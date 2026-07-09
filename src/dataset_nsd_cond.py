import os, re, glob
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from nsd_access import NSDAccess

TARGET_RES = 512  # 必须是 64 的倍数，比如 512/448/384


class NSDCondDataset(Dataset):
    """
    - 扫描 {root}/nsddata/ppdata/{subj}/{space}/condmaps/**/*.npy 作为条件图
    - 用文件名解析出 sess & 本地 idx，再通过 read_behavior 映射到 73KID
    - 用 NSDAccess.read_image(73KID) 读取 stimulus（自动处理 0/1 基准等细节）
    """
    def __init__(self, root, subj="subj01", space="func1pt8mm",
                 cond_dir=None, resize=512, limit=None,
                 use_captions=True,
                 caps_topk=3,
                 caps_max_chars=256,
                 early_edge_boost=0.0,
                 early_edge_blur_radius=2): 
        self.root = root
        self.subj = subj
        self.space = space
        self.resize = resize

        # NSD 访问器 + 行为表缓存
        self.nsda = NSDAccess(self.root)
        self._beh_cache = {}
        self._cap_cache = {} 
        self.use_captions = use_captions
        self.caps_topk = int(caps_topk)
        self.caps_max_chars = int(caps_max_chars)
        self.early_edge_boost = float(early_edge_boost)
        self.early_edge_blur_radius = int(early_edge_blur_radius)

        # 路径
        self.cond_dir = cond_dir or f"{root}/nsddata/ppdata/{subj}/{space}/condmaps"
        Path(self.cond_dir).mkdir(parents=True, exist_ok=True)

        # 扫描 cond 文件
        self.items = self._scan_items()
        if limit is not None:
            self.items = self.items[:int(limit)]
        if len(self.items) == 0:
            raise RuntimeError(f"[NSDCondDataset] No .npy found in {self.cond_dir}")

        print(f"[NSDCondDataset] {len(self.items)} pairs (subject={self.subj}, space={self.space})")

    # ---------- 行为表 ----------
    def _get_beh_df(self, sess: int):
        if sess not in self._beh_cache:
            df = self.nsda.read_behavior(self.subj, sess, [])
            if len(df) == 0:
                # 兼容 0/1 起始的两种标号
                df = self.nsda.read_behavior(self.subj, sess - 1, [])
            if len(df) == 0:
                raise RuntimeError(f"[NSDCondDataset] No behavior found for {self.subj} sess={sess} (0/1-based).")
            self._beh_cache[sess] = df.reset_index(drop=True)
        return self._beh_cache[sess]

    # ---------- 扫描 cond 文件 ----------
    def _scan_items(self):
        # 例：cond_betas_session01.nii_idx161.npy 或 .../sess01/...idx0161.npy
        pat_idx  = re.compile(r"idx(\d+)")
        pat_sess = re.compile(r"(?:session|sess)[^\d]*?(\d+)", re.IGNORECASE)

        items = []
        npys = sorted(glob.glob(os.path.join(self.cond_dir, "**", "*.npy"), recursive=True))
        for p in npys:
            base = os.path.basename(p)
            m_idx = pat_idx.search(base)
            if not m_idx:
                continue
            local_idx = int(m_idx.group(1))  # 会话内 trial 序号（可能 1-based 或 0-based）

            m_sess = pat_sess.search(p)
            if not m_sess:
                raise ValueError(f"No session number found in filename/path: {p}")
            sess = int(m_sess.group(1))

            items.append({"sess": sess, "local_idx": local_idx, "cond_npy": p})
        return items

    def _local_idx_to_kid(self, sess: int, local_idx: int) -> int:
        df = self._get_beh_df(sess)

        # 先按 1-based 试（idx161 -> 第 161 条）
        pos = local_idx - 1
        if 0 <= pos < len(df):
            return int(df.iloc[pos]["73KID"])

        # 再按 0-based 试
        pos = local_idx
        if 0 <= pos < len(df):
            return int(df.iloc[pos]["73KID"])

        # 兜底：直接匹配行为表的 LOCAL_TRIAL_INDEX
        if "LOCAL_TRIAL_INDEX" in df.columns:
            hit = df[df["LOCAL_TRIAL_INDEX"] == local_idx]
            if len(hit) == 1:
                return int(hit.iloc[0]["73KID"])

        raise IndexError(f"[NSDCondDataset] sess={sess} local_idx={local_idx} cannot map to 73KID (len={len(df)})")

    # ---------- 读取 COCO captions（带缓存） ----------
    def _prompt_from_captions(self, kid0: int) -> str:
        if kid0 in self._cap_cache:
            return self._cap_cache[kid0]

        try:
            caps = self.nsda.read_image_coco_info([kid0], info_type='captions',
                                                  show_annot=False, show_img=False)
            texts = []
            # 返回格式通常是 List[Dict]（单图），取前 k 条
            if isinstance(caps, list):
                for c in caps[:self.caps_topk]:
                    if isinstance(c, dict) and "caption" in c:
                        t = str(c["caption"]).strip()
                        if t: texts.append(t)
            prompt = " ".join(texts).strip()[:self.caps_max_chars]
        except Exception:
            prompt = ""

        # fallback：可能某些图无标注
        if not prompt:
            prompt = ""

        self._cap_cache[kid0] = prompt
        return prompt



    def _apply_early_edge_enhance(self, cond: np.ndarray) -> np.ndarray:
        """Apply a light high-pass / edge emphasis only to the early channel (index 0)."""
        if cond.shape[0] < 1 or self.early_edge_boost <= 0:
            return cond
        ch = np.clip(cond[0], 0.0, 1.0)
        img = Image.fromarray((ch * 255.0).astype(np.uint8))
        blur = img.filter(Image.Filter.GaussianBlur(radius=self.early_edge_blur_radius)) if hasattr(Image, 'Filter') else img
        # PIL exposes filters via ImageFilter, import lazily to avoid global change
        try:
            from PIL import ImageFilter
            blur = img.filter(ImageFilter.GaussianBlur(radius=self.early_edge_blur_radius))
        except Exception:
            blur = img
        blur_np = np.asarray(blur, dtype=np.float32) / 255.0
        hp = ch - blur_np
        cond[0] = np.clip(ch + self.early_edge_boost * hp, 0.0, 1.0)
        return cond

    # ---------- 数据读取 ----------
    def __getitem__(self, i):
        rec = self.items[i]

        # --- map to 73KID ---
        #kid = self._local_idx_to_kid(rec["sess"], rec["local_idx"])

        kid = self._local_idx_to_kid(rec["sess"], rec["local_idx"])  # 1..73000
        kid0 = kid - 1                                               # 0..72999
        img_np = self.nsda.read_images([kid0])[0]                    # HxWx3, uint8


        # 1) 刺激图：用 NSDAccess 读取（避免 off-by-one）
        #img_np = self.nsda.read_images(kid)          # HxWx3, uint8
        img = Image.fromarray(img_np)
        if img.size != (TARGET_RES, TARGET_RES):
            img = img.resize((TARGET_RES, TARGET_RES), Image.BICUBIC)
        img = np.asarray(img, dtype=np.float32) / 127.5 - 1.0   # [-1,1]
        img = torch.from_numpy(img.transpose(2, 0, 1))          # CHW

        # 2) condmap 原样加载 -> (3, 512, 512), [0,1]
        cond = np.load(rec["cond_npy"]).astype(np.float32)
        if cond.ndim == 2:
            cond = np.stack([cond] * 3, 0)
        if cond.shape[1] != TARGET_RES or cond.shape[2] != TARGET_RES:
            chans = []
            for c in range(cond.shape[0]):
                ch = Image.fromarray((cond[c] * 255.0).astype(np.uint8))
                ch = ch.resize((TARGET_RES, TARGET_RES), Image.BILINEAR)
                chans.append(np.asarray(ch, dtype=np.float32) / 255.0)
            cond = np.stack(chans, 0)
        cond = np.clip(cond, 0.0, 1.0)
        cond = self._apply_early_edge_enhance(cond)
        cond = torch.from_numpy(cond)

        prompt = self._prompt_from_captions(kid0) if self.use_captions else ""

        return {
            "pixel_values": img,
            "cond_values": cond,
            "prompt": prompt,                    # <<< 关键：非空时就会被 CLIP 编码
            "sess": rec["sess"], "local_idx": rec["local_idx"], "kid": kid, "cond_npy": rec["cond_npy"]
        }

    def __len__(self):
        return len(self.items)