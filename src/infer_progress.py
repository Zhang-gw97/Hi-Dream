import os, json, argparse
from pathlib import Path
from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from PIL import Image
import matplotlib.cm as mpl_cm

from dataset_nsd_cond import NSDCondDataset
from roi_xattn import ROIChannelCrossAttention

from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    UNet2DConditionModel,
)
from transformers import CLIPTextModel, CLIPTokenizer


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--subj", required=True)          # e.g. subj01
    ap.add_argument("--space", default="func1pt8mm")
    ap.add_argument("--cond_dir", default=None)       # default -> {root}/nsddata/ppdata/{subj}/{space}/condmaps

    ap.add_argument("--model_dir", required=True)     # your LOSO model dir, contains controlnet_finetuned + roi_mixer.pth
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--sd_id", default="runwayml/stable-diffusion-v1-5")
    ap.add_argument("--cache_dir", default=None)      # HF cache (optional)
    ap.add_argument("--local_files_only", action="store_true")  # if models are already cached

    ap.add_argument("--num_inference_steps", type=int, default=50)
    ap.add_argument("--guidance_scale", type=float, default=7.5)
    ap.add_argument("--cond_scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=512)

    ap.add_argument("--shared1000", required=True)    # path to shared1000.npy
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)   # 0 -> no limit
    ap.add_argument("--save_gt", action="store_true")
    ap.add_argument("--disable_xattn", action="store_true")  # disable roi_mixer usage
    ap.add_argument("--depth_scales", type=str, default=None,
                    help="override shallow,mid,deep scales; default loads from model_dir/depth_scale_config.json if present")

    # progress visualization
    ap.add_argument("--save_progress", action="store_true",
                    help="save step-wise intermediate predictions and heatmaps")
    ap.add_argument("--save_steps", type=str, default="1,10,20,30,40,50",
                    help="comma-separated denoising step indices to save (1-based)")
    ap.add_argument("--progress_subdir", type=str, default="progress",
                    help="subdirectory under out_dir to store progress outputs")
    ap.add_argument("--heatmap_alpha", type=float, default=0.38,
                    help="alpha blending for heatmap overlays")
    ap.add_argument("--progress_heatmap_mode", type=str, default="ablation",
                    choices=["ablation", "residual"],
                    help="ablation = group contribution via group-zeroing; residual = raw residual magnitude")
    return ap.parse_args()


def parse_int_list(s: str):
    vals = [int(x.strip()) for x in s.split(",") if x.strip()]
    if len(vals) == 0:
        raise ValueError(f"Expected at least one integer, got: {s}")
    return vals


def parse_depth_scales(s: str):
    vals = [float(x.strip()) for x in s.split(",") if x.strip()]
    if len(vals) != 3:
        raise ValueError(f"depth scales must be 3 comma-separated floats, got: {s}")
    return vals[0], vals[1], vals[2]


def load_depth_scales(model_dir: Path, override: Optional[str]):
    if override is not None and len(override.strip()) > 0:
        return parse_depth_scales(override)
    cfg_path = model_dir / "depth_scale_config.json"
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        vals = cfg.get("depth_scales", None)
        if isinstance(vals, (list, tuple)) and len(vals) == 3:
            return tuple(float(x) for x in vals)
        if isinstance(vals, str):
            return parse_depth_scales(vals)
    return (1.0, 1.0, 1.0)


def scale_controlnet_residuals(down_block_res, mid_block_res, scales):
    shallow_s, mid_s, deep_s = scales
    n = len(down_block_res)
    if n == 0:
        return down_block_res, mid_block_res

    b1 = max(1, (n + 2) // 3)
    b2 = max(b1 + 1, (2 * n + 2) // 3)

    scaled = []
    for i, res in enumerate(down_block_res):
        if i < b1:
            s = shallow_s
        elif i < b2:
            s = mid_s
        else:
            s = deep_s
        scaled.append(res * s)

    mid_scaled = mid_block_res * deep_s if mid_block_res is not None else None
    return tuple(scaled), mid_scaled


def zero_group_residuals(down_block_res, mid_block_res, group_name: str):
    n = len(down_block_res)
    if n == 0:
        return down_block_res, mid_block_res
    b1 = max(1, (n + 2) // 3)
    b2 = max(b1 + 1, (2 * n + 2) // 3)

    new_down = []
    for i, res in enumerate(down_block_res):
        if group_name == "shallow" and i < b1:
            new_down.append(torch.zeros_like(res))
        elif group_name == "mid" and b1 <= i < b2:
            new_down.append(torch.zeros_like(res))
        elif group_name == "deep" and i >= b2:
            new_down.append(torch.zeros_like(res))
        else:
            new_down.append(res)

    new_mid = mid_block_res
    if group_name == "deep" and mid_block_res is not None:
        new_mid = torch.zeros_like(mid_block_res)
    return tuple(new_down), new_mid


def tensor_to_uint8_img(x_chw_neg1_to1: torch.Tensor) -> Image.Image:
    x = (x_chw_neg1_to1.clamp(-1, 1) + 1) * 0.5
    x = (x * 255.0).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(x)


def tensor_to_pil_from_01(x_chw_01: torch.Tensor) -> Image.Image:
    x = x_chw_01.clamp(0, 1)
    x = (x * 255.0).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(x)


def save_heat_overlay(base_img: Image.Image, heat_2d: torch.Tensor, out_path: Path, alpha: float = 0.38):
    hm = heat_2d.detach().float().cpu().numpy()
    hm = np.clip(hm, 0.0, 1.0)
    rgb = (mpl_cm.get_cmap("magma")(hm)[..., :3] * 255.0).astype(np.uint8)
    heat_img = Image.fromarray(rgb).convert("RGB")
    overlay = Image.blend(base_img.convert("RGB"), heat_img, alpha=float(alpha))
    overlay.save(out_path)


def normalize_heatmaps(hm: torch.Tensor):
    """
    hm: [B, 1, H, W]
    normalize per-sample to [0, 1]
    """
    b = hm.shape[0]
    flat = hm.view(b, -1)
    mn = flat.min(dim=1)[0].view(b, 1, 1, 1)
    mx = flat.max(dim=1)[0].view(b, 1, 1, 1)
    return (hm - mn) / (mx - mn + 1e-6)


def residuals_to_group_heatmaps(down_block_res, mid_block_res, out_hw):
    """
    Raw residual-energy heatmaps (fallback mode).
    """
    h, w = out_hw
    n = len(down_block_res)
    if n == 0:
        ref = mid_block_res
        if ref is None:
            raise ValueError("No residuals available.")
        zeros = torch.zeros((ref.shape[0], 1, h, w), device=ref.device)
        return {"shallow": zeros, "mid": zeros, "deep": zeros}

    b1 = max(1, (n + 2) // 3)
    b2 = max(b1 + 1, (2 * n + 2) // 3)

    groups = {
        "shallow": list(down_block_res[:b1]),
        "mid": list(down_block_res[b1:b2]),
        "deep": list(down_block_res[b2:]),
    }
    if mid_block_res is not None:
        groups["deep"].append(mid_block_res)

    out = {}
    for name, res_list in groups.items():
        if len(res_list) == 0:
            ref = mid_block_res if mid_block_res is not None else down_block_res[0]
            out[name] = torch.zeros((ref.shape[0], 1, h, w), device=ref.device, dtype=torch.float32)
            continue

        acc = None
        for res in res_list:
            hm = res.abs().float().mean(dim=1, keepdim=True)   # [B, 1, h, w]
            hm = F.interpolate(hm, size=(h, w), mode="bilinear", align_corners=False)
            acc = hm if acc is None else acc + hm
        acc = acc / float(len(res_list))
        out[name] = normalize_heatmaps(acc)
    return out


def decode_latents_to_tensor(vae, latents):
    latents = latents / 0.18215
    image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    return image


def decode_latents(vae, latents):
    image = decode_latents_to_tensor(vae, latents)
    image = image.detach().cpu().permute(0, 2, 3, 1).float().numpy()
    images = (image * 255).round().astype("uint8")
    return [Image.fromarray(img) for img in images]


def predict_x0_from_noise(latents, noise_pred, timesteps, scheduler):
    """
    x0 = (x_t - sqrt(1 - alpha_t) * eps) / sqrt(alpha_t)
    latents, noise_pred: [B, C, H, W]
    timesteps: scalar tensor or batch tensor
    """
    alphas_cumprod = scheduler.alphas_cumprod.to(device=latents.device, dtype=latents.dtype)

    if torch.is_tensor(timesteps):
        t = timesteps.long().view(-1)
    else:
        t = torch.tensor([int(timesteps)], device=latents.device, dtype=torch.long)

    a_t = alphas_cumprod[t].view(-1, 1, 1, 1)
    x0 = (latents - (1.0 - a_t).sqrt() * noise_pred) / torch.clamp(a_t.sqrt(), min=1e-6)
    return x0

def load_roi_mixer(roi_path: Path, device: torch.device):
    if not roi_path.exists():
        return None
    roi_mixer = ROIChannelCrossAttention(d_model=64, n_heads=4, dropout=0.0).to(device)

    state = torch.load(str(roi_path), map_location="cpu")
    state = state.get("state_dict", state)
    state = state.get("model", state)
    if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

    roi_mixer.load_state_dict(state, strict=False)
    roi_mixer.eval()
    return roi_mixer


def prepare_text_embeddings(tokenizer, text_encoder, prompts, device, do_cfg: bool):
    text_inputs = tokenizer(
        prompts,
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    )
    text_embeds = text_encoder(text_inputs.input_ids.to(device))[0]

    if not do_cfg:
        return text_embeds

    uncond_inputs = tokenizer(
        [""] * len(prompts),
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    )
    uncond_embeds = text_encoder(uncond_inputs.input_ids.to(device))[0]
    return torch.cat([uncond_embeds, text_embeds], dim=0)


def build_shared_kid_set(shared1000_path: str):
    shared = np.load(shared1000_path)
    shared = np.asarray(shared).reshape(-1)
    if shared.size == 73000 and (shared.dtype == np.bool_ or np.isin(shared, [0, 1]).all()):
        mask = shared.astype(bool)
        return set((np.where(mask)[0] + 1).tolist())
    ids = shared.astype(np.int64).tolist()
    if len(ids) > 0 and min(ids) == 0:
        return set([i + 1 for i in ids])
    return set(ids)


def make_sample_name(kid: int, sess: int, lidx: int) -> str:
    return f"kid{kid:05d}_sess{sess:02d}_idx{lidx:04d}"


@torch.no_grad()
def main():
    args = get_args()

    if args.cache_dir:
        os.environ.setdefault("HF_HOME", args.cache_dir)
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", args.cache_dir)
        os.environ.setdefault("HF_HUB_CACHE", args.cache_dir)
        os.environ.setdefault("TRANSFORMERS_CACHE", args.cache_dir)
        os.environ.setdefault("DIFFUSERS_CACHE", args.cache_dir)
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "recon").mkdir(exist_ok=True)
    if args.save_gt:
        (out_dir / "gt").mkdir(exist_ok=True)

    progress_root = out_dir / args.progress_subdir
    save_steps = set(parse_int_list(args.save_steps)) if args.save_progress else set()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    weight_dtype = torch.float16 if device.type == "cuda" else torch.float32

    if args.cond_dir is None:
        args.cond_dir = str(Path(args.root) / f"nsddata/ppdata/{args.subj}/{args.space}/condmaps")

    ds = NSDCondDataset(
        root=args.root,
        subj=args.subj,
        space=args.space,
        cond_dir=args.cond_dir,
        use_captions=True,
    )

    shared_kid_set = build_shared_kid_set(args.shared1000)

    keep = []
    for i, rec in enumerate(ds.items):
        kid = ds._local_idx_to_kid(rec["sess"], rec["local_idx"])
        if kid in shared_kid_set:
            keep.append(i)

    if args.limit and args.limit > 0:
        keep = keep[: args.limit]

    print(f"[infer] subj={args.subj} | cond_dir={args.cond_dir}")
    print(f"[infer] shared1000 matched: {len(keep)} samples")

    loader = DataLoader(
        Subset(ds, keep),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    model_dir = Path(args.model_dir)
    ctrl_dir = model_dir / "controlnet_finetuned"
    roi_path = model_dir / "roi_mixer.pth"
    depth_scales = load_depth_scales(model_dir, args.depth_scales)
    print(f"[infer] explicit depth scales shallow/mid/deep = {depth_scales}")

    tokenizer = CLIPTokenizer.from_pretrained(
        args.sd_id, subfolder="tokenizer",
        cache_dir=args.cache_dir, local_files_only=args.local_files_only,
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.sd_id, subfolder="text_encoder",
        cache_dir=args.cache_dir, local_files_only=args.local_files_only,
        torch_dtype=weight_dtype,
    ).to(device)
    vae = AutoencoderKL.from_pretrained(
        args.sd_id, subfolder="vae",
        cache_dir=args.cache_dir, local_files_only=args.local_files_only,
        torch_dtype=weight_dtype,
    ).to(device)
    unet = UNet2DConditionModel.from_pretrained(
        args.sd_id, subfolder="unet",
        cache_dir=args.cache_dir, local_files_only=args.local_files_only,
        torch_dtype=weight_dtype,
    ).to(device)
    controlnet = ControlNetModel.from_pretrained(
        str(ctrl_dir),
        torch_dtype=weight_dtype,
        local_files_only=args.local_files_only,
        cache_dir=args.cache_dir,
    ).to(device)
    scheduler = DDPMScheduler.from_pretrained(
        args.sd_id, subfolder="scheduler",
        cache_dir=args.cache_dir, local_files_only=args.local_files_only,
    )

    unet.eval(); controlnet.eval(); text_encoder.eval(); vae.eval()

    roi_mixer = None
    if (not args.disable_xattn) and roi_path.exists():
        roi_mixer = load_roi_mixer(roi_path, device)
        print("[INFO] ROI mixer loaded:", roi_path)
    else:
        print("[INFO] ROI mixer disabled.")

    meta_f = open(out_dir / "meta.jsonl", "w", encoding="utf-8")
    global_idx = 0
    do_cfg = args.guidance_scale is not None and args.guidance_scale > 1.0

    if args.save_progress:
        progress_root.mkdir(parents=True, exist_ok=True)

    for batch in loader:
        cond = batch["cond_values"].to(device, dtype=torch.float32)
        if roi_mixer is not None:
            with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=torch.float16, enabled=False):
                cond = roi_mixer(cond).clamp(0, 1)
        cond = cond.to(dtype=weight_dtype)

        prompts = batch.get("prompt", None)
        prompts = [""] * cond.shape[0] if prompts is None else list(prompts)
        encoder_hidden_states = prepare_text_embeddings(tokenizer, text_encoder, prompts, device, do_cfg).to(dtype=weight_dtype)

        scheduler.set_timesteps(args.num_inference_steps, device=device)
        latents = torch.randn(
            (cond.shape[0], unet.config.in_channels, args.height // 8, args.width // 8),
            generator=torch.Generator(device=device).manual_seed(args.seed + global_idx),
            device=device,
            dtype=weight_dtype,
        )
        latents = latents * scheduler.init_noise_sigma

        for step_idx, t in enumerate(scheduler.timesteps, start=1):
            latent_model_input = torch.cat([latents, latents], dim=0) if do_cfg else latents
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)
            control_cond = torch.cat([cond, cond], dim=0) if do_cfg else cond

            base_down_block_res, base_mid_block_res = controlnet(
                sample=latent_model_input.to(dtype=weight_dtype),
                timestep=t,
                encoder_hidden_states=encoder_hidden_states.to(dtype=weight_dtype),
                controlnet_cond=control_cond.to(dtype=weight_dtype),
                conditioning_scale=args.cond_scale,
                return_dict=False,
            )
            full_down_block_res, full_mid_block_res = scale_controlnet_residuals(
                base_down_block_res, base_mid_block_res, depth_scales
            )

            noise_pred = unet(
                sample=latent_model_input,
                timestep=t,
                encoder_hidden_states=encoder_hidden_states,
                down_block_additional_residuals=full_down_block_res,
                mid_block_additional_residual=full_mid_block_res,
                return_dict=False,
            )[0]

            if do_cfg:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_text - noise_pred_uncond)

            if args.save_progress and (step_idx in save_steps):
                # Base prediction, from which we derive group contributions
                pred_latents = predict_x0_from_noise(latents, noise_pred, t, scheduler)
                pred_img_t = decode_latents_to_tensor(vae, pred_latents)  # [B,3,H,W] in [0,1]
                pred_img_pil = [tensor_to_pil_from_01(pred_img_t[b]) for b in range(pred_img_t.shape[0])]

                if args.progress_heatmap_mode == "ablation":
                    group_hmaps = {}
                    for group_name in ["shallow", "mid", "deep"]:
                        zero_down, zero_mid = zero_group_residuals(full_down_block_res, full_mid_block_res, group_name)
                        ab_noise = unet(
                            sample=latent_model_input,
                            timestep=t,
                            encoder_hidden_states=encoder_hidden_states,
                            down_block_additional_residuals=zero_down,
                            mid_block_additional_residual=zero_mid,
                            return_dict=False,
                        )[0]
                        if do_cfg:
                            ab_uncond, ab_text = ab_noise.chunk(2)
                            ab_noise = ab_uncond + args.guidance_scale * (ab_text - ab_uncond)

                        ab_pred_latents = predict_x0_from_noise(latents, ab_noise, t, scheduler)
                        ab_img_t = decode_latents_to_tensor(vae, ab_pred_latents)

                        diff = (pred_img_t - ab_img_t).abs().mean(dim=1, keepdim=True)  # [B,1,H,W]
                        diff = normalize_heatmaps(diff)
                        group_hmaps[group_name] = diff
                else:
                    group_hmaps = residuals_to_group_heatmaps(
                        down_block_res=full_down_block_res,
                        mid_block_res=full_mid_block_res,
                        out_hw=(args.height, args.width),
                    )

                curr_decode_imgs = decode_latents(vae, latents)
                for b in range(len(pred_img_pil)):
                    kid = int(batch["kid"][b])
                    sess = int(batch["sess"][b])
                    lidx = int(batch["local_idx"][b])
                    sample_name = make_sample_name(kid, sess, lidx)
                    sample_root = progress_root / sample_name
                    step_root = sample_root / f"step_{step_idx:03d}"
                    step_root.mkdir(parents=True, exist_ok=True)

                    # step-wise images
                    pred_img_pil[b].save(step_root / "pred_x0.png")
                    curr_decode_imgs[b].save(step_root / "latent_decode.png")

                    # per-group heatmaps and overlays
                    for group_name in ["shallow", "mid", "deep"]:
                        hm = group_hmaps[group_name][b, 0]
                        np.save(step_root / f"{group_name}_heat.npy", hm.detach().float().cpu().numpy())
                        save_heat_overlay(
                            pred_img_pil[b],
                            hm,
                            step_root / f"{group_name}_overlay.png",
                            alpha=args.heatmap_alpha,
                        )

                    if args.save_gt:
                        gt_img = tensor_to_uint8_img(batch["pixel_values"][b])
                        if not (sample_root / "gt.png").exists():
                            gt_img.save(sample_root / "gt.png")

                    meta_path = sample_root / "meta.json"
                    if not meta_path.exists():
                        meta = dict(
                            kid=kid,
                            sess=sess,
                            local_idx=lidx,
                            subj=args.subj,
                            depth_scales=[float(x) for x in depth_scales],
                            guidance=float(args.guidance_scale),
                            cond_scale=float(args.cond_scale),
                            save_steps=sorted(list(save_steps)),
                            num_inference_steps=int(args.num_inference_steps),
                            seed=int(args.seed),
                            progress_alpha=float(args.heatmap_alpha),
                            heatmap_mode=args.progress_heatmap_mode,
                        )
                        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

            latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        images = decode_latents(vae, latents)

        for b in range(len(images)):
            kid = int(batch["kid"][b])
            sess = int(batch["sess"][b])
            lidx = int(batch["local_idx"][b])
            name = make_sample_name(kid, sess, lidx) + ".png"
            images[b].save(out_dir / "recon" / name)

            if args.save_gt:
                gt = tensor_to_uint8_img(batch["pixel_values"][b])
                gt.save(out_dir / "gt" / name)

            if args.save_progress:
                sample_root = progress_root / make_sample_name(kid, sess, lidx)
                final_root = sample_root / "final"
                final_root.mkdir(parents=True, exist_ok=True)
                images[b].save(final_root / "recon.png")

                if (sample_root / f"step_{args.num_inference_steps:03d}" / "pred_x0.png").exists():
                    src = sample_root / f"step_{args.num_inference_steps:03d}" / "pred_x0.png"
                    dst = final_root / "pred_x0.png"
                    if not dst.exists():
                        dst.write_bytes(src.read_bytes())

            meta = dict(
                file=name,
                kid=kid, sess=sess, local_idx=lidx,
                seed=int(args.seed + global_idx),
                steps=int(args.num_inference_steps),
                guidance=float(args.guidance_scale),
                cond_scale=float(args.cond_scale),
                depth_scales=[float(x) for x in depth_scales],
                subj=args.subj,
            )
            meta_f.write(json.dumps(meta) + "\n")
            global_idx += 1

        if device.type == "cuda":
            torch.cuda.empty_cache()

    meta_f.close()
    print("[DONE] saved to:", out_dir)
    if args.save_progress:
        print("[DONE] progress saved to:", progress_root)


if __name__ == "__main__":
    main()
