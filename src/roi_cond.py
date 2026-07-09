#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch-build 3-channel ControlNet conditioning from NSD betas + ROI masks.

C0 (early) = V1 & V2
C1 (mid)   = V3 & hV4
C2 (late)  = FFA (+ LOC if present)

Outputs for each (session, idx):
  - <OUTDIR>/cond_betas_sessionXX.nii_idx{idx}.npy  (3x512x512, float32 in [0,1])
  - <OUTDIR>/cond_betas_sessionXX.nii_idx{idx}.png  (preview)
  - <OUTDIR>/cond_betas_sessionXX.nii_idx{idx}.json (metadata)
"""

import os, glob, argparse, json
from pathlib import Path
import numpy as np
import nibabel as nib
from PIL import Image, ImageFilter


# ----------------- Defaults -----------------
ROOT = None
SUBJ = "subj07"
SPACE = "func1pt8mm"

ROI_FILENAMES = {
    "V1" : "V1.nii.gz",
    "V2" : "V2.nii.gz",
    "V3" : "V3.nii.gz",
    "hV4": "hV4.nii.gz",
    "FFA": "FFA.nii.gz",
    "LOC": "LOC.nii.gz",  # optional
}

# ----------------- Helpers -----------------
def load_nii_bool(p: Path):
    if not p.exists(): return None, None, None
    img = nib.load(str(p))
    data = img.get_fdata()
    m = (data > 0).astype(np.uint8)
    return m, img.affine, img.shape

def minmax01(x, eps=1e-8):
    mn, mx = x.min(), x.max()
    if mx - mn < eps: return np.zeros_like(x, dtype=np.float32)
    return ((x - mn) / (mx - mn)).astype(np.float32)

def zmax_project(vol3d):
    # volume shape: X,Y,Z -> project over Z -> HxW
    return vol3d.max(axis=2)

def resize_to_512(x_hw):
    im = Image.fromarray((np.clip(x_hw,0,1)*255).astype(np.uint8))
    im = im.resize((512,512), Image.BICUBIC)
    arr = np.asarray(im).astype(np.float32)/255.0
    return arr

def _resize01(x_hw, size=512):
    im = Image.fromarray((np.clip(x_hw,0,1)*255).astype(np.uint8))
    im = im.resize((size,size), Image.BICUBIC)
    return np.asarray(im).astype(np.float32)/255.0

def _group_or(masks, *keys):
    acc = None
    for k in keys:
        m = masks.get(k)
        if m is None: continue
        acc = m if acc is None else np.maximum(acc, m)
    return acc


def build_cond_from_beta(beta_vol, roi_masks: dict,
                         norm="z_in_roi", proj="zmean",
                         percentiles=(1,99), smooth_sigma=0.8, out_size=512):
    """
    返回 (3, out_size, out_size) in [0,1]
    - C0=V1∪V2, C1=V3∪hV4, C2=FFA(∪LOC)
    - 先 ROI 内 z-score，再投影；投影后在 ROI 支撑内做 P1–P99 拉伸；ROI 外显式置0
    """
    assert beta_vol.ndim==3
    early = _group_or(roi_masks, "V1","V2")
    mid   = _group_or(roi_masks, "V3","hV4")
    late  = _group_or(roi_masks, "FFA","LOC")

    chans=[]
    for m in [early, mid, late]:
        if m is None or m.sum()==0:
            chans.append(np.zeros((out_size,out_size), np.float32)); continue

        roi_vals = beta_vol[m>0]
        if norm=="z_in_roi":
            mu, sd = roi_vals.mean(), roi_vals.std() + 1e-6
            v = (beta_vol - mu)/sd
        elif norm=="minmax_in_roi":
            mn, mx = roi_vals.min(), roi_vals.max()
            v = (beta_vol - mn)/(mx - mn + 1e-6)
        else:
            gmn, gmx = beta_vol.min(), beta_vol.max()
            v = (beta_vol - gmn)/(gmx - gmn + 1e-6)

        v = v*(m>0)  # ROI 外清零

        # 投影
        if proj=="zmean":
            ch2d = v.mean(axis=2)
        elif proj=="energy":
            ch2d = np.sqrt((v*v).mean(axis=2))
        else:  # zmax
            ch2d = v.max(axis=2)

        # 仅在 ROI 的 2D 支撑上做百分位拉伸
        m2d = (m>0).any(axis=2)
        if m2d.sum()>0:
            vals = ch2d[m2d]
            a,b = np.percentile(vals, percentiles)
            if b>a:
                ch2d = (ch2d - a)/(b - a)
            ch2d[~m2d] = 0.0
            ch2d = np.clip(ch2d, 0.0, 1.0)
        else:
            ch2d = np.zeros_like(ch2d, dtype=np.float32)

        # 轻度平滑（可关）
        if smooth_sigma and smooth_sigma>0:
            im = Image.fromarray((ch2d*255).astype(np.uint8))
            im = im.filter(ImageFilter.GaussianBlur(radius=float(smooth_sigma)))
            ch2d = np.asarray(im).astype(np.float32)/255.0

        chans.append(_resize01(ch2d, out_size))

    return np.stack(chans, axis=0).astype(np.float32)


def save_one(prefix: Path, cond: np.ndarray, meta: dict):
    np.save(str(prefix)+".npy", cond)
    preview = (cond.mean(axis=0)*255).astype(np.uint8)
    Image.fromarray(preview).save(str(prefix)+".png")
    with open(str(prefix)+".json","w") as f:
        json.dump(meta, f, indent=2)

# ----------------- Main (batch) -----------------
def parse_sessions(arg: str):
    """
    'all' -> [1..40]
    '1,2,5' -> [1,2,5]
    '1-4,10,20-22' -> [1,2,3,4,10,20,21,22]
    """
    if arg.lower() == "all":
        return list(range(1,41))
    out = []
    for chunk in arg.split(","):
        chunk = chunk.strip()
        if "-" in chunk:
            a,b = chunk.split("-")
            out.extend(range(int(a), int(b)+1))
        else:
            out.append(int(chunk))
    return sorted(set(out))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="Path to the NSD root directory containing nsddata/ and nsddata_betas/.")
    ap.add_argument("--subj", default=SUBJ)
    ap.add_argument("--space", default=SPACE)
    ap.add_argument("--sessions", default="all",
                    help="e.g. 'all' or '1-5,10,20-22'")
    ap.add_argument("--norm", default="z_in_roi",
                    choices=["z_in_roi","minmax_in_roi","minmax_global"])
    ap.add_argument("--proj", default="zmax",
                    choices=["zmax","zmean"])
    ap.add_argument("--stride", type=int, default=1,
                    help="take every k-th volume per session")
    ap.add_argument("--limit", type=int, default=0,
                    help="optional max volumes per session (0=all)")
    ap.add_argument("--overwrite", action="store_true",
                    help="recompute even if files exist")
    args = ap.parse_args()

    root  = Path(args.root)
    ppdir = root / "nsddata" / "ppdata" / args.subj / args.space
    betadir = root / "nsddata_betas" / "ppdata" / args.subj / args.space / "betas_fithrf_GLMdenoise_RR"
    roidir = ppdir / "rois"
    outdir  = ppdir / "condmaps"
    outdir.mkdir(parents=True, exist_ok=True)

    # Load ROI masks once
    roi_masks, rois_used = {}, []
    for k, filename in ROI_FILENAMES.items():
        p = roidir / filename
        m, aff, shp = load_nii_bool(p)
        if m is None:
            print(f"[WARN] missing ROI: {k}")
            continue
        roi_masks[k] = m
        rois_used.append(k)
    if not roi_masks:
        raise RuntimeError("No ROI masks found. Please generate V1/V2/V3/hV4/FFA(/LOC) first.")

    sessions = parse_sessions(args.sessions)
    print(f"[INFO] sessions: {sessions} | stride={args.stride} limit={args.limit or 'ALL'}")
    total_written = 0

    for s in sessions:
        pat = betadir / f"betas_session{str(s).zfill(2)}.nii.gz"
        if not pat.exists():
            print(f"[MISS] {pat} (skip)"); continue

        bimg = nib.load(str(pat))
        bdat = bimg.get_fdata()  # X,Y,Z,T
        assert bdat.ndim == 4, f"bad beta shape: {bdat.shape}"
        T = bdat.shape[3]
        print(f"[S{str(s).zfill(2)}] volumes={T}")

        # iterate indices
        count_this = 0
        for idx in range(0, T, args.stride):
            if args.limit and count_this >= args.limit:
                break
            prefix = outdir / f"cond_betas_session{str(s).zfill(2)}.nii_idx{idx}"
            if (not args.overwrite) and (prefix.with_suffix(".npy").exists()):
                # resume-friendly
                continue

            vol = bdat[..., idx]

            cond = build_cond_from_beta(vol, roi_masks, norm=args.norm, proj=args.proj)
            meta = dict(
                beta=str(pat), session=s, index=idx,
                norm=args.norm, proj=args.proj, rois=rois_used,
                subj=args.subj, space=args.space
            )
            save_one(prefix, cond, meta)

            count_this += 1
            total_written += 1

        print(f"  -> wrote {count_this} condmaps")

    print(f"[ALL DONE] total written: {total_written} | outdir={outdir}")

if __name__ == "__main__":
    main()

