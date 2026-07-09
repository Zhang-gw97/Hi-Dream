"""
MindEye2 Image Reconstruction Evaluation Script
Computes 8 metrics: PixCorr, SSIM, AlexNet(2), AlexNet(5), InceptionV3, CLIP, EffNet-B, SwAV

Usage:
    python evaluate_metrics.py \
        --gt_dir /path/to/ground_truth \
        --recon_dir /path/to/reconstructions

Images in both directories must be paired by filename (sorted order).
"""

import argparse
import os
from pathlib import Path
from typing import Optional
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models.feature_extraction import create_feature_extractor

import scipy as sp
import pandas as pd

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
CACHE_DIR = None


def configure_cache(cache_dir: Optional[str]):
    global CACHE_DIR
    if cache_dir is None:
        CACHE_DIR = str(Path.home() / ".cache")
        return

    CACHE_DIR = cache_dir
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.environ["TORCH_HOME"] = CACHE_DIR
    os.environ["HF_HOME"] = CACHE_DIR
    os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def load_images_from_dir(directory: str, imsize: int = 256) -> torch.Tensor:
    """Load all images from a directory, sorted by filename, resized to (imsize x imsize)."""
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
    fnames = sorted([f for f in os.listdir(directory) if os.path.splitext(f)[1].lower() in exts])
    if len(fnames) == 0:
        raise FileNotFoundError(f"No images found in {directory}")

    to_tensor = transforms.Compose([
        transforms.Resize((imsize, imsize), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),          # → [0, 1], shape (3, H, W)
    ])

    imgs = []
    for fname in tqdm(fnames, desc=f"Loading {os.path.basename(directory)}"):
        img = Image.open(os.path.join(directory, fname)).convert("RGB")
        imgs.append(to_tensor(img))
    return torch.stack(imgs)   # (N, 3, H, W)


@torch.no_grad()
def two_way_identification(all_recons, all_images, model, preprocess, feature_layer=None, device="cpu"):
    """
    2-way identification score (MindEye2 standard).
    Higher = better (max 1.0).
    """
    preds = model(torch.stack([preprocess(r) for r in all_recons]).to(device))
    reals = model(torch.stack([preprocess(im) for im in all_images]).to(device))

    if feature_layer is not None:
        preds = preds[feature_layer]
        reals = reals[feature_layer]

    preds = preds.float().flatten(1).cpu().numpy()
    reals = reals.float().flatten(1).cpu().numpy()

    r = np.corrcoef(reals, preds)
    r = r[:len(all_images), len(all_images):]
    congruents = np.diag(r)
    success = r < congruents
    success_cnt = np.sum(success, 0)
    perf = np.mean(success_cnt) / (len(all_images) - 1)
    return perf


# ─────────────────────────────────────────────
# Metric functions
# ─────────────────────────────────────────────

def compute_pixcorr(all_images: torch.Tensor, all_recons: torch.Tensor) -> float:
    """Pixel-level Pearson correlation (resize to 425 first, as in MindEye2)."""
    preprocess = transforms.Resize(425, interpolation=transforms.InterpolationMode.BILINEAR)
    flat_gt    = preprocess(all_images).reshape(len(all_images), -1).cpu()
    flat_recon = preprocess(all_recons).reshape(len(all_recons), -1).cpu()

    corrsum = 0.0
    for i in tqdm(range(len(all_images)), desc="PixCorr"):
        corrsum += np.corrcoef(flat_gt[i].numpy(), flat_recon[i].numpy())[0][1]
    return corrsum / len(all_images)


def compute_ssim(all_images: torch.Tensor, all_recons: torch.Tensor) -> float:
    """SSIM on grayscale images (resize to 425, gaussian_weights, sigma=1.5)."""
    from skimage.color import rgb2gray
    from skimage.metrics import structural_similarity as ssim_fn

    preprocess = transforms.Resize(425, interpolation=transforms.InterpolationMode.BILINEAR)
    img_gray   = rgb2gray(preprocess(all_images).permute(0, 2, 3, 1).cpu().numpy())
    recon_gray = rgb2gray(preprocess(all_recons).permute(0, 2, 3, 1).cpu().numpy())

    scores = []
    for im, rec in tqdm(zip(img_gray, recon_gray), total=len(all_images), desc="SSIM"):
        scores.append(
            ssim_fn(rec, im, multichannel=True,
                    gaussian_weights=True, sigma=1.5,
                    use_sample_covariance=False, data_range=1.0)
        )
    return float(np.mean(scores))


def compute_alexnet(all_images: torch.Tensor, all_recons: torch.Tensor, device: str) -> tuple:
    """AlexNet 2-way ID for layers features.4 and features.11."""
    from torchvision.models import alexnet, AlexNet_Weights

    weights = AlexNet_Weights.IMAGENET1K_V1
    model = create_feature_extractor(
        alexnet(weights=weights),
        return_nodes=["features.4", "features.11"]
    ).to(device).eval().requires_grad_(False)

    preprocess = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    score2  = two_way_identification(all_recons.float(), all_images, model, preprocess, "features.4",  device)
    score5  = two_way_identification(all_recons.float(), all_images, model, preprocess, "features.11", device)
    return float(score2), float(score5)


def compute_inception(all_images: torch.Tensor, all_recons: torch.Tensor, device: str) -> float:
    """InceptionV3 2-way ID on avgpool layer."""
    from torchvision.models import inception_v3, Inception_V3_Weights

    weights = Inception_V3_Weights.DEFAULT
    model = create_feature_extractor(
        inception_v3(weights=weights),
        return_nodes=["avgpool"]
    ).to(device).eval().requires_grad_(False)

    preprocess = transforms.Compose([
        transforms.Resize(342, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    score = two_way_identification(all_recons, all_images, model, preprocess, "avgpool", device)
    return float(score)


def compute_clip(all_images: torch.Tensor, all_recons: torch.Tensor, device: str) -> float:
    """CLIP ViT-L/14 2-way ID on final image embedding."""
    import clip

    clip_model, _ = clip.load("ViT-L/14", device=device, download_root=CACHE_DIR)
    clip_model.eval().requires_grad_(False)

    preprocess = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                             std=[0.26862954, 0.26130258, 0.27577711]),
    ])

    score = two_way_identification(all_recons, all_images,
                                   clip_model.encode_image, preprocess, None, device)
    return float(score)


def compute_effnet(all_images: torch.Tensor, all_recons: torch.Tensor) -> float:
    """EfficientNet-B1 correlation distance (lower = better in original, but we report as-is)."""
    from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights

    weights = EfficientNet_B1_Weights.DEFAULT
    model = create_feature_extractor(
        efficientnet_b1(weights=weights),
        return_nodes=["avgpool"]
    ).eval().requires_grad_(False)

    preprocess = transforms.Compose([
        transforms.Resize(255, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    with torch.no_grad():
        gt_feat   = model(preprocess(all_images))["avgpool"].reshape(len(all_images), -1).numpy()
        fake_feat = model(preprocess(all_recons))["avgpool"].reshape(len(all_recons), -1).numpy()

    dist = np.array([
        sp.spatial.distance.correlation(gt_feat[i], fake_feat[i])
        for i in range(len(gt_feat))
    ]).mean()
    return float(dist)


def compute_swav(all_images: torch.Tensor, all_recons: torch.Tensor) -> float:
    """SwAV ResNet-50 correlation distance."""
    swav_model = torch.hub.load(
        "facebookresearch/swav:main", "resnet50",
        verbose=False
    )
    swav_model = create_feature_extractor(swav_model, return_nodes=["avgpool"])
    swav_model.eval().requires_grad_(False)

    preprocess = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    with torch.no_grad():
        gt_feat   = swav_model(preprocess(all_images))["avgpool"].reshape(len(all_images), -1).numpy()
        fake_feat = swav_model(preprocess(all_recons))["avgpool"].reshape(len(all_recons), -1).numpy()

    dist = np.array([
        sp.spatial.distance.correlation(gt_feat[i], fake_feat[i])
        for i in range(len(gt_feat))
    ]).mean()
    return float(dist)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="MindEye2 Image Evaluation (8 metrics)")
    parser.add_argument("--gt_dir",    type=str, required=True,
                        help="Directory containing ground-truth images")
    parser.add_argument("--recon_dir", type=str, required=True,
                        help="Directory containing reconstructed images")
    parser.add_argument("--output",    type=str, default="eval_results.csv",
                        help="Path to save the CSV results (default: eval_results.csv)")
    parser.add_argument("--imsize",    type=int, default=256,
                        help="Resize all images to this square size before evaluation (default: 256)")
    parser.add_argument("--device",    type=str, default=None,
                        help="Device: 'cuda', 'cpu', etc. Auto-detected if not set.")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Optional cache directory for downloaded model weights.")
    parser.add_argument("--subj",      type=str, default="01",
                        help="Subject identifier, e.g. 01, 02 ... (default: 01)")
    args = parser.parse_args()
    configure_cache(args.cache_dir)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Cache dir:    {CACHE_DIR}")

    # ── Load images ──────────────────────────────────────────────────────────
    print("\n[1/9] Loading images...")
    all_images = load_images_from_dir(args.gt_dir,    imsize=args.imsize)
    all_recons = load_images_from_dir(args.recon_dir, imsize=args.imsize)

    if len(all_images) != len(all_recons):
        raise ValueError(
            f"Number of GT images ({len(all_images)}) ≠ "
            f"number of reconstructions ({len(all_recons)})"
        )
    print(f"  Loaded {len(all_images)} image pairs  (shape: {all_images.shape})")

    results = {}

    # ── PixCorr ──────────────────────────────────────────────────────────────
    print("\n[2/9] PixCorr...")
    results["PixCorr"] = compute_pixcorr(all_images, all_recons)
    print(f"  PixCorr = {results['PixCorr']:.4f}")

    # ── SSIM ─────────────────────────────────────────────────────────────────
    print("\n[3/9] SSIM...")
    results["SSIM"] = compute_ssim(all_images, all_recons)
    print(f"  SSIM = {results['SSIM']:.4f}")

    # ── AlexNet ──────────────────────────────────────────────────────────────
    print("\n[4/9] AlexNet(2) & AlexNet(5)...")
    results["AlexNet(2)"], results["AlexNet(5)"] = compute_alexnet(all_images, all_recons, device)
    print(f"  AlexNet(2) = {results['AlexNet(2)']:.4f}")
    print(f"  AlexNet(5) = {results['AlexNet(5)']:.4f}")

    # ── InceptionV3 ───────────────────────────────────────────────────────────
    print("\n[5/9] InceptionV3...")
    results["InceptionV3"] = compute_inception(all_images, all_recons, device)
    print(f"  InceptionV3 = {results['InceptionV3']:.4f}")

    # ── CLIP ──────────────────────────────────────────────────────────────────
    print("\n[6/9] CLIP ViT-L/14...")
    results["CLIP"] = compute_clip(all_images, all_recons, device)
    print(f"  CLIP = {results['CLIP']:.4f}")

    # ── EfficientNet-B1 ───────────────────────────────────────────────────────
    print("\n[7/9] EffNet-B (distance, lower = more similar)...")
    results["EffNet-B"] = compute_effnet(all_images, all_recons)
    print(f"  EffNet-B = {results['EffNet-B']:.4f}")

    # ── SwAV ──────────────────────────────────────────────────────────────────
    print("\n[8/9] SwAV (distance, lower = more similar)...")
    results["SwAV"] = compute_swav(all_images, all_recons)
    print(f"  SwAV = {results['SwAV']:.4f}")

    # ── Save & print ──────────────────────────────────────────────────────────
    print("\n[9/9] Saving results...")
    df = pd.DataFrame(list(results.items()), columns=["Metric", "Value"])
    df.to_csv(args.output, index=False)

    metrics_order = ["PixCorr", "SSIM", "AlexNet(2)", "AlexNet(5)",
                     "InceptionV3", "CLIP", "EffNet-B", "SwAV"]
    values_row = [results[m] for m in metrics_order]
 
    col_width = 12
    header_row  = "|" + "|".join(f"{m:^{col_width}}" for m in metrics_order) + "|"
    value_row   = "|" + "|".join(f"{v:^{col_width}.4f}" for v in values_row) + "|"
    total_width = len(header_row)
 
    title = f"Evaluation Results -- Subj{args.subj}"
    print("\n" + "=" * total_width)
    print(title.center(total_width))
    print("=" * total_width)
    print(header_row)
    print(value_row)
    print("=" * total_width)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
