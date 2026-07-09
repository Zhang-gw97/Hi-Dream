# train_controlnet_nsd.py
import os, math, argparse, csv
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
from torch.optim import AdamW
from tqdm.auto import tqdm
from roi_xattn import ROIChannelCrossAttention

from dataset_nsd_cond import NSDCondDataset
from kshot_utils import make_kshot_subset

import torch.multiprocessing as mp

mp.set_start_method("spawn", force=True)

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
    ControlNetModel,
)
from transformers import CLIPTextModel, CLIPTokenizer


# ----------------- 配置 -----------------
def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="Path to the NSD root directory containing nsddata/, nsddata_betas/, and nsddata_stimuli/.")
    ap.add_argument("--subj", default="subj02")
    ap.add_argument("--subjs", type=str, default=None,
                    help="comma-separated subject list, e.g., subj02,subj05,subj07. If set, overrides --subj.")

    ap.add_argument("--space", default="func1pt8mm")
    ap.add_argument("--cache_dir", default=None,
                    help="Optional Hugging Face/cache directory. If unset, library defaults are used.")
    ap.add_argument("--out_dir", default="runs",
                    help="Directory for checkpoints and training logs.")
    ap.add_argument("--cond_dir", type=str, default=None,  # <== 新增：外部指定condmaps目录
                    help="path to condmaps (will be scanned recursively)")
    ap.add_argument("--sd_id", default="runwayml/stable-diffusion-v1-5")
    ap.add_argument("--ctrl_id", default="lllyasviel/sd-controlnet-canny")

    ap.add_argument("--h5_key", type=str, default=None)  # e.g. imgBrick
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--num_workers", type=int, default=0)  # 先用 0，稳定
    ap.add_argument("--pin_memory", action="store_true", default=False)

    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max_steps", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=None,  # <== 新增：按epoch训练可选
                    help="if set, overrides max_steps = ceil(N/batch)*epochs")
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mixed_precision", action="store_true", default=True)

    ap.add_argument("--kshot", type=int, default=0)  # 0 -> 不做kshot
    ap.add_argument("--shared1000", type=str, default=None)
    ap.add_argument("--kshot_seed", type=int, default=0)
    ap.add_argument("--kshot_mode", type=str, default="all_repeats", choices=["one_per_kid", "all_repeats"])
    ap.add_argument("--kshot_max_rep", type=int, default=3)  # all_repeats时限制每图repeat数；不想限制就设0

    ap.add_argument("--init_model_dir", type=str, default=None,
                    help="path to base LOSO model dir (contains controlnet_finetuned/ and roi_mixer.pth). "
                         "If set, will load weights from there instead of ctrl_id.")
    ap.add_argument("--strict_load", action="store_true", default=False,
                    help="strictly load roi_mixer state dict")

    ap.add_argument("--warmup_only_mixer_steps", type=int, default=0)
    ap.add_argument("--roi_mixer_lr", type=float, default=5e-5)

    # Explicit shallow / mid / deep injection controls
    ap.add_argument("--depth_scales", type=str, default="1.2,0.5,0.3",
                    help="Explicit scales for shallow,mid,deep ControlNet residual injection")
    ap.add_argument("--early_edge_boost", type=float, default=0.20)
    ap.add_argument("--early_edge_blur_radius", type=int, default=2)

    # Low-level reconstruction losses
    ap.add_argument("--recon_loss_weight", type=float, default=0.10)
    ap.add_argument("--grad_loss_weight", type=float, default=0.05)
    ap.add_argument("--ssim_loss_weight", type=float, default=0.02)
    ap.add_argument("--recon_loss_tmax", type=float, default=0.35,
                    help="Enable image-space low-level losses only when normalized timestep <= this threshold")
    ap.add_argument("--log_every", type=int, default=10,
                    help="Print loss breakdown every N optimization steps and append to CSV")

    return ap.parse_args()


# ----------------- 训练工具 -----------------
def set_seed(seed=42):
    import random
    import numpy as np
    import torch
    random.seed(seed);
    np.random.seed(seed);
    torch.manual_seed(seed);
    torch.cuda.manual_seed_all(seed)


def prepare_text_embed(tokenizer, text_encoder, prompts):
    tokens = tokenizer(
        prompts, padding="max_length", truncation=True,
        max_length=tokenizer.model_max_length, return_tensors="pt"
    )
    with torch.no_grad():
        enc = text_encoder(tokens.input_ids.to(text_encoder.device))[0]
    return enc


def parse_depth_scales(s: str):
    vals = [float(x.strip()) for x in s.split(",") if x.strip()]
    if len(vals) != 3:
        raise ValueError(f"--depth_scales expects 3 comma-separated floats, got: {s}")
    return vals[0], vals[1], vals[2]


def scale_controlnet_residuals(down_block_res, mid_block_res, scales):
    """Explicitly scale shallow / mid / deep ControlNet residuals before UNet injection."""
    shallow_s, mid_s, deep_s = scales
    n = len(down_block_res)
    if n == 0:
        return down_block_res, mid_block_res

    # split by depth index into thirds: shallow / mid / deep
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


def decode_pred_x0(vae, noisy_latents, noise_pred, timesteps, scheduler):
    alphas_cumprod = scheduler.alphas_cumprod.to(device=noisy_latents.device, dtype=noisy_latents.dtype)
    a_t = alphas_cumprod[timesteps].view(-1, 1, 1, 1)
    sqrt_a = a_t.sqrt()
    sqrt_one_minus_a = (1.0 - a_t).sqrt()
    pred_x0 = (noisy_latents - sqrt_one_minus_a * noise_pred) / torch.clamp(sqrt_a, min=1e-6)
    pred_x0 = pred_x0 / 0.18215
    img = vae.decode(pred_x0).sample
    return img.clamp(-1.0, 1.0)


def gradient_map(x):
    # x: [B,C,H,W] in [-1,1]
    gray = x.mean(dim=1, keepdim=True)
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=x.device, dtype=x.dtype).view(1,1,3,3)
    ky = torch.tensor([[-1,-2,-1], [ 0, 0, 0], [ 1, 2, 1]], device=x.device, dtype=x.dtype).view(1,1,3,3)
    gx = torch.nn.functional.conv2d(gray, kx, padding=1)
    gy = torch.nn.functional.conv2d(gray, ky, padding=1)
    return gx, gy


def simple_ssim_loss(x, y, c1=0.01**2, c2=0.03**2):
    # simple global SSIM-like loss on images in [-1,1]
    mu_x = x.mean(dim=(2,3), keepdim=True)
    mu_y = y.mean(dim=(2,3), keepdim=True)
    sigma_x = ((x - mu_x) ** 2).mean(dim=(2,3), keepdim=True)
    sigma_y = ((y - mu_y) ** 2).mean(dim=(2,3), keepdim=True)
    sigma_xy = ((x - mu_x) * (y - mu_y)).mean(dim=(2,3), keepdim=True)
    ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2) + 1e-8)
    return 1.0 - ssim.mean()


def _worker_init_fn(worker_id):
    info = torch.utils.data.get_worker_info()
    ds = info.dataset
    # 保险起见，强制让每个 worker 在自己的进程里重新打开 HDF5
    if hasattr(ds, "h5"):
        ds.h5 = None
        ds.images = None


def main():
    args = get_args()
    set_seed(args.seed)

    if args.cache_dir:
        os.environ.setdefault("HF_HOME", args.cache_dir)
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", args.cache_dir)
        os.environ.setdefault("TRANSFORMERS_CACHE", args.cache_dir)
        os.environ.setdefault("DIFFUSERS_CACHE", args.cache_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using Device: {device}')
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    depth_scales = parse_depth_scales(args.depth_scales)
    print(f"[cfg] explicit depth scales shallow/mid/deep = {depth_scales}")

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # Dataset / Loader
    # Dataset / Loader (single-subject or multi-subject)
    if args.subjs is not None and len(args.subjs.strip()) > 0:
        subj_list = [s.strip() for s in args.subjs.split(",") if len(s.strip()) > 0]
    else:
        subj_list = [args.subj]

    datasets = []
    for subj in subj_list:
        if args.cond_dir is not None:
            cond_dir = args.cond_dir
        else:
            cond_dir = str(Path(args.root) / f"nsddata/ppdata/{subj}/{args.space}/condmaps")

        ds_one = NSDCondDataset(
            root=args.root, subj=subj, space=args.space, cond_dir=cond_dir,
            early_edge_boost=args.early_edge_boost,
            early_edge_blur_radius=args.early_edge_blur_radius,
        )
        datasets.append(ds_one)

    dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    print(f"[data] using subjects={subj_list} | total N={len(dataset)}")

    if args.kshot > 0:
        assert args.shared1000 is not None, "--shared1000 is required for k-shot (to avoid leakage)"
        assert len(datasets) == 1, "k-shot must run on ONE holdout subject only. Use --subj subjXX (no --subjs)."

        max_rep = None if args.kshot_max_rep <= 0 else int(args.kshot_max_rep)
        dataset, meta = make_kshot_subset(
            datasets[0],
            k=args.kshot,
            shared1000_path=args.shared1000,
            seed=args.kshot_seed,
            mode=args.kshot_mode,
            max_items_per_kid=max_rep
        )
        print("[kshot]", meta)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,  # 先用 0，稳定
        pin_memory=args.pin_memory,  # 先关；以后需要再开
        persistent_workers=(args.num_workers > 0),
        worker_init_fn=_worker_init_fn if args.num_workers > 0 else None,
        drop_last=True,
    )

    # steps_per_epoch = math.ceil(len(dataset) / args.batch_size)
    # target_epochs   = 5  # 想训几轮就写几
    # args.max_steps  = steps_per_epoch * target_epochs
    # print(f"[info] N={len(dataset)}, batch={args.batch_size}, steps/epoch={steps_per_epoch}, max_steps={args.max_steps}")

    steps_per_epoch = math.ceil(len(dataset) / args.batch_size)
    if args.epochs is not None and args.epochs > 0:
        args.max_steps = steps_per_epoch * int(args.epochs)
    print(f"[info] N={len(dataset)}, batch={args.batch_size}, "
          f"steps/epoch={steps_per_epoch}, max_steps={args.max_steps}")

    # 模型组件
    tokenizer = CLIPTokenizer.from_pretrained(args.sd_id, subfolder="tokenizer",
                                              cache_dir=args.cache_dir, local_files_only=False)

    text_encoder = CLIPTextModel.from_pretrained(args.sd_id, subfolder="text_encoder",
                                                 cache_dir=args.cache_dir,
                                                 torch_dtype=torch.float16).to(device)

    vae = AutoencoderKL.from_pretrained(args.sd_id, subfolder="vae",
                                        cache_dir=args.cache_dir,
                                        torch_dtype=torch.float16).to(device)
    unet = UNet2DConditionModel.from_pretrained(args.sd_id, subfolder="unet",
                                                cache_dir=args.cache_dir,
                                                torch_dtype=torch.float16).to(device)

    # controlnet = ControlNetModel.from_pretrained(
    #     args.ctrl_id, cache_dir=args.cache_dir
    # ).to(device, dtype=torch.float32)
    # controlnet.train()

    # roi_mixer = ROIChannelCrossAttention(d_model=64, n_heads=4, dropout=0.0).to(device)
    # roi_mixer.train()
    # ---- ControlNet ----
    if args.init_model_dir is not None and len(args.init_model_dir) > 0:
        init_dir = Path(args.init_model_dir)
        ctrl_ckpt = init_dir / "controlnet_finetuned"
        assert ctrl_ckpt.exists(), f"[init] missing: {ctrl_ckpt}"
        print(f"[init] loading ControlNet from: {ctrl_ckpt}")
        controlnet = ControlNetModel.from_pretrained(
            str(ctrl_ckpt),
            cache_dir=args.cache_dir
        ).to(device, dtype=torch.float32)
    else:
        print(f"[init] loading ControlNet from ctrl_id: {args.ctrl_id}")
        controlnet = ControlNetModel.from_pretrained(
            args.ctrl_id,
            cache_dir=args.cache_dir
        ).to(device, dtype=torch.float32)

    controlnet.train()

    # ---- ROI mixer ----
    roi_mixer = ROIChannelCrossAttention(d_model=64, n_heads=4, dropout=0.0).to(device)
    if args.init_model_dir is not None and len(args.init_model_dir) > 0:
        roi_path = Path(args.init_model_dir) / "roi_mixer.pth"
        if roi_path.exists():
            print(f"[init] loading roi_mixer from: {roi_path}")
            state = torch.load(str(roi_path), map_location="cpu")
            # 兼容 dict / module 前缀
            if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
                state = {k.replace("module.", "", 1): v for k, v in state.items()}
            roi_mixer.load_state_dict(state, strict=args.strict_load)
        else:
            print(f"[WARN] roi_mixer.pth not found at {roi_path}, start from random init.")
    roi_mixer.train()

    # 冻结 VAE/UNet/TextEnc 已有
    for p in vae.parameters(): p.requires_grad = False
    for p in unet.parameters(): p.requires_grad = False
    for p in text_encoder.parameters(): p.requires_grad = False

    # ===== Warm-up 策略与优化器初始化（只在这里建一次！）=====
    warmup_steps = int(getattr(args, "warmup_only_mixer_steps", 0))
    if warmup_steps > 0:
        # 阶段1：只训 mixer
        for p in controlnet.parameters(): p.requires_grad = False
        for p in roi_mixer.parameters():  p.requires_grad = True
        optim = AdamW([
            {"params": roi_mixer.parameters(), "lr": args.roi_mixer_lr},
        ])
    else:
        # 无 warm-up：一开始就联合训练
        for p in controlnet.parameters(): p.requires_grad = True
        for p in roi_mixer.parameters():  p.requires_grad = True
        optim = AdamW([
            {"params": controlnet.parameters(), "lr": args.lr},
            {"params": roi_mixer.parameters(), "lr": args.roi_mixer_lr},
        ])

    controlnet.train()

    # 调度器
    noise_scheduler = DDPMScheduler.from_pretrained(args.sd_id, subfolder="scheduler", cache_dir=args.cache_dir)

    scaler = torch.cuda.amp.GradScaler(enabled=args.mixed_precision)

    global_step = 0
    losses = []

    csv_path = Path(args.out_dir) / "train_losses.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_fieldnames = [
        "step", "total_loss", "diff_loss", "recon_l1_loss",
        "gradient_loss", "ssim_loss", "weighted_recon",
        "weighted_gradient", "weighted_ssim", "norm_t_mean"
    ]
    with open(csv_path, "w", newline="") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=csv_fieldnames)
        writer.writeheader()

    pbar = tqdm(total=args.max_steps, desc="train")
    while global_step < args.max_steps:
        for batch in loader:
            pixel_values = batch["pixel_values"].to(device, dtype=torch.float16)  # 给 VAE/UNet
            cond_values = batch["cond_values"].to(device, dtype=torch.float32)
            with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=torch.float16, enabled=False):
                cond_values = roi_mixer(cond_values).clamp(0.0, 1.0)

            prompts = batch.get("prompt", None)
            if prompts is None:
                prompts = [""] * pixel_values.size(0)

            with torch.no_grad():
                encoder_hidden_states = prepare_text_embed(tokenizer, text_encoder, prompts)
                latents = vae.encode(pixel_values).latent_dist.sample() * 0.18215
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                                          (latents.size(0),), device=device, dtype=torch.long)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            step_diff_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
            step_recon_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
            step_grad_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
            step_ssim_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
            step_norm_t_mean = torch.tensor(0.0, device=device, dtype=torch.float32)

            with torch.cuda.amp.autocast(enabled=args.mixed_precision, dtype=torch.float16):
                down_block_res, mid_block_res = controlnet(
                    sample=noisy_latents.to(torch.float32),
                    timestep=timesteps,
                    encoder_hidden_states=encoder_hidden_states.to(torch.float32),
                    controlnet_cond=cond_values,
                    conditioning_scale=1.0,
                    return_dict=False,
                )
                down_block_res, mid_block_res = scale_controlnet_residuals(down_block_res, mid_block_res, depth_scales)
                noise_pred = unet(
                    sample=noisy_latents,
                    timestep=timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=down_block_res,
                    mid_block_additional_residual=mid_block_res,
                ).sample

                diff_loss = torch.nn.functional.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
                step_diff_loss = diff_loss.detach().float()

                norm_t = timesteps.float() / float(noise_scheduler.config.num_train_timesteps - 1)
                step_norm_t_mean = norm_t.mean().detach().float()
                low_mask = (norm_t <= args.recon_loss_tmax).float().view(-1, 1, 1, 1)
                if low_mask.sum() > 0 and (args.recon_loss_weight > 0 or args.grad_loss_weight > 0 or args.ssim_loss_weight > 0):
                    pred_img = decode_pred_x0(vae, noisy_latents, noise_pred, timesteps, noise_scheduler)
                    gt_img = pixel_values
                    denom = low_mask.mean().clamp(min=1e-6)

                    recon_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
                    grad_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
                    ssim_loss = torch.tensor(0.0, device=device, dtype=torch.float32)

                    if args.recon_loss_weight > 0:
                        recon_loss = ((pred_img.float() - gt_img.float()).abs() * low_mask).mean() / denom
                    if args.grad_loss_weight > 0:
                        pgx, pgy = gradient_map(pred_img.float())
                        ggx, ggy = gradient_map(gt_img.float())
                        grad_loss = (((pgx - ggx).abs() + (pgy - ggy).abs()) * low_mask).mean() / denom
                    if args.ssim_loss_weight > 0:
                        ssim_loss = simple_ssim_loss(pred_img.float() * low_mask, gt_img.float() * low_mask)

                    loss = diff_loss \
                        + args.recon_loss_weight * recon_loss \
                        + args.grad_loss_weight * grad_loss \
                        + args.ssim_loss_weight * ssim_loss
                else:
                    loss = diff_loss

                step_recon_loss = recon_loss.detach().float() if 'recon_loss' in locals() else torch.tensor(0.0, device=device)
                step_grad_loss = grad_loss.detach().float() if 'grad_loss' in locals() else torch.tensor(0.0, device=device)
                step_ssim_loss = ssim_loss.detach().float() if 'ssim_loss' in locals() else torch.tensor(0.0, device=device)

            scaler.scale(loss / args.grad_accum).backward()
            if (global_step + 1) % args.grad_accum == 0:
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)

            global_step += 1

            # ★ 到达阈值：从“只训 mixer”切到“联合训练”
            if warmup_steps > 0 and global_step == warmup_steps:
                optim.zero_grad(set_to_none=True)
                for p in controlnet.parameters(): p.requires_grad = True
                for p in roi_mixer.parameters():  p.requires_grad = True
                optim = AdamW([
                    {"params": controlnet.parameters(), "lr": args.lr},
                    {"params": roi_mixer.parameters(), "lr": args.roi_mixer_lr},
                ])
                print(f"[SWITCH] warm-up finished @ step {global_step}: now joint training.")

            losses.append(loss.item())
            pbar.set_postfix(loss=f"{np.mean(losses[-50:]):.4f}")
            pbar.update(1)

            if global_step % args.log_every == 0:
                log_row = {
                    "step": int(global_step),
                    "total_loss": float(loss.detach().item()),
                    "diff_loss": float(step_diff_loss.item()),
                    "recon_l1_loss": float(step_recon_loss.item()),
                    "gradient_loss": float(step_grad_loss.item()),
                    "ssim_loss": float(step_ssim_loss.item()),
                    "weighted_recon": float(args.recon_loss_weight * step_recon_loss.item()),
                    "weighted_gradient": float(args.grad_loss_weight * step_grad_loss.item()),
                    "weighted_ssim": float(args.ssim_loss_weight * step_ssim_loss.item()),
                    "norm_t_mean": float(step_norm_t_mean.item()),
                }
                print(
                    f"[step {global_step}] total={log_row['total_loss']:.6f} "
                    f"diff={log_row['diff_loss']:.6f} "
                    f"l1={log_row['recon_l1_loss']:.6f} "
                    f"grad={log_row['gradient_loss']:.6f} "
                    f"ssim={log_row['ssim_loss']:.6f} "
                    f"w_l1={log_row['weighted_recon']:.6f} "
                    f"w_grad={log_row['weighted_gradient']:.6f} "
                    f"w_ssim={log_row['weighted_ssim']:.6f} "
                    f"tmean={log_row['norm_t_mean']:.4f}"
                )
                with open(csv_path, "a", newline="") as fcsv:
                    writer = csv.DictWriter(fcsv, fieldnames=csv_fieldnames)
                    writer.writerow(log_row)

            if global_step >= args.max_steps:
                break

    # 保存 ControlNet 权重
    save_dir = Path(args.out_dir) / "controlnet_finetuned"
    with open(Path(args.out_dir) / "depth_scale_config.json", "w") as f:
        import json
        json.dump({
            "depth_scales": depth_scales,
            "early_edge_boost": args.early_edge_boost,
            "early_edge_blur_radius": args.early_edge_blur_radius,
            "recon_loss_weight": args.recon_loss_weight,
            "grad_loss_weight": args.grad_loss_weight,
            "ssim_loss_weight": args.ssim_loss_weight,
            "recon_loss_tmax": args.recon_loss_tmax,
        }, f, indent=2)
    save_dir.mkdir(parents=True, exist_ok=True)
    controlnet.save_pretrained(str(save_dir))
    torch.save(roi_mixer.state_dict(), str(Path(args.out_dir) / "roi_mixer.pth"))
    print("[DONE] saved:", save_dir, "and roi_mixer.pth")


if __name__ == "__main__":
    main()
