import argparse
from pathlib import Path
import re

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def load_rgb(path: Path):
    if not path.exists():
        return None
    return mpimg.imread(str(path))


def list_step_dirs(sample_dir: Path):
    step_dirs = []
    for p in sample_dir.glob("step_*"):
        if p.is_dir():
            m = re.match(r"step_(\d+)", p.name)
            if m:
                step_dirs.append((int(m.group(1)), p))
    step_dirs.sort(key=lambda x: x[0])
    return step_dirs


def resolve_step_dirs(sample_dir: Path, steps_str: str = None):
    step_dirs = list_step_dirs(sample_dir)
    if steps_str:
        wanted = [int(x.strip()) for x in steps_str.split(",") if x.strip()]
        wanted_set = set(wanted)
        step_dirs = [(s, p) for (s, p) in step_dirs if s in wanted_set]
    return step_dirs


def show_img(ax, img, title=None):
    if img is None:
        ax.axis("off")
        return
    ax.imshow(img)
    ax.axis("off")
    if title is not None:
        ax.set_title(title, fontsize=12)


def pick_final_image(sample_dir: Path):
    final_predx0 = sample_dir / "final" / "pred_x0.png"
    final_recon = sample_dir / "final" / "recon.png"
    img = load_rgb(final_predx0)
    if img is None:
        img = load_rgb(final_recon)
    return img


def main():
    ap = argparse.ArgumentParser(description="Build Hi-DREAM progress figure from saved inference steps")
    ap.add_argument("--sample_dir", required=True,
                    help="One sample directory under progress/, e.g. progress/kid00012_sess01_idx0005")
    ap.add_argument("--output", required=True, help="Output figure path, e.g. progress_figure.png")
    ap.add_argument("--steps", default=None,
                    help="Optional comma-separated subset of steps to plot (e.g. 1,10,20,30,40,50)")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    sample_dir = Path(args.sample_dir)
    if not sample_dir.exists():
        raise FileNotFoundError(f"sample_dir does not exist: {sample_dir}")

    step_dirs = resolve_step_dirs(sample_dir, args.steps)
    if len(step_dirs) == 0:
        raise FileNotFoundError(f"No step_* directories found under {sample_dir}")

    gt_path = sample_dir / "gt.png"
    final_img = pick_final_image(sample_dir)

    row_specs = [
        ("Pred x0", "pred_x0.png"),
        ("Early contribution", "shallow_overlay.png"),
        ("Middle contribution", "mid_overlay.png"),
        ("Late contribution", "deep_overlay.png"),
    ]

    ncols = len(step_dirs) + 1  # final column
    col_titles = [f"step={s}" for s, _ in step_dirs] + ["final"]

    fig_w = max(12, 2.6 * ncols)
    fig_h = 3.2 + 2.5 * len(row_specs)
    fig = plt.figure(figsize=(fig_w, fig_h))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 4.2], hspace=0.18)

    # Top panel: GT vs Final
    gt_img = load_rgb(gt_path)
    top_cols = 2 if gt_img is not None and final_img is not None else 1
    top = outer[0].subgridspec(1, top_cols, wspace=0.06)

    if gt_img is not None and final_img is not None:
        ax0 = fig.add_subplot(top[0, 0])
        show_img(ax0, gt_img, title="Ground Truth")
        ax1 = fig.add_subplot(top[0, 1])
        show_img(ax1, final_img, title="Final Reconstruction")
    elif final_img is not None:
        ax1 = fig.add_subplot(top[0, 0])
        show_img(ax1, final_img, title="Final Reconstruction")

    # Bottom grid
    bottom = outer[1].subgridspec(len(row_specs), ncols, wspace=0.03, hspace=0.08)

    for r, (row_name, fname) in enumerate(row_specs):
        for c, (step_num, step_dir) in enumerate(step_dirs):
            ax = fig.add_subplot(bottom[r, c])
            img = load_rgb(step_dir / fname)
            show_img(ax, img, title=col_titles[c] if r == 0 else None)
            if c == 0:
                ax.set_ylabel(row_name, rotation=0, labelpad=55, va="center", fontsize=12)

        ax = fig.add_subplot(bottom[r, ncols - 1])
        if r == 0:
            img = load_rgb(sample_dir / "final" / "pred_x0.png")
            if img is None:
                img = load_rgb(sample_dir / "final" / "recon.png")
        else:
            img = load_rgb(sample_dir / "final" / fname)
        show_img(ax, img, title=col_titles[-1] if r == 0 else None)
        if r == 0:
            ax.set_ylabel(row_name, rotation=0, labelpad=55, va="center", fontsize=12)

    fig.suptitle(sample_dir.name, fontsize=14, y=0.995)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved figure to {args.output}")


if __name__ == "__main__":
    main()
