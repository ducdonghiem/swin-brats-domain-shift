"""
visualize.py — Visualize SwinBraTS predictions on random test samples.

Usage:
    python visualize.py                          # 3 random samples, best_model.pth
    python visualize.py --n 5                    # 5 random samples
    python visualize.py --n 1 --seed 42          # reproducible pick
    python visualize.py --checkpoint path/to/checkpoint.pth
    python visualize.py --case BraTS2021_00042   # specific patient ID

Output:
    results/viz/BraTS2021_00042_slice077.png  — one file per sample
    Each file shows: FLAIR | T1ce | Ground Truth | Prediction | Overlay
    with one representative slice chosen automatically (most tumor content).
"""

import argparse
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch

# ── Path setup ────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.data_loader import MRIDataset
from src.models.swinBraTS_full import SwinBraTS
from src.utils.config import load_config

# ── Label definitions ─────────────────────────────────────────────────────────
# BraTS label values in the ground truth mask
LABEL_NAMES = {0: "Background", 1: "NCR/NET", 2: "Edema", 4: "Enhancing"}

# Colors for each class in the overlay (RGBA, values 0-1)
# Background is transparent; tumor regions use distinct colours
LABEL_COLORS = {
    0: (0.00, 0.00, 0.00, 0.00),  # background — transparent
    1: (0.20, 0.60, 1.00, 0.55),  # NCR/NET — blue
    2: (0.20, 0.80, 0.20, 0.55),  # Edema — green
    4: (1.00, 0.25, 0.25, 0.55),  # Enhancing — red
}

# BraTS subregion definitions used in the metric legend
REGION_COLORS = {
    "Whole Tumour (WT)": "#4caf50",   # all non-background
    "Tumour Core (TC)":  "#2196f3",   # labels 1 + 4
    "Enhancing (ET)":    "#f44336",   # label 4 only
}


def build_colormap(mask: np.ndarray) -> np.ndarray:
    """
    Convert an integer label mask (H, W) with values {0,1,2,4}
    to an RGBA image (H, W, 4) using LABEL_COLORS.
    """
    rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
    for label, color in LABEL_COLORS.items():
        rgba[mask == label] = color
    return rgba


def pick_best_slice(mask_3d: np.ndarray) -> int:
    """
    Return the depth index with the most foreground (tumor) voxels.
    mask_3d: (D, H, W) integer array.
    """
    tumor_per_slice = (mask_3d > 0).sum(axis=(1, 2))
    return int(np.argmax(tumor_per_slice))


def run_inference(model, modalities, device, use_amp):
    """
    Run model inference on a single sample.

    Args:
        model: SwinBraTS model (eval mode)
        modalities: tuple of 4 tensors, each (D, H, W)
        device: torch device
        use_amp: bool

    Returns:
        pred_mask: (D, H, W) int64 numpy array with values {0,1,2,4}
        logits:    (4, D, H, W) float32 numpy array
    """
    # Add batch dim: (D, H, W) -> (1, D, H, W)
    inputs = [m.unsqueeze(0).to(device) for m in modalities]

    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=(use_amp and device.type == 'cuda')):
            logits = model(inputs)  # (1, 4, D, H, W)

    logits_np = logits.squeeze(0).cpu().float().numpy()  # (4, D, H, W)

    # Argmax over class dim -> {0,1,2,3}, remap 3->4 (BraTS label convention)
    pred = np.argmax(logits_np, axis=0).astype(np.int64)  # (D, H, W)
    pred[pred == 3] = 4

    return pred, logits_np


def save_figure(
    case_id: str,
    modalities_np: list,       # list of 4 (D, H, W) float32 arrays
    gt_mask: np.ndarray,       # (D, H, W) int64
    pred_mask: np.ndarray,     # (D, H, W) int64
    slice_idx: int,
    output_path: Path,
    modality_names: list,
):
    """
    Save a single visualization figure with 5 panels:
        FLAIR | T1ce | Ground Truth | Prediction | Overlay (on FLAIR)
    """

    def norm(vol_slice):
        """Normalize a 2D slice to [0, 1] for display."""
        v = vol_slice.astype(np.float32)
        lo, hi = v.min(), v.max()
        if hi - lo < 1e-6:
            return np.zeros_like(v)
        return (v - lo) / (hi - lo)

    # Pick FLAIR (index 0) and T1ce (index 2) for the MRI panels
    flair_slice = norm(modalities_np[0][slice_idx])
    t1ce_slice  = norm(modalities_np[2][slice_idx])
    gt_slice    = gt_mask[slice_idx]
    pred_slice  = pred_mask[slice_idx]

    gt_rgba   = build_colormap(gt_slice)
    pred_rgba = build_colormap(pred_slice)

    # Overlay: FLAIR as greyscale background + predicted RGBA on top
    flair_rgb = np.stack([flair_slice] * 3, axis=-1)

    # Dice scores for this slice (quick sanity numbers in the title)
    def slice_dice(gt, pred, label):
        g = (gt == label).astype(float)
        p = (pred == label).astype(float)
        denom = g.sum() + p.sum()
        return 2 * (g * p).sum() / denom if denom > 0 else float('nan')

    # BraTS regions for this slice
    gt_wt   = (gt_slice > 0).astype(float)
    pred_wt = (pred_slice > 0).astype(float)
    denom   = gt_wt.sum() + pred_wt.sum()
    dice_wt = 2 * (gt_wt * pred_wt).sum() / denom if denom > 0 else float('nan')

    gt_tc   = ((gt_slice == 1) | (gt_slice == 4)).astype(float)
    pred_tc = ((pred_slice == 1) | (pred_slice == 4)).astype(float)
    denom   = gt_tc.sum() + pred_tc.sum()
    dice_tc = 2 * (gt_tc * pred_tc).sum() / denom if denom > 0 else float('nan')

    gt_et   = (gt_slice == 4).astype(float)
    pred_et = (pred_slice == 4).astype(float)
    denom   = gt_et.sum() + pred_et.sum()
    dice_et = 2 * (gt_et * pred_et).sum() / denom if denom > 0 else float('nan')

    # ── Figure layout ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    fig.patch.set_facecolor('#1a1a2e')

    panel_titles = ['FLAIR', 'T1ce', 'Ground Truth', 'Prediction', 'Overlay']
    images = [flair_slice, t1ce_slice, gt_rgba, pred_rgba, None]

    for ax, title in zip(axes, panel_titles):
        ax.set_facecolor('#1a1a2e')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor('#444466')
        ax.set_title(title, color='#ccccee', fontsize=11, fontweight='bold', pad=6)

    axes[0].imshow(images[0], cmap='gray', vmin=0, vmax=1)
    axes[1].imshow(images[1], cmap='gray', vmin=0, vmax=1)
    axes[2].imshow(images[2])
    axes[3].imshow(images[3])

    # Overlay panel: greyscale FLAIR + coloured prediction
    axes[4].imshow(flair_rgb, vmin=0, vmax=1)
    axes[4].imshow(pred_rgba)

    # ── Legend for tumour classes ──────────────────────────────────────────────
    legend_patches = [
        mpatches.Patch(color=LABEL_COLORS[1][:3], label='NCR/NET (label 1)'),
        mpatches.Patch(color=LABEL_COLORS[2][:3], label='Edema (label 2)'),
        mpatches.Patch(color=LABEL_COLORS[4][:3], label='Enhancing (label 4)'),
    ]
    axes[4].legend(
        handles=legend_patches,
        loc='lower right',
        fontsize=7,
        framealpha=0.5,
        facecolor='#1a1a2e',
        labelcolor='white',
    )

    # ── Main title with per-slice Dice ─────────────────────────────────────────
    def fmt(v):
        return f"{v:.3f}" if not np.isnan(v) else "n/a"

    fig.suptitle(
        f"{case_id}  —  slice {slice_idx:03d}\n"
        f"Slice Dice:  WT {fmt(dice_wt)}  |  TC {fmt(dice_tc)}  |  ET {fmt(dice_et)}",
        color='#eeeeff',
        fontsize=12,
        fontweight='bold',
        y=1.02,
    )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved → {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize SwinBraTS predictions")
    parser.add_argument('--config',     default='configs/train_config.yml')
    parser.add_argument('--checkpoint', default=None,
                        help="Path to .pth checkpoint. Defaults to checkpoint_dir/best_model.pth")
    parser.add_argument('--n',          type=int, default=3,
                        help="Number of random test samples to visualize")
    parser.add_argument('--seed',       type=int, default=None,
                        help="Random seed for reproducible sample selection")
    parser.add_argument('--case',       default=None,
                        help="Specific patient ID to visualize (overrides --n)")
    parser.add_argument('--out_dir',    default=None,
                        help="Output directory. Defaults to results/viz/")
    args = parser.parse_args()

    # ── Config ────────────────────────────────────────────────────────────────
    config = load_config(args.config)

    device_str = config.get('device', 'cuda')
    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
    use_amp = config['training'].get('use_amp', True)

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else \
        Path(config['training']['checkpoint_dir']) / 'best_model.pth'

    out_dir = Path(args.out_dir) if args.out_dir else \
        Path(config['training']['results_dir']) / 'viz'

    # ── Dataset ───────────────────────────────────────────────────────────────
    test_dir = Path(config['data']['test_dir'])
    modality_order = config['data']['modality_order']

    dataset = MRIDataset(
        data_dir=test_dir,
        modalities=modality_order,
        transforms=None,
    )

    if len(dataset) == 0:
        print(f"No samples found in {test_dir}")
        sys.exit(1)

    # ── Select cases ──────────────────────────────────────────────────────────
    if args.case:
        # Find the index of the requested patient ID
        case_ids = [Path(c[0]).parent.name for c in dataset.cases]
        if args.case not in case_ids:
            print(f"Case '{args.case}' not found. Available: {case_ids[:5]} ...")
            sys.exit(1)
        indices = [case_ids.index(args.case)]
    else:
        if args.seed is not None:
            random.seed(args.seed)
        n = min(args.n, len(dataset))
        indices = random.sample(range(len(dataset)), n)

    print(f"Visualizing {len(indices)} sample(s) from {test_dir}")

    # ── Model ─────────────────────────────────────────────────────────────────
    print(f"Loading model from {checkpoint_path}")
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    model = SwinBraTS(
        in_channels=config['model']['in_channels'],
        num_classes=config['model']['num_classes'],
        embed_dim=config['model']['embed_dim'],
        window_size=config['model']['window_size'],
        patch_size=config['model']['patch_size'],
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    val_dice = checkpoint.get('best_metric', None)
    epoch    = checkpoint.get('epoch', None)
    if val_dice is not None:
        print(f"  Checkpoint: epoch {epoch}, best val Dice {val_dice:.4f}")

    # ── Inference + visualization ─────────────────────────────────────────────
    for idx in indices:
        modalities_tuple, mask = dataset[idx]
        case_id = Path(dataset.cases[idx][0]).parent.name

        print(f"\nProcessing {case_id} ...")

        # modalities_tuple: 4 × (D, H, W) tensors
        modalities_list = list(modalities_tuple)
        modalities_np   = [m.numpy() for m in modalities_list]
        gt_mask         = mask.numpy().astype(np.int64)  # (D, H, W)

        # Pick the most informative slice
        slice_idx = pick_best_slice(gt_mask)
        print(f"  Best slice: {slice_idx} "
              f"({int((gt_mask[slice_idx] > 0).sum())} tumor voxels)")

        # Run model
        pred_mask, _ = run_inference(model, modalities_list, device, use_amp)

        # Count predicted vs GT tumor voxels as a quick sanity check
        gt_tumor   = int((gt_mask > 0).sum())
        pred_tumor = int((pred_mask > 0).sum())
        print(f"  GT tumor voxels: {gt_tumor:,}  |  Predicted: {pred_tumor:,}")

        # Save figure
        out_path = out_dir / f"{case_id}_slice{slice_idx:03d}.png"
        save_figure(
            case_id=case_id,
            modalities_np=modalities_np,
            gt_mask=gt_mask,
            pred_mask=pred_mask,
            slice_idx=slice_idx,
            output_path=out_path,
            modality_names=modality_order,
        )

    print(f"\nDone. All figures saved to {out_dir}/")


if __name__ == '__main__':
    main()