"""
visualize_extremes.py — Find and visualise the best and worst predicted
samples from the test set according to volumetric Mean Dice.

Runs inference over the entire test set, ranks every sample by Mean Dice
(average of WT/TC/ET Dice), then saves a side-by-side comparison figure
showing the best and worst cases on a single page.

Usage:
    python visualize_extremes.py                          # default config + best_model.pth
    python visualize_extremes.py --metric mean_dice       # default ranking metric
    python visualize_extremes.py --metric dice_et         # rank by ET Dice only
    python visualize_extremes.py --metric dice_wt
    python visualize_extremes.py --metric dice_tc
    python visualize_extremes.py --checkpoint path/to/checkpoint.pth
    python visualize_extremes.py --out_dir results/extremes/

Output:
    results/viz/extremes_mean_dice.png
        A figure with 2 rows (best / worst) × 5 columns
        (FLAIR | T1ce | Ground Truth | Prediction | Overlay)
    results/viz/extremes_scores.txt
        Full ranked list of all test samples and their Dice scores
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
from tqdm import tqdm

# ── Path setup ────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.data_loader import MRIDataset
from src.models.swinBraTS_full import SwinBraTS
from src.utils.config import load_config

# ── Label / colour definitions (shared with visualize.py) ─────────────────────
LABEL_COLORS = {
    0: (0.00, 0.00, 0.00, 0.00),   # background — transparent
    1: (0.20, 0.60, 1.00, 0.55),   # NCR/NET — blue
    2: (0.20, 0.80, 0.20, 0.55),   # Edema — green
    4: (1.00, 0.25, 0.25, 0.55),   # Enhancing — red
}

METRIC_CHOICES = ['mean_dice', 'dice_wt', 'dice_tc', 'dice_et']


# ── Utilities ─────────────────────────────────────────────────────────────────

def build_colormap(mask: np.ndarray) -> np.ndarray:
    """Integer label mask (H, W) → RGBA image (H, W, 4)."""
    rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
    for label, color in LABEL_COLORS.items():
        rgba[mask == label] = color
    return rgba


def pick_best_slice(mask_3d: np.ndarray) -> int:
    """Return depth index with the most tumour voxels."""
    return int(np.argmax((mask_3d > 0).sum(axis=(1, 2))))


def norm_slice(arr: np.ndarray) -> np.ndarray:
    """Min-max normalise a 2D array to [0, 1] for display."""
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + 1e-8)


def volumetric_dice(gt: np.ndarray, pred: np.ndarray) -> dict:
    """
    Compute volumetric Dice for the three BraTS regions over a full (D,H,W) volume.

    Args:
        gt:   (D, H, W) int64 array with values {0,1,2,4}
        pred: (D, H, W) int64 array with values {0,1,2,4}

    Returns:
        dict with keys: dice_wt, dice_tc, dice_et, mean_dice
        Values are float; nan when both GT and pred are empty for a region.
    """
    def dice(a, b):
        a, b = a.astype(bool), b.astype(bool)
        num  = 2 * (a & b).sum()
        den  = a.sum() + b.sum()
        return float(num / den) if den > 0 else float('nan')

    wt = dice(gt > 0,
              pred > 0)
    tc = dice((gt == 1) | (gt == 4),
              (pred == 1) | (pred == 4))
    et = dice(gt == 4,
              pred == 4)

    scores = [s for s in [wt, tc, et] if not np.isnan(s)]
    mean   = float(np.mean(scores)) if scores else float('nan')

    return {'dice_wt': wt, 'dice_tc': tc, 'dice_et': et, 'mean_dice': mean}


def run_inference(model, modalities_list, device, use_amp):
    """
    Single-sample inference.

    Args:
        model:           SwinBraTS in eval mode
        modalities_list: list of 4 tensors, each (D, H, W)
        device:          torch.device
        use_amp:         bool

    Returns:
        pred_mask: (D, H, W) int64 numpy array with values {0,1,2,4}
    """
    inputs = [m.unsqueeze(0).to(device) for m in modalities_list]
    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=(use_amp and device.type == 'cuda')):
            logits = model(inputs)          # (1, 4, D, H, W)
    pred = logits.squeeze(0).argmax(dim=0).cpu().numpy().astype(np.int64)
    pred[pred == 3] = 4                     # remap channel 3 → label 4
    return pred


# ── Figure builder ─────────────────────────────────────────────────────────────

def save_comparison_figure(
    best: dict,
    worst: dict,
    metric_name: str,
    out_path: Path,
):
    """
    Save a 2-row × 5-column comparison figure.

    Each dict must have keys:
        case_id      str
        modalities   list of 4 (D,H,W) float32 numpy arrays
        gt_mask      (D,H,W) int64
        pred_mask    (D,H,W) int64
        scores       dict from volumetric_dice()
        slice_idx    int
    """
    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    fig.patch.set_facecolor('#1a1a2e')

    col_titles = ['FLAIR', 'T1ce', 'Ground Truth', 'Prediction', 'Overlay']
    row_labels  = [
        f'BEST  ({metric_name.replace("_", " ").upper()} = '
        f'{best["scores"][metric_name]:.4f})',
        f'WORST ({metric_name.replace("_", " ").upper()} = '
        f'{worst["scores"][metric_name]:.4f})',
    ]
    row_colors  = ['#00c853', '#ff1744']    # green for best, red for worst

    legend_patches = [
        mpatches.Patch(color=LABEL_COLORS[1][:3], label='NCR/NET (label 1)'),
        mpatches.Patch(color=LABEL_COLORS[2][:3], label='Edema (label 2)'),
        mpatches.Patch(color=LABEL_COLORS[4][:3], label='Enhancing (label 4)'),
    ]

    for row_idx, (case, row_color) in enumerate(zip([best, worst], row_colors)):
        sl = case['slice_idx']
        flair = norm_slice(case['modalities'][0][sl])
        t1ce  = norm_slice(case['modalities'][2][sl])
        gt_rgba   = build_colormap(case['gt_mask'][sl])
        pred_rgba = build_colormap(case['pred_mask'][sl])
        flair_rgb = np.stack([flair] * 3, axis=-1)

        panels = [flair, t1ce, gt_rgba, pred_rgba, None]

        for col_idx, (ax, title) in enumerate(zip(axes[row_idx], col_titles)):
            ax.set_facecolor('#1a1a2e')
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor('#444466')

            # Column title only on top row
            if row_idx == 0:
                ax.set_title(title, color='#ccccee', fontsize=11,
                             fontweight='bold', pad=6)

            if col_idx == 0:
                ax.imshow(panels[0], cmap='gray', vmin=0, vmax=1)
            elif col_idx == 1:
                ax.imshow(panels[1], cmap='gray', vmin=0, vmax=1)
            elif col_idx == 2:
                ax.imshow(panels[2])
            elif col_idx == 3:
                ax.imshow(panels[3])
            else:
                ax.imshow(flair_rgb, vmin=0, vmax=1)
                ax.imshow(pred_rgba)
                ax.legend(handles=legend_patches, loc='lower right',
                          fontsize=7, framealpha=0.5,
                          facecolor='#1a1a2e', labelcolor='white')

        # Row label on the left of first column
        axes[row_idx, 0].set_ylabel(
            row_labels[row_idx],
            color=row_color,
            fontsize=10,
            fontweight='bold',
            labelpad=8,
        )

        # Per-row subtitle with full volumetric scores + case ID
        sc = case['scores']

        def fmt(v):
            return f"{v:.3f}" if not np.isnan(v) else "n/a"

        subtitle = (
            f"{case['case_id']}  —  slice {sl:03d}\n"
            f"WT {fmt(sc['dice_wt'])}  |  TC {fmt(sc['dice_tc'])}"
            f"  |  ET {fmt(sc['dice_et'])}  |  Mean {fmt(sc['mean_dice'])}"
        )
        axes[row_idx, 2].set_title(
            subtitle,
            color='#eeeeff',
            fontsize=8.5,
            pad=6,
        )

    fig.suptitle(
        f'Best vs. Worst Prediction on Test Set\n'
        f'Ranked by {metric_name.replace("_", " ").title()}',
        color='white',
        fontsize=13,
        fontweight='bold',
        y=1.01,
    )

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Figure saved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Visualise best and worst SwinBraTS predictions on the test set.')
    parser.add_argument('--config',     default='configs/train_config.yml')
    parser.add_argument('--checkpoint', default=None,
                        help='Path to .pth checkpoint. Defaults to '
                             'checkpoint_dir/best_model.pth')
    parser.add_argument('--metric',     default='mean_dice',
                        choices=METRIC_CHOICES,
                        help='Metric used to rank samples (default: mean_dice)')
    parser.add_argument('--out_dir',    default=None,
                        help='Output directory. Defaults to results/viz/')
    args = parser.parse_args()

    # ── Config ────────────────────────────────────────────────────────────────
    config = load_config(args.config)

    device_str = config.get('device', 'cuda')
    device     = torch.device(device_str if torch.cuda.is_available() else 'cpu')
    use_amp    = config['training'].get('use_amp', True)

    checkpoint_path = (
        Path(args.checkpoint) if args.checkpoint
        else Path(config['training']['checkpoint_dir']) / 'best_model.pth'
    )
    out_dir = (
        Path(args.out_dir) if args.out_dir
        else Path(config['training']['results_dir']) / 'viz'
    )

    # ── Dataset ───────────────────────────────────────────────────────────────
    test_dir       = Path(config['data']['test_dir'])
    modality_order = config['data']['modality_order']

    dataset = MRIDataset(
        data_dir=test_dir,
        modalities=modality_order,
        transforms=None,
    )

    if len(dataset) == 0:
        print(f"No samples found in {test_dir}")
        sys.exit(1)

    print(f"Test set: {len(dataset)} samples in {test_dir}")

    # ── Model ─────────────────────────────────────────────────────────────────
    print(f"Loading model from {checkpoint_path}")
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = SwinBraTS(
        in_channels=config['model']['in_channels'],
        num_classes=config['model']['num_classes'],
        embed_dim=config['model']['embed_dim'],
        window_size=config['model']['window_size'],
        patch_size=config['model']['patch_size'],
        C=config['model']['C'],
        hidden_channels_projection=config['model']['hidden_channels_projection'],
        hidden_channels_reconstruction=config['model']['hidden_channels_reconstruction'],
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    ep  = checkpoint.get('epoch', '?')
    vd  = checkpoint.get('best_metric', None)
    msg = f"  Checkpoint: epoch {ep}"
    if vd is not None:
        msg += f", best val Dice {vd:.4f}"
    print(msg)

    # ── Run inference on entire test set ──────────────────────────────────────
    print(f"\nRunning inference on all {len(dataset)} test samples ...")
    print(f"Ranking metric: {args.metric}\n")

    records = []   # list of dicts, one per sample

    for idx in tqdm(range(len(dataset)), desc='Inference', unit='sample'):
        modalities_tuple, mask = dataset[idx]
        case_id = Path(dataset.cases[idx][0]).parent.name

        modalities_list = list(modalities_tuple)
        gt_mask         = mask.numpy().astype(np.int64)
        pred_mask       = run_inference(model, modalities_list, device, use_amp)
        scores          = volumetric_dice(gt_mask, pred_mask)

        records.append({
            'idx':        idx,
            'case_id':    case_id,
            'modalities': [m.numpy() for m in modalities_list],
            'gt_mask':    gt_mask,
            'pred_mask':  pred_mask,
            'scores':     scores,
            'slice_idx':  pick_best_slice(gt_mask),
        })

    # ── Rank and pick extremes ─────────────────────────────────────────────────
    # Filter out samples where the ranking metric is NaN (e.g. no ET in GT)
    valid   = [r for r in records if not np.isnan(r['scores'][args.metric])]
    invalid = [r for r in records if     np.isnan(r['scores'][args.metric])]

    if len(invalid) > 0:
        print(f"  Note: {len(invalid)} sample(s) skipped for ranking "
              f"(NaN {args.metric} — likely absent region in GT):")
        for r in invalid:
            print(f"    {r['case_id']}")

    if len(valid) < 2:
        print("Not enough valid samples to compare best vs. worst.")
        sys.exit(1)

    ranked = sorted(valid, key=lambda r: r['scores'][args.metric])
    worst  = ranked[0]
    best   = ranked[-1]

    print(f"\n  Best  → {best['case_id']:30s}  {args.metric} = "
          f"{best['scores'][args.metric]:.4f}")
    print(f"  Worst → {worst['case_id']:30s}  {args.metric} = "
          f"{worst['scores'][args.metric]:.4f}")

    # ── Save comparison figure ─────────────────────────────────────────────────
    fig_path = out_dir / f"extremes_{args.metric}.png"
    save_comparison_figure(best, worst, args.metric, fig_path)

    # ── Save full ranked score table ───────────────────────────────────────────
    scores_path = out_dir / f"extremes_{args.metric}_scores.txt"
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    with open(scores_path, 'w') as f:
        header = (f"{'Rank':>5}  {'Case ID':<30}  "
                  f"{'Dice_WT':>8}  {'Dice_TC':>8}  "
                  f"{'Dice_ET':>8}  {'Mean_Dice':>10}\n")
        f.write(header)
        f.write('-' * len(header) + '\n')
        for rank, r in enumerate(ranked, start=1):
            sc = r['scores']
            def fmt(v): return f"{v:.4f}" if not np.isnan(v) else "   nan"
            f.write(
                f"{rank:>5}  {r['case_id']:<30}  "
                f"{fmt(sc['dice_wt']):>8}  {fmt(sc['dice_tc']):>8}  "
                f"{fmt(sc['dice_et']):>8}  {fmt(sc['mean_dice']):>10}\n"
            )
        if invalid:
            f.write('\nSamples excluded from ranking (NaN metric):\n')
            for r in invalid:
                f.write(f"  {r['case_id']}\n")

    print(f"  Scores saved → {scores_path}")
    print(f"\nDone.")


if __name__ == '__main__':
    main()
