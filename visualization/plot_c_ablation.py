"""
Visualization of ablation study on channel dimension C hyperparameter.

Usage:
    python plot_c_ablation.py                           # saves plot_c_ablation.pdf + plot_c_ablation.png
    python plot_c_ablation.py --out results/c_ablation    # custom output path (no extension)

Outputs:
    c_ablation.png: Subplot figure showing metrics across values of C.
"""

import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# Data
C_VALS   = [24,        48,        72,        96,        120       ]
PARAMS_M = [3.622669,  8.730493,  17.222893, 29.099869, 44.361421 ]  # millions

DICE = {
    'WT':   [0.7761, 0.7570, 0.7513, 0.7536, 0.7619],
    'TC':   [0.6788, 0.6849, 0.6798, 0.6794, 0.6758],
    'ET':   [0.5384, 0.5427, 0.5373, 0.5293, 0.5402],
    'Mean': [0.6645, 0.6615, 0.6562, 0.6541, 0.6593],
}

HD95 = {
    'WT':   [11.300, 12.729, 13.421, 12.480, 13.309],
    'TC':   [11.775, 11.639, 11.963, 12.296, 11.821],
    'ET':   [11.063, 11.201, 11.677, 11.907, 11.164],
    'Mean': [11.379, 11.856, 12.354, 12.228, 12.098],
}

# Visual style 
COLORS     = {'WT': '#2196F3', 'TC': '#4CAF50', 'ET': '#F44336', 'Mean': '#FF9800'}
MARKERS    = {'WT': 'o',       'TC': 's',       'ET': '^',       'Mean': 'D'}
LINESTYLES = {'WT': '-',       'TC': '-',       'ET': '-',       'Mean': '--'}

plt.rcParams.update({
    'font.family':     'DejaVu Serif',
    'font.size':        9,
    'axes.titlesize':  10,
    'axes.labelsize':   9,
    'xtick.labelsize':  8,
    'ytick.labelsize':  8,
    'legend.fontsize':  8,
    'axes.linewidth':   0.8,
    'axes.grid':        True,
    'grid.alpha':       0.35,
    'grid.linewidth':   0.5,
    'lines.linewidth':  1.6,
    'lines.markersize': 6,
    'figure.dpi':      180,
})


def plot_line_panel(ax, x, metric_dict, ylabel, title, best_fn):
    """
    Plot one metric (Dice or HD95) for all regions against x (C or params).

    Args:
        ax:          matplotlib Axes
        x:           array-like x values
        metric_dict: dict mapping region name -> list of values
        ylabel:      y-axis label string
        title:       panel title string
        best_fn:     np.argmax or np.argmin - used to draw the best-C guideline
    """
    x = np.array(x)
    for key in ['WT', 'TC', 'ET', 'Mean']:
        ax.plot(x, metric_dict[key],
                color=COLORS[key],
                marker=MARKERS[key],
                linestyle=LINESTYLES[key],
                label=key,
                zorder=3)

    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight='bold', pad=5)

    # Dotted vertical at best-mean C
    best_idx = best_fn(np.array(metric_dict['Mean']))
    ax.axvline(x[best_idx], color='#888888', linestyle=':', linewidth=0.9, zorder=1)


# Main plot function 
def make_figure(c_vals, params_m, dice, hd95, out_base):
    x = np.array(c_vals)

    fig, axes = plt.subplots(2, 2, figsize=(7.16, 5.5))
    fig.subplots_adjust(hspace=0.42, wspace=0.32)

    # (a) Dice vs C 
    ax = axes[0, 0]
    plot_line_panel(ax, x, dice,
                    ylabel='Dice Similarity Coefficient',
                    title='(a) Dice vs. $C$',
                    best_fn=np.argmax)
    ax.set_xlabel('Base channel width $C$')
    ax.set_xticks(c_vals)
    ax.set_ylim(0.50, 0.84)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

    # (b) HD95 vs C 
    ax = axes[0, 1]
    plot_line_panel(ax, x, hd95,
                    ylabel='HD95 (mm)',
                    title='(b) HD95 vs. $C$',
                    best_fn=np.argmin)
    ax.set_xlabel('Base channel width $C$')
    ax.set_xticks(c_vals)
    ax.set_ylim(10.5, 14.5)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

    # (c) Dice vs #params 
    ax = axes[1, 0]
    for key in ['WT', 'TC', 'ET', 'Mean']:
        ax.plot(params_m, dice[key],
                color=COLORS[key],
                marker=MARKERS[key],
                linestyle=LINESTYLES[key],
                label=key,
                zorder=3)
        # Annotate C value above one set of points
        if key == 'WT':
            for xi, yi, ci in zip(params_m, dice[key], c_vals):
                label_pos = (10,6)
                if ci == 120:
                    label_pos = (-10,6)
                ax.annotate(f'$C$={ci}',
                            xy=(xi, yi), xytext=label_pos,
                            textcoords='offset points',
                            ha='center', fontsize=6.5, color='#444444')

    ax.set_xlabel('Parameters (M)')
    ax.set_ylabel('Dice Similarity Coefficient')
    ax.set_title('(c) Dice vs. Parameters', fontweight='bold', pad=5)
    ax.set_ylim(0.50, 0.84)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda v, _: f'{v:.1f}M'))

    # (d) Mean Dice vs Mean HD95 trade-off scatter
    ax = axes[1, 1]
    ax.set_xlim(11.0, 12.7)
    ax.set_ylim(0.652, 0.666)
    mean_dice = np.array(dice['Mean'])
    mean_hd95 = np.array(hd95['Mean'])

    sc = ax.scatter(mean_hd95, mean_dice,
                    c=params_m, cmap='viridis',
                    s=80, zorder=4,
                    edgecolors='white', linewidths=0.6)

    for xi, yi, ci in zip(mean_hd95, mean_dice, c_vals):
        ax.annotate(f'$C$={ci}',
                    xy=(xi, yi), xytext=(4, 3),
                    textcoords='offset points',
                    fontsize=7, color='#222222')

    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label('Parameters (M)', fontsize=7.5)
    cb.ax.tick_params(labelsize=7)

    ax.set_xlabel('Mean HD95 (mm)')
    ax.set_ylabel('Mean Dice')
    ax.set_title('(d) Dice\u2013HD95 Trade-off', fontweight='bold', pad=5)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

    # Legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.05,0.8), framealpha=0.7, edgecolor='#cccccc')

    # Save
    for ext in ('png'):
        path = f'{out_base}.{ext}'
        fig.savefig(path, bbox_inches='tight',
                    format=ext, dpi=200 if ext == 'png' else None)
        print(f'Saved {path}')

    plt.close(fig)


# CLI
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Plot SwinBraTS ablation results over channel width C.')
    parser.add_argument('--out', default='ablation_C',
                        help='Output path without extension (default: ablation_C)')
    args = parser.parse_args()

    make_figure(C_VALS, PARAMS_M, DICE, HD95, args.out)
