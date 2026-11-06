"""Render saved C3.1 CMOS samples and classification statistics."""
import argparse
import json
from pathlib import Path

import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('validation', type=Path)
    args = parser.parse_args()
    summary = json.loads((args.validation / 'summary.json').read_text(encoding='utf-8'))
    figures = args.validation / 'figures'
    figures.mkdir(exist_ok=False)
    records = {}
    for name, metrics in summary['conditions'].items():
        data = torch.load(args.validation / (name + '.pt'), map_location='cpu', weights_only=True)
        records[name] = data
        rois = metrics['roi_bounds_tile']
        lo, hi = rois[0][0], rois[3][1]
        margin = max(4, (hi - lo) // 8)
        lo, hi = max(0, lo - margin), min(768, hi + margin)
        fig, axes = plt.subplots(4, 5, figsize=(16, 12), constrained_layout=True)
        for label in range(4):
            choices = torch.where(data['example_labels'] == label)[0]
            if len(choices) != 5:
                raise ValueError(f'{name} needs five saved examples per class')
            for col, ix in enumerate(choices.tolist()):
                row = torch.where(data['indices'] == data['example_indices'][ix])[0]
                if len(row) != 1:
                    raise ValueError('Example index is not unique')
                j = int(row[0]); ax = axes[label, col]
                tile = data['example_cmos'][ix]
                ax.imshow(tile[lo:hi, lo:hi], cmap='inferno', vmin=0, vmax=max(float(tile.max()), 1e-30))
                for r0, r1, c0, c1 in rois:
                    ax.add_patch(Rectangle((c0-lo, r0-lo), c1-c0, r1-r0, fill=False, edgecolor='cyan', linewidth=.7))
                s = data['scores'][j].tolist(); pred = int(data['predictions'][j])
                ax.set_title(f'true={label} pred={pred} {"OK" if pred == label else "wrong"}\nS=' + '/'.join(f'{v:.1f}' for v in s), fontsize=7)
                ax.axis('off')
        fig.suptitle(f'{name}: pixel-integrated relative energy; each panel scaled for display')
        fig.savefig(figures / f'{name}_examples.png', dpi=150); plt.close(fig)
        fig, axes = plt.subplots(2, 2, figsize=(9, 8), constrained_layout=True)
        means = data['class_mean_cmos']
        vmax = float(means.max())
        for label, ax in enumerate(axes.flat):
            im = ax.imshow(means[label, lo:hi, lo:hi], cmap='inferno', vmin=0, vmax=vmax)
            for r0, r1, c0, c1 in rois:
                ax.add_patch(Rectangle((c0-lo, r0-lo), c1-c0, r1-r0, fill=False, edgecolor='cyan', linewidth=.7))
            ax.set_title(f'class {label}'); ax.axis('off')
        fig.colorbar(im, ax=axes, label='mean relative pixel energy')
        fig.savefig(figures / f'{name}_class_means.png', dpi=150); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for ax, (name, data) in zip(axes, records.items(), strict=True):
        matrix = torch.stack([data['scores'][data['labels'] == label].mean(0) for label in range(4)])
        matrix = matrix / matrix.sum(1, keepdim=True).clamp_min(1e-30)
        im = ax.imshow(matrix, vmin=0, vmax=1, cmap='viridis')
        for r in range(4):
            for c in range(4): ax.text(c, r, f'{matrix[r,c]:.2f}', ha='center', va='center', color='white')
        ax.set(title=name, xlabel='ROI', ylabel='true class', xticks=range(4), yticks=range(4))
    fig.colorbar(im, ax=axes, label='fraction of four-ROI energy')
    fig.savefig(figures / 'roi_energy_matrix.png', dpi=150); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for ax, (name, item) in zip(axes, summary['conditions'].items(), strict=True):
        matrix = item['confusion_4x4']; im = ax.imshow(matrix, cmap='Blues')
        for r in range(4):
            for c in range(4): ax.text(c, r, str(matrix[r][c]), ha='center', va='center')
        ax.set(title=f'{name}; abstain={sum(item["abstentions_by_class"])}', xlabel='prediction', ylabel='true class', xticks=range(4), yticks=range(4))
    fig.savefig(figures / 'confusion_matrices.png', dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    names = list(summary['conditions']); inside=[]; outside=[]
    for name in names:
        data=records[name]; inside.append(float(data['scores'].sum(1).mean())); outside.append(float((data['detected_energy']-data['scores'].sum(1)).clamp_min(0).mean()))
    ax.bar(names, inside, label='four ROIs'); ax.bar(names, outside, bottom=inside, label='outside ROIs')
    ax.set_ylabel('mean detected relative energy'); ax.legend(); ax.set_title('Finite-window diagnostic; leakage beyond the modeled relay field is unavailable')
    fig.savefig(figures / 'energy_distribution.png', dpi=150); plt.close(fig)


if __name__ == '__main__':
    main()
