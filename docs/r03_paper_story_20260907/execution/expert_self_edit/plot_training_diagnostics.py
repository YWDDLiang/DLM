"""Render the recorded training curves and paired development outcomes."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
curves = json.loads((ROOT / 'LEARNING_CURVES.json').read_text())
dev = json.loads((ROOT / 'DEV_COMPARISON_FINAL.json').read_text())
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.titleweight': 'bold', 'svg.fonttype': 'none'})

fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
for row, (key, title) in enumerate((('expert_full_stage1', 'Stage 1: content and geometry'),
                                   ('expert_full_stage2', 'Stage 2: student feedback and preservation'))):
    data = curves[key]
    points = data['points']
    x = [p['step'] for p in points]
    for task, color in [('G', '#276FBF'), ('S', '#C86226')]:
        for split, style in [('train', '--'), ('dev', '-')]:
            axes[row, 0].plot(x, [p[split][task + '_real_content_ce'] for p in points],
                             style, color=color, linewidth=1.8,
                             label=f'{task} {"training" if split == "train" else "development"}')
    for split, color in [('train', '#7057A3'), ('dev', '#238A79')]:
        axes[row, 1].plot(x, [p[split]['quality_bce'] for p in points],
                         color=color, linewidth=1.8,
                         label='Training views' if split == 'train' else 'Fixed development views')
    axes[row, 0].set_title(f'{title}\nContent prediction', loc='left', fontsize=11)
    axes[row, 1].set_title('Quality heads (mixed supervised views)', loc='left', fontsize=11)
    axes[row, 0].set_ylabel('Cross-entropy per supervised token')
    axes[row, 1].set_ylabel('Reported mixed quality BCE')
    for column, ax in enumerate(axes[row]):
        ax.set_xlabel('Optimizer updates within stage')
        ax.grid(axis='y', alpha=.18)
        if row == 0 or column == 1:
            ax.legend(frameon=False, fontsize=9)
fig.suptitle('Training diagnostics: 2,400 updates across two stages', fontsize=16)
fig.supxlabel('Stage 2 changes the training-view mixture. Quality BCE is not an acceptance error rate.\n'
              'Content axes use separate scales for each stage; the development views remain fixed.', fontsize=10)
for suffix in ('png', 'svg'):
    path = ROOT / ('training_curves.' + suffix)
    fig.savefig(path, dpi=180)
    if suffix == 'svg':
        path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
plt.close(fig)

labels = ['Original B0', 'Stage 1\n1,400 steps', 'Stage 2\n800 steps (selected)', 'Stage 2\n1,000 steps']
ordered = [dev[k] for k in ('stage1_1400', 'stage2_800', 'stage2_1000')]
retained = np.asarray([42] + [d['pairs']['11'] for d in ordered])
repaired = np.asarray([0] + [d['pairs']['01'] for d in ordered])
harmed = np.asarray([0] + [d['pairs']['10'] for d in ordered])
x = np.arange(len(labels))
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
axes[0].bar(x, retained, color='#276FBF', label='Original reliable inputs still classified reliable')
axes[0].bar(x, repaired, bottom=retained, color='#238A79', label='Previously unreliable inputs now classified reliable')
axes[0].axhline(42, color='#5A6470', linewidth=1, linestyle='--', alpha=.7)
for i, (a, b) in enumerate(zip(retained, repaired)):
    axes[0].text(i, a / 2, str(a), ha='center', va='center', color='white', weight='bold')
    if b:
        axes[0].text(i, a + b / 2, str(b), ha='center', va='center', color='white', weight='bold')
    axes[0].text(i, a + b + 1, str(a + b), ha='center', weight='bold')
axes[0].set_ylim(0, 62)
axes[0].set_title('Reliable structures after editing', loc='left')
axes[0].legend(frameon=False, fontsize=8, loc='upper left')
axes[1].bar(x, harmed, color='#C44C49')
for i, value in enumerate(harmed):
    axes[1].text(i, value + .7, str(value), ha='center', weight='bold')
axes[1].set_ylim(0, 42)
axes[1].set_title('Observed reliable-to-unreliable changes', loc='left')
for ax in axes:
    ax.set_xticks(x, labels, fontsize=9)
    ax.set_ylabel('Structures')
    ax.grid(axis='y', alpha=.18)
    ax.set_axisbelow(True)
fig.suptitle('Observed reliability on the fixed development cohort', fontsize=15)
fig.supxlabel('259 original B0 inputs; 42 initially reliable. Repeated R can vary even for unchanged inputs.\n'
              'These counts describe independent evaluations; common R reliability is not SUN/MSUN.', fontsize=10)
for suffix in ('png', 'svg'):
    path = ROOT / ('development_reliability.' + suffix)
    fig.savefig(path, dpi=180)
    if suffix == 'svg':
        path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
plt.close(fig)
print('Wrote training_curves and development_reliability as PNG and SVG.')
