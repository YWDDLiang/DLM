"""Standalone figures from source-bound mechanism-study result capsules."""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main(directory):
    directory = Path(directory)
    data = json.loads((directory/'DATAFACTOR_COMPACT.json').read_text())
    rounds = json.loads((directory/'ROUNDS64_COMPACT.json').read_text())
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.titlesize': 12,
        'figure.facecolor': 'white', 'axes.labelcolor': '#263241', 'text.color': '#263241',
        'xtick.color': '#425268', 'ytick.color': '#425268', 'svg.fonttype': 'none'})
    fig, axs = plt.subplots(2, 2, figsize=(12.5, 8.2))
    colors = {'64': '#2967A6', '256': '#B86B2E'}
    series = {}
    for task, ax in zip(['G', 'S'], axs[0]):
        for arm in ['64', '256']:
            events = list(data[f'{arm}:curves.jsonl'])
            final = data[f'{arm}:TRAIN_FINAL.json']
            events.append({'step': 1600, **final['metrics']})
            for split, style in [('train', '--'), ('dev', '-')]:
                x = [row['step'] for row in events]
                y = [row[split][task+'_content_ce'] for row in events]
                ax.plot(x, y, style, color=colors[arm], lw=2,
                        label=f'{arm} sources · {"development" if split == "dev" else "training"}')
                series[f'{arm}:{split}:{task}'] = list(zip(x, y))
        ax.set(title=f'{task} content: fitting versus generalization', xlabel='Optimizer updates', ylabel='Common next-token CE (nats)')
        ax.grid(axis='y', alpha=.18)
        ax.set_xlim(0, 1650)
        ax.legend(fontsize=8, frameon=False, loc='lower left')
    m = 66
    h = np.concatenate([[0.], np.cumsum(1/np.arange(1, m+1))])
    positions = np.arange(1, m+1)
    suffix = h[m] - h[m-positions]
    ax = axs[1, 0]
    ax.plot(positions, suffix, color='#AA4D53', lw=2.2, label='All pending, averaged by suffix length')
    ax.axhline(1., color='#24846D', lw=2, label='Next pending token only')
    ax.set_yscale('log')
    ax.set(xlabel='Numeric token position in a 66-token action', ylabel='Expected loss weight / uniform weight',
           title='The original objective underweights early decisions')
    ax.text(3, .023, f'Last / first coefficient = {m*h[m]:.1f}', color='#AA4D53', fontsize=10)
    ax.set_ylim(.01, 8)
    ax.grid(axis='y', alpha=.18)
    ax.legend(fontsize=8, frameon=False, loc='upper left')
    ax = axs[1, 1]
    endpoints = [0, 1, 2, 4]
    geo = [sum(x['status'] != 'invalid_raw' for x in rounds[str(n)]['rows']) for n in endpoints]
    verified = [sum(bool(x['verified']) for x in rounds[str(n)]['rows']) for n in endpoints]
    calls = [0.] + [np.mean([x['rounds'][n-1]['cumulative_forward_calls'] for x in rounds['samples']]) for n in [1, 2, 4]]
    ax.plot(endpoints, geo, 'o-', color='#2967A6', lw=2, label='Pass common R input geometry support')
    ax.plot(endpoints, verified, 's--', color='#B86B2E', lw=2, label='R verified (not SUN/MSUN)')
    for n, g, v in zip(endpoints, geo, verified):
        ax.annotate(str(g), (n, g), xytext=(0, 7), textcoords='offset points', ha='center', color='#2967A6')
        ax.annotate(str(v), (n, v), xytext=(0, -16), textcoords='offset points', ha='center', color='#B86B2E')
    ax.set(ylim=(0, 64), title='More rounds reach a plateau; S submits no edits', ylabel='Count among the same 64 development sources',
           xlabel='G/S rounds\n(mean cumulative forward calls)')
    ax.set_xticks(endpoints, [f'{n}\n({value:.1f})' for n, value in zip(endpoints, calls)])
    ax.grid(axis='y', alpha=.18)
    ax.legend(fontsize=8, frameon=False, loc='upper left')
    fig.suptitle('Mechanism checks: objective weighting, generalization and actual editing', fontsize=15, x=.5, y=.99)
    fig.text(.5, .012, 'Development diagnostics only. Round 2→4 R-verified loss occurred with unchanged structure tokens; ordinary R can vary.',
             ha='center', fontsize=9, color='#596779')
    fig.tight_layout(rect=(0, .045, 1, .955), h_pad=2.4, w_pad=2.2)
    fig.savefig(directory/'MECHANISM_OVERVIEW.png', dpi=190)
    fig.savefig(directory/'MECHANISM_OVERVIEW.svg')
    svg = directory/'MECHANISM_OVERVIEW.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n', encoding='utf-8')
    plt.close(fig)
    (directory/'MECHANISM_PLOT_DATA.json').write_text(json.dumps({'loss_curves': series,
        'position_coefficients_relative_to_uniform': suffix.tolist(),
        'rounds': endpoints, 'geometry_support_counts': geo, 'R_verified_counts': verified,
        'mean_cumulative_forward_calls': calls}, indent=2)+'\n')
    print(directory/'MECHANISM_OVERVIEW.png')


if __name__ == '__main__':
    main(sys.argv[1])
