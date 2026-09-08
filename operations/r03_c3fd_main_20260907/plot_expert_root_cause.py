"""Plot tiny-set fitting and source-cluster physics checks from saved analyses."""
import hashlib
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main(directory):
    root = Path(directory)
    names = ['FIT8_ANALYSIS.json', 'GAUGE64_REPEATS_ANALYSIS.json', 'T2T64_ANALYSIS.json', 'TEACHER64_ANALYSIS.json']
    fit, gauge, t2t, teacher = [json.loads((root/name).read_text()) for name in names]
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.titlesize': 12,
        'figure.facecolor': 'white', 'axes.labelcolor': '#263241', 'text.color': '#263241',
        'xtick.color': '#425268', 'ytick.color': '#425268', 'svg.fonttype': 'none'})
    data = {'source_sha256': {n: hashlib.sha256((root/n).read_bytes()).hexdigest() for n in names}, 'curves': {}}
    fig, axs = plt.subplots(2, 2, figsize=(13, 8.8))
    events = list(fit['curves'])
    final = fit['training_final']
    events.append({'step': 1600, **final['metrics']})
    events = list({x['step']: x for x in events}.values())
    events.sort(key=lambda x: x['step'])
    for task, ax in zip(['G', 'S'], axs[0]):
        for split, color, label in [('train', '#2967A6', 'Eight training sources'),
                                     ('dev', '#B86B2E', 'Unseen development sources')]:
            points = [(x['step'], x[split][task+'_content_ce']) for x in events]
            data['curves'][task+':'+split] = points
            ax.plot(*zip(*points), color=color, lw=2.2, label=label)
            ax.annotate(f'{points[-1][1]:.3f}', points[-1], xytext=(5, 0), textcoords='offset points', color=color)
        ax.set(title=f'{task}: the tiny training set can be fitted', xlabel='Optimizer updates',
               ylabel='Next-token CE (nats)', xlim=(0, 1770), ylim=(0, 6))
        ax.grid(axis='y', alpha=.18)
        ax.legend(frameon=False, fontsize=8, loc='center right')
    ax = axs[1, 0]
    metric = 'S:verified_gain_ge_0.01'
    labels = ['Teacher', '64-source\ncontrol', 'Aligned\ntarget', 'Stage2\nmasked', 'Stage2\nvisible OLD', '8-source\nfit']
    numerators = [teacher['splits']['dev']['S']['S_positive_reconfirmed'],
        gauge['paired_effects'][metric]['control'], gauge['paired_effects'][metric]['aligned'],
        t2t['paired_effects'][metric]['masked'], t2t['paired_effects'][metric]['old_values'],
        fit['splits']['dev']['tasks']['S']['verified_gain_ge_0.01']['count']]
    denominators = [32, 128, 128, 128, 128, 128]
    rates = 100*np.array(numerators)/np.array(denominators)
    bars = ax.bar(np.arange(len(labels)), rates, color=['#24846D']+['#6D88A8']*5, width=.65)
    for bar, n, d in zip(bars, numerators, denominators):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2, f'{n}/{d}', ha='center', fontsize=9)
    ax.set_xticks(np.arange(len(labels)), labels, fontsize=8)
    ax.set(title='Real teacher headroom remains largely unrealized', ylabel='Verified OLD-to-target gain >= 0.01 eV/atom (%)', ylim=(0, 111))
    ax.grid(axis='y', alpha=.15)
    ax.set_axisbelow(True)
    data['common_dev_S_gain'] = {'labels': labels, 'counts': numerators, 'denominators': denominators,
        'independent_sources': 32, 'student_generation_seeds': 4,
        'comparison': 'descriptive across checkpoints; fixed paired interventions only within each A/B family'}
    effects = []
    for name, item, delta in [('Target alignment', gauge, 'aligned_minus_control_pp'),
                              ('Visible OLD input', t2t, 'old_values_minus_masked_pp')]:
        x = item['paired_effects'][metric]
        effects.append({'name': name, 'delta_pp': x[delta], 'interval': x['paired_source_bootstrap_95pct_interval_pp']})
    ax = axs[1, 1]
    reliability_labels = ['Original OLD', 'Teacher', '64-source control', 'Aligned target', 'Stage2 masked', 'Stage2 visible OLD', '8-source fit']
    reliability_n = [fit['splits']['dev']['tasks']['S']['old_R_verified'], teacher['splits']['dev']['S']['target_R_verified'],
        gauge['paired_effects']['S:R_verified']['control'], gauge['paired_effects']['S:R_verified']['aligned'],
        t2t['paired_effects']['S:R_verified']['masked'], t2t['paired_effects']['S:R_verified']['old_values'],
        fit['splits']['dev']['tasks']['S']['R_verified']['count']]
    reliability_d = [32, 32, 128, 128, 128, 128, 128]
    rates = 100*np.array(reliability_n)/np.array(reliability_d)
    bars = ax.barh(np.arange(7), rates, height=.62, color=['#24846D', '#24846D']+['#6D88A8']*5)
    for bar, n, d in zip(bars, reliability_n, reliability_d):
        ax.text(bar.get_width()+1.5, bar.get_y()+bar.get_height()/2, f'{n}/{d}', va='center', fontsize=9)
    ax.set_yticks(np.arange(7), reliability_labels, fontsize=9)
    ax.set(xlim=(0, 125), title='S proposals do not preserve OLD reliability',
           xlabel='R verified (%) — not SUN/MSUN')
    ax.set_xticks(np.arange(0, 101, 20))
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=.15)
    ax.set_axisbelow(True)
    data['paired_S_gain_effects'] = effects
    data['common_dev_S_R_verified'] = {'labels': reliability_labels, 'counts': reliability_n, 'denominators': reliability_d}
    fig.suptitle('Root-cause checks: fitting is possible; reliable development gains remain unproven', fontsize=15, y=.99)
    fig.text(.5, .019, 'Exploratory development diagnostics, not SUN/MSUN. Four student seeds reuse the same 32 S sources.\n'
             'Counts across checkpoints are descriptive, not isolated causal effects. Paired A/B intervals are recorded in the companion data and report.',
             ha='center', fontsize=9, color='#596779')
    fig.tight_layout(rect=(0, .063, 1, .95), h_pad=2.6, w_pad=2.5)
    for suffix in ['png', 'svg', 'pdf']:
        fig.savefig(root/('ROOT_CAUSE_RESULTS.'+suffix), dpi=190)
    svg = root/'ROOT_CAUSE_RESULTS.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n', encoding='utf-8')
    plt.close(fig)
    (root/'ROOT_CAUSE_PLOT_DATA.json').write_text(json.dumps(data, indent=2)+'\n')
    print(root/'ROOT_CAUSE_RESULTS.png')


if __name__ == '__main__':
    main(sys.argv[1])
