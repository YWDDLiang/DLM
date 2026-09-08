"""Paired loss-control summaries; bootstrap units are original source groups."""
from collections import Counter
import json
from pathlib import Path
import random
import statistics as st
import sys


def summarize(path, output):
    data = json.loads(Path(path).read_text())
    seeds = [2026090813, 2026090913, 2026091013, 2026091113]
    summary = {'source_count': 64, 'repeated_generation_seeds': seeds, 'seed_source_observations': 256,
        'not_256_independent_sources': True, 'primary_metric_claim': False,
        'geometry_definition': 'common R input support (not frozen benchmark Struct_valid)',
        'R_verified_is_not_SUN_or_MSUN': True, 'per_seed': [], 'per_task': {}}
    grouped = {}
    pairs = []
    for seed in seeds:
        arms = {arm: {row['group_id']: row for row in data[f'{arm}:{seed}']['rows']} for arm in ['suffix', 'next']}
        assert arms['suffix'].keys() == arms['next'].keys() and len(arms['next']) == 64
        summary['per_seed'].append({'seed': seed, **{arm: dict(Counter(row['status'] for row in rows.values())) for arm, rows in arms.items()}})
        for group in arms['next']:
            a, b = arms['suffix'][group], arms['next'][group]
            assert a['trajectory_id'].split(':')[-1] == b['trajectory_id'].split(':')[-1]
            pair = {'group': group, 'seed': seed, 'task': a['trajectory_id'].split(':')[-1], 'suffix': a, 'next': b}
            grouped.setdefault(group, []).append(pair)
            pairs.append(pair)
    assert len(grouped) == 64 and all(len(rows) == 4 for rows in grouped.values())
    for task in ['all', 'G', 'S']:
        selected = [p for p in pairs if task == 'all' or p['task'] == task]
        result = {}
        for metric, condition in [('raw_geometry_valid', lambda x: x['status'] != 'invalid_raw'), ('R_verified', lambda x: bool(x['verified']))]:
            result[metric] = {arm: sum(condition(p[arm]) for p in selected) for arm in ['suffix', 'next']}
            result[metric].update(denominator=len(selected), paired_gains=sum(condition(p['next']) and not condition(p['suffix']) for p in selected),
                                  paired_losses=sum(condition(p['suffix']) and not condition(p['next']) for p in selected))
            cluster_deltas = [st.mean(int(condition(p['next'])) - int(condition(p['suffix'])) for p in rows)
                              for rows in grouped.values() if task == 'all' or rows[0]['task'] == task]
            rng = random.Random(2026090829)
            boot = sorted(st.mean(rng.choices(cluster_deltas, k=len(cluster_deltas))) for _ in range(10000))
            result[metric]['paired_source_bootstrap_95pct_interval_pp'] = [100*boot[250], 100*boot[9749]]
        for verified in [False, True]:
            ds = [p['next']['terminal_energy'] - p['suffix']['terminal_energy'] for p in selected
                  if p['next']['terminal_energy'] is not None and p['suffix']['terminal_energy'] is not None
                  and (not verified or (p['next']['verified'] and p['suffix']['verified']))]
            result['both_verified_energy' if verified else 'all_finite_descriptive_energy'] = {'n': len(ds),
                'mean_next_minus_suffix_eV_atom': st.mean(ds) if ds else None,
                'median_next_minus_suffix_eV_atom': st.median(ds) if ds else None,
                'lower_by_0.01': sum(d < -.01 for d in ds), 'higher_by_0.01': sum(d > .01 for d in ds)}
        summary['per_task'][task] = result
    fields = ['G_content_ce', 'S_content_ce', 'first_lattice_content_ce', 'length_content_ce', 'angle_content_ce', 'coord_content_ce']
    summary['common_next_token_evaluation_at_400'] = {arm: {split: {k: data[f'{arm}:TRAIN_FINAL.json']['metrics'][split][k] for k in fields}
                                                         for split in ['train', 'dev']} for arm in ['suffix', 'next']}
    Path(output).write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    summarize(sys.argv[1], sys.argv[2])
