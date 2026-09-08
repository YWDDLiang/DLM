"""Derive compact summaries from the preserved local mechanism capsules."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics as st
import sys


def main(directory):
    root = Path(directory)
    files = ['LOSS64_ANALYSIS.json', 'COMPOSITION16_ANALYSIS.json', 'ROUNDS64_COMPACT.json',
             'DATAFACTOR_COMPACT.json', 'DETERMINISTIC6_SAUDIT_COMPACT.json', 'R_REPEAT8_COMPACT.json', 'GAUGE32_AUDIT.json']
    source = {name: json.loads((root/name).read_text()) for name in files}
    report = {'schema': 'expert_mechanism_result_index_v1', 'primary_metrics_rerun': False,
        'source_capsule_sha256': {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in files}}
    rounds = source['ROUNDS64_COMPACT.json']
    report['sequential_rounds'] = []
    for n in [0, 1, 2, 4]:
        rows = rounds[str(n)]['rows']
        report['sequential_rounds'].append({'rounds': n, 'sources': len(rows),
            'common_R_input_support': sum(x['status'] != 'invalid_raw' for x in rows),
            'R_verified': sum(x['verified'] for x in rows), 'statuses': dict(Counter(x['status'] for x in rows)),
            'mean_cumulative_forwards': st.mean(s['rounds'][n-1]['cumulative_forward_calls'] for s in rounds['samples']) if n else 0})
    report['round2_to4_unchanged_input_status_changes'] = []
    labels = {n: {row['group_id']: row for row in rounds[str(n)]['rows']} for n in [2, 4]}
    for sample in rounds['samples']:
        if sample['rounds'][1]['body_sha256'] == sample['rounds'][3]['body_sha256']:
            a, b = (labels[n][sample['ancestor_id']] for n in [2, 4])
            if a['status'] != b['status']:
                report['round2_to4_unchanged_input_status_changes'].append({'case_idx': sample['probe_case'],
                    'R2': a['status'], 'R4': b['status'], 'R2_energy': a['terminal_energy'], 'R4_energy': b['terminal_energy']})
    det = source['DETERMINISTIC6_SAUDIT_COMPACT.json']
    prints = {(x['case_idx'], x['variant'], x['repeat']): x for x in det['refined/FINGERPRINTS.json']}
    pairs = [(prints[k[0], k[1], 0], x) for k, x in prints.items() if k[2] == 1]
    report['F800_deterministic_repeat'] = {'independent_source_count': 6, 'paired_source_variants': len(pairs),
        **{name+'_equal_pairs': sum(a[name] == b[name] for a, b in pairs) for name in ['cuda_rng_before', 'cuda_rng_after', 'output_tensors']},
        'source_subset_fixed_before_outcomes': True}
    gate_labels = {row['trajectory_id']: row for row in det['labels_S_audit']['rows']}
    decisions = []
    for row in det['decisions']:
        key = f'round-S-audit:{row["decision_idx"]}:'
        a, b = gate_labels[key+'old'], gate_labels[key+'proposal']
        gain = a['terminal_energy'] - b['terminal_energy'] if a['verified'] and b['verified'] else None
        decisions.append({'case_idx': row['case_idx'], 'round': row['round'], 'old_status': a['status'],
            'proposal_status': b['status'], 'both_verified_gain_eV_atom': gain, 'p_accept': row['quality'][3],
            'meets_S_gain_contract': gain is not None and gain >= .01})
    report['rejected_S_proposals'] = {'attempts': len(decisions), 'source_groups': len({x['ancestor_id'] for x in det['decisions']}),
        'accepted_by_original_policy': sum(x['accepted'] for x in det['decisions']),
        'old_statuses': dict(Counter(x['old_status'] for x in decisions)),
        'proposal_statuses': dict(Counter(x['proposal_status'] for x in decisions)),
        'both_verified_pairs': sum(x['both_verified_gain_eV_atom'] is not None for x in decisions),
        'meets_S_gain_contract': sum(x['meets_S_gain_contract'] for x in decisions), 'decisions': decisions}
    report['R_numerical_repeat'] = {}
    r_data = source['R_REPEAT8_COMPACT.json']
    for mode in ['ordinary', 'deterministic']:
        by = {rep: {row['trajectory_id']: row for row in r_data[mode+'_R'+str(rep)]['rows']} for rep in [1, 2]}
        pairs = [(x, by[2][k]) for k, x in by[1].items()]
        differences = [abs(b['terminal_energy'] - a['terminal_energy']) for a, b in pairs
                       if a['terminal_energy'] is not None and b['terminal_energy'] is not None]
        report['R_numerical_repeat'][mode] = {'inputs': len(pairs), 'finite_energy_pairs': len(differences),
            'max_abs_terminal_difference_eV_atom': max(differences),
            'mean_abs_terminal_difference_eV_atom': st.mean(differences),
            'status_changes': sum(a['status'] != b['status'] for a, b in pairs),
            'step_count_changes': sum(a['actual_steps'] != b['actual_steps'] for a, b in pairs),
            'runtime': r_data[mode+'_R1']['report']['runtime_identities']}
    data = source['DATAFACTOR_COMPACT.json']
    report['data_factor_at_equal_1600_updates'] = {}
    for arm in ['64', '256']:
        result = {'train_seconds': data[f'{arm}:TRAIN_FINAL.json']['train_seconds']}
        for split in ['train', 'dev']:
            rows = data[f'{arm}:{split}']['rows']
            result[split] = {'sources': len(rows), 'common_R_input_support': sum(x['status'] != 'invalid_raw' for x in rows),
                'R_verified': sum(x['verified'] for x in rows),
                'common_next_token_CE': {k: data[f'{arm}:TRAIN_FINAL.json']['metrics'][split][k]
                                         for k in ['G_content_ce', 'S_content_ce', 'first_lattice_content_ce', 'coord_content_ce']}}
        report['data_factor_at_equal_1600_updates'][arm] = result
    report['cumulative_conservative_training_seconds_before_gauge64'] = 3678.791464943438 + sum(data[f'{a}:TRAIN_FINAL.json']['train_seconds'] for a in ['64', '256'])
    (root/'MECHANISM_RESULT_INDEX.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ['rejected_S_proposals', 'source_capsule_sha256']}, indent=2))


if __name__ == '__main__':
    main(sys.argv[1])
