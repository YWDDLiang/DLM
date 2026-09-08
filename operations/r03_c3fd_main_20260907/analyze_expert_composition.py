"""Summarize paired F800 and R repeats without treating repeats as source groups."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics as st


def distribution(values):
    values = [x for x in values if x is not None]
    return {'n': len(values), 'mean': st.mean(values) if values else None,
            'median': st.median(values) if values else None,
            'min': min(values) if values else None, 'max': max(values) if values else None,
            'lower_by_more_than_0.01': sum(x < -.01 for x in values),
            'higher_by_more_than_0.01': sum(x > .01 for x in values)}


def delta(a, b, verified=False):
    if (verified and not (a['verified'] and b['verified'])) or a['terminal_energy'] is None or b['terminal_energy'] is None:
        return None
    return a['terminal_energy'] - b['terminal_energy']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    raw = {tuple(row['trajectory_id'].split(':')[1:]): row for row in data['mechanism_composition16/labels_raw']['rows']}
    refined = {tuple(row['trajectory_id'].split(':')[1:]): row for row in data['mechanism_composition16_labels/labels_refined']['rows']}
    r_repeats = {n: {tuple(row['trajectory_id'].split(':')[1:4]): row for row in data[f'mechanism_composition16_labels/labels_R_repeat{n}']['rows']} for n in (1, 2)}
    variants = ('old', 'teacher', 'student')
    result = {'source_count': 16, 'technical_F_repeats_per_source': 2,
              'interpretation': 'composition-matched terminal energy differences in eV/atom; finite unverified R outputs are descriptive only; no new SUN/MSUN claim'}
    result['status_counts'] = {}
    for name, lookup in [('raw', raw), ('F800', refined)]:
        result['status_counts'][name] = {v: dict(Counter(x['status'] for k, x in lookup.items() if k[-1] == v)) for v in variants}
    result['paired_energy'] = {}
    for a, b in [('teacher', 'old'), ('student', 'old'), ('student', 'teacher')]:
        name = a + '_minus_' + b
        result['paired_energy'][name] = {}
        for verified in (False, True):
            raw_values = [delta(raw[str(i), a], raw[str(i), b], verified) for i in range(16)]
            by_rep = [[delta(refined[str(rep), str(i), a], refined[str(rep), str(i), b], verified) for i in range(16)] for rep in (0, 1)]
            paired = [(x, y) for x, y in zip(*by_rep) if x is not None and y is not None]
            result['paired_energy'][name]['both_verified' if verified else 'all_finite'] = {
                'raw': distribution(raw_values), 'F_repeat0': distribution(by_rep[0]), 'F_repeat1': distribution(by_rep[1]),
                'F_source_mean_of_two': distribution([(x + y)/2 for x, y in paired]),
                'F_same_source_available_both_repeats': len(paired),
                'F_better_in_both_repeats_0.01': sum(x < -.01 and y < -.01 for x, y in paired),
                'F_worse_in_both_repeats_0.01': sum(x > .01 and y > .01 for x, y in paired),
                'F_opposite_sign_beyond_0.01': sum((x < -.01 and y > .01) or (y < -.01 and x > .01) for x, y in paired)}
    result['technical_variance'] = {}
    for variant in variants:
        pairs = [(refined['0', str(i), variant], refined['1', str(i), variant]) for i in range(16)]
        result['technical_variance'][variant] = {'F_then_R_abs_terminal_difference': distribution([abs(d) for a, b in pairs if (d := delta(a, b)) is not None]),
            'verified_status_changed': sum(a['verified'] != b['verified'] for a, b in pairs)}
    result['R_only_repeats'] = {}
    for verified in (False, True):
        pairs = [(r_repeats[1][k], r_repeats[2][k]) for k in r_repeats[1]]
        result['R_only_repeats']['both_verified' if verified else 'all_finite'] = {
            'abs_terminal_difference': distribution([abs(d) for a, b in pairs if (d := delta(a, b, verified)) is not None]),
            'verified_status_changed': sum(a['verified'] != b['verified'] for a, b in pairs)}
    result['cases'] = []
    for case in data['cases']:
        i = str(case['case_idx'])
        result['cases'].append({'case_idx': int(i), 'stratum': case['stratum'],
            'raw_terminal_energy': {v: raw[i, v]['terminal_energy'] for v in variants},
            'raw_verified': {v: raw[i, v]['verified'] for v in variants},
            'F_terminal_energy_by_repeat': [{v: refined[str(rep), i, v]['terminal_energy'] for v in variants} for rep in (0, 1)],
            'F_verified_by_repeat': [{v: refined[str(rep), i, v]['verified'] for v in variants} for rep in (0, 1)]})
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'cases'}, indent=2))


if __name__ == '__main__':
    main()
