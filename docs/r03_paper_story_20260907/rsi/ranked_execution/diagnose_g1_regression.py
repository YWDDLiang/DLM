"""Audit same-condition G1 regression without changing the running experiment."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import sys


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def collect(root):
    source = Path((root / 'SOURCE_PATH').read_text().strip())
    sys.path.insert(0, str(source / 'src'))
    from crystal_dlm.ranked_feedback import endpoint_quality
    cohorts = [root / 'fit', root / 'rounds/round1/fit']
    hashes = [digest(p / 'cohort/plans.jsonl') for p in cohorts]
    if hashes[0] != hashes[1]:
        raise ValueError('Plan, prompt, or per-request seeds changed')
    records = [[json.loads(p.read_text()) for p in sorted((c / 'construction/records').glob('*.json'))]
               for c in cohorts]
    if any(len(items) != 256 for items in records):
        raise ValueError('complete denominator required')
    result = {'plans_sha256': hashes[0], 'same_plans_prompts_G_and_F_seeds': True,
              'denominator': 256, 'stage_diagnostics': {}}
    for name, cohort, items in zip(('S0', 'S1'), cohorts, records):
        reasons, axes = Counter(), Counter()
        xy = []
        for item in items:
            record = item['record']
            tokens = record.get('body_token_ids')
            if not record['success']:
                failure = item['trace']['construction_failure']
                reasons[failure['reason']] += 1
                for pos in failure.get('failure_positions', []):
                    axes['cell' if pos < 7 else ('X', 'Y', 'Z', 'element')[(pos - 8) % 4]] += 1
                tokens = failure['partial_body_token_ids'][0]
            n = (len(tokens) - 7) // 4
            columns = Counter((tokens[8 + 4*i], tokens[9 + 4*i]) for i in range(n))
            xy.append((len(columns)/n, max(columns.values())))
        result['stage_diagnostics'][name] = {
            'generation_success': sum(x['record']['success'] for x in items),
            'failure_reasons': dict(reasons), 'failure_position_counts': dict(axes),
            'mean_unique_XY_fraction': statistics.mean(x[0] for x in xy),
            'mean_max_XY_column_multiplicity': statistics.mean(x[1] for x in xy),
            'requests_with_XY_column_at_least_four_atoms': sum(x[1] >= 4 for x in xy),
            'XY_note': 'Final successful body or retained failed partial body; descriptive, not a causal test.',
            'raw_scores': json.loads((cohort/'construction/scoring/result/RANKED_METRICS.json').read_text()),
            'token_scores': json.loads((cohort/'tokenized/scoring/result/RANKED_METRICS.json').read_text())}
    before, after = [rows(c/'tokenized/scoring/result/attempt_results.jsonl') for c in cohorts]
    transitions = Counter()
    for a, b, ra, rb in zip(before, after, *records):
        if not a['sample_idx'] == b['sample_idx'] == ra['record']['sample_idx'] == rb['record']['sample_idx']:
            raise ValueError('request identity changed')
        sa, sb = (endpoint_quality(x)['rank'] == 4 for x in (a, b))
        transitions[f'{sa} -> {sb}'] += 1
        if sa and not sb:
            transitions['lost_SUN_with_G_success' if rb['record']['success'] else 'lost_SUN_to_G_failure'] += 1
    result['SUN_transitions'] = dict(transitions)
    data_path = root/'fit/pairs/G.jsonl'
    data = rows(data_path)
    counts = Counter(x['objective_level'] for x in data)
    weights = {'SUN': 16, 'strict_stable': 8, 'meta_stable': 4, 'ordinary_improvement': 1}
    result['training_data'] = {'records': len(data), 'sha256': digest(data_path),
        'records_by_level': dict(counts),
        'group_draw_probabilities': {k: weights[k]/sum(weights[n] for n in counts) for k in counts},
        'pair_and_anchor_counts': dict(Counter(x['objective_level'] + (' pair' if x.get('chosen_tokens') is not None else ' anchor') for x in data))}
    receipt = json.loads((root/'training/round1/G/result/TRAINING_FINAL.json').read_text())
    result['training'] = {k: receipt[k] for k in ('optimizer_steps', 'parameter_delta_squared', 'training_seconds')}
    result['causality_limit'] = 'Same-condition regression and failure mechanism established; oversampling and surrogate mismatch remain hypotheses, without an intervention test.'
    result['analysis_sha256'] = digest(Path(__file__))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = collect(args.root)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
    print(json.dumps(report))
