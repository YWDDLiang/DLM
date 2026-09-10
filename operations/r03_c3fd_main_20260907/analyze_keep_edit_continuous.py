"""Diagnose proposal supply and learned acceptance on TRAIN/DEV only."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

SOURCE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SOURCE / 'src'))
from scripts.run_post_refine_cycle import read_rows, write_json, file_hash
from scripts.run_rsi_stages import scores, score_directory
from compile_keep_edit_utility import utility
from editor_trial_analysis import flags


def main(root):
    split = read_rows(root / 'SOURCE_SPLIT.jsonl')
    before = scores(root / 'fit', 'native')
    proposal = scores(root / 'fit', 'hybrid_proposal')
    predicted = {r['ordinal']: r for r in read_rows(root / 'evaluation_features/UTILITY_PREDICTIONS.jsonl')}
    reports = {}
    for role in ('train', 'dev'):
        transitions = Counter()
        cases = []
        for ordinal, source in enumerate(split):
            if source['split'] != role:
                continue
            a, b = before[ordinal], proposal[ordinal]
            if a['sample_idx'] != b['sample_idx']:
                raise ValueError('proposal diagnostic source alignment failed')
            left, right = utility(a), utility(b)
            transitions[f'{left}->{right}'] += 1
            af, bf = flags(a), flags(b)
            trace = json.loads((root / f'fit/proposal/records/{ordinal:04d}.json').read_text())['editor_trace']
            commit = json.loads((root / f'fit/hybrid_proposal/records/{ordinal:04d}.json').read_text())['commit_trace']
            prediction = predicted.get(ordinal)
            cases.append(dict(ordinal=ordinal, old_E_seen_source=source['old_E_seen_source'],
                before=af, proposal=bf, before_utility=left, proposal_utility=right,
                utility_delta=None if left is None or right is None else right-left,
                continuous_patch_applied=commit['applied'], proposal_generated=trace.get('proposal_generated', False),
                known_SUN_guard=bool(trace.get('known_sun') and trace.get('learned_mode') == 0),
                action=trace.get('action'), prediction=prediction))
        diagnostics = []
        for epoch in (4, 8, 16, 32):
            for margin in (0., .05, .1, .2):
                accepted = Counter()
                missed = Counter()
                signed = Counter()
                gains = {name: [] for name in ('Stable', 'MS', 'SUN', 'MSUN')}
                losses = {name: [] for name in gains}
                for row in cases:
                    prediction = row['prediction']
                    chose = bool(prediction and row['proposal_generated'] and row['continuous_patch_applied']
                        and not row['known_SUN_guard'] and prediction['learned_utilities'][str(epoch)] >= margin)
                    delta = row['utility_delta']
                    kind = 'unknown' if delta is None else 'beneficial' if delta > 0 else 'harmful' if delta < 0 else 'neutral'
                    (accepted if chose else missed)[kind] += 1
                    if chose:
                        if delta is not None:
                            signed['sum_signed_utility'] += delta
                        for name in gains:
                            if row['proposal'][name] and not row['before'][name]:
                                gains[name].append(row['ordinal'])
                            if row['before'][name] and not row['proposal'][name]:
                                losses[name].append(row['ordinal'])
                diagnostics.append(dict(epoch=epoch, margin=margin, accepted=dict(accepted),
                    kept=dict(missed), signed=dict(signed),
                    full_proposal_panel_flag_gains=gains, full_proposal_panel_flag_losses=losses,
                    U_needs_actual_policy_panel=True))
        reports[role] = dict(requests=len(cases), utility_transition_counts=dict(transitions),
            available_Stable_promotions=[r['ordinal'] for r in cases if r['proposal']['Stable'] and not r['before']['Stable']],
            available_SUN_gains_without_Stable_promotion=[r['ordinal'] for r in cases
                if r['proposal']['SUN'] and not r['before']['SUN'] and r['before']['Stable']],
            available_novel_Stable_promotions=[r['ordinal'] for r in cases
                if r['proposal_utility'] == 2 and r['before_utility'] is not None and r['before_utility'] < 2],
            available_positive_utility=[r['ordinal'] for r in cases if r['utility_delta'] is not None and r['utility_delta'] > 0],
            policies=diagnostics, cases=cases)
    report = dict(schema='continuous_KEEP_EDIT_TRAIN_DEV_diagnostic_v1',
        final_quality_consulted=False, reports=reports,
        binding={str(path): file_hash(path) for path in (
            root / 'SOURCE_SPLIT.jsonl', root / 'evaluation_features/UTILITY_PREDICTIONS.jsonl',
            score_directory(root / 'fit', 'native') / 'attempt_results.jsonl',
            score_directory(root / 'fit', 'hybrid_proposal') / 'attempt_results.jsonl')})
    write_json(root / 'analysis/CONTINUOUS_PROPOSAL_TRAIN_DEV_DIAGNOSTIC.json', report)
    print(json.dumps({role: {k: v for k, v in result.items() if k not in ('policies', 'cases')}
        for role, result in reports.items()}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    main(parser.parse_args().root)
