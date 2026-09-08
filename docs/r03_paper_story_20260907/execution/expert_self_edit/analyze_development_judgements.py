"""Audit saved development transactions without rerunning or changing the policy."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--b0-checkpoint', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source_root / 'src'))
    from crystal_dlm.expert_edit_data import (bound_labels, read_rows, sha256,
        replay_feedback_trace, feedback_decision, decode_body, certify_geometry,
        _proposal_inputs_for_split)
    from transformers import AutoTokenizer

    root = args.run_root
    sample_dir = root / 'stage2_diagnostics/autonomous_selected'
    extra_dir = root / 'development_proposal_audit'
    samples_file = sample_dir / 'samples.jsonl'
    samples = read_rows(samples_file)
    tokenizer = AutoTokenizer.from_pretrained(args.b0_checkpoint, local_files_only=True, trust_remote_code=True)
    inverse = {int(value): key for key, value in tokenizer.get_vocab().items()}
    prep = json.loads((extra_dir / 'prepared/PROPOSALS_FINAL.json').read_text())
    assert prep['schema'] == 'development_proposal_physics_v1'
    assert prep['training_use_allowed'] is False and prep['source_split'] == 'dev'
    assert prep['sample_sha256'] == sha256(samples_file)
    actual_inputs = read_rows(extra_dir / 'prepared/inputs.jsonl')
    mapping = read_rows(extra_dir / 'prepared/proposal_map.jsonl')
    expected_inputs, expected_mapping = _proposal_inputs_for_split(samples, tokenizer, 'dev')
    assert json.dumps(expected_mapping, sort_keys=True) == json.dumps(mapping, sort_keys=True)
    roundoff = {'float64_scaled_epsilon_limit': 128, 'maximum_absolute_difference': 0., 'different_float_leaves': 0}
    def check_reconstruction(expected, actual):
        if isinstance(expected, dict):
            assert isinstance(actual, dict) and expected.keys() == actual.keys()
            for key in expected:
                check_reconstruction(expected[key], actual[key])
        elif isinstance(expected, (list, tuple)):
            assert isinstance(actual, (list, tuple)) and len(expected) == len(actual)
            for left, right in zip(expected, actual):
                check_reconstruction(left, right)
        elif isinstance(expected, float) and isinstance(actual, float):
            if math.isnan(expected) and math.isnan(actual):
                return
            if expected != actual:
                delta = abs(expected - actual)
                assert math.isfinite(delta) and delta <= 128 * sys.float_info.epsilon * max(1., abs(expected), abs(actual))
                roundoff['maximum_absolute_difference'] = max(roundoff['maximum_absolute_difference'], delta)
                roundoff['different_float_leaves'] += 1
        else:
            assert type(expected) is type(actual) and expected == actual
    # CPU-dependent trigonometric roundoff can affect derived MSON values.
    # Every token/identifier remains exact. The original recorded inputs, their
    # hashes and all actual physics labels are kept byte-for-byte unchanged.
    check_reconstruction(json.loads(json.dumps(expected_inputs)), actual_inputs)
    old, old_report = bound_labels(sample_dir / 'old_physics.jsonl', root / 'stage2_diagnostics/labels_original')
    final, final_report = bound_labels(sample_dir / 'physics.jsonl', root / 'stage2_diagnostics/labels_selected')
    additional, additional_report = bound_labels(extra_dir / 'prepared/inputs.jsonl', extra_dir / 'labels')
    assert old_report['runtime_identities'] == final_report['runtime_identities'] == additional_report['runtime_identities']
    old = {row['group_id']: row for row in old.values()}
    final = {row['group_id']: row for row in final.values()}
    mappings = defaultdict(list)
    for row in mapping:
        mappings[row['ancestor_id']].append(row)

    counts, by_old_reliability, transactions = Counter(), Counter(), []
    for sample in samples:
        ancestor = sample['ancestor_id']
        initial_body = tuple(sample['old_body'])
        final_body = tuple(sample['output']['canonical_body'])
        # A token-identical no-op has one R observation within this analysis.
        # Do not manufacture an action gain from independently repeated R noise.
        known = {initial_body: old[ancestor]}
        geometry = {initial_body: sample['old_geometry']}
        if final_body != initial_body:
            known[final_body] = final[ancestor]
            geometry[final_body] = sample['proposal_geometry']
        for row in mappings[ancestor]:
            body = tuple(row['body'])
            assert body not in known
            known[body] = additional[row['trajectory_id']]
            geometry[body] = certify_geometry(decode_body(body, inverse))
        for index, trace in replay_feedback_trace(sample['output'], sample['old_body'], sample['num_atoms']):
            before, after = tuple(trace['old_body']), tuple(trace['proposal_body'])
            assert before in known and after in known
            decision = feedback_decision(trace['task'], before, after, geometry[before], geometry[after],
                                         known[before], known[after])
            outcome = 'helpful' if decision['accept_label'] is True else (
                      'not_helpful' if decision['accept_label'] is False else 'unknown')
            action = 'accepted' if trace['accepted'] else 'rejected'
            key = f'{trace["task"]}:{action}:{outcome}'
            counts[key] += 1
            by_old_reliability[f'{key}:old_reliable={decision["old_reliable"]}'] += 1
            transactions.append({'ancestor_id': ancestor, 'trace_index': index, 'task': trace['task'],
                                 'accepted': trace['accepted'], **decision})
    report = {'source_split': 'dev', 'training_use_allowed': False, 'requests': len(samples),
              'complete_transactions': len(transactions), 'counts': dict(counts),
              'by_transaction_input_reliability': dict(by_old_reliability),
              'R_observation_for_identical_tokens': 'initial observation retained; final observation used only when tokens change',
              'interpretation': 'fixed saved single transactions; not an accept-all rollout or a deployed alternate policy',
              'runtime_identities': old_report['runtime_identities'],
              'reconstructed_MSON_roundoff_check': roundoff,
              'source_files_sha256': {str(p): sha256(p) for p in [samples_file,
                  extra_dir / 'prepared/PROPOSALS_FINAL.json', extra_dir / 'prepared/proposal_map.jsonl',
                  root / 'stage2_diagnostics/labels_original/labels.jsonl',
                  root / 'stage2_diagnostics/labels_selected/labels.jsonl', extra_dir / 'labels/labels.jsonl']},
              'analysis_source_sha256': sha256(Path(__file__))}
    (extra_dir / 'JUDGEMENT_AUDIT_FINAL.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    (extra_dir / 'transaction_judgements.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in transactions))
    print(json.dumps(report))


if __name__ == '__main__':
    main()
