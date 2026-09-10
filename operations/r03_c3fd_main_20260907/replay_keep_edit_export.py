"""Prepare and verify an actual decoder replay of the exported utility editor."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

SOURCE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SOURCE / 'src'))
from scripts.run_post_refine_cycle import read_rows, write_json, file_hash, validate_rsi_checkpoint
from scripts.run_rsi_stages import scores, score_directory


def prepare(root):
    fit = root / 'fit'
    export = root / 'models/selected_utility/EXPORT_FINAL.json'
    receipt = json.loads(export.read_text())
    if not receipt['all_1000_continuous_outputs_reproduced']:
        raise ValueError('the selected utility checkpoint has not passed reload verification')
    checkpoint = Path(receipt['checkpoint'])
    validate_rsi_checkpoint(checkpoint, 'E')
    panel = root / 'decoder_replay'
    if panel.exists():
        raise ValueError('decoder replay already has a registration')
    panel.mkdir()
    shutil.copytree(fit / 'cohort', panel / 'cohort')
    current = panel / 'current'
    current.mkdir()
    for name in ('inputs.jsonl', 'FEEDBACK_MANIFEST.json', 'SCORING_DIRECTORY.json'):
        original = fit / 'current' / name
        if original.exists():
            shutil.copy2(original, current / name)
    observed = score_directory(fit, 'current')
    relative = observed.relative_to(fit / 'current')
    shutil.copytree(observed, current / relative)
    # The decoder reads current quality only for the original known-SUN guard.
    # Byte-identical input and score files preserve that decision context.
    scores(panel, 'current')
    spec = json.loads((fit / 'RUN_SPEC.json').read_text())
    spec.update(run_root=str(panel), run_id='keep_edit_focus:decoder_replay',
                focus_replay_role='same_inputs_scope_seed_and_batch_shape_with_exported_quality_head')
    spec['assets']['editor_checkpoint'] = str(checkpoint)
    spec['updated_checkpoint_receipts']['E'] = dict(path=str(checkpoint),
        receipt_sha256=file_hash(checkpoint / 'RSI_TRAINING_DONE.json'))
    write_json(panel / 'RUN_SPEC.json', spec)
    write_json(panel / 'REPLAY_REGISTRATION.json', dict(
        reference=str(fit / 'proposal'), export_receipt_sha256=file_hash(export),
        current_inputs_sha256=file_hash(current / 'inputs.jsonl'),
        plans_sha256=file_hash(panel / 'cohort/plans.jsonl'),
        current_scores_sha256=file_hash(observed / 'attempt_results.jsonl'),
        decoder_shards=2, editor_batch_size=spec['parallelism']['editor_batch_size'],
        seed_derivation="derived_seed(str(body_noise_seed), 'E')",
        new_G_or_F_sampling=False, endpoint_quality_used_for_comparison=False))
    print(json.dumps(dict(config=str(panel / 'RUN_SPEC.json'))), flush=True)


def verify(root):
    panel = root / 'decoder_replay'
    reg = json.loads((panel / 'REPLAY_REGISTRATION.json').read_text())
    for rank in range(reg['decoder_shards']):
        if not (panel / f'proposal/worker_{rank}_DONE.json').exists():
            raise ValueError('decoder replay is incomplete')
    if file_hash(panel / 'current/inputs.jsonl') != reg['current_inputs_sha256']:
        raise ValueError('decoder replay changed the original current inputs')
    if file_hash(panel / 'cohort/plans.jsonl') != reg['plans_sha256']:
        raise ValueError('decoder replay changed the original plans')
    fields = ('current_tokens', 'proposal_tokens', 'sampled_attempt_tokens', 'initial_masked_tokens',
              'known_sun', 'proposal_generated', 'proposal_failure', 'sampling_seed',
              'sampling_temperature', 'forward_calls', 'action', 'learned_mode',
              'mode_logits', 'unshifted_mode_logits', 'sampling_trace', 'upstream_failure')
    mismatches = []
    reference_calls = replay_calls = proposals = 0
    plans = read_rows(panel / 'cohort/plans.jsonl')
    pins = {}
    for plan in plans:
        ordinal = plan['original_ordinal']
        old_path = root / f'fit/proposal/records/{ordinal:04d}.json'
        new_path = panel / f'proposal/records/{ordinal:04d}.json'
        before = json.loads(old_path.read_text())['editor_trace']
        after = json.loads(new_path.read_text())['editor_trace']
        different = [name for name in fields if before.get(name) != after.get(name)]
        if different:
            mismatches.append(dict(ordinal=ordinal, fields=different))
        reference_calls += before.get('forward_calls', 0)
        replay_calls += after.get('forward_calls', 0)
        proposals += int(after.get('proposal_generated', False))
        pins[str(ordinal)] = dict(reference=file_hash(old_path), replay=file_hash(new_path))
    report = dict(schema='exported_utility_decoder_replay_v1', requests=len(plans),
        all_proposals_scope_seeds_and_conditional_logp_identical=not mismatches,
        mismatches=mismatches, proposal_generated=proposals,
        reference_forward_rows=reference_calls, replay_forward_rows=replay_calls,
        registration_sha256=file_hash(panel / 'REPLAY_REGISTRATION.json'),
        config_sha256=file_hash(panel / 'RUN_SPEC.json'), record_bindings=pins,
        acceptance_outputs_excluded='exported head is applied by canonical continuous_decision, verified in EXPORT_FINAL.json',
        new_physics_calls=0)
    write_json(panel / 'DECODER_REPLAY_FINAL.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'record_bindings'}), flush=True)
    if mismatches:
        raise ValueError('exported head decoder replay changed proposal execution')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--mode', choices=['prepare', 'verify'], required=True)
    args = parser.parse_args()
    (prepare if args.mode == 'prepare' else verify)(args.root)
