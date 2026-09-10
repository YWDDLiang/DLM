"""Run the preregistered E seed repetitions with one frozen utility policy."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import sys

SOURCE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SOURCE / 'src'))
from scripts.run_post_refine_cycle import read_rows, write_rows, write_json, file_hash, load_config, validate_rsi_checkpoint
from scripts.run_rsi_stages import materialize
from crystal_dlm.post_refine_contract import derived_seed
from replay_keep_edit_export import clone_decoder_inputs

STREAMS = ('keep_edit_focus_repeat1', 'keep_edit_focus_repeat2')


def register(root):
    path = root / 'REPEAT_SEED_REGISTRATION.json'
    if path.exists():
        raise ValueError('repeat streams already registered')
    if (root / 'UTILITY_DEV_SELECTION_FINAL.json').exists():
        raise ValueError('repeat streams must be registered before the first DEV comparison')
    write_json(path, dict(schema='frozen_utility_seed_repetitions_v1',
        created_utc=dt.datetime.now(dt.timezone.utc).isoformat(), streams=list(STREAMS), original_stream='E',
        plans_sha256=file_hash(root / 'fit/cohort/plans.jsonl'),
        current_sha256=file_hash(root / 'fit/current/inputs.jsonl'),
        native_sha256=file_hash(root / 'fit/native/inputs.jsonl'),
        source_split_sha256=file_hash(root / 'SOURCE_SPLIT.jsonl'),
        plan_sha256=file_hash(SOURCE / 'docs/r03_paper_story_20260907/KEEP_EDIT_FOCUS_PLAN_20260910.md'),
        source_commit=(SOURCE / '_CODE_READY').read_text().strip(),
        decoder_shards=2, editor_batch_size=64, parameter_or_margin_reselection=False,
        report_all_registered_streams=True, new_G_or_F_sampling=False))
    print(json.dumps(dict(registered=str(path), sha256=file_hash(path))), flush=True)


def prepare(root, index):
    seed_reg = json.loads((root / 'REPEAT_SEED_REGISTRATION.json').read_text())
    stream = seed_reg['streams'][index - 1]
    if seed_reg['streams'] != list(STREAMS):
        raise ValueError('registered repeat streams changed')
    for name, path in (('plans_sha256', root / 'fit/cohort/plans.jsonl'),
                       ('current_sha256', root / 'fit/current/inputs.jsonl'),
                       ('native_sha256', root / 'fit/native/inputs.jsonl'),
                       ('source_split_sha256', root / 'SOURCE_SPLIT.jsonl')):
        if file_hash(path) != seed_reg[name]:
            raise ValueError('frozen repeat input changed:' + name)
    export_path = root / 'models/selected_utility/EXPORT_FINAL.json'
    export = json.loads(export_path.read_text())
    if not export['all_1000_continuous_outputs_reproduced']:
        raise ValueError('repeat needs a verified full-model utility export')
    checkpoint = Path(export['checkpoint'])
    validate_rsi_checkpoint(checkpoint, 'E')
    panel = root / f'repeats/repeat{index}'
    clone_decoder_inputs(root / 'fit', panel)
    spec = json.loads((root / 'fit/RUN_SPEC.json').read_text())
    spec.update(run_root=str(panel), run_id=f'keep_edit_focus:repeat{index}', editor_seed_stream=stream,
                focus_role='frozen_policy_seed_repetition')
    if spec['parallelism']['editor_batch_size'] != seed_reg['editor_batch_size']:
        raise ValueError('repeat decoder batch size changed')
    spec['assets']['editor_checkpoint'] = str(checkpoint)
    spec['assets']['official_cache'] = json.loads((root / 'REFERENCE_CACHE_OVERRIDE.json').read_text())['directory']
    spec['updated_checkpoint_receipts']['E'] = dict(path=str(checkpoint),
        receipt_sha256=file_hash(checkpoint / 'RSI_TRAINING_DONE.json'))
    write_json(panel / 'RUN_SPEC.json', spec)
    write_json(panel / 'REPETITION_BINDING.json', dict(stream=stream, index=index,
        selection_sha256=file_hash(root / 'FROZEN_SELECTION.json'),
        export_sha256=file_hash(export_path), seed_registration_sha256=file_hash(root / 'REPEAT_SEED_REGISTRATION.json'),
        config_sha256=file_hash(panel / 'RUN_SPEC.json'), original_seed_count=1000))
    print(json.dumps(dict(config=str(panel / 'RUN_SPEC.json'), stream=stream)), flush=True)


def decide(root, index, completion):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import torch
    from crystal_dlm.expert_edit import load_editor_model
    from crystal_dlm.utility_acceptance import judge_completed_proposals, continuous_decision
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('canonical repeated utility decisions need a GPU allocation')
    torch.cuda.set_device(0)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    panel = root / f'repeats/repeat{index}'
    spec = load_config(panel / 'RUN_SPEC.json')
    binding = json.loads((panel / 'REPETITION_BINDING.json').read_text())
    if binding['selection_sha256'] != file_hash(root / 'FROZEN_SELECTION.json'):
        raise ValueError('repeat decision policy was reselected')
    checkpoint = Path(spec['assets']['editor_checkpoint'])
    validate_rsi_checkpoint(checkpoint, 'E')
    policy = json.loads((checkpoint / 'QUALITY_UTILITY_POLICY.json').read_text())
    if policy['selection_sha256'] != binding['selection_sha256']:
        raise ValueError('repeat checkpoint selection differs')
    for shard in range(2):
        if not (panel / f'proposal/worker_{shard}_DONE.json').exists():
            raise ValueError('repeat proposal decoding is incomplete')
    plans = read_rows(panel / 'cohort/plans.jsonl')
    current = read_rows(panel / 'current/inputs.jsonl')
    proposal = read_rows(panel / 'proposal/inputs.jsonl')
    views, traces, compared = [], {}, []
    for plan, before, after in zip(plans, current, proposal, strict=True):
        i = plan['original_ordinal']
        trace = json.loads((panel / f'proposal/records/{i:04d}.json').read_text())['editor_trace']
        traces[i] = trace
        if before.get('body_token_ids') and after.get('body_token_ids'):
            expected = derived_seed(str(plan['body_noise_seed']), binding['stream'])
            original = derived_seed(str(plan['body_noise_seed']), 'E')
            if trace['sampling_seed'] != expected or expected == original:
                raise ValueError('repeat did not actually change its E seed')
            compared.append(i)
            views.append(dict(ordinal=i, prompt=plan['body_prompt'], current_tokens=before['body_token_ids'],
                proposal_tokens=after['body_token_ids'], num_sites=plan['plan_state']['N'],
                action_positions=trace.get('action', {}).get('positions', [])))
    model, tokenizer = load_editor_model(spec['assets']['base_model'], checkpoint, torch.device('cuda', 0))
    raw = judge_completed_proposals(model, tokenizer, views)
    inverse = {int(v): k for k, v in tokenizer.get_vocab().items()}
    margin = policy['raw_utility_margin']
    decisions = []
    for plan, before in zip(plans, current, strict=True):
        i = plan['original_ordinal']
        native = json.loads((root / f'fit/native/records/{i:04d}.json').read_text())
        record, decision = continuous_decision(native, before.get('body_token_ids', []), traces[i], inverse, raw.get(i), margin)
        record['trajectory_id'] = f'{spec["run_id"]}:edited:{i}'
        # The legacy decode-stage edited view stays in its original directory;
        # this canonical stage is the registered deployment output.
        write_json(panel / f'continuous_edited/records/{i:04d}.json', dict(record=record, editor_trace=decision))
        decisions.append(dict(ordinal=i, **decision))
    materialize(spec, 'continuous_edited')
    write_json(panel / 'DECISION_BINDING.json', dict(policy=policy, decisions=decisions,
        native_inputs_sha256=file_hash(root / 'fit/native/inputs.jsonl'),
        proposal_inputs_sha256=file_hash(panel / 'proposal/inputs.jsonl'), model_scores_only=True))
    write_rows(panel / 'CANONICAL_UTILITY_SCORES.jsonl', [dict(ordinal=i, raw_utility=v) for i, v in raw.items()])
    report = dict(index=index, stream=binding['stream'], requests=len(plans), actual_seed_changes=len(compared),
        model_proposals=sum(int(t.get('proposal_generated', False)) for t in traces.values()),
        actual_edits=sum(int(d['actual_edit']) for d in decisions), raw_margin=margin,
        selection_sha256=binding['selection_sha256'],
        checkpoint_receipt_sha256=file_hash(checkpoint / 'RSI_TRAINING_DONE.json'),
        inputs_sha256=file_hash(panel / 'continuous_edited/inputs.jsonl'),
        decision_sha256=file_hash(panel / 'DECISION_BINDING.json'), physical_labels_used_to_accept=False)
    write_json(completion / 'REPEAT_DECISION_FINAL.json', report)
    print(json.dumps(report), flush=True)


def report(root, index):
    from evaluate_keep_edit_utility import summary
    panel = root / f'repeats/repeat{index}'
    binding = json.loads((panel / 'REPETITION_BINDING.json').read_text())
    if binding['selection_sha256'] != file_hash(root / 'FROZEN_SELECTION.json'):
        raise ValueError('repeat summary policy identity differs')
    reports = {role: summary(root, panel, role, after_stage='continuous_edited')
        for role in ('train', 'dev', 'final', 'all')}
    final = reports['final']
    result = dict(index=index, stream=binding['stream'], reports=reports,
        primary_unseen_FINAL_SUN_and_MSUN_gain=final['delta']['SUN']>0 and final['delta']['MSUN']>0,
        selection_sha256=binding['selection_sha256'], no_policy_reselection=True)
    write_json(panel / 'REPEAT_COMPARISON_FINAL.json', result)
    print(json.dumps(dict(index=index, final_counts=final['counts'], KEEP=final['KEEP'], delta=final['delta'],
        primary_pass=result['primary_unseen_FINAL_SUN_and_MSUN_gain'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--mode', choices=['register', 'prepare', 'decide', 'report'], required=True)
    parser.add_argument('--index', type=int, choices=[1, 2])
    parser.add_argument('--completion-dir', type=Path)
    args = parser.parse_args()
    if args.mode == 'register':
        register(args.root)
    elif args.mode == 'prepare':
        if args.index is None:
            parser.error('prepare requires a repeat index')
        prepare(args.root, args.index)
    elif args.mode == 'report':
        if args.index is None:
            parser.error('report requires a repeat index')
        report(args.root, args.index)
    else:
        if args.index is None or args.completion_dir is None:
            parser.error('decide requires a repeat index and a component completion directory')
        decide(args.root, args.index, args.completion_dir)
