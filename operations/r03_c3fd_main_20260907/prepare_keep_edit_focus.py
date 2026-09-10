"""Freeze KEEP/EDIT source partitions and materialize continuous endpoints."""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil
import sys

SOURCE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SOURCE/'src'))
from scripts.run_post_refine_cycle import read_rows, write_rows, write_json, file_hash
from scripts.run_rsi_stages import materialize
from crystal_dlm.continuous_keep_edit import native_current, quantization_error, commit_patch


def partition(plans, old_plans):
    ancestors = {p['ancestor_id'] for p in old_plans}
    formulas = {p['plan_state']['reduced_formula'] for p in old_plans}
    extra = {p['plan_state']['reduced_formula'] for p in plans} - formulas
    ordered = sorted(extra, key=lambda f: hashlib.sha256(('keep_edit_focus_20260910_v1'+f).encode()).hexdigest())
    cut = len(ordered)//3
    roles = {f: ('train' if i < cut else 'dev' if i < 2*cut else 'final') for i, f in enumerate(ordered)}
    rows = []
    for p in plans:
        f = p['plan_state']['reduced_formula']; seen = p['ancestor_id'] in ancestors
        role = 'train' if f in formulas or seen else roles[f]
        rows.append(dict(ordinal=p['original_ordinal'], ancestor_id=p['ancestor_id'],
            reduced_formula=f, split=role, old_E_seen_source=seen, old_E_seen_formula=f in formulas))
    return rows


def register(previous, old, trial, root):
    previous, old, trial, root = map(lambda p: Path(p).resolve(), (previous, old, trial, root))
    fixed = previous/'rounds/round1/fit'
    if not (previous/'RANKED_RUN_PAUSED.json').exists(): raise ValueError('1000 run must be paused')
    plans = read_rows(fixed/'cohort/plans.jsonl'); old_plans = read_rows(old/'fit/cohort/plans.jsonl')
    if len(plans) != 1000 or len(old_plans) != 256: raise ValueError('unexpected frozen source count')
    split = partition(plans, old_plans)
    evidence = {}
    old_ids = {p['ancestor_id'] for p in old_plans}
    for k in range(1, 4):
        config = old/f'training_configs/E{k}.json'
        if not config.exists():
            options = [p for p in (old/'training_configs').glob('*.json')
                       if json.loads(p.read_text()).get('output_dir') == str(old/f'training/round{k}/E/result')]
            if len(options) != 1: raise ValueError('cannot resolve old E training config')
            config = options[0]
        cfg = json.loads(config.read_text()); data = Path(cfg['data']); rows = read_rows(data)
        if any(r['source_id'] not in old_ids or r['source_split'] != 'train' for r in rows):
            raise ValueError('old E3 source provenance escaped old256')
        receipt = old/f'training/round{k}/E/result/checkpoint/RSI_TRAINING_DONE.json'
        evidence[str(config)] = file_hash(config); evidence[str(data)] = file_hash(data)
        evidence[str(receipt)] = file_hash(receipt)
    if root.exists() and (root/'PREREGISTRATION.json').exists():
        raise ValueError('focus source registration already exists')
    root.mkdir(parents=True, exist_ok=True)
    budget = json.loads((previous/'BUDGET.json').read_text())
    if dt.datetime.now(dt.timezone.utc) >= dt.datetime.fromisoformat(budget['deadline_utc']):
        raise ValueError('original budget expired')
    write_rows(root/'SOURCE_SPLIT.jsonl', split)
    registration = dict(schema='keep_edit_focus_v1', previous_run=str(previous), old256_run=str(old),
        previous_trial=str(trial), fixed_input=str(fixed), old_editor=str(old/'training/round3/E/result/checkpoint'),
        plan_sha256=file_hash(SOURCE/'docs/r03_paper_story_20260907/KEEP_EDIT_FOCUS_PLAN_20260910.md'),
        plans_sha256=file_hash(fixed/'cohort/plans.jsonl'), old_plans_sha256=file_hash(old/'fit/cohort/plans.jsonl'),
        source_split_sha256=file_hash(root/'SOURCE_SPLIT.jsonl'), split_counts=dict(Counter(r['split'] for r in split)),
        old_E_source_provenance=evidence, old_budget_sha256=file_hash(previous/'BUDGET.json'),
        absolute_deadline_utc=budget['deadline_utc'], created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        final_quality_sealed_until_policy_frozen=True, original_source_split='MP20_train',
        head_training=dict(seed=20260910, learning_rate=1e-5, epochs=32, snapshots=[4,8,16,32],
            loss='SmoothL1_signed_novel_Stable_plus_novel_MS_gain', batch_size=256,
            thresholds=[0.,.05,.1,.2], frozen='all_except_quality_head'))
    write_json(root/'PREREGISTRATION.json', registration)
    shutil.copy2(previous/'BUDGET.json', root/'BUDGET.json')
    pipe = json.loads((previous/'RAW0_PIPELINE.json').read_text()); pipe['run_root'] = str(root)
    pipe['resources'].update(gpus_before_extra_window=5, gpus_after_extra_window=5)
    write_json(root/'RAW0_PIPELINE.json', pipe)
    write_json(root/'PHYSICS_SOURCE_PIN.json', dict(source=pipe['source_root'], identity=pipe['source_identity']))
    fit = root/'fit'; fit.mkdir(exist_ok=True)
    shutil.copytree(fixed/'cohort', fit/'cohort')
    for name in ('PREPARATION_FINAL.json', 'parents.jsonl'):
        if not (fit/'cohort'/name).exists(): shutil.copy2(previous/'fit/cohort'/name, fit/'cohort'/name)
    shutil.copytree(fixed/'current', fit/'current')
    spec = json.loads((fixed/'RUN_SPEC.json').read_text())
    spec.update(run_root=str(fit), run_id='keep_edit_focus_20260910', editor_trial=True,
                collect_training_proposals=False, training_parent_root=str(fit/'cohort'))
    spec['assets']['editor_checkpoint'] = registration['old_editor']
    spec['assets']['nu_cache'] = str(root/'nu_cache')
    spec['assets']['official_cache'] = str(previous/'fit_hull/official_mp_cache')
    spec['updated_checkpoint_receipts'].pop('E', None)
    spec['resources']['budget_receipt'] = str(root/'BUDGET.json')
    spec['execution_policy'].update(single_GPUs=5, parallel_main_GPUs=3, parallel_other_GPUs=2, training_GPUs=1)
    write_json(fit/'RUN_SPEC.json', spec); write_json(root/'RUN_SPEC.json', spec)
    token = read_rows(fixed/'current/inputs.jsonl'); audit = []
    for p, before, role in zip(plans, token, split, strict=True):
        i = p['original_ordinal']; path = fixed/f'refined/records/{i:04d}.json'
        wrapper = json.loads(path.read_text()); native, trace = native_current(wrapper, before)
        native['trajectory_id'] = f'keep_edit_focus_20260910:native:{i}'
        write_json(fit/f'native/records/{i:04d}.json', dict(record=native, continuous_trace=trace,
            source_path=str(path), source_sha256=file_hash(path), source_partition=role))
        error = quantization_error(native, before) if trace['source'] == 'continuous_F' and trace['editable'] else None
        audit.append(dict(ordinal=i, split=role['split'], trace=trace, quantization=error))
    materialize(spec, 'native')
    write_rows(root/'REPRESENTATION_AUDIT.jsonl', audit)
    write_json(root/'PREPARATION_FINAL.json', dict(registration_sha256=file_hash(root/'PREREGISTRATION.json'),
        native_inputs_sha256=file_hash(fit/'native/inputs.jsonl'), audit_sha256=file_hash(root/'REPRESENTATION_AUDIT.jsonl'),
        split_counts=registration['split_counts'], representation_counts=dict(Counter(a['trace']['source'] for a in audit)),
        noneditable=[a['ordinal'] for a in audit if not a['trace']['editable']]))
    print((root/'PREPARATION_FINAL.json').read_text(), flush=True)


def finish_registration(root):
    """Complete manifests after a recorded preparation-only metadata failure."""
    root = Path(root); reg = json.loads((root/'PREREGISTRATION.json').read_text())
    fit = root/'fit'; previous = Path(reg['previous_run']); fixed = Path(reg['fixed_input'])
    if (root/'PREPARATION_FINAL.json').exists(): raise ValueError('preparation already finalized')
    if (file_hash(root/'SOURCE_SPLIT.jsonl') != reg['source_split_sha256'] or
            file_hash(fixed/'cohort/plans.jsonl') != reg['plans_sha256']):
        raise ValueError('registered source changed')
    for name in ('PREPARATION_FINAL.json', 'parents.jsonl'):
        original = previous/'fit/cohort'/name; destination = fit/'cohort'/name
        if destination.exists() and file_hash(destination) != file_hash(original):
            raise ValueError('existing parent manifest differs')
        if not destination.exists(): shutil.copy2(original, destination)
    spec = json.loads((fit/'RUN_SPEC.json').read_text())
    split = read_rows(root/'SOURCE_SPLIT.jsonl'); token = read_rows(fit/'current/inputs.jsonl'); audit = []
    for role, before in zip(split, token, strict=True):
        i = role['ordinal']; wrapper = json.loads((fit/f'native/records/{i:04d}.json').read_text())
        if file_hash(wrapper['source_path']) != wrapper['source_sha256']: raise ValueError('native source changed')
        expected, trace = native_current(json.loads(Path(wrapper['source_path']).read_text()), before)
        expected['trajectory_id'] = f'keep_edit_focus_20260910:native:{i}'
        if wrapper['record'] != expected or wrapper['continuous_trace'] != trace:
            raise ValueError('retained native endpoint differs')
        error = quantization_error(expected, before) if trace['source'] == 'continuous_F' and trace['editable'] else None
        audit.append(dict(ordinal=i, split=role['split'], trace=trace, quantization=error))
    materialize(spec, 'native'); write_rows(root/'REPRESENTATION_AUDIT.jsonl', audit)
    write_json(root/'PREPARATION_FINAL.json', dict(registration_sha256=file_hash(root/'PREREGISTRATION.json'),
        native_inputs_sha256=file_hash(fit/'native/inputs.jsonl'), audit_sha256=file_hash(root/'REPRESENTATION_AUDIT.jsonl'),
        split_counts=reg['split_counts'], representation_counts=dict(Counter(a['trace']['source'] for a in audit)),
        noneditable=[a['ordinal'] for a in audit if not a['trace']['editable']],
        metadata_recovery='completed_missing_training_parent_files_with_original_hashes'))
    print((root/'PREPARATION_FINAL.json').read_text(), flush=True)


def hybrid(root, candidate_root, output_stage):
    from transformers import AutoTokenizer
    root, candidate_root = Path(root), Path(candidate_root)
    spec = json.loads((root/'fit/RUN_SPEC.json').read_text()); fit = root/'fit'
    token = read_rows(fit/'current/inputs.jsonl'); proposal = read_rows(candidate_root/'proposal/inputs.jsonl')
    tokenizer = AutoTokenizer.from_pretrained(spec['assets']['b0_checkpoint'], trust_remote_code=True)
    inverse = {int(v): k for k, v in tokenizer.get_vocab().items()}; traces = []
    for a, b in zip(token, proposal, strict=True):
        i = a['original_ordinal']
        if b['original_ordinal'] != i: raise ValueError('hybrid candidate order changed')
        native = json.loads((fit/f'native/records/{i:04d}.json').read_text())
        value, trace = commit_patch(native['record'], a.get('body_token_ids', []), b.get('body_token_ids', []),
            inverse, editable=native['continuous_trace']['editable'])
        value['trajectory_id'] = f'keep_edit_focus_20260910:{output_stage}:{i}'
        write_json(fit/f'{output_stage}/records/{i:04d}.json', dict(record=value, commit_trace=trace,
            proposal_source=str(candidate_root/'proposal/inputs.jsonl')))
        traces.append(dict(ordinal=i, **trace))
    materialize(spec, output_stage)
    write_rows(fit/output_stage/'COMMIT_AUDIT.jsonl', traces)
    print(json.dumps(dict(stage=output_stage, counts=dict(Counter(t['reason'] for t in traces)),
        inputs_sha256=file_hash(fit/output_stage/'inputs.jsonl'))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True); parser.add_argument('--previous')
    parser.add_argument('--old'); parser.add_argument('--trial'); parser.add_argument('--candidate-root')
    parser.add_argument('--output-stage', default='hybrid_proposal')
    parser.add_argument('--finish-registration', action='store_true')
    args = parser.parse_args()
    if args.finish_registration: finish_registration(args.root)
    elif args.candidate_root: hybrid(args.root, args.candidate_root, args.output_stage)
    else: register(args.previous, args.old, args.trial, args.root)
