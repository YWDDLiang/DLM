"""Audit completed clean-1000 updates without changing training inputs or weights."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def stable(quality):
    hull = quality.get('hull')
    return bool(quality.get('reliable') and hull is not None and math.isfinite(hull) and hull <= 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--round', type=int, choices=(1, 2, 3), required=True)
    parser.add_argument('--branch', choices=('G', 'E'), required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    frozen = read(root/'FROZEN_METHOD.json')
    source = Path(frozen['source'])
    sys.path[:0] = [str(source/'operations/r03_c3fd_main_20260907'), str(source/'src')]
    from run_component import verify_deployed_source
    from scripts.run_post_refine_cycle import validate_rsi_checkpoint
    from crystal_dlm.rsi_minibatch import epoch_indices, has_head_supervision

    require(verify_deployed_source(source) == frozen['source_identity'], 'scientific source changed')
    require(frozen['requests'] == 1000, 'registered panel is not 1000')
    panel_plans = root/'fit/cohort/plans.jsonl'
    require(digest(panel_plans) == frozen['plans_sha256'], 'registered Plans changed')
    plans = rows(panel_plans)
    require(len(plans) == 1000 and all(p['source_split'] == 'train' for p in plans), 'invalid TRAIN panel')
    by_source = {p['ancestor_id']: p for p in plans}
    require(len(by_source) == 1000, 'duplicate TRAIN ancestors')
    training = root/'training'/f'round{args.round}'/args.branch
    output = training/'result'
    require((training/'_SUCCESS').is_file() and (output/'_SUCCESS').is_file(), 'training is incomplete')
    checkpoint = output/'checkpoint'
    receipt = validate_rsi_checkpoint(checkpoint, args.branch)
    contract = receipt['contract']
    require(read(output/'TRAIN_CONFIG.json') == contract, 'receipt and actual training contract differ')
    overrides = root/'TRAINING_CONFIG_OVERRIDES.json'
    config_name = read(overrides).get(f'{args.branch}{args.round}') if overrides.exists() else None
    config_path = root/'training_configs'/(config_name or f'TRAIN_{args.branch}{args.round}.json')
    require(config_path.parent == root/'training_configs', 'invalid training config override')
    spec = read(config_path)
    require(all(contract.get(k) == v for k, v in spec.items()), 'actual contract differs from dispatch config')
    require(spec['replay_data'] == [] and spec['bounded_minibatch_training'], 'unexpected training mode')
    require(not spec.get('editor_dense_t2t') and not spec.get('continue_heads_after_content_KL'), 'frozen recipe changed')
    world = spec['training_gpus']
    require(contract['world_size'] == world and spec['batch_size']*world == 192, 'actual GPU/batch contract differs')
    data_path = Path(spec['data']).resolve()
    require(root in data_path.parents, 'data is outside the clean run')
    manifest_path = data_path.with_name(f'PAIRS_{args.branch}_FINAL.json')
    manifest = read(manifest_path)
    require(manifest['source_split'] == 'train' and digest(manifest_path) == contract['pair_manifest_sha256'],
            'training manifest changed or is not TRAIN')
    require(digest(data_path) == contract['data_sha256'] == manifest['files_sha256'][data_path.name], 'data changed')
    records = rows(data_path)
    require(len(records) == manifest['examples'] and records, 'training row count differs')
    pair_ids = [row['pair_id'] for row in records]
    require(len(pair_ids) == len(set(pair_ids)), 'duplicate training pair IDs')
    expected_round = args.round-1 if args.branch == 'G' else args.round
    for row in records:
        plan = by_source.get(row['source_id'])
        require(plan is not None and row['source_split'] == 'train' and row['source_round'] == expected_round,
                'training source or round changed')
        require((row['source_row_idx'], row['prompt'], row['plan_state']) ==
                (plan['source_row_idx'], plan['body_prompt'], plan['plan_state']), 'Plan conditioning changed')
    pins = {str(p.relative_to(root)): digest(p) for p in
            (panel_plans, config_path, data_path, manifest_path, output/'TRAIN_CONFIG.json', checkpoint/'RSI_TRAINING_DONE.json')}
    for stage_name, identity in manifest['stage_evidence'].items():
        stage = Path(stage_name).resolve()
        require(root in stage.parents, 'feedback is outside the clean run')
        for filename, key in [('inputs.jsonl', 'inputs_sha256'), ('scoring/result/attempt_results.jsonl', 'scores_sha256')]:
            path = stage/filename
            require(digest(path) == identity[key], 'source feedback changed')
            pins[str(path.relative_to(root))] = digest(path)
    exposures = []
    totals = {key: Counter() for key in ('pair_visits', 'content_updated_pair_visits', 'head_updated_pair_visits')}
    require(len(list(output.glob('EXPOSURE_rank*.json'))) == world, 'exposure rank count differs')
    for rank in range(world):
        path = output/f'EXPOSURE_rank{rank}.json'
        exposure = read(path)
        require(exposure['batch_size'] == spec['batch_size'] and exposure['epochs_cap'] == spec['epochs'],
                'rank exposure contract changed')
        require(exposure['content_optimizer_steps'] == receipt['content_optimizer_steps'] and
                exposure['head_optimizer_steps'] == receipt['head_optimizer_steps'], 'ranks disagree on applied updates')
        for key, total in totals.items():
            require(set(exposure[key]) <= set(pair_ids) and all(v > 0 for v in exposure[key].values()),
                    'exposure contains unknown or invalid pair visits')
            total.update(exposure[key])
        exposures.append({k: v for k, v in exposure.items() if k not in totals})
        pins[str(path.relative_to(root))] = digest(path)
    inspected, content, heads = (totals[key] for key in totals)
    require(all(content[p] <= inspected[p] and heads[p] <= inspected[p] for p in pair_ids),
            'applied updates exceed inspected visits')
    content_ids = {row['pair_id'] for row in records if (args.branch == 'G' or row.get('action_positions')) and
                   ((row.get('chosen_tokens') is not None and row.get('rejected_tokens') is not None) or
                    row.get('healthy_anchor_tokens' if args.branch == 'G' else 'content_target_tokens') is not None)}
    head_ids = {row['pair_id'] for row in records if args.branch == 'E' and has_head_supervision(row)}
    require(set(content) <= content_ids and set(heads) <= head_ids, 'updated visits have no corresponding supervision')

    def coverage(ids):
        return dict(eligible_rows=len(ids), inspected_unique=sum(inspected[p] > 0 for p in ids),
                    content_updated_unique=sum(content[p] > 0 for p in ids), content_visits=sum(content[p] for p in ids),
                    head_updated_unique=sum(heads[p] > 0 for p in ids), head_visits=sum(heads[p] for p in ids),
                    min_content_visits=min((content[p] for p in ids), default=0),
                    max_content_visits=max((content[p] for p in ids), default=0),
                    min_head_visits=min((heads[p] for p in ids), default=0),
                    max_head_visits=max((heads[p] for p in ids), default=0))

    groups = {'all_training_rows': set(pair_ids), 'eligible_content': content_ids, 'eligible_heads': head_ids}
    for name, predicate in (
        ('Stable_promotion_content', lambda q: not stable(q['before']) and stable(q['after']) and q['chosen'] == 'after'),
        ('Stable_damage_rejection_content', lambda q: stable(q['before']) and not stable(q['after']) and q['chosen'] == 'before'),
    ):
        groups[name] = {row['pair_id'] for row in records if row['pair_id'] in content_ids and predicate(row['preference'])}
    expected = Counter()
    for _, indices in epoch_indices(len(records), batch_size=spec['batch_size'], world=world,
                                   epochs=spec['epochs'], seed=spec['seed']):
        expected.update(pair_ids[i] for i in indices)
    full_content = all(content[p] == expected[p] for p in content_ids)
    full_heads = all(heads[p] == expected[p] for p in head_ids)
    history = receipt['loss_history']
    all_kl_stop = [x['KL_stop'] for x in exposures]
    require(len(set(all_kl_stop)) == 1, 'ranks disagree on KL stop')
    stop = ('reference_KL' if all_kl_stop[0] else 'epoch_cap' if full_content and full_heads else
            'wall_time_observed_after_update' if history[-1]['seconds'] > spec['max_training_seconds'] else 'unresolved')
    require(stop != 'unresolved', 'partial training has no supported stopping explanation')
    report = dict(schema='clean_1000_completed_training_analysis_v1', update=f'{args.branch}{args.round}',
        purpose='derived_report_only_not_training_feedback', source_identity=frozen['source_identity'],
        analyzer_sha256=digest(__file__), checkpoint=str(checkpoint), checkpoint_validation_passed=True,
        input_bindings=pins, training_config=spec, actual_world_size=world, actual_source_batch=world*spec['batch_size'],
        training_rows=len(records), unique_TRAIN_sources=len({row['source_id'] for row in records}),
        objective_levels=dict(Counter(row['objective_level'] for row in records)),
        origins=dict(Counter(row['origin'] for row in records)),
        optimizer_steps=receipt['optimizer_steps'], content_optimizer_steps=receipt['content_optimizer_steps'],
        head_optimizer_steps=receipt['head_optimizer_steps'], parameter_delta_squared=receipt['parameter_delta_squared'],
        training_seconds=receipt['training_seconds'], max_logged_rank0_GPU_GB=max(x['peak_GPU_GB'] for x in history),
        stopping_evidence=stop, rank_stop_details=exposures,
        full_epoch_cap_content_coverage=full_content, full_epoch_cap_head_coverage=full_heads,
        coverage={name: coverage(ids) for name, ids in groups.items()},
        exposure_by_pair={pair: {key: total[pair] for key, total in totals.items()} for pair in pair_ids})
    target = root/'analysis'/f'{args.branch}{args.round}_COMPLETED_TRAINING.json'
    target.parent.mkdir(exist_ok=True)
    if target.exists():
        require(read(target) == report, 'existing derived training report differs; inspect without overwriting')
    else:
        with target.open('x', encoding='utf-8') as stream:
            stream.write(json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+'\n')
    compact = {k: v for k, v in report.items() if k not in ('exposure_by_pair', 'input_bindings')}
    compact.update(report=str(target), report_sha256=digest(target))
    print(json.dumps(compact, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
