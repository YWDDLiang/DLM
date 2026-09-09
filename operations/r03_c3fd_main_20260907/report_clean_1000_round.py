"""Read completed frozen TRAIN outputs and archive paired Stable/SUN results."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--round', type=int, choices=range(4), required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    frozen = json.loads((root/'FROZEN_METHOD.json').read_text())
    source = Path(frozen['source']).resolve()
    sys.path.insert(0, str(source/'operations/r03_c3fd_main_20260907'))
    sys.path.insert(0, str(source/'src'))
    from run_component import verify_deployed_source
    from editor_trial_analysis import flags, paired
    from scripts.run_post_refine_cycle import read_rows
    from scripts.run_rsi_stages import score_directory, scores

    require(verify_deployed_source(source) == frozen['source_identity'], 'frozen scientific source changed')
    require(frozen['requests'] == 1000, 'this report requires the registered 1000 panel')
    base_plans = read_rows(root/'fit/cohort/plans.jsonl')
    require(len(base_plans) == 1000 and all(p['source_split'] == 'train' for p in base_plans),
            'TRAIN denominator or source split changed')
    require(digest(root/'fit/cohort/plans.jsonl') == frozen['plans_sha256'], 'registered Plans changed')
    indices = [p['sample_idx'] for p in base_plans]
    require(indices == list(range(1000)), 'registered ordinal ordering changed')
    bindings = {}

    def panel_path(index):
        return root/'fit' if index == 0 else root/'rounds'/f'round{index}'/'fit'

    def stage(panel, name):
        require(digest(panel/'cohort/plans.jsonl') == frozen['plans_sha256'], 'round Plans changed')
        directory = score_directory(panel, name)
        require((directory.parent/'_SUCCESS').is_file(), f'incomplete scoring component: {directory}')
        measured = scores(panel, name)
        inputs = read_rows(panel/name/'inputs.jsonl')
        require(len(measured) == len(inputs) == 1000, f'incomplete denominator: {panel}/{name}')
        require([r['sample_idx'] for r in inputs] == indices and
                [r['sample_idx'] for r in measured] == indices, 'input/score ordering changed')
        for record, plan in zip(inputs, base_plans, strict=True):
            require(record['source_split'] == 'train' and record['source_row_idx'] == plan['source_row_idx'],
                    'input source identity differs from its Plan')
        counts = json.loads((directory/'RANKED_METRICS.json').read_text())
        quality = [flags(row) for row in measured]
        require(counts['requested'] == 1000 and
                counts['strict_Stable'] == sum(q['Stable'] for q in quality) and
                counts['reliable_SUN'] == sum(q['SUN'] for q in quality), 'primary count mismatch')
        require(sum(counts[k] for k in ['strict_Stable', 'exclusive_MetaStable', 'unstable',
                    'unknown', 'known_physical_failure']) == 1000, 'quality classes do not cover the denominator')
        pins = {str(p.relative_to(root)): digest(p) for p in [
            panel/'cohort/plans.jsonl', panel/name/'inputs.jsonl', panel/name/'FEEDBACK_MANIFEST.json',
            panel/name/'labeling/result/LABEL_FINAL.json', directory/'attempt_results.jsonl',
            directory/'RANKED_METRICS.json', directory/'BASIC_METRICS.json']}
        require(counts['input_sha256'] == pins[str((panel/name/'inputs.jsonl').relative_to(root))] and
                counts['score_sha256'] == pins[str((directory/'attempt_results.jsonl').relative_to(root))],
                'saved metrics no longer match input or score bytes')
        bindings.update(pins)
        return measured, inputs, counts

    panel = panel_path(args.round)
    data = {name: stage(panel, name) for name in ['construction', 'tokenized', 'current', 'proposal', 'edited']}
    edit_config = panel/('EDIT_SPEC.json' if args.round else 'RUN_SPEC.json')
    spec = json.loads(edit_config.read_text())
    edit_config_sha256 = digest(edit_config)
    checkpoint = Path(spec['assets']['editor_checkpoint'])
    require(root in checkpoint.resolve().parents, 'editor checkpoint is outside this clean run')
    checkpoint_receipt = checkpoint/('RSI_TRAINING_DONE.json' if args.round else 'INITIALIZATION_FINAL.json')
    bindings.update({str(path.relative_to(root)): digest(path) for path in [edit_config, checkpoint_receipt]})
    traces = []
    known_sun_changes = []
    for i in indices:
        path = panel/'edited/records'/f'{i:04d}.json'
        value = json.loads(path.read_text())
        require(value['record'] == data['edited'][1][i], 'edited record differs from materialized input')
        require(value['config_sha256'] == edit_config_sha256 and value['checkpoint'] == str(checkpoint),
                'edited record differs from its registered configuration or checkpoint')
        trace = value['editor_trace']
        before = data['current'][1][i].get('body_token_ids')
        after = data['edited'][1][i].get('body_token_ids')
        proposal = data['proposal'][1][i].get('body_token_ids')
        require(after == (proposal if trace.get('applied') else before), 'final output differs from recorded decision')
        if flags(data['current'][0][i])['SUN'] and after != before:
            known_sun_changes.append(i)
        traces.append(trace)
        bindings[str(path.relative_to(root))] = digest(path)
    effect = paired(data['current'][0], data['edited'][0], indices,
                    traces=traces, proposals=data['proposal'][0],
                    current_inputs=data['current'][1], final_inputs=data['edited'][1])
    reliability = Counter()
    for row in effect['records']:
        a, b = row['before'], row['after']
        if a['Stable'] == b['Stable']:
            continue
        changed = a if b['Stable'] else b
        status = 'reliable_nonStable' if changed['reliable'] else 'known_physical_failure' if changed['known_failure'] else 'unknown'
        reliability[('gain_from_' if b['Stable'] else 'loss_to_')+status] += 1
    cross_round = {}
    for prior in sorted({0, args.round-1} if args.round else set()):
        old = stage(panel_path(prior), 'edited')
        comparison = paired(old[0], data['edited'][0], indices,
                            current_inputs=old[1], final_inputs=data['edited'][1])
        comparison['counts']['changed_outputs'] = comparison['counts'].pop('actual_edits')
        comparison['scope'] = 'same-Plan full G/F/E pipeline comparison, not E-only attribution'
        cross_round[f'S{prior}_to_S{args.round}'] = comparison
    report = dict(schema='clean_1000_completed_round_analysis_v1', round_index=args.round,
        purpose='derived_report_only_not_training_feedback', panel=str(panel), requests=1000,
        source_identity=frozen['source_identity'], analyzer_sha256=digest(__file__),
        frozen_method_sha256=digest(root/'FROZEN_METHOD.json'), input_bindings=bindings,
        editor_checkpoint=str(checkpoint), known_SUN_changed_ordinals=known_sun_changes,
        stages={name: value[2] for name, value in data.items()}, E_effect=effect,
        Stable_change_reliability=dict(reliability), same_Plan_full_flow_changes=cross_round)
    target = root/'analysis'/f'S{args.round}_COMPLETED_ROUND.json'
    target.parent.mkdir(exist_ok=True)
    if target.exists():
        require(json.loads(target.read_text()) == report, 'completed derived report differs; preserve and inspect it')
    else:
        with target.open('x', encoding='utf-8') as stream:
            stream.write(json.dumps(report, sort_keys=True, indent=2)+'\n')
    def compact(value):
        return {key: item for key, item in value.items() if key != 'records'}
    print(json.dumps(dict(report=str(target), report_sha256=digest(target), stages=report['stages'],
        editor_checkpoint=str(checkpoint), known_SUN_changed_ordinals=known_sun_changes,
        E_effect=compact(effect), Stable_change_reliability=dict(reliability),
        same_Plan_full_flow_changes={key: compact(value) for key, value in cross_round.items()}), sort_keys=True))


if __name__ == '__main__':
    main()
