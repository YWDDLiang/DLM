"""Fixed-input trial panels, sealed selection, and paired SUN/MSUN accounting."""
import argparse
from collections import Counter
import copy
import datetime as dt
import importlib.util
import json
from pathlib import Path
import shutil
import sys

SOURCE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SOURCE/'src'))
from scripts.run_post_refine_cycle import read_rows, write_rows, write_json, file_hash, load_config
from scripts.run_rsi_stages import scores, materialize, rebind_labels
from crystal_dlm.ranked_feedback import endpoint_quality, ranked_preference


def clone_bound_stage(source, destination):
    """Copy identical observations and bind their path-dependent TRAIN scope."""
    from crystal_dlm.sun_feedback_contract import validate_training_feedback
    from crystal_dlm.ranked_feedback import ranked_relaxation_protocol
    source, destination = Path(source), Path(destination)
    original_inputs = source/'inputs.jsonl'
    manifest = json.loads((source/'FEEDBACK_MANIFEST.json').read_text())
    declared = Path(manifest['paths']['path'])
    if file_hash(original_inputs) != file_hash(declared):
        raise ValueError('copied stage differs from its declared original input bytes')
    records = read_rows(original_inputs)
    old_scope = validate_training_feedback(records, declared, source/'FEEDBACK_MANIFEST.json')
    binding = importlib.util.spec_from_file_location('_trial_exact_labels', SOURCE/'scripts/evaluate_programmed_paths.py')
    api = importlib.util.module_from_spec(binding); binding.loader.exec_module(api)
    expected = ranked_relaxation_protocol(api.COMMON_RELAXATION_PROTOCOL)
    original_report = source/'labeling/result/LABEL_FINAL.json'
    api.load_bound_evaluation_labels(records, [source/'labeling/result/labels.jsonl'], paths_file=declared,
        endpoint='native', purpose='training_feedback', feedback_scope=old_scope, expected_protocol=expected)
    if not destination.exists(): shutil.copytree(source, destination)
    inputs = destination/'inputs.jsonl'
    if file_hash(inputs) != file_hash(original_inputs): raise ValueError('stage copy changed input bytes')
    marker = destination/'COPIED_STAGE_BINDING.json'
    if marker.exists():
        prior = json.loads(marker.read_text())
        if prior['source_inputs_sha256'] != file_hash(original_inputs): raise ValueError('bound copy changed source')
        return
    manifest['paths'] = dict(path=str(inputs), sha256=file_hash(inputs))
    write_json(destination/'FEEDBACK_MANIFEST.json', manifest)
    scope = validate_training_feedback(records, inputs, destination/'FEEDBACK_MANIFEST.json')
    report = json.loads(original_report.read_text())
    pins = dict(source_inputs=str(original_inputs), source_inputs_sha256=file_hash(original_inputs),
        source_manifest=str(source/'FEEDBACK_MANIFEST.json'), source_manifest_sha256=file_hash(source/'FEEDBACK_MANIFEST.json'),
        source_report=str(original_report), source_report_sha256=file_hash(original_report),
        source_labels_sha256=file_hash(source/'labeling/result/labels.jsonl'),
        identical_input_bytes=True, label_values_unchanged=True, new_physics_calls=0)
    report.update(input_file=str(inputs), input_sha256=file_hash(inputs), training_feedback_scope=scope,
                  distinct_endpoint_evaluations=0, new_endpoint_evaluations=0, copied_stage_binding=pins)
    write_json(destination/'labeling/result/LABEL_FINAL.json', report)
    api.load_bound_evaluation_labels(records, [destination/'labeling/result/labels.jsonl'], paths_file=inputs,
        endpoint='native', purpose='training_feedback', feedback_scope=scope, expected_protocol=expected)
    write_json(marker, pins)


def clone_collection(root, arm, checkpoint, stream='E'):
    root, checkpoint = Path(root), Path(checkpoint)
    name = arm if stream == 'E' else arm+'_'+stream
    panel = root/'collections'/name
    spec = json.loads((root/'fit/RUN_SPEC.json').read_text())
    spec.update(run_root=str(panel), run_id='editor_trial:'+name,
                collect_training_proposals=True, editor_seed_stream=stream,
                trial_role='single_proposal_diagnostic_collection_no_new_training')
    spec['assets']['editor_checkpoint'] = str(checkpoint)
    spec['updated_checkpoint_receipts']['E'] = dict(path=str(checkpoint),
        receipt_sha256=file_hash(checkpoint/'RSI_TRAINING_DONE.json'))
    config = panel/'RUN_SPEC.json'
    if config.exists():
        if json.loads(config.read_text()) != spec: raise ValueError('registered collection changed')
        return config
    panel.mkdir(parents=True)
    shutil.copytree(root/'fit/cohort', panel/'cohort')
    clone_bound_stage(root/'fit/current', panel/'current')
    write_json(config, spec)
    return config


def policy_decision(trace, *, threshold, respect_keep):
    if not 0 <= threshold <= 1: raise ValueError('invalid probability threshold')
    proposed = trace.get('proposal_generated') is True
    protected = trace.get('learned_mode') == 0 and (respect_keep or trace.get('known_sun'))
    return bool(proposed and not protected and trace['learned_accept_probability'] >= threshold)


def policy_panel(root, collection, *, threshold, respect_keep):
    root, collection = Path(root), Path(collection)
    tag = collection.name+f'_keep{int(respect_keep)}_a{round(100*threshold):02d}'
    panel = root/'policies'/tag
    spec = json.loads((collection/'RUN_SPEC.json').read_text())
    spec.update(run_root=str(panel), run_id='editor_trial_policy:'+tag,
                collect_training_proposals=False, trial_role='autonomous_single_output_policy')
    spec['trial_decision_policy'] = dict(acceptance_threshold=threshold, respect_learned_KEEP=respect_keep,
                                      physics_used_for_decision=False, collection=str(collection))
    config = panel/'RUN_SPEC.json'
    if config.exists():
        if json.loads(config.read_text()) != spec: raise ValueError('registered decision policy changed')
        if (panel/'edited/labeling/result/_SUCCESS').exists(): return config
        raise ValueError('incomplete existing policy panel requires inspection')
    panel.mkdir(parents=True)
    shutil.copytree(collection/'cohort', panel/'cohort')
    for name in ('current', 'proposal'):
        clone_bound_stage(collection/name, panel/name)
    write_json(config, spec)
    current, proposal = read_rows(panel/'current/inputs.jsonl'), read_rows(panel/'proposal/inputs.jsonl')
    bindings = []
    for i, (before, candidate) in enumerate(zip(current, proposal, strict=True)):
        path = collection/'proposal/records'/f'{i:04d}.json'
        wrapper = json.loads(path.read_text()); trace = copy.deepcopy(wrapper['editor_trace'])
        applied = policy_decision(trace, threshold=threshold, respect_keep=respect_keep)
        selected = candidate if applied else before
        record = dict(selected, trajectory_id=f"{spec['run_id']}:edited:{i}")
        trace.update(applied=applied, learned_decision='EDIT' if applied else 'KEEP',
                     final_tokens=record.get('body_token_ids'), trial_decision_policy=spec['trial_decision_policy'])
        write_json(panel/'edited/records'/f'{i:04d}.json', dict(record=record, editor_trace=trace,
                   checkpoint=spec['assets']['editor_checkpoint'], source_wrapper_sha256=file_hash(path)))
        bindings.append(dict(sample_idx=i, applied=applied, source_trajectory_id=selected['trajectory_id'],
                             collection_trace_sha256=file_hash(path)))
    write_json(panel/'DECISION_BINDING.json', dict(policy=spec['trial_decision_policy'], decisions=bindings,
                model_scores_only=True, source_current_sha256=file_hash(collection/'current/inputs.jsonl'),
                source_proposal_sha256=file_hash(collection/'proposal/inputs.jsonl')))
    live = load_config(config)
    materialize(live, 'edited'); rebind_labels(live, 'edited')
    return config


def flags(row):
    q = endpoint_quality(row)
    return dict(Stable=bool(q['reliable'] and q['hull'] <= 0),
        MS=bool(q['reliable'] and q['hull'] <= .1),
        SUN=bool(q['reliable'] and row.get('strict_sun')),
        MSUN=bool(q['reliable'] and row.get('meta_sun')),
        reliable=q['reliable'], N=row.get('novel'), U=row.get('unique_representative'),
        known_failure=q['known_failure'], hull=q['hull'])


def paired(before, after, indices, *, traces=None, proposals=None, current_inputs=None, final_inputs=None):
    counts = {name: 0 for name in ('requested', 'Stable', 'MS', 'SUN', 'MSUN', 'reliable', 'known_failure', 'unknown', 'actual_edits')}
    transitions = {name: dict(gains=[], losses=[]) for name in ('Stable', 'MS', 'SUN', 'MSUN')}
    decision = Counter(); causes = Counter(); records = []
    for i in indices:
        a, b = flags(before[i]), flags(after[i])
        counts['requested'] += 1
        for k in ('Stable', 'MS', 'SUN', 'MSUN', 'reliable', 'known_failure'): counts[k] += int(b[k])
        counts['unknown'] += int(not b['reliable'] and not b['known_failure'])
        edited = bool(current_inputs and final_inputs and current_inputs[i].get('body_token_ids') != final_inputs[i].get('body_token_ids'))
        counts['actual_edits'] += int(edited)
        for name in transitions:
            if b[name] and not a[name]: transitions[name]['gains'].append(i)
            if a[name] and not b[name]: transitions[name]['losses'].append(i)
        for metric, physical in (('SUN', 'Stable'), ('MSUN', 'MS')):
            if a[metric] != b[metric]:
                changed = [name for name in (physical, 'N', 'U') if a[name] != b[name]]
                causes[metric+('_gain:' if b[metric] else '_loss:')+'+'.join(changed)] += 1
        record = dict(ordinal=i, before=a, after=b, actual_edit=edited)
        if traces is not None and proposals is not None:
            trace = traces[i]; q = flags(proposals[i])
            pref = ranked_preference(dict(before[i], strict_sun=False), dict(proposals[i], strict_sun=False))
            quality = 'beneficial' if pref['chosen'] == 'after' else 'harmful' if pref['chosen'] == 'before' else 'tie_or_unknown'
            chosen = 'accepted' if trace.get('applied') else 'rejected'
            if trace.get('proposal_generated'):
                decision[quality+'_'+chosen] += 1
                if q['Stable'] and not a['Stable']: decision['Stable_promotion_'+chosen] += 1
                if a['Stable'] and not q['Stable']: decision['Stable_damage_'+chosen] += 1
                if q['MS'] and not a['MS']: decision['MS_promotion_'+chosen] += 1
                if a['MS'] and not q['MS']: decision['MS_damage_'+chosen] += 1
            else: decision['no_complete_proposal'] += 1
            decision['learned_KEEP' if trace.get('learned_mode') == 0 else 'learned_EDIT'] += 1
            record.update(proposal=q, proposal_relation=quality, applied=bool(trace.get('applied')),
                accept_probability=trace.get('learned_accept_probability'), learned_mode=trace.get('learned_mode'))
        records.append(record)
    return dict(counts=counts, transitions=transitions, proposal_decisions=dict(decision),
                predicate_change_accounting=dict(causes), records=records)


def panel_report(root, panel, split_name):
    root, panel = Path(root), Path(panel)
    source_split = read_rows(root/'SOURCE_SPLIT.jsonl')
    indices = [r['ordinal'] for r in source_split if split_name == 'all' or r['split'] == split_name]
    before, after = scores(panel, 'current'), scores(panel, 'edited')
    current_inputs = read_rows(panel/'current/inputs.jsonl'); final_inputs = read_rows(panel/'edited/inputs.jsonl')
    traces = [json.loads((panel/'edited/records'/f'{i:04d}.json').read_text())['editor_trace'] for i in range(len(before))]
    result = paired(before, after, indices, traces=traces, proposals=scores(panel, 'proposal'),
                    current_inputs=current_inputs, final_inputs=final_inputs)
    result.update(panel=str(panel), split=split_name)
    return result


def coverage(root, arm):
    root = Path(root); rows = read_rows(root/'data/E.jsonl')
    content, heads = Counter(), Counter(); reports = []
    for path in sorted((root/'training'/arm/'result').glob('EXPOSURE_rank*.json')):
        report = json.loads(path.read_text()); reports.append(report)
        content.update(report['content_updated_pair_visits']); heads.update(report['head_updated_pair_visits'])
    promotions = [r['pair_id'] for r in rows if r.get('content_kind') == 'Stable_promotion']
    dense = [r['pair_id'] for r in rows if r.get('content_target_tokens')]
    return dict(ranks=len(reports), content_optimizer_steps=[r['content_optimizer_steps'] for r in reports],
        head_optimizer_steps=[r['head_optimizer_steps'] for r in reports],
        Stable_promotion_targets=len(promotions), minimum_Stable_promotion_visits=min((content[i] for i in promotions), default=0),
        minimum_dense_target_visits=min((content[i] for i in dense), default=0),
        content_targets=len(dense), missing_content_targets=sum(content[i] == 0 for i in dense),
        total_content_target_visits=sum(content.values()), total_head_pair_visits=sum(heads.values()),
        qualifies=bool(len(reports) == 6 and all(r['content_optimizer_steps'] == 8 for r in reports)
                       and promotions and all(content[i] >= 8 for i in dense)))


def freeze_selection(root, arm, policy_panels):
    root = Path(root)
    destination = root/'FROZEN_SELECTION.json'
    if destination.exists(): return json.loads(destination.read_text())
    candidates = []
    for panel in policy_panels:
        panel = Path(panel); report = panel_report(root, panel, 'dev')
        policy = json.loads((panel/'RUN_SPEC.json').read_text())['trial_decision_policy']
        counts = report['counts']
        key = [counts['Stable'], counts['SUN'], -len(report['transitions']['Stable']['losses']),
               -counts['actual_edits'], policy['acceptance_threshold'], int(policy['respect_learned_KEEP'])]
        candidates.append(dict(panel=str(panel), policy=policy, dev=report, ranking_key=key,
            score_sha256=file_hash(panel/'edited/scoring/result/attempt_results.jsonl')))
    selected = max(candidates, key=lambda x: x['ranking_key'])
    result = dict(frozen_utc=dt.datetime.now(dt.timezone.utc).isoformat(), selected=selected, arm=arm,
        calibration_split='dev_only', candidates=candidates, coverage=coverage(root, arm),
        preregistration_sha256=file_hash(root/'PREREGISTRATION.json'),
        scope_amendment_sha256=file_hash(root/'SCOPE_AMENDMENT_STABLE_SUN.json'),
        final_metrics_used_for_selection=False)
    write_json(destination, result)
    return result


def final_report(root, heads_panel):
    root = Path(root); frozen = json.loads((root/'FROZEN_SELECTION.json').read_text())
    reg = json.loads((root/'PREREGISTRATION.json').read_text())
    old = Path(reg['fixed_current']); candidate = Path(frozen['selected']['panel'])
    reports = {}
    for name in ('train', 'dev', 'final', 'all'):
        indices = [r['ordinal'] for r in read_rows(root/'SOURCE_SPLIT.jsonl') if name == 'all' or r['split'] == name]
        current = scores(old, 'current')
        reports[name] = dict(KEEP=paired(current, current, indices), old_E3=panel_report(root, old, name),
            T2T=panel_report(root, candidate, name), heads_only=panel_report(root, Path(heads_panel), name))
    test = reports['final']; result = test['T2T']; a, b, c = [test[n]['counts'] for n in ('KEEP', 'old_E3', 'T2T')]
    criteria = dict(actual_training_coverage=frozen['coverage']['qualifies'],
        Stable_above_both=c['Stable'] > max(a['Stable'], b['Stable']),
        SUN_not_lower=c['SUN'] >= max(a['SUN'], b['SUN']),
        Stable_protection=len(result['transitions']['Stable']['losses']) <= min(1, len(test['old_E3']['transitions']['Stable']['losses'])),
        physical_failures=c['known_failure'] <= max(a['known_failure'], b['known_failure']))
    report = dict(schema='fixed_GF_editor_trial_result_v1', work=all(criteria.values()), criteria=criteria,
        frozen_selection_sha256=file_hash(root/'FROZEN_SELECTION.json'),
        coverage=frozen['coverage'], selected_policy=frozen['selected']['policy'], reports=reports,
        denominator_policy='all_registered_requests_in_each_source_split_including_failures',
        heads_ablation='identical_selected_acceptance_threshold_and_KEEP_rule',
        heldout_scope=reg['split_scope'], measured_effect='this_fixed_GF_panel_and_shared_rng_only')
    write_json(root/'PRIMARY_TRIAL_RESULT.json', report)
    print(json.dumps(dict(work=report['work'], criteria=criteria,
                         final={k: v['counts'] for k, v in test.items()}, coverage=frozen['coverage']), sort_keys=True))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--heads-panel', type=Path, required=True)
    args = parser.parse_args()
    final_report(args.root, args.heads_panel)
