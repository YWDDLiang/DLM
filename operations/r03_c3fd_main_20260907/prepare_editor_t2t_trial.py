"""Register and compile a fixed-G/F editor trial before any new optimizer run."""
import argparse
from collections import Counter, defaultdict
import copy
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil
import sys

SOURCE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SOURCE/'src'))
from scripts.run_post_refine_cycle import read_rows, write_rows, write_json, file_hash, load_config
from scripts.run_rsi_stages import scores, score_directory
from scripts.run_ranked_rsi import require_train
from crystal_dlm.post_refine_contract import fingerprint
from crystal_dlm.ranked_feedback import (ranked_preference, endpoint_quality, align_fixed_slots,
    target_action, validate_action_target, assign_current_mode_targets)


def group_split(plans):
    keys = {p['plan_state']['reduced_formula'] for p in plans}
    ordered = sorted(keys, key=lambda key: hashlib.sha256(('editor_t2t_20260910_v1|'+key).encode()).hexdigest())
    a, b = int(.6*len(ordered)), int(.8*len(ordered))
    roles = {key: 'train' if i < a else 'dev' if i < b else 'final' for i, key in enumerate(ordered)}
    return [dict(ordinal=i, ancestor_id=p['ancestor_id'], reduced_formula=p['plan_state']['reduced_formula'],
                 split=roles[p['plan_state']['reduced_formula']]) for i, p in enumerate(plans)]


def register(previous, root):
    previous, root = Path(previous).resolve(), Path(root).resolve()
    current = previous/'rounds/round3/fit'
    spec = load_config(current/'RUN_SPEC.json')
    _, plans = require_train(spec)
    split = group_split(plans)
    registration = dict(schema='fixed_GF_editor_T2T_trial_v1', previous_run=str(previous),
        fixed_current=str(current), fixed_current_sha256=file_hash(current/'current/inputs.jsonl'),
        plans_sha256=file_hash(current/'cohort/plans.jsonl'),
        split_sha256=fingerprint(split), split_counts=dict(Counter(r['split'] for r in split)),
        split_scope='excluded_from_this_editor_update; old_RSI_training_may_have_seen_all_256',
        source_group='reduced_formula_and_ancestor', seed=20260910,
        controls=['same_input_KEEP', 'completed_old_E3', 'same_data_detached_heads_only'],
        candidate='dense_T2T_M2T_plus_detached_heads',
        target_sources='identical_Plan_actual_scored_token_states_in_completed_S0_to_S3; includes_rejected_proposals',
        target_quality='verified_physics_Stable_or_MetaStable_or_fixed_0.01_eV_improvement; historical_U_never_a_target',
        healthy_rule='verified_Stable_current_has_KEEP_mode_and_identity_content_anchor',
        data_limits=dict(max_positive_candidates_per_current=2, max_negative_candidates_per_current=4,
                         content_targets_per_current=1),
        training=dict(epochs=8, content_learning_rate=2e-6, head_learning_rate=1e-4,
            max_reference_KL=.02, reference_KL_weight=1., source_batch_per_GPU=8, training_GPUs=6,
            max_seconds_per_arm=3600, dense_supervision='all_action_numeric_tokens_in_each_view',
            schedule='alternating_full_T2T_and_full_mask_M2T; old_structure_always_visible',
            content_optimizer='one_accumulated_complete_data_pass_per_step; discard_interrupted_pass',
            heads='each_minibatch; content_features_detached',
            parameter_groups='all_content_modules_share_content_LR; only_four_decision_heads_use_head_LR',
            max_candidate_training_attempts=2,
            second_attempt_rule='only_if_first_has_fewer_than_8_complete_content_passes; content_LR_2e-7; same_head_LR_seed_data'),
        calibration=dict(split='dev_only', respect_learned_KEEP=[True, False], acceptance_thresholds=[.5,.65,.8],
            tie_break='Stable_then_SUN_then_MSUN_then_fewer_Stable_losses_then_fewer_applied_edits_then_larger_threshold_then_respect_KEEP',
            candidate_weights='T2T_arm_only; heads_only_is_an_ablation', final_selection='one_frozen_policy_before_final_metrics'),
        work_rule=dict(required_complete_content_passes=8, min_actual_visits_every_Stable_promotion=8,
            min_T2T_visits_every_content_target=4, final_Stable='strictly_more_than_both_KEEP_and_old_E3',
            final_SUN='at_least_both_controls', final_MSUN='at_least_both_controls',
            final_Stable_losses='no_more_than_old_E3_and_no_more_than_one',
            final_known_physical_failures='no_more_than_worse_control',
            primary_output='single_autonomous_E_output_per_fixed_input; no_physics_candidate_selection',
            no_success_from_training_loss=True),
        diagnostics=dict(SUN_MSUN='paired_gains_losses_attribution_to_Stable_MS_N_U_and_acceptance',
            proposal='beneficial_or_harmful_by_fresh_physics_then_accepted_rejected',
            novelty='report_input_N_separately_from_common_relaxed_terminal_N',
            uniqueness='four_prespecified_E_seed_streams_same_Plan_on_final_groups; all_candidates_and_failures_count',
            K=4, seed_streams=['E','E_DIAG_1','E_DIAG_2','E_DIAG_3']),
        continuation='1000_TRAIN_S0_plus_three_G_E_updates; use_trial_method_if_work_rule_passes_else_completed_old_method',
        resource_cap=dict(wall_hours=12, max_GPUs=6, max_submitted_jobs=3),
        created_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    path = root/'PREREGISTRATION.json'
    if path.exists():
        old = json.loads(path.read_text())
        registration['created_utc'] = old['created_utc']
        if old != registration: raise ValueError('trial preregistration is immutable')
        return
    root.mkdir(parents=True, exist_ok=True)
    write_json(path, registration); write_rows(root/'SOURCE_SPLIT.jsonl', split)
    fit = root/'fit'; fit.mkdir()
    shutil.copytree(current/'cohort', fit/'cohort')
    shutil.copytree(current/'current', fit/'current')
    target = copy.deepcopy({k: v for k, v in spec.items() if not k.startswith('_')})
    target.update(run_root=str(fit), run_id='fixed_GF_editor_T2T_20260910:fit',
                  editor_trial=True, complete_flow_before_judgment=True)
    target['updated_checkpoint_receipts'].pop('E', None)
    target['resources'].update(budget_receipt=str(root/'BUDGET.json'), wall_hours=12, start_utc=None, deadline_utc=None)
    target['execution_policy'] = dict(single_GPUs=6, parallel_main_GPUs=3, parallel_other_GPUs=3,
                                    training_GPUs=6, training_batch_size=8, cpus_per_gpu=4)
    write_json(fit/'RUN_SPEC.json', target); write_json(root/'RUN_SPEC.json', target)
    pipeline = json.loads((previous/'RAW0_PIPELINE_TEMPLATE.json').read_text())
    pipeline['run_root'] = str(root)
    pipeline['resources'].update(deadline_utc=None, extra_gpu_until_utc=None)
    write_json(root/'RAW0_PIPELINE_TEMPLATE.json', pipeline)


def compile_data(root):
    from transformers import AutoTokenizer
    from crystal_dlm.r03_physics_transfer import build_repair_constraints, geometry_support_report
    root = Path(root).resolve()
    reg = json.loads((root/'PREREGISTRATION.json').read_text())
    previous = Path(reg['previous_run']); fixed = Path(reg['fixed_current'])
    plans = read_rows(fixed/'cohort/plans.jsonl')
    split = read_rows(root/'SOURCE_SPLIT.jsonl')
    if fingerprint(split) != reg['split_sha256']: raise ValueError('trial source split changed')
    spec = load_config(fixed/'RUN_SPEC.json')
    tokenizer = AutoTokenizer.from_pretrained(spec['assets']['b0_checkpoint'], trust_remote_code=True)
    support = build_repair_constraints(tokenizer)
    physical = json.loads((fixed/'current/labeling/result/LABEL_FINAL.json').read_text())
    candidates = defaultdict(dict); currents = defaultdict(dict); pins = {}; exclusions = Counter()
    roots = [previous/'fit', previous/'bootstrap_comparator'] + [previous/f'rounds/round{i}/fit' for i in range(1, 4)]
    locations = [(p, name) for p in roots for name in ('current', 'proposal', 'edited')]
    locations += [(p/'editor_collection', name) for p in roots[2:] for name in ('teacher', 'proposal')]
    def supported(tokens):
        key = tuple(tokens)
        if key not in support_cache:
            support_cache[key] = geometry_support_report(tokens, constraints=support)['supported']
        return support_cache[key]
    support_cache = {}
    for parent, name in locations:
        stage = parent/name
        if not (stage/'inputs.jsonl').exists(): continue
        old_plans = read_rows(parent/'cohort/plans.jsonl')
        if len(old_plans) != len(plans) or any(any(p[k] != q[k] for k in ('ancestor_id', 'body_prompt', 'plan_state'))
                for p, q in zip(plans, old_plans, strict=True)):
            raise ValueError('trial source Plan identity changed')
        inputs = stage/'inputs.jsonl'; report_path = stage/'labeling/result/LABEL_FINAL.json'
        report = json.loads(report_path.read_text())
        if (not (report_path.parent/'_SUCCESS').exists() or report['input_sha256'] != file_hash(inputs)
                or any(report[k] != physical[k] for k in ('protocol', 'verification_protocol',
                           'geometry_validation_protocol', 'runtime_identities'))):
            raise ValueError('trial source physics protocol or binding changed')
        records, measured = read_rows(inputs), scores(parent, name)
        if len(records) != len(plans) or len(measured) != len(plans): raise ValueError('truncated trial source')
        pins[str(stage)] = {str(p): file_hash(p) for p in (inputs, report_path, report_path.parent/'labels.jsonl',
                             score_directory(parent, name)/'attempt_results.jsonl', parent/'cohort/plans.jsonl')}
        for i, (record, score) in enumerate(zip(records, measured, strict=True)):
            if (record['trajectory_id'] != score['trajectory_id'] or record['sample_idx'] != i or score['sample_idx'] != i):
                raise ValueError('trial endpoint and score identities differ')
            if split[i]['split'] != 'train': continue
            tokens = record.get('body_token_ids')
            if not tokens:
                exclusions['missing_token_state'] += 1
                continue
            physical_score = dict(score, strict_sun=False)
            entry = dict(tokens=tokens, score=physical_score, source=str(stage),
                record_sha256=fingerprint(record), score_sha256=fingerprint(score),
                actual_current_SUN=score.get('strict_sun') is True, historical_U_reused=False)
            token_key = fingerprint(tokens)
            if name == 'current':
                currents[i].setdefault(token_key, entry)
            if not supported(tokens):
                exclusions['unsupported_candidate_kept_only_as_current_condition'] += 1
                continue
            # A duplicate body carries one conservative measured quality label.
            prior = candidates[i].get(token_key)
            q = endpoint_quality(physical_score)
            if q['reliable'] or q['known_failure']:
                worst = (q['rank'] if q['rank'] is not None else -1, -(q['hull'] or 0.))
                if prior is None or worst < prior['_quality_order']:
                    candidates[i][token_key] = dict(entry, _quality_order=worst)
    examples = []; audit = []; state_counts = Counter()
    for i, group in sorted(currents.items()):
        plan = plans[i]
        for current_key, before in sorted(group.items()):
            left = before['tokens']; before_q = endpoint_quality(before['score'])
            condition = fingerprint(dict(Plan=plan['body_prompt'], current=left))
            possible = []
            for candidate in candidates[i].values():
                try:
                    right, permutation = align_fixed_slots(candidate['tokens'], left)
                    if not supported(right): raise ValueError('aligned_target_outside_support')
                    action = target_action(left, right)
                    validate_action_target(left, right, action['positions'])
                except (ValueError, KeyError, TypeError) as error:
                    exclusions[str(error)] += 1
                    continue
                pref = ranked_preference(before['score'], candidate['score'])
                known = ((pref['before']['reliable'] and pref['after']['reliable']) or pref['chosen'] is not None)
                if not known: continue
                positive = pref['chosen'] == 'after' and left != right and before_q['rank'] not in (3, 4)
                trace = dict(conditioning_sha256=condition, before_source=before['source'],
                    before_record_sha256=before['record_sha256'], before_score_sha256=before['score_sha256'],
                    target_source=candidate['source'], target_record_sha256=candidate['record_sha256'],
                    target_score_sha256=candidate['score_sha256'], historical_U_reused=False)
                row = dict(pair_id=fingerprint(trace), source_id=plan['ancestor_id'], source_split='train',
                    trial_split='train', source_row_idx=plan['source_row_idx'], source_ordinal=i,
                    prompt=plan['body_prompt'], plan_state=plan['plan_state'], num_sites=plan['plan_state']['N'],
                    current_tokens=left, proposal_tokens=right, action_positions=action['positions'], action_spec=action,
                    mode_target=action['mode'] if positive else 0, site_targets=[float(j in action['sites']) for j in range(plan['plan_state']['N'])],
                    accept_target=int(positive), known_sun=before['actual_current_SUN'],
                    conditioning_sha256=condition, priority=pref['priority'], preference=pref,
                    objective_level=pref['objective_level'] or 'decision', origin='verified_same_Plan_history',
                    chosen_tokens=None, rejected_tokens=None, atom_permutation=permutation,
                    Stable_promotion=bool(positive and pref['after']['rank'] == 3 and before_q['rank'] != 3),
                    Stable_protection=before_q['rank'] == 3, source_trace=trace)
                possible.append(row)
            positives = sorted((r for r in possible if r['accept_target']),
                key=lambda r: (-r['preference']['after']['rank'], r['preference']['after']['hull'], len(r['action_positions']), r['pair_id']))
            negatives = sorted((r for r in possible if not r['accept_target']), key=lambda r: r['pair_id'])
            retained = positives[:2] + negatives[:4]
            if not retained: continue
            assign_current_mode_targets(retained)
            chosen = next((r for r in retained if r['mode_target'] not in (None, 0)), None)
            if chosen:
                chosen['content_target_tokens'] = chosen['proposal_tokens']
                chosen['content_positions'] = chosen['action_positions']
                chosen['content_kind'] = 'Stable_promotion' if chosen['Stable_promotion'] else 'verified_repair'
            elif before_q['rank'] in (2, 3) and supported(left):
                chosen = next(r for r in retained if r['mode_target'] == 0)
                chosen['content_target_tokens'] = left
                chosen['content_positions'] = list(range(1, 7)) + [8 + 4*j + axis for j in range(plan['plan_state']['N']) for axis in range(3)]
                chosen['content_kind'] = 'Stable_identity_anchor' if before_q['rank'] == 3 else 'MS_identity_anchor'
            state_counts['conditions'] += 1
            if chosen: state_counts[chosen['content_kind']] += 1
            examples.extend(retained)
            audit.append(dict(conditioning_sha256=condition, source_ordinal=i, pool_candidates=len(possible),
                verified_positive_candidates=len(positives), retained_pairs=[r['pair_id'] for r in retained]))
    if len({r['pair_id'] for r in examples}) != len(examples): raise ValueError('duplicate trial pair identity')
    if any(split[r['source_ordinal']]['split'] != 'train' for r in examples): raise ValueError('trial split leakage')
    directory = root/'data'; directory.mkdir(exist_ok=True)
    data, audit_path = directory/'E.jsonl', directory/'AUDIT.jsonl'
    if data.exists(): raise ValueError('trial data is immutable')
    write_rows(data, examples); write_rows(audit_path, audit)
    write_json(directory/'PAIRS_E_FINAL.json', dict(schema='editor_T2T_trusted_history_v1', source_split='train',
        files_sha256={p.name: file_hash(p) for p in (data, audit_path)}, stage_evidence=pins,
        preregistration_sha256=file_hash(root/'PREREGISTRATION.json'), split_sha256=file_hash(root/'SOURCE_SPLIT.jsonl'),
        examples=len(examples), states=dict(state_counts), exclusions=dict(exclusions),
        source_group_counts=dict(Counter(r['split'] for r in split)),
        historical_U_reused=False, no_unscored_quantization_targets=True,
        content_targets=sum(bool(r.get('content_target_tokens')) for r in examples),
        Stable_promotion_content_targets=sum(r.get('content_kind') == 'Stable_promotion' for r in examples),
        heldout_groups_used_for_this_update=False))
    print(json.dumps(dict(examples=len(examples), states=dict(state_counts), split=dict(Counter(r['split'] for r in split)),
                          data_sha256=file_hash(data), exclusions=dict(exclusions)), sort_keys=True), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--register-only', action='store_true')
    args = parser.parse_args()
    register(args.previous, args.root)
    if not args.register_only: compile_data(args.root)
