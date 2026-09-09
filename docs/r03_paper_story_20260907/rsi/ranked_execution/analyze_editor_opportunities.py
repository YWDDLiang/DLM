#!/usr/bin/env python3
"""Attribute completed TRAIN editor outcomes without selecting new outputs."""
import argparse
from collections import Counter, defaultdict
import hashlib
from itertools import islice
import json
from pathlib import Path
import sys


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def analyze(root, source, index):
    sys.path.insert(0, str(source/'src'))
    from crystal_dlm.ranked_feedback import endpoint_quality, ranked_preference
    root = root.resolve()
    cohort = root/'fit' if index == 0 else root/'rounds'/f'round{index}'/'fit'
    hashes = {}

    def scores(base, stage):
        directory = base/stage/'scoring/result'
        if not (directory/'_SUCCESS').exists():
            raise ValueError(f'incomplete scores: {directory}')
        path = directory/'attempt_results.jsonl'
        data = read_rows(path)
        if len(data) != 256 or [x['sample_idx'] for x in data] != list(range(256)):
            raise ValueError('expected complete ordered TRAIN256')
        hashes[str(path)] = digest(path)
        return data

    def stable(q):
        return q['reliable'] and q['hull'] <= 0

    def transition(before, after):
        counts = Counter()
        changes = []
        for a, b in zip(before, after, strict=True):
            x, y = endpoint_quality(a), endpoint_quality(b)
            preference = ranked_preference(a, b)
            counts['preference_'+str(preference['chosen'])] += 1
            counts['Stable_retained'] += stable(x) and stable(y)
            counts['Stable_gained'] += not stable(x) and stable(y)
            counts['Stable_lost'] += stable(x) and not stable(y)
            counts['Stable_lost_to_unverified'] += stable(x) and not y['reliable']
            counts['SUN_retained'] += x['rank'] == 4 and y['rank'] == 4
            counts['SUN_gained'] += x['rank'] != 4 and y['rank'] == 4
            counts['SUN_lost'] += x['rank'] == 4 and y['rank'] != 4
            if stable(x) != stable(y) or preference['chosen'] == 'after':
                changes.append(dict(ordinal=a['sample_idx'], before=x, after=y,
                                    preference=preference['chosen']))
        return dict(counts=dict(counts), changes=changes)

    current, proposal, final = [scores(cohort, stage) for stage in ('current', 'proposal', 'edited')]
    result = dict(scope='TRAIN diagnostic; no new generation, threshold selection, or mixed-output SUN claim',
                  root=str(root), source=str(source), round=index, analysis_sha256=digest(Path(__file__)),
                  proposal_transition=transition(current, proposal), actual_transition=transition(current, final))
    traces, decisions, positive, negative, population = {}, Counter(), [], [], []
    for a, b in zip(current, proposal, strict=True):
        ordinal = a['sample_idx']
        path = cohort/'edited/records'/f'{ordinal:04d}.json'
        hashes[str(path)] = digest(path)
        trace = json.loads(path.read_text())['editor_trace']
        traces[ordinal] = trace
        decisions['requested'] += 1
        if trace.get('upstream_failure'):
            decisions['upstream_failure'] += 1
            continue
        decisions['mode_'+str(trace.get('learned_mode'))] += 1
        for key in ('known_sun', 'proposal_generated', 'applied'):
            decisions[key] += trace.get(key) is True
        decisions['proposal_failure'] += bool(trace.get('proposal_failure'))
        probability = trace.get('learned_accept_probability')
        if probability is None:
            continue
        p = ranked_preference(a, b)
        x, y = endpoint_quality(a), endpoint_quality(b)
        population.append(dict(probability=probability, preference=p['chosen'],
                               gain=not stable(x) and stable(y), loss=stable(x) and not stable(y)))
        if p['chosen'] == 'after': positive.append(probability)
        if p['chosen'] == 'before': negative.append(probability)
    for change in result['proposal_transition']['changes']:
        trace = traces[change['ordinal']]
        change.update(action=trace.get('action'), applied=trace.get('applied'),
                      acceptance_probability=trace.get('learned_accept_probability'))
    result['decisions'] = dict(decisions)
    result['acceptance'] = dict(beneficial=len(positive), harmful=len(negative),
        pairwise_AUROC=(sum((a>b)+.5*(a==b) for a in positive for b in negative)/
                       (len(positive)*len(negative))) if positive and negative else None,
        thresholds=[])
    for threshold in (.03, .04, .05, .06, .065, .07, .08, .5):
        selected = [x for x in population if x['probability'] >= threshold]
        result['acceptance']['thresholds'].append(dict(threshold=threshold, accepted=len(selected),
            beneficial=sum(x['preference'] == 'after' for x in selected),
            harmful=sum(x['preference'] == 'before' for x in selected),
            Stable_gain=sum(x['gain'] for x in selected), Stable_loss=sum(x['loss'] for x in selected)))
    result['Stable_without_SUN'] = [{k:row.get(k) for k in
        ('sample_idx', 'novel', 'unique_representative', 'e_above_hull_eV_atom', 'terminal_status')}
        for row in current if endpoint_quality(row)['rank'] == 3]
    result['reliable_near_Stable_with_NU'] = [row['sample_idx'] for row in current
        if endpoint_quality(row)['reliable'] and 0 < row['e_above_hull_eV_atom'] <= .03
        and row.get('novel_unique') is True]
    if index > 0:
        input_path = cohort/'current/inputs.jsonl'
        inputs = read_rows(input_path); hashes[str(input_path)] = digest(input_path)
        plan_path = cohort/'cohort/plans.jsonl'
        plans = read_rows(plan_path); hashes[str(plan_path)] = digest(plan_path)
        targets = defaultdict(list)
        history = [root/'fit', root/'bootstrap_comparator',
                   *[root/'rounds'/f'round{i}'/'fit' for i in range(1, index)]]
        for prior in history:
            prior_scores = scores(prior, 'current')
            prior_inputs = prior/'current/inputs.jsonl'
            candidate_inputs = read_rows(prior_inputs)
            hashes[str(prior_inputs)] = digest(prior_inputs)
            prior_plans = prior/'cohort/plans.jsonl'
            candidate_plans = read_rows(prior_plans)
            hashes[str(prior_plans)] = digest(prior_plans)
            for i, (plan, old_plan) in enumerate(zip(plans, candidate_plans, strict=True)):
                if any(plan[k] != old_plan[k] for k in ('ancestor_id', 'body_prompt', 'plan_state')):
                    raise ValueError('historical teacher Plan or ancestor differs')
                a, b = endpoint_quality(current[i]), endpoint_quality(prior_scores[i])
                if a['reliable'] and not stable(a) and stable(b) and inputs[i].get('body_token_ids') \
                        and candidate_inputs[i].get('body_token_ids'):
                    if inputs[i]['sample_idx'] != candidate_inputs[i]['sample_idx']:
                        raise ValueError('historical teacher input ordering differs')
                    targets[i].append(dict(source=str(prior), hull=b['hull'], SUN=b['SUN']))
        result['historical_Stable_targets'] = dict(
            current_states_with_targets=len(targets), targets=dict(targets),
            note='Earlier same-Plan verified Stable endpoints available for reliable non-Stable inputs; '
                 'not compiled targets, learned successes, or mixed-output scores.')
    collection = cohort/'editor_collection'
    if collection.exists():
        initial = scores(collection, 'current')
        result['training_candidates'] = {stage:transition(initial, scores(collection, stage))
                                         for stage in ('proposal', 'teacher')}
        path = collection/'pairs/E.jsonl'
        examples = read_rows(path); hashes[str(path)] = digest(path)
        result['training_examples'] = dict(count=len(examples),
            edit_positive=sum(bool(x.get('mode_target')) for x in examples),
            accept_positive=sum(x.get('accept_target') == 1 for x in examples))
        conditions = defaultdict(list)
        for example in examples:
            conditions[example['conditioning_sha256']].append(example)
        conflicting = [group for group in conditions.values()
                       if len({x['mode_target'] for x in group}) > 1]
        result['mode_supervision'] = dict(
            distinct_current_conditions=len(conditions),
            target_counts=dict(Counter(x['mode_target'] for x in examples)),
            differing_mode_conditions=len(conflicting),
            keep_and_edit_conditions=sum(any(x['mode_target'] == 0 for x in group)
                and any(x['mode_target'] != 0 for x in group) for group in conflicting),
            note='Mode sees the current state; acceptance also sees the actual proposal. '
                 'Differing candidate labels do not establish a causal effect on the trained model.',
            positive_scope=[dict(pair_id=x['pair_id'], origin=x['origin'], mode=x['mode_target'],
                changed_numeric_tokens=sum(a != b for a, b in
                    zip(x['current_tokens'], x['proposal_tokens'], strict=True)),
                opened_tokens=len(x['action_positions']))
                for x in examples if x.get('mode_target')])
    training = root/'training'/f'round{index}'/'E/result'
    if (training/'TRAINING_FINAL.json').exists():
        path = training/'TRAINING_FINAL.json'; hashes[str(path)] = digest(path)
        receipt = json.loads(path.read_text())
        result['training'] = {k:receipt[k] for k in
            ('optimizer_steps', 'training_seconds', 'parameter_delta_squared')}
        result['training']['KL_stopped_ranks'] = [int(path.stem.split('rank')[-1])
            for path in sorted(training.glob('EXPOSURE_rank*.json')) if json.loads(path.read_text())['KL_stop']]
        result['training']['stop_KL_value'] = None
        result['training']['stop_KL_note'] = 'Existing trainer records the stop flag, not the triggering KL value.'
        contract = receipt['contract']
        if contract.get('bounded_minibatch_training'):
            from crystal_dlm.rsi_minibatch import epoch_indices
            data_path = Path(contract['data'])
            if digest(data_path) != contract['data_sha256']:
                raise ValueError('training examples changed after the actual update')
            examples = read_rows(data_path)
            batches = epoch_indices(len(examples), batch_size=contract['batch_size'],
                world=contract['world_size'], epochs=contract['epochs'], seed=contract['seed'])
            updated = [examples[i] for _, indices in islice(batches, receipt['optimizer_steps']) for i in indices]
            promotions = [x for x in updated if x.get('mode_target')
                and x['preference']['before']['rank'] not in (3,4)
                and x['preference']['after']['rank'] in (3,4)]
            result['training']['actual_updated_exposure'] = dict(
                visits=len(updated), unique_examples=len({x['pair_id'] for x in updated}),
                dataset_examples=len(examples),
                edit_positive_visits=sum(bool(x.get('mode_target')) for x in updated),
                Stable_promotion_visits=len(promotions),
                Stable_promotion_pair_ids=sorted({x['pair_id'] for x in promotions}),
                SUN_positive_visits=sum(bool(x.get('mode_target')) and x['objective_level']=='SUN' for x in updated),
                note='Reconstructed optimizer batches; excludes the next inspected batch that triggered KL stop.')
    result['input_sha256'] = hashes
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'source', 'output'): parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--round', type=int, required=True)
    args = parser.parse_args()
    report = analyze(args.root, args.source, args.round)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream: json.dump(report, stream, indent=2, sort_keys=True)
    print(json.dumps({'output':str(args.output), 'round':args.round, 'decisions':report['decisions']}))
