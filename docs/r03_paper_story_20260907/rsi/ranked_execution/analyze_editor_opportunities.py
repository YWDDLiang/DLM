#!/usr/bin/env python3
"""Attribute completed TRAIN editor outcomes without selecting new outputs."""
import argparse
from collections import Counter
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
