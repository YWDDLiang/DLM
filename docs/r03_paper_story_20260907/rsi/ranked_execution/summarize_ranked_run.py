#!/usr/bin/env python3
"""Read completed TRAIN endpoints and preserve comparable round diagnostics."""
import argparse
from collections import Counter
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def collect(root):
    root=Path(root).resolve()
    source=Path((root/'SOURCE_PATH').read_text().strip())
    sys.path.insert(0,str(source/'src'))
    from crystal_dlm.ranked_feedback import ranked_preference,endpoint_quality
    from scripts.run_rsi_stages import score_directory
    result={'observed_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
        'root':str(root),'analysis_sha256':digest(__file__),'source_path':str(source),
        'budget':json.loads((root/'BUDGET.json').read_text()),'rounds':[],'weight_updates':{}}
    previous_final=None
    for index,cohort in enumerate([root/'fit',*[root/'rounds'/f'round{i}'/'fit' for i in (1,2,3)]]):
        record={'index':index,'stages':{},'editor':None,'relative_to_S0_final':None}
        labels={}
        for stage in ('construction','tokenized','current','proposal','edited'):
            directory=score_directory(cohort,stage)
            if not (directory/'_SUCCESS').exists(): continue
            path=directory/'attempt_results.jsonl';measured=rows(path)
            inputs=cohort/stage/'inputs.jsonl';input_rows=rows(inputs)
            reports=list(directory.glob('*FINAL.json'))
            if len(reports)!=1: raise ValueError('ambiguous scoring report')
            report=json.loads(reports[0].read_text())
            if report['status']!='complete' or report['input_sha256']!=digest(inputs):
                raise ValueError('scoring report does not bind the current inputs')
            if len(measured)!=256 or len(input_rows)!=256:
                raise ValueError('complete 256-request denominator required')
            if any(a['sample_idx']!=b['sample_idx'] for a,b in zip(measured,input_rows,strict=True)):
                raise ValueError('source ordering changed')
            physical_rows=rows(cohort/stage/'labeling/result/labels.jsonl')
            by_identity={row['trajectory_id']:row for row in physical_rows}
            if len(by_identity)!=256 or set(by_identity)!={row['trajectory_id'] for row in measured}:
                raise ValueError('physical labels do not match complete scored occurrences')
            physics=[by_identity[row['trajectory_id']] for row in measured]
            quality=[endpoint_quality(row) for row in measured]
            categories=Counter()
            for q,p in zip(quality,physics,strict=True):
                categories[q['level'] if q['reliable'] else
                    'generation_failure' if p['status']=='generation_failure' else
                    'physical_failure' if q['known_failure'] else 'unverified_or_reference_unknown']+=1
            basic=json.loads((directory/'BASIC_METRICS.json').read_text())
            record['stages'][stage]={'requested':256,'quality_categories':dict(categories),
                'strict_Stable_including_SUN':sum(q['reliable'] and q['hull']<=0 for q in quality),
                'physics_statuses':dict(Counter(p['status'] for p in physics)),
                'comp_valid':basic['comp_valid'],'Struct_valid':basic['Struct_valid'],
                'input_sha256':digest(inputs),'score_sha256':digest(path)}
            labels[stage]=measured
        if all(stage in labels for stage in ('current','proposal','edited')):
            counts=Counter();transitions=Counter()
            for i in range(256):
                trace=json.loads((cohort/'proposal/records'/f'{i:04d}.json').read_text())['editor_trace']
                current,proposal,final=(labels[stage][i] for stage in ('current','proposal','edited'))
                preference=ranked_preference(current,proposal)
                actual=ranked_preference(current,final)
                applied=trace.get('applied') is True
                counts['applied']+=applied
                counts['proposal_generated']+=trace.get('proposal_generated') is True
                if trace.get('known_sun'):
                    counts['known_SUN']+=1
                    counts['known_SUN_kept_exact']+=trace['final_tokens']==trace['current_tokens']
                if applied: counts['accepted_'+str(preference['chosen'])]+=1
                for label,choice in [('beneficial','after'),('harmful','before')]:
                    if preference['chosen']==choice:
                        counts[label+'_proposals']+=1;counts[label+'_accepted']+=applied
                counts['final_'+str(actual['chosen'])]+=1
                a,b=endpoint_quality(current),endpoint_quality(final)
                transitions[str(a['level'])+' -> '+str(b['level'])]+=1
            record['editor']={'counts':dict(counts),'quality_transitions':dict(transitions),
                'comparison':'ranked quality; chosen after=improvement, before=regression; None=tie or unavailable'}
        if 'edited' in labels:
            if index==0: previous_final=labels['edited']
            elif previous_final:
                counts=Counter();sun=Counter()
                for before,after in zip(previous_final,labels['edited'],strict=True):
                    if before['sample_idx']!=after['sample_idx']: raise ValueError('round source IDs differ')
                    p=ranked_preference(before,after);counts[str(p['chosen'])]+=1
                    a,b=endpoint_quality(before),endpoint_quality(after)
                    sun[f"{a['rank']==4} -> {b['rank']==4}"]+=1
                record['relative_to_S0_final']={'ranked_comparisons':dict(counts),'reliable_SUN_transitions':dict(sun)}
        if record['stages']: result['rounds'].append(record)
    for index in (1,2,3):
        for branch in ('G','E'):
            path=root/'training'/f'round{index}'/branch/'result/checkpoint/RSI_TRAINING_DONE.json'
            if not path.exists(): continue
            receipt=json.loads(path.read_text())
            if receipt['optimizer_steps']<1 or receipt['parameter_delta_squared']<=0:
                raise ValueError('a reported update did not change parameters')
            for name,expected in receipt['checkpoint_files'].items():
                if digest(path.parent/name)!=expected: raise ValueError('updated checkpoint bytes changed')
            result['weight_updates'][f'{branch}{index}']={key:receipt[key] for key in
                ('optimizer_steps','parameter_delta_squared','training_seconds','local_preference_examples','local_decision_examples')}
            result['weight_updates'][f'{branch}{index}']['receipt_sha256']=digest(path)
    result['complete']=len(result['weight_updates'])==6 and any(
        row['index']==3 and 'edited' in row['stages'] for row in result['rounds'])
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();report=collect(args.root)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream: json.dump(report,stream,indent=2,sort_keys=True)
    print(json.dumps({'output':str(args.output),'complete':report['complete'],
                      'rounds':len(report['rounds']),'actual_updates':list(report['weight_updates'])}))
