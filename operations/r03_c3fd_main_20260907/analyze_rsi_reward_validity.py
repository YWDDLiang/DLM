"""Read existing relaxation validity and preference inputs; never change labels."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys


def analyze(root, source):
    sys.path.insert(0, str(source/'src'))
    from scripts.run_rsi_stages import scores, score_directory, read_rows, file_hash
    report={'scope':'Read-only existing labels; formal SUN is unchanged. Invalid geometry is distinct from non-convergence.',
            'stages':[], 'energy_outliers':[], 'training_label_exposure':[],
            'editor_SUN_changes':[], 'input_hashes':{}}
    for cohort in ['MAIN','FIT']:
        for i in range(4):
            base=root if i==0 else root/'rounds'/f'round{i}'
            if cohort=='FIT': base=base/'fit'
            plans={p['original_ordinal']:p for p in read_rows(base/'cohort/plans.jsonl')}
            panels={}
            for stage in ['construction','refined','tokenized','edited']+(['proposal'] if cohort=='FIT' and i<3 else []):
                rows=scores(base,stage);assert len(rows)==256
                panels[stage]={x['sample_idx']:x for x in rows}
                report['input_hashes'][str(score_directory(base,stage)/'attempt_results.jsonl')]=file_hash(score_directory(base,stage)/'attempt_results.jsonl')
                sun=[x for x in rows if x['strict_sun'] is True]
                report['stages'].append(dict(cohort=cohort,theta=i,stage=stage,SUN=len(sun),
                    SUN_terminal_status=dict(Counter(x['terminal_status'] for x in sun)),
                    MSUN_invalid_terminal=sum(x['meta_sun'] is True and x['terminal_status']=='invalid_terminal' for x in rows),
                    verified_SUN=sum(x['verified_strict_sun'] is True for x in rows)))
                for x in rows:
                    if x.get('e_above_hull_eV_atom') is not None and x['e_above_hull_eV_atom'] < -1:
                        report['energy_outliers'].append(dict(cohort=cohort,theta=i,stage=stage,
                            formula=plans[x['sample_idx']]['plan_state']['formula'],score=x))
            a,b=panels['tokenized'],panels['edited']
            for k in a:
                if a[k]['strict_sun']!=b[k]['strict_sun']:
                    report['editor_SUN_changes'].append(dict(cohort=cohort,theta=i,id=k,
                        formula=plans[k]['plan_state']['formula'],before=a[k],after=b[k]))
            if cohort=='FIT' and i<3:
                for branch,left,right in [('G','construction','tokenized'),('E','tokenized','proposal')]:
                    examples=read_rows(base/'pairs'/f'{branch}.jsonl')
                    report['input_hashes'][str(base/'pairs'/f'{branch}.jsonl')]=file_hash(base/'pairs'/f'{branch}.jsonl')
                    ids={p['ancestor_id']:k for k,p in plans.items()};counts=Counter();details=[]
                    for e in examples:
                        k=ids[e['source_id']];a,b=panels[left][k],panels[right][k]
                        invalid_a=a['terminal_status']=='invalid_terminal'
                        invalid_b=b['terminal_status']=='invalid_terminal'
                        counts['examples']+=1
                        counts['any_invalid_terminal']+=invalid_a or invalid_b
                        counts['known_SUN_invalid_terminal']+=invalid_a and e.get('known_sun') is True
                        if invalid_a or invalid_b:
                            details.append(dict(source_id=e['source_id'],id=k,invalid_before=invalid_a,
                                invalid_after=invalid_b,known_sun=e['known_sun'],mode_target=e.get('mode_target'),
                                accept_target=e.get('accept_target'),has_preference=e.get('chosen_tokens') is not None,
                                before_hull=a['e_above_hull_eV_atom'],after_hull=b['e_above_hull_eV_atom']))
                    report['training_label_exposure'].append(dict(source_theta=i,branch=branch,counts=dict(counts),details=details))
    for name in ['scripts/evaluate_programmed_paths.py','scripts/label_programmed_paths.py',
                 'src/crystal_dlm/post_refine_contract.py','src/scripts/run_rsi_stages.py']:
        report['input_hashes'][str(source/name)]=file_hash(source/name)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['root','source','output']: parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args();report=analyze(args.root,args.source)
    raw=(json.dumps(report,sort_keys=True,indent=2)+'\n').encode()
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_bytes(raw)
    print(json.dumps({'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),
                      'stages':len(report['stages']),'outliers':len(report['energy_outliers'])}))
