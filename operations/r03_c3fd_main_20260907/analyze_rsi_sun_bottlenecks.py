"""TRAIN-only, read-only attribution of existing SUN bottlenecks and labels."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys


def analyze(root,source):
    sys.path.insert(0,str(source/'src'))
    from scripts.run_rsi_stages import scores,score_directory,read_rows,file_hash
    result={'scope':'Existing TRAIN only; no new neural calls or physical scoring.',
            'stages':[],'transitions':[],'editor_targets':[],'generator_targets':[],
            'input_hashes':{}}
    for i in range(4):
        base=(root if i==0 else root/'rounds'/f'round{i}')/'fit'
        plans=read_rows(base/'cohort/plans.jsonl');ids={p['ancestor_id']:p['original_ordinal'] for p in plans}
        panels={}
        for stage in ['construction','refined','tokenized','edited']+(['proposal'] if i<3 else []):
            rows=scores(base,stage);assert len(rows)==256
            panels[stage]={x['sample_idx']:x for x in rows}
            result['input_hashes'][str(score_directory(base,stage)/'attempt_results.jsonl')]=file_hash(score_directory(base,stage)/'attempt_results.jsonl')
            c=Counter();near=Counter()
            for x in rows:
                c['requested']+=1;c['SUN']+=x['strict_sun'] is True
                c['strict_stable']+=x['strict_stable'] is True
                c['meta_stable']+=x['meta_stable'] is True
                if x['strict_stable'] is True and x['strict_sun'] is not True:
                    c['stable_but_not_SUN']+=1
                    c['stable_non_novel']+=x['novel'] is False
                    c['stable_non_unique']+=x['unique_representative'] is False
                e=x.get('e_above_hull_eV_atom')
                if e is not None and e>0 and x['novel_unique'] is True:
                    tag='0_to_0.01' if e<=.01 else '0.01_to_0.03' if e<=.03 else '0.03_to_0.1' if e<=.1 else 'above_0.1'
                    near[tag]+=1
                    if x['terminal_verified']: near['verified_'+tag]+=1
            result['stages'].append(dict(theta=i,stage=stage,counts=dict(c),nonSUN_novel_unique_hull_bins=dict(near)))
        for before,after in [('construction','refined'),('refined','tokenized'),('tokenized','edited')]:
            a,b=panels[before],panels[after]
            gains=[k for k in a if a[k]['strict_sun'] is False and b[k]['strict_sun'] is True]
            losses=[k for k in a if a[k]['strict_sun'] is True and b[k]['strict_sun'] is False]
            result['transitions'].append(dict(theta=i,before=before,after=after,gains=len(gains),losses=len(losses),
                gain_status=dict(Counter(b[k]['terminal_status'] for k in gains)),gain_ids=gains,loss_ids=losses))
        if i==3: continue
        for branch,before,after in [('G','construction','tokenized'),('E','tokenized','proposal')]:
            examples=read_rows(base/'pairs'/f'{branch}.jsonl')
            result['input_hashes'][str(base/'pairs'/f'{branch}.jsonl')]=file_hash(base/'pairs'/f'{branch}.jsonl')
            c=Counter();new_sun=[]
            for e in examples:
                a,b=panels[before][ids[e['source_id']]],panels[after][ids[e['source_id']]]
                if branch=='E': positive=e.get('mode_target')==1
                else:
                    raw=json.loads((base/'construction/records'/f"{ids[e['source_id']]:04d}.json").read_text())['record']['body_token_ids']
                    positive=e.get('chosen_tokens') is not None and e['chosen_tokens']!=raw
                c['examples']+=1;c['positive_after_targets']+=positive
                if positive:
                    category='SUN_gain' if b['strict_sun'] is True and a['strict_sun'] is not True else 'SUN_retained' if b['strict_sun'] is True else 'no_SUN_after'
                    c[category]+=1
                    if category=='no_SUN_after':
                        c['nonSUN_positive_MSUN_gain']+=b['meta_sun'] is True and a['meta_sun'] is not True
                    if category=='SUN_gain': c['SUN_gain_'+b['terminal_status']]+=1
                if branch=='E' and b['strict_sun'] is True and a['strict_sun'] is False:
                    k=ids[e['source_id']];t=json.loads((base/'edited/records'/f'{k:04d}.json').read_text())['editor_trace']
                    new_sun.append(dict(source_id=e['source_id'],id=k,positive_target=positive,
                        status=b['terminal_status'],hull=b['e_above_hull_eV_atom'],mode=t.get('learned_decision'),
                        accept=t.get('learned_accept_probability'),applied=t.get('applied',False)))
            result['editor_targets' if branch=='E' else 'generator_targets'].append(dict(source_theta=i,counts=dict(c),SUN_gain_proposals=new_sun))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['root','source','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();q=analyze(a.root,a.source);raw=(json.dumps(q,sort_keys=True,indent=2)+'\n').encode()
    a.output.write_bytes(raw)
    print(json.dumps({'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}))
