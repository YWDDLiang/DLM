"""Separate measured candidate coverage from deployable SUN policy outcomes."""
from collections import Counter
import argparse
import json
from pathlib import Path
import sys

SOURCE=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(SOURCE/'src'),str(SOURCE/'operations/r03_c3fd_main_20260907')]
from scripts.run_post_refine_cycle import read_rows,write_json,file_hash
from scripts.run_rsi_stages import scores,score_directory
from scripts.run_sun_rank_scope import STREAMS
from crystal_dlm.sun_ranker import endpoint_targets
from editor_trial_analysis import flags


def analyze(root):
    frozen=root/'FROZEN_SELECTION.json'
    if not frozen.exists():raise ValueError('complete old-FINAL supply analysis waits for policy freeze')
    reg=json.loads((root/'PREREGISTRATION.json').read_text());previous=Path(reg['previous_run'])
    before=scores(previous/'fit','native');roles=read_rows(root/'SOURCE_SPLIT.jsonl')
    panels={'primary':previous/'fit/hybrid_proposal'}
    panels.update({name:root/f'fit/bank/{name}/candidate' for name in STREAMS if name!='primary'})
    labels={name:scores(path.parent,path.name) for name,path in panels.items()}
    chosen=Path(json.loads(frozen.read_text())['selected']['panel'])
    final=scores(chosen,'edited');decisions=json.loads((chosen/'DECISION_BINDING.json').read_text())['decisions']
    rows=[]
    for i,a in enumerate(before):
        source=roles[i];current=flags(a);target,_=endpoint_targets(a);candidates={}
        for name,bank in labels.items():
            b=bank[i];q=flags(b);value,status=endpoint_targets(b)
            candidates[name]=dict(Stable=q['Stable'],MS=q['MS'],novel=q['N'],known=q['reliable'],
                target_status=status,hull=q['hull'],new_novel_Stable=bool(target is not None and value is not None and value[0]>target[0]),
                true_Stable_promotion=bool(q['Stable'] and not current['Stable']),
                true_Stable_loss=bool(current['Stable'] and not q['Stable']))
        potential=[name for name,c in candidates.items() if c['new_novel_Stable']]
        got=flags(final[i]);ns_after=endpoint_targets(final[i])[0]
        captured=bool(target is not None and ns_after is not None and ns_after[0]>target[0])
        rows.append(dict(ordinal=i,source_id=source['ancestor_id'],split=source['split'],current=current,
            candidates=candidates,new_NS_candidate_streams=potential,selected_stream=decisions[i]['chosen_stream'],
            selected_actual_edit=decisions[i]['actual_edit'],selected_NS_promotion=captured,
            selected_SUN_promotion=not current['SUN'] and got['SUN'],
            selected_SUN_loss=current['SUN'] and not got['SUN'],
            missed_measured_NS_opportunity=bool(potential and not captured)))
    result={}
    for role in ('train','dev','final','all'):
        selected=[r for r in rows if (role=='all' or r['split']==role) and not (role=='final' and r['ordinal']==300)]
        result[role]=dict(requests=len(selected),
            per_stream={name:dict(new_NS_candidate_rows=sum(r['candidates'][name]['new_novel_Stable'] for r in selected),
                true_Stable_promotion_rows=sum(r['candidates'][name]['true_Stable_promotion'] for r in selected),
                true_Stable_loss_rows=sum(r['candidates'][name]['true_Stable_loss'] for r in selected)) for name in STREAMS},
            pool_sources_with_NS_promotion=sum(bool(r['new_NS_candidate_streams']) for r in selected),
            pool_sources_with_true_Stable_promotion=sum(any(c['true_Stable_promotion'] for c in r['candidates'].values()) for r in selected),
            selected_NS_promotions=sum(r['selected_NS_promotion'] for r in selected),
            selected_SUN_promotions=sum(r['selected_SUN_promotion'] for r in selected),
            selected_SUN_losses=sum(r['selected_SUN_loss'] for r in selected),
            missed_NS_sources=[r['ordinal'] for r in selected if r['missed_measured_NS_opportunity']],
            selected_stream_counts=dict(Counter(str(r['selected_stream']) for r in selected)))
    path=root/'analysis/CANDIDATE_SUPPLY_FINAL.json'
    write_json(path,dict(schema='measured_SUN_supply_vs_selection_v1',reports=result,rows=rows,
        candidate_coverage_is_not_a_deployable_policy=True,candidate_NS_excludes_cross_candidate_pool_U=True,
        selected_policy_sha256=file_hash(chosen/'DECISION_BINDING.json'),
        input_score_pins={str(p):file_hash(score_directory(p.parent,p.name)/'attempt_results.jsonl') for p in panels.values()}))
    print(json.dumps(dict(reports=result,path=str(path),sha256=file_hash(path))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args();analyze(a.root)
