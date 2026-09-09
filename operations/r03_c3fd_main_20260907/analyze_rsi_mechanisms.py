"""Read-only mechanism diagnostics on an existing complete RSI experiment.

No model inference, relaxation, training, or outcome-based cohort selection.
Counterfactual action counts describe the logged TRAIN proposals; they are not
SUN scores for a newly mixed cohort, whose N/U must be recomputed separately.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
import random
from pathlib import Path
import statistics
import sys


def describe(values):
    x=sorted(float(v) for v in values if v is not None and math.isfinite(float(v)))
    if not x: return {'n':0}
    return {'n':len(x),'mean':statistics.fmean(x),'median':statistics.median(x),
            'p10':x[int((len(x)-1)*.1)],'p90':x[int((len(x)-1)*.9)],'min':x[0],'max':x[-1]}


def failure_reason(row,metric):
    stable='strict_stable' if metric=='SUN' else 'meta_stable'
    if not row.get('reconstructed') or not row.get('native_execution_success'):
        return 'execution_or_geometry'
    if row.get(stable) is not True: return 'energy_stability'
    n,u=row.get('novel'),row.get('unique_representative')
    if n is False and u is False: return 'novelty_and_uniqueness'
    if n is False: return 'novelty'
    if u is False: return 'uniqueness'
    return 'unknown_or_other'


def confusion(rows):
    out=Counter()
    for actual,predicted in rows:
        out[('T' if actual==predicted else 'F')+('P' if predicted else 'N')]+=1
    tp,fp,fn=out['TP'],out['FP'],out['FN']
    return {**{k:out[k] for k in ['TP','FP','FN','TN']},
            'precision':tp/(tp+fp) if tp+fp else None,
            'recall':tp/(tp+fn) if tp+fn else None}


def analyze(root,source):
    sys.path.insert(0,str(source/'src'))
    from scripts.run_post_refine_cycle import read_rows,file_hash
    from scripts.run_rsi_stages import scores,score_directory
    result={'scope':'Existing fixed 256 MAIN and 256 TRAIN, one fixed seed per Plan, no new model calls.',
            'stage_physics':[],'availability_decomposition':[],
            'version_transitions':[],'editor_diagnostics':[],
            'generator_pair_diagnostics':[],'training_exposure':[],'source_files':{},
            'fixed_slot_diagnostics':[],'input_hashes':{}}
    for name in ['src/crystal_dlm/rsi_preference.py','src/scripts/train_rsi_preferences.py',
                 'src/scripts/run_rsi_stages.py','scripts/evaluate_programmed_paths.py',
                 'scripts/label_programmed_paths.py']:
        result['source_files'][name]=file_hash(source/name)
    panels={};bases={};plans_by={}
    for i in range(4):
        for cohort,base in [('MAIN',root if i==0 else root/'rounds'/f'round{i}'),
                            ('FIT',root/'fit' if i==0 else root/'rounds'/f'round{i}'/'fit')]:
            bases[cohort,i]=base;plans=read_rows(base/'cohort/plans.jsonl');plans_by[cohort,i]=plans
            assert len(plans)==256
            for stage in ['construction','refined','tokenized','edited']+(['proposal'] if cohort=='FIT' and i<3 else []):
                rows=scores(base,stage);assert len(rows)==256
                panels[cohort,i,stage]={v['sample_idx']:v for v in rows}
                row=dict(cohort=cohort,theta=i,stage=stage,requested=256,
                    reconstructed=sum(x['reconstructed'] for x in rows),
                    verified=sum(x['terminal_verified'] for x in rows),
                    strict_stable=sum(x['strict_stable'] is True for x in rows),
                    meta_stable=sum(x['meta_stable'] is True for x in rows),
                    SUN=sum(x['strict_sun'] is True for x in rows),MSUN=sum(x['meta_sun'] is True for x in rows),
                    score_sha256=file_hash(score_directory(base,stage)/'attempt_results.jsonl'))
                for field in ['e_above_hull_eV_atom','raw_energy_eV_atom','gap_eV_atom','actual_relaxation_steps']:
                    row[field]=describe(x.get(field) for x in rows)
                row['force_max']=describe((x.get('raw') or {}).get('force_max_eV_A') for x in rows)
                result['stage_physics'].append(row)

    for cohort in ['MAIN','FIT']:
        for i in [1,2,3]:
            oldraw,newraw=panels[cohort,i-1,'construction'],panels[cohort,i,'construction']
            for stage in ['construction','refined','tokenized','edited']:
                old,new=panels[cohort,i-1,stage],panels[cohort,i,stage]
                groups={};changes={}
                for name,flag in [('SUN','strict_sun'),('MSUN','meta_sun')]:
                    gains=[k for k in old if old[k][flag] is False and new[k][flag] is True]
                    losses=[k for k in old if old[k][flag] is True and new[k][flag] is False]
                    details=[]
                    for k in losses:
                        a,b=old[k],new[k];threshold=0 if name=='SUN' else .1
                        margin=b.get('e_above_hull_eV_atom')
                        details.append(dict(id=k,reason=failure_reason(b,name),
                            before_hull=a.get('e_above_hull_eV_atom'),after_hull=margin,
                            after_threshold_margin=margin-threshold if margin is not None else None,
                            raw_before_hull=oldraw[k].get('e_above_hull_eV_atom'),
                            raw_after_hull=newraw[k].get('e_above_hull_eV_atom'),
                            formula=next(p['plan_state']['formula'] for p in plans_by[cohort,i] if p['original_ordinal']==k)))
                    changes[name]=dict(gains=len(gains),losses=len(losses),gain_ids=gains,loss_ids=losses,
                        loss_reasons=dict(Counter(x['reason'] for x in details)),loss_details=details,
                        near_threshold_losses=sum(x['reason']=='energy_stability' and x['after_threshold_margin'] is not None and 0<x['after_threshold_margin']<=.02 for x in details))
                for k in old:
                    a,b=oldraw[k]['native_execution_success'],newraw[k]['native_execution_success']
                    group='both_valid' if a and b else 'recovered' if b else 'new_failure' if a else 'both_failed'
                    g=groups.setdefault(group,dict(n=0,SUN_before=0,SUN_after=0,MSUN_before=0,MSUN_after=0))
                    g['n']+=1
                    for label,flag in [('SUN','strict_sun'),('MSUN','meta_sun')]:
                        g[label+'_before']+=old[k][flag] is True;g[label+'_after']+=new[k][flag] is True
                result['availability_decomposition'].append(dict(cohort=cohort,before=i-1,after=i,stage=stage,groups=groups))
                result['version_transitions'].append(dict(cohort=cohort,before=i-1,after=i,stage=stage,metrics=changes))

    for cohort in ['MAIN','FIT']:
        for i in range(4):
            base=bases[cohort,i];plans=plans_by[cohort,i]
            traces={p['original_ordinal']:json.loads((base/'edited/records'/f"{p['original_ordinal']:04d}.json").read_text())['editor_trace'] for p in plans}
            healthy=[t for t in traces.values() if t.get('known_sun')]
            row=dict(cohort=cohort,theta=i,mode_counts=dict(Counter(t.get('learned_decision','upstream_failure') for t in traces.values())),
                known_SUN=len(healthy),known_SUN_unshifted_keep=sum(t['mode_logits_keep_edit'][0]-math.log(9)>=t['mode_logits_keep_edit'][1] for t in healthy),
                known_SUN_unshifted_margin=describe(t['mode_logits_keep_edit'][0]-math.log(9)-t['mode_logits_keep_edit'][1] for t in healthy),
                known_SUN_prior_flipped_to_keep=sum(t['learned_decision']=='KEEP' and t['mode_logits_keep_edit'][0]-math.log(9)<t['mode_logits_keep_edit'][1]-1e-6 for t in healthy),
                generated=sum(t.get('proposal_generated',False) for t in traces.values()),applied=sum(t.get('applied',False) for t in traces.values()))
            numeric_changes=[];cell_changes=[]
            for k,t in traces.items():
                if not t.get('proposal_generated'): continue
                before,after=t['current_tokens'],t['proposal_tokens']
                cell_changes.append(sum(before[j]!=after[j] for j in range(1,7)))
                numeric_changes.append(sum(a!=b for a,b in zip(before,after)))
            row.update(proposal_token_changes=describe(numeric_changes),proposal_cell_changes=describe(cell_changes))
            if cohort=='FIT' and i<3:
                examples=read_rows(base/'pairs/E.jsonl');by_source={x['source_id']:x for x in examples}
                decisions=[];acceptances=[];effective=[];accept_only=[];missed=Counter();errors=[]
                for p in plans:
                    e=by_source.get(p['ancestor_id']);t=traces[p['original_ordinal']]
                    if e is None or e.get('mode_target') is None: continue
                    y=bool(e['mode_target']);mode=t.get('learned_decision')=='EDIT';applied=bool(t.get('applied'))
                    accept=t.get('proposal_generated',False) and t.get('learned_accept_probability',0)>=.5
                    decisions.append((y,mode));effective.append((y,applied))
                    accept_only.append((y,accept and not e.get('known_sun',False)))
                    if e.get('accept_target') is not None and t.get('proposal_generated'):
                        acceptances.append((bool(e['accept_target']),accept))
                    if y:
                        missed['captured' if applied else 'mode_keep' if not mode else 'proposal_failure' if not t.get('proposal_generated') else 'accept_reject']+=1
                    if t.get('proposal_generated') and e.get('accept_target') is not None:
                        errors.append((t['learned_accept_probability']-e['accept_target'])**2)
                row.update(target_examples=len(decisions),positive_targets=sum(y for y,_ in decisions),
                    mode_confusion=confusion(decisions),accept_confusion=confusion(acceptances),
                    effective_confusion=confusion(effective),accept_only_confusion=confusion(accept_only),
                    beneficial_proposal_fate=dict(missed),accept_Brier=statistics.fmean(errors) if errors else None)
                a,b=panels[cohort,i,'tokenized'],panels[cohort,i,'proposal']
                row['proposal_survival']={label:dict(before=sum(x[flag] is True for x in a.values()),
                    retained=sum(a[k][flag] is True and b[k][flag] is True for k in a),
                    new=sum(a[k][flag] is False and b[k][flag] is True for k in a))
                    for label,flag in [('SUN','strict_sun'),('MSUN','meta_sun')]}
            result['editor_diagnostics'].append(row)

    for i in range(3):
        base=bases['FIT',i];examples=read_rows(base/'pairs/G.jsonl')
        result['input_hashes'][str(base/'pairs/G.jsonl')]=file_hash(base/'pairs/G.jsonl')
        raw_by_source={p['ancestor_id']:json.loads((base/'construction/records'/f"{p['original_ordinal']:04d}.json").read_text())['record'] for p in plans_by['FIT',i]}
        slot_counts=Counter();slot_examples=[]
        for e in examples:
            if e.get('chosen_tokens') is None: continue
            original=raw_by_source[e['source_id']]['body_token_ids']
            winner,loser=e['chosen_tokens'],e['rejected_tokens']
            assert original==winner or original==loser
            expected=original[7::4]
            slot_counts['pairs']+=1
            slot_counts['winner_unreachable_fixed_slots']+=winner[7::4]!=expected
            slot_counts['loser_unreachable_fixed_slots']+=loser[7::4]!=expected
            slot_counts['type_order_mismatch']+=winner[7::4]!=loser[7::4]
            assert Counter(winner[7::4])==Counter(loser[7::4])==Counter(expected)
            slot_counts['same_composition']+=1
            for target in [winner,loser]:
                blocks=[tuple(target[j:j+4]) for j in range(7,len(target),4)]
                remaining=blocks.copy();ordered=[]
                for element in expected:
                    index=next(j for j,b in enumerate(remaining) if b[0]==element)
                    ordered.append(remaining.pop(index))
                aligned=target[:7]+[v for b in ordered for v in b]
                assert not remaining and aligned[7::4]==expected and aligned[0]==original[0]
                assert aligned[:7]==target[:7] and Counter(ordered)==Counter(blocks)
                slot_counts['permutation_only_alignment_proved']+=1
            if winner[7::4]!=loser[7::4] and len(slot_examples)<2:
                slot_examples.append(dict(source_id=e['source_id'],expected_types=expected,
                    winner_types=winner[7::4],loser_types=loser[7::4]))
        result['fixed_slot_diagnostics'].append(dict(source_theta=i,counts=dict(slot_counts),examples=slot_examples,
            proof_scope='CPU only: unchanged cell and exact multiset of (element,x,y,z) token blocks. No training data written, no full dynamic numeric-support or neural validation.'))
        total=changed=zero=context_only=fixed_mismatch=0;counts=[];kinds=Counter()
        for e in examples:
            a,b=e.get('chosen_tokens'),e.get('rejected_tokens')
            if a is None or b is None: continue
            n=e['num_sites'];order=list(range(1,7))+[8+4*s+axis for axis in range(3) for s in range(n)]
            assert len(a)==len(b)==7+4*n
            prefix_equal=all(a[j]==b[j] for j in range(len(a)) if j not in order)
            fixed_mismatch+=not prefix_equal;d=0
            for pos in order:
                total+=1
                if a[pos]!=b[pos]: changed+=1;d+=1;kinds['cell' if pos<7 else 'coordinate']+=1
                elif prefix_equal: zero+=1
                else: context_only+=1
                prefix_equal=prefix_equal and a[pos]==b[pos]
            counts.append(d/len(order))
        result['generator_pair_diagnostics'].append(dict(source_theta=i,pairs=len(counts),numeric_cut_count=total,
            direct_changed_targets=changed,identical_context_and_target_zero_DPO=zero,
            same_target_different_context=context_only,changed_fraction=describe(counts),changed_kinds=dict(kinds),
            fixed_slot_mismatch_pairs=fixed_mismatch,
            initial_winner_SFT_coefficient=.2,initial_winner_DPO_coefficient=.05,initial_rejected_DPO_coefficient=.05))

    # Replay the exact Python RNG choices used by the trainer. This counts
    # supervision exposure only; no neural computation or fitted claim.
    for update in [1,2,3]:
        for branch in ['G','E']:
            path=root/'training'/f'round{update}'/branch/'result/checkpoint/RSI_TRAINING_DONE.json'
            receipt=json.loads(path.read_text());cfg=receipt['contract']
            assert file_hash(cfg['data'])==cfg['data_sha256']
            for pin in cfg.get('historical_replay',[]): assert file_hash(pin['path'])==pin['sha256']
            current=read_rows(cfg['data']);replay=[]
            for value in cfg.get('replay_data',[]): replay.extend(read_rows(value))
            pools={'current':current,'replay':replay};draws=Counter();seen=set();cuts=set();geometry_cuts=set();anchors=Counter()
            ranks=[]
            for rank in range(cfg['world_size']):
                rng=random.Random(cfg['seed']+rank);pairs=heads=0
                for step in range(receipt['optimizer_steps']):
                    for micro in range(cfg['accumulation']):
                        name='replay' if replay and rng.random()<.25 else 'current'
                        pool=pools[name];index=rng.randrange(len(pool));e=pool[index]
                        cut=rng.randrange(6+3*e['num_sites'])
                        draws[name]+=1;seen.add((name,index));cuts.add((name,index,cut))
                        paired=e.get('chosen_tokens') is not None and e.get('rejected_tokens') is not None
                        pairs+=paired
                        if paired or branch=='G' and e.get('healthy_anchor_tokens') is not None:
                            geometry_cuts.add((name,index,cut))
                        heads+=branch=='E' and e.get('mode_target') is not None
                        anchors[name]+=branch=='G' and e.get('healthy_anchor_tokens') is not None
                ranks.append(dict(rank=rank,pairs=pairs,heads=heads))
            assert ranks[0]['pairs']==receipt['local_preference_examples']
            assert ranks[0]['heads']==receipt['local_decision_examples']
            current_seen=sum(name=='current' for name,_ in seen)
            current_cuts=sum(name=='current' for name,_,_ in cuts)
            current_geometry_cuts=sum(name=='current' for name,_,_ in geometry_cuts)
            potential=sum(6+3*e['num_sites'] for e in current if e.get('chosen_tokens') is not None or branch=='G' and e.get('healthy_anchor_tokens') is not None)
            result['training_exposure'].append(dict(update=update,branch=branch,
                actual_optimizer_steps=receipt['optimizer_steps'],global_micro_draws=dict(draws),
                current_examples=len(current),current_unique_seen=current_seen,
                current_unseen=len(current)-current_seen,current_unique_drawn_position_cuts=current_cuts,
                current_unique_supervised_position_cuts=current_geometry_cuts,
                current_possible_supervised_position_cuts=potential,current_position_coverage=current_geometry_cuts/potential,
                healthy_anchor_draws=dict(anchors),ranks=ranks,rank0_receipt_counts_reproduced=True))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    report=analyze(a.root,a.source)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    raw=(json.dumps(report,sort_keys=True,indent=2)+'\n').encode()
    a.output.write_bytes(raw)
    print(json.dumps({'output':str(a.output),'sha256':hashlib.sha256(raw).hexdigest(),
                     'stages':len(report['stage_physics']),'transitions':len(report['version_transitions']),
                     'editors':len(report['editor_diagnostics'])}))
