"""Compare registered SUN policies on actual selected and rescored endpoints."""
from __future__ import annotations
import argparse
import copy
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

SOURCE=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(SOURCE/'src'))
from scripts.run_post_refine_cycle import read_rows,write_rows,write_json,file_hash
from scripts.run_rsi_stages import materialize,rebind_labels,scores,score_directory,validity
from scripts.run_sun_rank_scope import STREAMS
from crystal_dlm.sun_ranker import select_candidate
from crystal_dlm.post_refine_contract import fingerprint
from editor_trial_analysis import flags


def build(root,panel_name,tag,sun_threshold,ms_floor,*,operational=False,primary_only=False,
          prediction_path=None,score_kind='signed_gains'):
    data=root/panel_name;panel=root/'policies'/panel_name/tag
    if (panel/'DECISION_BINDING.json').exists():
        if file_hash(panel/'edited/inputs.jsonl')!=json.loads((panel/'DECISION_BINDING.json').read_text())['inputs_sha256']:
            raise ValueError('registered policy inputs changed')
        return panel
    if panel.exists():raise ValueError('partial policy requires explicit inspection')
    previous=Path(json.loads((root/'PREREGISTRATION.json').read_text())['previous_run'])
    prediction_path=prediction_path or root/('training/sun_ranker/FIT_PREDICTIONS.jsonl' if panel_name=='fit' else 'fresh/ranker_inference/FRESH_PREDICTIONS.jsonl')
    predictions={}
    for row in read_rows(prediction_path):predictions[(row['stream'],row['ordinal'])]=row
    current=read_rows(data/'native/inputs.jsonl')
    observed=scores(previous/'fit','native') if panel_name=='fit' else scores(data,'native')
    spec=json.loads((data/'RUN_SPEC.json').read_text());spec.update(run_root=str(panel),run_id='sun_rank_policy:'+panel_name+':'+tag,
        training_parent_root=str(panel/'cohort'))
    streams=['primary'] if primary_only else list(STREAMS)
    panel.mkdir(parents=True);shutil.copytree(data/'cohort',panel/'cohort');write_json(panel/'RUN_SPEC.json',spec)
    decisions=[]
    for i,native in enumerate(current):
        candidates=[predictions.get((name,i),dict(stream=name,valid=False,sun_gain=0.,ms_gain=0.,operational_utility=0.)) for name in streams]
        guard=flags(observed[i])['SUN']
        if operational:
            eligible=[(c['operational_utility'],-k,c['stream']) for k,c in enumerate(candidates)
                if c['valid'] and c['operational_utility']>=.05]
            chosen=max(eligible)[2] if eligible and not guard else None
        else:chosen=select_candidate(candidates,sun_threshold=sun_threshold,ms_floor=ms_floor,known_sun=guard)
        record=native
        if chosen is not None:
            bank=data/'bank'/chosen;bound=json.loads((bank/f'bound/{i:04d}.json').read_text())
            registered=json.loads((bank/'COLLECTION_FINAL.json').read_text())['bindings'][i]
            if fingerprint(bound['record'])!=registered['record_sha256']:raise ValueError('selected endpoint bytes changed')
            if not bound['commit_trace']['applied']:raise ValueError('selected proposal has no actual edit')
            record=bound['record']
        calls=0
        for name in streams:
            trace=json.loads((data/f'bank/{name}/candidate/records/{i:04d}.json').read_text())['editor_trace']
            calls+=trace.get('forward_calls',0)+int((name,i) in predictions)
        if calls>80:raise ValueError('ranker exceeded the per-request DLM budget')
        decision=dict(ordinal=i,chosen_stream=chosen,actual_edit=chosen is not None,known_SUN_guard=guard,
            source_trajectory_id=record['trajectory_id'],source_record_sha256=fingerprint(record),DLM_forward_calls=calls)
        result=copy.deepcopy(record);result['trajectory_id']=f'{spec["run_id"]}:edited:{i}'
        write_json(panel/f'edited/records/{i:04d}.json',dict(record=result,editor_trace=decision));decisions.append(decision)
    materialize(spec,'edited')
    write_json(panel/'DECISION_BINDING.json',dict(decisions=decisions,sun_threshold=sun_threshold,ms_floor=ms_floor,
        operational_comparator=operational,primary_only=primary_only,score_kind=score_kind,model_choice_without_candidate_physics=True,
        predictions_path=str(prediction_path),predictions_sha256=file_hash(prediction_path),
        inputs_sha256=file_hash(panel/'edited/inputs.jsonl')))
    return panel


def evaluate(root,panel,*,cached=True,nu_workers=7):
    spec=json.loads((panel/'RUN_SPEC.json').read_text());output=panel/'edited/scoring/result'
    if (output/'_SUCCESS').exists():return
    if cached and not (panel/'edited/labeling/result/_SUCCESS').exists():
        reg=json.loads((root/'PREREGISTRATION.json').read_text());previous=Path(reg['previous_run'])
        source=[previous/'fit/native',previous/'fit/hybrid_proposal']
        source += [root/f'fit/bank/{name}/candidate' for name in STREAMS
                   if name!='primary' or reg.get('editor_content_changed')]
        rebind_labels(spec,'edited',source_directories=source)
    if not (panel/'edited/labeling/result/_SUCCESS').exists():raise ValueError('policy needs complete endpoint labels')
    command=[sys.executable,str(SOURCE/'scripts/evaluate_programmed_paths.py'),
        '--paths-jsonl',str(panel/'edited/inputs.jsonl'),'--labels-jsonl',str(panel/'edited/labeling/result/labels.jsonl'),
        '--frozen-config',spec['assets']['frozen_config'],'--official-cache',spec['assets']['official_cache'],
        '--output-dir',str(output),'--expected-requests',str(spec['requests']),
        '--endpoint','native','--cohort-role','training_feedback','--policy-stage','round0_diagnostic',
        '--sun-only','--nu-workers',str(nu_workers),'--nu-cache',str(root/'nu_cache'),
        '--feedback-manifest',str(panel/'edited/FEEDBACK_MANIFEST.json'),'--joint-physical-stop']
    write_json(panel/'SCORE_COMMAND.json',dict(command=command))
    with (panel/'score.out').open('x') as out,(panel/'score.err').open('x') as err:
        completed=subprocess.run(command,stdout=out,stderr=err)
    if completed.returncode:raise RuntimeError('SUN policy scoring failed:'+str(panel))
    validity(spec,'edited')


def summary(root,panel,role):
    reg=json.loads((root/'PREREGISTRATION.json').read_text());previous=Path(reg['previous_run'])
    fresh=role=='fresh';before=scores(root/'fresh','native') if fresh else scores(previous/'fit','native')
    after=scores(panel,'edited');decisions=json.loads((panel/'DECISION_BINDING.json').read_text())['decisions']
    split=read_rows(root/'SOURCE_SPLIT.jsonl')
    indices=range(len(after)) if role in ('all','fresh') else [i for i,r in enumerate(split) if r['split']==role]
    if role=='final':indices=[i for i in indices if i!=300]
    metrics=('Stable','MS','SUN','MSUN');base={k:0 for k in metrics};count=dict(base)
    gains={k:[] for k in metrics};losses={k:[] for k in metrics}
    for i in indices:
        a,b=flags(before[i]),flags(after[i])
        for k in metrics:
            base[k]+=int(a[k]);count[k]+=int(b[k])
            if b[k] and not a[k]:gains[k].append(i)
            if a[k] and not b[k]:losses[k].append(i)
    return dict(role=role,requests=len(indices),KEEP=base,counts=count,delta={k:count[k]-base[k] for k in metrics},
        gains=gains,losses=losses,actual_edits=sum(decisions[i]['actual_edit'] for i in indices),
        max_DLM_calls=max(decisions[i]['DLM_forward_calls'] for i in indices),panel=str(panel),
        scores_sha256=file_hash(score_directory(panel,'edited')/'attempt_results.jsonl'))


def policies(root,output):
    reg=json.loads((root/'PREREGISTRATION.json').read_text());rule=reg['selection'];candidates=[]
    training=root/'training/sun_ranker/TRAINING_FINAL.json';report=json.loads(training.read_text())
    if file_hash(root/'training/sun_ranker/FIT_PREDICTIONS.jsonl')!=report['predictions_sha256']:
        raise ValueError('trained predictions changed')
    for threshold in rule['sun_thresholds']:
        for floor in rule['ms_floors']:
            tag=f'sun{round(threshold*100):02d}_ms{round(floor*100):03d}'
            panel=build(root,'fit',tag,threshold,floor);evaluate(root,panel);dev=summary(root,panel,'dev')
            candidate=dict(sun_threshold=threshold,ms_floor=floor,panel=str(panel),DEV=dev)
            candidates.append(candidate);write_json(panel/'DEV_RESULT.json',dev)
            print(json.dumps(dict(policy=tag,DEV=dev['counts'],delta=dev['delta'],edits=dev['actual_edits'])),flush=True)
    def accepted(c):
        delta=c['DEV']['delta'];return delta['SUN']>0 and delta['MSUN']>=0 and delta['Stable']>=0
    eligible=[c for c in candidates if accepted(c)]
    def key(c):
        d=c['DEV'];return (d['counts']['SUN'],d['counts']['MSUN'],-d['actual_edits'],c['sun_threshold'],c['ms_floor'])
    selected=max(eligible,key=key) if eligible else next(c for c in candidates if c['sun_threshold']==.05 and c['ms_floor']==0.)
    frozen=dict(schema='frozen_SUN_ranker_DEV_selection_v1',created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        selected=selected,DEV_admitted=bool(eligible),adoption='candidate' if eligible else 'KEEP; candidate is diagnostic only',
        model_path=report['model_path'],model_sha256=report['model_sha256'],
        training_receipt_sha256=file_hash(training),fresh_quality_consulted=False,rule=rule)
    if (root/'FROZEN_SELECTION.json').exists():raise ValueError('policy selection is already frozen')
    write_json(root/'DEV_SELECTION_FINAL.json',dict(candidates=candidates,selected=selected,DEV_admitted=bool(eligible)))
    write_json(root/'FROZEN_SELECTION.json',frozen)
    op=build(root,'fit','operational_diverse_K4',0.,0.,operational=True);evaluate(root,op)
    single=build(root,'fit','SUN_ranker_primary_K1',selected['sun_threshold'],selected['ms_floor'],primary_only=True);evaluate(root,single)
    panels=dict(SUN_diverse_K4=Path(selected['panel']),operational_diverse_K4=op,SUN_primary_K1=single)
    result={name:{role:summary(root,panel,role) for role in ('train','dev','final','all')} for name,panel in panels.items()}
    write_json(output/'POLICY_EVALUATION_FINAL.json',dict(status='complete',frozen_sha256=file_hash(root/'FROZEN_SELECTION.json'),
        results=result,old_FINAL_is_exploratory=True,DEV_admitted=bool(eligible)))
    print(json.dumps(dict(event='DEV_selection_frozen',admitted=bool(eligible),selected=selected)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--completion-dir',type=Path,required=True);args=parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('policy comparisons need an allocation')
    policies(args.root,args.completion_dir)
