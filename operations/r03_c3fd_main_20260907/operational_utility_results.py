"""Bind exact cached endpoints and report every preregistered repair-pool policy."""
import argparse
import json
from pathlib import Path
import sys

SOURCE=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(SOURCE/'src'))
from scripts.run_post_refine_cycle import read_rows,write_json,file_hash,load_config
from scripts.run_rsi_stages import rebind_labels,score_directory,scores
from evaluate_keep_edit_utility import light_clone
from audit_keep_edit_final import compare
from editor_trial_analysis import flags


def rebind(root,name):
    panel=root/'operational/panels'/name
    if name=='primary_K1':parents=[root/'fit/native',root/'fit/hybrid_proposal']
    elif name=='all_K4':parents=[root/'operational/panels/pool_A_K2/edited',root/'operational/panels/pool_B_K2/edited']
    else:raise ValueError('this pool contains endpoints needing actual new labels')
    if (panel/'edited/labeling/result/_SUCCESS').exists():raise ValueError('pool labels already bound')
    for source,stage in zip(parents,('current','proposal')):light_clone(source,panel/stage)
    rebind_labels(load_config(panel/'RUN_SPEC.json'),'edited')
    print(json.dumps(dict(panel=name,label_receipt_sha256=file_hash(panel/'edited/labeling/result/LABEL_FINAL.json'),
        new_physical_evaluations=0)),flush=True)


def report(root):
    model_path=root/'operational/MODEL_DEFINITION.json';model=json.loads(model_path.read_text())
    roles=read_rows(root/'SOURCE_SPLIT.jsonl')
    admission=json.loads((root/'analysis/FINAL_COMPOSITION_ADMISSION.json').read_text())
    before=scores(root/'fit','native');reports={};pending=[]
    subsets={role:[i for i,r in enumerate(roles) if role=='all' or r['split']==role] for role in ('train','dev','final','all')}
    subsets['strict_composition_final']=admission['strict_composition_isolated_FINAL']
    for name in model['pools']:
        panel=root/'operational/panels'/name
        if not (score_directory(panel,'edited')/'_SUCCESS').exists():pending.append(name);continue
        decisions=json.loads((panel/'DECISION_BINDING.json').read_text())['decisions'];after=scores(panel,'edited')
        values={role:compare(root,panel,'edited',indices) for role,indices in subsets.items()}
        causes={};cases=[]
        for i in subsets['strict_composition_final']:
            a,b=flags(before[i]),flags(after[i])
            if not any(a[k]!=b[k] for k in ('Stable','MS','SUN','MSUN')):continue
            cases.append(dict(ordinal=i,before=a,after=b,decision=decisions[i]))
        values['strict_cases']=cases
        values['max_DLM_calls']=max(x['dlm_calls'] for x in decisions)
        values['known_SUN_guarded']=sum(x['known_continuous_SUN_guard'] for x in decisions)
        values['before_not_SUN_with_candidate']=sum(not x['known_continuous_SUN_guard'] and x['valid_candidate_count']>0 for x in decisions)
        values['no_candidate_indices']=[x['ordinal'] for x in decisions if not x['known_continuous_SUN_guard'] and x['valid_candidate_count']==0]
        values['model_definition_sha256']=file_hash(model_path);reports[name]=values
    result=dict(schema='operational_utility_repair_pool_comparison_v1',reports=reports,pending=pending,
        model_definition_sha256=file_hash(model_path),original_frozen_model_unchanged=True,
        no_DEV_FINAL_training=True,evaluation_status='exploratory_reuse_of_previously_examined_sources',
        fresh_random_stream='keep_edit_operational_fresh1',
        formal_original_trial_is_reported_separately=True,all_policies_reported_without_posthoc_selection=True)
    path=root/'operational'/('COMPARISON_FINAL.json' if not pending else 'COMPARISON_PROGRESS.json')
    write_json(path,result)
    print(json.dumps(dict(pending=pending,reports={name:{role:{k:v[k] for k in ('requests','KEEP','counts','delta')}
        for role,v in values.items() if role in subsets} for name,values in reports.items()})),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--mode',choices=['rebind','report'],required=True);parser.add_argument('--panel')
    args=parser.parse_args()
    if args.mode=='report':report(args.root)
    elif args.panel:rebind(args.root,args.panel)
    else:parser.error('rebind requires a panel name')
