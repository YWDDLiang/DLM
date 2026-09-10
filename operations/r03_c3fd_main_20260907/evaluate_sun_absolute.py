"""Register and select the candidate-state comparison before fresh outcomes."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import sys

SOURCE=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(SOURCE/'src'))
from scripts.run_post_refine_cycle import write_json,file_hash
from evaluate_sun_ranker import build,evaluate,summary


def require_fresh_unopened(root):
    if any((p/'edited/labeling/result/LABEL_FINAL.json').exists() for p in (root/'policies/fresh').glob('*')):
        raise ValueError('new candidate endpoint results were already opened')


def register(root):
    require_fresh_unopened(root);path=root/'ABSOLUTE_STATE_REGISTRATION.json'
    if path.exists():raise ValueError('absolute-state comparison already registered')
    training=root/'training/sun_ranker/TRAINING_FINAL.json'
    source=json.loads(training.read_text())
    write_json(path,dict(schema='sun_absolute_state_pretest_supplement_v1',created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        plan_sha256=file_hash(SOURCE/'docs/r03_paper_story_20260907/SUN_ABSOLUTE_STATE_SUPPLEMENT_20260910.md'),
        base_training_sha256=file_hash(training),base_train_rows_sha256=source['train_data_sha256'],
        training_settings=source['settings'],sun_probability_thresholds=[.05,.1,.2,.3],
        per_candidate_MS_gain_veto=False,current_verified_SUN_guard=True,
        DEV_constraints=['SUN>KEEP','MSUN>=KEEP','Stable>=KEEP'],
        new_FINAL_candidate_results_consulted=False,original_delta_model_retained=True))
    print(json.dumps(dict(registration=str(path),sha256=file_hash(path))),flush=True)


def policies(root,output):
    require_fresh_unopened(root)
    reg=json.loads((root/'ABSOLUTE_STATE_REGISTRATION.json').read_text())
    training=root/'training/nested_sun_ranker/TRAINING_FINAL.json';receipt=json.loads(training.read_text())
    predictions=root/'training/nested_sun_ranker/FIT_PREDICTIONS.jsonl'
    if file_hash(predictions)!=receipt['predictions_sha256']:raise ValueError('nested model predictions changed')
    rows=[]
    for threshold in reg['sun_probability_thresholds']:
        panel=build(root,'fit',f'nested_sun{round(threshold*100):02d}',threshold,0.,prediction_path=predictions,
            score_kind='absolute_NS_NMS_probabilities')
        evaluate(root,panel);dev=summary(root,panel,'dev')
        rows.append(dict(sun_threshold=threshold,ms_floor=0.,panel=str(panel),DEV=dev))
        write_json(panel/'DEV_RESULT.json',dev)
        print(json.dumps(dict(threshold=threshold,DEV=dev['counts'],delta=dev['delta'],actual_edits=dev['actual_edits'])),flush=True)
    eligible=[r for r in rows if r['DEV']['delta']['SUN']>0 and r['DEV']['delta']['MSUN']>=0 and r['DEV']['delta']['Stable']>=0]
    selected=max(eligible,key=lambda r:(r['DEV']['counts']['SUN'],r['DEV']['counts']['MSUN'],-r['DEV']['actual_edits'],r['sun_threshold'])) if eligible else next(r for r in rows if r['sun_threshold']==.1)
    target=root/'NESTED_FROZEN_SELECTION.json'
    if target.exists():raise ValueError('nested policy is already frozen')
    frozen=dict(schema='nested_SUN_DEV_selection_v1',created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        selected=selected,candidates=rows,DEV_admitted=bool(eligible),head_kind='nested_absolute_states',
        model_path=receipt['model_path'],model_sha256=receipt['model_sha256'],
        training_receipt_sha256=file_hash(training),fresh_quality_consulted=False,
        registration_sha256=file_hash(root/'ABSOLUTE_STATE_REGISTRATION.json'))
    write_json(target,frozen)
    original=json.loads((root/'FROZEN_SELECTION.json').read_text())
    primary='nested' if eligible else 'original' if original['DEV_admitted'] else 'KEEP'
    write_json(root/'FINAL_MODEL_SELECTION.json',dict(primary=primary,created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        original_selection_sha256=file_hash(root/'FROZEN_SELECTION.json'),nested_selection_sha256=file_hash(target),
        new_FINAL_candidate_results_consulted=False,rule='nested_state_comparison_if_DEV_admitted_else_original_if_DEV_admitted_else_KEEP'))
    results={role:summary(root,Path(selected['panel']),role) for role in ('train','dev','final','all')}
    write_json(output/'POLICY_EVALUATION_FINAL.json',dict(status='complete',DEV_admitted=bool(eligible),
        selected=selected,results=results,new_FINAL_remains_unopened=True,old_FINAL_is_exploratory=True))
    print(json.dumps(dict(event='nested_policy_frozen',DEV_admitted=bool(eligible),selected=selected,primary=primary)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--register',action='store_true');p.add_argument('--completion-dir',type=Path)
    a=p.parse_args()
    if a.register:register(a.root)
    else:
        if not os.environ.get('SLURM_JOB_ID'):raise ValueError('absolute-state policy evaluation needs allocation')
        policies(a.root,a.completion_dir)
