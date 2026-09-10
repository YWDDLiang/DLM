"""Materialize frozen fresh policies and bind final source-isolated results."""
from __future__ import annotations
import argparse
import copy
import datetime as dt
import json
from pathlib import Path
import shutil
import subprocess
import sys

SOURCE=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(SOURCE/'src'),str(SOURCE/'operations/r03_c3fd_main_20260907')]
from scripts.run_post_refine_cycle import read_rows,write_json,file_hash
from scripts.run_rsi_stages import scores,score_directory
from scripts.run_sun_rank_scope import STREAMS
from crystal_dlm.post_refine_contract import fingerprint
from editor_trial_analysis import flags
from evaluate_sun_ranker import build,summary


def fresh_panels(root):
    frozen=json.loads((root/'FROZEN_SELECTION.json').read_text());selected=frozen['selected']
    inference=json.loads((root/'fresh/ranker_inference/INFERENCE_FINAL.json').read_text())
    if inference['model_sha256']!=frozen['model_sha256'] or file_hash(root/'fresh/ranker_inference/FRESH_PREDICTIONS.jsonl')!=inference['predictions_sha256']:
        raise ValueError('fresh inference is not the frozen model')
    panels={}
    for name,op in [('SUN_diverse_K4',False),('operational_diverse_K4',True)]:
        panel=build(root,'fresh',name,selected['sun_threshold'],selected['ms_floor'],operational=op)
        panels[name]=dict(panel=str(panel),inputs_sha256=file_hash(panel/'edited/inputs.jsonl'),
            decision_sha256=file_hash(panel/'DECISION_BINDING.json'))
    if (root/'ABSOLUTE_STATE_REGISTRATION.json').exists():
        nested=json.loads((root/'NESTED_FROZEN_SELECTION.json').read_text());selection=json.loads((root/'FINAL_MODEL_SELECTION.json').read_text())
        infer=json.loads((root/'fresh/nested_ranker_inference/INFERENCE_FINAL.json').read_text())
        predictions=root/'fresh/nested_ranker_inference/FRESH_PREDICTIONS.jsonl'
        if infer['model_sha256']!=nested['model_sha256'] or file_hash(predictions)!=infer['predictions_sha256']:
            raise ValueError('nested fresh inference changed')
        panel=build(root,'fresh','nested_SUN_diverse_K4',nested['selected']['sun_threshold'],0.,
            prediction_path=predictions,score_kind='absolute_NS_NMS_probabilities')
        panels['nested_SUN_diverse_K4']=dict(panel=str(panel),inputs_sha256=file_hash(panel/'edited/inputs.jsonl'),
            decision_sha256=file_hash(panel/'DECISION_BINDING.json'))
    write_json(root/'fresh/PANELS_FROZEN.json',dict(panels=panels,frozen_selection_sha256=file_hash(root/'FROZEN_SELECTION.json'),
        before_candidate_endpoint_evaluation=True,secondary_comparator_fixed_before_new_final_results=True))
    print(json.dumps(panels),flush=True)


def paired_ci(before,after):
    import numpy as np
    rng=np.random.default_rng(20260910);indices=rng.integers(0,len(before),size=(10000,len(before)))
    result={}
    for metric in ('Stable','SUN','MSUN'):
        delta=np.array([int(flags(b)[metric])-int(flags(a)[metric]) for a,b in zip(before,after,strict=True)])
        interval=np.quantile(delta[indices].mean(axis=1)*100,[.025,.975]).tolist()
        result[metric]=dict(net_count=int(delta.sum()),gain_rows=int((delta>0).sum()),loss_rows=int((delta<0).sum()),
            delta_percentage_points=float(delta.mean()*100),source_bootstrap_95_percent_interval_pp=interval)
    return dict(method='paired_bootstrap_by_unique_composition_source',draws=10000,seed=20260910,metrics=result,
        one_fixed_candidate_random_stream_set=True,does_not_measure_repeat_seed_variance=True)


def finalize(root):
    from pymatgen.core import Composition
    audit_path=root/'models/sun_ranker_runtime_audit/AUDIT_FINAL.json';audit=json.loads(audit_path.read_text())
    frozen_path=root/'FROZEN_SELECTION.json';frozen=json.loads(frozen_path.read_text())
    training_path=root/'training/sun_ranker/TRAINING_FINAL.json';training=json.loads(training_path.read_text())
    if file_hash(training_path)!=frozen['training_receipt_sha256'] or training['model_sha256']!=frozen['model_sha256']:
        raise ValueError('frozen training binding changed')
    if audit['status']!='complete' or audit['exported_ranker_sha256']!=frozen['model_sha256']:
        raise ValueError('exported full-model parity is not complete')
    for value in audit['results'].values():
        if value['decision_mismatches'] or value['max_prediction_abs_difference']>1e-5:
            raise ValueError('runtime decisions changed')
    definition_path=root/'fresh/PANELS_FROZEN.json';panels=json.loads(definition_path.read_text())['panels']
    before=scores(root/'fresh','native');newplans=read_rows(root/'fresh/cohort/plans.jsonl')
    oldplans=read_rows(root/'fit/cohort/plans.jsonl')
    canonical=lambda p:Composition(p['plan_state']['reduced_formula']).reduced_formula
    new_comp={canonical(p) for p in newplans};old_comp={canonical(p) for p in oldplans}
    if len(new_comp)!=256 or new_comp&old_comp:raise ValueError('new FINAL is not composition excluded')
    if {p['ancestor_id'] for p in newplans}&{p['ancestor_id'] for p in oldplans}:raise ValueError('new FINAL ancestor overlap')
    outcome={};cases={};parity={};interval={};pins={}
    for name,value in panels.items():
        panel=Path(value['panel']);inputs=panel/'edited/inputs.jsonl';decision_path=panel/'DECISION_BINDING.json'
        if file_hash(inputs)!=value['inputs_sha256'] or file_hash(decision_path)!=value['decision_sha256']:
            raise ValueError('fresh evaluated panel changed after freezing')
        after=scores(panel,'edited');decisions=json.loads(decision_path.read_text())['decisions']
        actual=read_rows(inputs);native=read_rows(root/'fresh/native/inputs.jsonl');verified=[]
        for i,(record,decision) in enumerate(zip(actual,decisions,strict=True)):
            chosen=decision['chosen_stream'];original=native[i] if chosen is None else json.loads((root/f'fresh/bank/{chosen}/bound/{i:04d}.json').read_text())['record']
            if fingerprint(original)!=decision['source_record_sha256']:raise ValueError('selected source bytes changed')
            expected=copy.deepcopy(original);expected['trajectory_id']=record['trajectory_id']
            if expected!=record:raise ValueError('actual evaluated output differs from selected bound record')
            verified.append(dict(ordinal=i,KEEP=chosen is None,record_sha256=fingerprint(record)))
        parity[name]=dict(requests=len(verified),all_outputs_exact=True,all_KEEP_originals_exact=True,records=verified)
        outcome[name]=summary(root,panel,'fresh');interval[name]=paired_ci(before,after)
        cases[name]=[dict(ordinal=i,composition=canonical(newplans[i]),ancestor_id=newplans[i]['ancestor_id'],
            before=flags(a),after=flags(b),decision=decisions[i]) for i,(a,b) in enumerate(zip(before,after,strict=True))
            if any(flags(a)[k]!=flags(b)[k] for k in ('Stable','SUN','MSUN'))]
        for p in [inputs,decision_path,panel/'edited/labeling/result/LABEL_FINAL.json',
            score_directory(panel,'edited')/'attempt_results.jsonl',score_directory(panel,'edited')/'BASIC_METRICS.json']:
            pins[str(p)]=file_hash(p)
    write_json(root/'models/MATERIALIZED_OUTPUT_PARITY.json',parity)
    primary='original';selected_frozen=frozen;selected_training=training;selected_training_path=training_path;selected_audit=audit_path
    if (root/'FINAL_MODEL_SELECTION.json').exists():
        primary=json.loads((root/'FINAL_MODEL_SELECTION.json').read_text())['primary']
        if primary=='nested':
            selected_frozen=json.loads((root/'NESTED_FROZEN_SELECTION.json').read_text())
            selected_training_path=root/'training/nested_sun_ranker/TRAINING_FINAL.json';selected_training=json.loads(selected_training_path.read_text())
            selected_audit=root/'models/nested_sun_ranker_runtime_audit/AUDIT_FINAL.json'
            nested_audit=json.loads(selected_audit.read_text())
            if (nested_audit['status']!='complete' or nested_audit['exported_ranker_sha256']!=selected_frozen['model_sha256']
                or file_hash(selected_training_path)!=selected_frozen['training_receipt_sha256']):
                raise ValueError('selected nested model has no verified runtime binding')
            if any(v['decision_mismatches'] or v['max_prediction_abs_difference']>1e-5 for v in nested_audit['results'].values()):
                raise ValueError('nested full-model audit failed')
    final=outcome['nested_SUN_diverse_K4' if primary=='nested' else 'SUN_diverse_K4'];delta=final['delta']
    passes=bool(primary!='KEEP' and selected_frozen['DEV_admitted'] and delta['SUN']>0 and delta['MSUN']>0 and delta['Stable']>=0)
    model_dir=root/'models/sun_ranker';model_dir.mkdir(parents=True,exist_ok=True)
    exported=model_dir/'SUN_RANKER.pt';shutil.copy2(selected_frozen['model_path'],exported)
    if file_hash(exported)!=selected_frozen['model_sha256']:raise ValueError('ranker export bytes changed')
    reg=json.loads((root/'PREREGISTRATION.json').read_text())
    model_definition=dict(schema='SUN_rank_scope_model_bundle_v1',editor_checkpoint=reg['old_editor'],
        editor_receipt_sha256=reg['old_editor_receipt_sha256'],ranker=str(exported),ranker_sha256=file_hash(exported),
        training_receipt_path=str(selected_training_path),training_receipt_sha256=file_hash(selected_training_path),
        source_commit=(SOURCE/'_CODE_READY').read_text().strip(),
        head_variant=primary,feature_schema=selected_training['feature_schema'],output_targets=selected_training['target_definition'],
        scope_candidates=reg['streams'],current_quality_features='only already measured native current; no candidate physical outcomes',
        decision=selected_frozen['selected'],known_native_SUN_guard=True,max_DLM_calls=24,
        composition_and_unmodified_continuous_fields_preserved=True,
        runtime_audit_sha256=file_hash(selected_audit),materialized_parity_sha256=file_hash(root/'models/MATERIALIZED_OUTPUT_PARITY.json'),
        validated_adoption=passes,adoption='candidate_passes_point_estimate_gate' if passes else 'retain_current_policy; candidate_not_admitted')
    write_json(model_dir/'MODEL_DEFINITION.json',model_definition)
    result=dict(schema='sun_rank_scope_fresh_comparison_v1',status='complete',created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        reports=outcome,paired_uncertainty=interval,cases=cases,DEV_selected_primary_variant=primary,
        passes_predeclared_DEV_and_fresh_joint_improvement=passes,
        fresh_compositions=256,fresh_sources=256,overlap_with_all_prior_1000_compositions=0,
        original_MP20_source_split='train',editor_final_role='new_source_and_composition_isolated_final',
        generator_domain='cached_original_B0_F800; differs_from_G1_F1000_fit',
        selected_before_new_final=True,one_fixed_candidate_stream_set=True,
        no_G_F_training_or_sampling=True,all_policies_reported_without_posthoc_new_FINAL_selection=True,
        training_receipt_sha256=file_hash(training_path),frozen_selection_sha256=file_hash(frozen_path),
        model_definition_path=str(model_dir/'MODEL_DEFINITION.json'),model_definition_sha256=file_hash(model_dir/'MODEL_DEFINITION.json'),
        full_model_runtime_audit_sha256=file_hash(audit_path),endpoint_and_score_pins=pins)
    write_json(root/'FRESH_COMPARISON_FINAL.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('cases','endpoint_and_score_pins')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--mode',choices=['fresh_panels','finalize'],required=True);a=p.parse_args()
    (fresh_panels if a.mode=='fresh_panels' else finalize)(a.root)
