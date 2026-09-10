"""Build learned continuous-output panels, recompute N/U, and seal DEV choice."""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import datetime as dt
import json
from pathlib import Path
import shutil
import subprocess
import sys

SOURCE=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(SOURCE/'src'))
from scripts.run_post_refine_cycle import read_rows,write_rows,write_json,file_hash,load_config
from scripts.run_rsi_stages import materialize,rebind_labels,scores,score_directory
from editor_trial_analysis import clone_bound_stage,flags
from crystal_dlm.post_refine_contract import fingerprint


def light_clone(source,destination):
    for relative in ['inputs.jsonl','FEEDBACK_MANIFEST.json','labeling/result/labels.jsonl',
                     'labeling/result/LABEL_FINAL.json','labeling/result/_SUCCESS']:
        target=destination/relative;target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():shutil.copy2(source/relative,target)
    clone_bound_stage(source,destination)


def build_panel(root,epoch,threshold,*,old=False,recorded=False,consensus=False):
    fit=root/'fit';tag=('old_E3_recorded_continuous' if recorded else 'old_E3_canonical_continuous') if old else f'{"consensus" if consensus else "utility"}_e{epoch}_m{round(threshold*100):02d}'
    panel=root/'policies'/tag;config=panel/'RUN_SPEC.json'
    if config.exists():
        if not (panel/'edited/labeling/result/_SUCCESS').exists():raise ValueError('unfinished policy requires inspection')
        return config
    predictions={r['ordinal']:r for r in read_rows(root/'evaluation_features/UTILITY_PREDICTIONS.jsonl')}
    native=read_rows(fit/'native/inputs.jsonl');hybrid=read_rows(fit/'hybrid_proposal/inputs.jsonl')
    spec=json.loads((fit/'RUN_SPEC.json').read_text());spec.update(run_root=str(panel),run_id='keep_edit_focus:'+tag)
    spec['assets']['official_cache']=json.loads((root/'REFERENCE_CACHE_OVERRIDE.json').read_text())['directory']
    policy=dict(kind=('old_E3_recorded_probability' if recorded else 'old_E3_canonical_probability') if old else 'quality_only_signed_utility',epoch=epoch,raw_margin=threshold,
        known_SUN_guard='same_original_E3_guard',physics_used_for_acceptance=False,
        predictions_sha256=file_hash(root/'evaluation_features/UTILITY_PREDICTIONS.jsonl'))
    if consensus:
        policy.update(kind='original_accept_AND_learned_signed_utility',reference_acceptance_required=True,
            reference_raw_margin=0.,registration_sha256=file_hash(root/'CONSENSUS_REGISTRATION.json'))
    spec['focus_decision_policy']=policy
    panel.mkdir(parents=True,exist_ok=True);shutil.copytree(fit/'cohort',panel/'cohort')
    light_clone(fit/'native',panel/'current');light_clone(fit/'hybrid_proposal',panel/'proposal')
    write_json(config,spec);decisions=[]
    for a,b in zip(native,hybrid,strict=True):
        i=a['original_ordinal'];prediction=predictions.get(i)
        trace=json.loads((fit/f'proposal/records/{i:04d}.json').read_text())['editor_trace']
        commit=json.loads((fit/f'hybrid_proposal/records/{i:04d}.json').read_text())['commit_trace']
        guarded=trace.get('known_sun') and trace.get('learned_mode')==0
        chosen=bool(prediction and prediction['proposal_generated'] and not guarded and
            ((prediction['old_applied'] if recorded else prediction['original_quality_logit']>=0.) if old
             else prediction['learned_utilities'][str(epoch)]>=threshold) and
            (not consensus or prediction['original_quality_logit']>=0.))
        actual=chosen and commit['applied'];source=b if actual else a
        record=dict(source,trajectory_id=f'keep_edit_focus:{tag}:edited:{i}')
        write_json(panel/f'edited/records/{i:04d}.json',dict(record=record,learned_accept=chosen,
            actual_edit=actual,commit_trace=commit,decision_policy=policy))
        decisions.append(dict(ordinal=i,learned_accept=chosen,actual_edit=actual,
            original_E3_known_SUN_guard=bool(guarded),source_trajectory=source['trajectory_id'],
            raw_utility=None if old or prediction is None else prediction['learned_utilities'][str(epoch)]))
    write_json(panel/'DECISION_BINDING.json',dict(policy=policy,decisions=decisions,
        native_inputs_sha256=file_hash(fit/'native/inputs.jsonl'),hybrid_inputs_sha256=file_hash(fit/'hybrid_proposal/inputs.jsonl'),
        no_physical_candidate_selection=True))
    materialize(spec,'edited');rebind_labels(spec,'edited')
    return config


def evaluate(root,config):
    spec=load_config(config);panel=Path(spec['run_root']);output=panel/'edited/scoring/result'
    if (output/'_SUCCESS').exists():return
    # The common scorer owns creation of the result directory and rejects
    # pre-existing output, including an empty directory.
    output.parent.mkdir(parents=True,exist_ok=True)
    command=[sys.executable,str(SOURCE/'scripts/evaluate_programmed_paths.py'),
        '--paths-jsonl',str(panel/'edited/inputs.jsonl'),'--labels-jsonl',str(panel/'edited/labeling/result/labels.jsonl'),
        '--frozen-config',spec['assets']['frozen_config'],'--official-cache',spec['assets']['official_cache'],
        '--output-dir',str(output),'--expected-requests','1000','--endpoint','native','--cohort-role','training_feedback',
        '--policy-stage','round0_diagnostic','--sun-only','--nu-workers','7','--nu-cache',str(root/'nu_cache'),
        '--feedback-manifest',str(panel/'edited/FEEDBACK_MANIFEST.json'),'--joint-physical-stop']
    write_json(panel/'SCORE_COMMAND.json',dict(command=command))
    with (panel/'SCORE.stdout').open('x') as out,(panel/'SCORE.stderr').open('x') as err:
        result=subprocess.run(command,stdout=out,stderr=err)
    if result.returncode:raise RuntimeError('continuous policy scoring failed:'+str(panel))


def summary(root,panel,role,*,after_stage='edited'):
    if role=='final' and not (root/'FROZEN_SELECTION.json').exists():raise ValueError('FINAL policy is not frozen')
    roles=read_rows(root/'SOURCE_SPLIT.jsonl');before=scores(root/'fit','native');after=scores(panel,after_stage)
    decisions=json.loads((panel/'DECISION_BINDING.json').read_text())['decisions']
    indices=[i for i,r in enumerate(roles) if role=='all' or r['split']==role]
    if role=='all' and not (root/'FROZEN_SELECTION.json').exists():raise ValueError('all includes sealed FINAL')
    counts={k:0 for k in ['Stable','MS','SUN','MSUN','known_failure','unknown','actual_edits']}
    gains={k:[] for k in ['Stable','MS','SUN','MSUN']};losses=copy.deepcopy(gains);base={k:0 for k in gains}
    for i in indices:
        a,b=flags(before[i]),flags(after[i])
        for key in gains:
            counts[key]+=int(b[key]);base[key]+=int(a[key])
            if b[key] and not a[key]:gains[key].append(i)
            if a[key] and not b[key]:losses[key].append(i)
        counts['known_failure']+=int(b['known_failure']);counts['unknown']+=int(not b['reliable'] and not b['known_failure'])
        counts['actual_edits']+=int(decisions[i]['actual_edit'])
    return dict(panel=str(panel),split=role,requests=len(indices),counts=counts,KEEP=base,gains=gains,losses=losses,
        delta={k:counts[k]-base[k] for k in gains},decision_sha256=file_hash(panel/'DECISION_BINDING.json'),
        scores_sha256=file_hash(score_directory(panel,after_stage)/'attempt_results.jsonl'))


def register_consensus(root):
    root=Path(root);prior=root/'UTILITY_DEV_SELECTION_FINAL.json';path=root/'CONSENSUS_REGISTRATION.json'
    if path.exists():raise ValueError('consensus method already registered')
    if (root/'FROZEN_SELECTION.json').exists():raise ValueError('FINAL was already opened for a frozen policy')
    report=json.loads(prior.read_text())
    if report['admitted_to_FINAL'] or report['final_quality_consulted']:
        raise ValueError('consensus requires a failed DEV stage with sealed FINAL')
    write_json(path,dict(schema='original_accept_AND_signed_utility_v1',
        created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        plan_sha256=file_hash(SOURCE/'docs/r03_paper_story_20260907/KEEP_EDIT_ACCEPTANCE_CONSENSUS_PLAN_20260910.md'),
        preceding_failed_DEV_sha256=file_hash(prior),source_split_sha256=file_hash(root/'SOURCE_SPLIT.jsonl'),
        training_receipt_sha256=file_hash(root/'training/utility/result/TRAINING_FINAL.json'),
        predictions_sha256=file_hash(root/'evaluation_features/UTILITY_PREDICTIONS.jsonl'),
        fixed_reference_logit_threshold=0.,additional_optimizer_steps=0,final_quality_consulted=False,
        selection_scope='same_16_snapshots_and_margins_DEV_only',physics_used_for_acceptance=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--completion-dir',type=Path)
    parser.add_argument('--method',choices=['utility','consensus'],default='utility')
    args=parser.parse_args();root=args.root;completion=args.completion_dir or root/'analysis/utility_policies'
    import os
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('policy evaluation requires its CPU allocation')
    consensus=args.method=='consensus';prefix='CONSENSUS' if consensus else 'UTILITY'
    if consensus:
        admission=json.loads((root/'CONSENSUS_REGISTRATION.json').read_text())
        if admission['predictions_sha256']!=file_hash(root/'evaluation_features/UTILITY_PREDICTIONS.jsonl'):
            raise ValueError('registered consensus predictions changed')
    reg=json.loads((root/'PREREGISTRATION.json').read_text());receipt=json.loads((root/'evaluation_features/FEATURES_FINAL.json').read_text())
    if receipt['predictions_sha256']!=file_hash(root/'evaluation_features/UTILITY_PREDICTIONS.jsonl'):
        raise ValueError('learned predictions changed')
    rule_path=root/('CONSENSUS_SELECTION_RULE.json' if consensus else 'DECISION_SELECTION_RULE.json')
    if not rule_path.exists():
        write_json(rule_path,dict(created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
            committed_plan_sha256=file_hash(SOURCE/'docs/r03_paper_story_20260907'/
                ('KEEP_EDIT_ACCEPTANCE_CONSENSUS_PLAN_20260910.md' if consensus else 'KEEP_EDIT_FOCUS_PLAN_20260910.md')),
            snapshots=reg['head_training']['snapshots'],thresholds=reg['head_training']['thresholds'],
            rule='both_gains_then_SUN_MSUN_Stable_fewer_Stable_losses_fewer_edits_higher_margin_earlier_epoch',
            acceptance='original_logit_ge_zero_AND_raw_utility_ge_margin' if consensus else 'raw_quality_output_greater_than_or_equal_to_margin',
            FINAL_admission='DEV_SUN_and_MSUN_both_above_continuous_KEEP',
            before_first_matched_proposal_DEV_policy_score=True))
    candidates=[]
    for epoch in reg['head_training']['snapshots']:
        for margin in reg['head_training']['thresholds']:
            config=build_panel(root,epoch,margin,consensus=consensus);evaluate(root,config);dev=summary(root,config.parent,'dev')
            entry=dict(epoch=epoch,margin=margin,dev=dev);candidates.append(entry)
            write_json(config.parent/'DEV_RESULT.json',dev)
            print(json.dumps(dict(epoch=epoch,margin=margin,dev=dev['counts'],delta=dev['delta'])),flush=True)
    def key(candidate):
        d=candidate['dev'];c=d['counts']
        return (d['delta']['SUN']>0 and d['delta']['MSUN']>0,c['SUN'],c['MSUN'],c['Stable'],
            -len(d['losses']['Stable']),-c['actual_edits'],candidate['margin'],-candidate['epoch'])
    selected=max(candidates,key=key)
    control=build_panel(root,0,.5,old=True);evaluate(root,control)
    original_control=build_panel(root,0,.5,old=True,recorded=True);evaluate(root,original_control)
    dev_controls=dict(old_E3_canonical=summary(root,control.parent,'dev'),
        old_E3_recorded=summary(root,original_control.parent,'dev'))
    admitted=bool(key(selected)[0])
    dev_path=root/(prefix+'_DEV_SELECTION_FINAL.json')
    write_json(dev_path,dict(method=args.method,selected=selected,candidates=candidates,
        admitted_to_FINAL=admitted,final_quality_consulted=False,selection_rule_sha256=file_hash(rule_path),controls=dev_controls))
    if not admitted:
        write_json(completion/'POLICY_EVALUATION_FINAL.json',dict(status='DEV_failed_FINAL_sealed',
            dev_selection_sha256=file_hash(dev_path)))
        print(json.dumps(dict(event='DEV_FAILED_FINAL_REMAINS_SEALED',selected=selected)),flush=True)
        return
    frozen=dict(schema='keep_edit_utility_DEV_selection_v1',method=args.method,created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        selected=selected,candidates=candidates,selection_split='dev',final_quality_consulted=False,
        training_receipt_sha256=file_hash(root/'training/utility/result/TRAINING_FINAL.json'),
        feature_receipt_sha256=file_hash(root/'evaluation_features/FEATURES_FINAL.json'),
        selection_rule_sha256=file_hash(rule_path),
        selection_rule='both_gains_then_SUN_MSUN_Stable_fewer_Stable_losses_fewer_edits_higher_margin_earlier_epoch')
    if (root/'FROZEN_SELECTION.json').exists():raise ValueError('selection is already frozen')
    write_json(root/'FROZEN_SELECTION.json',frozen)
    panel=Path(selected['dev']['panel'])
    reports={role:{'learned':summary(root,panel,role),'old_E3':summary(root,control.parent,role),
        'old_E3_original_execution':summary(root,original_control.parent,role)} for role in ['train','dev','final','all']}
    final=reports['final']['learned'];passed=final['delta']['SUN']>0 and final['delta']['MSUN']>0
    report=dict(schema='continuous_keep_edit_utility_comparison_v1',method=args.method,selected=dict(epoch=selected['epoch'],margin=selected['margin'],panel=str(panel)),
        frozen_selection_sha256=file_hash(root/'FROZEN_SELECTION.json'),reports=reports,
        primary_unseen_FINAL_SUN_and_MSUN_gain=passed,actual_training_performed=True,
        old_editor_source_exclusion='old256_and_same_formula_all_TRAIN',original_data_domain='MP20_TRAIN',
        E_unseen_does_not_mean_G_F_B0_unseen=True)
    final_path=root/(prefix+'_COMPARISON_FINAL.json');write_json(final_path,report)
    write_json(completion/'POLICY_EVALUATION_FINAL.json',dict(status='complete',
        dev_selection_sha256=file_hash(dev_path),
        final_report_path=str(final_path),final_report_sha256=file_hash(final_path)))
    print(json.dumps(dict(event='FINAL_AFTER_FROZEN_SELECTION',primary_pass=passed,
        learned=final['counts'],KEEP=final['KEEP'],delta=final['delta'],old_E3=reports['final']['old_E3']['counts'])),flush=True)


if __name__=='__main__':main()
