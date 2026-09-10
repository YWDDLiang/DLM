"""Run the registered utility work through the shared deadline/resource guard."""
import argparse
import json
from pathlib import Path
from run_component import verify_deployed_source
from submit_stage import configured_dispatch


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--mode',choices=['train','evaluate','policies','audit','export','export_audit','decide_repeat','train_readout','train_operational','infer_operational'],required=True)
    parser.add_argument('--repeat-index',type=int,choices=[1,2])
    parser.add_argument('--policy-method',choices=['utility','consensus','readout'],default='utility')
    parser.add_argument('--minutes',type=int,default=35)
    parser.add_argument('--job-suffix',default='')
    args=parser.parse_args();root=args.root.resolve();source=Path(__file__).resolve().parents[2]
    pipeline=json.loads((root/'RAW0_PIPELINE.json').read_text())
    pipeline.update(source_root=str(source),source_identity=verify_deployed_source(source))
    if args.job_suffix and not args.job_suffix.replace('_','').isalnum():raise ValueError('invalid job suffix')
    job='focus_utility_'+args.mode+('_'+args.job_suffix if args.job_suffix else '')
    output='training/utility' if args.mode=='train' else 'evaluation_features' if args.mode=='evaluate' else 'models/selected_utility' if args.mode=='export' else 'analysis/utility_'+args.mode
    if args.mode=='train_readout':output='training/readout_matched'
    if args.mode=='train_operational':output='training/operational_utility'
    if args.mode=='infer_operational':output='operational/model_inference'
    if args.mode=='export_audit':output='models/selected_utility/parity_audit'
    if args.policy_method!='utility':
        if args.mode!='policies':raise ValueError('policy method applies only to policy evaluation')
        job+='_'+args.policy_method;output+='_'+args.policy_method
    if args.mode=='decide_repeat':
        if args.repeat_index is None:raise ValueError('repeat decision needs an index')
        job+='_'+str(args.repeat_index);output=f'repeats/repeat{args.repeat_index}/decision'
    if args.job_suffix:
        if args.mode not in ('audit','policies'):raise ValueError('training/feature retries require explicit data recovery')
        output+='_'+args.job_suffix
    required=[root/'PREREGISTRATION.json',root/'SOURCE_SPLIT.jsonl',root/'fit/RUN_SPEC.json']
    if args.mode=='train':
        required += [root/'data/UTILITY_TRAIN.jsonl',root/'data/UTILITY_DATA_FINAL.json']
        products=['{output}/result/TRAINING_FINAL.json','{output}/result/_SUCCESS']
    elif args.mode in ('train_readout','train_operational'):
        required += [root/('OPERATIONAL_REGISTRATION.json' if args.mode=='train_operational' else 'READOUT_REGISTRATION.json'),root/'data/UTILITY_TRAIN.jsonl',
            root/'training/utility/result/TRAIN_FEATURES.pt',root/'evaluation_features/EVAL_FEATURES.pt']
        products=['{output}/result/TRAINING_FINAL.json','{output}/result/_SUCCESS',
            '{output}/evaluation/FEATURES_FINAL.json','{output}/evaluation/UTILITY_PREDICTIONS.jsonl']
    elif args.mode=='infer_operational':
        required += [root/'operational/MODEL_DEFINITION.json']
        products=['{output}/worker_0_DONE.json','{output}/worker_1_DONE.json']
    elif args.mode=='evaluate':
        required += [root/'fit/proposal/inputs.jsonl',root/'training/utility/result/TRAINING_FINAL.json']
        products=['{output}/FEATURES_FINAL.json','{output}/UTILITY_PREDICTIONS.jsonl']
    elif args.mode=='policies':
        feature_dir=root/('training/readout_matched/evaluation' if args.policy_method=='readout' else 'evaluation_features')
        required += [feature_dir/'FEATURES_FINAL.json',feature_dir/'UTILITY_PREDICTIONS.jsonl',
            root/'fit/native/labeling/result/LABEL_FINAL.json',root/'fit/hybrid_proposal/labeling/result/LABEL_FINAL.json']
        products=['{output}/POLICY_EVALUATION_FINAL.json']
        if args.policy_method=='consensus':required+=[root/'CONSENSUS_REGISTRATION.json']
        if args.policy_method=='readout':required+=[root/'READOUT_REGISTRATION.json']
    elif args.mode=='audit':
        required += [root/'evaluation_features/FEATURES_FINAL.json',root/'evaluation_features/UTILITY_PREDICTIONS.jsonl']
        products=['{output}/AUDIT_FINAL.json']
    elif args.mode=='export':
        required += [root/'FROZEN_SELECTION.json',root/'training/utility/result/TRAINING_FINAL.json',
            root/'evaluation_features/FEATURES_FINAL.json',root/'materialized_primary/MATERIALIZATION_FINAL.json']
        if (root/'FROZEN_SELECTION.json').exists():
            frozen=json.loads((root/'FROZEN_SELECTION.json').read_text())
            required += [Path(frozen[k]) for k in ('training_receipt_path','feature_receipt_path','predictions_path') if k in frozen]
        products=['{output}/EXPORT_FINAL.json','{output}/checkpoint/RSI_TRAINING_DONE.json']
    elif args.mode=='export_audit':
        required += [root/'FROZEN_SELECTION.json',root/'models/selected_utility/checkpoint/RSI_TRAINING_DONE.json']
        products=['{output}/EXPORT_AUDIT_FINAL.json']
    else:
        required += [root/'FROZEN_SELECTION.json',root/'REPEAT_SEED_REGISTRATION.json',
            root/'models/selected_utility/EXPORT_FINAL.json',
            root/f'repeats/repeat{args.repeat_index}/proposal/inputs.jsonl',
            root/f'repeats/repeat{args.repeat_index}/materialized_proposal/MATERIALIZATION_FINAL.json']
        products=['{output}/REPEAT_DECISION_FINAL.json']
    if any(not p.is_file() for p in required):raise ValueError('utility admission inputs are incomplete')
    gpu=0 if args.mode=='policies' else 2 if args.mode=='infer_operational' else 1
    script='operations/r03_c3fd_main_20260907/evaluate_keep_edit_utility.py' if args.mode=='policies' else 'src/scripts/audit_keep_edit_judgement.py' if args.mode=='audit' else 'src/scripts/export_keep_edit_utility.py' if args.mode=='export' else 'src/scripts/train_keep_edit_utility.py'
    if args.mode=='decide_repeat':script='operations/r03_c3fd_main_20260907/repeat_keep_edit_utility.py'
    if args.mode in ('train_readout','train_operational'):script='src/scripts/train_keep_edit_readout.py'
    if args.mode=='export_audit':script='src/scripts/audit_exported_utility.py'
    if args.mode=='infer_operational':script='src/scripts/run_operational_utility.py'
    stage_args=['--root',str(root)] + ([] if args.mode in ('policies','audit','export','export_audit','decide_repeat','train_readout','train_operational','infer_operational') else ['--mode',args.mode])
    if args.mode=='infer_operational':stage_args+=['--mode','infer']
    if args.mode=='train_operational':stage_args+=['--operational']
    if args.mode=='policies':stage_args+=['--method',args.policy_method]
    if args.mode=='decide_repeat':stage_args+=['--mode','decide','--index',str(args.repeat_index)]
    if args.mode in ('policies','audit','export','export_audit','decide_repeat','infer_operational'):stage_args+=['--completion-dir','{output}']
    pipeline['components']=[dict(id=job,output_dir=output,gpus=gpu,stages=[dict(name=args.mode,
        script=script,args=stage_args,
        inputs=[str(p) for p in required],outputs=products)])]
    if args.mode=='infer_operational':pipeline['components'][0]['stages'][0]['distributed_processes']=2
    pipeline['jobs']={job:dict(component_indices=[0],gpus_per_task=gpu,cpus_per_task=4*gpu if gpu else 8,parallel_tasks=1,
        wall_minutes=args.minutes,memory=f'{96*gpu}G' if gpu else '64G',partition='gpu' if gpu else 'normal')}
    path=root/(job+'_PIPELINE.json')
    if path.exists() and json.loads(path.read_text())!=pipeline:raise ValueError('utility registration differs')
    if not path.exists():path.write_text(json.dumps(pipeline,sort_keys=True,indent=2)+'\n')
    configured_dispatch(['--config',str(path),'--job',job])


if __name__=='__main__':main()
