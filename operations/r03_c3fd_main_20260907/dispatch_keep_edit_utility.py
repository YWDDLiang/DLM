"""Run the registered utility work through the shared deadline/resource guard."""
import argparse
import json
from pathlib import Path
from run_component import verify_deployed_source
from submit_stage import configured_dispatch


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--mode',choices=['train','evaluate','policies','audit','export','decide_repeat'],required=True)
    parser.add_argument('--repeat-index',type=int,choices=[1,2])
    parser.add_argument('--minutes',type=int,default=35)
    parser.add_argument('--job-suffix',default='')
    args=parser.parse_args();root=args.root.resolve();source=Path(__file__).resolve().parents[2]
    pipeline=json.loads((root/'RAW0_PIPELINE.json').read_text())
    pipeline.update(source_root=str(source),source_identity=verify_deployed_source(source))
    if args.job_suffix and not args.job_suffix.replace('_','').isalnum():raise ValueError('invalid job suffix')
    job='focus_utility_'+args.mode+('_'+args.job_suffix if args.job_suffix else '')
    output='training/utility' if args.mode=='train' else 'evaluation_features' if args.mode=='evaluate' else 'models/selected_utility' if args.mode=='export' else 'analysis/utility_'+args.mode
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
    elif args.mode=='evaluate':
        required += [root/'fit/proposal/inputs.jsonl',root/'training/utility/result/TRAINING_FINAL.json']
        products=['{output}/FEATURES_FINAL.json','{output}/UTILITY_PREDICTIONS.jsonl']
    elif args.mode=='policies':
        required += [root/'evaluation_features/FEATURES_FINAL.json',root/'evaluation_features/UTILITY_PREDICTIONS.jsonl',
            root/'fit/native/labeling/result/LABEL_FINAL.json',root/'fit/hybrid_proposal/labeling/result/LABEL_FINAL.json']
        products=['{output}/POLICY_EVALUATION_FINAL.json']
    elif args.mode=='audit':
        required += [root/'evaluation_features/FEATURES_FINAL.json',root/'evaluation_features/UTILITY_PREDICTIONS.jsonl']
        products=['{output}/AUDIT_FINAL.json']
    elif args.mode=='export':
        required += [root/'FROZEN_SELECTION.json',root/'training/utility/result/TRAINING_FINAL.json',
            root/'evaluation_features/FEATURES_FINAL.json']
        products=['{output}/EXPORT_FINAL.json','{output}/checkpoint/RSI_TRAINING_DONE.json']
    else:
        required += [root/'FROZEN_SELECTION.json',root/'REPEAT_SEED_REGISTRATION.json',
            root/'models/selected_utility/EXPORT_FINAL.json',
            root/f'repeats/repeat{args.repeat_index}/proposal/inputs.jsonl']
        products=['{output}/REPEAT_DECISION_FINAL.json']
    if any(not p.is_file() for p in required):raise ValueError('utility admission inputs are incomplete')
    gpu=0 if args.mode=='policies' else 1
    script='operations/r03_c3fd_main_20260907/evaluate_keep_edit_utility.py' if args.mode=='policies' else 'src/scripts/audit_keep_edit_judgement.py' if args.mode=='audit' else 'src/scripts/export_keep_edit_utility.py' if args.mode=='export' else 'src/scripts/train_keep_edit_utility.py'
    if args.mode=='decide_repeat':script='operations/r03_c3fd_main_20260907/repeat_keep_edit_utility.py'
    stage_args=['--root',str(root)] + ([] if args.mode in ('policies','audit','export','decide_repeat') else ['--mode',args.mode])
    if args.mode=='decide_repeat':stage_args+=['--mode','decide','--index',str(args.repeat_index)]
    if args.mode in ('policies','audit','export','decide_repeat'):stage_args+=['--completion-dir','{output}']
    pipeline['components']=[dict(id=job,output_dir=output,gpus=gpu,stages=[dict(name=args.mode,
        script=script,args=stage_args,
        inputs=[str(p) for p in required],outputs=products)])]
    pipeline['jobs']={job:dict(component_indices=[0],gpus_per_task=gpu,cpus_per_task=4 if gpu else 8,parallel_tasks=1,
        wall_minutes=args.minutes,memory='96G' if gpu else '64G',partition='gpu' if gpu else 'normal')}
    path=root/(job+'_PIPELINE.json')
    if path.exists() and json.loads(path.read_text())!=pipeline:raise ValueError('utility registration differs')
    if not path.exists():path.write_text(json.dumps(pipeline,sort_keys=True,indent=2)+'\n')
    configured_dispatch(['--config',str(path),'--job',job])


if __name__=='__main__':main()
