"""Dispatch the new SUN pilot under its own bounded resource receipt."""
import argparse
import json
from pathlib import Path
from run_component import verify_deployed_source
from submit_stage import configured_dispatch


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--panel',choices=['fit','fresh'],default='fit')
    parser.add_argument('--mode',choices=['collect','train','infer','policies','audit'],default='collect')
    parser.add_argument('--minutes',type=int,default=30)
    parser.add_argument('--nested',action='store_true')
    args=parser.parse_args();root=args.root.resolve();source=Path(__file__).resolve().parents[2]
    pipeline=json.loads((root/'RAW0_PIPELINE.json').read_text())
    pipeline.update(source_root=str(source),source_identity=verify_deployed_source(source))
    job='sun_rank_'+args.mode+'_'+args.panel;gpus=2 if args.mode=='collect' else 0 if args.mode=='policies' else 1
    required=[root/'PREREGISTRATION.json',root/'SOURCE_SPLIT.jsonl',root/args.panel/'RUN_SPEC.json',
        root/args.panel/'native/inputs.jsonl',root/args.panel/'current/inputs.jsonl']
    if args.panel=='fresh':required+=[root/'FROZEN_SELECTION.json']
    if args.mode=='train':
        required += [root/f'fit/bank/{s}/COLLECTION_FINAL.json' for s in ('primary','rank1','rank2','rank3')]
        required += [root/f'fit/bank/{s}/candidate/scoring/result/_SUCCESS' for s in ('rank1','rank2','rank3')]
    if args.mode=='policies':required += [root/'training/sun_ranker/TRAINING_FINAL.json']
    if args.mode in ('infer','audit'):required += [root/'FROZEN_SELECTION.json',root/'fresh/native/labeling/result/LABEL_FINAL.json']
    if args.nested:
        if args.mode=='collect':raise ValueError('nested head shares the exact content candidates')
        job+='__nested';required += [root/'ABSOLUTE_STATE_REGISTRATION.json']
        if args.mode=='policies':required += [root/'training/nested_sun_ranker/TRAINING_FINAL.json']
        if args.mode in ('infer','audit'):required += [root/'NESTED_FROZEN_SELECTION.json']
    if any(not p.exists() for p in required):raise ValueError('SUN pilot inputs are incomplete')
    stage=dict(name='collect',script='src/scripts/run_sun_rank_scope.py',
        args=['--root',str(root),'--mode','collect','--panel',args.panel,'--completion-dir','{output}'],
        inputs=[str(p) for p in required],outputs=['{output}/worker_0_DONE.json','{output}/worker_1_DONE.json'],
        distributed_processes=2)
    output=args.panel+'/collection'
    if args.mode!='collect':
        output='training/sun_ranker' if args.mode=='train' else 'fresh/ranker_inference' if args.mode=='infer' else 'analysis/DEV_policies'
        script='src/scripts/train_sun_ranker.py' if args.mode in ('train','infer') else 'operations/r03_c3fd_main_20260907/evaluate_sun_ranker.py'
        required_outputs=['{output}/TRAINING_FINAL.json','{output}/_SUCCESS'] if args.mode=='train' else ['{output}/INFERENCE_FINAL.json'] if args.mode=='infer' else ['{output}/POLICY_EVALUATION_FINAL.json']
        extra=[] if args.mode=='policies' else ['--mode',args.mode]
        stage=dict(name=args.mode,script=script,
            args=['--root',str(root),'--completion-dir','{output}',*extra],
            inputs=[str(p) for p in required],outputs=required_outputs)
    if args.mode=='audit':
        output='models/sun_ranker_runtime_audit'
        stage=dict(name='audit',script='src/scripts/audit_sun_ranker.py',
            args=['--root',str(root),'--completion-dir','{output}'],inputs=[str(p) for p in required],
            outputs=['{output}/AUDIT_FINAL.json'])
    if args.nested:
        output={'train':'training/nested_sun_ranker','infer':'fresh/nested_ranker_inference',
            'policies':'analysis/nested_DEV_policies','audit':'models/nested_sun_ranker_runtime_audit'}[args.mode]
        if args.mode=='policies':stage['script']='operations/r03_c3fd_main_20260907/evaluate_sun_absolute.py'
        else:stage['args'].append('--nested')
    pipeline['components']=[dict(id=job,output_dir=output,gpus=gpus,stages=[stage])]
    pipeline['jobs']={job:dict(component_indices=[0],gpus_per_task=gpus,cpus_per_task=4*gpus if gpus else 8,
        parallel_tasks=1,wall_minutes=args.minutes,memory=f'{96*gpus}G' if gpus else '64G',partition='gpu' if gpus else 'normal')}
    path=root/(job+'_PIPELINE.json')
    if path.exists() and json.loads(path.read_text())!=pipeline:raise ValueError('SUN pilot job differs')
    if not path.exists():path.write_text(json.dumps(pipeline,sort_keys=True,indent=2)+'\n')
    configured_dispatch(['--config',str(path),'--job',job])


if __name__=='__main__':main()
