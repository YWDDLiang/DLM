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
    parser.add_argument('--mode',choices=['collect'],default='collect')
    parser.add_argument('--minutes',type=int,default=30)
    args=parser.parse_args();root=args.root.resolve();source=Path(__file__).resolve().parents[2]
    pipeline=json.loads((root/'RAW0_PIPELINE.json').read_text())
    pipeline.update(source_root=str(source),source_identity=verify_deployed_source(source))
    job='sun_rank_'+args.mode+'_'+args.panel;gpus=2
    required=[root/'PREREGISTRATION.json',root/'SOURCE_SPLIT.jsonl',root/args.panel/'RUN_SPEC.json',
        root/args.panel/'native/inputs.jsonl',root/args.panel/'current/inputs.jsonl']
    if args.panel=='fresh':required+=[root/'FROZEN_SELECTION.json']
    if any(not p.exists() for p in required):raise ValueError('SUN pilot inputs are incomplete')
    stage=dict(name='collect',script='src/scripts/run_sun_rank_scope.py',
        args=['--root',str(root),'--mode','collect','--panel',args.panel,'--completion-dir','{output}'],
        inputs=[str(p) for p in required],outputs=['{output}/worker_0_DONE.json','{output}/worker_1_DONE.json'],
        distributed_processes=2)
    pipeline['components']=[dict(id=job,output_dir=args.panel+'/collection',gpus=gpus,stages=[stage])]
    pipeline['jobs']={job:dict(component_indices=[0],gpus_per_task=gpus,cpus_per_task=8,
        parallel_tasks=1,wall_minutes=args.minutes,memory='192G',partition='gpu')}
    path=root/(job+'_PIPELINE.json')
    if path.exists() and json.loads(path.read_text())!=pipeline:raise ValueError('SUN pilot job differs')
    if not path.exists():path.write_text(json.dumps(pipeline,sort_keys=True,indent=2)+'\n')
    configured_dispatch(['--config',str(path),'--job',job])


if __name__=='__main__':main()
