"""Run the registered utility work through the shared deadline/resource guard."""
import argparse
import json
from pathlib import Path
from run_component import verify_deployed_source
from submit_stage import configured_dispatch


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--mode',choices=['train','evaluate'],required=True)
    parser.add_argument('--minutes',type=int,default=35)
    args=parser.parse_args();root=args.root.resolve();source=Path(__file__).resolve().parents[2]
    pipeline=json.loads((root/'RAW0_PIPELINE.json').read_text())
    pipeline.update(source_root=str(source),source_identity=verify_deployed_source(source))
    job='focus_utility_'+args.mode
    output='training/utility' if args.mode=='train' else 'evaluation_features'
    required=[root/'PREREGISTRATION.json',root/'SOURCE_SPLIT.jsonl',root/'fit/RUN_SPEC.json']
    if args.mode=='train':
        required += [root/'data/UTILITY_TRAIN.jsonl',root/'data/UTILITY_DATA_FINAL.json']
        products=['{output}/result/TRAINING_FINAL.json','{output}/result/_SUCCESS']
    else:
        required += [root/'fit/proposal/inputs.jsonl',root/'training/utility/result/TRAINING_FINAL.json']
        products=['{output}/FEATURES_FINAL.json','{output}/UTILITY_PREDICTIONS.jsonl']
    if any(not p.is_file() for p in required):raise ValueError('utility admission inputs are incomplete')
    pipeline['components']=[dict(id=job,output_dir=output,gpus=1,stages=[dict(name=args.mode,
        script='src/scripts/train_keep_edit_utility.py',args=['--root',str(root),'--mode',args.mode],
        inputs=[str(p) for p in required],outputs=products)])]
    pipeline['jobs']={job:dict(component_indices=[0],gpus_per_task=1,cpus_per_task=4,parallel_tasks=1,
        wall_minutes=args.minutes,memory='96G',partition='gpu')}
    path=root/(job+'_PIPELINE.json')
    if path.exists() and json.loads(path.read_text())!=pipeline:raise ValueError('utility registration differs')
    if not path.exists():path.write_text(json.dumps(pipeline,sort_keys=True,indent=2)+'\n')
    configured_dispatch(['--config',str(path),'--job',job])


if __name__=='__main__':main()
