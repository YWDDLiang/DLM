#!/usr/bin/env python3
"""Register the next weight version against the exact original Plan bytes."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.run_post_refine_cycle import (file_hash, read_rows, write_json,
                                          validate_rsi_checkpoint)


def prepare_round(root, index):
    root=Path(root).resolve()
    if index not in (1,2,3): raise ValueError('only the three registered updates are allowed')
    destination=root/'rounds'/f'round{index}'
    checkpoints={branch:root/'training'/f'round{index}'/branch/'result/checkpoint' for branch in ('G','E')}
    receipts={branch:validate_rsi_checkpoint(path,branch) for branch,path in checkpoints.items()}
    registration={'round_index':index,'checkpoints':{b:{'path':str(p),
        'receipt_sha256':file_hash(p/'RSI_TRAINING_DONE.json'),
        'optimizer_steps':receipts[b]['optimizer_steps'],
        'parameter_delta_squared':receipts[b]['parameter_delta_squared']} for b,p in checkpoints.items()},
        'cohorts':{},'same_plan_and_seed_bytes':True}
    if (destination/'ROUND_READY.json').exists():
        old=json.loads((destination/'ROUND_READY.json').read_text())
        if old['checkpoints']!=registration['checkpoints']: raise ValueError('registered weight version changed')
        for value in old['cohorts'].values():
            if file_hash(value['plans'])!=value['plans_sha256'] or file_hash(value['config'])!=value['config_sha256']:
                raise ValueError('registered round files changed')
        return old
    if destination.exists(): raise ValueError('partial round registration requires inspection')
    for cohort,original,output in [('MAIN',root,destination),('FIT',root/'fit',destination/'fit')]:
        spec=copy.deepcopy(json.loads((original/'RUN_SPEC.json').read_text()))
        spec.update(run_root=str(output),run_id=f"{spec['run_id']}:theta{index}",round_index=index)
        spec['policy']={k:v for k,v in spec['policy'].items() if k not in ('max_edited_sites','max_numeric_bin_delta')}
        spec['policy']['prefer_raw_if_refiner_degrades']='training_preference_only'
        spec['assets'].update(generator_checkpoint=str(checkpoints['G']),editor_checkpoint=str(checkpoints['E']))
        if cohort=='FIT': spec['assets']['official_cache']=str(root/'fit_hull/official_mp_cache')
        spec['updated_checkpoint_receipts']=registration['checkpoints']
        plans=original/'cohort/plans.jsonl';rows=read_rows(plans)
        if len(rows)!=spec['requests'] or any((r.get('source_split')=='train')!=(cohort=='FIT') for r in rows):
            raise ValueError('original cohort size or TRAIN/EVAL identity changed')
        (output/'cohort').mkdir(parents=True)
        shutil.copyfile(plans,output/'cohort/plans.jsonl')
        write_json(output/'RUN_SPEC.json',spec)
        manifest={'parent_manifest':str(original/'cohort/MANIFEST.json'),
            'parent_manifest_sha256':file_hash(original/'cohort/MANIFEST.json'),
            'plans_sha256':file_hash(plans),'config_sha256':file_hash(output/'RUN_SPEC.json'),
            'requests':len(rows),'round_index':index,'source_split':'train' if cohort=='FIT' else 'evaluation',
            'preserved_plan_prompt_ids_and_all_seeds':True}
        write_json(output/'cohort/MANIFEST.json',manifest)
        (output/'cohort/_SUCCESS').touch()
        registration['cohorts'][cohort]={'plans':str(output/'cohort/plans.jsonl'),
            'plans_sha256':file_hash(plans),'config':str(output/'RUN_SPEC.json'),
            'config_sha256':file_hash(output/'RUN_SPEC.json')}
    write_json(destination/'ROUND_READY.json',registration)
    return registration


def prepare_training(root, index, branch):
    root=Path(root).resolve()
    if index not in (2,3) or branch not in ('G','E'): raise ValueError('invalid registered training update')
    previous=index-1
    checkpoint=root/'training'/f'round{previous}'/branch/'result/checkpoint'
    validate_rsi_checkpoint(checkpoint,branch)
    cohort=root/'rounds'/f'round{previous}'/'fit'
    data=cohort/'pairs'/f'{branch}.jsonl'
    manifest=cohort/'pairs'/f'PAIRS_{branch}_FINAL.json'
    report=json.loads(manifest.read_text())
    if report['source_split']!='train' or report['files_sha256'][data.name]!=file_hash(data):
        raise ValueError('new training preferences not bound to TRAIN')
    config=json.loads((root/'training_configs'/f'TRAIN_{branch}1.json').read_text())
    replay=[root/'fit/pairs'/f'{branch}.jsonl']
    replay += [root/'rounds'/f'round{i}'/'fit/pairs'/f'{branch}.jsonl' for i in range(1,previous)]
    config.update(checkpoint=str(checkpoint),data=str(data),seed=config['seed']+100*previous,
        output_dir=str(root/'training'/f'round{index}'/branch/'result'),replay_data=[str(p) for p in replay])
    path=root/'training_configs'/f'TRAIN_{branch}{index}.json'
    if path.exists():
        if json.loads(path.read_text())!=config: raise ValueError('existing training configuration changed')
    else: write_json(path,config)
    return {'config':str(path),'sha256':file_hash(path)}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True,type=Path)
    parser.add_argument('--index',required=True,type=int)
    parser.add_argument('--training-branch',choices=['G','E'])
    args=parser.parse_args()
    print(json.dumps(prepare_training(args.root,args.index,args.training_branch) if args.training_branch
                     else prepare_round(args.root,args.index)))
