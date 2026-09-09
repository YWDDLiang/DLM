#!/usr/bin/env python3
"""Freeze the approved TRAIN panel and fresh base assets before GPU execution."""
import argparse
import copy
import datetime as dt
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.run_post_refine_cycle import read_rows,write_rows,write_json,file_hash
from crystal_dlm.post_refine_contract import derived_seed, registered_requests


def prepare(old,root,requests=None):
    old=Path(old).resolve();root=Path(root).resolve();fit=root/'fit'
    if (root/'PREPARED.json').exists():
        receipt=json.loads((root/'PREPARED.json').read_text())
        if requests is not None and receipt['requests']!=requests:
            raise ValueError('registered request count cannot change on resume')
        return receipt
    if fit.exists(): raise ValueError('partial TRAIN registration requires inspection')
    spec=json.loads((old/'fit/RUN_SPEC.json').read_text())
    requests=registered_requests(spec if requests is None else {'requests':requests})
    spec.update(run_id=f'{root.name}:S0',run_root=str(fit),ranked_training=True,requests=requests,
        round_index=0,training_parent_root=str(fit/'cohort'),collect_training_proposals=False,
        selection_tag=f'same_{requests}_TRAIN_Plans_ranked_v2')
    spec.pop('updated_checkpoint_receipts',None)
    spec['assets'].update(generator_checkpoint=spec['assets']['b0_checkpoint'],
        editor_checkpoint=str(fit/'initialization/checkpoint'),
        editor_reference=str(fit/'initialization/checkpoint'),
        official_cache=str(root/'fit_hull/official_mp_cache'),nu_cache=str(root/'nu_cache'))
    spec['policy'].update(preference_order=['SUN','strict_Stable','MetaStable','ordinary_improvement','unstable'],
        prefer_raw_if_refiner_degrades='training_preference_only',joint_physical_stop=True,
        physical_max_steps=1000,physical_teacher_runtime_selection=False)
    for key in ('max_edited_sites','max_numeric_bin_delta'): spec['policy'].pop(key,None)
    spec.setdefault('training_policy',{}).update(continue_heads_after_content_KL=True,
                                                historical_Stable_teachers=True)
    spec['complete_flow_before_judgment']=True
    if requests > 256:
        spec['execution_policy']={'single_GPUs':6,'parallel_main_GPUs':3,'parallel_other_GPUs':3,
            'training_GPUs':6,'training_batch_size':32,'cpus_per_gpu':4}
    spec['resources']={'wall_hours':12,'max_gpus':6,'base_gpus':6 if requests>256 else 4,'max_submitted_slurm_jobs':3,
        'start_utc':None,'deadline_utc':None,'budget_receipt':str(root/'BUDGET.json')}
    original=read_rows(old/'fit/cohort/plans.jsonl')
    if requests == len(original):
        shutil.copytree(old/'fit/cohort',fit/'cohort')
    else:
        from scripts.run_rsi_stages import prepare_fit
        prepare_fit(dict(spec,run_root=str(root)))
    plans=read_rows(fit/'cohort/plans.jsonl')
    if len(plans)!=requests or any(p.get('source_split')!='train' for p in plans):
        raise ValueError('original TRAIN cohort identity changed')
    prefix=min(len(original),requests)
    if plans[:prefix] != original[:prefix]:
        raise ValueError('expanded fixed panel changed the original Plan/prompt/seed prefix')
    shutil.copytree(old/'fit_hull/official_mp_cache',root/'fit_hull/official_mp_cache')
    write_json(fit/'RUN_SPEC.json',spec)
    write_json(root/'RUN_SPEC.json',dict(spec,run_root=str(root),role='TRAIN_experiment_registry'))
    twin=root/'bootstrap_comparator';twin_spec=copy.deepcopy(spec)
    twin_spec.update(run_root=str(twin),run_id=spec['run_id']+':fixed_second_RAW',role='G_preference_bootstrap')
    twin_plans=[dict(p,body_noise_seed=derived_seed(p['ancestor_id'],'ranked_fixed_second_RAW')) for p in plans]
    write_rows(twin/'cohort/plans.jsonl',twin_plans)
    write_json(twin/'cohort/MANIFEST.json',{'source_split':'train','requests':requests,
        'parent_plans_sha256':file_hash(fit/'cohort/plans.jsonl'),
        'plans_sha256':file_hash(twin/'cohort/plans.jsonl'),
        'replicate_seed_rule':'derived_seed(ancestor_id,ranked_fixed_second_RAW)',
        'seed_search':False,'purpose':'two_actual_RAW_candidates_for_first_G_update'})
    write_json(twin/'RUN_SPEC.json',twin_spec)
    pipeline=json.loads((old/'RAW0_PIPELINE.json').read_text())
    pipeline.update(run_root=str(root),purpose='training_feedback',components=[],jobs={})
    pipeline['resources']={'max_submitted_slurm_jobs':3,'gpus_before_extra_window':6,'gpus_after_extra_window':6,
                          'deadline_utc':None,'extra_gpu_until_utc':None}
    write_json(root/'RAW0_PIPELINE_TEMPLATE.json',pipeline)
    receipt={'source_split':'train','requests':requests,'plans_sha256':file_hash(fit/'cohort/plans.jsonl'),
        'source_plans_sha256':file_hash(old/'fit/cohort/plans.jsonl'),'base_generator':spec['assets']['b0_checkpoint'],
        'initial_E_seed':20260909,'old_editor_checkpoint_used':False,'GPU_experiment_started':False,
        'preserved_original_prefix_requests':prefix,
        'prepared_utc':dt.datetime.now(dt.timezone.utc).isoformat()}
    write_json(root/'PREPARED.json',receipt)
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous-root',type=Path,required=True)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--requests',type=int)
    args=parser.parse_args();print(json.dumps(prepare(args.previous_root,args.root,args.requests)))
