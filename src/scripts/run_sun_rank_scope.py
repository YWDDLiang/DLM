"""Registered SUN rank supervision and independent local-site exploration."""
from __future__ import annotations
import argparse
import copy
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import sys

SOURCE=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(SOURCE/'src'))
sys.path.insert(0,str(SOURCE/'operations/r03_c3fd_main_20260907'))
from scripts.run_post_refine_cycle import read_rows,write_rows,write_json,file_hash,physics_record,validate_rsi_checkpoint
from scripts.run_rsi_stages import materialize
from crystal_dlm.post_refine_contract import fingerprint,derived_seed

STREAMS={'primary':None,'rank1':'keep_edit_focus_repeat1','rank2':'keep_edit_focus_repeat2',
         'rank3':'keep_edit_operational_fresh1'}


def register(root,previous,cache):
    from pymatgen.core import Composition
    from run_component import verify_deployed_source
    if (root/'PREREGISTRATION.json').exists():raise ValueError('SUN rank registration already exists')
    spec=json.loads((previous/'fit/RUN_SPEC.json').read_text())
    oldplans=read_rows(previous/'fit/cohort/plans.jsonl')
    canonical=lambda p:Composition(p['plan_state']['reduced_formula']).reduced_formula
    forbidden={canonical(p) for p in oldplans}
    for p in read_rows(Path(spec['assets']['cohort'])):
        try:forbidden.add(canonical(p))
        except (KeyError,ValueError,TypeError):
            if p.get('body_eligible'):raise
    pool=Path(spec['assets']['training_preparation'])/'pairs_pending.jsonl'
    covered={p['chemsys'] for p in read_rows(cache/'official_slim_cache.jsonl')}
    eligible=[]
    for row in read_rows(pool):
        p=row['plan_state']
        if (row['source_split']=='train' and p.get('rich_field_valid') and p.get('plan_end_marker_present')
            and canonical(row) not in forbidden and row.get('teacher_available') and row.get('target_body')
            and '-'.join(sorted(p['elements'])) in covered):eligible.append(row)
    unique={}
    for row in sorted(eligible,key=lambda r:fingerprint(dict(domain='sun_rank_fresh_source_v1',source=r['ancestor_id']))):
        unique.setdefault(canonical(row),row)
    keys=sorted(unique,key=lambda f:fingerprint(dict(domain='sun_rank_fresh_composition_v1',formula=f)))
    if len(keys)<256:raise ValueError('fewer than 256 fully excluded fresh compositions')
    fresh=[unique[f] for f in keys[:256]]
    now=dt.datetime.now(dt.timezone.utc);deadline=now+dt.timedelta(hours=3)
    root.mkdir(parents=True,exist_ok=True)
    write_json(root/'BUDGET.json',dict(schema='new_user_requested_sun_pilot_v1',dispatch_start_utc=now.isoformat(),
        deadline_utc=deadline.isoformat(),wall_hours=3,max_GPUs=5,max_queued_and_running_jobs=3,
        authorization='user request 2026-09-10T10:09:53Z; duration is an agent-selected stage cap',
        previous_budget_path=str(previous/'BUDGET.json'),previous_budget_sha256=file_hash(previous/'BUDGET.json')))
    pipe=json.loads((previous/'RAW0_PIPELINE.json').read_text())
    pipe.update(run_root=str(root),source_root=str(SOURCE),source_identity=verify_deployed_source(SOURCE))
    pipe['resources']=dict(deadline_utc=deadline.isoformat(),extra_gpu_until_utc=deadline.isoformat(),
        gpus_before_extra_window=5,gpus_after_extra_window=5,max_submitted_slurm_jobs=3)
    pipe.pop('components',None);pipe.pop('jobs',None);write_json(root/'RAW0_PIPELINE.json',pipe)
    shutil.copy2(previous/'PHYSICS_SOURCE_PIN.json',root/'PHYSICS_SOURCE_PIN.json')
    spec['assets']['official_cache']=str(cache);spec['assets']['nu_cache']=str(root/'nu_cache')
    spec['execution_policy'].update(single_GPUs=5,parallel_main_GPUs=3,parallel_other_GPUs=2,training_GPUs=1,cpus_per_gpu=4)
    spec['resources'].update(budget_receipt=str(root/'BUDGET.json'),base_gpus=5,max_gpus=5,
        wall_hours=3,start_utc=now.isoformat(),deadline_utc=deadline.isoformat())
    spec['editor_trial']=True
    fit=root/'fit';shutil.copytree(previous/'fit/cohort',fit/'cohort')
    spec.update(run_root=str(fit),run_id='sun_rank_scope_fit',training_parent_root=str(fit/'cohort'))
    write_json(fit/'RUN_SPEC.json',spec);write_json(root/'RUN_SPEC.json',spec)
    for stage in ('current','native'):
        original=read_rows(previous/f'fit/{stage}/inputs.jsonl')
        for i,row in enumerate(original):
            value=json.loads((previous/f'fit/native/records/{i:04d}.json').read_text()) if stage=='native' else dict(record=row)
            write_json(fit/f'{stage}/records/{i:04d}.json',value)
        materialize(spec,stage)
        if file_hash(fit/stage/'inputs.jsonl')!=file_hash(previous/f'fit/{stage}/inputs.jsonl'):
            raise ValueError('copied current inputs changed')
    shutil.copy2(previous/'SOURCE_SPLIT.jsonl',root/'SOURCE_SPLIT.jsonl')
    new=root/'fresh';parents=new/'cohort/parents.jsonl';write_rows(parents,fresh)
    plans=[]
    for i,row in enumerate(fresh):
        plans.append(dict(original_ordinal=i,evaluation_ordinal=i,sample_idx=i,body_eligible=True,
            ancestor_id=row['ancestor_id'],source_row_idx=row['source_row_idx'],source_split='train',
            plan_state=row['plan_state'],body_prompt=row['prompt'],
            body_noise_seed=derived_seed(row['ancestor_id'],'sun_rank_fresh_E'),
            cached_teacher=True,E_evaluation_role='final',canonical_reduced_formula=canonical(row)))
    write_rows(new/'cohort/plans.jsonl',plans)
    prep=dict(schema='sun_rank_fresh_cached_GF_sources_v1',files_sha256={'parents.jsonl':file_hash(parents)},
        heldout_cohort_sha256=file_hash(spec['assets']['cohort']),source_pool=str(pool),source_pool_sha256=file_hash(pool),
        plans_sha256=file_hash(new/'cohort/plans.jsonl'),sources=len(plans),eligible_sources=len(eligible),
        eligible_compositions=len(unique),excluded_previous_1000_and_MAIN_compositions=len(forbidden),
        original_source_split='MP20_train',E_role='sealed_final',no_physical_or_novelty_outcomes_loaded=True)
    write_json(new/'cohort/PREPARATION_FINAL.json',prep);write_json(new/'cohort/MANIFEST.json',prep)
    (new/'cohort/_SUCCESS').touch()
    fcfg=copy.deepcopy(spec);fcfg.update(run_root=str(new),run_id='sun_rank_scope_fresh',requests=256,
        training_parent_root=str(new/'cohort'),cached_GF_distribution='original_B0_F800',
        E_evaluation_role='sealed_final',updated_checkpoint_receipts={})
    fcfg['assets']['generator_checkpoint']=fcfg['assets']['b0_checkpoint']
    write_json(new/'RUN_SPEC.json',fcfg)
    reg=dict(schema='sun_rank_scope_v1',previous_run=str(previous),created_utc=now.isoformat(),
        plan_sha256=file_hash(SOURCE/'docs/r03_paper_story_20260907/SUN_RANK_SCOPE_PLAN_20260910.md'),
        source_split_sha256=file_hash(root/'SOURCE_SPLIT.jsonl'),fresh_preparation_sha256=file_hash(new/'cohort/PREPARATION_FINAL.json'),
        fresh_sources_sha256=file_hash(parents),old_editor=spec['assets']['editor_checkpoint'],
        old_editor_receipt_sha256=file_hash(Path(spec['assets']['editor_checkpoint'])/'RSI_TRAINING_DONE.json'),
        streams=STREAMS,source_scope='only E proposals and judgement; no G/F training or sampling',
        head_training=dict(seed=20260910,epochs=64,batch_size=256,learning_rate=.001,ridge=.1,SUN_pairwise_coefficient=.2),
        selection=dict(sun_thresholds=[0.,.02,.05,.1],ms_floors=[-.02,0.],sun_score_tie_width=.01,
            known_continuous_SUN_guard=True,DEV_constraints=['Stable>=KEEP','MSUN>=KEEP','SUN>KEEP']),
        fresh_endpoint_results_sealed_until_frozen_selection=True)
    write_json(root/'PREREGISTRATION.json',reg)
    print(json.dumps(dict(root=str(root),deadline_utc=deadline.isoformat(),fresh=prep)),flush=True)


def fresh_inputs(root):
    from transformers import AutoTokenizer
    from crystal_dlm.expert_edit_data import refined_arrays,decode_body,align_complete_target,quantize_arrays
    from crystal_dlm.dynamic_crystal import arrays_to_structure
    from crystal_dlm.continuous_keep_edit import geometry
    panel=root/'fresh';spec=json.loads((panel/'RUN_SPEC.json').read_text())
    if (panel/'INPUTS_FINAL.json').exists():raise ValueError('fresh inputs are already bound')
    tokenizer=AutoTokenizer.from_pretrained(spec['assets']['b0_checkpoint'],trust_remote_code=True)
    vocab=tokenizer.get_vocab();inverse={int(v):k for k,v in vocab.items()}
    parents=read_rows(panel/'cohort/parents.jsonl');plans=read_rows(panel/'cohort/plans.jsonl')
    expert=Path(spec['assets']['training_preparation']).parents[1];cache={};pins={};traces=[]
    for i,(parent,plan) in enumerate(zip(parents,plans,strict=True)):
        source_idx=parent['source_row_idx'];shard=(source_idx-2048)//1024
        if shard not in cache:
            candidates=sorted((expert/f'collection_expanded/shard_{shard}/teacher_refine').glob('dlm_refined_mp_*.pt'))
            if len(candidates)!=1:raise ValueError('cached teacher output is ambiguous or absent')
            path=candidates[0]
            conf=json.loads((path.parent/'run_config.json').read_text())
            if conf['diff_steps']!=800 or conf['checkpoint']!=spec['assets']['model494']:
                raise ValueError('fresh cached teacher differs')
            cache[shard]=refined_arrays(path);pins[str(path)]=file_hash(path)
        aligned,order=align_complete_target(decode_body(parent['old_body'],inverse),cache[shard][source_idx])
        body,_,_=quantize_arrays(aligned,vocab)
        if body!=parent['target_body'] or order!=parent['alignment']['target_order']:
            raise ValueError('fresh teacher/token correspondence changed')
        token=physics_record(plan,state_id=f'sun_rank_scope_fresh:current:{i}',body=''.join(inverse[t] for t in body))
        token.update(body_token_ids=body,body_prompt=plan['body_prompt'])
        native=physics_record(plan,state_id=f'sun_rank_scope_fresh:native:{i}',structure=arrays_to_structure(aligned).as_dict())
        check=geometry(native);trace=dict(source='cached_continuous_F800',editable=True,geometry=check)
        if check['valid'] is not True:
            native=copy.deepcopy(token);native['trajectory_id']=f'sun_rank_scope_fresh:native:{i}'
            trace.update(source='cached_token_geometry_fallback',fallback_reason=check['reason'])
        write_json(panel/f'current/records/{i:04d}.json',dict(record=token))
        write_json(panel/f'native/records/{i:04d}.json',dict(record=native,continuous_trace=trace,cached_source_row=source_idx))
        traces.append(dict(ordinal=i,source_row_idx=source_idx,trace=trace))
    for stage in ('current','native'):materialize(spec,stage)
    write_json(panel/'INPUTS_FINAL.json',dict(requests=len(plans),cached_teacher_pins=pins,traces=traces,
        native_sha256=file_hash(panel/'native/inputs.jsonl'),current_sha256=file_hash(panel/'current/inputs.jsonl'),
        no_physical_or_novelty_outcomes_loaded=True,new_GF_calls=0))
    print(json.dumps(dict(fresh_requests=len(plans),fallbacks=sum(x['trace']['source'].endswith('fallback') for x in traces))),flush=True)


def collect(root,panel_name,output):
    import torch
    from crystal_dlm.expert_edit import load_editor_model
    from crystal_dlm.rsi_minibatch import propose_ranked_batch
    from crystal_dlm.r03_physics_transfer import build_repair_constraints
    from crystal_dlm.utility_acceptance import materialize_continuous_patch
    from scripts.train_keep_edit_utility import feature_rows
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():raise ValueError('collection needs GPU allocation')
    rank=int(os.environ.get('LOCAL_RANK',0));world=int(os.environ.get('WORLD_SIZE',1))
    torch.set_num_threads(1);torch.cuda.set_device(rank);torch.use_deterministic_algorithms(True)
    panel=root/panel_name;spec=json.loads((panel/'RUN_SPEC.json').read_text())
    reg=json.loads((root/'PREREGISTRATION.json').read_text());previous=Path(reg['previous_run'])
    if panel_name=='fresh' and not (root/'FROZEN_SELECTION.json').exists():raise ValueError('fresh proposals await frozen policy')
    validate_rsi_checkpoint(reg['old_editor'],'E')
    model,tokenizer=load_editor_model(spec['assets']['base_model'],reg['old_editor'],torch.device('cuda',rank))
    support=build_repair_constraints(tokenizer);inverse={int(v):k for k,v in tokenizer.get_vocab().items()}
    plans=read_rows(panel/'cohort/plans.jsonl');current=read_rows(panel/'current/inputs.jsonl')
    native=[json.loads((panel/f'native/records/{i:04d}.json').read_text()) for i in range(len(plans))]
    completed=[]
    for k,(name,seed_stream) in enumerate(STREAMS.items()):
        if k%world!=rank:continue
        destination=panel/'bank'/name;destination.mkdir(parents=True,exist_ok=True)
        if (destination/'COLLECTION_FINAL.json').exists():raise ValueError('stream already finalized')
        traces={}
        if name=='primary' and panel_name=='fit':
            for i in range(len(plans)):
                traces[i]=json.loads((previous/f'fit/proposal/records/{i:04d}.json').read_text())['editor_trace']
        else:
            eligible=[i for i,c in enumerate(current) if c.get('body_token_ids') and c.get('success')]
            for offset in range(0,len(eligible),64):
                indices=eligible[offset:offset+64]
                requests=[dict(prompt=plans[i]['body_prompt'],body=current[i]['body_token_ids'],n=plans[i]['plan_state']['N'],
                    seed=derived_seed(str(plans[i]['body_noise_seed']),seed_stream or 'E'),known_sun=False,
                    force_proposal=True,**({'exploration_local_rank':k} if k else {})) for i in indices]
                values=propose_ranked_batch(model,tokenizer,requests,support=support,batch_size=64)
                traces.update(zip(indices,values,strict=True))
                print(json.dumps(dict(panel=panel_name,stream=name,generated=len(traces),requests=len(plans))),flush=True)
        rows=[];bindings=[]
        for i,(plan,before) in enumerate(zip(plans,current,strict=True)):
            trace=traces.get(i,dict(upstream_failure=True,proposal_generated=False,proposal_tokens=[],forward_calls=0))
            old_tokens=before.get('body_token_ids') or []
            if name=='primary' and panel_name=='fit':
                bound=json.loads((previous/f'materialized_primary/records/{i:04d}.json').read_text())
            else:bound=materialize_continuous_patch(native[i],old_tokens,trace.get('proposal_tokens') or [],inverse)
            write_json(destination/f'bound/{i:04d}.json',bound)
            write_json(destination/f'candidate/records/{i:04d}.json',dict(record=bound['record'],editor_trace=trace,
                bound_record_sha256=fingerprint(bound['record'])))
            bindings.append(dict(ordinal=i,record_sha256=fingerprint(bound['record']),trace_sha256=fingerprint(trace)))
            if old_tokens and trace.get('proposal_tokens'):
                rows.append(dict(ordinal=i,pair_id=fingerprint(dict(panel=panel_name,stream=name,source=plan['ancestor_id'])),
                    source_id=plan['ancestor_id'],prompt=plan['body_prompt'],num_sites=plan['plan_state']['N'],
                    current_tokens=old_tokens,proposal_tokens=trace['proposal_tokens'],action_positions=trace.get('action',{}).get('positions',[])))
        shutil.copytree(panel/'cohort',destination/'cohort')
        cfg=copy.deepcopy(spec);cfg.update(run_root=str(destination),training_parent_root=str(destination/'cohort'))
        write_json(destination/'RUN_SPEC.json',cfg);materialize(cfg,'candidate')
        write_rows(destination/'FEATURE_ROWS.jsonl',rows)
        feature_rows(model,tokenizer,rows,torch.device('cuda',rank),destination,'CANDIDATE')
        write_json(destination/'COLLECTION_FINAL.json',dict(rows=len(plans),stream=name,bindings=bindings,
            feature_rows=len(rows),features_sha256=file_hash(destination/'CANDIDATE_FEATURES.pt'),
            feature_rows_sha256=file_hash(destination/'FEATURE_ROWS.jsonl'),
            max_generation_calls=max(t.get('forward_calls',0) for t in traces.values()),
            no_candidate_endpoint_labels_loaded=True,content_checkpoint=reg['old_editor'],
            candidate_inputs_sha256=file_hash(destination/'candidate/inputs.jsonl')))
        completed.append(name)
    write_json(output/f'worker_{rank}_DONE.json',dict(panel=panel_name,streams=completed,world=world))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--mode',choices=['register','fresh_inputs','collect'],required=True)
    parser.add_argument('--previous',type=Path);parser.add_argument('--cache',type=Path)
    parser.add_argument('--panel',choices=['fit','fresh'],default='fit');parser.add_argument('--completion-dir',type=Path)
    args=parser.parse_args()
    if args.mode=='register':register(args.root,args.previous,args.cache)
    elif args.mode=='fresh_inputs':fresh_inputs(args.root)
    else:collect(args.root,args.panel,args.completion_dir)
