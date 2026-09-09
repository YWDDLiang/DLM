#!/usr/bin/env python3
"""TRAIN-only ranked RSI initialization, teachers, datasets and retention."""
from __future__ import annotations
import argparse
import copy
import gzip
import json
import os
from pathlib import Path
import random
import shutil
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.run_post_refine_cycle import read_rows,write_rows,write_json,file_hash,load_config,physics_record
from scripts.run_rsi_stages import scores,score_directory,materialize
from crystal_dlm.post_refine_contract import fingerprint
from crystal_dlm.ranked_feedback import (ranked_preference,endpoint_quality,align_fixed_slots,
    target_action,validate_action_target,SCHEMA)


def require_train(spec):
    if not spec.get('ranked_training') or not spec.get('training_parent_root'):
        raise ValueError('ranked work requires a registered TRAIN-only cohort')
    root=Path(spec['run_root']);plans=read_rows(root/'cohort/plans.jsonl')
    if len(plans)!=256 or any(p.get('source_split')!='train' for p in plans):
        raise ValueError('all original 256 TRAIN requests are required')
    return root,plans


def initialize(spec):
    import torch
    from crystal_dlm.expert_edit import load_editor_model,inference_view,materialize_edit_batch
    from scripts.train_rsi_preferences import tensor_hash
    root,plans=require_train(spec)
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('initialization requires the registered GPU allocation')
    device=torch.device('cuda',0);torch.cuda.set_device(0);torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    seed=20260909;torch.manual_seed(seed);random.seed(seed)
    destination=Path(spec['assets']['editor_checkpoint'])
    if destination.exists(): raise ValueError('initial E must be written once')
    original=Path(spec['assets']['b0_checkpoint'])
    if any((original/name).exists() for name in ('EXPERT_EDITOR.json','expert_edit_config.json')):
        raise ValueError('fresh E cannot start from a previously trained editor')
    model,tokenizer=load_editor_model(spec['assets']['base_model'],original,device,trainable=True)
    model.training_modes={'G':[0,1,2,3],'S':[0,1,2,3]}
    model.eval()
    # Use a real fixed-Plan generated body to verify the saved initializer.
    row=next(r for r in read_rows(root/'construction/inputs.jsonl') if r.get('body_token_ids'))
    plan=plans[row['evaluation_ordinal']];prefix=tokenizer(plan['body_prompt'],add_special_tokens=False)['input_ids']
    probe=[inference_view(prefix,row['body_token_ids'],row['body_token_ids'],plan['plan_state']['N'],1)]
    model.save_pretrained(destination,safe_serialization=True,save_embedding_layers=False)
    tokenizer.save_pretrained(destination)
    batch=materialize_edit_batch(probe,tokenizer,device)
    with torch.no_grad(): output=model(batch['input_ids'],attention_mask=batch['attention_mask'],edit_context=batch['edit_context'])
    torch.save({'examples':probe,**{name:getattr(output,name).cpu() for name in
        ('logits','mode_logits','site_logits','count_logits','quality_logits')}},destination/'roundtrip_probe.pt')
    receipt={'schema':'ranked_fresh_E_initialization_v1','seed':seed,'base_checkpoint':str(original),
        'module_initialization_order':['state_conditioner',*model.extra_modules()],
        'torch_version':torch.__version__,'optimizer_steps':0,'old_editor_weights_used':False,
        'IO_sha256':{name:tensor_hash(layer.weight) for name,layer in
            [('input',model.get_input_embeddings()),('output',model.get_output_embeddings())]},
        'files_sha256':{p.name:file_hash(p) for p in destination.iterdir() if p.is_file()}}
    write_json(destination/'INITIALIZATION_FINAL.json',receipt)


def teachers(spec):
    """Quantize actual CHGNet terminals; their new endpoints must be rescored."""
    from transformers import AutoTokenizer
    from pymatgen.core import Structure
    from crystal_dlm.expert_edit_data import quantize_arrays,arrays_from_structure,certify_geometry
    from crystal_dlm.r03_physics_transfer import build_repair_constraints,geometry_support_report
    root,plans=require_train(spec)
    tokenizer=AutoTokenizer.from_pretrained(spec['assets']['b0_checkpoint'],trust_remote_code=True)
    vocab=tokenizer.get_vocab();inverse={int(v):k for k,v in vocab.items()}
    support=build_repair_constraints(tokenizer)
    current=read_rows(root/'current/inputs.jsonl')
    labels={r['trajectory_id']:r for r in read_rows(root/'current/labeling/result/labels.jsonl')}
    for plan,old in zip(plans,current,strict=True):
        label=labels[old['trajectory_id']];ordinal=plan['original_ordinal']
        trace={'origin':'CHGNet_terminal_teacher','label_trajectory_id':label['trajectory_id'],
               'label_sha256':fingerprint(label),'physical_teacher_used_at_runtime':False,
               'teacher_distribution':None,'actual_quantized_endpoint_rescoring_required':True}
        record=physics_record(plan,state_id=f"{spec['run_id']}:teacher:{ordinal}",reason='unavailable_physical_teacher')
        if label.get('final_structure') and old.get('body_token_ids'):
            try:
                ids,decoded,diagnostic=quantize_arrays(arrays_from_structure(Structure.from_dict(label['final_structure'])),vocab)
                if not certify_geometry(decoded)['valid']: raise ValueError('invalid quantized teacher')
                aligned,permutation=align_fixed_slots(ids,old['body_token_ids'])
                if not geometry_support_report(aligned,constraints=support)['supported']:
                    raise ValueError('teacher outside unchanged hard support')
                action=target_action(old['body_token_ids'],aligned)
                validate_action_target(old['body_token_ids'],aligned,action['positions'])
                record=physics_record(plan,state_id=record['trajectory_id'],body=''.join(inverse[i] for i in aligned))
                record.update(body_token_ids=aligned,body_prompt=plan['body_prompt'])
                trace.update(action=action,atom_permutation=permutation,quantization=diagnostic)
            except (ValueError,TypeError,KeyError) as error: trace['failure']=str(error)
        write_json(root/'teacher/records'/f'{ordinal:04d}.json',{'record':record,'teacher_trace':trace,
            'config_sha256':spec['_config_sha256']})
    materialize(spec,'teacher')


def collection(spec,checkpoint):
    root,plans=require_train(spec);destination=root/'editor_collection'
    target=copy.deepcopy({k:v for k,v in spec.items() if not k.startswith('_')})
    target.update(run_root=str(destination),run_id=spec['run_id']+':collect',collect_training_proposals=True,
                  collection_current_config=str(root/'RUN_SPEC.json'))
    target['assets']['editor_checkpoint']=str(checkpoint)
    target.get('updated_checkpoint_receipts',{}).pop('E',None)
    if destination.exists():
        if json.loads((destination/'RUN_SPEC.json').read_text())!=target: raise ValueError('collection already changed')
        return
    destination.mkdir()
    shutil.copytree(root/'cohort',destination/'cohort')
    # Exact immutable current inputs and physical evidence retain their identities.
    shutil.copytree(root/'current',destination/'current')
    write_json(destination/'RUN_SPEC.json',target)
    write_json(destination/'COLLECTION_BINDING.json',{'current_inputs_sha256':file_hash(root/'current/inputs.jsonl'),
        'source_config_sha256':spec['_config_sha256'],'editor_checkpoint':str(checkpoint),
        'role':'E_previous_on_fresh_G_current_states'})


def compile_dataset(spec,branch,comparators=()):
    root,plans=require_train(spec);examples=[];audit=[];pins={}
    def stage(at,name):
        at=Path(at);key=str(at/name)
        records=read_rows(at/name/'inputs.jsonl');measured=scores(at,name)
        pins[key]={'inputs_sha256':file_hash(at/name/'inputs.jsonl'),
                   'scores_sha256':file_hash(score_directory(at,name)/'attempt_results.jsonl')}
        if len(records)!=256 or len(measured)!=256: raise ValueError('pair cohort truncated')
        return records,measured
    if branch=='G':
        current,value=stage(root,'construction')[0],stage(root,'tokenized')[1]
        comparisons=[]
        for other in comparators:
            other=Path(other);old_plans=read_rows(other/'cohort/plans.jsonl')
            if any((p['ancestor_id'],p['body_prompt'],p['plan_state'])!=(q['ancestor_id'],q['body_prompt'],q['plan_state'])
                   for p,q in zip(plans,old_plans,strict=True)): raise ValueError('G conditioning changed between RAW candidates')
            comparisons.append(('G_RAW_F_value',*stage(other,'construction')[:1],stage(other,'tokenized')[1],current,value,other))
        if not comparisons: raise ValueError('G requires two actual RAW candidates with F values')
    else:
        current,value=stage(root,'current')
        comparisons=[(name,current,value,*stage(root,name),root) for name in ('proposal','teacher')]
    for name,before,first,after,second,source in comparisons:
        for index,plan in enumerate(plans):
            a,b=before[index],after[index];qa,qb=first[index],second[index]
            if len({r['sample_idx'] for r in (a,b,qa,qb)})!=1: raise ValueError('pair source alignment failed')
            preference=ranked_preference(qa,qb)
            item={'source_id':plan['ancestor_id'],'source_split':'train','source_row_idx':plan['source_row_idx'],
                  'prompt':plan['body_prompt'],'num_sites':plan['plan_state']['N'],
                  'known_sun':endpoint_quality(qa)['rank']==4,'objective_level':preference['objective_level'] or 'decision',
                  'priority':preference['priority'],'chosen_tokens':None,'rejected_tokens':None,
                  'origin':name,'source_round':spec.get('round_index',0),
                  'teacher_distribution':None,'log_probability_kind':'masked_conditional_surrogate'}
            trace={'preference':preference,'before_record_sha256':fingerprint(a),'after_record_sha256':fingerprint(b),
                   'before_score_sha256':fingerprint(qa),'after_score_sha256':fingerprint(qb),
                   'source':str(source),'conditioning_sha256':fingerprint({'Plan':plan['body_prompt'],
                        'current':a.get('body_token_ids') if branch=='E' else None})}
            item['pair_id']=fingerprint(trace);audit.append(dict(trace,pair_id=item['pair_id']))
            if not a.get('body_token_ids') or not b.get('body_token_ids'): continue
            left=a['body_token_ids']
            try: right,permutation=align_fixed_slots(b['body_token_ids'],left)
            except ValueError as error:
                audit[-1]['excluded']=str(error);continue
            choice=preference['chosen']
            if branch=='E':
                wrapper=json.loads((root/name/'records'/f"{plan['original_ordinal']:04d}.json").read_text())
                action=(wrapper.get('editor_trace') or wrapper.get('teacher_trace') or {}).get('action')
                if not action: action=target_action(left,right)
                try: validate_action_target(left,right,action['positions'])
                except ValueError as error:
                    audit[-1]['excluded']=str(error);continue
                known=(endpoint_quality(qa)['reliable'] and endpoint_quality(qb)['reliable']) or choice is not None
                positive=choice=='after' and left!=right
                item.update(current_tokens=left,proposal_tokens=right,action_positions=action['positions'],action_spec=action,
                    mode_target=action['mode'] if positive else 0 if known or item['known_sun'] else None,
                    site_targets=[float(i in action['sites']) for i in range(item['num_sites'])],
                    accept_target=int(positive) if known else None)
                if item['mode_target'] is None: continue
                if not action['positions']: choice=None
            elif endpoint_quality(qa)['rank'] in (2,3,4):
                item['healthy_anchor_tokens']=left
            if choice is not None and left!=right:
                item.update(chosen_tokens=right if choice=='after' else left,rejected_tokens=left if choice=='after' else right)
            if branch=='G' and item['chosen_tokens'] is None and 'healthy_anchor_tokens' not in item: continue
            item.update(atom_permutation=permutation,preference=preference,conditioning_sha256=trace['conditioning_sha256'])
            examples.append(item)
    directory=root/'pairs';data=directory/f'{branch}.jsonl';audit_file=directory/f'pair_audit_{branch}.jsonl'
    write_rows(data,examples);write_rows(audit_file,audit)
    write_json(directory/f'PAIRS_{branch}_FINAL.json',{'schema':SCHEMA,'source_split':'train',
        'source_config_sha256':spec['_config_sha256'],'plans_sha256':file_hash(root/'cohort/plans.jsonl'),
        'files_sha256':{p.name:file_hash(p) for p in (data,audit_file)},'stage_evidence':pins,
        'examples':len(examples),'round_index':spec.get('round_index',0),'DPO_training_performed':False})


def archive(spec):
    root,plans=require_train(spec);directory=root/'archive';directory.mkdir(exist_ok=True)
    all_records=[];all_labels=[];files={}
    for stage in ('construction','refined','tokenized','current','proposal','teacher','edited'):
        for path in sorted((root/stage/'records').glob('*.json')):
            all_records.append({'stage':stage,'record_file_sha256':file_hash(path),**json.loads(path.read_text())})
        label=root/stage/'labeling/result/labels.jsonl'
        if label.exists(): all_labels.extend(dict(stage=stage,**row) for row in read_rows(label))
    for name,rows in [('TRAJECTORIES.jsonl.gz',all_records),('PHYSICS_LABELS.jsonl.gz',all_labels)]:
        path=directory/name
        with gzip.GzipFile(filename=str(path),mode='wb',mtime=0) as stream:
            for row in rows: stream.write((json.dumps(row,sort_keys=True)+'\n').encode())
        files[name]=file_hash(path)
    for branch in ('G','E'):
        source=root/'pairs'/f'{branch}.jsonl'
        if not source.exists(): continue
        path=directory/f'{branch}_PAIRS.jsonl';shutil.copyfile(source,path);files[path.name]=file_hash(path)
        if branch=='E':
            rows=read_rows(source)
            for name,subset in [('EDIT_TARGETS.jsonl',[r for r in rows if r.get('mode_target',0)]),
                                ('KEEP_ANCHORS.jsonl',[r for r in rows if r.get('mode_target')==0])]:
                write_rows(directory/name,subset);files[name]=file_hash(directory/name)
    write_json(directory/'DATASET_MANIFEST.json',{'schema':SCHEMA,'source_split':'train','requests':256,
        'round_index':spec.get('round_index',0),'config_sha256':spec['_config_sha256'],
        'plans_sha256':file_hash(root/'cohort/plans.jsonl'),'files_sha256':files,
        'origin_checkpoint_assets':spec['assets'],'future_replay_is_off_policy':True,
        'teacher_token_distribution':None,'full_trajectory_likelihood_available':False})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--action',choices=['initialize','teachers','collection','compile','archive'],required=True)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--branch',choices=['G','E'])
    parser.add_argument('--comparators',type=Path,nargs='*',default=[])
    args=parser.parse_args();spec=load_config(args.config)
    if args.action=='initialize': initialize(spec)
    elif args.action=='teachers': teachers(spec)
    elif args.action=='collection': collection(spec,args.checkpoint)
    elif args.action=='compile': compile_dataset(spec,args.branch,args.comparators)
    else: archive(spec)
