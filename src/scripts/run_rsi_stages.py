#!/usr/bin/env python3
"""Bound cohort, materialization and SUN-gated refiner stages for RSI."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_post_refine_cycle import (read_rows, file_hash, write_json, write_rows,
    load_config, physics_record, load_refiner, refine_one, structure_from_refined)
from crystal_dlm.post_refine_contract import derived_seed, fingerprint, make_gate, Quality, preference, make_pair
from crystal_dlm.sun_feedback_contract import composition_counts, reduced_key


def prepare_fit(spec):
    root = Path(spec['run_root']) / 'fit'
    parent = Path(spec['assets']['training_preparation']) / 'pairs_pending.jsonl'
    heldout = Path(spec['assets']['cohort'])
    forbidden = set()
    for row in read_rows(heldout):
        try:
            forbidden.add(reduced_key(composition_counts(row['plan_state'])))
        except (KeyError, ValueError, TypeError):
            if row.get('body_eligible'):
                raise
    # Hash selection is frozen before scoring and one occurrence per source.
    sources = {}
    for row in read_rows(parent):
        plan = row['plan_state']
        if (row['source_split'] == 'train' and plan.get('rich_field_valid') is True
                and plan.get('plan_end_marker_present') is True
                and reduced_key(composition_counts(plan)) not in forbidden):
            sources.setdefault(row['ancestor_id'], row)
    selected = sorted(sources.values(), key=lambda r: fingerprint({'fit_panel': 'rsi_v1',
                                                     'ancestor': r['ancestor_id']}))[:256]
    if len(selected) != 256:
        raise ValueError('not enough composition-excluded training conditions')
    plans = []
    for index, row in enumerate(selected):
        plans.append({'original_ordinal': index, 'sample_idx': index, 'evaluation_ordinal': index,
            'body_eligible': True, 'plan_state': row['plan_state'], 'body_prompt': row['prompt'],
            'ancestor_id': row['ancestor_id'], 'source_row_idx': row['source_row_idx'],
            'source_split': 'train', 'body_noise_seed': derived_seed(row['ancestor_id'], 'fit_G'),
            'refiner_noise_seed': derived_seed(row['ancestor_id'], 'fit_F')})
    write_rows(root/'cohort/plans.jsonl', plans)
    write_rows(root/'cohort/parents.jsonl', selected)
    registration = {'selection': 'rsi_v1_hash_before_outcomes', 'count': len(plans),
        'original_pool': {'path': str(parent), 'sha256': file_hash(parent)},
        'heldout_cohort_sha256': file_hash(heldout),
        'files_sha256': {'parents.jsonl': file_hash(root/'cohort/parents.jsonl')},
        'plans_sha256': file_hash(root/'cohort/plans.jsonl'),
        'use': 'training_only_all_256', 'heldout_compositions': len(forbidden)}
    write_json(root/'cohort/PREPARATION_FINAL.json', registration)
    write_json(root/'cohort/MANIFEST.json', registration)
    (root/'cohort/_SUCCESS').touch()
    fit_spec = {k: v for k, v in spec.items() if not k.startswith('_')}
    fit_spec.update(run_root=str(root), run_id=spec['run_id']+':fit', requests=len(plans),
                    training_parent_root=str(root/'cohort'))
    fit_spec['policy'] = {k:v for k,v in fit_spec['policy'].items()
                          if k not in ('max_edited_sites','max_numeric_bin_delta')}
    write_json(root/'RUN_SPEC.json', fit_spec)
    print(json.dumps(registration), flush=True)


def materialize(spec, stage):
    root = Path(spec['run_root'])
    plans = read_rows(root/'cohort/plans.jsonl')
    source = root / stage
    records = [json.loads((source/'records'/f"{p['original_ordinal']:04d}.json").read_text())['record'] for p in plans]
    if any(r['evaluation_ordinal'] != i for i,r in enumerate(records)):
        raise ValueError('stage record ordering changed')
    path = source/'inputs.jsonl'
    write_rows(path, records)
    if plans[0].get('source_split') == 'train':
        parent_root = Path(spec['training_parent_root'])
        files = {'paths': path, 'parent_preparation': parent_root/'PREPARATION_FINAL.json',
                 'parent_pairs': parent_root/'parents.jsonl', 'heldout_cohort': Path(spec['assets']['cohort'])}
        manifest = {'schema': 'sun_training_feedback_inputs_v1', 'purpose': 'training_feedback',
            'expected_requests': len(records), 'endpoint': 'native',
            **{key: {'path':str(value), 'sha256':file_hash(value)} for key,value in files.items()}}
        write_json(source/'FEEDBACK_MANIFEST.json', manifest)
        from crystal_dlm.sun_feedback_contract import validate_training_feedback
        validate_training_feedback(records, path, source/'FEEDBACK_MANIFEST.json')
    write_json(source/'MATERIALIZED.json', {'inputs_sha256':file_hash(path), 'requests':len(records)})


def gate(spec):
    root = Path(spec['run_root']) / 'construction'
    score_dir = root/'scoring/result'
    reports = list(score_dir.glob('*REPORT.json')) + list(score_dir.glob('*FINAL.json'))
    if len(reports) != 1:
        raise ValueError('exactly one complete score report required: '+str(reports))
    report = json.loads(reports[0].read_text())
    if report['status'] != 'complete':
        raise ValueError('incomplete raw scoring cannot gate F')
    decision = make_gate(read_rows(root/'inputs.jsonl'), read_rows(score_dir/'attempt_results.jsonl'),
        input_sha256=file_hash(root/'inputs.jsonl'), scored_input_sha256=report['input_sha256'],
        score_identity={'report_sha256':file_hash(reports[0]),'scores_sha256':file_hash(score_dir/'attempt_results.jsonl')})
    write_json(root/'GATE.json', decision)


def refine(spec, shard, shards):
    import torch
    from transformers import AutoTokenizer
    from crystal_dlm.expert_edit_data import quantize_arrays, arrays_from_structure, certify_geometry
    from crystal_dlm.r03_physics_transfer import build_repair_constraints, geometry_support_report
    if 'SLURM_JOB_ID' not in os.environ or not torch.cuda.is_available():
        raise RuntimeError('refiner requires a Slurm GPU')
    rank = int(os.environ.get('LOCAL_RANK', '0'))
    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    root = Path(spec['run_root'])
    plans = read_rows(root/'cohort/plans.jsonl')
    raw = read_rows(root/'construction/inputs.jsonl')
    frozen_gate = json.loads((root/'construction/GATE.json').read_text())
    if frozen_gate['draft_input_sha256'] != file_hash(root/'construction/inputs.jsonl'):
        raise ValueError('raw gate input changed')
    tokenizer = AutoTokenizer.from_pretrained(spec['assets']['b0_checkpoint'], trust_remote_code=True)
    vocab = tokenizer.get_vocab()
    inverse = {int(v):k for k,v in vocab.items()}
    support = build_repair_constraints(tokenizer)
    model, Data, DataLoader = load_refiner(spec, torch.device('cuda',rank))
    started = time.monotonic()
    for index in range(shard, len(plans), shards):
        plan, before, decision = plans[index], raw[index], frozen_gate['decisions'][index]
        original = plan['original_ordinal']
        if decision['input_fingerprint'] != fingerprint(before):
            raise ValueError('raw gate occurrence changed')
        result = None
        failure = None
        if decision['run_diffusion']:
            graph_path = root/'construction/graphs'/f'{original:04d}.pt'
            if graph_path.is_file():
                graph = torch.load(graph_path, map_location='cpu', weights_only=False)
                try:
                    result = refine_one(graph, model=model, Data=Data, DataLoader=DataLoader,
                                        seed=plan['refiner_noise_seed'], steps=800)
                except (ValueError,FloatingPointError) as error:
                    failure = str(error)
            else:
                failure = 'raw_graph_unavailable'
        float_record = dict(before, trajectory_id=f"{spec['run_id']}:F:{original}")
        token_record = dict(before, trajectory_id=f"{spec['run_id']}:tokenF:{original}")
        token_trace = {'fallback_to_raw': False, 'reason': decision['reason']}
        if result is not None:
            structure = structure_from_refined(result)
            float_record = physics_record(plan,state_id=float_record['trajectory_id'],structure=structure)
            try:
                ids, decoded, diagnostic = quantize_arrays(arrays_from_structure(structure),vocab)
                report = geometry_support_report(ids, constraints=support)
                if not report['supported'] or not certify_geometry(decoded)['valid']:
                    raise ValueError('token_F_geometry_outside_hard_support')
                token_record = physics_record(plan,state_id=token_record['trajectory_id'],
                                               body=''.join(inverse[i] for i in ids))
                token_record.update(body_token_ids=ids,body_prompt=plan['body_prompt'])
                token_trace.update(quantization=diagnostic, geometry=report)
            except ValueError as error:
                token_trace.update(fallback_to_raw=True,reason=str(error))
        elif failure:
            float_record = physics_record(plan,state_id=float_record['trajectory_id'],reason=failure)
            token_trace.update(fallback_to_raw=True,reason=failure)
        for stage,record in [('refined',float_record),('tokenized',token_record)]:
            write_json(root/stage/'records'/f'{original:04d}.json',
                {'record':record,'raw_refiner_output':result,'tokenization':token_trace,
                 'gate':decision,'config_sha256':spec['_config_sha256']})
        if index//shards % 5 == 0:
            print(json.dumps({'shard':shard,'index':index,'seconds':time.monotonic()-started}),flush=True)
    write_json(root/'refined'/f'worker_{shard}_DONE.json',{'seconds':time.monotonic()-started})


def scores(root, stage):
    directory=root/stage/'scoring/result'
    report=next(directory.glob('*FINAL.json'))
    summary=json.loads(report.read_text())
    if summary['status']!='complete' or summary['input_sha256']!=file_hash(root/stage/'inputs.jsonl'):
        raise ValueError('stage scores are incomplete or not bound to the current inputs')
    return read_rows(directory/'attempt_results.jsonl')


def select_current(spec):
    root=Path(spec['run_root'])
    before,after=(read_rows(root/stage/'inputs.jsonl') for stage in ['construction','tokenized'])
    left,right=(scores(root,stage) for stage in ['construction','tokenized'])
    for raw,token,a,b in zip(before,after,left,right,strict=True):
        if not raw['sample_idx']==token['sample_idx']==a['sample_idx']==b['sample_idx']:
            raise ValueError('raw/F comparison source IDs differ')
        choice=preference(Quality.from_score(a),Quality.from_score(b))
        source=raw if choice['chosen']=='before' else token
        original=source['original_ordinal']
        record=dict(source,trajectory_id=f"{spec['run_id']}:current:{original}")
        write_json(root/'current/records'/f'{original:04d}.json',{'record':record,
            'source_stage':'construction' if source is raw else 'tokenized','preference':choice})
    materialize(spec,'current')


def edit(spec,shard,shards):
    import torch
    from crystal_dlm.expert_edit import load_editor_model
    from crystal_dlm.r03_physics_transfer import build_repair_constraints
    from crystal_dlm.rsi_preference import propose_editor
    rank=int(os.environ.get('LOCAL_RANK','0'))
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('editor requires a Slurm GPU')
    torch.cuda.set_device(rank);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    root=Path(spec['run_root']);plans=read_rows(root/'cohort/plans.jsonl')
    current=read_rows(root/'current/inputs.jsonl');quality=scores(root,'current')
    checkpoint=spec['assets'].get('editor_checkpoint',spec['assets']['editor_reference'])
    model,tokenizer=load_editor_model(spec['assets']['base_model'],checkpoint,torch.device('cuda',rank))
    support=build_repair_constraints(tokenizer); inverse={int(v):k for k,v in tokenizer.get_vocab().items()}
    started=time.monotonic()
    for index in range(shard,len(plans),shards):
        plan,record,score=plans[index],current[index],quality[index]
        original=plan['original_ordinal'];trace={'upstream_failure':True}
        proposed=dict(record,trajectory_id=f"{spec['run_id']}:proposal:{original}")
        edited=dict(record,trajectory_id=f"{spec['run_id']}:edited:{original}")
        if record['success'] and record.get('body_token_ids'):
            trace=propose_editor(model,tokenizer,prompt=plan['body_prompt'],body=record['body_token_ids'],
                n=plan['plan_state']['N'],support=support,seed=derived_seed(str(plan['body_noise_seed']),'E'),
                known_sun=score['strict_sun'] is True,force_proposal=plan.get('source_split')=='train',
                keep_prior=spec['policy']['known_SUN_keep_prior'])
            for output,key in [(proposed,'proposal_tokens'),(edited,'final_tokens')]:
                output.update(body_token_ids=trace[key],body=''.join(inverse[i] for i in trace[key]),structure=None)
        for stage,value in [('proposal',proposed),('edited',edited)]:
            write_json(root/stage/'records'/f'{original:04d}.json',{'record':value,'editor_trace':trace,
                       'checkpoint':checkpoint,'config_sha256':spec['_config_sha256']})
        if index//shards%5==0: print(json.dumps({'index':index,'shard':shard,'seconds':time.monotonic()-started}),flush=True)
    write_json(root/'proposal'/f'worker_{shard}_DONE.json',{'seconds':time.monotonic()-started})


def compile_pairs(spec):
    root=Path(spec['run_root']);plans=read_rows(root/'cohort/plans.jsonl')
    if any(p.get('source_split')!='train' for p in plans):
        raise ValueError('evaluation plans cannot enter preference training')
    stages=['construction','tokenized','current','proposal']
    data={stage:read_rows(root/stage/'inputs.jsonl') for stage in stages}
    labels={stage:scores(root,stage) for stage in stages}
    result={'G':[],'E':[]};audit=[]
    for index,plan in enumerate(plans):
        for branch,before_stage,after_stage in [('G','construction','tokenized'),('E','current','proposal')]:
            before=data[before_stage][index];after=data[after_stage][index]
            a=labels[before_stage][index];b=labels[after_stage][index]
            if not before.get('body_token_ids') or not after.get('body_token_ids'): continue
            if len({p['sample_idx'] for p in [before,after,a,b]})!=1: raise ValueError('pair source misalignment')
            pair=make_pair(source_id=plan['ancestor_id'],conditioning={'prompt':plan['body_prompt'],
                **({'current_tokens':before['body_token_ids']} if branch=='E' else {})},
                before_tokens=before['body_token_ids'],after_tokens=after['body_token_ids'],before_score=a,after_score=b,
                before_identity={'trajectory_id':before['trajectory_id'],'record_sha256':fingerprint(before)},
                after_identity={'trajectory_id':after['trajectory_id'],'record_sha256':fingerprint(after)},
                round_index=spec.get('round_index',0),purpose='train')
            audit.append(dict(pair,branch=branch));choice=pair['preference']['chosen']
            example={'source_id':plan['ancestor_id'],'source_split':'train','source_row_idx':plan['source_row_idx'],
                'prompt':plan['body_prompt'],'num_sites':plan['plan_state']['N'],'pair_id':pair['pair_id'],
                'known_sun':a['strict_sun'] is True,'chosen_tokens':None,'rejected_tokens':None}
            # Preservation on equal successful SUN states is a decision label,
            # never a fabricated physical preference against lower energy.
            same_success=branch=='E' and a['strict_sun'] is True and b['strict_sun'] is True
            if choice is not None and not same_success:
                chosen,rejected=(after,before) if choice=='after' else (before,after)
                example.update(chosen_tokens=chosen['body_token_ids'],rejected_tokens=rejected['body_token_ids'])
            if branch=='G':
                if a['strict_sun'] is True: example['healthy_anchor_tokens']=before['body_token_ids']
                if example['chosen_tokens'] is None and 'healthy_anchor_tokens' not in example: continue
            else:
                example.update(current_tokens=before['body_token_ids'],proposal_tokens=after['body_token_ids'])
                known=Quality.from_score(a).tier is not None and Quality.from_score(b).tier is not None
                prefer_edit=choice=='after' and not same_success
                example.update(mode_target=int(prefer_edit) if known else (0 if example['known_sun'] else None),
                               accept_target=int(prefer_edit) if known else None)
                if example['mode_target'] is None and example['chosen_tokens'] is None: continue
            result[branch].append(example)
    directory=root/'pairs'
    write_rows(directory/'pair_audit.jsonl',audit)
    for branch,rows in result.items(): write_rows(directory/f'{branch}.jsonl',rows)
    write_json(directory/'PAIRS_FINAL.json',{'source_split':'train','round_index':spec.get('round_index',0),
        'source_config_sha256':spec['_config_sha256'],'plans_sha256':file_hash(root/'cohort/plans.jsonl'),
        'files_sha256':{name:file_hash(directory/name) for name in ['G.jsonl','E.jsonl','pair_audit.jsonl']},
        'stage_inputs':{stage:file_hash(root/stage/'inputs.jsonl') for stage in stages},
        'stage_scores':{stage:file_hash(root/stage/'scoring/result/attempt_results.jsonl') for stage in stages},
        'examples':{branch:len(rows) for branch,rows in result.items()},'DPO_training_performed':False})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--action',choices=['prepare-fit','materialize','gate','refine','select','edit','pairs'],required=True)
    parser.add_argument('--stage',default='construction')
    args=parser.parse_args()
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    spec=load_config(args.config)
    if args.action=='prepare-fit': prepare_fit(spec)
    elif args.action=='materialize': materialize(spec,args.stage)
    elif args.action=='gate': gate(spec)
    elif args.action=='select': select_current(spec)
    elif args.action=='pairs': compile_pairs(spec)
    elif args.action=='edit': edit(spec,int(os.environ.get('RANK','0')),int(os.environ.get('WORLD_SIZE','1')))
    else: refine(spec,int(os.environ.get('RANK','0')),int(os.environ.get('WORLD_SIZE','1')))


if __name__=='__main__': main()
