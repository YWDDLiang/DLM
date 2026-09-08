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
    score_dir = score_directory(Path(spec['run_root']),'construction')
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


def score_directory(root,stage):
    pointer=root/stage/'SCORING_DIRECTORY.json'
    relative=json.loads(pointer.read_text())['directory'] if pointer.exists() else 'scoring'
    directory=(root/stage/relative/'result').resolve()
    if (root/stage).resolve() not in directory.parents: raise ValueError('score directory escaped stage')
    return directory


def scores(root, stage):
    directory=score_directory(root,stage)
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
        # A physical preference labels training; it does not rerank deployment.
        # The editor's authoritative current state is always token-F. The only
        # online physical branch is the previously measured raw-SUN F bypass.
        source=token
        original=source['original_ordinal']
        record=dict(source,trajectory_id=f"{spec['run_id']}:current:{original}")
        write_json(root/'current/records'/f'{original:04d}.json',{'record':record,
            'source_stage':'tokenized','preference_for_training_only':choice,
            'physical_reranking_performed':False})
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
    expected=spec.get('updated_checkpoint_receipts',{}).get('E')
    if expected and (expected.get('pending_training_receipt') or
            expected['receipt_sha256']!=file_hash(Path(checkpoint)/'RSI_TRAINING_DONE.json')):
        raise ValueError('editor update is not finalized for this weight version')
    if (Path(checkpoint)/'RSI_TRAINING_DONE.json').exists():
        from scripts.run_post_refine_cycle import validate_rsi_checkpoint
        validate_rsi_checkpoint(checkpoint,'E')
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


def compile_pairs(spec, requested_branch='both'):
    root=Path(spec['run_root']);plans=read_rows(root/'cohort/plans.jsonl')
    if any(p.get('source_split')!='train' for p in plans):
        raise ValueError('evaluation plans cannot enter preference training')
    comparisons=[('G','construction','tokenized'),('E','current','proposal')]
    comparisons=[v for v in comparisons if requested_branch=='both' or v[0]==requested_branch]
    stages=list(dict.fromkeys(stage for _,a,b in comparisons for stage in (a,b)))
    data={stage:read_rows(root/stage/'inputs.jsonl') for stage in stages}
    labels={stage:scores(root,stage) for stage in stages}
    result={branch:[] for branch,_,_ in comparisons};audit=[]
    for index,plan in enumerate(plans):
        for branch,before_stage,after_stage in comparisons:
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
    audit_name='pair_audit.jsonl' if requested_branch=='both' else f'pair_audit_{requested_branch}.jsonl'
    manifest_name='PAIRS_FINAL.json' if requested_branch=='both' else f'PAIRS_{requested_branch}_FINAL.json'
    write_rows(directory/audit_name,audit)
    for branch,rows in result.items(): write_rows(directory/f'{branch}.jsonl',rows)
    write_json(directory/manifest_name,{'source_split':'train','round_index':spec.get('round_index',0),
        'source_config_sha256':spec['_config_sha256'],'plans_sha256':file_hash(root/'cohort/plans.jsonl'),
        'files_sha256':{name:file_hash(directory/name) for name in [*[f'{branch}.jsonl' for branch in result],audit_name]},
        'stage_inputs':{stage:file_hash(root/stage/'inputs.jsonl') for stage in stages},
        'stage_scores':{stage:file_hash(score_directory(root,stage)/'attempt_results.jsonl') for stage in stages},
        'examples':{branch:len(rows) for branch,rows in result.items()},'DPO_training_performed':False})


def rebind_labels(spec, stage):
    """Reuse exactly identical physical endpoints, retaining the complete cohort.

    Current is token-F; edited is chosen by the learned current/proposal decision.
    This only copies proven identical physical labels. Every N/U stage is rerun.
    """
    from collections import Counter
    import importlib.util
    from crystal_dlm.sun_feedback_contract import validate_training_feedback
    source_root=Path(__file__).resolve().parents[2]
    binding=importlib.util.spec_from_file_location('_rsi_bound_evaluation',source_root/'scripts/evaluate_programmed_paths.py')
    api=importlib.util.module_from_spec(binding);binding.loader.exec_module(api)
    root=Path(spec['run_root']);training='training_parent_root' in spec
    purpose='training_feedback' if training else 'evaluation'
    sources={'current':['construction','tokenized'],'edited':['current','proposal']}
    if stage not in sources: raise ValueError('this stage contains newly sampled endpoints')
    known={};pins=[];runtime=None;protocol=None
    def key(record):
        return (record.get('group_id'),record.get('source_row_idx'),record['sample_idx'],
                api.endpoint_cache_key(record) if record['success'] else 'explicit_generation_failure')
    for parent in sources[stage]:
        inputs=root/parent/'inputs.jsonl';records=read_rows(inputs)
        scope=validate_training_feedback(records,inputs,root/parent/'FEEDBACK_MANIFEST.json') if training else None
        directory=root/parent/'labeling/result'
        labels,identities=api.load_bound_evaluation_labels(records,[directory/'labels.jsonl'],
            paths_file=inputs,endpoint='native',purpose=purpose,feedback_scope=scope)
        report=json.loads((directory/'LABEL_FINAL.json').read_text())
        actual=report['runtime_identities'][0]
        if actual['labeler_sha256']!=file_hash(source_root/'scripts/label_programmed_paths.py'):
            raise ValueError('cached endpoint uses a different physical implementation')
        if runtime is not None and runtime!=actual: raise ValueError('physical caches use different runtimes')
        runtime=actual;protocol=report['protocol']
        pins.extend(identities)
        for record in records:
            label=labels[record['trajectory_id']]
            k=key(record)
            if k in known:
                # Proven duplicate endpoints must carry the same physical evidence.
                for field in ['raw_energy','terminal_energy','status','verified']:
                    if known[k][field]!=label[field]: raise ValueError('identical endpoints have contradictory cached labels')
            known[k]=label
    inputs=root/stage/'inputs.jsonl';records=read_rows(inputs)
    scope=validate_training_feedback(records,inputs,root/stage/'FEEDBACK_MANIFEST.json') if training else None
    labels=[];occurrence_bindings=[]
    for record in records:
        if key(record) not in known: raise ValueError('new endpoint cannot inherit a previous physical label')
        label=dict(known[key(record)])
        origin=label['trajectory_id']
        label.update({name:record.get(name) for name in ['trajectory_id','group_id','source_row_idx','source_split','endpoint']})
        label.update(endpoint_cache_key=api.endpoint_cache_key(record))
        occurrence_bindings.append({'trajectory_id':record['trajectory_id'],'rebound_from_trajectory_id':origin,
                                    'endpoint_cache_key':label['endpoint_cache_key']})
        labels.append(label)
    directory=root/stage/'labeling/result';write_rows(directory/'labels.jsonl',labels)
    report={'requested':len(labels),'completed':len(labels),'statuses':dict(Counter(r['status'] for r in labels)),
            'purpose':purpose,'protocol':protocol,'verification_protocol':api.TERMINAL_VERIFICATION_PROTOCOL,
            'geometry_validation_protocol':api.LABEL_GEOMETRY_PROTOCOL,'runtime_identities':[runtime],
            'input_file':str(inputs),'input_sha256':file_hash(inputs),'training_feedback_scope':scope,
            'distinct_endpoint_evaluations':0,'new_endpoint_evaluations':0,'physical_reuse_sources':pins,
            'binding_implementation_sha256':file_hash(Path(__file__)), 'occurrence_bindings':occurrence_bindings,
            'N_U_copied':False,'exact_endpoint_and_source_identity_checked':True}
    write_json(directory/'LABEL_FINAL.json',report);(directory/'_SUCCESS').touch()
    api.load_bound_evaluation_labels(records,[directory/'labels.jsonl'],paths_file=inputs,endpoint='native',
                                    purpose=purpose,feedback_scope=scope)


def validity(spec,stage):
    from collections import Counter
    import math
    import importlib.util
    from crystal_dlm.post_refine_contract import stage_summary
    from crystal_dlm.dynamic_crystal import arrays_to_structure,parse_dynamic_answer
    from pymatgen.core import Structure
    if not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('validity requires its CPU allocation')
    project=Path(spec['assets']['b0_checkpoint']).parents[4]
    snapshot=project/'runs/20260814_h1a2_epoch2_exactplan1200_h1a2_r03_refine800_fullsun1000_v3/frozen/best/workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis_refined_repeats4_v1/runtime/crystal_dlm/wqcodiff/crysllmgen/upstream'
    utility=snapshot/'eval_utils.py'
    expected='68e6d0a9703f412cfd3215e6d0ae687e5b153e941d16f3fa4f2fffeedb505cb6'
    if file_hash(utility)!=expected: raise ValueError('frozen validity implementation changed')
    sys.path.insert(0,str(snapshot))
    binding=importlib.util.spec_from_file_location('_rsi_frozen_validity',utility)
    api=importlib.util.module_from_spec(binding);binding.loader.exec_module(api)
    root=Path(spec['run_root']);records=read_rows(root/stage/'inputs.jsonl')
    measured=scores(root,stage);rows=[]
    for record,score in zip(records,measured,strict=True):
        comp=False;struct=False;reason=record.get('reason')
        if record['success']:
            try:
                structure=Structure.from_dict(record['structure']) if record.get('structure') else arrays_to_structure(parse_dynamic_answer(record['body'],strict=True))
                counts=Counter(int(v) for v in structure.atomic_numbers);elements=tuple(sorted(counts))
                amounts=[counts[e] for e in elements];divisor=math.gcd(*amounts)
                comp=bool(api.smact_validity(elements,tuple(v//divisor for v in amounts)))
                struct=bool(api.structure_validity(structure))
            except (ValueError,TypeError,KeyError) as error: reason=str(error)
        rows.append({'sample_idx':record['sample_idx'],'trajectory_id':record['trajectory_id'],
                     'comp_valid':comp,'Struct_valid':struct,'SUN':score['strict_sun'],'MSUN':score['meta_sun'],'reason':reason})
    output=score_directory(root,stage)
    write_rows(output/'four_metrics.jsonl',rows)
    write_json(output/'BASIC_METRICS.json',{'requested':len(rows),'stage':stage,
        'comp_valid':{'count':sum(r['comp_valid'] for r in rows),'percent':100*sum(r['comp_valid'] for r in rows)/len(rows)},
        'Struct_valid':{'count':sum(r['Struct_valid'] for r in rows),'percent':100*sum(r['Struct_valid'] for r in rows)/len(rows)},
        'stability':stage_summary(measured),'frozen_validity_sha256':expected,
        'inputs_sha256':file_hash(root/stage/'inputs.jsonl'),'score_sha256':file_hash(output/'attempt_results.jsonl'),
        'metrics_sha256':file_hash(output/'four_metrics.jsonl')})


def baseline_raw(spec,arm):
    """Select the registered baseline bodies under the identical frozen Plans."""
    import hashlib
    root=Path(spec['run_root']);plans=read_rows(root/'cohort/plans.jsonl')
    asset='control_body' if arm=='H1A2' else 'candidate_body'
    source=Path(spec['assets'][asset])
    source_digest=file_hash(source)
    if source_digest!=spec['source_sha256'][asset]: raise ValueError('saved baseline bodies changed')
    records={row['ordinal']:row for row in read_rows(source)}
    stage='baselines/'+arm+'_raw'
    for plan in plans:
        original=plan['original_ordinal'];saved=records[original]
        if (saved['body_noise_seed']!=plan['body_noise_seed'] or
                saved['body_prompt_sha256']!=hashlib.sha256(plan['body_prompt'].encode()).hexdigest()):
            raise ValueError('saved baseline Plan prompt or generation seed differs')
        success=saved['status']=='succeeded' and saved['body_generation_complete'] is True
        record=physics_record(plan,state_id=f"{spec['run_id']}:{arm}:raw:{original}",
                              body=saved['text'] if success else None,reason=None if success else saved.get('reason'))
        if success: record['body_token_ids']=saved['raw_body_token_ids']
        write_json(root/stage/'records'/f'{original:04d}.json',{'record':record,
            'source_file':str(source),'source_sha256':source_digest,'original_ordinal':original,
            'saved_generation_policy':saved['generation_policy'],'config_sha256':spec['_config_sha256']})
    materialize(spec,stage)


def baseline_chunk(spec,arm,chunk,chunks,shard,shards):
    """Independent fixed-Plan baseline chunks; reuse only bound identical F800."""
    import torch
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available(): raise RuntimeError('Slurm GPU required')
    rank=int(os.environ.get('LOCAL_RANK','0'));torch.cuda.set_device(rank);torch.set_num_threads(1)
    root=Path(spec['run_root']);plans=read_rows(root/'cohort/plans.jsonl')
    asset='control_graphs' if arm=='H1A2' else 'candidate_graphs'
    if file_hash(spec['assets'][asset])!=spec['source_sha256'][asset]: raise ValueError('baseline graph archive changed')
    entries=torch.load(spec['assets'][asset],map_location='cpu',weights_only=False)
    graphs={int(row['ordinal']):row['graph'] for row in entries}
    source=root.parent/'RUN_SPEC.json';old_spec=json.loads(source.read_text())
    reusable=(old_spec['source_sha256'][asset]==spec['source_sha256'][asset]
              and old_spec['assets']['model494']==spec['assets']['model494'])
    model,Data,DataLoader=load_refiner(spec,torch.device('cuda',rank))
    selected=plans[chunk::chunks][shard::shards];started=time.monotonic();copied=0;sampled=0
    for plan in selected:
        original=plan['original_ordinal'];output=root/'baselines'/arm/'records'/f'{original:04d}.json'
        if output.exists(): raise ValueError('baseline chunk overlaps an already stored endpoint')
        raw=None;reuse=None;reason=None
        old=root.parent/'baselines'/arm/'records'/f'{original:04d}.json'
        if reusable and old.exists():
            previous=json.loads(old.read_text());candidate=previous.get('raw_refiner_output')
            if (previous['config_sha256']==file_hash(source) and candidate is not None
                    and candidate['seed']==plan['refiner_noise_seed'] and candidate['diffusion_steps']==800):
                raw=candidate;reuse={'path':str(old),'sha256':file_hash(old),'config_sha256':file_hash(source),
                    'matched_graph_archive_sha256':spec['source_sha256'][asset],'same_original_seed_and_F800':True};copied+=1
        if raw is None and original in graphs:
            try:
                raw=refine_one(dict(graphs[original],sample_idx=original),model=model,Data=Data,DataLoader=DataLoader,
                               seed=plan['refiner_noise_seed'],steps=800);sampled+=1
            except (ValueError,FloatingPointError) as error: reason=str(error)
        elif raw is None: reason='baseline_saved_graph_missing'
        record=physics_record(plan,state_id=f"{spec['run_id']}:{arm}:F:{original}",
                              structure=structure_from_refined(raw) if raw else None,reason=reason)
        write_json(output,{'record':record,'raw_refiner_output':raw,'reused':reuse,'config_sha256':spec['_config_sha256']})
        print(json.dumps({'arm':arm,'chunk':chunk,'shard':shard,'copied':copied,'sampled':sampled,'seconds':time.monotonic()-started}),flush=True)
    write_json(root/'baselines'/arm/f'chunk{chunk}'/f'worker_{shard}_DONE.json',
               {'copied':copied,'sampled':sampled,'requests':len(selected),'seconds':time.monotonic()-started})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--action',choices=['prepare-fit','materialize','gate','refine','select','edit','pairs','rebind','validity','baseline-raw','baseline-chunk'],required=True)
    parser.add_argument('--stage',default='construction')
    parser.add_argument('--arm',choices=['H1A2','R03'],default='H1A2')
    parser.add_argument('--chunk',type=int,default=0)
    parser.add_argument('--chunks',type=int,default=4)
    parser.add_argument('--branch',choices=['G','E','both'],default='both')
    args=parser.parse_args()
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    spec=load_config(args.config)
    if args.action=='prepare-fit': prepare_fit(spec)
    elif args.action=='materialize': materialize(spec,args.stage)
    elif args.action=='gate': gate(spec)
    elif args.action=='select': select_current(spec)
    elif args.action=='pairs': compile_pairs(spec,args.branch)
    elif args.action=='rebind': rebind_labels(spec,args.stage)
    elif args.action=='validity': validity(spec,args.stage)
    elif args.action=='baseline-raw': baseline_raw(spec,args.arm)
    elif args.action=='baseline-chunk':
        if not 0<=args.chunk<args.chunks: parser.error('invalid baseline chunk')
        baseline_chunk(spec,args.arm,args.chunk,args.chunks,int(os.environ.get('RANK','0')),int(os.environ.get('WORLD_SIZE','1')))
    elif args.action=='edit': edit(spec,int(os.environ.get('RANK','0')),int(os.environ.get('WORLD_SIZE','1')))
    else: refine(spec,int(os.environ.get('RANK','0')),int(os.environ.get('WORLD_SIZE','1')))


if __name__=='__main__': main()
