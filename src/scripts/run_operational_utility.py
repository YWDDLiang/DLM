"""Evaluate a fixed operational-utility head and model-selected repair pools."""
import argparse
import copy
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import sys

SOURCE=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(SOURCE/'src'));sys.path.insert(0,str(SOURCE/'operations/r03_c3fd_main_20260907'))
from scripts.run_post_refine_cycle import read_rows,write_rows,write_json,file_hash,load_config,validate_rsi_checkpoint
from scripts.run_rsi_stages import materialize,scores
from crystal_dlm.post_refine_contract import fingerprint


def sources(root):
    return dict(primary=(root/'fit',root/'materialized_primary'),
        repeat1=(root/'repeats/repeat1',root/'repeats/repeat1/materialized_proposal'),
        repeat2=(root/'repeats/repeat2',root/'repeats/repeat2/materialized_proposal'),
        fresh1=(root/'operational/fresh_open_generation',root/'operational/fresh_open_generation/materialized_proposal'))


def prepare(root):
    output=root/'operational';path=output/'MODEL_DEFINITION.json'
    if path.exists():raise ValueError('operational utility model is already frozen')
    training_path=root/'training/operational_utility/result/TRAINING_FINAL.json'
    training=json.loads(training_path.read_text());reg=json.loads((root/'OPERATIONAL_REGISTRATION.json').read_text())
    if training['settings']['snapshots']!=[64] or training['settings']['thresholds']!=[.05]:
        raise ValueError('operational policy was not fixed before training')
    snapshot=training['snapshots']['64'];base=Path(training['original_checkpoint']);validate_rsi_checkpoint(base,'E')
    for name,(panel,bound) in sources(root).items():
        if not (panel/'proposal/inputs.jsonl').exists() or not (bound/'MATERIALIZATION_FINAL.json').exists():
            raise ValueError('operational proposal inputs are incomplete:'+name)
    write_json(path,dict(schema='frozen_operational_utility_patch_v1',created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        base_editor=str(base),base_editor_receipt_sha256=file_hash(base/'RSI_TRAINING_DONE.json'),
        quality_head=snapshot['path'],quality_head_sha256=snapshot['sha256'],epoch=64,raw_margin=.05,
        training_receipt_path=str(training_path),training_receipt_sha256=file_hash(training_path),
        source_split_sha256=file_hash(root/'SOURCE_SPLIT.jsonl'),
        registration_sha256=file_hash(root/'OPERATIONAL_REGISTRATION.json'),
        committed_plan_sha256=file_hash(SOURCE/'docs/r03_paper_story_20260907/KEEP_EDIT_OPERATIONAL_UTILITY_PLAN_20260910.md'),
        sources={name:dict(proposals=str(panel),materialized=str(bound),
            proposal_inputs_sha256=file_hash(panel/'proposal/inputs.jsonl'),
            materialization_sha256=file_hash(bound/'MATERIALIZATION_FINAL.json')) for name,(panel,bound) in sources(root).items()},
        pools=dict(primary_K1=['primary'],pool_A_K2=['primary','repeat1'],pool_B_K2=['repeat2','fresh1'],
            all_K4=['primary','repeat1','repeat2','fresh1']),
        known_current_SUN_guard='verified_continuous_native_current',
        failed_or_not_converged_current_with_valid_token_view_is_editable=True,
        selection='highest_model_utility_above_fixed_margin; ties_use_registered_stream_order',
        no_physical_candidate_reranking=True,validation_status=reg['evaluation_status']))
    print(json.dumps(dict(model_definition=str(path),sha256=file_hash(path))),flush=True)


def infer(root,output):
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import torch
    from crystal_dlm.expert_edit import load_editor_model,materialize_edit_batch
    from crystal_dlm.utility_acceptance import canonical_judgements
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():raise ValueError('operational inference needs GPU allocation')
    rank=int(os.environ.get('LOCAL_RANK',0));world=int(os.environ.get('WORLD_SIZE',1));torch.cuda.set_device(rank)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);device=torch.device('cuda',rank)
    definition=json.loads((root/'operational/MODEL_DEFINITION.json').read_text())
    base=Path(definition['base_editor']);validate_rsi_checkpoint(base,'E')
    if file_hash(definition['quality_head'])!=definition['quality_head_sha256']:raise ValueError('operational weights changed')
    spec=json.loads((root/'fit/RUN_SPEC.json').read_text())
    model,tokenizer=load_editor_model(spec['assets']['base_model'],base,device)
    before={k:v.detach().cpu().clone() for k,v in model.quality_head.state_dict().items()}
    state=torch.load(definition['quality_head'],map_location=device,weights_only=True);model.quality_head.load_state_dict(state)
    if any(not torch.equal(v.cpu(),before[k]) for k,v in state.items() if not k.startswith('layers.2.')):
        raise ValueError('operational inference changed the quality hidden representation')
    if any(not torch.equal(state[k][:3].cpu(),before[k][:3]) for k in ('layers.2.weight','layers.2.bias')):
        raise ValueError('operational inference changed another quality output')
    probe=torch.load(base/'roundtrip_probe.pt',map_location='cpu',weights_only=False)
    batch=materialize_edit_batch(probe['examples'],tokenizer,device)
    with torch.no_grad():actual=model(batch['input_ids'],attention_mask=batch['attention_mask'],edit_context=batch['edit_context'])
    if any(not torch.equal(getattr(actual,k).cpu(),probe[k]) for k in ('logits','mode_logits','site_logits','count_logits')):
        raise ValueError('operational utility changed content or scope probe outputs')
    plans=read_rows(root/'fit/cohort/plans.jsonl');current=read_rows(root/'fit/current/inputs.jsonl');done=[]
    for index,(name,origin) in enumerate(definition['sources'].items()):
        if index%world!=rank:continue
        panel=Path(origin['proposals']);proposal=read_rows(panel/'proposal/inputs.jsonl');views=[]
        if file_hash(panel/'proposal/inputs.jsonl')!=origin['proposal_inputs_sha256']:raise ValueError('operational proposal cache changed')
        for i,(plan,a,b) in enumerate(zip(plans,current,proposal,strict=True)):
            if not a.get('body_token_ids') or not b.get('body_token_ids'):continue
            trace=json.loads((panel/f'proposal/records/{i:04d}.json').read_text())['editor_trace']
            views.append(dict(ordinal=i,prompt=plan['body_prompt'],current_tokens=a['body_token_ids'],
                proposal_tokens=b['body_token_ids'],num_sites=plan['plan_state']['N'],
                action_positions=trace.get('action',{}).get('positions',[])))
        values,_=canonical_judgements(model,tokenizer,views)
        if name=='primary':
            cached={r['ordinal']:r['learned_utilities']['64'] for r in read_rows(root/'training/operational_utility/evaluation/UTILITY_PREDICTIONS.jsonl')}
            drift=max(abs(v-cached[i]) for i,v in values.items())
            if drift>1e-5:raise ValueError('operational full model differs from trained cached head')
        else:drift=None
        destination=output/name;write_rows(destination/'MODEL_SCORES.jsonl',[dict(ordinal=i,raw_utility=v) for i,v in values.items()])
        write_json(destination/'SCORES_FINAL.json',dict(rows=len(values),stream=name,source=origin,
            full_model_checkpoint=str(base),quality_head_sha256=definition['quality_head_sha256'],
            full_model_vs_cached_max_drift=drift,content_scope_probe_identical=True,
            no_endpoint_labels_loaded=True,scores_sha256=file_hash(destination/'MODEL_SCORES.jsonl')))
        done.append(name);print(json.dumps(dict(rank=rank,completed_stream=name,rows=len(values),drift=drift)),flush=True)
    write_json(output/f'worker_{rank}_DONE.json',dict(streams=done,world=world,
        model_definition_sha256=file_hash(root/'operational/MODEL_DEFINITION.json')))


def combine(root):
    definition=json.loads((root/'operational/MODEL_DEFINITION.json').read_text());score_root=root/'operational/model_inference'
    for rank in range(2):
        if not (score_root/f'worker_{rank}_DONE.json').exists():raise ValueError('operational model inference incomplete')
    current=read_rows(root/'fit/current/inputs.jsonl');native_scores=scores(root/'fit','native')
    from crystal_dlm.utility_acceptance import materialized_continuous_decision
    from editor_trial_analysis import flags
    predictions={name:{r['ordinal']:r['raw_utility'] for r in read_rows(score_root/name/'MODEL_SCORES.jsonl')}
        for name in definition['sources']}
    wrappers={};traces={}
    for name,origin in definition['sources'].items():
        path=Path(origin['materialized']);marker=json.loads((path/'MATERIALIZATION_FINAL.json').read_text())
        if file_hash(path/'MATERIALIZATION_FINAL.json')!=origin['materialization_sha256']:raise ValueError('operational candidate realization changed')
        wrappers[name]=[];traces[name]=[]
        for i in range(1000):
            record=path/f'records/{i:04d}.json'
            if file_hash(record)!=marker['bound_record_sha256'][i]:raise ValueError('operational candidate bytes changed')
            wrappers[name].append(json.loads(record.read_text()))
            traces[name].append(json.loads((Path(origin['proposals'])/f'proposal/records/{i:04d}.json').read_text())['editor_trace'])
    summary={}
    for name,streams in definition['pools'].items():
        panel=root/'operational/panels'/name
        if panel.exists():raise ValueError('operational selection panel already exists:'+name)
        panel.mkdir(parents=True);shutil.copytree(root/'fit/cohort',panel/'cohort')
        spec=json.loads((root/'fit/RUN_SPEC.json').read_text());spec.update(run_root=str(panel),run_id='operational:'+name,
            experimental_policy=name,decision_model_definition_sha256=file_hash(root/'operational/MODEL_DEFINITION.json'))
        spec['assets']['official_cache']=json.loads((root/'REFERENCE_CACHE_OVERRIDE.json').read_text())['directory']
        write_json(panel/'RUN_SPEC.json',spec);decisions=[]
        for i,record in enumerate(current):
            native=json.loads((root/f'fit/native/records/{i:04d}.json').read_text())
            known_sun=flags(native_scores[i])['SUN'];valid=[]
            for order,stream in enumerate(streams):
                trace=traces[stream][i];bound=wrappers[stream][i];raw=predictions[stream].get(i)
                if trace.get('proposal_generated') and bound['commit_trace']['applied'] and raw is not None:
                    valid.append((raw,-order,stream))
            chosen=max(valid)[2] if valid and not known_sun else None
            if chosen is not None:
                value,decision=materialized_continuous_decision(native,record.get('body_token_ids') or [],traces[chosen][i],
                    wrappers[chosen][i],predictions[chosen][i],definition['raw_margin'],known_sun_guard=known_sun)
            else:value,decision=copy.deepcopy(native['record']),dict(learned_accept=False,actual_edit=False,raw_utility=None)
            calls=sum(traces[stream][i].get('forward_calls',0)+(i in predictions[stream]) for stream in streams)
            if calls>80:raise ValueError('repair pool exceeds original per-request DLM budget')
            value['trajectory_id']=f'{spec["run_id"]}:edited:{i}'
            decision.update(ordinal=i,chosen_stream=chosen,known_continuous_SUN_guard=known_sun,
                valid_candidate_count=len(valid),dlm_calls=calls)
            write_json(panel/f'edited/records/{i:04d}.json',dict(record=value,editor_trace=decision));decisions.append(decision)
        materialize(spec,'edited')
        write_json(panel/'DECISION_BINDING.json',dict(decisions=decisions,streams=streams,
            model_definition_sha256=file_hash(root/'operational/MODEL_DEFINITION.json'),
            physics_used_for_acceptance=False,failed_current_is_not_a_veto=True))
        summary[name]=dict(actual_edits=sum(x['actual_edit'] for x in decisions),
            max_dlm_calls=max(x['dlm_calls'] for x in decisions),inputs_sha256=file_hash(panel/'edited/inputs.jsonl'))
    write_json(root/'operational/SELECTION_FINAL.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--mode',choices=['prepare','infer','combine'],required=True);parser.add_argument('--completion-dir',type=Path)
    args=parser.parse_args()
    if args.mode=='prepare':prepare(args.root)
    elif args.mode=='combine':combine(args.root)
    else:
        if args.completion_dir is None:parser.error('inference needs a completion directory')
        infer(args.root,args.completion_dir)
