"""Frozen-E3 feature extraction and full-pass quality-head utility regression."""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.run_post_refine_cycle import read_rows,write_rows,write_json,file_hash


def tensor_hash(tensor):
    import torch
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def feature_rows(model, tokenizer, rows, device, output, name, batch_size=16):
    import torch
    from crystal_dlm.expert_edit import inference_view,materialize_edit_batch
    features=[]; original=[]; captured=[]; started=time.monotonic()
    def capture(module, inputs): captured.append(inputs[0].detach().float().cpu())
    hook=model.quality_head.register_forward_pre_hook(capture)
    try:
        for start in range(0,len(rows),batch_size):
            selected=rows[start:start+batch_size]
            views=[inference_view(tokenizer(r['prompt'],add_special_tokens=False)['input_ids'],
                r['current_tokens'],r['proposal_tokens'],r['num_sites'],1,r['action_positions'],remaining=80,reveal=1.) for r in selected]
            batch=materialize_edit_batch(views,tokenizer,device)
            with torch.no_grad():
                actual=model(batch['input_ids'],attention_mask=batch['attention_mask'],edit_context=batch['edit_context'])
            if len(captured)!=1: raise ValueError('quality feature hook count changed')
            features.append(captured.pop()); original.append(actual.quality_logits.detach().cpu())
            del actual,batch
            if start%256==0:
                event=dict(stage='feature_extraction',kind=name,rows=min(start+batch_size,len(rows)),total=len(rows),
                    seconds=time.monotonic()-started,peak_GPU_GB=torch.cuda.max_memory_allocated()/1e9)
                write_json(output/f'{name}_PROGRESS.json',event);print(json.dumps(event),flush=True)
    finally: hook.remove()
    values=torch.cat(features); logits=torch.cat(original)
    payload=dict(features=values,original_quality_logits=logits,pair_ids=[r['pair_id'] for r in rows],
        feature_tensor_sha256=tensor_hash(values),original_logit_sha256=tensor_hash(logits),
        feature_context=dict(task_id=1,remaining=80,reveal=1.,active='registered_action_positions'))
    torch.save(payload,output/f'{name}_FEATURES.pt')
    return payload


def protected_hashes(model):
    return {name:tensor_hash(value) for name,value in model.named_parameters()
        if not name.startswith('quality_head.') and ('lora_' in name or not name.startswith('base_model.'))}


def train(root, model, tokenizer, device, reg):
    import torch
    output=root/'training/utility/result';output.mkdir(parents=True,exist_ok=True)
    data=root/'data/UTILITY_TRAIN.jsonl'; manifest=json.loads((root/'data/UTILITY_DATA_FINAL.json').read_text())
    if manifest['data_sha256']!=file_hash(data):raise ValueError('registered utility data changed')
    rows=read_rows(data)
    roles={r['ancestor_id']:r['split'] for r in read_rows(root/'SOURCE_SPLIT.jsonl')}
    if any(r['focus_split']!='train' or roles[r['source_id']]!='train' for r in rows):raise ValueError('utility training split leakage')
    frozen_before=protected_hashes(model)
    for parameter in model.parameters():parameter.requires_grad_(False)
    if any(p.requires_grad for p in model.parameters()):raise ValueError('feature model is not frozen')
    feature=feature_rows(model,tokenizer,rows,device,output,'TRAIN')
    head=copy.deepcopy(model.quality_head).float().to(device)
    for p in head.parameters():p.requires_grad_(True)
    initial={k:v.detach().clone() for k,v in head.state_dict().items()}
    inputs=feature['features'].to(device);targets=torch.tensor([r['utility_target'] for r in rows],device=device,dtype=torch.float32)
    weights=torch.tensor([r['source_weight'] for r in rows],device=device,dtype=torch.float32)
    with torch.no_grad():
        reconstructed=torch.cat([head(inputs[i:i+16]).cpu() for i in range(0,len(rows),16)])
    if not torch.equal(reconstructed,feature['original_quality_logits']):
        raise ValueError('cached features do not exactly reproduce original quality-head outputs')
    settings=reg['head_training'];optimizer=torch.optim.AdamW(head.parameters(),lr=settings['learning_rate'],weight_decay=0.)
    rng=torch.Generator(device='cpu').manual_seed(settings['seed']);history=[];steps=0;started=time.monotonic()
    exposures=torch.zeros(len(rows),dtype=torch.int64);snapshots={}
    for epoch in range(1,settings['epochs']+1):
        order=torch.randperm(len(rows),generator=rng)
        for start in range(0,len(rows),settings['batch_size']):
            cpu_indices=order[start:start+settings['batch_size']]; indices=cpu_indices.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction=head(inputs[indices])[:,3]
            losses=torch.nn.functional.smooth_l1_loss(prediction,targets[indices],reduction='none')
            loss=(losses*weights[indices]).mean()*len(rows)/weights.sum()
            loss.backward();optimizer.step();steps+=1;exposures[cpu_indices]+=1
        with torch.no_grad():
            predicted=torch.cat([head(inputs[i:i+256])[:,3] for i in range(0,len(rows),256)])
            objective=float((torch.nn.functional.smooth_l1_loss(predicted,targets,reduction='none')*weights).sum()/weights.sum())
            accepted=predicted>0
            counts={str(t):dict(total=int((targets==t).sum()),positive_prediction=int(((targets==t)&accepted).sum())) for t in [-2,-1,0,1,2]}
        event=dict(epoch=epoch,optimizer_steps=steps,source_weighted_SmoothL1=objective,
            utility_target_counts=counts,seconds=time.monotonic()-started)
        history.append(event);write_json(output/'PROGRESS.json',event)
        if epoch in settings['snapshots']:
            path=output/f'quality_head_epoch{epoch}.pt';torch.save(head.state_dict(),path)
            snapshots[str(epoch)]=dict(path=str(path),sha256=file_hash(path),
                parameter_delta_squared=sum(float((v-initial[k]).double().square().sum()) for k,v in head.state_dict().items()),
                optimizer_steps=steps,complete_passes=epoch)
            print(json.dumps(event),flush=True)
    frozen_after=protected_hashes(model)
    if frozen_before!=frozen_after:raise ValueError('frozen content or scope parameters changed')
    if not bool((exposures==settings['epochs']).all()):raise ValueError('incomplete registered feature passes')
    torch.save(dict(optimizer=optimizer.state_dict(),torch_rng=torch.get_rng_state(),sampler_rng=rng.get_state(),
        exposures=exposures),output/'OPTIMIZER_AND_COVERAGE.pt')
    report=dict(schema='quality_only_signed_utility_v1',status='complete',train_rows=len(rows),
        train_sources=len({r['source_id'] for r in rows}),optimizer_steps=steps,complete_passes=settings['epochs'],
        minimum_row_visits=int(exposures.min()),maximum_row_visits=int(exposures.max()),
        source_target_counts=dict(Counter(r['utility_target'] for r in rows)),settings=settings,
        data_sha256=file_hash(data),data_manifest_sha256=file_hash(root/'data/UTILITY_DATA_FINAL.json'),
        features_sha256=file_hash(output/'TRAIN_FEATURES.pt'),cached_head_reproduction_exact=True,
        all_model_parameters_frozen_during_extraction=True,optimized_parameters='detached_existing_quality_head_copy_only',
        frozen_content_and_scope_parameters_before=frozen_before,frozen_content_and_scope_parameters_after=frozen_after,
        original_checkpoint=reg['old_editor'],snapshots=snapshots,history=history,training_seconds=time.monotonic()-started)
    write_json(output/'TRAINING_FINAL.json',report);(output/'_SUCCESS').touch()


def evaluation_features(root, model, tokenizer, device, reg):
    from crystal_dlm.post_refine_contract import fingerprint
    output=root/'evaluation_features';output.mkdir(parents=True,exist_ok=True)
    fit=root/'fit';plans=read_rows(fit/'cohort/plans.jsonl');currents=read_rows(fit/'current/inputs.jsonl')
    proposals=read_rows(fit/'proposal/inputs.jsonl');rows=[]
    for plan,current,proposal in zip(plans,currents,proposals,strict=True):
        i=plan['original_ordinal'];wrapper=json.loads((fit/f'proposal/records/{i:04d}.json').read_text());trace=wrapper['editor_trace']
        if not current.get('body_token_ids') or not proposal.get('body_token_ids'):continue
        rows.append(dict(pair_id=fingerprint(dict(source=plan['ancestor_id'],current=current['body_token_ids'],proposal=proposal['body_token_ids'])),
            ordinal=i,prompt=plan['body_prompt'],current_tokens=current['body_token_ids'],proposal_tokens=proposal['body_token_ids'],
            num_sites=plan['plan_state']['N'],action_positions=trace.get('action',{}).get('positions',[]),
            proposal_generated=trace.get('proposal_generated',False),old_applied=trace.get('applied',False),
            old_probability=trace.get('learned_accept_probability'),known_sun_guard=trace.get('learned_mode')==0 and
            trace.get('known_sun',False)))
    write_rows(output/'EVAL_ROWS.jsonl',rows)
    payload=feature_rows(model,tokenizer,rows,device,output,'EVAL')
    training=json.loads((root/'training/utility/result/TRAINING_FINAL.json').read_text())
    import torch
    values=payload['features'].to(device);head=copy.deepcopy(model.quality_head).to(device);predictions={}
    with torch.no_grad():
        for epoch,snapshot in training['snapshots'].items():
            if file_hash(snapshot['path'])!=snapshot['sha256']:raise ValueError('utility head snapshot changed')
            head.load_state_dict(torch.load(snapshot['path'],map_location=device,weights_only=True))
            predictions[epoch]=torch.cat([head(values[i:i+256])[:,3].cpu() for i in range(0,len(rows),256)]).tolist()
    result=[dict(ordinal=row['ordinal'],proposal_generated=row['proposal_generated'],old_applied=row['old_applied'],
        old_probability=row['old_probability'],original_quality_logit=float(payload['original_quality_logits'][i,3]),
        learned_utilities={epoch:v[i] for epoch,v in predictions.items()}) for i,row in enumerate(rows)]
    write_rows(output/'UTILITY_PREDICTIONS.jsonl',result)
    drift=[abs(float(payload['original_quality_logits'][i,3].sigmoid())-row['old_probability'])
           for i,row in enumerate(rows) if row['proposal_generated'] and row['old_probability'] is not None]
    report=dict(rows=len(rows),checkpoint=reg['old_editor'],checkpoint_receipt_sha256=file_hash(Path(reg['old_editor'])/'RSI_TRAINING_DONE.json'),
        proposal_inputs_sha256=file_hash(fit/'proposal/inputs.jsonl'),rows_sha256=file_hash(output/'EVAL_ROWS.jsonl'),
        predictions_sha256=file_hash(output/'UTILITY_PREDICTIONS.jsonl'),features_sha256=file_hash(output/'EVAL_FEATURES.pt'),
        old_probability_max_abs_drift=max(drift,default=0.),no_endpoint_labels_loaded=True,
        training_receipt_sha256=file_hash(root/'training/utility/result/TRAINING_FINAL.json'))
    write_json(output/'FEATURES_FINAL.json',report);print(json.dumps(report),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--mode',choices=['train','evaluate'],default='train');args=parser.parse_args()
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import torch
    from crystal_dlm.expert_edit import load_editor_model
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():raise RuntimeError('utility work requires a Slurm GPU')
    torch.cuda.set_device(0);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    root=args.root;reg=json.loads((root/'PREREGISTRATION.json').read_text());torch.manual_seed(reg['head_training']['seed'])
    spec=json.loads((root/'fit/RUN_SPEC.json').read_text());device=torch.device('cuda',0)
    model,tokenizer=load_editor_model(spec['assets']['base_model'],reg['old_editor'],device)
    model.eval()
    for parameter in model.parameters():parameter.requires_grad_(False)
    if args.mode=='train':train(root,model,tokenizer,device,reg)
    else:evaluation_features(root,model,tokenizer,device,reg)


if __name__=='__main__':main()
