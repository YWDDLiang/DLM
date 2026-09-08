#!/usr/bin/env python3
"""Update existing G/E weights from isolated, bound online SUN preferences."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.run_post_refine_cycle import read_rows,file_hash,write_json


def tensor_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().view(__import__('torch').uint8).numpy().tobytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,type=Path)
    args=parser.parse_args()
    spec=json.loads(args.config.read_text())
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import torch
    import torch.distributed as dist
    from crystal_dlm.r03_physics_transfer import build_repair_constraints
    from crystal_dlm.rsi_preference import conditional_logp,forward_view,numeric_order
    from crystal_dlm.expert_edit import load_editor_model,inference_view,materialize_edit_batch
    from scripts.sample_llada_dynamic_crystals import load_model_and_tokenizer
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('training requires a Slurm GPU allocation')
    rank=int(os.environ.get('RANK','0')); world=int(os.environ.get('WORLD_SIZE','1'))
    local=int(os.environ.get('LOCAL_RANK','0')); device=torch.device('cuda',local)
    torch.cuda.set_device(local); torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if world>1: dist.init_process_group('nccl')
    torch.manual_seed(spec['seed']); random.seed(spec['seed'])
    branch=spec['branch']; output=Path(spec['output_dir'])
    input_path=Path(spec['data'])
    manifest=json.loads(input_path.with_name('PAIRS_FINAL.json').read_text())
    if file_hash(input_path)!=manifest['files_sha256'][input_path.name] or manifest['source_split']!='train':
        raise ValueError('training preferences are not registered training-only data')
    examples=read_rows(input_path)
    if not examples or any(r['source_split']!='train' for r in examples):
        raise ValueError('empty or contaminated training preferences')
    replay=[];replay_pins=[]
    for value in spec.get('replay_data',[]):
        path=Path(value);report_path=path.with_name('PAIRS_FINAL.json')
        report=json.loads(report_path.read_text())
        if report['source_split']!='train' or report['files_sha256'][path.name]!=file_hash(path):
            raise ValueError('historical replay preferences are not bound TRAIN data')
        rows=read_rows(path)
        if any(r['source_split']!='train' for r in rows): raise ValueError('heldout replay is forbidden')
        replay.extend(rows);replay_pins.append({'path':str(path),'sha256':file_hash(path),
                                             'manifest_sha256':file_hash(report_path),'examples':len(rows)})
    if branch=='G':
        model,tokenizer=load_model_and_tokenizer(spec['base_model'],spec['checkpoint'],device,mean_resizing=False)
        for name,param in model.named_parameters():
            param.requires_grad_('.lora_A.' in name or '.lora_B.' in name)
    else:
        model,tokenizer=load_editor_model(spec['base_model'],spec['checkpoint'],device,trainable=True)
        model.training_modes={**model.training_modes,'S':[0,3]}
    for module in model.modules():
        if isinstance(module,torch.nn.Dropout): module.p=0.
    selected=[(name,param) for name,param in model.named_parameters() if param.requires_grad]
    if not selected: raise ValueError('no existing trainable parameters')
    for _,param in selected: param.data=param.data.float()
    # A frozen copy of just the trainable state implements the previous-policy
    # reference on the same unchanged backbone. Copies happen before graphs.
    reference={name:param.detach().clone() for name,param in selected}
    original_tables={name:tensor_hash(layer.weight) for name,layer in
                     [('input',model.get_input_embeddings()),('output',model.get_output_embeddings())]}
    if model.get_input_embeddings().weight.requires_grad or model.get_output_embeddings().weight.requires_grad:
        raise ValueError('B0 IO tables must remain frozen')
    base=model.base_model if branch=='E' else model
    # The frozen LLaDA implementation exposes block checkpointing directly.
    checkpoint_modules=[]
    for name,module in base.named_modules():
        setter=getattr(module,'set_activation_checkpointing',None)
        if callable(setter) and hasattr(module,'transformer'):
            setter('whole_layer'); checkpoint_modules.append(name)
    if not checkpoint_modules: raise RuntimeError('native LLaDA activation checkpointing unavailable')
    base.enable_input_require_grads()
    model.eval()
    support=build_repair_constraints(tokenizer)
    optimizer=torch.optim.AdamW([
        {'params':[p for n,p in selected if 'lora_' in n],'lr':spec['learning_rate']},
        {'params':[p for n,p in selected if 'lora_' not in n],'lr':spec['head_learning_rate']}],weight_decay=0.)
    if rank==0: output.mkdir(parents=True,exist_ok=False)
    if world>1: dist.barrier()
    contract={**spec,'data_sha256':file_hash(input_path),'pair_manifest_sha256':file_hash(input_path.with_name('PAIRS_FINAL.json')),
              'objective':'one_shared_prefix_cut_conditional_DPO_surrogate',
              'hard_support':'unchanged_K8_periodic_and_cell_support',
              'trainable_parameters':sum(p.numel() for _,p in selected),'world_size':world,
              'reference':'previous_round_trainable_state_on_identical_frozen_backbone',
              'original_IO_sha256':original_tables,'historical_replay':replay_pins,
              'replay_probability':.25 if replay else 0.}
    if rank==0: write_json(output/'TRAIN_CONFIG.json',contract)
    started=time.monotonic(); steps=0; total_pairs=0; total_heads=0; history=[]
    rng=random.Random(spec['seed']+rank)
    for step in range(spec['steps']):
        optimizer.zero_grad(set_to_none=True)
        losses=[]; margins=[]; pair_count=0; head_count=0
        for micro in range(spec['accumulation']):
            pool=replay if replay and rng.random()<.25 else examples
            example=pool[rng.randrange(len(pool))]
            cut=rng.randrange(len(numeric_order(example['num_sites'],branch)))
            chosen=example.get('chosen_tokens'); rejected=example.get('rejected_tokens')
            if chosen is not None and rejected is not None:
                live={name:param.detach().clone() for name,param in selected}
                with torch.no_grad():
                    for name,param in selected: param.copy_(reference[name])
                    ref_w=conditional_logp(model,tokenizer,example,chosen,cut,branch,support).detach()
                    ref_l=conditional_logp(model,tokenizer,example,rejected,cut,branch,support).detach()
                    for name,param in selected: param.copy_(live[name])
                del live
                win=conditional_logp(model,tokenizer,example,chosen,cut,branch,support)
                lose=conditional_logp(model,tokenizer,example,rejected,cut,branch,support)
                margin=(win-ref_w)-(lose-ref_l)
                if step==0 and micro==0 and abs(float(margin.detach()))>1e-5:
                    raise ValueError('step-zero policy/reference are not identical')
                loss=-torch.nn.functional.logsigmoid(spec['beta']*margin)-spec['anchor_weight']*win
                (loss/spec['accumulation']).backward()
                losses.append(float(loss.detach())); margins.append(float(margin.detach())); pair_count+=1
            elif branch=='G' and example.get('healthy_anchor_tokens') is not None:
                win=conditional_logp(model,tokenizer,example,example['healthy_anchor_tokens'],cut,branch,support)
                loss=-spec['anchor_weight']*win
                (loss/spec['accumulation']).backward(); losses.append(float(loss.detach()))
            if branch=='E' and example.get('mode_target') is not None:
                current=example['current_tokens']; n=example['num_sites']
                out,_=forward_view(model,tokenizer,example['prompt'],current,current,n,'E')
                # Unshifted logits learn retention; the same explicit SUN prior
                # is applied only to the runtime action decision.
                target=torch.tensor([example['mode_target']],device=device)
                mode=torch.nn.functional.cross_entropy(out.mode_logits[:,[0,3]],target)
                weight=2. if example.get('known_sun') else 1.
                loss=weight*mode
                proposal=example.get('proposal_tokens')
                if proposal is not None and example.get('accept_target') is not None:
                    judge,_=forward_view(model,tokenizer,example['prompt'],current,proposal,n,'E',numeric_order(n,'E'),1.)
                    loss=loss+weight*torch.nn.functional.binary_cross_entropy_with_logits(judge.quality_logits[0,3],
                                            torch.tensor(float(example['accept_target']),device=device))
                (loss/spec['accumulation']).backward();losses.append(float(loss.detach()));head_count+=1
        if world>1:
            for _,param in selected:
                if param.grad is None: param.grad=torch.zeros_like(param)
                dist.all_reduce(param.grad); param.grad.div_(world)
        grad=torch.nn.utils.clip_grad_norm_([p for _,p in selected],1.,error_if_nonfinite=True)
        if float(grad)==0: raise ValueError('no effective training gradient in optimizer step')
        optimizer.step();steps+=1;total_pairs+=pair_count;total_heads+=head_count
        event={'step':steps,'mean_loss':sum(losses)/len(losses) if losses else None,
            'mean_preference_margin':sum(margins)/len(margins) if margins else None,
            'gradient_norm':float(grad),'pairs':pair_count,'head_examples':head_count,
            'seconds':time.monotonic()-started,'peak_GPU_GB':torch.cuda.max_memory_allocated()/1e9}
        history.append(event)
        if rank==0 and (steps%4==0 or steps==1):
            write_json(output/'PROGRESS.json',event);print(json.dumps(event),flush=True)
        stop=torch.tensor(int(time.monotonic()-started>spec['max_training_seconds']),device=device)
        if world>1: dist.all_reduce(stop,op=dist.ReduceOp.MAX)
        if bool(stop): break
    if world>1: dist.barrier()
    if rank==0:
        delta=sum(float((p.detach()-reference[n]).square().sum()) for n,p in selected)
        if not delta>0: raise ValueError('optimizer did not change model weights')
        if any(tensor_hash(layer.weight)!=original_tables[name] for name,layer in
               [('input',model.get_input_embeddings()),('output',model.get_output_embeddings())]):
            raise ValueError('training changed frozen B0 IO tables')
        model.eval(); checkpoint=output/'checkpoint'
        model.save_pretrained(checkpoint,safe_serialization=True,save_embedding_layers=False)
        tokenizer.save_pretrained(checkpoint)
        if branch=='E':
            row=examples[0];prefix=tokenizer(row['prompt'],add_special_tokens=False)['input_ids']
            probe=[inference_view(prefix,row['current_tokens'],row['current_tokens'],row['num_sites'],1)]
            batch=materialize_edit_batch(probe,tokenizer,device)
            with torch.no_grad():
                actual=model(batch['input_ids'],attention_mask=batch['attention_mask'],edit_context=batch['edit_context'])
            torch.save({'examples':probe,**{name:getattr(actual,name).cpu() for name in
                        ('logits','mode_logits','site_logits','count_logits','quality_logits')}},checkpoint/'roundtrip_probe.pt')
        files={p.name:file_hash(p) for p in checkpoint.iterdir() if p.is_file()}
        receipt={'optimizer_steps':steps,'parameter_delta_squared':delta,'local_preference_examples':total_pairs,
            'local_decision_examples':total_heads,'training_seconds':time.monotonic()-started,
            'checkpoint_files':files,'contract':contract,'loss_history':history}
        write_json(checkpoint/'RSI_TRAINING_DONE.json',receipt)
        write_json(output/'TRAINING_FINAL.json',receipt)
        (output/'_SUCCESS').touch()
    if world>1: dist.barrier();dist.destroy_process_group()


if __name__=='__main__': main()
