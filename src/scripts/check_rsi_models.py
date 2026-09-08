#!/usr/bin/env python3
"""Actual G/E backward compatibility and memory canary, without weight updates."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.run_post_refine_cycle import read_rows,write_json,load_config,fingerprint


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();spec=load_config(args.config)
    import torch
    from crystal_dlm.expert_edit import load_editor_model
    from crystal_dlm.rsi_preference import conditional_logp,forward_view,numeric_order
    from crystal_dlm.r03_physics_transfer import build_repair_constraints
    from crystal_dlm.state_training import enable_native_checkpointing
    from scripts.sample_llada_dynamic_crystals import load_model_and_tokenizer
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available(): raise RuntimeError('Slurm GPU required')
    rank=int(os.environ.get('LOCAL_RANK','0'));torch.cuda.set_device(rank);torch.set_num_threads(1)
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';torch.use_deterministic_algorithms(True)
    device=torch.device('cuda',rank);branch='G' if rank==0 else 'E';started=time.monotonic()
    if branch=='G':
        model,tokenizer=load_model_and_tokenizer(spec['assets']['base_model'],spec['assets']['b0_checkpoint'],device,mean_resizing=False)
        for name,param in model.named_parameters(): param.requires_grad_('.lora_A.' in name or '.lora_B.' in name)
        base=model
    else:
        model,tokenizer=load_editor_model(spec['assets']['base_model'],spec['assets']['editor_reference'],device,trainable=True)
        base=model.base_model
    for module in model.modules():
        if isinstance(module,torch.nn.Dropout): module.p=0.
    for param in model.parameters():
        if param.requires_grad: param.data=param.data.float()
    enabled=enable_native_checkpointing(base);base.enable_input_require_grads();model.eval()
    if not enabled: raise RuntimeError('missing native checkpointing')
    rows=read_rows(Path(spec['run_root'])/'fit/construction/inputs.jsonl')
    row=max((r for r in rows if r.get('body_token_ids')),key=lambda r:len(r['body_token_ids'])+len(r['body_prompt']))
    body=row['body_token_ids'];n=(len(body)-7)//4
    example={'prompt':row['body_prompt'],'num_sites':n,'current_tokens':body}
    support=build_repair_constraints(tokenizer);cut=len(numeric_order(n,branch))//2
    with torch.no_grad(): ref=conditional_logp(model,tokenizer,example,body,cut,branch,support).detach()
    win=conditional_logp(model,tokenizer,example,body,cut,branch,support)
    loss=-.2*win
    if abs(float((win-ref).detach()))>1e-5: raise RuntimeError('policy/reference logits disagree at initialization')
    loss.backward()
    if branch=='E':
        out,_=forward_view(model,tokenizer,example['prompt'],body,body,n,'E')
        # This arbitrary target checks only that the mode head has a gradient.
        # It is not a material preference label and no optimizer is constructed.
        torch.nn.functional.cross_entropy(out.mode_logits[:,[0,3]],torch.tensor([0],device=device)).backward()
    norms={}
    for name,param in model.named_parameters():
        if param.grad is None: continue
        if not torch.isfinite(param.grad).all(): raise RuntimeError('nonfinite actual model gradient')
        key='lora' if 'lora_' in name else 'mode' if 'mode_head' in name else 'conditioning'
        norms[key]=norms.get(key,0.)+float(param.grad.float().square().sum())
    if norms.get('lora',0)<=0 or branch=='E' and norms.get('mode',0)<=0: raise RuntimeError('missing actual policy/head gradient')
    write_json(Path(args.output)/f'CHECK_DONE_{rank}.json',{'branch':branch,'gradient_squared':norms,
        'sample_identity':fingerprint(row),'num_sites':n,'seconds':time.monotonic()-started,
        'GPU_peak_GB':torch.cuda.max_memory_allocated()/1e9,'optimizer_steps':0,'weights_updated':False,
        'physical_reward_claimed':False,'initial_reference_logp':float(ref),'policy_logp':float(win.detach())})


if __name__=='__main__': main()
