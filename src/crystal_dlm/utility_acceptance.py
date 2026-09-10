"""Canonical completed-proposal judgement and lossless continuous decisions."""
from __future__ import annotations
import copy
import math


def judge_completed_proposals(model,tokenizer,requests,*,batch_size=16):
    """Use the same ordered complete views for the old and learned quality heads."""
    import torch
    from crystal_dlm.expert_edit import inference_view,materialize_edit_batch
    if batch_size!=16:raise ValueError('this registered utility head requires canonical batches of 16')
    ordinals=[r['ordinal'] for r in requests]
    if ordinals!=sorted(set(ordinals)):raise ValueError('canonical judgement needs unique original ordinal order')
    device=next(model.parameters()).device;values={}
    valid=[r for r in requests if r.get('current_tokens') and r.get('proposal_tokens')]
    model.eval()
    with torch.no_grad():
        for start in range(0,len(valid),batch_size):
            rows=valid[start:start+batch_size]
            examples=[inference_view(tokenizer(r['prompt'],add_special_tokens=False)['input_ids'],r['current_tokens'],
                r['proposal_tokens'],r['num_sites'],1,r['action_positions'],remaining=80,reveal=1.) for r in rows]
            batch=materialize_edit_batch(examples,tokenizer,device)
            actual=model(batch['input_ids'],attention_mask=batch['attention_mask'],edit_context=batch['edit_context'])
            for row,value in zip(rows,actual.quality_logits[:,3].float().tolist(),strict=True):
                if not math.isfinite(value):raise ValueError('nonfinite learned utility')
                values[row['ordinal']]=value
    return values


def accept_utility(raw,margin,*,proposal_generated,known_sun_guard=False):
    if not isinstance(margin,(int,float)) or not math.isfinite(margin):raise ValueError('finite utility margin required')
    if raw is None:return False
    if not isinstance(raw,(int,float)) or not math.isfinite(raw):raise ValueError('finite learned utility required')
    return bool(proposal_generated and not known_sun_guard and raw>=margin)


def continuous_decision(native_wrapper,current_tokens,trace,inverse,raw_utility,margin):
    from crystal_dlm.continuous_keep_edit import commit_patch
    guarded=bool(trace.get('known_sun') and trace.get('learned_mode')==0)
    accepted=accept_utility(raw_utility,margin,proposal_generated=trace.get('proposal_generated',False),known_sun_guard=guarded)
    if not accepted:
        return copy.deepcopy(native_wrapper['record']),dict(learned_accept=False,actual_edit=False,
            raw_utility=raw_utility,margin=margin,reason='learned_KEEP_or_original_SUN_guard')
    result,commit=commit_patch(native_wrapper['record'],current_tokens,trace['proposal_tokens'],inverse,
        editable=native_wrapper['continuous_trace']['editable'])
    return result,dict(learned_accept=True,actual_edit=commit['applied'],raw_utility=raw_utility,margin=margin,commit=commit)
