"""Shared hard-support sampler and prefix-denoising preference surrogate.

The loss is a sampled conditional preference surrogate, not the marginal
likelihood of the complete diffusion trajectory. Support is not learned.
"""
from __future__ import annotations
import math
import torch
from crystal_dlm.fixed_slot import MASK_TOKEN_ID
from crystal_dlm.r03_physics_transfer import supported_scalar_logits, geometry_support_report
from crystal_dlm.llada_generation import _apply_lightweight_decoding_masks
from crystal_dlm.expert_edit import inference_view, materialize_edit_batch


def numeric_order(n, branch):
    cell=list(range(1,7))
    if branch=='G':
        return cell+[8+4*site+axis for axis in range(3) for site in range(n)]
    return cell+[8+4*site+axis for site in range(n) for axis in range(3)]


def legal_vector(vector, body, n, position, support):
    if position>=8:
        return supported_scalar_logits(vector,body,0,n,position,constraints=support)
    minimum=torch.finfo(vector.dtype).min
    family='length_token_to_bin' if position<4 else 'angle_token_to_bin'
    axis=('LA','LB','LC')[position-1] if position<4 else ('AA','AB','AG')[position-4]
    ids=list(support[family][axis])
    # The detached probe computes only the denominator support. No physical
    # force/energy tensor enters the differentiable policy graph.
    with torch.no_grad():
        probe=vector.new_full((1,len(body),vector.numel()),minimum)
        probe[0,position,ids]=0.
        canvas=torch.tensor([body],device=vector.device)
        active=torch.zeros_like(canvas,dtype=torch.bool)
        active[0,position]=True
        _apply_lightweight_decoding_masks(probe,canvas,0,len(body),support,active,MASK_TOKEN_ID)
        legal=torch.isfinite(probe[0,position]) & (probe[0,position]>minimum)
    return vector.masked_fill(~legal,minimum),{'available':bool(legal.any()),'legal_count':int(legal.sum())}


def forward_view(model,tokenizer,prompt,old,current,n,branch,active=(),reveal=0.):
    device=next(model.parameters()).device
    prefix=tokenizer(prompt,add_special_tokens=False)['input_ids']
    if branch=='G':
        ids=torch.tensor([prefix+list(current)],device=device)
        return model(ids,attention_mask=torch.ones_like(ids)),len(prefix)
    example=inference_view(prefix,old,current,n,1,active,remaining=80,reveal=reveal)
    batch=materialize_edit_batch([example],tokenizer,device)
    return model(batch['input_ids'],attention_mask=batch['attention_mask'],edit_context=batch['edit_context']),len(prefix)


def conditional_logp(model,tokenizer,example,target,cut,branch,support):
    n=example['num_sites']
    order=example.get('action_positions',numeric_order(n,branch)) if branch=='E' else numeric_order(n,branch)
    if branch=='E' and 'action_positions' in example:
        from crystal_dlm.ranked_feedback import validate_action_target
        validate_action_target(example['current_tokens'],target,order)
    if not order or not 0 <= cut < len(order): raise ValueError('invalid conditional mask cut')
    current=list(target)
    for position in order[cut:]: current[position]=MASK_TOKEN_ID
    output,prefix=forward_view(model,tokenizer,example['prompt'],example.get('current_tokens',target),
                                current,n,branch,order,reveal=cut/len(order))
    position=order[cut]
    vector,report=legal_vector(output.logits[0,prefix+position].float(),current,n,position,support)
    if not report['available'] or vector[int(target[position])]<=torch.finfo(vector.dtype).min:
        raise ValueError('preference target absent from unchanged hard support')
    return vector.log_softmax(-1)[int(target[position])]


@torch.no_grad()
def propose_ranked_editor(model,tokenizer,*,prompt,body,n,support,seed,known_sun=False,
                          force_proposal=False,keep_prior=9.):
    """One learned adaptive action, then one learned accept decision."""
    from crystal_dlm.ranked_feedback import action_positions, COUNTS, MODES
    before=list(body)
    inspect,_=forward_view(model,tokenizer,prompt,before,before,n,'E')
    logits=inspect.mode_logits[0].float().clone()
    if known_sun: logits[0]+=math.log(keep_prior)
    selected=int(logits.argmax())
    # Non-SUN current states always receive a proposal before the final judge.
    mode=selected if selected else int(logits[1:].argmax())+1
    result=dict(current_tokens=before,proposal_tokens=before,final_tokens=before,
        known_sun=known_sun,learned_mode=selected,mode_logits=logits.tolist(),
        proposal_generated=False,forward_calls=1,applied=False,
        origin='forced_training_proposal' if force_proposal else 'current_policy',
        counterfactual_training_proposal=bool(force_proposal and selected==0))
    if known_sun and selected==0 and not force_proposal:
        result.update(action=dict(mode=0,name=MODES[0],sites=[],positions=[]),learned_decision='KEEP')
        return result
    if mode==1:
        allowed=[i for i,count in enumerate(COUNTS) if count<=n]
        count=COUNTS[allowed[int(inspect.count_logits[0,allowed].argmax())]]
        sites=sorted(inspect.site_logits[0,:n].topk(count).indices.tolist())
    else: sites=list(range(n))
    order=action_positions(n,mode,sites)
    result['action']=dict(mode=mode,name=MODES[mode],sites=sites,positions=order)
    candidate=before.copy()
    for position in order: candidate[position]=MASK_TOKEN_ID
    generator=torch.Generator(device=next(model.parameters()).device).manual_seed(seed)
    for offset,position in enumerate(order):
        output,prefix=forward_view(model,tokenizer,prompt,before,candidate,n,'E',order,offset/len(order))
        result['forward_calls']+=1
        vector,report=legal_vector(output.logits[0,prefix+position].float(),candidate,n,position,support)
        if not report['available']:
            result.update(proposal_failure='empty_hard_support',learned_decision='KEEP')
            return result
        candidate[position]=int(torch.multinomial((vector/.7).softmax(-1),1,generator=generator))
    report=geometry_support_report(candidate,constraints=support)
    if not report['supported']:
        result.update(proposal_failure=report,learned_decision='KEEP')
        return result
    judge,_=forward_view(model,tokenizer,prompt,before,candidate,n,'E',order,1.)
    accept=float(judge.quality_logits[0,3].sigmoid())
    applied=accept>=.5 and not (known_sun and selected==0)
    result.update(proposal_generated=True,proposal_tokens=candidate,final_tokens=candidate if applied else before,
                  learned_accept_probability=accept,applied=applied,forward_calls=result['forward_calls']+1,
                  learned_decision='EDIT' if applied else 'KEEP')
    if result['forward_calls']>80: raise RuntimeError('editor forward budget exceeded')
    return result


@torch.no_grad()
def propose_editor(model,tokenizer,*,prompt,body,n,support,seed,known_sun=False,force_proposal=False,keep_prior=9.):
    before=list(body)
    inspect,_=forward_view(model,tokenizer,prompt,before,before,n,'E')
    scores=inspect.mode_logits[0,[0,3]].float().clone()
    if known_sun: scores[0]+=math.log(keep_prior)
    chosen_mode=int(scores.argmax())
    result={'current_tokens':before,'proposal_tokens':before,'final_tokens':before,
            'known_sun':known_sun,'learned_decision':'KEEP' if chosen_mode==0 else 'EDIT',
            'mode_logits_keep_edit':scores.tolist(),'proposal_generated':False,'forward_calls':1,
            'counterfactual_training_proposal':force_proposal and chosen_mode==0}
    if chosen_mode==0 and not force_proposal:
        return result
    order=numeric_order(n,'E')
    candidate=before.copy()
    for position in order: candidate[position]=MASK_TOKEN_ID
    generator=torch.Generator(device=next(model.parameters()).device).manual_seed(seed)
    for offset,position in enumerate(order):
        output,prefix=forward_view(model,tokenizer,prompt,before,candidate,n,'E',order,offset/len(order))
        result['forward_calls']+=1
        vector,report=legal_vector(output.logits[0,prefix+position].float(),candidate,n,position,support)
        if not report['available']:
            result['proposal_failure']='empty_hard_support'
            return result
        candidate[position]=int(torch.multinomial((vector/.7).softmax(-1),1,generator=generator))
    report=geometry_support_report(candidate,constraints=support)
    if not report['supported']:
        result['proposal_failure']=report
        return result
    output,_=forward_view(model,tokenizer,prompt,before,candidate,n,'E',order,1.)
    result['forward_calls']+=1
    accept=float(output.quality_logits[0,3].sigmoid())
    applied=chosen_mode==1 and accept>=.5
    result.update(proposal_generated=True,proposal_tokens=candidate,
                  final_tokens=candidate if applied else before,learned_accept_probability=accept,
                  applied=applied)
    if result['forward_calls']>80: raise RuntimeError('editor forward budget exceeded')
    return result
