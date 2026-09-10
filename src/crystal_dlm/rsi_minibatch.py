"""Bounded data passes and batched conditional updates for ranked RSI."""
import math
import random
import time
from collections import Counter

import torch

from crystal_dlm.fixed_slot import MASK_TOKEN_ID
from crystal_dlm.rsi_preference import legal_vector, numeric_order
from crystal_dlm.expert_edit import inference_view, materialize_edit_batch


def epoch_indices(size, *, batch_size, world, epochs, seed):
    """Every record once per epoch, with at most one padding repeat per record."""
    if min(size, batch_size, world, epochs) < 1:
        raise ValueError('positive data and batch dimensions required')
    rng = random.Random(seed)
    width = batch_size * world
    for epoch in range(epochs):
        indices = list(range(size))
        rng.shuffle(indices)
        # Pad only to world size, not to a full large minibatch. The final
        # minibatch is smaller; increasing capacity must not multiply exposure.
        padding = (-len(indices)) % world
        # Small datasets may need more than one padding copy; expose this in receipts.
        extra = []
        while len(extra) < padding:
            copy = list(range(size)); rng.shuffle(copy); extra.extend(copy)
        indices.extend(extra[:padding])
        for start in range(0, len(indices), width):
            yield epoch, indices[start:start + width]


def training_view(example, target, cut, branch, *, mask_seed=0):
    n = example['num_sites']
    if branch == 'G' and example.get('plan_state'):
        groups = example['generation_groups']
        expected=set(numeric_order(n,'G'))
        positions=[pos for group in groups for pos in group]
        if len(positions)!=len(expected) or set(positions)!=expected:
            raise ValueError('registered native groups changed numeric support')
        rng = random.Random(mask_seed)
        # Native semantic groups stay in order; cover possible reveal subsets
        # within each confidence-remasked group instead of one site ordering.
        order = []
        for positions in groups:
            positions = list(positions); rng.shuffle(positions); order.extend(positions)
    else:
        order = example.get('action_positions', numeric_order(n, branch)) if branch == 'E' else numeric_order(n, branch)
    if not 0 <= cut < len(order):
        raise ValueError('invalid mask cut')
    if branch == 'E':
        from crystal_dlm.ranked_feedback import validate_action_target
        validate_action_target(example['current_tokens'], target, order)
    current = list(target)
    for pos in order[cut:]: current[pos] = MASK_TOKEN_ID
    return {'example': example, 'target': target, 'current': current,
            'position': order[cut], 'order': order, 'reveal': cut/len(order)}


def conditional_batch(model, tokenizer, views, branch, support):
    """One transformer forward for multiple independent conditional views."""
    device = next(model.parameters()).device
    prefixes = [tokenizer(v['example']['prompt'], add_special_tokens=False)['input_ids'] for v in views]
    if branch == 'G':
        width = max(len(p) + len(v['current']) for p, v in zip(prefixes, views))
        ids = torch.full((len(views), width), int(tokenizer.pad_token_id), device=device, dtype=torch.long)
        attention = torch.zeros_like(ids)
        for i, (prefix, view) in enumerate(zip(prefixes, views)):
            body = prefix + view['current']; ids[i, :len(body)] = torch.tensor(body, device=device)
            attention[i, :len(body)] = 1
        output = model(ids, attention_mask=attention)
    else:
        examples = [inference_view(prefix, v['example']['current_tokens'], v['current'],
                    v['example']['num_sites'], 1, v['order'], remaining=80, reveal=v['reveal'])
                    for prefix, v in zip(prefixes, views)]
        batch = materialize_edit_batch(examples, tokenizer, device)
        output = model(batch['input_ids'], attention_mask=batch['attention_mask'], edit_context=batch['edit_context'])
    values, distributions = [], []
    for i, (prefix, view) in enumerate(zip(prefixes, views)):
        pos = view['position']; target = int(view['target'][pos])
        vector, report = legal_vector(output.logits[i, len(prefix) + pos].float(),
                                      view['current'], view['example']['num_sites'], pos, support)
        minimum = torch.finfo(vector.dtype).min
        if not report['available'] or vector[target] <= minimum:
            raise ValueError('minibatch target absent from unchanged hard support')
        legal = (vector.detach() > minimum).nonzero().flatten()
        logp = (vector[legal]/.7).log_softmax(-1)
        values.append(logp[(legal == target).nonzero().item()])
        distributions.append(logp)
    return torch.stack(values), distributions


def editor_head_loss(model, tokenizer, examples, device, *, detach_content=False):
    from crystal_dlm.ranked_feedback import COUNTS
    chosen = [x for x in examples if has_head_supervision(x)]
    if not chosen: return None, 0
    views, modes, judges = [], [], []
    for row in chosen:
        prefix = tokenizer(row['prompt'], add_special_tokens=False)['input_ids']
        if row.get('mode_target') is not None:
            modes.append(len(views))
            views.append(inference_view(prefix, row['current_tokens'], row['current_tokens'], row['num_sites'], 1))
        else: modes.append(None)
        if row.get('proposal_tokens') is not None and row.get('accept_target') is not None:
            judges.append(len(views))
            views.append(inference_view(prefix, row['current_tokens'], row['proposal_tokens'], row['num_sites'],
                                        1, row.get('action_positions', []), remaining=80, reveal=1.))
        else: judges.append(None)
    batch = materialize_edit_batch(views, tokenizer, device)
    kwargs = {'detach_head_features': True} if detach_content else {}
    out = model(batch['input_ids'], attention_mask=batch['attention_mask'], edit_context=batch['edit_context'], **kwargs)
    losses = []
    for row, i, j in zip(chosen, modes, judges):
        weight = 2. if row.get('known_sun') else 1.
        loss = out.quality_logits.new_zeros(())
        if i is not None:
            loss = weight * torch.nn.functional.cross_entropy(out.mode_logits[i:i+1], torch.tensor([row['mode_target']], device=device))
        if row['mode_target'] == 1:
            sites = torch.tensor(row['site_targets'], device=device, dtype=torch.float32)
            loss = loss + torch.nn.functional.binary_cross_entropy_with_logits(out.site_logits[i, :row['num_sites']], sites)
            loss = loss + torch.nn.functional.cross_entropy(out.count_logits[i:i+1], torch.tensor([COUNTS.index(int(sites.sum()))], device=device))
        if j is not None:
            loss = loss + weight * torch.nn.functional.binary_cross_entropy_with_logits(out.quality_logits[j, 3],
                                                    torch.tensor(float(row['accept_target']), device=device))
        losses.append(loss)
    return torch.stack(losses).sum()/len(examples), len(chosen)


def has_head_supervision(row):
    return (row.get('mode_target') is not None or
            (row.get('proposal_tokens') is not None and row.get('accept_target') is not None))


def decision_head_parameter(name):
    return name.split('.', 1)[0] in ('mode_head', 'site_head', 'count_head', 'quality_head')


def train_bounded(model, tokenizer, examples, spec, selected, reference, optimizer, support, output, write_json):
    import torch.distributed as dist
    device = next(model.parameters()).device
    world = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    branch = spec['branch']; batch_size = spec['batch_size']; rng = random.Random(spec['seed'] + rank)
    started = time.monotonic(); history = []; exposures = Counter(); pairs = heads = 0
    content_exposures, head_exposures = Counter(), Counter()
    kl_stop = False; kl_trigger = None; content_frozen = False
    content_steps = head_steps = 0
    head_parameters = [(n,p) for n,p in selected if decision_head_parameter(n)]
    content_parameters = [(n,p) for n,p in selected if not decision_head_parameter(n)]
    freeze_after_kl = branch == 'E' and spec.get('continue_heads_after_content_KL', False)
    if freeze_after_kl and not head_parameters:
        raise ValueError('head continuation requires the existing editor decision heads')
    for epoch, indices in epoch_indices(len(examples), batch_size=batch_size, world=world,
                                       epochs=spec['epochs'], seed=spec['seed']):
        local_size=len(indices)//world
        rows = [examples[i] for i in indices[rank*local_size:(rank+1)*local_size]]
        optimizer.zero_grad(set_to_none=True)
        views, pair_slots, anchor_slots, content_rows = [], [], [], []
        for row in rows:
            exposures[row['pair_id']] += 1
            if content_frozen: continue
            order = row.get('action_positions', numeric_order(row['num_sites'], branch)) if branch == 'E' else numeric_order(row['num_sites'], branch)
            if not order: continue
            first_view = len(views)
            for cut in rng.sample(range(len(order)), min(spec['mask_cuts'], len(order))):
                seed = rng.randrange(2**63)
                if row.get('chosen_tokens') is not None and row.get('rejected_tokens') is not None:
                    pair_slots.append((len(views), len(views)+1))
                    views.extend(training_view(row, row[k], cut, branch, mask_seed=seed) for k in ('chosen_tokens', 'rejected_tokens'))
                else:
                    anchor = row.get('healthy_anchor_tokens') if branch == 'G' else row.get('content_target_tokens')
                    if anchor is not None:
                        anchor_slots.append(len(views)); views.append(training_view(row, anchor, cut, branch, mask_seed=seed))
            if len(views) > first_view: content_rows.append(row['pair_id'])
        loss = None; kl_value = None if content_frozen else 0.; margin_value = None
        if views:
            live = {n: p.detach().clone() for n, p in selected}
            with torch.no_grad():
                for n, p in selected: p.copy_(reference[n])
                reference_logp, reference_distribution = conditional_batch(model, tokenizer, views, branch, support)
                for n, p in selected: p.copy_(live[n])
            del live
            actual, distribution = conditional_batch(model, tokenizer, views, branch, support)
            terms = []
            margins = []
            for a, b in pair_slots:
                margin = (actual[a]-reference_logp[a])-(actual[b]-reference_logp[b]); margins.append(margin)
                terms.append(-torch.nn.functional.logsigmoid(spec['beta']*margin)-spec['anchor_weight']*actual[a])
            terms.extend(-spec['anchor_weight']*actual[i] for i in anchor_slots)
            kl = torch.stack([(q.exp()*(q-p)).sum() for q, p in zip(reference_distribution, distribution)]).mean()
            kl_value = float(kl.detach())
            loss = torch.stack(terms).sum()/(local_size*spec['mask_cuts']) + spec['reference_kl_weight']*kl
            if margins: margin_value = float(torch.stack(margins).mean().detach())
        if not content_frozen:
            # The same maximum is observed on every rank, including ranks with
            # no local content views, so the active parameter set stays aligned.
            observed = torch.tensor(kl_value, device=device)
            if world > 1: dist.all_reduce(observed, op=dist.ReduceOp.MAX)
            if float(observed) > spec['max_reference_kl']:
                kl_stop = True
                kl_trigger = {'after_optimizer_steps':len(history), 'epoch':epoch,
                              'maximum_rank_KL':float(observed), 'local_KL':kl_value}
                if not freeze_after_kl: break
                content_frozen = True; loss = None
                # grad=None is essential: zero gradients still let AdamW
                # momentum or weight decay change the content parameters.
                for _, p in content_parameters:
                    p.requires_grad_(False); p.grad = None
                if views:
                    if margins: del margin
                    del actual, distribution, reference_logp, reference_distribution, terms, margins, kl
        count = 0
        if branch == 'E':
            head_loss, count = editor_head_loss(model, tokenizer, rows, device)
            if head_loss is not None: loss = head_loss if loss is None else loss+head_loss
        supervised = torch.tensor(int(loss is not None), device=device)
        if world > 1: dist.all_reduce(supervised, op=dist.ReduceOp.SUM)
        if not bool(supervised): raise ValueError('minibatch contains no supervised examples')
        if loss is not None: loss.backward()
        active = head_parameters if content_frozen else selected
        if world > 1:
            for _, p in active:
                if p.grad is None: p.grad = torch.zeros_like(p)
                dist.all_reduce(p.grad); p.grad.div_(world)
        grad = torch.nn.utils.clip_grad_norm_([p for _, p in active], 1., error_if_nonfinite=True)
        if float(grad) == 0: raise ValueError('no effective minibatch gradient')
        optimizer.step(); heads += count
        if not content_frozen:
            pairs += len(pair_slots); content_exposures.update(content_rows)
            content_steps += 1
        if branch == 'E':
            head_exposures.update(row['pair_id'] for row in rows if has_head_supervision(row))
            head_steps += 1
        event = {'step': len(history)+1, 'epoch': epoch, 'mean_loss': float(loss.detach()) if loss is not None else None,
                 'mean_preference_margin': margin_value, 'reference_KL': kl_value,
                 'update_kind':'heads_only' if content_frozen else 'joint' if branch=='E' else 'content',
                 'content_optimizer_steps':content_steps, 'head_optimizer_steps':head_steps,
                 'content_frozen_by_KL':content_frozen,
                 'gradient_norm': float(grad), 'source_examples_per_GPU': local_size,
                 'source_batch_capacity_per_GPU': batch_size,
                 'conditional_forward_rows': len(views), 'seconds': time.monotonic()-started,
                 'peak_GPU_GB': torch.cuda.max_memory_allocated()/1e9 if device.type=='cuda' else 0.}
        history.append(event)
        if rank == 0: write_json(output/'PROGRESS.json', event); print(__import__('json').dumps(event), flush=True)
        stop = torch.tensor(int(time.monotonic()-started > spec['max_training_seconds']), device=device)
        if world > 1: dist.all_reduce(stop, op=dist.ReduceOp.MAX)
        if bool(stop): break
    write_json(output/f'EXPOSURE_rank{rank}.json', {'pair_visits': dict(exposures),
        'pair_visits_semantics':'inspected_batches_including_KL_trigger_batch',
        'content_updated_pair_visits':dict(content_exposures), 'head_updated_pair_visits':dict(head_exposures),
        'content_optimizer_steps':content_steps, 'head_optimizer_steps':head_steps,
        'sampling':'shuffled_complete_data_passes', 'batch_size':batch_size,
        'epochs_cap':spec['epochs'], 'KL_stop':kl_stop, 'KL_trigger':kl_trigger,
        'head_continuation_after_KL':content_frozen})
    if not history: raise ValueError('no bounded optimizer update completed')
    return len(history), pairs, heads, history, rng


@torch.no_grad()
def propose_ranked_batch(model, tokenizer, requests, *, support, batch_size=16, keep_prior=9.):
    """Batch independent E requests while preserving each action and RNG stream."""
    from crystal_dlm.ranked_feedback import COUNTS, MODES, action_positions
    from crystal_dlm.r03_physics_transfer import geometry_support_report
    if not 1 <= batch_size <= 64: raise ValueError('invalid editor batch size')
    device = next(model.parameters()).device
    results = []
    for start in range(0, len(requests), batch_size):
        group = requests[start:start+batch_size]; states = []
        for request in group:
            before = list(request['body']); seed = request['seed']
            result = dict(current_tokens=before, proposal_tokens=before, final_tokens=before,
                known_sun=request['known_sun'], proposal_generated=False, forward_calls=0, applied=False,
                sampling_seed=seed, sampling_temperature=.7, sampling_trace=[],
                scalar_logp_kind='actual_tempered_hard_support_token_sampling_conditional',
                origin='forced_training_proposal' if request.get('force_proposal') else 'current_policy')
            states.append(dict(request=request, result=result, stage='inspect', before=before,
                current=before.copy(), order=[], offset=0,
                prefix=tokenizer(request['prompt'], add_special_tokens=False)['input_ids'],
                generator=torch.Generator(device=device).manual_seed(seed)))
        while any(s['stage']!='done' for s in states):
            active = [s for s in states if s['stage']!='done']; views=[]
            for s in active:
                reveal = s['offset']/len(s['order']) if s['order'] else 0.
                views.append(inference_view(s['prefix'],s['before'],s['current'],s['request']['n'],1,
                                            s['order'],remaining=80,reveal=reveal))
            batch = materialize_edit_batch(views,tokenizer,device)
            out=model(batch['input_ids'],attention_mask=batch['attention_mask'],edit_context=batch['edit_context'])
            for i,s in enumerate(active):
                request,result=s['request'],s['result'];n=request['n'];result['forward_calls']+=1
                if s['stage']=='inspect':
                    logits=out.mode_logits[i].float().clone()
                    if request['known_sun']:logits[0]+=math.log(keep_prior)
                    selected=int(logits.argmax());mode=selected if selected else int(logits[1:].argmax())+1
                    result.update(learned_mode=selected,mode_logits=logits.tolist(),
                        unshifted_mode_logits=out.mode_logits[i].float().tolist(),
                        counterfactual_training_proposal=bool(request.get('force_proposal') and selected==0))
                    if request['known_sun'] and selected==0 and not request.get('force_proposal'):
                        result.update(action=dict(mode=0,name=MODES[0],sites=[],positions=[]),learned_decision='KEEP')
                        s['stage']='done';continue
                    exploration=request.get('exploration_local_rank')
                    if exploration is not None:
                        if type(exploration) is not int or not 0<=exploration<=3 or not request.get('force_proposal'):
                            raise ValueError('registered site exploration needs a forced proposal and rank 0..3')
                        mode=1
                        ranked=out.site_logits[i,:n].float().argsort(descending=True,stable=True).tolist()
                        sites=[ranked[exploration % n]]
                        result.update(scope_policy='registered_local_site_rank',exploration_local_rank=exploration,
                            available_site_ranks=n,site_logits=out.site_logits[i,:n].float().tolist(),
                            count_logits=out.count_logits[i].float().tolist(),site_rank_wrapped=exploration>=n)
                    elif mode==1:
                        allowed=[j for j,count in enumerate(COUNTS) if count<=n]
                        count=COUNTS[allowed[int(out.count_logits[i,allowed].argmax())]]
                        sites=sorted(out.site_logits[i,:n].topk(count).indices.tolist())
                    else:sites=list(range(n))
                    s['order']=action_positions(n,mode,sites)
                    result['action']=dict(mode=mode,name=MODES[mode],sites=sites,positions=s['order'])
                    for pos in s['order']:s['current'][pos]=MASK_TOKEN_ID
                    result['initial_masked_tokens']=s['current'].copy();s['stage']='fill'
                elif s['stage']=='fill':
                    pos=s['order'][s['offset']]
                    vector,report=legal_vector(out.logits[i,len(s['prefix'])+pos].float(),s['current'],n,pos,support)
                    if not report['available']:
                        result.update(proposal_failure='empty_hard_support',learned_decision='KEEP',sampled_attempt_tokens=s['current'].copy())
                        s['stage']='done';continue
                    probability=(vector/.7).softmax(-1)
                    token=int(torch.multinomial(probability,1,generator=s['generator']));s['current'][pos]=token
                    result['sampling_trace'].append(dict(position=pos,token_id=token,scalar_logp=float(probability[token].log())))
                    s['offset']+=1
                    if s['offset']==len(s['order']):
                        result['sampled_attempt_tokens']=s['current'].copy()
                        report=geometry_support_report(s['current'],constraints=support)
                        if not report['supported']:
                            result.update(proposal_failure=report,learned_decision='KEEP');s['stage']='done'
                        else:s['stage']='judge'
                else:
                    accept=float(out.quality_logits[i,3].sigmoid())
                    applied=accept>=.5 and not (request['known_sun'] and result['learned_mode']==0)
                    result.update(proposal_generated=True,proposal_tokens=s['current'].copy(),
                        final_tokens=s['current'].copy() if applied else s['before'],learned_accept_probability=accept,
                        applied=applied,learned_decision='EDIT' if applied else 'KEEP');s['stage']='done'
                if result['forward_calls']>80:raise RuntimeError('batched editor exceeded per-request budget')
            del out
        results.extend(s['result'] for s in states)
    return results
