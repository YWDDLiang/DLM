"""Dense editor distillation with complete-pass content updates and detached heads.

Content uses the typed periodic vocabulary, not an approximation to the joint
geometric acceptance region. Every complete teacher was checked under that
region at compilation; the deployed sequential sampler keeps its hard masks.
"""
from collections import Counter
import random
import time

import torch

from crystal_dlm.expert_edit import ExpertEditObjective, inference_view, materialize_edit_batch
from crystal_dlm.fixed_slot import MASK_TOKEN_ID
from crystal_dlm.ranked_feedback import validate_action_target
from crystal_dlm.rsi_minibatch import decision_head_parameter, editor_head_loss, epoch_indices, has_head_supervision


def dense_view(row, prefix, *, masked):
    target = row['content_target_tokens']
    order = row['content_positions']
    validate_action_target(row['current_tokens'], target, order)
    current = list(row['current_tokens'])
    cut = row.get('training_cut')
    if cut is not None:
        if not 0 <= cut < len(order):
            raise ValueError('training cut must leave at least one masked target')
        for index, pos in enumerate(order):
            current[pos] = target[pos] if index < cut else MASK_TOKEN_ID
    if masked:
        for pos in order if cut is None else []:
            current[pos] = MASK_TOKEN_ID
    view = inference_view(prefix, row['current_tokens'], current, row['num_sites'], 1,
                          order, remaining=80, reveal=0. if cut is None else cut / len(order))
    for pos in order if cut is None else order[cut:]:
        view['targets'][pos] = target[pos]
    return view


def dense_vectors(model, tokenizer, rows, objective, *, masked):
    device = next(model.parameters()).device
    views = [dense_view(r, tokenizer(r['prompt'], add_special_tokens=False)['input_ids'], masked=masked) for r in rows]
    batch = materialize_edit_batch(views, tokenizer, device)
    output = model(batch['input_ids'], attention_mask=batch['attention_mask'], edit_context=batch['edit_context'])
    rr, pp = torch.nonzero(batch['targets'] != -100, as_tuple=True)
    relative = pp - batch['edit_context'].prompt_lengths[rr]
    vectors = []
    for kind in range(9):
        if kind < 6:
            selected = relative == kind + 1
            family, axis = ('length', 'ABC'[kind]) if kind < 3 else ('angle', 'ABG'[kind - 3])
        else:
            selected = (relative >= 8) & ((relative - 8).remainder(4) == kind - 6)
            family, axis = 'coord', 'XYZ'[kind - 6]
        r, p = rr[selected], pp[selected]
        if not len(r):
            continue
        logits, ids = objective.typed_vector(output.logits, r, p, family, axis)
        target = batch['targets'][r, p]
        if family == 'coord':
            original_ids, values = objective.tables[(family, axis)]
            alias, zero = original_ids[values == 1].item(), original_ids[values == 0].item()
            target = torch.where(target == alias, zero, target)
        matches = target[:, None] == ids[None]
        if not bool(matches.any(-1).all()):
            raise ValueError('dense target is outside canonical typed periodic vocabulary')
        vectors.append((r, logits.log_softmax(-1), matches.long().argmax(-1)))
    if sum(len(r) for r, _, _ in vectors) != len(rr):
        raise ValueError('dense supervision omitted or duplicated a target token')
    return vectors


def dense_loss(live, reference, rows):
    device = live[0][1].device
    sums = torch.zeros(len(rows), device=device)
    counts = torch.zeros_like(sums)
    kl_sums = torch.zeros_like(sums)
    for (r, p, target), (qr, q, qtarget) in zip(live, reference, strict=True):
        if not torch.equal(r, qr) or not torch.equal(target, qtarget):
            raise ValueError('reference dense supervision changed')
        ce = -p.gather(1, target[:, None]).squeeze(1)
        kl = (q.exp() * (q - p)).sum(-1)
        sums = sums.scatter_add(0, r, ce)
        kl_sums = kl_sums.scatter_add(0, r, kl)
        counts = counts.scatter_add(0, r, torch.ones_like(ce))
    if bool((counts == 0).any()):
        raise ValueError('empty dense target')
    return (sums / counts).sum(), (kl_sums / counts).sum(), int(counts.sum())


def train_t2t(model, tokenizer, examples, spec, selected, reference, optimizer, support, output, write_json):
    import torch.distributed as dist
    device = next(model.parameters()).device
    world = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    heads = [(n, p) for n, p in selected if decision_head_parameter(n)]
    content = [(n, p) for n, p in selected if not decision_head_parameter(n)]
    if spec['branch'] != 'E' or not heads or not content:
        raise ValueError('T2T requires the existing editor content modules and decision heads')
    objective = ExpertEditObjective(tokenizer, device, temperature=.7)
    frozen = spec.get('editor_heads_only', False)
    content_count = sum(bool(r.get('content_target_tokens')) for r in examples)
    if not content_count:
        raise ValueError('T2T dataset contains no trusted dense targets')
    if frozen:
        for _, p in content:
            p.requires_grad_(False)
    inspected, head_visits, content_visits = Counter(), Counter(), Counter()
    token_visits = Counter()
    history, trigger = [], None
    hard_stop = spec.get('reference_KL_hard_stop', True)
    threshold_crossings = 0; maximum_observed_kl = 0.
    started = time.monotonic()
    content_steps = head_steps = total_heads = 0
    timed_out = False
    for epoch in range(spec['epochs']):
        # Parameters affecting content stay fixed until the entire pass finishes.
        # A KL/time interruption discards this pass's accumulated content gradient.
        buffers = {}
        pending_visits, pending_tokens = Counter(), Counter()
        pass_kl = pass_ce = 0.
        for _, indices in epoch_indices(len(examples), batch_size=spec['batch_size'], world=world,
                                        epochs=1, seed=spec['seed'] + epoch):
            width = len(indices) // world
            rows = [examples[i] for i in indices[rank * width:(rank + 1) * width]]
            inspected.update(r['pair_id'] for r in rows)
            optimizer.zero_grad(set_to_none=True)
            targets = [r for r in rows if r.get('content_target_tokens')] if not frozen else []
            measured_kl = 0.
            ce = kl = None
            if targets:
                live_state = {n: p.detach().clone() for n, p in content}
                with torch.no_grad():
                    try:
                        for n, p in content:
                            p.copy_(reference[n])
                        reference_vectors = dense_vectors(model, tokenizer, targets, objective, masked=bool(epoch % 2))
                    finally:
                        for n, p in content:
                            p.copy_(live_state[n])
                del live_state
                actual_vectors = dense_vectors(model, tokenizer, targets, objective, masked=bool(epoch % 2))
                ce, kl, tokens = dense_loss(actual_vectors, reference_vectors, targets)
                measured_kl = float(kl.detach()) / len(targets)
            observed = torch.tensor(measured_kl, device=device)
            if world > 1:
                dist.all_reduce(observed, op=dist.ReduceOp.MAX)
            pass_kl = max(pass_kl, float(observed))
            maximum_observed_kl = max(maximum_observed_kl, float(observed))
            crossed = not frozen and float(observed) > spec['max_reference_kl']
            threshold_crossings += int(crossed)
            if crossed and hard_stop:
                trigger = dict(epoch=epoch, after_content_optimizer_steps=content_steps,
                               maximum_rank_KL=float(observed), local_KL=measured_kl,
                               discarded_partial_pass=True)
                frozen = True
                buffers.clear(); pending_visits.clear(); pending_tokens.clear()
                for _, p in content:
                    p.requires_grad_(False); p.grad = None
            if ce is not None:
                if not frozen:
                    # Summing rank means at the end yields the full data-pass mean.
                    loss = (ce + spec['reference_kl_weight'] * kl) * world / content_count
                    loss.backward()
                    pass_ce += float(ce.detach())
                    for n, p in content:
                        if p.grad is not None:
                            if n not in buffers: buffers[n] = p.grad.detach().clone()
                            else: buffers[n].add_(p.grad)
                    pending_visits.update(r['pair_id'] for r in targets)
                    for r in targets: pending_tokens[r['pair_id']] += len(r['content_positions'])
                    del loss
                del ce, kl, actual_vectors, reference_vectors
            optimizer.zero_grad(set_to_none=True)
            head_loss, count = editor_head_loss(model, tokenizer, rows, device, detach_content=True)
            if head_loss is not None:
                head_loss.backward()
            for _, p in heads:
                if p.grad is None: p.grad = torch.zeros_like(p)
                if world > 1:
                    dist.all_reduce(p.grad); p.grad.div_(world)
            if any(p.grad is not None for _, p in content):
                raise ValueError('detached decision loss leaked gradients into content')
            grad = torch.nn.utils.clip_grad_norm_([p for _, p in heads], 1., error_if_nonfinite=True)
            if float(grad) == 0: raise ValueError('decision update has no gradient')
            optimizer.step(); head_steps += 1; total_heads += count
            head_visits.update(r['pair_id'] for r in rows if has_head_supervision(r))
            event = dict(step=head_steps, epoch=epoch, update_kind='detached_heads',
                mean_loss=float(head_loss.detach()) if head_loss is not None else None,
                reference_KL=measured_kl, content_optimizer_steps=content_steps,
                reference_KL_hard_stop=hard_stop, maximum_rank_KL=float(observed),
                head_optimizer_steps=head_steps, content_frozen_by_KL=trigger is not None,
                gradient_norm=float(grad), source_examples_per_GPU=width,
                seconds=time.monotonic() - started,
                peak_GPU_GB=torch.cuda.max_memory_allocated() / 1e9 if device.type == 'cuda' else 0.)
            history.append(event)
            if rank == 0:
                write_json(output/'PROGRESS.json', event)
                print(__import__('json').dumps(event), flush=True)
            stop = torch.tensor(int(time.monotonic() - started >= spec['max_training_seconds']), device=device)
            if world > 1: dist.all_reduce(stop, op=dist.ReduceOp.MAX)
            if bool(stop):
                timed_out = True
                break
        if not frozen and not timed_out:
            optimizer.zero_grad(set_to_none=True)
            for n, p in content:
                p.grad = buffers.get(n, torch.zeros_like(p))
                if world > 1:
                    dist.all_reduce(p.grad); p.grad.div_(world)
            grad = torch.nn.utils.clip_grad_norm_([p for _, p in content], 1., error_if_nonfinite=True)
            if float(grad) == 0: raise ValueError('complete content pass has no gradient')
            optimizer.step(); content_steps += 1
            content_visits.update(pending_visits); token_visits.update(pending_tokens)
            event = dict(step=head_steps + content_steps, epoch=epoch, update_kind='complete_pass_content',
                content_optimizer_steps=content_steps, head_optimizer_steps=head_steps,
                local_dense_CE_sum=pass_ce, maximum_batch_rank_KL=pass_kl,
                gradient_norm=float(grad), seconds=time.monotonic() - started)
            history.append(event)
            if rank == 0: print(__import__('json').dumps(event), flush=True)
        del buffers
        if timed_out: break
    report = dict(pair_visits=dict(inspected), pair_visits_semantics='inspected_batches',
        content_updated_pair_visits=dict(content_visits), head_updated_pair_visits=dict(head_visits),
        content_updated_token_visits=dict(token_visits), content_optimizer_steps=content_steps,
        head_optimizer_steps=head_steps, KL_stop=trigger is not None, KL_trigger=trigger,
        head_continuation_after_KL=trigger is not None, content_frozen_from_start=spec.get('editor_heads_only', False),
        content_update_unit='one_complete_dense_data_pass', epochs_cap=spec['epochs'], timed_out=timed_out,
        reference_KL_hard_stop=hard_stop, KL_threshold_crossing_batches=threshold_crossings,
        maximum_observed_batch_rank_KL=maximum_observed_kl,
        dense_view_schedule='even_epoch_T2T_odd_epoch_all_mask_M2T',
        current_structure_always_visible=True, detached_decision_head_gradients=True)
    write_json(output/f'EXPOSURE_rank{rank}.json', report)
    if not history: raise ValueError('no T2T or decision update completed')
    return head_steps + content_steps, 0, total_heads, history, random.Random(spec['seed'] + rank)
