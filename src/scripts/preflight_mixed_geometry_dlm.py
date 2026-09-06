#!/usr/bin/env python3
"""Bounded real-model verification of mixed geometry training and execution.

The saved checkpoint is an engineering artifact and is never an eligible
sampling policy. No physics outcomes or SUN labels are computed here.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from crystal_dlm.llada_hidden import llada_hidden_forward, resolve_llada_core
from crystal_dlm.mixed_geometry_diffusion import GeometryState, LatticeNormalizer, sample_geometry, sample_prior
from crystal_dlm.mixed_geometry_training_data import (
    MixedBatchSchedule, MixedGeometrySources, forward_mixed, materialize_mixed_batch, read_rows, source_key, source_seed,
)
from crystal_dlm.periodic_v2_objective import PeriodicV2Objective
from crystal_dlm.state_training import enable_native_checkpointing
from scripts.train_mixed_geometry_dlm import autocast_for, file_sha256, global_counter, mixed_batch_loss, write_json


def parameter_group(name):
    if any(part in name for part in ("numeric_adapter.", "repair_task_projection.", "new_token_rows.output_delta")):
        return "token_only"
    if "lora_" in name:
        return "lora"
    if "state_conditioner." in name:
        return "conditioner"
    if "geometry_attention." in name:
        return "geometry_attention"
    if "new_token_rows.input_delta" in name:
        return "input_delta"
    if "geometry_heads.v_head." in name:
        return "v_head"
    if "geometry_heads.u_head." in name:
        return "u_head"
    if "geometry_heads." in name:
        return "geometry_time"
    return "other"


def gradients(model):
    return {name: None if p.grad is None else p.grad.detach().float().cpu().clone()
            for name, p in model.named_parameters() if p.requires_grad}


def gradient_summary(model, before=None):
    result = defaultdict(float)
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        previous = None if before is None else before[name]
        if parameter.grad is None and previous is None:
            continue
        value = torch.zeros_like(previous) if parameter.grad is None else parameter.grad.detach().float().cpu()
        if previous is not None:
            value = value - previous
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"nonfinite gradient increment in {name}")
        result[parameter_group(name)] += float(value.abs().sum())
    return dict(result)


def compare_gradients(model, reference):
    squared_reference, squared_error, maximum = 0., 0., 0.
    by_group = defaultdict(lambda: [0., 0.])
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        expected = reference[name]
        actual = None if parameter.grad is None else parameter.grad.detach().float().cpu()
        if expected is None and actual is None:
            continue
        if expected is None:
            expected = torch.zeros_like(actual)
        if actual is None:
            actual = torch.zeros_like(expected)
        error = (actual - expected).double()
        ref_norm = float(expected.double().square().sum())
        err_norm = float(error.square().sum())
        squared_reference += ref_norm
        squared_error += err_norm
        by_group[parameter_group(name)][0] += ref_norm
        by_group[parameter_group(name)][1] += err_norm
        maximum = max(maximum, float(error.abs().max()))
    relative = (squared_error / max(squared_reference, 1e-30))**.5
    groups = {key: {"reference_l2": a**.5, "error_l2": b**.5,
                    "relative_l2": (b / a)**.5 if a > 0 else None,
                    "passed": b**.5 <= (max(1e-8, .02 * a**.5))}
              for key, (a, b) in by_group.items()}
    return {"relative_l2_error": relative, "maximum_absolute_error": maximum, "groups": groups,
            "relative_l2_tolerance": .02, "group_absolute_l2_floor": 1e-8,
            "passed": relative <= .02 and all(group["passed"] for group in groups.values())}


def tensor_delta(a, b):
    value = (a.detach().float() - b.detach().float()).abs()
    return {"max_abs": float(value.max()), "nonzero_fraction": float((value != 0).float().mean())}


def to_device_state(state, device, dtype):
    return GeometryState(state.z.to(device=device, dtype=dtype), state.fractional.to(device=device, dtype=dtype),
                         state.t.to(device=device, dtype=dtype), state.atom_mask.to(device))


def batch_for(data, tokenizer, device, index, mode, *, epoch=0, padding=False, fixed_time=None):
    example = data.example(index, mode=mode, epoch=epoch, is_padding=padding, fixed_time=fixed_time)
    return materialize_mixed_batch([example], tokenizer, device=device, max_sites=data.max_sites)


def scalar_backward(model, batch, objective, coefficient=1.):
    with autocast_for(batch["input_ids"].device):
        output = forward_mixed(model, batch)
        loss, _, _ = mixed_batch_loss(output, batch, objective)
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("nonfinite preflight loss")
    (loss * coefficient).backward()
    return float(loss.detach())


@torch.no_grad()
def precision_case(model, batch, *, axis, displacement):
    state = batch["geometry_state"]
    z, fractional = state.z.clone(), state.fractional.clone()
    if axis == "fractional":
        fractional[:, 0, 0] = (fractional[:, 0, 0] + displacement).remainder(1)
    else:
        z[:, 0] += displacement
    changed = {**batch, "geometry_state": GeometryState(z, fractional, state.t, state.atom_mask)}

    def encode(current):
        geometry, *_ = model.geometry_inputs(current["input_ids"], current["geometry_context"],
                                             current["geometry_state"], current["species"], current["attention_mask"])
        encoded = model.state_conditioner(**geometry)
        with autocast_for(current["input_ids"].device):
            output = forward_mixed(model, current, output_hidden_states=True)
        return encoded, output

    before_features, before = encode(batch)
    after_features, after = encode(changed)
    return {"axis": axis, "displacement": displacement, "t": float(state.t[0]),
            "cell_features": tensor_delta(before_features["cell_embedding"], after_features["cell_embedding"]),
            "site_features": tensor_delta(before_features["site_embeddings"], after_features["site_embeddings"]),
            "hidden": tensor_delta(before.hidden_states[-1], after.hidden_states[-1]),
            "v_prediction": tensor_delta(before.v_prediction, after.v_prediction),
            "u_prediction": tensor_delta(before.u_prediction, after.u_prediction)}


def main():
    from crystal_dlm.mixed_geometry_model import initialize_mixed_geometry_model, load_mixed_geometry_model
    from scripts.sample_llada_dynamic_crystals import build_dynamic_lightweight_constraints

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    for name in ("v2-checkpoint", "val-data", "val-identity", "normalizer", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--seed", type=int, default=82018)
    parser.add_argument("--benchmark-updates", type=int, default=32)
    args = parser.parse_args()
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    dist.init_process_group("nccl")
    rank, world, device = dist.get_rank(), dist.get_world_size(), torch.device("cuda", local)
    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    start = time.monotonic()
    normalizer = LatticeNormalizer.load(args.normalizer)
    model, tokenizer = initialize_mixed_geometry_model(args.model_path, args.v2_checkpoint, normalizer, device)
    checkpointing = enable_native_checkpointing(model.base_model)
    if not checkpointing:
        raise RuntimeError("native activation checkpointing was not enabled")
    core = resolve_llada_core(model.base_model)
    checkpoint_function = repr(core._activation_checkpoint_fn)
    if "use_reentrant=False" not in checkpoint_function.replace(" ", ""):
        # Native partial/repr can change; inspect the bound partial keywords too.
        if getattr(core._activation_checkpoint_fn, "keywords", {}).get("use_reentrant") is not False:
            raise RuntimeError("native checkpoint function must explicitly use nonreentrant autograd")
    constraints = build_dynamic_lightweight_constraints(
        tokenizer, duplicate_coordinate_mask=True, lattice_volume_mask=True, min_lattice_rad=1e-4,
        canonicalize_periodic_alias=True, pbc_min_distance_mask=True, pbc_min_distance_A=.5, pbc_image_radius=2)
    all_rows = read_rows(args.val_data)
    identities = {source_key(row): row for row in read_rows(args.val_identity)}
    ordered = sorted(all_rows, key=lambda row: source_seed(202609063, source_key(row), 0, "preflight_selection"))
    selected = []
    for count in (1, 10, 20):
        candidate = next(row for row in ordered if len(identities[source_key(row)]["continuous_aligned"]["species"]) == count)
        selected.append(candidate)
    selected_keys = {source_key(row) for row in selected}
    selected.extend(row for row in ordered if source_key(row) not in selected_keys)
    selected = selected[:16]
    data = MixedGeometrySources(selected, [identities[source_key(row)] for row in selected], tokenizer, constraints,
                                normalizer, split="val", seed=202609063, max_sites=model.state_config.max_sites)
    del all_rows, ordered, identities
    objective = PeriodicV2Objective(tokenizer, constraints, device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=5e-7, betas=(.9, .999), eps=1e-8, weight_decay=0.)
    report = {"eligible_policy": False, "world_size": world, "checkpointing": checkpointing,
              "checkpoint_function": checkpoint_function, "torch_version": torch.__version__,
              "input_sha256": {name: file_sha256(getattr(args, name)) for name in ("val_data", "val_identity", "normalizer")},
              "source_keys": data.keys, "trainable_parameters": sum(p.numel() for p in parameters),
              "physics_or_SUN_evaluation": False}

    model.eval()
    token_batch = materialize_mixed_batch([data.example(i, mode="token", epoch=0) for i in (0, 2)],
                                          tokenizer, device=device, max_sites=data.max_sites)
    with torch.no_grad(), autocast_for(device):
        expected = model.token_model(token_batch["input_ids"], attention_mask=token_batch["attention_mask"],
                                     geometry_context=token_batch["geometry_context"]).logits
        actual = forward_mixed(model, token_batch).logits
        report["token_step_zero"] = tensor_delta(expected, actual)
        if report["token_step_zero"]["max_abs"] != 0:
            raise RuntimeError("mixed token step-zero differs from loaded V2")
        del expected, actual
        length = token_batch["input_ids"].shape[1]
        query = torch.linspace(-1., 1., length, device=device)
        key = torch.linspace(.1, -.1, length, device=device)
        heads = torch.linspace(.5, 1.5, core.config.n_heads, device=device)
        bias = heads[None, :, None, None] * query[None, None, :, None] * key[None, None, None, :]
        full = core(input_ids=token_batch["input_ids"], attention_mask=token_batch["attention_mask"],
                    attention_bias=bias, output_hidden_states=True)
        hidden = llada_hidden_forward(model.base_model, token_batch["input_ids"],
                                      attention_mask=token_batch["attention_mask"], attention_bias=bias)
        report["native_hidden_parity_with_bias_and_padding"] = tensor_delta(full.hidden_states[-1], hidden)
        if report["native_hidden_parity_with_bias_and_padding"]["max_abs"] != 0:
            raise RuntimeError("hidden-only encoder differs from the actual final ln_f")
        del full, hidden, bias
    geometry_batch = batch_for(data, tokenizer, device, 1, "geometry", fixed_time=.002)
    report["precision_before_head_open"] = precision_case(model, geometry_batch, axis="fractional", displacement=.002)
    model.zero_grad(set_to_none=True)
    scalar_backward(model, geometry_batch, objective)
    report["zero_head_gradient"] = gradient_summary(model)
    if any(report["zero_head_gradient"].get(name, 0) <= 0 for name in ("v_head", "u_head")):
        raise RuntimeError("zero initialized geometry heads did not receive a gradient")
    if any(value != 0 for name, value in report["zero_head_gradient"].items() if name not in ("v_head", "u_head")):
        raise RuntimeError("zero heads unexpectedly propagated a geometry gradient upstream")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    # Independent reference: eight real states in the 16-source tail window,
    # normalized by all 24 states. Padding contributes exactly zero.
    schedule = MixedBatchSchedule(len(data), world)
    tail = schedule.updates_per_epoch - 1
    reference = None
    if rank == 0:
        reference_states = 0
        for source_rank in range(world):
            for window, _, mode, slots in schedule.rank_batches(rank=source_rank, seed=31, epoch=0):
                if window != tail:
                    continue
                for index, padding in slots:
                    if not padding:
                        scalar_backward(model, batch_for(data, tokenizer, device, index, mode), objective,
                                        schedule.epoch_normalization / schedule.effective_batch)
                        reference_states += 1
        reference = gradients(model)
        report["reference_tail_real_states"] = reference_states
    model.zero_grad(set_to_none=True)
    dist.barrier()
    wrapped = DistributedDataParallel(model, device_ids=[local], find_unused_parameters=True)
    for window, _, mode, slots in schedule.rank_batches(rank=rank, seed=31, epoch=0):
        if window == tail:
            index, padding = slots[0]
            scalar_backward(wrapped, batch_for(data, tokenizer, device, index, mode, padding=padding), objective,
                            schedule.epoch_normalization / schedule.accumulation)
    parity_ok = torch.tensor(1, device=device)
    if rank == 0:
        report["actual_mixed_padding_gradient_equivalence"] = compare_gradients(model, reference)
        parity_ok.fill_(int(report["actual_mixed_padding_gradient_equivalence"]["passed"]))
        del reference
    dist.broadcast(parity_ok, 0)
    if not int(parity_ok):
        raise RuntimeError("DDP mixed accumulation differs from the serial population gradient")
    optimizer.zero_grad(set_to_none=True)

    model.train()
    step, epoch, timed_start = 0, 0, None
    benchmark_start = time.monotonic()
    while step < args.benchmark_updates:
        for _, microstep, mode, slots in schedule.rank_batches(rank=rank, seed=37, epoch=epoch):
            index, padding = slots[0]
            batch = batch_for(data, tokenizer, device, index, mode, epoch=epoch, padding=padding)
            before_g = gradients(model) if step == 0 and microstep == 1 else None
            scalar_backward(wrapped, batch, objective, schedule.epoch_normalization / schedule.accumulation)
            if before_g is not None:
                increments = global_counter(gradient_summary(model, before_g), device)
                report["opened_head_G_gradient_increment_after_T"] = increments
                if any(increments.get(name, 0) <= 0 for name in
                       ("v_head", "u_head", "lora", "conditioner", "geometry_attention", "input_delta", "geometry_time")):
                    raise RuntimeError(f"shared geometry gradient is disconnected: {increments}")
                if increments.get("token_only", 0) > 1e-7:
                    raise RuntimeError("geometry backward changed token-only accumulated gradients")
                del before_g
            if microstep + 1 == schedule.accumulation:
                torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if step == 4:
                    torch.cuda.synchronize(device)
                    timed_start = time.monotonic()
                if rank == 0 and (step <= 2 or step % 8 == 0):
                    print(json.dumps({"event": "preflight_update", "step": step,
                                      "elapsed_seconds": time.monotonic() - start}), flush=True)
                if step == args.benchmark_updates:
                    break
        epoch += 1
    torch.cuda.synchronize(device)
    report["benchmark"] = {"updates": step, "elapsed_seconds": time.monotonic() - benchmark_start,
                           "steady_seconds_per_update": ((time.monotonic() - timed_start) / (step - 4)
                                                         if timed_start is not None and step > 4 else None),
                           "peak_memory_GiB": torch.cuda.max_memory_allocated() / 2**30,
                           "accumulation": schedule.accumulation, "microbatch": 1, "effective_batch": 24,
                           "subset_has_padding": True, "formal_all_sources_may_have_different_length_distribution": True}
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    hashes = [None] * world
    dist.all_gather_object(hashes, digest.hexdigest())
    report["trainable_parameter_hashes_by_rank"] = hashes
    if len(set(hashes)) != 1:
        raise RuntimeError("trained parameter values diverged between DDP ranks")

    model.eval()
    report["precision_after_updates"] = []
    for index in (0, 1, 2):
        batch = batch_for(data, tokenizer, device, index, "geometry", fixed_time=.002)
        for axis in ("fractional", "z"):
            case = precision_case(model, batch, axis=axis, displacement=.002)
            case["num_atoms"] = int(batch["geometry_state"].num_atoms[0])
            report["precision_after_updates"].append(case)
            if case["hidden"]["max_abs"] <= 0 or max(case["v_prediction"]["max_abs"], case["u_prediction"]["max_abs"]) <= 0:
                raise RuntimeError("small-time floating input perturbation disappeared through the shared model")
    batch = batch_for(data, tokenizer, device, rank % 3, "geometry", fixed_time=.002)
    initial = sample_prior(batch["geometry_state"].atom_mask.cpu(),
                           generator=torch.Generator().manual_seed(330019 + rank))
    calls, actual_times = 0, []

    @torch.no_grad()
    def field(state):
        nonlocal calls
        calls += 1
        actual_times.append(float(state.t[0]))
        current = {**batch, "geometry_state": to_device_state(state, device, torch.float32)}
        with autocast_for(device):
            output = forward_mixed(model, current)
        return output.v_prediction.cpu().double(), output.u_prediction.cpu().double()

    sampled = sample_geometry(field, initial, config=model.diffusion_config)
    sampled_lattice = normalizer.decode(sampled.z, sampled.atom_mask.sum(-1))
    report["actual_neural_sampler"] = {"forward_calls": calls, "sampler_nfe": sampled.nfe,
                                        "times": actual_times, "output_dtype": str(sampled.z.dtype),
                                        "finite": bool(torch.isfinite(sampled.z).all() and torch.isfinite(sampled.fractional).all()),
                                        "final_lattice_finite": bool(torch.isfinite(sampled_lattice).all()),
                                        "final_volume_A3": torch.linalg.det(sampled_lattice).tolist()}
    if calls != 33 or sampled.nfe != 33 or actual_times[-1] != model.diffusion_config.epsilon:
        raise RuntimeError("actual sampler did not execute the registered 33rd terminal forward")

    token_check = batch_for(data, tokenizer, device, 0, "token")
    geometry_check = batch_for(data, tokenizer, device, 1, "geometry", fixed_time=.002)
    with torch.no_grad(), autocast_for(device):
        token_expected = forward_mixed(model, token_check).logits.detach().cpu()
        geometry_expected = forward_mixed(model, geometry_check)
        v_expected, u_expected = geometry_expected.v_prediction.cpu(), geometry_expected.u_prediction.cpu()
    checkpoint = args.output_dir / "engineering_checkpoint"
    if rank == 0:
        model.save_pretrained(checkpoint)
        tokenizer.save_pretrained(checkpoint)
        write_json(checkpoint / "CHECKPOINT_FINAL.json", {"eligible_policy": False, "purpose": "implementation_preflight"})
    dist.barrier()
    del wrapped, optimizer, parameters, core, model, geometry_expected
    gc.collect()
    torch.cuda.empty_cache()
    restored, _ = load_mixed_geometry_model(args.model_path, checkpoint, device, trainable=False)
    with torch.no_grad(), autocast_for(device):
        token_actual = forward_mixed(restored, token_check).logits.cpu()
        geometry_actual = forward_mixed(restored, geometry_check)
    report["actual_checkpoint_roundtrip"] = {
        "token": tensor_delta(token_expected, token_actual),
        "v": tensor_delta(v_expected, geometry_actual.v_prediction.cpu()),
        "u": tensor_delta(u_expected, geometry_actual.u_prediction.cpu())}
    if any(value["max_abs"] != 0 for value in report["actual_checkpoint_roundtrip"].values()):
        raise RuntimeError("saved/reloaded mixed checkpoint changed a mode's actual output")
    report["elapsed_seconds"] = time.monotonic() - start
    write_json(args.output_dir / f"PREFLIGHT.rank{rank}.json", report)
    dist.barrier()
    if rank == 0:
        reports = [json.loads((args.output_dir / f"PREFLIGHT.rank{i}.json").read_text()) for i in range(world)]
        write_json(args.output_dir / "SUMMARY.json", {"status": "PASS", "eligible_policy": False,
                    "ranks": reports, "maximum_peak_memory_GiB": max(r["benchmark"]["peak_memory_GiB"] for r in reports),
                    "physics_or_SUN_evaluation": False})
        (args.output_dir / "_SUCCESS").touch()
        print(json.dumps({"event": "preflight_complete", "status": "PASS", "elapsed_seconds": time.monotonic() - start}), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
