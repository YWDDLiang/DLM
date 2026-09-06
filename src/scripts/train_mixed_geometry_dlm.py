#!/usr/bin/env python3
"""Train the shared token/continuous crystal model with a fixed final policy."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import nullcontext
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from crystal_dlm.mixed_geometry_diffusion import LatticeNormalizer, geometry_denoising_risk
from crystal_dlm.mixed_geometry_training_data import (
    MixedBatchSchedule, MixedGeometrySources, VALIDATION_TIME_BANDS,
    forward_mixed, materialize_mixed_batch,
)
from crystal_dlm.periodic_v2_objective import PeriodicV2Objective, V2_OBJECTIVE_PROTOCOL
from crystal_dlm.state_training import enable_native_checkpointing
from crystal_dlm.two_stage_lr import two_stage_lr_multiplier


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=str) + "\n",
                          encoding="utf-8")


def append_json(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def global_counter(counter, device):
    if not dist.is_initialized():
        return dict(counter)
    names = [None] * dist.get_world_size()
    dist.all_gather_object(names, sorted(counter))
    keys = sorted(set().union(*(set(item) for item in names)))
    values = torch.tensor([counter.get(key, 0.) for key in keys], dtype=torch.float64, device=device)
    dist.all_reduce(values)
    return dict(zip(keys, values.tolist()))


def autocast_for(device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if torch.device(device).type == "cuda" else nullcontext()


def mixed_batch_loss(output, batch, token_objective):
    """Fixed microbatch mean; padding never renormalizes a local real subset."""
    if batch["mode"] == "token":
        loss, metrics, conflicts = token_objective(output.logits, batch)
        return loss, {"token_" + key: value for key, value in metrics.items()}, conflicts
    risk = geometry_denoising_risk(output.v_prediction, output.u_prediction,
                                  batch["v_target"], batch["u_target"], batch["geometry_state"].atom_mask,
                                  source_weights=batch["source_weights"])
    weights = batch["source_weights"]
    loss = (risk.per_example * weights).mean()
    weight_sum = float(weights.sum())
    metrics = {"geometry_loss_sum": float(loss.detach()) * len(weights),
               "geometry_lattice_sum": float(risk.lattice.detach()) * weight_sum,
               "geometry_coordinates_sum": float(risk.coordinates.detach()) * weight_sum,
               "geometry_weight_sum": weight_sum, "geometry_state_count": len(weights),
               "geometry_real_states": sum(not e["is_padding"] for e in batch["examples"])}
    return loss, metrics, []


def make_sources(args, tokenizer, constraints, normalizer, *, max_sites):
    common = dict(tokenizer=tokenizer, constraints=constraints, normalizer=normalizer, max_sites=max_sites)
    train = MixedGeometrySources(args.train_data, args.train_identity, split="train", seed=args.data_seed, **common)
    val = MixedGeometrySources(args.val_data, args.val_identity, split="val", seed=args.validation_seed, **common)
    for name, actual in (("train", len(train)), ("val", len(val))):
        expected = getattr(args, f"expected_{name}_sources", None)
        if expected is not None and expected != actual:
            raise ValueError(f"registered {name} source count differs: {actual} != {expected}")
    if normalizer.source_count != len(train):
        raise ValueError("normalizer population differs from the complete training split")
    return train, val


@torch.no_grad()
def validate(model, sources, indices, tokenizer, objective, *, rank, world, device, max_length, max_sites):
    model.eval()
    metrics = Counter()
    for index in indices[rank::world]:
        token = sources.example(index, mode="token", epoch=0)
        batch = materialize_mixed_batch([token], tokenizer, device=device, max_length=max_length, max_sites=max_sites)
        with autocast_for(device):
            _, values, _ = mixed_batch_loss(forward_mixed(model, batch), batch, objective)
        metrics.update(values)
        for band, (lower, upper) in enumerate(VALIDATION_TIME_BANDS):
            example = sources.example(index, mode="geometry", epoch=0, fixed_time=math.sqrt(lower * upper),
                                      noise_stream=f"validation_band_{band}")
            batch = materialize_mixed_batch([example], tokenizer, device=device, max_length=max_length, max_sites=max_sites)
            with autocast_for(device):
                _, values, _ = mixed_batch_loss(forward_mixed(model, batch), batch, objective)
            metrics.update(values)
            metrics.update({f"t_band_{band}_" + key: value for key, value in values.items()})
    return global_counter(metrics, device)


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--v2-checkpoint", type=Path, required=True)
    for name in ("train-data", "val-data", "train-identity", "val-identity", "normalizer", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--seed", type=int, default=82018)
    parser.add_argument("--data-seed", type=int, default=202609062)
    parser.add_argument("--validation-seed", type=int, default=202609063)
    parser.add_argument("--validation-sources", type=int, default=100)
    parser.add_argument("--expected-train-sources", type=int)
    parser.add_argument("--expected-val-sources", type=int)
    parser.add_argument("--epochs", type=int, default=2, choices=(2,))
    parser.add_argument("--microbatch", type=int, default=1)
    parser.add_argument("--effective-batch", type=int, default=24)
    parser.add_argument("--max-length", type=int, default=382)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--log-steps", type=int, default=10)
    parser.add_argument("--launch-manifest", type=Path, required=True)
    return parser


def main():
    from crystal_dlm.mixed_geometry_model import initialize_mixed_geometry_model, set_mixed_geometry_trainable
    from scripts.sample_llada_dynamic_crystals import build_dynamic_lightweight_constraints

    args = argument_parser().parse_args()
    world, local = int(os.environ.get("WORLD_SIZE", 1)), int(os.environ.get("LOCAL_RANK", 0))
    if not torch.cuda.is_available():
        raise RuntimeError("this model training entry point requires CUDA")
    torch.cuda.set_device(local)
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    dist.init_process_group("nccl")
    rank, device = dist.get_rank(), torch.device("cuda", local)
    launch = json.loads(args.launch_manifest.read_text(encoding="utf-8"))
    # The launcher creates an immutable git archive and records its object ID.
    # That archive deliberately has no .git directory or mutable checkout.
    code_sha = launch["code_commit"]
    input_names = ("train_data", "val_data", "train_identity", "val_identity", "normalizer")
    input_hashes = {name: file_sha256(getattr(args, name)) for name in input_names}
    if input_hashes != launch["input_sha256"]:
        raise ValueError("training input hashes differ from launch manifest")
    normalizer = LatticeNormalizer.load(args.normalizer)
    model, tokenizer = initialize_mixed_geometry_model(args.model_path, args.v2_checkpoint, normalizer, device)
    trainable = set_mixed_geometry_trainable(model)
    checkpointing = enable_native_checkpointing(model.base_model)
    if not checkpointing:
        raise RuntimeError("shared native checkpointing is unavailable")
    max_sites = model.state_config.max_sites
    constraints = build_dynamic_lightweight_constraints(
        tokenizer, duplicate_coordinate_mask=True, lattice_volume_mask=True, min_lattice_rad=1e-4,
        canonicalize_periodic_alias=True, pbc_min_distance_mask=True, pbc_min_distance_A=.5, pbc_image_radius=2)
    train, val = make_sources(args, tokenizer, constraints, normalizer, max_sites=max_sites)
    schedule = MixedBatchSchedule(len(train), world, args.microbatch, args.effective_batch)
    allocation = schedule.manifest(args.epochs)
    indices = val.validation_indices(args.validation_sources, args.validation_seed)
    wrapped = DistributedDataParallel(model, device_ids=[local], find_unused_parameters=True)
    objective = PeriodicV2Objective(tokenizer, constraints, device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=5e-5, betas=(.9, .999), eps=1e-8, weight_decay=0.)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: two_stage_lr_multiplier(
        step, total_steps=allocation["updates"], stage_boundary=allocation["epoch_updates"],
        stage1_warmup=100, stage1_min_ratio=.2, stage2_base_ratio=.2,
        stage2_warmup=100, stage2_min_ratio=.1))
    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        config = {**vars(args), **allocation, "code_commit": code_sha, "launch_manifest_content": launch,
                  "input_sha256": input_hashes, "trainable": trainable, "checkpointing": checkpointing,
                  "normalizer": normalizer.to_dict(), "token_objective": V2_OBJECTIVE_PROTOCOL,
                  "model_config": asdict(model.model_config), "diffusion_config": asdict(model.diffusion_config),
                  "parent_provenance": model.parent_provenance,
                  "geometry_objective": "0.5*mean6(v-v_target)^2 + 0.5*mean3N(u-u_target)^2",
                  "time_distribution": "LogUniform[0.002,1]", "ddp_microbatch_sync": True,
                  "warm_start": str(args.v2_checkpoint), "new_epochs": args.epochs,
                  "checkpoint_selection": "last_complete_epoch_only; no validation or SUN selection",
                  "monitor_source_keys": [val.keys[i] for i in indices],
                  "monitor_time_bands": VALIDATION_TIME_BANDS,
                  "monitor_fixed_times": [math.sqrt(a*b) for a, b in VALIDATION_TIME_BANDS],
                  "monitor_states_per_epoch": args.validation_sources * 7}
        write_json(args.output_dir / "training_config.json", config)
    dist.barrier()
    start, step = time.monotonic(), 0
    totals, block = Counter(), Counter()
    coverage = torch.zeros((len(train), 2), dtype=torch.long, device=device)
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        model.train()
        for window, microstep, mode, slots in schedule.rank_batches(rank=rank, seed=args.data_seed, epoch=epoch):
            examples = [train.example(index, mode=mode, epoch=epoch, is_padding=padding) for index, padding in slots]
            batch = materialize_mixed_batch(examples, tokenizer, device=device,
                                             max_length=args.max_length, max_sites=max_sites)
            with autocast_for(device):
                output = forward_mixed(wrapped, batch)
                loss, metrics, conflicts = mixed_batch_loss(output, batch, objective)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"nonfinite {mode} training loss at epoch={epoch}, window={window}")
            (loss * schedule.epoch_normalization / schedule.accumulation).backward()
            block.update(metrics)
            for example in examples:
                if example["is_padding"]:
                    totals[mode + "_padding_states"] += 1
                else:
                    coverage[example["source_index"], 0 if mode == "token" else 1] += 1
                    totals[mode + "_real_states"] += 1
            for conflict in conflicts:
                append_json(args.output_dir / f"support_conflicts.rank{rank}.jsonl", conflict)
            del output, loss, batch
            if microstep + 1 != schedule.accumulation:
                continue
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step <= 3 or step % args.log_steps == 0 or step % schedule.updates_per_epoch == 0:
                values = global_counter(block, device)
                if rank == 0:
                    event = {**values, "event": "train", "step": step, "epoch": epoch + 1,
                             "lr_next": scheduler.get_last_lr()[0], "grad_norm_rank0": float(norm),
                             "elapsed_seconds": time.monotonic() - start,
                             "peak_memory_GiB_rank0": torch.cuda.max_memory_allocated() / 2**30}
                    append_json(args.output_dir / "training_log.jsonl", event)
                    print(json.dumps(event), flush=True)
                block = Counter()
        values = validate(model, val, indices, tokenizer, objective, rank=rank, world=world, device=device,
                          max_length=args.max_length, max_sites=max_sites)
        all_coverage = coverage.clone()
        dist.all_reduce(all_coverage)
        if not bool((all_coverage == epoch + 1).all()):
            raise RuntimeError("source T/G coverage is incomplete or duplicated")
        dist.barrier()
        if rank == 0:
            event = {**values, "event": "validation", "step": step, "epoch": epoch + 1,
                     "source_count": len(indices), "checkpoint_selection": False}
            append_json(args.output_dir / "validation_log.jsonl", event)
            print(json.dumps(event), flush=True)
            checkpoint = args.output_dir / "checkpoints" / f"step-{step}"
            model.save_pretrained(checkpoint)
            tokenizer.save_pretrained(checkpoint)
            torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                        "completed_step": step, "completed_epochs": epoch + 1,
                        "source_view_epoch_coverage": all_coverage.cpu()}, checkpoint / "MIXED_TRAIN_STATE.pt")
            write_json(checkpoint / "CHECKPOINT_FINAL.json", {
                "completed_step": step, "completed_epochs": epoch + 1, "expected_epochs": args.epochs,
                "eligible_policy": epoch + 1 == args.epochs, "method": "mixed_geometry_h_p33"})
        dist.barrier()
    totals = global_counter(totals, device)
    if (step != allocation["updates"]
            or sum(totals.get(mode + "_real_states", 0) for mode in ("token", "geometry")) != allocation["effective_states"]):
        raise RuntimeError("training did not complete its declared state and update budget")
    if rank == 0:
        destination = args.output_dir / "checkpoints" / f"step-{step}"
        report = {"method": "mixed_geometry_h_p33", "eligible_policy": True, "policy_path": str(destination),
                  **allocation, "totals": totals, "coverage_min": int(all_coverage.min()),
                  "coverage_max": int(all_coverage.max()), "elapsed_seconds": time.monotonic() - start,
                  "physical_benefit": "unmeasured_until_complete_registered_SUN_evaluation"}
        write_json(args.output_dir / "TRAIN_FINAL.json", report)
        (args.output_dir / "POLICY_PATH").write_text(str(destination) + "\n", encoding="utf-8")
        (args.output_dir / "_SUCCESS").touch()
        print(json.dumps(report), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
