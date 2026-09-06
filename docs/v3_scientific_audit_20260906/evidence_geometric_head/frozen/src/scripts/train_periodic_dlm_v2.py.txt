#!/usr/bin/env python3
"""Six-A800 original-LLaDA V2 training with complete original MP20 coverage."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
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
from torch.utils.data import DataLoader, DistributedSampler, Subset

from crystal_dlm.periodic_repair_initialization import set_fresh_repair_trainable
from crystal_dlm.periodic_v2_initialization import initialize_periodic_v2_model
from crystal_dlm.periodic_v2_training_data import PeriodicV2TrainingDataset, V2_TRAINING_DATA_PROTOCOL
from crystal_dlm.periodic_v2_objective import PeriodicV2Objective, V2_OBJECTIVE_PROTOCOL
from crystal_dlm.state_training import enable_native_checkpointing, materialize_state_batch
from crystal_dlm.two_stage_lr import two_stage_lr_multiplier


def budget(source_count=27136, *, world=6, microbatch=2, accumulation=2, epochs=2):
    effective = world * microbatch * accumulation
    real = 2 * source_count
    padded = ((real + effective - 1) // effective) * effective
    return {"effective_batch": effective, "real_states_per_epoch": real,
            "padded_states_per_epoch": padded, "padding_per_epoch": padded - real,
            "epoch_updates": padded // effective, "updates": epochs * (padded // effective),
            "effective_states": epochs * real, "epochs": epochs}


def global_counter(counter, device):
    names = [None] * dist.get_world_size()
    dist.all_gather_object(names, sorted(counter))
    keys = sorted(set().union(*(set(item) for item in names)))
    values = torch.tensor([counter.get(key, 0.) for key in keys], dtype=torch.float64, device=device)
    dist.all_reduce(values)
    return dict(zip(keys, values.tolist()))


def gradient_modules(model):
    result = {"state_conditioner": model.state_conditioner, **model.repair_modules()}
    # Separate residual heads from the legacy path: one active module cannot
    # hide a disconnected lattice-to-site interaction.
    for name, child in model.geometry_attention.named_children():
        if any(parameter.requires_grad for parameter in child.parameters()):
            result[f"geometry_attention.{name}"] = child
    return result


def append_json(path, value):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def validate_input_manifest(root):
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if not (root / "_SUCCESS").is_file():
        raise ValueError("frozen Planner preparation has not completed")
    required = {"schema": "periodic_v2_frozen_plan_conditioning_v1",
                "planner_training": False, "composition_resampled": False,
                "outcomes_read": False, "generated_path_files_read": False,
                "all_source_clean_answers_preserved": True,
                "pointer_uses_actual_sampled_soft_ids": True}
    if any(manifest.get(key) != value for key, value in required.items()):
        raise ValueError("frozen Planner source contract differs")
    for split, count in (("train", 27136), ("val", 9047)):
        if (manifest["splits"][split]["source_rows"] != count
                or manifest["splits"][split]["all_clean_answers_unchanged"] is not True
                or hashlib.sha256((root / f"{split}.jsonl").read_bytes()).hexdigest()
                != manifest["output_sha256"][f"{split}.jsonl"]):
            raise ValueError("prepared source coverage or immutable content differs")
    return manifest


def main():
    from scripts.sample_llada_dynamic_crystals import build_dynamic_lightweight_constraints
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=82017)
    p.add_argument("--data-seed", type=int, default=20260906)
    args = p.parse_args()
    world, local = int(os.environ.get("WORLD_SIZE", 1)), int(os.environ.get("LOCAL_RANK", 0))
    if world != 6 or "SLURM_JOB_ID" not in os.environ or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("V2 starts only inside the registered six-A800 allocation")
    torch.cuda.set_device(local)
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    dist.init_process_group("nccl")
    rank, device = dist.get_rank(), torch.device("cuda", local)
    allocation = budget()
    plans = validate_input_manifest(args.data_dir)
    model, tokenizer = initialize_periodic_v2_model(args.model_path, device)
    trainable = set_fresh_repair_trainable(model)
    checkpointing = enable_native_checkpointing(model.base_model)
    if not checkpointing:
        raise RuntimeError("registered LLaDA checkpointing route is unavailable")
    constraints = build_dynamic_lightweight_constraints(
        tokenizer, duplicate_coordinate_mask=True, lattice_volume_mask=True, min_lattice_rad=1e-4,
        canonicalize_periodic_alias=True, pbc_min_distance_mask=True,
        pbc_min_distance_A=.5, pbc_image_radius=2)
    data = PeriodicV2TrainingDataset(args.data_dir / "train.jsonl", tokenizer, constraints,
                                     seed=args.data_seed, effective_batch=24)
    val = PeriodicV2TrainingDataset(args.data_dir / "val.jsonl", tokenizer, constraints,
                                    seed=args.data_seed, expected_rows=9047,
                                    expected_split="val", effective_batch=24)
    if (len(data) != allocation["padded_states_per_epoch"]
            or data.real_length != allocation["real_states_per_epoch"]
            or allocation["updates"] != 4524):
        raise ValueError("complete original source/view accounting differs from the frozen budget")
    sampler = DistributedSampler(data, num_replicas=6, rank=rank, shuffle=True, seed=args.data_seed)
    loader = DataLoader(data, batch_size=2, sampler=sampler, num_workers=1, collate_fn=list)
    val_loader = DataLoader(Subset(val, list(range(rank, 200, world))), batch_size=2,
                            num_workers=0, collate_fn=list)
    wrapped = DistributedDataParallel(model, device_ids=[local], find_unused_parameters=True)
    objective = PeriodicV2Objective(tokenizer, constraints, device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=5e-5, weight_decay=0.)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: two_stage_lr_multiplier(
        step, total_steps=allocation["updates"], stage_boundary=allocation["epoch_updates"],
        stage1_warmup=100, stage1_min_ratio=.2, stage2_base_ratio=.2,
        stage2_warmup=100, stage2_min_ratio=.1))
    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        config = {**vars(args), **allocation, "world_size": 6, "microbatch": 2,
                  "gradient_accumulation": 2, "lr_stage1": 5e-5, "lr_stage2": 1e-5,
                  "initialization": model.raw_initialization, "trainable": trainable,
                  "geometry_config": asdict(model.v2_config),
                  "data_protocol": V2_TRAINING_DATA_PROTOCOL, "objective_protocol": V2_OBJECTIVE_PROTOCOL,
                  "checkpointing": checkpointing, "frozen_plan_manifest": plans,
                  "data_sha256": {split: hashlib.sha256((args.data_dir / f"{split}.jsonl").read_bytes()).hexdigest()
                                   for split in ("train", "val")},
                  "original_sources": 27136, "validation_sources": 9047,
                  "monitor_sources": 100, "monitor_states": 200, "checkpoint_selection": "registered_final_only",
                  "prefix_branch_probability": .75, "lattice_object_weight": .5,
                  "coordinate_object_weight": .5, "exact_CE_temperature": .7,
                  "ordinal_smoothing_weight": 0., "old_dlm_weights_loaded": False,
                  "K4_K8_data_read": False, "native_deployment": "construct_then_full_cell_repair",
                  "layer_bias": "head_specific_shared_across_layers",
                  "padding_normalization": "batch mean times padded_epoch_states/real_epoch_states"}
        (args.output_dir / "training_config.json").write_text(json.dumps(config, indent=2, default=str) + "\n")
    dist.barrier()
    start, step, initial_delta = time.monotonic(), 0, None
    optimizer.zero_grad(set_to_none=True)
    totals, block, gradients = Counter(), Counter(), Counter()
    source_index = {int(row["source_row_idx"]): i for i, row in enumerate(data.rows)}
    coverage = torch.zeros((len(data.rows), 2), dtype=torch.long)
    target_coverage = Counter()
    for epoch in range(2):
        data.epoch = epoch
        sampler.set_epoch(epoch)
        model.train()
        for microstep, examples in enumerate(loader):
            batch = materialize_state_batch(examples, tokenizer, device=device, max_length=382)
            if initial_delta is None:
                model.eval()
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    expected = model.base_model(input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"], attention_bias=torch.zeros(
                            len(examples), 1, batch["input_ids"].shape[1], batch["input_ids"].shape[1],
                            device=device)).logits
                    actual = model(batch["input_ids"], attention_mask=batch["attention_mask"],
                                   geometry_context=batch["geometry_context"]).logits
                    delta = (expected - actual).abs().max().detach().float()
                dist.all_reduce(delta, op=dist.ReduceOp.MAX)
                initial_delta = float(delta)
                if initial_delta > 1e-5:
                    raise RuntimeError(f"V2 raw zero-increment equality failed: {initial_delta}")
                del actual, expected
                model.train()
                if rank == 0:
                    print(json.dumps({"event": "v2_initialization_verified", "zero_logit_delta": initial_delta,
                                      "checkpointing": checkpointing, "trainable": trainable}), flush=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = wrapped(batch["input_ids"], attention_mask=batch["attention_mask"],
                                 geometry_context=batch["geometry_context"]).logits
                loss, metrics, conflicts = objective(logits, batch)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("nonfinite V2 loss")
            normalized = loss * (len(data) / data.real_length)
            (normalized / 2).backward()
            block.update(metrics)
            block["state_loss_sum"] += float(loss.detach()) * len(examples)
            block["padded_states"] += len(examples)
            for example in examples:
                if example["is_padding"]:
                    totals["padding_states"] += 1
                    continue
                identity, view = int(example["source_row_idx"]), int(example["view"])
                coverage[source_index[identity], view] += 1
                totals["real_states"] += 1
                totals[f"view{view}_{example.get('state_kind', example.get('branch', 'unspecified'))}"] += 1
                totals["empty_states"] += int(example["empty_supervision"])
                totals["corruption_fallbacks"] += int(example.get("structured_corruption_fallback", False))
                for position in example["positions"]:
                    target_coverage[f"{identity}:{view}:{position}"] += 1
                if source_index[identity] < 64:
                    keys = ("source_row_idx", "view", "epoch", "phase", "branch", "species_program",
                            "species_program_source", "old_body", "input_body", "transaction_positions",
                            "positions", "targets", "legal_eligible", "prefix_reachable",
                            "old_state_admitted", "numeric_noise_components", "noise_metadata_dropped",
                            "mask_probability", "corruption_info")
                    append_json(args.output_dir / f"state_audit.rank{rank}.jsonl",
                                {key: example.get(key) for key in keys})
            if conflicts:
                for conflict in conflicts:
                    append_json(args.output_dir / f"support_conflicts.rank{rank}.jsonl", conflict)
            if (microstep + 1) % 2:
                continue
            if step < 64:
                for name, module in gradient_modules(model).items():
                    gradients[name] += sum(float(parameter.grad.detach().float().abs().sum())
                                           for parameter in module.parameters() if parameter.grad is not None)
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step == 64:
                global_gradients = global_counter(gradients, device)
                if any(value <= 0 for value in global_gradients.values()):
                    raise RuntimeError(f"a V2 module is disconnected: {global_gradients}")
                if rank == 0:
                    (args.output_dir / "EARLY_GRADIENT_CHECK.json").write_text(json.dumps(global_gradients, indent=2) + "\n")
            if step <= 3 or step % 10 == 0 or step % allocation["epoch_updates"] == 0:
                values = global_counter(block, device)
                if rank == 0:
                    event = dict(values, event="train", step=step, epoch=epoch + 1,
                                 grad_norm_rank0=float(norm), lr=scheduler.get_last_lr()[0],
                                 elapsed_seconds=time.monotonic() - start,
                                 peak_memory_GiB_rank0=torch.cuda.max_memory_allocated() / 2**30)
                    append_json(args.output_dir / "training_log.jsonl", event)
                    print(json.dumps(event), flush=True)
                block = Counter()
        model.eval()
        val.epoch = 0
        val_sum = torch.zeros(2, dtype=torch.float64, device=device)
        val_metrics = Counter()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for examples in val_loader:
                batch = materialize_state_batch(examples, tokenizer, device=device, max_length=382)
                logits = model(batch["input_ids"], attention_mask=batch["attention_mask"],
                               geometry_context=batch["geometry_context"]).logits
                value, metrics, _ = objective(logits, batch)
                # The objective is a batch mean, including a possibly short last batch.
                val_sum += torch.tensor([float(value) * len(examples), len(examples)], device=device)
                val_metrics.update(metrics)
        dist.all_reduce(val_sum)
        val_metrics = global_counter(val_metrics, device)
        all_coverage = coverage.to(device).clone()
        dist.all_reduce(all_coverage)
        if not bool((all_coverage == epoch + 1).all()):
            raise RuntimeError("source/view coverage was not complete at the epoch boundary")
        dist.barrier()
        if rank == 0:
            event = dict(val_metrics, event="validation_monitor", step=step, states=int(val_sum[1]),
                         loss=float(val_sum[0] / val_sum[1]), checkpoint_selection=False)
            append_json(args.output_dir / "validation_log.jsonl", event)
            print(json.dumps(event), flush=True)
            checkpoint = args.output_dir / "checkpoints" / f"step-{step}"
            model.save_pretrained(checkpoint)
            tokenizer.save_pretrained(checkpoint)
            torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                        "completed_step": step, "completed_epochs": epoch + 1,
                        "source_view_epoch_coverage": all_coverage.detach().cpu()}, checkpoint / "V2_TRAIN_STATE.pt")
            (checkpoint / "CHECKPOINT_FINAL.json").write_text(json.dumps({
                "completed_step": step, "completed_epochs": epoch + 1,
                "eligible_policy": epoch == 1, "method": "periodic_dlm_v2_from_original_llada"}, indent=2) + "\n")
        dist.barrier()
    (args.output_dir / f"target_coverage.rank{rank}.json").write_text(json.dumps(dict(target_coverage)) + "\n")
    totals = global_counter(totals, device)
    if step != allocation["updates"] or totals["real_states"] != allocation["effective_states"]:
        raise RuntimeError("registered final effective state budget was not completed")
    dist.barrier()
    if rank == 0:
        destination = args.output_dir / "checkpoints" / f"step-{step}"
        report = dict(method="periodic_dlm_v2_from_original_llada", eligible_policy=True,
                      policy_path=str(destination), **allocation, totals=totals,
                      coverage_min=int(all_coverage.min()), coverage_max=int(all_coverage.max()),
                      zero_increment_max_logit_delta=initial_delta, old_dlm_weights_loaded=False,
                      K4_K8_data_read=False, native_full_cell_repair_required=True,
                      elapsed_seconds=time.monotonic() - start,
                      physical_benefit="unmeasured_until_registered_native_and_tau800_evaluation")
        (args.output_dir / "TRAIN_FINAL.json").write_text(json.dumps(report, indent=2) + "\n")
        (args.output_dir / "POLICY_PATH").write_text(str(destination) + "\n")
        (args.output_dir / "_SUCCESS").touch()
        print(json.dumps(report), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
