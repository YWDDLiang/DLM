#!/usr/bin/env python3
"""Original LLaDA + original MP20; no earlier DLM checkpoint or K4/K8 input."""
from __future__ import annotations
import argparse
from collections import Counter
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
from crystal_dlm.periodic_repair_initialization import initialize_fresh_periodic_repair_model, set_fresh_repair_trainable
from crystal_dlm.periodic_base_training_data import PeriodicBaseTrainingDataset, BASE_TRAINING_DATA_PROTOCOL
from crystal_dlm.periodic_base_objective import PeriodicBaseObjective
from crystal_dlm.state_training import enable_native_checkpointing, materialize_state_batch
from crystal_dlm.two_stage_lr import two_stage_lr_multiplier
from scripts.sample_llada_dynamic_crystals import build_dynamic_lightweight_constraints


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--train-jsonl", type=Path, required=True)
    p.add_argument("--val-jsonl", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=82017)
    p.add_argument("--data-seed", type=int, default=20260515)
    args = p.parse_args()
    world, local = int(os.environ.get("WORLD_SIZE", 1)), int(os.environ.get("LOCAL_RANK", 0))
    if world != 2 or "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("raw periodic DLM training requires its two-A800 allocation")
    torch.cuda.set_device(local)
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    dist.init_process_group("nccl")
    rank, device = dist.get_rank(), torch.device("cuda", local)
    model, tokenizer = initialize_fresh_periodic_repair_model(args.model_path, device)
    trainable = set_fresh_repair_trainable(model)
    checkpointing = enable_native_checkpointing(model.base_model)
    constraints = build_dynamic_lightweight_constraints(
        tokenizer, duplicate_coordinate_mask=True, lattice_volume_mask=True, min_lattice_rad=1e-4,
        canonicalize_periodic_alias=True, pbc_min_distance_mask=True, pbc_min_distance_A=.5, pbc_image_radius=2)
    data = PeriodicBaseTrainingDataset(args.train_jsonl, tokenizer, constraints, seed=args.data_seed)
    val = PeriodicBaseTrainingDataset(args.val_jsonl, tokenizer, constraints, seed=args.data_seed,
                                      expected_rows=9047, expected_split="val")
    sampler = DistributedSampler(data, num_replicas=world, rank=rank, shuffle=True, seed=args.data_seed)
    loader = DataLoader(data, batch_size=1, sampler=sampler, num_workers=1, collate_fn=list)
    val_indices = list(range(rank, 200, world))
    val_loader = DataLoader(Subset(val, val_indices), batch_size=1, num_workers=0, collate_fn=list)
    wrapped = DistributedDataParallel(model, device_ids=[local])
    objective = PeriodicBaseObjective(tokenizer, constraints, device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=5e-5, weight_decay=0.)
    accumulation = 8  # two ranks * microbatch1 * accumulation8 = 16 states
    epoch_updates = len(data) // 16
    updates = 2 * epoch_updates
    if updates != 6784 or data.real_length != 54272 or len(data) != data.real_length:
        raise ValueError("complete two-view/two-epoch source accounting changed")
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: two_stage_lr_multiplier(
        step, total_steps=updates, stage_boundary=epoch_updates, stage1_warmup=100,
        stage1_min_ratio=.2, stage2_base_ratio=.2, stage2_warmup=100, stage2_min_ratio=.1))
    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        config = {
            **vars(args), "initialization": model.raw_initialization, "trainable": trainable,
            "world_size": 2, "batch_size": 1, "gradient_accumulation": 8, "effective_batch": 16,
            "epochs": 2, "updates": updates, "epoch_updates": epoch_updates,
            "lr_stage1": 5e-5, "lr_stage2": 1e-5, "warmup_per_stage": 100,
            "weight_decay": 0., "data_protocol": BASE_TRAINING_DATA_PROTOCOL,
            "data_hashes": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                            for path in (args.train_jsonl, args.val_jsonl)},
            "train_sources": 27136, "validation_sources": 9047,
            "validation_monitor_sources": 100, "validation_monitor_states": 200,
            "alias_rows": data.alias_rows, "checkpointing": checkpointing,
            "old_dlm_weights_loaded": False, "K4_K8_data_read": False,
            "schema_temperature": 1., "legal_temperature": .7, "legal_loss_weight": .25,
            "numeric_smoothing_weight": .1, "max_length": 382,
            "native_deployment": "construct_then_full_cell_repair",
        }
        (args.output_dir / "training_config.json").write_text(json.dumps(config, indent=2, default=str) + "\n")
    dist.barrier()
    optimizer.zero_grad(set_to_none=True)
    start, step, initial_delta = time.monotonic(), 0, None
    totals, block, seen_gradients = Counter(), Counter(), Counter()
    source_index = {int(row["source_row_idx"]): i for i, row in enumerate(data.rows)}
    coverage = torch.zeros((len(data.rows), 2), dtype=torch.long)
    for epoch in range(2):
        data.epoch = epoch
        sampler.set_epoch(epoch)
        model.train()
        for microstep, examples in enumerate(loader):
            batch = materialize_state_batch(examples, tokenizer, device=device, max_length=382)
            if initial_delta is None:
                model.eval()
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    expected = model.base_model(
                        input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                        attention_bias=torch.zeros(1, 1, batch["input_ids"].shape[1],
                                                   batch["input_ids"].shape[1], device=device)).logits
                    actual = model(batch["input_ids"], attention_mask=batch["attention_mask"],
                                   geometry_context=batch["geometry_context"]).logits
                    difference = (expected - actual).abs().max().detach().float()
                dist.all_reduce(difference, op=dist.ReduceOp.MAX)
                initial_delta = float(difference)
                if initial_delta > 1e-5:
                    raise RuntimeError(f"raw zero-increment equality failed: {initial_delta}")
                del expected, actual
                model.train()
                if rank == 0:
                    print(json.dumps({"event": "raw_initialization_verified", "zero_logit_delta": initial_delta,
                                      "old_checkpoint": None, "new_tokens": model.raw_initialization["new_token_count"],
                                      "trainable": trainable}), flush=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = wrapped(batch["input_ids"], attention_mask=batch["attention_mask"],
                                 geometry_context=batch["geometry_context"]).logits
                loss, metrics, conflicts = objective(logits, batch, sigma_bins=.75 if epoch == 0 else .25)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("nonfinite raw periodic training loss")
            (loss / accumulation).backward()
            block.update({key: value for key, value in metrics.items()})
            block["loss"] += float(loss.detach()) / accumulation
            for example in examples:
                if not example["is_padding"]:
                    coverage[source_index[int(example["source_row_idx"])], example["view"]] += 1
                    totals["real_states"] += 1
                    totals["construct_states" if example["view"] == 0 else "repair_states"] += 1
                    totals["empty_states"] += int(example["empty_supervision"])
                    totals["corruption_fallbacks"] += int(example["structured_corruption_fallback"])
            totals["legal_conflicts"] += len(conflicts)
            if conflicts:
                with (args.output_dir / f"support_conflicts.rank{rank}.jsonl").open("a") as handle:
                    for conflict in conflicts:
                        handle.write(json.dumps(conflict) + "\n")
            if (microstep + 1) % accumulation:
                continue
            if step < 10:
                for name, module in {**model.repair_modules(), "state_conditioner": model.state_conditioner}.items():
                    seen_gradients[name] += float(sum(
                        parameter.grad.float().abs().sum() for parameter in module.parameters()
                        if parameter.grad is not None))
                for name, parameter in (("new_input_rows", model.new_token_rows.input_delta),
                                        ("new_output_rows", model.new_token_rows.output_delta)):
                    if parameter.grad is not None:
                        seen_gradients[name] += float(parameter.grad.float().abs().sum())
            gradient = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step == 10:
                if any(value <= 0 for value in seen_gradients.values()):
                    raise RuntimeError(f"a new module received no gradient in the first ten updates: {seen_gradients}")
                if rank == 0:
                    (args.output_dir / "EARLY_GRADIENT_CHECK.json").write_text(json.dumps(dict(seen_gradients), indent=2) + "\n")
            if rank == 0:
                event = dict(block, event="train", step=step, epoch=epoch + 1, grad_norm=float(gradient),
                             lr=scheduler.get_last_lr()[0], elapsed_seconds=time.monotonic() - start,
                             peak_memory_GiB=torch.cuda.max_memory_allocated() / 2**30)
                with (args.output_dir / "training_log.jsonl").open("a") as handle:
                    handle.write(json.dumps(event) + "\n")
                if step <= 3 or step % 10 == 0:
                    print(json.dumps(event), flush=True)
            block = Counter()
        model.eval()
        val.epoch = 0  # frozen monitor, no checkpoint selection
        val_sum = torch.zeros(2, device=device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for examples in val_loader:
                batch = materialize_state_batch(examples, tokenizer, device=device, max_length=382)
                logits = model(batch["input_ids"], attention_mask=batch["attention_mask"],
                               geometry_context=batch["geometry_context"]).logits
                value, _, _ = objective(logits, batch, sigma_bins=.25)
                val_sum += torch.tensor([float(value), len(examples)], device=device)
        dist.all_reduce(val_sum)
        if rank == 0:
            print(json.dumps({"event": "validation_monitor", "step": step, "states": int(val_sum[1]),
                              "sources": 100, "loss": float(val_sum[0] / val_sum[1])}), flush=True)
    all_coverage = coverage.to(device)
    dist.all_reduce(all_coverage)
    if step != updates or not bool((all_coverage == 2).all()):
        raise RuntimeError("complete original source/view/epoch coverage was not achieved")
    keys = sorted(totals)
    counts = torch.tensor([totals[key] for key in keys], device=device, dtype=torch.long)
    dist.all_reduce(counts)
    dist.barrier()
    if rank == 0:
        destination = args.output_dir / "checkpoints" / f"step-{step}"
        model.save_pretrained(destination)
        tokenizer.save_pretrained(destination)
        report = {
            "method": "periodic_dlm_from_original_llada_v1", "eligible_policy": True,
            "policy_path": str(destination), "initialization": model.raw_initialization,
            "updates": step, "epochs": 2, "source_rows": len(data.rows),
            "views_per_source_epoch": 2, "coverage_min": int(all_coverage.min()),
            "coverage_max": int(all_coverage.max()), "totals": dict(zip(keys, counts.tolist())),
            "zero_increment_max_logit_delta": initial_delta, "new_module_gradients": dict(seen_gradients),
            "elapsed_seconds": time.monotonic() - start, "K4_K8_data_read": False,
            "old_dlm_weights_loaded": False, "native_full_cell_repair_required": True,
        }
        (args.output_dir / "TRAIN_FINAL.json").write_text(json.dumps(report, indent=2) + "\n")
        (args.output_dir / "POLICY_PATH").write_text(str(destination) + "\n")
        (args.output_dir / "_SUCCESS").touch()
        print(json.dumps(report), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
