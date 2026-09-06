#!/usr/bin/env python3
"""Two declared repair stages: completed K4 now, then full K4+K8 feedback."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import sys
import time

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Dataset, DataLoader, DistributedSampler

from crystal_dlm.periodic_repair_model import load_repair_model, set_repair_trainable
from crystal_dlm.programmed_path_data import load_path_model, read_jsonl
from crystal_dlm.state_training import enable_native_checkpointing, materialize_state_batch
from crystal_dlm.terminal_repair_data import (
    TARGET_ADMISSION, STRUCTURED_NOISE_LEVELS, make_terminal_repair_example,
)
from crystal_dlm.terminal_repair_objective import RepairObjective
from scripts.sample_llada_dynamic_crystals import build_dynamic_lightweight_constraints
from scripts.train_state_conditioned_spad import RevisionDataset


class TerminalRepairDataset(Dataset):
    def __init__(self, path, tokenizer, constraints, seed, effective_batch):
        self.rows = read_jsonl(path)
        if not self.rows or any(row["repair_split"] != "train" or not row["target_supervision_ready"]
                                for row in self.rows):
            raise ValueError("repair training requires admitted train-only targets")
        self.tokenizer, self.constraints, self.seed = tokenizer, constraints, seed
        self.epoch = 0
        self.real_length = 3 * len(self.rows)
        self.padded_length = math.ceil(self.real_length / effective_batch) * effective_batch

    def __len__(self):
        return self.padded_length

    def __getitem__(self, index):
        padding = index >= self.real_length
        source = index % self.real_length
        pair = self.rows[source // 3]
        from crystal_dlm.fixed_slot import MASK_TOKEN_ID
        row = make_terminal_repair_example(
            pair, family=source % 3, epoch=self.epoch, seed=self.seed, mask_id=MASK_TOKEN_ID,
            tokenizer=self.tokenizer, constraints=self.constraints,
        )
        row["is_padding"] = padding
        if padding:
            row["sample_weight"] = 0.
        else:
            row["sample_weight"] *= self.padded_length / self.real_length
        row["reference_state"]["sample_weight"] = row["sample_weight"]
        return row


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--checkpoint-path", type=Path, required=True)
    p.add_argument("--reference-checkpoint-path", type=Path, required=True)
    p.add_argument("--repair-round", type=int, choices=(0, 1), required=True)
    p.add_argument("--data-jsonl", type=Path, required=True)
    p.add_argument("--ce-data-jsonl", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--effective-batch", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--new-module-learning-rate", type=float, default=1e-4)
    p.add_argument("--kl-coefficient", type=float, default=.01)
    p.add_argument("--seed", type=int, default=20260906)
    p.add_argument("--engineering-updates", type=int, default=0)
    return p.parse_args()


def main():
    args = arguments()
    if "SLURM_JOB_ID" not in os.environ or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("repair training requires its declared two-GPU allocation")
    world, local = int(os.environ.get("WORLD_SIZE", 1)), int(os.environ.get("LOCAL_RANK", 0))
    if world != 2 or args.effective_batch % (world * args.batch_size):
        raise ValueError("registered repair training uses exactly two GPUs and global batch 16")
    base_report_path = args.checkpoint_path.parent.parent / "TRAIN_FINAL.json"
    if not base_report_path.is_file():
        raise ValueError("the preceding completed training report is required")
    base_report = json.loads(base_report_path.read_text())
    if base_report.get("eligible_policy") is not True:
        raise ValueError("engineering checkpoints cannot initialize scientific repair training")
    if args.repair_round == 0 and (base_report.get("collection_round") != 0
                                   or (args.checkpoint_path / "periodic_repair_config.json").is_file()):
        raise ValueError("first repair round starts from the completed original K4 policy")
    if args.repair_round == 1 and (base_report.get("method") != "periodic_discrete_self_repair_v1"
                                   or base_report.get("repair_collection_round") != 0):
        raise ValueError("second repair round continues the completed first repair model")
    reference_report = json.loads((args.reference_checkpoint_path.parent.parent / "TRAIN_FINAL.json").read_text())
    if (reference_report.get("eligible_policy") is not True
            or reference_report.get("collection_round") != args.repair_round):
        raise ValueError("reference must be original K4 for round0 and final K8 for round1")
    data_report = json.loads((args.data_jsonl.parent / "DATA_FINAL.json").read_text())
    if not (args.data_jsonl.parent / "_SUCCESS").is_file() or data_report["target_admission"] != TARGET_ADMISSION:
        raise ValueError("target data is incomplete or its frozen admission changed")
    if data_report["parent_requests"] != (4096 if args.repair_round == 0 else 12288):
        raise ValueError("the repair round does not match its complete source denominator")
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    model, tokenizer = load_repair_model(args.model_path, args.checkpoint_path, device, trainable=True)
    counts = set_repair_trainable(model)
    checkpointing = enable_native_checkpointing(model.base_model)
    reference, _ = load_path_model(args.model_path, args.reference_checkpoint_path, device)
    reference.eval().requires_grad_(False)
    constraints = build_dynamic_lightweight_constraints(
        tokenizer, duplicate_coordinate_mask=True, lattice_volume_mask=True, min_lattice_rad=1e-4,
        canonicalize_periodic_alias=True, pbc_min_distance_mask=True,
        pbc_min_distance_A=.5, pbc_image_radius=2,
    )
    dataset = TerminalRepairDataset(args.data_jsonl, tokenizer, constraints, args.seed, args.effective_batch)
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True, seed=args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                        num_workers=1, collate_fn=list)
    anchors = RevisionDataset(args.ce_data_jsonl, tokenizer, constraints, 20260905)
    if len(anchors) != 27136:
        raise ValueError("the original complete MP20 anchor source changed")
    anchors.epoch = 1
    anchor_order = np.random.default_rng(20260905).permutation(len(anchors))
    objective = RepairObjective(tokenizer, constraints, device)
    wrapped = DistributedDataParallel(model, device_ids=[local])
    new_prefixes = tuple(name + "." for name in model.repair_modules())
    old_parameters, new_parameters = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (new_parameters if name.startswith(new_prefixes) else old_parameters).append(parameter)
    parameters = old_parameters + new_parameters
    optimizer = torch.optim.AdamW([
        {"params": old_parameters, "lr": args.learning_rate},
        {"params": new_parameters, "lr": args.new_module_learning_rate},
    ], weight_decay=.01)
    accumulation = args.effective_batch // (world * args.batch_size)
    repair_updates = args.epochs * len(dataset) // args.effective_batch
    planned_updates = repair_updates + repair_updates // 4
    def schedule(step):
        if step < 50:
            return (step + 1) / 50
        fraction = min(1., (step - 50) / max(1, planned_updates - 50))
        return .1 + .9 * .5 * (1 + math.cos(math.pi * fraction))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        config = {
            **vars(args), "world_size": world, "trainable": counts, "checkpoint_modules": checkpointing,
            "repair_pair_count": len(dataset.rows), "real_scalar_states_per_epoch": dataset.real_length,
            "zero_weight_padding_per_epoch": len(dataset) - dataset.real_length,
            "planned_repair_updates": repair_updates, "planned_ce_updates": repair_updates // 4,
            "actual_reference_policy_kl": "KL(old || student), actual replay states, legal support, T=.7",
            "structured_noise_levels": STRUCTURED_NOISE_LEVELS,
            "real_error_fraction": .75, "numeric_ce_weight": .1, "family_weighting": "one third each",
            "supervision": "own-generated error -> separately admitted quantized own terminal",
            "additional_mp20_construction_supervision": False, "inference_mlip": False,
        }
        (args.output_dir / "training_config.json").write_text(json.dumps(config, indent=2, default=str) + "\n")
    dist.barrier()
    started = time.monotonic()
    completed = repair_completed = ce_completed = 0
    zero_delta = None
    event_totals = Counter()
    optimizer.zero_grad(set_to_none=True)

    def optimize():
        nonlocal completed
        gradient = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        completed += 1
        return float(gradient)

    def record(event):
        if rank == 0:
            event.update(step=completed, repair_updates=repair_completed, ce_updates=ce_completed,
                         elapsed_seconds=time.monotonic() - started,
                         lr=scheduler.get_last_lr(), peak_memory_GiB=torch.cuda.max_memory_allocated() / 2**30)
            with (args.output_dir / "training_log.jsonl").open("a") as handle:
                handle.write(json.dumps(event) + "\n")
            if completed <= 3 or completed % 10 == 0:
                print(json.dumps(event), flush=True)

    stop = False
    for epoch in range(args.epochs):
        dataset.epoch = epoch
        sampler.set_epoch(epoch)
        block = Counter()
        for microstep, examples in enumerate(loader):
            batch = materialize_state_batch(examples, tokenizer, device=device)
            reference_examples = [row["reference_state"] for row in examples]
            reference_batch = materialize_state_batch(reference_examples, tokenizer, device=device)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                old_logits = reference(
                    reference_batch["input_ids"], attention_mask=reference_batch["attention_mask"],
                    geometry_context=reference_batch["geometry_context"],
                    attention_bias=torch.zeros(len(examples), 1, reference_batch["input_ids"].shape[1],
                                               reference_batch["input_ids"].shape[1], device=device),
                ).logits
            with torch.autocast("cuda", dtype=torch.bfloat16):
                actual_logits = wrapped(
                    reference_batch["input_ids"], attention_mask=reference_batch["attention_mask"],
                    geometry_context=reference_batch["geometry_context"],
                ).logits
                if zero_delta is None and args.repair_round == 0:
                    delta = (actual_logits - old_logits).abs().max().detach().float()
                    dist.all_reduce(delta, op=dist.ReduceOp.MAX)
                    zero_delta = float(delta)
                    if zero_delta > 1e-5:
                        raise RuntimeError(f"zero-increment K8 reproduction failed: {zero_delta}")
                kl = objective.policy_kl(actual_logits, old_logits, reference_batch)
            # Finish this DDP forward/backward before the next one. This avoids
            # two outstanding reducer graphs and never backpropagates into old.
            (args.kl_coefficient * kl / accumulation).backward()
            del actual_logits, old_logits
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = wrapped(batch["input_ids"], attention_mask=batch["attention_mask"],
                                 geometry_context=batch["geometry_context"]).logits
                sigma = .75 - .5 * epoch / max(args.epochs - 1, 1)
                loss, stats = objective.terminal_ce(logits, batch, sigma_bins=sigma, smooth_weight=.1)
            if not bool(torch.isfinite(loss) & torch.isfinite(kl)):
                raise FloatingPointError("nonfinite repair CE or actual-policy KL")
            (loss / accumulation).backward()
            block.update(loss=float(loss.detach()) / accumulation, kl=float(kl.detach()) / accumulation)
            block.update({key: value / accumulation for key, value in stats.items()})
            event_totals.update(real_examples=sum(not row["is_padding"] for row in examples),
                                padding_examples=sum(row["is_padding"] for row in examples),
                                structured_examples=sum(row["phase"] == "structured_denoise" for row in examples),
                                structured_fallbacks=sum(row["structured_corruption_fallback"] for row in examples))
            if (microstep + 1) % accumulation:
                continue
            gradient = optimize()
            repair_completed += 1
            record(dict(block, kind="repair", epoch=epoch + 1, gradient=gradient, sigma_bins=sigma))
            block = Counter()
            if repair_completed % 4 == 0:
                ce_loss = 0.
                indices = [(ce_completed * args.effective_batch + i) % len(anchors)
                           for i in range(rank, args.effective_batch, world)]
                for offset in range(0, len(indices), args.batch_size):
                    rows = [anchors[int(anchor_order[index])] for index in indices[offset:offset + args.batch_size]]
                    anchor_batch = materialize_state_batch(rows, tokenizer, device=device)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        output = wrapped(anchor_batch["input_ids"], attention_mask=anchor_batch["attention_mask"],
                                         geometry_context=anchor_batch["geometry_context"]).logits
                        selected = output[torch.arange(len(rows), device=device), anchor_batch["positions"]].float()
                        ce = torch.nn.functional.cross_entropy(selected, anchor_batch["targets"])
                    (ce / accumulation).backward()
                    ce_loss += float(ce.detach()) / accumulation
                gradient = optimize()
                ce_completed += 1
                record(dict(kind="original_ce_anchor", epoch=epoch + 1, loss=ce_loss, gradient=gradient))
            if args.engineering_updates and completed >= args.engineering_updates:
                stop = True
                break
        if stop:
            break
    if not args.engineering_updates and (repair_completed != repair_updates or completed != planned_updates):
        raise RuntimeError("the registered number of full passes was not completed")
    totals = torch.tensor([event_totals[key] for key in sorted(event_totals)], device=device, dtype=torch.long)
    dist.all_reduce(totals)
    dist.barrier()
    if rank == 0:
        checkpoint_dir = args.output_dir / "checkpoints" / f"step-{completed}"
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        report = {
            "method": "periodic_discrete_self_repair_v1", "stage": "geometry_repair",
            "initial_policy": str(args.checkpoint_path),
            "reference_policy": str(args.reference_checkpoint_path),
            "repair_collection_round": args.repair_round, "policy_path": str(checkpoint_dir),
            "eligible_policy": not bool(args.engineering_updates),
            "complete_updates": completed, "repair_updates": repair_completed, "ce_updates": ce_completed,
            "zero_increment_max_logit_delta": zero_delta,
            "epochs": args.epochs, "train_pairs": len(dataset.rows),
            "totals": dict(zip(sorted(event_totals), totals.tolist())),
            "elapsed_seconds": time.monotonic() - started,
            "inference_mlip": False, "new_mp20_construction_supervision": False,
            "native_full_cell_repair_required": True,
            "stage_optimizer_restart": True,
        }
        (args.output_dir / "TRAIN_FINAL.json").write_text(json.dumps(report, indent=2) + "\n")
        (args.output_dir / "POLICY_PATH").write_text(str(checkpoint_dir) + "\n")
        (args.output_dir / "_SUCCESS").touch()
        print(json.dumps(report), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
