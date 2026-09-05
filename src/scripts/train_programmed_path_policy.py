#!/usr/bin/env python3
"""Fit the verified full-path teacher with retained LoRA and periodic state input."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import sys
import time

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from crystal_dlm.programmed_path_data import load_path_model, read_jsonl
from crystal_dlm.programmed_path_training import (
    PathLogProbability, minibatch_path_loss, sampled_training_examples,
)
from crystal_dlm.state_conditioned_model import set_state_lora_trainable
from crystal_dlm.state_training import enable_native_checkpointing, materialize_state_batch
from scripts.sample_llada_dynamic_crystals import build_dynamic_lightweight_constraints
from scripts.train_state_conditioned_spad import RevisionDataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--checkpoint-path", type=Path, required=True)
    p.add_argument("--teacher-json", type=Path, required=True)
    p.add_argument("--ce-data-jsonl", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--optimizer-state", type=Path)
    p.add_argument("--passes", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--effective-batch", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=5e-6)
    p.add_argument("--warmup-updates", type=int, default=100)
    p.add_argument("--schedule-updates", type=int, default=7680)
    p.add_argument("--seed", type=int, default=20260905)
    p.add_argument("--engineering-steps", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    if "SLURM_JOB_ID" not in os.environ or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("path policy training requires its declared allocation")
    if args.effective_batch != 16 or args.passes != 2:
        raise ValueError("retain the fixed effective batch16 and two complete passes")
    teacher = json.loads(args.teacher_json.read_text(encoding="utf-8"))
    summary, provenance = teacher["summary"], teacher["provenance"]
    if not summary.get("trainable_teacher") and not args.engineering_steps:
        raise ValueError("full pool has no certified positive-gain trainable teacher")
    normalizer_groups = summary.get("supervised_condition_count", summary["validated_groups"])
    if normalizer_groups < 1:
        raise ValueError("no conditions have actual path supervision")
    if args.checkpoint_path.resolve() != Path(provenance["checkpoint"]).resolve():
        raise ValueError("training must start at this collection round's reference policy")
    collection_round = int(provenance["collection_round"])
    if collection_round not in (0, 1) or (collection_round == 1 and args.optimizer_state is None):
        raise ValueError("only one refresh, continuing the original optimizer, is allowed")
    paths = [row for p in provenance["paths_jsonl"] for row in read_jsonl(p)]
    world, local = int(os.environ.get("WORLD_SIZE", 1)), int(os.environ.get("LOCAL_RANK", 0))
    if args.effective_batch % (world * args.batch_size):
        raise ValueError("world size and microbatch must preserve global batch16")
    accumulation = args.effective_batch // (world * args.batch_size)
    torch.cuda.set_device(local)
    torch.set_num_threads(2)
    device = torch.device("cuda", local)
    if world > 1:
        dist.init_process_group("nccl")
    rank = dist.get_rank() if world > 1 else 0
    torch.manual_seed(args.seed)
    model, tokenizer = load_path_model(args.model_path, args.checkpoint_path, device, trainable=True)
    counts = set_state_lora_trainable(model)
    checkpoint_modules = enable_native_checkpointing(model.base_model)
    if not checkpoint_modules:
        raise RuntimeError("native activation checkpointing was not enabled")
    model.train()
    wrapped = DistributedDataParallel(model, device_ids=[local]) if world > 1 else model
    constraints = build_dynamic_lightweight_constraints(
        tokenizer, duplicate_coordinate_mask=True, lattice_volume_mask=True, min_lattice_rad=1e-4,
        canonicalize_periodic_alias=True, pbc_min_distance_mask=True, pbc_min_distance_A=.5, pbc_image_radius=2,
    )
    scorer = PathLogProbability(tokenizer, constraints)
    anchors = RevisionDataset(args.ce_data_jsonl, tokenizer, constraints, args.seed)
    if len(anchors) != 27136:
        raise ValueError("CE anchors must come from retained full MP20 train")
    anchors.epoch = collection_round
    anchor_order = np.random.default_rng(args.seed).permutation(len(anchors))
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=.01)
    def lr_multiplier(step):
        if step < args.warmup_updates:
            return (step + 1) / max(1, args.warmup_updates)
        fraction = (step - args.warmup_updates) / max(1, args.schedule_updates - args.warmup_updates)
        return .1 + .9 * .5 * (1 + math.cos(math.pi * min(fraction, 1)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)
    global_updates = path_updates = ce_updates = 0
    if args.optimizer_state:
        saved = torch.load(args.optimizer_state, map_location="cpu", weights_only=False)
        for key in ("seed", "learning_rate", "schedule_updates", "warmup_updates"):
            if saved[key] != getattr(args, key):
                raise ValueError(f"refresh cannot silently change optimizer setting {key}")
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        global_updates, path_updates, ce_updates = (int(saved[k]) for k in ("global_updates", "path_updates", "ce_updates"))
    initial_updates = global_updates
    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        config = {**vars(args), "world_size": world, "gradient_accumulation": accumulation,
                  "trainable": counts, "checkpoint_modules": checkpoint_modules,
                  "collection_round": collection_round, "teacher_summary": summary,
                  "likelihood_dropout": 0., "ce_every_path_updates": 4,
                  "objective": "HT full-deployment path NLL, condition mean"}
        (args.output_dir / "training_config.json").write_text(json.dumps(config, indent=2, default=str) + "\n", encoding="utf-8")
    if world > 1:
        dist.barrier()
    started = time.monotonic()
    optimizer.zero_grad(set_to_none=True)
    pass_reports = []
    initial_replay_errors = []

    def update(examples, *, kind, padded_size, groups):
        block_loss = 0.
        for micro in range(accumulation):
            chunk = examples[micro * args.batch_size:(micro + 1) * args.batch_size]
            batch = materialize_state_batch(chunk, tokenizer, device=device)
            sync = wrapped.no_sync() if world > 1 and micro + 1 < accumulation else nullcontext()
            with sync:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = wrapped(batch["input_ids"], attention_mask=batch["attention_mask"],
                                     geometry_context=batch["geometry_context"]).logits
                    if kind == "path":
                        logp = scorer(logits, batch)
                        if global_updates == initial_updates:
                            recorded = torch.tensor([e["recorded_log_probability"] for e in chunk],
                                                    device=device, dtype=torch.float64)
                            initial_replay_errors.extend((logp.detach().double() - recorded).abs().cpu().tolist())
                        loss = minibatch_path_loss(logp, chunk, dataset_size=padded_size, validated_groups=groups)
                    else:
                        selected = logits[torch.arange(len(chunk), device=device), batch["positions"]].float()
                        loss = torch.nn.functional.cross_entropy(selected, batch["targets"])
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(f"nonfinite {kind} likelihood")
                (loss / accumulation).backward()
            block_loss += float(loss.detach()) / accumulation
        gradient = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        return block_loss, float(gradient)

    def log_event(kind, loss, gradient, pass_index):
        if rank == 0 and (global_updates - initial_updates <= 5 or global_updates % 10 == 0):
            event = {"global_update": global_updates, "path_updates": path_updates, "ce_updates": ce_updates,
                     "kind": kind, "group_pass": pass_index, "loss_rank0": loss, "gradient_norm": gradient,
                     "learning_rate": scheduler.get_last_lr()[0], "elapsed_seconds": time.monotonic() - started}
            print(json.dumps(event), flush=True)
            with (args.output_dir / "training_log.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event) + "\n")

    for group_pass in range(args.passes):
        pass_index = 2 * collection_round + group_pass
        examples = sampled_training_examples(paths, teacher, seed=args.seed, pass_index=pass_index)
        if not examples:
            raise ValueError("no positive-weight path states")
        real_size = len(examples)
        padded_size = math.ceil(real_size / args.effective_batch) * args.effective_batch
        # Ghost rows have ZERO teacher mass, not repeated positive trajectories.
        examples += [dict(examples[0], weight=0., inclusion_probability=1.) for _ in range(padded_size - real_size)]
        order = np.random.default_rng(np.random.SeedSequence([args.seed, pass_index, 71])).permutation(padded_size)
        processed = 0
        for begin in range(0, padded_size, args.effective_batch):
            local_ids = order[begin:begin + args.effective_batch][rank::world]
            loss, gradient = update([examples[int(i)] for i in local_ids], kind="path",
                                    padded_size=padded_size, groups=normalizer_groups)
            path_updates += 1
            global_updates += 1
            processed += len(local_ids)
            log_event("path", loss, gradient, pass_index)
            if path_updates % 4 == 0:
                source_ids = [(ce_updates * args.effective_batch + i) % len(anchors) for i in range(rank, args.effective_batch, world)]
                loss, gradient = update([anchors[int(anchor_order[i])] for i in source_ids], kind="ce",
                                        padded_size=0, groups=0)
                ce_updates += 1
                global_updates += 1
                log_event("ce", loss, gradient, pass_index)
            if args.engineering_steps and global_updates - initial_updates >= args.engineering_steps:
                break
        pass_reports.append({"pass_index": pass_index, "real_scalar_states": real_size, "padded_scalar_slots": padded_size,
                             "processed_scalar_slots": processed * world,
                             "complete_pass": processed * world == padded_size})
        if args.engineering_steps:
            break
    if world > 1:
        dist.barrier()
    replay_reports = [{"rank": rank, "decisions": len(initial_replay_errors),
                       "mean_error": float(np.mean(initial_replay_errors)),
                       "maximum_error": max(initial_replay_errors)}]
    if world > 1:
        gathered = [None] * world
        dist.all_gather_object(gathered, replay_reports[0])
        replay_reports = gathered
    if rank == 0:
        destination = args.output_dir / "checkpoints" / f"step-{global_updates}"
        model.save_pretrained(destination)
        tokenizer.save_pretrained(destination)
        training_state = {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                          "global_updates": global_updates, "path_updates": path_updates, "ce_updates": ce_updates,
                          **{key: getattr(args, key) for key in ("seed", "learning_rate", "schedule_updates", "warmup_updates")}}
        torch.save(training_state, destination / "POST_STATE.pt")
        report = {"phase": "dual_objective_full_path_policy", "collection_round": collection_round,
                  "eligible_policy": not bool(args.engineering_steps), "policy_path": str(destination),
                  "global_updates": global_updates, "updates_this_run": global_updates - initial_updates,
                  "path_updates": path_updates, "ce_updates": ce_updates, "passes": pass_reports,
                  "teacher_summary": summary, "trainable": counts, "elapsed_seconds": time.monotonic() - started}
        report["initial_minibatch_replay"] = replay_reports
        (args.output_dir / "TRAIN_FINAL.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        (args.output_dir / "POLICY_PATH").write_text(str(destination) + "\n", encoding="utf-8")
        (args.output_dir / "_SUCCESS").touch()
        print(json.dumps(report), flush=True)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
