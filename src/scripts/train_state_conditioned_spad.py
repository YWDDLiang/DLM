#!/usr/bin/env python3
"""Full-source MP20 warmup of existing LoRA plus explicit periodic state input."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from crystal_dlm.periodic_state_conditioning import PeriodicStateConfig
from crystal_dlm.state_conditioned_model import StateConditionedDLM, set_state_lora_trainable
from crystal_dlm.state_revision_data import make_state_revision_example
from crystal_dlm.state_training import enable_native_checkpointing, materialize_state_batch
from scripts.sample_llada_dynamic_crystals import build_dynamic_lightweight_constraints, load_model_and_tokenizer


class RevisionDataset(Dataset):
    def __init__(self, path, tokenizer, constraints, seed):
        with Path(path).open(encoding="utf-8") as handle:
            self.rows = [json.loads(line) for line in handle if line.strip()]
        self.tokenizer, self.constraints, self.seed, self.epoch = tokenizer, constraints, seed, 0
        self.alias_rows = 0
        self.prefixes = []
        for row in self.rows:
            answer = row["answer"]
            normalized = answer
            for axis in "XYZ":
                normalized = normalized.replace(f"<{axis}_100>", f"<{axis}_000>")
            self.alias_rows += int(answer != normalized)
            row["answer"] = normalized
            if row.get("source_split") not in (None, "train"):
                raise ValueError("warmup accepts MP20-train sources only")
            prefix = tokenizer(row["prompt"].rstrip() + "\n", add_special_tokens=False)["input_ids"]
            self.prefixes.append(prefix)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        example = make_state_revision_example(self.rows[index], self.tokenizer, self.constraints,
                                              seed=self.seed, epoch=self.epoch)
        example["prompt_token_ids"] = self.prefixes[index]
        return example


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--checkpoint-path", required=True)
    p.add_argument("--data-jsonl", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--expected-rows", type=int, default=27136)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--effective-batch", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=20260905)
    return p.parse_args()


def main():
    args = parse_args()
    if "SLURM_JOB_ID" not in os.environ or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("formal training requires its declared allocation")
    world, local = int(os.environ.get("WORLD_SIZE", 1)), int(os.environ.get("LOCAL_RANK", 0))
    if args.effective_batch % (world * args.batch_size):
        raise ValueError("use a compatible rank/microbatch count without changing global batch")
    accumulation = args.effective_batch // (world * args.batch_size)
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    if world > 1:
        dist.init_process_group("nccl")
    rank = dist.get_rank() if world > 1 else 0
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    base, tokenizer = load_model_and_tokenizer(args.model_path, args.checkpoint_path, device)
    model = StateConditionedDLM(base, tokenizer, PeriodicStateConfig(base.get_input_embeddings().weight.shape[1])).to(device)
    parameter_counts = set_state_lora_trainable(model)
    checkpoint_modules = enable_native_checkpointing(base)
    model.train()
    wrapped = DistributedDataParallel(model, device_ids=[local]) if world > 1 else model
    constraints = build_dynamic_lightweight_constraints(
        tokenizer, duplicate_coordinate_mask=True, lattice_volume_mask=True,
        min_lattice_rad=1e-4,
        canonicalize_periodic_alias=True, pbc_min_distance_mask=True,
        pbc_min_distance_A=.5, pbc_image_radius=2,
    )
    data = RevisionDataset(args.data_jsonl, tokenizer, constraints, args.seed)
    if len(data) != args.expected_rows or len(data) % args.effective_batch:
        raise ValueError("full-source row count/effective-batch accounting differs")
    sampler = DistributedSampler(data, num_replicas=world, rank=rank, shuffle=True, seed=args.seed)
    loader = DataLoader(data, batch_size=args.batch_size, sampler=sampler,
                        num_workers=args.num_workers, collate_fn=list, pin_memory=False)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=.01)
    updates = args.epochs * len(data) // args.effective_batch
    def lr_multiplier(step):
        if step < args.warmup_steps:
            return (step + 1) / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, updates - args.warmup_steps)
        return .1 + .9 * .5 * (1 + math.cos(math.pi * min(progress, 1)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)
    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        config = {**vars(args), "world_size": world, "accumulation": accumulation,
                  "updates": updates, "trainable": parameter_counts, "checkpoint_modules": checkpoint_modules,
                  "canonical_periodic_alias_rows": data.alias_rows,
                  "warmup_loss": "MP20_teacher_active_scalar_full_vocab_CE", "inference_mlip": False}
        (args.output_dir / "training_config.json").write_text(json.dumps(config, indent=2, default=str) + "\n", encoding="utf-8")
    if world > 1:
        dist.barrier()
    completed, local_sources, block_loss = 0, 0, 0.
    region_fallbacks = 0
    optimizer.zero_grad(set_to_none=True)
    start = time.monotonic()
    for epoch in range(args.epochs):
        data.epoch = epoch
        sampler.set_epoch(epoch)
        for microstep, examples in enumerate(loader):
            batch = materialize_state_batch(examples, tokenizer, device=device)
            local_sources += len(examples)
            region_fallbacks += sum(bool(e["corruption_info"].get("region_fallback")) for e in examples)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = wrapped(batch["input_ids"], attention_mask=batch["attention_mask"],
                                 geometry_context=batch["geometry_context"]).logits
                selected = logits[torch.arange(len(examples), device=device), batch["positions"]].float()
                loss = torch.nn.functional.cross_entropy(selected, batch["targets"])
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("nonfinite warmup CE")
            (loss / accumulation).backward()
            block_loss += float(loss.detach()) / accumulation
            if (microstep + 1) % accumulation:
                continue
            gradient = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            completed += 1
            if rank == 0 and (completed <= 3 or completed % 10 == 0):
                event = {"step": completed, "epoch": epoch + 1, "loss": block_loss,
                         "grad_norm": float(gradient), "lr": scheduler.get_last_lr()[0],
                         "elapsed_seconds": time.monotonic() - start,
                         "global_sources": local_sources * world}
                print(json.dumps(event), flush=True)
                with (args.output_dir / "training_log.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(event) + "\n")
            block_loss = 0.
    if completed != updates:
        raise RuntimeError(f"expected {updates} complete updates, got {completed}")
    if world > 1:
        dist.barrier()
    if rank == 0:
        checkpoint_dir = args.output_dir / "checkpoints" / f"step-{completed}"
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        report = {"phase": "state_conditioned_mp20_warmup", "complete_updates": completed,
                  "source_rows": len(data), "source_epochs": args.epochs, "world_size": world,
                  "trainable": parameter_counts, "elapsed_seconds": time.monotonic() - start,
                  "rank0_region_fallbacks": region_fallbacks, "policy_path": str(checkpoint_dir)}
        (args.output_dir / "TRAIN_FINAL.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        (args.output_dir / "POLICY_PATH").write_text(str(checkpoint_dir) + "\n", encoding="utf-8")
        (args.output_dir / "_SUCCESS").touch()
        print(json.dumps(report), flush=True)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
