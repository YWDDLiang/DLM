#!/usr/bin/env python3
"""Frozen 400-update Planner training for PlanGraph-DLM G1."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Iterable

SOURCE_ROOT = Path(__file__).resolve().parent
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

from crystal_dlm.h1_llm_planner import (
    disable_peft_bnb_autodetect,
    ensure_peft_cache_compat,
    load_llama3_compatible_config,
)
from g1_protocol import sha256_file, write_json


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError(f"non-object row in {path}")
                yield payload


class PlannerDataset(Dataset):
    def __init__(
        self,
        path: Path,
        tokenizer,
        *,
        max_length: int,
        limit: int = 0,
    ) -> None:
        rows = list(iter_jsonl(path))
        self.rows = rows[: int(limit)] if int(limit) > 0 else rows
        if not self.rows:
            raise ValueError(f"no rows in {path}")
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.maximum_observed_length = 0
        for row in self.rows:
            prompt_ids, answer_ids = self._token_ids(row)
            self.maximum_observed_length = max(
                self.maximum_observed_length,
                len(prompt_ids) + len(answer_ids),
            )
            if len(answer_ids) >= self.max_length:
                raise ValueError(
                    f"answer length {len(answer_ids)} does not fit max_length={self.max_length}"
                )

    def _token_ids(self, row: dict[str, Any]) -> tuple[list[int], list[int]]:
        prompt = str(row["prompt"])
        answer = str(row["answer"]).strip()
        eos = self.tokenizer.eos_token or ""
        prompt_ids = self.tokenizer(
            prompt,
            add_special_tokens=False,
        )["input_ids"]
        answer_ids = self.tokenizer(
            answer + eos,
            add_special_tokens=False,
        )["input_ids"]
        return list(prompt_ids), list(answer_ids)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        prompt_ids, answer_ids = self._token_ids(row)
        max_prompt = self.max_length - len(answer_ids)
        if len(prompt_ids) > max_prompt:
            prompt_ids = prompt_ids[-max_prompt:]
        input_ids = prompt_ids + answer_ids
        labels = [-100] * len(prompt_ids) + answer_ids
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "sample_weight": float(row.get("sample_weight", 1.0) or 1.0),
        }


def collate(
    batch: list[dict[str, Any]],
    *,
    pad_token_id: int,
) -> dict[str, torch.Tensor]:
    maximum = max(int(row["input_ids"].numel()) for row in batch)
    input_ids = torch.full(
        (len(batch), maximum),
        int(pad_token_id),
        dtype=torch.long,
    )
    attention_mask = torch.zeros((len(batch), maximum), dtype=torch.long)
    labels = torch.full((len(batch), maximum), -100, dtype=torch.long)
    weights = torch.ones((len(batch),), dtype=torch.float32)
    for index, row in enumerate(batch):
        length = int(row["input_ids"].numel())
        input_ids[index, :length] = row["input_ids"]
        attention_mask[index, :length] = 1
        labels[index, :length] = row["labels"]
        weights[index] = float(row["sample_weight"])
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "sample_weight": weights,
    }


def weighted_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    shifted_logits = logits[:, :-1, :].contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    token_losses = torch.nn.functional.cross_entropy(
        shifted_logits.view(-1, shifted_logits.shape[-1]),
        shifted_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view(shifted_labels.shape)
    mask = (shifted_labels != -100).to(token_losses.dtype)
    per_row = (token_losses * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    active_weights = weights.to(device=per_row.device, dtype=per_row.dtype)
    return (per_row * active_weights).sum() / active_weights.sum().clamp_min(1.0)


@torch.no_grad()
def evaluate(model, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total = 0.0
    count = 0
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        output = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        loss = weighted_loss(
            output.logits,
            batch["labels"],
            batch["sample_weight"],
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite validation loss")
        total += float(loss.item())
        count += 1
    model.train()
    return total / max(1, count)


def save_final(model, tokenizer, output_dir: Path, *, step: int) -> tuple[Path, str]:
    checkpoint = output_dir / f"checkpoint-{int(step):06d}"
    checkpoint.mkdir(exist_ok=False)
    model.save_pretrained(checkpoint)
    tokenizer.save_pretrained(checkpoint)
    files = []
    for path in sorted(checkpoint.rglob("*")):
        if path.is_file():
            files.append(
                {
                    "path": str(path.relative_to(checkpoint)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    manifest_path = checkpoint / "checkpoint_manifest.json"
    write_json(
        manifest_path,
        {
            "schema": "plangraph-dlm-g1-checkpoint@1",
            "step": int(step),
            "files": files,
        },
    )
    return checkpoint, sha256_file(manifest_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("PG", "PG-shuffle"), required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--initial-adapter-sha256", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-updates", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-6)
    parser.add_argument("--warmup-updates", type=int, default=25)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--eval-rows", type=int, default=64)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    frozen = {
        "max_updates": 400,
        "batch_size": 1,
        "grad_accum": 8,
        "learning_rate": 2e-6,
        "warmup_updates": 25,
        "weight_decay": 0.0,
        "eval_steps": 50,
        "seed": 17,
    }
    for name, expected in frozen.items():
        observed = getattr(args, name)
        if observed != expected:
            raise ValueError(f"G1 freezes {name}={expected}, got {observed}")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing existing output {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if not torch.cuda.is_available():
        raise RuntimeError("G1 Planner training requires one CUDA device")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"G1 Planner training requires exactly one visible GPU, got {torch.cuda.device_count()}"
        )
    torch.cuda.manual_seed_all(int(args.seed))
    torch.cuda.reset_peak_memory_stats()
    device = torch.device("cuda")

    tokenizer_source = (
        args.checkpoint_path
        if (Path(args.checkpoint_path) / "tokenizer_config.json").exists()
        else args.model_path
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    config = load_llama3_compatible_config(args.model_path)
    load_started = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    ensure_peft_cache_compat()
    disable_peft_bnb_autodetect()
    from peft import PeftModel

    model = PeftModel.from_pretrained(
        model,
        args.checkpoint_path,
        is_trainable=True,
    )
    model.to(device)
    model.train()
    model_load_seconds = time.time() - load_started

    train_dataset = PlannerDataset(
        args.data_dir / "train.jsonl",
        tokenizer,
        max_length=int(args.max_length),
    )
    validation_dataset = PlannerDataset(
        args.data_dir / "val.jsonl",
        tokenizer,
        max_length=int(args.max_length),
        limit=int(args.eval_rows),
    )
    generator = torch.Generator()
    generator.manual_seed(int(args.seed))
    collator = lambda batch: collate(  # noqa: E731
        batch,
        pad_token_id=int(tokenizer.pad_token_id),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        generator=generator,
        num_workers=0,
        collate_fn=collator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=collator,
    )
    required_microbatches = int(args.max_updates) * int(args.grad_accum)
    if len(train_loader) < required_microbatches:
        raise ValueError(
            f"train loader has {len(train_loader)} rows but G1 needs "
            f"{required_microbatches} without cycling"
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_updates),
        num_training_steps=int(args.max_updates),
    )
    train_config = {
        "schema": "plangraph-dlm-g1-training-config@1",
        "arm": args.arm,
        "model_path": args.model_path,
        "checkpoint_path": args.checkpoint_path,
        "initial_adapter_sha256": args.initial_adapter_sha256,
        "data_dir": str(args.data_dir),
        "train_rows": len(train_dataset),
        "validation_rows": len(validation_dataset),
        "max_observed_train_tokens": train_dataset.maximum_observed_length,
        "max_observed_validation_tokens": validation_dataset.maximum_observed_length,
        "max_length": int(args.max_length),
        **frozen,
        "eval_rows": int(args.eval_rows),
        "precision": "bf16",
        "gradient_checkpointing": True,
        "gradient_clip": 1.0,
        "checkpoint_selection": False,
        "final_checkpoint_only": True,
        "energy_or_sun_signal": False,
    }
    write_json(args.output_dir / "train_config.json", train_config)

    events: list[dict[str, Any]] = []
    train_losses: list[float] = []
    gradient_norms: list[float] = []
    validation_losses: list[float] = []
    started = time.time()
    update = 0
    microbatch = 0
    recent = 0.0
    model.zero_grad(set_to_none=True)
    for batch in train_loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        output = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        loss = weighted_loss(
            output.logits,
            batch["labels"],
            batch["sample_weight"],
        )
        if not torch.isfinite(loss) or float(loss.item()) <= 0.0:
            raise FloatingPointError(f"invalid train loss at microbatch {microbatch}")
        (loss / int(args.grad_accum)).backward()
        train_losses.append(float(loss.item()))
        recent += float(loss.item())
        microbatch += 1
        if microbatch % int(args.grad_accum) != 0:
            continue
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        norm_value = float(norm.item() if hasattr(norm, "item") else norm)
        if not math.isfinite(norm_value) or norm_value <= 0.0:
            raise FloatingPointError(f"invalid gradient norm at update {update + 1}")
        gradient_norms.append(norm_value)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        update += 1
        if update == 1 or update % int(args.logging_steps) == 0:
            event = {
                "event": "train",
                "arm": args.arm,
                "update": update,
                "microbatches": microbatch,
                "recent_loss": recent
                / (
                    int(args.grad_accum)
                    * (1 if update == 1 else int(args.logging_steps))
                ),
                "gradient_norm": norm_value,
                "learning_rate": float(scheduler.get_last_lr()[0]),
                "elapsed_seconds": time.time() - started,
            }
            recent = 0.0
            print(json.dumps(event, sort_keys=True), flush=True)
            events.append(event)
        if update % int(args.eval_steps) == 0:
            value = evaluate(model, validation_loader, device)
            validation_losses.append(value)
            event = {
                "event": "validation",
                "arm": args.arm,
                "update": update,
                "validation_loss": value,
                "elapsed_seconds": time.time() - started,
            }
            print(json.dumps(event, sort_keys=True), flush=True)
            events.append(event)
        if update >= int(args.max_updates):
            break
    if update != int(args.max_updates) or microbatch != required_microbatches:
        raise RuntimeError(
            f"training schedule mismatch updates={update} microbatches={microbatch}"
        )
    checkpoint_dir, checkpoint_manifest_sha = save_final(
        model,
        tokenizer,
        args.output_dir,
        step=update,
    )
    report = {
        "schema": "plangraph-dlm-g1-training-report@1",
        "status": "complete",
        "arm": args.arm,
        "optimizer_updates": update,
        "train_microbatches": microbatch,
        "train_loss_min": min(train_losses),
        "train_loss_max": max(train_losses),
        "gradient_norm_min": min(gradient_norms),
        "gradient_norm_max": max(gradient_norms),
        "validation_losses": validation_losses,
        "validation_updates": [
            int(event["update"])
            for event in events
            if event["event"] == "validation"
        ],
        "all_numeric_finite_positive": all(
            math.isfinite(value) and value > 0.0
            for value in [*train_losses, *gradient_norms, *validation_losses]
        ),
        "model_load_seconds": model_load_seconds,
        "total_seconds": time.time() - started,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_manifest_sha256": checkpoint_manifest_sha,
        "events": events,
        "scientific_selection_role": "fixed_final_checkpoint_for_G1",
        "checkpoint_selection": False,
        "retry": False,
        "replacement": False,
        "repair": False,
        "filter": False,
        "rerank": False,
    }
    if not report["all_numeric_finite_positive"]:
        raise FloatingPointError("G1 training report contains non-positive numerics")
    write_json(args.output_dir / "training_report.json", report)
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

