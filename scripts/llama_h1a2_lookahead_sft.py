#!/usr/bin/env python3
"""Frozen P0 evaluation and P-control/P* LoRA training for H1-A2 V3."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

from crystal_dlm.a800_llama_compat import (  # noqa: E402
    disable_peft_bnb_autodetect,
    ensure_peft_cache_compat,
    load_llama3_compatible_config,
)
from crystal_dlm.h1a2_planner_batch import (  # noqa: E402
    PreparedPlannerExample,
    prepare_planner_example,
)
from crystal_dlm.h1a2_planner_objective import (  # noqa: E402
    FIELD_GROUP_NAMES,
    LOOKAHEAD_FIELDS,
    build_lookahead_heads,
    torch_field_balanced_loss,
    torch_lookahead_loss,
)


TRAINING_SCHEMA = "h1a2_lookahead_planner_training_v1"
OFFICIAL_TRAIN_ROWS = 3200
OFFICIAL_VAL_ROWS = 256
OFFICIAL_MAX_UPDATES = 400
OFFICIAL_EVAL_STEPS = 50
OFFICIAL_WARMUP_STEPS = 25
OFFICIAL_GRAD_ACCUM = 8
OFFICIAL_BATCH_SIZE = 1
OFFICIAL_MAX_LENGTH = 768
OFFICIAL_LR = 2e-6
OFFICIAL_SEED = 17


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise TypeError(f"{path}:{line_number} is not a JSON object")
            yield payload


def validate_data_directory(
    data_dir: Path,
    *,
    engineering_smoke: bool,
) -> tuple[dict[str, Any], dict[str, tuple[str, ...]]]:
    manifest_path = data_dir / "manifest.json"
    success_path = data_dir / "_SUCCESS"
    vocab_path = data_dir / "lookahead_vocabs.json"
    for path in (
        manifest_path,
        success_path,
        vocab_path,
        data_dir / "train.jsonl",
        data_dir / "val.jsonl",
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    success = json.loads(success_path.read_text(encoding="utf-8"))
    if success.get("status") != "complete":
        raise ValueError("Planner data _SUCCESS status is not complete")
    if success.get("manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("Planner data manifest SHA does not match _SUCCESS")
    output = manifest.get("output") or {}
    for key, filename in (
        ("train", "train.jsonl"),
        ("val", "val.jsonl"),
        ("lookahead_vocabs", "lookahead_vocabs.json"),
    ):
        expected = str((output.get(key) or {}).get("sha256") or "")
        observed = sha256_file(data_dir / filename)
        if expected != observed:
            raise ValueError(
                f"Planner data {key} SHA mismatch: expected {expected}, "
                f"observed {observed}"
            )
    selection = manifest.get("selection") or {}
    train_rows = int(selection.get("train_rows", 0))
    val_rows = int(selection.get("val_rows", 0))
    if engineering_smoke:
        if train_rows != 32 or val_rows != 32:
            raise ValueError("engineering smoke requires exactly 32/32 rows")
    elif train_rows != OFFICIAL_TRAIN_ROWS or val_rows != OFFICIAL_VAL_ROWS:
        raise ValueError(
            "official Planner training requires exactly "
            f"{OFFICIAL_TRAIN_ROWS}/{OFFICIAL_VAL_ROWS} rows"
        )
    if selection.get("same_order_for_arms") != ["P-control", "Pstar"]:
        raise ValueError("P-control/P* data-order contract changed")
    if bool(selection.get("replacement")):
        raise ValueError("Planner stream unexpectedly enables replacement")
    if bool(selection.get("validity_filter")):
        raise ValueError("Planner stream unexpectedly enables filtering")
    raw_vocabs = json.loads(vocab_path.read_text(encoding="utf-8"))
    vocabs: dict[str, tuple[str, ...]] = {}
    for field in LOOKAHEAD_FIELDS:
        labels = tuple(str(value) for value in raw_vocabs.get(field, ()))
        if not labels or labels != tuple(sorted(set(labels))):
            raise ValueError(f"invalid look-ahead vocabulary for {field!r}")
        vocabs[field] = labels
    return manifest, vocabs


class LookaheadPlannerDataset(Dataset):
    def __init__(
        self,
        path: Path,
        tokenizer: Any,
        *,
        max_length: int,
        lookahead_vocabs: Mapping[str, tuple[str, ...]],
    ) -> None:
        self.examples = [
            prepare_planner_example(
                row,
                tokenizer,
                max_length=max_length,
                lookahead_vocabs=lookahead_vocabs,
            )
            for row in iter_jsonl(path)
        ]
        if not self.examples:
            raise ValueError(f"{path} contains no Planner rows")
        identities = [
            example.source_line_sha256
            for example in self.examples
            if example.source_line_sha256 is not None
        ]
        if len(identities) != len(self.examples):
            raise ValueError(f"{path} has rows without source-line identities")
        if len(identities) != len(set(identities)):
            raise ValueError(f"{path} has duplicate source-line identities")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> PreparedPlannerExample:
        return self.examples[index]


def collate(
    examples: list[PreparedPlannerExample],
    *,
    pad_token_id: int,
) -> dict[str, Any]:
    max_length = max(len(example.input_ids) for example in examples)
    batch_size = len(examples)
    input_ids = torch.full(
        (batch_size, max_length),
        int(pad_token_id),
        dtype=torch.long,
    )
    attention_mask = torch.zeros((batch_size, max_length), dtype=torch.long)
    labels = torch.full((batch_size, max_length), -100, dtype=torch.long)
    group_ids = torch.full((batch_size, max_length), -1, dtype=torch.long)
    formula_positions = torch.empty((batch_size,), dtype=torch.long)
    lattice_positions = torch.empty((batch_size,), dtype=torch.long)
    lookahead_labels = {
        field: torch.empty((batch_size,), dtype=torch.long)
        for field in LOOKAHEAD_FIELDS
    }
    for row_index, example in enumerate(examples):
        length = len(example.input_ids)
        input_ids[row_index, :length] = torch.tensor(
            example.input_ids,
            dtype=torch.long,
        )
        attention_mask[row_index, :length] = 1
        labels[row_index, :length] = torch.tensor(
            example.labels,
            dtype=torch.long,
        )
        group_ids[row_index, :length] = torch.tensor(
            example.field_group_ids,
            dtype=torch.long,
        )
        formula_positions[row_index] = int(example.formula_boundary)
        lattice_positions[row_index] = int(example.lattice_boundary)
        for field in LOOKAHEAD_FIELDS:
            lookahead_labels[field][row_index] = int(
                example.lookahead_labels[field]
            )
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "field_group_ids": group_ids,
        "formula_positions": formula_positions,
        "lattice_positions": lattice_positions,
        "lookahead_labels": lookahead_labels,
    }


def batch_to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        "input_ids": batch["input_ids"].to(device),
        "attention_mask": batch["attention_mask"].to(device),
        "labels": batch["labels"].to(device),
        "field_group_ids": batch["field_group_ids"].to(device),
        "formula_positions": batch["formula_positions"].to(device),
        "lattice_positions": batch["lattice_positions"].to(device),
        "lookahead_labels": {
            field: values.to(device)
            for field, values in batch["lookahead_labels"].items()
        },
    }


def compute_objective(
    model: Any,
    batch: Mapping[str, Any],
    *,
    arm: str,
    lookahead_heads: Any | None,
) -> tuple[Any, dict[str, Any]]:
    use_lookahead = arm == "pstar"
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        output_hidden_states=use_lookahead,
        use_cache=False,
    )
    shift_logits = outputs.logits[:, :-1, :].contiguous()
    shift_labels = batch["labels"][:, 1:].contiguous()
    token_losses = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view(shift_labels.shape)
    supervision_mask = shift_labels != -100
    if not bool(torch.all(supervision_mask.any(dim=1)).item()):
        raise ValueError("a Planner sample has no supervised answer tokens")
    field_loss, field_diagnostics = torch_field_balanced_loss(
        token_losses,
        batch["field_group_ids"][:, 1:],
        supervision_mask,
    )
    target_nll = token_losses[supervision_mask].mean()
    metrics: dict[str, Any] = {
        "target_nll": target_nll,
        "field_loss": field_loss,
        **{
            f"field_{group}": field_diagnostics[group]
            for group in FIELD_GROUP_NAMES
        },
    }
    if use_lookahead:
        if lookahead_heads is None:
            raise ValueError("P* requires the registered look-ahead heads")
        if not outputs.hidden_states:
            raise ValueError("P* model did not return hidden states")
        lookahead_loss, lookahead_diagnostics = torch_lookahead_loss(
            outputs.hidden_states[-1],
            {
                "formula": batch["formula_positions"],
                "lattice": batch["lattice_positions"],
            },
            batch["lookahead_labels"],
            lookahead_heads,
        )
        total = 0.80 * field_loss + 0.20 * lookahead_loss
        metrics["lookahead_loss"] = lookahead_loss
        metrics.update(
            {
                key: value
                for key, value in lookahead_diagnostics.items()
                if key != "lookahead"
            }
        )
    else:
        total = field_loss
        metrics["lookahead_loss"] = field_loss.detach().new_tensor(float("nan"))
    metrics["total_loss"] = total
    return total, metrics


@torch.no_grad()
def evaluate(
    model: Any,
    loader: DataLoader,
    device: torch.device,
    *,
    arm: str,
    lookahead_heads: Any | None,
) -> dict[str, float]:
    model.eval()
    if lookahead_heads is not None:
        lookahead_heads.eval()
    totals: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    rows = 0
    for raw_batch in loader:
        batch = batch_to_device(raw_batch, device)
        _, metrics = compute_objective(
            model,
            batch,
            arm=arm,
            lookahead_heads=lookahead_heads,
        )
        rows += int(batch["input_ids"].shape[0])
        for key, value in metrics.items():
            observed = float(value.detach().float().item())
            if not torch.isfinite(value.detach()).item():
                if key == "lookahead_loss" and arm != "pstar":
                    continue
                raise FloatingPointError(f"non-finite evaluation metric {key}")
            totals[key] += observed
            counts[key] += 1
    model.train(arm != "p0")
    if lookahead_heads is not None:
        lookahead_heads.train(arm == "pstar")
    result = {
        key: totals[key] / counts[key]
        for key in sorted(totals)
        if counts[key]
    }
    result["rows"] = float(rows)
    return result


def file_inventory(root: Path) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name == "checkpoint_manifest.json":
            continue
        inventory.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return inventory


def publish_checkpoint(
    output_dir: Path,
    *,
    step: int,
    arm: str,
    model: Any,
    tokenizer: Any,
    lookahead_heads: Any | None,
    lookahead_vocabs: Mapping[str, tuple[str, ...]],
    metrics: Mapping[str, float],
) -> dict[str, Any]:
    target = output_dir / f"checkpoint-{int(step):04d}"
    temporary = output_dir / f".checkpoint-{int(step):04d}.tmp"
    if target.exists() or temporary.exists():
        raise FileExistsError(target if target.exists() else temporary)
    temporary.mkdir()
    adapter_dir = temporary / "adapter"
    tokenizer_dir = temporary / "tokenizer"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(tokenizer_dir)
    head_record: dict[str, Any] | None = None
    if lookahead_heads is not None:
        head_path = temporary / "lookahead_heads.pt"
        torch.save(lookahead_heads.state_dict(), head_path)
        head_config_path = temporary / "lookahead_head_config.json"
        head_config_path.write_text(
            json.dumps(
                {
                    "seed": OFFICIAL_SEED,
                    "vocabs": {
                        key: list(value)
                        for key, value in sorted(lookahead_vocabs.items())
                    },
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        head_record = {
            "state_path": head_path.name,
            "state_sha256": sha256_file(head_path),
            "config_path": head_config_path.name,
            "config_sha256": sha256_file(head_config_path),
            "discarded_for_inference": True,
        }
    manifest = {
        "schema": TRAINING_SCHEMA,
        "status": "complete",
        "arm": arm,
        "optimizer_step": int(step),
        "metrics": {key: float(value) for key, value in sorted(metrics.items())},
        "lookahead_heads": head_record,
        "inventory": file_inventory(temporary),
    }
    (temporary / "checkpoint_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)
    return {
        "step": int(step),
        "path": target.name,
        "manifest_sha256": sha256_file(target / "checkpoint_manifest.json"),
        "metrics": manifest["metrics"],
    }


def append_event(path: Path, event: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps(dict(event), ensure_ascii=False, sort_keys=True) + "\n"
        )
        handle.flush()


def validate_args(args: argparse.Namespace) -> None:
    if args.arm not in {"p0", "pcontrol", "pstar"}:
        raise ValueError(f"unsupported Planner arm {args.arm!r}")
    if int(args.batch_size) != OFFICIAL_BATCH_SIZE:
        raise ValueError("registered Planner per-device batch is 1")
    if int(args.grad_accum) != OFFICIAL_GRAD_ACCUM:
        raise ValueError("registered Planner gradient accumulation is 8")
    if int(args.max_length) != OFFICIAL_MAX_LENGTH:
        raise ValueError("registered Planner maximum sequence length is 768")
    if int(args.seed) != OFFICIAL_SEED:
        raise ValueError("registered Planner seed is 17")
    if float(args.lr) != OFFICIAL_LR:
        raise ValueError("registered Planner learning rate is 2e-6")
    if float(args.weight_decay) != 0.0:
        raise ValueError("registered Planner weight decay is zero")
    expected_updates = 4 if args.engineering_smoke else OFFICIAL_MAX_UPDATES
    expected_eval = 4 if args.engineering_smoke else OFFICIAL_EVAL_STEPS
    expected_warmup = 1 if args.engineering_smoke else OFFICIAL_WARMUP_STEPS
    if int(args.max_updates) != expected_updates:
        raise ValueError(f"this mode requires max_updates={expected_updates}")
    if int(args.eval_steps) != expected_eval:
        raise ValueError(f"this mode requires eval_steps={expected_eval}")
    if int(args.warmup_steps) != expected_warmup:
        raise ValueError(f"this mode requires warmup_steps={expected_warmup}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("p0", "pcontrol", "pstar"), required=True)
    parser.add_argument(
        "--model-path",
        default="/public/home/jiaosz/ywliang/models/Meta-Llama-3-8B/",
    )
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=OFFICIAL_MAX_LENGTH)
    parser.add_argument("--batch-size", type=int, default=OFFICIAL_BATCH_SIZE)
    parser.add_argument("--grad-accum", type=int, default=OFFICIAL_GRAD_ACCUM)
    parser.add_argument("--max-updates", type=int, default=OFFICIAL_MAX_UPDATES)
    parser.add_argument("--lr", type=float, default=OFFICIAL_LR)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-steps", type=int, default=OFFICIAL_WARMUP_STEPS)
    parser.add_argument("--eval-steps", type=int, default=OFFICIAL_EVAL_STEPS)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=OFFICIAL_SEED)
    parser.add_argument("--engineering-smoke", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
    parser.add_argument(
        "--no-gradient-checkpointing",
        dest="gradient_checkpointing",
        action="store_false",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_args(args)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if not args.checkpoint_path.is_dir():
        raise FileNotFoundError(args.checkpoint_path)
    if not torch.cuda.is_available():
        raise RuntimeError("registered Planner execution requires one CUDA GPU")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("registered Planner execution requires BF16 support")
    args.output_dir.mkdir(parents=True)
    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed_all(int(args.seed))
    device = torch.device("cuda", 0)

    data_manifest, lookahead_vocabs = validate_data_directory(
        args.data_dir,
        engineering_smoke=bool(args.engineering_smoke),
    )
    tokenizer_source = (
        args.checkpoint_path
        if (args.checkpoint_path / "tokenizer_config.json").is_file()
        else args.model_path
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        trust_remote_code=True,
        use_fast=True,
    )
    if not tokenizer.is_fast:
        raise ValueError("registered Planner field loss requires a fast tokenizer")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_dataset = LookaheadPlannerDataset(
        args.data_dir / "train.jsonl",
        tokenizer,
        max_length=int(args.max_length),
        lookahead_vocabs=lookahead_vocabs,
    )
    val_dataset = LookaheadPlannerDataset(
        args.data_dir / "val.jsonl",
        tokenizer,
        max_length=int(args.max_length),
        lookahead_vocabs=lookahead_vocabs,
    )
    expected_train = 32 if args.engineering_smoke else OFFICIAL_TRAIN_ROWS
    expected_val = 32 if args.engineering_smoke else OFFICIAL_VAL_ROWS
    if len(train_dataset) != expected_train or len(val_dataset) != expected_val:
        raise ValueError("materialized Planner dataset row count changed")
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=lambda rows: collate(
            rows,
            pad_token_id=int(tokenizer.pad_token_id),
        ),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=lambda rows: collate(
            rows,
            pad_token_id=int(tokenizer.pad_token_id),
        ),
    )
    expected_microbatches = int(args.max_updates) * int(args.grad_accum)
    if len(train_loader) != expected_microbatches:
        raise ValueError(
            f"training stream has {len(train_loader)} microbatches; expected "
            f"{expected_microbatches}"
        )

    config = load_llama3_compatible_config(args.model_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    if args.gradient_checkpointing and args.arm != "p0":
        base_model.gradient_checkpointing_enable()
        if hasattr(base_model, "enable_input_require_grads"):
            base_model.enable_input_require_grads()
        if hasattr(base_model.config, "use_cache"):
            base_model.config.use_cache = False
    ensure_peft_cache_compat()
    disable_peft_bnb_autodetect()
    from peft import PeftModel

    model = PeftModel.from_pretrained(
        base_model,
        str(args.checkpoint_path),
        is_trainable=args.arm != "p0",
    )
    model.to(device)
    torch.cuda.reset_peak_memory_stats(device)
    model.train(args.arm != "p0")
    lookahead_heads = None
    if args.arm == "pstar":
        hidden_size = int(getattr(model.config, "hidden_size"))
        lookahead_heads = build_lookahead_heads(
            hidden_size,
            {field: len(labels) for field, labels in lookahead_vocabs.items()},
            seed=int(args.seed),
        ).to(device=device, dtype=torch.float32)
        lookahead_heads.train()

    configuration = {
        "schema": TRAINING_SCHEMA,
        "status": "running",
        "arm": args.arm,
        "engineering_smoke": bool(args.engineering_smoke),
        "model_path": args.model_path,
        "checkpoint_path": str(args.checkpoint_path),
        "data_dir": str(args.data_dir),
        "data_manifest_sha256": sha256_file(args.data_dir / "manifest.json"),
        "data_source": data_manifest.get("source"),
        "precision": "bfloat16",
        "gpus": 1,
        "batch_size": int(args.batch_size),
        "gradient_accumulation": int(args.grad_accum),
        "global_effective_batch": int(args.batch_size) * int(args.grad_accum),
        "max_updates": int(args.max_updates),
        "learning_rate": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "warmup_steps": int(args.warmup_steps),
        "eval_steps": int(args.eval_steps),
        "max_length": int(args.max_length),
        "seed": int(args.seed),
        "shuffle": False,
        "loss": (
            "0.80_field_balanced_plus_0.20_lookahead"
            if args.arm == "pstar"
            else "field_balanced_target_only"
        ),
        "generation_or_sun_selection": False,
    }
    (args.output_dir / "train_config.json").write_text(
        json.dumps(configuration, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    event_path = args.output_dir / "events.jsonl"
    start_time = time.time()
    initial_metrics = evaluate(
        model,
        val_loader,
        device,
        arm=args.arm,
        lookahead_heads=lookahead_heads,
    )
    append_event(
        event_path,
        {
            "event": "initial_validation",
            "step": 0,
            "metrics": initial_metrics,
            "elapsed_sec": time.time() - start_time,
        },
    )
    if args.arm == "p0":
        report = {
            **configuration,
            "status": "complete",
            "validation": initial_metrics,
            "elapsed_sec": time.time() - start_time,
            "cuda": {
                "device_name": torch.cuda.get_device_name(device),
                "peak_memory_allocated_bytes": int(
                    torch.cuda.max_memory_allocated(device)
                ),
                "peak_memory_reserved_bytes": int(
                    torch.cuda.max_memory_reserved(device)
                ),
            },
            "checkpoints": [],
        }
        (args.output_dir / "evaluation_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
        return

    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if lookahead_heads is not None:
        trainable_parameters.extend(lookahead_heads.parameters())
    if not trainable_parameters:
        raise RuntimeError("Planner training has no trainable parameters")
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_steps),
        num_training_steps=int(args.max_updates),
    )
    model.zero_grad(set_to_none=True)
    if lookahead_heads is not None:
        lookahead_heads.zero_grad(set_to_none=True)
    global_step = 0
    micro_step = 0
    running_loss = 0.0
    running_microbatches = 0
    checkpoints: list[dict[str, Any]] = []
    for raw_batch in train_loader:
        batch = batch_to_device(raw_batch, device)
        loss, metrics = compute_objective(
            model,
            batch,
            arm=args.arm,
            lookahead_heads=lookahead_heads,
        )
        if not bool(torch.isfinite(loss.detach()).item()):
            raise FloatingPointError("non-finite Planner training loss")
        (loss / int(args.grad_accum)).backward()
        running_loss += float(loss.detach().float().item())
        running_microbatches += 1
        micro_step += 1
        if micro_step % int(args.grad_accum) != 0:
            continue
        grad_norm = torch.nn.utils.clip_grad_norm_(
            trainable_parameters,
            max_norm=1.0,
        )
        if not bool(torch.isfinite(grad_norm.detach()).item()):
            raise FloatingPointError("non-finite Planner gradient norm")
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1
        if (
            global_step == 1
            or global_step % int(args.logging_steps) == 0
        ):
            append_event(
                event_path,
                {
                    "event": "training",
                    "step": global_step,
                    "micro_step": micro_step,
                    "train_loss_recent": running_loss
                    / max(1, running_microbatches),
                    "grad_norm": float(grad_norm.detach().float().item()),
                    "lr": float(scheduler.get_last_lr()[0]),
                    "elapsed_sec": time.time() - start_time,
                },
            )
            running_loss = 0.0
            running_microbatches = 0
        if global_step % int(args.eval_steps) == 0:
            validation = evaluate(
                model,
                val_loader,
                device,
                arm=args.arm,
                lookahead_heads=lookahead_heads,
            )
            event = {
                "event": "validation",
                "step": global_step,
                "metrics": validation,
                "elapsed_sec": time.time() - start_time,
            }
            append_event(event_path, event)
            checkpoints.append(
                publish_checkpoint(
                    args.output_dir,
                    step=global_step,
                    arm=args.arm,
                    model=model,
                    tokenizer=tokenizer,
                    lookahead_heads=lookahead_heads,
                    lookahead_vocabs=lookahead_vocabs,
                    metrics=validation,
                )
            )
        if global_step >= int(args.max_updates):
            break
    if micro_step != expected_microbatches:
        raise RuntimeError(
            f"consumed {micro_step} microbatches; expected {expected_microbatches}"
        )
    if global_step != int(args.max_updates):
        raise RuntimeError(
            f"completed {global_step} updates; expected {int(args.max_updates)}"
        )
    expected_checkpoints = int(args.max_updates) // int(args.eval_steps)
    if len(checkpoints) != expected_checkpoints:
        raise RuntimeError(
            f"published {len(checkpoints)} checkpoints; expected "
            f"{expected_checkpoints}"
        )
    report = {
        **configuration,
        "status": "complete",
        "microbatches": micro_step,
        "optimizer_updates": global_step,
        "initial_validation": initial_metrics,
        "checkpoints": checkpoints,
        "elapsed_sec": time.time() - start_time,
        "cuda": {
            "device_name": torch.cuda.get_device_name(device),
            "peak_memory_allocated_bytes": int(
                torch.cuda.max_memory_allocated(device)
            ),
            "peak_memory_reserved_bytes": int(
                torch.cuda.max_memory_reserved(device)
            ),
        },
        "auxiliary_heads_discarded_for_inference": args.arm == "pstar",
    }
    (args.output_dir / "training_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
