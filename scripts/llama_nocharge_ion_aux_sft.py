#!/usr/bin/env python3
"""Matched C0/C1 LoRA continuation for the no-charge H1 Planner route."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from crystal_dlm.h1_llm_planner import (  # noqa: E402
    disable_peft_bnb_autodetect,
    ensure_peft_cache_compat,
    load_llama3_compatible_config,
)
from crystal_dlm.h1_nocharge_ion_aux import (  # noqa: E402
    H1_NOCHARGE_ION_AUX_SCHEMA,
    validation_anchor_nll_gate,
)
from crystal_dlm.peft_adapter_identity import (  # noqa: E402
    PROTECTED_P0_ADAPTER_CONFIG_SHA256,
    PROTECTED_P0_ADAPTER_WEIGHT_SHA256,
    adapter_source_identity_report,
    adapter_pair_identity_report,
    copy_adapter_state_exact,
)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def format_prompt(tokenizer, record: Mapping[str, Any]) -> str:
    messages = record.get("messages")
    if isinstance(messages, list) and hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        return str(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    if isinstance(messages, list) and len(messages) >= 2:
        return f"System: {messages[0]['content']}\n\nUser: {messages[1]['content']}\n\nAssistant:"
    raise ValueError(f"record {record.get('record_id')} has no valid messages")


def answer_token_weights(
    tokenizer,
    answer: str,
    weighted_spans: Sequence[Mapping[str, Any]],
    *,
    eos_text: str,
) -> tuple[list[int], list[float]]:
    text = str(answer).strip() + str(eos_text)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    if "offset_mapping" not in encoded:
        raise RuntimeError("a fast tokenizer with offset_mapping is required for frozen token weighting")
    token_ids = [int(value) for value in encoded["input_ids"]]
    offsets = [(int(start), int(end)) for start, end in encoded["offset_mapping"]]
    weights = [1.0 for _ in token_ids]
    answer_length = len(str(answer).strip())
    for span in weighted_spans:
        start = int(span["start"])
        end = int(span["end"])
        weight = float(span["weight"])
        if not (0 <= start < end <= answer_length):
            raise ValueError(f"invalid weighted answer span {dict(span)} for answer length {answer_length}")
        covered = False
        for token_idx, (token_start, token_end) in enumerate(offsets):
            if token_end <= start or token_start >= end:
                continue
            weights[token_idx] = max(weights[token_idx], weight)
            covered = True
        if not covered:
            raise ValueError(f"weighted answer span {dict(span)} covers no tokenizer token")
    return token_ids, weights


class NochargeIonAuxDataset(Dataset):
    def __init__(self, path: Path, tokenizer, max_length: int, *, expected_arm: str) -> None:
        self.rows = list(iter_jsonl(path))
        if not self.rows:
            raise ValueError(f"no rows found in {path}")
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        for row in self.rows:
            if row.get("schema") != H1_NOCHARGE_ION_AUX_SCHEMA:
                raise ValueError(f"record {row.get('record_id')} has wrong schema {row.get('schema')!r}")
            if row.get("arm") != expected_arm:
                raise ValueError(f"record {row.get('record_id')} belongs to {row.get('arm')!r}, expected {expected_arm!r}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        prompt = format_prompt(self.tokenizer, row)
        prompt_ids = [int(value) for value in self.tokenizer(prompt, add_special_tokens=False)["input_ids"]]
        answer = str(row["answer"]).strip()
        answer_ids, answer_weights = answer_token_weights(
            self.tokenizer,
            answer,
            row.get("weighted_answer_spans") or [],
            eos_text=self.tokenizer.eos_token or "",
        )
        if len(answer_ids) >= self.max_length:
            raise ValueError(
                f"record {row['record_id']} answer uses {len(answer_ids)} tokens for max_length={self.max_length}"
            )
        max_prompt = self.max_length - len(answer_ids)
        if len(prompt_ids) > max_prompt:
            prompt_ids = prompt_ids[-max_prompt:]
        input_ids = prompt_ids + answer_ids
        loss_mode = str(row.get("loss_mode", "sft"))
        if loss_mode not in {"sft", "kl_only"}:
            raise ValueError(f"record {row['record_id']} has unsupported loss_mode {loss_mode!r}")
        labels = [-100] * len(prompt_ids) + (answer_ids if loss_mode == "sft" else [-100] * len(answer_ids))
        token_weight = [0.0] * len(prompt_ids) + (answer_weights if loss_mode == "sft" else [0.0] * len(answer_ids))
        kl_mask = [0.0] * len(prompt_ids) + ([1.0] * len(answer_ids) if loss_mode == "kl_only" else [0.0] * len(answer_ids))
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "token_weight": torch.tensor(token_weight, dtype=torch.float32),
            "kl_mask": torch.tensor(kl_mask, dtype=torch.float32),
            "sample_weight": float(row.get("sample_weight", 1.0)),
            "record_id": str(row["record_id"]),
            "task": str(row["task"]),
        }


def collate(batch: list[dict[str, Any]], pad_token_id: int) -> dict[str, Any]:
    max_len = max(item["input_ids"].numel() for item in batch)
    batch_size = len(batch)
    input_ids = torch.full((batch_size, max_len), int(pad_token_id), dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
    token_weight = torch.zeros((batch_size, max_len), dtype=torch.float32)
    kl_mask = torch.zeros((batch_size, max_len), dtype=torch.float32)
    sample_weight = torch.ones((batch_size,), dtype=torch.float32)
    for idx, item in enumerate(batch):
        length = item["input_ids"].numel()
        input_ids[idx, :length] = item["input_ids"]
        attention_mask[idx, :length] = 1
        labels[idx, :length] = item["labels"]
        token_weight[idx, :length] = item["token_weight"]
        kl_mask[idx, :length] = item["kl_mask"]
        sample_weight[idx] = float(item["sample_weight"])
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "token_weight": token_weight,
        "kl_mask": kl_mask,
        "sample_weight": sample_weight,
        "record_id": [item["record_id"] for item in batch],
        "task": [item["task"] for item in batch],
    }


def weighted_sft_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    token_weight: torch.Tensor,
    sample_weight: torch.Tensor,
) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_weight = token_weight[:, 1:].to(device=logits.device, dtype=torch.float32)
    losses = torch.nn.functional.cross_entropy(
        shift_logits.float().view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view(shift_labels.shape)
    mask = (shift_labels != -100).to(dtype=torch.float32)
    weights = shift_weight * mask * sample_weight.to(device=logits.device, dtype=torch.float32).unsqueeze(1)
    denominator = weights.sum()
    if float(denominator.detach().item()) == 0.0:
        return logits.sum() * 0.0
    return (losses * weights).sum() / denominator


def masked_forward_kl(
    candidate_logits: torch.Tensor,
    reference_logits: torch.Tensor,
    kl_mask: torch.Tensor,
) -> torch.Tensor:
    mask = kl_mask[:, 1:].to(device=candidate_logits.device, dtype=torch.float32)
    denominator = mask.sum()
    if float(denominator.detach().item()) == 0.0:
        return candidate_logits.sum() * 0.0
    candidate_log_probs = torch.nn.functional.log_softmax(candidate_logits[:, :-1, :].float(), dim=-1)
    reference_probs = torch.nn.functional.softmax(reference_logits[:, :-1, :].float(), dim=-1)
    per_token = torch.nn.functional.kl_div(candidate_log_probs, reference_probs, reduction="none").sum(dim=-1)
    return (per_token * mask).sum() / denominator


def tensor_batch_to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def dual_adapter_identity_report(model) -> dict[str, Any]:
    return adapter_pair_identity_report(
        model,
        candidate_name="candidate",
        reference_name="reference",
        expected_dtype=torch.float32,
        expected_active_adapter="candidate",
    )


def reference_logits(model, batch: Mapping[str, Any]) -> torch.Tensor | None:
    if float(batch["kl_mask"].sum().item()) == 0.0:
        return None
    was_training = bool(model.training)
    model.set_adapter("reference")
    for name, parameter in model.named_parameters():
        if ".reference." in name or ".candidate." in name:
            parameter.requires_grad_(False)
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False)
        logits = outputs.logits.detach()
    model.set_adapter("candidate")
    for name, parameter in model.named_parameters():
        if ".candidate." in name:
            parameter.requires_grad_(True)
        elif ".reference." in name:
            parameter.requires_grad_(False)
    model.train(was_training)
    return logits


def batch_losses(model, batch: Mapping[str, Any], *, kl_beta: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    frozen_logits = reference_logits(model, batch)
    outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False)
    sft_loss = weighted_sft_loss(
        outputs.logits,
        batch["labels"],
        batch["token_weight"],
        batch["sample_weight"],
    )
    kl_loss = (
        outputs.logits.sum() * 0.0
        if frozen_logits is None
        else masked_forward_kl(outputs.logits, frozen_logits, batch["kl_mask"])
    )
    return sft_loss + float(kl_beta) * kl_loss, sft_loss, kl_loss


@torch.no_grad()
def evaluate(model, loader: DataLoader, device: torch.device, *, kl_beta: float, max_batches: int) -> dict[str, float]:
    model.set_adapter("candidate")
    model.eval()
    totals = {"loss": 0.0, "sft_loss": 0.0, "kl_loss": 0.0}
    counts = {"loss": 0, "sft_loss": 0, "kl_loss": 0}
    for batch_idx, raw_batch in enumerate(loader):
        if int(max_batches) > 0 and batch_idx >= int(max_batches):
            break
        batch = tensor_batch_to_device(raw_batch, device)
        loss, sft_loss, kl_loss = batch_losses(model, batch, kl_beta=float(kl_beta))
        totals["loss"] += float(loss.item())
        counts["loss"] += 1
        if float(batch["token_weight"].sum().item()) > 0:
            totals["sft_loss"] += float(sft_loss.item())
            counts["sft_loss"] += 1
        if float(batch["kl_mask"].sum().item()) > 0:
            totals["kl_loss"] += float(kl_loss.item())
            counts["kl_loss"] += 1
    model.set_adapter("candidate")
    model.train()
    return {key: totals[key] / max(1, counts[key]) for key in totals} | {
        f"{key}_batch_count": float(counts[key]) for key in counts
    }


@torch.no_grad()
def sft_nll_by_task(
    model,
    loader: DataLoader,
    device: torch.device,
    *,
    adapter_name: str,
) -> dict[str, dict[str, float]]:
    """Measure unweighted answer-token NLL without changing a checkpoint."""

    if adapter_name not in {"candidate", "reference"}:
        raise ValueError(f"unknown adapter {adapter_name!r}")
    was_training = bool(model.training)
    model.set_adapter(adapter_name)
    for name, parameter in model.named_parameters():
        if ".candidate." in name or ".reference." in name:
            parameter.requires_grad_(False)
    model.eval()
    totals: dict[str, float] = {}
    token_counts: dict[str, int] = {}
    record_counts: dict[str, int] = {}
    for raw_batch in loader:
        batch = tensor_batch_to_device(raw_batch, device)
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
        )
        shift_logits = outputs.logits[:, :-1, :].float()
        shift_labels = batch["labels"][:, 1:]
        per_token = torch.nn.functional.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).reshape(shift_labels.shape)
        mask = shift_labels != -100
        for row_idx, task in enumerate(batch["task"]):
            count = int(mask[row_idx].sum().item())
            if count == 0:
                continue
            key = str(task)
            totals[key] = totals.get(key, 0.0) + float(
                per_token[row_idx][mask[row_idx]].sum().item()
            )
            token_counts[key] = token_counts.get(key, 0) + count
            record_counts[key] = record_counts.get(key, 0) + 1
    model.set_adapter("candidate")
    for name, parameter in model.named_parameters():
        if ".candidate." in name:
            parameter.requires_grad_(True)
        elif ".reference." in name:
            parameter.requires_grad_(False)
    model.train(was_training)
    return {
        task: {
            "nll": totals[task] / token_counts[task],
            "answer_token_count": float(token_counts[task]),
            "record_count": float(record_counts[task]),
        }
        for task in sorted(totals)
    }


def save_candidate_adapter(model, tokenizer, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    model.set_adapter("candidate")
    model.save_pretrained(output_dir, selected_adapters=["candidate"], safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    candidates = [
        path.parent
        for path in output_dir.rglob("adapter_config.json")
        if path.is_file()
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected one saved candidate adapter below {output_dir}, found {[str(path) for path in candidates]}"
        )
    adapter_dir = candidates[0]
    weight_files = sorted(
        path.name
        for path in adapter_dir.iterdir()
        if path.name in {"adapter_model.safetensors", "adapter_model.bin"}
    )
    if len(weight_files) != 1:
        raise RuntimeError(f"candidate adapter at {adapter_dir} has unexpected weight files {weight_files}")
    (output_dir / "adapter_location.json").write_text(
        json.dumps(
            {
                "adapter_name": "candidate",
                "adapter_directory": str(adapter_dir),
                "adapter_directory_relative": str(adapter_dir.relative_to(output_dir)),
                "weight_file": weight_files[0],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--p0-adapter-path", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--arm", choices=("c0", "c1"), required=True)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-updates", type=int, default=400)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-steps", type=int, default=25)
    parser.add_argument("--kl-beta", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=26080617)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--eval-max-batches", type=int, default=160)
    parser.add_argument("--checkpoint-steps", default="100,200,300,400")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--allow-nonfrozen-fixture", action="store_true")
    parser.add_argument(
        "--expected-p0-adapter-weight-sha256",
        default=PROTECTED_P0_ADAPTER_WEIGHT_SHA256,
    )
    parser.add_argument(
        "--expected-p0-adapter-config-sha256",
        default=PROTECTED_P0_ADAPTER_CONFIG_SHA256,
    )
    args = parser.parse_args()

    expected_arm = "c0_neutral_aux" if args.arm == "c0" else "c1_ion_aux"
    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not args.preflight_only and device.type != "cuda":
        raise RuntimeError("frozen H1 no-charge SFT requires CUDA")
    if device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("frozen H1 no-charge SFT requires BF16 support")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    tokenizer_source = (
        args.p0_adapter_path
        if (Path(args.p0_adapter_path) / "tokenizer_config.json").exists()
        else args.model_path
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    config = load_llama3_compatible_config(args.model_path)
    base = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
    )
    if hasattr(base.config, "use_cache"):
        base.config.use_cache = False
    ensure_peft_cache_compat()
    from peft import PeftModel
    disable_peft_bnb_autodetect()

    model = PeftModel.from_pretrained(
        base,
        args.p0_adapter_path,
        adapter_name="candidate",
        is_trainable=True,
        autocast_adapter_dtype=True,
    )
    candidate_source_before_reference = adapter_source_identity_report(
        model,
        "candidate",
        Path(args.p0_adapter_path),
        expected_weight_sha256=str(args.expected_p0_adapter_weight_sha256),
        expected_config_sha256=str(args.expected_p0_adapter_config_sha256),
    )
    if not candidate_source_before_reference["passed"]:
        raise RuntimeError("candidate adapter is not byte-identical to protected P0")
    model.load_adapter(
        args.p0_adapter_path,
        adapter_name="reference",
        is_trainable=True,
        autocast_adapter_dtype=True,
    )
    pre_copy_identity = dual_adapter_identity_report(model)
    copy_report = copy_adapter_state_exact(
        model,
        source_adapter="candidate",
        target_adapter="reference",
        expected_dtype=torch.float32,
    )
    model.set_adapter("candidate")
    model.to(device)
    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.train()

    identity = dual_adapter_identity_report(model)
    candidate_source_after_setup = adapter_source_identity_report(
        model,
        "candidate",
        Path(args.p0_adapter_path),
        expected_weight_sha256=str(args.expected_p0_adapter_weight_sha256),
        expected_config_sha256=str(args.expected_p0_adapter_config_sha256),
    )
    reference_source_after_setup = adapter_source_identity_report(
        model,
        "reference",
        Path(args.p0_adapter_path),
        expected_weight_sha256=str(args.expected_p0_adapter_weight_sha256),
        expected_config_sha256=str(args.expected_p0_adapter_config_sha256),
    )
    protected_identity = {
        "schema": "h1_protected_p0_candidate_reference_identity_v2",
        "candidate_source_before_reference": candidate_source_before_reference,
        "pre_copy_pair_diagnostic": pre_copy_identity,
        "copy_report": copy_report,
        "post_setup_pair_identity": identity,
        "candidate_source_after_setup": candidate_source_after_setup,
        "reference_source_after_setup": reference_source_after_setup,
        "passed": bool(
            candidate_source_before_reference["passed"]
            and copy_report["passed"]
            and identity["passed"]
            and candidate_source_after_setup["passed"]
            and reference_source_after_setup["passed"]
        ),
    }
    (args.output_dir / "protected_p0_triplet_identity.json").write_text(
        json.dumps(protected_identity, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "dual_adapter_identity.json").write_text(
        json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not protected_identity["passed"]:
        raise RuntimeError("candidate/reference P0 adapter identity failed")

    train_ds = NochargeIonAuxDataset(args.data_dir / "train.jsonl", tokenizer, args.max_length, expected_arm=expected_arm)
    val_ds = NochargeIonAuxDataset(args.data_dir / "val.jsonl", tokenizer, args.max_length, expected_arm=expected_arm)
    if not args.allow_nonfrozen_fixture:
        if len(train_ds) != 3200 or len(val_ds) != 640:
            raise RuntimeError(f"frozen ledger count mismatch train={len(train_ds)} val={len(val_ds)}")
        if (int(args.batch_size), int(args.grad_accum), int(args.max_updates)) != (1, 8, 400):
            raise RuntimeError("frozen optimizer geometry must be batch=1, grad_accum=8, updates=400")
        if int(args.batch_size) * int(args.grad_accum) * int(args.max_updates) != len(train_ds):
            raise RuntimeError("frozen run must consume exactly one 3,200-record epoch")

    generator = torch.Generator()
    generator.manual_seed(int(args.seed))
    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        generator=generator,
        num_workers=0,
        collate_fn=lambda batch: collate(batch, int(tokenizer.pad_token_id)),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=lambda batch: collate(batch, int(tokenizer.pad_token_id)),
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=float(args.lr), weight_decay=float(args.weight_decay))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_steps),
        num_training_steps=int(args.max_updates),
    )
    checkpoints = {int(value) for value in str(args.checkpoint_steps).split(",") if value.strip()}
    if checkpoints != {100, 200, 300, 400} and not args.allow_nonfrozen_fixture:
        raise RuntimeError(f"frozen checkpoint steps changed: {sorted(checkpoints)}")

    config_payload = {
        "schema": H1_NOCHARGE_ION_AUX_SCHEMA,
        "arm": args.arm,
        "expected_record_arm": expected_arm,
        "model_path": args.model_path,
        "p0_adapter_path": args.p0_adapter_path,
        "data_dir": str(args.data_dir),
        "train_rows": len(train_ds),
        "val_rows": len(val_ds),
        "max_length": int(args.max_length),
        "batch_size": int(args.batch_size),
        "grad_accum": int(args.grad_accum),
        "max_updates": int(args.max_updates),
        "lr": float(args.lr),
        "warmup_steps": int(args.warmup_steps),
        "weight_decay": float(args.weight_decay),
        "kl_beta": float(args.kl_beta),
        "seed": int(args.seed),
        "checkpoint_steps": sorted(checkpoints),
        "device": str(device),
        "bf16": device.type == "cuda",
        "dual_adapter_identity": identity,
        "protected_p0_triplet_identity": protected_identity,
    }
    if args.preflight_only:
        raw_batch = next(
            batch
            for batch in train_loader
            if float(batch["token_weight"].sum().item()) > 0.0
        )
        batch = tensor_batch_to_device(raw_batch, device)
        smoke_loss, smoke_sft, smoke_kl = batch_losses(
            model,
            batch,
            kl_beta=float(args.kl_beta),
        )
        if not all(
            torch.isfinite(value).item()
            for value in (smoke_loss, smoke_sft, smoke_kl)
        ):
            raise RuntimeError("preflight forward loss is non-finite")
        smoke_loss.backward()
        finite_gradients = all(
            torch.isfinite(parameter.grad).all().item()
            for parameter in trainable
            if parameter.grad is not None
        )
        gradient_tensor_count = sum(
            parameter.grad is not None for parameter in trainable
        )
        model.zero_grad(set_to_none=True)
        if not finite_gradients or gradient_tensor_count == 0:
            raise RuntimeError("preflight backward gradients are absent or non-finite")
        config_payload["preflight_smoke"] = {
            "loss": float(smoke_loss.item()),
            "sft_loss": float(smoke_sft.item()),
            "kl_loss": float(smoke_kl.item()),
            "finite_gradients": finite_gradients,
            "gradient_tensor_count": gradient_tensor_count,
            "optimizer_step_performed": False,
        }
        (args.output_dir / "train_config.json").write_text(
            json.dumps(config_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"status": "preflight_pass", **config_payload}, ensure_ascii=False, indent=2))
        return

    reference_validation_nll = sft_nll_by_task(
        model,
        val_loader,
        device,
        adapter_name="reference",
    )
    if "conditional_mp20_anchor" not in reference_validation_nll:
        raise RuntimeError(
            "reference validation omitted the conditional_mp20_anchor task"
        )
    config_payload["reference_validation_nll_by_task"] = (
        reference_validation_nll
    )
    (args.output_dir / "train_config.json").write_text(
        json.dumps(config_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    start = time.time()
    global_step = 0
    micro_step = 0
    running = {"loss": 0.0, "sft_loss": 0.0, "kl_loss": 0.0, "microbatches": 0}
    history_path = args.output_dir / "history.jsonl"
    model.zero_grad(set_to_none=True)
    progress = tqdm(total=int(args.max_updates), desc=f"H1 no-charge {args.arm} SFT")
    for raw_batch in train_loader:
        batch = tensor_batch_to_device(raw_batch, device)
        loss, sft_loss, kl_loss = batch_losses(model, batch, kl_beta=float(args.kl_beta))
        if not all(
            torch.isfinite(value).item()
            for value in (loss, sft_loss, kl_loss)
        ):
            raise RuntimeError(
                f"non-finite training loss at micro_step={micro_step + 1}"
            )
        (loss / int(args.grad_accum)).backward()
        running["loss"] += float(loss.item())
        running["sft_loss"] += float(sft_loss.item())
        running["kl_loss"] += float(kl_loss.item())
        running["microbatches"] += 1
        micro_step += 1
        if micro_step % int(args.grad_accum) != 0:
            continue
        if not all(
            torch.isfinite(parameter.grad).all().item()
            for parameter in trainable
            if parameter.grad is not None
        ):
            raise RuntimeError(
                f"non-finite training gradient before update={global_step + 1}"
            )
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1
        progress.update(1)
        events: list[dict[str, Any]] = []
        if global_step == 1 or global_step % int(args.logging_steps) == 0:
            denominator = max(1, int(running["microbatches"]))
            events.append({
                "event": "train",
                "step": global_step,
                "loss": running["loss"] / denominator,
                "sft_loss": running["sft_loss"] / denominator,
                "kl_loss": running["kl_loss"] / denominator,
                "lr": float(scheduler.get_last_lr()[0]),
                "elapsed_sec": time.time() - start,
            })
            running = {"loss": 0.0, "sft_loss": 0.0, "kl_loss": 0.0, "microbatches": 0}
        if int(args.eval_steps) > 0 and global_step % int(args.eval_steps) == 0:
            evaluation = evaluate(
                model,
                val_loader,
                device,
                kl_beta=float(args.kl_beta),
                max_batches=int(args.eval_max_batches),
            )
            events.append({
                "event": "teacher_forced_eval",
                "step": global_step,
                **evaluation,
                "elapsed_sec": time.time() - start,
                "selection_authorized": False,
            })
        if events:
            with history_path.open("a", encoding="utf-8") as handle:
                for event in events:
                    handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
            for event in events:
                print(json.dumps(event, ensure_ascii=False), flush=True)
        if global_step in checkpoints:
            save_candidate_adapter(model, tokenizer, args.output_dir / f"checkpoint-{global_step:04d}")
        if global_step >= int(args.max_updates):
            break

    progress.close()
    if global_step != int(args.max_updates) or micro_step != len(train_ds) // int(args.batch_size):
        raise RuntimeError(
            f"training ledger consumption mismatch: global_step={global_step}, micro_step={micro_step}, rows={len(train_ds)}"
        )
    final_eval = evaluate(
        model,
        val_loader,
        device,
        kl_beta=float(args.kl_beta),
        max_batches=int(args.eval_max_batches),
    )
    candidate_validation_nll = sft_nll_by_task(
        model,
        val_loader,
        device,
        adapter_name="candidate",
    )
    if "conditional_mp20_anchor" not in candidate_validation_nll:
        raise RuntimeError(
            "candidate validation omitted the conditional_mp20_anchor task"
        )
    anchor_gate = validation_anchor_nll_gate(
        reference_validation_nll["conditional_mp20_anchor"]["nll"],
        candidate_validation_nll["conditional_mp20_anchor"]["nll"],
        maximum_relative_degradation=0.01,
    )
    terminal = {
        "status": (
            "pass"
            if anchor_gate["passed"]
            else "anchor_nll_noninferiority_failure"
        ),
        "arm": args.arm,
        "global_step": global_step,
        "micro_step": micro_step,
        "fixed_endpoint": "checkpoint-0400",
        "teacher_forced_eval": final_eval,
        "reference_validation_nll_by_task": reference_validation_nll,
        "candidate_validation_nll_by_task": candidate_validation_nll,
        "conditional_mp20_anchor_nll_gate": anchor_gate,
        "generated_metric_checkpoint_selection": False,
        "elapsed_sec": time.time() - start,
    }
    (args.output_dir / "terminal_report.json").write_text(
        json.dumps(terminal, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    marker = "_SUCCESS" if anchor_gate["passed"] else "_FAILED"
    (args.output_dir / marker).write_text(
        json.dumps(terminal, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(terminal, ensure_ascii=False, indent=2), flush=True)
    if not anchor_gate["passed"]:
        raise RuntimeError(
            "conditional MP20 anchor validation NLL degraded by more than 1%"
        )


if __name__ == "__main__":
    main()
