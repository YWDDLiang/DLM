#!/usr/bin/env python3
"""Short epoch-2-anchored H1-A2 ValidReplay/JointChem continuation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Iterable


THIS_DIR = Path(__file__).resolve().parent
REACTIVATION_ROOT = THIS_DIR.parents[1]
PROJECT_ROOT = THIS_DIR.parents[3]
RESTORED_BASELINE_ROOT = REACTIVATION_ROOT / "baseline"
RUNTIME_ROOT = RESTORED_BASELINE_ROOT if (RESTORED_BASELINE_ROOT / "crystal_dlm").is_dir() else PROJECT_ROOT
if str(RUNTIME_ROOT) not in sys.path:
    sys.path.insert(0, str(RUNTIME_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from crystal_dlm.h1_llm_planner import (  # noqa: E402
    disable_peft_bnb_autodetect,
    ensure_peft_cache_compat,
    load_llama3_compatible_config,
)
from jointchem_loss import (  # noqa: E402
    build_target_only_token_lists,
    combine_active_losses,
    pairwise_preference_loss,
    target_only_mean_nll,
)


ARMS = ("valid_replay", "jointchem")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise TypeError(f"{path} contains a non-object row")
                yield payload


def format_prompt(tokenizer, record: dict[str, Any]) -> str:
    prompt = record.get("prompt")
    if prompt:
        return str(prompt)
    messages = record.get("messages")
    if isinstance(messages, list) and hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        return str(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    if isinstance(messages, list) and len(messages) >= 2:
        return f"System: {messages[0]['content']}\n\nUser: {messages[1]['content']}\n\nAssistant:"
    raise ValueError("JointChem row has neither messages nor prompt")


def tokenize_plan(tokenizer, prompt: str, answer: str, max_length: int) -> dict[str, torch.Tensor]:
    tokenized = build_target_only_token_lists(tokenizer, prompt, answer, max_length)
    return {
        "input_ids": torch.tensor(tokenized["input_ids"], dtype=torch.long),
        "labels": torch.tensor(tokenized["labels"], dtype=torch.long),
    }


class JointChemDataset(Dataset):
    def __init__(self, path: Path, tokenizer, max_length: int) -> None:
        self.rows = list(iter_jsonl(path))
        if not self.rows:
            raise ValueError(f"No rows found in {path}")
        self.tokenizer = tokenizer
        self.max_length = int(max_length)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        prompt = format_prompt(self.tokenizer, row)
        positive = tokenize_plan(self.tokenizer, prompt, str(row["answer"]), self.max_length)
        output: dict[str, Any] = {
            "positive": positive,
            "stream_role": str(row.get("stream_role", "validation")),
            "row_key": str(row["row_key"]),
            "composition_valid": bool(row.get("composition_valid")),
        }
        for key, answer_key in (
            ("chemistry_negative", "chemistry_negative_answer"),
            ("joint_negative", "joint_negative_answer"),
        ):
            answer = row.get(answer_key)
            output[key] = None if answer is None else tokenize_plan(
                self.tokenizer,
                prompt,
                str(answer),
                self.max_length,
            )
        return output


def _pad_sequence_group(items, *, pad_token_id: int) -> dict[str, torch.Tensor] | None:
    if any(item is None for item in items):
        if all(item is None for item in items):
            return None
        raise ValueError("mixed missing/present negatives require batch_size=1")
    max_len = max(item["input_ids"].numel() for item in items)
    input_ids = torch.full((len(items), max_len), int(pad_token_id), dtype=torch.long)
    attention_mask = torch.zeros((len(items), max_len), dtype=torch.long)
    labels = torch.full((len(items), max_len), -100, dtype=torch.long)
    for index, item in enumerate(items):
        length = item["input_ids"].numel()
        input_ids[index, :length] = item["input_ids"]
        attention_mask[index, :length] = 1
        labels[index, :length] = item["labels"]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def collate(batch: list[dict[str, Any]], pad_token_id: int) -> dict[str, Any]:
    if len(batch) != 1:
        raise ValueError("JointChem v1 requires batch_size=1")
    return {
        "positive": _pad_sequence_group([item["positive"] for item in batch], pad_token_id=pad_token_id),
        "chemistry_negative": _pad_sequence_group(
            [item["chemistry_negative"] for item in batch],
            pad_token_id=pad_token_id,
        ),
        "joint_negative": _pad_sequence_group(
            [item["joint_negative"] for item in batch],
            pad_token_id=pad_token_id,
        ),
        "stream_role": batch[0]["stream_role"],
        "row_key": batch[0]["row_key"],
        "composition_valid": bool(batch[0]["composition_valid"]),
    }


def move_group(group: dict[str, torch.Tensor] | None, device: torch.device):
    if group is None:
        return None
    return {key: value.to(device) for key, value in group.items()}


def sequence_nll(model, group: dict[str, torch.Tensor]):
    output = model(input_ids=group["input_ids"], attention_mask=group["attention_mask"])
    return target_only_mean_nll(output.logits, group["labels"])


def compute_loss(
    model,
    batch: dict[str, Any],
    *,
    arm: str,
    margin: float,
    sft_weight: float,
    chemistry_weight: float,
    joint_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    positive_nll = sequence_nll(model, batch["positive"])
    positive_loss = positive_nll.mean()
    components: list[tuple[float, torch.Tensor | None]] = [(float(sft_weight), positive_loss)]
    metrics = {"positive_nll": float(positive_loss.detach().item())}
    use_ranking = (
        arm == "jointchem"
        and batch["stream_role"] != "epoch2_anchor"
        and batch["composition_valid"] is True
    )

    chemistry_loss = None
    if use_ranking and batch["chemistry_negative"] is not None:
        negative_nll = sequence_nll(model, batch["chemistry_negative"])
        chemistry_loss = pairwise_preference_loss(positive_nll, negative_nll, margin=margin)
        metrics["chemistry_negative_nll"] = float(negative_nll.mean().detach().item())
        metrics["chemistry_paired_margin"] = float(
            (negative_nll - positive_nll).mean().detach().item()
        )
        metrics["chemistry_rank_loss"] = float(chemistry_loss.detach().item())
    components.append((float(chemistry_weight), chemistry_loss))

    joint_loss = None
    if use_ranking and batch["joint_negative"] is not None:
        negative_nll = sequence_nll(model, batch["joint_negative"])
        joint_loss = pairwise_preference_loss(positive_nll, negative_nll, margin=margin)
        metrics["joint_negative_nll"] = float(negative_nll.mean().detach().item())
        metrics["joint_paired_margin"] = float(
            (negative_nll - positive_nll).mean().detach().item()
        )
        metrics["joint_rank_loss"] = float(joint_loss.detach().item())
    components.append((float(joint_weight), joint_loss))

    loss = combine_active_losses(components)
    metrics["loss"] = float(loss.detach().item())
    return loss, metrics


@torch.no_grad()
def evaluate(
    model,
    loader: DataLoader,
    device: torch.device,
    *,
    arm: str,
    max_batches: int,
    margin: float,
    sft_weight: float,
    chemistry_weight: float,
    joint_weight: float,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    evaluated_rows = 0
    for batch_idx, batch in enumerate(loader):
        if int(max_batches) > 0 and batch_idx >= int(max_batches):
            break
        evaluated_rows += 1
        batch["positive"] = move_group(batch["positive"], device)
        batch["chemistry_negative"] = move_group(batch["chemistry_negative"], device)
        batch["joint_negative"] = move_group(batch["joint_negative"], device)
        _, metrics = compute_loss(
            model,
            batch,
            # Use the common JointChem diagnostic objective for both trained
            # arms so likelihood margins are comparable at every checkpoint.
            arm="jointchem",
            margin=margin,
            sft_weight=sft_weight,
            chemistry_weight=chemistry_weight,
            joint_weight=joint_weight,
        )
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + float(value)
            counts[key] = counts.get(key, 0) + 1
    model.train()
    result = {key: totals[key] / max(1, counts[key]) for key in sorted(totals)}
    result["evaluated_rows"] = int(evaluated_rows)
    result["chemistry_paired_rows"] = int(counts.get("chemistry_paired_margin", 0))
    result["joint_paired_rows"] = int(counts.get("joint_paired_margin", 0))
    return result


def assert_resources(*, require_one_a800: bool) -> dict[str, Any]:
    gpu_count = int(torch.cuda.device_count())
    cpu_count = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    gpu_names = [str(torch.cuda.get_device_name(index)) for index in range(gpu_count)]
    if require_one_a800:
        if gpu_count != 1:
            raise RuntimeError(f"JointChem v1 requires exactly one visible GPU, observed {gpu_count}")
        if "A800" not in gpu_names[0]:
            raise RuntimeError(f"JointChem v1 requires an A800, observed {gpu_names[0]!r}")
        if cpu_count > 8:
            raise RuntimeError(f"CPU policy violation: {cpu_count} CPU for one A800")
    return {"gpu_count": gpu_count, "gpu_names": gpu_names, "cpu_count": cpu_count}


def save_checkpoint(model, tokenizer, output_dir: Path, step: int) -> tuple[Path, str]:
    checkpoint_dir = output_dir / f"checkpoint-{int(step):06d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    inventory = []
    for path in sorted(checkpoint_dir.rglob("*")):
        if path.is_file():
            inventory.append(
                {
                    "path": str(path.relative_to(checkpoint_dir)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    manifest_path = checkpoint_dir / "checkpoint_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "h1a2_jointchem_checkpoint_manifest_v1",
                "step": int(step),
                "files": inventory,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return checkpoint_dir, sha256_file(manifest_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-updates", type=int, default=400)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-steps", type=int, default=25)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--eval-max-batches", type=int, default=128)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--margin", type=float, default=0.10)
    parser.add_argument("--sft-weight", type=float, default=0.75)
    parser.add_argument("--chemistry-weight", type=float, default=0.15)
    parser.add_argument("--joint-weight", type=float, default=0.10)
    parser.add_argument("--execution-manifest-sha256", required=True)
    parser.add_argument("--initial-adapter-sha256", required=True)
    parser.add_argument("--require-one-a800", action="store_true")
    args = parser.parse_args()

    if int(args.batch_size) != 1 or int(args.grad_accum) != 8:
        raise ValueError("JointChem v1 freezes batch_size=1 and grad_accum=8")
    if int(args.max_updates) != 400:
        raise ValueError("JointChem v1 freezes max_updates=400")
    if abs(float(args.lr) - 2e-6) > 1e-15:
        raise ValueError("JointChem v1 freezes lr=2e-6")
    if int(args.eval_steps) != 50:
        raise ValueError("JointChem v1 freezes eval_steps=50")
    if int(args.eval_max_batches) != 128:
        raise ValueError("JointChem v1 freezes eval_max_batches=128")
    if int(args.seed) != 17:
        raise ValueError("JointChem v1 freezes seed=17")
    if int(args.max_length) != 768:
        raise ValueError("JointChem v1 freezes max_length=768")
    if int(args.warmup_steps) != 25:
        raise ValueError("JointChem v1 freezes warmup_steps=25")
    if abs(float(args.weight_decay)) > 1e-15:
        raise ValueError("JointChem v1 freezes weight_decay=0")
    if abs(float(args.margin) - 0.10) > 1e-12:
        raise ValueError("JointChem v1 freezes margin=0.10")
    expected_weights = (0.75, 0.15, 0.10)
    observed_weights = (
        float(args.sft_weight),
        float(args.chemistry_weight),
        float(args.joint_weight),
    )
    if any(abs(observed - expected) > 1e-12 for observed, expected in zip(observed_weights, expected_weights)):
        raise ValueError(f"JointChem v1 freezes loss weights={expected_weights}")

    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    resources = assert_resources(require_one_a800=bool(args.require_one_a800))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    train_path = args.data_dir / "train.jsonl"
    val_path = args.data_dir / "val.jsonl"
    data_manifest = args.data_dir / "manifest.json"
    success_path = args.data_dir / "_SUCCESS"
    for path in (train_path, val_path, data_manifest, success_path):
        if not path.exists():
            raise FileNotFoundError(path)
    train_rows = sum(1 for _ in iter_jsonl(train_path))
    if train_rows != 3200:
        raise ValueError(f"JointChem v1 requires 3200 train rows, observed {train_rows}")
    adapter_weights = Path(args.checkpoint_path) / "adapter_model.safetensors"
    if not adapter_weights.is_file():
        raise FileNotFoundError(adapter_weights)
    observed_adapter_sha256 = sha256_file(adapter_weights)
    if observed_adapter_sha256 != str(args.initial_adapter_sha256):
        raise ValueError(
            "epoch-2 adapter SHA mismatch: "
            f"expected {args.initial_adapter_sha256}, observed {observed_adapter_sha256}"
        )

    tokenizer_source = (
        args.checkpoint_path
        if (Path(args.checkpoint_path) / "tokenizer_config.json").exists()
        else args.model_path
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    config = load_llama3_compatible_config(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        config=config,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    model.gradient_checkpointing_enable()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    ensure_peft_cache_compat()
    from peft import PeftModel
    disable_peft_bnb_autodetect()

    model = PeftModel.from_pretrained(model, args.checkpoint_path, is_trainable=True)
    model.to(device)
    model.train()

    train_ds = JointChemDataset(train_path, tokenizer, int(args.max_length))
    val_ds = JointChemDataset(val_path, tokenizer, int(args.max_length))
    train_loader = DataLoader(
        train_ds,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda batch: collate(batch, int(tokenizer.pad_token_id)),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda batch: collate(batch, int(tokenizer.pad_token_id)),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_steps),
        num_training_steps=int(args.max_updates),
    )
    config_payload = {
        "schema": "h1a2_jointchem_train_config_v1",
        "arm": args.arm,
        "model_path": args.model_path,
        "checkpoint_path": args.checkpoint_path,
        "initial_adapter_sha256": observed_adapter_sha256,
        "execution_manifest_sha256": str(args.execution_manifest_sha256),
        "data_dir": str(args.data_dir),
        "data_sha256": {
            "train": sha256_file(train_path),
            "val": sha256_file(val_path),
            "manifest": sha256_file(data_manifest),
            "success": sha256_file(success_path),
        },
        "max_length": int(args.max_length),
        "batch_size": 1,
        "grad_accum": 8,
        "max_updates": 400,
        "lr": 2e-6,
        "weight_decay": 0.0,
        "warmup_steps": 25,
        "scheduler": "cosine",
        "eval_steps": 50,
        "seed": 17,
        "margin": float(args.margin),
        "loss_weights": {
            "sft": float(args.sft_weight),
            "chemistry": float(args.chemistry_weight),
            "joint": float(args.joint_weight),
        },
        "resources": resources,
        "automatic_crystal_evaluation_authorized": False,
        "sun_or_mlip_used_for_selection": False,
    }
    (args.output_dir / "train_config.json").write_text(
        json.dumps(config_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    initial_eval = evaluate(
        model,
        val_loader,
        device,
        arm="jointchem",
        max_batches=int(args.eval_max_batches),
        margin=float(args.margin),
        sft_weight=float(args.sft_weight),
        chemistry_weight=float(args.chemistry_weight),
        joint_weight=float(args.joint_weight),
    )
    if int(initial_eval.get("evaluated_rows", -1)) != int(args.eval_max_batches):
        raise RuntimeError(f"incomplete initial validation: {initial_eval}")
    if int(initial_eval.get("chemistry_paired_rows", 0)) <= 0:
        raise RuntimeError("initial validation has no chemistry-paired rows")
    if int(initial_eval.get("joint_paired_rows", 0)) <= 0:
        raise RuntimeError("initial validation has no joint-paired rows")
    print(
        json.dumps(
            {"event": "initial_epoch2_validation", "eval": initial_eval},
            sort_keys=True,
        ),
        flush=True,
    )

    start = time.time()
    global_step = 0
    micro_step = 0
    running_loss = 0.0
    running_micro_count = 0
    history: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    model.zero_grad(set_to_none=True)
    progress = tqdm(total=400, desc=f"H1-A2C {args.arm}")

    for batch in train_loader:
        batch["positive"] = move_group(batch["positive"], device)
        batch["chemistry_negative"] = move_group(batch["chemistry_negative"], device)
        batch["joint_negative"] = move_group(batch["joint_negative"], device)
        loss, _ = compute_loss(
            model,
            batch,
            arm=args.arm,
            margin=float(args.margin),
            sft_weight=float(args.sft_weight),
            chemistry_weight=float(args.chemistry_weight),
            joint_weight=float(args.joint_weight),
        )
        (loss / 8.0).backward()
        running_loss += float(loss.detach().item())
        running_micro_count += 1
        micro_step += 1
        if micro_step % 8 != 0:
            continue

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1
        progress.update(1)

        if global_step == 1 or global_step % int(args.logging_steps) == 0:
            event = {
                "event": "train",
                "step": global_step,
                "mean_loss_since_log": running_loss / max(1, running_micro_count),
                "lr": float(scheduler.get_last_lr()[0]),
                "elapsed_sec": time.time() - start,
            }
            running_loss = 0.0
            running_micro_count = 0
            history.append(event)
            print(json.dumps(event, sort_keys=True), flush=True)

        if global_step % 50 == 0:
            eval_metrics = evaluate(
                model,
                val_loader,
                device,
                arm=args.arm,
                max_batches=int(args.eval_max_batches),
                margin=float(args.margin),
                sft_weight=float(args.sft_weight),
                chemistry_weight=float(args.chemistry_weight),
                joint_weight=float(args.joint_weight),
            )
            checkpoint_dir, checkpoint_manifest_sha = save_checkpoint(
                model,
                tokenizer,
                args.output_dir,
                global_step,
            )
            event = {
                "event": "checkpoint",
                "step": global_step,
                "eval": eval_metrics,
                "checkpoint_dir": str(checkpoint_dir),
                "checkpoint_manifest_sha256": checkpoint_manifest_sha,
                "elapsed_sec": time.time() - start,
            }
            checkpoints.append(event)
            history.append(event)
            print(json.dumps(event, sort_keys=True), flush=True)

        if global_step >= 400:
            break

    progress.close()
    if global_step != 400 or micro_step != 3200:
        raise RuntimeError(f"incomplete training: global_step={global_step}, micro_step={micro_step}")
    report = {
        "schema": "h1a2_jointchem_training_report_v1",
        "ok": True,
        "arm": args.arm,
        "global_step": global_step,
        "micro_step": micro_step,
        "elapsed_sec": time.time() - start,
        "execution_manifest_sha256": str(args.execution_manifest_sha256),
        "initial_adapter_sha256": observed_adapter_sha256,
        "initial_epoch2_eval": initial_eval,
        "checkpoints": checkpoints,
        "history": history,
        "automatic_crystal_evaluation_authorized": False,
        "retry_or_replacement_used": False,
        "sun_or_mlip_used_for_selection": False,
    }
    report_path = args.output_dir / "training_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "_SUCCESS").write_text(
        json.dumps(
            {
                "schema": "h1a2_jointchem_training_success_v1",
                "report_sha256": sha256_file(report_path),
                "ok": True,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
