#!/usr/bin/env python3
"""LoRA SFT for the H1 Llama formula planner."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterable

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
from crystal_dlm.c3fd_rich_expander import (  # noqa: E402
    FEATURE_DIM,
    SoftPrefixProjector,
    SoftPrefixProjectorConfig,
    build_expander_prompt,
)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def format_prompt(tokenizer, record: dict[str, Any]) -> str:
    expander_plan = record.get("expander_plan_state")
    if isinstance(expander_plan, dict):
        return build_expander_prompt(tokenizer, expander_plan)
    prompt = record.get("prompt")
    if prompt:
        return str(prompt)
    messages = record.get("messages")
    if isinstance(messages, list) and hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        return str(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    if isinstance(messages, list) and len(messages) >= 2:
        return f"System: {messages[0]['content']}\n\nUser: {messages[1]['content']}\n\nAssistant:"
    raise ValueError("H1 SFT record has neither messages nor prompt")


class FormulaPlanDataset(Dataset):
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
        answer = str(row["answer"]).strip()
        eos = self.tokenizer.eos_token or ""
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        answer_ids = self.tokenizer(answer + eos, add_special_tokens=False)["input_ids"]
        if len(answer_ids) >= self.max_length:
            raise ValueError(
                f"H1 formula answer uses {len(answer_ids)} tokens, which does not fit max_length={self.max_length}"
            )
        max_prompt_tokens = self.max_length - len(answer_ids)
        if len(prompt_ids) > max_prompt_tokens:
            # Preserve the task instruction nearest the answer boundary.  This
            # also guarantees that answer + EOS are always supervised.
            prompt_ids = prompt_ids[-max_prompt_tokens:]
        input_ids = prompt_ids + answer_ids
        labels = [-100] * len(prompt_ids) + answer_ids
        item = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "sample_weight": float(row.get("sample_weight", 1.0) or 1.0),
        }
        features = row.get("soft_prefix_features")
        if features is not None:
            if not isinstance(features, list) or len(features) != FEATURE_DIM:
                raise ValueError("soft-prefix feature vector has the wrong width")
            item["soft_prefix_features"] = torch.tensor(
                [float(value) for value in features], dtype=torch.float32
            )
        return item


def collate(batch: list[dict[str, Any]], pad_token_id: int) -> dict[str, torch.Tensor]:
    max_len = max(item["input_ids"].numel() for item in batch)
    input_ids = torch.full((len(batch), max_len), int(pad_token_id), dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    weights = torch.ones((len(batch),), dtype=torch.float32)
    for idx, item in enumerate(batch):
        length = item["input_ids"].numel()
        input_ids[idx, :length] = item["input_ids"]
        attention_mask[idx, :length] = 1
        labels[idx, :length] = item["labels"]
        weights[idx] = float(item["sample_weight"])
    output = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "sample_weight": weights,
    }
    prefix_rows = [item.get("soft_prefix_features") for item in batch]
    if any(value is not None for value in prefix_rows):
        if not all(value is not None for value in prefix_rows):
            raise ValueError("one batch cannot mix F and M conditioning rows")
        output["soft_prefix_features"] = torch.stack(prefix_rows)
    return output


def weighted_token_loss(logits: torch.Tensor, labels: torch.Tensor, sample_weight: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    losses = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view(shift_labels.shape)
    mask = (shift_labels != -100).float()
    per_sample = (losses * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    weights = sample_weight.to(device=per_sample.device, dtype=per_sample.dtype)
    return (per_sample * weights).sum() / weights.sum().clamp_min(1.0)


def forward_batch(model, batch, projector=None):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
    features = batch.get("soft_prefix_features")
    if projector is None:
        if features is not None:
            raise ValueError("M rows require a soft-prefix projector")
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs, labels
    if features is None:
        raise ValueError("soft-prefix projector requires M feature rows")
    token_embeddings = model.get_input_embeddings()(input_ids)
    prefix = projector(features.to(dtype=torch.float32)).to(
        dtype=token_embeddings.dtype
    )
    inputs_embeds = torch.cat([prefix, token_embeddings], dim=1)
    prefix_mask = torch.ones(
        (attention_mask.shape[0], prefix.shape[1]),
        dtype=attention_mask.dtype,
        device=attention_mask.device,
    )
    extended_attention = torch.cat([prefix_mask, attention_mask], dim=1)
    prefix_labels = torch.full(
        (labels.shape[0], prefix.shape[1]),
        -100,
        dtype=labels.dtype,
        device=labels.device,
    )
    extended_labels = torch.cat([prefix_labels, labels], dim=1)
    outputs = model(inputs_embeds=inputs_embeds, attention_mask=extended_attention)
    return outputs, extended_labels


@torch.no_grad()
def evaluate(model, loader: DataLoader, device: torch.device, max_batches: int, projector=None) -> float:
    model.eval()
    if projector is not None:
        projector.eval()
    total = 0.0
    count = 0
    for batch_idx, batch in enumerate(loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        batch = {key: value.to(device) for key, value in batch.items()}
        outputs, labels = forward_batch(model, batch, projector=projector)
        loss = weighted_token_loss(outputs.logits, labels, batch["sample_weight"])
        total += float(loss.item())
        count += 1
    model.train()
    if projector is not None:
        projector.train()
    return total / max(1, count)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        help="Optional PEFT/LoRA adapter checkpoint to continue training from.",
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--logging-steps", type=int, default=20)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--eval-max-batches", type=int, default=50)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--soft-prefix-length", type=int, default=0)
    parser.add_argument("--soft-prefix-projector-hidden", type=int, default=1024)
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    args = parser.parse_args()

    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_source = (
        args.checkpoint_path
        if args.checkpoint_path and (Path(args.checkpoint_path) / "tokenizer_config.json").exists()
        else args.model_path
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    config = load_llama3_compatible_config(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
    ensure_peft_cache_compat()
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    disable_peft_bnb_autodetect()

    if args.checkpoint_path:
        model = PeftModel.from_pretrained(model, args.checkpoint_path, is_trainable=True)
    else:
        lora = LoraConfig(
            r=int(args.lora_r),
            lora_alpha=int(args.lora_alpha),
            lora_dropout=float(args.lora_dropout),
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, lora)
    model.to(device)
    model.train()

    projector = None
    if int(args.soft_prefix_length) > 0:
        if SoftPrefixProjector is None:
            raise RuntimeError("PyTorch soft-prefix projector is unavailable")
        projector_config = SoftPrefixProjectorConfig(
            input_dim=FEATURE_DIM,
            prefix_length=int(args.soft_prefix_length),
            model_hidden_dim=int(config.hidden_size),
            projector_hidden_dim=int(args.soft_prefix_projector_hidden),
        )
        projector = SoftPrefixProjector(projector_config).to(device)
        projector.train()

    train_ds = FormulaPlanDataset(args.data_dir / "train.jsonl", tokenizer, args.max_length)
    val_ds = FormulaPlanDataset(args.data_dir / "val.jsonl", tokenizer, args.max_length)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        collate_fn=lambda batch: collate(batch, int(tokenizer.pad_token_id)),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        collate_fn=lambda batch: collate(batch, int(tokenizer.pad_token_id)),
    )

    updates_per_epoch = math.ceil(len(train_loader) / max(1, int(args.grad_accum)))
    total_updates = max(1, int(math.ceil(float(args.epochs) * updates_per_epoch)))
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if projector is not None:
        trainable_parameters.extend(projector.parameters())
    optimizer = torch.optim.AdamW(trainable_parameters, lr=float(args.lr), weight_decay=float(args.weight_decay))
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_steps), num_training_steps=total_updates)

    write_payload = {
        "model_path": args.model_path,
        "checkpoint_path": args.checkpoint_path,
        "data_dir": str(args.data_dir),
        "max_length": int(args.max_length),
        "epochs": float(args.epochs),
        "batch_size": int(args.batch_size),
        "grad_accum": int(args.grad_accum),
        "lr": float(args.lr),
        "seed": int(args.seed),
        "total_updates": total_updates,
        "train_rows": len(train_ds),
        "val_rows": len(val_ds),
        "lora": {
            "r": int(args.lora_r),
            "alpha": int(args.lora_alpha),
            "dropout": float(args.lora_dropout),
            "continued_from_checkpoint": bool(args.checkpoint_path),
        },
        "soft_prefix": (
            None
            if projector is None
            else projector.config.to_dict()
        ),
    }
    (args.output_dir / "train_config.json").write_text(json.dumps(write_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    start = time.time()
    global_step = 0
    micro_step = 0
    running = 0.0
    history: list[dict[str, Any]] = []
    model.zero_grad(set_to_none=True)
    progress = tqdm(total=total_updates, desc="H1 Llama planner SFT")
    while global_step < total_updates:
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs, labels = forward_batch(model, batch, projector=projector)
            loss = weighted_token_loss(outputs.logits, labels, batch["sample_weight"])
            (loss / max(1, int(args.grad_accum))).backward()
            running += float(loss.item())
            micro_step += 1
            if micro_step % max(1, int(args.grad_accum)) == 0:
                torch.nn.utils.clip_grad_norm_(trainable_parameters, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                progress.update(1)
                if global_step % int(args.logging_steps) == 0 or global_step == 1:
                    event = {
                        "step": global_step,
                        "train_loss_recent": running / max(1, int(args.logging_steps)),
                        "lr": float(scheduler.get_last_lr()[0]),
                        "elapsed_sec": time.time() - start,
                    }
                    running = 0.0
                    print(json.dumps(event, ensure_ascii=False), flush=True)
                    history.append(event)
                if int(args.eval_steps) > 0 and global_step % int(args.eval_steps) == 0:
                    eval_loss = evaluate(
                        model,
                        val_loader,
                        device,
                        int(args.eval_max_batches),
                        projector=projector,
                    )
                    event = {"step": global_step, "eval_loss": eval_loss, "elapsed_sec": time.time() - start}
                    print(json.dumps(event, ensure_ascii=False), flush=True)
                    history.append(event)
                if global_step >= total_updates:
                    break
        if global_step >= total_updates:
            break

    final_eval = evaluate(
        model,
        val_loader,
        device,
        int(args.eval_max_batches),
        projector=projector,
    )
    final_dir = args.output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    if projector is not None:
        torch.save(projector.state_dict(), final_dir / "soft_prefix_projector.pt")
        (final_dir / "soft_prefix_projector_config.json").write_text(
            json.dumps(projector.config.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    metrics = {
        "global_step": global_step,
        "final_eval_loss": final_eval,
        "elapsed_sec": time.time() - start,
        "history": history,
    }
    (args.output_dir / "train_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
