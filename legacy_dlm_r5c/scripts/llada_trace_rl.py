#!/usr/bin/env python3
"""Lightweight TraceRL-style policy update for fixed-slot LLaDA crystals."""

from __future__ import annotations

import argparse
import json
import math
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from crystal_dlm.fixed_slot import MASK_TOKEN_ID, PROMPT_POOL, write_json
from crystal_dlm.rl_utils import normalize_rewards, trace_items_from_sequence, trace_policy_loss
from scripts.llada_sft import build_lr_scheduler, init_distributed, load_tokenizer_and_model, save_checkpoint


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def rollout_group_key(row: dict[str, Any]) -> str:
    if "prompt_id" in row:
        return f"prompt_id:{row['prompt_id']}"
    return str(row.get("prompt") or "default_prompt")


def group_normalized_advantages(rows: list[dict[str, Any]]) -> list[float]:
    groups: dict[str, list[int]] = {}
    for idx, row in enumerate(rows):
        groups.setdefault(rollout_group_key(row), []).append(idx)
    advantages = [0.0] * len(rows)
    for indices in groups.values():
        rewards = torch.tensor([float(rows[idx].get("reward", 0.0)) for idx in indices], dtype=torch.float32)
        normalized = normalize_rewards(rewards)
        for idx, value in zip(indices, normalized.tolist()):
            advantages[idx] = float(value)
    return advantages


class TraceRlDataset(Dataset):
    def __init__(
        self,
        path: Path,
        tokenizer,
        *,
        max_length: int,
        trace_shrink: int,
        max_trace_states_per_sample: int | None,
    ):
        source_rows = read_jsonl(path)
        rewards = group_normalized_advantages(source_rows)
        self.items = []
        self.old_logp: list[torch.Tensor | None] = []
        for row_idx, (row, advantage) in enumerate(zip(source_rows, rewards)):
            prompt_text = str(row["prompt"]).rstrip() + "\n"
            response = str(row["response"])
            full_text = prompt_text + response
            input_ids_list = tokenizer(
                full_text,
                add_special_tokens=False,
                truncation=True,
                max_length=max_length,
            )["input_ids"]
            prompt_length = len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])
            input_ids = torch.tensor(input_ids_list, dtype=torch.long)
            step_map = row.get("step_map")
            if not step_map:
                answer_len = max(0, int(input_ids.numel()) - int(prompt_length))
                step_map = list(range(answer_len))
            trace_items = trace_items_from_sequence(
                input_ids,
                prompt_length=min(prompt_length, input_ids.numel()),
                step_map=step_map,
                reward=float(advantage),
                mask_id=MASK_TOKEN_ID,
                shrink=trace_shrink,
                max_states=max_trace_states_per_sample,
                sample_idx=int(row.get("sample_idx", row_idx)),
                token_weights=row.get("cepo_token_weights"),
            )
            self.items.extend(trace_items)
        self.old_logp = [None] * len(self.items)
        self.source_count = len(source_rows)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.items[index]
        return {
            "id": index,
            "input_ids": item.input_ids,
            "noisy_ids": item.noisy_ids,
            "attention_mask": item.attention_mask,
            "p_mask": item.p_mask,
            "token_weights": item.token_weights,
            "advantage": torch.tensor(float(item.reward), dtype=torch.float32),
            "old_logp": self.old_logp[index],
        }


class TraceCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = int(pad_token_id)

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_len = max(item["input_ids"].numel() for item in batch)
        size = len(batch)
        input_ids = torch.full((size, max_len), self.pad_token_id, dtype=torch.long)
        noisy_ids = torch.full((size, max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((size, max_len), dtype=torch.long)
        p_mask = torch.zeros((size, max_len), dtype=torch.bool)
        old_logp = torch.zeros((size, max_len), dtype=torch.float32)
        token_weights = torch.ones((size, max_len), dtype=torch.float32)
        ids = torch.zeros((size,), dtype=torch.long)
        advantages = torch.zeros((size,), dtype=torch.float32)
        for idx, item in enumerate(batch):
            length = item["input_ids"].numel()
            ids[idx] = int(item["id"])
            input_ids[idx, :length] = item["input_ids"]
            noisy_ids[idx, :length] = item["noisy_ids"]
            attention_mask[idx, :length] = item["attention_mask"]
            p_mask[idx, :length] = item["p_mask"]
            token_weights[idx, :length] = item["token_weights"]
            advantages[idx] = item["advantage"]
            if item["old_logp"] is not None:
                old = item["old_logp"]
                old_logp[idx, : min(length, old.numel())] = old[:length]
        return {
            "ids": ids,
            "input_ids": input_ids,
            "noisy_ids": noisy_ids,
            "attention_mask": attention_mask,
            "p_mask": p_mask,
            "token_weights": token_weights,
            "advantages": advantages,
            "old_logp": old_logp,
        }


@torch.no_grad()
def fill_old_logp(model, dataset: TraceRlDataset, loader: DataLoader, device: torch.device) -> None:
    model.eval()
    for batch in tqdm(loader, desc="old logp", leave=False):
        noisy = batch["noisy_ids"].to(device)
        labels = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        logits = model(noisy, attention_mask=attention_mask).logits
        log_probs = F.log_softmax(logits, dim=-1)
        logp = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1).detach().cpu()
        for row_idx, item_id in enumerate(batch["ids"].tolist()):
            length = int(batch["attention_mask"][row_idx].sum().item())
            dataset.old_logp[int(item_id)] = logp[row_idx, :length].float().clone()
    model.train()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct/")
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--rollout-jsonl", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data/dlm_sft/mp_20")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument(
        "--skip-data-vocab-resize",
        action="store_true",
        help="Do not add data_dir/vocab_tokens.txt. Mirrors llada_sft.py for shared model loading.",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=5e-7)
    parser.add_argument("--lr-scheduler", choices=["constant", "cosine"], default="constant")
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--min-lr-ratio", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--trace-shrink", type=int, default=8)
    parser.add_argument("--max-trace-states-per-sample", type=int, default=16)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--use-lora", action="store_true", default=True)
    parser.add_argument("--no-lora", dest="use_lora", action="store_false")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", default="q_proj,k_proj,v_proj,ff_proj,up_proj")
    parser.add_argument("--modules-to-save", default="model.transformer.wte,model.transformer.ff_out")
    args = parser.parse_args()

    dist_info = init_distributed()
    is_main = dist_info["is_main"]
    device = dist_info["device"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if is_main:
        run_config = {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
        }
        run_config["distributed"] = dist_info["distributed"]
        run_config["world_size"] = dist_info["world_size"]
        write_json(str(args.output_dir / "run_config.json"), run_config)
        write_json(str(args.output_dir / "prompt_pool.json"), {"prompt_pool": PROMPT_POOL})

    tokenizer, model, num_new_tokens, tokenizer_source, model_source, *_ = load_tokenizer_and_model(args, is_main=is_main)
    if is_main:
        write_json(
            str(args.output_dir / "tokenizer_report.json"),
            {
                "tokenizer_source": tokenizer_source,
                "model_source": model_source,
                "vocab_size": len(tokenizer),
                "num_new_tokens": num_new_tokens,
                "mask_token_id": MASK_TOKEN_ID,
                "pad_token_id": tokenizer.pad_token_id,
            },
        )
    model.to(device)
    if dist_info["distributed"]:
        model = DDP(
            model,
            device_ids=[dist_info["local_rank"]],
            output_device=dist_info["local_rank"],
            find_unused_parameters=False,
        )
    model.train()

    dataset = TraceRlDataset(
        args.rollout_jsonl,
        tokenizer,
        max_length=args.max_length,
        trace_shrink=args.trace_shrink,
        max_trace_states_per_sample=args.max_trace_states_per_sample,
    )
    sampler = (
        DistributedSampler(dataset, num_replicas=dist_info["world_size"], rank=dist_info["rank"], shuffle=False)
        if dist_info["distributed"]
        else None
    )
    collator = TraceCollator(tokenizer.pad_token_id)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, shuffle=False, collate_fn=collator)
    fill_old_logp(model, dataset, loader, device)
    if dist_info["distributed"]:
        dist.barrier()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    update_steps = math.ceil(len(loader) * args.epochs / max(1, args.grad_accum))
    if args.max_train_steps:
        update_steps = min(update_steps, int(args.max_train_steps))
    scheduler = build_lr_scheduler(optimizer, args, max(1, update_steps))
    log_path = args.output_dir / "training_log.jsonl"
    log_context = log_path.open("a", encoding="utf-8") if is_main else nullcontext()
    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(total=update_steps, desc="LLaDA TraceRL", disable=not is_main)
    with log_context as log_handle:
        if is_main:
            log_handle.write(
                json.dumps(
                    {
                        "event": "start",
                        "source_rollouts": dataset.source_count,
                        "trace_items": len(dataset),
                        "distributed": dist_info["distributed"],
                        "world_size": dist_info["world_size"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            log_handle.flush()
        for _epoch in range(args.epochs):
            for micro_step, batch in enumerate(loader):
                if args.max_train_steps is not None and global_step >= int(args.max_train_steps):
                    break
                is_last_micro = micro_step == len(loader) - 1
                sync_now = (micro_step + 1) % max(1, args.grad_accum) == 0 or is_last_micro
                sync_context = (
                    model.no_sync()
                    if dist_info["distributed"] and not sync_now
                    else nullcontext()
                )
                with sync_context:
                    logits = model(
                        batch["noisy_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device),
                    ).logits
                    loss, details = trace_policy_loss(
                        logits,
                        batch["input_ids"].to(device),
                        batch["p_mask"].to(device),
                        batch["old_logp"].to(device),
                        batch["advantages"].to(device),
                        batch["token_weights"].to(device),
                        clip_eps=args.clip_eps,
                        beta=args.beta,
                    )
                    (loss / max(1, args.grad_accum)).backward()
                if not sync_now:
                    continue
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                progress.update(1)
                if is_main and global_step % args.logging_steps == 0:
                    log_handle.write(
                        json.dumps(
                            {
                                "event": "train",
                                "step": global_step,
                                "loss": float(loss.detach().cpu()),
                                "lr": optimizer.param_groups[0]["lr"],
                                **details,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    log_handle.flush()
                if global_step % args.save_steps == 0:
                    save_checkpoint(model, tokenizer, args.output_dir, global_step, is_main=is_main)
                    if dist_info["distributed"]:
                        dist.barrier()
            if args.max_train_steps is not None and global_step >= int(args.max_train_steps):
                break

    save_checkpoint(model, tokenizer, args.output_dir, global_step, is_main=is_main)
    if is_main:
        final_dir = args.output_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        target_model = model.module if hasattr(model, "module") else model
        target_model.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
    if dist_info["distributed"]:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
