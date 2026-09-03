#!/usr/bin/env python3
"""Train SPAD reference-control or terminal-energy backfill posterior LoRA."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback
from types import SimpleNamespace
from typing import Any, Iterable, Mapping

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from crystal_dlm.fixed_slot import MASK_TOKEN_ID
from crystal_dlm.r5_dynamic_length import exact_body_token_count
from crystal_dlm.spad_energy_posterior import (
    SPAD_E_KL_BUDGET_NATS,
    build_spad_energy_posterior,
    spad_energy_posterior_loss,
)


D3PO_PATH = Path(__file__).with_name("llada_d3po.py")
D3PO_SPEC = importlib.util.spec_from_file_location("llada_d3po_for_spad_energy", D3PO_PATH)
if D3PO_SPEC is None or D3PO_SPEC.loader is None:
    raise RuntimeError(D3PO_PATH)
D3PO = importlib.util.module_from_spec(D3PO_SPEC)
sys.modules[D3PO_SPEC.name] = D3PO
D3PO_SPEC.loader.exec_module(D3PO)


SCHEMA = "spad_energy_group_v1"
TRAIN_SCHEMA = "spad_energy_train_v1"
UPDATES = 348
GRADIENT_ACCUMULATION = 4
LEARNING_RATE = 5.0e-6
NOOP_CE_WEIGHT = 0.1
REFERENCE_KL_WEIGHT = 1.0


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n")


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(value), sort_keys=True) + "\n")


class SPADEnergyDataset(Dataset):
    def __init__(self, path: Path, tokenizer: Any) -> None:
        self.rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.total_groups = len(self.rows)
        self.rows = [row for row in self.rows if row.get("trainable") is True]
        self.untrainable_groups = self.total_groups - len(self.rows)
        if self.total_groups != 2048 or not self.rows:
            raise ValueError("SPAD-E group accounting changed")
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        if row.get("schema") != SCHEMA or int(row.get("K", 0)) != 4:
            raise ValueError("SPAD-E group schema changed")
        candidates = row["candidates"]
        if len(candidates) != 4 or candidates[0].get("mandatory_noop") is not True:
            raise ValueError("SPAD-E no-op/K contract changed")
        prompt_ids = self.tokenizer(
            str(row["prompt"]), add_special_tokens=False
        )["input_ids"]
        source_ids = self.tokenizer(
            str(row["source_answer"]), add_special_tokens=False
        )["input_ids"]
        num_atoms = int(row["plan_state"]["N"])
        if len(source_ids) != exact_body_token_count(num_atoms):
            raise ValueError("SPAD-E source is not exact 7+4N")
        active = tuple(int(value) for value in row["active_positions"])
        if len(active) != 3 or len(set(active)) != 3:
            raise ValueError("SPAD-E active positions are not one XYZ triplet")
        candidate_triplets: list[list[int]] = []
        legal: list[bool] = []
        energies: list[float | None] = []
        for candidate in candidates:
            answer = candidate.get("answer")
            action = candidate.get("action_triplet_tokens")
            is_legal = candidate.get("valid_action") is True
            energy = candidate.get("terminal_energy_per_atom")
            legal.append(bool(is_legal))
            energies.append(None if energy is None else float(energy))
            if not is_legal:
                candidate_triplets.append([source_ids[position] for position in active])
                continue
            if not isinstance(answer, str) or not isinstance(action, list) or len(action) != 3:
                raise ValueError("legal SPAD-E action lacks tokens")
            answer_ids = self.tokenizer(answer, add_special_tokens=False)["input_ids"]
            if len(answer_ids) != len(source_ids):
                raise ValueError("SPAD-E candidate length changed")
            differences = [
                position
                for position, (left, right) in enumerate(
                    zip(source_ids, answer_ids, strict=True)
                )
                if left != right
            ]
            if not set(differences) <= set(active):
                raise ValueError("SPAD-E candidate escaped active XYZ")
            candidate_triplets.append([int(answer_ids[position]) for position in active])
        no_op = [source_ids[position] for position in active]
        if candidate_triplets[0] != no_op or not legal[0] or energies[0] is None:
            raise ValueError("SPAD-E trainable group lacks legal energy-known no-op")
        input_ids = torch.tensor(prompt_ids + source_ids, dtype=torch.long)
        absolute = torch.tensor(
            [len(prompt_ids) + position for position in active], dtype=torch.long
        )
        input_ids[absolute] = int(MASK_TOKEN_ID)
        return {
            "group_idx": int(row["group_idx"]),
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "active_positions": absolute,
            "action_triplets": torch.tensor(candidate_triplets, dtype=torch.long),
            "legal_mask": torch.tensor(legal, dtype=torch.bool),
            "terminal_energies": torch.tensor(
                [math.nan if value is None else value for value in energies],
                dtype=torch.float64,
            ),
        }


def collate_one(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != 1:
        raise ValueError("SPAD-E trainer uses one common state per microbatch")
    return rows[0]


def move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def model_logits(model: Any, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    output = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = getattr(output, "logits", None)
    if logits is None or logits.ndim != 3:
        raise RuntimeError("DLM did not return rank-three logits")
    return logits


def sequential_action_scores(
    runtime: Any,
    batch: Mapping[str, Any],
    *,
    reference: bool,
) -> torch.Tensor:
    if reference:
        runtime.activate_reference()
        context = torch.no_grad()
    else:
        runtime.activate_policy(trainable=True)
        context = nullcontext()
    source = batch["input_ids"].reshape(1, -1)
    attention = batch["attention_mask"].reshape(1, -1)
    positions = batch["active_positions"]
    actions = batch["action_triplets"]
    count = int(actions.shape[0])
    with context, autocast_context(source.device):
        logits_x = model_logits(runtime.model, source, attention)
        logp_x = F.log_softmax(logits_x[0, positions[0]].float(), dim=-1).index_select(
            0, actions[:, 0]
        )
        stage_y = source.repeat(count, 1)
        stage_attention = attention.repeat(count, 1)
        stage_y[:, positions[0]] = actions[:, 0]
        logits_y = model_logits(runtime.model, stage_y, stage_attention)
        rows = torch.arange(count, device=source.device)
        logp_y = F.log_softmax(logits_y[rows, positions[1]].float(), dim=-1).gather(
            1, actions[:, 1].unsqueeze(1)
        ).squeeze(1)
        stage_z = stage_y.clone()
        stage_z[:, positions[1]] = actions[:, 1]
        logits_z = model_logits(runtime.model, stage_z, stage_attention)
        logp_z = F.log_softmax(logits_z[rows, positions[2]].float(), dim=-1).gather(
            1, actions[:, 2].unsqueeze(1)
        ).squeeze(1)
    return logp_x + logp_y + logp_z


def group_loss(
    runtime: Any,
    batch: Mapping[str, Any],
    *,
    mode: str,
    reference_cache: dict[int, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor | float]:
    group_idx = int(batch["group_idx"])
    cached = None if reference_cache is None else reference_cache.get(group_idx)
    if cached is None:
        reference_scores = sequential_action_scores(runtime, batch, reference=True)
        if reference_cache is not None:
            reference_cache[group_idx] = reference_scores.detach().cpu()
    else:
        reference_scores = cached.to(batch["input_ids"].device)
    policy_scores = sequential_action_scores(runtime, batch, reference=False)
    posterior = build_spad_energy_posterior(
        reference_scores,
        batch["terminal_energies"],
        batch["legal_mask"],
        action_triplets=batch["action_triplets"].detach().cpu().tolist(),
        no_op_triplet=batch["action_triplets"][0].detach().cpu().tolist(),
        kl_budget_nats=SPAD_E_KL_BUDGET_NATS,
    )
    target = spad_energy_posterior_loss(policy_scores, posterior)
    legal = batch["legal_mask"]
    policy_log = torch.log_softmax(policy_scores.masked_fill(~legal, -torch.inf), dim=0)
    reference_prob = posterior.reference_probabilities.to(
        device=policy_scores.device, dtype=policy_scores.dtype
    )
    support = reference_prob > 0
    reference_kl = torch.sum(
        reference_prob[support]
        * (
            torch.log(reference_prob[support])
            - policy_log[support]
        )
    )
    noop_nll = -policy_scores[0] / 3.0
    terminal_kl = target.kl if mode == "energy" else reference_kl
    loss = terminal_kl + REFERENCE_KL_WEIGHT * reference_kl + NOOP_CE_WEIGHT * noop_nll
    return {
        "loss": loss,
        "terminal_kl": terminal_kl,
        "reference_kl": reference_kl,
        "noop_nll": noop_nll,
        "teacher_kl_nats": float(posterior.kl_nats),
        "teacher_tilt": float(posterior.tilt),
        "legal_actions": float(posterior.legal_action_count),
        "known_actions": float(posterior.legal_known_energy_count),
        "max_policy_reference_score_delta": float(
            torch.max(torch.abs(policy_scores.detach() - reference_scores.detach())).cpu()
        ),
    }


def next_batch(loader: DataLoader, iterator: Iterable[Any]) -> tuple[Any, Iterable[Any]]:
    try:
        return next(iterator), iterator  # type: ignore[arg-type]
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--groups", type=Path, required=True)
    parser.add_argument("--vocab-data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("control", "energy"), required=True)
    parser.add_argument("--seed", type=int, default=98017)
    args = parser.parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("SPAD-E uses one process and one visible GPU per cell")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("SPAD-E cell requires exactly one visible GPU")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    log = args.output_dir / "training_log.jsonl"
    started = time.time()
    config = {
        "schema": TRAIN_SCHEMA,
        "mode": args.mode,
        "seed": int(args.seed),
        "updates": UPDATES,
        "gradient_accumulation": GRADIENT_ACCUMULATION,
        "learning_rate": LEARNING_RATE,
        "noop_ce_weight": NOOP_CE_WEIGHT,
        "reference_kl_weight": REFERENCE_KL_WEIGHT,
        "teacher_kl_budget_nats": SPAD_E_KL_BUDGET_NATS,
        "validity_before_energy": True,
        "inference_time_critic": False,
    }
    write_json(args.output_dir / "RUN_CONFIG.json", config)
    append_jsonl(log, {"event": "start", **config})
    try:
        random.seed(int(args.seed))
        torch.manual_seed(int(args.seed))
        torch.cuda.manual_seed_all(int(args.seed))
        loader_args = SimpleNamespace(
            model_path=args.model_path,
            checkpoint_path=args.checkpoint_path,
            data_dir=args.vocab_data_dir,
        )
        tokenizer, runtime, adapter_report = D3PO.load_policy_and_reference_adapters(
            loader_args
        )
        device = torch.device("cuda", 0)
        runtime.model.to(device)
        dataset = SPADEnergyDataset(args.groups, tokenizer)
        reference_cache: dict[int, torch.Tensor] = {}
        sampler = RandomSampler(
            dataset,
            replacement=True,
            num_samples=len(dataset),
            generator=torch.Generator().manual_seed(int(args.seed) + 101),
        )
        loader = DataLoader(
            dataset,
            batch_size=1,
            sampler=sampler,
            collate_fn=collate_one,
            num_workers=0,
            pin_memory=True,
        )
        canary = move_batch(dataset[0], device)
        step0 = group_loss(
            runtime,
            canary,
            mode=args.mode,
            reference_cache=reference_cache,
        )
        if step0["max_policy_reference_score_delta"] > 1.0e-6:
            raise RuntimeError("policy/reference action scores differ at step0")
        if step0["teacher_kl_nats"] > SPAD_E_KL_BUDGET_NATS + 1.0e-9:
            raise RuntimeError("teacher posterior escaped KL budget")
        append_jsonl(
            log,
            {
                "event": "step0",
                **{
                    key: float(value.detach().cpu()) if isinstance(value, torch.Tensor) else value
                    for key, value in step0.items()
                },
            },
        )
        runtime.activate_policy(trainable=True)
        parameters = [parameter for parameter in runtime.policy_parameters if parameter.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=LEARNING_RATE, weight_decay=0.0)
        optimizer.zero_grad(set_to_none=True)
        iterator = iter(loader)
        for step in range(1, UPDATES + 1):
            sums = CounterFloat()
            for _ in range(GRADIENT_ACCUMULATION):
                raw, iterator = next_batch(loader, iterator)
                batch = move_batch(raw, device)
                values = group_loss(
                    runtime,
                    batch,
                    mode=args.mode,
                    reference_cache=reference_cache,
                )
                loss = values["loss"]
                if not isinstance(loss, torch.Tensor) or not bool(torch.isfinite(loss).item()):
                    raise FloatingPointError("SPAD-E loss is nonfinite")
                (loss / GRADIENT_ACCUMULATION).backward()
                for key in ("loss", "terminal_kl", "reference_kl", "noop_nll"):
                    value = values[key]
                    assert isinstance(value, torch.Tensor)
                    sums[key] += float(value.detach().cpu())
                for key in ("teacher_kl_nats", "legal_actions", "known_actions"):
                    sums[key] += float(values[key])
            gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            if not bool(torch.isfinite(torch.as_tensor(gradient_norm)).item()):
                raise FloatingPointError("SPAD-E gradient is nonfinite")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if step == 1 or step % 10 == 0 or step == UPDATES:
                append_jsonl(
                    log,
                    {
                        "event": "train",
                        "step": step,
                        **{
                            key: value / GRADIENT_ACCUMULATION
                            for key, value in sums.items()
                        },
                        "gradient_norm": float(torch.as_tensor(gradient_norm).cpu()),
                        "elapsed_sec": time.time() - started,
                    },
                )
        checkpoint = D3PO.save_policy_step348(runtime, tokenizer, args.output_dir)
        report = {
            "schema": TRAIN_SCHEMA,
            "status": "success",
            "mode": args.mode,
            "seed": int(args.seed),
            "updates": UPDATES,
            "groups_total": dataset.total_groups,
            "groups_trainable": len(dataset),
            "groups_untrainable": dataset.untrainable_groups,
            "step0": {
                key: float(value.detach().cpu()) if isinstance(value, torch.Tensor) else value
                for key, value in step0.items()
            },
            "adapter_report": adapter_report,
            "checkpoint": checkpoint,
            "elapsed_sec": time.time() - started,
        }
        write_json(args.output_dir / "TRAIN_FINAL.json", report)
        append_jsonl(log, {"event": "success", **report})
        (args.output_dir / "_SUCCESS").touch()
        print(json.dumps(report, sort_keys=True))
    except Exception as exc:
        failure = {
            "schema": TRAIN_SCHEMA,
            "status": "failed",
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        write_json(args.output_dir / "_FAILED.json", failure)
        append_jsonl(log, {"event": "failure", **failure})
        raise


class CounterFloat(dict[str, float]):
    def __missing__(self, key: str) -> float:
        return 0.0


if __name__ == "__main__":
    main()
