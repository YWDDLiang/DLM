#!/usr/bin/env python3
"""One-GPU shared-noise K-way safety alignment for the native crystal DLM."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
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
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler

from crystal_dlm.d3po import (
    SharedGeometryCorruption,
    legal_target_log_probs,
    masked_sequence_log_ratio,
    shared_geometry_corruption,
    winner_denoising_anchor,
)
from crystal_dlm.dynamic_crystal import parse_dynamic_answer
from crystal_dlm.fixed_slot import MASK_TOKEN_ID
from crystal_dlm.listwise_alignment import (
    ListwiseLossOutput,
    shared_noise_listwise_alignment_loss,
)
from crystal_dlm.r5_dynamic_length import exact_dynamic_schema_constraints


D3PO_PATH = PROJECT_ROOT / "scripts" / "llada_d3po.py"
D3PO_SPEC = importlib.util.spec_from_file_location("llada_d3po_for_listwise", D3PO_PATH)
if D3PO_SPEC is None or D3PO_SPEC.loader is None:
    raise RuntimeError(D3PO_PATH)
D3PO = importlib.util.module_from_spec(D3PO_SPEC)
sys.modules[D3PO_SPEC.name] = D3PO
D3PO_SPEC.loader.exec_module(D3PO)


GROUP_SCHEMA = "c3fd_native_alignment_group_v1"
MANIFEST_SCHEMA = "c3fd_native_alignment_group_manifest_v1"
TRAIN_SCHEMA = "c3fd_native_listwise_safety_train_manifest_v1"
TOTAL_UPDATES = 348
GRADIENT_ACCUMULATION = 8
LEARNING_RATE = 5e-6
MAX_SEQUENCE_LENGTH = 382
LOGGING_STEPS = 10
MAD_SCALE = 1.4826
SCALE_FLOOR = 0.01
REWARD_CLIP = 5.0
REWARD_TEMPERATURE = 1.0
LINEAR_WEIGHT = 1.0
QUADRATIC_WEIGHT = 0.05
BEST_ANCHOR_WEIGHT = 0.2
ADVANTAGE_SUM_TOLERANCE = 1e-5
ALLOWED_POLICY_SEEDS = (82017, 82018)
ALLOWED_TRAINING_SEEDS = (83017, 83018)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(value), sort_keys=True) + "\n")


def validate_answer(answer: str, plan_state: Mapping[str, Any]) -> None:
    if not answer or answer != answer.strip():
        raise ValueError("candidate answer must be nonempty canonical text")
    parsed = parse_dynamic_answer(answer, strict=True)
    n_atom = int(plan_state["N"])
    if int(parsed["num_atoms"]) != n_atom or len(parsed["tokens"]) != 7 + 4 * n_atom:
        raise ValueError("candidate answer violates dynamic 7+4N")
    expected: dict[str, int] = {}
    for symbol, count in zip(plan_state["elements"], plan_state["counts"]):
        expected[str(symbol)] = int(count)
    observed: dict[str, int] = {}
    for symbol in parsed["species"]:
        observed[str(symbol)] = observed.get(str(symbol), 0) + 1
    if observed != expected:
        raise ValueError("candidate answer composition differs from Plan")


def validate_group_row(row: Mapping[str, Any], *, policy_seed: int) -> dict[str, Any]:
    value = dict(row)
    if value.get("schema") != GROUP_SCHEMA:
        raise ValueError("alignment group schema changed")
    if int(value.get("policy_seed", -1)) != int(policy_seed):
        raise ValueError("alignment group policy seed changed")
    if int(value.get("K", -1)) != 4 or float(value.get("group_weight", 0.0)) != 1.0:
        raise ValueError("alignment group K/weight changed")
    if value.get("trainable") is not True:
        raise ValueError("non-trainable groups must be preserved but excluded from loader")
    candidates = value.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 4:
        raise ValueError("alignment group candidates changed")
    best = int(value.get("best_valid_candidate_index", -1))
    if not 0 <= best < 4 or candidates[best].get("is_best_valid_anchor") is not True:
        raise ValueError("alignment group best-valid anchor changed")
    plan = value.get("plan_state")
    if not isinstance(plan, Mapping):
        raise ValueError("alignment group Plan is missing")
    prompt = str(value.get("prompt") or "")
    if not prompt.endswith("dynamic_crystal_body:"):
        raise ValueError("alignment group prompt framing changed")
    target_energies: list[float] = []
    for index, candidate in enumerate(candidates):
        if int(candidate.get("candidate_index", -1)) != index:
            raise ValueError("candidate order changed")
        answer = str(candidate.get("answer") or "")
        validate_answer(answer, plan)
        if hashlib.sha256(answer.encode("utf-8")).hexdigest() != str(
            candidate.get("answer_sha256")
        ):
            raise ValueError("candidate answer SHA changed")
        target = float(candidate["target_energy_per_atom"])
        if not math.isfinite(target):
            raise ValueError("candidate target energy must be finite")
        target_energies.append(target)
    if int(min(range(4), key=lambda index: (target_energies[index], index))) != best:
        raise ValueError("best-valid anchor is not target-energy minimum")
    invalid = [
        target_energies[index]
        for index, candidate in enumerate(candidates)
        if candidate.get("raw_direct_joint_valid") is not True
    ]
    valid = [
        target_energies[index]
        for index, candidate in enumerate(candidates)
        if candidate.get("raw_direct_joint_valid") is True
        and candidate.get("eligible_valid_energy") is True
    ]
    if invalid and valid and not min(invalid) > max(valid):
        raise ValueError("raw-invalid target is not lexicographically worst")
    return value


class ListwiseGroupDataset(Dataset):
    def __init__(self, path: Path, tokenizer: Any, *, policy_seed: int) -> None:
        self.path = path
        self.tokenizer = tokenizer
        self.policy_seed = int(policy_seed)
        self._supports: dict[int, tuple[frozenset[int], ...]] = {}
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.total_groups = len(rows)
        self.untrainable_groups = sum(row.get("trainable") is not True for row in rows)
        self.rows = [
            validate_group_row(row, policy_seed=self.policy_seed)
            for row in rows
            if row.get("trainable") is True
        ]
        if not self.rows:
            raise ValueError("alignment dataset contains no trainable groups")
        self.items = [self._tokenize(row) for row in self.rows]

    def _tokenize(self, row: Mapping[str, Any]) -> dict[str, Any]:
        prompt = str(row["prompt"]).rstrip() + "\n"
        prompt_ids = list(self.tokenizer(prompt, add_special_tokens=False)["input_ids"])
        n_atom = int(row["plan_state"]["N"])
        body_length = 7 + 4 * n_atom
        if n_atom not in self._supports:
            self._supports[n_atom] = tuple(
                frozenset(int(value) for value in support)
                for support in exact_dynamic_schema_constraints(self.tokenizer, n_atom)
            )
        supports = self._supports[n_atom]
        sequences: list[list[int]] = []
        targets: list[float] = []
        for candidate in row["candidates"]:
            answer = str(candidate["answer"])
            answer_ids = list(self.tokenizer(answer, add_special_tokens=False)["input_ids"])
            full_ids = list(self.tokenizer(prompt + answer, add_special_tokens=False)["input_ids"])
            if len(answer_ids) != body_length or full_ids[-body_length:] != answer_ids:
                raise ValueError("candidate is not one token per dynamic field")
            if len(full_ids) != len(prompt_ids) + body_length or len(full_ids) > MAX_SEQUENCE_LENGTH:
                raise ValueError("candidate prompt/body boundary or length changed")
            for token, legal in zip(answer_ids, supports):
                if int(token) not in legal:
                    raise ValueError("candidate token lies outside typed legal support")
            sequences.append(full_ids)
            targets.append(float(candidate["target_energy_per_atom"]))
        if len({len(sequence) for sequence in sequences}) != 1:
            raise ValueError("group sequence lengths differ")
        return {
            "input_ids": torch.tensor(sequences, dtype=torch.long),
            "attention_mask": torch.ones((4, len(sequences[0])), dtype=torch.long),
            "prompt_length": len(prompt_ids),
            "num_atoms": n_atom,
            "target_energies": torch.tensor(targets, dtype=torch.float32),
            "best_index": int(row["best_valid_candidate_index"]),
            "group_id": str(row["group_id"]),
        }

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.items[index]


def collate_one(batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(batch) != 1:
        raise ValueError("listwise trainer freezes one group per microbatch")
    return dict(batch[0])


def move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


@dataclass
class GroupComputation:
    output: ListwiseLossOutput
    corruption: SharedGeometryCorruption
    max_policy_reference_delta: float


def compute_group_loss(
    runtime: Any,
    batch: Mapping[str, Any],
    support_cache: Any,
    *,
    generator: torch.Generator | None,
    require_grad: bool,
    p_mask: torch.Tensor | None = None,
) -> GroupComputation:
    input_ids = batch["input_ids"]
    attention = batch["attention_mask"]
    k, _length = input_ids.shape
    if k != 4:
        raise ValueError("listwise group K changed")
    first_attention = attention[0:1]
    corruption_pair = shared_geometry_corruption(
        input_ids[0:1],
        input_ids[1:2],
        torch.tensor([batch["prompt_length"]], device=input_ids.device),
        torch.tensor([batch["num_atoms"]], device=input_ids.device),
        attention_mask=first_attention,
        generator=generator,
        p_mask=p_mask,
    )
    masked = corruption_pair.masked_positions.expand(k, -1).clone()
    geometry = corruption_pair.geometry_mask.expand(k, -1).clone()
    probabilities = corruption_pair.p_mask.expand(k).clone()
    noisy = torch.where(masked, torch.full_like(input_ids, MASK_TOKEN_ID), input_ids)
    supports = support_cache.selected(
        num_atoms=torch.full((k,), int(batch["num_atoms"]), device=input_ids.device),
        prompt_lengths=torch.full((k,), int(batch["prompt_length"]), device=input_ids.device),
        masked_positions=masked,
    )
    runtime.activate_reference()
    with torch.no_grad(), D3PO._autocast_context(input_ids.device):
        reference_logits = D3PO._model_logits(runtime.model, noisy, attention)
    reference_probs = legal_target_log_probs(
        reference_logits[masked], input_ids[masked], supports
    )
    del reference_logits
    runtime.activate_policy(trainable=require_grad)
    policy_context = nullcontext() if require_grad else torch.no_grad()
    with policy_context, D3PO._autocast_context(input_ids.device):
        policy_logits = D3PO._model_logits(runtime.model, noisy, attention)
    zero_anchor = policy_logits.reshape(-1)[0].to(torch.float32) * 0.0
    policy_probs = legal_target_log_probs(policy_logits[masked], input_ids[masked], supports)
    del policy_logits
    if not bool(torch.isfinite(reference_probs).all().item()) or not bool(
        torch.isfinite(policy_probs).all().item()
    ):
        raise FloatingPointError("policy/reference log probabilities are nonfinite")
    scores = masked_sequence_log_ratio(
        policy_probs,
        reference_probs,
        masked,
        probabilities,
        geometry_mask=geometry,
    )
    anchors = winner_denoising_anchor(policy_probs, masked, probabilities, geometry)
    output = shared_noise_listwise_alignment_loss(
        scores,
        batch["target_energies"],
        anchors,
        mad_scale=MAD_SCALE,
        scale_floor=SCALE_FLOOR,
        reward_clip=REWARD_CLIP,
        reward_temperature=REWARD_TEMPERATURE,
        linear_weight=LINEAR_WEIGHT,
        quadratic_weight=QUADRATIC_WEIGHT,
        best_anchor_weight=BEST_ANCHOR_WEIGHT,
        group_weight=1.0,
        advantage_sum_tolerance=ADVANTAGE_SUM_TOLERANCE,
    )
    if int(output.best_index) != int(batch["best_index"]):
        raise RuntimeError("runtime listwise best index differs from best-valid anchor")
    if require_grad:
        output = ListwiseLossOutput(
            loss=output.loss + zero_anchor,
            linear_loss=output.linear_loss,
            quadratic_loss=output.quadratic_loss,
            best_anchor_loss=output.best_anchor_loss,
            rewards=output.rewards,
            probabilities=output.probabilities,
            advantages=output.advantages,
            implicit_scores=output.implicit_scores,
            best_index=output.best_index,
        )
    max_delta = (
        float((policy_probs - reference_probs).abs().max().detach().cpu())
        if policy_probs.numel()
        else 0.0
    )
    return GroupComputation(output, corruption_pair, max_delta)


def validate_runtime() -> torch.device:
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("listwise alignment requires one process per policy")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("listwise alignment requires one visible GPU")
    if "A800" not in torch.cuda.get_device_name(0).upper():
        raise RuntimeError("listwise alignment requires one A800")
    return torch.device("cuda", 0)


def verify_inputs(args: argparse.Namespace) -> dict[str, Any]:
    if int(args.policy_seed) not in ALLOWED_POLICY_SEEDS:
        raise ValueError("policy_seed is not frozen")
    if int(args.seed) not in ALLOWED_TRAINING_SEEDS:
        raise ValueError("alignment training seed is not frozen")
    if (int(args.policy_seed), int(args.seed)) not in {(82017, 83017), (82018, 83018)}:
        raise ValueError("policy/training seed ledger changed")
    if Path(args.output_dir).exists():
        raise FileExistsError(args.output_dir)
    for path in (Path(args.model_path), Path(args.checkpoint_path), Path(args.data_dir)):
        if not path.exists():
            raise FileNotFoundError(path)
    manifest_path = Path(args.data_dir) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("alignment data manifest schema changed")
    if manifest.get("historical_3614_used") is not False or manifest.get(
        "prospective_outcomes_read"
    ) is not False:
        raise ValueError("alignment data used forbidden outcomes")
    data_path = Path(args.data_dir) / f"policy{int(args.policy_seed)}.jsonl"
    observed = sha256_file(data_path)
    if manifest["output_sha256"].get(data_path.name) != observed:
        raise ValueError("alignment policy data hash changed")
    return {
        "manifest": manifest,
        "manifest_sha256": sha256_file(manifest_path),
        "data_path": data_path,
        "data_sha256": observed,
    }


def run_step0_canary(runtime: Any, batch: Mapping[str, Any], support_cache: Any) -> dict[str, Any]:
    p_mask = torch.ones((1,), dtype=torch.float32, device=batch["input_ids"].device)
    result = compute_group_loss(
        runtime,
        batch,
        support_cache,
        generator=None,
        require_grad=False,
        p_mask=p_mask,
    )
    if result.max_policy_reference_delta > 1e-6:
        raise RuntimeError("policy/reference differ at step0")
    if not bool((result.output.implicit_scores.abs() <= 1e-6).all().item()):
        raise RuntimeError("step0 reference-corrected scores are nonzero")
    runtime.activate_policy(trainable=True)
    return {
        "passed": True,
        "max_policy_reference_log_prob_delta": result.max_policy_reference_delta,
        "max_abs_implicit_score": float(result.output.implicit_scores.abs().max().cpu()),
        "advantages_sum": float(result.output.advantages.sum().cpu()),
        "loss": float(result.output.loss.cpu()),
        "masked_geometry_tokens_per_candidate": int(
            result.corruption.masked_positions.sum().item()
        ),
    }


def next_batch(loader: DataLoader, iterator: Iterable[Any]) -> tuple[Any, Iterable[Any]]:
    try:
        return next(iterator), iterator  # type: ignore[arg-type]
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def train(args: argparse.Namespace) -> dict[str, Any]:
    inputs = verify_inputs(args)
    device = validate_runtime()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    log_path = output / "training_log.jsonl"
    started = time.time()
    config = {
        "schema": "c3fd_native_listwise_safety_run_v1",
        "model_path": str(Path(args.model_path).resolve()),
        "checkpoint_path": str(Path(args.checkpoint_path).resolve()),
        "data_dir": str(Path(args.data_dir).resolve()),
        "policy_seed": int(args.policy_seed),
        "seed": int(args.seed),
        "updates": TOTAL_UPDATES,
        "gradient_accumulation": GRADIENT_ACCUMULATION,
        "learning_rate": LEARNING_RATE,
        "coefficients": {
            "mad_scale": MAD_SCALE,
            "scale_floor": SCALE_FLOOR,
            "reward_clip": REWARD_CLIP,
            "reward_temperature": REWARD_TEMPERATURE,
            "linear_weight": LINEAR_WEIGHT,
            "quadratic_weight": QUADRATIC_WEIGHT,
            "best_anchor_weight": BEST_ANCHOR_WEIGHT,
        },
        "selection_or_grid": False,
    }
    write_json(output / "RUN_CONFIG.json", config)
    append_jsonl(log_path, {"event": "start", **config})
    try:
        random.seed(int(args.seed))
        torch.manual_seed(int(args.seed))
        torch.cuda.manual_seed_all(int(args.seed))
        tokenizer, runtime, adapter_report = D3PO.load_policy_and_reference_adapters(args)
        runtime.model.to(device)
        dataset = ListwiseGroupDataset(
            inputs["data_path"], tokenizer, policy_seed=int(args.policy_seed)
        )
        sampler_generator = torch.Generator().manual_seed(int(args.seed) + 101)
        sampler = RandomSampler(
            dataset,
            replacement=True,
            num_samples=len(dataset),
            generator=sampler_generator,
        )
        loader = DataLoader(
            dataset,
            batch_size=1,
            sampler=sampler,
            collate_fn=collate_one,
            num_workers=0,
            pin_memory=True,
        )
        support_cache = D3PO.DynamicLegalSupportCache(tokenizer)
        canary = move_batch(collate_one([dataset[0]]), device)
        step0 = run_step0_canary(runtime, canary, support_cache)
        append_jsonl(log_path, {"event": "step0_canary", **step0})
        runtime.activate_policy(trainable=True)
        parameters = [p for p in runtime.policy_parameters if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=LEARNING_RATE, weight_decay=0.0)
        optimizer.zero_grad(set_to_none=True)
        iterator = iter(loader)
        corruption_generator = torch.Generator(device=device).manual_seed(
            int(args.seed) + 202
        )
        for step in range(1, TOTAL_UPDATES + 1):
            sums = {
                "loss": 0.0,
                "linear_loss": 0.0,
                "quadratic_loss": 0.0,
                "best_anchor_loss": 0.0,
                "masked_tokens": 0.0,
            }
            for _ in range(GRADIENT_ACCUMULATION):
                raw, iterator = next_batch(loader, iterator)
                batch = move_batch(raw, device)
                result = compute_group_loss(
                    runtime,
                    batch,
                    support_cache,
                    generator=corruption_generator,
                    require_grad=True,
                )
                (result.output.loss / GRADIENT_ACCUMULATION).backward()
                sums["loss"] += float(result.output.loss.detach().cpu())
                sums["linear_loss"] += float(result.output.linear_loss.detach().cpu())
                sums["quadratic_loss"] += float(result.output.quadratic_loss.detach().cpu())
                sums["best_anchor_loss"] += float(
                    result.output.best_anchor_loss.detach().cpu()
                )
                sums["masked_tokens"] += float(
                    result.corruption.masked_positions.sum().item() * 4
                )
            gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
            if not bool(torch.isfinite(torch.as_tensor(gradient_norm)).item()):
                raise FloatingPointError("listwise gradient norm is nonfinite")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if step % LOGGING_STEPS == 0 or step == TOTAL_UPDATES:
                append_jsonl(
                    log_path,
                    {
                        "event": "train",
                        "step": step,
                        **{key: value / GRADIENT_ACCUMULATION for key, value in sums.items()},
                        "gradient_norm": float(torch.as_tensor(gradient_norm).cpu()),
                        "learning_rate": LEARNING_RATE,
                    },
                )
        checkpoint = D3PO.save_policy_step348(runtime, tokenizer, output)
        manifest = {
            "schema": TRAIN_SCHEMA,
            "status": "success",
            "policy_seed": int(args.policy_seed),
            "seed": int(args.seed),
            "optimizer_updates": TOTAL_UPDATES,
            "dataset": {
                "manifest_sha256": inputs["manifest_sha256"],
                "data_sha256": inputs["data_sha256"],
                "groups_total": dataset.total_groups,
                "groups_trainable": len(dataset),
                "groups_untrainable_preserved": dataset.untrainable_groups,
            },
            "adapter_report": adapter_report,
            "step0_canary": step0,
            "checkpoint": checkpoint,
            "elapsed_seconds": time.time() - started,
            "selection_or_grid": False,
        }
        write_json(output / "LISTWISE_TRAIN_MANIFEST.json", manifest)
        append_jsonl(log_path, {"event": "success", **manifest})
        (output / "_SUCCESS").touch()
        return manifest
    except Exception as exc:
        failure = {
            "schema": "c3fd_native_listwise_safety_failure_v1",
            "status": "failed",
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        write_json(output / "_FAILED.json", failure)
        append_jsonl(log_path, {"event": "failure", **failure})
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--policy-seed", type=int, choices=ALLOWED_POLICY_SEEDS, required=True)
    parser.add_argument("--seed", type=int, choices=ALLOWED_TRAINING_SEEDS, required=True)
    args = parser.parse_args()
    result = train(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
