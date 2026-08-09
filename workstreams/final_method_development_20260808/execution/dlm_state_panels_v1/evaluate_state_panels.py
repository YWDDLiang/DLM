#!/usr/bin/env python3
"""Freeze and score common IID/D1/safe-axis/actual-rollout DLM states."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[3]
SAFE_AXIS_ROOT = (
    PROJECT_ROOT
    / "workstreams"
    / "plangraph_dlm_iclr_20260731"
    / "execution"
    / "h1_body_safeaxis256_v1"
)
for location in (PROJECT_ROOT, SAFE_AXIS_ROOT):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from crystal_dlm.fixed_slot import MASK_TOKEN_ID  # noqa: E402
from crystal_dlm.h1a2_factorial_runtime import (  # noqa: E402
    assert_body_tokenizer_identity,
)
from crystal_dlm.llada_generation import (  # noqa: E402
    _apply_lightweight_decoding_masks,
    _apply_schema_masks,
    _model_logits,
    _prepare_atom_count_grammar,
    _validate_generation_position_groups,
    get_num_transfer_tokens,
)
from crystal_dlm.planned_corruption import (  # noqa: E402
    corruption_key_for_record,
    current_order_groups,
    safe_axis_dependency_groups,
    sample_iid_corruption,
    sample_planned_corruption,
)
from crystal_dlm.r5_dynamic_length import (  # noqa: E402
    count_prefill_for_batch,
    exact_dynamic_schema_constraints,
)
from paired_llada import _paired_suffix_candidates  # noqa: E402
from scripts.sample_llada_dynamic_crystals import (  # noqa: E402
    build_dynamic_lightweight_constraints,
    load_model_and_tokenizer,
)
from scripts.sample_llada_r5_exact_length import (  # noqa: E402
    element_prefill_for_batch,
    merge_prefill_maps,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_sha256(value: Any) -> str:
    return sha256_text(
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path} contains a non-object row")
            rows.append(value)
    return rows


def write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_jsonl_exclusive(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def require_sha(path: Path, expected: str, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} is missing: {resolved}")
    observed = sha256_file(resolved)
    if observed != str(expected):
        raise ValueError(f"{label} SHA changed: {observed}")
    return resolved


def validate_runtime() -> torch.device:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("state panels must run through Slurm")
    if os.environ.get("SLURM_JOB_PARTITION") != "gpu":
        raise RuntimeError("state panels require the gpu partition")
    if int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) != 8:
        raise RuntimeError("state panels require exactly eight CPUs")
    if os.environ.get("CONDA_DEFAULT_ENV") != "diff_meets_diff":
        raise RuntimeError("state panels require diff_meets_diff")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("state panels require exactly one CUDA device")
    name = torch.cuda.get_device_name(0)
    if "A800" not in name:
        raise RuntimeError(f"state panels require A800, observed {name}")
    return torch.device("cuda", 0)


def stable_seed(*parts: Any) -> int:
    digest = hashlib.sha256(
        json.dumps(parts, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) & ((1 << 63) - 1)


def quantile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * float(probability)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def graph_plan(graph: Mapping[str, Any]) -> dict[str, Any]:
    composition = graph["composition"]
    return {
        "N": int(composition["N"]),
        "elements": [str(value) for value in composition["elements"]],
        "counts": [int(value) for value in composition["counts"]],
    }


def tokenize_row(tokenizer, row: Mapping[str, Any]) -> dict[str, Any]:
    prompt = str(row["prompt"]).rstrip() + "\n"
    answer = str(row["answer"])
    prompt_ids = list(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    answer_ids = list(tokenizer(answer, add_special_tokens=False)["input_ids"])
    full_ids = list(
        tokenizer(prompt + answer, add_special_tokens=False)["input_ids"]
    )
    if full_ids != [*prompt_ids, *answer_ids]:
        raise ValueError("prompt/answer tokenization is not additive")
    num_atoms = int(row["plangraph"]["composition"]["N"])
    expected = 7 + 4 * num_atoms
    if len(answer_ids) != expected:
        raise ValueError(
            f"dynamic-v1 token count changed: {len(answer_ids)} != {expected}"
        )
    return {
        "prompt": prompt,
        "prompt_ids": prompt_ids,
        "answer_ids": answer_ids,
        "full_ids": full_ids,
        "prompt_length": len(prompt_ids),
        "num_atoms": num_atoms,
    }


def state_identity(record: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {
            "panel_type": record["panel_type"],
            "validation_ordinal": record["validation_ordinal"],
            "active_group_index": record.get("active_group_index"),
            "step_in_group": record.get("step_in_group"),
            "input_ids": record["input_ids"],
            "attention_mask": record["attention_mask"],
            "target_positions": record["target_positions"],
            "target_token_ids": record["target_token_ids"],
        }
    )


def synthetic_record(
    *,
    panel_type: str,
    validation_ordinal: int,
    row: Mapping[str, Any],
    tokenized: Mapping[str, Any],
    masked_positions: Sequence[int],
    target_positions: Sequence[int],
    active_group_index: int | None,
    active_group: str | None,
    active_group_width: int,
) -> dict[str, Any]:
    prompt_length = int(tokenized["prompt_length"])
    input_ids = list(tokenized["full_ids"])
    for relative in masked_positions:
        input_ids[prompt_length + int(relative)] = int(MASK_TOKEN_ID)
    targets = [int(tokenized["answer_ids"][int(value)]) for value in target_positions]
    absolute_targets = [prompt_length + int(value) for value in target_positions]
    remaining = sum(token == int(MASK_TOKEN_ID) for token in input_ids[prompt_length:])
    record: dict[str, Any] = {
        "schema": "evidence_first_dlm_state_v1",
        "panel_type": str(panel_type),
        "validation_ordinal": int(validation_ordinal),
        "training_pair_sha256": str(row["training_pair_sha256"]),
        "num_atoms": int(tokenized["num_atoms"]),
        "active_group_index": active_group_index,
        "active_group": active_group,
        "step_in_group": None,
        "active_group_width": int(active_group_width),
        "active_masked_count": len(targets),
        "remaining_mask_count": int(remaining),
        "remaining_mask_fraction": remaining / len(tokenized["answer_ids"]),
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "target_positions": absolute_targets,
        "target_token_ids": targets,
        "visible_wrong_commitments": 0,
        "commit_count": None,
        "commit_correct_count": None,
    }
    record["state_id"] = state_identity(record)
    return record


def build_synthetic_states(
    rows: Sequence[Mapping[str, Any]],
    *,
    tokenizer,
    panel_seed: int,
) -> list[dict[str, Any]]:
    states: list[dict[str, Any]] = []
    for ordinal, row in enumerate(rows):
        tokenized = tokenize_row(tokenizer, row)
        answer_length = len(tokenized["answer_ids"])
        key = corruption_key_for_record(row)

        iid = sample_iid_corruption(
            answer_length,
            rng=random.Random(stable_seed(panel_seed, key, "iid")),
        )
        states.append(
            synthetic_record(
                panel_type="iid",
                validation_ordinal=ordinal,
                row=row,
                tokenized=tokenized,
                masked_positions=iid.masked_input_positions,
                target_positions=iid.loss_positions,
                active_group_index=None,
                active_group=None,
                active_group_width=answer_length,
            )
        )

        for panel_type, groups in (
            ("d1", current_order_groups(int(tokenized["num_atoms"]))),
            ("safe_axis_synthetic", safe_axis_dependency_groups(row["plangraph"])),
        ):
            for group_index, group in enumerate(groups):
                sample = sample_planned_corruption(
                    groups,
                    rng=random.Random(
                        stable_seed(panel_seed, key, panel_type, group_index)
                    ),
                    active_group_index=group_index,
                    policy_name=panel_type,
                )
                states.append(
                    synthetic_record(
                        panel_type=panel_type,
                        validation_ordinal=ordinal,
                        row=row,
                        tokenized=tokenized,
                        masked_positions=sample.masked_input_positions,
                        target_positions=sample.loss_positions,
                        active_group_index=group_index,
                        active_group=group.name,
                        active_group_width=len(group.positions),
                    )
                )
    if len({row["state_id"] for row in states}) != len(states):
        raise ValueError("synthetic state identities are not unique")
    return states


def _raw_state_metrics(
    logits: torch.Tensor,
    *,
    positions: Sequence[int],
    targets: Sequence[int],
) -> dict[str, Any]:
    device = logits.device
    selected = logits[
        torch.tensor(positions, dtype=torch.long, device=device)
    ].float()
    target_tensor = torch.tensor(targets, dtype=torch.long, device=device)
    losses = F.cross_entropy(selected, target_tensor, reduction="none")
    probabilities = F.softmax(selected, dim=-1)
    confidence, predicted = probabilities.max(dim=-1)
    correct = predicted.eq(target_tensor)
    return {
        "target_count": len(targets),
        "nll_sum": float(losses.sum().detach().cpu()),
        "mean_nll": float(losses.mean().detach().cpu()),
        "correct_count": int(correct.sum().detach().cpu()),
        "confidence_sum": float(confidence.sum().detach().cpu()),
        "brier_sum": float(
            (
                probabilities.square().sum(dim=-1)
                - 2.0
                * probabilities.gather(1, target_tensor.unsqueeze(1)).squeeze(1)
                + 1.0
            )
            .sum()
            .detach()
            .cpu()
        ),
        "calibration": [
            [float(conf), bool(ok)]
            for conf, ok in zip(
                confidence.detach().cpu().tolist(),
                correct.detach().cpu().tolist(),
                strict=True,
            )
        ],
    }


def build_allowed_mask(
    *,
    tokenizer,
    model,
    num_atoms: int,
) -> tuple[torch.Tensor, Any]:
    allowed = exact_dynamic_schema_constraints(tokenizer, int(num_atoms))
    vocab_size = model.get_output_embeddings().weight.shape[0]
    device = next(model.parameters()).device
    mask = torch.zeros(
        (len(allowed), vocab_size), dtype=torch.bool, device=device
    )
    for position, token_ids in enumerate(allowed):
        if not token_ids:
            raise ValueError(f"schema position {position} has no legal token")
        mask[
            position,
            torch.tensor(token_ids, dtype=torch.long, device=device),
        ] = True
    grammar = _prepare_atom_count_grammar(None, vocab_size, device)
    return mask, grammar


def build_actual_rollout_states(
    rows: Sequence[Mapping[str, Any]],
    *,
    tokenizer,
    model,
    decoder: Mapping[str, Any],
    panel_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    device = next(model.parameters()).device
    lightweight = build_dynamic_lightweight_constraints(
        tokenizer,
        duplicate_coordinate_mask=bool(decoder["duplicate_coordinate_mask"]),
        lattice_volume_mask=bool(decoder["lattice_volume_mask"]),
        min_lattice_rad=float(decoder["min_lattice_rad"]),
    )
    tasks: list[dict[str, Any]] = []
    for ordinal, row in enumerate(rows):
        tokenized = tokenize_row(tokenizer, row)
        groups = safe_axis_dependency_groups(row["plangraph"])
        schedule = [list(group.positions) for group in groups]
        tasks.append(
            {
                "ordinal": ordinal,
                "row": row,
                "tokenized": tokenized,
                "groups": groups,
                "schedule": schedule,
                "schedule_sha256": canonical_sha256(schedule),
                "plan": graph_plan(row["plangraph"]),
                "noise_seed": stable_seed(
                    panel_seed, row["training_pair_sha256"], "actual_rollout"
                ),
            }
        )

    buckets: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        buckets[(task["tokenized"]["num_atoms"], task["schedule_sha256"])].append(task)

    states: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    max_batch = int(decoder["max_batch_size"])
    for key in sorted(buckets):
        bucket = sorted(buckets[key], key=lambda item: item["ordinal"])
        for offset in range(0, len(bucket), max_batch):
            batch = bucket[offset : offset + max_batch]
            prompts = [task["tokenized"]["prompt"] for task in batch]
            encoded = tokenizer(
                prompts,
                add_special_tokens=False,
                padding=True,
                return_tensors="pt",
            )
            prompt_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            prompt_width = int(prompt_ids.shape[1])
            num_atoms = int(batch[0]["tokenized"]["num_atoms"])
            gen_length = 7 + 4 * num_atoms
            x = torch.full(
                (len(batch), prompt_width + gen_length),
                int(MASK_TOKEN_ID),
                dtype=torch.long,
                device=device,
            )
            x[:, :prompt_width] = prompt_ids
            prefill = merge_prefill_maps(
                count_prefill_for_batch(tokenizer, num_atoms, len(batch)),
                element_prefill_for_batch(
                    tokenizer, [task["plan"] for task in batch]
                ),
            )
            for generation_position, values in prefill.items():
                x[:, prompt_width + int(generation_position)] = torch.tensor(
                    values, dtype=torch.long, device=device
                )
            full_attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones(
                        (len(batch), gen_length),
                        dtype=attention_mask.dtype,
                        device=device,
                    ),
                ],
                dim=-1,
            )
            prompt_index = x != int(MASK_TOKEN_ID)
            allowed_mask, atom_count_grammar = build_allowed_mask(
                tokenizer=tokenizer, model=model, num_atoms=num_atoms
            )
            schedule = batch[0]["schedule"]
            groups = batch[0]["groups"]
            base_seeds = [int(task["noise_seed"]) for task in batch]
            ground_truth = [task["tokenized"]["answer_ids"] for task in batch]

            for group_index, group_positions in enumerate(
                _validate_generation_position_groups(schedule, gen_length)
            ):
                absolute = torch.tensor(
                    [prompt_width + position for position in group_positions],
                    dtype=torch.long,
                    device=device,
                )
                group_allowed = torch.zeros_like(x, dtype=torch.bool)
                group_allowed[:, absolute] = True
                group_mask = (x == int(MASK_TOKEN_ID)) & group_allowed
                group_steps = int(group_mask.sum(dim=1).max().detach().item())
                if group_steps <= 0:
                    continue
                transfers = get_num_transfer_tokens(group_mask, group_steps)
                for step_in_group in range(group_steps):
                    mask_index = x == int(MASK_TOKEN_ID)
                    raw_logits = _model_logits(
                        model,
                        x,
                        full_attention_mask,
                        prompt_index,
                        float(decoder["cfg_scale"]),
                        int(MASK_TOKEN_ID),
                    )
                    constrained_logits = raw_logits.clone()
                    _apply_schema_masks(
                        constrained_logits,
                        x,
                        prompt_width,
                        gen_length,
                        allowed_mask,
                        atom_count_grammar,
                    )
                    _apply_lightweight_decoding_masks(
                        constrained_logits,
                        x,
                        prompt_width,
                        gen_length,
                        lightweight,
                    )
                    x0, confidence = _paired_suffix_candidates(
                        constrained_logits,
                        current_tokens=x,
                        prompt_length=prompt_width,
                        gen_length=gen_length,
                        temperature=float(decoder["temperature"]),
                        remasking=str(decoder["remasking"]),
                        base_seeds=base_seeds,
                        semantic_group=group_index,
                        step_in_group=step_in_group,
                    )
                    x0 = torch.where(mask_index, x0, x)
                    confidence = torch.where(
                        mask_index & group_allowed, confidence, -float("inf")
                    )
                    transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                    selected_by_row: list[list[int]] = []
                    for row_index in range(len(batch)):
                        count = int(transfers[row_index, step_in_group].item())
                        if count <= 0:
                            selected_by_row.append([])
                            continue
                        _values, selected = torch.topk(
                            confidence[row_index], k=count
                        )
                        selected_list = [int(value) for value in selected.tolist()]
                        if any(
                            not bool(group_allowed[row_index, value])
                            or not bool(mask_index[row_index, value])
                            for value in selected_list
                        ):
                            raise ValueError("decoder selected outside the active mask")
                        transfer_index[row_index, selected] = True
                        selected_by_row.append(selected_list)

                    for row_index, task in enumerate(batch):
                        active_absolute = [
                            int(value)
                            for value in absolute.tolist()
                            if bool(mask_index[row_index, int(value)])
                        ]
                        if not active_absolute:
                            raise ValueError("actual rollout state has no active target")
                        relative_targets = [
                            value - prompt_width for value in active_absolute
                        ]
                        target_ids = [
                            int(ground_truth[row_index][value])
                            for value in relative_targets
                        ]
                        suffix = x[row_index, prompt_width : prompt_width + gen_length]
                        target_tensor = torch.tensor(
                            ground_truth[row_index], dtype=torch.long, device=device
                        )
                        visible = suffix != int(MASK_TOKEN_ID)
                        visible_wrong = int(
                            (visible & suffix.ne(target_tensor)).sum().detach().cpu()
                        )
                        metrics = _raw_state_metrics(
                            raw_logits[row_index],
                            positions=active_absolute,
                            targets=target_ids,
                        )
                        selected = selected_by_row[row_index]
                        selected_relative = [value - prompt_width for value in selected]
                        selected_tokens = [int(x0[row_index, value]) for value in selected]
                        commit_correct = sum(
                            token == int(ground_truth[row_index][relative])
                            for relative, token in zip(
                                selected_relative, selected_tokens, strict=True
                            )
                        )
                        remaining = int(
                            (suffix == int(MASK_TOKEN_ID)).sum().detach().cpu()
                        )
                        record: dict[str, Any] = {
                            "schema": "evidence_first_dlm_state_v1",
                            "panel_type": "safe_axis_actual_b0",
                            "validation_ordinal": int(task["ordinal"]),
                            "training_pair_sha256": str(
                                task["row"]["training_pair_sha256"]
                            ),
                            "num_atoms": num_atoms,
                            "active_group_index": group_index,
                            "active_group": groups[group_index].name,
                            "step_in_group": step_in_group,
                            "active_group_width": len(group_positions),
                            "active_masked_count": len(active_absolute),
                            "remaining_mask_count": remaining,
                            "remaining_mask_fraction": remaining / gen_length,
                            "input_ids": [int(value) for value in x[row_index].tolist()],
                            "attention_mask": [
                                int(value)
                                for value in full_attention_mask[row_index].tolist()
                            ],
                            "target_positions": active_absolute,
                            "target_token_ids": target_ids,
                            "visible_wrong_commitments": visible_wrong,
                            "commit_count": len(selected),
                            "commit_positions": selected_relative,
                            "commit_token_ids": selected_tokens,
                            "commit_correct_count": int(commit_correct),
                            "commit_confidences": [
                                float(confidence[row_index, value].detach().cpu())
                                for value in selected
                            ],
                            "producer_mean_nll": metrics["mean_nll"],
                            "producer_correct_count": metrics["correct_count"],
                            "producer_confidence_sum": metrics["confidence_sum"],
                            "producer_brier_sum": metrics["brier_sum"],
                        }
                        record["state_id"] = state_identity(record)
                        states.append(record)
                    x[transfer_index] = x0[transfer_index]

            if bool((x[:, prompt_width:] == int(MASK_TOKEN_ID)).any()):
                raise RuntimeError("actual safe-axis rollout left masked tokens")
            for row_index, task in enumerate(batch):
                generated = [int(value) for value in x[row_index, prompt_width:].tolist()]
                truth = [int(value) for value in ground_truth[row_index]]
                attempts.append(
                    {
                        "schema": "evidence_first_dlm_actual_rollout_attempt_v1",
                        "validation_ordinal": int(task["ordinal"]),
                        "training_pair_sha256": str(
                            task["row"]["training_pair_sha256"]
                        ),
                        "num_atoms": num_atoms,
                        "noise_seed": int(task["noise_seed"]),
                        "schedule_sha256": str(task["schedule_sha256"]),
                        "generated_token_ids": generated,
                        "ground_truth_token_ids": truth,
                        "exact_token_match": generated == truth,
                        "token_error_count": sum(
                            left != right
                            for left, right in zip(generated, truth, strict=True)
                        ),
                    }
                )
            del x, full_attention_mask, prompt_ids, attention_mask
    states.sort(
        key=lambda row: (
            int(row["validation_ordinal"]),
            int(row["active_group_index"]),
            int(row["step_in_group"]),
        )
    )
    attempts.sort(key=lambda row: int(row["validation_ordinal"]))
    if len({row["state_id"] for row in states}) != len(states):
        raise ValueError("actual rollout state identities are not unique")
    if [row["validation_ordinal"] for row in attempts] != list(range(len(rows))):
        raise ValueError("actual rollout attempts lost or duplicated an ordinal")
    return states, attempts


def score_states(
    states: Sequence[Mapping[str, Any]],
    *,
    model,
    batch_size: int,
) -> list[dict[str, Any]]:
    device = next(model.parameters()).device
    by_length: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for state in states:
        by_length[len(state["input_ids"])].append(state)
    scored: list[dict[str, Any]] = []
    for length in sorted(by_length):
        bucket = by_length[length]
        for offset in range(0, len(bucket), int(batch_size)):
            batch = bucket[offset : offset + int(batch_size)]
            input_ids = torch.tensor(
                [row["input_ids"] for row in batch],
                dtype=torch.long,
                device=device,
            )
            attention_mask = torch.tensor(
                [row["attention_mask"] for row in batch],
                dtype=torch.long,
                device=device,
            )
            with torch.inference_mode():
                logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                ).logits
            for row_index, state in enumerate(batch):
                metrics = _raw_state_metrics(
                    logits[row_index],
                    positions=state["target_positions"],
                    targets=state["target_token_ids"],
                )
                bins = [
                    {"count": 0, "confidence_sum": 0.0, "correct_count": 0}
                    for _ in range(10)
                ]
                for confidence, correct in metrics.pop("calibration"):
                    bin_index = min(9, int(float(confidence) * 10.0))
                    bins[bin_index]["count"] += 1
                    bins[bin_index]["confidence_sum"] += float(confidence)
                    bins[bin_index]["correct_count"] += int(bool(correct))
                score = {
                    "schema": "evidence_first_dlm_state_score_v1",
                    "state_id": state["state_id"],
                    "panel_type": state["panel_type"],
                    "validation_ordinal": int(state["validation_ordinal"]),
                    "num_atoms": int(state["num_atoms"]),
                    "active_group_index": state.get("active_group_index"),
                    "active_group": state.get("active_group"),
                    "step_in_group": state.get("step_in_group"),
                    "active_group_width": int(state["active_group_width"]),
                    "remaining_mask_fraction": float(
                        state["remaining_mask_fraction"]
                    ),
                    "visible_wrong_commitments": int(
                        state.get("visible_wrong_commitments", 0)
                    ),
                    "commit_count": state.get("commit_count"),
                    "commit_correct_count": state.get("commit_correct_count"),
                    "calibration_bins": bins,
                    **metrics,
                }
                if state["panel_type"] == "safe_axis_actual_b0":
                    delta = abs(
                        float(score["mean_nll"])
                        - float(state["producer_mean_nll"])
                    )
                    score["producer_rescore_abs_delta"] = delta
                    if delta > 5e-4:
                        raise ValueError(
                            f"actual rollout producer/rescore NLL changed by {delta}"
                        )
                scored.append(score)
            del logits, input_ids, attention_mask
    scored.sort(key=lambda row: str(row["state_id"]))
    if {row["state_id"] for row in scored} != {row["state_id"] for row in states}:
        raise ValueError("state scoring lost or substituted an identity")
    return scored


def aggregate_scores(
    scored: Sequence[Mapping[str, Any]],
    *,
    states_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    panels = sorted({str(row["panel_type"]) for row in scored})
    for panel in panels:
        rows = [row for row in scored if row["panel_type"] == panel]
        token_count = sum(int(row["target_count"]) for row in rows)
        nll_sum = sum(float(row["nll_sum"]) for row in rows)
        correct_count = sum(int(row["correct_count"]) for row in rows)
        confidence_sum = sum(float(row["confidence_sum"]) for row in rows)
        brier_sum = sum(float(row["brier_sum"]) for row in rows)
        bins = [
            {"count": 0, "confidence_sum": 0.0, "correct_count": 0}
            for _ in range(10)
        ]
        for row in rows:
            for index, source in enumerate(row["calibration_bins"]):
                bins[index]["count"] += int(source["count"])
                bins[index]["confidence_sum"] += float(source["confidence_sum"])
                bins[index]["correct_count"] += int(source["correct_count"])
        ece = 0.0
        for item in bins:
            if item["count"] <= 0:
                continue
            average_confidence = item["confidence_sum"] / item["count"]
            accuracy = item["correct_count"] / item["count"]
            ece += item["count"] / token_count * abs(average_confidence - accuracy)
        panel_states = [states_by_id[str(row["state_id"])] for row in rows]
        widths = [float(row["active_group_width"]) for row in panel_states]
        mask_fractions = [float(row["remaining_mask_fraction"]) for row in panel_states]
        commit_count = sum(
            int(row.get("commit_count") or 0) for row in panel_states
        )
        commit_correct = sum(
            int(row.get("commit_correct_count") or 0) for row in panel_states
        )
        result[panel] = {
            "states": len(rows),
            "validation_rows": len(
                {int(row["validation_ordinal"]) for row in rows}
            ),
            "target_tokens": token_count,
            "token_weighted_mean_nll": nll_sum / token_count,
            "state_weighted_mean_nll": sum(
                float(row["mean_nll"]) for row in rows
            )
            / len(rows),
            "top1_accuracy": correct_count / token_count,
            "mean_top1_confidence": confidence_sum / token_count,
            "brier": brier_sum / token_count,
            "ece10": ece,
            "active_group_width": {
                "mean": sum(widths) / len(widths),
                "p50": quantile(widths, 0.5),
                "p95": quantile(widths, 0.95),
            },
            "remaining_mask_fraction": {
                "mean": sum(mask_fractions) / len(mask_fractions),
                "p50": quantile(mask_fractions, 0.5),
                "p95": quantile(mask_fractions, 0.95),
            },
            "num_atoms_distribution": dict(
                sorted(Counter(int(row["num_atoms"]) for row in panel_states).items())
            ),
            "active_group_distribution": dict(
                sorted(Counter(str(row.get("active_group")) for row in panel_states).items())
            ),
            "visible_wrong_commitments_sum": sum(
                int(row.get("visible_wrong_commitments") or 0)
                for row in panel_states
            ),
            "commit_count": commit_count,
            "commit_correct_count": commit_correct,
            "commit_accuracy": (
                commit_correct / commit_count if commit_count else None
            ),
            "calibration_bins": bins,
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    started = time.monotonic()
    device = validate_runtime()
    config_path = args.config.resolve()
    config = read_json(config_path)
    if (
        config.get("schema") != "evidence_first_dlm_state_panels_v1"
        or config.get("training") is not False
        or config.get("sun") is not False
        or config.get("automatic_b3_submission") is not False
        or config.get("retry_replacement_repair_filter_rerank") is not False
        or config["checkpoint"].get("arm") != "B0"
    ):
        raise ValueError("state-panel configuration changed")
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    data_root = Path(config["data"]["root"]).resolve()
    require_sha(
        data_root / "manifest.json",
        config["data"]["manifest_sha256"],
        "R5-C sidecar manifest",
    )
    validation_path = require_sha(
        data_root / "val.jsonl",
        config["data"]["validation_jsonl_sha256"],
        "R5-C validation JSONL",
    )
    all_rows = read_jsonl(validation_path)
    if len(all_rows) != int(config["data"]["validation_rows"]):
        raise ValueError("R5-C validation row count changed")
    synthetic_rows = all_rows[:100]
    actual_rows = all_rows[:64]
    if len(synthetic_rows) != 100 or len(actual_rows) != 64:
        raise ValueError("frozen panel ordinals are unavailable")

    checkpoint = Path(config["checkpoint"]["path"]).resolve()
    adapter = checkpoint / str(config["checkpoint"]["adapter_file"])
    if (
        not adapter.is_file()
        or adapter.stat().st_size != int(config["checkpoint"]["adapter_bytes"])
    ):
        raise ValueError("protected B0 adapter path or byte size changed")
    require_sha(
        checkpoint / "tokenizer.json",
        config["checkpoint"]["tokenizer_json_sha256"],
        "B0 tokenizer.json",
    )
    require_sha(
        checkpoint / "tokenizer_config.json",
        config["checkpoint"]["tokenizer_config_sha256"],
        "B0 tokenizer_config.json",
    )
    model, tokenizer = load_model_and_tokenizer(
        str(Path(config["base_model"]).resolve()), str(checkpoint), device
    )
    tokenizer_identity = assert_body_tokenizer_identity(
        tokenizer,
        expected_vocab_sha256=config["checkpoint"]["tokenizer_vocab_sha256"],
    )
    if int(tokenizer_identity["vocab_size"]) != int(
        config["checkpoint"]["tokenizer_size"]
    ):
        raise ValueError("B0 tokenizer size changed")
    write_json_exclusive(output / "tokenizer_identity.json", tokenizer_identity)

    panel_seed = int(config["panels"]["seed"])
    synthetic_states = build_synthetic_states(
        synthetic_rows, tokenizer=tokenizer, panel_seed=panel_seed
    )
    actual_states, actual_attempts = build_actual_rollout_states(
        actual_rows,
        tokenizer=tokenizer,
        model=model,
        decoder=config["decoder"],
        panel_seed=panel_seed,
    )
    synthetic_path = output / "synthetic_states.jsonl"
    actual_path = output / "actual_rollout_states.jsonl"
    attempts_path = output / "actual_rollout_attempts.jsonl"
    write_jsonl_exclusive(synthetic_path, synthetic_states)
    write_jsonl_exclusive(actual_path, actual_states)
    write_jsonl_exclusive(attempts_path, actual_attempts)

    all_states = [*synthetic_states, *actual_states]
    scored = score_states(
        all_states,
        model=model,
        batch_size=int(config["decoder"]["score_batch_size"]),
    )
    score_path = output / "B0_state_scores.jsonl"
    write_jsonl_exclusive(score_path, scored)
    states_by_id = {str(row["state_id"]): row for row in all_states}
    summary = aggregate_scores(scored, states_by_id=states_by_id)

    required_panels = set(config["panels"]["panel_types"])
    if set(summary) != required_panels:
        raise ValueError(f"state panel coverage changed: {set(summary)}")
    if summary["iid"]["states"] != 100 or summary["d1"]["states"] != 600:
        raise ValueError("fixed IID/D1 panel sizes changed")
    if summary["safe_axis_synthetic"]["validation_rows"] != 100:
        raise ValueError("safe-axis synthetic panel lost validation rows")
    if summary["safe_axis_actual_b0"]["validation_rows"] != 64:
        raise ValueError("actual rollout panel lost validation rows")
    if any(not math.isfinite(value["token_weighted_mean_nll"]) for value in summary.values()):
        raise ValueError("state panel NLL is non-finite")

    manifest = {
        "schema": "evidence_first_dlm_frozen_state_panel_manifest_v1",
        "status": "complete",
        "identity": config["identity"],
        "config_sha256": sha256_file(config_path),
        "validation_jsonl_sha256": sha256_file(validation_path),
        "synthetic_validation_ordinals": list(range(100)),
        "actual_rollout_validation_ordinals": list(range(64)),
        "panel_seed": panel_seed,
        "checkpoint_arm": "B0",
        "checkpoint_adapter_sha256_recorded": config["checkpoint"][
            "adapter_sha256"
        ],
        "actual_rollout_target": config["panels"]["actual_rollout_target"],
        "actual_wrong_visible_commitments": config["panels"][
            "actual_wrong_visible_commitments"
        ],
        "state_counts": dict(
            sorted(Counter(row["panel_type"] for row in all_states).items())
        ),
        "state_identity_sha256": canonical_sha256(
            sorted(str(row["state_id"]) for row in all_states)
        ),
        "files": {
            "synthetic_states.jsonl": sha256_file(synthetic_path),
            "actual_rollout_states.jsonl": sha256_file(actual_path),
            "actual_rollout_attempts.jsonl": sha256_file(attempts_path),
            "B0_state_scores.jsonl": sha256_file(score_path),
        },
        "retry_replacement_repair_filter_rerank": False,
        "training": False,
        "sun": False,
        "automatic_b3_submission": False,
    }
    manifest_path = output / "state_panel_manifest.json"
    write_json_exclusive(manifest_path, manifest)
    report = {
        "schema": "evidence_first_dlm_state_panel_terminal_v1",
        "status": "complete",
        "identity": config["identity"],
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "checkpoint_arm": "B0",
        "checkpoint_path": str(checkpoint),
        "checkpoint_adapter_sha256_recorded": config["checkpoint"][
            "adapter_sha256"
        ],
        "state_panel_manifest_sha256": sha256_file(manifest_path),
        "summary": summary,
        "actual_rollout_attempts": {
            "attempts": len(actual_attempts),
            "exact_token_matches": sum(
                row["exact_token_match"] is True for row in actual_attempts
            ),
            "token_errors": sum(
                int(row["token_error_count"]) for row in actual_attempts
            ),
        },
        "cuda_peak_memory_bytes": int(torch.cuda.max_memory_reserved(0)),
        "walltime_s": time.monotonic() - started,
        "training": False,
        "generation_science_endpoint": False,
        "sun": False,
        "automatic_b3_submission": False,
    }
    terminal_path = output / "terminal_report.json"
    write_json_exclusive(terminal_path, report)
    (output / "_SUCCESS").touch(exist_ok=False)
    print(json.dumps(report, indent=2, sort_keys=True))
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
