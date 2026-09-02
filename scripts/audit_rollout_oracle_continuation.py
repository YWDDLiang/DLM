#!/usr/bin/env python3
"""Audit whether MP20 continuation can recover real exact-axis rollout states.

Each captured ``source_answer`` contains the model's already committed tokens
and the original MP20 tokens only at positions that were still masked.  It is
therefore the teacher-forced continuation limit of the proposed active-group
CE objective, not a generated teacher.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Callable, Iterable

from crystal_dlm.canonical_site_order import canonicalize_dynamic_answer_to_plan
from crystal_dlm.dynamic_crystal import arrays_to_structure, parse_dynamic_answer
from crystal_dlm.fixed_slot import tokenize_answer_text


STAGES = ("lattice", "x", "y", "z")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def replace_masked_with_target(
    source_answer: str,
    target_answer: str,
    forced_mask_positions: Iterable[int],
) -> str:
    source = tokenize_answer_text(source_answer)
    target = tokenize_answer_text(target_answer)
    if len(source) != len(target):
        raise ValueError("source/target token lengths differ")
    for position in forced_mask_positions:
        source[int(position)] = target[int(position)]
    return "".join(source)


def load_direct_functions(snapshot_root: Path) -> tuple[Callable[..., Any], Callable[..., Any]]:
    sys.path.insert(0, str(snapshot_root.resolve()))
    try:
        from eval_utils import smact_validity, structure_validity
    finally:
        sys.path.pop(0)
    return smact_validity, structure_validity


def evaluate_answer(
    answer: str,
    *,
    smact_validity: Callable[..., Any],
    structure_validity: Callable[..., Any],
) -> dict[str, Any]:
    result = {
        "parsed": False,
        "composition_valid": False,
        "structure_valid": False,
        "direct": False,
        "reason": "",
    }
    try:
        arrays = parse_dynamic_answer(answer, strict=True)
        structure = arrays_to_structure(arrays)
        counts = Counter(int(value) for value in structure.atomic_numbers)
        elements = tuple(sorted(counts))
        amounts = [counts[element] for element in elements]
        divisor = math.gcd(*amounts)
        reduced = tuple(int(value // divisor) for value in amounts)
        result["parsed"] = True
        result["composition_valid"] = bool(smact_validity(elements, reduced))
        result["structure_valid"] = bool(structure_validity(structure))
        result["direct"] = bool(
            result["composition_valid"] and result["structure_valid"]
        )
        if not result["direct"]:
            result["reason"] = "crysllmgen_invalid"
    except Exception as exc:  # preserve every failed hybrid
        result["reason"] = f"{type(exc).__name__}:{exc}"
    return result


def counts(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    values = list(rows)
    return {
        "rows": len(values),
        "parsed": sum(bool(row["parsed"]) for row in values),
        "composition_valid": sum(bool(row["composition_valid"]) for row in values),
        "structure_valid": sum(bool(row["structure_valid"]) for row in values),
        "direct": sum(bool(row["direct"]) for row in values),
    }


def paired_summary(
    base: dict[int, dict[str, Any]],
    candidate: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    if set(base) != set(candidate):
        raise ValueError("base/candidate sample sets differ")
    invalid_to_valid = sum(
        not bool(base[index]["direct"]) and bool(candidate[index]["direct"])
        for index in base
    )
    valid_to_invalid = sum(
        bool(base[index]["direct"]) and not bool(candidate[index]["direct"])
        for index in base
    )
    return {
        "base": counts(base.values()),
        "candidate": counts(candidate.values()),
        "invalid_to_valid": invalid_to_valid,
        "valid_to_invalid": valid_to_invalid,
        "net_invalid_to_valid": invalid_to_valid - valid_to_invalid,
        "direct_delta": sum(bool(row["direct"]) for row in candidate.values())
        - sum(bool(row["direct"]) for row in base.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transitions", type=Path, required=True)
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--plans-jsonl", type=Path)
    parser.add_argument("--canonicalize-target-to-plan-order", action="store_true")
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    transitions = read_jsonl(args.transitions)
    trajectories = read_jsonl(args.trajectories)
    if len(transitions) != 512 or len(trajectories) != 128:
        raise ValueError("expected 512 transitions and 128 final trajectories")
    stage_counts = Counter(str(row["active_group"]) for row in transitions)
    if stage_counts != Counter({stage: 128 for stage in STAGES}):
        raise ValueError(f"stage accounting changed: {stage_counts}")
    if args.canonicalize_target_to_plan_order and args.plans_jsonl is None:
        raise ValueError("canonical target audit requires --plans-jsonl")
    plans = None
    if args.plans_jsonl is not None:
        plan_rows = read_jsonl(args.plans_jsonl)
        if len(plan_rows) != 128:
            raise ValueError("plans JSONL must contain 128 rows in sample order")
        plans = {index: dict(row["plan_state"]) for index, row in enumerate(plan_rows)}

    smact_validity, structure_validity = load_direct_functions(args.snapshot_root)
    base = {
        int(row["sample_idx"]): evaluate_answer(
            str(row["answer"]),
            smact_validity=smact_validity,
            structure_validity=structure_validity,
        )
        for row in trajectories
    }
    if set(base) != set(range(128)):
        raise ValueError("final trajectories do not cover sample_idx 0..127")

    by_stage: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    target_by_sample: dict[int, str] = {}
    committed_errors: dict[str, list[int]] = defaultdict(list)
    row_output: list[dict[str, Any]] = []
    for row in transitions:
        stage = str(row["active_group"])
        sample_idx = int(row["sample_idx"])
        original_target = str(row["answer"])
        target = (
            canonicalize_dynamic_answer_to_plan(
                original_target, plans[sample_idx]
            )[0]
            if args.canonicalize_target_to_plan_order and plans is not None
            else original_target
        )
        if sample_idx in target_by_sample and target_by_sample[sample_idx] != target:
            raise ValueError("MP20 target changed across rollout stages")
        target_by_sample[sample_idx] = target
        source_answer = (
            replace_masked_with_target(
                str(row["source_answer"]),
                target,
                row["forced_mask_positions"],
            )
            if args.canonicalize_target_to_plan_order
            else str(row["source_answer"])
        )
        metric = evaluate_answer(
            source_answer,
            smact_validity=smact_validity,
            structure_validity=structure_validity,
        )
        if sample_idx in by_stage[stage]:
            raise ValueError("duplicate sample/stage transition")
        by_stage[stage][sample_idx] = metric
        source_tokens = tokenize_answer_text(source_answer)
        target_tokens = tokenize_answer_text(target)
        forced = {int(value) for value in row["forced_mask_positions"]}
        current_committed_errors = sum(
            source_token != target_token and position not in forced
            for position, (source_token, target_token) in enumerate(
                zip(source_tokens, target_tokens, strict=True)
            )
        )
        committed_errors[stage].append(current_committed_errors)
        row_output.append(
            {
                "sample_idx": sample_idx,
                "source_row_idx": int(row["source_row_idx"]),
                "active_group": stage,
                "committed_error_count": current_committed_errors,
                **metric,
            }
        )

    target = {
        sample_idx: evaluate_answer(
            answer,
            smact_validity=smact_validity,
            structure_validity=structure_validity,
        )
        for sample_idx, answer in target_by_sample.items()
    }
    stage_summaries = {
        stage: {
            **paired_summary(base, by_stage[stage]),
            "committed_errors": {
                "mean": statistics.fmean(committed_errors[stage]),
                "median": statistics.median(committed_errors[stage]),
                "positive": sum(value > 0 for value in committed_errors[stage]),
                "max": max(committed_errors[stage]),
            },
        }
        for stage in STAGES
    }
    mean_stage_net = statistics.fmean(
        float(stage_summaries[stage]["net_invalid_to_valid"]) for stage in STAGES
    )
    report = {
        "schema": "rollout_oracle_continuation_audit_v1",
        "status": "complete",
        "interpretation": (
            "source_answer keeps committed model tokens and fills only still-masked "
            "positions with the original MP20 target"
        ),
        "teacher": "original_MP20_only",
        "generated_structure_used_as_teacher": False,
        "target_site_order": (
            "plan_expanded_inference_order"
            if args.canonicalize_target_to_plan_order
            else "original_MP20_site_order"
        ),
        "base_final": counts(base.values()),
        "mp20_target": counts(target.values()),
        "stages": stage_summaries,
        "mean_stage_net_invalid_to_valid": mean_stage_net,
        "preregistered_training_necessary_conditions": {
            "all_stage_net_nonnegative": all(
                stage_summaries[stage]["net_invalid_to_valid"] >= 0
                for stage in STAGES
            ),
            "z_stage_net_at_least_12": stage_summaries["z"][
                "net_invalid_to_valid"
            ]
            >= 12,
            "each_stage_valid_to_invalid_at_most_4": all(
                stage_summaries[stage]["valid_to_invalid"] <= 4
                for stage in STAGES
            ),
            "mean_stage_net_at_least_12": mean_stage_net >= 12.0,
        },
    }
    conditions = report["preregistered_training_necessary_conditions"]
    report["supports_active_group_ce_training"] = bool(all(conditions.values()))

    args.output_dir.mkdir(parents=True)
    (args.output_dir / "attempts.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in row_output)
    )
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "_SUCCESS").touch()
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
