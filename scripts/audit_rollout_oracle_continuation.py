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

from crystal_dlm.dynamic_crystal import arrays_to_structure, parse_dynamic_answer


STAGES = ("lattice", "x", "y", "z")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


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
        target = str(row["answer"])
        if sample_idx in target_by_sample and target_by_sample[sample_idx] != target:
            raise ValueError("MP20 target changed across rollout stages")
        target_by_sample[sample_idx] = target
        metric = evaluate_answer(
            str(row["source_answer"]),
            smact_validity=smact_validity,
            structure_validity=structure_validity,
        )
        if sample_idx in by_stage[stage]:
            raise ValueError("duplicate sample/stage transition")
        by_stage[stage][sample_idx] = metric
        committed_errors[stage].append(int(row["committed_error_count"]))
        row_output.append(
            {
                "sample_idx": sample_idx,
                "source_row_idx": int(row["source_row_idx"]),
                "active_group": stage,
                "committed_error_count": int(row["committed_error_count"]),
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
