#!/usr/bin/env python3
"""Assemble the complete current-run Planner x Body four-cell matrix."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

from protocol import (
    CELLS,
    DENOMINATOR,
    read_json,
    read_jsonl,
    require_source_manifest,
    sha256_file,
    validate_config,
    write_json_exclusive,
)


ENDPOINTS = (
    "generation_complete",
    "composition_valid",
    "structure_valid",
    "joint_valid",
    "novel",
    "unique_representative",
    "novel_unique",
    "strict_full_sun",
    "meta_full_sun",
)


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * float(probability)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _mcnemar(candidate: Sequence[bool], baseline: Sequence[bool]) -> dict[str, Any]:
    candidate_only = sum(
        bool(left) and not bool(right)
        for left, right in zip(candidate, baseline, strict=True)
    )
    baseline_only = sum(
        not bool(left) and bool(right)
        for left, right in zip(candidate, baseline, strict=True)
    )
    discordant = candidate_only + baseline_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(
            math.comb(discordant, value)
            for value in range(min(candidate_only, baseline_only) + 1)
        ) / (2**discordant)
        p_value = min(1.0, 2.0 * tail)
    return {
        "candidate_only": candidate_only,
        "baseline_only": baseline_only,
        "discordant": discordant,
        "two_sided_exact_p_value": p_value,
    }


def _paired_effect(
    candidate: Sequence[bool],
    baseline: Sequence[bool],
    *,
    candidate_cell: str,
    baseline_cell: str,
    seed: int,
    draws: int,
) -> dict[str, Any]:
    if len(candidate) != DENOMINATOR or len(baseline) != DENOMINATOR:
        raise ValueError("paired effect lost the all-attempt denominator")
    differences = [
        float(bool(left)) - float(bool(right))
        for left, right in zip(candidate, baseline, strict=True)
    ]
    rng = random.Random(seed)
    samples = [
        100.0
        * sum(differences[rng.randrange(DENOMINATOR)] for _ in range(DENOMINATOR))
        / DENOMINATOR
        for _ in range(draws)
    ]
    candidate_count = sum(bool(value) for value in candidate)
    baseline_count = sum(bool(value) for value in baseline)
    return {
        "candidate_cell": candidate_cell,
        "baseline_cell": baseline_cell,
        "attempts": DENOMINATOR,
        "candidate_count": candidate_count,
        "baseline_count": baseline_count,
        "candidate_rate": candidate_count / DENOMINATOR,
        "baseline_rate": baseline_count / DENOMINATOR,
        "difference_count": candidate_count - baseline_count,
        "difference_percentage_points": (
            100.0 * (candidate_count - baseline_count) / DENOMINATOR
        ),
        "paired_bootstrap": {
            "draws": draws,
            "seed": seed,
            "ci95_lower_percentage_points": _quantile(samples, 0.025),
            "ci95_upper_percentage_points": _quantile(samples, 0.975),
        },
        "exact_mcnemar": _mcnemar(candidate, baseline),
    }


def _interaction(
    vectors: Mapping[str, Sequence[bool]], *, seed: int, draws: int
) -> dict[str, Any]:
    per_ordinal = [
        100.0
        * (
            float(vectors["M11"][index])
            - float(vectors["M10"][index])
            - float(vectors["M01"][index])
            + float(vectors["M00"][index])
        )
        for index in range(DENOMINATOR)
    ]
    rng = random.Random(seed)
    samples = [
        sum(per_ordinal[rng.randrange(DENOMINATOR)] for _ in range(DENOMINATOR))
        / DENOMINATOR
        for _ in range(draws)
    ]
    return {
        "definition": "M11-M10-M01+M00",
        "interaction_percentage_points": sum(per_ordinal) / DENOMINATOR,
        "paired_bootstrap": {
            "draws": draws,
            "seed": seed,
            "ci95_lower_percentage_points": _quantile(samples, 0.025),
            "ci95_upper_percentage_points": _quantile(samples, 0.975),
        },
    }


def _cell_evidence(
    run_root: Path, cell: str, config: Mapping[str, Any]
) -> dict[str, Any]:
    root = run_root / "cells" / cell
    if not (root / "_SUCCESS").is_file() or not (
        root / "evaluation/_SUCCESS"
    ).is_file():
        raise FileNotFoundError(f"{cell} did not reach its complete marker")
    evaluation = read_json(root / "evaluation/evaluation_report.json")
    generation = read_jsonl(root / "generation/generation.jsonl")
    direct = read_jsonl(root / "evaluation/crysllmgen_metrics/attempt_metrics.jsonl")
    direct_report = read_json(root / "evaluation/crysllmgen_metrics/report.json")
    sun = read_jsonl(root / "evaluation/r5c_a100_sun/attempt_results.jsonl")
    sun_summary = read_json(root / "evaluation/r5c_a100_sun/attempt_summary.json")
    expected_ids = [f"h1-ef-fourcell-{cell.lower()}-{index:04d}" for index in range(DENOMINATOR)]
    if (
        evaluation.get("ok") is not True
        or evaluation.get("cell") != cell
        or len(generation) != DENOMINATOR
        or len(direct) != DENOMINATOR
        or len(sun) != DENOMINATOR
        or [str(row.get("attempt_id")) for row in generation] != expected_ids
        or [str(row.get("attempt_id")) for row in direct] != expected_ids
        or [str(row.get("attempt_id")) for row in sun] != expected_ids
        or any(int(row.get("ordinal", -1)) != index for index, row in enumerate(generation))
        or {str(row.get("pair_id")) for row in generation}
        != {f"h1-ef-fourcell:{index:04d}" for index in range(DENOMINATOR)}
    ):
        raise ValueError(f"{cell} current-run all-attempt evidence changed")
    vectors = {
        "generation_complete": [row.get("status") == "succeeded" for row in generation],
        "composition_valid": [bool(row.get("comp_valid")) for row in direct],
        "structure_valid": [bool(row.get("struct_valid")) for row in direct],
        "joint_valid": [bool(row.get("valid")) for row in direct],
        "novel": [bool((row.get("metrics") or {}).get("novel")) for row in sun],
        "unique_representative": [
            bool((row.get("metrics") or {}).get("unique_representative"))
            for row in sun
        ],
        "novel_unique": [
            bool((row.get("metrics") or {}).get("novel_unique")) for row in sun
        ],
        "strict_full_sun": [
            bool((row.get("metrics") or {}).get("strict_full_sun")) for row in sun
        ],
        "meta_full_sun": [
            bool((row.get("metrics") or {}).get("meta_full_sun")) for row in sun
        ],
    }
    if tuple(vectors) != ENDPOINTS:
        raise ValueError("endpoint vector order changed")
    specification = config["cells"][cell]
    return {
        "vectors": vectors,
        "summary": {
            "planner": specification["planner"],
            "body": specification["body"],
            "role": specification["role"],
            "method": specification["method"],
            "attempts": DENOMINATOR,
            "counts": {key: sum(values) for key, values in vectors.items()},
            "rates": {
                key: sum(values) / DENOMINATOR for key, values in vectors.items()
            },
            "cov_precision_recall": direct_report["metrics_unchanged_upstream"],
            "sun_exact_legacy": sun_summary["exact_legacy_r5c_a100"],
            "sun_hull": evaluation["sun_hull"],
            "artifacts": evaluation["artifacts"],
        },
    }


def _assemble(args: argparse.Namespace) -> dict[str, Any]:
    source = args.source_dir.resolve()
    require_source_manifest(source, args.source_manifest_sha256)
    config = read_json(args.config.resolve())
    validate_config(config)
    run_root = args.run_root.resolve()
    evidence = {
        cell: _cell_evidence(run_root, cell, config) for cell in CELLS
    }
    seed = int(config["analysis"]["bootstrap_seed"])
    draws = int(config["analysis"]["bootstrap_draws"])
    effects: dict[str, Any] = {}
    leaders: dict[str, list[str]] = {}
    for endpoint_index, endpoint in enumerate(ENDPOINTS):
        vectors = {
            cell: evidence[cell]["vectors"][endpoint] for cell in CELLS
        }
        counts = {cell: sum(vectors[cell]) for cell in CELLS}
        maximum = max(counts.values())
        leaders[endpoint] = [cell for cell in CELLS if counts[cell] == maximum]
        offset = 10 * endpoint_index
        effects[endpoint] = {
            "planner_at_B0": _paired_effect(
                vectors["M10"], vectors["M00"],
                candidate_cell="M10", baseline_cell="M00",
                seed=seed + offset, draws=draws,
            ),
            "planner_at_B3": _paired_effect(
                vectors["M11"], vectors["M01"],
                candidate_cell="M11", baseline_cell="M01",
                seed=seed + offset + 1, draws=draws,
            ),
            "body_at_P0": _paired_effect(
                vectors["M01"], vectors["M00"],
                candidate_cell="M01", baseline_cell="M00",
                seed=seed + offset + 2, draws=draws,
            ),
            "body_at_SFT_v2": _paired_effect(
                vectors["M11"], vectors["M10"],
                candidate_cell="M11", baseline_cell="M10",
                seed=seed + offset + 3, draws=draws,
            ),
            "joint_M11_vs_M00": _paired_effect(
                vectors["M11"], vectors["M00"],
                candidate_cell="M11", baseline_cell="M00",
                seed=seed + offset + 4, draws=draws,
            ),
            "factorial_interaction": _interaction(
                vectors, seed=seed + offset + 5, draws=draws
            ),
        }
    return {
        "schema": "h1_ef_fourcell_direct_sun_terminal_report_v1",
        "status": "complete",
        "decision": "complete_evidence_report_no_automatic_promotion",
        "run_id": config["run_id"],
        "run_root": str(run_root),
        "cells": {cell: evidence[cell]["summary"] for cell in CELLS},
        "paired_effects": effects,
        "descriptive_endpoint_leaders": leaders,
        "all_attempt_denominator_per_cell": DENOMINATOR,
        "same_current_run_pipeline_all_cells": True,
        "historical_summary_substituted": False,
        "same_ordinal_body_and_refiner_seeds": True,
        "safe_axis_all_cells": True,
        "refiner_model": config["refiner"]["name"],
        "diffusion_steps": 800,
        "raw_failures_in_denominator": True,
        "retry_replacement_repair_filter_rerank": False,
        "protected_incumbent": "M00",
        "formal_promotion": False,
        "automatic_checkpoint_reselection": False,
        "automatic_training": False,
        "automatic_downstream": False,
        "automatic_rl": False,
        "source_manifest_sha256": args.source_manifest_sha256,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    try:
        report = _assemble(args)
    except Exception as exc:
        failure = {
            "schema": "h1_ef_fourcell_direct_sun_terminal_report_v1",
            "status": "failed",
            "decision": "engineering_failure_fail_closed",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
            "formal_promotion": False,
            "automatic_downstream": False,
            "automatic_rl": False,
        }
        write_json_exclusive(output, failure)
        print(json.dumps(failure, sort_keys=True))
        raise
    write_json_exclusive(output, report)
    with (output.parent / "_SUCCESS").open("x", encoding="ascii") as handle:
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
