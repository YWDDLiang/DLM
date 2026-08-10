#!/usr/bin/env python3
"""Validate and preserve complete Direct/S.U.N. detail for one stage."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from protocol import (
    DENOMINATOR,
    ordered_rows,
    read_json,
    read_jsonl,
    require_source_manifest,
    sha256_file,
    validate_arm,
    validate_config,
    validate_repeat,
    write_json_exclusive,
)


def identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def finite_summary(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"count": 0, "min": None, "q25": None, "median": None, "q75": None, "max": None, "mean": None}
    q25, median, q75 = np.quantile(array, [0.25, 0.5, 0.75])
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "q25": float(q25),
        "median": float(median),
        "q75": float(q75),
        "max": float(array.max()),
        "mean": float(array.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True)
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument("--stage", choices=("pre_model494", "post_model494"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--generation-dir", type=Path, required=True)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--completion-manifest", type=Path, required=True)
    args = parser.parse_args()

    arm = validate_arm(args.arm)
    repeat = validate_repeat(args.repeat)
    stage = args.stage
    source = args.source_dir.resolve()
    require_source_manifest(source, args.source_manifest_sha256)
    config = read_json(args.config.resolve())
    validate_config(config)
    generation_dir = args.generation_dir.resolve()
    evaluation_dir = args.evaluation_dir.resolve()
    completion = read_json(args.completion_manifest.resolve())
    cache_spec = completion.get("completed_mp_hull_cache") or {}
    cache_path = Path(str(cache_spec.get("path", ""))).resolve()
    cache_sha = str(cache_spec.get("sha256", ""))
    if (
        completion.get("status") != "complete_all_wanted_chemsys_resolved"
        or completion.get("api_key_serialized") is not False
        or completion.get("mp_query_inside_slurm") is not False
        or cache_spec.get("all_rows_populated") is not True
        or not cache_path.is_file()
        or sha256_file(cache_path) != cache_sha
    ):
        raise ValueError("completed MP cache contract changed")

    generation_report = read_json(generation_dir / "generation_report.json")
    generation = ordered_rows(
        read_jsonl(generation_dir / "generation.jsonl"), ordinal_field="ordinal"
    )
    method = f"P0-{arm}-SAFEAXIS-{stage}"
    expected_refined = stage == "post_model494"
    if (
        not (generation_dir / "_SUCCESS").is_file()
        or generation_report.get("ok") is not True
        or generation_report.get("arm") != arm
        or int(generation_report.get("repeat", -1)) != repeat
        or generation_report.get("stage") != stage
        or int(generation_report.get("attempts", -1)) != DENOMINATOR
        or {str(row.get("method")) for row in generation} != {method}
        or {str(row.get("arm")) for row in generation} != {arm}
        or {int(row.get("repeat", -1)) for row in generation} != {repeat}
        or any(row.get("retry_or_replacement_used") is not False for row in generation)
        or any(
            row.get("status") == "succeeded"
            and bool(row.get("diffusion_refinement_applied")) != expected_refined
            for row in generation
        )
    ):
        raise ValueError("generation stage contract changed")

    direct_dir = evaluation_dir / "crysllmgen_metrics"
    sun_dir = evaluation_dir / "r5c_a100_sun"
    direct_report = read_json(direct_dir / "report.json")
    direct_attempts = read_jsonl(direct_dir / "attempt_metrics.jsonl")
    sun_summary = read_json(sun_dir / "attempt_summary.json")
    sun_attempts = read_jsonl(sun_dir / "attempt_results.jsonl")
    expected_ids = [str(row["attempt_id"]) for row in generation]
    if (
        direct_report.get("ok") is not True
        or int(direct_report.get("attempts", -1)) != DENOMINATOR
        or direct_report.get("denominator") != "all_generation_attempts"
        or direct_report.get("method") != method
        or len(direct_attempts) != DENOMINATOR
        or [str(row.get("attempt_id")) for row in direct_attempts] != expected_ids
        or sun_summary.get("ok") is not True
        or int((sun_summary.get("counts") or {}).get("total_attempts", -1)) != DENOMINATOR
        or sun_summary.get("denominator") != "all_generation_attempts"
        or sun_summary.get("method") != method
        or sun_summary.get("execution_patch_sha256") != args.source_manifest_sha256
        or sun_summary.get("retry_or_replacement_used") is not False
        or len(sun_attempts) != DENOMINATOR
        or [str(row.get("attempt_id")) for row in sun_attempts] != expected_ids
    ):
        raise ValueError("Direct/S.U.N. attempt mapping changed")

    direct_counts = {
        "composition_valid": sum(bool(row.get("comp_valid")) for row in direct_attempts),
        "structure_valid": sum(bool(row.get("struct_valid")) for row in direct_attempts),
        "joint_valid": sum(bool(row.get("valid")) for row in direct_attempts),
    }
    if (
        direct_counts["composition_valid"] != int(direct_report["comp_valid_count"])
        or direct_counts["structure_valid"] != int(direct_report["struct_valid_count"])
        or direct_counts["joint_valid"] != int(direct_report["valid_count"])
    ):
        raise ValueError("Direct summary disagrees with attempts")

    sun_counts = {
        "reconstructed": int(sun_summary["counts"]["reconstructed"]),
        "novel": sum(bool((row.get("metrics") or {}).get("novel")) for row in sun_attempts),
        "unique": sum(bool((row.get("metrics") or {}).get("unique_representative")) for row in sun_attempts),
        "novel_unique": sum(bool((row.get("metrics") or {}).get("novel_unique")) for row in sun_attempts),
        "strict_full_sun": sum(bool((row.get("metrics") or {}).get("strict_full_sun")) for row in sun_attempts),
        "meta_full_sun": sum(bool((row.get("metrics") or {}).get("meta_full_sun")) for row in sun_attempts),
        "hull_evaluated": sum((row.get("metrics") or {}).get("e_above_hull") is not None for row in sun_attempts),
    }
    native_counts = sun_summary["counts"]
    for key in ("novel", "unique", "novel_unique", "strict_full_sun", "meta_full_sun"):
        if sun_counts[key] != int(native_counts[key]):
            raise ValueError(f"S.U.N. {key} summary disagrees with attempts")
    statuses = Counter(str(row.get("evaluation_status")) for row in sun_attempts)
    reasons = Counter(str(row.get("generation_reason") or "") for row in sun_attempts if row.get("generation_status") != "succeeded")
    hull_values = [
        float((row.get("metrics") or {})["e_above_hull"])
        for row in sun_attempts
        if (row.get("metrics") or {}).get("e_above_hull") is not None
    ]
    energy_values = [
        float((row.get("metrics") or {})["energy_per_atom"])
        for row in sun_attempts
        if (row.get("metrics") or {}).get("energy_per_atom") is not None
    ]
    exact_legacy = sun_summary.get("exact_legacy_r5c_a100") or {}
    if exact_legacy.get("denominator") != "reconstructed_structures":
        raise ValueError("exact legacy S.U.N. denominator changed")

    report = {
        "schema": "h1_plan1200_stage_evaluation_report_v1",
        "status": "complete",
        "ok": True,
        "arm": arm,
        "repeat": repeat,
        "stage": stage,
        "method": method,
        "attempts": DENOMINATOR,
        "generation_succeeded": sum(row.get("status") == "succeeded" for row in generation),
        "direct_counts": direct_counts,
        "direct_rates_all_attempts": {key: value / DENOMINATOR for key, value in direct_counts.items()},
        "direct_failure_reasons": dict(sorted(Counter(str(row.get("reason") or "") for row in direct_attempts if not row.get("valid")).items())),
        "direct_native_report_complete": direct_report,
        "sun_counts": sun_counts,
        "sun_rates_all_attempts": {key: value / DENOMINATOR for key, value in sun_counts.items()},
        "sun_exact_legacy_reconstructed_denominator": exact_legacy,
        "sun_native_summary_complete": sun_summary,
        "sun_diagnostics": {
            "evaluation_status_counts": dict(sorted(statuses.items())),
            "generation_failure_reasons": dict(sorted(reasons.items())),
            "hull_evaluated_denominator": sun_counts["hull_evaluated"],
            "hull_unknown": int(native_counts["relaxation_or_hull_unknown"]),
            "e_above_hull_ev_per_atom": finite_summary(hull_values),
            "energy_per_atom_ev": finite_summary(energy_values),
        },
        "denominator_policy": {
            "headline": "reconstructed_structures_exact_legacy",
            "secondary": "all_1000_attempts",
            "evaluated_or_stable": "diagnostic_only",
        },
        "cache": {**cache_spec, "completion_manifest": identity(args.completion_manifest)},
        "artifacts": {
            "generation": identity(generation_dir / "generation.jsonl"),
            "generation_report": identity(generation_dir / "generation_report.json"),
            "direct_report": identity(direct_dir / "report.json"),
            "direct_attempts": identity(direct_dir / "attempt_metrics.jsonl"),
            "sun_summary": identity(sun_dir / "attempt_summary.json"),
            "sun_attempts": identity(sun_dir / "attempt_results.jsonl"),
            "strict_summary": identity(sun_dir / "exact_strict" / "RESULTS_SUMMARY.md"),
            "meta_summary": identity(sun_dir / "exact_meta_like" / "RESULTS_SUMMARY.md"),
        },
        "source_manifest_sha256": args.source_manifest_sha256,
        "retry_replacement_repair_filter_rerank": False,
        "automatic_training": False,
        "automatic_promotion": False,
        "automatic_rl": False,
    }
    output = evaluation_dir / "stage_report.json"
    write_json_exclusive(output, report)
    with (evaluation_dir / "_SUCCESS").open("x", encoding="ascii") as handle:
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
