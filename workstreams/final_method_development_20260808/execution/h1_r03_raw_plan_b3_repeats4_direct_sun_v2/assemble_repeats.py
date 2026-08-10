#!/usr/bin/env python3
"""Assemble paired R03 B0 versus new R03-plan B3 repeat evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np


CELLS = ("M00", "M10", "M01", "M11")
DENOMINATOR = 256


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def require_sha(path: Path, expected: str, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    observed = sha256_file(resolved)
    if observed != expected:
        raise ValueError(f"{label} changed: expected={expected} observed={observed}")
    return resolved


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected one JSON object")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected one object")
            rows.append(value)
    return rows


def ordered(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    result = sorted(rows, key=lambda row: int(row["generation_ordinal"]))
    if (
        len(result) != DENOMINATOR
        or [int(row.get("generation_ordinal", -1)) for row in result]
        != list(range(DENOMINATOR))
    ):
        raise ValueError("generation ordinal coverage changed")
    return result


def require_source_manifest(source: Path, expected_sha: str) -> None:
    manifest = require_sha(
        source / "SOURCE_SHA256.txt", expected_sha, "execution source manifest"
    )
    listed: set[str] = set()
    for line_number, line in enumerate(
        manifest.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        pieces = line.split("  ", 1)
        if len(pieces) != 2:
            raise ValueError(f"manifest line {line_number} is malformed")
        expected, relative = pieces
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"manifest line {line_number} has unsafe path")
        require_sha(source / relative_path, expected, f"source file {relative}")
        listed.add(relative_path.as_posix())
    observed = {
        path.relative_to(source).as_posix()
        for path in source.rglob("*")
        if path.is_file()
        and path.name != "SOURCE_SHA256.txt"
        and "__pycache__" not in path.parts
        and path.suffix not in {".pyc", ".pyo"}
    }
    if observed != listed:
        raise ValueError(
            f"source file set changed: missing={sorted(listed-observed)} "
            f"extra={sorted(observed-listed)}"
        )


def endpoint(row: dict[str, Any], name: str) -> bool:
    metrics = row.get("metrics") or {}
    if name == "generation_complete":
        return row.get("generation_status") == "succeeded"
    if name == "hull_evaluated":
        return metrics.get("e_above_hull") is not None
    return bool(metrics.get(name))


def exact_mcnemar(baseline: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    baseline_only = int(np.count_nonzero(baseline & ~candidate))
    candidate_only = int(np.count_nonzero(candidate & ~baseline))
    discordant = baseline_only + candidate_only
    if discordant == 0:
        p_value = 1.0
    else:
        lower = min(baseline_only, candidate_only)
        numerator = sum(math.comb(discordant, index) for index in range(lower + 1))
        p_value = min(1.0, 2.0 * numerator / (2**discordant))
    return {
        "baseline_only": baseline_only,
        "candidate_only": candidate_only,
        "discordant": discordant,
        "two_sided_exact_p_value": p_value,
    }


def paired_summary(baseline: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    attempts = int(baseline.size)
    baseline_count = int(np.count_nonzero(baseline))
    candidate_count = int(np.count_nonzero(candidate))
    return {
        "attempts": attempts,
        "baseline_count": baseline_count,
        "candidate_count": candidate_count,
        "difference_count": candidate_count - baseline_count,
        "baseline_rate": baseline_count / attempts,
        "candidate_rate": candidate_count / attempts,
        "difference_percentage_points": 100.0
        * (candidate_count - baseline_count)
        / attempts,
        "exact_mcnemar": exact_mcnemar(baseline, candidate),
    }


def hierarchical_bootstrap(
    differences: np.ndarray, *, draws: int, seed: int
) -> dict[str, Any]:
    if differences.shape != (4, DENOMINATOR):
        raise ValueError("hierarchical bootstrap matrix shape changed")
    rng = np.random.default_rng(seed)
    values = np.empty(draws, dtype=np.float64)
    batch_size = 1000
    for start in range(0, draws, batch_size):
        stop = min(draws, start + batch_size)
        size = stop - start
        repeat_indices = rng.integers(0, 4, size=(size, 4))
        ordinal_indices = rng.integers(
            0, DENOMINATOR, size=(size, 4, DENOMINATOR)
        )
        sampled = differences[repeat_indices[:, :, None], ordinal_indices]
        values[start:stop] = sampled.mean(axis=(1, 2)) * 100.0
    lower, upper = np.quantile(values, [0.025, 0.975])
    return {
        "design": "resample repeat blocks, then paired ordinals within repeats",
        "draws": draws,
        "seed": seed,
        "observed_mean_difference_percentage_points": float(
            differences.mean() * 100.0
        ),
        "ci95_lower_percentage_points": float(lower),
        "ci95_upper_percentage_points": float(upper),
        "probability_difference_gt_zero": float(np.mean(values > 0.0)),
        "probability_difference_lt_zero": float(np.mean(values < 0.0)),
    }


def write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = args.source_dir.resolve()
    run_root = args.run_root.resolve()
    output = args.output.resolve()
    require_source_manifest(source, args.source_manifest_sha256)
    config = read_json(args.config.resolve())
    if (
        config.get("run_root") != str(run_root)
        or config.get("run_id")
        != "20260810_h1_r03_raw_plan_b3_safeaxis_direct_sun_runtime_path_repair_v2"
        or output != run_root / "terminal_report.json"
        or output.exists()
    ):
        raise ValueError("assembly run/output identity changed")

    preflight = read_json(run_root / "status/preflight_report.json")
    completion = read_json(run_root / config["sun"]["completion_manifest"])
    if (
        preflight.get("status") != "pass"
        or preflight.get("source_manifest_sha256")
        != args.source_manifest_sha256
        or completion.get("external_query_performed") is not False
        or (completion.get("completed_mp_hull_cache") or {}).get("sha256")
        != config["sun"]["r03f_snapshot_sha256"]
    ):
        raise ValueError("preflight/cache evidence changed")

    endpoints = list(config["analysis"]["sun_endpoints"])
    historical_specs = config["analysis"]["historical_r03_b0"][
        "attempt_results"
    ]
    baseline_arrays: dict[str, list[np.ndarray]] = {name: [] for name in endpoints}
    candidate_arrays: dict[str, list[np.ndarray]] = {name: [] for name in endpoints}
    repeat_reports: list[dict[str, Any]] = []

    for repeat, cell in enumerate(CELLS):
        cell_root = run_root / "cells" / cell
        exit_code = run_root / f"status/repeat_{repeat}_exit_code.txt"
        if (
            not (cell_root / "_SUCCESS").is_file()
            or not (cell_root / "evaluation/_SUCCESS").is_file()
            or not exit_code.is_file()
            or exit_code.read_text(encoding="ascii").strip() != "0"
        ):
            raise ValueError(f"repeat {repeat} is not an engineering success")

        evaluation_path = cell_root / "evaluation/evaluation_report.json"
        evaluation = read_json(evaluation_path)
        expected_method = config["cells"][cell]["method"]
        if (
            evaluation.get("status") != "complete"
            or evaluation.get("ok") is not True
            or evaluation.get("cell") != cell
            or evaluation.get("planner") != "P0"
            or evaluation.get("body") != "B3"
            or evaluation.get("method") != expected_method
            or int(evaluation.get("attempts", -1)) != DENOMINATOR
            or (evaluation.get("evaluation_contract") or {}).get(
                "direct_comp_valid_order"
            )
            != "gcd_then_smact_validity"
            or (evaluation.get("evaluation_contract") or {}).get(
                "completed_mp_hull_cache_sha256"
            )
            != config["sun"]["r03f_snapshot_sha256"]
        ):
            raise ValueError(f"repeat {repeat} evaluation contract changed")

        candidate_path = cell_root / "evaluation/r5c_a100_sun/attempt_results.jsonl"
        candidate = ordered(read_jsonl(candidate_path))
        historical_spec = historical_specs[repeat]
        baseline_path = require_sha(
            Path(historical_spec["path"]),
            historical_spec["sha256"],
            f"historical R03 B0 repeat {repeat}",
        )
        baseline = ordered(read_jsonl(baseline_path))
        if any(
            row.get("retry_or_replacement_used") is not False
            or row.get("schema") != "crysllmgen_r5c_a100_sun_attempt_v1"
            for row in candidate + baseline
        ):
            raise ValueError(f"repeat {repeat} S.U.N. attempt contract changed")

        per_endpoint: dict[str, Any] = {}
        for name in endpoints:
            baseline_values = np.asarray(
                [endpoint(row, name) for row in baseline], dtype=np.bool_
            )
            candidate_values = np.asarray(
                [endpoint(row, name) for row in candidate], dtype=np.bool_
            )
            baseline_arrays[name].append(baseline_values)
            candidate_arrays[name].append(candidate_values)
            per_endpoint[name] = paired_summary(baseline_values, candidate_values)

        repeat_reports.append(
            {
                "repeat": repeat,
                "cell_slot": cell,
                "method": expected_method,
                "candidate_direct_counts": evaluation["direct_counts"],
                "candidate_direct_rates": evaluation["direct_rates"],
                "candidate_sun_counts": evaluation["sun_counts"],
                "candidate_sun_rates": evaluation["sun_rates"],
                "candidate_sun_hull": evaluation["sun_hull"],
                "paired_sun_endpoints": per_endpoint,
                "artifacts": {
                    "candidate_evaluation_report": identity(evaluation_path),
                    "candidate_attempt_results": identity(candidate_path),
                    "historical_r03_b0_attempt_results": identity(baseline_path),
                },
            }
        )

    historical = config["analysis"]["historical_r03_b0"]
    historical_terminal = require_sha(
        Path(historical["terminal_report"]),
        historical["terminal_report_sha256"],
        "historical R03G terminal report",
    )
    draws = int(config["analysis"]["bootstrap_draws"])
    seed = int(config["analysis"]["bootstrap_seed"])
    pooled: dict[str, Any] = {}
    inference: dict[str, Any] = {}
    for endpoint_index, name in enumerate(endpoints):
        baseline_matrix = np.stack(baseline_arrays[name], axis=0)
        candidate_matrix = np.stack(candidate_arrays[name], axis=0)
        differences = candidate_matrix.astype(np.int8) - baseline_matrix.astype(
            np.int8
        )
        pooled[name] = {
            **paired_summary(baseline_matrix.ravel(), candidate_matrix.ravel()),
            "interpretation": "descriptive only; repeated ordinals are not independent",
        }
        repeat_differences = candidate_matrix.sum(axis=1) - baseline_matrix.sum(
            axis=1
        )
        inference[name] = {
            "per_repeat_difference_counts": [
                int(value) for value in repeat_differences
            ],
            "sign_stability": {
                "positive_repeats": int(np.count_nonzero(repeat_differences > 0)),
                "zero_repeats": int(np.count_nonzero(repeat_differences == 0)),
                "negative_repeats": int(np.count_nonzero(repeat_differences < 0)),
            },
            "hierarchical_paired_bootstrap": hierarchical_bootstrap(
                differences, draws=draws, seed=seed + endpoint_index
            ),
        }

    strict_signs = inference["strict_full_sun"]["sign_stability"]
    meta_signs = inference["meta_full_sun"]["sign_stability"]
    report = {
        "schema": "h1_r03_raw_plan_b3_repeats4_terminal_report_v2",
        "status": "complete",
        "run_id": config["run_id"],
        "scientific_question": "B3 minus historical B0 under byte-frozen R03 raw Plan",
        "candidate": "R03 raw P0 + B3 + D2 safe-axis + model_494 + Direct + S.U.N.",
        "baseline": "historical R03G raw P0 + B0 + D2 safe-axis + model_494 + S.U.N.",
        "attempts_per_repeat": DENOMINATOR,
        "process_repeats": 4,
        "candidate_attempts_total": 4 * DENOMINATOR,
        "historical_control_rerun": False,
        "repeat_interpretation": (
            "independent A800 process realizations with frozen scientific seeds; "
            "not four new planner samples"
        ),
        "repeat_reports": repeat_reports,
        "paired_sun_inference": inference,
        "pooled_1024_descriptive": pooled,
        "primary_interpretation": {
            "strict_repeat_signs": strict_signs,
            "meta_repeat_signs": meta_signs,
            "promotion_claim": False,
            "reason": (
                "exploratory B3-on-R03 repeat evidence is reported with paired "
                "uncertainty; no automatic promotion was authorized"
            ),
        },
        "statistical_mouth": {
            "all_raw_failures_in_256_denominator": True,
            "ordinal_pairing": True,
            "per_repeat_exact_mcnemar": True,
            "hierarchical_paired_bootstrap_draws": draws,
            "pooled_1024_is_descriptive_only": True,
        },
        "evaluation_contract": {
            "r03_raw_plan_first256_sha256": config["planner_sources"]["P0"][
                "raw_generations_sha256"
            ],
            "b3_adapter_sha256": config["body"]["models"]["B3"][
                "adapter_sha256"
            ],
            "direct_comp_valid_order": "gcd_then_smact_validity",
            "r03f_cache_sha256": config["sun"]["r03f_snapshot_sha256"],
            "r03f_cache_rows": int(config["sun"]["wanted_chemsys_count"]),
            "external_mp_query": False,
            "retry_replacement_repair_filter_rerank": False,
        },
        "artifacts": {
            "preflight_report": identity(run_root / "status/preflight_report.json"),
            "completion_manifest": identity(
                run_root / config["sun"]["completion_manifest"]
            ),
            "r03f_cache": identity(
                run_root / config["sun"]["completed_mp_hull_cache"]
            ),
            "historical_r03g_terminal_report": identity(historical_terminal),
            "submission_record": identity(
                run_root / "status/submission_record.json"
            ),
        },
        "source_manifest_sha256": args.source_manifest_sha256,
        "formal_promotion": False,
        "automatic_checkpoint_reselection": False,
        "automatic_training": False,
        "automatic_downstream": False,
        "automatic_rl": False,
    }
    write_json_exclusive(output, report)
    print(json.dumps(report, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
