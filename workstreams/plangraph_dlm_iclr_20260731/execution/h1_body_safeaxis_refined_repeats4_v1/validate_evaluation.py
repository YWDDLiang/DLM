#!/usr/bin/env python3
"""Validate one R03E arm/repeat direct and frozen-cache S.U.N. result."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

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


def _identity(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True)
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--generation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    arm = validate_arm(args.arm)
    repeat = validate_repeat(args.repeat)
    source = args.source_dir.resolve()
    require_source_manifest(source, args.source_manifest_sha256)
    config = read_json(args.config.resolve())
    validate_config(config)
    method = str(config["arms"][arm]["method"])

    generation_dir = args.generation_dir.resolve()
    output = args.output_dir.resolve()
    if not (generation_dir / "_SUCCESS").is_file():
        raise FileNotFoundError(generation_dir / "_SUCCESS")
    generation_report = read_json(generation_dir / "generation_report.json")
    generation = ordered_rows(
        read_jsonl(generation_dir / "generation.jsonl"),
        ordinal_field="ordinal",
    )
    expected_ids = [str(row.get("attempt_id")) for row in generation]
    if (
        generation_report.get("ok") is not True
        or generation_report.get("all_successes_diffusion_refined") is not True
        or int(generation_report.get("diffusion_steps", -1)) != 800
        or int(generation_report.get("repeat", -1)) != repeat
        or generation_report.get("arm") != arm
        or len(generation) != DENOMINATOR
        or {str(row.get("method")) for row in generation} != {method}
        or {int(row.get("repeat", -1)) for row in generation} != {repeat}
        or {str(row.get("arm")) for row in generation} != {arm}
        or len(set(expected_ids)) != DENOMINATOR
        or any(
            row.get("retry_or_replacement_used") is not False
            or row.get("new_scientific_seed_per_repeat") is not False
            for row in generation
        )
        or any(
            row.get("status") == "succeeded"
            and (
                row.get("diffusion_refinement_applied") is not True
                or int(row.get("diffusion_refinement_steps", -1)) != 800
            )
            for row in generation
        )
    ):
        raise ValueError("R03E generation/refinement denominator changed")

    direct_dir = output / "crysllmgen_metrics"
    sun_dir = output / "r5c_a100_sun"
    direct_report = read_json(direct_dir / "report.json")
    direct_attempts = read_jsonl(direct_dir / "attempt_metrics.jsonl")
    sun_summary = read_json(sun_dir / "attempt_summary.json")
    sun_attempts = read_jsonl(sun_dir / "attempt_results.jsonl")
    if (
        direct_report.get("ok") is not True
        or int(direct_report.get("attempts", -1)) != DENOMINATOR
        or direct_report.get("denominator") != "all_generation_attempts"
        or direct_report.get("method") != method
        or len(direct_attempts) != DENOMINATOR
        or [str(row.get("attempt_id")) for row in direct_attempts] != expected_ids
        or any(
            row.get("schema") != "crysllmgen_metric_attempt_v1"
            or row.get("method") != method
            for row in direct_attempts
        )
    ):
        raise ValueError("R03E direct metric attempt mapping changed")
    if (
        sun_summary.get("ok") is not True
        or int((sun_summary.get("counts") or {}).get("total_attempts", -1))
        != DENOMINATOR
        or sun_summary.get("denominator") != "all_generation_attempts"
        or sun_summary.get("method") != method
        or sun_summary.get("execution_patch_sha256")
        != args.source_manifest_sha256
        or sun_summary.get("retry_or_replacement_used") is not False
        or len(sun_attempts) != DENOMINATOR
        or [str(row.get("attempt_id")) for row in sun_attempts] != expected_ids
        or any(
            row.get("schema") != "crysllmgen_r5c_a100_sun_attempt_v1"
            or row.get("method") != method
            or row.get("retry_or_replacement_used") is not False
            for row in sun_attempts
        )
    ):
        raise ValueError("R03E S.U.N. all-attempt mapping changed")

    direct_counts = {
        "composition_valid": sum(
            bool(row.get("comp_valid")) for row in direct_attempts
        ),
        "structure_valid": sum(
            bool(row.get("struct_valid")) for row in direct_attempts
        ),
        "joint_valid": sum(bool(row.get("valid")) for row in direct_attempts),
    }
    sun_counts = {
        "novel": sum(
            bool((row.get("metrics") or {}).get("novel")) for row in sun_attempts
        ),
        "unique_representative": sum(
            bool((row.get("metrics") or {}).get("unique_representative"))
            for row in sun_attempts
        ),
        "novel_unique": sum(
            bool((row.get("metrics") or {}).get("novel_unique"))
            for row in sun_attempts
        ),
        "strict_full_sun": sum(
            bool((row.get("metrics") or {}).get("strict_full_sun"))
            for row in sun_attempts
        ),
        "meta_full_sun": sum(
            bool((row.get("metrics") or {}).get("meta_full_sun"))
            for row in sun_attempts
        ),
    }
    expected_sun = sun_summary["counts"]
    if (
        direct_counts["composition_valid"]
        != int(direct_report["comp_valid_count"])
        or direct_counts["structure_valid"]
        != int(direct_report["struct_valid_count"])
        or direct_counts["joint_valid"] != int(direct_report["valid_count"])
        or sun_counts["novel"] != int(expected_sun["novel"])
        or sun_counts["unique_representative"] != int(expected_sun["unique"])
        or sun_counts["novel_unique"] != int(expected_sun["novel_unique"])
        or sun_counts["strict_full_sun"] != int(expected_sun["strict_full_sun"])
        or sun_counts["meta_full_sun"] != int(expected_sun["meta_full_sun"])
    ):
        raise ValueError("R03E endpoint count parity changed")

    report = {
        "schema": "h1_r03e_arm_evaluation_v1",
        "status": "complete",
        "ok": True,
        "arm": arm,
        "repeat": repeat,
        "method": method,
        "attempts": DENOMINATOR,
        "generation_succeeded": sum(
            row.get("status") == "succeeded" for row in generation
        ),
        "all_generation_successes_diffusion_refined": True,
        "diffusion_steps": 800,
        "direct_counts": direct_counts,
        "sun_counts": sun_counts,
        "rates": {
            **{key: value / DENOMINATOR for key, value in direct_counts.items()},
            **{key: value / DENOMINATOR for key, value in sun_counts.items()},
        },
        "artifacts": {
            "generation": _identity(generation_dir / "generation.jsonl"),
            "direct_report": _identity(direct_dir / "report.json"),
            "direct_attempts": _identity(
                direct_dir / "attempt_metrics.jsonl"
            ),
            "sun_summary": _identity(sun_dir / "attempt_summary.json"),
            "sun_attempts": _identity(sun_dir / "attempt_results.jsonl"),
        },
        "source_manifest_sha256": args.source_manifest_sha256,
        "formal_g3": False,
        "automatic_promotion": False,
        "automatic_training": False,
        "automatic_downstream": False,
    }
    write_json_exclusive(output / "evaluation_report.json", report)
    with (output / "_SUCCESS").open("x", encoding="ascii") as handle:
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
