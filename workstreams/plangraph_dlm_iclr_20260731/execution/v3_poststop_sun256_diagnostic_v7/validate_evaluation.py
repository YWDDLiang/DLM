#!/usr/bin/env python3
"""Validate one arm's direct metrics and frozen all-attempt S.U.N. mapping."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[3]
for location in (PROJECT_ROOT, HERE):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

from protocol import (  # noqa: E402
    read_json,
    read_jsonl,
    require_hex_sha,
    require_runtime_manifest,
    require_source_manifest,
    sha256_file,
    validate_arm,
    write_json_exclusive,
)


DENOMINATOR = 256


def _identity(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--generation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execution-manifest-sha256", required=True)
    args = parser.parse_args()

    arm = validate_arm(args.arm)
    execution_sha = require_hex_sha(
        args.execution_manifest_sha256,
        "execution source manifest",
    )
    source = args.source_dir.resolve()
    require_source_manifest(source, execution_sha)
    require_runtime_manifest(args.project_root.resolve(), source)
    config = read_json(args.config.resolve())
    method = str(config["arms"][arm]["method"])
    if (
        config["sun"].get("mp_api_enabled") is not False
        or config["decision_firewall"].get("formal_g3") is not False
        or config["decision_firewall"].get("automatic_downstream") is not False
    ):
        raise ValueError("frozen S.U.N. or decision firewall changed")

    generation_dir = args.generation_dir.resolve()
    output = args.output_dir.resolve()
    if not (generation_dir / "_SUCCESS").is_file():
        raise FileNotFoundError(generation_dir / "_SUCCESS")
    generation_report = read_json(generation_dir / "generation_report.json")
    generation = read_jsonl(generation_dir / "generation.jsonl")
    expected_ids = [str(row.get("attempt_id")) for row in generation]
    if (
        generation_report.get("ok") is not True
        or generation_report.get("all_successes_diffusion_refined") is not True
        or int(generation_report.get("diffusion_steps", -1)) != 800
        or len(generation) != DENOMINATOR
        or [int(row.get("ordinal", -1)) for row in generation]
        != list(range(DENOMINATOR))
        or {str(row.get("method")) for row in generation} != {method}
        or len(set(expected_ids)) != DENOMINATOR
        or any(
            row.get("retry_or_replacement_used") is not False for row in generation
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
        raise ValueError("generation/refinement denominator changed")

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
        raise ValueError("direct metric attempt mapping changed")
    if (
        sun_summary.get("ok") is not True
        or int((sun_summary.get("counts") or {}).get("total_attempts", -1))
        != DENOMINATOR
        or sun_summary.get("denominator") != "all_generation_attempts"
        or sun_summary.get("method") != method
        or sun_summary.get("execution_patch_sha256") != execution_sha
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
        raise ValueError("S.U.N. all-attempt mapping changed")

    report = {
        "schema": "h1a2_v3_poststop_sun256_arm_evaluation_v1",
        "status": "complete",
        "ok": True,
        "arm": arm,
        "method": method,
        "attempts": DENOMINATOR,
        "generation_succeeded": sum(
            row.get("status") == "succeeded" for row in generation
        ),
        "all_generation_successes_diffusion_refined": True,
        "diffusion_steps": 800,
        "direct": {
            "composition_valid_count": int(direct_report["comp_valid_count"]),
            "structure_valid_count": int(direct_report["struct_valid_count"]),
            "joint_valid_count": int(direct_report["valid_count"]),
            "composition_valid_rate": int(direct_report["comp_valid_count"])
            / DENOMINATOR,
            "structure_valid_rate": int(direct_report["struct_valid_count"])
            / DENOMINATOR,
            "joint_valid_rate": int(direct_report["valid_count"]) / DENOMINATOR,
            "report": _identity(direct_dir / "report.json"),
            "attempt_metrics": _identity(
                direct_dir / "attempt_metrics.jsonl"
            ),
        },
        "sun": {
            "counts": sun_summary["counts"],
            "rates": sun_summary["rates"],
            "summary": _identity(sun_dir / "attempt_summary.json"),
            "attempt_results": _identity(sun_dir / "attempt_results.jsonl"),
        },
        "generation": _identity(generation_dir / "generation.jsonl"),
        "execution_manifest_sha256": execution_sha,
        "formal_g3": False,
        "automatic_promotion": False,
        "automatic_downstream": False,
    }
    write_json_exclusive(output / "evaluation_report.json", report)
    with (output / "_SUCCESS").open("x", encoding="ascii") as handle:
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
