#!/usr/bin/env python3
"""Validate one evaluation-only S.U.N. arm against frozen v7 evidence."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from protocol import (
    DENOMINATOR,
    read_json,
    read_jsonl,
    require_source_manifest,
    sha256_file,
    validate_arm,
    verify_frozen_arm,
    write_json_exclusive,
)


V7_SOURCE_SHA256 = "5d05e23da6ba4e0e49f4646a5db8181c27f3dd47d67148cac9baaa842fa6a42f"


def identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    args = parser.parse_args()

    arm = validate_arm(args.arm)
    source = args.source_dir.resolve()
    require_source_manifest(source, args.source_manifest_sha256)
    frozen = verify_frozen_arm(args.input_manifest.resolve(), arm)
    output = args.output_dir.resolve()
    preflight = read_json(output / "preflight_report.json")
    sun_dir = output / "r5c_a100_sun"
    if (sun_dir / "executor_failure.json").exists():
        raise RuntimeError("S.U.N. executor recorded a failure")
    run_contract = read_json(sun_dir / "run_contract.json")
    sun_summary = read_json(sun_dir / "attempt_summary.json")
    sun_attempts = read_jsonl(sun_dir / "attempt_results.jsonl")
    expected_ids = frozen["attempt_ids"]
    method = frozen["method"]
    if (
        preflight.get("status") != "pass"
        or preflight.get("arm") != arm
        or preflight.get("mp_credentials_present") is not False
        or preflight.get("generation_or_refinement_rerun") is not False
        or preflight.get("direct_metrics_rerun") is not False
    ):
        raise ValueError("arm preflight report changed")
    if (
        run_contract.get("offline") is not True
        or int(run_contract.get("expected_attempts", -1)) != DENOMINATOR
        or run_contract.get("base_source_bundle_sha256") != V7_SOURCE_SHA256
        or run_contract.get("execution_patch_sha256")
        != args.source_manifest_sha256
        or run_contract.get("retry_or_replacement_used") is not False
        or "A800" not in str(run_contract.get("cuda_device", ""))
    ):
        raise ValueError("S.U.N. execution contract changed")
    if (
        sun_summary.get("ok") is not True
        or sun_summary.get("method") != method
        or sun_summary.get("denominator") != "all_generation_attempts"
        or int((sun_summary.get("counts") or {}).get("total_attempts", -1))
        != DENOMINATOR
        or sun_summary.get("base_source_bundle_sha256") != V7_SOURCE_SHA256
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
        raise ValueError("S.U.N. all-attempt mapping changed")
    counts = {
        "novel": sum(bool(row["metrics"]["novel"]) for row in sun_attempts),
        "unique": sum(
            bool(row["metrics"]["unique_representative"]) for row in sun_attempts
        ),
        "novel_unique": sum(
            bool(row["metrics"]["novel_unique"]) for row in sun_attempts
        ),
        "strict_full_sun": sum(
            bool(row["metrics"]["strict_full_sun"]) for row in sun_attempts
        ),
        "meta_full_sun": sum(
            bool(row["metrics"]["meta_full_sun"]) for row in sun_attempts
        ),
    }
    summary_counts = sun_summary["counts"]
    if (
        any(int(summary_counts[key]) != value for key, value in counts.items())
        or counts["meta_full_sun"] < counts["strict_full_sun"]
    ):
        raise ValueError("S.U.N. summary/attempt counts disagree")

    report = {
        "schema": "h1a2_v3_sun_evaluation_repair_arm_v1",
        "status": "complete",
        "ok": True,
        "arm": arm,
        "method": method,
        "attempts": DENOMINATOR,
        "frozen_v7": {
            "generation_succeeded": frozen["counts"]["generation_succeeded"],
            "composition_valid": frozen["counts"]["composition_valid"],
            "structure_valid": frozen["counts"]["structure_valid"],
            "joint_valid": frozen["counts"]["joint_valid"],
            "all_successes_diffusion_refined": True,
            "diffusion_steps": 800,
            "generation_jsonl": frozen["specification"]["generation_jsonl"],
            "direct_attempt_metrics": frozen["specification"][
                "direct_attempt_metrics"
            ],
            "generation_or_refinement_rerun": False,
            "direct_metrics_rerun": False,
        },
        "sun": {
            "counts": counts,
            "rates": {
                key: value / DENOMINATOR for key, value in counts.items()
            },
            "attempt_summary": identity(sun_dir / "attempt_summary.json"),
            "attempt_results": identity(sun_dir / "attempt_results.jsonl"),
            "run_contract": identity(sun_dir / "run_contract.json"),
            "mp_api_enabled": False,
            "frozen_cache_only": True,
        },
        "source_manifest_sha256": args.source_manifest_sha256,
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
