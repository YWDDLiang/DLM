#!/usr/bin/env python3
"""Validate one arm's direct metrics and exact S.U.N. attempt mapping."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
PROJECT_ROOT_FALLBACK = HERE.parents[3]
for location in (PROJECT_ROOT_FALLBACK, HERE):
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
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--generation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execution-manifest-sha256", required=True)
    args = parser.parse_args()

    arm = validate_arm(args.arm)
    execution_sha = require_hex_sha(
        args.execution_manifest_sha256, "execution source manifest"
    )
    require_source_manifest(args.source_dir.resolve(), execution_sha)
    require_runtime_manifest(args.project_root.resolve(), args.source_dir.resolve())
    config = read_json(args.config.resolve())
    data_manifest = read_json(args.data_dir.resolve() / "ledger_manifest.json")
    if (
        data_manifest.get("execution_manifest_sha256") != execution_sha
        or data_manifest.get("ok") is not True
    ):
        raise ValueError("evaluation ledger identity changed")
    generation_dir = args.generation_dir.resolve()
    output = args.output_dir.resolve()
    if not (generation_dir / "_SUCCESS").is_file():
        raise FileNotFoundError(generation_dir / "_SUCCESS")
    generation_report = read_json(generation_dir / "generation_report.json")
    generation = read_jsonl(generation_dir / "generation.jsonl")
    method = str(config["source_plan_run"]["arms"][arm]["method"])
    if (
        generation_report.get("ok") is not True
        or len(generation) != 256
        or [int(row.get("ordinal", -1)) for row in generation] != list(range(256))
        or {str(row.get("method")) for row in generation} != {method}
        or len({str(row.get("attempt_id")) for row in generation}) != 256
        or any(row.get("retry_or_replacement_used") is not False for row in generation)
    ):
        raise ValueError("generation denominator is not the frozen 256 attempts")
    attempt_ids = [str(row["attempt_id"]) for row in generation]

    direct_dir = output / "crysllmgen_metrics"
    sun_dir = output / "r5c_a100_sun"
    direct_report = read_json(direct_dir / "report.json")
    direct_attempts = read_jsonl(direct_dir / "attempt_metrics.jsonl")
    sun_summary = read_json(sun_dir / "attempt_summary.json")
    sun_attempts = read_jsonl(sun_dir / "attempt_results.jsonl")
    if (
        direct_report.get("ok") is not True
        or int(direct_report.get("attempts", -1)) != 256
        or direct_report.get("denominator") != "all_generation_attempts"
        or direct_report.get("method") != method
        or direct_report.get("retry_or_replacement_used") is not False
        or len(direct_attempts) != 256
        or [str(row.get("attempt_id")) for row in direct_attempts] != attempt_ids
        or any(
            row.get("schema") != "crysllmgen_metric_attempt_v1"
            or row.get("method") != method
            for row in direct_attempts
        )
    ):
        raise ValueError("direct CrysLLMGen metric attempt mapping changed")
    if (
        sun_summary.get("ok") is not True
        or int((sun_summary.get("counts") or {}).get("total_attempts", -1)) != 256
        or sun_summary.get("denominator") != "all_generation_attempts"
        or sun_summary.get("method") != method
        or sun_summary.get("execution_patch_sha256") != execution_sha
        or sun_summary.get("retry_or_replacement_used") is not False
        or len(sun_attempts) != 256
        or [str(row.get("attempt_id")) for row in sun_attempts] != attempt_ids
        or any(
            row.get("schema") != "crysllmgen_r5c_a100_sun_attempt_v1"
            or row.get("method") != method
            or row.get("retry_or_replacement_used") is not False
            for row in sun_attempts
        )
    ):
        raise ValueError("exact S.U.N. attempt mapping changed")

    report = {
        "schema": "h1a2c_p0_p1_arm_evaluation_report_v1",
        "ok": True,
        "arm": arm,
        "method": method,
        "attempts": 256,
        "generation_succeeded": sum(
            row.get("status") == "succeeded" for row in generation
        ),
        "direct": {
            "comp_valid_count": int(direct_report["comp_valid_count"]),
            "struct_valid_count": int(direct_report["struct_valid_count"]),
            "joint_valid_count": int(direct_report["valid_count"]),
            "comp_valid_rate": int(direct_report["comp_valid_count"]) / 256,
            "struct_valid_rate": int(direct_report["struct_valid_count"]) / 256,
            "joint_valid_rate": int(direct_report["valid_count"]) / 256,
            "report": _identity(direct_dir / "report.json"),
            "attempt_metrics": _identity(direct_dir / "attempt_metrics.jsonl"),
        },
        "sun": {
            "counts": sun_summary["counts"],
            "rates": sun_summary["rates"],
            "summary": _identity(sun_dir / "attempt_summary.json"),
            "attempt_results": _identity(sun_dir / "attempt_results.jsonl"),
        },
        "generation": _identity(generation_dir / "generation.jsonl"),
        "execution_manifest_sha256": execution_sha,
        "retry_or_replacement_used": False,
        "automatic_downstream_authorized": False,
    }
    write_json_exclusive(output / "evaluation_report.json", report)
    with (output / "_SUCCESS").open("x", encoding="ascii") as handle:
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
