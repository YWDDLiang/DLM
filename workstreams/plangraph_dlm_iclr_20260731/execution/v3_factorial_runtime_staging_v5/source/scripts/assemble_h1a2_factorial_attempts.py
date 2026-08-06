#!/usr/bin/env python3
"""Assemble four H1-A2 arm ledgers without dropping failed attempts."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from crystal_dlm.fixed_slot import write_json  # noqa: E402
from crystal_dlm.h1a2_factorial_contract import (  # noqa: E402
    FACTORIAL_ARMS,
    ordered_factorial_attempts,
)
from crystal_dlm.h1a2_factorial_runtime import (  # noqa: E402
    assert_factorial_pairing,
    ordered_single_arm_attempts,
    read_jsonl_objects,
)


def _write_jsonl(path: Path, records: list[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m00-attempts", type=Path, required=True)
    parser.add_argument("--m10-attempts", type=Path, required=True)
    parser.add_argument("--m01-attempts", type=Path, required=True)
    parser.add_argument("--m11-attempts", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if int(args.num_samples) <= 0:
        raise ValueError("--num-samples must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "M00": args.m00_attempts,
        "M10": args.m10_attempts,
        "M01": args.m01_attempts,
        "M11": args.m11_attempts,
    }
    records: list[Mapping[str, Any]] = []
    arm_reports: dict[str, Any] = {}
    for arm in FACTORIAL_ARMS:
        ordered = ordered_single_arm_attempts(
            read_jsonl_objects(paths[arm]),
            expected_count=int(args.num_samples),
            expected_factorial_arm=arm,
        )
        records.extend(ordered)
        failures = Counter(
            str(record.get("earliest_failure_stage") or "none")
            for record in ordered
            if record.get("attempt_status") == "failed"
        )
        complete = sum(
            record.get("attempt_status") == "complete" for record in ordered
        )
        arm_reports[arm] = {
            "all_attempt_denominator": int(args.num_samples),
            "complete": complete,
            "failed": int(args.num_samples) - complete,
            "completion_rate_all_attempt": complete
            / max(1, int(args.num_samples)),
            "earliest_failure_counts": dict(sorted(failures.items())),
        }

    pairing = assert_factorial_pairing(
        records,
        expected_count=int(args.num_samples),
    )
    ordered_four_arm = ordered_factorial_attempts(
        records,
        expected_count=int(args.num_samples),
    )
    _write_jsonl(
        args.output_dir / "factorial_attempts.jsonl",
        list(ordered_four_arm),
    )
    write_json(
        str(args.output_dir / "factorial_assembly_report.json"),
        {
            "schema": "h1a2_factorial_assembly_v1",
            "status": "complete",
            "all_attempt_denominator_per_arm": int(args.num_samples),
            "total_attempts": 4 * int(args.num_samples),
            "arms": arm_reports,
            "pairing": pairing,
            "raw_all_attempt_primary": True,
            "retry": False,
            "replacement": False,
            "repair": False,
            "filter": False,
            "rerank": False,
            "scientific_selection_performed": False,
            "automatic_downstream": False,
        },
    )


if __name__ == "__main__":
    main()
