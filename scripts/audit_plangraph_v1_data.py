#!/usr/bin/env python3
"""Audit PlanGraph v1 conversion coverage for one source JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from crystal_dlm.h1_readonly_guard import assert_writable_output_path  # noqa: E402
from crystal_dlm.plangraph_audit import audit_plangraph_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-failure-examples", type=int, default=20)
    parser.add_argument(
        "--min-conversion-rate",
        type=float,
        default=0.98,
        help="Exit non-zero below this preregistered coverage gate",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = audit_plangraph_jsonl(
        args.input,
        max_failure_examples=args.max_failure_examples,
    )
    report["minimum_conversion_rate"] = float(args.min_conversion_rate)
    report["coverage_gate_passed"] = report["conversion_rate"] >= float(
        args.min_conversion_rate
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is None:
        print(rendered)
    else:
        output = assert_writable_output_path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as handle:
            handle.write(rendered)
            handle.write("\n")
        print(
            json.dumps(
                {
                    "output": str(output),
                    "coverage_gate_passed": report["coverage_gate_passed"],
                },
                sort_keys=True,
            )
        )
    return 0 if report["coverage_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
