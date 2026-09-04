#!/usr/bin/env python3
"""Authorize the paper-scale endpoint from a frozen fixed-256 result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


STRICT_EXACT = 26
META_EXACT = 128
STRICT_NEAR = 23
META_NEAR = 125


def authorize(result: Mapping[str, Any]) -> dict[str, Any]:
    cells = list(result.get("cells") or [])
    evaluated = []
    qualifying = []
    for cell in cells:
        denominator = int(cell.get("fixed_denominator", -1))
        if denominator != 256:
            raise ValueError("paper authorization requires fixed denominator 256")
        endpoint = str(cell["endpoint"])
        strict = int(cell["strict_sun"]["count"])
        meta = int(cell["meta_sun"]["count"])
        exact = strict >= STRICT_EXACT and meta >= META_EXACT
        near = strict >= STRICT_NEAR and meta >= META_NEAR
        record = {
            "endpoint": endpoint,
            "strict_sun": strict,
            "meta_sun": meta,
            "exact_10_50": exact,
            "within_three_counts_of_both_targets": near,
        }
        evaluated.append(record)
        if exact or near:
            qualifying.append(record)
    return {
        "schema": "spad_paper1000_authorization_v1",
        "fixed_denominator": 256,
        "exact_threshold_counts": {"strict": STRICT_EXACT, "meta": META_EXACT},
        "preregistered_near_miss_counts": {"strict": STRICT_NEAR, "meta": META_NEAR},
        "evaluated_endpoints": evaluated,
        "qualifying_endpoints": qualifying,
        "paper1000_authorized": bool(qualifying),
        "authorization_rule": (
            "same_endpoint_exact_26_128_or_preregistered_within_three_23_125"
        ),
        "paper_run_reports_raw_and_tau800_from_one_shared_generation": True,
        "outcome_based_row_selection": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed256-final", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = json.loads(args.fixed256_final.read_text(encoding="utf-8"))
    report = authorize(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))
    if not report["paper1000_authorized"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
