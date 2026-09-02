#!/usr/bin/env python3
"""Insert explicit failed body rows for failed prospective Planner requests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def pad_rows(
    body_rows: Sequence[Mapping[str, Any]],
    ledger: Sequence[Mapping[str, Any]],
    *,
    denominator: int,
) -> list[dict[str, Any]]:
    body = {int(row["sample_idx"]): dict(row) for row in body_rows}
    plans = {int(row["sample_idx"]): row for row in ledger}
    expected = set(range(denominator))
    if set(plans) != expected or not set(body) <= expected or len(body) != len(body_rows):
        raise ValueError("body or Planner indices are malformed")
    output = []
    for sample_idx in range(denominator):
        planner = plans[sample_idx]
        row = body.get(sample_idx)
        if planner.get("planner_valid") is True:
            if row is None:
                raise ValueError("valid Planner request is missing its body attempt")
            output.append(row)
            continue
        if row is not None:
            raise ValueError("failed Planner request unexpectedly has a body attempt")
        output.append(
            {
                "sample_idx": sample_idx,
                "parsed": False,
                "plan_match": False,
                "pymatgen_valid": False,
                "graph_valid": False,
                "cif": None,
                "plan_state": None,
                "reason": f"planner:{planner.get('failure') or 'invalid_plan'}",
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--body-rows", type=Path, required=True)
    parser.add_argument("--planner-ledger", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--denominator", type=int, default=256)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    rows = pad_rows(
        read_jsonl(args.body_rows),
        read_jsonl(args.planner_ledger),
        denominator=int(args.denominator),
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "raw_generations.jsonl").open(
        "x", encoding="utf-8", newline="\n"
    ) as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    report = {
        "schema": "spad_body_planner_denominator_v1",
        "denominator": int(args.denominator),
        "body_attempts": sum(
            not str(row.get("reason") or "").startswith("planner:") for row in rows
        ),
        "planner_failures": sum(str(row.get("reason") or "").startswith("planner:") for row in rows),
        "replacement": False,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "_SUCCESS").touch()
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
