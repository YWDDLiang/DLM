#!/usr/bin/env python3
"""Exact validity-only subset of the frozen CrysLLMGen Direct evaluator."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import sys


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation-jsonl", type=Path, required=True)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-report", type=Path)
    parser.add_argument("--expected-denominator", type=int, default=256)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("registered fast Direct must run through Slurm")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    sys.path.insert(0, str(args.snapshot_root.resolve()))
    try:
        from eval_utils import smact_validity, structure_validity
    finally:
        sys.path.pop(0)
    from pymatgen.core import Structure

    rows = read_jsonl(args.generation_jsonl.resolve())
    if args.expected_denominator <= 0:
        raise ValueError("expected-denominator must be positive")
    if len(rows) != args.expected_denominator or [int(row["ordinal"]) for row in rows] != list(range(args.expected_denominator)):
        raise ValueError("generation denominator or ordinal order changed")
    attempts = []
    for row in rows:
        comp_valid = False
        struct_valid = False
        reason = ""
        if row.get("status") != "succeeded" or not isinstance(row.get("structure"), dict):
            reason = "upstream_generation:" + str(row.get("reason", row.get("status")))
        else:
            try:
                structure = Structure.from_dict(dict(row["structure"]))
                counts = Counter(int(value) for value in structure.atomic_numbers)
                elements = tuple(sorted(counts))
                amounts = [counts[element] for element in elements]
                divisor = math.gcd(*amounts)
                reduced = tuple(int(value // divisor) for value in amounts)
                comp_valid = bool(smact_validity(elements, reduced))
                struct_valid = bool(structure_validity(structure))
                if not (comp_valid and struct_valid):
                    reason = "crysllmgen_invalid"
            except Exception as exc:  # preserve attempt as invalid
                reason = f"{type(exc).__name__}:{exc}"
        attempts.append(
            {
                "ordinal": int(row["ordinal"]),
                "attempt_id": str(row["attempt_id"]),
                "comp_valid": comp_valid,
                "struct_valid": struct_valid,
                "valid": comp_valid and struct_valid,
                "reason": reason,
            }
        )
    report = {
        "schema": "crysllmgen_direct_validity_fast_v1",
        "attempts": int(args.expected_denominator),
        "generation_succeeded": sum(row.get("status") == "succeeded" for row in rows),
        "comp_valid_count": sum(row["comp_valid"] for row in attempts),
        "struct_valid_count": sum(row["struct_valid"] for row in attempts),
        "valid_count": sum(row["valid"] for row in attempts),
        "omitted_metrics": [
            "density_wasserstein",
            "num_elements_wasserstein",
            "composition_fingerprint_coverage",
            "CrystalNN_fingerprint_coverage",
        ],
        "validity_functions": "frozen_upstream_eval_utils",
        "retry_or_replacement_used": False,
    }
    if args.expected_report is not None:
        expected = json.loads(args.expected_report.read_text())
        for key in (
            "attempts",
            "generation_succeeded",
            "comp_valid_count",
            "struct_valid_count",
            "valid_count",
        ):
            if int(report[key]) != int(expected[key]):
                raise ValueError(f"fast Direct regression mismatch for {key}")
        report["expected_report_exact_count_match"] = True
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "attempt_metrics.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in attempts)
    )
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "_SUCCESS").touch()
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
