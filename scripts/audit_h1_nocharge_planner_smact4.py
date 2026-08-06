#!/usr/bin/env python3
"""Audit raw H1 Planner formulas with the frozen SMACT 4.0.0 contract.

This process is intentionally separate from the paper-comparable legacy
CrysLLMGen evaluator.  Running both evaluators in one interpreter would let a
SMACT package upgrade silently change the primary metric.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from crystal_dlm.h1_nocharge_ion_aux import (  # noqa: E402
    SMACT4_VERSION,
    canonical_json_sha256,
    load_smact4_icsd24_oxidation_map,
    smact4_validity_with_witness,
)


SCHEMA = "h1_nocharge_planner_smact4_audit_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_raw_rows(path: Path, denominator: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path} contains a non-object JSONL row")
                rows.append(value)
    ordinals = [int(row.get("sample_idx", -1)) for row in rows]
    if len(rows) != int(denominator) or ordinals != list(range(int(denominator))):
        raise ValueError(
            f"{path} must contain ordered all-attempt ordinals 0..{int(denominator) - 1}"
        )
    return rows


def audit_rows(
    rows: list[Mapping[str, Any]],
    oxidation_map: Mapping[str, list[int]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    attempts: list[dict[str, Any]] = []
    parsed_count = 0
    valid_count = 0
    parity_count = 0
    strata: dict[str, int] = {}
    for ordinal, row in enumerate(rows):
        parsed = row.get("parsed") is True
        attempt: dict[str, Any] = {
            "ordinal": ordinal,
            "parsed": parsed,
            "formula": None,
            "valid": False,
            "witness_valid": False,
            "official_witness_parity": True,
            "stratum": "planner_parse_failure",
            "charge_sum": None,
            "witness_sha256": None,
        }
        if parsed:
            plan = row.get("plan_state")
            if not isinstance(plan, Mapping):
                raise ValueError(f"parsed ordinal {ordinal} is missing plan_state")
            formula = str(plan.get("formula") or "")
            symbols = [str(value) for value in (plan.get("elements") or ())]
            counts = [int(value) for value in (plan.get("counts") or ())]
            if not formula or not symbols or len(symbols) != len(counts) or any(value <= 0 for value in counts):
                raise ValueError(f"parsed ordinal {ordinal} has an invalid composition payload")
            result = smact4_validity_with_witness(
                formula,
                symbols,
                counts,
                oxidation_map,
            )
            attempt.update(
                {
                    "formula": formula,
                    "valid": result["valid"] is True,
                    "witness_valid": result["witness_valid"] is True,
                    "official_witness_parity": result["official_witness_parity"] is True,
                    "stratum": str(result["stratum"]),
                    "charge_sum": result["charge_sum"],
                    "witness_sha256": (
                        None
                        if result["witness"] is None
                        else canonical_json_sha256(result["witness"])
                    ),
                }
            )
            parsed_count += 1
            valid_count += int(attempt["valid"])
            parity_count += int(attempt["official_witness_parity"])
        stratum = str(attempt["stratum"])
        strata[stratum] = strata.get(stratum, 0) + 1
        attempts.append(attempt)
    return attempts, {
        "parsed_count": parsed_count,
        "valid_count": valid_count,
        "official_witness_parity_count": parity_count,
        "official_witness_parity_100pct": parity_count == parsed_count,
        "stratum_counts": dict(sorted(strata.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-generations-jsonl", type=Path, required=True)
    parser.add_argument("--denominator", type=int, choices=(64, 256), required=True)
    parser.add_argument("--arm", choices=("p0", "c0", "c1"), required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    oxidation_map, contract = load_smact4_icsd24_oxidation_map()
    if contract.get("smact_version") != SMACT4_VERSION:
        raise SystemExit(
            f"exact SMACT {SMACT4_VERSION} is required, found {contract.get('smact_version')!r}"
        )
    rows = read_raw_rows(args.raw_generations_jsonl, args.denominator)
    attempts, summary = audit_rows(rows, oxidation_map)
    if not summary["official_witness_parity_100pct"]:
        raise SystemExit("SMACT4 official/deterministic-witness parity failed")

    report = {
        "schema": SCHEMA,
        "status": "pass",
        "arm": args.arm,
        "denominator": int(args.denominator),
        "raw_generations_sha256": sha256_file(args.raw_generations_jsonl),
        "smact4_contract": contract,
        "summary": summary,
        "attempts": attempts,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: report[key] for key in ("schema", "status", "arm", "denominator", "summary")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
