#!/usr/bin/env python3
"""Run the exact frozen SMACT4 audit for Evidence-First Planner arms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from crystal_dlm.h1_nocharge_ion_aux import (  # noqa: E402
    SMACT4_VERSION,
    load_smact4_icsd24_oxidation_map,
)
from scripts.audit_h1_nocharge_planner_smact4 import (  # noqa: E402
    SCHEMA,
    audit_rows,
    read_raw_rows,
    sha256_file,
)


ALLOWED_ARMS = {"p0", "sft_v2", "sft_v2_c"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-generations-jsonl", type=Path, required=True)
    parser.add_argument("--denominator", type=int, choices=(64, 256), required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if str(args.arm) not in ALLOWED_ARMS:
        raise SystemExit(f"unsupported chemistry-first Planner arm {args.arm!r}")

    oxidation_map, contract = load_smact4_icsd24_oxidation_map()
    if contract.get("smact_version") != SMACT4_VERSION:
        raise SystemExit(
            f"exact SMACT {SMACT4_VERSION} is required, found {contract.get('smact_version')!r}"
        )
    rows = read_raw_rows(args.raw_generations_jsonl, int(args.denominator))
    attempts, summary = audit_rows(rows, oxidation_map)
    if not summary["official_witness_parity_100pct"]:
        raise SystemExit("SMACT4 official/deterministic-witness parity failed")
    report = {
        "schema": SCHEMA,
        "status": "pass",
        "arm": str(args.arm),
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
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("schema", "status", "arm", "denominator", "summary")
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
