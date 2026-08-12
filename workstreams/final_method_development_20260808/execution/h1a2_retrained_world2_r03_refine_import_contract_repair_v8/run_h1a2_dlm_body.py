#!/usr/bin/env python3
"""Run the frozen H1-A2 B0/D1 body with the registered recovery seed ledger."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

from protocol import DENOMINATOR, read_jsonl, require_file


def main() -> None:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--base-script", type=Path, required=True)
    parser.add_argument("--base-script-sha256", required=True)
    parser.add_argument("--planner-attempts", type=Path, required=True)
    args, forwarded = parser.parse_known_args()
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    if not forwarded:
        raise ValueError("frozen H1-A2 body arguments are required")
    base = require_file(
        args.base_script, args.base_script_sha256, "frozen H1-A2 body script"
    )
    attempts = read_jsonl(args.planner_attempts.resolve())
    if (
        len(attempts) != DENOMINATOR
        or [int(row.get("sample_idx", -1)) for row in attempts]
        != list(range(DENOMINATOR))
    ):
        raise ValueError("H1-A2 planner-attempt coverage changed")
    seed_rows: dict[int, dict[str, int]] = {}
    for ordinal, row in enumerate(attempts):
        seed_rows[ordinal] = {
            "planner_sampling_seed": int(row["planner_sampling_seed"]),
            "body_sampling_seed": int(row["registered_body_sampling_seed"]),
            "refiner_sampling_seed": int(row["registered_refiner_sampling_seed"]),
        }

    if str(base.parent.parent) not in sys.path:
        sys.path.insert(0, str(base.parent.parent))
    spec = importlib.util.spec_from_file_location("_frozen_h1a2_body", base)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen H1-A2 body script")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def build_factorial_ordinal_record(
        base_seed: int, *, sample_idx: int
    ) -> dict[str, Any]:
        if int(base_seed) != 17:
            raise ValueError("H1-A2 controlled body base seed changed")
        ordinal = int(sample_idx)
        if ordinal not in seed_rows:
            raise ValueError("H1-A2 body ordinal outside registered denominator")
        return {
            "schema": "h1a2_factorial_contract_v1",
            "sample_idx": ordinal,
            **seed_rows[ordinal],
            "evaluation_order": ordinal,
        }

    module.build_factorial_ordinal_record = build_factorial_ordinal_record
    if module.build_factorial_ordinal_record is not build_factorial_ordinal_record:
        raise RuntimeError("H1-A2 ordinal-ledger override did not bind")
    sys.argv = [str(base), *forwarded]
    module.main()


if __name__ == "__main__":
    main()
