#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from crystal_dlm.ordinal_rng import derive_ordinal_seed


IDENTITY = "h1_chemistry_first_sft_v2_v1"
SCHEMA = "h1_chemistry_first_planner_science_ledger_v1"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--denominator", type=int, choices=(64, 256), required=True)
    parser.add_argument("--base-seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = {
        "schema": SCHEMA,
        "identity": IDENTITY,
        "denominator": args.denominator,
        "base_seed": args.base_seed,
        "stage": f"planner{args.denominator}",
        "independent_from_training_and_other_stages": True,
        "rows": [
            {
                "ordinal": ordinal,
                "role": "shared",
                "planner_sampling_seed": derive_ordinal_seed(
                    args.base_seed,
                    sample_idx=ordinal,
                    stage="planner_sampling",
                    role="shared",
                ),
            }
            for ordinal in range(args.denominator)
        ],
    }
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
