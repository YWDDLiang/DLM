#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, choices=(64, 256), required=True)
    parser.add_argument("--candidate", choices=("sft_v2", "sft_v2_c"), required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = {
        "schema": "h1_chemistry_first_planner_gate_v1",
        "stage": args.stage,
        "candidate_id": args.candidate,
        "status": "engineering_failure",
        "decision": f"stop_{args.candidate}_engineering",
        "gate_passed": False,
        "reason": args.reason,
        "automatic_downstream": False,
        "automatic_rl": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
