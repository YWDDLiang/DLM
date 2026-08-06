#!/usr/bin/env python3
"""Preserve a machine-readable stage terminal when an upstream arm failed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, choices=(64, 256), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reason", action="append", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "h1_nocharge_ion_aux_planner_gate_v1",
        "stage": int(args.stage),
        "status": "engineering_failure",
        "decision": "stop_nocharge_ion_aux_sft_route",
        "engineering_passed": False,
        "scientific_passed": False,
        "gate_passed": False,
        "failure_reasons": list(args.reason),
        "automatic_rl": False,
        "automatic_downstream": False,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
