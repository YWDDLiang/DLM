#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, choices=(64, 256), required=True)
    parser.add_argument("--candidate", choices=("sft_v2", "sft_v2_c"), required=True)
    parser.add_argument("--training-terminal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.training_terminal.read_text(encoding="utf-8"))
    if (
        source.get("candidate") != args.candidate
        or source.get("status") != "anchor_nll_noninferiority_failure"
        or (source.get("conditional_structural_anchor_nll_gate") or {}).get("passed")
        is not False
    ):
        raise SystemExit("training terminal is not the registered anchor-NLL stop")
    payload = {
        "schema": "h1_chemistry_first_planner_gate_v1",
        "stage": args.stage,
        "candidate_id": args.candidate,
        "status": "scientific_stop",
        "decision": f"stop_{args.candidate}_anchor_nll_no_rl",
        "gate_passed": False,
        "reason": "conditional_structural_anchor_nll_degradation_gt_1pct",
        "training_terminal_sha256": sha256_file(args.training_terminal),
        "conditional_structural_anchor_nll_gate": source[
            "conditional_structural_anchor_nll_gate"
        ],
        "raw_generation_performed": False,
        "automatic_downstream": False,
        "automatic_rl": False,
    }
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
