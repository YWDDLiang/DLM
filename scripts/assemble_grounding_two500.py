#!/usr/bin/env python3
"""Assemble two global-ordinal 500 Plan rounds into one 1000-attempt ledger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from assemble_grounding_repeat import canonical_sha256, read_jsonl, refined_structures


ATTEMPTS = 1000
ROUND_SIZE = 500


def load_refined(path: Path) -> dict[int, dict]:
    files = sorted(path.glob("dlm_refined_mp_*.pt"))
    files = [value for value in files if ".rank" not in value.name]
    if len(files) != 1:
        raise ValueError(f"expected one merged refined tensor in {path}, found {files}")
    return refined_structures(torch.load(files[0], map_location="cpu"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round1-body-dir", type=Path, required=True)
    parser.add_argument("--round1-refine-dir", type=Path, required=True)
    parser.add_argument("--round2-body-dir", type=Path, required=True)
    parser.add_argument("--round2-refine-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--arm", choices=("control", "candidate"), required=True)
    parser.add_argument("--epoch", type=int, choices=(2, 3), required=True)
    parser.add_argument("--dlm-seed", type=int, required=True)
    parser.add_argument("--refiner-seed", type=int, required=True)
    args = parser.parse_args()

    round1 = read_jsonl(args.round1_body_dir / "raw_generations.jsonl")
    round2 = read_jsonl(args.round2_body_dir / "raw_generations.jsonl")
    round1_metrics = json.loads(
        (args.round1_body_dir / "sample_metrics.json").read_text(encoding="utf-8")
    )
    round2_metrics = json.loads(
        (args.round2_body_dir / "sample_metrics.json").read_text(encoding="utf-8")
    )
    round1_by_idx = {int(row["sample_idx"]): row for row in round1}
    round2_by_idx = {int(row["sample_idx"]): row for row in round2}
    if len(round1) != ROUND_SIZE or set(round1_by_idx) != set(range(0, ROUND_SIZE)):
        raise ValueError("round1 does not cover global ordinals 0..499")
    if len(round2) != ROUND_SIZE or set(round2_by_idx) != set(range(ROUND_SIZE, ATTEMPTS)):
        raise ValueError("round2 does not cover global ordinals 500..999")
    by_idx = {**round1_by_idx, **round2_by_idx}

    refined1 = load_refined(args.round1_refine_dir)
    refined2 = load_refined(args.round2_refine_dir)
    if set(refined1) - set(range(0, ROUND_SIZE)):
        raise ValueError("round1 refined payload contains out-of-range global ordinals")
    if set(refined2) - set(range(ROUND_SIZE, ATTEMPTS)):
        raise ValueError("round2 refined payload contains out-of-range global ordinals")
    structures = {**refined1, **refined2}
    if len(structures) != len(refined1) + len(refined2):
        raise ValueError("refined rounds overlap in global ordinal")

    method = f"H1-A2-DLM-SUFFICIENT-EPOCH{args.epoch}"
    rows: list[dict] = []
    for ordinal in range(ATTEMPTS):
        source = by_idx[ordinal]
        structure = structures.get(ordinal)
        succeeded = structure is not None
        plan_state = source.get("plan_state")
        rows.append(
            {
                "schema": "wqcodiff_generation_attempt_v1",
                "attempt_id": f"h1a2-dlm-e{args.epoch}-{ordinal:04d}",
                "method": method,
                "ordinal": ordinal,
                "sample_idx": ordinal,
                "repeat": 0,
                "experiment_repeat": args.epoch,
                "pair_id": f"h1a2-dlm-sufficient:{ordinal:04d}",
                "arm": args.arm,
                "planner_arm": "raw1000-P0-frozen",
                "body_arm": f"B0-total-epoch{args.epoch}",
                "schedule_arm": "D1",
                "status": "succeeded" if succeeded else "failed",
                "reason": (
                    None
                    if succeeded
                    else f"body:{source.get('reason') or source.get('message') or 'graph_failure'}"
                ),
                "structure": structure,
                "body_noise_seed": int(args.dlm_seed) + ordinal,
                "refiner_noise_seed": int(args.refiner_seed) + ordinal,
                "source_plan_state_sha256": (
                    None if plan_state is None else canonical_sha256(plan_state)
                ),
                "plan_state": plan_state,
                "source_round": 1 if ordinal < ROUND_SIZE else 2,
                "diffusion_refinement_applied": succeeded,
                "diffusion_refinement_steps": 800 if succeeded else None,
                "new_scientific_seed_per_repeat": True,
                "retry_or_replacement_used": False,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "generation.jsonl").open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    for round_idx, bounds in ((1, (0, ROUND_SIZE)), (2, (ROUND_SIZE, ATTEMPTS))):
        with (args.output_dir / f"generation_round{round_idx}.jsonl").open(
            "x", encoding="utf-8"
        ) as handle:
            for row in rows[bounds[0] : bounds[1]]:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    report = {
        "schema": "h1a2_grounding_two500_generation_report_v1",
        "arm": args.arm,
        "epoch": args.epoch,
        "attempts": ATTEMPTS,
        "rounds": {
            "round1": {
                "global_ordinals": [0, 499],
                "parsed": int(round1_metrics["parse_success"]),
                "body_success": int(round1_metrics["graph_success"]),
                "refined": len(refined1),
            },
            "round2": {
                "global_ordinals": [500, 999],
                "parsed": int(round2_metrics["parse_success"]),
                "body_success": int(round2_metrics["graph_success"]),
                "refined": len(refined2),
            },
        },
        "parsed": int(round1_metrics["parse_success"]) + int(round2_metrics["parse_success"]),
        "body_success": int(round1_metrics["graph_success"])
        + int(round2_metrics["graph_success"]),
        "refined": len(structures),
        "dlm_seed": args.dlm_seed,
        "refiner_seed": args.refiner_seed,
        "seed_rule": "base_seed + global sample_idx",
    }
    (args.output_dir / "generation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "_SUCCESS").touch()


if __name__ == "__main__":
    main()
