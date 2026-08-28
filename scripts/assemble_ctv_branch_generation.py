#!/usr/bin/env python3
"""Assemble refined CTV branches into frozen 256-attempt evaluation chunks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def chunk_coordinates(global_ordinal: int, *, chunk_size: int = 256) -> tuple[int, int]:
    ordinal = int(global_ordinal)
    size = int(chunk_size)
    if ordinal < 0 or size <= 0:
        raise ValueError("CTV chunk coordinates require non-negative ordinal and size")
    return ordinal // size, ordinal % size


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-run", type=Path, required=True)
    parser.add_argument("--plans-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--expected-branches", type=int, required=True)
    parser.add_argument("--chunk-size", type=int, default=256)
    args = parser.parse_args()

    import torch
    from assemble_grounding_repeat import refined_structures

    expected = int(args.expected_branches)
    chunk_size = int(args.chunk_size)
    if expected <= 0 or expected % chunk_size:
        raise ValueError("formal CTV denominator must be a positive multiple of 256")
    branches = read_jsonl(args.rollout_run / "branch/branches.jsonl")
    branches.sort(key=lambda row: int(row["branch_ordinal"]))
    if [int(row["branch_ordinal"]) for row in branches] != list(range(expected)):
        raise ValueError("CTV branch ledger does not cover the formal denominator")
    plans = read_jsonl(args.plans_jsonl)
    plan_by_index = {index: row for index, row in enumerate(plans)}
    refined_files = [
        path
        for path in sorted((args.rollout_run / "refine").glob("dlm_refined_mp_*.pt"))
        if ".rank" not in path.name
    ]
    if len(refined_files) != 1:
        raise ValueError(f"expected one merged CTV refined tensor: {refined_files}")
    structures = refined_structures(torch.load(refined_files[0], map_location="cpu"))
    if set(structures) != set(range(expected)):
        raise ValueError("CTV refined structures do not cover every global branch ordinal")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    metadata_rows: list[dict[str, Any]] = []
    chunk_rows: dict[int, list[dict[str, Any]]] = {
        index: [] for index in range(expected // chunk_size)
    }
    for branch in branches:
        global_ordinal = int(branch["branch_ordinal"])
        chunk_index, local_ordinal = chunk_coordinates(
            global_ordinal, chunk_size=chunk_size
        )
        plan_index = int(branch["canary_plan_idx"])
        if plan_index not in plan_by_index:
            raise ValueError("CTV branch references an unavailable frozen Plan")
        plan_state = plan_by_index[plan_index].get("plan_state")
        if not isinstance(plan_state, dict):
            raise ValueError("CTV frozen Plan lacks plan_state")
        structure = structures[global_ordinal]
        attempt_id = f"ctv-{args.split}-{global_ordinal:05d}"
        row = {
            "schema": "wqcodiff_generation_attempt_v1",
            "attempt_id": attempt_id,
            "method": f"H1-A2-CTV-BRANCH-{args.split.upper()}",
            "ordinal": local_ordinal,
            "sample_idx": local_ordinal,
            "global_branch_ordinal": global_ordinal,
            "chunk_index": chunk_index,
            "repeat": 0,
            "experiment_repeat": chunk_index,
            "pair_id": f"{branch['state_id']}:{int(branch['continuation_seed'])}",
            "arm": "candidate",
            "planner_arm": "C3FD-v2.5-frozen",
            "body_arm": "CTV-formal-branch",
            "schedule_arm": "exact-axis",
            "status": "succeeded",
            "reason": None,
            "structure": structure,
            "body_noise_seed": int(branch["noise_group"]),
            "refiner_noise_seed": int(branch["noise_group"]) & ((1 << 31) - 1),
            "source_plan_state_sha256": canonical_sha256(plan_state),
            "plan_state": plan_state,
            "ctv_state_id": str(branch["state_id"]),
            "ctv_action_token": int(branch["action_token"]),
            "ctv_action_rank": int(branch["action_rank"]),
            "ctv_continuation_seed": int(branch["continuation_seed"]),
            "ctv_milestone": float(branch["milestone"]),
            "diffusion_refinement_applied": True,
            "diffusion_refinement_steps": 800,
            "new_scientific_seed_per_repeat": True,
            "retry_or_replacement_used": False,
        }
        chunk_rows[chunk_index].append(row)
        metadata_rows.append(
            {
                "global_branch_ordinal": global_ordinal,
                "chunk_index": chunk_index,
                "local_ordinal": local_ordinal,
                "attempt_id": attempt_id,
                "state_id": str(branch["state_id"]),
                "action_token": int(branch["action_token"]),
                "continuation_seed": int(branch["continuation_seed"]),
            }
        )

    chunk_reports = []
    for chunk_index, rows in chunk_rows.items():
        rows.sort(key=lambda row: int(row["ordinal"]))
        if [int(row["ordinal"]) for row in rows] != list(range(chunk_size)):
            raise ValueError(f"CTV chunk {chunk_index} is incomplete")
        chunk_dir = output / f"chunk{chunk_index:02d}"
        chunk_dir.mkdir()
        generation_path = chunk_dir / "generation.jsonl"
        write_jsonl(generation_path, rows)
        chunk_reports.append(
            {
                "chunk_index": chunk_index,
                "attempts": len(rows),
                "generation_sha256": hashlib.sha256(
                    generation_path.read_bytes()
                ).hexdigest(),
            }
        )
    write_jsonl(output / "branch_metadata.jsonl", metadata_rows)
    report = {
        "schema": "h1a2_ctv_branch_generation_chunks_v1",
        "split": args.split,
        "expected_branches": expected,
        "chunk_size": chunk_size,
        "chunks": len(chunk_rows),
        "chunk_reports": chunk_reports,
        "rollout_run": str(args.rollout_run.resolve()),
        "plans_jsonl": str(args.plans_jsonl.resolve()),
        "common_refiner_noise": True,
        "outcome_labels_used": False,
    }
    (output / "CTV_BRANCH_GENERATION_MANIFEST.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "_SUCCESS").touch()
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
