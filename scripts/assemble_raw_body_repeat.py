#!/usr/bin/env python3
"""Assemble unrefined DLM body CIFs into the fixed-attempt evaluator schema."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def canonical_sha256(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--body-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--dlm-seed", type=int, required=True)
    parser.add_argument("--source-arm", choices=("full_axis", "hard_axis"), required=True)
    parser.add_argument("--arm-label", choices=("control", "candidate"), required=True)
    parser.add_argument("--denominator", type=int, default=256)
    parser.add_argument("--method-label", default="H1-A2-DLM-RAW-BODY-NO-MODEL494")
    parser.add_argument("--planner-arm-label", default="raw-P0-frozen")
    parser.add_argument("--body-arm-label", default=None)
    parser.add_argument("--schedule-arm-label", default="D1-exact-axis")
    args = parser.parse_args()

    from pymatgen.core import Structure  # noqa: PLC0415

    raw_rows = read_jsonl(args.body_dir / "raw_generations.jsonl")
    by_idx = {int(row["sample_idx"]): row for row in raw_rows}
    expected = set(range(args.denominator))
    if len(raw_rows) != args.denominator or set(by_idx) != expected:
        raise ValueError("body rows do not cover the requested ordinal denominator")

    rows: list[dict] = []
    reconstructed = 0
    for ordinal in range(args.denominator):
        source = by_idx[ordinal]
        structure = None
        failure = None
        if source.get("parsed") is True and source.get("cif"):
            try:
                structure = Structure.from_str(str(source["cif"]), fmt="cif").as_dict()
            except Exception as exc:  # noqa: BLE001
                failure = f"raw_cif:{type(exc).__name__}:{exc}"
        else:
            failure = f"body:{source.get('reason') or source.get('message') or 'graph_failure'}"
        succeeded = structure is not None
        reconstructed += int(succeeded)
        plan_state = source.get("plan_state")
        rows.append(
            {
                "schema": "wqcodiff_generation_attempt_v1",
                "attempt_id": f"h1a2-raw-{args.source_arm}-s{args.seed}-{ordinal:04d}",
                "method": str(args.method_label),
                "ordinal": ordinal,
                "sample_idx": ordinal,
                "repeat": 0,
                "experiment_repeat": args.seed * 10 + 9,
                "pair_id": f"h1a2-refiner-{args.source_arm}-s{args.seed}:{ordinal:04d}",
                "arm": args.arm_label,
                "planner_arm": str(args.planner_arm_label),
                "body_arm": str(
                    args.body_arm_label
                    or f"public-H1-A2-DLM-raw-{args.source_arm}"
                ),
                "schedule_arm": str(args.schedule_arm_label),
                "status": "succeeded" if succeeded else "failed",
                "reason": failure,
                "structure": structure,
                "body_noise_seed": int(args.dlm_seed) + ordinal,
                "refiner_noise_seed": None,
                "source_plan_state_sha256": (
                    None if plan_state is None else canonical_sha256(plan_state)
                ),
                "plan_state": plan_state,
                "diffusion_refinement_applied": False,
                "diffusion_refinement_steps": 0,
                "new_scientific_seed_per_repeat": True,
                "retry_or_replacement_used": False,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "generation.jsonl").open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    report = {
        "schema": "h1a2_raw_body_generation_report_v1",
        "seed": args.seed,
        "source_arm": args.source_arm,
        "attempts": args.denominator,
        "body_success": sum(row.get("parsed") is True for row in raw_rows),
        "reconstructed": reconstructed,
        "dlm_seed": args.dlm_seed,
        "refiner": None,
        "diagnostic_only": True,
    }
    (args.output_dir / "generation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "_SUCCESS").touch()


if __name__ == "__main__":
    main()
