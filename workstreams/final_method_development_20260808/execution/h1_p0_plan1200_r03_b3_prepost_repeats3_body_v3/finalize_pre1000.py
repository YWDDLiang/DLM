#!/usr/bin/env python3
"""Serialize pre-model_494 body proposals as a 1,000-attempt generation ledger."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

from crystal_dlm.fixed_slot import arrays_to_structure
from protocol import (
    DENOMINATOR,
    attempt_id,
    ordered_rows,
    read_json,
    read_jsonl,
    require_source_manifest,
    sha256_file,
    validate_arm,
    validate_config,
    validate_repeat,
    write_json_exclusive,
    write_jsonl_exclusive,
)


def _serialize(arrays: object) -> dict:
    if not isinstance(arrays, dict):
        raise TypeError("body arrays are missing")
    structure = arrays_to_structure(arrays)
    scalars = np.concatenate(
        [structure.frac_coords.reshape(-1), structure.lattice.matrix.reshape(-1)]
    )
    if (
        not 1 <= structure.num_sites <= 20
        or not np.isfinite(scalars).all()
        or not math.isfinite(float(structure.volume))
        or structure.volume < 0.1
    ):
        raise ValueError("pre-model_494 structure failed serialization sanity")
    return structure.as_dict()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True)
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--body-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    arm = validate_arm(args.arm)
    repeat = validate_repeat(args.repeat)
    source = args.source_dir.resolve()
    require_source_manifest(source, args.source_manifest_sha256)
    config = read_json(args.config.resolve())
    validate_config(config)
    body_dir = args.body_dir.resolve()
    attempts = ordered_rows(
        read_jsonl(body_dir / "body_attempts.jsonl"), ordinal_field="ordinal"
    )
    body_report = read_json(body_dir / "generation_report.json")
    if (
        body_report.get("status") != "complete"
        or body_report.get("arm") != arm
        or int(body_report.get("repeat", -1)) != repeat
        or int(body_report.get("attempts", -1)) != DENOMINATOR
    ):
        raise ValueError("body report contract changed")

    rows: list[dict] = []
    serialize_failures = 0
    for ordinal, body in enumerate(attempts):
        structure = None
        reason = str(body.get("reason") or "")
        if body.get("status") == "succeeded":
            try:
                structure = _serialize(body.get("arrays"))
            except Exception as exc:  # noqa: BLE001 - retained per attempt.
                serialize_failures += 1
                reason = f"pre_serialize:{type(exc).__name__}:{exc}"
        succeeded = structure is not None
        rows.append(
            {
                "schema": "wqcodiff_generation_attempt_v1",
                "attempt_id": attempt_id(arm, repeat, ordinal, "pre_model494"),
                "method": f"P0-{arm}-SAFEAXIS-pre_model494",
                "ordinal": ordinal,
                "sample_idx": ordinal,
                "pair_id": f"h1-plan1200-r{repeat}:{ordinal:04d}",
                "repeat": repeat,
                "arm": arm,
                "planner_arm": "P0",
                "body_arm": arm,
                "evaluation_stage": "pre_model494",
                "schedule_arm": "D2_SAFE_AXIS",
                "status": "succeeded" if succeeded else "failed",
                "reason": "" if succeeded else (reason or "body_failed"),
                "structure": structure,
                "body_noise_seed": int(body["body_noise_seed"]),
                "refiner_noise_seed": int(body["refiner_noise_seed"]),
                "source_plan_state_sha256": body.get("plan_state_sha256"),
                "diffusion_refinement_applied": False,
                "diffusion_refinement_steps": None,
                "retry_or_replacement_used": False,
            }
        )

    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    generation = output / "generation.jsonl"
    write_jsonl_exclusive(generation, rows)
    succeeded = sum(row["status"] == "succeeded" for row in rows)
    report = {
        "schema": "h1_plan1200_generation_report_v1",
        "status": "complete",
        "ok": True,
        "arm": arm,
        "repeat": repeat,
        "stage": "pre_model494",
        "planner": "P0",
        "body": arm,
        "method": f"P0-{arm}-SAFEAXIS-pre_model494",
        "attempts": DENOMINATOR,
        "body_succeeded": int(body_report["succeeded"]),
        "generation_succeeded": succeeded,
        "generation_failed": DENOMINATOR - succeeded,
        "pre_serialize_failures": serialize_failures,
        "diffusion_refiner": None,
        "diffusion_steps": 0,
        "all_successes_diffusion_refined": False,
        "generation_jsonl_sha256": sha256_file(generation),
        "body_attempts_sha256": sha256_file(body_dir / "body_attempts.jsonl"),
        "retry_replacement_repair_filter_rerank": False,
        "source_manifest_sha256": args.source_manifest_sha256,
        "automatic_training": False,
        "automatic_promotion": False,
        "automatic_rl": False,
    }
    write_json_exclusive(output / "generation_report.json", report)
    with (output / "_SUCCESS").open("x", encoding="ascii") as handle:
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
