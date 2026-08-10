#!/usr/bin/env python3
"""Convert one refined arm/repeat into the 1,000-row generation ledger."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from pymatgen.core import Lattice, Structure

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


def _finite(values: np.ndarray) -> bool:
    return bool(np.isfinite(np.asarray(values, dtype=float)).all())


def _structure(
    payload: Mapping[str, Any], *, success_index: int, atom_offset: int
) -> tuple[dict[str, Any], int]:
    num_atoms = int(payload["num_atoms"][0, success_index].item())
    if not 1 <= num_atoms <= 20:
        raise ValueError(f"refined atom count {num_atoms} outside 1..20")
    stop = atom_offset + num_atoms
    atom_types = (
        payload["atom_types"][0, atom_offset:stop]
        .detach()
        .cpu()
        .numpy()
        .astype(int)
    )
    frac_coords = (
        payload["frac_coords"][0, atom_offset:stop]
        .detach()
        .cpu()
        .numpy()
        .astype(float)
    )
    lengths = (
        payload["lengths"][0, success_index].detach().cpu().numpy().astype(float)
    )
    angles = (
        payload["angles"][0, success_index].detach().cpu().numpy().astype(float)
    )
    if (
        len(atom_types) != num_atoms
        or frac_coords.shape != (num_atoms, 3)
        or lengths.shape != (3,)
        or angles.shape != (3,)
        or not _finite(frac_coords)
        or not _finite(lengths)
        or not _finite(angles)
        or np.any(lengths <= 0.0)
        or np.any(angles <= 0.0)
        or np.any(angles >= 180.0)
    ):
        raise ValueError("refined tensor geometry is malformed or nonfinite")
    lattice = Lattice.from_parameters(
        float(lengths[0]),
        float(lengths[1]),
        float(lengths[2]),
        float(angles[0]),
        float(angles[1]),
        float(angles[2]),
    )
    structure = Structure(
        lattice,
        atom_types.tolist(),
        frac_coords,
        coords_are_cartesian=False,
        to_unit_cell=True,
    )
    scalars = np.concatenate(
        [structure.frac_coords.reshape(-1), structure.lattice.matrix.reshape(-1)]
    )
    if (
        structure.num_sites != num_atoms
        or not _finite(scalars)
        or not math.isfinite(float(structure.volume))
        or structure.volume < 0.1
    ):
        raise ValueError("refined structure failed serialization sanity")
    return structure.as_dict(), stop


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True)
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--refinement-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    arm = validate_arm(args.arm)
    repeat = validate_repeat(args.repeat)
    source = args.source_dir.resolve()
    require_source_manifest(source, args.source_manifest_sha256)
    config = read_json(args.config.resolve())
    validate_config(config)
    method = f"P0-{arm}-SAFEAXIS-post_model494"

    refinement = args.refinement_dir.resolve()
    attempts = ordered_rows(
        read_jsonl(refinement / "refinement_attempts.jsonl"),
        ordinal_field="ordinal",
    )
    metrics = read_json(refinement / "refinement_metrics.json")
    if (
        {str(row.get("arm")) for row in attempts} != {arm}
        or {int(row.get("repeat", -1)) for row in attempts} != {repeat}
        or int(metrics.get("all_attempt_denominator", -1)) != DENOMINATOR
        or int(metrics.get("diff_steps", -1)) != 800
        or int(metrics.get("num_evals", -1)) != 1
        or any(
            metrics.get(key) is not False
            for key in ("retry", "replacement", "repair", "filter", "rerank")
        )
    ):
        raise ValueError("refinement all-attempt contract changed")

    payload_paths = sorted(refinement.glob("dlm_refined_mp_*.pt"))
    if len(payload_paths) != 1:
        raise ValueError("expected exactly one merged refined payload")
    payload = torch.load(payload_paths[0], map_location="cpu")
    success_ordinals = [
        int(value) for value in payload["sample_idx"].detach().cpu().tolist()
    ]
    expected_success = [
        ordinal
        for ordinal, row in enumerate(attempts)
        if row.get("refiner_status") == "complete"
    ]
    if (
        success_ordinals != expected_success
        or len(set(success_ordinals)) != len(success_ordinals)
    ):
        raise ValueError("refined payload does not match successful attempts")

    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    successful: dict[int, dict[str, Any]] = {}
    atom_offset = 0
    for success_index, ordinal in enumerate(success_ordinals):
        structure, atom_offset = _structure(
            payload, success_index=success_index, atom_offset=atom_offset
        )
        successful[ordinal] = structure
    if atom_offset != int(payload["atom_types"].shape[1]):
        raise ValueError("refined atom tensor contains trailing sites")

    rows: list[dict[str, Any]] = []
    for ordinal, source_attempt in enumerate(attempts):
        structure = successful.get(ordinal)
        succeeded = structure is not None
        reason = ""
        if not succeeded:
            stage = str(source_attempt.get("earliest_failure_stage") or "unknown")
            failure = str(
                source_attempt.get("failure_reason")
                or source_attempt.get("reason")
                or "unknown"
            )
            reason = f"{stage}:{failure}"
        rows.append(
            {
                "schema": "wqcodiff_generation_attempt_v1",
                "attempt_id": attempt_id(arm, repeat, ordinal, "post_model494"),
                "method": method,
                "ordinal": ordinal,
                "sample_idx": ordinal,
                "pair_id": f"h1-plan1200-r{repeat}:{ordinal:04d}",
                "repeat": repeat,
                "arm": arm,
                "planner_arm": "P0",
                "body_arm": arm,
                "evaluation_stage": "post_model494",
                "schedule_arm": "D2_SAFE_AXIS",
                "status": "succeeded" if succeeded else "failed",
                "reason": reason,
                "structure": structure,
                "body_noise_seed": (
                    int(source_attempt["body_noise_seed"])
                    if source_attempt.get("body_noise_seed") is not None
                    else None
                ),
                "refiner_noise_seed": int(
                    source_attempt["refiner_sampling_seed"]
                ),
                "source_plan_state_sha256": source_attempt.get(
                    "plan_state_sha256"
                ),
                "diffusion_refinement_applied": succeeded,
                "diffusion_refinement_steps": 800 if succeeded else None,
                "retry_or_replacement_used": False,
            }
        )
    generation_path = output / "generation.jsonl"
    write_jsonl_exclusive(generation_path, rows)
    report = {
        "schema": "h1_plan1200_generation_report_v1",
        "status": "complete",
        "ok": True,
        "arm": arm,
        "repeat": repeat,
        "stage": "post_model494",
        "planner": "P0",
        "body": arm,
        "method": method,
        "attempts": DENOMINATOR,
        "body_succeeded": int(metrics["body_complete"]),
        "generation_succeeded": len(success_ordinals),
        "generation_failed": DENOMINATOR - len(success_ordinals),
        "refiner_complete": len(success_ordinals),
        "diffusion_refiner": config["refiner"]["name"],
        "diffusion_steps": 800,
        "all_successes_diffusion_refined": True,
        "same_frozen_seed_ledger": True,
        "generation_jsonl_sha256": sha256_file(generation_path),
        "refinement_attempts_sha256": sha256_file(
            refinement / "refinement_attempts.jsonl"
        ),
        "refined_payload_sha256": sha256_file(payload_paths[0]),
        "retry_replacement_repair_filter_rerank": False,
        "formal_promotion": False,
        "automatic_training": False,
        "automatic_downstream": False,
        "automatic_rl": False,
        "source_manifest_sha256": args.source_manifest_sha256,
    }
    write_json_exclusive(output / "generation_report.json", report)
    with (output / "_SUCCESS").open("x", encoding="ascii") as handle:
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
