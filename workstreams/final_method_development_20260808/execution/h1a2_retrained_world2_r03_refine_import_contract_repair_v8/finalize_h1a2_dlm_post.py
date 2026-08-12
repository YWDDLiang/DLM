#!/usr/bin/env python3
"""Convert the controlled H1-A2 B0/D1 refined payload to 256 attempts."""

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
    read_json,
    read_jsonl,
    require_source_manifest,
    sha256_file,
    validate_config,
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
        raise ValueError("refined atom count outside 1..20")
    stop = atom_offset + num_atoms
    atom_types = payload["atom_types"][0, atom_offset:stop].cpu().numpy().astype(int)
    frac_coords = payload["frac_coords"][0, atom_offset:stop].cpu().numpy().astype(float)
    lengths = payload["lengths"][0, success_index].cpu().numpy().astype(float)
    angles = payload["angles"][0, success_index].cpu().numpy().astype(float)
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
    lattice = Lattice.from_parameters(*[float(v) for v in lengths], *[float(v) for v in angles])
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
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--refinement-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    source = args.source_dir.resolve()
    require_source_manifest(source, args.source_manifest_sha256)
    config = read_json(args.config.resolve())
    validate_config(config)
    refinement = args.refinement_dir.resolve()
    attempts = read_jsonl(refinement / "refinement_attempts.jsonl")
    metrics = read_json(refinement / "refinement_metrics.json")
    if (
        len(attempts) != DENOMINATOR
        or [int(row.get("sample_idx", -1)) for row in attempts]
        != list(range(DENOMINATOR))
        or {str(row.get("factorial_arm")) for row in attempts} != {"M00"}
        or int(metrics.get("all_attempt_denominator", -1)) != DENOMINATOR
        or int(metrics.get("diff_steps", -1)) != 800
        or int(metrics.get("num_evals", -1)) != 1
        or any(
            metrics.get(key) is not False
            for key in ("retry", "replacement", "repair", "filter", "rerank")
        )
    ):
        raise ValueError("H1-A2 B0/D1 refinement contract changed")
    payload_paths = sorted(refinement.glob("dlm_refined_mp_*.pt"))
    if len(payload_paths) != 1:
        raise ValueError("expected one merged H1-A2 refined payload")
    payload = torch.load(payload_paths[0], map_location="cpu")
    success_ordinals = [int(v) for v in payload["sample_idx"].cpu().tolist()]
    expected_success = [
        int(row["sample_idx"])
        for row in attempts
        if row.get("refiner_status") == "complete"
    ]
    if success_ordinals != expected_success or len(set(success_ordinals)) != len(success_ordinals):
        raise ValueError("H1-A2 refined payload does not match attempts")

    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    successful: dict[int, dict[str, Any]] = {}
    atom_offset = 0
    for success_index, ordinal in enumerate(success_ordinals):
        successful[ordinal], atom_offset = _structure(
            payload, success_index=success_index, atom_offset=atom_offset
        )
    if atom_offset != int(payload["atom_types"].shape[1]):
        raise ValueError("refined payload contains trailing sites")

    rows: list[dict[str, Any]] = []
    for ordinal, attempt in enumerate(attempts):
        structure = successful.get(ordinal)
        succeeded = structure is not None
        reason = ""
        if not succeeded:
            reason = (
                f"{attempt.get('earliest_failure_stage') or 'unknown'}:"
                f"{attempt.get('failure_reason') or 'unknown'}"
            )
        rows.append(
            {
                "schema": "wqcodiff_generation_attempt_v1",
                "attempt_id": f"h1a2-retrained-b0-d1-post_model494-{ordinal:04d}",
                "method": "H1A2-RETRAINED-P0-B0-D1-REFINE800",
                "ordinal": ordinal,
                "sample_idx": ordinal,
                "pair_id": f"h1a2-retrained-seed17-world2:{ordinal:04d}",
                "factorial_arm": "M00",
                "planner_arm": "P0",
                "body_arm": "B0",
                "evaluation_stage": "post_model494",
                "generation_policy": "d1",
                "status": "succeeded" if succeeded else "failed",
                "reason": reason,
                "structure": structure,
                "body_noise_seed": int(attempt["body_sampling_seed"]),
                "refiner_noise_seed": int(attempt["refiner_sampling_seed"]),
                "source_plan_state_sha256": attempt.get("plan_text_sha256"),
                "diffusion_refinement_applied": succeeded,
                "diffusion_refinement_steps": 800 if succeeded else None,
                "retry_or_replacement_used": False,
            }
        )
    generation_path = output / "generation.jsonl"
    write_jsonl_exclusive(generation_path, rows)
    report = {
        "schema": "h1a2_retrained_b0_d1_post_generation_report_v1",
        "status": "complete",
        "ok": True,
        "panel": "historical_h1a2_dlm_control",
        "planner": "retrained_P0_seed17_world2_b4",
        "body": "historical_H1A2_B0",
        "generation_policy": "d1_exact_plan_schedule",
        "stage": "post_model494",
        "attempts": DENOMINATOR,
        "generation_succeeded": len(success_ordinals),
        "generation_failed": DENOMINATOR - len(success_ordinals),
        "refiner_complete": len(success_ordinals),
        "diffusion_steps": 800,
        "generation_jsonl_sha256": sha256_file(generation_path),
        "refinement_attempts_sha256": sha256_file(
            refinement / "refinement_attempts.jsonl"
        ),
        "refined_payload_sha256": sha256_file(payload_paths[0]),
        "retry_replacement_repair_filter_rerank": False,
        "source_manifest_sha256": args.source_manifest_sha256,
    }
    write_json_exclusive(output / "generation_report.json", report)
    with (output / "_SUCCESS").open("x", encoding="ascii") as handle:
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
