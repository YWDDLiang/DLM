#!/usr/bin/env python3
"""Convert strict refined tensors into a 256-row all-attempt generation ledger."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[3]
for location in (PROJECT_ROOT, HERE):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from pymatgen.core import Lattice, Structure  # noqa: E402

from protocol import (  # noqa: E402
    attempt_id,
    read_json,
    read_jsonl,
    require_hex_sha,
    require_runtime_manifest,
    require_source_manifest,
    sha256_file,
    validate_arm,
    write_json_exclusive,
    write_jsonl_exclusive,
)


DENOMINATOR = 256


def _finite(values: np.ndarray) -> bool:
    return bool(np.isfinite(np.asarray(values, dtype=float)).all())


def _structure(
    payload: Mapping[str, Any],
    *,
    success_index: int,
    atom_offset: int,
) -> tuple[dict[str, Any], int]:
    num_atoms = int(payload["num_atoms"][0, success_index].item())
    if not 1 <= num_atoms <= 20:
        raise ValueError(f"refined atom count {num_atoms} outside 1..20")
    stop = atom_offset + num_atoms
    atom_types = (
        payload["atom_types"][0, atom_offset:stop].detach().cpu().numpy().astype(int)
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
        raise ValueError("refined structure failed the serialization sanity gate")
    return structure.as_dict(), stop


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--refinement-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execution-manifest-sha256", required=True)
    args = parser.parse_args()

    arm = validate_arm(args.arm)
    execution_sha = require_hex_sha(
        args.execution_manifest_sha256,
        "execution source manifest",
    )
    source = args.source_dir.resolve()
    require_source_manifest(source, execution_sha)
    require_runtime_manifest(args.project_root.resolve(), source)
    config = read_json(args.config.resolve())
    if (
        config.get("status") != "user_authorized_diagnostic_execution"
        or config["decision_firewall"].get("formal_g3") is not False
        or config["decision_firewall"].get("automatic_downstream") is not False
    ):
        raise ValueError("diagnostic decision firewall changed")

    refinement = args.refinement_dir.resolve()
    attempts = read_jsonl(refinement / "refinement_attempts.jsonl")
    metrics = read_json(refinement / "refinement_metrics.json")
    if (
        len(attempts) != DENOMINATOR
        or [int(row.get("sample_idx", -1)) for row in attempts]
        != list(range(DENOMINATOR))
        or {str(row.get("factorial_arm")) for row in attempts} != {arm}
        or int(metrics.get("all_attempt_denominator", -1)) != DENOMINATOR
        or int(metrics.get("diff_steps", -1)) != 800
        or int(metrics.get("num_evals", -1)) != 1
        or metrics.get("retry") is not False
        or metrics.get("replacement") is not False
        or metrics.get("repair") is not False
        or metrics.get("filter") is not False
        or metrics.get("rerank") is not False
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
        int(row["sample_idx"])
        for row in attempts
        if row.get("refiner_status") == "complete"
    ]
    if success_ordinals != expected_success or len(set(success_ordinals)) != len(
        success_ordinals
    ):
        raise ValueError("refined payload does not match successful attempts")

    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    method = str(config["arms"][arm]["method"])
    successful_by_ordinal: dict[int, dict[str, Any]] = {}
    atom_offset = 0
    for success_index, sample_idx in enumerate(success_ordinals):
        structure, atom_offset = _structure(
            payload,
            success_index=success_index,
            atom_offset=atom_offset,
        )
        successful_by_ordinal[sample_idx] = structure
    if atom_offset != int(payload["atom_types"].shape[1]):
        raise ValueError("refined atom tensor contains trailing sites")

    rows: list[dict[str, Any]] = []
    for sample_idx, attempt in enumerate(attempts):
        structure = successful_by_ordinal.get(sample_idx)
        succeeded = structure is not None
        reason = ""
        if not succeeded:
            stage = str(attempt.get("earliest_failure_stage") or "unknown")
            failure = str(attempt.get("failure_reason") or "unknown")
            reason = f"{stage}:{failure}"
        rows.append(
            {
                "schema": "wqcodiff_generation_attempt_v1",
                "attempt_id": attempt_id(arm, sample_idx),
                "method": method,
                "ordinal": sample_idx,
                "sample_idx": sample_idx,
                "pair_id": f"h1a2-v3-poststop-sun256:{sample_idx:04d}",
                "factorial_arm": arm,
                "planner_arm": config["arms"][arm]["planner"],
                "body_arm": config["arms"][arm]["body"],
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
        "schema": "h1a2_v3_poststop_sun256_generation_report_v1",
        "status": "complete",
        "ok": True,
        "arm": arm,
        "method": method,
        "attempts": DENOMINATOR,
        "generation_succeeded": sum(
            row["status"] == "succeeded" for row in rows
        ),
        "generation_failed": sum(row["status"] == "failed" for row in rows),
        "refiner_complete": len(success_ordinals),
        "diffusion_refiner": config["refiner"]["name"],
        "diffusion_steps": 800,
        "all_successes_diffusion_refined": all(
            row["diffusion_refinement_applied"]
            for row in rows
            if row["status"] == "succeeded"
        ),
        "generation_jsonl_sha256": sha256_file(generation_path),
        "refinement_attempts_sha256": sha256_file(
            refinement / "refinement_attempts.jsonl"
        ),
        "refined_payload_sha256": sha256_file(payload_paths[0]),
        "retry_replacement_repair_filter_rerank": False,
        "formal_g3": False,
        "automatic_promotion": False,
        "automatic_downstream": False,
        "execution_manifest_sha256": execution_sha,
    }
    write_json_exclusive(output / "generation_report.json", report)
    with (output / "_SUCCESS").open("x", encoding="ascii") as handle:
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
