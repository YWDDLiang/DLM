#!/usr/bin/env python3
"""Assemble one fixed-256 grounding repeat into the evaluator generation schema."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from pymatgen.core import Lattice, Structure


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def canonical_sha256(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def refined_structures(payload: dict) -> dict[int, dict]:
    required = ("frac_coords", "num_atoms", "atom_types", "lengths", "angles", "sample_indices")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"refined payload misses {missing}")
    sample_indices = torch.as_tensor(payload["sample_indices"]).view(-1).tolist()
    num_atoms = torch.as_tensor(payload["num_atoms"])
    atom_types = torch.as_tensor(payload["atom_types"])
    frac_coords = torch.as_tensor(payload["frac_coords"])
    lengths = torch.as_tensor(payload["lengths"])
    angles = torch.as_tensor(payload["angles"])
    if num_atoms.ndim != 2 or int(num_atoms.shape[0]) != 1:
        raise ValueError("expected one refinement evaluation")
    if len(sample_indices) != int(num_atoms.shape[1]) or len(set(sample_indices)) != len(sample_indices):
        raise ValueError("sample_indices do not uniquely cover refined proposals")

    structures: dict[int, dict] = {}
    atom_offset = 0
    for proposal_idx, sample_idx in enumerate(sample_indices):
        n_atom = int(num_atoms[0, proposal_idx].item())
        atoms = atom_types[0, atom_offset : atom_offset + n_atom].detach().cpu().numpy()
        coords = frac_coords[0, atom_offset : atom_offset + n_atom].detach().cpu().numpy()
        lens = lengths[0, proposal_idx].detach().cpu().numpy()
        angs = angles[0, proposal_idx].detach().cpu().numpy()
        atom_offset += n_atom
        if not np.isfinite(np.concatenate((coords.reshape(-1), lens, angs))).all():
            raise ValueError(f"nonfinite refined geometry for sample {sample_idx}")
        lattice = Lattice.from_parameters(*[float(value) for value in lens], *[float(value) for value in angs])
        structure = Structure(lattice, atoms.tolist(), coords, coords_are_cartesian=False, to_unit_cell=True)
        if structure.num_sites != n_atom or not math.isfinite(float(structure.volume)) or structure.volume < 0.1:
            raise ValueError(f"invalid refined structure for sample {sample_idx}")
        structures[int(sample_idx)] = structure.as_dict()
    if atom_offset != int(atom_types.shape[1]):
        raise ValueError("refined atom tensor contains trailing sites")
    return structures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--body-dir", type=Path, required=True)
    parser.add_argument("--refine-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--arm", choices=("control", "candidate"), required=True)
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument("--dlm-seed", type=int, required=True)
    parser.add_argument("--refiner-seed", type=int, required=True)
    parser.add_argument("--denominator", type=int, default=256)
    args = parser.parse_args()

    raw_rows = read_jsonl(args.body_dir / "raw_generations.jsonl")
    by_idx = {int(row["sample_idx"]): row for row in raw_rows}
    if len(raw_rows) != args.denominator or set(by_idx) != set(range(args.denominator)):
        raise ValueError("body rows do not cover the requested ordinal denominator")
    refined_files = sorted(args.refine_dir.glob("dlm_refined_mp_*.pt"))
    refined_files = [path for path in refined_files if ".rank" not in path.name]
    if len(refined_files) != 1:
        raise ValueError(f"expected one merged refined tensor, found {refined_files}")
    structures = refined_structures(torch.load(refined_files[0], map_location="cpu"))

    method = "H1-A2-DLM-CE-CONTROL" if args.arm == "control" else "H1-A2-DLM-COUNTERFACTUAL-GROUNDING"
    rows: list[dict] = []
    for ordinal in range(args.denominator):
        source = by_idx[ordinal]
        structure = structures.get(ordinal)
        succeeded = structure is not None
        plan_state = source.get("plan_state")
        failure = None if succeeded else f"body:{source.get('reason') or source.get('message') or 'graph_failure'}"
        rows.append(
            {
                "schema": "wqcodiff_generation_attempt_v1",
                "attempt_id": f"h1a2-ground-r{args.repeat}-{args.arm}-{ordinal:04d}",
                "method": method,
                "ordinal": ordinal,
                "sample_idx": ordinal,
                "repeat": 0,
                "experiment_repeat": args.repeat,
                "pair_id": f"h1a2-ground-r{args.repeat}:{ordinal:04d}",
                "arm": args.arm,
                "planner_arm": "P0-frozen",
                "body_arm": "B0" if args.arm == "control" else "B0-counterfactual-grounding",
                "schedule_arm": "D1",
                "status": "succeeded" if succeeded else "failed",
                "reason": failure,
                "structure": structure,
                "body_noise_seed": int(args.dlm_seed) + ordinal,
                "refiner_noise_seed": int(args.refiner_seed) + ordinal,
                "source_plan_state_sha256": None if plan_state is None else canonical_sha256(plan_state),
                "plan_state": plan_state,
                "diffusion_refinement_applied": succeeded,
                "diffusion_refinement_steps": 800 if succeeded else None,
                "new_scientific_seed_per_repeat": True,
                "retry_or_replacement_used": False,
            }
        )

    if set(structures) - set(range(args.denominator)):
        raise ValueError("refined payload contains out-of-range sample indices")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "generation.jsonl").open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    report = {
        "schema": "h1a2_grounding_repeat_generation_report_v1",
        "arm": args.arm,
        "repeat": args.repeat,
        "attempts": args.denominator,
        "body_success": sum(row.get("parsed") is True for row in raw_rows),
        "refined": len(structures),
        "reconstructed": len(structures),
        "dlm_seed": args.dlm_seed,
        "refiner_seed": args.refiner_seed,
        "seed_rule": "base_seed + sample_idx",
    }
    (args.output_dir / "generation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "_SUCCESS").touch()


if __name__ == "__main__":
    main()
