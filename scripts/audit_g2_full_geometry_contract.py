#!/usr/bin/env python3
"""Audit bounded PBC and species-margin assumptions before G2-full training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import torch

from crystal_dlm.dynamic_crystal import parse_dynamic_answer
from crystal_dlm.fixed_slot import SYMBOL_TO_Z
from crystal_dlm.periodic_geometry_ops import (
    ELEMENT_RADII_SHA256,
    element_radius,
    minimum_image_distances_125,
    minimum_image_distances_27,
)
from crystal_dlm.periodic_geometry_objective import lattice_matrix_from_parameters


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def shell_distances(
    deltas: torch.Tensor,
    lattice: torch.Tensor,
    radius: int,
) -> torch.Tensor:
    values = torch.arange(-radius, radius + 1, dtype=deltas.dtype)
    shifts = torch.cartesian_prod(values, values, values).reshape(-1, 3)
    candidates = deltas.unsqueeze(1) + shifts.unsqueeze(0)
    return torch.linalg.vector_norm(candidates @ lattice, dim=-1).min(dim=-1).values


def arrays_from_generation(row: dict[str, Any]) -> dict[str, Any] | None:
    if row.get("parsed") is not True or not row.get("cif"):
        return None
    from pymatgen.core import Structure

    structure = Structure.from_str(str(row["cif"]), fmt="cif")
    return {
        "lengths": list(structure.lattice.abc),
        "angles": list(structure.lattice.angles),
        "species": [site.specie.symbol for site in structure],
        "frac_coords": structure.frac_coords.tolist(),
    }


def audit_arrays(
    arrays: dict[str, Any],
    *,
    margin_scale: float,
    margin_floor: float,
    margin_ceiling: float,
) -> dict[str, Any]:
    coordinates = torch.tensor(arrays["frac_coords"], dtype=torch.float64)
    lattice = lattice_matrix_from_parameters(
        torch.tensor(arrays["lengths"], dtype=torch.float64),
        torch.tensor(arrays["angles"], dtype=torch.float64),
    )
    if coordinates.shape[0] < 2:
        return {
            "pairs": 0,
            "max_27_vs_125": 0.0,
            "max_125_vs_343": 0.0,
            "max_125_vs_pymatgen": 0.0,
            "mismatch_27_pairs": 0,
            "margin_violations": 0,
        }
    pairs = torch.triu_indices(coordinates.shape[0], coordinates.shape[0], offset=1)
    deltas = coordinates.index_select(0, pairs[0]) - coordinates.index_select(0, pairs[1])
    distance27 = minimum_image_distances_27(deltas, lattice)
    distance125 = minimum_image_distances_125(deltas, lattice)
    distance343 = shell_distances(deltas, lattice, radius=3)

    from pymatgen.core import Lattice

    pmg = torch.tensor(
        Lattice(lattice.detach().cpu().numpy()).get_all_distances(
            coordinates.detach().cpu().numpy(), coordinates.detach().cpu().numpy()
        ),
        dtype=torch.float64,
    )[pairs[0], pairs[1]]
    species = [str(value) for value in arrays["species"]]
    radii = torch.tensor(
        [element_radius(SYMBOL_TO_Z[symbol]) for symbol in species],
        dtype=torch.float64,
    )
    margins = (
        float(margin_scale)
        * (radii.index_select(0, pairs[0]) + radii.index_select(0, pairs[1]))
    ).clamp(min=float(margin_floor), max=float(margin_ceiling))
    return {
        "pairs": int(pairs.shape[1]),
        "max_27_vs_125": float((distance27 - distance125).abs().max()),
        "max_125_vs_343": float((distance125 - distance343).abs().max()),
        "max_125_vs_pymatgen": float((distance125 - pmg).abs().max()),
        "mismatch_27_pairs": int(((distance27 - distance125).abs() > 1.0e-7).sum()),
        "margin_violations": int((distance125 < margins).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--generation-jsonl", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--margin-scale", type=float, default=0.55)
    parser.add_argument("--margin-floor", type=float, default=0.60)
    parser.add_argument("--margin-ceiling", type=float, default=1.40)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    totals = {
        "structures": 0,
        "pairs": 0,
        "margin_violations": 0,
        "mismatch_27_pairs": 0,
        "max_27_vs_125": 0.0,
        "max_125_vs_343": 0.0,
        "max_125_vs_pymatgen": 0.0,
    }

    def consume(arrays: dict[str, Any]) -> None:
        result = audit_arrays(
            arrays,
            margin_scale=args.margin_scale,
            margin_floor=args.margin_floor,
            margin_ceiling=args.margin_ceiling,
        )
        totals["structures"] += 1
        totals["pairs"] += int(result["pairs"])
        totals["margin_violations"] += int(result["margin_violations"])
        totals["mismatch_27_pairs"] += int(result["mismatch_27_pairs"])
        totals["max_27_vs_125"] = max(
            totals["max_27_vs_125"], float(result["max_27_vs_125"])
        )
        totals["max_125_vs_343"] = max(
            totals["max_125_vs_343"], float(result["max_125_vs_343"])
        )
        totals["max_125_vs_pymatgen"] = max(
            totals["max_125_vs_pymatgen"], float(result["max_125_vs_pymatgen"])
        )

    train_rows = 0
    for row in iter_jsonl(args.train_jsonl):
        if args.max_train_rows > 0 and train_rows >= args.max_train_rows:
            break
        consume(parse_dynamic_answer(str(row["answer"]), strict=True))
        train_rows += 1
    generation_rows = 0
    generation_failures = 0
    for path in args.generation_jsonl:
        for row in iter_jsonl(path):
            arrays = arrays_from_generation(row)
            if arrays is None:
                generation_failures += 1
                continue
            consume(arrays)
            generation_rows += 1

    tolerance = 1.0e-7
    margin_rate = (
        totals["margin_violations"] / totals["pairs"] if totals["pairs"] else 0.0
    )
    gates = {
        "all_train_rows_audited": args.max_train_rows <= 0,
        "125_matches_343": totals["max_125_vs_343"] <= tolerance,
        "125_matches_pymatgen": totals["max_125_vs_pymatgen"] <= tolerance,
        "target_margin_violation_rate_le_0p5pct": margin_rate <= 0.005,
    }
    report = {
        "schema": "g2_full_geometry_contract_audit_v2",
        "train_rows": train_rows,
        "generation_rows": generation_rows,
        "generation_failures_preserved": generation_failures,
        "totals": totals,
        "target_margin_violation_rate": margin_rate,
        "margin": {
            "scale": args.margin_scale,
            "floor_A": args.margin_floor,
            "ceiling_A": args.margin_ceiling,
        },
        "element_radii_sha256": ELEMENT_RADII_SHA256,
        "inputs": {
            "train": {"path": str(args.train_jsonl), "sha256": sha256_file(args.train_jsonl)},
            "generation": [
                {"path": str(path), "sha256": sha256_file(path)}
                for path in args.generation_jsonl
            ],
        },
        "gates": gates,
        "overall_pass": all(gates.values()),
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "GEOMETRY_CONTRACT_AUDIT.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"overall_pass": report["overall_pass"], **totals}))
    if not report["overall_pass"]:
        (args.output_dir / "_FAILED").touch()
        raise SystemExit(2)
    (args.output_dir / "_SUCCESS").touch()


if __name__ == "__main__":
    main()
