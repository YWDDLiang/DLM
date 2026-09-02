#!/usr/bin/env python3
"""Build a deterministic 64-structure × 8-perturbation Force-Score preflight."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random

import numpy as np
import pandas as pd
from pymatgen.core import Lattice, Structure

from crystal_dlm.dynamic_crystal import (
    arrays_to_structure,
    parse_dynamic_answer,
    structure_to_dynamic_answer,
)


CORE_PERTURBATIONS = (
    "collision_0p15A",
    "collision_0p25A",
    "collision_0p35A",
    "collision_0p45A",
    "coord_jitter_0p03A",
    "coord_jitter_0p08A",
    "near_threshold_0p60A",
)
SENTINEL_PERTURBATIONS = (
    "isotropic_compression_6pct",
    "anisotropic_strain_6pct",
    "shear_4pct",
    "wrap_boundary",
)


def minimum_distance(structure: Structure) -> float:
    if len(structure) < 2:
        return float("inf")
    matrix = structure.distance_matrix
    return min(
        float(matrix[left, right])
        for left in range(len(structure))
        for right in range(left + 1, len(structure))
    )


def n_bin(count: int) -> str:
    if count <= 5:
        return "N02_05"
    if count <= 10:
        return "N06_10"
    if count <= 15:
        return "N11_15"
    return "N16_20"


def arity_bin(arity: int) -> str:
    return str(arity) if arity <= 3 else "4plus"


def cartesian_jitter(
    structure: Structure, *, rms_angstrom: float, rng: np.random.Generator
) -> Structure:
    shifts = rng.normal(size=(len(structure), 3))
    shifts -= shifts.mean(axis=0, keepdims=True)
    rms = float(np.sqrt(np.mean(np.sum(shifts * shifts, axis=1))))
    if rms > 0:
        shifts *= float(rms_angstrom) / rms
    fractional = structure.frac_coords + shifts @ np.linalg.inv(structure.lattice.matrix)
    return Structure(
        structure.lattice,
        structure.species,
        np.mod(fractional, 1.0),
        coords_are_cartesian=False,
    )


def controlled_collision(
    structure: Structure, *, target_angstrom: float, rng: np.random.Generator
) -> Structure:
    matrix = structure.distance_matrix.copy()
    np.fill_diagonal(matrix, np.inf)
    left, right = np.unravel_index(np.argmin(matrix), matrix.shape)
    if left > right:
        left, right = right, left
    frac = structure.frac_coords.copy()
    _distance, image = structure.lattice.get_distance_and_image(frac[left], frac[right])
    delta_frac = frac[right] + np.asarray(image, dtype=float) - frac[left]
    delta_cart = delta_frac @ structure.lattice.matrix
    norm = float(np.linalg.norm(delta_cart))
    if norm < 1.0e-8:
        unit = rng.normal(size=3)
        unit /= np.linalg.norm(unit)
    else:
        unit = delta_cart / norm
    target_frac = (unit * float(target_angstrom)) @ np.linalg.inv(
        structure.lattice.matrix
    )
    frac[right] = np.mod(frac[left] + target_frac - image, 1.0)
    return Structure(
        structure.lattice,
        structure.species,
        frac,
        coords_are_cartesian=False,
    )


def strained(structure: Structure, transform: np.ndarray) -> Structure:
    matrix = np.asarray(structure.lattice.matrix) @ np.asarray(transform)
    return Structure(
        Lattice(matrix),
        structure.species,
        structure.frac_coords,
        coords_are_cartesian=False,
    )


def perturb(
    structure: Structure, name: str, rng: np.random.Generator
) -> Structure:
    if name == "coord_jitter_0p03A":
        return cartesian_jitter(structure, rms_angstrom=0.03, rng=rng)
    if name == "coord_jitter_0p08A":
        return cartesian_jitter(structure, rms_angstrom=0.08, rng=rng)
    if name == "collision_0p35A":
        return controlled_collision(structure, target_angstrom=0.35, rng=rng)
    if name == "collision_0p15A":
        return controlled_collision(structure, target_angstrom=0.15, rng=rng)
    if name == "collision_0p25A":
        return controlled_collision(structure, target_angstrom=0.25, rng=rng)
    if name == "collision_0p45A":
        return controlled_collision(structure, target_angstrom=0.45, rng=rng)
    if name == "near_threshold_0p60A":
        return controlled_collision(structure, target_angstrom=0.60, rng=rng)
    if name == "isotropic_compression_6pct":
        return strained(structure, np.eye(3) * 0.94)
    if name == "anisotropic_strain_6pct":
        return strained(structure, np.diag([0.94, 1.03, 1.03]))
    if name == "shear_4pct":
        return strained(
            structure,
            np.asarray([[1.0, 0.04, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        )
    if name == "wrap_boundary":
        frac = structure.frac_coords.copy()
        frac[0, 0] = 0.995
        return Structure(
            structure.lattice,
            structure.species,
            frac,
            coords_are_cartesian=False,
        )
    raise ValueError(name)


def select_structures(
    csv_path: Path, *, count: int, seed: int
) -> list[tuple[int, str, Structure, str]]:
    table = pd.read_csv(csv_path, usecols=["material_id", "cif"])
    indices = list(range(len(table)))
    random.Random(seed).shuffle(indices)
    groups: dict[tuple[str, str], list[tuple[int, str, Structure, str]]] = defaultdict(list)
    for source_index in indices:
        row = table.iloc[source_index]
        try:
            structure = Structure.from_str(str(row["cif"]), fmt="cif")
        except Exception:
            continue
        if not 2 <= len(structure) <= 20:
            continue
        arity = len({site.specie.symbol for site in structure})
        key = (n_bin(len(structure)), arity_bin(arity))
        groups[key].append((source_index, str(row["material_id"]), structure, "/".join(key)))
        if sum(len(values) for values in groups.values()) >= 4096:
            break
    selected = []
    positions = Counter()
    keys = sorted(groups)
    while len(selected) < count:
        progressed = False
        for key in keys:
            position = positions[key]
            if position < len(groups[key]):
                selected.append(groups[key][position])
                positions[key] += 1
                progressed = True
                if len(selected) == count:
                    break
        if not progressed:
            raise RuntimeError("insufficient stratified MP20 structures")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-structures", type=int, default=64)
    parser.add_argument("--selection-seed", type=int, default=20260902)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    selected = select_structures(
        args.train_csv.resolve(), count=args.base_structures, seed=args.selection_seed
    )

    rows = []
    for base_index, (source_index, material_id, teacher, stratum) in enumerate(selected):
        perturbations = CORE_PERTURBATIONS + (
            SENTINEL_PERTURBATIONS[base_index % len(SENTINEL_PERTURBATIONS)],
        )
        for perturbation_index, name in enumerate(perturbations):
            rng = np.random.default_rng(
                args.selection_seed + 1009 * base_index + 17 * perturbation_index
            )
            candidate = perturb(teacher, name, rng)
            answer, diagnostics = structure_to_dynamic_answer(candidate)
            quantized = arrays_to_structure(parse_dynamic_answer(answer, strict=True))
            rows.append(
                {
                    "schema": "force_score_preflight_row_v1",
                    "row_index": len(rows),
                    "base_index": base_index,
                    "source_row_index": source_index,
                    "material_id": material_id,
                    "stratum": stratum,
                    "perturbation": name,
                    "teacher_structure": teacher.as_dict(),
                    "perturbed_structure": candidate.as_dict(),
                    "quantized_perturbed_structure": quantized.as_dict(),
                    "dynamic_answer": answer,
                    "encoding_diagnostics": diagnostics.to_dict(),
                    "num_atoms": len(candidate),
                    "arity": len({site.specie.symbol for site in candidate}),
                    "minimum_distance_A": minimum_distance(candidate),
                    "volume_A3": float(candidate.volume),
                    "outcomes_read": False,
                }
            )
    perturbations_per_structure = len(CORE_PERTURBATIONS) + 1
    if len(rows) != args.base_structures * perturbations_per_structure:
        raise RuntimeError("preflight row count changed")
    output.mkdir(parents=True)
    data = output / "preflight_rows.jsonl"
    data.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    manifest = {
        "schema": "force_score_preflight_data_v1",
        "status": "complete",
        "source": "MP20 train only",
        "base_structures": len(selected),
        "perturbations_per_structure": perturbations_per_structure,
        "rows": len(rows),
        "selection_seed": args.selection_seed,
        "perturbation_counts": dict(Counter(row["perturbation"] for row in rows)),
        "stratum_counts": dict(Counter(row["stratum"] for row in rows)),
        "outcomes_read": False,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    (output / "_SUCCESS").touch()
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
