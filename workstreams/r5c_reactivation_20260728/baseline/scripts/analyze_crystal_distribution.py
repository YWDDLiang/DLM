#!/usr/bin/env python3
"""Compare generated crystal distributions with MP-20 reference data."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from pymatgen.core import Lattice, Structure
from scipy.stats import wasserstein_distance

from crystal_dlm.fixed_slot import Z_TO_SYMBOL, write_json


HIGH_SYMMETRY_VALUES = (0.0, 0.25, 0.5, 0.75, 1.0)


def close_to_high_symmetry(value: float, tol: float) -> bool:
    wrapped = value % 1.0
    return any(abs(wrapped - target) <= tol for target in HIGH_SYMMETRY_VALUES)


def coord_key(coord: Sequence[float], decimals: int = 6) -> Tuple[float, float, float]:
    return tuple(round(float(value) % 1.0, decimals) for value in coord)  # type: ignore[return-value]


def min_fractional_distance(coords: Sequence[Sequence[float]]) -> float | None:
    if len(coords) < 2:
        return None
    best = math.inf
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            delta = [abs(float(coords[i][axis]) - float(coords[j][axis])) for axis in range(3)]
            delta = [min(value, 1.0 - value) for value in delta]
            best = min(best, math.sqrt(sum(value * value for value in delta)))
    return None if math.isinf(best) else best


def species_from_atomic_numbers(atom_types: Sequence[int]) -> List[str]:
    species = []
    for atomic_number in atom_types:
        atomic_number = int(atomic_number)
        species.append(Z_TO_SYMBOL.get(atomic_number, "H"))
    return species


def structure_from_arrays(
    frac_coords: Sequence[Sequence[float]],
    atom_types: Sequence[int],
    lengths: Sequence[float],
    angles: Sequence[float],
) -> Structure:
    return Structure(
        lattice=Lattice.from_parameters(*(list(map(float, lengths)) + list(map(float, angles)))),
        species=species_from_atomic_numbers(atom_types),
        coords=np.asarray(frac_coords, dtype=float) % 1.0,
        coords_are_cartesian=False,
    )


def load_pt_structures(path: Path, eval_index: int = 0) -> List[Structure]:
    data = torch.load(path, map_location="cpu")
    frac_coords = data["frac_coords"]
    atom_types = data["atom_types"]
    lengths = data["lengths"]
    angles = data["angles"]
    num_atoms = data["num_atoms"]

    if frac_coords.dim() == 3:
        frac_coords = frac_coords[eval_index]
    if atom_types.dim() == 2:
        atom_types = atom_types[eval_index]
    if lengths.dim() == 3:
        lengths = lengths[eval_index]
    if angles.dim() == 3:
        angles = angles[eval_index]
    if num_atoms.dim() == 2:
        num_atoms = num_atoms[eval_index]

    structures: List[Structure] = []
    start = 0
    for idx, n_atom in enumerate(num_atoms.tolist()):
        n_atom = int(n_atom)
        cur_frac = frac_coords[start : start + n_atom].numpy()
        cur_atom = atom_types[start : start + n_atom].numpy()
        structures.append(
            structure_from_arrays(
                cur_frac,
                cur_atom,
                lengths[idx].numpy(),
                angles[idx].numpy(),
            )
        )
        start += n_atom
    return structures


def load_csv_structures(path: Path, limit: int | None = None) -> List[Structure]:
    structures: List[Structure] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader):
            if limit is not None and idx >= limit:
                break
            structures.append(Structure.from_str(row["cif"], fmt="cif"))
    return structures


def quantiles(values: Sequence[float]) -> Dict[str, float | None]:
    if not values:
        return {key: None for key in ("min", "p01", "p05", "p25", "median", "p75", "p95", "p99", "max", "mean")}
    arr = np.asarray(values, dtype=float)
    return {
        "min": float(np.min(arr)),
        "p01": float(np.quantile(arr, 0.01)),
        "p05": float(np.quantile(arr, 0.05)),
        "p25": float(np.quantile(arr, 0.25)),
        "median": float(np.quantile(arr, 0.50)),
        "p75": float(np.quantile(arr, 0.75)),
        "p95": float(np.quantile(arr, 0.95)),
        "p99": float(np.quantile(arr, 0.99)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


def summarize_structures(structures: Sequence[Structure], high_symmetry_tol: float) -> Dict[str, Any]:
    densities: List[float] = []
    volumes_per_atom: List[float] = []
    num_atoms: List[int] = []
    num_elements: List[int] = []
    min_distances: List[float] = []
    high_sym_fracs: List[float] = []
    all_angles_90 = 0
    all_lengths_equal = 0
    duplicate_records = 0
    same_species_duplicate_records = 0
    element_histogram: Counter[str] = Counter()
    atom_count_histogram: Counter[str] = Counter()
    num_element_histogram: Counter[str] = Counter()

    for structure in structures:
        densities.append(float(structure.density))
        volumes_per_atom.append(float(structure.volume / max(1, len(structure))))
        num_atoms.append(len(structure))
        species = [site.specie.symbol for site in structure.sites]
        element_histogram.update(species)
        n_elements = len(set(species))
        num_elements.append(n_elements)
        atom_count_histogram[str(len(structure))] += 1
        num_element_histogram[str(n_elements)] += 1

        lengths = list(map(float, structure.lattice.abc))
        angles = list(map(float, structure.lattice.angles))
        all_angles_90 += int(all(abs(value - 90.0) < 1e-6 for value in angles))
        all_lengths_equal += int(max(lengths) - min(lengths) < 1e-6)

        coords = (np.asarray(structure.frac_coords, dtype=float) % 1.0).tolist()
        coord_counts = Counter(coord_key(coord) for coord in coords)
        species_coord_counts = Counter((symbol, coord_key(coord)) for symbol, coord in zip(species, coords))
        duplicate_records += int(any(count > 1 for count in coord_counts.values()))
        same_species_duplicate_records += int(any(count > 1 for count in species_coord_counts.values()))
        min_distance = min_fractional_distance(coords)
        if min_distance is not None:
            min_distances.append(float(min_distance))
        high_sym_count = sum(
            int(all(close_to_high_symmetry(float(axis), high_symmetry_tol) for axis in coord))
            for coord in coords
        )
        high_sym_fracs.append(high_sym_count / max(1, len(coords)))

    return {
        "count": len(structures),
        "density": quantiles(densities),
        "volume_per_atom": quantiles(volumes_per_atom),
        "num_atoms": quantiles([float(value) for value in num_atoms]),
        "num_elements": quantiles([float(value) for value in num_elements]),
        "min_fractional_distance": quantiles(min_distances),
        "high_symmetry_coord_fraction": quantiles(high_sym_fracs),
        "records_all_angles_90": all_angles_90,
        "records_all_lengths_equal": all_lengths_equal,
        "records_with_exact_duplicate_sites": duplicate_records,
        "records_with_same_species_duplicate_sites": same_species_duplicate_records,
        "atom_count_histogram": dict(atom_count_histogram.most_common()),
        "num_element_histogram": dict(num_element_histogram.most_common()),
        "element_histogram_top30": dict(element_histogram.most_common(30)),
        "_raw": {
            "density": densities,
            "volume_per_atom": volumes_per_atom,
            "num_atoms": num_atoms,
            "num_elements": num_elements,
            "high_symmetry_coord_fraction": high_sym_fracs,
        },
    }


def compare_summaries(generated: Mapping[str, Any], reference: Mapping[str, Any]) -> Dict[str, Any]:
    gen_raw = generated["_raw"]
    ref_raw = reference["_raw"]
    comparisons: Dict[str, Any] = {}
    for key in ("density", "volume_per_atom", "num_atoms", "num_elements", "high_symmetry_coord_fraction"):
        if gen_raw.get(key) and ref_raw.get(key):
            comparisons[f"{key}_wasserstein"] = float(
                wasserstein_distance(gen_raw[key], ref_raw[key])
            )
    ref_density = np.asarray(ref_raw.get("density", []), dtype=float)
    gen_density = np.asarray(gen_raw.get("density", []), dtype=float)
    if len(ref_density) and len(gen_density):
        low, high = np.quantile(ref_density, [0.01, 0.99])
        comparisons["density_outside_ref_p01_p99_fraction"] = float(
            np.mean((gen_density < low) | (gen_density > high))
        )
        comparisons["density_ref_p01"] = float(low)
        comparisons["density_ref_p99"] = float(high)
    return comparisons


def drop_raw(summary: Dict[str, Any]) -> Dict[str, Any]:
    copied = dict(summary)
    copied.pop("_raw", None)
    return copied


def write_markdown(payload: Mapping[str, Any], path: Path) -> None:
    gen = payload["generated"]
    ref = payload["reference"]
    cmp = payload["comparison"]
    lines = [
        "# Crystal Distribution Analysis",
        "",
        "## 诊断摘要",
        "",
        f"- 生成样本数：{gen['count']}；参考样本数：{ref['count']}",
        f"- density Wasserstein：{cmp.get('density_wasserstein')}",
        f"- volume/atom Wasserstein：{cmp.get('volume_per_atom_wasserstein')}",
        f"- num_atoms Wasserstein：{cmp.get('num_atoms_wasserstein')}",
        f"- num_elements Wasserstein：{cmp.get('num_elements_wasserstein')}",
        f"- 高对称坐标比例 Wasserstein：{cmp.get('high_symmetry_coord_fraction_wasserstein')}",
        f"- density 超出参考 p01-p99 的比例：{cmp.get('density_outside_ref_p01_p99_fraction')}",
        f"- 生成样本精确重复坐标结构数：{gen['records_with_exact_duplicate_sites']}",
        f"- 生成样本同元素同坐标重复结构数：{gen['records_with_same_species_duplicate_sites']}",
        f"- 生成样本高对称坐标比例中位数：{gen['high_symmetry_coord_fraction']['median']}",
        f"- 参考样本高对称坐标比例中位数：{ref['high_symmetry_coord_fraction']['median']}",
        "",
        "## Density Quantiles",
        "",
        "```json",
        json.dumps({"generated": gen["density"], "reference": ref["density"]}, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Volume Per Atom Quantiles",
        "",
        "```json",
        json.dumps({"generated": gen["volume_per_atom"], "reference": ref["volume_per_atom"]}, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Atom Count Histogram",
        "",
        "```json",
        json.dumps({"generated": gen["atom_count_histogram"], "reference": ref["atom_count_histogram"]}, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Element Histogram Top30",
        "",
        "```json",
        json.dumps({"generated": gen["element_histogram_top30"], "reference": ref["element_histogram_top30"]}, ensure_ascii=False, indent=2),
        "```",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated-pt", type=Path, required=True)
    parser.add_argument("--reference-csv", type=Path, default=PROJECT_ROOT / "reference/crysllmgen/data/mp_20/test.csv")
    parser.add_argument("--eval-index", type=int, default=0)
    parser.add_argument("--reference-limit", type=int, default=None)
    parser.add_argument("--high-symmetry-tol", type=float, default=1e-4)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    generated_structures = load_pt_structures(args.generated_pt, eval_index=args.eval_index)
    reference_structures = load_csv_structures(args.reference_csv, limit=args.reference_limit)
    generated = summarize_structures(generated_structures, args.high_symmetry_tol)
    reference = summarize_structures(reference_structures, args.high_symmetry_tol)
    comparison = compare_summaries(generated, reference)
    payload = {
        "generated_pt": str(args.generated_pt),
        "reference_csv": str(args.reference_csv),
        "eval_index": args.eval_index,
        "generated": drop_raw(generated),
        "reference": drop_raw(reference),
        "comparison": comparison,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    write_json(str(args.output_json), payload)
    write_markdown(payload, args.output_md)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
