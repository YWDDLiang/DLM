#!/usr/bin/env python3
"""Summarize distribution shifts that explain MP-20 COV recall changes."""

from __future__ import annotations

import argparse
import math
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pymatgen.core import Composition, Element, Lattice, Structure
from scipy.stats import wasserstein_distance


warnings.filterwarnings("ignore")


def _pct_counter(counter: Counter, total: int) -> dict[str, float]:
    def key_fn(item):
        key = str(item[0])
        return int(key) if key.isdigit() else key

    return {str(k): round(v / total * 100.0, 2) for k, v in sorted(counter.items(), key=key_fn)}


def _top_share(counter: Counter, total: int, k: int) -> float:
    return round(sum(v for _, v in counter.most_common(k)) / total * 100.0, 2)


def _entropy_eff(counter: Counter) -> float:
    total = sum(counter.values())
    probs = [v / total for v in counter.values() if v]
    entropy = -sum(p * math.log(p) for p in probs)
    return round(math.exp(entropy), 1)


def summarize_records(records: list[dict], label: str) -> dict:
    total = len(records)
    num_atoms = Counter(r["num_atoms"] for r in records)
    num_elems = Counter(r["num_elems"] for r in records)
    formulas = Counter(r["formula"] for r in records)
    elem_occ = Counter()
    elem_sites = Counter()
    densities = []

    for record in records:
        elem_occ.update(set(record["symbols"]))
        elem_sites.update(record["symbols"])
        if record.get("density") is not None:
            densities.append(record["density"])

    return {
        "label": label,
        "n": total,
        "mean_num_atoms": round(float(np.mean([r["num_atoms"] for r in records])), 3),
        "mean_num_elems": round(float(np.mean([r["num_elems"] for r in records])), 3),
        "num_atoms_pct": _pct_counter(num_atoms, total),
        "num_elems_pct": _pct_counter(num_elems, total),
        "unique_formulas": len(formulas),
        "formula_top1_share": _top_share(formulas, total, 1),
        "formula_top10_share": _top_share(formulas, total, 10),
        "formula_entropy_eff": _entropy_eff(formulas),
        "top_formulas": formulas.most_common(12),
        "elem_occ_top": elem_occ.most_common(20),
        "elem_sites_top": elem_sites.most_common(20),
        "density_p10_p50_p90": [round(float(x), 4) for x in np.percentile(densities, [10, 50, 90])]
        if densities
        else None,
        "records": records,
    }


def load_reference(test_csv: Path) -> list[dict]:
    df = pd.read_csv(test_csv)
    records = []
    for cif in df["cif"]:
        try:
            structure = Structure.from_str(cif, fmt="cif")
        except Exception:
            continue
        symbols = [str(specie.symbol) for specie in structure.species]
        records.append(
            {
                "num_atoms": len(structure),
                "num_elems": len(set(symbols)),
                "symbols": symbols,
                "formula": structure.composition.reduced_formula,
                "density": structure.density,
            }
        )
    return records


def load_generated(pt_path: Path) -> list[dict]:
    data = torch.load(pt_path, map_location="cpu")
    num_atoms = data["num_atoms"][0].long().tolist()
    atom_types = data["atom_types"][0].long().tolist()
    lengths = data["lengths"][0].float().numpy()
    angles = data["angles"][0].float().numpy()

    records = []
    offset = 0
    for idx, atom_count in enumerate(num_atoms):
        atomic_numbers = atom_types[offset : offset + atom_count]
        offset += atom_count
        symbols = []
        for atomic_number in atomic_numbers:
            try:
                symbols.append(Element.from_Z(int(atomic_number)).symbol)
            except Exception:
                symbols.append(f"Z{atomic_number}")

        density = None
        try:
            lattice = Lattice.from_parameters(
                *(list(map(float, lengths[idx])) + list(map(float, angles[idx])))
            )
            structure = Structure(lattice, symbols, [[0, 0, 0]] * len(symbols), coords_are_cartesian=False)
            density = structure.density
        except Exception:
            pass

        records.append(
            {
                "num_atoms": atom_count,
                "num_elems": len(set(symbols)),
                "symbols": symbols,
                "formula": Composition(Counter(symbols)).reduced_formula,
                "density": density,
            }
        )
    return records


def print_summary(summaries: dict[str, dict]) -> None:
    for label, summary in summaries.items():
        print(f"==== {label}")
        print(
            "n",
            summary["n"],
            "meanN",
            summary["mean_num_atoms"],
            "meanElems",
            summary["mean_num_elems"],
            "uniqFormula",
            summary["unique_formulas"],
            "top1%",
            summary["formula_top1_share"],
            "top10%",
            summary["formula_top10_share"],
            "effFormula",
            summary["formula_entropy_eff"],
            "density_p10_p50_p90",
            summary["density_p10_p50_p90"],
        )
        print("num_elems_pct", summary["num_elems_pct"])
        keys = ["1", "2", "3", "4", "5", "6", "7", "8", "10", "12", "16", "20"]
        print("num_atoms_key_pct", {key: summary["num_atoms_pct"].get(key, 0) for key in keys})
        print("top_formulas", summary["top_formulas"][:8])
        print("elem_occ_top", summary["elem_occ_top"][:12])

    reference = summaries["test"]["records"]
    print("==== W distances vs test")
    for label in [key for key in summaries if key != "test"]:
        records = summaries[label]["records"]
        print(
            label,
            {
                "num_atoms_w": round(
                    float(wasserstein_distance([r["num_atoms"] for r in records], [r["num_atoms"] for r in reference])),
                    4,
                ),
                "num_elems_w": round(
                    float(wasserstein_distance([r["num_elems"] for r in records], [r["num_elems"] for r in reference])),
                    4,
                ),
                "density_w": round(
                    float(
                        wasserstein_distance(
                            [r["density"] for r in records if r["density"] is not None],
                            [r["density"] for r in reference if r["density"] is not None],
                        )
                    ),
                    4,
                ),
            },
        )

    print("==== element occurrence deltas vs test, top abs")
    test_occ = Counter()
    for record in reference:
        test_occ.update(set(record["symbols"]))
    test_pct = {key: value / len(reference) * 100.0 for key, value in test_occ.items()}

    for label in [key for key in summaries if key != "test"]:
        records = summaries[label]["records"]
        occ = Counter()
        for record in records:
            occ.update(set(record["symbols"]))
        pct = {key: value / len(records) * 100.0 for key, value in occ.items()}
        elems = set(test_pct) | set(pct)
        deltas = sorted(
            [
                (
                    elem,
                    round(pct.get(elem, 0.0) - test_pct.get(elem, 0.0), 2),
                    round(pct.get(elem, 0.0), 2),
                    round(test_pct.get(elem, 0.0), 2),
                )
                for elem in elems
            ],
            key=lambda item: abs(item[1]),
            reverse=True,
        )
        print(label, deltas[:18])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-csv", default="reference/crysllmgen/data/mp_20/test.csv")
    parser.add_argument("--old-final07", required=True)
    parser.add_argument("--sftbest", required=True)
    parser.add_argument("--lowlr", required=True)
    args = parser.parse_args()

    summaries = {
        "test": summarize_records(load_reference(Path(args.test_csv)), "test"),
        "old_final07_recall93": summarize_records(load_generated(Path(args.old_final07)), "old_final07_recall93"),
        "sftbest_nelemseq": summarize_records(load_generated(Path(args.sftbest)), "sftbest_nelemseq"),
        "lowlr_nelemseq": summarize_records(load_generated(Path(args.lowlr)), "lowlr_nelemseq"),
    }
    print_summary(summaries)


if __name__ == "__main__":
    main()
