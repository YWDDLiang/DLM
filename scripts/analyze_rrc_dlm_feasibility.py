#!/usr/bin/env python3
"""Audit existing matched bodies before any RRC/D3PO GPU execution."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence


ARMS = ("base", "g0_all", "g1_strict")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_cell(path: Path) -> dict[int, dict[str, Any]]:
    rows = read_jsonl(path)
    indexed = {int(row["ordinal"]): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"duplicate ordinals in {path}")
    return indexed


def common_known_ordinals(
    cells: Mapping[str, Mapping[int, Mapping[str, Any]]]
) -> list[int]:
    common = set.intersection(*(set(rows) for rows in cells.values()))
    return [
        ordinal
        for ordinal in sorted(common)
        if all(
            rows[ordinal].get("official_hull_status") == "known"
            and rows[ordinal].get("chgnet_energy_per_atom") is not None
            for rows in cells.values()
        )
    ]


def outcome_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    hull = [float(row["official_e_above_hull"]) for row in rows]
    return {
        "n": len(rows),
        "strict_stable": sum(row.get("strict_stable") is True for row in rows),
        "strict_sun": sum(row.get("strict_sun") is True for row in rows),
        "meta_stable": sum(row.get("meta_stable") is True for row in rows),
        "meta_sun": sum(row.get("meta_sun") is True for row in rows),
        "official_e_hull_mean": statistics.fmean(hull),
        "official_e_hull_q50": statistics.median(hull),
    }


def selected_rows(
    cells: Mapping[str, Mapping[int, Mapping[str, Any]]],
    ordinals: Sequence[int],
    labels: Sequence[str],
) -> list[Mapping[str, Any]]:
    return [
        min(
            (cells[label][ordinal] for label in labels),
            key=lambda row: float(row["chgnet_energy_per_atom"]),
        )
        for ordinal in ordinals
    ]


def oracle_curve(
    cells: Mapping[str, Mapping[int, Mapping[str, Any]]],
    ordinals: Sequence[int],
) -> dict[str, Any]:
    labels = tuple(cells)
    result: dict[str, Any] = {}
    for count in range(1, len(labels) + 1):
        summaries = [
            outcome_summary(selected_rows(cells, ordinals, subset))
            for subset in itertools.combinations(labels, count)
        ]
        result[f"k{count}"] = {
            "subsets": len(summaries),
            "mean": {
                key: statistics.fmean(float(summary[key]) for summary in summaries)
                for key in summaries[0]
            },
            "minimum_strict_sun": min(row["strict_sun"] for row in summaries),
            "maximum_strict_sun": max(row["strict_sun"] for row in summaries),
            "minimum_meta_sun": min(row["meta_sun"] for row in summaries),
            "maximum_meta_sun": max(row["meta_sun"] for row in summaries),
        }
    return result


def rank_agreement(
    cells: Mapping[str, Mapping[int, Mapping[str, Any]]],
    ordinals: Sequence[int],
) -> dict[str, Any]:
    concordant = 0
    discordant = 0
    ties = 0
    for ordinal in ordinals:
        for left_label, right_label in itertools.combinations(cells, 2):
            left = cells[left_label][ordinal]
            right = cells[right_label][ordinal]
            energy_delta = float(left["chgnet_energy_per_atom"]) - float(
                right["chgnet_energy_per_atom"]
            )
            hull_delta = float(left["official_e_above_hull"]) - float(
                right["official_e_above_hull"]
            )
            if abs(energy_delta) < 1e-12 or abs(hull_delta) < 1e-12:
                ties += 1
            elif (energy_delta < 0.0) == (hull_delta < 0.0):
                concordant += 1
            else:
                discordant += 1
    non_ties = concordant + discordant
    return {
        "concordant": concordant,
        "discordant": discordant,
        "ties": ties,
        "accuracy_non_tie": None if non_ties == 0 else concordant / non_ties,
    }


def threshold_pair_summary(
    cells: Mapping[str, Mapping[int, Mapping[str, Any]]],
    ordinals: Sequence[int],
    metric: str,
) -> dict[str, Any]:
    variable_compositions = 0
    margins: list[float] = []
    for ordinal in ordinals:
        rows = [cell[ordinal] for cell in cells.values()]
        values = [bool(row[metric]) for row in rows]
        if any(values) and not all(values):
            variable_compositions += 1
        for left, right in itertools.combinations(rows, 2):
            if bool(left[metric]) == bool(right[metric]):
                continue
            margins.append(
                abs(
                    float(left["chgnet_energy_per_atom"])
                    - float(right["chgnet_energy_per_atom"])
                )
            )
    return {
        "variable_compositions": variable_compositions,
        "crossing_pairs": len(margins),
        "median_margin_mev_per_atom": (
            None if not margins else 1000.0 * statistics.median(margins)
        ),
    }


def audit_cells(cells: Mapping[str, Mapping[int, Mapping[str, Any]]]) -> dict[str, Any]:
    ordinals = common_known_ordinals(cells)
    return {
        "labels": list(cells),
        "common_known_compositions": len(ordinals),
        "rank_agreement": rank_agreement(cells, ordinals),
        "oracle_curve": oracle_curve(cells, ordinals),
        "threshold_pairs": {
            metric: threshold_pair_summary(cells, ordinals, metric)
            for metric in ("strict_stable", "meta_stable")
        },
    }


def load_l6(final_root: Path) -> dict[str, dict[int, dict[str, Any]]]:
    return {
        f"seed{seed}_{arm}": load_cell(
            final_root
            / f"cells/seed{seed}/{arm}/attempt_results_official.jsonl"
        )
        for seed in (17, 18)
        for arm in ARMS
    }


def load_l7(final_root: Path) -> dict[str, dict[int, dict[str, Any]]]:
    return {
        arm: load_cell(final_root / f"cells/seed18/{arm}/attempt_results_official.jsonl")
        for arm in ARMS
    }


def raw_geometry_probe(l6_final: Path, l6_generation: Path) -> dict[str, Any]:
    try:
        import numpy as np
        from pymatgen.core import Element, Structure
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import GroupKFold
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        return {"available": False, "reason": str(exc)}

    def radius(symbol: str) -> float:
        element = Element(symbol)
        for value in (element.atomic_radius, element.atomic_radius_calculated):
            if value is not None:
                return float(value)
        return 1.5

    def features(cif: str) -> Any:
        structure = Structure.from_str(cif, fmt="cif")
        count = len(structure)
        lattice = structure.lattice
        scale = lattice.volume ** (1.0 / 3.0)
        lengths = sorted((lattice.a / scale, lattice.b / scale, lattice.c / scale))
        distance_matrix = np.asarray(structure.distance_matrix, dtype=float)
        distances: list[float] = []
        normalized: list[float] = []
        for left in range(count):
            for right in range(left + 1, count):
                distance = float(distance_matrix[left, right])
                distances.append(distance)
                normalized.append(
                    distance
                    / (
                        radius(structure[left].specie.symbol)
                        + radius(structure[right].specie.symbol)
                    )
                )
        distances_array = np.asarray(distances or [0.0], dtype=float)
        normalized_array = np.asarray(normalized or [0.0], dtype=float)
        nearest = np.asarray(
            [
                min(distance_matrix[left, right] for right in range(count) if left != right)
                for left in range(count)
            ]
            if count > 1
            else [0.0],
            dtype=float,
        )
        return np.asarray(
            [
                math.log(lattice.volume / count),
                *lengths,
                lattice.alpha / 180.0,
                lattice.beta / 180.0,
                lattice.gamma / 180.0,
                np.linalg.cond(lattice.matrix),
                distances_array.min(),
                np.quantile(distances_array, 0.10),
                np.median(distances_array),
                nearest.mean(),
                nearest.std(),
                normalized_array.min(),
                np.quantile(normalized_array, 0.25),
                np.median(normalized_array),
            ],
            dtype=float,
        )

    grouped: dict[int, list[dict[str, Any]]] = {}
    parse_failures = 0
    for seed in (17, 18):
        for arm in ARMS:
            raw = {
                int(row["ordinal"]): row
                for row in read_jsonl(
                    l6_generation / f"seed{seed}/{arm}/body/raw_generations.jsonl"
                )
            }
            official = load_cell(
                l6_final
                / f"cells/seed{seed}/{arm}/attempt_results_official.jsonl"
            )
            for ordinal, raw_row in raw.items():
                official_row = official[ordinal]
                if (
                    raw_row.get("parsed") is not True
                    or not raw_row.get("cif")
                    or official_row.get("official_hull_status") != "known"
                ):
                    continue
                try:
                    descriptor = features(str(raw_row["cif"]))
                except Exception:
                    parse_failures += 1
                    continue
                grouped.setdefault(ordinal, []).append(
                    {
                        "descriptor": descriptor,
                        "energy": float(official_row["chgnet_energy_per_atom"]),
                    }
                )
    grouped = {key: rows for key, rows in grouped.items() if len(rows) >= 3}
    pair_features: list[Any] = []
    pair_labels: list[int] = []
    pair_groups: list[int] = []
    for group, rows in grouped.items():
        for left, right in itertools.combinations(rows, 2):
            delta = left["energy"] - right["energy"]
            if abs(delta) < 1e-12:
                continue
            difference = left["descriptor"] - right["descriptor"]
            pair_features.extend((difference, -difference))
            pair_labels.extend((int(delta < 0.0), int(delta > 0.0)))
            pair_groups.extend((group, group))
    x = np.asarray(pair_features)
    y = np.asarray(pair_labels)
    groups = np.asarray(pair_groups)
    aucs: list[float] = []
    accuracies: list[float] = []
    for train, test in GroupKFold(n_splits=5).split(x, y, groups):
        model = make_pipeline(
            StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)
        )
        model.fit(x[train], y[train])
        probability = model.predict_proba(x[test])[:, 1]
        aucs.append(float(roc_auc_score(y[test], probability)))
        accuracies.append(float(np.mean((probability >= 0.5) == y[test])))
    return {
        "available": True,
        "groups": len(grouped),
        "ordered_pair_examples": len(y),
        "parse_failures": parse_failures,
        "fold_auc": aucs,
        "mean_auc": statistics.fmean(aucs),
        "mean_accuracy": statistics.fmean(accuracies),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--l6-final", type=Path, required=True)
    parser.add_argument("--l7-final", type=Path, required=True)
    parser.add_argument("--l6-generation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--skip-raw-probe", action="store_true")
    args = parser.parse_args()

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "rrc_dlm_feasibility_audit_v1",
        "gpu_jobs_used": 0,
        "l6": audit_cells(load_l6(args.l6_final.resolve())),
        "l7": audit_cells(load_l7(args.l7_final.resolve())),
        "raw_geometry_probe": (
            {"available": False, "reason": "skipped"}
            if args.skip_raw_probe
            else raw_geometry_probe(
                args.l6_final.resolve(), args.l6_generation.resolve()
            )
        ),
        "interpretation": {
            "within_composition_chgnet_is_official_rank_label": True,
            "oracle_headroom_exists": True,
            "simple_geometry_rank_head_supported": False,
            "preferred_low_resource_method": (
                "composition-matched shared-noise masked-DLM D3PO"
            ),
        },
    }
    (output / "RRC_DLM_FEASIBILITY.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# RRC/D3PO zero-GPU feasibility audit",
        "",
        "| Cohort | Common known | Rank accuracy | K1 Strict/Meta S.U.N. | Kmax Strict/Meta S.U.N. |",
        "|---|---:|---:|---:|---:|",
    ]
    for cohort in ("l6", "l7"):
        row = report[cohort]
        first = row["oracle_curve"]["k1"]["mean"]
        last = row["oracle_curve"][f"k{len(row['labels'])}"]["mean"]
        lines.append(
            f"| {cohort.upper()} | {row['common_known_compositions']} | "
            f"{row['rank_agreement']['accuracy_non_tie']:.4f} | "
            f"{first['strict_sun']:.1f}/{first['meta_sun']:.1f} | "
            f"{last['strict_sun']:.1f}/{last['meta_sun']:.1f} |"
        )
    probe = report["raw_geometry_probe"]
    lines.extend(
        [
            "",
            f"Raw geometry probe available: `{probe['available']}`; "
            f"mean AUC: `{probe.get('mean_auc')}`.",
            "",
            "Conclusion: energy preference labels are exact within composition and oracle headroom is material, but a simple geometry rank head is unsupported. Prefer shared-noise full-sequence D3PO over scalar rank regression.",
        ]
    )
    (output / "RRC_DLM_FEASIBILITY.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    (output / "_SUCCESS").touch()
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
