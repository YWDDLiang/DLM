#!/usr/bin/env python3
"""Diagnose where stable and S.U.N. yield is lost for a fixed DLM cohort."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any, Iterable


LATTICE_TO_SPACEGROUP = {
    "triclinic": "sg_001_002",
    "monoclinic": "sg_003_015",
    "orthorhombic": "sg_016_074",
    "tetragonal": "sg_075_142",
    "trigonal": "sg_143_167",
    "hexagonal": "sg_168_194",
    "cubic": "sg_195_230",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def rate(value: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else value / denominator


def determinant3(matrix: list[list[float]]) -> float:
    a, b, c = matrix
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


def volume_bin_bounds(value: Any) -> tuple[float, float] | None:
    text = str(value or "")
    parts = text.split("_")
    if len(parts) != 3 or parts[0] != "volpa":
        return None
    try:
        return float(int(parts[1])), float(int(parts[2]))
    except ValueError:
        return None


def n_bin(value: int) -> str:
    if value <= 4:
        return "01_04"
    if value <= 8:
        return "05_08"
    if value <= 12:
        return "09_12"
    return "13_20"


def quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    values = list(rows)
    reconstructed = sum(row["reconstructed"] for row in values)
    hull_known = sum(row["official_hull_status"] == "known" for row in values)
    strict_stable = sum(row["strict_stable"] for row in values)
    meta_stable = sum(row["meta_stable"] for row in values)
    strict_sun = sum(row["strict_sun"] for row in values)
    meta_sun = sum(row["meta_sun"] for row in values)
    novel_unique = sum(row["novel_unique"] for row in values)
    energies = [
        float(row["official_e_above_hull"])
        for row in values
        if row.get("official_e_above_hull") is not None
    ]
    return {
        "attempts": len(values),
        "reconstructed": reconstructed,
        "hull_known": hull_known,
        "novel_unique": novel_unique,
        "strict_stable": strict_stable,
        "strict_sun": strict_sun,
        "meta_stable": meta_stable,
        "meta_sun": meta_sun,
        "reconstructed_rate": rate(reconstructed, len(values)),
        "novel_unique_rate": rate(novel_unique, len(values)),
        "strict_stable_rate": rate(strict_stable, len(values)),
        "strict_sun_rate": rate(strict_sun, len(values)),
        "meta_stable_rate": rate(meta_stable, len(values)),
        "meta_sun_rate": rate(meta_sun, len(values)),
        "strict_retention": rate(strict_sun, strict_stable),
        "meta_retention": rate(meta_sun, meta_stable),
        "e_hull_q25": quantile(energies, 0.25),
        "e_hull_q50": quantile(energies, 0.50),
        "e_hull_q75": quantile(energies, 0.75),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt-results", type=Path, required=True)
    parser.add_argument("--generation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    attempts = sorted(read_jsonl(args.attempt_results), key=lambda row: int(row["ordinal"]))
    generation = sorted(read_jsonl(args.generation), key=lambda row: int(row["ordinal"]))
    if len(attempts) != len(generation) or [int(row["ordinal"]) for row in attempts] != list(
        range(len(attempts))
    ):
        raise ValueError("attempt/generation alignment changed")
    joined: list[dict[str, Any]] = []
    for attempt, generated in zip(attempts, generation):
        if str(attempt["attempt_id"]) != str(generated["attempt_id"]):
            raise ValueError("attempt id alignment changed")
        plan = generated.get("plan_state") or {}
        lattice = str(plan.get("lattice_system") or "unknown")
        spacegroup = str(plan.get("spacegroup_bucket") or "unknown")
        lattice_sg_match = LATTICE_TO_SPACEGROUP.get(lattice) == spacegroup
        structure = generated.get("structure")
        volpa = None
        if isinstance(structure, dict):
            matrix = structure.get("lattice", {}).get("matrix")
            sites = structure.get("sites") or []
            if isinstance(matrix, list) and len(matrix) == 3 and sites:
                volpa = abs(determinant3(matrix)) / len(sites)
        bounds = volume_bin_bounds(plan.get("volume_per_atom_bin"))
        volume_match = (
            None
            if volpa is None or bounds is None
            else bounds[0] <= float(volpa) < bounds[1]
        )
        joined.append(
            {
                **attempt,
                "family": str(plan.get("anion_framework") or "other"),
                "charge": str(plan.get("charge_bucket") or "unknown"),
                "lattice_sg_match": lattice_sg_match,
                "volume_match": volume_match,
                "soft_consistent": lattice_sg_match and volume_match is True,
                "arity": len(plan.get("elements") or ()),
                "n_bin": n_bin(int(plan.get("N") or 0)),
                "model494_volume_per_atom": volpa,
            }
        )

    group_specs = {
        "lattice_sg_match": lambda row: str(bool(row["lattice_sg_match"])),
        "volume_match": lambda row: "unknown" if row["volume_match"] is None else str(bool(row["volume_match"])),
        "soft_consistent": lambda row: str(bool(row["soft_consistent"])),
        "family": lambda row: row["family"],
        "charge": lambda row: row["charge"],
        "arity": lambda row: str(row["arity"]),
        "n_bin": lambda row: row["n_bin"],
    }
    grouped: dict[str, dict[str, Any]] = {}
    for name, key_fn in group_specs.items():
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in joined:
            buckets[key_fn(row)].append(row)
        grouped[name] = {
            key: summarize(value)
            for key, value in sorted(buckets.items(), key=lambda item: item[0])
        }

    overall = summarize(joined)
    stable_not_sun = {
        "strict": overall["strict_stable"] - overall["strict_sun"],
        "meta": overall["meta_stable"] - overall["meta_sun"],
    }
    target_math = {
        "strict_additional_stable_needed_at_current_retention": max(
            0,
            math.ceil(100 / max(overall["strict_retention"], 1e-9))
            - overall["strict_stable"],
        ),
        "meta_additional_stable_needed_at_current_retention": max(
            0,
            math.ceil(500 / max(overall["meta_retention"], 1e-9))
            - overall["meta_stable"],
        ),
    }
    report = {
        "schema": "h1a2_stability_mechanism_diagnostic_v1",
        "overall": overall,
        "stable_not_sun": stable_not_sun,
        "target_math": target_math,
        "groups": grouped,
        "interpretation_contract": {
            "stable_scope": "official hull known after model494 and CHGNet relaxation",
            "soft_consistency": "Plan lattice-SG hard match and model494 volume-per-atom inside Plan bin",
            "causal_claim": False,
        },
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "STABILITY_MECHANISM_DIAGNOSTIC.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Stability mechanism diagnostic",
        "",
        f"Overall Strict stable/SUN: `{overall['strict_stable']}/{overall['strict_sun']}`; "
        f"Meta: `{overall['meta_stable']}/{overall['meta_sun']}`.",
        f"At current retention, additional stable structures needed for 10/50: "
        f"Strict `{target_math['strict_additional_stable_needed_at_current_retention']}`, "
        f"Meta `{target_math['meta_additional_stable_needed_at_current_retention']}`.",
        "",
    ]
    for name in ("lattice_sg_match", "volume_match", "soft_consistent"):
        lines.extend(
            [
                f"## {name}",
                "",
                "| Group | n | Strict stable/SUN | Meta stable/SUN | E_hull median |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for key, value in grouped[name].items():
            median = value["e_hull_q50"]
            lines.append(
                f"| {key} | {value['attempts']} | {value['strict_stable']}/{value['strict_sun']} | "
                f"{value['meta_stable']}/{value['meta_sun']} | "
                f"{'NA' if median is None else f'{median:.4f}'} |"
            )
        lines.append("")
    (output / "STABILITY_MECHANISM_DIAGNOSTIC.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    (output / "_SUCCESS").touch()
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
