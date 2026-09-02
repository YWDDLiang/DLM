#!/usr/bin/env python3
"""Summarize matched SPAD raw/refined Direct and pre-official CHGNet cells."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
from typing import Any, Mapping, Sequence


STREAMS = (17, 18)
ENDPOINTS = ("raw", "refined")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def direct_index(row: Mapping[str, Any]) -> int:
    return int(str(row["attempt_id"]).rsplit("-", 1)[1])


def distribution(values: Sequence[float]) -> dict[str, Any]:
    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not ordered:
        return {"count": 0, "mean": None, "median": None, "minimum": None, "maximum": None}
    return {
        "count": len(ordered),
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "minimum": ordered[0],
        "maximum": ordered[-1],
    }


def bootstrap_mean(values: Sequence[float], *, label: str, replicates: int = 10_000) -> dict[str, Any]:
    data = [float(value) for value in values]
    if not data:
        return {"replicates": 0, "ci95_lower": None, "ci95_upper": None}
    seed = int.from_bytes(hashlib.sha256(label.encode("utf-8")).digest()[:8], "big")
    rng = random.Random(seed)
    means = sorted(
        statistics.fmean(data[rng.randrange(len(data))] for _ in data)
        for _ in range(replicates)
    )
    return {
        "replicates": replicates,
        "ci95_lower": means[int(0.025 * replicates)],
        "ci95_upper": means[int(0.975 * replicates) - 1],
    }


def average_maps(maps: Sequence[Mapping[int, float]]) -> dict[int, float]:
    if not maps:
        return {}
    shared = set(maps[0])
    for mapping in maps[1:]:
        shared &= set(mapping)
    return {
        index: statistics.fmean(mapping[index] for mapping in maps)
        for index in sorted(shared)
    }


def cluster_map(
    values: Mapping[int, float], clusters: Mapping[int, str] | None
) -> dict[int | str, float]:
    if clusters is None:
        return dict(values)
    grouped: dict[str, list[float]] = {}
    for sample_idx, value in values.items():
        grouped.setdefault(clusters[sample_idx], []).append(float(value))
    return {key: statistics.fmean(group) for key, group in grouped.items()}


def paired_delta(
    first: Mapping[int, float], second: Mapping[int, float]
) -> dict[int, float]:
    return {
        index: float(second[index]) - float(first[index])
        for index in sorted(set(first) & set(second))
    }


def summarize_delta(
    per_stream: Sequence[Mapping[int, float]],
    *,
    label: str,
    clusters: Mapping[int, str] | None,
) -> dict[str, Any]:
    averaged = cluster_map(average_maps(per_stream), clusters)
    values = list(averaged.values())
    return {
        "direction": "negative is favorable",
        "per_stream": [distribution(list(mapping.values())) for mapping in per_stream],
        "compositions_observed": len(values),
        "stream_averaged": distribution(values),
        "fraction_negative": (
            sum(value < 0.0 for value in values) / len(values) if values else None
        ),
        "bootstrap": bootstrap_mean(values, label=label),
    }


def load_cell(eval_run: Path, arm: str, stream: int, endpoint: str) -> tuple[dict[str, Any], dict[int, float], dict[int, bool]]:
    root = eval_run / arm / f"stream{stream}" / endpoint / "evaluation"
    direct_rows = read_jsonl(root / "direct/attempt_metrics.jsonl")
    labels = read_jsonl(root / "full_reconstructed/attempt_labels_preofficial.jsonl")
    direct = {direct_index(row): row for row in direct_rows}
    labelled = {int(row["ordinal"]): row for row in labels}
    if set(direct) != set(range(256)) or set(labelled) != set(range(256)):
        raise ValueError(f"{arm}/{stream}/{endpoint} does not cover fixed256")
    energies = {
        index: float(row["chgnet_energy_per_atom"])
        for index, row in labelled.items()
        if row.get("chgnet_energy_per_atom") is not None
        and math.isfinite(float(row["chgnet_energy_per_atom"]))
    }
    valid = {index: row.get("valid") is True for index, row in direct.items()}
    counts = {
        "arm": arm,
        "stream": stream,
        "endpoint": endpoint,
        "requested": 256,
        "composition_valid": sum(row.get("comp_valid") is True for row in direct.values()),
        "structure_valid": sum(row.get("struct_valid") is True for row in direct.values()),
        "direct_joint": sum(valid.values()),
        "reconstructed": sum(row.get("reconstructed") is True for row in labelled.values()),
        "novel": sum(row.get("novel") is True for row in labelled.values()),
        "unique": sum(row.get("unique_representative") is True for row in labelled.values()),
        "novel_unique": sum(row.get("novel_unique") is True for row in labelled.values()),
        "energy": distribution(list(energies.values())),
    }
    return counts, energies, valid


def mcnemar(first: Mapping[int, bool], second: Mapping[int, bool]) -> dict[str, int]:
    shared = sorted(set(first) & set(second))
    return {
        "paired": len(shared),
        "second_only": sum((not first[index]) and second[index] for index in shared),
        "first_only": sum(first[index] and (not second[index]) for index in shared),
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# SPAD raw/refined offline result",
        "",
        "All rows use the fixed 256 Plan ledger. Official MP hull values are not used here.",
        "",
        "| Arm | Endpoint | Stream | Comp | Struct | Direct | N/U/NU | Energy known | Mean CHGNet E |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in report["cells"]:
        energy = cell["energy"]
        mean = "NA" if energy["mean"] is None else f"{energy['mean']:.6f}"
        lines.append(
            f"| {cell['arm']} | {cell['endpoint']} | {cell['stream']} | "
            f"{cell['composition_valid']}/256 | {cell['structure_valid']}/256 | "
            f"{cell['direct_joint']}/256 | {cell['novel']}/{cell['unique']}/{cell['novel_unique']} | "
            f"{energy['count']}/256 | {mean} |"
        )
    lines.extend(["", "## Two-stream aggregates", ""])
    for aggregate in report["aggregates"]:
        lines.append(
            f"- {aggregate['arm']} {aggregate['endpoint']}: Direct "
            f"{aggregate['direct_joint']}/512 ({100*aggregate['direct_joint']/512:.2f}%), "
            f"NU {aggregate['novel_unique']}/512."
        )
    lines.extend(["", "## Paired CHGNet effects", ""])
    for name, effect in report["energy_effects"].items():
        value = effect["stream_averaged"]["mean"]
        ci = effect["bootstrap"]
        if value is None:
            lines.append(f"- {name}: no common finite CHGNet rows.")
        else:
            lines.append(
                f"- {name}: mean delta {value:.6f} eV/atom; 95% composition-bootstrap "
                f"[{ci['ci95_lower']:.6f}, {ci['ci95_upper']:.6f}] "
                f"over {effect['compositions_observed']} compositions."
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--arms", default="BC,BR,BS")
    parser.add_argument("--ledger", type=Path)
    args = parser.parse_args()
    arms = tuple(value.strip() for value in args.arms.split(",") if value.strip())
    clusters = None
    if args.ledger is not None:
        ledger = read_jsonl(args.ledger)
        if {int(row["sample_idx"]) for row in ledger} != set(range(256)):
            raise ValueError("cluster ledger does not cover fixed256")
        clusters = {
            int(row["sample_idx"]): (
                str(row["exact_composition_identity"])
                if row.get("exact_composition_identity") is not None
                else f"failed:{int(row['sample_idx'])}"
            )
            for row in ledger
        }
    if not (
        (args.eval_run / "_OFFLINE_SUCCESS").is_file()
        or (args.eval_run / "OFFLINE_FINAL.json").is_file()
    ):
        raise FileNotFoundError("offline run is incomplete")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    cells: list[dict[str, Any]] = []
    energies: dict[tuple[str, str, int], dict[int, float]] = {}
    valid: dict[tuple[str, str, int], dict[int, bool]] = {}
    for arm in arms:
        for endpoint in ENDPOINTS:
            for stream in STREAMS:
                cell, energy, direct = load_cell(args.eval_run, arm, stream, endpoint)
                cells.append(cell)
                energies[(arm, endpoint, stream)] = energy
                valid[(arm, endpoint, stream)] = direct

    aggregates = []
    for arm in arms:
        for endpoint in ENDPOINTS:
            selected = [
                cell for cell in cells
                if cell["arm"] == arm and cell["endpoint"] == endpoint
            ]
            aggregates.append(
                {
                    "arm": arm,
                    "endpoint": endpoint,
                    "requested": 512,
                    **{
                        key: sum(int(cell[key]) for cell in selected)
                        for key in (
                            "composition_valid", "structure_valid", "direct_joint",
                            "reconstructed", "novel", "unique", "novel_unique",
                        )
                    },
                }
            )

    energy_effects: dict[str, Any] = {}
    direct_effects: dict[str, Any] = {}
    for arm in arms:
        deltas = [
            paired_delta(energies[(arm, "raw", stream)], energies[(arm, "refined", stream)])
            for stream in STREAMS
        ]
        energy_effects[f"{arm}:refined-minus-raw"] = summarize_delta(
            deltas,
            label=f"{args.eval_run}:{arm}:refined-minus-raw",
            clusters=clusters,
        )
        direct_effects[f"{arm}:refined-vs-raw"] = [
            {"stream": stream, **mcnemar(valid[(arm, "raw", stream)], valid[(arm, "refined", stream)])}
            for stream in STREAMS
        ]
    for comparator in ("BC", "BR"):
        if "BS" not in arms or comparator not in arms:
            continue
        for endpoint in ENDPOINTS:
            deltas = [
                paired_delta(
                    energies[(comparator, endpoint, stream)],
                    energies[("BS", endpoint, stream)],
                )
                for stream in STREAMS
            ]
            energy_effects[f"{endpoint}:BS-minus-{comparator}"] = summarize_delta(
                deltas,
                label=f"{args.eval_run}:{endpoint}:BS-minus-{comparator}",
                clusters=clusters,
            )
            direct_effects[f"{endpoint}:BS-vs-{comparator}"] = [
                {
                    "stream": stream,
                    **mcnemar(
                        valid[(comparator, endpoint, stream)],
                        valid[("BS", endpoint, stream)],
                    ),
                }
                for stream in STREAMS
            ]

    report = {
        "schema": "spad_offline_final_v1",
        "eval_run": str(args.eval_run.resolve()),
        "arms": list(arms),
        "official": False,
        "cells": cells,
        "aggregates": aggregates,
        "energy_effects": energy_effects,
        "direct_effects": direct_effects,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "SPAD_OFFLINE_FINAL.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "SPAD_OFFLINE_FINAL.md").write_text(
        render_markdown(report), encoding="utf-8"
    )
    with (args.output_dir / "SPAD_OFFLINE_CELLS.csv").open(
        "x", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "arm", "endpoint", "stream", "requested", "composition_valid",
                "structure_valid", "direct_joint", "reconstructed", "novel",
                "unique", "novel_unique", "energy_known", "energy_mean",
            ],
        )
        writer.writeheader()
        for cell in cells:
            writer.writerow(
                {
                    **{key: cell[key] for key in writer.fieldnames if key in cell},
                    "energy_known": cell["energy"]["count"],
                    "energy_mean": cell["energy"]["mean"],
                }
            )
    (args.output_dir / "_SUCCESS").touch()
    print(json.dumps({"output": str(args.output_dir), "cells": len(cells)}))


if __name__ == "__main__":
    main()
