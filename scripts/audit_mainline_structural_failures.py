#!/usr/bin/env python3
"""Audit structural failures across mainline generation/evaluation profiles."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
from typing import Any

from pymatgen.core import Structure


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def minimum_distance(structure: Structure) -> float:
    if len(structure) < 2:
        return math.inf
    matrix = structure.distance_matrix
    return min(
        float(matrix[left, right])
        for left in range(len(structure))
        for right in range(left + 1, len(structure))
    )


def distance_bucket(value: float) -> str:
    if value < 0.25:
        return "lt_0p25_A"
    if value < 0.5:
        return "0p25_to_0p5_A"
    if value < 0.75:
        return "0p5_to_0p75_A"
    return "ge_0p75_A"


def audit_cell(run_root: Path, direct_path: Path) -> dict[str, Any]:
    cell = direct_path.parents[2]
    generation_path = cell / "generation/generation.jsonl"
    labels_path = cell / "evaluation/full_reconstructed/attempt_labels_preofficial.jsonl"
    if not generation_path.is_file():
        raise FileNotFoundError(generation_path)
    direct = read_jsonl(direct_path)
    generation = read_jsonl(generation_path)
    generation_by_id = {str(row["attempt_id"]): row for row in generation}
    if len(generation_by_id) != len(generation):
        raise ValueError(f"duplicate generation attempt ID: {cell}")

    reasons: Counter[str] = Counter()
    geometry: Counter[str] = Counter()
    parseable = 0
    for row in direct:
        attempt_id = str(row["attempt_id"])
        source = generation_by_id.get(attempt_id)
        if source is None:
            raise ValueError(f"missing generation row: {cell}:{attempt_id}")
        if row.get("valid") is True:
            reasons["direct_valid"] += 1
        elif str(row.get("reason") or "").startswith("upstream_generation"):
            reasons["upstream_parse_or_body_failure"] += 1
        elif row.get("comp_valid") is not True:
            reasons["composition_invalid"] += 1
        elif row.get("struct_valid") is not True:
            reasons["direct_structure_invalid"] += 1
        else:
            reasons["other_direct_failure"] += 1

        payload = source.get("structure")
        if source.get("status") != "succeeded" or not isinstance(payload, dict):
            continue
        try:
            structure = Structure.from_dict(payload)
            parseable += 1
            min_distance = minimum_distance(structure)
            volume = float(structure.volume)
            geometry[distance_bucket(min_distance)] += 1
            geometry["volume_lt_0p1_A3"] += int(volume < 0.1)
            if row.get("comp_valid") is True and row.get("struct_valid") is not True:
                if min_distance < 0.5:
                    geometry["invalid_explained_by_collision"] += 1
                elif volume < 0.1:
                    geometry["invalid_explained_by_volume"] += 1
                else:
                    geometry["invalid_unexplained_by_frozen_metric"] += 1
        except Exception:
            geometry["structure_dict_parse_failure"] += 1

    reconstruction = Counter()
    if labels_path.is_file():
        for row in read_jsonl(labels_path):
            reconstruction["reconstructed"] += int(row.get("reconstructed") is True)
            reconstruction["not_reconstructed"] += int(row.get("reconstructed") is not True)
            reconstruction["chgnet_known"] += int(
                row.get("chgnet_relaxation_known") is True
            )
            reconstruction["chgnet_unknown"] += int(
                row.get("chgnet_relaxation_known") is not True
            )

    return {
        "cell": cell.relative_to(run_root).as_posix(),
        "attempts": len(direct),
        "parseable_structures": parseable,
        "direct": dict(sorted(reasons.items())),
        "geometry": dict(sorted(geometry.items())),
        "reconstruction": dict(sorted(reconstruction.items())),
    }


def audit_accounting(name: str, path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    reasons = Counter(
        str(row.get("reason") or "parsed")
        for row in rows
        if row.get("parsed") is not True
    )
    return {
        "cell": name,
        "attempts": len(rows),
        "parsed": sum(row.get("parsed") is True for row in rows),
        "parse_failures": sum(reasons.values()),
        "reasons": dict(sorted(reasons.items())),
    }


def aggregate(cells: list[dict[str, Any]]) -> dict[str, Any]:
    direct: Counter[str] = Counter()
    geometry: Counter[str] = Counter()
    reconstruction: Counter[str] = Counter()
    for cell in cells:
        direct.update(cell["direct"])
        geometry.update(cell["geometry"])
        reconstruction.update(cell["reconstruction"])
    invalid = direct["direct_structure_invalid"]
    explained = geometry["invalid_explained_by_collision"] + geometry[
        "invalid_explained_by_volume"
    ]
    return {
        "cells": len(cells),
        "attempts": sum(cell["attempts"] for cell in cells),
        "direct": dict(sorted(direct.items())),
        "geometry": dict(sorted(geometry.items())),
        "reconstruction": dict(sorted(reconstruction.items())),
        "structural_invalid_explained_by_force_or_G2_fraction": (
            None if invalid == 0 else explained / invalid
        ),
    }


def render(report: dict[str, Any]) -> str:
    lines = [
        "# Mainline structural-failure taxonomy",
        "",
        "| Cell | Attempts | Direct valid | Structural invalid | Upstream failure | CHGNet unknown |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for cell in report["cells"]:
        direct = cell["direct"]
        reconstruction = cell["reconstruction"]
        lines.append(
            f"| {cell['cell']} | {cell['attempts']} | {direct.get('direct_valid', 0)} | "
            f"{direct.get('direct_structure_invalid', 0)} | "
            f"{direct.get('upstream_parse_or_body_failure', 0)} | "
            f"{reconstruction.get('chgnet_unknown', 0)} |"
        )
    overall = report["overall"]
    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            f"- observations: `{overall['attempts']}` across `{overall['cells']}` cells",
            f"- direct structural invalid: `{overall['direct'].get('direct_structure_invalid', 0)}`",
            f"- upstream parse/body failure: `{overall['direct'].get('upstream_parse_or_body_failure', 0)}`",
            f"- invalid explained by <0.5 Å collision or volume <0.1 Å³: "
            f"`{overall['structural_invalid_explained_by_force_or_G2_fraction']}`",
            "",
            "Force-Score targets collision/local-energy failures; G2-valid also covers "
            "analytic collision/volume failures. CE/schema supervision remains responsible "
            "for upstream text/body failures.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-run", type=Path, action="append", default=[])
    parser.add_argument(
        "--include-cell",
        action="append",
        default=[],
        help="Regex applied to each eval-run-relative Direct path",
    )
    parser.add_argument(
        "--accounting", action="append", default=[], help="NAME=JSONL"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)

    cells = []
    include = [re.compile(value) for value in args.include_cell]
    for run in args.eval_run:
        run = run.resolve()
        for path in sorted(run.rglob("attempt_metrics.jsonl")):
            if path.parent.name != "direct":
                continue
            relative = path.relative_to(run).as_posix()
            if include and not any(pattern.search(relative) for pattern in include):
                continue
            cells.append(audit_cell(run, path))
    accounting = []
    for item in args.accounting:
        if "=" not in item:
            raise ValueError("accounting must be NAME=JSONL")
        name, raw_path = item.split("=", 1)
        accounting.append(audit_accounting(name, Path(raw_path).resolve()))
    report = {
        "schema": "mainline_structural_failure_taxonomy_v1",
        "status": "complete",
        "cells": cells,
        "accounting": accounting,
        "overall": aggregate(cells),
    }
    output.mkdir(parents=True)
    (output / "STRUCTURAL_FAILURE_TAXONOMY.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    (output / "STRUCTURAL_FAILURE_TAXONOMY.md").write_text(render(report))
    (output / "_SUCCESS").touch()
    print(json.dumps(report["overall"], sort_keys=True))


if __name__ == "__main__":
    main()
