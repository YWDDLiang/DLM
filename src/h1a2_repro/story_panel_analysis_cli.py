"""Analyze E1 raw outputs without filtering failed requested tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .attribution_cli import load_jsonl
from .story_panel_analysis import (
    cluster_with_matcher,
    multiplicity_gate,
    summarize_plan_clusters,
    summarize_story_records,
)


def minimum_distance(structure) -> float | None:
    if len(structure) < 2:
        return None
    matrix = structure.distance_matrix
    return min(
        float(matrix[i, j])
        for i in range(len(structure))
        for j in range(len(structure))
        if i != j
    )


def local_environment_fingerprint(structure, crystal_nn) -> str | None:
    try:
        entries = []
        for index, site in enumerate(structure):
            coordination = float(crystal_nn.get_cn(structure, index, use_weights=True))
            entries.append((str(site.specie.symbol), round(coordination, 3)))
        payload = json.dumps(sorted(entries), separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
    except Exception:
        return None


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--sample-metrics", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--duplicate-tolerance", type=float, default=1e-6)
    args = parser.parse_args()

    from pymatgen.analysis.local_env import CrystalNN
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.core import Structure
    from crystal_dlm.r5_plan_state import lattice_system_from_lattice, spacegroup_bucket, volume_per_atom_bin

    tasks = load_jsonl(args.tasks)
    raw_rows = load_jsonl(args.raw)
    task_index = {str(row["task_id"]): dict(row) for row in tasks}
    if len(task_index) != len(tasks):
        raise ValueError("E1 task ledger contains duplicate task_id values")
    raw_index = {str(row["task_id"]): dict(row) for row in raw_rows}
    if len(raw_index) != len(raw_rows):
        raise ValueError("E1 raw output contains duplicate task_id values")
    unexpected = sorted(set(raw_index) - set(task_index))
    if unexpected:
        raise ValueError(f"raw output contains unexpected task ids: {unexpected[:5]}")

    matcher = StructureMatcher()
    crystal_nn = CrystalNN(weighted_cn=True)
    records: list[dict[str, Any]] = []
    structures_by_group: dict[tuple[str, str], list[tuple[int, Any]]] = defaultdict(list)
    for task in tasks:
        task_id = str(task["task_id"])
        raw = raw_index.get(task_id)
        record = {
            key: task.get(key)
            for key in (
                "task_id",
                "pair_id",
                "plan_id",
                "plan_source",
                "arm",
                "replicate",
                "scientific_seed",
                "shuffle_donor_plan_id",
            )
        }
        record.update(
            {
                "parsed": False,
                "plan_match": False,
                "graph_success": False,
                "model_forward_calls": None,
                "structure_cluster": None,
            }
        )
        if raw is None:
            record["failure"] = "missing_raw_output"
            records.append(record)
            continue
        record["parsed"] = bool(raw.get("parsed"))
        record["plan_match"] = bool(raw.get("parsed"))
        record["model_forward_calls"] = raw.get("model_forward_calls")
        cif = raw.get("cif")
        if not raw.get("parsed") or not cif:
            record["failure"] = raw.get("reason") or "no_graph_cif"
            records.append(record)
            continue
        try:
            structure = Structure.from_str(str(cif), fmt="cif")
            lengths = list(structure.lattice.abc)
            angles = list(structure.lattice.angles)
            sg_number = int(structure.get_space_group_info(symprec=0.1)[1])
            min_distance = minimum_distance(structure)
            actual_lattice = lattice_system_from_lattice(lengths, angles)
            actual_spacegroup = spacegroup_bucket({"spacegroup.number": sg_number})
            actual_volume = volume_per_atom_bin(lengths, angles, len(structure))
            plan = task["plan_state"]
            record.update(
                {
                    "graph_success": True,
                    "lattice_legal": all(float(value) > 0 for value in lengths)
                    and float(structure.lattice.volume) > 0,
                    "minimum_distance": min_distance,
                    "duplicate_free": min_distance is None or min_distance > float(args.duplicate_tolerance),
                    "actual_lattice": actual_lattice,
                    "actual_spacegroup": actual_spacegroup,
                    "actual_volume": actual_volume,
                    "plan_lattice_match": actual_lattice == str(plan.get("lattice_system", "unknown")),
                    "plan_spacegroup_match": actual_spacegroup == str(plan.get("spacegroup_bucket", "unknown")),
                    "plan_volume_match": actual_volume == str(plan.get("volume_per_atom_bin", "unknown")),
                    "local_environment_fingerprint": local_environment_fingerprint(structure, crystal_nn),
                }
            )
            structures_by_group[(str(task["plan_id"]), str(task["arm"]))].append((len(records), structure))
        except Exception as exc:
            record["failure"] = f"{type(exc).__name__}: {exc}"
        records.append(record)

    for indexed_structures in structures_by_group.values():
        labels = cluster_with_matcher([structure for _, structure in indexed_structures], matcher)
        for (record_index, _), label in zip(indexed_structures, labels):
            records[record_index]["structure_cluster"] = int(label)

    plan_clusters = summarize_plan_clusters(records)
    report = {
        "schema": "h1a2_story_e1_analysis_v1",
        "requested": len(tasks),
        "raw_rows": len(raw_rows),
        "groups": summarize_story_records(records),
        "plan_clusters": plan_clusters,
        "multiplicity_gate": multiplicity_gate(plan_clusters),
        "sample_metrics": (
            None
            if args.sample_metrics is None
            else json.loads(args.sample_metrics.read_text(encoding="utf-8"))
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    write_jsonl(args.output_jsonl, records)


if __name__ == "__main__":
    main()
