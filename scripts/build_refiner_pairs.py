#!/usr/bin/env python3
"""Align E2 proposal graphs, refined tensors, and the requested-attempt ledger."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def mean_absolute(left, right) -> float:
    values = [abs(float(a) - float(b)) for a, b in zip(left, right)]
    return sum(values) / len(values)


def periodic_rms(pre, post) -> float:
    squared: list[float] = []
    for pre_row, post_row in zip(pre, post):
        for before, after in zip(pre_row, post_row):
            delta = ((float(after) - float(before) + 0.5) % 1.0) - 0.5
            squared.append(delta * delta)
    return math.sqrt(sum(squared) / len(squared))


def structure_metrics(lengths, angles, atom_types, frac_coords) -> tuple[Any, dict[str, Any]]:
    from pymatgen.core import Element, Lattice, Structure

    lattice = Lattice.from_parameters(*[float(value) for value in lengths], *[float(value) for value in angles])
    species = [Element.from_Z(int(value)) for value in atom_types]
    structure = Structure(lattice, species, frac_coords, coords_are_cartesian=False)
    minimum_distance = None
    if len(structure) > 1:
        distances = structure.distance_matrix
        minimum_distance = min(
            float(distances[i, j])
            for i in range(len(structure))
            for j in range(len(structure))
            if i != j
        )
    spacegroup_number = int(structure.get_space_group_info(symprec=0.1)[1])
    return structure, {
        "min_distance": minimum_distance,
        "spacegroup_number": spacegroup_number,
        "p1": spacegroup_number == 1,
    }


def single_point_energy_per_atom(structure, calculator) -> float:
    from pymatgen.io.ase import AseAtomsAdaptor

    atoms = AseAtomsAdaptor.get_atoms(structure)
    atoms.calc = calculator
    return float(atoms.get_potential_energy()) / len(structure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposal-graphs", type=Path, required=True)
    parser.add_argument("--refined", type=Path, required=True)
    parser.add_argument("--metadata-jsonl", type=Path, required=True)
    parser.add_argument("--attempt-ledger", type=Path, required=True)
    parser.add_argument("--evaluation-jsonl", type=Path, default=None)
    parser.add_argument("--chgnet-model", type=Path, default=None)
    parser.add_argument("--chgnet-device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from crystal_dlm.composition_validity import smact_validity_from_atom_types
    from crystal_dlm.r5_plan_state import lattice_system_from_lattice, spacegroup_bucket, volume_per_atom_bin
    from h1a2_repro.story_panel_analysis import cluster_with_matcher

    chgnet_calculator = None
    if args.chgnet_model is not None:
        from chgnet.model.dynamics import CHGNetCalculator

        if not args.chgnet_model.is_file():
            raise FileNotFoundError(args.chgnet_model)
        chgnet_calculator = CHGNetCalculator.from_file(
            str(args.chgnet_model), use_device=str(args.chgnet_device)
        )

    graphs = torch.load(args.proposal_graphs, map_location="cpu")
    payload = torch.load(args.refined, map_location="cpu")
    metadata = load_jsonl(args.metadata_jsonl)
    ledger = load_jsonl(args.attempt_ledger)
    if len(graphs) != len(metadata):
        raise ValueError(f"proposal graphs {len(graphs)} != metadata rows {len(metadata)}")

    refined_num_atoms = torch.as_tensor(payload["num_atoms"])[0]
    refined_atom_types = torch.as_tensor(payload["atom_types"])[0]
    refined_frac_coords = torch.as_tensor(payload["frac_coords"])[0]
    refined_lengths = torch.as_tensor(payload["lengths"])[0]
    refined_angles = torch.as_tensor(payload["angles"])[0]
    if len(refined_num_atoms) != len(graphs):
        raise ValueError(f"refined structures {len(refined_num_atoms)} != proposal graphs {len(graphs)}")

    evaluation = {}
    if args.evaluation_jsonl is not None:
        evaluation = {str(row["task_id"]): row for row in load_jsonl(args.evaluation_jsonl)}
    graph_by_task = {str(row["task_id"]): index for index, row in enumerate(metadata)}
    rows: list[dict[str, Any]] = []
    atom_offset = 0
    post_slices: list[tuple[int, int]] = []
    for value in refined_num_atoms.tolist():
        count = int(value)
        post_slices.append((atom_offset, atom_offset + count))
        atom_offset += count
    if atom_offset != len(refined_atom_types) or atom_offset != len(refined_frac_coords):
        raise ValueError("refined flattened atom arrays do not match num_atoms")

    matcher = StructureMatcher()
    structures_by_group: dict[tuple[str, str], list[tuple[int, Any, Any]]] = defaultdict(list)
    for attempt in ledger:
        task_id = str(attempt["task_id"])
        row = dict(attempt)
        row.update(evaluation.get(task_id, {}))
        row["body_success"] = bool(attempt.get("body_success"))
        row["refined"] = False
        if task_id not in graph_by_task:
            rows.append(row)
            continue
        index = graph_by_task[task_id]
        graph = graphs[index]
        start, end = post_slices[index]
        pre_n = int(torch.as_tensor(graph["n_atom"]).view(-1)[0].item())
        post_n = int(refined_num_atoms[index].item())
        pre_types = [int(value) for value in torch.as_tensor(graph["a_type"]).view(-1).tolist()]
        post_types = [int(value) for value in refined_atom_types[start:end].view(-1).tolist()]
        pre_frac = torch.as_tensor(graph["x_coord"], dtype=torch.float64).tolist()
        post_frac = refined_frac_coords[start:end].to(dtype=torch.float64).tolist()
        pre_lengths = torch.as_tensor(graph["length"], dtype=torch.float64).view(-1).tolist()
        pre_angles = torch.as_tensor(graph["angle"], dtype=torch.float64).view(-1).tolist()
        post_lengths = refined_lengths[index].to(dtype=torch.float64).view(-1).tolist()
        post_angles = refined_angles[index].to(dtype=torch.float64).view(-1).tolist()
        row.update(
            {
                "refined": True,
                "n_pre": pre_n,
                "n_post": post_n,
                "n_invariant": pre_n == post_n,
                "composition_invariant": sorted(pre_types) == sorted(post_types),
                "coordinate_periodic_rms": (
                    periodic_rms(pre_frac, post_frac)
                    if pre_n == post_n and len(pre_frac) == len(post_frac)
                    else None
                ),
                "lattice_length_mae": mean_absolute(pre_lengths, post_lengths),
                "lattice_angle_mae": mean_absolute(pre_angles, post_angles),
            }
        )
        try:
            pre_structure, pre_info = structure_metrics(pre_lengths, pre_angles, pre_types, pre_frac)
            post_structure, post_info = structure_metrics(post_lengths, post_angles, post_types, post_frac)
            structures_by_group[(str(row.get("plan_id")), str(row.get("arm")))].append(
                (len(rows), pre_structure, post_structure)
            )
            row.update(
                {
                    "structure_match": bool(matcher.fit(pre_structure, post_structure)),
                    "min_distance_pre": pre_info["min_distance"],
                    "min_distance_post": post_info["min_distance"],
                    "spacegroup_pre": pre_info["spacegroup_number"],
                    "spacegroup_post": post_info["spacegroup_number"],
                    "spacegroup_same": pre_info["spacegroup_number"] == post_info["spacegroup_number"],
                    "pre_p1": pre_info["p1"],
                    "post_p1": post_info["p1"],
                    "pre_struct_valid": (
                        float(pre_structure.volume) >= 0.1
                        and (pre_info["min_distance"] is None or pre_info["min_distance"] >= 0.5)
                    ),
                    "post_struct_valid": (
                        float(post_structure.volume) >= 0.1
                        and (post_info["min_distance"] is None or post_info["min_distance"] >= 0.5)
                    ),
                }
            )
            try:
                pre_comp_valid = bool(smact_validity_from_atom_types(pre_types))
                post_comp_valid = bool(smact_validity_from_atom_types(post_types))
                row.update(
                    {
                        "pre_comp_valid": pre_comp_valid,
                        "post_comp_valid": post_comp_valid,
                        "pre_joint_valid": pre_comp_valid and bool(row["pre_struct_valid"]),
                        "post_joint_valid": post_comp_valid and bool(row["post_struct_valid"]),
                    }
                )
                row.setdefault("body_good", bool(row["pre_joint_valid"]))
                row.setdefault("final_good", bool(row["post_joint_valid"]))
                row.setdefault("good_definition", "CrysLLMGen-compatible Direct joint validity")
            except Exception as exc:
                row["direct_comp_error"] = f"{type(exc).__name__}: {exc}"
            if chgnet_calculator is not None:
                try:
                    row["energy_pre"] = single_point_energy_per_atom(pre_structure, chgnet_calculator)
                    row["energy_post"] = single_point_energy_per_atom(post_structure, chgnet_calculator)
                    row["energy_kind"] = "CHGNet single-point eV/atom"
                except Exception as exc:
                    row["chgnet_error"] = f"{type(exc).__name__}: {exc}"
            plan = metadata[index].get("plan_state") or {}
            planned_lattice = str(plan.get("lattice_system", "unknown"))
            planned_spacegroup = str(plan.get("spacegroup_bucket", "unknown"))
            planned_volume = str(plan.get("volume_per_atom_bin", "unknown"))
            actual_lattice_pre = lattice_system_from_lattice(pre_lengths, pre_angles)
            actual_lattice_post = lattice_system_from_lattice(post_lengths, post_angles)
            actual_spacegroup_pre = spacegroup_bucket({"spacegroup.number": pre_info["spacegroup_number"]})
            actual_spacegroup_post = spacegroup_bucket({"spacegroup.number": post_info["spacegroup_number"]})
            actual_volume_pre = volume_per_atom_bin(pre_lengths, pre_angles, pre_n)
            actual_volume_post = volume_per_atom_bin(post_lengths, post_angles, post_n)
            row.update(
                {
                    "plan_lattice_match_pre": actual_lattice_pre == planned_lattice,
                    "plan_lattice_match_post": actual_lattice_post == planned_lattice,
                    "plan_spacegroup_match_pre": actual_spacegroup_pre == planned_spacegroup,
                    "plan_spacegroup_match_post": actual_spacegroup_post == planned_spacegroup,
                    "plan_volume_match_pre": actual_volume_pre == planned_volume,
                    "plan_volume_match_post": actual_volume_post == planned_volume,
                }
            )
        except Exception as exc:  # keep the requested attempt in the output
            row["structure_metrics_error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)

    for indexed_structures in structures_by_group.values():
        pre_labels = cluster_with_matcher(
            [pre_structure for _, pre_structure, _ in indexed_structures], matcher
        )
        post_labels = cluster_with_matcher(
            [post_structure for _, _, post_structure in indexed_structures], matcher
        )
        for (row_index, _, _), pre_label, post_label in zip(
            indexed_structures, pre_labels, post_labels
        ):
            rows[row_index]["pre_structure_cluster"] = int(pre_label)
            rows[row_index]["post_structure_cluster"] = int(post_label)

    if len(rows) != len(ledger):
        raise RuntimeError("output row count changed from requested-attempt ledger")
    write_jsonl(args.output, rows)


if __name__ == "__main__":
    main()
