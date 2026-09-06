#!/usr/bin/env python3
"""Exercise the actual continuous CIF/export/CrysLLMGen graph/refiner input path."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
from pymatgen.core import Lattice, Structure
from scipy.optimize import linear_sum_assignment
from torch_geometric.data import Data

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from crystal_dlm.dynamic_crystal import arrays_to_dynamic_tokens
from crystal_dlm.mixed_geometry_training_data import read_rows
from scripts.refine_dlm_with_crysllmgen import ProposalDataset


def periodic_coordinate_error(reference, candidate):
    if reference.composition != candidate.composition or len(reference) != len(candidate):
        raise ValueError("coordinate matching requires the same exact composition")
    error = 0.
    for symbol in set(reference.atomic_numbers):
        original_ids = np.flatnonzero(np.asarray(reference.atomic_numbers) == symbol)
        candidate_ids = np.flatnonzero(np.asarray(candidate.atomic_numbers) == symbol)
        delta = reference.frac_coords[original_ids, None] - candidate.frac_coords[None, candidate_ids]
        costs = np.abs(delta - np.round(delta)).max(-1)
        left, right = linear_sum_assignment(costs)
        error = max(error, float(costs[left, right].max()))
    return error


def graph_geometry_checks(expected, actual):
    gram_error = float(np.max(np.abs(expected.lattice.metric_tensor - actual.lattice.metric_tensor)))
    relative_gram = gram_error / max(1., float(np.abs(expected.lattice.metric_tensor).max()))
    coordinate_error = periodic_coordinate_error(expected, actual)
    return {"gram_error_A2": gram_error, "relative_gram_error": relative_gram,
            "coordinate_error": coordinate_error, "passed": relative_gram < 1e-10 and coordinate_error < 1e-11}


def record(structure, ordinal, key):
    elements = list(dict.fromkeys(str(site.specie) for site in structure))
    counts = [sum(str(site.specie) == element for site in structure) for element in elements]
    tokens, diagnostics = arrays_to_dynamic_tokens(structure.lattice.abc, structure.lattice.angles,
                                                   [str(site.specie) for site in structure], structure.frac_coords)
    if diagnostics.length_clips or diagnostics.angle_clips or diagnostics.coord_clips:
        raise ValueError("engineering fixture must have an ordinary valid diagnostic Q encoding")
    return {"source_split": "evaluation", "sample_idx": ordinal, "evaluation_ordinal": ordinal,
            "group_id": f"continuous_graph_preflight:{ordinal}", "trajectory_id": f"engineering:{ordinal}",
            "success": True, "parseable": True, "structure": structure.as_dict(), "body": " ".join(tokens),
            "num_atoms": structure.num_sites, "plan_state": {"N": structure.num_sites, "elements": elements, "counts": counts},
            "engineering_source_key": key, "eligible_policy": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--crysllmgen-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    single = Structure(Lattice(np.diag([4.0234, 20 / 4.0234, 6.])), ["He"], [[0., 0., 0.]])
    rounded = Structure(Lattice(np.diag([4., 5., 6.])), ["He"], [[0., 0., 0.]])
    checker_counterexample = graph_geometry_checks(single, rounded)
    if checker_counterexample["passed"]:
        raise RuntimeError("graph checker failed to reject equal-volume one-atom lattice quantization")
    originals = [Structure(Lattice.from_parameters(4.0234, 5.0432, 6.0321, 85.213, 95.327, 105.413),
                           ["Na", "Cl"], [[.003137, .202314, .398721], [.507913, .604822, .796132]])]
    keys = ["analytic_sub_bin_fixture"]
    identities = read_rows(args.identity)
    for count in (1, 10, 20):
        row = next(row for row in identities if row["identity_verified"] is True
                   and len(row["continuous_aligned"]["species"]) == count)
        geometry = row["continuous_aligned"]
        originals.append(Structure(Lattice(geometry["lattice_matrix_A"]), geometry["species"], geometry["fractional"]))
        keys.append([row["source_split"], row["source_row_idx"]])
    input_path = args.output_dir / "input.jsonl"
    with input_path.open("x", encoding="utf-8") as stream:
        for ordinal, (structure, key) in enumerate(zip(originals, keys)):
            stream.write(json.dumps(record(structure, ordinal, key)) + "\n")
    artifact_dir = args.output_dir / "artifacts"
    subprocess.run([sys.executable, str(ROOT / "scripts/export_programmed_path_artifacts.py"),
                    "--paths-jsonl", str(input_path), "--output-dir", str(artifact_dir),
                    "--crysllmgen-dir", str(args.crysllmgen_dir), "--native-source", "structure"], check=True)
    exported = read_rows(artifact_dir / "paths.jsonl")
    graphs = torch.load(artifact_dir / "proposal_graphs.pt", weights_only=False, map_location="cpu")
    if len(graphs) != len(originals) or len(exported) != len(originals):
        raise ValueError("actual continuous export/graph preprocessing did not preserve all engineering requests")
    indexed = {int(graph["sample_idx"]): graph for graph in graphs}
    checks = []
    for index, original in enumerate(originals):
        row, graph = exported[index], indexed[index]
        native = Structure.from_dict(row["native_structure"])
        parsed = Structure.from_str(Path(row["cif_path"]).read_text(), fmt="cif")
        prepared = Structure(Lattice.from_parameters(*np.asarray(graph["length"]).reshape(-1),
                                                       *np.asarray(graph["angle"]).reshape(-1)),
                             np.asarray(graph["a_type"]).reshape(-1).tolist(), np.asarray(graph["x_coord"]))
        if original.composition != prepared.composition or len(original) != len(prepared):
            raise ValueError("actual graph changed composition")
        # Match the inspected build_crystal(cif,niggli=True,primitive=False)
        # operations explicitly; no graph function or quantizer supplies the
        # expected geometry. This also checks one-atom lattice shape.
        reduced = parsed.get_reduced_structure()
        expected_graph = Structure(Lattice.from_parameters(*reduced.lattice.parameters),
                                   reduced.species, reduced.frac_coords, coords_are_cartesian=False)
        graph_check = graph_geometry_checks(expected_graph, prepared)
        # Physical fingerprints remain diagnostics, not identity substitutes.
        pair_error = float(np.max(np.abs(np.sort(original.distance_matrix.reshape(-1))
                                          - np.sort(prepared.distance_matrix.reshape(-1)))))
        volume_error = abs(prepared.volume / original.volume - 1)
        native_coordinate_error = float(np.max(np.abs(native.frac_coords - original.frac_coords)))
        cif_gram_error = float(np.max(np.abs(parsed.lattice.metric_tensor - original.lattice.metric_tensor)))
        cif_coordinate_error = periodic_coordinate_error(original, parsed)
        proposal = ProposalDataset([graph], Data, seed_from_graph_field="refiner_seed")[0]
        fp32_coordinate_error = float(np.max(np.abs(proposal.frac_coords.numpy().astype(np.float64)
                                                  - np.asarray(graph["x_coord"], dtype=np.float64))))
        off_grid = float(np.max(np.abs(proposal.frac_coords.numpy() * 100
                                      - np.round(proposal.frac_coords.numpy() * 100))))
        passed = (graph_check["passed"] and pair_error < 1e-6 and volume_error < 1e-7 and native_coordinate_error == 0
                  and cif_gram_error < 1e-9 and cif_coordinate_error < 1e-12 and fp32_coordinate_error < 6e-8)
        if index == 0:
            passed = passed and off_grid > .01
        checks.append({"source_key": keys[index], "num_atoms": len(original), "passed": passed,
                       "graph_identity": graph_check,
                       "pair_distance_error_A": pair_error, "relative_volume_error": volume_error,
                       "native_coordinate_error": native_coordinate_error, "cif_gram_error_A2": cif_gram_error,
                       "cif_coordinate_error": cif_coordinate_error,
                       "refiner_fp32_coordinate_error": fp32_coordinate_error,
                       "refiner_input_max_off_0p01_grid_in_bins": off_grid,
                       "refiner_seed": int(proposal.refiner_seed[0])})
    result = {"status": "PASS" if all(item["passed"] for item in checks) else "FAIL",
              "actual_process_one": str(args.crysllmgen_dir / "data_utils.py"),
              "process_one_sha256": hashlib.sha256((args.crysllmgen_dir / "data_utils.py").read_bytes()).hexdigest(),
              "input_identity_sha256": hashlib.sha256(args.identity.read_bytes()).hexdigest(),
              "rejected_equal_volume_single_atom_quantization": checker_counterexample,
              "model_forwards": 0, "refiner_forwards": 0, "physics_or_SUN_evaluation": False,
              "checks": checks}
    (args.output_dir / "SUMMARY.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if result["status"] != "PASS":
        raise ValueError("continuous graph roundtrip lost geometry precision; inspect SUMMARY.json")
    (args.output_dir / "_SUCCESS").touch()
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
