#!/usr/bin/env python3
"""Read-only CPU audit of the actual G sample/export/CIF/graph/refiner-input chain.

No model or optimizer is constructed. Existing proposal graphs are inspected;
process_one is not rerun, and no source file is rewritten.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback

os.environ["CUDA_VISIBLE_DEVICES"] = ""


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def indexed(rows, label, *, key="evaluation_ordinal"):
    result = {}
    for row in rows:
        value = int(row.get(key, row.get("sample_idx")))
        if value in result:
            raise ValueError(f"duplicate {label} ordinal {value}")
        result[value] = row
    return result


def import_checker(code_root):
    sys.path.insert(0, str(code_root / "src"))
    sys.path.insert(0, str(code_root))
    path = code_root / "operations/mixed_geometry/verify_continuous_graphs.py"
    spec = importlib.util.spec_from_file_location("actual_chain_original_40058_checker", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def audit(args):
    import numpy as np
    import torch
    from pymatgen.core import Lattice, Structure

    torch.set_num_threads(args.cpu_threads)
    checker, checker_path = import_checker(args.code_root)
    run = args.run_dir.resolve()
    paths = {"sample": run/"sample/native/paths.jsonl", "native": run/"native/paths.jsonl",
             "graphs": run/"native/proposal_graphs.pt", "process_one_source": args.crysllmgen_dir/"data_utils.py",
             "checker_source": checker_path, "dataset_source": args.code_root/"src/scripts/refine_dlm_with_crysllmgen.py"}
    if (run/"tau800/paths.jsonl").is_file():
        paths["tau800"] = run/"tau800/paths.jsonl"
    hashes = {str(path.resolve()): sha256(path) for path in paths.values()}
    sample = indexed(read_rows(paths["sample"]), "sample")
    native = indexed(read_rows(paths["native"]), "native")
    tau = indexed(read_rows(paths["tau800"]), "tau800") if "tau800" in paths else None
    expected = set(range(args.expected_requests))
    if set(sample) != expected or set(native) != expected or (tau is not None and set(tau) != expected):
        raise ValueError("source ledgers do not preserve the complete requested ordinal population")
    graph_list = torch.load(paths["graphs"], weights_only=False, map_location="cpu")
    graphs = indexed(graph_list, "graph", key="sample_idx")
    if not set(graphs) <= expected:
        raise ValueError("proposal graph contains an out-of-ledger ordinal")
    checks = []
    identity_fields = ("sample_idx", "group_id", "trajectory_id", "sampling_seed", "plan_state",
                       "species_program", "species_program_source", "num_atoms")
    active_fields = ("body", "structure", "cif", "cif_path")

    def require(condition, message):
        if not condition:
            raise ValueError(message)

    def direct_structure_check(a, b):
        require(list(a.atomic_numbers) == list(b.atomic_numbers), "direct structure changed ordered atom identities")
        matrix_error = float(np.abs(a.lattice.matrix-b.lattice.matrix).max())
        coordinate_error = float(np.abs(a.frac_coords-b.frac_coords).max())
        require(matrix_error <= 1e-12 and coordinate_error <= 1e-12,
                "direct sample/native structure geometry changed")
        return {"matrix_max_abs_A": matrix_error, "coordinate_max_abs": coordinate_error}

    for ordinal in sorted(expected):
        source, exported = sample[ordinal], native[ordinal]
        row = {"evaluation_ordinal": ordinal, "sample_idx": source.get("sample_idx"),
               "trajectory_id": source.get("trajectory_id"), "source_success": bool(source.get("success")),
               "native_success": bool(exported.get("success")), "graph_present": ordinal in graphs,
               "integrity_pass": False, "availability": "unknown", "stage": "identity", "error": None}
        try:
            require(all(source.get(k) == exported.get(k) for k in identity_fields), "sample/native identity changed")
            if tau is not None:
                require(all(source.get(k) == tau[ordinal].get(k) for k in identity_fields), "sample/tau identity changed")
            if not source.get("success"):
                require(not exported.get("success") and exported.get("parseable") is False,
                        "native exporter readmitted a failed sample")
                require(all(source.get(k) is None and exported.get(k) is None for k in active_fields),
                        "failed source retained an active preview")
                require(ordinal not in graphs, "failed source has a refiner graph")
                row.update(integrity_pass=True, availability="generation_failure_preserved", stage="complete")
                checks.append(row)
                continue
            row["stage"] = "sample_to_native"
            require(exported.get("success") and exported.get("parseable"),
                    f"successful sample unavailable after export: {exported.get('artifact_error')}")
            original = Structure.from_dict(source["structure"])
            true_native = Structure.from_dict(exported["native_structure"])
            label_input = Structure.from_dict(exported["structure"])
            row["sample_to_native_structure"] = direct_structure_check(original, true_native)
            row["sample_to_label_structure"] = direct_structure_check(original, label_input)
            plan = source["plan_state"]
            expected_species = Counter({str(e): int(n) for e,n in zip(plan["elements"], plan["counts"], strict=True)})
            require(original.num_sites == int(plan["N"]) and Counter(str(site.specie) for site in original) == expected_species,
                    "sample structure differs from fixed Plan composition")
            if tau is not None:
                require(tau[ordinal].get("native_structure") is not None, "tau ledger lost its genuine native upstream")
                row["tau_native_upstream"] = direct_structure_check(original, Structure.from_dict(tau[ordinal]["native_structure"]))

            row["stage"] = "actual_cif"
            cif_path = Path(exported["cif_path"]).resolve()
            require(cif_path.is_relative_to(run/"native/cifs"), "recorded CIF path is outside this native output directory")
            hashes[str(cif_path)] = sha256(cif_path)
            parsed = Structure.from_str(cif_path.read_text(encoding="utf-8"), fmt="cif")
            row["actual_cif"] = checker.graph_geometry_checks(original, parsed)
            require(row["actual_cif"]["passed"], "actual CIF changed the native Gram or periodic coordinates")
            row["volume_per_atom_A3"] = original.volume/original.num_sites

            row["stage"] = "stored_graph"
            if ordinal not in graphs:
                require(bool(exported.get("refiner_graph_error")), "missing graph has no recorded graph failure")
                row.update(integrity_pass=True, availability="recorded_graph_unavailable",
                           recorded_graph_error=exported["refiner_graph_error"], stage="complete")
                checks.append(row)
                continue
            graph = graphs[ordinal]
            require(not exported.get("refiner_graph_error"), "a graph and a graph-failure record coexist")
            prepared = Structure(Lattice.from_parameters(*np.asarray(graph["length"]).reshape(-1),
                                                         *np.asarray(graph["angle"]).reshape(-1)),
                                 np.asarray(graph["a_type"]).reshape(-1).tolist(), np.asarray(graph["x_coord"]))
            require(int(np.asarray(graph["n_atom"]).reshape(-1)[0]) == original.num_sites,
                    "graph atom count differs from original sample")
            reduced = parsed.get_reduced_structure()
            expected_graph = Structure(Lattice.from_parameters(*reduced.lattice.parameters),
                                       reduced.species, reduced.frac_coords, coords_are_cartesian=False)
            row["graph_geometry"] = checker.graph_geometry_checks(expected_graph, prepared)
            row["graph_relative_volume_error"] = abs(prepared.volume/original.volume-1.)
            row["graph_pair_distance_max_abs_A"] = float(np.abs(np.sort(original.distance_matrix.reshape(-1))
                                                                   -np.sort(prepared.distance_matrix.reshape(-1))).max())
            require(row["graph_geometry"]["passed"] and row["graph_relative_volume_error"] < 1e-7
                    and row["graph_pair_distance_max_abs_A"] < 1e-6,
                    "stored graph differs from actual CIF after the declared Niggli transform")

            row["stage"] = "proposal_dataset_fp32"
            proposal = checker.ProposalDataset([graph], checker.Data, seed_from_graph_field="refiner_seed")[0]
            require(int(proposal.sample_idx.reshape(-1)[0]) == ordinal, "ProposalDataset changed original ordinal")
            expected_seed = (20260905+int(source["sample_idx"])) % (2**32)
            require(int(proposal.refiner_seed.reshape(-1)[0]) == int(graph["refiner_seed"]) == expected_seed,
                    "refiner seed changed after graph packing")
            require(proposal.num_nodes == original.num_sites
                    and proposal.atom_types.reshape(-1).tolist() == np.asarray(graph["a_type"]).reshape(-1).tolist(),
                    "ProposalDataset changed ordered graph atoms")
            fp32 = {}
            for field, graph_key in (("lengths", "length"), ("angles", "angle"), ("frac_coords", "x_coord")):
                actual = getattr(proposal, field).detach().cpu().numpy()
                source_array = np.asarray(graph[graph_key], dtype=np.float64).reshape(actual.shape)
                require(np.array_equal(actual, source_array.astype(np.float32)) and np.isfinite(actual).all(),
                        f"ProposalDataset {field} differs from declared FP32 conversion")
                fp32[field+"_max_abs_rounding"] = float(np.abs(actual.astype(np.float64)-source_array).max())
            coords64 = np.asarray(graph["x_coord"], dtype=np.float64)
            coords32 = proposal.frac_coords.detach().cpu().numpy().astype(np.float64)
            fp32["graph_max_off_0p01_grid_in_bins"] = float(np.abs(coords64*100-np.rint(coords64*100)).max())
            fp32["input_max_off_0p01_grid_in_bins"] = float(np.abs(coords32*100-np.rint(coords32*100)).max())
            row.update(proposal_fp32=fp32, refiner_seed=expected_seed, integrity_pass=True,
                       availability="complete", stage="complete")
        except Exception as error:
            row["error"] = f"{type(error).__name__}: {error}"
        checks.append(row)

    changed = [name for name,value in hashes.items() if sha256(name) != value]
    integrity = not changed and all(row["integrity_pass"] for row in checks)
    complete = all(row["availability"] in {"complete", "generation_failure_preserved"} for row in checks)
    counts = Counter(row["availability"] for row in checks)
    return {"status": "PASS" if integrity and complete else "INCOMPLETE" if integrity else "FAIL",
            "integrity_pass": integrity, "complete_available_graph_coverage": complete,
            "run_dir": str(run), "requests": len(checks), "graphs": len(graphs), "counts": dict(counts),
            "mismatches": sum(not row["integrity_pass"] for row in checks),
            "mismatches_by_stage": dict(Counter(row["stage"] for row in checks if not row["integrity_pass"])),
            "input_sha256": hashes, "inputs_changed_during_audit": changed,
            "reused_checker": str(checker_path), "actual_process_one_source": str(paths["process_one_source"]),
            "process_one_reexecuted": False, "new_graphs_constructed": 0,
            "model_forwards": 0, "optimizer_steps": 0, "energy_evaluations": 0,
            "torch_cuda_initialized": torch.cuda.is_initialized(), "checks": checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--crysllmgen-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--expected-requests", type=int, default=256)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    if args.expected_requests < 1 or args.cpu_threads < 1:
        raise ValueError("positive population and CPU thread count are required")
    if args.output_dir.resolve().is_relative_to(args.run_dir.resolve()):
        raise ValueError("write this diagnostic outside the immutable source run")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        result = audit(args)
    except Exception as error:
        result = {"status": "ERROR", "error": f"{type(error).__name__}: {error}",
                  "traceback": traceback.format_exc(), "model_forwards": 0, "optimizer_steps": 0}
    rows = result.pop("checks", [])
    (args.output_dir/"per_request.jsonl").write_text("".join(json.dumps(row, allow_nan=False)+"\n" for row in rows), encoding="utf-8")
    (args.output_dir/"SUMMARY.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    passed = result["status"] == "PASS"
    (args.output_dir/("_SUCCESS" if passed else "_FAILED")).touch()
    print(json.dumps(result, allow_nan=False), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
