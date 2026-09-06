#!/usr/bin/env python3
"""CPU-only comparison of already saved native graphs and repeated tau800 outputs."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

import numpy as np
import torch


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path, receipts, required=True):
    path = Path(path)
    if not path.is_file():
        if required:
            raise FileNotFoundError(path)
        return None
    raw = path.read_bytes()
    receipts[str(path)] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def read_rows(path, receipts):
    raw = Path(path).read_bytes()
    receipts[str(path)] = hashlib.sha256(raw).hexdigest()
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    result = {r["trajectory_id"]: r for r in rows}
    if len(rows) != 256 or len(result) != 256:
        raise ValueError(f"expected all256 unique request records: {path}")
    return result


def graph_records(path, receipts):
    raw = Path(path).read_bytes()
    receipts[str(path)] = hashlib.sha256(raw).hexdigest()
    graphs = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
    result = {}
    for index, graph in enumerate(graphs):
        sample_idx = int(torch.as_tensor(graph["sample_idx"]).reshape(-1)[0])
        if sample_idx in result:
            raise ValueError("duplicate graph sample_idx")
        result[sample_idx] = (index, graph)
    return result


def array(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def fingerprint(value):
    value = np.ascontiguousarray(array(value))
    header = json.dumps({"dtype": str(value.dtype), "shape": list(value.shape)}, sort_keys=True).encode()
    return {"dtype": str(value.dtype), "shape": list(value.shape),
        "sha256": hashlib.sha256(header + b"\0" + value.tobytes()).hexdigest()}


def array_pair(left, right):
    a, b = array(left), array(right)
    result = {"old": fingerprint(a), "new": fingerprint(b), "same_shape": a.shape == b.shape,
              "same_dtype": a.dtype == b.dtype}
    result["exact_values_same"] = a.shape == b.shape and bool(np.array_equal(a, b))
    result["dtype_shape_bytes_same"] = result["old"] == result["new"]
    if a.shape == b.shape and np.issubdtype(a.dtype, np.number) and np.issubdtype(b.dtype, np.number):
        finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
        result["finite"] = finite
        if finite:
            delta = b.astype(np.float64) - a.astype(np.float64)
            result["max_abs_difference"] = float(np.abs(delta).max()) if delta.size else 0.
            result["rms_difference"] = float(np.sqrt(np.square(delta).mean())) if delta.size else 0.
    return result


def effective_inputs(graph):
    # Exact casts/shapes used by ProposalDataset. CrystalNN edges are audited
    # separately; the vendored fc decoder does not consume those edge fields.
    return {"num_atoms": torch.tensor([int(torch.as_tensor(graph["n_atom"]).view(-1)[0])], dtype=torch.long),
        "lengths": torch.as_tensor(graph["length"], dtype=torch.float32).view(1, 3),
        "angles": torch.as_tensor(graph["angle"], dtype=torch.float32).view(1, 3),
        "frac_coords": torch.as_tensor(graph["x_coord"], dtype=torch.float32),
        "atom_types": torch.as_tensor(graph["a_type"], dtype=torch.long)}


def sample_seed(graph, graph_index, config):
    world = int(config.get("world_size", 1))
    rank = graph_index % world if bool(config.get("distributed", world > 1)) else 0
    base = int(config.get("seed", 27017))
    name = config.get("seed_from_graph_field")
    if name:
        effective = int(torch.as_tensor(graph[name]).reshape(-1)[0])
        mode = "graph_field_per_sample"
    elif config.get("seed_by_sample_index"):
        effective = base + int(graph["sample_idx"]) * max(1, int(config.get("num_evals", 1)))
        mode = "sample_index_per_sample"
    else:
        effective = None
        mode = "rank_stream_sample_noise_depends_on_prior_consumption"
    return {"mode": mode, "graph_seed_field": name, "effective_eval0_seed": effective,
        "base_seed": base, "rank_seed": base + rank, "graph_list_index": graph_index,
        "assigned_rank": rank, "world_size": world, "num_evals": config.get("num_evals", 1),
        "batch_size": config.get("batch_size"), "sample_idx": int(graph["sample_idx"])}


def structure_arrays(value):
    if not isinstance(value, dict) or not isinstance(value.get("lattice"), dict) or not isinstance(value.get("sites"), list) or not value["sites"]:
        return None
    try:
        lattice = np.asarray(value["lattice"]["matrix"], dtype=float)
        coords = np.asarray([s["abc"] for s in value["sites"]], dtype=float)
        species = [s["species"] for s in value["sites"]]
        if lattice.shape != (3, 3) or coords.shape != (len(species), 3):
            return None
        if not np.isfinite(lattice).all() or not np.isfinite(coords).all() or abs(float(np.linalg.det(lattice))) <= 0:
            return None
        return lattice, coords, species
    except (KeyError, TypeError, ValueError):
        return None


def compare_structures(left, right):
    a, b = structure_arrays(left), structure_arrays(right)
    result = {"old_valid_dict": a is not None, "new_valid_dict": b is not None,
        "both_missing": left is None and right is None,
        "exact_valid_dict_same": a is not None and b is not None and left == right,
        "comparison": "same stored site order and basis; componentwise torus difference, not StructureMatcher or atom reassignment"}
    if a is None or b is None:
        return result
    la, fa, sa = a
    lb, fb, sb = b
    result.update(old_num_atoms=len(sa), new_num_atoms=len(sb), same_ordered_species=sa == sb,
        lattice_max_abs_A=float(np.abs(lb - la).max()),
        lattice_relative_frobenius=float(np.linalg.norm(lb - la) / max(np.linalg.norm(la), 1e-30)),
        gram_max_abs_A2=float(np.abs(lb @ lb.T - la @ la.T).max()))
    if fa.shape == fb.shape and sa == sb:
        delta = (fb - fa + .5) % 1. - .5
        result.update(frac_periodic_max_abs=float(np.abs(delta).max()),
            frac_periodic_rms=float(np.sqrt(np.square(delta).mean())),
            periodic_displacement_in_old_cell_max_A=float(np.linalg.norm(delta @ la, axis=-1).max()),
            numeric_lattice_exact=bool(np.array_equal(la, lb)),
            numeric_fractional_exact=bool(np.array_equal(fa, fb)))
    return result


def evidence_for_path(value, target):
    found = []
    if isinstance(value, dict):
        if str(value.get("path", "")) == str(target) and isinstance(value.get("sha256"), str):
            found.append(value["sha256"])
        for child in value.values():
            found.extend(evidence_for_path(child, target))
    elif isinstance(value, list):
        for child in value:
            found.extend(evidence_for_path(child, target))
    return sorted(set(found))


def provenance(run, code_root, extra_manifest, receipts, external_cache):
    config = read_json(run / "refine/run_config.json", receipts)
    metrics = read_json(run / "refine/refinement_metrics.json", receipts)
    result = {"run": str(run), "run_config": config, "metrics": metrics,
              "frozen_code_root": str(code_root)}
    commit = code_root.parent / "CODE_COMMIT"
    result["recorded_code_commit"] = commit.read_text().strip() if commit.is_file() else None
    if commit.is_file():
        receipts[str(commit)] = file_sha(commit)
    manifest = read_json(extra_manifest, receipts, required=False) if extra_manifest else None
    files = ["src/scripts/refine_dlm_with_crysllmgen.py", "scripts/export_programmed_path_artifacts.py",
        "src/scripts/sample_llada_dynamic_crystals.py", "scripts/assemble_grounding_repeat.py"]
    result["frozen_source_sha256"] = {name: file_sha(code_root / name) if (code_root / name).is_file() else None for name in files}
    external = Path(config["crysllmgen_dir"])
    paths = [external / name for name in ("config.py", "data_utils.py", "models_ddpm/diffusion.py",
        "models_ddpm/cspnet.py", "models_ddpm/diff_utils.py", "models_ddpm/data_utils.py")]
    paths.append(Path(config["checkpoint"]))
    fingerprints = {}
    for path in paths:
        key = str(path)
        if key not in external_cache:
            external_cache[key] = {"exists_now": path.is_file(),
                "sha256_observed_now": file_sha(path) if path.is_file() else None}
        fingerprints[key] = {**external_cache[key], "historical_manifest_sha256": evidence_for_path(manifest, path),
            "historical_execution_bytes_proven": False}
        fingerprints[key]["current_matches_recorded_manifest"] = bool(fingerprints[key]["historical_manifest_sha256"]) and fingerprints[key]["sha256_observed_now"] in fingerprints[key]["historical_manifest_sha256"]
    result["external_dependencies"] = fingerprints
    result["external_hash_limit"] = "Current mutable CRYS paths cannot establish old executed bytes. Historical hashes are reported only when explicitly present in that run's manifest; a vendored copy is not proof it was imported."
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate-root", type=Path, required=True)
    p.add_argument("--contact-run", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    if not (args.contact_run / "_SUCCESS").is_file():
        raise ValueError("contact run is incomplete")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    receipts, external_cache, methods, cases = {}, {}, {}, []
    for name, native_id, tau_id in (("k4", 39910, 39910), ("k8", 39945, 39948)):
        old_native = args.candidate_root / f"runs/spad_state_eval_method_{native_id}"
        old_tau = args.candidate_root / f"runs/spad_state_tau800_{tau_id}"
        current = args.contact_run / name
        a = read_rows(old_native / "native/paths.jsonl", receipts)
        b = read_rows(current / "native/paths.jsonl", receipts)
        ta = read_rows(old_tau / "tau800/paths.jsonl", receipts)
        tb = read_rows(current / "tau800/paths.jsonl", receipts)
        if not (set(a) == set(b) == set(ta) == set(tb)):
            raise ValueError("original and repeated request identities differ")
        ga = graph_records(old_native / "native/proposal_graphs.pt", receipts)
        gb = graph_records(current / "native/proposal_graphs.pt", receipts)
        old_info = provenance(old_tau, old_tau / "code", old_tau / "LAUNCH_MANIFEST.json", receipts, external_cache)
        new_info = provenance(current, args.contact_run / "code", args.contact_run / "LAUNCH_MANIFEST.json", receipts, external_cache)
        summary = Counter()
        for key in a:
            if a[key].get("body") != b[key].get("body"):
                continue
            summary["same_native_body_requests"] += 1
            ordinal_a = int(a[key].get("evaluation_ordinal", a[key]["sample_idx"]))
            ordinal_b = int(b[key].get("evaluation_ordinal", b[key]["sample_idx"]))
            if ordinal_a != ordinal_b or a[key]["sample_idx"] != b[key]["sample_idx"]:
                raise ValueError("same request changed evaluation/original sample index")
            entry = {"method": name, "trajectory_id": key, "sample_idx": a[key]["sample_idx"],
                "evaluation_ordinal": ordinal_a, "native_body_present": isinstance(a[key].get("body"), str),
                "native": compare_structures(a[key].get("structure"), b[key].get("structure")),
                "tau800": compare_structures(ta[key].get("structure"), tb[key].get("structure")),
                "old_graph_present": ordinal_a in ga, "new_graph_present": ordinal_a in gb}
            summary["native_exact_valid_dict_same"] += int(entry["native"]["exact_valid_dict_same"])
            summary["tau_exact_valid_dict_same"] += int(entry["tau800"]["exact_valid_dict_same"])
            summary["tau_both_missing"] += int(entry["tau800"]["both_missing"])
            summary["tau_both_valid_dicts"] += int(entry["tau800"]["old_valid_dict"] and entry["tau800"]["new_valid_dict"])
            if ordinal_a in ga and ordinal_a in gb:
                summary["both_graphs_present"] += 1
                ia, old_graph = ga[ordinal_a]
                ib, new_graph = gb[ordinal_a]
                raw_fields = ("n_atom", "length", "angle", "x_coord", "a_type", "edge_indices", "to_jimages", "sample_idx", "refiner_seed")
                entry["raw_graph_fields"] = {field: array_pair(old_graph[field], new_graph[field]) for field in raw_fields if field in old_graph and field in new_graph}
                entry["raw_graph_missing_fields"] = {"old": [f for f in raw_fields if f not in old_graph], "new": [f for f in raw_fields if f not in new_graph]}
                ea, eb = effective_inputs(old_graph), effective_inputs(new_graph)
                entry["effective_model_inputs"] = {field: array_pair(ea[field], eb[field]) for field in ea}
                same_inputs = all(v["dtype_shape_bytes_same"] for v in entry["effective_model_inputs"].values())
                sa = sample_seed(old_graph, ia, old_info["run_config"])
                sb = sample_seed(new_graph, ib, new_info["run_config"])
                same_seed = sa["effective_eval0_seed"] is not None and sa["effective_eval0_seed"] == sb["effective_eval0_seed"]
                entry.update(old_seed=sa, new_seed=sb, effective_inputs_identical=same_inputs, effective_eval0_seed_equal=same_seed)
                summary["effective_inputs_identical"] += int(same_inputs)
                summary["effective_eval0_seed_equal"] += int(same_seed)
                summary["effective_inputs_and_seed_identical"] += int(same_inputs and same_seed)
                summary["assigned_rank_changed"] += int(sa["assigned_rank"] != sb["assigned_rank"])
            else:
                summary["both_graphs_missing"] += int(ordinal_a not in ga and ordinal_a not in gb)
                summary["one_graph_missing"] += int((ordinal_a in ga) != (ordinal_a in gb))
            cases.append(entry)
        selected_cases = [c for c in cases if c["method"] == name]
        differences = {}
        subsets = {"all_same_native_body": selected_cases,
            "same_effective_model_inputs_and_seed": [c for c in selected_cases if c.get("effective_inputs_identical") and c.get("effective_eval0_seed_equal")]}
        for subset_name, subset in subsets.items():
            tau_deltas = {}
            for field in ("lattice_max_abs_A", "lattice_relative_frobenius", "gram_max_abs_A2", "frac_periodic_max_abs", "frac_periodic_rms", "periodic_displacement_in_old_cell_max_A"):
                values = [c["tau800"][field] for c in subset if field in c["tau800"]]
                tau_deltas[field] = {"coverage": len(values), "median": float(np.median(values)) if values else None,
                    "p90": float(np.quantile(values, .9)) if values else None, "max": max(values) if values else None}
            differences[subset_name] = {"requests": len(subset), "tau_same_index_differences": tau_deltas}
        config_fields = ("timesteps", "run_type", "diff_steps", "num_evals", "batch_size", "seed", "seed_from_graph_field", "seed_by_sample_index", "checkpoint", "crysllmgen_dir", "world_size")
        config_comparison = {field: {"old": old_info["run_config"].get(field), "new": new_info["run_config"].get(field),
            "equal": old_info["run_config"].get(field) == new_info["run_config"].get(field)} for field in config_fields}
        methods[name] = {"old": old_info, "new": new_info, "summary": dict(summary),
            "configuration_comparison": config_comparison, "difference_groups": differences}
    versions = {}
    for package in ("torch-geometric", "torch-scatter", "pymatgen", "spglib"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    report = {"status": "PASS", "scope": "saved-data and provenance comparison only",
        "methods": methods, "source_sha256": receipts, "diagnostic_script_sha256": file_sha(__file__),
        "runtime_observed_now_only": {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__, "numpy": np.__version__, "packages": versions},
        "model_forwards": 0, "CUDA_calls": 0, "energy_evaluations": 0, "optimizer_steps": 0,
        "structural_comparison_limit": "Stored site order and cell basis only; not a structure matcher or optimal periodic site assignment.",
        "causal_limit": "Equal seed does not certify identical numerical execution. This inspection reports actual inputs/configurations and endpoint differences, without declaring a cause absent execution evidence."}
    (args.output_dir / "REPEATABILITY.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (args.output_dir / "CASES.json").write_text(json.dumps(cases, indent=2, allow_nan=False) + "\n")
    (args.output_dir / "_SUCCESS").touch()
    print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
