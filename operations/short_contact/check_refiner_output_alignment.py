#!/usr/bin/env python3
"""Last CPU-only check of saved refinement tensor alignment and coordinate gauges."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from pymatgen.core import Element, Lattice, Structure
from pymatgen.util.coord import pbc_shortest_vectors
from pymatgen.analysis.structure_matcher import StructureMatcher

sys.path.insert(0, str(Path(__file__).resolve().parent))
import inspect_refiner_repeatability as common


def tensor_payload(path, receipts):
    path = Path(path)
    receipts[str(path)] = common.file_sha(path)
    return torch.load(path, map_location="cpu", weights_only=False)


def same_tensor(a, b):
    return common.fingerprint(a) == common.fingerprint(b)


def ordered_atomic_numbers(structure):
    return np.asarray([Element(site["species"][0]["element"]).Z for site in structure["sites"]], dtype=int)


def inspect_run(run, graph_path, tau_path, receipts):
    graphs = common.graph_records(graph_path, receipts)
    ordered = [v[1] for v in sorted(graphs.values(), key=lambda item: item[0])]
    rows = common.read_rows(tau_path, receipts)
    by_ordinal = {int(r.get("evaluation_ordinal", r["sample_idx"])): r for r in rows.values()}
    config = common.read_json(run / "refine/run_config.json", receipts)
    metrics = common.read_json(run / "refine/refinement_metrics.json", receipts)
    merged = tensor_payload(metrics["output_file"], receipts)
    world = int(metrics.get("world_size", 1))
    rank_payloads, rank_reports = [], []
    for rank in range(world):
        rank_metrics = common.read_json(run / f"refine/refinement_metrics.rank{rank}.json", receipts) if world > 1 else metrics
        payload = tensor_payload(rank_metrics["output_file"], receipts) if world > 1 else merged
        expected = ordered[rank::world] if world > 1 else ordered
        indices = torch.tensor([int(g["sample_idx"]) for g in expected], dtype=torch.long)
        counts = torch.tensor([[int(torch.as_tensor(g["n_atom"]).view(-1)[0]) for g in expected]], dtype=torch.long)
        types = torch.cat([torch.as_tensor(g["a_type"], dtype=torch.long) for g in expected]).view(1, -1) if expected else torch.empty((1, 0), dtype=torch.long)
        rank_reports.append({"rank": rank, "expected_graphs": len(expected),
            "sample_indices_exact_expected_rank_stride": same_tensor(payload["sample_indices"].view(-1), indices),
            "num_atoms_exact_input_graphs": same_tensor(payload["num_atoms"], counts),
            "atom_types_exact_input_graph_order": same_tensor(payload["atom_types"], types),
            "sample_indices": payload["sample_indices"].view(-1).tolist()})
        rank_payloads.append(payload)
    fields = ("frac_coords", "num_atoms", "atom_types", "lengths", "angles", "sample_indices")
    merge_checks = {}
    for field in fields:
        expected = torch.cat([v[field] for v in rank_payloads], dim=0 if field == "sample_indices" else 1)
        merge_checks[field] = {"merged_sha": common.fingerprint(merged[field]),
            "concatenated_rank_sha": common.fingerprint(expected), "exact_dtype_shape_bytes": same_tensor(merged[field], expected)}
    indices = merged["sample_indices"].view(-1).tolist()
    if len(set(indices)) != len(indices):
        raise ValueError("merged payload repeats a request identity")
    offset, bindings = 0, []
    for index, ordinal in enumerate(indices):
        count = int(merged["num_atoms"][0, index])
        atom_types = merged["atom_types"][0, offset:offset + count].numpy()
        frac = merged["frac_coords"][0, offset:offset + count].numpy().astype(float) % 1.
        lengths = merged["lengths"][0, index].numpy().astype(float)
        angles = merged["angles"][0, index].numpy().astype(float)
        offset += count
        record = by_ordinal[int(ordinal)]
        stored = record.get("structure")
        parsed = common.structure_arrays(stored)
        binding = {"sample_idx_in_payload": int(ordinal), "trajectory_id": record["trajectory_id"], "num_atoms": count,
            "valid_stored_structure": parsed is not None}
        if parsed is not None:
            matrix, saved_frac, _ = parsed
            expected_matrix = Lattice.from_parameters(*lengths, *angles).matrix
            equal_species = np.array_equal(atom_types, ordered_atomic_numbers(stored))
            same_shape = frac.shape == saved_frac.shape
            delta = (saved_frac - frac + .5) % 1. - .5 if same_shape else None
            f_error = float(np.abs(delta).max()) if delta is not None else None
            l_error = float(np.abs(matrix - expected_matrix).max())
            binding.update(ordered_species_exact=bool(equal_species), same_fractional_shape=same_shape,
                max_fractional_periodic_error=f_error, max_lattice_A_error=l_error,
                stored_tau_matches_payload=bool(equal_species and same_shape and f_error <= 1e-10 and l_error <= 1e-10))
        bindings.append(binding)
    rank_ok = all(r[k] for r in rank_reports for k in ("sample_indices_exact_expected_rank_stride", "num_atoms_exact_input_graphs", "atom_types_exact_input_graph_order"))
    merged_ok = all(v["exact_dtype_shape_bytes"] for v in merge_checks.values())
    valid_bindings = [b for b in bindings if b["valid_stored_structure"]]
    mapping_ok = all(b["stored_tau_matches_payload"] for b in valid_bindings)
    return {"run": str(run), "world_size": world, "batch_size": config["batch_size"],
        "rank_checks": rank_reports, "merged_tensor_checks": merge_checks,
        "rank_input_binding_PASS": rank_ok, "merged_exact_rank_cat_PASS": merged_ok,
        "stored_valid_tau_matches_payload_PASS": mapping_ok, "valid_tau_bindings": len(valid_bindings),
        "nonvalid_tau_bindings": len(bindings) - len(valid_bindings), "payload_atom_offset_exhausted": offset == merged["atom_types"].shape[1],
        "maximum_tau_payload_fractional_error": max((b["max_fractional_periodic_error"] for b in valid_bindings), default=None),
        "maximum_tau_payload_lattice_error": max((b["max_lattice_A_error"] for b in valid_bindings), default=None)}, bindings


def translation_species_residual(old, new):
    """Anchor-seeded translation/assignment search in the old cell metric.

    This returns an achieved upper bound, not a proof of global minimum. The
    old/new cell basis is retained; cell mismatch is measured separately.
    """
    a, b = common.structure_arrays(old), common.structure_arrays(new)
    if a is None or b is None:
        return {"comparable": False, "reason": "one endpoint lacks a valid structure"}
    la, fa, _ = a
    lb, fb, _ = b
    za, zb = ordered_atomic_numbers(old), ordered_atomic_numbers(new)
    if len(za) != len(zb) or sorted(za.tolist()) != sorted(zb.tolist()):
        return {"comparable": False, "reason": "composition differs"}
    lattice = Lattice(la)
    inverse = np.linalg.inv(la)
    groups = [(np.flatnonzero(za == z), np.flatnonzero(zb == z)) for z in sorted(set(za.tolist()))]
    anchor_old, anchor_new = min(groups, key=lambda pair: (len(pair[0]), int(za[pair[0][0]])))
    translations = [np.zeros(3)] + [(fa[anchor_old[0]] - fb[j]) % 1. for j in anchor_new]
    best = None
    initial_rms = None
    for seed_index, starting in enumerate(translations):
        shift = starting.copy()
        for iteration in range(8):
            vectors, costs = pbc_shortest_vectors(lattice, fa, fb + shift, return_d2=True)
            assignment = np.empty(len(za), dtype=int)
            for old_indices, new_indices in groups:
                rows, columns = linear_sum_assignment(costs[np.ix_(old_indices, new_indices)])
                assignment[old_indices[rows]] = new_indices[columns]
            residual = vectors[np.arange(len(za)), assignment]
            rms = float(np.sqrt(np.mean(np.sum(residual ** 2, axis=1))))
            if seed_index == 0 and iteration == 0:
                initial_rms = rms
            if best is None or rms < best["rms_A"]:
                fractional = residual @ inverse
                best = {"rms_A": rms, "max_A": float(np.linalg.norm(residual, axis=1).max()),
                    "fractional_rms": float(np.sqrt(np.mean(fractional ** 2))),
                    "translation_fractional": shift.tolist(), "new_site_for_each_old_site": assignment.tolist(),
                    "changed_correspondences": int(np.sum(assignment != np.arange(len(za)))),
                    "search_seed_index": seed_index, "search_iteration": iteration}
            correction = -residual.mean(axis=0)
            if float(np.linalg.norm(correction)) < 1e-10:
                break
            shift = (shift + correction @ inverse) % 1.
    return {"comparable": True, "assignment_only_rms_A": initial_rms,
        "translation_and_species_assignment": best, "old_cell_metric": True,
        "same_cell_basis_assumed": True, "global_optimum_certified": False,
        "lattice_relative_frobenius": float(np.linalg.norm(lb - la) / max(np.linalg.norm(la), 1e-30)),
        "tested_translation_seeds": len(translations), "iterations_per_seed_max": 8}


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
    receipts, alignment, payload_bindings, gauges, summaries = {}, {}, {}, [], {}
    matcher_settings = dict(ltol=.2, stol=.3, angle_tol=5, primitive_cell=False, scale=False,
                            attempt_supercell=False, allow_subset=False)
    matcher = StructureMatcher(**matcher_settings)
    for name, native_id, tau_id in (("k4", 39910, 39910), ("k8", 39945, 39948)):
        old_native = args.candidate_root / f"runs/spad_state_eval_method_{native_id}"
        old_tau = args.candidate_root / f"runs/spad_state_tau800_{tau_id}"
        new_run = args.contact_run / name
        for tag, run, graph_path, tau_path in (
            (name + "_old", old_tau, old_native / "native/proposal_graphs.pt", old_tau / "tau800/paths.jsonl"),
            (name + "_new", new_run, new_run / "native/proposal_graphs.pt", new_run / "tau800/paths.jsonl")):
            alignment[tag], payload_bindings[tag] = inspect_run(run, graph_path, tau_path, receipts)
        old_rows = common.read_rows(old_native / "native/paths.jsonl", receipts)
        new_rows = common.read_rows(new_run / "native/paths.jsonl", receipts)
        old_refined = common.read_rows(old_tau / "tau800/paths.jsonl", receipts)
        new_refined = common.read_rows(new_run / "tau800/paths.jsonl", receipts)
        if not (set(old_rows) == set(new_rows) == set(old_refined) == set(new_refined)):
            raise ValueError("request identities differ")
        selected = [key for key in sorted(old_rows, key=lambda k: old_rows[k]["sample_idx"])
                    if old_rows[key].get("body") == new_rows[key].get("body")]
        matcher_count = 0
        for key in selected:
            left, right = old_refined[key].get("structure"), new_refined[key].get("structure")
            result = translation_species_residual(left, right)
            row = {"method": name, "trajectory_id": key, "sample_idx": old_rows[key]["sample_idx"], **result}
            if result["comparable"] and matcher_count < 8:
                try:
                    a, b = Structure.from_dict(left), Structure.from_dict(right)
                    rms = matcher.get_rms_dist(a, b)
                    row["fixed_structure_matcher"] = {"settings": matcher_settings, "matched": rms is not None,
                        "rms_normalized_if_matched": float(rms[0]) if rms is not None else None,
                        "max_normalized_if_matched": float(rms[1]) if rms is not None else None}
                except Exception as error:
                    row["fixed_structure_matcher"] = {"settings": matcher_settings, "error": str(error)}
                matcher_count += 1
            gauges.append(row)
        valid = [g for g in gauges if g["method"] == name and g["comparable"]]
        values = [g["translation_and_species_assignment"]["rms_A"] for g in valid]
        fractional = [g["translation_and_species_assignment"]["fractional_rms"] for g in valid]
        summaries[name] = {"same_body_requests": len(selected), "gauge_comparable": len(valid),
            "after_translation_species_assignment_rms_A": {"median": float(np.median(values)), "p90": float(np.quantile(values, .9)), "max": max(values)},
            "after_translation_species_assignment_fractional_rms": {"median": float(np.median(fractional)), "p90": float(np.quantile(fractional, .9)), "max": max(fractional)},
            "achieved_rms_A_below_1e_6": sum(v < 1e-6 for v in values),
            "achieved_rms_A_below_1e_3": sum(v < 1e-3 for v in values),
            "fixed_matcher_first_valid_pairs": matcher_count,
            "fixed_matcher_matches": sum(g.get("fixed_structure_matcher", {}).get("matched", False) for g in valid),
            "matcher_subset_selection": "first eight valid same-body requests in original sample_idx order; no energy/outcome selection"}
        print(json.dumps({"method_complete": name, "same_body_pairs": len(selected)}), flush=True)
    flags = ("rank_input_binding_PASS", "merged_exact_rank_cat_PASS", "stored_valid_tau_matches_payload_PASS", "payload_atom_offset_exhausted")
    passed = all(run[k] for run in alignment.values() for k in flags)
    report = {"status": "PASS" if passed else "FAIL", "alignment_checks": alignment, "gauge_summary": summaries,
        "source_sha256": receipts, "script_sha256": common.file_sha(__file__),
        "helper_sha256": common.file_sha(Path(common.__file__)), "model_forwards": 0, "CUDA_calls": 0,
        "energy_evaluations": 0, "optimizer_steps": 0,
        "limits": "Translation/species search yields an achieved residual upper bound in the old cell basis, not a globally certified structural distance. First-eight StructureMatcher has fixed geometric tolerances and does not measure SUN or energy."}
    (args.output_dir / "OUTPUT_ALIGNMENT.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (args.output_dir / "PAYLOAD_BINDINGS.json").write_text(json.dumps(payload_bindings, indent=2, allow_nan=False) + "\n")
    (args.output_dir / "SITE_GAUGE_CASES.json").write_text(json.dumps(gauges, indent=2, allow_nan=False) + "\n")
    if passed:
        (args.output_dir / "_SUCCESS").touch()
    print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
