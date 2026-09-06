#!/usr/bin/env python3
"""CPU-only measurements from the supplied actual native/Q/refined casepack."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from pymatgen.core import Lattice, Structure

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/"src"))
from crystal_dlm.dynamic_crystal import arrays_to_dynamic_tokens, arrays_to_structure, parse_dynamic_answer
from crystal_dlm.mixed_geometry_diffusion import LatticeNormalizer
from crystal_dlm.periodic_base_training_data import _canonical_tokens


def structure(value):
    if value is None:
        return None
    species = []
    for site in value["sites"]:
        if len(site["species"]) != 1 or site["species"][0]["occu"] != 1:
            raise ValueError("casepack is not an ordered generated structure")
        species.append(site["species"][0]["element"])
    return Structure(Lattice(value["lattice"]), species, [site["abc"] for site in value["sites"]])


def minimum_distance(s):
    shifts = np.stack(np.meshgrid(*([np.arange(-2, 3)]*3), indexing="ij"), axis=-1).reshape(-1, 3)
    lattice, f = s.lattice.matrix, s.frac_coords
    minimum = float(np.linalg.norm(shifts[np.any(shifts != 0, axis=1)]@lattice, axis=-1).min())
    if len(f) > 1:
        delta = f[:, None]-f[None, :]
        delta -= np.rint(delta)
        distances = np.linalg.norm((delta[:, :, None]+shifts)@lattice, axis=-1).min(-1)
        np.fill_diagonal(distances, np.inf)
        minimum = min(minimum, float(distances.min()))
    return minimum


def describe(values):
    supplied = list(values)
    a = np.asarray([float(v) for v in supplied if v is not None and np.isfinite(float(v))])
    return {"n": int(len(a)), "missing_or_nonfinite": len(supplied)-len(a),
            **({"mean": float(a.mean()), "median": float(np.median(a)),
                "p10": float(np.quantile(a, .1)), "p90": float(np.quantile(a, .9)),
                "min": float(a.min()), "max": float(a.max())} if len(a) else {})}


def get(row, path):
    for part in path.split("."):
        row = row.get(part) if isinstance(row, dict) else None
    return row


def pairs(reference, method, field):
    a = np.asarray([bool(reference[k]["attempt"][field]) for k in sorted(reference)])
    b = np.asarray([bool(method[k]["attempt"][field]) for k in sorted(reference)])
    return {"reference": int(a.sum()), "method": int(b.sum()), "both": int((a&b).sum()),
            "reference_only": int((a&~b).sum()), "method_only": int((~a&b).sum()), "neither": int((~a&~b).sum())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--casepack", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.casepack.read_text(encoding="utf-8"))
    endpoints = {name: {int(row["sample_idx"]): row for row in value["rows"]} for name,value in data["endpoints"].items()}
    reference = endpoints["native"]
    if any(set(rows) != set(range(256)) for rows in endpoints.values()):
        raise ValueError("casepack changed the full256 request ledger")
    normalizer = LatticeNormalizer.from_dict(data["training_config"]["normalizer"])
    torch.set_num_threads(2)
    measured = []
    for index in sorted(reference):
        original = reference[index]
        others = [rows[index] for rows in endpoints.values()]
        if any(any(row[k] != original[k] for k in ("trajectory_id", "group_id", "sampling_seed", "plan_state", "species_program"))
               for row in others):
            raise ValueError(f"endpoint identity changed at {index}")
        native, q = structure(original["structure"]), structure(endpoints["quantized"][index]["structure"])
        if native is None or q is None:
            raise ValueError("this supplied casepack advertises complete native/Q geometry")
        symbols = [str(site.specie) for site in native]
        tokens, diagnostic = arrays_to_dynamic_tokens(native.lattice.abc, native.lattice.angles, symbols, native.frac_coords)
        tokens, aliases = _canonical_tokens(tokens)
        expected_q = arrays_to_structure(parse_dynamic_answer("".join(tokens), strict=True))
        if list(expected_q.atomic_numbers) != list(q.atomic_numbers):
            raise ValueError("quantized output changed canonical ordered species")
        q_gram_error = float(np.abs(expected_q.lattice.metric_tensor-q.lattice.metric_tensor).max())
        q_gram_relative = q_gram_error/max(1.,float(np.abs(expected_q.lattice.metric_tensor).max()))
        q_coordinates = expected_q.frac_coords-q.frac_coords
        q_coordinates -= np.rint(q_coordinates)
        q_coordinate_error = float(np.abs(q_coordinates).max())
        delta = native.frac_coords-q.frac_coords
        delta -= np.rint(delta)
        prior = original["initial_geometry_prior"]
        if prior["num_atoms"] != len(native) or int(prior["seed"]) != int(original["sampling_seed"]):
            raise ValueError("stored prior identity differs")
        prior_z = torch.tensor(prior["z"], dtype=torch.float64)
        prior_lattice = normalizer.decode(prior_z, len(native)).numpy()
        prior_structure = Structure(prior_lattice, symbols, prior["fractional"])
        final_z = normalizer.encode(torch.from_numpy(native.lattice.matrix), len(native)).numpy()
        fractional_move = native.frac_coords-np.asarray(prior["fractional"])
        fractional_move -= np.rint(fractional_move)
        actual_minimum = minimum_distance(native)
        original_gram = native.lattice.metric_tensor
        q_real_delta = q.lattice.metric_tensor-original_gram
        measured.append({"sample_idx": index, "N": len(native), "trajectory_id": original["trajectory_id"],
            "native_status": original["label_details"]["status"], "q_status": endpoints["quantized"][index]["label_details"]["status"],
            "native_strict_sun": original["attempt"]["strict_sun"], "native_meta_sun": original["attempt"]["meta_sun"],
            "q_expected_gram_relative_error": q_gram_relative, "q_expected_coordinate_error": q_coordinate_error,
            "q_matches_fixed_codec": q_gram_relative < 1e-10 and q_coordinate_error < 1e-12,
            "quantization": asdict(diagnostic), "periodic_aliases": aliases,
            "q_fractional_component_max_change": float(np.abs(delta).max()),
            "q_fractional_rms_change": float(np.sqrt(np.mean(delta**2))),
            "q_relative_gram_change": float(np.abs(q_real_delta).max()/max(1.,float(np.abs(original_gram).max()))),
            "q_relative_volume_change": q.volume/native.volume-1.,
            "native_volume_per_atom_A3": native.volume/len(native),
            "prior_volume_per_atom_A3": prior_structure.volume/len(native),
            "native_over_prior_volume": native.volume/prior_structure.volume,
            "native_min_distance_A": actual_minimum, "q_min_distance_A": minimum_distance(q),
            "prior_min_distance_A": minimum_distance(prior_structure),
            "native_vs_recorded_min_distance_error_A": (actual_minimum-original["label_details"]["raw_min_distance_A"]
                                                        if original["label_details"]["raw_min_distance_A"] is not None else None),
            "native_lattice_condition_number": float(np.linalg.cond(native.lattice.matrix)),
            "native_min_lattice_singular_value_A": float(np.linalg.svd(native.lattice.matrix, compute_uv=False).min()),
            "net_fractional_rms_from_prior": float(np.sqrt(np.mean(fractional_move**2))),
            "final_z": final_z.tolist(), "prior_z": prior["z"],
            "raw_energy_eV_atom": original["attempt"]["raw_energy_eV_atom"],
            "raw_force_max_eV_A": get(original,"attempt.raw.force_max_eV_A"),
            "raw_stress_max_GPa": get(original,"attempt.raw.stress_max_GPa")})
    endpoint_summaries = {}
    for name,rows in endpoints.items():
        endpoint_summaries[name] = {"counts": data["endpoints"][name]["report"]["counts"],
            "statuses": dict(Counter(row["label_details"]["status"] for row in rows.values())),
            "headline_by_status": {status: {key: sum(row["label_details"]["status"] == status and row["attempt"][key] for row in rows.values())
                                              for key in ("strict_sun", "meta_sun")}
                                   for status in sorted({row["label_details"]["status"] for row in rows.values()})},
            **{key: describe(get(row,"attempt."+path) for row in rows.values()) for key,path in
               (("raw_energy_eV_atom","raw_energy_eV_atom"),("raw_force_max_eV_A","raw.force_max_eV_A"),
                ("raw_stress_max_GPa","raw.stress_max_GPa"),("gap_eV_atom","gap_eV_atom"))}}
    summary = {"casepack_sha256": hashlib.sha256(args.casepack.read_bytes()).hexdigest(),
               "source_files_sha256": data["files"], "requests": 256, "endpoints": endpoint_summaries,
               "quantized_matches_fixed_codec_count": sum(row["q_matches_fixed_codec"] for row in measured),
               "native_vs_q_binary": {key:pairs(reference,endpoints["quantized"],key)
                                        for key in ("strict_sun","meta_sun","terminal_verified")},
               "native_vs_tau_binary": {key:pairs(reference,endpoints["tau800"],key)
                                        for key in ("strict_sun","meta_sun","terminal_verified","novel_unique")},
               "geometry": {key: describe(row[key] for row in measured) for key in measured[0]
                            if key not in {"sample_idx","N","trajectory_id","native_status","q_status","native_strict_sun",
                                           "native_meta_sun","q_matches_fixed_codec","quantization","final_z","prior_z"}},
               "support_counts": {key: sum(row[key] < .5-1e-8 for row in measured)
                                  for key in ("prior_min_distance_A","native_min_distance_A","q_min_distance_A")},
               "native_by_N": {f"{lo}-{hi}": {"requests":sum(lo<=r["N"]<=hi for r in measured),
                                   "strict_sun":sum(lo<=r["N"]<=hi and r["native_strict_sun"] for r in measured),
                                   "meta_sun":sum(lo<=r["N"]<=hi and r["native_meta_sun"] for r in measured),
                                   "force":describe(r["raw_force_max_eV_A"] for r in measured if lo<=r["N"]<=hi)}
                               for lo,hi in ((1,5),(6,10),(11,15),(16,20))},
               "limits": "casepack has no sample-before-export structure, actual CIF files, saved graph or intermediate field; these checks do not certify that chain or assign solver causality",
               "model_forwards":0,"optimizer_steps":0,"new_energy_evaluations":0}
    args.output_dir.mkdir(parents=True,exist_ok=False)
    (args.output_dir/"SUMMARY.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    (args.output_dir/"per_request.jsonl").write_text("".join(json.dumps(row,allow_nan=False)+"\n" for row in measured),encoding="utf-8")
    print(json.dumps({"requests":256,"quantized_matches_fixed_codec_count":summary["quantized_matches_fixed_codec_count"],
                      "support_counts":summary["support_counts"],"output_dir":str(args.output_dir)},indent=2))


if __name__ == "__main__":
    main()
