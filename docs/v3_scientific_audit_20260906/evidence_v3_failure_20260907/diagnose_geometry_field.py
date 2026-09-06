"""Read-only analysis of saved 40066 outputs and source-keyed zero-field risks.

No neural forward, sampling, training, MLIP or network call is made.  The optional
zero-target reconstruction uses the production CPU noise/target functions only.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import itertools
import json
import math
from pathlib import Path
import sys

import numpy as np


def summary(values):
    values = np.asarray([value for value in values if value is not None and math.isfinite(value)], dtype=float)
    if not values.size:
        return {"count": 0}
    return {"count": int(values.size), "mean": float(values.mean()),
            **dict(zip(("min", "p10", "median", "p90", "max"), np.quantile(values, [0, .1, .5, .9, 1]).tolist()))}


def basis():
    result = [np.diag([1., -1., 0.]) / math.sqrt(2), np.diag([1., 1., -2.]) / math.sqrt(6)]
    for a, b in ((0, 1), (0, 2), (1, 2)):
        value = np.zeros((3, 3)); value[a, b] = value[b, a] = 1 / math.sqrt(2)
        result.append(value)
    return np.asarray([*result, np.eye(3) / math.sqrt(3)])


def encode(lattice, n, normalizer):
    unit = normalizer["reference_length"]
    values, vectors = np.linalg.eigh(lattice @ lattice.T / unit**2)
    if np.min(values) <= 0:
        raise ValueError("non-SPD saved metric")
    half_log = (vectors * (0.5 * np.log(values))) @ vectors.T
    y = np.einsum("ab,jab->j", half_log, basis())
    y[5] -= math.log(n) / math.sqrt(3)
    return (y - np.asarray(normalizer["mean"])) / np.asarray(normalizer["std"])


def decode(z, n, normalizer):
    coefficients = np.asarray(normalizer["mean"]) + np.asarray(normalizer["std"]) * z
    coefficients[5] += math.log(n) / math.sqrt(3)
    half_log = np.einsum("j,jab->ab", coefficients, basis())
    values, vectors = np.linalg.eigh(half_log)
    metric = (vectors * np.exp(2 * values)) @ vectors.T * normalizer["reference_length"]**2
    return np.linalg.cholesky(metric)


def structure_arrays(record):
    lattice = record["lattice"]
    if isinstance(lattice, dict):
        lattice = lattice["matrix"]
    return np.asarray(lattice, dtype=float), np.asarray([site["abc"] for site in record["sites"]], dtype=float)


def shell_minimum(lattice, fractional):
    shifts = np.asarray(list(itertools.product(range(-2, 3), repeat=3)), dtype=float)
    delta = fractional[:, None] - fractional[None, :]
    delta -= np.rint(delta)
    vectors = (delta[:, :, None] + shifts) @ lattice
    distances = np.linalg.norm(vectors, axis=-1)
    index = np.arange(len(fractional))
    zero = int(np.flatnonzero((shifts == 0).all(-1))[0])
    distances[index, index, zero] = np.inf
    return float(distances.min())


def analyze(bundle):
    config = bundle["training_config"]
    normalizer = config["normalizer"]
    epsilon = config["diffusion_config"]["epsilon"]
    steps = config["diffusion_config"]["euler_steps"]
    times = config["monitor_fixed_times"]
    validation = []
    for event in bundle["validation"]:
        bands = []
        for i, time in enumerate(times):
            prefix = f"t_band_{i}_geometry_"
            weight = event[prefix + "weight_sum"]
            v_mse, u_mse = event[prefix + "lattice_sum"] / weight, event[prefix + "coordinates_sum"] / weight
            bands.append({"band": i, "time": time, "weight": weight, "v_MSE": v_mse, "u_MSE": u_mse,
                          "paired_z0_readout_MSE_from_v_ignoring_FP32_roundoff": math.sin(math.pi*time/2)**2 * v_mse})
        validation.append({"epoch": event["epoch"], "step": event["step"], "bands": bands,
                           "token_risk": event["token_loss_sum"] / event["token_source_weight_sum"]})
    grid = [epsilon+(1-epsilon)*(j/steps)**2 for j in range(steps, -1, -1)]
    delta = np.asarray(grid[:-1]) - np.asarray(grid[1:])
    time_scale = config["model_config"]["time_input_scale"]
    records, initial_z, final_z = [], [], []
    for row in bundle["endpoints"]["native"]["rows"]:
        if row.get("structure") is None or row.get("initial_geometry_prior") is None:
            records.append({"sample_idx": row["sample_idx"], "missing_geometry_or_prior": True})
            continue
        lattice, fractional = structure_arrays(row["structure"])
        n = len(fractional)
        prior = row["initial_geometry_prior"]
        before_f, before_z = np.asarray(prior["fractional"]), np.asarray(prior["z"])
        z = encode(lattice, n, normalizer)
        before_l = decode(before_z, n, normalizer)
        shift = (fractional-before_f+.5) % 1 - .5
        centered = shift - shift.mean(0, keepdims=True)
        internal = (((fractional[:, None]-fractional[None, :])
                     - (before_f[:, None]-before_f[None, :]) + .5) % 1 - .5)
        total_square = float(np.square(shift).sum())
        attempt = row["attempt"]
        item = {"sample_idx": row["sample_idx"], "trajectory_id": row["trajectory_id"], "N": n,
                "fractional_shift_RMS": float(np.sqrt(np.mean(shift**2))),
                "fractional_max_abs_shift": float(np.abs(shift).max()),
                "centered_fractional_shift_RMS": float(np.sqrt(np.mean(centered**2))) if n > 1 else None,
                "translation_fraction_of_squared_shift": float(n*np.square(shift.mean(0)).sum()/total_square) if total_square else None,
                "pair_fractional_change_RMS": float(np.sqrt(np.mean(internal**2))) if n > 1 else None,
                "Cartesian_shift_RMS_A_using_final_cell": float(np.sqrt(np.mean(np.sum((shift@lattice)**2, -1)))),
                "chart_z_shift_RMS": float(np.sqrt(np.mean((z-before_z)**2))),
                "vpa_A3": float(np.linalg.det(lattice)/n), "prior_vpa_A3": float(np.linalg.det(before_l)/n),
                "log_volume_change": float(math.log(np.linalg.det(lattice)/np.linalg.det(before_l))),
                "cell_condition_number": float(np.linalg.cond(lattice)),
                "prior_min_distance_A_shell2": shell_minimum(before_l, before_f),
                "final_min_distance_A_shell2": shell_minimum(lattice, fractional),
                "label_raw_min_distance_A": row["label_details"].get("raw_min_distance_A"),
                "strict_sun": bool(attempt["strict_sun"]), "meta_sun": bool(attempt["meta_sun"]),
                "raw_force_max_eV_A": (attempt.get("raw") or {}).get("force_max_eV_A"),
                "raw_stress_max_GPa": (attempt.get("raw") or {}).get("stress_max_GPa"),
                "terminal_status": attempt.get("terminal_status")}
        records.append(item); initial_z.append(before_z); final_z.append(z)
    valid = [row for row in records if "N" in row]
    by_n = defaultdict(list)
    for row in valid:
        by_n[row["N"]].append(row)
    metric_names = [name for name in valid[0] if isinstance(valid[0][name], (float, int))
                    and name not in ("N", "sample_idx", "strict_sun", "meta_sun")]
    return {"scope": "Saved outputs/validation only; no neural forwards or physical model calls.",
            "training_coverage": {key: config.get(key) for key in ("epochs", "updates", "source_count", "effective_states")},
            "endpoint_counts": {name: endpoint["report"]["counts"] for name, endpoint in bundle["endpoints"].items()},
            "validation": validation,
            "time_grid_audit": {"euler_steps": steps, "epsilon": epsilon, "largest_dt": float(delta.max()),
                                "smallest_dt": float(delta.min()), "time_feature_max_frequency_rad_per_unit_t": time_scale,
                                "largest_highest_frequency_phase_change_rad": float(delta.max()*time_scale),
                                "boundary": "Large phase change does not prove learned field oscillation or solver error."},
            "geometry_summaries": {name: summary([row.get(name) for row in valid]) for name in metric_names},
            "chart_statistics": {"prior_mean": np.mean(initial_z, 0).tolist(), "prior_std": np.std(initial_z, 0).tolist(),
                                 "final_mean": np.mean(final_z, 0).tolist(), "final_std": np.std(final_z, 0).tolist()},
            "near_distance_counts": {str(threshold): {"prior": sum(row["prior_min_distance_A_shell2"] < threshold for row in valid),
                                                       "final": sum(row["final_min_distance_A_shell2"] < threshold for row in valid)}
                                     for threshold in (.5, 1., 1.5)},
            "by_N": {str(n): {"requests": len(rows), "strict_sun": sum(r["strict_sun"] for r in rows),
                                "meta_sun": sum(r["meta_sun"] for r in rows),
                                "fractional_shift_RMS": summary([r["fractional_shift_RMS"] for r in rows])}
                     for n, rows in sorted(by_n.items())},
            "rows": records,
            "limits": ["No pre-epsilon state or neural fields were saved in the input bundle.",
                       "Shell-2 distances are finite-shell diagnostics, not universally exact MIC.",
                       "Near-one denoising MSE alone cannot separate irreducible conditional variance from underlearning."]}


def zero_baseline(config, identity_path, prepared_path, max_sites, source_root):
    """Exactly replay source-keyed CPU noise/targets, without loading a model."""
    sys.path.insert(0, str(source_root / "src"))
    import torch
    from crystal_dlm.mixed_geometry_diffusion import LatticeNormalizer, MixedGeometryConfig, corrupt_geometry
    from crystal_dlm.mixed_geometry_training_data import source_seed
    normalizer = LatticeNormalizer.from_dict(config["normalizer"])
    diffusion = MixedGeometryConfig(**config["diffusion_config"])
    keys = [tuple(key) for key in config["monitor_source_keys"]]
    needed = set(keys)
    identities, weights = {}, {}
    for path, destination, kind in ((identity_path, identities, "identity"), (prepared_path, weights, "weight")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip(): continue
                row = json.loads(line)
                key = (row["source_split"], int(row["source_row_idx"]))
                if key not in needed: continue
                if key in destination: raise ValueError("duplicate monitor source")
                destination[key] = row if kind == "identity" else float(row.get("sample_weight", 1.0))
    if set(identities) != needed or set(weights) != needed:
        raise ValueError("incomplete monitor source identity/weight coverage")
    result = []
    for band, time in enumerate(config["monitor_fixed_times"]):
        v_sum = u_sum = weight_sum = 0.0
        for key in keys:
            geometry = identities[key]["continuous_aligned"]
            n = len(geometry["species"])
            lattice = torch.tensor(geometry["lattice_matrix_A"], dtype=torch.float64)
            clean_z = normalizer.encode(lattice, n)
            fractional = torch.zeros(max_sites, 3, dtype=torch.float64)
            fractional[:n] = torch.tensor(geometry["fractional"], dtype=torch.float64).remainder(1)
            mask = torch.arange(max_sites) < n
            generator = torch.Generator().manual_seed(source_seed(config["validation_seed"], key, 0,
                                                                     f"geometry:validation_band_{band}"))
            noisy = corrupt_geometry(clean_z, fractional, torch.tensor(time, dtype=torch.float64),
                                     atom_mask=mask, generator=generator, config=diffusion)
            weight = weights[key]
            v_sum += weight * float(noisy.v_target.float().square().mean())
            u_sum += weight * float(noisy.u_target[mask].float().square().mean())
            weight_sum += weight
        result.append({"band": band, "time": time, "weight": weight_sum,
                       "zero_v_MSE": v_sum/weight_sum, "zero_u_MSE": u_sum/weight_sum})
    return {"torch_version": str(torch.__version__), "model_forwards": 0, "GPU_calls": 0,
            "max_sites_for_exact_noise_draw_shape": max_sites, "sources": len(keys), "bands": result}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--zero-identity", type=Path)
    parser.add_argument("--zero-prepared", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--max-sites", type=int, default=20)
    args = parser.parse_args()
    raw = args.cases.read_bytes()
    bundle = json.loads(raw)
    result = analyze(bundle)
    result["input_sha256"] = sha256(raw).hexdigest()
    if args.zero_identity:
        if not args.zero_prepared or not args.source_root:
            raise ValueError("zero baseline requires identity, prepared rows and source-root")
        result["zero_field_validation"] = zero_baseline(bundle["training_config"], args.zero_identity,
                                                         args.zero_prepared, args.max_sites, args.source_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    compact = {key: result[key] for key in ("input_sha256", "endpoint_counts", "validation", "geometry_summaries",
                                           "near_distance_counts", "time_grid_audit")}
    print(json.dumps(compact, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
