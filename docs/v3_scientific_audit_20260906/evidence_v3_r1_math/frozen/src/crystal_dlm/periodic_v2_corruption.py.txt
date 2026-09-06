"""Quantized log-volume/shape and affine-Cartesian corruption for v2.

The three scales are a registered design prior, not fitted train statistics.
Only integer tokens leave the corruption path as model-visible geometry.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from typing import Any, Mapping

import numpy as np
import torch

from crystal_dlm.dynamic_crystal import arrays_to_dynamic_tokens
from crystal_dlm.manifold_corruption import (
    lattice_matrix_from_parameters,
    lattice_parameters_from_matrix,
    symmetric_matrix_exp,
    symmetric_matrix_log,
)
from crystal_dlm.periodic_base_training_data import (
    PreparedBaseSource,
    _canonical_tokens,
    _config,
)
from crystal_dlm.programmed_path_runtime import complete_geometry_supported


V2_NOISE_GRID = ((.03, .01, .05), (.10, .03, .15), (.20, .06, .30))
V2_MAX_CORRUPTION_ATTEMPTS = 8
V2_CORRUPTION_PROTOCOL = {
    "schema": "periodic_v2_log_spd_affine_cartesian_v1",
    "noise_grid": [list(level) for level in V2_NOISE_GRID],
    "noise_grid_provenance": "frozen_design_prior_not_statistical_optimum",
    "components": ["log_volume_std", "traceless_shape_coefficient_std", "cartesian_component_std_A"],
    "base_variates": "independent_standard_normal",
    "coordinate_reference": "affinely_deformed_clean_fractional_sites",
    "max_total_proposals": V2_MAX_CORRUPTION_ATTEMPTS,
    "rejection": "quantize_then_check_current_complete_geometry_support",
    "exhaustion": "retain_source_with_clean_old_and_explicit_admission_flag",
}


def traceless_shape_basis() -> np.ndarray:
    """Five Frobenius-orthonormal symmetric traceless 3x3 matrices."""
    basis = [np.diag([1., -1., 0.]) / np.sqrt(2),
             np.diag([1., 1., -2.]) / np.sqrt(6)]
    for left, right in ((0, 1), (0, 2), (1, 2)):
        value = np.zeros((3, 3), dtype=np.float64)
        value[left, right] = value[right, left] = 1 / np.sqrt(2)
        basis.append(value)
    return np.stack(basis)


def log_spd_volume_shape(lattice: np.ndarray) -> tuple[float, np.ndarray]:
    """Return log(V/1 A^3) and traceless half-log metric for row lattice L."""
    lattice = np.asarray(lattice, dtype=np.float64)
    if lattice.shape != (3, 3) or not np.isfinite(lattice).all():
        raise ValueError("lattice must be a finite row-vector 3x3 matrix")
    half_log = .5 * symmetric_matrix_log(lattice @ lattice.T)
    log_volume = float(np.trace(half_log))
    return log_volume, half_log - (log_volume / 3) * np.eye(3)


def sample_log_spd_corruption(
    lattice: np.ndarray,
    fractional: np.ndarray,
    rng: np.random.Generator,
    *,
    sigma_v: float,
    sigma_shape: float,
    sigma_F: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Sample a continuous proposal; callers must quantize before conditioning.

    Before wrapping, F_t L_t = F_0 L_t + delta_R. Thus sigma_F controls
    displacement relative to the affinely deformed reference, not total motion
    relative to F_0 L_0. Shape coefficient RMS is sqrt(5) * sigma_shape.
    """
    scales = np.asarray((sigma_v, sigma_shape, sigma_F), dtype=np.float64)
    if not np.isfinite(scales).all() or bool((scales < 0).any()):
        raise ValueError("noise standard deviations must be finite and nonnegative")
    fractional = np.asarray(fractional, dtype=np.float64)
    if fractional.ndim != 2 or fractional.shape[1] != 3 or not np.isfinite(fractional).all():
        raise ValueError("fractional coordinates must be a finite Nx3 array")
    log_volume, shape = log_spd_volume_shape(lattice)
    delta_v = float(sigma_v * rng.normal())
    coefficients = sigma_shape * rng.normal(size=5)
    delta_shape = np.einsum("r,rij->ij", coefficients, traceless_shape_basis())
    half_log = shape + delta_shape + ((log_volume + delta_v) / 3) * np.eye(3)
    metric = symmetric_matrix_exp(2 * half_log)
    # np.linalg.cholesky returns lower L_t with L_t L_t^T = G_t.
    noisy_lattice = np.linalg.cholesky(metric)
    cartesian = sigma_F * rng.normal(size=fractional.shape)
    fractional_displacement = np.linalg.solve(noisy_lattice.T, cartesian.T).T
    noisy_fractional = np.mod(fractional + fractional_displacement, 1.)
    if not np.isfinite(noisy_fractional).all() or not np.isfinite(noisy_lattice).all():
        raise FloatingPointError("nonfinite log-SPD corruption")
    return noisy_lattice, noisy_fractional, {
        "sampled_log_volume_delta": delta_v,
        "sampled_shape_frobenius_delta": float(np.linalg.norm(delta_shape)),
        "sampled_max_affine_displacement_A": float(np.linalg.norm(cartesian, axis=-1).max(initial=0.)),
        "sampled_rms_affine_displacement_A": float(np.sqrt(np.square(cartesian).sum(-1).mean())),
    }


def corrupt_periodic_v2_source(
    source: PreparedBaseSource,
    rng: np.random.Generator,
    vocabulary: Mapping[str, int],
    constraints: Mapping[str, Any],
    *,
    max_attempts: int = V2_MAX_CORRUPTION_ATTEMPTS,
) -> tuple[list[int], tuple[float, float, float], dict[str, Any]]:
    """Try at most eight proposals at one sampled grid level, retaining failures.

    Accepted but quantized-unchanged proposals retain their requested sigmas.
    Exhaustion falls back to clean with zero applied sigmas; its old-state
    admission is separately reported and may be false. No source is filtered.
    """
    if not 1 <= int(max_attempts) <= V2_MAX_CORRUPTION_ATTEMPTS:
        raise ValueError("v2 permits one to eight total corruption proposals")
    level = int(rng.integers(len(V2_NOISE_GRID)))
    requested = V2_NOISE_GRID[level]
    clean = list(source.clean_tokens)
    protected = (0, *(7 + 4 * site for site in range(int(source.arrays["num_atoms"]))))
    attempts: list[dict[str, Any]] = []
    rejections: Counter[str] = Counter()
    old: list[int] | None = None
    config = _config(constraints)
    for attempt in range(1, int(max_attempts) + 1):
        record: dict[str, Any] = {"attempt": attempt, "accepted": False}
        reason = None
        try:
            lattice = lattice_matrix_from_parameters(source.arrays["lengths"], source.arrays["angles"])
            noisy_lattice, noisy_fractional, continuous = sample_log_spd_corruption(
                lattice, np.asarray(source.arrays["frac_coords"]), rng,
                sigma_v=requested[0], sigma_shape=requested[1], sigma_F=requested[2],
            )
            record.update(continuous)
            lengths, angles = lattice_parameters_from_matrix(noisy_lattice)
            tokens, diagnostics = arrays_to_dynamic_tokens(
                lengths, angles, source.arrays["species"], noisy_fractional, config=config,
            )
            tokens, aliases = _canonical_tokens(tokens)
            record.update(encoding=asdict(diagnostics), alias_tokens_canonicalized=aliases)
            if diagnostics.length_clips or diagnostics.angle_clips or diagnostics.coord_clips:
                reason = "quantization_clipped"
            else:
                candidate = [int(vocabulary[token]) for token in tokens]
                if len(candidate) != len(clean) or any(candidate[p] != clean[p] for p in protected):
                    raise RuntimeError("log-SPD corruption changed protected composition slots")
                if not complete_geometry_supported(torch.tensor(candidate, dtype=torch.long), dict(constraints)):
                    reason = "quantized_old_outside_complete_support"
                else:
                    old = candidate
                    record["accepted"] = True
        except (ValueError, np.linalg.LinAlgError, FloatingPointError, OverflowError) as error:
            reason = f"continuous_or_codec_unavailable:{type(error).__name__}"
            record["detail"] = str(error)
        record["reason"] = reason
        attempts.append(record)
        if old is not None:
            break
        rejections[str(reason)] += 1
    fallback = old is None
    if fallback:
        old = clean.copy()
    admitted = complete_geometry_supported(torch.tensor(old, dtype=torch.long), dict(constraints))
    changed = [p for p, (before, after) in enumerate(zip(clean, old, strict=True)) if before != after]
    applied_sigmas = (0., 0., 0.) if fallback else requested
    info = {
        "schema": V2_CORRUPTION_PROTOCOL["schema"],
        "target_source": "original_MP20_clean_native",
        "noise_grid_provenance": V2_CORRUPTION_PROTOCOL["noise_grid_provenance"],
        "level_index": level,
        "requested_noise_components": list(requested),
        "applied_noise_components": list(applied_sigmas),
        "attempt_count": len(attempts),
        "rejected_attempts": sum(rejections.values()),
        "rejection_counts": dict(rejections),
        "attempts": attempts,
        "fallback_to_clean": fallback,
        "fallback_reason": "all_proposals_rejected" if fallback else None,
        "old_state_admitted": bool(admitted),
        "changed_positions": changed,
        "applied": bool(changed),
        "quantized_unchanged": not changed,
    }
    return old, tuple(map(float, applied_sigmas)), info


__all__ = [
    "V2_NOISE_GRID", "V2_MAX_CORRUPTION_ATTEMPTS", "V2_CORRUPTION_PROTOCOL",
    "traceless_shape_basis", "log_spd_volume_shape", "sample_log_spd_corruption",
    "corrupt_periodic_v2_source",
]
