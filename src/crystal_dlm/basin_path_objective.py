"""Empirical, two-objective full-path teachers (plans 17/18).

Only verified occurrences with both finite energies enter the reference. Each
such group has equal mass; occurrences within it have uniform reference mass.
No trajectory IDs are deduplicated and no energies are imputed. This module
does not score/replay paths or import Torch unless the optional loss is called.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq, minimize
from scipy.special import logsumexp, xlogy


class _EmpiricalPool:
    """Padded, vectorized labels; padding always has zero probability."""

    def __init__(self, labels: list[np.ndarray]) -> None:
        counts = np.array([len(row) for row in labels])
        self.mask = np.arange(int(counts.max()))[None, :] < counts[:, None]
        self.log_counts = np.log(counts)[:, None]
        self.uniform = self.mask / counts[:, None]
        self.r = np.zeros((*self.mask.shape, 2), dtype=np.float64)
        for i, row in enumerate(labels):
            # Subtract a local origin before centering large absolute energies.
            offsets = row - row[0]
            self.r[i, :len(row)] = offsets - offsets.mean(axis=0)
        if not np.isfinite(self.r).all():
            raise ValueError("energy differences must be finite")
        # A single common rescaling improves conditioning, not the objective.
        # It is NOT a per-group/axis normalization or a MAD transformation.
        self.scale = float(np.abs(self.r).max()) or 1.0
        self.z = self.r / self.scale

    def tilt(self, costs: np.ndarray, eta: float = 1.0):
        minimum = np.where(self.mask, costs, np.inf).min(axis=1, keepdims=True)
        if eta == 0.0:
            support = self.mask & (costs == minimum)
            weights = support / support.sum(axis=1, keepdims=True)
            return weights, -minimum[:, 0]  # eta * log Z, continuous limit
        logits = np.where(self.mask, -(costs - minimum) / eta, -np.inf)
        normalizer = logsumexp(logits, axis=1, keepdims=True)
        weights = np.exp(logits - normalizer)
        eta_log_z = eta * (normalizer - self.log_counts) - minimum
        return weights, eta_log_z[:, 0]

    def delta(self, weights: np.ndarray) -> np.ndarray:
        return np.einsum("cj,cjk->k", weights, self.z) / len(weights)

    def kl(self, weights: np.ndarray) -> float:
        terms = xlogy(weights, weights) + weights * self.log_counts
        return max(0.0, float(terms.sum() / len(weights)))


def _max_common_gain(pool: _EmpiricalPool, kappa: float):
    """Solve the 2-D convex (a, eta) dual via its monotone KKT equations.

    For fixed eta, dF/da = D_B-D_A; after minimizing over a, dF/deta
    is kappa-KL. Bracketed roots solve these equations, including the a endpoints,
    without a beta/weight grid. A tiny positive eta recovers primal weights on a
    possibly tied eta=0 face; the exact eta=0 limit remains an upper bound only.
    """

    def at_temperature(eta: float):
        def at_mixture(a: float):
            cost = a * pool.z[..., 0] + (1.0 - a) * pool.z[..., 1]
            weights, eta_log_z = pool.tilt(cost, eta)
            delta = pool.delta(weights)
            return weights, float(delta[1] - delta[0]), eta_log_z

        if at_mixture(0.0)[1] >= 0.0:
            a = 0.0
        elif at_mixture(1.0)[1] <= 0.0:
            a = 1.0
        else:
            a = brentq(lambda a: at_mixture(a)[1], 0.0, 1.0, xtol=1e-15)
        weights, _, eta_log_z = at_mixture(a)
        return weights, eta * kappa + float(eta_log_z.mean()), a

    eta_floor = 1e-8
    weights, bound, a = at_temperature(eta_floor)
    eta = eta_floor
    if pool.kl(weights) > kappa:
        high = 1.0
        while pool.kl(at_temperature(high)[0]) > kappa:
            high *= 2.0
        eta = brentq(
            lambda value: kappa - pool.kl(at_temperature(value)[0]),
            eta_floor, high, xtol=1e-13, rtol=1e-13,
        )
        weights, bound, a = at_temperature(eta)

    cost = a * pool.z[..., 0] + (1.0 - a) * pool.z[..., 1]
    _, zero_limit = pool.tilt(cost, 0.0)
    bound = min(bound, float(zero_limit.mean()))
    # Enforce the actual KL constraint, even if a root ended on its wrong side
    # by rounding. Mixing with u preserves signs of both centered gains.
    if pool.kl(weights) > kappa:
        fraction = brentq(
            lambda t: pool.kl((1.0 - t) * pool.uniform + t * weights) - kappa,
            0.0, 1.0, xtol=1e-14,
        )
        fraction = np.nextafter(fraction, 0.0)
        weights = (1.0 - fraction) * pool.uniform + fraction * weights
    gain = max(0.0, float(-pool.delta(weights).max()))
    # Roundoff must not make a reported upper bound lower than its witness.
    return weights, gain, max(gain, bound)


def _minimum_kl_projection(pool: _EmpiricalPool, witness, target: float):
    """Nonnegative two-multiplier dual, with a primal-feasible roundoff repair."""

    def feasible_fallback():
        fraction = min(1.0, target / float(-pool.delta(witness).max()))
        weights = (1.0 - fraction) * pool.uniform + fraction * witness
        return weights, None  # Feasible, but minimum KL is not certified.

    def objective(multipliers):
        costs = pool.z @ multipliers
        weights, log_z = pool.tilt(costs)
        value = float(log_z.mean() - target * multipliers.sum())
        gradient = -pool.delta(weights) - target
        return value, gradient

    result = minimize(
        objective, np.zeros(2), jac=True, method="L-BFGS-B",
        bounds=[(0.0, None), (0.0, None)],
        options={"ftol": 1e-15, "gtol": 1e-11, "maxiter": 500, "maxls": 50},
    )
    if not np.isfinite(result.x).all():
        return feasible_fallback()
    weights, log_z = pool.tilt(pool.z @ result.x)
    dual_value = float(log_z.mean() - target * result.x.sum())
    if not np.isfinite(weights).all() or not np.isfinite(dual_value):
        return feasible_fallback()
    delta = pool.delta(weights)
    witness_delta = pool.delta(witness)
    # The first-stage witness meets both targets. A convex mixture repairs any
    # residual without inventing feasible evidence from an optimizer flag.
    repair = 0.0
    for axis in range(2):
        if delta[axis] > -target:
            denominator = delta[axis] - witness_delta[axis]
            ratio = (delta[axis] + target) / denominator if denominator > 0 else 1.0
            repair = max(repair, ratio)
    if repair > 0.0:
        repair = min(1.0, repair + 1e-12)
        weights = (1.0 - repair) * weights + repair * witness
    # -Phi is a lower bound on the minimum KL. Check the returned primal, not
    # merely scipy's termination flag, to certify the projection.
    gap = max(0.0, pool.kl(weights) + dual_value)
    return weights, gap


def solve_basin_path_teacher(
    groups: list[dict],
    kappa: float = 0.2,
    energy_scale: float = 0.1,
    retained_fraction: float = 0.5,
) -> dict:
    """Return copied ``groups``/``candidates`` with occurrence-level ``weight``.

    Positive weights sum to one in each labeled group. Unverified, missing or
    nonfinite-energy occurrences receive zero; all original rows/order remain.
    ``validated_groups`` counts groups with at least one usable verified label.

    Summary ``mean_delta_gap``/``mean_delta_terminal`` are signed changes from
    the within-group empirical reference, in input energy units (negative is
    better). ``target_gain`` is in those units; ``rho_max`` is dimensionless,
    divided by ``energy_scale``. It is the achieved first-stage primal value;
    ``rho_dual_upper_bound`` and their gap report numerical optimality separately.
    ``ESS`` is 1/sum((w/C)**2), the ESS of the equal-group joint distribution.
    ``primal_residual`` is the maximum normalization/nonnegativity/target
    violation, with energy violations divided by ``energy_scale``. Empty
    coverage has no energy deltas (None), not fabricated zero-energy labels.

    A non-``optimal`` numerical status must not be interpreted as a certified
    minimum-KL teacher. These are empirical-pool guarantees, not student or SUN
    guarantees. No paths/conditions are resampled and no coefficient grid is used.
    """

    kappa, energy_scale, retained_fraction = map(
        float, (kappa, energy_scale, retained_fraction)
    )
    if not np.isfinite(kappa) or kappa < 0.0:
        raise ValueError("kappa must be finite and nonnegative")
    if not np.isfinite(energy_scale) or energy_scale <= 0.0:
        raise ValueError("energy_scale must be finite and positive")
    if not np.isfinite(retained_fraction) or not 0.0 <= retained_fraction <= 1.0:
        raise ValueError("retained_fraction must be between zero and one")

    output, labels, locations = [], [], []
    for group in groups:
        candidates = [dict(candidate, weight=0.0) for candidate in group["candidates"]]
        output.append(dict(group, candidates=candidates))
        row, indices = [], []
        for index, candidate in enumerate(candidates):
            if candidate.get("verified") is not True:
                continue
            raw, terminal = candidate.get("raw_energy"), candidate.get("terminal_energy")
            if raw is None or terminal is None:
                continue
            raw, terminal = float(raw), float(terminal)
            if not np.isfinite([raw, terminal]).all():
                continue
            gap = raw - terminal
            if not np.isfinite(gap):
                raise ValueError("raw minus terminal energy must be finite")
            row.append((gap, terminal))
            indices.append(index)
        if row:
            labels.append(np.asarray(row, dtype=np.float64))
            locations.append((len(output) - 1, indices))

    summary = {
        "rho_max": 0.0, "rho_dual_upper_bound": 0.0, "rho_duality_gap": 0.0,
        "target_gain": 0.0, "mean_delta_gap": None, "mean_delta_terminal": None,
        "mean_kl": 0.0, "ESS": 0.0, "validated_groups": len(labels),
        "total_groups": len(groups), "validated_candidates": sum(map(len, labels)),
        "total_candidates": sum(len(group["candidates"]) for group in groups),
        "solver_status": "no_verified_groups", "primal_residual": 0.0,
        "max_common_mean_kl": 0.0, "projection_duality_gap": 0.0,
    }
    if not labels:
        return {"groups": output, "summary": summary}

    pool = _EmpiricalPool(labels)
    weights = pool.uniform.copy()
    target = 0.0
    if not np.any(pool.r):
        status = "uniform_constant_labels"
    elif kappa == 0.0:
        status = "uniform_zero_kl_budget"
    else:
        witness, gain, bound = _max_common_gain(pool, kappa)
        conversion = pool.scale / energy_scale
        summary.update(
            rho_max=gain * conversion,
            rho_dual_upper_bound=bound * conversion,
            rho_duality_gap=max(0.0, bound - gain) * conversion,
            max_common_mean_kl=pool.kl(witness),
        )
        if gain == 0.0 or bound <= 1e-12:
            summary["rho_max"] = 0.0
            status = "uniform_no_common_gain" if bound <= 1e-6 else "uniform_solver_unresolved"
        elif retained_fraction == 0.0:
            status = "uniform_zero_target"
        else:
            target = retained_fraction * gain
            weights, projection_gap = _minimum_kl_projection(pool, witness, target)
            summary["projection_duality_gap"] = projection_gap
            status = "optimal"
            if bound - gain > 1e-6:
                status = "feasible_max_common_not_converged"
            elif projection_gap is None or projection_gap > 1e-7:
                status = "feasible_projection_not_converged"

    delta = pool.delta(weights) * pool.scale
    target_gain = target * pool.scale
    residual = max(
        0.0, float((delta + target_gain).max()) / energy_scale,
        float(np.abs(weights.sum(axis=1) - 1.0).max()), float(-weights.min()),
    )
    summary.update(
        target_gain=target_gain, mean_delta_gap=float(delta[0]),
        mean_delta_terminal=float(delta[1]), mean_kl=pool.kl(weights),
        ESS=float(len(labels) ** 2 / np.square(weights).sum()),
        solver_status=status, primal_residual=residual,
    )
    for row_index, (group_index, indices) in enumerate(locations):
        for column, candidate_index in enumerate(indices):
            output[group_index]["candidates"][candidate_index]["weight"] = float(
                weights[row_index, column]
            )
    return {"groups": output, "summary": summary}


def weighted_sampled_path_nll(log_probs, weights, inclusion_probs, group_normalizer):
    """Torch HT loss: ``-sum(stopgrad(w) * log_p / pi) / C``.

    ``log_probs`` contains actual sampled scalar-decision log probabilities,
    not a log-softmax over candidate paths. Weights/pi must broadcast to its
    shape (e.g. path weights ``[paths, 1]`` for ``[paths, sampled_states]``).
    There is no path-length normalization. Repeated sampled occurrences are
    summed; the caller must supply probabilities appropriate to its design.
    Only positive-weight entries need valid pi in (0, 1]. Zero-weight entries
    are indexed out BEFORE arithmetic, even when log_p=-inf or pi=0/unknown.
    An all-zero batch returns a differentiable zero with zero log_p gradients.
    """

    import torch

    if not isinstance(log_probs, torch.Tensor) or not log_probs.is_floating_point():
        raise TypeError("log_probs must be a floating-point Torch tensor")
    dtype = torch.float64 if log_probs.dtype == torch.float64 else torch.float32
    log_probs = log_probs.to(dtype=dtype)

    def constant(value):
        return torch.as_tensor(value, dtype=dtype, device=log_probs.device).detach()

    weights = torch.broadcast_to(constant(weights), log_probs.shape)
    inclusion_probs = torch.broadcast_to(constant(inclusion_probs), log_probs.shape)
    normalizer = constant(group_normalizer)
    if (normalizer.numel() != 1 or not bool(torch.isfinite(normalizer).all())
            or not bool(normalizer > 0)):
        raise ValueError("group_normalizer must be a finite positive scalar")
    if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
        raise ValueError("weights must be finite and nonnegative")
    active = weights > 0
    pi = inclusion_probs[active]
    if not bool(torch.isfinite(pi).all()) or bool(((pi <= 0) | (pi > 1)).any()):
        raise ValueError("positive-weight entries need inclusion probabilities in (0, 1]")
    return -(log_probs[active] * (weights[active] / pi)).sum() / normalizer.reshape(())
