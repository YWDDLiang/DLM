"""Deterministic summaries for matched SGTC L6 evaluation."""

from __future__ import annotations

import math
import random
from statistics import mean, median
from typing import Mapping, Sequence


def quantile(values: Sequence[float], probability: float) -> float | None:
    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not ordered:
        return None
    q = float(probability)
    if not 0.0 <= q <= 1.0:
        raise ValueError("SGTC quantile probability must lie in [0,1]")
    position = q * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def paired_energy_stats(
    candidate: Mapping[tuple[int, int], float],
    control: Mapping[tuple[int, int], float],
    *,
    bootstrap_draws: int = 2000,
    seed: int = 82017,
) -> dict[str, float | int | None]:
    keys = sorted(set(candidate) & set(control))
    deltas = [float(candidate[key]) - float(control[key]) for key in keys]
    if not deltas:
        return {
            "paired": 0,
            "mean_delta": None,
            "median_delta": None,
            "fraction_candidate_lower": None,
            "mean_delta_lcb_95": None,
            "mean_delta_ucb_95": None,
        }
    rng = random.Random(int(seed))
    bootstrap = []
    for _ in range(int(bootstrap_draws)):
        sample = [deltas[rng.randrange(len(deltas))] for _ in deltas]
        bootstrap.append(mean(sample))
    bootstrap.sort()
    return {
        "paired": len(deltas),
        "mean_delta": mean(deltas),
        "median_delta": median(deltas),
        "fraction_candidate_lower": sum(value < 0.0 for value in deltas)
        / len(deltas),
        "mean_delta_lcb_95": bootstrap[int(0.025 * (len(bootstrap) - 1))],
        "mean_delta_ucb_95": bootstrap[int(0.975 * (len(bootstrap) - 1))],
    }


def rate_delta_pp(candidate_count: int, control_count: int, denominator: int) -> float:
    if int(denominator) <= 0:
        raise ValueError("SGTC rate denominator must be positive")
    return 100.0 * (int(candidate_count) - int(control_count)) / int(denominator)


__all__ = ["paired_energy_stats", "quantile", "rate_delta_pp"]
