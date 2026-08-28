"""Frozen architecture and bookkeeping helpers for two CTV Q heads."""

from __future__ import annotations

from collections import defaultdict
import math
import random
from typing import Any, Mapping, Sequence

from crystal_dlm.ctv_value_data import spearman


def disjoint_plan_group(plan_ordinal: int) -> int:
    value = int(plan_ordinal)
    if value < 0:
        raise ValueError("CTV Plan ordinal must be non-negative")
    return value % 2


def token_support_counts(
    rows: Sequence[Mapping[str, Any]],
) -> dict[int, dict[str, int]]:
    returns: dict[int, int] = defaultdict(int)
    plans: dict[int, set[int]] = defaultdict(set)
    for row in rows:
        if row.get("energy") is None:
            continue
        token = int(row["action_token"])
        returns[token] += 1
        plans[token].add(int(row["plan_ordinal"]))
    return {
        token: {"known_returns": returns[token], "unique_plans": len(plans[token])}
        for token in sorted(returns)
    }


def supported_token_ids(
    first: Mapping[int, Mapping[str, int]],
    second: Mapping[int, Mapping[str, int]],
    *,
    minimum_returns: int = 8,
    minimum_plans: int = 4,
) -> set[int]:
    tokens = set(int(token) for token in first) & set(int(token) for token in second)
    return {
        token
        for token in tokens
        if int(first[token]["known_returns"]) >= int(minimum_returns)
        and int(second[token]["known_returns"]) >= int(minimum_returns)
        and int(first[token]["unique_plans"]) >= int(minimum_plans)
        and int(second[token]["unique_plans"]) >= int(minimum_plans)
    }


def robust_scale(values: Sequence[float]) -> tuple[float, float]:
    numbers = [float(value) for value in values]
    if not numbers or any(not math.isfinite(value) for value in numbers):
        raise ValueError("CTV Q normalization requires finite train targets")
    center = sum(numbers) / len(numbers)
    variance = sum((value - center) ** 2 for value in numbers) / len(numbers)
    scale = max(math.sqrt(variance), 1e-3)
    return center, scale


def advantage_is_supported(
    first: float,
    second: float,
    *,
    neutral_band: float = 0.005,
    maximum_disagreement: float = 0.02,
) -> bool:
    left = float(first)
    right = float(second)
    if not math.isfinite(left) or not math.isfinite(right):
        return False
    if abs(left - right) > float(maximum_disagreement):
        return False
    left_neutral = abs(left) <= float(neutral_band)
    right_neutral = abs(right) <= float(neutral_band)
    if left_neutral or right_neutral:
        return left_neutral and right_neutral
    return (left > 0.0) == (right > 0.0)


def pairwise_order_accuracy(
    predicted: Sequence[float],
    observed: Sequence[float],
    *,
    tie_epsilon: float = 1e-4,
) -> tuple[float | None, int, float]:
    if len(predicted) != len(observed):
        raise ValueError("CTV predicted/observed action lengths differ")
    correct = 0.0
    comparisons = 0
    for left in range(len(observed)):
        for right in range(left + 1, len(observed)):
            truth = float(observed[left]) - float(observed[right])
            if abs(truth) < float(tie_epsilon):
                continue
            estimate = float(predicted[left]) - float(predicted[right])
            comparisons += 1
            if abs(estimate) < float(tie_epsilon):
                correct += 0.5
            elif (estimate > 0.0) == (truth > 0.0):
                correct += 1.0
    return (
        correct / comparisons if comparisons else None,
        comparisons,
        correct,
    )


def centered_prediction_pairs(
    predicted: Sequence[float], observed: Sequence[float]
) -> list[tuple[float, float]]:
    if len(predicted) != len(observed) or not predicted:
        raise ValueError("CTV centered prediction rows require equal non-empty values")
    prediction_mean = sum(float(value) for value in predicted) / len(predicted)
    observed_mean = sum(float(value) for value in observed) / len(observed)
    return [
        (float(estimate) - prediction_mean, float(truth) - observed_mean)
        for estimate, truth in zip(predicted, observed)
    ]


def plan_bootstrap_spearman(
    rows_by_plan: Mapping[int, Sequence[tuple[float, float]]],
    *,
    draws: int = 2000,
    seed: int = 76017,
) -> dict[str, float | int | None]:
    plans = sorted(int(plan) for plan in rows_by_plan)
    if not plans:
        raise ValueError("CTV Spearman bootstrap has no Plans")

    def evaluate(sampled: Sequence[int]) -> float | None:
        predicted = []
        observed = []
        for plan in sampled:
            for estimate, truth in rows_by_plan[int(plan)]:
                predicted.append(float(estimate))
                observed.append(float(truth))
        return spearman(predicted, observed)

    point = evaluate(plans)
    rng = random.Random(int(seed))
    values = []
    for _ in range(int(draws)):
        sampled = [plans[rng.randrange(len(plans))] for _ in plans]
        value = evaluate(sampled)
        if value is not None and math.isfinite(value):
            values.append(float(value))
    values.sort()
    if not values:
        lower = upper = None
    else:
        lower = values[int(0.025 * (len(values) - 1))]
        upper = values[int(0.975 * (len(values) - 1))]
    return {
        "point": point,
        "bootstrap_draws": int(draws),
        "finite_draws": len(values),
        "lcb_95": lower,
        "ucb_95": upper,
        "seed": int(seed),
    }


def build_q_head(projection_dim: int = 256) -> Any:
    import torch

    class CTVQHead(torch.nn.Module):
        def __init__(self, dimension: int) -> None:
            super().__init__()
            self.dimension = int(dimension)
            self.state_norm = torch.nn.LayerNorm(self.dimension)
            self.action_norm = torch.nn.LayerNorm(self.dimension)
            self.baseline = torch.nn.Sequential(
                torch.nn.Linear(self.dimension, 64),
                torch.nn.SiLU(),
                torch.nn.Linear(64, 1),
            )
            self.advantage = torch.nn.Sequential(
                torch.nn.Linear(3 * self.dimension + 11, 128),
                torch.nn.SiLU(),
                torch.nn.Linear(128, 1),
            )

        def forward(
            self,
            state_features: Any,
            action_features: Any,
            log_probabilities: Any,
            milestones: Any,
            family_one_hot: Any,
        ) -> tuple[Any, Any]:
            if state_features.ndim != 2 or action_features.ndim != 3:
                raise ValueError("CTV Q head expects [S,D] states and [S,A,D] actions")
            if action_features.shape[0] != state_features.shape[0]:
                raise ValueError("CTV Q state/action batch sizes differ")
            state = self.state_norm(state_features)
            action = self.action_norm(action_features)
            expanded = state.unsqueeze(1).expand_as(action)
            inputs = torch.cat(
                [
                    expanded,
                    action,
                    expanded * action,
                    log_probabilities,
                    milestones,
                    family_one_hot,
                ],
                dim=-1,
            )
            raw_advantage = self.advantage(inputs).squeeze(-1)
            centered_advantage = raw_advantage - raw_advantage.mean(dim=1, keepdim=True)
            baseline = self.baseline(state)
            q_value = baseline + centered_advantage
            return q_value, centered_advantage

    if int(projection_dim) != 256:
        raise ValueError("CTV Q projection dimension changed")
    return CTVQHead(int(projection_dim))


__all__ = [
    "build_q_head",
    "centered_prediction_pairs",
    "advantage_is_supported",
    "disjoint_plan_group",
    "pairwise_order_accuracy",
    "plan_bootstrap_spearman",
    "robust_scale",
    "supported_token_ids",
    "token_support_counts",
]
