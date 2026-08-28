"""Validation and descriptive statistics for frozen CTV Branch returns."""

from __future__ import annotations

from collections import defaultdict
import math
from statistics import mean, pstdev
from typing import Any, Iterable, Mapping, Sequence


CTV_TIE_EPS = 1e-4


def linear_quantile(values: Sequence[float], probability: float) -> float | None:
    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not ordered:
        return None
    q = float(probability)
    if not 0.0 <= q <= 1.0:
        raise ValueError("quantile probability must lie in [0,1]")
    position = q * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def average_ranks(values: Sequence[float]) -> list[float]:
    indexed = sorted(enumerate(float(value) for value in values), key=lambda item: item[1])
    ranks = [0.0] * len(indexed)
    start = 0
    while start < len(indexed):
        end = start + 1
        while end < len(indexed) and indexed[end][1] == indexed[start][1]:
            end += 1
        average = (start + end - 1) / 2.0
        for cursor in range(start, end):
            ranks[indexed[cursor][0]] = average
        start = end
    return ranks


def pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    x = [float(value) for value in left]
    y = [float(value) for value in right]
    x_mean = mean(x)
    y_mean = mean(y)
    numerator = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y, strict=True))
    x_norm = math.sqrt(sum((a - x_mean) ** 2 for a in x))
    y_norm = math.sqrt(sum((b - y_mean) ** 2 for b in y))
    if x_norm == 0.0 or y_norm == 0.0:
        return None
    return numerator / (x_norm * y_norm)


def spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    return pearson(average_ranks(left), average_ranks(right))


def signed_difference(value: float, *, epsilon: float = CTV_TIE_EPS) -> int:
    number = float(value)
    if abs(number) < float(epsilon):
        return 0
    return 1 if number > 0.0 else -1


def cross_continuation_pair_agreement(
    energy_by_action_continuation: Mapping[int, Mapping[int, float]],
    *,
    epsilon: float = CTV_TIE_EPS,
) -> tuple[int, int]:
    actions = sorted(int(action) for action in energy_by_action_continuation)
    agreements = 0
    comparisons = 0
    for left_index, left_action in enumerate(actions):
        for right_action in actions[left_index + 1 :]:
            left = energy_by_action_continuation[left_action]
            right = energy_by_action_continuation[right_action]
            shared = sorted(set(left) & set(right))
            if len(shared) < 2:
                continue
            signs = [
                signed_difference(
                    float(left[continuation]) - float(right[continuation]),
                    epsilon=epsilon,
                )
                for continuation in shared
            ]
            non_ties = [value for value in signs if value != 0]
            if len(non_ties) < 2:
                continue
            comparisons += 1
            agreements += int(len(set(non_ties)) == 1)
    return agreements, comparisons


def _finite_energy(row: Mapping[str, Any]) -> float | None:
    if row.get("chgnet_relaxation_known") is not True:
        return None
    value = row.get("chgnet_energy_per_atom")
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def validate_branch_split(
    *,
    split: str,
    branches: Sequence[Mapping[str, Any]],
    states: Sequence[Mapping[str, Any]],
    labels: Sequence[Mapping[str, Any]],
    expected_branches: int,
    expected_states: int,
    continuations_per_action: int,
) -> dict[str, Any]:
    expected_branch_count = int(expected_branches)
    expected_state_count = int(expected_states)
    continuation_count = int(continuations_per_action)
    if continuation_count <= 0:
        raise ValueError("CTV continuation count must be positive")
    if len(branches) != expected_branch_count or len(labels) != expected_branch_count:
        raise ValueError(f"CTV {split} branch or label denominator changed")
    if len(states) != expected_state_count:
        raise ValueError(f"CTV {split} state denominator changed")

    branch_by_ordinal = {int(row["branch_ordinal"]): row for row in branches}
    label_by_ordinal = {int(row["global_branch_ordinal"]): row for row in labels}
    expected_ordinals = set(range(expected_branch_count))
    if set(branch_by_ordinal) != expected_ordinals or set(label_by_ordinal) != expected_ordinals:
        raise ValueError(f"CTV {split} global ordinals are incomplete or duplicated")
    state_by_id = {str(row["state_id"]): row for row in states}
    if len(state_by_id) != expected_state_count:
        raise ValueError(f"CTV {split} state ids are duplicated")

    joined: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ordinal in range(expected_branch_count):
        branch = branch_by_ordinal[ordinal]
        label = label_by_ordinal[ordinal]
        for key in ("state_id", "action_token", "continuation_seed"):
            if str(branch[key]) != str(label[key]):
                raise ValueError(f"CTV {split} branch/label {key} alignment changed")
        state_id = str(branch["state_id"])
        state = state_by_id.get(state_id)
        if state is None:
            raise ValueError(f"CTV {split} label references an unknown state")
        if str(branch["composition_id"]) != str(state["composition_id"]):
            raise ValueError(f"CTV {split} state composition alignment changed")
        row = {
            "split": str(split),
            "global_branch_ordinal": ordinal,
            "state_id": state_id,
            "composition_id": str(branch["composition_id"]),
            "sample_idx": int(branch["sample_idx"]),
            "plan_ordinal": int(branch["canary_plan_idx"]),
            "milestone": float(branch["milestone"]),
            "intervention_position": int(branch["intervention_position"]),
            "action_token": int(branch["action_token"]),
            "action_rank": int(branch["action_rank"]),
            "base_action_probability": float(branch["base_action_probability"]),
            "continuation_seed": int(branch["continuation_seed"]),
            "continuation_rank": int(branch["continuation_rank"]),
            "num_atoms": int(branch["num_atoms"]),
            "direct_valid": label.get("direct_valid") is True,
            "reconstructed": label.get("reconstructed") is True,
            "energy": _finite_energy(label),
        }
        joined.append(row)
        grouped[state_id].append(row)

    state_summaries: list[dict[str, Any]] = []
    all_pair_agreements = 0
    all_pair_comparisons = 0
    continuation_spearman: list[float] = []
    for state_id in sorted(grouped):
        rows = grouped[state_id]
        state = state_by_id[state_id]
        expected_actions = [int(value) for value in state["action_token_ids"]]
        if len(expected_actions) != 8 or len(set(expected_actions)) != 8:
            raise ValueError(f"CTV {split} state does not contain eight actions")
        by_action: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_action[int(row["action_token"])].append(row)
        if set(by_action) != set(expected_actions):
            raise ValueError(f"CTV {split} observed action support changed")
        if any(len(action_rows) != continuation_count for action_rows in by_action.values()):
            raise ValueError(f"CTV {split} continuation count changed")
        continuation_sets = {
            tuple(sorted(int(row["continuation_seed"]) for row in action_rows))
            for action_rows in by_action.values()
        }
        if len(continuation_sets) != 1:
            raise ValueError(f"CTV {split} actions do not share continuation seeds")

        energy_by_action_continuation: dict[int, dict[int, float]] = {}
        action_means: list[float] = []
        for action, action_rows in by_action.items():
            known = {
                int(row["continuation_seed"]): float(row["energy"])
                for row in action_rows
                if row["energy"] is not None
            }
            energy_by_action_continuation[action] = known
            if known:
                action_means.append(mean(known.values()))
        agreements, comparisons = cross_continuation_pair_agreement(
            energy_by_action_continuation
        )
        all_pair_agreements += agreements
        all_pair_comparisons += comparisons

        state_spearman: float | None = None
        continuations = sorted(next(iter(continuation_sets)))
        if len(continuations) >= 2 and all(
            all(continuation in values for continuation in continuations[:2])
            for values in energy_by_action_continuation.values()
        ):
            first = [
                energy_by_action_continuation[action][continuations[0]]
                for action in expected_actions
            ]
            second = [
                energy_by_action_continuation[action][continuations[1]]
                for action in expected_actions
            ]
            state_spearman = spearman(first, second)
            if state_spearman is not None:
                continuation_spearman.append(state_spearman)

        composition_numbers = {
            int(part.split(":", 1)[0])
            for part in str(rows[0]["composition_id"]).split("|")
            if ":" in part
        }
        state_summaries.append(
            {
                "split": str(split),
                "state_id": state_id,
                "composition_id": str(rows[0]["composition_id"]),
                "sample_idx": int(rows[0]["sample_idx"]),
                "plan_ordinal": int(rows[0]["plan_ordinal"]),
                "milestone": float(rows[0]["milestone"]),
                "intervention_position": int(rows[0]["intervention_position"]),
                "num_atoms": int(rows[0]["num_atoms"]),
                "branches": len(rows),
                "known_returns": sum(row["energy"] is not None for row in rows),
                "direct_valid": sum(row["direct_valid"] for row in rows),
                "known_actions": len(action_means),
                "action_energy_mean": mean(action_means) if action_means else None,
                "action_energy_std": pstdev(action_means) if len(action_means) > 1 else 0.0,
                "action_energy_range": max(action_means) - min(action_means)
                if action_means
                else None,
                "cross_continuation_spearman": state_spearman,
                "cross_pair_agreements": agreements,
                "cross_pair_comparisons": comparisons,
                "oxide": 8 in composition_numbers,
                "sulfide": 16 in composition_numbers,
                "n13_20": 13 <= int(rows[0]["num_atoms"]) <= 20,
            }
        )

    plan_milestones: dict[tuple[str, int], set[float]] = defaultdict(set)
    for summary in state_summaries:
        plan_milestones[
            (str(summary["composition_id"]), int(summary["sample_idx"]))
        ].add(float(summary["milestone"]))
    if len(plan_milestones) * 2 != expected_state_count or any(
        values != {0.6, 0.8} for values in plan_milestones.values()
    ):
        raise ValueError(f"CTV {split} Plan/milestone accounting changed")

    energies = [float(row["energy"]) for row in joined if row["energy"] is not None]
    ranges = [
        float(row["action_energy_range"])
        for row in state_summaries
        if row["action_energy_range"] is not None
    ]
    report = {
        "split": str(split),
        "branches": len(joined),
        "states": len(state_summaries),
        "plans": len(plan_milestones),
        "known_returns": len(energies),
        "unknown_returns": len(joined) - len(energies),
        "direct_valid": sum(row["direct_valid"] for row in joined),
        "reconstructed": sum(row["reconstructed"] for row in joined),
        "energy_quantiles": {
            str(q): linear_quantile(energies, q) for q in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
        },
        "within_state_range_quantiles": {
            str(q): linear_quantile(ranges, q) for q in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
        },
        "cross_pair_agreements": all_pair_agreements,
        "cross_pair_comparisons": all_pair_comparisons,
        "cross_pair_agreement": all_pair_agreements / all_pair_comparisons
        if all_pair_comparisons
        else None,
        "cross_continuation_spearman_mean": mean(continuation_spearman)
        if continuation_spearman
        else None,
        "cross_continuation_states": len(continuation_spearman),
    }
    return {
        "report": report,
        "joined_rows": joined,
        "state_summaries": state_summaries,
        "composition_ids": sorted({row["composition_id"] for row in joined}),
    }


__all__ = [
    "CTV_TIE_EPS",
    "average_ranks",
    "cross_continuation_pair_agreement",
    "linear_quantile",
    "pearson",
    "signed_difference",
    "spearman",
    "validate_branch_split",
]
