"""Fail-closed helpers for CTV-DLM counterfactual branch rollouts.

The module deliberately separates protocol bookkeeping from the expensive
model/refiner pipeline.  In particular, action identity never enters the
continuation-noise key, so actions from one state receive common random
numbers rather than accidentally different sampling luck.
"""

from __future__ import annotations

from collections import Counter
import math
from typing import Any, Hashable, Mapping, Sequence

from crystal_dlm.ctv_protocol import counter_seed, select_eight_legal_actions


CTV_MILESTONES = (0.60, 0.80)


def free_geometry_positions(num_atoms: int) -> tuple[int, ...]:
    """Return relative suffix positions for lattice and XYZ, not N/elements."""

    count = int(num_atoms)
    if not 1 <= count <= 20:
        raise ValueError("CTV num_atoms must lie in 1..20")
    positions = list(range(1, 7))
    for slot in range(count):
        base = 7 + 4 * slot
        positions.extend((base + 1, base + 2, base + 3))
    expected = 6 + 3 * count
    if len(positions) != expected or len(set(positions)) != expected:
        raise RuntimeError("CTV free-geometry position construction changed")
    return tuple(positions)


def visible_free_geometry_fraction(
    suffix_token_ids: Sequence[int], *, mask_id: int, num_atoms: int
) -> float:
    positions = free_geometry_positions(num_atoms)
    if not positions or max(positions) >= len(suffix_token_ids):
        raise ValueError("suffix is too short for the declared atom count")
    visible = sum(int(suffix_token_ids[position]) != int(mask_id) for position in positions)
    return float(visible) / float(len(positions))


def newly_crossed_milestones(
    before: float,
    after: float,
    *,
    pending: Sequence[float] = CTV_MILESTONES,
) -> tuple[float, ...]:
    lower = float(before)
    upper = float(after)
    if not (0.0 <= lower <= upper <= 1.0):
        raise ValueError("visible fractions must be monotone inside [0,1]")
    values = tuple(float(value) for value in pending)
    if tuple(sorted(values)) != values or len(set(values)) != len(values):
        raise ValueError("CTV milestones must be sorted and unique")
    return tuple(value for value in values if lower < value <= upper)


def make_branch_layout(
    *,
    composition_id: str,
    sample_idx: int,
    milestone: float,
    intervention_position: int,
    action_token_ids: Sequence[int],
    continuation_seeds: Sequence[int],
) -> list[dict[str, Any]]:
    actions = [int(value) for value in action_token_ids]
    continuations = [int(value) for value in continuation_seeds]
    if len(actions) != 8 or len(set(actions)) != 8:
        raise ValueError("CTV branch state requires exactly eight distinct actions")
    if len(continuations) != 2 or len(set(continuations)) != 2:
        raise ValueError("CTV resource canary requires two distinct continuations")
    if not composition_id:
        raise ValueError("CTV composition identity must be non-empty")
    rows: list[dict[str, Any]] = []
    for action_rank, action in enumerate(actions):
        for continuation_rank, continuation in enumerate(continuations):
            # Deliberately omit action from the continuation-noise group.  The
            # branch id still includes action; only the common random numbers do not.
            noise_group = counter_seed(
                "ctv-continuation-v1",
                composition_id,
                int(sample_idx),
                f"{float(milestone):.2f}",
                int(intervention_position),
                int(continuation),
            )
            rows.append(
                {
                    "composition_id": str(composition_id),
                    "sample_idx": int(sample_idx),
                    "milestone": float(milestone),
                    "intervention_position": int(intervention_position),
                    "action_token": action,
                    "action_rank": action_rank,
                    "continuation_seed": continuation,
                    "continuation_rank": continuation_rank,
                    "noise_group": int(noise_group),
                }
            )
    return rows


def validate_canary_layout(
    rows: Sequence[Mapping[str, Any]], *, expected_plans: int = 8
) -> dict[str, int]:
    expected = int(expected_plans) * 2 * 8 * 2
    if len(rows) != expected:
        raise ValueError(f"CTV canary has {len(rows)} rows, expected {expected}")
    keys: set[tuple[Any, ...]] = set()
    state_counts: Counter[tuple[str, int, float, int]] = Counter()
    noise_actions: dict[tuple[str, int, float, int, int], set[int]] = {}
    plan_milestones: dict[tuple[str, int], set[float]] = {}
    for row in rows:
        state = (
            str(row["composition_id"]),
            int(row["sample_idx"]),
            float(row["milestone"]),
            int(row["intervention_position"]),
        )
        key = (*state, int(row["action_token"]), int(row["continuation_seed"]))
        if key in keys:
            raise ValueError("CTV canary repeats a branch identity")
        keys.add(key)
        state_counts[state] += 1
        plan_milestones.setdefault(state[:2], set()).add(state[2])
        noise_key = (*state, int(row["continuation_seed"]))
        noise_actions.setdefault(noise_key, set()).add(int(row["noise_group"]))
    if len(plan_milestones) != int(expected_plans):
        raise ValueError("CTV canary Plan count changed")
    if any(values != set(CTV_MILESTONES) for values in plan_milestones.values()):
        raise ValueError("every CTV canary Plan requires both frozen milestones")
    if any(count != 16 for count in state_counts.values()):
        raise ValueError("every CTV state requires eight actions by two continuations")
    if any(len(values) != 1 for values in noise_actions.values()):
        raise ValueError("actions in one continuation do not share common random numbers")
    return {
        "rows": len(rows),
        "plans": len(plan_milestones),
        "states": len(state_counts),
        "unique_branches": len(keys),
        "common_noise_groups": len(noise_actions),
    }


def select_intervention_from_masked_logits(
    *,
    logits: Any,
    suffix_token_ids: Any,
    allowed_token_ids_by_generation_pos: Sequence[Sequence[int]],
    num_atoms: int,
    mask_id: int,
    eligible_positions: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Select the masked geometry position with highest frozen-base confidence.

    ``logits`` must already include every frozen schema/lightweight mask.  The
    function is kept here (instead of in the sampler) so its exact confidence
    and tie rules are unit-testable.
    """

    import torch

    if logits.ndim != 2 or suffix_token_ids.ndim != 1:
        raise ValueError("CTV intervention selection expects [length,vocab] logits")
    if logits.shape[0] != suffix_token_ids.shape[0]:
        raise ValueError("CTV logits/token suffix lengths differ")
    geometry = set(free_geometry_positions(num_atoms))
    positions = (
        list(free_geometry_positions(num_atoms))
        if eligible_positions is None
        else [int(value) for value in eligible_positions]
    )
    if not positions or any(position not in geometry for position in positions):
        raise ValueError("CTV intervention positions must be non-empty free geometry")
    candidates: list[tuple[float, int, list[int], list[float]]] = []
    for position in positions:
        if int(suffix_token_ids[position].detach().item()) != int(mask_id):
            continue
        legal_ids = [int(value) for value in allowed_token_ids_by_generation_pos[position]]
        if len(legal_ids) < 8:
            continue
        indices = torch.tensor(legal_ids, dtype=torch.long, device=logits.device)
        legal_logits = logits[position].index_select(0, indices).to(torch.float64)
        if not bool(torch.isfinite(legal_logits).any()):
            continue
        probabilities = torch.softmax(legal_logits, dim=0)
        raw_values = [
            float(value) for value in probabilities.detach().cpu().tolist()
        ]
        positive = [
            (token, probability)
            for token, probability in zip(legal_ids, raw_values, strict=True)
            if probability > 0.0 and math.isfinite(probability)
        ]
        if len(positive) < 8:
            continue
        legal_ids = [token for token, _probability in positive]
        total = sum(probability for _token, probability in positive)
        values = [probability / total for _token, probability in positive]
        confidence = max(values)
        candidates.append((confidence, int(position), legal_ids, values))
    if not candidates:
        raise ValueError("CTV state has no masked legal free-geometry position")
    # Higher confidence first; a lower relative suffix position wins exact ties.
    confidence, position, legal_ids, values = min(
        candidates, key=lambda value: (-value[0], value[1])
    )
    actions = select_eight_legal_actions(values, legal_ids)
    probability_by_token = dict(zip(legal_ids, values, strict=True))
    return {
        "position": position,
        "confidence": float(confidence),
        "action_token_ids": actions,
        "action_probabilities": tuple(probability_by_token[token] for token in actions),
        "legal_token_count": len(legal_ids),
        "legal_probability_sum": float(sum(values)),
    }


def stateless_gumbel_scores(
    logits: Any,
    *,
    temperature: float,
    noise_groups: Sequence[Hashable],
    denoise_step: int,
) -> Any:
    """Apply per-step counter-based Gumbel noise with common branch randomness."""

    import torch

    if logits.ndim != 3 or logits.shape[0] != len(noise_groups):
        raise ValueError("one CTV noise group is required for every logits row")
    value = float(temperature)
    if value < 0.0 or not math.isfinite(value):
        raise ValueError("CTV temperature must be finite and non-negative")
    if value == 0.0:
        return logits
    rows = []
    for row, noise_group in zip(logits, noise_groups, strict=True):
        generator = torch.Generator(device=row.device)
        seed = counter_seed(
            "ctv-gumbel-v1", noise_group, int(denoise_step), row.shape[0], row.shape[1]
        ) & ((1 << 63) - 1)
        generator.manual_seed(seed)
        uniform = torch.rand(
            row.shape,
            generator=generator,
            device=row.device,
            dtype=torch.float64,
        )
        rows.append(row.to(torch.float64).exp() / ((-torch.log(uniform)) ** value))
    return torch.stack(rows)


def require_gamma_zero_identity(
    base_logits: Any, guided_logits: Any, *, atol: float = 0.0
) -> dict[str, float | bool]:
    import torch

    if base_logits.shape != guided_logits.shape:
        raise ValueError("gamma-zero logits shape changed")
    absolute = (base_logits.to(torch.float64) - guided_logits.to(torch.float64)).abs()
    maximum = float(absolute.max().detach().cpu().item()) if absolute.numel() else 0.0
    passed = bool(maximum <= float(atol))
    if not passed:
        raise ValueError(f"gamma-zero logits differ from base by {maximum}")
    return {"passed": passed, "max_abs_delta": maximum, "atol": float(atol)}


__all__ = [
    "CTV_MILESTONES",
    "free_geometry_positions",
    "make_branch_layout",
    "newly_crossed_milestones",
    "require_gamma_zero_identity",
    "select_intervention_from_masked_logits",
    "stateless_gumbel_scores",
    "validate_canary_layout",
    "visible_free_geometry_fraction",
]
