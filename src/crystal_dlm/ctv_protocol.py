"""Fail-closed protocol helpers for Counterfactual Terminal-cost Value DLM."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Iterable, Mapping, Sequence


CTV_ACTION_QUANTILES = (0.05, 0.15, 0.30, 0.50, 0.70, 0.85, 0.95)


def counter_seed(*parts: object) -> int:
    payload = json.dumps(parts, ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def branch_record_id(
    *,
    composition_id: str,
    sample_idx: int,
    milestone: float,
    position: int,
    action_token: int,
    continuation_seed: int,
) -> str:
    payload = {
        "composition_id": str(composition_id),
        "sample_idx": int(sample_idx),
        "milestone": f"{float(milestone):.2f}",
        "position": int(position),
        "action_token": int(action_token),
        "continuation_seed": int(continuation_seed),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def select_eight_legal_actions(
    probabilities: Sequence[float],
    legal_token_ids: Sequence[int],
    *,
    quantiles: Sequence[float] = CTV_ACTION_QUANTILES,
) -> tuple[int, ...]:
    if len(probabilities) != len(legal_token_ids) or not probabilities:
        raise ValueError("probabilities and legal tokens must be aligned and non-empty")
    if len(set(int(value) for value in legal_token_ids)) != len(legal_token_ids):
        raise ValueError("legal token ids must be unique")
    values = [float(value) for value in probabilities]
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("legal probabilities must be finite and non-negative")
    total = sum(values)
    if total <= 0.0:
        raise ValueError("legal probability mass must be positive")
    normalized = [value / total for value in values]
    ordered = sorted(
        zip((int(value) for value in legal_token_ids), normalized),
        key=lambda pair: pair[0],
    )
    argmax_token = min(
        (
            (int(token), float(probability))
            for token, probability in zip(legal_token_ids, normalized)
        ),
        key=lambda pair: (-pair[1], pair[0]),
    )[0]
    selected = [argmax_token]
    for raw_quantile in quantiles:
        quantile = float(raw_quantile)
        if not 0.0 < quantile < 1.0:
            raise ValueError("action quantiles must lie strictly inside (0,1)")
        cumulative = 0.0
        token_at_quantile = ordered[-1][0]
        for token, probability in ordered:
            cumulative += probability
            if cumulative >= quantile:
                token_at_quantile = token
                break
        selected.append(int(token_at_quantile))
    if len(selected) != 8 or len(set(selected)) != 8:
        raise ValueError(
            "CTV action protocol requires argmax plus seven distinct quantile tokens"
        )
    return tuple(selected)


def apply_energy_guidance(
    base_logits: Sequence[float],
    centered_costs: Sequence[float],
    supported: Sequence[bool],
    *,
    gamma: float,
) -> tuple[float, ...]:
    if not (
        len(base_logits) == len(centered_costs) == len(supported)
        and len(base_logits) > 0
    ):
        raise ValueError("guidance arrays must share a positive length")
    coefficient = float(gamma)
    if coefficient < 0.0 or not math.isfinite(coefficient):
        raise ValueError("gamma must be finite and non-negative")
    guided: list[float] = []
    for logit, cost, is_supported in zip(base_logits, centered_costs, supported):
        value = float(logit)
        if not math.isfinite(value):
            guided.append(value)
            continue
        advantage = float(cost) if bool(is_supported) else 0.0
        if not math.isfinite(advantage):
            raise ValueError("supported centered cost must be finite")
        guided.append(value - coefficient * advantage)
    return tuple(guided)


def validate_branch_ledger(
    rows: Iterable[Mapping[str, Any]], *, expected_rows: int
) -> dict[str, int]:
    values = list(rows)
    ids = [str(row.get("branch_id") or "") for row in values]
    if len(values) != int(expected_rows):
        raise ValueError(f"branch ledger has {len(values)} rows, expected {expected_rows}")
    if any(not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("branch ids must be non-empty and unique")
    state_actions: dict[str, set[int]] = {}
    for row in values:
        state_id = str(row.get("state_id") or "")
        action = int(row["action_token"])
        if not state_id:
            raise ValueError("branch row lacks state_id")
        actions = state_actions.setdefault(state_id, set())
        if action in actions:
            raise ValueError(f"state {state_id} repeats action token {action}")
        actions.add(action)
    bad = {state: len(actions) for state, actions in state_actions.items() if len(actions) != 8}
    if bad:
        raise ValueError(f"every state requires exactly eight actions: {bad}")
    return {
        "rows": len(values),
        "unique_branch_ids": len(set(ids)),
        "states": len(state_actions),
    }


__all__ = [
    "CTV_ACTION_QUANTILES",
    "apply_energy_guidance",
    "branch_record_id",
    "counter_seed",
    "select_eight_legal_actions",
    "validate_branch_ledger",
]
