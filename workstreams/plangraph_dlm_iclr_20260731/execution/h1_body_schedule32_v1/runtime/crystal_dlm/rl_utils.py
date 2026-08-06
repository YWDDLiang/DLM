"""Small RL utilities for MP-20 fixed-slot DLM experiments."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError:  # pragma: no cover
    torch = None
    F = None


REWARD_PARSE_FAILURE = -1.0
REWARD_GRAPH_FAILURE = -0.8
REWARD_SMACT_INVALID = -0.4
REWARD_SMACT_VALID = 1.0
REWARD_MIN = -1.2
REWARD_MAX = 1.0
REWARD_MODES = (
    "simple",
    "shortcut_aware",
    "shortcut_strict",
    "reason_aware",
    "comp_valid_priority",
    "comp_valid_binary",
)
SHORTCUT_REASONS = {"single_element_shortcut", "all_metal_shortcut"}


def formula_key(diagnostics: Mapping[str, Any]) -> str | None:
    formula = diagnostics.get("formula")
    return str(formula) if formula else None


def element_set_key(diagnostics: Mapping[str, Any]) -> str | None:
    symbols = diagnostics.get("symbols")
    if not symbols:
        return None
    return "|".join(sorted(str(symbol) for symbol in symbols))


def base_reward(parse_ok: bool, graph_ok: bool, smact_valid: bool) -> float:
    if not parse_ok:
        return REWARD_PARSE_FAILURE
    if not graph_ok:
        return REWARD_GRAPH_FAILURE
    return REWARD_SMACT_VALID if smact_valid else REWARD_SMACT_INVALID


def base_reward_for_reason(
    parse_ok: bool,
    graph_ok: bool,
    smact_valid: bool,
    reason: str | None,
    *,
    reward_mode: str = "simple",
) -> float:
    """Return the non-duplicate reward for a diagnosed rollout.

    ``simple`` preserves the first reward version. ``shortcut_aware`` keeps the
    reward intentionally small, but avoids rewarding trivial SMACT shortcuts as
    if they were MP-20-like charge-neutral compositions. ``shortcut_strict`` is
    the v2 fallback when shortcuts still rise: all-metal receives the same base
    reward as an invalid composition and single-element receives a stronger
    negative reward.
    """

    if reward_mode not in REWARD_MODES:
        raise ValueError(f"Unknown reward_mode: {reward_mode}")
    if reward_mode == "simple":
        return base_reward(parse_ok, graph_ok, smact_valid)
    if not parse_ok:
        return REWARD_PARSE_FAILURE
    if not graph_ok:
        return REWARD_GRAPH_FAILURE
    if reward_mode == "reason_aware":
        if reason == "charge_neutral_pauling_valid":
            return 1.0
        if reason == "all_metal_shortcut":
            return 0.1
        if reason == "single_element_shortcut":
            return -0.6
        if reason == "charge_neutrality_fail":
            return -0.55
        if reason == "pauling_fail_or_ratio_rejected":
            return -0.35
        if reason == "oxidation_state_missing":
            return -0.25
        return REWARD_SMACT_INVALID
    if reward_mode == "comp_valid_priority":
        if reason == "charge_neutral_pauling_valid":
            return 1.0
        if reason == "all_metal_shortcut":
            return 0.55
        if reason == "single_element_shortcut":
            return 0.25
        if reason == "charge_neutrality_fail":
            return -0.8
        if reason == "pauling_fail_or_ratio_rejected":
            return -0.6
        if reason == "oxidation_state_missing":
            return -0.45
        return REWARD_SMACT_INVALID
    if reward_mode == "comp_valid_binary":
        if reason in {"charge_neutral_pauling_valid", "all_metal_shortcut", "single_element_shortcut"}:
            return 1.0
        return -1.0
    if not smact_valid:
        return REWARD_SMACT_INVALID
    if reason == "charge_neutral_pauling_valid":
        return REWARD_SMACT_VALID
    if reward_mode == "shortcut_strict":
        if reason == "all_metal_shortcut":
            return REWARD_SMACT_INVALID
        if reason == "single_element_shortcut":
            return -0.6
    if reason == "all_metal_shortcut":
        return 0.0
    if reason == "single_element_shortcut":
        return -0.2
    return 0.0


def duplicate_penalty(count: int, *, per_extra: float, floor: float) -> float:
    count = max(1, int(count))
    return max(float(floor), -float(per_extra) * max(0, count - 1))


def score_rollout_records(
    records: Sequence[dict[str, Any]], *, reward_mode: str = "simple"
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Add reward fields to rollout diagnostics.

    The default first reward version stays simple: failure class, SMACT
    validity, and batch-level duplicate formula / element-set penalties. The v2
    ``shortcut_aware`` mode keeps the same interface but does not give full
    reward to all-metal or single-element shortcuts.
    """

    if reward_mode not in REWARD_MODES:
        raise ValueError(f"Unknown reward_mode: {reward_mode}")
    formula_counts = Counter(
        key for row in records if (key := formula_key(row.get("diagnostics", {}))) is not None
    )
    element_set_counts = Counter(
        key for row in records if (key := element_set_key(row.get("diagnostics", {}))) is not None
    )
    scored: list[dict[str, Any]] = []
    reward_values: list[float] = []
    for row in records:
        new_row = dict(row)
        diagnostics = dict(new_row.get("diagnostics") or {})
        parse_ok = bool(diagnostics.get("parse_ok", False))
        graph_ok = bool(diagnostics.get("graph_ok", False))
        smact_valid = bool(diagnostics.get("smact_valid", False))
        reason = str(diagnostics.get("smact_reason", diagnostics.get("reason", "unknown")))
        formula = formula_key(diagnostics)
        element_set = element_set_key(diagnostics)
        raw_reward = base_reward_for_reason(
            parse_ok,
            graph_ok,
            smact_valid,
            reason,
            reward_mode=reward_mode,
        )
        formula_count = formula_counts.get(formula, 1) if formula else 1
        element_set_count = element_set_counts.get(element_set, 1) if element_set else 1
        if reward_mode == "comp_valid_binary":
            formula_penalty = 0.0
            element_set_penalty = 0.0
        elif reward_mode == "comp_valid_priority":
            formula_penalty = duplicate_penalty(formula_count, per_extra=0.02, floor=-0.1)
            element_set_penalty = duplicate_penalty(element_set_count, per_extra=0.01, floor=-0.05)
        else:
            formula_penalty = duplicate_penalty(formula_count, per_extra=0.1, floor=-0.3)
            element_set_penalty = duplicate_penalty(element_set_count, per_extra=0.05, floor=-0.2)
        reward = min(REWARD_MAX, max(REWARD_MIN, raw_reward + formula_penalty + element_set_penalty))
        if bool(diagnostics.get("has_pbc_equivalent_duplicate", False)):
            reward = min(reward, REWARD_GRAPH_FAILURE)
        diagnostics.update(
            {
                "base_reward": raw_reward,
                "reward_mode": reward_mode,
                "formula_duplicate_count": int(formula_count),
                "element_set_duplicate_count": int(element_set_count),
                "formula_duplicate_penalty": formula_penalty,
                "element_set_duplicate_penalty": element_set_penalty,
            }
        )
        new_row["diagnostics"] = diagnostics
        new_row["reward"] = float(reward)
        scored.append(new_row)
        reward_values.append(float(reward))

    summary = summarize_scored_records(scored, reward_values)
    summary["reward_mode"] = reward_mode
    return scored, summary


def summarize_scored_records(records: Sequence[Mapping[str, Any]], rewards: Sequence[float] | None = None) -> dict[str, Any]:
    rewards = list(rewards if rewards is not None else [float(row.get("reward", 0.0)) for row in records])
    reason_counts: Counter[str] = Counter()
    formula_counts: Counter[str] = Counter()
    element_set_counts: Counter[str] = Counter()
    parse_ok = graph_ok = smact_valid = 0
    shortcut = 0
    charge_neutral_pauling_valid = 0
    all_metal_shortcut = 0
    single_element_shortcut = 0
    pbc_equivalent_duplicate = 0
    for row in records:
        diagnostics = row.get("diagnostics", {}) or {}
        parse_ok += int(bool(diagnostics.get("parse_ok", False)))
        graph_ok += int(bool(diagnostics.get("graph_ok", False)))
        smact_valid += int(bool(diagnostics.get("smact_valid", False)))
        reason = str(diagnostics.get("smact_reason", diagnostics.get("reason", "unknown")))
        reason_counts[reason] += 1
        shortcut += int(reason in SHORTCUT_REASONS)
        charge_neutral_pauling_valid += int(reason == "charge_neutral_pauling_valid")
        all_metal_shortcut += int(reason == "all_metal_shortcut")
        single_element_shortcut += int(reason == "single_element_shortcut")
        pbc_equivalent_duplicate += int(bool(diagnostics.get("has_pbc_equivalent_duplicate", False)))
        if (key := formula_key(diagnostics)) is not None:
            formula_counts[key] += 1
        if (key := element_set_key(diagnostics)) is not None:
            element_set_counts[key] += 1
    count = len(records)
    reward_mean = sum(rewards) / max(1, len(rewards))
    reward_var = sum((value - reward_mean) ** 2 for value in rewards) / max(1, len(rewards))
    return {
        "count": count,
        "parse_ok": parse_ok,
        "graph_ok": graph_ok,
        "smact_valid": smact_valid,
        "parse_rate": parse_ok / max(1, count),
        "graph_rate": graph_ok / max(1, count),
        "smact_valid_rate": smact_valid / max(1, count),
        "charge_neutral_pauling_valid_rate": charge_neutral_pauling_valid / max(1, count),
        "shortcut_rate": shortcut / max(1, count),
        "all_metal_shortcut_rate": all_metal_shortcut / max(1, count),
        "single_element_shortcut_rate": single_element_shortcut / max(1, count),
        "pbc_equivalent_duplicate_rate": pbc_equivalent_duplicate / max(1, count),
        "reward_mean": reward_mean,
        "reward_std": math.sqrt(reward_var),
        "reward_min": min(rewards) if rewards else None,
        "reward_max": max(rewards) if rewards else None,
        "reason_counts": dict(reason_counts.most_common()),
        "formula_top20": dict(formula_counts.most_common(20)),
        "element_set_top20": dict(element_set_counts.most_common(20)),
    }


def normalize_rewards(rewards: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if torch is None:
        raise RuntimeError("torch is required for reward normalization")
    rewards = rewards.float()
    if rewards.numel() == 0:
        return rewards
    std = rewards.std(unbiased=False)
    if not torch.isfinite(std) or std.item() < eps:
        return torch.zeros_like(rewards)
    return (rewards - rewards.mean()) / (std + eps)


def collapse_step_map(step_map: Sequence[int], shrink: int) -> list[int]:
    """Collapse unique nonnegative denoising steps into groups of ``shrink``."""

    shrink = max(1, int(shrink))
    unique_steps = sorted({int(step) for step in step_map if int(step) >= 0})
    mapping: dict[int, int] = {}
    for group_start in range(0, len(unique_steps), shrink):
        group = unique_steps[group_start : group_start + shrink]
        if not group:
            continue
        representative = group[-1]
        for step in group:
            mapping[step] = representative
    return [mapping.get(int(step), -1) for step in step_map]


@dataclass
class TraceItem:
    input_ids: torch.Tensor
    noisy_ids: torch.Tensor
    attention_mask: torch.Tensor
    p_mask: torch.Tensor
    token_weights: torch.Tensor
    reward: float
    sample_idx: int
    old_logp: torch.Tensor | None = None


def trace_items_from_sequence(
    input_ids: torch.Tensor,
    prompt_length: int,
    step_map: Sequence[int],
    reward: float,
    *,
    mask_id: int,
    shrink: int = 8,
    max_states: int | None = None,
    sample_idx: int = -1,
    token_weights: Sequence[float] | None = None,
) -> list[TraceItem]:
    """Build TraceRL masked states for one prompt+answer sequence."""

    if torch is None:
        raise RuntimeError("torch is required for TraceRL item construction")
    seq_len = int(input_ids.numel())
    prompt_length = max(0, min(int(prompt_length), seq_len))
    answer_len = seq_len - prompt_length
    collapsed = collapse_step_map(list(step_map)[:answer_len], shrink=shrink)
    if len(collapsed) < answer_len:
        collapsed.extend([-1] * (answer_len - len(collapsed)))
    unique_steps = sorted({step for step in collapsed if step >= 0})
    if max_states is not None and len(unique_steps) > max_states:
        stride = max(1, math.ceil(len(unique_steps) / int(max_states)))
        unique_steps = unique_steps[::stride][: int(max_states)]

    items: list[TraceItem] = []
    full_steps = torch.full((seq_len,), -1, dtype=torch.long)
    if answer_len > 0:
        full_steps[prompt_length:] = torch.tensor(collapsed[:answer_len], dtype=torch.long)
    attention_mask = torch.ones((seq_len,), dtype=torch.long)
    full_weights = torch.ones((seq_len,), dtype=torch.float32)
    if token_weights is not None and answer_len > 0:
        answer_weights = list(float(value) for value in token_weights[:answer_len])
        if len(answer_weights) < answer_len:
            answer_weights.extend([1.0] * (answer_len - len(answer_weights)))
        full_weights[prompt_length:] = torch.tensor(answer_weights[:answer_len], dtype=torch.float32)
    for step in unique_steps:
        p_mask = full_steps == int(step)
        if not p_mask.any():
            continue
        if token_weights is not None and float(full_weights[p_mask].sum().item()) <= 0.0:
            continue
        noisy = input_ids.clone()
        noisy[full_steps >= int(step)] = int(mask_id)
        items.append(
            TraceItem(
                input_ids=input_ids.clone(),
                noisy_ids=noisy,
                attention_mask=attention_mask.clone(),
                p_mask=p_mask,
                token_weights=full_weights.clone(),
                reward=float(reward),
                sample_idx=int(sample_idx),
            )
        )
    return items


def trace_policy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    p_mask: torch.Tensor,
    old_logp: torch.Tensor,
    advantages: torch.Tensor,
    token_weights: torch.Tensor | None = None,
    *,
    clip_eps: float = 0.2,
    beta: float = 0.01,
    use_kl_estimator_k3: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    """PPO-like token-level objective used by the lightweight TraceRL trainer."""

    if torch is None or F is None:
        raise RuntimeError("torch is required for TraceRL policy loss")
    log_probs = F.log_softmax(logits, dim=-1)
    safe_labels = labels.masked_fill(labels < 0, 0)
    logp_new = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    diff = (logp_new - old_logp).clamp(min=-20.0, max=20.0)
    ratio = torch.exp(diff)
    clipped = torch.clamp(ratio, 1.0 - float(clip_eps), 1.0 + float(clip_eps))
    adv = advantages.float().unsqueeze(1)
    weights = p_mask.float()
    if token_weights is not None:
        weights = weights * token_weights.float().clamp_min(0.0)
    surrogate = torch.minimum(ratio * adv, clipped * adv) * weights
    token_counts = weights.sum(dim=1).clamp_min(1.0)
    policy_loss = -((surrogate.sum(dim=1) / token_counts).mean())

    kl_loss = torch.zeros((), dtype=policy_loss.dtype, device=policy_loss.device)
    if beta > 0.0:
        if use_kl_estimator_k3:
            kl_tokens = torch.exp(-diff) - 1.0 + diff
        else:
            kl_tokens = -diff
        kl_loss = float(beta) * ((kl_tokens * weights).sum(dim=1) / token_counts).mean()
    total = policy_loss + kl_loss
    return total, {
        "policy_loss": float(policy_loss.detach().cpu()),
        "kl_loss": float(kl_loss.detach().cpu()),
        "ratio_mean": float((ratio[p_mask].mean() if p_mask.any() else ratio.mean()).detach().cpu()),
        "token_weight_mean": float((weights[p_mask].mean() if p_mask.any() else weights.mean()).detach().cpu()),
    }
