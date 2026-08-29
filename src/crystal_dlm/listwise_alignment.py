"""Pure LAIR-style listwise alignment utilities for masked crystal DLMs.

The caller is responsible for producing every candidate score in a group with
one shared timestep, mask probability, and typed geometry mask.  This module
contains no model, tokenizer, trainer, data loading, sampling, or job logic.
All scientific coefficients are required call arguments so they cannot be
silently selected from main or sealed holdout outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch


@dataclass(frozen=True)
class ListwiseTargets:
    """Robust continuous rewards and centered softmax advantages."""

    rewards: torch.Tensor
    probabilities: torch.Tensor
    advantages: torch.Tensor


@dataclass(frozen=True)
class ListwiseLossOutput:
    """Scalar loss components plus unreduced group diagnostics."""

    loss: torch.Tensor
    linear_loss: torch.Tensor
    quadratic_loss: torch.Tensor
    best_anchor_loss: torch.Tensor
    rewards: torch.Tensor
    probabilities: torch.Tensor
    advantages: torch.Tensor
    implicit_scores: torch.Tensor
    best_index: int


def _group_vector(
    value: torch.Tensor | Sequence[float],
    *,
    name: str,
    minimum_candidates: int = 2,
) -> torch.Tensor:
    result = torch.as_tensor(value)
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if result.numel() < minimum_candidates:
        raise ValueError(
            f"{name} must contain at least {minimum_candidates} candidates"
        )
    result = result.to(torch.float32)
    if not bool(torch.isfinite(result).all().item()):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_finite(value: float, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _nonnegative_finite(value: float, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _median(values: torch.Tensor) -> torch.Tensor:
    ordered = torch.sort(values).values
    count = ordered.numel()
    midpoint = count // 2
    if count % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2.0


def robust_centered_rewards(
    energies: torch.Tensor | Sequence[float],
    *,
    mad_scale: float,
    scale_floor: float,
    reward_clip: float,
) -> torch.Tensor:
    """Map lower energies to higher robust within-composition rewards.

    The reward is ``-(E - median(E)) / max(mad_scale * MAD(E), scale_floor)``
    followed by symmetric clipping.  ``mad_scale``, ``scale_floor``, and
    ``reward_clip`` are explicit train-only calibration constants.
    """

    values = _group_vector(energies, name="energies")
    mad_multiplier = _positive_finite(mad_scale, name="mad_scale")
    minimum_scale = _positive_finite(scale_floor, name="scale_floor")
    clipping = _positive_finite(reward_clip, name="reward_clip")
    center = _median(values)
    mad = _median(torch.abs(values - center))
    scale = torch.clamp(mad * mad_multiplier, min=minimum_scale)
    rewards = -(values - center) / scale
    return torch.clamp(rewards, min=-clipping, max=clipping).to(torch.float32)


def softmax_advantages(
    rewards: torch.Tensor | Sequence[float],
    *,
    temperature: float,
) -> ListwiseTargets:
    """Return softmax probabilities and exactly centered group advantages."""

    values = _group_vector(rewards, name="rewards")
    tau = _positive_finite(temperature, name="temperature")
    probabilities = torch.softmax(values / tau, dim=0).to(torch.float32)
    advantages = probabilities - probabilities.mean()
    # One additional centering pass removes the last FP32 reduction residue.
    advantages = advantages - advantages.mean()
    return ListwiseTargets(
        rewards=values,
        probabilities=probabilities,
        advantages=advantages.to(torch.float32),
    )


def listwise_lair_loss(
    implicit_scores: torch.Tensor | Sequence[float],
    advantages: torch.Tensor | Sequence[float],
    candidate_denoising_losses: torch.Tensor | Sequence[float],
    *,
    best_index: int,
    linear_weight: float,
    quadratic_weight: float,
    best_anchor_weight: float,
    group_weight: float,
    advantage_sum_tolerance: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the bounded linear listwise objective for one composition.

    ``implicit_scores`` must be reference-corrected ``1/(p|G|)`` masked
    sequence scores evaluated under one shared-noise corruption.  The returned
    tuple is ``(total, weighted_linear, weighted_quadratic, best_anchor_nll)``.
    """

    scores = _group_vector(implicit_scores, name="implicit_scores")
    group_advantages = _group_vector(advantages, name="advantages")
    anchors = _group_vector(
        candidate_denoising_losses,
        name="candidate_denoising_losses",
    )
    if scores.device != group_advantages.device or scores.device != anchors.device:
        raise ValueError("scores, advantages, and anchors must share a device")
    if scores.shape != group_advantages.shape or scores.shape != anchors.shape:
        raise ValueError("scores, advantages, and anchors must have identical shape")
    index = int(best_index)
    if index < 0 or index >= scores.numel():
        raise ValueError("best_index is outside the candidate group")

    linear_coefficient = _nonnegative_finite(
        linear_weight, name="linear_weight"
    )
    quadratic_coefficient = _positive_finite(
        quadratic_weight, name="quadratic_weight"
    )
    anchor_coefficient = _nonnegative_finite(
        best_anchor_weight, name="best_anchor_weight"
    )
    composition_weight = _positive_finite(group_weight, name="group_weight")
    if composition_weight != 1.0:
        raise ValueError("group_weight must equal one for composition normalization")
    tolerance = _positive_finite(
        advantage_sum_tolerance,
        name="advantage_sum_tolerance",
    )
    if abs(float(group_advantages.sum().item())) > tolerance:
        raise ValueError("advantages must sum to zero within the composition")

    linear_loss = -linear_coefficient * torch.sum(group_advantages * scores)
    quadratic_loss = quadratic_coefficient * torch.mean(scores.square())
    best_anchor_loss = anchors[index]
    total = composition_weight * (
        linear_loss
        + quadratic_loss
        + anchor_coefficient * best_anchor_loss
    )
    return (
        total.to(torch.float32),
        linear_loss.to(torch.float32),
        quadratic_loss.to(torch.float32),
        best_anchor_loss.to(torch.float32),
    )


def shared_noise_listwise_alignment_loss(
    implicit_scores: torch.Tensor | Sequence[float],
    energies: torch.Tensor | Sequence[float],
    candidate_denoising_losses: torch.Tensor | Sequence[float],
    *,
    mad_scale: float,
    scale_floor: float,
    reward_clip: float,
    reward_temperature: float,
    linear_weight: float,
    quadratic_weight: float,
    best_anchor_weight: float,
    group_weight: float,
    advantage_sum_tolerance: float,
) -> ListwiseLossOutput:
    """Build continuous targets and evaluate one shared-noise listwise group."""

    scores = _group_vector(implicit_scores, name="implicit_scores")
    energy_values = _group_vector(energies, name="energies")
    anchors = _group_vector(
        candidate_denoising_losses,
        name="candidate_denoising_losses",
    )
    if scores.device != energy_values.device or scores.device != anchors.device:
        raise ValueError("scores, energies, and anchors must share a device")
    if scores.shape != energy_values.shape or scores.shape != anchors.shape:
        raise ValueError("scores, energies, and anchors must have identical shape")

    rewards = robust_centered_rewards(
        energy_values,
        mad_scale=mad_scale,
        scale_floor=scale_floor,
        reward_clip=reward_clip,
    )
    targets = softmax_advantages(rewards, temperature=reward_temperature)
    best_index = int(torch.argmin(energy_values).item())
    total, linear, quadratic, anchor = listwise_lair_loss(
        scores,
        targets.advantages,
        anchors,
        best_index=best_index,
        linear_weight=linear_weight,
        quadratic_weight=quadratic_weight,
        best_anchor_weight=best_anchor_weight,
        group_weight=group_weight,
        advantage_sum_tolerance=advantage_sum_tolerance,
    )
    return ListwiseLossOutput(
        loss=total,
        linear_loss=linear,
        quadratic_loss=quadratic,
        best_anchor_loss=anchor,
        rewards=targets.rewards,
        probabilities=targets.probabilities,
        advantages=targets.advantages,
        implicit_scores=scores,
        best_index=best_index,
    )


__all__ = [
    "ListwiseLossOutput",
    "ListwiseTargets",
    "listwise_lair_loss",
    "robust_centered_rewards",
    "shared_noise_listwise_alignment_loss",
    "softmax_advantages",
]
