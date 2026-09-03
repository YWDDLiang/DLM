"""Pure finite-action energy posterior used by SPAD-E.

This module deliberately contains no tokenizer, model, data-loader, or job
code.  It turns exactly four already-enumerated XYZ action draws into a target
distribution.  Structural validity is lexicographic: illegal actions have
zero support before terminal energies are inspected.  Energy can only
reweight the remaining reference-DLM action probabilities inside a fixed KL
trust region.

The four entries are *draws*, not unique coordinate triplets.  Consequently,
duplicate sampled triplets remain separate entries in all probabilities and
accounting.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
import operator
from typing import Any

import torch
from torch import Tensor


SPAD_E_ACTION_COUNT = 4
SPAD_E_KL_BUDGET_NATS = 0.05

_MAD_NORMAL_CONSISTENCY = 1.4826
_NORMALIZED_ENERGY_CLIP = 8.0
_BISECTION_STEPS = 96
_BRACKET_STEPS = 64


@dataclass(frozen=True)
class CommonStateValidation:
    """Normalized diagnostics for one common-state finite-action group."""

    action_count: int
    duplicate_draw_count: int
    active_positions: tuple[int, int, int]
    differing_positions: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class RobustEnergyNormalization:
    """Robust, within-group terminal-energy normalization.

    ``normalized_energies`` is zero for invalid or energy-unknown actions.
    ``observed_mask`` records finite labels independently of legality, while
    ``preference_mask`` identifies finite labels that may shape the posterior.
    """

    normalized_energies: Tensor
    observed_mask: Tensor
    preference_mask: Tensor
    center: float | None
    scale: float | None
    informative: bool


@dataclass(frozen=True)
class SPADEnergyPosterior:
    """Reference and terminal-energy-shaped distributions for four draws."""

    target_probabilities: Tensor
    reference_probabilities: Tensor
    reference_log_probabilities: Tensor
    normalized_terminal_energies: Tensor
    legal_mask: Tensor
    terminal_energy_known_mask: Tensor
    energy_preference_mask: Tensor
    tilt: float
    kl_nats: float
    kl_budget_nats: float
    action_count: int
    legal_action_count: int
    known_energy_count: int
    unknown_energy_count: int
    legal_known_energy_count: int
    duplicate_draw_count: int
    infinite_tilt_limit: bool


@dataclass(frozen=True)
class SPADEnergyPosteriorLoss:
    """Differentiable policy loss and equivalent cross-entropy diagnostics."""

    loss: Tensor
    kl: Tensor
    cross_entropy: Tensor
    target_entropy: Tensor
    policy_log_probabilities: Tensor


def _exact_vector(value: Any, *, name: str, length: int) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence of {length} integers")
    if len(value) != length:
        raise ValueError(f"{name} must contain exactly {length} entries")
    result: list[int] = []
    for item in value:
        if isinstance(item, bool):
            raise TypeError(f"{name} entries must be integers, not bool")
        try:
            result.append(operator.index(item))
        except TypeError as exc:
            raise TypeError(f"{name} entries must be integers") from exc
    return tuple(result)


def _normalize_action_triplets(
    action_triplets: Sequence[Sequence[int]],
    *,
    no_op_triplet: Sequence[int],
) -> tuple[tuple[tuple[int, int, int], ...], tuple[int, int, int], int]:
    if isinstance(action_triplets, (str, bytes)) or not isinstance(
        action_triplets, Sequence
    ):
        raise TypeError("action_triplets must be a sequence")
    if len(action_triplets) != SPAD_E_ACTION_COUNT:
        raise ValueError("SPAD-E requires K exactly 4")
    actions = tuple(
        _exact_vector(action, name=f"action_triplets[{index}]", length=3)
        for index, action in enumerate(action_triplets)
    )
    no_op = _exact_vector(no_op_triplet, name="no_op_triplet", length=3)
    if actions[0] != no_op:
        raise ValueError("action 0 must be the mandatory no-op triplet")
    duplicate_draw_count = SPAD_E_ACTION_COUNT - len(set(actions))
    return actions, no_op, duplicate_draw_count


def _metadata_equal(left: Any, right: Any) -> bool:
    """Type-aware equality for JSON-like metadata and small tensors."""

    if isinstance(left, Tensor) or isinstance(right, Tensor):
        return (
            isinstance(left, Tensor)
            and isinstance(right, Tensor)
            and left.dtype == right.dtype
            and tuple(left.shape) == tuple(right.shape)
            and bool(torch.equal(left, right))
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            return False
        if set(left) != set(right):
            return False
        return all(_metadata_equal(left[key], right[key]) for key in left)
    sequence_types = (list, tuple)
    if isinstance(left, sequence_types) or isinstance(right, sequence_types):
        if type(left) is not type(right) or len(left) != len(right):
            return False
        return all(_metadata_equal(a, b) for a, b in zip(left, right))
    if type(left) is not type(right):
        return False
    return bool(left == right)


def validate_common_state_group(
    *,
    action_triplets: Sequence[Sequence[int]],
    no_op_triplet: Sequence[int],
    state_metadata_by_action: Sequence[Mapping[str, Any]],
    active_positions: Sequence[int],
    differing_positions_by_action: Sequence[Sequence[int]],
) -> CommonStateValidation:
    """Validate that four action draws alter one XYZ transaction only.

    The caller supplies metadata already extracted from its dataset.  This
    function neither tokenizes bodies nor loads a model.  All four metadata
    mappings must be exactly equal.  For every draw, reported differing body
    positions must equal the XYZ positions whose triplet values differ from
    the no-op.  Duplicate triplets are valid separate draws.
    """

    actions, no_op, duplicate_count = _normalize_action_triplets(
        action_triplets,
        no_op_triplet=no_op_triplet,
    )
    positions = _exact_vector(active_positions, name="active_positions", length=3)
    if any(position < 0 for position in positions):
        raise ValueError("active_positions must be non-negative")
    if len(set(positions)) != 3:
        raise ValueError("active_positions must contain three unique positions")

    if isinstance(state_metadata_by_action, (str, bytes)) or not isinstance(
        state_metadata_by_action, Sequence
    ):
        raise TypeError("state_metadata_by_action must be a sequence")
    if len(state_metadata_by_action) != SPAD_E_ACTION_COUNT:
        raise ValueError("state_metadata_by_action must contain exactly 4 mappings")
    if not all(isinstance(item, Mapping) for item in state_metadata_by_action):
        raise TypeError("every common-state metadata entry must be a mapping")
    common = state_metadata_by_action[0]
    for index, metadata in enumerate(state_metadata_by_action[1:], start=1):
        if not _metadata_equal(common, metadata):
            raise ValueError(
                f"action {index} does not share bitwise-identical state metadata"
            )

    if isinstance(differing_positions_by_action, (str, bytes)) or not isinstance(
        differing_positions_by_action, Sequence
    ):
        raise TypeError("differing_positions_by_action must be a sequence")
    if len(differing_positions_by_action) != SPAD_E_ACTION_COUNT:
        raise ValueError("differing_positions_by_action must contain exactly 4 entries")

    normalized_differences: list[tuple[int, ...]] = []
    for index, (action, reported) in enumerate(
        zip(actions, differing_positions_by_action)
    ):
        if isinstance(reported, (str, bytes)) or not isinstance(reported, Sequence):
            raise TypeError(f"differing_positions_by_action[{index}] must be a sequence")
        normalized = _exact_vector(
            reported,
            name=f"differing_positions_by_action[{index}]",
            length=len(reported),
        )
        if len(set(normalized)) != len(normalized):
            raise ValueError(f"action {index} reports duplicate differing positions")
        expected = tuple(
            position
            for position, candidate, source in zip(positions, action, no_op)
            if candidate != source
        )
        if set(normalized) != set(expected):
            raise ValueError(
                f"action {index} differing positions do not match its XYZ triplet"
            )
        normalized_differences.append(expected)

    return CommonStateValidation(
        action_count=SPAD_E_ACTION_COUNT,
        duplicate_draw_count=duplicate_count,
        active_positions=(positions[0], positions[1], positions[2]),
        differing_positions=tuple(normalized_differences),
    )


def _as_log_score_vector(value: Tensor | Sequence[float], *, name: str) -> Tensor:
    if isinstance(value, Tensor):
        if not value.is_floating_point():
            raise TypeError(f"{name} must be a floating-point tensor")
        result = value
    else:
        result = torch.as_tensor(value, dtype=torch.float64)
    if result.ndim != 1 or result.numel() != SPAD_E_ACTION_COUNT:
        raise ValueError(f"{name} must contain exactly 4 action scores")
    if not bool(torch.isfinite(result).all().item()):
        raise ValueError(f"{name} must be finite")
    return result


def _as_legal_mask(value: Tensor | Sequence[bool], *, device: torch.device) -> Tensor:
    if isinstance(value, Tensor):
        if value.dtype is not torch.bool:
            raise TypeError("legal_mask must be a bool tensor")
        result = value.to(device=device)
    else:
        result = torch.as_tensor(value, device=device)
        if result.dtype is not torch.bool:
            raise TypeError("legal_mask must contain booleans")
    if result.ndim != 1 or result.numel() != SPAD_E_ACTION_COUNT:
        raise ValueError("legal_mask must contain exactly 4 entries")
    if not bool(result[0].item()):
        raise ValueError("action 0 (the mandatory no-op) must be legal")
    return result


def _as_terminal_energies(
    value: Tensor | Sequence[float | None], *, device: torch.device
) -> Tensor:
    if isinstance(value, Tensor):
        if value.is_complex():
            raise TypeError("terminal_energies must be real-valued")
        result = value.detach().to(device=device, dtype=torch.float64)
    else:
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise TypeError("terminal_energies must be a sequence")
        converted: list[float] = []
        for item in value:
            if item is None:
                converted.append(math.nan)
            elif isinstance(item, bool):
                raise TypeError("terminal energies must be real values, not bool")
            else:
                converted.append(float(item))
        result = torch.tensor(converted, device=device, dtype=torch.float64)
    if result.ndim != 1 or result.numel() != SPAD_E_ACTION_COUNT:
        raise ValueError("terminal_energies must contain exactly 4 entries")
    return result


def _median(values: Tensor) -> Tensor:
    ordered = torch.sort(values).values
    midpoint = ordered.numel() // 2
    if ordered.numel() % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2.0


def robust_normalize_terminal_energies(
    terminal_energies: Tensor | Sequence[float | None],
    legal_mask: Tensor | Sequence[bool],
    *,
    device: torch.device | None = None,
) -> RobustEnergyNormalization:
    """Robustly normalize finite legal energies within one four-draw group.

    Median/MAD scaling is used.  If MAD is zero despite unequal labels, the
    median positive absolute deviation is the deterministic fallback.  Values
    are clipped only after normalization.  Unknown and invalid entries receive
    normalized value zero, so they have no direct energy tilt.
    """

    target_device = device or (
        terminal_energies.device
        if isinstance(terminal_energies, Tensor)
        else torch.device("cpu")
    )
    energies = _as_terminal_energies(terminal_energies, device=target_device)
    legal = _as_legal_mask(legal_mask, device=target_device)
    observed = torch.isfinite(energies)
    preference = observed & legal
    normalized = torch.zeros_like(energies)
    values = energies[preference]

    if values.numel() == 0:
        return RobustEnergyNormalization(
            normalized_energies=normalized,
            observed_mask=observed,
            preference_mask=preference,
            center=None,
            scale=None,
            informative=False,
        )

    center_tensor = _median(values)
    center = float(center_tensor.item())
    if values.numel() < 2 or bool(torch.all(values == values[0]).item()):
        return RobustEnergyNormalization(
            normalized_energies=normalized,
            observed_mask=observed,
            preference_mask=preference,
            center=center,
            scale=None,
            informative=False,
        )

    deviations = torch.abs(values - center_tensor)
    mad = _median(deviations)
    scale_tensor = mad * _MAD_NORMAL_CONSISTENCY
    epsilon = torch.finfo(torch.float64).eps
    if float(scale_tensor.item()) <= epsilon:
        positive_deviations = deviations[deviations > epsilon]
        if positive_deviations.numel() == 0:
            return RobustEnergyNormalization(
                normalized_energies=normalized,
                observed_mask=observed,
                preference_mask=preference,
                center=center,
                scale=None,
                informative=False,
            )
        scale_tensor = _median(positive_deviations)

    scaled = (values - center_tensor) / scale_tensor
    normalized[preference] = torch.clamp(
        scaled,
        min=-_NORMALIZED_ENERGY_CLIP,
        max=_NORMALIZED_ENERGY_CLIP,
    )
    return RobustEnergyNormalization(
        normalized_energies=normalized,
        observed_mask=observed,
        preference_mask=preference,
        center=center,
        scale=float(scale_tensor.item()),
        informative=True,
    )


def _tilted_distribution(
    reference_log_probabilities: Tensor,
    normalized_energies: Tensor,
    legal_mask: Tensor,
    tilt: float,
) -> tuple[Tensor, float]:
    legal_log_reference = reference_log_probabilities[legal_mask]
    legal_energy = normalized_energies[legal_mask]
    logits = legal_log_reference - float(tilt) * legal_energy
    legal_log_target = torch.log_softmax(logits, dim=0)
    legal_target = torch.exp(legal_log_target)
    target = torch.zeros_like(reference_log_probabilities)
    target[legal_mask] = legal_target
    kl = torch.sum(legal_target * (legal_log_target - legal_log_reference))
    return target, float(kl.item())


def _solve_strongest_tilt(
    reference_log_probabilities: Tensor,
    normalized_energies: Tensor,
    legal_mask: Tensor,
    *,
    kl_budget_nats: float,
) -> tuple[Tensor, float, float, bool]:
    budget = float(kl_budget_nats)
    if not math.isfinite(budget) or budget < 0.0:
        raise ValueError("kl_budget_nats must be finite and non-negative")
    if budget > SPAD_E_KL_BUDGET_NATS:
        raise ValueError("SPAD-E KL budget cannot exceed 0.05 nat")

    reference = torch.exp(reference_log_probabilities)
    legal_energies = normalized_energies[legal_mask]
    if (
        budget == 0.0
        or legal_energies.numel() == 1
        or bool(torch.all(legal_energies == legal_energies[0]).item())
    ):
        return reference, 0.0, 0.0, False

    # At infinite tilt, reference mass is renormalized over minimum-energy
    # actions.  This handles the rare case where even the limiting posterior
    # remains inside the KL budget and no largest finite coefficient exists.
    minimum_energy = torch.min(legal_energies)
    limiting_support = legal_mask & (normalized_energies == minimum_energy)
    limiting_mass = torch.sum(reference[limiting_support])
    limiting_target = torch.zeros_like(reference)
    limiting_target[limiting_support] = (
        reference[limiting_support] / limiting_mass
    )
    limiting_kl = -math.log(float(limiting_mass.item()))
    if limiting_kl <= budget:
        return limiting_target, math.inf, limiting_kl, True

    lower = 0.0
    upper = 1.0
    upper_target, upper_kl = _tilted_distribution(
        reference_log_probabilities,
        normalized_energies,
        legal_mask,
        upper,
    )
    for _ in range(_BRACKET_STEPS):
        if upper_kl >= budget:
            break
        lower = upper
        upper *= 2.0
        upper_target, upper_kl = _tilted_distribution(
            reference_log_probabilities,
            normalized_energies,
            legal_mask,
            upper,
        )
    else:  # Defensive numerical fallback to the analytically valid limit.
        return limiting_target, math.inf, limiting_kl, True

    lower_target, lower_kl = _tilted_distribution(
        reference_log_probabilities,
        normalized_energies,
        legal_mask,
        lower,
    )
    for _ in range(_BISECTION_STEPS):
        midpoint = (lower + upper) / 2.0
        midpoint_target, midpoint_kl = _tilted_distribution(
            reference_log_probabilities,
            normalized_energies,
            legal_mask,
            midpoint,
        )
        if midpoint_kl <= budget:
            lower = midpoint
            lower_target = midpoint_target
            lower_kl = midpoint_kl
        else:
            upper = midpoint
            upper_target = midpoint_target
            upper_kl = midpoint_kl
    del upper_target, upper_kl
    return lower_target, lower, lower_kl, False


def build_spad_energy_posterior(
    reference_action_log_scores: Tensor | Sequence[float],
    terminal_energies: Tensor | Sequence[float | None],
    legal_mask: Tensor | Sequence[bool],
    *,
    action_triplets: Sequence[Sequence[int]],
    no_op_triplet: Sequence[int],
    kl_budget_nats: float = SPAD_E_KL_BUDGET_NATS,
    state_metadata_by_action: Sequence[Mapping[str, Any]] | None = None,
    active_positions: Sequence[int] | None = None,
    differing_positions_by_action: Sequence[Sequence[int]] | None = None,
) -> SPADEnergyPosterior:
    """Construct the strongest legal energy tilt inside the SPAD-E KL budget.

    Optional common-state arguments are all-or-none.  Supplying them makes the
    builder validate that every draw came from the same suffix-visible state
    and changed only the declared XYZ transaction.
    """

    actions, _, duplicate_count = _normalize_action_triplets(
        action_triplets,
        no_op_triplet=no_op_triplet,
    )
    metadata_arguments = (
        state_metadata_by_action,
        active_positions,
        differing_positions_by_action,
    )
    if any(argument is not None for argument in metadata_arguments):
        if not all(argument is not None for argument in metadata_arguments):
            raise ValueError(
                "common-state validation arguments must be supplied together"
            )
        validation = validate_common_state_group(
            action_triplets=actions,
            no_op_triplet=no_op_triplet,
            state_metadata_by_action=state_metadata_by_action,
            active_positions=active_positions,
            differing_positions_by_action=differing_positions_by_action,
        )
        duplicate_count = validation.duplicate_draw_count

    raw_reference_scores = _as_log_score_vector(
        reference_action_log_scores,
        name="reference_action_log_scores",
    )
    device = raw_reference_scores.device
    legal = _as_legal_mask(legal_mask, device=device)
    reference_scores = raw_reference_scores.detach().to(torch.float64)
    masked_reference_scores = reference_scores.masked_fill(~legal, -torch.inf)
    reference_log_probabilities = torch.log_softmax(
        masked_reference_scores,
        dim=0,
    )
    reference_probabilities = torch.exp(reference_log_probabilities)

    normalization = robust_normalize_terminal_energies(
        terminal_energies,
        legal,
        device=device,
    )
    target, tilt, kl_nats, infinite_limit = _solve_strongest_tilt(
        reference_log_probabilities,
        normalization.normalized_energies,
        legal,
        kl_budget_nats=kl_budget_nats,
    )
    known_count = int(normalization.observed_mask.sum().item())
    return SPADEnergyPosterior(
        target_probabilities=target.detach(),
        reference_probabilities=reference_probabilities.detach(),
        reference_log_probabilities=reference_log_probabilities.detach(),
        normalized_terminal_energies=normalization.normalized_energies.detach(),
        legal_mask=legal.detach(),
        terminal_energy_known_mask=normalization.observed_mask.detach(),
        energy_preference_mask=normalization.preference_mask.detach(),
        tilt=tilt,
        kl_nats=kl_nats,
        kl_budget_nats=float(kl_budget_nats),
        action_count=SPAD_E_ACTION_COUNT,
        legal_action_count=int(legal.sum().item()),
        known_energy_count=known_count,
        unknown_energy_count=SPAD_E_ACTION_COUNT - known_count,
        legal_known_energy_count=int(normalization.preference_mask.sum().item()),
        duplicate_draw_count=duplicate_count,
        infinite_tilt_limit=infinite_limit,
    )


def spad_energy_posterior_loss(
    policy_action_log_scores: Tensor | Sequence[float],
    posterior: SPADEnergyPosterior,
) -> SPADEnergyPosteriorLoss:
    """Return differentiable ``KL(q || p_policy)`` and cross entropy.

    ``policy_action_log_scores`` are unnormalized scores for the same four
    action draws.  Illegal entries are masked before normalization and receive
    exactly zero gradient.
    """

    scores = _as_log_score_vector(
        policy_action_log_scores,
        name="policy_action_log_scores",
    )
    working_dtype = torch.float64 if scores.dtype == torch.float64 else torch.float32
    scores_for_loss = scores.to(dtype=working_dtype)
    legal = posterior.legal_mask.to(device=scores.device)
    if legal.ndim != 1 or legal.numel() != SPAD_E_ACTION_COUNT:
        raise ValueError("posterior legal mask must contain exactly 4 entries")
    policy_log_probabilities = torch.log_softmax(
        scores_for_loss.masked_fill(~legal, -torch.inf),
        dim=0,
    )
    target = posterior.target_probabilities.to(
        device=scores.device,
        dtype=working_dtype,
    ).detach()
    if bool((target[~legal] != 0).any().item()):
        raise ValueError("posterior assigns support to an illegal action")
    if not bool(torch.isclose(target.sum(), target.new_tensor(1.0)).item()):
        raise ValueError("posterior target probabilities must sum to one")
    support = target > 0
    cross_entropy = -torch.sum(target[support] * policy_log_probabilities[support])
    target_entropy = -torch.sum(target[support] * torch.log(target[support]))
    kl = torch.sum(
        target[support]
        * (torch.log(target[support]) - policy_log_probabilities[support])
    )
    return SPADEnergyPosteriorLoss(
        loss=kl,
        kl=kl,
        cross_entropy=cross_entropy,
        target_entropy=target_entropy,
        policy_log_probabilities=policy_log_probabilities,
    )


__all__ = [
    "CommonStateValidation",
    "RobustEnergyNormalization",
    "SPAD_E_ACTION_COUNT",
    "SPAD_E_KL_BUDGET_NATS",
    "SPADEnergyPosterior",
    "SPADEnergyPosteriorLoss",
    "build_spad_energy_posterior",
    "robust_normalize_terminal_energies",
    "spad_energy_posterior_loss",
    "validate_common_state_group",
]
