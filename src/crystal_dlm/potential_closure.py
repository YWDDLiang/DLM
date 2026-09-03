"""Pure finite-action posterior utilities for potential-closed crystal DLMs.

The caller enumerates one complete lattice (six-token) or coordinate
(three-token) transaction and supplies one *joint* log-score per action.  This
module validates that all actions came from the same composition-conditioned
state, constructs a raw-energy posterior inside a small KL trust region, and
computes a transaction-length-normalized policy loss.

This module intentionally owns no tokenizer, model, sampler, data loader, or
trainer.  In particular, it does not inherit SPAD-E's rigid K=4 draw semantics:
groups contain two to four unique complete actions, never duplicate draws.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
import operator
from typing import Any

import torch
from torch import Tensor


MIN_ACTION_COUNT = 2
MAX_ACTION_COUNT = 4
VALID_TRANSACTION_LENGTHS = frozenset((3, 6))
MAX_KL_BUDGET_NATS = 0.05
MIN_RAW_ENERGY_SPREAD_EV_PER_ATOM = 1.0e-3

_MAD_NORMAL_CONSISTENCY = 1.4826
_NORMALIZED_ENERGY_CLIP = 8.0
_BISECTION_STEPS = 96
_BRACKET_STEPS = 64


@dataclass(frozen=True)
class TransactionGroupValidation:
    """Normalized contract for one common-state transaction group."""

    action_count: int
    transaction_length: int
    actions: tuple[tuple[int, ...], ...]
    active_positions: tuple[int, ...]
    differing_positions: tuple[tuple[int, ...], ...]
    legal_mask: tuple[bool, ...]


@dataclass(frozen=True)
class RawEnergyNormalization:
    """Robust within-composition normalization of raw energies per atom."""

    normalized_energies: Tensor
    known_mask: Tensor
    preference_mask: Tensor
    center_ev_per_atom: float | None
    scale_ev_per_atom: float | None
    spread_ev_per_atom: float | None
    informative: bool


@dataclass(frozen=True)
class PotentialClosurePosterior:
    """Reference and raw-energy-shaped probabilities for one transaction."""

    target_probabilities: Tensor
    reference_probabilities: Tensor
    reference_log_probabilities: Tensor
    normalized_raw_energies: Tensor
    legal_mask: Tensor
    raw_energy_known_mask: Tensor
    energy_preference_mask: Tensor
    informative: bool
    tilt: float
    kl_nats: float
    kl_budget_nats: float
    action_count: int
    transaction_length: int
    legal_action_count: int
    legal_known_energy_count: int
    energy_center_ev_per_atom: float | None
    energy_scale_ev_per_atom: float | None
    energy_spread_ev_per_atom: float | None
    infinite_tilt_limit: bool


@dataclass(frozen=True)
class PotentialClosureLoss:
    """Transaction-normalized loss plus unnormalized group diagnostics."""

    loss: Tensor
    kl: Tensor
    cross_entropy: Tensor
    target_entropy: Tensor
    group_kl: Tensor
    group_cross_entropy: Tensor
    group_target_entropy: Tensor
    policy_log_probabilities: Tensor


def _sequence_length(value: Any, *, name: str) -> int:
    if isinstance(value, Tensor):
        if value.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional")
        return int(value.numel())
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence")
    return len(value)


def _exact_int_vector(
    value: Any,
    *,
    name: str,
    length: int | None = None,
) -> tuple[int, ...]:
    if isinstance(value, Tensor):
        if value.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional")
        items = value.detach().cpu().tolist()
    elif isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence of integers")
    else:
        items = value
    if length is not None and len(items) != length:
        raise ValueError(f"{name} must contain exactly {length} entries")
    result: list[int] = []
    for item in items:
        if isinstance(item, bool):
            raise TypeError(f"{name} entries must be integers, not bool")
        try:
            result.append(operator.index(item))
        except TypeError as exc:
            raise TypeError(f"{name} entries must be integers") from exc
    return tuple(result)


def _normalize_actions(
    action_tokens: Sequence[Sequence[int]],
    *,
    no_op_tokens: Sequence[int],
) -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...], int]:
    if isinstance(action_tokens, (str, bytes)) or not isinstance(
        action_tokens, Sequence
    ):
        raise TypeError("action_tokens must be a sequence")
    action_count = len(action_tokens)
    if not MIN_ACTION_COUNT <= action_count <= MAX_ACTION_COUNT:
        raise ValueError("action_tokens must contain K in [2, 4]")

    transaction_length = _sequence_length(no_op_tokens, name="no_op_tokens")
    if transaction_length not in VALID_TRANSACTION_LENGTHS:
        raise ValueError("transaction length must be exactly 3 or 6")
    no_op = _exact_int_vector(
        no_op_tokens,
        name="no_op_tokens",
        length=transaction_length,
    )
    actions = tuple(
        _exact_int_vector(
            action,
            name=f"action_tokens[{index}]",
            length=transaction_length,
        )
        for index, action in enumerate(action_tokens)
    )
    if actions[0] != no_op:
        raise ValueError("action 0 must be the no-op transaction")
    if len(set(actions)) != action_count:
        raise ValueError("complete transaction actions must be unique")
    return actions, no_op, transaction_length


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
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if type(left) is not type(right) or len(left) != len(right):
            return False
        return all(_metadata_equal(a, b) for a, b in zip(left, right))
    if type(left) is not type(right):
        return False
    return bool(left == right)


def _normalize_bool_mask(
    value: Tensor | Sequence[bool],
    *,
    name: str,
    length: int,
    device: torch.device,
) -> Tensor:
    if isinstance(value, Tensor):
        if value.dtype is not torch.bool:
            raise TypeError(f"{name} must be a bool tensor")
        result = value.to(device=device)
    else:
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise TypeError(f"{name} must be a sequence of booleans")
        if not all(type(item) is bool for item in value):
            raise TypeError(f"{name} must contain booleans")
        result = torch.tensor(value, dtype=torch.bool, device=device)
    if result.ndim != 1 or result.numel() != length:
        raise ValueError(f"{name} must contain exactly {length} entries")
    return result


def validate_potential_closure_group(
    *,
    action_tokens: Sequence[Sequence[int]],
    no_op_tokens: Sequence[int],
    legal_mask: Tensor | Sequence[bool],
    state_metadata_by_action: Sequence[Mapping[str, Any]],
    active_positions: Sequence[int],
    differing_positions_by_action: Sequence[Sequence[int]],
) -> TransactionGroupValidation:
    """Validate unique complete actions from one composition-conditioned state.

    The common metadata is caller-defined and must include the composition
    identity used by the data pipeline.  Exact metadata equality therefore
    prevents cross-composition or cross-state energy comparisons.  Reported
    token differences must exactly match changes inside the declared active
    three- or six-token block.
    """

    actions, no_op, transaction_length = _normalize_actions(
        action_tokens,
        no_op_tokens=no_op_tokens,
    )
    action_count = len(actions)
    legal = _normalize_bool_mask(
        legal_mask,
        name="legal_mask",
        length=action_count,
        device=torch.device("cpu"),
    )
    if not bool(legal[0].item()):
        raise ValueError("action 0 (the no-op transaction) must be legal")

    positions = _exact_int_vector(
        active_positions,
        name="active_positions",
        length=transaction_length,
    )
    if any(position < 0 for position in positions):
        raise ValueError("active_positions must be non-negative")
    if len(set(positions)) != transaction_length:
        raise ValueError("active_positions must be unique")

    if isinstance(state_metadata_by_action, (str, bytes)) or not isinstance(
        state_metadata_by_action, Sequence
    ):
        raise TypeError("state_metadata_by_action must be a sequence")
    if len(state_metadata_by_action) != action_count:
        raise ValueError(
            "state_metadata_by_action must contain one mapping per action"
        )
    if not all(isinstance(item, Mapping) for item in state_metadata_by_action):
        raise TypeError("every state metadata entry must be a mapping")
    common_metadata = state_metadata_by_action[0]
    for index, metadata in enumerate(state_metadata_by_action[1:], start=1):
        if not _metadata_equal(common_metadata, metadata):
            raise ValueError(
                f"action {index} does not share identical state metadata"
            )

    if isinstance(differing_positions_by_action, (str, bytes)) or not isinstance(
        differing_positions_by_action, Sequence
    ):
        raise TypeError("differing_positions_by_action must be a sequence")
    if len(differing_positions_by_action) != action_count:
        raise ValueError(
            "differing_positions_by_action must contain one entry per action"
        )

    normalized_differences: list[tuple[int, ...]] = []
    for index, (action, reported) in enumerate(
        zip(actions, differing_positions_by_action)
    ):
        reported_positions = _exact_int_vector(
            reported,
            name=f"differing_positions_by_action[{index}]",
        )
        if len(set(reported_positions)) != len(reported_positions):
            raise ValueError(f"action {index} reports duplicate differing positions")
        expected = tuple(
            position
            for position, candidate, source in zip(positions, action, no_op)
            if candidate != source
        )
        if set(reported_positions) != set(expected):
            raise ValueError(
                f"action {index} differences must match changes in the active block"
            )
        normalized_differences.append(expected)

    return TransactionGroupValidation(
        action_count=action_count,
        transaction_length=transaction_length,
        actions=actions,
        active_positions=positions,
        differing_positions=tuple(normalized_differences),
        legal_mask=tuple(bool(item) for item in legal.tolist()),
    )


def _as_log_score_vector(
    value: Tensor | Sequence[float],
    *,
    name: str,
    action_count: int | None = None,
) -> Tensor:
    if isinstance(value, Tensor):
        if not value.is_floating_point() or value.is_complex():
            raise TypeError(f"{name} must be a real floating-point tensor")
        result = value
    else:
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise TypeError(f"{name} must be a sequence")
        result = torch.as_tensor(value, dtype=torch.float64)
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    count = int(result.numel())
    if not MIN_ACTION_COUNT <= count <= MAX_ACTION_COUNT:
        raise ValueError(f"{name} must contain K in [2, 4]")
    if action_count is not None and count != action_count:
        raise ValueError(f"{name} must contain exactly {action_count} entries")
    if not bool(torch.isfinite(result).all().item()):
        raise ValueError(f"{name} must be finite")
    return result


def _as_raw_energy_vector(
    value: Tensor | Sequence[float | None],
    *,
    action_count: int,
    device: torch.device,
) -> Tensor:
    if isinstance(value, Tensor):
        if value.is_complex() or value.dtype is torch.bool:
            raise TypeError("raw_energies_per_atom must be real-valued")
        result = value.detach().to(device=device, dtype=torch.float64)
    else:
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise TypeError("raw_energies_per_atom must be a sequence")
        converted: list[float] = []
        for item in value:
            if item is None:
                converted.append(math.nan)
            elif isinstance(item, bool):
                raise TypeError("raw energies must be real values, not bool")
            else:
                converted.append(float(item))
        result = torch.tensor(converted, dtype=torch.float64, device=device)
    if result.ndim != 1 or result.numel() != action_count:
        raise ValueError(
            f"raw_energies_per_atom must contain exactly {action_count} entries"
        )
    return result


def _median(values: Tensor) -> Tensor:
    ordered = torch.sort(values).values
    midpoint = ordered.numel() // 2
    if ordered.numel() % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2.0


def robust_normalize_raw_energies(
    raw_energies_per_atom: Tensor | Sequence[float | None],
    legal_mask: Tensor | Sequence[bool],
    *,
    device: torch.device | None = None,
) -> RawEnergyNormalization:
    """Normalize finite legal raw energies with within-group median and MAD.

    A group is informative only when at least two legal actions have finite
    raw energies and their spread is at least 1 meV/atom.  Non-finite labels
    are unknown, receive normalized value zero, and never define a preference.
    """

    action_count = _sequence_length(
        raw_energies_per_atom,
        name="raw_energies_per_atom",
    )
    if not MIN_ACTION_COUNT <= action_count <= MAX_ACTION_COUNT:
        raise ValueError("raw_energies_per_atom must contain K in [2, 4]")
    target_device = device or (
        raw_energies_per_atom.device
        if isinstance(raw_energies_per_atom, Tensor)
        else torch.device("cpu")
    )
    energies = _as_raw_energy_vector(
        raw_energies_per_atom,
        action_count=action_count,
        device=target_device,
    )
    legal = _normalize_bool_mask(
        legal_mask,
        name="legal_mask",
        length=action_count,
        device=target_device,
    )
    known = torch.isfinite(energies)
    preference = legal & known
    values = energies[preference]
    normalized = torch.zeros_like(energies)

    if values.numel() == 0:
        return RawEnergyNormalization(
            normalized_energies=normalized,
            known_mask=known,
            preference_mask=preference,
            center_ev_per_atom=None,
            scale_ev_per_atom=None,
            spread_ev_per_atom=None,
            informative=False,
        )

    center_tensor = _median(values)
    center = float(center_tensor.item())
    spread_tensor = torch.max(values) - torch.min(values)
    spread = float(spread_tensor.item())
    informative = (
        values.numel() >= 2
        and spread >= MIN_RAW_ENERGY_SPREAD_EV_PER_ATOM
    )
    if not informative:
        return RawEnergyNormalization(
            normalized_energies=normalized,
            known_mask=known,
            preference_mask=preference,
            center_ev_per_atom=center,
            scale_ev_per_atom=None,
            spread_ev_per_atom=spread,
            informative=False,
        )

    deviations = torch.abs(values - center_tensor)
    mad = _median(deviations)
    scale_tensor = mad * _MAD_NORMAL_CONSISTENCY
    epsilon = torch.finfo(torch.float64).eps
    if float(scale_tensor.item()) <= epsilon:
        positive_deviations = deviations[deviations > epsilon]
        if positive_deviations.numel() == 0:
            return RawEnergyNormalization(
                normalized_energies=normalized,
                known_mask=known,
                preference_mask=preference,
                center_ev_per_atom=center,
                scale_ev_per_atom=None,
                spread_ev_per_atom=spread,
                informative=False,
            )
        scale_tensor = _median(positive_deviations)

    normalized_values = (values - center_tensor) / scale_tensor
    normalized[preference] = torch.clamp(
        normalized_values,
        min=-_NORMALIZED_ENERGY_CLIP,
        max=_NORMALIZED_ENERGY_CLIP,
    )
    return RawEnergyNormalization(
        normalized_energies=normalized,
        known_mask=known,
        preference_mask=preference,
        center_ev_per_atom=center,
        scale_ev_per_atom=float(scale_tensor.item()),
        spread_ev_per_atom=spread,
        informative=True,
    )


def _validate_kl_budget(kl_budget_nats: float) -> float:
    budget = float(kl_budget_nats)
    if not math.isfinite(budget) or budget < 0.0:
        raise ValueError("kl_budget_nats must be finite and non-negative")
    if budget > MAX_KL_BUDGET_NATS:
        raise ValueError("kl_budget_nats cannot exceed 0.05 nat")
    return budget


def _tilted_known_target(
    reference_probabilities: Tensor,
    normalized_energies: Tensor,
    preference_mask: Tensor,
    tilt: float,
) -> tuple[Tensor, float]:
    known_reference = reference_probabilities[preference_mask]
    known_mass = torch.sum(known_reference)
    conditional_reference = known_reference / known_mass
    conditional_logits = (
        torch.log(conditional_reference)
        - float(tilt) * normalized_energies[preference_mask]
    )
    conditional_log_target = torch.log_softmax(conditional_logits, dim=0)
    conditional_target = torch.exp(conditional_log_target)

    target = reference_probabilities.clone()
    target[preference_mask] = known_mass * conditional_target
    kl = known_mass * torch.sum(
        conditional_target
        * (conditional_log_target - torch.log(conditional_reference))
    )
    return target, float(kl.item())


def _solve_strongest_known_tilt(
    reference_probabilities: Tensor,
    normalized_energies: Tensor,
    preference_mask: Tensor,
    *,
    kl_budget_nats: float,
) -> tuple[Tensor, float, float, bool]:
    budget = _validate_kl_budget(kl_budget_nats)
    reference = reference_probabilities
    known_reference = reference[preference_mask]
    known_mass = torch.sum(known_reference)
    if budget == 0.0 or float(known_mass.item()) <= 0.0:
        return reference.clone(), 0.0, 0.0, False

    conditional_reference = known_reference / known_mass
    known_energies = normalized_energies[preference_mask]
    if bool(torch.all(known_energies == known_energies[0]).item()):
        return reference.clone(), 0.0, 0.0, False

    minimum_energy = torch.min(known_energies)
    minimum_mask = known_energies == minimum_energy
    limiting_conditional_mass = torch.sum(conditional_reference[minimum_mask])
    limiting_conditional = torch.zeros_like(conditional_reference)
    limiting_conditional[minimum_mask] = (
        conditional_reference[minimum_mask] / limiting_conditional_mass
    )
    limiting_target = reference.clone()
    limiting_target[preference_mask] = known_mass * limiting_conditional
    limiting_kl = float(
        (known_mass * -torch.log(limiting_conditional_mass)).item()
    )
    if limiting_kl <= budget:
        return limiting_target, math.inf, limiting_kl, True

    lower = 0.0
    upper = 1.0
    upper_target, upper_kl = _tilted_known_target(
        reference,
        normalized_energies,
        preference_mask,
        upper,
    )
    for _ in range(_BRACKET_STEPS):
        if upper_kl >= budget:
            break
        lower = upper
        upper *= 2.0
        upper_target, upper_kl = _tilted_known_target(
            reference,
            normalized_energies,
            preference_mask,
            upper,
        )
    else:
        return limiting_target, math.inf, limiting_kl, True

    lower_target, lower_kl = _tilted_known_target(
        reference,
        normalized_energies,
        preference_mask,
        lower,
    )
    for _ in range(_BISECTION_STEPS):
        midpoint = (lower + upper) / 2.0
        midpoint_target, midpoint_kl = _tilted_known_target(
            reference,
            normalized_energies,
            preference_mask,
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


def build_potential_closure_posterior(
    reference_action_log_scores: Tensor | Sequence[float],
    raw_energies_per_atom: Tensor | Sequence[float | None],
    legal_mask: Tensor | Sequence[bool],
    *,
    action_tokens: Sequence[Sequence[int]],
    no_op_tokens: Sequence[int],
    state_metadata_by_action: Sequence[Mapping[str, Any]],
    active_positions: Sequence[int],
    differing_positions_by_action: Sequence[Sequence[int]],
    kl_budget_nats: float = MAX_KL_BUDGET_NATS,
) -> PotentialClosurePosterior:
    """Build the strongest valid raw-energy tilt within the KL budget.

    Scores are complete joint transaction log-scores supplied by the caller;
    this function never reconstructs them from token averages.  Illegal
    actions receive zero support.  Legal unknown-energy actions keep exactly
    their reference probability, while energy only redistributes probability
    among legal known-energy actions from the same state and composition.
    """

    budget = _validate_kl_budget(kl_budget_nats)
    validation = validate_potential_closure_group(
        action_tokens=action_tokens,
        no_op_tokens=no_op_tokens,
        legal_mask=legal_mask,
        state_metadata_by_action=state_metadata_by_action,
        active_positions=active_positions,
        differing_positions_by_action=differing_positions_by_action,
    )
    scores = _as_log_score_vector(
        reference_action_log_scores,
        name="reference_action_log_scores",
        action_count=validation.action_count,
    )
    device = scores.device
    legal = _normalize_bool_mask(
        legal_mask,
        name="legal_mask",
        length=validation.action_count,
        device=device,
    )
    detached_scores = scores.detach().to(dtype=torch.float64)
    reference_log_probabilities = torch.log_softmax(
        detached_scores.masked_fill(~legal, -torch.inf),
        dim=0,
    )
    reference_probabilities = torch.exp(reference_log_probabilities)

    normalization = robust_normalize_raw_energies(
        raw_energies_per_atom,
        legal,
        device=device,
    )
    if normalization.informative:
        target, tilt, kl_nats, infinite_limit = _solve_strongest_known_tilt(
            reference_probabilities,
            normalization.normalized_energies,
            normalization.preference_mask,
            kl_budget_nats=budget,
        )
    else:
        target = reference_probabilities.clone()
        tilt = 0.0
        kl_nats = 0.0
        infinite_limit = False

    return PotentialClosurePosterior(
        target_probabilities=target.detach(),
        reference_probabilities=reference_probabilities.detach(),
        reference_log_probabilities=reference_log_probabilities.detach(),
        normalized_raw_energies=normalization.normalized_energies.detach(),
        legal_mask=legal.detach(),
        raw_energy_known_mask=normalization.known_mask.detach(),
        energy_preference_mask=normalization.preference_mask.detach(),
        informative=normalization.informative,
        tilt=tilt,
        kl_nats=kl_nats,
        kl_budget_nats=budget,
        action_count=validation.action_count,
        transaction_length=validation.transaction_length,
        legal_action_count=int(legal.sum().item()),
        legal_known_energy_count=int(
            normalization.preference_mask.sum().item()
        ),
        energy_center_ev_per_atom=normalization.center_ev_per_atom,
        energy_scale_ev_per_atom=normalization.scale_ev_per_atom,
        energy_spread_ev_per_atom=normalization.spread_ev_per_atom,
        infinite_tilt_limit=infinite_limit,
    )


def potential_closure_loss(
    policy_action_log_scores: Tensor | Sequence[float],
    posterior: PotentialClosurePosterior,
) -> PotentialClosureLoss:
    """Return KL/CE divided by the complete transaction length.

    The supplied policy values must again be complete joint action log-scores.
    Normalization is over legal actions only, so every illegal score has
    exactly zero gradient.  KL and cross-entropy have identical policy
    gradients; ``loss`` is the transaction-normalized KL.
    """

    if posterior.transaction_length not in VALID_TRANSACTION_LENGTHS:
        raise ValueError("posterior transaction length must be exactly 3 or 6")
    scores = _as_log_score_vector(
        policy_action_log_scores,
        name="policy_action_log_scores",
        action_count=posterior.action_count,
    )
    working_dtype = torch.float64 if scores.dtype == torch.float64 else torch.float32
    working_scores = scores.to(dtype=working_dtype)
    legal = posterior.legal_mask.to(device=scores.device)
    if legal.ndim != 1 or legal.numel() != posterior.action_count:
        raise ValueError("posterior legal mask has the wrong shape")

    legal_scores = working_scores[legal]
    legal_log_probabilities = torch.log_softmax(legal_scores, dim=0)
    policy_log_probabilities = torch.full_like(working_scores, -torch.inf)
    policy_log_probabilities[legal] = legal_log_probabilities

    target = posterior.target_probabilities.to(
        device=scores.device,
        dtype=working_dtype,
    ).detach()
    if target.ndim != 1 or target.numel() != posterior.action_count:
        raise ValueError("posterior target probabilities have the wrong shape")
    if bool((target[~legal] != 0).any().item()):
        raise ValueError("posterior assigns support to an illegal action")
    if not bool(
        torch.isclose(
            target.sum(),
            target.new_tensor(1.0),
            rtol=1.0e-7,
            atol=1.0e-9,
        ).item()
    ):
        raise ValueError("posterior target probabilities must sum to one")

    legal_target = target[legal]
    support = legal_target > 0
    supported_target = legal_target[support]
    supported_policy_log = legal_log_probabilities[support]
    group_cross_entropy = -torch.sum(supported_target * supported_policy_log)
    group_target_entropy = -torch.sum(
        supported_target * torch.log(supported_target)
    )
    group_kl = torch.sum(
        supported_target
        * (torch.log(supported_target) - supported_policy_log)
    )
    divisor = group_kl.new_tensor(float(posterior.transaction_length))
    normalized_kl = group_kl / divisor
    normalized_cross_entropy = group_cross_entropy / divisor
    normalized_target_entropy = group_target_entropy / divisor

    return PotentialClosureLoss(
        loss=normalized_kl,
        kl=normalized_kl,
        cross_entropy=normalized_cross_entropy,
        target_entropy=normalized_target_entropy,
        group_kl=group_kl,
        group_cross_entropy=group_cross_entropy,
        group_target_entropy=group_target_entropy,
        policy_log_probabilities=policy_log_probabilities,
    )


__all__ = [
    "MAX_ACTION_COUNT",
    "MAX_KL_BUDGET_NATS",
    "MIN_ACTION_COUNT",
    "MIN_RAW_ENERGY_SPREAD_EV_PER_ATOM",
    "PotentialClosureLoss",
    "PotentialClosurePosterior",
    "RawEnergyNormalization",
    "TransactionGroupValidation",
    "VALID_TRANSACTION_LENGTHS",
    "build_potential_closure_posterior",
    "potential_closure_loss",
    "robust_normalize_raw_energies",
    "validate_potential_closure_group",
]
