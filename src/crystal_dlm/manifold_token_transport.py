"""Sparse continuous-to-token logit transport for crystal numeric families."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from crystal_dlm.manifold_geometry import wrap_fractional, wrapped_fractional_delta


def _as_values(
    values: Tensor | Sequence[float], *, reference: Tensor, name: str
) -> Tensor:
    result = torch.as_tensor(values, dtype=reference.dtype, device=reference.device)
    if result.ndim != 1 or result.numel() < 2:
        raise ValueError(f"{name} must be a one-dimensional sequence with at least two values")
    if not bool(torch.isfinite(result).all().item()):
        raise ValueError(f"{name} contains non-finite values")
    if not bool((result[1:] > result[:-1]).all().item()):
        raise ValueError(f"{name} must be strictly increasing")
    return result


def _as_token_ids(
    token_ids: Tensor | Sequence[int],
    *,
    device: torch.device,
    expected: int,
    vocab_size: int,
) -> Tensor:
    result = torch.as_tensor(token_ids, dtype=torch.long, device=device)
    if result.ndim != 1 or result.numel() != expected:
        raise ValueError("legal_token_ids must align one-to-one with legal_values")
    if bool(((result < 0) | (result >= int(vocab_size))).any().item()):
        raise ValueError("legal_token_ids contain an out-of-range vocabulary id")
    if int(torch.unique(result).numel()) != expected:
        raise ValueError("legal_token_ids must be unique")
    return result


def _broadcast_inputs(
    old_value: Tensor,
    predicted_target: Tensor,
    gain: Tensor | float,
    active_mask: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if not isinstance(old_value, Tensor) or not isinstance(predicted_target, Tensor):
        raise TypeError("old_value and predicted_target must be torch tensors")
    if not old_value.is_floating_point() or not predicted_target.is_floating_point():
        raise TypeError("old_value and predicted_target must be floating point")
    if old_value.device != predicted_target.device or old_value.dtype != predicted_target.dtype:
        raise ValueError("old_value and predicted_target must share device and dtype")
    old, target = torch.broadcast_tensors(old_value, predicted_target)
    gain_tensor = torch.as_tensor(gain, dtype=target.dtype, device=target.device)
    gain_tensor = torch.broadcast_to(gain_tensor, target.shape)
    if active_mask is None:
        active = torch.ones(target.shape, dtype=torch.bool, device=target.device)
    else:
        active = torch.as_tensor(active_mask, dtype=torch.bool, device=target.device)
        active = torch.broadcast_to(active, target.shape)
    if not bool(torch.isfinite(old).all().item()):
        raise ValueError("old_value contains non-finite values")
    if not bool(torch.isfinite(target).all().item()):
        raise ValueError("predicted_target contains non-finite values")
    if not bool(torch.isfinite(gain_tensor).all().item()):
        raise ValueError("gain contains non-finite values")
    return old, target, gain_tensor, active


def _scatter_two(
    lower_ids: Tensor,
    upper_ids: Tensor,
    lower_weights: Tensor,
    upper_weights: Tensor,
    *,
    vocab_size: int,
) -> Tensor:
    output = lower_weights.new_zeros((*lower_weights.shape, int(vocab_size)))
    output = output.scatter_add(-1, lower_ids.unsqueeze(-1), lower_weights.unsqueeze(-1))
    output = output.scatter_add(-1, upper_ids.unsqueeze(-1), upper_weights.unsqueeze(-1))
    return output


def render_bracketed_token_residual(
    legal_values: Tensor | Sequence[float],
    legal_token_ids: Tensor | Sequence[int],
    old_value: Tensor,
    predicted_target: Tensor,
    *,
    vocab_size: int,
    gain: Tensor | float = 1.0,
    active_mask: Tensor | None = None,
) -> Tensor:
    """Render transport on the two legal bins bracketing a predicted target.

    The residual is sparse in vocabulary space.  Its magnitude is the absolute
    continuous correction times ``gain`` and its split between the two target
    bins is linearly differentiable.  Consequently either a zero correction or
    a zero gain produces an exactly-zero tensor.  Values outside the registered
    family range are projected to the nearest endpoint without ever touching a
    non-family token.
    """

    if int(vocab_size) <= 0:
        raise ValueError("vocab_size must be positive")
    old, target, gain_tensor, active = _broadcast_inputs(
        old_value, predicted_target, gain, active_mask
    )
    values = _as_values(legal_values, reference=target, name="legal_values")
    token_ids = _as_token_ids(
        legal_token_ids,
        device=target.device,
        expected=int(values.numel()),
        vocab_size=int(vocab_size),
    )

    projected = target.clamp(min=values[0], max=values[-1])
    upper_index = torch.searchsorted(values, projected.contiguous(), right=True)
    upper_index = upper_index.clamp(min=1, max=int(values.numel()) - 1)
    lower_index = upper_index - 1
    lower_value = values[lower_index]
    upper_value = values[upper_index]
    fraction = (projected - lower_value) / (upper_value - lower_value)
    magnitude = gain_tensor * torch.abs(target - old) * active.to(target.dtype)
    lower_weights = magnitude * (1.0 - fraction)
    upper_weights = magnitude * fraction
    return _scatter_two(
        token_ids[lower_index],
        token_ids[upper_index],
        lower_weights,
        upper_weights,
        vocab_size=int(vocab_size),
    )


def _canonical_periodic_support(
    values: Tensor,
    token_ids: Tensor,
    *,
    period: float,
    alias_tolerance: float,
) -> tuple[Tensor, Tensor]:
    canonical = torch.remainder(values, float(period))
    canonical = torch.where(
        torch.isclose(
            canonical,
            canonical.new_tensor(float(period)),
            atol=float(alias_tolerance),
            rtol=0.0,
        ),
        torch.zeros_like(canonical),
        canonical,
    )
    order = torch.argsort(canonical, stable=True)
    canonical = canonical[order]
    ordered_ids = token_ids[order]

    keep = torch.ones_like(canonical, dtype=torch.bool)
    if canonical.numel() > 1:
        duplicates = torch.isclose(
            canonical[1:], canonical[:-1], atol=float(alias_tolerance), rtol=0.0
        )
        keep[1:] = ~duplicates
    canonical = canonical[keep]
    ordered_ids = ordered_ids[keep]
    if canonical.numel() < 2:
        raise ValueError("periodic support must contain at least two physical bins")
    if not bool(
        torch.isclose(
            canonical[0],
            canonical.new_zeros(()),
            atol=float(alias_tolerance),
            rtol=0.0,
        ).item()
    ):
        raise ValueError("periodic support must contain a zero/period alias bin")
    return canonical, ordered_ids


def render_periodic_coordinate_token_residual(
    legal_values: Tensor | Sequence[float],
    legal_token_ids: Tensor | Sequence[int],
    old_value: Tensor,
    predicted_target: Tensor,
    *,
    vocab_size: int,
    gain: Tensor | float = 1.0,
    active_mask: Tensor | None = None,
    period: float = 1.0,
    alias_tolerance: float = 1.0e-7,
) -> Tensor:
    """Render a periodic coordinate correction without a 0/1 discontinuity.

    A physical zero/period alias is represented by one canonical token (the
    first supplied alias, normally the ``0`` token).  The final circular
    interval connects the largest non-alias bin back to that token.  Residuals
    therefore approach the same vocabulary coordinate continuously from both
    sides of the wrap boundary.
    """

    if int(vocab_size) <= 0:
        raise ValueError("vocab_size must be positive")
    if float(period) <= 0.0:
        raise ValueError("period must be positive")
    if float(alias_tolerance) < 0.0:
        raise ValueError("alias_tolerance must be non-negative")
    old, target, gain_tensor, active = _broadcast_inputs(
        old_value, predicted_target, gain, active_mask
    )
    raw_values = _as_values(legal_values, reference=target, name="legal_values")
    raw_ids = _as_token_ids(
        legal_token_ids,
        device=target.device,
        expected=int(raw_values.numel()),
        vocab_size=int(vocab_size),
    )
    if bool((raw_values < -float(alias_tolerance)).any().item()) or bool(
        (raw_values > float(period) + float(alias_tolerance)).any().item()
    ):
        raise ValueError("periodic legal values must lie in [0, period]")
    values, token_ids = _canonical_periodic_support(
        raw_values,
        raw_ids,
        period=float(period),
        alias_tolerance=float(alias_tolerance),
    )

    wrapped_target = wrap_fractional(target, period=float(period))
    upper_index = torch.searchsorted(values, wrapped_target.contiguous(), right=True)
    wraps = upper_index == int(values.numel())
    lower_index = torch.where(
        wraps,
        torch.full_like(upper_index, int(values.numel()) - 1),
        (upper_index - 1).clamp_min(0),
    )
    upper_index = torch.where(wraps, torch.zeros_like(upper_index), upper_index)
    lower_value = values[lower_index]
    upper_value = torch.where(
        wraps,
        values[upper_index] + float(period),
        values[upper_index],
    )
    target_for_interval = torch.where(
        wraps, wrapped_target + 0.0, wrapped_target
    )
    fraction = (target_for_interval - lower_value) / (upper_value - lower_value)
    correction = wrapped_fractional_delta(
        target - old, period=float(period)
    )
    magnitude = gain_tensor * torch.abs(correction) * active.to(target.dtype)
    lower_weights = magnitude * (1.0 - fraction)
    upper_weights = magnitude * fraction
    return _scatter_two(
        token_ids[lower_index],
        token_ids[upper_index],
        lower_weights,
        upper_weights,
        vocab_size=int(vocab_size),
    )


__all__ = [
    "render_bracketed_token_residual",
    "render_periodic_coordinate_token_residual",
]
