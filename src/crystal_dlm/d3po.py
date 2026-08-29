"""Pure shared-noise masked-D3PO primitives for dynamic crystal bodies.

The functions in this module intentionally know nothing about models, PEFT,
datasets, or trainers.  They implement the frozen mathematical core from
``docs/D3PO_256_MIN_CONTRACT_V1.md`` while preserving the dynamic ``7 + 4N``
representation used by the existing crystal DLM.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Sequence

import torch
import torch.nn.functional as F

from crystal_dlm.fixed_slot import MASK_TOKEN_ID
from crystal_dlm.stable_geometry_curriculum import dynamic_geometry_relative_positions


@dataclass(frozen=True)
class SharedGeometryCorruption:
    """A winner/loser corruption coupled by one probability and one mask."""

    winner_noisy_ids: torch.Tensor
    loser_noisy_ids: torch.Tensor
    masked_positions: torch.Tensor
    geometry_mask: torch.Tensor
    p_mask: torch.Tensor


@dataclass(frozen=True)
class D3POLossOutput:
    """Scalar loss components plus unreduced pair diagnostics."""

    loss: torch.Tensor
    preference_loss: torch.Tensor
    winner_anchor_loss: torch.Tensor
    margin: torch.Tensor
    target_probability: torch.Tensor
    per_pair_preference_loss: torch.Tensor


def _batch_vector(
    value: torch.Tensor | Sequence[float] | float | int,
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    name: str,
) -> torch.Tensor:
    result = torch.as_tensor(value, device=device, dtype=dtype)
    if result.ndim == 0:
        result = result.expand(batch_size)
    if result.shape != (batch_size,):
        raise ValueError(f"{name} must be scalar or have shape ({batch_size},)")
    return result


def _probability_vector(
    p_mask: torch.Tensor | Sequence[float] | float,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    probabilities = _batch_vector(
        p_mask,
        batch_size=batch_size,
        device=device,
        dtype=torch.float32,
        name="p_mask",
    )
    if not bool(torch.isfinite(probabilities).all().item()):
        raise ValueError("p_mask must be finite")
    if not bool(((probabilities > 0.0) & (probabilities <= 1.0)).all().item()):
        raise ValueError("p_mask must lie in (0, 1]")
    return probabilities


def shared_geometry_corruption(
    winner_input_ids: torch.Tensor,
    loser_input_ids: torch.Tensor,
    prompt_lengths: torch.Tensor | Sequence[int] | int,
    num_atoms: torch.Tensor | Sequence[int] | int,
    *,
    mask_token_id: int = MASK_TOKEN_ID,
    attention_mask: torch.Tensor | None = None,
    p_mask: torch.Tensor | Sequence[float] | float | None = None,
    shared_mask: torch.Tensor | None = None,
    eps: float = 1e-3,
    generator: torch.Generator | None = None,
) -> SharedGeometryCorruption:
    """Corrupt winner and loser with one geometry-only mask per pair.

    Dynamic body position 0 (``N``) and every site element position remain
    visible.  Lattice positions 1--6 and each site's X/Y/Z positions are the
    only mask candidates.  Unlike the historical SFT corruption helper, this
    function never forces a token to be masked: an empty sampled mask is a
    valid zero-contribution event under the masking-state objective.

    ``shared_mask`` is primarily useful for deterministic ledgers and tests.
    When supplied, ``p_mask`` must also be supplied because the D3PO score
    requires the probability that produced that mask.
    """

    if winner_input_ids.ndim != 2 or loser_input_ids.ndim != 2:
        raise ValueError("winner_input_ids and loser_input_ids must be rank-2")
    if winner_input_ids.shape != loser_input_ids.shape:
        raise ValueError("winner_input_ids and loser_input_ids must have identical shape")
    if winner_input_ids.device != loser_input_ids.device:
        raise ValueError("winner_input_ids and loser_input_ids must share a device")
    if winner_input_ids.dtype != loser_input_ids.dtype:
        raise ValueError("winner_input_ids and loser_input_ids must share a dtype")

    batch_size, sequence_length = winner_input_ids.shape
    device = winner_input_ids.device
    prompts = _batch_vector(
        prompt_lengths,
        batch_size=batch_size,
        device=device,
        dtype=torch.long,
        name="prompt_lengths",
    )
    atoms = _batch_vector(
        num_atoms,
        batch_size=batch_size,
        device=device,
        dtype=torch.long,
        name="num_atoms",
    )
    if attention_mask is None:
        attended = torch.ones_like(winner_input_ids, dtype=torch.bool)
    else:
        attended = torch.as_tensor(attention_mask, device=device, dtype=torch.bool)
        if attended.shape != winner_input_ids.shape:
            raise ValueError("attention_mask must match the input-id shape")

    geometry_mask = torch.zeros_like(winner_input_ids, dtype=torch.bool)
    prompt_values = prompts.detach().cpu().tolist()
    atom_values = atoms.detach().cpu().tolist()
    for row, (prompt_length, atom_count) in enumerate(zip(prompt_values, atom_values)):
        prompt_length = int(prompt_length)
        atom_count = int(atom_count)
        body_length = 7 + 4 * atom_count
        if prompt_length < 0 or prompt_length + body_length > sequence_length:
            raise ValueError(
                f"row {row} cannot contain a 7+4N body at prompt length {prompt_length}"
            )
        if not bool(attended[row, prompt_length : prompt_length + body_length].all().item()):
            raise ValueError(f"row {row} has an unattended token inside its dynamic body")
        relative_positions = dynamic_geometry_relative_positions(atom_count)
        absolute_positions = torch.tensor(
            [prompt_length + position for position in relative_positions],
            dtype=torch.long,
            device=device,
        )
        geometry_mask[row, absolute_positions] = True

    if shared_mask is not None:
        if p_mask is None:
            raise ValueError("p_mask is required when shared_mask is supplied")
        masked_positions = torch.as_tensor(shared_mask, device=device, dtype=torch.bool)
        if masked_positions.shape != winner_input_ids.shape:
            raise ValueError("shared_mask must match the input-id shape")
        if bool((masked_positions & ~geometry_mask).any().item()):
            raise ValueError("shared_mask contains a non-geometry position")
        probabilities = _probability_vector(p_mask, batch_size=batch_size, device=device)
    else:
        if not 0.0 < float(eps) <= 1.0:
            raise ValueError("eps must lie in (0, 1]")
        if p_mask is None:
            random_t = torch.rand(
                (batch_size,),
                dtype=torch.float32,
                device=device,
                generator=generator,
            )
            probabilities = float(eps) + (1.0 - float(eps)) * random_t
        else:
            probabilities = _probability_vector(p_mask, batch_size=batch_size, device=device)
        draws = torch.rand(
            winner_input_ids.shape,
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        masked_positions = geometry_mask & (draws < probabilities.unsqueeze(1))

    mask_value = torch.full_like(winner_input_ids, int(mask_token_id))
    return SharedGeometryCorruption(
        winner_noisy_ids=torch.where(masked_positions, mask_value, winner_input_ids),
        loser_noisy_ids=torch.where(masked_positions, mask_value, loser_input_ids),
        masked_positions=masked_positions,
        geometry_mask=geometry_mask,
        p_mask=probabilities.to(torch.float32),
    )


def legal_target_log_probs(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    legal_token_ids_by_position: Sequence[Sequence[int]],
    *,
    mask_token_id: int = MASK_TOKEN_ID,
) -> torch.Tensor:
    """Return FP32 target log-probabilities over each position's legal support.

    ``logits`` contains only the selected masked positions and has shape
    ``[K, vocab]``.  The support sequence therefore also has length ``K``.
    Illegal vocabulary entries never enter the denominator, and the mask token
    is removed even if a caller accidentally includes it in a support list.
    """

    if logits.ndim != 2:
        raise ValueError("logits must have shape [K, vocab]")
    if target_ids.ndim != 1 or target_ids.shape[0] != logits.shape[0]:
        raise ValueError("target_ids must have shape [K]")
    if len(legal_token_ids_by_position) != logits.shape[0]:
        raise ValueError("one legal-token support is required for each selected position")
    if target_ids.device != logits.device:
        raise ValueError("target_ids and logits must share a device")
    if logits.shape[0] == 0:
        return torch.empty((0,), dtype=torch.float32, device=logits.device)

    vocab_size = logits.shape[1]
    grouped_rows: dict[tuple[int, ...], list[int]] = {}
    for row, token_ids in enumerate(legal_token_ids_by_position):
        support = tuple(
            dict.fromkeys(int(token_id) for token_id in token_ids if int(token_id) != int(mask_token_id))
        )
        if not support:
            raise ValueError(f"position {row} has no legal clean-token support")
        if any(token_id < 0 or token_id >= vocab_size for token_id in support):
            raise ValueError(f"position {row} has an out-of-range legal token id")
        grouped_rows.setdefault(support, []).append(row)

    fp32_logits = logits.to(torch.float32)
    targets = target_ids.to(torch.long)
    result = torch.zeros((logits.shape[0],), dtype=torch.float32, device=logits.device)
    for support, rows in grouped_rows.items():
        row_indices = torch.tensor(rows, dtype=torch.long, device=logits.device)
        support_ids = torch.tensor(support, dtype=torch.long, device=logits.device)
        row_targets = targets.index_select(0, row_indices)
        target_matches = row_targets.unsqueeze(1) == support_ids.unsqueeze(0)
        if not bool(target_matches.any(dim=1).all().item()):
            bad_row = rows[int(torch.nonzero(~target_matches.any(dim=1), as_tuple=False)[0].item())]
            raise ValueError(f"target token at position {bad_row} is outside its legal support")
        legal_logits = fp32_logits.index_select(0, row_indices).index_select(1, support_ids)
        target_offsets = target_matches.to(torch.long).argmax(dim=1, keepdim=True)
        values = F.log_softmax(legal_logits, dim=-1).gather(1, target_offsets).squeeze(1)
        result = result.index_copy(0, row_indices, values)
    return result


def _selected_log_probs(
    values: torch.Tensor,
    masked_positions: torch.Tensor,
    *,
    name: str,
) -> torch.Tensor:
    if values.device != masked_positions.device:
        raise ValueError(f"{name} and masked_positions must share a device")
    if values.shape == masked_positions.shape:
        return values[masked_positions]
    selected_count = torch.nonzero(masked_positions, as_tuple=False).shape[0]
    if values.ndim == 1 and values.shape[0] == selected_count:
        return values
    raise ValueError(
        f"{name} must match masked_positions or contain one value per masked position"
    )


def masked_sequence_log_ratio(
    policy_target_log_probs: torch.Tensor,
    reference_target_log_probs: torch.Tensor,
    masked_positions: torch.Tensor,
    p_mask: torch.Tensor | Sequence[float] | float,
    *,
    geometry_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute length-normalized ``(1/p)`` sequence scores in FP32.

    When ``geometry_mask`` is supplied (required by the scientific trainer),
    divide by the full number of geometry positions.  This prevents the retired
    fixed-107 versus dynamic-``7+4N`` length difference from becoming an
    implicit preference weight across compositions.
    """

    if masked_positions.ndim != 2:
        raise ValueError("masked_positions must have shape [batch, sequence]")
    masked = masked_positions.to(torch.bool)
    geometry_counts: torch.Tensor | None = None
    if geometry_mask is not None:
        geometry = torch.as_tensor(geometry_mask, device=masked.device, dtype=torch.bool)
        if geometry.shape != masked.shape:
            raise ValueError("geometry_mask must match masked_positions")
        if bool((masked & ~geometry).any().item()):
            raise ValueError("masked_positions contains a non-geometry position")
        geometry_counts = geometry.sum(dim=1).to(torch.float32)
        if not bool((geometry_counts > 0).all().item()):
            raise ValueError("every sequence must contain at least one geometry position")

    policy = _selected_log_probs(policy_target_log_probs, masked, name="policy_target_log_probs")
    reference = _selected_log_probs(
        reference_target_log_probs,
        masked,
        name="reference_target_log_probs",
    )
    if policy.shape != reference.shape:
        raise ValueError("policy and reference selected log-probabilities must match")

    batch_size = masked.shape[0]
    probabilities = _probability_vector(p_mask, batch_size=batch_size, device=masked.device)
    sample_indices = torch.nonzero(masked, as_tuple=True)[0]
    deltas = policy.to(torch.float32) - reference.to(torch.float32)
    scores = torch.zeros((batch_size,), dtype=torch.float32, device=masked.device)
    scores = scores.scatter_add(0, sample_indices, deltas)
    scores = scores / probabilities
    return scores if geometry_counts is None else scores / geometry_counts


def winner_denoising_anchor(
    winner_target_log_probs: torch.Tensor,
    masked_positions: torch.Tensor,
    p_mask: torch.Tensor | Sequence[float] | float,
    geometry_mask: torch.Tensor,
) -> torch.Tensor:
    """Return the unbiased per-sequence winner denoising NLL.

    The masked-token NLL is corrected by ``1/p`` and normalized by the full
    number of geometry candidates, matching the existing masked-DLM objective.
    An empty sampled mask contributes exactly zero.
    """

    if masked_positions.ndim != 2:
        raise ValueError("masked_positions must have shape [batch, sequence]")
    masked = masked_positions.to(torch.bool)
    geometry = torch.as_tensor(geometry_mask, device=masked.device, dtype=torch.bool)
    if geometry.shape != masked.shape:
        raise ValueError("geometry_mask must match masked_positions")
    if bool((masked & ~geometry).any().item()):
        raise ValueError("masked_positions contains a non-geometry position")

    selected = _selected_log_probs(
        winner_target_log_probs,
        masked,
        name="winner_target_log_probs",
    ).to(torch.float32)
    batch_size = masked.shape[0]
    probabilities = _probability_vector(p_mask, batch_size=batch_size, device=masked.device)
    geometry_counts = geometry.sum(dim=1).to(torch.float32)
    if not bool((geometry_counts > 0).all().item()):
        raise ValueError("every sequence must contain at least one geometry position")
    sample_indices = torch.nonzero(masked, as_tuple=True)[0]
    nll_sums = torch.zeros((batch_size,), dtype=torch.float32, device=masked.device)
    nll_sums = nll_sums.scatter_add(0, sample_indices, -selected)
    return nll_sums / probabilities / geometry_counts


def composition_normalized_pair_weights(
    composition_ids: torch.Tensor | Sequence[Hashable],
    raw_weights: torch.Tensor | Sequence[float] | None = None,
) -> torch.Tensor:
    """Normalize pair weights so every composition has total weight one."""

    if isinstance(composition_ids, torch.Tensor):
        if composition_ids.ndim != 1:
            raise ValueError("composition_ids must be one-dimensional")
        device = composition_ids.device
        count = composition_ids.shape[0]
        if count == 0:
            return torch.empty((0,), dtype=torch.float32, device=device)
        _, inverse = torch.unique(composition_ids, return_inverse=True)
    else:
        values = list(composition_ids)
        count = len(values)
        if isinstance(raw_weights, torch.Tensor):
            device = raw_weights.device
        else:
            device = torch.device("cpu")
        if count == 0:
            return torch.empty((0,), dtype=torch.float32, device=device)
        codes: dict[Hashable, int] = {}
        encoded: list[int] = []
        for value in values:
            try:
                code = codes.setdefault(value, len(codes))
            except TypeError as exc:
                raise TypeError("composition_ids must be hashable") from exc
            encoded.append(code)
        inverse = torch.tensor(encoded, dtype=torch.long, device=device)

    if raw_weights is None:
        weights = torch.ones((count,), dtype=torch.float32, device=device)
    else:
        weights = torch.as_tensor(raw_weights, dtype=torch.float32, device=device)
        if weights.shape != (count,):
            raise ValueError(f"raw_weights must have shape ({count},)")
    if not bool(torch.isfinite(weights).all().item()) or bool((weights < 0).any().item()):
        raise ValueError("raw_weights must be finite and non-negative")

    group_count = int(inverse.max().item()) + 1
    totals = torch.zeros((group_count,), dtype=torch.float32, device=device)
    totals = totals.scatter_add(0, inverse, weights)
    if not bool((totals > 0).all().item()):
        raise ValueError("every composition must have positive total pair weight")
    return weights / totals.index_select(0, inverse)


def _loss_vector(
    value: torch.Tensor | Sequence[float] | float,
    *,
    pair_count: int,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    return _batch_vector(
        value,
        batch_size=pair_count,
        device=device,
        dtype=torch.float32,
        name=name,
    )


def d3po_pair_loss(
    winner_scores: torch.Tensor,
    loser_scores: torch.Tensor,
    *,
    energy_gaps: torch.Tensor | Sequence[float] | float | None = None,
    target_probabilities: torch.Tensor | Sequence[float] | float | None = None,
    winner_denoising_losses: torch.Tensor | Sequence[float] | float | None = None,
    pair_weights: torch.Tensor | Sequence[float] | None = None,
    beta: float = 0.1,
    energy_temperature: float = 0.03,
    winner_anchor_weight: float = 0.2,
) -> D3POLossOutput:
    """Compute soft D3PO preference loss plus a winner denoising anchor.

    With neither ``energy_gaps`` nor ``target_probabilities`` supplied, the
    target is the hard winner label.  Otherwise the frozen soft target is
    ``sigmoid((E_loser - E_winner) / energy_temperature)``.
    """

    if winner_scores.device != loser_scores.device:
        raise ValueError("winner_scores and loser_scores must share a device")
    winners = winner_scores.to(torch.float32).reshape(-1)
    losers = loser_scores.to(torch.float32).reshape(-1)
    if winners.shape != losers.shape:
        raise ValueError("winner_scores and loser_scores must have identical shape")
    if winners.numel() == 0:
        raise ValueError("at least one preference pair is required")
    if energy_gaps is not None and target_probabilities is not None:
        raise ValueError("supply energy_gaps or target_probabilities, not both")
    if not float(beta) > 0.0:
        raise ValueError("beta must be positive")
    if not float(energy_temperature) > 0.0:
        raise ValueError("energy_temperature must be positive")
    if not float(winner_anchor_weight) >= 0.0:
        raise ValueError("winner_anchor_weight must be non-negative")

    pair_count = winners.numel()
    device = winners.device
    if target_probabilities is not None:
        targets = _loss_vector(
            target_probabilities,
            pair_count=pair_count,
            device=device,
            name="target_probabilities",
        )
        if not bool(((targets >= 0.0) & (targets <= 1.0)).all().item()):
            raise ValueError("target_probabilities must lie in [0, 1]")
    elif energy_gaps is not None:
        gaps = _loss_vector(
            energy_gaps,
            pair_count=pair_count,
            device=device,
            name="energy_gaps",
        )
        if not bool(torch.isfinite(gaps).all().item()) or bool((gaps < 0.0).any().item()):
            raise ValueError("energy_gaps must be finite and non-negative")
        targets = torch.sigmoid(gaps / float(energy_temperature))
    else:
        targets = torch.ones((pair_count,), dtype=torch.float32, device=device)

    if pair_weights is None:
        weights = torch.ones((pair_count,), dtype=torch.float32, device=device)
    else:
        weights = torch.as_tensor(pair_weights, dtype=torch.float32, device=device)
        if weights.shape != (pair_count,):
            raise ValueError(f"pair_weights must have shape ({pair_count},)")
    if not bool(torch.isfinite(weights).all().item()) or bool((weights < 0.0).any().item()):
        raise ValueError("pair_weights must be finite and non-negative")
    weight_sum = weights.sum()
    if not bool((weight_sum > 0.0).item()):
        raise ValueError("pair_weights must have positive total weight")

    margin = winners - losers
    preference_logits = float(beta) * margin
    per_pair_preference = (
        targets * F.softplus(-preference_logits)
        + (1.0 - targets) * F.softplus(preference_logits)
    )
    preference_loss = (per_pair_preference * weights).sum() / weight_sum

    if winner_denoising_losses is None:
        anchor_per_pair = torch.zeros((pair_count,), dtype=torch.float32, device=device)
    else:
        anchor_per_pair = _loss_vector(
            winner_denoising_losses,
            pair_count=pair_count,
            device=device,
            name="winner_denoising_losses",
        )
        if not bool(torch.isfinite(anchor_per_pair).all().item()):
            raise ValueError("winner_denoising_losses must be finite")
    winner_anchor_loss = (anchor_per_pair * weights).sum() / weight_sum
    loss = preference_loss + float(winner_anchor_weight) * winner_anchor_loss
    return D3POLossOutput(
        loss=loss,
        preference_loss=preference_loss,
        winner_anchor_loss=winner_anchor_loss,
        margin=margin,
        target_probability=targets,
        per_pair_preference_loss=per_pair_preference,
    )


__all__ = [
    "D3POLossOutput",
    "SharedGeometryCorruption",
    "composition_normalized_pair_weights",
    "d3po_pair_loss",
    "legal_target_log_probs",
    "masked_sequence_log_ratio",
    "shared_geometry_corruption",
    "winner_denoising_anchor",
]
