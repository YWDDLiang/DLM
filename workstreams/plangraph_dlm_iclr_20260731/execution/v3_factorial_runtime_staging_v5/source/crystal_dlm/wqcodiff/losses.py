"""Per-valid-term normalized co-denoising objectives."""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn.functional as functional
from torch import Tensor

from .model import WQModelOutput, WQPriorOutput


class WQLossTargets(NamedTuple):
    space_group: Tensor
    species: Tensor
    wyckoff: Tensor
    event: Tensor
    event_orbit: Tensor
    event_orbit_mask: Tensor
    birth_species: Tensor
    birth_wyckoff: Tensor
    birth_coordinate: Tensor
    birth_coordinate_mask: Tensor
    revision: Tensor
    revision_mask: Tensor
    coordinate_score: Tensor
    coordinate_mask: Tensor
    coordinate_weight: Tensor
    lattice_score: Tensor
    lattice_mask: Tensor
    bridge_coordinate: Tensor
    bridge_mask: Tensor


class WQPriorTargets(NamedTuple):
    space_group: Tensor
    first_species: Tensor
    first_wyckoff: Tensor
    first_coordinate: Tensor
    first_coordinate_mask: Tensor
    lattice_chart: Tensor
    lattice_chart_mask: Tensor


class WQLosses(NamedTuple):
    total: Tensor
    space_group: Tensor
    species: Tensor
    wyckoff: Tensor
    event: Tensor
    event_payload: Tensor
    revision: Tensor
    geometry: Tensor
    bridge: Tensor
    prior: Tensor


class WQLossTerms(NamedTuple):
    """Atomic normalized terms used by training and gradient diagnostics."""

    space_group: Tensor
    species: Tensor
    wyckoff: Tensor
    event: Tensor
    event_pointer: Tensor
    birth_species: Tensor
    birth_wyckoff: Tensor
    birth_coordinate: Tensor
    revision: Tensor
    coordinate_score: Tensor
    lattice_score: Tensor
    bridge: Tensor
    prior_space_group: Tensor
    prior_species: Tensor
    prior_wyckoff: Tensor
    prior_coordinate: Tensor
    prior_lattice: Tensor


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    mask = mask.to(values.dtype)
    denominator = mask.sum().clamp_min(1.0)
    return (values * mask).sum() / denominator


def _masked_cross_entropy(logits: Tensor, target: Tensor, ignore_index: int = -100) -> Tensor:
    values = functional.cross_entropy(logits, target, reduction="none", ignore_index=ignore_index)
    return _masked_mean(values, target != ignore_index)


def _gaussian_nll(mean: Tensor, log_scale: Tensor, value: Tensor) -> Tensor:
    variance = (2.0 * log_scale).exp()
    return 0.5 * ((value - mean).square() / variance + 2.0 * log_scale)


def _wrapped_gaussian_nll(
    mean: Tensor,
    log_scale: Tensor,
    value: Tensor,
    *,
    integer_image_radius: int = 8,
) -> Tensor:
    """Truncated wrapped-normal NLL on a unit torus.

    Sampling draws an ordinary Gaussian and reduces it modulo one.  Its exact
    likelihood is therefore the sum of all integer-translated Gaussian
    images.  The registered radius of eight is numerically exhaustive for the
    bounded periodic scales emitted by the model.  Computation is forced to
    float32 so BF16 autocast cannot underflow the log-sum-exp tails.
    """

    if mean.shape != log_scale.shape or mean.shape != value.shape:
        raise ValueError("wrapped Gaussian mean/scale/value shape mismatch")
    if integer_image_radius < 1:
        raise ValueError("wrapped Gaussian image radius must be positive")
    residual = (value.float() - mean.float() + 0.5).remainder(1.0) - 0.5
    scale_log = log_scale.float()
    shifts = mean.new_tensor(
        tuple(range(-integer_image_radius, integer_image_radius + 1)),
        dtype=torch.float32,
    )
    standardized = (
        residual.unsqueeze(-1) + shifts
    ) / scale_log.exp().unsqueeze(-1)
    log_images = -0.5 * standardized.square() - scale_log.unsqueeze(-1)
    return -log_images.logsumexp(dim=-1)


def _coordinate_score_loss(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
    weight: Tensor,
) -> Tensor:
    """Sigma-squared weighted denoising score-matching objective.

    Wrapped-Gaussian score targets scale as ``1 / sigma``.  Weighting their
    squared error by ``sigma**2`` keeps the objective finite and comparable
    across timesteps while preserving the score parameterization required by
    the reverse-time sampler.  The final mean is still taken over valid atoms,
    so ragged orbit or atom counts cannot change a structure's padding weight.
    """

    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("coordinate score prediction/target shape mismatch")
    if mask.shape != prediction.shape[:1] or weight.shape != mask.shape:
        raise ValueError("coordinate score mask/weight shape mismatch")
    values = (prediction - target).square().mean(dim=-1) * weight
    return _masked_mean(values, mask)


def compute_wq_loss_terms(
    output: WQModelOutput,
    target: WQLossTargets,
    *,
    masked_prior: WQPriorOutput | None = None,
    conditioned_prior: WQPriorOutput | None = None,
    prior_target: WQPriorTargets | None = None,
) -> WQLossTerms:
    sg_loss = _masked_cross_entropy(output.space_group_logits, target.space_group)
    species_loss = _masked_cross_entropy(output.species_logits, target.species)
    wyckoff_loss = _masked_cross_entropy(output.wyckoff_logits, target.wyckoff)
    event_loss = _masked_cross_entropy(output.event_logits, target.event)
    pointer_values = functional.binary_cross_entropy_with_logits(
        output.event_orbit_logits,
        target.event_orbit.to(output.event_orbit_logits.dtype),
        reduction="none",
    )
    pointer_loss = _masked_mean(pointer_values, target.event_orbit_mask)
    birth_species_loss = _masked_cross_entropy(output.birth_species_logits, target.birth_species)
    birth_wyckoff_loss = _masked_cross_entropy(output.birth_wyckoff_logits, target.birth_wyckoff)
    birth_nll = _wrapped_gaussian_nll(
        output.birth_coordinate_mean,
        output.birth_coordinate_log_scale,
        target.birth_coordinate,
    )
    birth_coordinate_loss = _masked_mean(birth_nll, target.birth_coordinate_mask)
    revision_values = functional.binary_cross_entropy_with_logits(
        output.revision_logits,
        target.revision.to(output.revision_logits.dtype),
        reduction="none",
    )
    revision_loss = _masked_mean(revision_values, target.revision_mask)
    lattice_values = (output.lattice_score - target.lattice_score).square()
    coordinate_score_loss = _coordinate_score_loss(
        output.atom_coordinate_score,
        target.coordinate_score,
        target.coordinate_mask,
        target.coordinate_weight,
    )
    lattice_score_loss = _masked_mean(lattice_values, target.lattice_mask)
    bridge_nll = _wrapped_gaussian_nll(
        output.bridge_mean,
        output.bridge_log_scale,
        target.bridge_coordinate,
    )
    bridge_loss = _masked_mean(bridge_nll, target.bridge_mask)
    if (masked_prior is None) != (conditioned_prior is None) or (
        masked_prior is None
    ) != (prior_target is None):
        raise ValueError("masked prior, conditioned prior, and prior target must be provided together")
    prior_sg = output.space_group_logits.new_zeros(())
    prior_species = output.space_group_logits.new_zeros(())
    prior_wyckoff = output.space_group_logits.new_zeros(())
    prior_coordinate = output.space_group_logits.new_zeros(())
    prior_lattice = output.space_group_logits.new_zeros(())
    if masked_prior is not None and conditioned_prior is not None and prior_target is not None:
        prior_sg = _masked_cross_entropy(
            masked_prior.space_group_logits,
            prior_target.space_group,
        )
        prior_species = _masked_cross_entropy(
            conditioned_prior.first_species_logits,
            prior_target.first_species,
        )
        prior_wyckoff = _masked_cross_entropy(
            conditioned_prior.first_wyckoff_logits,
            prior_target.first_wyckoff,
        )
        prior_coordinate = _masked_mean(
            _wrapped_gaussian_nll(
                conditioned_prior.first_coordinate_mean,
                conditioned_prior.first_coordinate_log_scale,
                prior_target.first_coordinate,
            ),
            prior_target.first_coordinate_mask,
        )
        prior_lattice = _masked_mean(
            _gaussian_nll(
                conditioned_prior.lattice_chart_mean,
                conditioned_prior.lattice_chart_log_scale,
                prior_target.lattice_chart,
            ),
            prior_target.lattice_chart_mask,
        )
    return WQLossTerms(
        space_group=sg_loss,
        species=species_loss,
        wyckoff=wyckoff_loss,
        event=event_loss,
        event_pointer=pointer_loss,
        birth_species=birth_species_loss,
        birth_wyckoff=birth_wyckoff_loss,
        birth_coordinate=birth_coordinate_loss,
        revision=revision_loss,
        coordinate_score=coordinate_score_loss,
        lattice_score=lattice_score_loss,
        bridge=bridge_loss,
        prior_space_group=prior_sg,
        prior_species=prior_species,
        prior_wyckoff=prior_wyckoff,
        prior_coordinate=prior_coordinate,
        prior_lattice=prior_lattice,
    )


def compute_wq_losses(
    output: WQModelOutput,
    target: WQLossTargets,
    *,
    masked_prior: WQPriorOutput | None = None,
    conditioned_prior: WQPriorOutput | None = None,
    prior_target: WQPriorTargets | None = None,
) -> WQLosses:
    terms = compute_wq_loss_terms(
        output,
        target,
        masked_prior=masked_prior,
        conditioned_prior=conditioned_prior,
        prior_target=prior_target,
    )
    event_payload_loss = (
        terms.event_pointer
        + terms.birth_species
        + terms.birth_wyckoff
        + terms.birth_coordinate
    )
    geometry_loss = terms.coordinate_score + terms.lattice_score
    prior_loss = (
        terms.prior_space_group
        + terms.prior_species
        + terms.prior_wyckoff
        + terms.prior_coordinate
        + terms.prior_lattice
    )
    # Every atomic term has already been normalized over its valid supervision
    # count.  Equal weighting is the registered objective and cannot change
    # with padding, orbit count, or diagnostic grouping.
    # Keep the registered objective and the component-gradient audit on the
    # same canonical left-fold.  Regrouping these float32 additions is
    # mathematically equivalent but can differ by a few ULPs, which makes an
    # exact provenance audit needlessly dependent on diagnostic grouping.
    total = sum(terms)
    return WQLosses(
        total=total,
        space_group=terms.space_group,
        species=terms.species,
        wyckoff=terms.wyckoff,
        event=terms.event,
        event_payload=event_payload_loss,
        revision=terms.revision,
        geometry=geometry_loss,
        bridge=terms.bridge,
        prior=prior_loss,
    )
