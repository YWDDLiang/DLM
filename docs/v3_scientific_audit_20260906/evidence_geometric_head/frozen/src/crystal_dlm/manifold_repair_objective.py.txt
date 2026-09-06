"""Loss primitives for program-matched manifold repair.

This module deliberately returns separately normalized components.  The PMTR
trainer alternates clean-token and corrupted-repair microbatches, so it must
not hide an uncalibrated fixed mixture of unrelated gradient scales here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import Tensor
import torch.nn.functional as F

from crystal_dlm.manifold_repair_head import ManifoldRepairOutput


@dataclass(frozen=True)
class ManifoldRepairLossConfig:
    lattice_tangent_scale: float
    cartesian_step_scale_A: float
    step_regularization: float = 1.0e-3

    def __post_init__(self) -> None:
        if float(self.lattice_tangent_scale) <= 0.0:
            raise ValueError("lattice_tangent_scale must be positive")
        if float(self.cartesian_step_scale_A) <= 0.0:
            raise ValueError("cartesian_step_scale_A must be positive")
        if float(self.step_regularization) < 0.0:
            raise ValueError("step_regularization must be non-negative")


class ManifoldRepairLosses(NamedTuple):
    repair: Tensor
    lattice: Tensor
    coordinate: Tensor
    step: Tensor


def masked_transaction_cross_entropy(
    logits: Tensor,
    target_ids: Tensor,
    active_mask: Tensor,
) -> Tensor:
    """Mean CE over one or more active transaction components."""

    if logits.ndim != 3 or target_ids.shape != logits.shape[:2]:
        raise ValueError("logits/target_ids shape mismatch")
    if active_mask.shape != target_ids.shape or active_mask.dtype is not torch.bool:
        raise ValueError("active_mask must be boolean and match target_ids")
    if not bool(active_mask.any().item()):
        raise ValueError("transaction CE requires at least one active token")
    return F.cross_entropy(logits[active_mask], target_ids[active_mask], reduction="mean")


def reference_logit_kl(
    adapted_logits: Tensor,
    reference_logits: Tensor,
    active_mask: Tensor,
) -> Tensor:
    """Forward KL from the frozen reference to the adapted token distribution."""

    if adapted_logits.shape != reference_logits.shape or adapted_logits.ndim != 3:
        raise ValueError("adapted/reference logits must have identical [B,L,V] shape")
    if active_mask.shape != adapted_logits.shape[:2] or active_mask.dtype is not torch.bool:
        raise ValueError("active_mask must match the logit prefix dimensions")
    if not bool(active_mask.any().item()):
        raise ValueError("reference KL requires at least one active token")
    reference_log_prob = F.log_softmax(reference_logits[active_mask].float(), dim=-1)
    adapted_log_prob = F.log_softmax(adapted_logits[active_mask].float(), dim=-1)
    reference_prob = reference_log_prob.exp()
    return (reference_prob * (reference_log_prob - adapted_log_prob)).sum(dim=-1).mean()


def manifold_repair_losses(
    output: ManifoldRepairOutput,
    *,
    target_lattice_tangent: Tensor,
    target_cartesian_site_delta: Tensor,
    site_mask: Tensor,
    lattice_active: Tensor,
    active_site_mask: Tensor,
    config: ManifoldRepairLossConfig,
) -> ManifoldRepairLosses:
    """Dimensionless SPD and translation-free Cartesian repair losses.

    ``lattice_active`` and ``active_site_mask`` make the supervision match the
    transaction visited by the SPAD repair program.  They are mutually
    exclusive per row; a batch may contain both transaction kinds.
    """

    batch = output.lattice_tangent.shape[0]
    sites = output.cartesian_site_delta.shape[1]
    if output.lattice_tangent.shape != (batch, 3, 3):
        raise ValueError("lattice_tangent must have shape [batch,3,3]")
    if target_lattice_tangent.shape != (batch, 3, 3):
        raise ValueError("target_lattice_tangent must have shape [batch,3,3]")
    expected_sites = (batch, sites)
    if output.cartesian_site_delta.shape != (batch, sites, 3):
        raise ValueError("cartesian_site_delta must have shape [batch,sites,3]")
    if target_cartesian_site_delta.shape != (batch, sites, 3):
        raise ValueError("target_cartesian_site_delta has the wrong shape")
    if site_mask.shape != expected_sites or site_mask.dtype is not torch.bool:
        raise ValueError("site_mask must be boolean [batch,sites]")
    if active_site_mask.shape != expected_sites or active_site_mask.dtype is not torch.bool:
        raise ValueError("active_site_mask must be boolean [batch,sites]")
    if lattice_active.shape != (batch,) or lattice_active.dtype is not torch.bool:
        raise ValueError("lattice_active must be boolean [batch]")
    if bool((active_site_mask & ~site_mask).any().item()):
        raise ValueError("active site lies outside site_mask")
    site_counts = active_site_mask.sum(dim=1)
    if bool(((site_counts > 1) | (lattice_active & (site_counts > 0))).any().item()):
        raise ValueError("each row must supervise one cell or at most one site transaction")
    if not bool((lattice_active | (site_counts == 1)).all().item()):
        raise ValueError("every row must select one repair transaction")

    tangent_error = (
        output.lattice_tangent.float() - target_lattice_tangent.float()
    ) / float(config.lattice_tangent_scale)
    # Frobenius error is the natural local metric for a symmetric SPD tangent.
    per_lattice = tangent_error.square().sum(dim=(-2, -1)) / 6.0
    if bool(lattice_active.any().item()):
        lattice_loss = per_lattice[lattice_active].mean()
    else:
        lattice_loss = output.lattice_tangent.sum() * 0.0

    coordinate_error = (
        output.cartesian_site_delta.float()
        - target_cartesian_site_delta.float()
    ) / float(config.cartesian_step_scale_A)
    per_site = coordinate_error.square().sum(dim=-1) / 3.0
    if bool(active_site_mask.any().item()):
        coordinate_loss = per_site[active_site_mask].mean()
    else:
        coordinate_loss = output.cartesian_site_delta.sum() * 0.0

    predicted_lattice_step = (
        output.lattice_tangent.float() / float(config.lattice_tangent_scale)
    ).square().sum(dim=(-2, -1)) / 6.0
    predicted_site_step = (
        output.cartesian_site_delta.float() / float(config.cartesian_step_scale_A)
    ).square().sum(dim=-1) / 3.0
    selected_steps = torch.cat(
        (
            predicted_lattice_step[lattice_active],
            predicted_site_step[active_site_mask],
        )
    )
    step_loss = selected_steps.mean() * float(config.step_regularization)
    repair = lattice_loss + coordinate_loss + step_loss
    return ManifoldRepairLosses(
        repair=repair,
        lattice=lattice_loss,
        coordinate=coordinate_loss,
        step=step_loss,
    )


def trainable_gradient_l2(parameters) -> Tensor:
    """Return a finite L2 gradient norm for preflight scale diagnostics."""

    squared = None
    for parameter in parameters:
        if not parameter.requires_grad or parameter.grad is None:
            continue
        term = parameter.grad.detach().float().square().sum()
        squared = term if squared is None else squared + term
    if squared is None:
        raise ValueError("no trainable gradients were populated")
    norm = squared.sqrt()
    if not bool(torch.isfinite(norm).item()):
        raise FloatingPointError("gradient norm is not finite")
    return norm


__all__ = [
    "ManifoldRepairLossConfig",
    "ManifoldRepairLosses",
    "manifold_repair_losses",
    "masked_transaction_cross_entropy",
    "reference_logit_kl",
    "trainable_gradient_l2",
]
