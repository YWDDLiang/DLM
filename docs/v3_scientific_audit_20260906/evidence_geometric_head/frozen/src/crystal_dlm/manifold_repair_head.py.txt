"""Small program-conditioned repair head for periodic crystal transactions.

The head is deliberately independent of the SPAD sampler and of any MLIP.  A
caller supplies DLM hidden states and exact minimum-image vectors calculated
from the current committed crystal.  The module predicts a bounded tangent in
SPD lattice-metric space and translation-free Cartesian site corrections.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ManifoldRepairConfig:
    hidden_size: int
    width: int = 128
    max_sites: int = 20
    max_atomic_number: int = 118
    radial_basis_count: int = 16
    radial_cutoff_A: float = 8.0
    max_metric_tangent: float = 0.20
    max_cartesian_step_A: float = 0.75

    def __post_init__(self) -> None:
        for name in (
            "hidden_size",
            "width",
            "max_sites",
            "max_atomic_number",
            "radial_basis_count",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if float(self.radial_cutoff_A) <= 0.0:
            raise ValueError("radial_cutoff_A must be positive")
        if float(self.max_metric_tangent) <= 0.0:
            raise ValueError("max_metric_tangent must be positive")
        if float(self.max_cartesian_step_A) <= 0.0:
            raise ValueError("max_cartesian_step_A must be positive")


class ManifoldRepairOutput(NamedTuple):
    lattice_tangent: Tensor
    cartesian_site_delta: Tensor
    site_states: Tensor
    pair_scalars: Tensor


def _symmetric_from_six(values: Tensor) -> Tensor:
    if values.shape[-1] != 6:
        raise ValueError("symmetric tangent requires six components")
    result = values.new_zeros((*values.shape[:-1], 3, 3))
    result[..., 0, 0] = values[..., 0]
    result[..., 1, 1] = values[..., 1]
    result[..., 2, 2] = values[..., 2]
    result[..., 0, 1] = result[..., 1, 0] = values[..., 3]
    result[..., 0, 2] = result[..., 2, 0] = values[..., 4]
    result[..., 1, 2] = result[..., 2, 1] = values[..., 5]
    return result


def _bound_translation_free_vectors(values: Tensor, maximum_norm: float) -> Tensor:
    """Bound a site's displacement without reintroducing translation.

    Per-site clipping does not commute with removal of the centre-of-mass mode:
    clipping each vector by a different factor can make an exactly centred set
    translate again.  A single scale per crystal preserves the zero-sum
    invariant while guaranteeing every active site's Cartesian step is bounded.
    """

    norms = torch.linalg.vector_norm(values, dim=-1)
    largest = norms.amax(dim=1, keepdim=True).unsqueeze(-1)
    factor = (float(maximum_norm) / largest.clamp_min(1.0e-12)).clamp(max=1.0)
    return values * factor


class ManifoldRepairHead(nn.Module):
    """Predict lattice and site repair vectors from one committed crystal.

    ``mic_vectors[i, j]`` is the minimum-image Cartesian vector from site i to
    site j.  ``pair_mask`` must exclude diagonal and padded pairs.  Output
    projections are exactly zero initialized, making the complete head an
    identity-preserving addition to an existing DLM.
    """

    def __init__(self, config: ManifoldRepairConfig) -> None:
        super().__init__()
        self.config = config
        width = int(config.width)
        hidden = int(config.hidden_size)
        self.hidden_norm = nn.LayerNorm(hidden)
        self.cell_projection = nn.Linear(hidden, width)
        self.site_projection = nn.Linear(hidden, width)
        self.plan_projection = nn.Linear(hidden, width, bias=False)
        self.species_embedding = nn.Embedding(
            int(config.max_atomic_number) + 1, width, padding_idx=0
        )
        self.program_rank_embedding = nn.Embedding(
            int(config.max_sites) + 1, width, padding_idx=int(config.max_sites)
        )
        self.pair_projection = nn.Sequential(
            nn.Linear(2 * width + int(config.radial_basis_count), width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
        )
        self.pair_output = nn.Linear(width, 1, bias=False)
        self.metric_hidden = nn.Sequential(
            nn.Linear(2 * width, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
        )
        self.metric_output = nn.Linear(width, 6, bias=False)
        centers = torch.linspace(
            0.0, float(config.radial_cutoff_A), int(config.radial_basis_count)
        )
        spacing = float(config.radial_cutoff_A) / max(
            int(config.radial_basis_count) - 1, 1
        )
        self.register_buffer("radial_centers", centers, persistent=True)
        self.register_buffer(
            "radial_gamma",
            torch.tensor(1.0 / max(spacing * spacing, 1.0e-6)),
            persistent=True,
        )
        nn.init.zeros_(self.pair_output.weight)
        nn.init.zeros_(self.metric_output.weight)

    def _validate(
        self,
        lattice_hidden: Tensor,
        site_hidden: Tensor,
        species: Tensor,
        program_rank: Tensor,
        site_mask: Tensor,
        mic_vectors: Tensor,
        pair_mask: Tensor,
        plan_hidden: Tensor | None,
    ) -> None:
        if lattice_hidden.ndim != 3 or lattice_hidden.shape[1:] != (
            6,
            int(self.config.hidden_size),
        ):
            raise ValueError("lattice_hidden must have shape [batch, 6, hidden]")
        if site_hidden.ndim != 4 or site_hidden.shape[2:] != (
            3,
            int(self.config.hidden_size),
        ):
            raise ValueError("site_hidden must have shape [batch, sites, 3, hidden]")
        batch, sites = site_hidden.shape[:2]
        if sites > int(self.config.max_sites):
            raise ValueError("site count exceeds configured maximum")
        expected = (batch, sites)
        if species.shape != expected or program_rank.shape != expected:
            raise ValueError("species/program_rank must have shape [batch, sites]")
        if site_mask.shape != expected or site_mask.dtype is not torch.bool:
            raise ValueError("site_mask must be boolean [batch, sites]")
        if mic_vectors.shape != (batch, sites, sites, 3):
            raise ValueError("mic_vectors must have shape [batch, sites, sites, 3]")
        if pair_mask.shape != (batch, sites, sites) or pair_mask.dtype is not torch.bool:
            raise ValueError("pair_mask must be boolean [batch, sites, sites]")
        if lattice_hidden.shape[0] != batch:
            raise ValueError("lattice and site batch sizes differ")
        if plan_hidden is not None and plan_hidden.shape != (
            batch,
            int(self.config.hidden_size),
        ):
            raise ValueError("plan_hidden must have shape [batch, hidden]")
        active_species = species[site_mask]
        if bool(
            (
                (active_species < 1)
                | (active_species > int(self.config.max_atomic_number))
            ).any()
        ):
            raise ValueError("active species lies outside configured range")
        active_rank = program_rank[site_mask]
        if bool(((active_rank < 0) | (active_rank >= sites)).any()):
            raise ValueError("active program rank lies outside site range")
        diagonal = torch.eye(sites, dtype=torch.bool, device=pair_mask.device).unsqueeze(0)
        valid_pairs = site_mask.unsqueeze(1) & site_mask.unsqueeze(2) & ~diagonal
        if bool((pair_mask & ~valid_pairs).any()):
            raise ValueError("pair_mask includes diagonal or padded pair")
        if not bool(torch.isfinite(mic_vectors[pair_mask]).all()):
            raise ValueError("active minimum-image vectors must be finite")

    def forward(
        self,
        *,
        lattice_hidden: Tensor,
        site_hidden: Tensor,
        species: Tensor,
        program_rank: Tensor,
        site_mask: Tensor,
        mic_vectors: Tensor,
        pair_mask: Tensor,
        plan_hidden: Tensor | None = None,
    ) -> ManifoldRepairOutput:
        self._validate(
            lattice_hidden,
            site_hidden,
            species,
            program_rank,
            site_mask,
            mic_vectors,
            pair_mask,
            plan_hidden,
        )
        dtype = self.cell_projection.weight.dtype
        # The retained LLaDA exposes bfloat16 hidden states while this small
        # repair head is intentionally trained in float32.  Cast before
        # LayerNorm so its input and parameters share dtype.
        lattice_hidden = self.hidden_norm(lattice_hidden.to(dtype=dtype))
        site_hidden = self.hidden_norm(site_hidden.to(dtype=dtype))
        species = species.to(device=site_hidden.device, dtype=torch.long)
        site_mask = site_mask.to(device=site_hidden.device)
        program_rank = program_rank.to(device=site_hidden.device, dtype=torch.long)
        padded_rank = torch.full_like(program_rank, int(self.config.max_sites))
        program_rank = torch.where(site_mask, program_rank, padded_rank)

        cell_state = self.cell_projection(lattice_hidden.mean(dim=1))
        site_state = self.site_projection(site_hidden.mean(dim=2))
        site_state = site_state + self.species_embedding(species)
        site_state = site_state + self.program_rank_embedding(program_rank)
        if plan_hidden is not None:
            context = self.plan_projection(
                self.hidden_norm(
                    plan_hidden.to(device=site_hidden.device, dtype=dtype)
                )
            )
            cell_state = cell_state + context
            site_state = site_state + context.unsqueeze(1)
        site_state = site_state * site_mask.unsqueeze(-1).to(dtype)

        vectors = mic_vectors.to(device=site_hidden.device, dtype=dtype)
        distances = torch.linalg.vector_norm(vectors, dim=-1)
        radial = torch.exp(
            -self.radial_gamma.to(dtype=dtype)
            * (distances.unsqueeze(-1) - self.radial_centers.to(dtype=dtype)) ** 2
        )
        left = site_state.unsqueeze(2).expand(-1, -1, site_state.shape[1], -1)
        right = site_state.unsqueeze(1).expand(-1, site_state.shape[1], -1, -1)
        pair_features = torch.cat((left + right, torch.abs(left - right), radial), dim=-1)
        pair_features = self.pair_projection(pair_features)
        pair_scalars = self.pair_output(pair_features).squeeze(-1)
        symmetric_mask = pair_mask & pair_mask.transpose(1, 2)
        pair_scalars = 0.5 * (pair_scalars + pair_scalars.transpose(1, 2))
        pair_scalars = pair_scalars * symmetric_mask.to(dtype)
        unit = vectors / distances.clamp_min(1.0e-8).unsqueeze(-1)
        site_delta = (pair_scalars.unsqueeze(-1) * unit).sum(dim=2)
        denominator = (site_mask.sum(dim=1, keepdim=True) - 1).clamp_min(1)
        site_delta = site_delta / denominator.to(dtype).unsqueeze(-1)
        site_delta = site_delta * site_mask.unsqueeze(-1).to(dtype)
        # Numerical and imperfect-input antisymmetry cannot introduce a global
        # translation mode.
        count = site_mask.sum(dim=1, keepdim=True).clamp_min(1).to(dtype)
        center = site_delta.sum(dim=1, keepdim=True) / count.unsqueeze(-1)
        site_delta = (site_delta - center) * site_mask.unsqueeze(-1).to(dtype)
        site_delta = _bound_translation_free_vectors(
            site_delta, float(self.config.max_cartesian_step_A)
        )

        pooled_sites = site_state.sum(dim=1) / count
        metric_state = self.metric_hidden(torch.cat((cell_state, pooled_sites), dim=-1))
        metric_values = torch.tanh(self.metric_output(metric_state))
        metric_values = metric_values * float(self.config.max_metric_tangent)
        lattice_tangent = _symmetric_from_six(metric_values)

        output_dtype = site_hidden.dtype
        return ManifoldRepairOutput(
            lattice_tangent=lattice_tangent.to(dtype=output_dtype),
            cartesian_site_delta=site_delta.to(dtype=output_dtype),
            site_states=site_state.to(dtype=output_dtype),
            pair_scalars=pair_scalars.to(dtype=output_dtype),
        )


__all__ = [
    "ManifoldRepairConfig",
    "ManifoldRepairHead",
    "ManifoldRepairOutput",
]
