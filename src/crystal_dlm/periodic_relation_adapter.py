"""Bounded periodic residual relation adapter for dynamic crystal states.

The adapter operates on the final hidden state of a ``7 + 4N`` crystal body.
It gathers the six lattice-token states and each ``element/X/Y/Z`` site block,
builds a small periodic site graph, and scatters a residual only to lattice and
coordinate token positions.  The output projection is initialized to exactly
zero so inserting the module preserves the base model at optimizer step zero.

The module deliberately does not decode token logits itself.  Callers provide
soft lattice and fractional-coordinate tensors derived from ``q0``.  The
``acyclic_periodic_residual_forward`` helper makes the intended dependency
explicit and never feeds ``q1`` back into the geometry builder.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class PeriodicRelationConfig:
    """Configuration for :class:`PeriodicRelationAdapter`."""

    hidden_size: int
    rank: int = 64
    num_species: int = 119
    num_rbf: int = 16
    rbf_max_distance: float = 10.0
    max_sites: int = 20
    image_radius: int = 1

    def __post_init__(self) -> None:
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.rank <= 0 or self.rank > self.hidden_size:
            raise ValueError("rank must be in 1..hidden_size")
        if self.num_species <= 1:
            raise ValueError("num_species must include padding and real species")
        if self.num_rbf <= 0:
            raise ValueError("num_rbf must be positive")
        if self.rbf_max_distance <= 0.0:
            raise ValueError("rbf_max_distance must be positive")
        if self.max_sites <= 0 or self.max_sites > 20:
            raise ValueError("max_sites must be in 1..20")
        if self.image_radius != 1:
            raise ValueError("image_radius is fixed at one bounded 27-image shell")


@dataclass(frozen=True)
class SoftCrystalGeometry:
    """Soft geometry and dynamic-body metadata derived from base logits.

    ``fractional_coordinates`` and ``species`` are padded to a common site
    width.  ``num_sites`` determines which rows are active.  ``prompt_lengths``
    points to the first token of the dynamic body, whose fixed layout is
    ``N, LA, LB, LC, AA, AB, AG, (E, X, Y, Z) * N``.
    """

    lattice: Tensor
    fractional_coordinates: Tensor
    species: Tensor
    prompt_lengths: Tensor
    num_sites: Tensor


@dataclass(frozen=True)
class PeriodicRelationOutput:
    """Corrected states plus bounded diagnostics used by preflight tests."""

    hidden_states: Tensor
    residual: Tensor
    internal_activation: Tensor
    pair_distances: Tensor
    pair_mask: Tensor

    @property
    def allocated_directed_pair_slots(self) -> int:
        return int(self.pair_mask.shape[-2] * self.pair_mask.shape[-1])

    @property
    def active_directed_pairs(self) -> Tensor:
        return self.pair_mask.sum(dim=(-2, -1))


@dataclass(frozen=True)
class AcyclicPeriodicResidualOutput:
    """Outputs of the one-pass ``q0 -> geometry -> residual -> q1`` helper."""

    q0: Tensor
    q1: Tensor
    relation: PeriodicRelationOutput
    geometry: SoftCrystalGeometry


class _LowRankMessageLayer(nn.Module):
    """One directed low-rank message-passing layer."""

    def __init__(self, rank: int) -> None:
        super().__init__()
        self.sender = nn.Linear(rank, rank, bias=False)
        self.receiver = nn.Linear(rank, rank, bias=False)
        self.edge = nn.Linear(rank, rank, bias=True)
        self.update_norm = nn.LayerNorm(rank)
        self.update_in = nn.Linear(rank, rank)
        self.update_out = nn.Linear(rank, rank)

    def forward(
        self,
        states: Tensor,
        edge_states: Tensor,
        pair_mask: Tensor,
        site_mask: Tensor,
    ) -> Tensor:
        sender = self.sender(states).unsqueeze(1)
        receiver = self.receiver(states).unsqueeze(2)
        messages = F.silu(sender + receiver + self.edge(edge_states))
        messages = messages * pair_mask.unsqueeze(-1).to(messages.dtype)
        degree = pair_mask.sum(dim=-1, keepdim=True).clamp_min(1).to(messages.dtype)
        aggregate = messages.sum(dim=2) / degree
        update = self.update_out(F.silu(self.update_in(self.update_norm(states + aggregate))))
        return (states + update) * site_mask.unsqueeze(-1).to(states.dtype)


class PeriodicRelationAdapter(nn.Module):
    """Two-layer periodic relation adapter with an exactly-zero initial output.

    Pair tensors scale as ``O(N^2 * rank)`` and ``N`` is rejected above
    ``config.max_sites``.  Self edges are allocated but masked, so a 20-site
    structure uses 400 pair slots and 380 active directed edges.
    """

    _NUM_LATTICE_TOKENS = 6
    _NUM_SITE_TOKENS = 4
    _NUM_XYZ_TOKENS = 3

    def __init__(self, config: PeriodicRelationConfig) -> None:
        super().__init__()
        self.config = config
        hidden_size = int(config.hidden_size)
        rank = int(config.rank)

        self.hidden_norm = nn.LayerNorm(hidden_size)
        self.site_projection = nn.Linear(hidden_size, rank)
        self.lattice_projection = nn.Linear(hidden_size, rank)
        self.species_embedding = nn.Embedding(int(config.num_species), rank)
        self.edge_projection = nn.Linear(2 * rank + int(config.num_rbf), rank)
        self.message_layers = nn.ModuleList([_LowRankMessageLayer(rank) for _ in range(2)])

        # Six lattice channels followed by X/Y/Z channels.  Channel embeddings
        # keep the single W_out projection compact while allowing axis-specific
        # corrections.
        self.output_channels = nn.Parameter(torch.empty(9, rank))
        nn.init.normal_(self.output_channels, mean=0.0, std=0.02)
        self.output_projection = nn.Linear(rank, hidden_size, bias=False)
        nn.init.zeros_(self.output_projection.weight)

        centers = torch.linspace(0.0, float(config.rbf_max_distance), int(config.num_rbf))
        if config.num_rbf == 1:
            width = float(config.rbf_max_distance)
        else:
            width = float(config.rbf_max_distance) / float(config.num_rbf - 1)
        self.register_buffer("rbf_centers", centers, persistent=True)
        self.register_buffer("rbf_gamma", torch.tensor(1.0 / max(width * width, 1.0e-12)))

        values = torch.arange(-int(config.image_radius), int(config.image_radius) + 1)
        shifts = torch.cartesian_prod(values, values, values).reshape(-1, 3)
        self.register_buffer("image_shifts", shifts.to(torch.get_default_dtype()), persistent=True)

    def _validate_and_prepare(
        self,
        hidden_states: Tensor,
        geometry: SoftCrystalGeometry,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape [batch, sequence, hidden]")
        batch, sequence_length, hidden_size = hidden_states.shape
        if hidden_size != int(self.config.hidden_size):
            raise ValueError("hidden_states width does not match adapter hidden_size")

        lattice = geometry.lattice.to(device=hidden_states.device, dtype=hidden_states.dtype)
        coordinates = geometry.fractional_coordinates.to(
            device=hidden_states.device, dtype=hidden_states.dtype
        )
        species = geometry.species.to(device=hidden_states.device, dtype=torch.long)
        prompt_lengths = geometry.prompt_lengths.to(device=hidden_states.device, dtype=torch.long)
        num_sites = geometry.num_sites.to(device=hidden_states.device, dtype=torch.long)

        if lattice.shape != (batch, 3, 3):
            raise ValueError("lattice must have shape [batch, 3, 3]")
        if coordinates.ndim != 3 or coordinates.shape[0] != batch or coordinates.shape[-1] != 3:
            raise ValueError("fractional_coordinates must have shape [batch, sites, 3]")
        max_sites = int(coordinates.shape[1])
        if max_sites > int(self.config.max_sites):
            raise ValueError("site width exceeds configured max_sites")
        if species.shape != (batch, max_sites):
            raise ValueError("species must have shape [batch, sites]")
        if prompt_lengths.shape != (batch,) or num_sites.shape != (batch,):
            raise ValueError("prompt_lengths and num_sites must have shape [batch]")
        if bool(((num_sites < 1) | (num_sites > max_sites)).any().item()):
            raise ValueError("num_sites must select 1..site_width sites")
        required_length = prompt_lengths + 7 + 4 * num_sites
        if bool(((prompt_lengths < 0) | (required_length > sequence_length)).any().item()):
            raise ValueError("dynamic 7+4N body lies outside hidden-state sequence")
        if not bool(torch.isfinite(lattice).all().item()):
            raise ValueError("lattice contains non-finite values")
        if not bool(torch.isfinite(coordinates).all().item()):
            raise ValueError("fractional coordinates contain non-finite values")

        site_mask = torch.arange(max_sites, device=hidden_states.device).unsqueeze(0) < num_sites.unsqueeze(1)
        active_species = species.masked_select(site_mask)
        if bool(((active_species < 0) | (active_species >= int(self.config.num_species))).any().item()):
            raise ValueError("active species index outside configured embedding range")
        species = torch.where(site_mask, species, torch.zeros_like(species))
        return lattice, coordinates, species, prompt_lengths, num_sites, site_mask

    @staticmethod
    def _gather(hidden_states: Tensor, positions: Tensor) -> Tensor:
        batch, _, hidden_size = hidden_states.shape
        flat_positions = positions.reshape(batch, -1)
        index = flat_positions.unsqueeze(-1).expand(-1, -1, hidden_size)
        gathered = torch.gather(hidden_states, dim=1, index=index)
        return gathered.reshape(*positions.shape, hidden_size)

    def _dynamic_positions(
        self,
        prompt_lengths: Tensor,
        site_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        batch, max_sites = site_mask.shape
        lattice_offsets = torch.arange(1, 7, device=prompt_lengths.device)
        lattice_positions = prompt_lengths.unsqueeze(1) + lattice_offsets.unsqueeze(0)
        site_offsets = 7 + 4 * torch.arange(max_sites, device=prompt_lengths.device)
        field_offsets = torch.arange(4, device=prompt_lengths.device)
        site_positions = (
            prompt_lengths.reshape(batch, 1, 1)
            + site_offsets.reshape(1, max_sites, 1)
            + field_offsets.reshape(1, 1, 4)
        )
        site_positions = torch.where(site_mask.unsqueeze(-1), site_positions, torch.zeros_like(site_positions))
        return lattice_positions, site_positions

    def _minimum_image_graph(
        self,
        lattice: Tensor,
        coordinates: Tensor,
        site_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        # Axis convention: lattice vectors are rows and Cartesian vectors are
        # fractional row vectors multiplied by the lattice matrix.
        delta = coordinates.unsqueeze(1) - coordinates.unsqueeze(2)
        centered = delta - torch.round(delta)
        shifts = self.image_shifts.to(device=centered.device, dtype=centered.dtype)
        candidates = centered.unsqueeze(-2) + shifts.reshape(1, 1, 1, -1, 3)
        cartesian = torch.einsum("bijnc,bcd->bijnd", candidates, lattice)
        squared = cartesian.square().sum(dim=-1)
        selected = squared.argmin(dim=-1, keepdim=True)
        minimum_squared = torch.gather(squared, dim=-1, index=selected).squeeze(-1)
        distances = torch.sqrt(minimum_squared.clamp_min(0.0))

        max_sites = int(coordinates.shape[1])
        active = site_mask.unsqueeze(1) & site_mask.unsqueeze(2)
        diagonal = torch.eye(max_sites, dtype=torch.bool, device=coordinates.device).unsqueeze(0)
        pair_mask = active & ~diagonal
        return distances.masked_fill(~pair_mask, 0.0), pair_mask

    def _radial_basis(self, distances: Tensor) -> Tensor:
        centers = self.rbf_centers.to(device=distances.device, dtype=distances.dtype)
        gamma = self.rbf_gamma.to(device=distances.device, dtype=distances.dtype)
        return torch.exp(-gamma * (distances.unsqueeze(-1) - centers) ** 2)

    def forward(
        self,
        hidden_states: Tensor,
        geometry: SoftCrystalGeometry,
    ) -> PeriodicRelationOutput:
        lattice, coordinates, species, prompt_lengths, _, site_mask = self._validate_and_prepare(
            hidden_states, geometry
        )
        lattice_positions, site_positions = self._dynamic_positions(prompt_lengths, site_mask)

        lattice_hidden = self._gather(hidden_states, lattice_positions)
        site_hidden = self._gather(hidden_states, site_positions)
        lattice_context = self.lattice_projection(self.hidden_norm(lattice_hidden.mean(dim=1)))
        site_context = self.site_projection(self.hidden_norm(site_hidden.mean(dim=2)))
        species_states = self.species_embedding(species)
        states = site_context + species_states + lattice_context.unsqueeze(1)
        states = states * site_mask.unsqueeze(-1).to(states.dtype)

        pair_distances, pair_mask = self._minimum_image_graph(lattice, coordinates, site_mask)
        radial = self._radial_basis(pair_distances)
        receiver_species = species_states.unsqueeze(2).expand(-1, -1, species_states.shape[1], -1)
        sender_species = species_states.unsqueeze(1).expand(-1, species_states.shape[1], -1, -1)
        edge_inputs = torch.cat((receiver_species, sender_species, radial), dim=-1)
        edge_states = self.edge_projection(edge_inputs)
        edge_states = edge_states * pair_mask.unsqueeze(-1).to(edge_states.dtype)

        for layer in self.message_layers:
            states = layer(states, edge_states, pair_mask, site_mask)

        count = site_mask.sum(dim=1, keepdim=True).clamp_min(1).to(states.dtype)
        pooled = states.sum(dim=1) / count
        lattice_activation = F.silu(pooled.unsqueeze(1) + self.output_channels[:6].unsqueeze(0))
        xyz_activation = F.silu(
            states.unsqueeze(2) + self.output_channels[6:].reshape(1, 1, 3, -1)
        )
        lattice_updates = self.output_projection(lattice_activation)
        xyz_updates = self.output_projection(xyz_activation)
        xyz_updates = xyz_updates * site_mask.reshape(*site_mask.shape, 1, 1).to(xyz_updates.dtype)

        batch, sequence_length, hidden_size = hidden_states.shape
        residual = hidden_states.new_zeros((batch, sequence_length, hidden_size))
        lattice_index = lattice_positions.unsqueeze(-1).expand(-1, -1, hidden_size)
        residual = residual.scatter_add(1, lattice_index, lattice_updates)
        xyz_positions = site_positions[..., 1:].reshape(batch, -1)
        xyz_index = xyz_positions.unsqueeze(-1).expand(-1, -1, hidden_size)
        residual = residual.scatter_add(1, xyz_index, xyz_updates.reshape(batch, -1, hidden_size))

        return PeriodicRelationOutput(
            hidden_states=hidden_states + residual,
            residual=residual,
            internal_activation=states,
            pair_distances=pair_distances,
            pair_mask=pair_mask,
        )


def acyclic_periodic_residual_forward(
    hidden_states: Tensor,
    lm_head: Callable[[Tensor], Tensor],
    geometry_from_q0: Callable[[Tensor], SoftCrystalGeometry],
    adapter: PeriodicRelationAdapter,
) -> AcyclicPeriodicResidualOutput:
    """Apply exactly one acyclic periodic correction before a second LM head.

    ``geometry_from_q0`` is invoked exactly once and receives only ``q0``.
    The returned ``q1`` is never passed back to the geometry builder during
    this call.
    """

    q0 = lm_head(hidden_states)
    geometry = geometry_from_q0(q0)
    if not isinstance(geometry, SoftCrystalGeometry):
        raise TypeError("geometry_from_q0 must return SoftCrystalGeometry")
    relation = adapter(hidden_states, geometry)
    q1 = lm_head(relation.hidden_states)
    return AcyclicPeriodicResidualOutput(q0=q0, q1=q1, relation=relation, geometry=geometry)


__all__ = [
    "AcyclicPeriodicResidualOutput",
    "PeriodicRelationAdapter",
    "PeriodicRelationConfig",
    "PeriodicRelationOutput",
    "SoftCrystalGeometry",
    "acyclic_periodic_residual_forward",
]
