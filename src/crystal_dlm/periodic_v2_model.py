"""Head-specific periodic attention, shared across Transformer layers.

The unchanged legacy scalar bias is broadcast to all heads. New site/site and
six-cell-slot/site residuals start at zero, with gates starting at one. This
supports an exact identity migration check; it does not choose a checkpoint or
change the fresh-initialization policy of the enclosing model.

``geometry`` must be the enclosing model's ``geometry_inputs(context)`` result,
decoded from the old integer token canvas. No clean target, continuous pre-codec
corruption, probability-averaged coordinate, or learned geometric output is
accepted by this interface. Geometry is fixed conditioning, so its small matrix
logarithm is evaluated without autograd; learned feature encoders retain their
gradients. The final bias is recomputed on every call, including mask/noise
gates. There is no physical-feature cache or mutable checkpoint-time hook.

Cost is O(B N^2 (2r+1)^3) for the finite-image geometry and O(B H L^2) for the
returned attention tensor, in addition to the retained legacy computation. The
same tensor may be consumed by every layer; no layer-specific routing is claimed.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import torch
from torch import nn

from crystal_dlm.periodic_repair_model import (
    FP32Module,
    PeriodicAttentionBias,
    PeriodicRepairConfig,
)


@dataclass(frozen=True)
class PeriodicV2Config:
    hidden_size: int
    heads: int = 32
    width: int = 64
    radial_bins: int = 16
    fourier_modes: int = 4
    image_radius: int = 2
    radial_cutoff_A: float = 6.0
    species_width: int = 16
    cell_slot_width: int = 8
    max_sites: int = 20
    schema: str = "periodic_v2_head_specific_layer_shared_v1"

    def __post_init__(self) -> None:
        dimensions = (
            self.hidden_size, self.heads, self.width, self.radial_bins,
            self.fourier_modes, self.species_width, self.cell_slot_width,
            self.max_sites,
        )
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 1
               for value in dimensions):
            raise ValueError("periodic V2 dimensions must be positive integers")
        if self.hidden_size % self.heads:
            raise ValueError("hidden size must be divisible by attention heads")
        if self.image_radius not in (1, 2):
            raise ValueError("periodic V2 uses a declared finite image radius 1 or 2")
        if not math.isfinite(self.radial_cutoff_A) or self.radial_cutoff_A <= 0:
            raise ValueError("radial cutoff must be finite and positive")


def numeric_noise_component_features(context: Any) -> torch.Tensor:
    """Return three declared sigma values plus their joint known flag.

    Missing metadata and (-1,-1,-1) mean unknown. Partial hiding is rejected:
    otherwise a supposedly dropped noise level would remain observable through
    another channel. The legacy scalar noise value is deliberately not consulted;
    V2 can keep it unknown while exposing separately typed sigma channels.
    """
    batch = context.old_token_ids.shape[0]
    device = context.old_token_ids.device
    values = getattr(context, "numeric_noise_components", None)
    if values is None:
        return torch.zeros(batch, 4, dtype=torch.float32, device=device)
    if not isinstance(values, torch.Tensor) or values.shape != (batch, 3):
        raise ValueError("numeric_noise_components must have shape [batch, 3]")
    values = values.detach().to(device=device, dtype=torch.float32)
    unknown = (values == -1).all(-1)
    known = (values >= 0).all(-1)
    if not bool(torch.isfinite(values).all()) or not bool((unknown | known).all()):
        raise ValueError("typed noise is nonnegative in every channel or all -1")
    return torch.cat((torch.where(known[:, None], values, 0.), known[:, None].float()), -1)


def _symmetric_components(matrix: torch.Tensor) -> torch.Tensor:
    """Frobenius-preserving six-vector for a symmetric 3x3 matrix."""
    root_two = math.sqrt(2.)
    return torch.stack((
        matrix[..., 0, 0], matrix[..., 1, 1], matrix[..., 2, 2],
        root_two * matrix[..., 0, 1], root_two * matrix[..., 0, 2],
        root_two * matrix[..., 1, 2],
    ), -1)


class PeriodicV2AttentionBias(FP32Module):
    """Old scalar bias plus zero-initialized head-wise geometric residuals.

    ``forward(geometry, context, length, tasks)`` retains the V1 signature and
    returns FP32 ``[batch, heads, length, length]``. Six distinct learned cell-slot
    embeddings and independent directional heads define cell->site and site->cell
    interactions. Per-site neighbor pooling is never replaced by a global mean.
    A known single-atom crystal has zero neighbors but retains its real cell/site
    descriptors; unknown cell or target site always contributes zero geometry.
    """

    def __init__(
        self,
        legacy_config: PeriodicRepairConfig,
        v2_config: PeriodicV2Config,
        legacy: PeriodicAttentionBias | None = None,
    ) -> None:
        super().__init__()
        if legacy_config.hidden_size != v2_config.hidden_size:
            raise ValueError("legacy and V2 hidden sizes differ")
        if legacy_config.image_radius != v2_config.image_radius:
            raise ValueError("legacy and V2 must use the same declared periodic shell")
        if legacy is not None and legacy.config != legacy_config:
            raise ValueError("supplied legacy bias does not match its config")
        self.config = v2_config
        self.v2_config = v2_config
        self.legacy_config = legacy_config
        self.legacy = PeriodicAttentionBias(legacy_config).float() if legacy is None else legacy
        self.species = nn.Embedding(119, v2_config.species_width, padding_idx=0, dtype=torch.float32)
        self.cell_slots = nn.Embedding(6, v2_config.cell_slot_width, dtype=torch.float32)

        # abs RBF + log(1+d_A) + d/cell_scale + periodic Fourier + two species
        # + nine cell descriptors + seven pair flags/ranks + nine old task
        # channels + three typed sigma values and their known flag.
        pair_width = (v2_config.radial_bins + 6 * v2_config.fourier_modes
                      + 2 * v2_config.species_width + 31)
        self.pair_encoder = nn.Sequential(
            nn.Linear(pair_width, v2_config.width, dtype=torch.float32), nn.SiLU(),
            nn.Linear(v2_config.width, v2_config.width, dtype=torch.float32), nn.SiLU(),
        )
        self.site_to_site = nn.Linear(v2_config.width, v2_config.heads, bias=False, dtype=torch.float32)
        # Per-site neighborhood + species + cell9 + site flags5 + task/noise13,
        # then the six-valued slot embedding and that cell slot's active flag.
        cell_width = (v2_config.width + v2_config.species_width + 27
                      + v2_config.cell_slot_width + 1)
        self.cell_site_encoder = nn.Sequential(
            nn.Linear(cell_width, v2_config.width, dtype=torch.float32), nn.SiLU(),
        )
        self.cell_to_site = nn.Linear(v2_config.width, v2_config.heads, bias=False, dtype=torch.float32)
        self.site_to_cell = nn.Linear(v2_config.width, v2_config.heads, bias=False, dtype=torch.float32)
        self.noise_gate = nn.Linear(13, v2_config.heads, dtype=torch.float32)
        for projection in (self.site_to_site, self.cell_to_site, self.site_to_cell):
            nn.init.zeros_(projection.weight)
        nn.init.zeros_(self.noise_gate.weight)
        nn.init.zeros_(self.noise_gate.bias)

        axis = torch.arange(-v2_config.image_radius, v2_config.image_radius + 1,
                            dtype=torch.float32)
        self.register_buffer("shifts", torch.cartesian_prod(axis, axis, axis))
        self.register_buffer("radial_centers", torch.linspace(
            0., v2_config.radial_cutoff_A, v2_config.radial_bins, dtype=torch.float32))
        self.register_buffer("fourier_frequencies", torch.arange(
            1, v2_config.fourier_modes + 1, dtype=torch.float32))

    def _validate(self, geometry: Mapping[str, torch.Tensor], context: Any,
                  length: int, tasks: torch.Tensor) -> tuple[int, int]:
        ids = context.old_token_ids
        if ids.ndim != 2 or ids.shape[1] != length:
            raise ValueError("old integer canvas must have shape [batch, length]")
        if ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("old canvas must contain integer token ids")
        batch = ids.shape[0]
        if tasks.shape != (batch, 9) or not bool(torch.isfinite(tasks).all()):
            raise ValueError("the unchanged legacy task input must be finite [batch, 9]")
        if context.prompt_lengths.shape != (batch,) or context.num_sites.shape != (batch,):
            raise ValueError("one prompt length and site count is required per crystal")
        if context.active_token_mask.shape != ids.shape:
            raise ValueError("active token mask must match the old canvas")
        if geometry["fractional"].ndim != 3 or geometry["fractional"].shape[::2] != (batch, 3):
            raise ValueError("fractional coordinates must have shape [batch, sites, 3]")
        sites = geometry["fractional"].shape[1]
        if sites < 1 or bool(((context.num_sites < 1)
                             | (context.num_sites > min(sites, self.config.max_sites))).any()):
            raise ValueError("site count exceeds the declared padded geometry")
        if bool(((context.prompt_lengths < 0)
                 | (context.prompt_lengths + 7 + 4 * context.num_sites > length)).any()):
            raise ValueError("crystal body lies outside the token canvas")
        if geometry["lattice"].shape != (batch, 3, 3):
            raise ValueError("lattice must have shape [batch, 3, 3]")
        if geometry["lattice_known"].shape != (batch,) or geometry["lattice_known"].dtype != torch.bool:
            raise ValueError("lattice-known mask must be boolean [batch]")
        for name in ("site_known", "active_sites", "species", "program_rank"):
            if geometry[name].shape != (batch, sites):
                raise ValueError(f"{name} must have shape [batch, sites]")
        if context.program_rank.shape != (batch, sites):
            raise ValueError("context program ranks must match padded geometry")
        if geometry["site_known"].dtype != torch.bool or geometry["active_sites"].dtype != torch.bool:
            raise ValueError("site-known and active-site masks must be boolean")
        if any(value.device != ids.device for value in geometry.values() if isinstance(value, torch.Tensor)):
            raise ValueError("geometry must share the old canvas device")
        return batch, sites

    def geometry_features(self, geometry: Mapping[str, torch.Tensor], context: Any,
                          tasks: torch.Tensor) -> dict[str, torch.Tensor]:
        """Build observable features afresh; returned masks also support diagnostics."""
        batch, sites = self._validate(geometry, context, context.old_token_ids.shape[1], tasks)
        device = context.old_token_ids.device
        with torch.autocast(device_type=device.type, enabled=False):
            metadata = torch.cat((tasks.float(), numeric_noise_component_features(context)), -1)
            # These are decoded discrete observations, never differentiable model
            # outputs. FP64 is confined to the tiny log-SPD calculation to avoid
            # loss of positive definiteness from forming an anisotropic FP32 Gram.
            with torch.no_grad():
                present = torch.arange(sites, device=device)[None] < context.num_sites[:, None]
                cell_known = geometry["lattice_known"]
                known = geometry["site_known"] & present & cell_known[:, None]
                raw_lattice = geometry["lattice"].detach().float()
                raw_frac = geometry["fractional"].detach().float()
                if not bool(torch.isfinite(raw_lattice[cell_known]).all()):
                    raise ValueError("a known lattice must be finite")
                if not bool(torch.isfinite(raw_frac[known]).all()):
                    raise ValueError("a known site must have finite coordinates")
                identity = torch.eye(3, dtype=torch.float32, device=device)[None]
                lattice = torch.where(cell_known[:, None, None], raw_lattice, identity)
                frac = torch.where(known[..., None], raw_frac, 0.).remainder(1.)
                sign, log_volume = torch.linalg.slogdet(lattice.double())
                gram = lattice.double() @ lattice.double().transpose(-1, -2)
                values, vectors = torch.linalg.eigh(gram)
                if bool(((sign <= 0) | (values <= 0).any(-1)).logical_and(cell_known).any()):
                    raise ValueError("known lattice must define a positive oriented SPD cell")
                log_gram = (vectors * values.clamp_min(torch.finfo(torch.float64).tiny).log()[:, None, :]) @ vectors.transpose(-1, -2)
                log_shape = log_gram - torch.diag_embed(log_gram.diagonal(dim1=-2, dim2=-1).mean(-1)[:, None].expand(-1, 3))
                log_vpa = log_volume - context.num_sites.double().log()
                known_fraction = known.sum(-1).float() / context.num_sites.float()
                cell_features = torch.cat((
                    log_vpa[:, None].float(), _symmetric_components(log_shape).float(),
                    (context.num_sites.float() / self.config.max_sites)[:, None],
                    known_fraction[:, None],
                ), -1)
                cell_features = torch.where(cell_known[:, None], cell_features, 0.)

                delta = frac[:, :, None] - frac[:, None, :]
                delta = delta - delta.round()
                cart = torch.einsum("bijsc,bcd->bijsd", delta[..., None, :] + self.shifts, lattice)
                distances = torch.linalg.vector_norm(cart, dim=-1).amin(-1)
                scale = (log_volume / 3.).exp().float()
                relative = distances / scale[:, None, None]
                radial_width = self.config.radial_cutoff_A / max(self.config.radial_bins - 1, 1)
                radial = torch.exp(-((distances[..., None] - self.radial_centers) / radial_width).square())
                phase = (2. * math.pi * delta[..., None] * self.fourier_frequencies)
                periodic = torch.cat((phase.sin().flatten(-2), phase.cos().flatten(-2)), -1)
                pair_known = known[:, :, None] & known[:, None, :]
                pair_known &= ~torch.eye(sites, dtype=torch.bool, device=device)[None]
                neighbors = pair_known.sum(-1)
                active = geometry["active_sites"] & present
                rank = torch.where(present, geometry["program_rank"], 0).float() / max(self.config.max_sites - 1, 1)
                flags = torch.stack((
                    known[:, :, None].expand(-1, -1, sites),
                    known[:, None, :].expand(-1, sites, -1),
                    active[:, :, None].expand(-1, -1, sites),
                    active[:, None, :].expand(-1, sites, -1),
                    rank[:, :, None].expand(-1, -1, sites),
                    rank[:, None, :].expand(-1, sites, -1),
                    rank[:, :, None] - rank[:, None, :],
                ), -1).float()
                species_ids = torch.where(present, geometry["species"], 0).long()
                if bool(((species_ids < 1) | (species_ids > 118)).logical_and(known).any()):
                    raise ValueError("known sites require a supported species id")
                species_ids = torch.where(known, species_ids, 0)

            # Learned species and encoders remain in the autograd graph.
            species = self.species(species_ids)
            pair = torch.cat((
                radial, distances.log1p()[..., None], relative[..., None], periodic,
                species[:, :, None].expand(-1, -1, sites, -1),
                species[:, None, :].expand(-1, sites, -1, -1),
                cell_features[:, None, None].expand(-1, sites, sites, -1), flags,
                metadata[:, None, None].expand(-1, sites, sites, -1),
            ), -1)
            pair = torch.where(pair_known[..., None], pair, 0.)
            return {
                "pair_features": pair, "pair_known": pair_known,
                "site_known": known, "cell_known": cell_known,
                "neighbor_count": neighbors, "cell_features": cell_features,
                "species_features": species, "active_sites": active,
                "program_rank": rank, "metadata": metadata,
            }

    def forward(self, geometry, context, length, tasks):
        batch, sites = self._validate(geometry, context, length, tasks)
        device = context.old_token_ids.device
        with torch.autocast(device_type=device.type, enabled=False):
            legacy = self.legacy(geometry, context, length, tasks.float())
            if legacy.shape != (batch, 1, length, length):
                raise ValueError("legacy bias must have shape [batch, 1, length, length]")
            features = self.geometry_features(geometry, context, tasks)
            known, pair_known = features["site_known"], features["pair_known"]
            encoded = self.pair_encoder(features["pair_features"])
            encoded = torch.where(pair_known[..., None], encoded, 0.)
            site_bias = self.site_to_site(encoded)
            neighbors = features["neighbor_count"]
            pooled = encoded.sum(2) / neighbors.clamp_min(1)[..., None]
            # With zero neighbors pooled is exactly zero, not a learned bias from
            # a fabricated pair. Real one-site cell descriptors remain available.
            site_flags = torch.stack((
                known.float(), features["active_sites"].float(), features["program_rank"],
                neighbors.float() / self.config.max_sites, (neighbors > 0).float(),
            ), -1)
            local = torch.cat((
                pooled, features["species_features"],
                features["cell_features"][:, None].expand(-1, sites, -1), site_flags,
                features["metadata"][:, None].expand(-1, sites, -1),
            ), -1)
            local = torch.where(known[..., None], local, 0.)
            cell_positions = context.prompt_lengths[:, None] + torch.arange(1, 7, device=device)[None]
            cell_active = context.active_token_mask.gather(1, cell_positions).float()
            slot_features = self.cell_slots(torch.arange(6, device=device))
            cell_input = torch.cat((
                local[:, None].expand(-1, 6, -1, -1),
                slot_features[None, :, None].expand(batch, -1, sites, -1),
                cell_active[:, :, None, None].expand(-1, -1, sites, -1),
            ), -1)
            cell_encoded = self.cell_site_encoder(cell_input)
            cell_to_site = torch.where(known[:, None, :, None], self.cell_to_site(cell_encoded), 0.)
            site_to_cell = torch.where(known[:, None, :, None], self.site_to_cell(cell_encoded), 0.)

            # Build a compact cell-slot/site matrix and lift it once to tokens;
            # avoid materializing three independent [B,H,L,L] residuals.
            top = torch.cat((cell_to_site.new_zeros(batch, 6, 6, self.config.heads), cell_to_site), 2)
            bottom = torch.cat((site_to_cell.transpose(1, 2), site_bias), 2)
            nodes = torch.cat((top, bottom), 1)
            gate = 2. * torch.sigmoid(self.noise_gate(features["metadata"]))
            nodes = nodes * gate[:, None, None]
            relative = torch.arange(length, device=device)[None] - context.prompt_lengths[:, None]
            cell_token = (relative >= 1) & (relative <= 6)
            site_token = (relative >= 7) & (relative < 7 + 4 * context.num_sites[:, None])
            site_index = torch.div(relative - 7, 4, rounding_mode="floor").clamp(0, sites - 1)
            node_index = torch.where(cell_token, relative - 1, 6 + site_index)
            rows = torch.arange(batch, device=device)[:, None, None]
            residual = nodes[rows, node_index[:, :, None], node_index[:, None, :]]
            valid_token = cell_token | site_token
            residual = torch.where((valid_token[:, :, None] & valid_token[:, None, :])[..., None], residual, 0.)
            return legacy.expand(-1, self.config.heads, -1, -1) + residual.permute(0, 3, 1, 2)


__all__ = [
    "PeriodicV2Config", "PeriodicV2AttentionBias", "numeric_noise_component_features",
]
