"""Native-PyTorch periodic CSP + dynamic orbit-set co-denoiser.

This module intentionally has no torch-geometric or torch-scatter dependency.
All graph and set reductions use native ``index_add_`` operations.
"""

from __future__ import annotations

import dataclasses
import enum
import math
from typing import NamedTuple

import torch
from torch import Tensor, nn


PERIODIC_COORDINATE_SCALE_MIN = 0.02
PERIODIC_COORDINATE_SCALE_MAX = 0.5


class WQVariant(str, enum.Enum):
    ATOM_JOINT = "B-ATOM-JOINT"
    AR = "B-WQ-AR"
    D3PM = "B-WQ-D3PM"
    DLM_MONO = "B-WQ-DLM-MONO"
    JOINT_NOREV = "B-WQ-JOINT-NOREV"
    DISC_ONCE = "B-WQ-DISC-ONCE"
    STRAT_CONF = "M-WQ-STRAT-CONF"
    STRAT_GEO = "M-WQ-STRAT-GEO"

    @property
    def causal(self) -> bool:
        return self is WQVariant.AR


@dataclasses.dataclass(frozen=True, slots=True)
class WQModelConfig:
    hidden_dim: int = 256
    csp_layers: int = 6
    set_layers: int = 4
    attention_heads: int = 8
    ffn_dim: int = 1024
    time_dim: int = 256
    cutoff: float = 7.0
    max_atoms: int = 20
    species_count: int = 89
    space_group_count: int = 230
    # 26 lower-case ITA letters plus SG 47's upper-case ``A`` position.
    wyckoff_type_count: int = 27
    event_type_count: int = 5
    radial_frequencies: int = 16

    def __post_init__(self) -> None:
        if self.hidden_dim != 256 or self.csp_layers != 6 or self.set_layers != 4:
            raise ValueError("registered architecture is hidden=256, CSP=6, set=4")
        if self.attention_heads != 8 or self.ffn_dim != 1024 or self.time_dim != 256:
            raise ValueError("registered attention/time architecture changed")
        if self.max_atoms != 20:
            raise ValueError("MP20 max_atoms must remain 20")


@dataclasses.dataclass(slots=True)
class WQTensorBatch:
    """Ragged semantic batch with no fixed orbit canvas."""

    atom_species: Tensor
    frac_coords: Tensor
    lattices: Tensor
    atom_batch: Tensor
    atom_to_orbit: Tensor
    orbit_species: Tensor
    orbit_wyckoff: Tensor
    orbit_batch: Tensor
    space_group: Tensor
    time: Tensor
    geometry_evidence: Tensor

    def to(self, device: torch.device | str) -> "WQTensorBatch":
        return WQTensorBatch(
            **{
                field.name: getattr(self, field.name).to(device)
                for field in dataclasses.fields(self)
            }
        )

    def validate(self, config: WQModelConfig) -> None:
        atom_count = int(self.atom_species.numel())
        orbit_count = int(self.orbit_species.numel())
        graph_count = int(self.space_group.numel())
        if self.frac_coords.shape != (atom_count, 3):
            raise ValueError("frac_coords shape mismatch")
        if self.lattices.shape != (graph_count, 3, 3):
            raise ValueError("lattices shape mismatch")
        if self.atom_batch.shape != (atom_count,) or self.atom_to_orbit.shape != (atom_count,):
            raise ValueError("atom mappings shape mismatch")
        if self.orbit_wyckoff.shape != (orbit_count,) or self.orbit_batch.shape != (orbit_count,):
            raise ValueError("orbit mappings shape mismatch")
        if self.time.shape != (graph_count,):
            raise ValueError("time shape mismatch")
        if self.geometry_evidence.shape != (orbit_count, 6):
            raise ValueError("geometry evidence must be [num_orbits, 6]")
        if atom_count == 0 or orbit_count == 0 or graph_count == 0:
            raise ValueError("semantic batches cannot be empty")
        counts = torch.bincount(self.atom_batch, minlength=graph_count)
        if bool(torch.any(counts < 1)) or bool(torch.any(counts > config.max_atoms)):
            raise ValueError("each MP20 graph must contain 1-20 expanded atoms")
        if int(self.atom_to_orbit.min()) < 0 or int(self.atom_to_orbit.max()) >= orbit_count:
            raise ValueError("atom_to_orbit index out of range")


class WQModelOutput(NamedTuple):
    space_group_logits: Tensor
    species_logits: Tensor
    wyckoff_logits: Tensor
    event_logits: Tensor
    event_orbit_logits: Tensor
    birth_species_logits: Tensor
    birth_wyckoff_logits: Tensor
    birth_coordinate_mean: Tensor
    birth_coordinate_log_scale: Tensor
    revision_logits: Tensor
    atom_coordinate_score: Tensor
    lattice_score: Tensor
    bridge_mean: Tensor
    bridge_log_scale: Tensor
    orbit_features: Tensor


class WQPriorOutput(NamedTuple):
    """Outputs needed before a non-empty semantic orbit set exists.

    The registered process first samples a space group from ``MASK`` and then
    commits it.  A second, SG-conditioned call samples the mandatory first
    orbit and a lattice-chart base point.  Keeping this path inside the same
    model avoids an empirical-data initializer and makes generation genuinely
    unconditional.
    """

    space_group_logits: Tensor
    first_species_logits: Tensor
    first_wyckoff_logits: Tensor
    first_coordinate_mean: Tensor
    first_coordinate_log_scale: Tensor
    lattice_chart_mean: Tensor
    lattice_chart_log_scale: Tensor


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        if dimension % 2:
            raise ValueError("time embedding dimension must be even")
        self.dimension = dimension

    def forward(self, time: Tensor) -> Tensor:
        half = self.dimension // 2
        frequencies = torch.exp(
            torch.arange(half, device=time.device, dtype=time.dtype)
            * (-math.log(10000.0) / max(half - 1, 1))
        )
        phase = time[:, None] * frequencies[None, :]
        return torch.cat([phase.sin(), phase.cos()], dim=-1)


def _index_mean(values: Tensor, index: Tensor, count: int) -> Tensor:
    output = values.new_zeros((count, values.shape[-1]))
    output.index_add_(0, index, values)
    denominator = torch.bincount(index, minlength=count).clamp_min(1).to(values.dtype)
    return output / denominator[:, None]


def _bounded_periodic_log_scale(raw: Tensor) -> Tensor:
    """Map an unconstrained head to an identifiable wrapped-normal scale.

    On a unit torus, scales much larger than one half are indistinguishable
    from a nearly uniform law, while an unconstrained Gaussian likelihood can
    collapse its variance and turn a rare cross-boundary target into an
    arbitrarily large gradient.  A smooth sigmoid keeps the scale inside the
    registered interval without saturating the head at initialization.
    """

    values = raw.float().sigmoid()
    scale = PERIODIC_COORDINATE_SCALE_MIN + (
        PERIODIC_COORDINATE_SCALE_MAX - PERIODIC_COORDINATE_SCALE_MIN
    ) * values
    return scale.log()


def fully_connected_periodic_edges(atom_batch: Tensor) -> tuple[Tensor, Tensor]:
    sources: list[Tensor] = []
    targets: list[Tensor] = []
    graph_count = int(atom_batch.max().item()) + 1
    for graph in range(graph_count):
        indices = torch.nonzero(atom_batch == graph, as_tuple=False).flatten()
        count = int(indices.numel())
        if count <= 1:
            continue
        source = indices.repeat_interleave(count)
        target = indices.repeat(count)
        keep = source != target
        sources.append(source[keep])
        targets.append(target[keep])
    if not sources:
        empty = atom_batch.new_empty((0,))
        return empty, empty
    return torch.cat(sources), torch.cat(targets)


class PeriodicMessageLayer(nn.Module):
    def __init__(self, config: WQModelConfig) -> None:
        super().__init__()
        radial_dim = 1 + 2 * config.radial_frequencies
        self.cutoff = config.cutoff
        self.radial_frequencies = config.radial_frequencies
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * config.hidden_dim + radial_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        self.node_mlp = nn.Sequential(
            nn.LayerNorm(2 * config.hidden_dim),
            nn.Linear(2 * config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )

    def forward(
        self,
        features: Tensor,
        frac_coords: Tensor,
        lattices: Tensor,
        atom_batch: Tensor,
    ) -> Tensor:
        source, target = fully_connected_periodic_edges(atom_batch)
        if source.numel() == 0:
            aggregate = torch.zeros_like(features)
        else:
            delta = frac_coords[source] - frac_coords[target]
            delta = torch.remainder(delta + 0.5, 1.0) - 0.5
            edge_graph = atom_batch[target]
            cartesian = torch.einsum("ei,eij->ej", delta, lattices[edge_graph])
            distance = torch.linalg.vector_norm(cartesian, dim=-1)
            scaled = distance / self.cutoff
            frequencies = torch.arange(
                1,
                self.radial_frequencies + 1,
                device=features.device,
                dtype=features.dtype,
            )
            phase = math.pi * scaled[:, None] * frequencies[None, :]
            radial = torch.cat([scaled[:, None], phase.sin(), phase.cos()], dim=-1)
            messages = self.edge_mlp(
                torch.cat([features[source], features[target], radial], dim=-1)
            )
            cutoff_weight = 0.5 * (torch.cos(math.pi * scaled.clamp(max=1.0)) + 1.0)
            messages = messages * cutoff_weight[:, None] * (scaled <= 1.0)[:, None]
            # CUDA BF16 autocast deliberately leaves the geometric distance
            # path in FP32.  Multiplying the BF16 edge-MLP result by that
            # cutoff therefore promotes ``messages`` to FP32.  Accumulate in
            # the promoted dtype (which is also numerically preferable), then
            # return to the node-feature dtype before the residual MLP.  An
            # in-place index_add_ requires source and destination to match.
            aggregate = messages.new_zeros(features.shape)
            aggregate.index_add_(0, target, messages)
            degree = torch.bincount(target, minlength=features.shape[0]).clamp_min(1)
            aggregate = (aggregate / degree[:, None].to(aggregate.dtype)).to(
                features.dtype
            )
        return features + self.node_mlp(torch.cat([features, aggregate], dim=-1))


class RaggedSetBlock(nn.Module):
    """Masked batched self-attention over a semantically ragged orbit set.

    A transient dense tensor is used only inside the attention kernel.  Padded
    keys are masked and padded queries are discarded before any head, loss,
    probability, or serialized output is formed.  This preserves the ragged
    protocol while avoiding one GPU kernel launch per crystal and layer.
    """

    def __init__(self, config: WQModelConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_dim)
        self.attention = nn.MultiheadAttention(
            config.hidden_dim,
            config.attention_heads,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(config.hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_dim, config.ffn_dim),
            nn.SiLU(),
            nn.Linear(config.ffn_dim, config.hidden_dim),
        )

    def forward(self, features: Tensor, orbit_batch: Tensor, *, causal: bool) -> Tensor:
        graph_count = int(orbit_batch.max().item()) + 1
        counts = torch.bincount(orbit_batch, minlength=graph_count)
        maximum = int(counts.max().item())
        padded = features.new_zeros((graph_count, maximum, features.shape[-1]))
        key_padding_mask = torch.ones(
            (graph_count, maximum), dtype=torch.bool, device=features.device
        )
        indices_by_graph: list[Tensor] = []
        for graph in range(graph_count):
            indices = torch.nonzero(orbit_batch == graph, as_tuple=False).flatten()
            indices_by_graph.append(indices)
            count = int(indices.numel())
            padded[graph, :count] = features[indices]
            key_padding_mask[graph, :count] = False
        normalized = self.norm1(padded)
        attention_mask = None
        if causal:
            attention_mask = torch.triu(
                torch.ones(
                    (maximum, maximum),
                    dtype=torch.bool,
                    device=features.device,
                ),
                diagonal=1,
            )
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=key_padding_mask,
            attn_mask=attention_mask,
            need_weights=False,
        )
        values = padded + attended
        values = values + self.ffn(self.norm2(values))
        output = torch.empty_like(features)
        for graph, indices in enumerate(indices_by_graph):
            output[indices] = values[graph, : int(indices.numel())]
        return output


class WQCoDenoiser(nn.Module):
    def __init__(self, config: WQModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or WQModelConfig()
        config = self.config
        # Index 0 is MASK; atomic number/type/SG IDs are shifted as documented
        # by the tensorizer.
        self.atom_embedding = nn.Embedding(config.species_count + 1, config.hidden_dim)
        self.orbit_species_embedding = nn.Embedding(config.species_count + 1, config.hidden_dim)
        self.wyckoff_embedding = nn.Embedding(config.wyckoff_type_count + 1, config.hidden_dim)
        self.space_group_embedding = nn.Embedding(config.space_group_count + 1, config.hidden_dim)
        self.time_embedding = SinusoidalTimeEmbedding(config.time_dim)
        self.time_projection = nn.Linear(config.time_dim, config.hidden_dim)
        self.atom_input = nn.Sequential(nn.LayerNorm(config.hidden_dim), nn.Linear(config.hidden_dim, config.hidden_dim))
        self.csp = nn.ModuleList(PeriodicMessageLayer(config) for _ in range(config.csp_layers))
        self.evidence_projection = nn.Sequential(nn.Linear(6, config.hidden_dim), nn.SiLU())
        self.set_blocks = nn.ModuleList(RaggedSetBlock(config) for _ in range(config.set_layers))
        self.graph_norm = nn.LayerNorm(config.hidden_dim)
        self.space_group_head = nn.Linear(config.hidden_dim, config.space_group_count)
        self.species_head = nn.Linear(config.hidden_dim, config.species_count)
        self.wyckoff_head = nn.Linear(config.hidden_dim, config.wyckoff_type_count)
        self.event_head = nn.Linear(config.hidden_dim, config.event_type_count)
        self.event_orbit_head = nn.Linear(config.hidden_dim, 1)
        self.birth_species_head = nn.Linear(config.hidden_dim, config.species_count)
        self.birth_wyckoff_head = nn.Linear(config.hidden_dim, config.wyckoff_type_count)
        self.birth_coordinate_mean_head = nn.Linear(config.hidden_dim, 3)
        self.birth_coordinate_log_scale_head = nn.Linear(config.hidden_dim, 3)
        self.revision_head = nn.Linear(config.hidden_dim, 3)
        self.coordinate_head = nn.Linear(config.hidden_dim, 3, bias=False)
        self.lattice_head = nn.Linear(config.hidden_dim, 6)
        self.bridge_mean_head = nn.Linear(config.hidden_dim, 3)
        self.bridge_log_scale_head = nn.Linear(config.hidden_dim, 3)
        self.start_token = nn.Parameter(torch.zeros(config.hidden_dim))
        self.prior_mlp = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.ffn_dim),
            nn.SiLU(),
            nn.Linear(config.ffn_dim, config.hidden_dim),
        )
        self.prior_space_group_head = nn.Linear(config.hidden_dim, config.space_group_count)
        self.prior_species_head = nn.Linear(config.hidden_dim, config.species_count)
        self.prior_wyckoff_head = nn.Linear(config.hidden_dim, config.wyckoff_type_count)
        self.prior_coordinate_mean_head = nn.Linear(config.hidden_dim, 3)
        self.prior_coordinate_log_scale_head = nn.Linear(config.hidden_dim, 3)
        self.prior_lattice_mean_head = nn.Linear(config.hidden_dim, 6)
        self.prior_lattice_log_scale_head = nn.Linear(config.hidden_dim, 6)

    def forward_prior(self, time: Tensor, space_group: Tensor) -> WQPriorOutput:
        """Evaluate the empty-set prior.

        ``space_group`` uses the same convention as :class:`WQTensorBatch`:
        zero is ``MASK`` and committed groups are 1--230.  Callers use a
        masked input for the SG logits and a committed input for all remaining
        prior fields.
        """

        if time.ndim != 1 or space_group.shape != time.shape:
            raise ValueError("prior time and space_group must be matching vectors")
        if time.numel() == 0:
            raise ValueError("prior batch cannot be empty")
        if bool(torch.any(space_group < 0)) or bool(
            torch.any(space_group > self.config.space_group_count)
        ):
            raise ValueError("prior space_group input is outside [MASK,230]")
        features = (
            self.start_token[None, :]
            + self.time_projection(self.time_embedding(time))
            + self.space_group_embedding(space_group)
        )
        features = features + self.prior_mlp(features)
        return WQPriorOutput(
            space_group_logits=self.prior_space_group_head(features),
            first_species_logits=self.prior_species_head(features),
            first_wyckoff_logits=self.prior_wyckoff_head(features),
            first_coordinate_mean=self.prior_coordinate_mean_head(features),
            first_coordinate_log_scale=_bounded_periodic_log_scale(
                self.prior_coordinate_log_scale_head(features)
            ),
            lattice_chart_mean=self.prior_lattice_mean_head(features),
            lattice_chart_log_scale=self.prior_lattice_log_scale_head(features).clamp(
                -8.0, 4.0
            ),
        )

    def forward(self, batch: WQTensorBatch, *, variant: WQVariant = WQVariant.STRAT_GEO) -> WQModelOutput:
        batch.validate(self.config)
        graph_count = int(batch.space_group.numel())
        orbit_count = int(batch.orbit_species.numel())
        time = self.time_projection(self.time_embedding(batch.time))
        sg = self.space_group_embedding(batch.space_group)
        atom_features = self.atom_embedding(batch.atom_species)
        atom_features = atom_features + time[batch.atom_batch] + sg[batch.atom_batch]
        atom_features = self.atom_input(atom_features)
        for layer in self.csp:
            atom_features = layer(
                atom_features,
                batch.frac_coords,
                batch.lattices,
                batch.atom_batch,
            )
        atom_coordinate_score = self.coordinate_head(atom_features)
        evidence = batch.geometry_evidence
        if variant is WQVariant.STRAT_GEO:
            # The score-norm evidence must be available identically in
            # training and sampling, without using the clean score target or a
            # previous-step teacher signal.  The coordinate head is upstream
            # of all geometry evidence, so its current detached orbit RMS is
            # causal and cannot create a circular shortcut.  Log compression
            # keeps this channel on the same bounded scale as the other five.
            squared_norm = atom_coordinate_score.detach().float().square().sum(
                dim=-1, keepdim=True
            )
            orbit_rms = torch.sqrt(
                _index_mean(squared_norm, batch.atom_to_orbit, orbit_count).clamp_min(0.0)
            )
            bounded_norm = (
                torch.log1p(orbit_rms) / math.log1p(200.0)
            ).clamp(0.0, 1.0)
            evidence = torch.cat(
                (evidence[:, :4], bounded_norm.to(evidence.dtype), evidence[:, 5:]),
                dim=-1,
            )
        orbit_features = _index_mean(atom_features, batch.atom_to_orbit, orbit_count)
        orbit_features = (
            orbit_features
            + self.orbit_species_embedding(batch.orbit_species)
            + self.wyckoff_embedding(batch.orbit_wyckoff)
            + sg[batch.orbit_batch]
            + time[batch.orbit_batch]
            + self.evidence_projection(evidence)
        )
        for block in self.set_blocks:
            orbit_features = block(orbit_features, batch.orbit_batch, causal=variant.causal)
        graph_features = self.graph_norm(_index_mean(orbit_features, batch.orbit_batch, graph_count))
        bridge_log_scale = _bounded_periodic_log_scale(
            self.bridge_log_scale_head(orbit_features)
        )
        birth_coordinate_log_scale = _bounded_periodic_log_scale(
            self.birth_coordinate_log_scale_head(graph_features)
        )
        return WQModelOutput(
            space_group_logits=self.space_group_head(graph_features),
            species_logits=self.species_head(orbit_features),
            wyckoff_logits=self.wyckoff_head(orbit_features),
            event_logits=self.event_head(graph_features),
            event_orbit_logits=self.event_orbit_head(orbit_features).squeeze(-1),
            birth_species_logits=self.birth_species_head(graph_features),
            birth_wyckoff_logits=self.birth_wyckoff_head(graph_features),
            birth_coordinate_mean=self.birth_coordinate_mean_head(graph_features),
            birth_coordinate_log_scale=birth_coordinate_log_scale,
            revision_logits=self.revision_head(orbit_features),
            atom_coordinate_score=atom_coordinate_score,
            lattice_score=self.lattice_head(graph_features),
            bridge_mean=self.bridge_mean_head(orbit_features),
            bridge_log_scale=bridge_log_scale,
            orbit_features=orbit_features,
        )

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
