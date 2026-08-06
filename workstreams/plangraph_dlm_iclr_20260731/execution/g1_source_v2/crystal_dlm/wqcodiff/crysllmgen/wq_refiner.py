"""Wyckoff quotient heads on the registered CrysLLMGen CSPDiffusion backbone."""

from __future__ import annotations

import dataclasses
import hashlib
import math
import sys
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from ..model import WQModelOutput, WQTensorBatch
from ..vocabulary import MP20_ATOMIC_NUMBERS
from .schedules import (
    OFFICIAL_REVERSE_START_TIMESTEP,
    PARENT_RUN_TYPE,
    PARENT_SCHEDULER_TIMESTEPS,
)


@dataclasses.dataclass(frozen=True, slots=True)
class CrysLLMGenWQRefinerConfig:
    backbone_hidden_dim: int = 512
    set_layers: int = 4
    attention_heads: int = 8
    ffn_dim: int = 1024
    species_count: int = 89
    wyckoff_type_count: int = 27
    space_group_count: int = 230
    event_type_count: int = 5
    max_atoms: int = 20
    geometry_evidence_dim: int = 6

    def __post_init__(self) -> None:
        if (
            self.backbone_hidden_dim,
            self.set_layers,
            self.attention_heads,
            self.ffn_dim,
        ) != (512, 4, 8, 1024):
            raise ValueError("registered CrysLLMGen WQ architecture changed")
        if self.max_atoms != 20 or self.species_count != 89:
            raise ValueError("registered MP20 support changed")


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _index_mean(values: Tensor, index: Tensor, count: int) -> Tensor:
    result = values.new_zeros((count, values.shape[-1]))
    result.index_add_(0, index, values)
    denominator = torch.bincount(index, minlength=count).clamp_min(1).to(values.dtype)
    return result / denominator[:, None]


def _periodic_log_scale(raw: Tensor) -> Tensor:
    minimum, maximum = 0.02, 0.5
    scale = minimum + (maximum - minimum) * raw.float().sigmoid()
    return scale.log()


class _RaggedSetBlock(nn.Module):
    def __init__(self, config: CrysLLMGenWQRefinerConfig) -> None:
        super().__init__()
        hidden = config.backbone_hidden_dim
        self.norm1 = nn.LayerNorm(hidden)
        self.attention = nn.MultiheadAttention(
            hidden,
            config.attention_heads,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, config.ffn_dim),
            nn.SiLU(),
            nn.Linear(config.ffn_dim, hidden),
        )

    def forward(self, values: Tensor, orbit_batch: Tensor) -> Tensor:
        graph_count = int(orbit_batch.max().item()) + 1
        counts = torch.bincount(orbit_batch, minlength=graph_count)
        maximum = int(counts.max().item())
        padded = values.new_zeros((graph_count, maximum, values.shape[-1]))
        padding = torch.ones(
            (graph_count, maximum),
            dtype=torch.bool,
            device=values.device,
        )
        groups: list[Tensor] = []
        for graph in range(graph_count):
            indices = torch.nonzero(orbit_batch == graph, as_tuple=False).flatten()
            groups.append(indices)
            count = int(indices.numel())
            padded[graph, :count] = values[indices]
            padding[graph, :count] = False
        normalized = self.norm1(padded)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=padding,
            need_weights=False,
        )
        updated = padded + attended
        updated = updated + self.ffn(self.norm2(updated))
        output = torch.empty_like(values)
        for graph, indices in enumerate(groups):
            output[indices] = updated[graph, : int(indices.numel())]
        return output


class CrysLLMGenWQRefiner(nn.Module):
    """Preserve the parent CSP geometry path and add equivariant orbit heads."""

    def __init__(
        self,
        *,
        decoder: nn.Module,
        time_embedding: nn.Module,
        config: CrysLLMGenWQRefinerConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or CrysLLMGenWQRefinerConfig()
        self.decoder = decoder
        self.time_embedding = time_embedding
        if not hasattr(decoder, "final_layer_norm"):
            raise ValueError("registered CSP decoder must expose final_layer_norm")
        self._captured_node_features: Tensor | None = None
        self._feature_hook = decoder.final_layer_norm.register_forward_hook(
            self._capture_features
        )
        hidden = self.config.backbone_hidden_dim
        self.orbit_species_embedding = nn.Embedding(
            self.config.species_count + 1, hidden
        )
        self.wyckoff_embedding = nn.Embedding(
            self.config.wyckoff_type_count + 1, hidden
        )
        self.space_group_embedding = nn.Embedding(
            self.config.space_group_count + 1, hidden
        )
        self.evidence_projection = nn.Sequential(
            nn.Linear(self.config.geometry_evidence_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.set_blocks = nn.ModuleList(
            _RaggedSetBlock(self.config) for _ in range(self.config.set_layers)
        )
        self.graph_norm = nn.LayerNorm(hidden)
        self.space_group_head = nn.Linear(hidden, self.config.space_group_count)
        self.species_head = nn.Linear(hidden, self.config.species_count)
        self.wyckoff_head = nn.Linear(hidden, self.config.wyckoff_type_count)
        self.event_head = nn.Linear(hidden, self.config.event_type_count)
        self.event_orbit_head = nn.Linear(hidden, 1)
        self.birth_species_head = nn.Linear(hidden, self.config.species_count)
        self.birth_wyckoff_head = nn.Linear(hidden, self.config.wyckoff_type_count)
        self.birth_coordinate_mean_head = nn.Linear(hidden, 3)
        self.birth_coordinate_log_scale_head = nn.Linear(hidden, 3)
        self.revision_head = nn.Linear(hidden, 3)
        self.bridge_mean_head = nn.Linear(hidden, 3)
        self.bridge_log_scale_head = nn.Linear(hidden, 3)
        self.lattice_chart_head = nn.Linear(hidden, 6)
        self.parent_lattice_projection = nn.Linear(9, 6, bias=False)
        nn.init.zeros_(self.parent_lattice_projection.weight)
        with torch.no_grad():
            # A deterministic residual initialization: diagonal and lower
            # triangle parent scores seed the six registered chart channels.
            for output, source in enumerate((0, 4, 8, 3, 6, 7)):
                self.parent_lattice_projection.weight[output, source] = 1.0
        atomic_lookup = torch.tensor(MP20_ATOMIC_NUMBERS, dtype=torch.long)
        self.register_buffer("atomic_number_lookup", atomic_lookup, persistent=True)

    def _capture_features(
        self,
        _module: nn.Module,
        _inputs: tuple[Any, ...],
        output: Tensor,
    ) -> None:
        self._captured_node_features = output

    def set_inherited_backbone_trainable(self, enabled: bool) -> None:
        for parameter in self.decoder.parameters():
            parameter.requires_grad_(bool(enabled))

    def inherited_parameter_count(self) -> int:
        return sum(value.numel() for value in self.decoder.parameters())

    def parameter_count(self) -> int:
        return sum(value.numel() for value in self.parameters())

    def forward(
        self,
        batch: WQTensorBatch,
        *,
        use_geometry_evidence: bool = True,
    ) -> WQModelOutput:
        batch.validate(
            # Validation depends only on these frozen fields, and using the
            # registered legacy config avoids a parallel ragged-batch schema.
            __import__(
                "crystal_dlm.wqcodiff.model", fromlist=["WQModelConfig"]
            ).WQModelConfig()
        )
        if bool(torch.any(batch.space_group < 1)):
            raise ValueError("CrysLLMGen handoff requires a committed space group")
        if bool(torch.any(batch.atom_species < 1)) or bool(
            torch.any(batch.orbit_species < 1)
        ):
            raise ValueError("CrysLLMGen WQ refiner never consumes MASK species")
        atom_types = self.atomic_number_lookup[batch.atom_species - 1]
        timesteps = (
            (batch.time.float() * (OFFICIAL_REVERSE_START_TIMESTEP - 1))
            .round()
            .long()
            .clamp(1, OFFICIAL_REVERSE_START_TIMESTEP)
        )
        time_features = self.time_embedding(timesteps)
        self._captured_node_features = None
        parent_lattice, coordinate_score = self.decoder(
            time_features,
            atom_types,
            batch.frac_coords,
            batch.lattices,
            torch.bincount(
                batch.atom_batch,
                minlength=int(batch.space_group.numel()),
            ),
            batch.atom_batch,
        )
        node_features = self._captured_node_features
        if node_features is None or node_features.shape != (
            batch.atom_species.numel(),
            self.config.backbone_hidden_dim,
        ):
            raise RuntimeError("registered CSP feature hook did not fire exactly once")
        orbit_count = int(batch.orbit_species.numel())
        graph_count = int(batch.space_group.numel())
        orbit_features = _index_mean(node_features, batch.atom_to_orbit, orbit_count)
        evidence = batch.geometry_evidence if use_geometry_evidence else torch.zeros_like(
            batch.geometry_evidence
        )
        if use_geometry_evidence:
            squared = coordinate_score.detach().float().square().sum(-1, keepdim=True)
            score_norm = torch.sqrt(
                _index_mean(squared, batch.atom_to_orbit, orbit_count).clamp_min(0.0)
            )
            bounded = (torch.log1p(score_norm) / math.log1p(200.0)).clamp(0.0, 1.0)
            evidence = torch.cat(
                (evidence[:, :4], bounded.to(evidence.dtype), evidence[:, 5:]),
                dim=-1,
            )
        sg = self.space_group_embedding(batch.space_group)
        orbit_features = (
            orbit_features
            + self.orbit_species_embedding(batch.orbit_species)
            + self.wyckoff_embedding(batch.orbit_wyckoff)
            + sg[batch.orbit_batch]
            + self.evidence_projection(evidence)
        )
        for block in self.set_blocks:
            orbit_features = block(orbit_features, batch.orbit_batch)
        graph_features = self.graph_norm(
            _index_mean(orbit_features, batch.orbit_batch, graph_count)
        )
        lattice_score = self.lattice_chart_head(graph_features)
        lattice_score = lattice_score + self.parent_lattice_projection(
            parent_lattice.reshape(graph_count, 9).to(lattice_score.dtype)
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
            birth_coordinate_log_scale=_periodic_log_scale(
                self.birth_coordinate_log_scale_head(graph_features)
            ),
            revision_logits=self.revision_head(orbit_features),
            atom_coordinate_score=coordinate_score,
            lattice_score=lattice_score,
            bridge_mean=self.bridge_mean_head(orbit_features),
            bridge_log_scale=_periodic_log_scale(
                self.bridge_log_scale_head(orbit_features)
            ),
            orbit_features=orbit_features,
        )


def load_registered_csp_refiner(
    *,
    snapshot_root: str | Path,
    checkpoint: str | Path,
) -> tuple[CrysLLMGenWQRefiner, dict[str, Any]]:
    """Strictly map every inherited parameter before constructing WQ heads."""

    root = Path(snapshot_root).resolve()
    checkpoint_path = Path(checkpoint).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    sys.path.insert(0, str(root))
    try:
        from models_ddpm.diffusion import CSPDiffusion
    finally:
        sys.path.pop(0)
    parent = CSPDiffusion(PARENT_SCHEDULER_TIMESTEPS, PARENT_RUN_TYPE)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = payload.get("model")
    if not isinstance(state, Mapping):
        raise ValueError("registered CrysLLMGen checkpoint has no model state")
    incompatible = parent.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError("registered CrysLLMGen checkpoint mapping is not strict")
    model = CrysLLMGenWQRefiner(
        decoder=parent.decoder,
        time_embedding=parent.time_embedding,
    )
    inherited = model.inherited_parameter_count()
    report = {
        "schema": "crysllmgen_wq_csp_mapping_v1",
        "ok": True,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_keys": len(state),
        "missing_keys": [],
        "unexpected_keys": [],
        "inherited_parameters": inherited,
        "total_parameters": model.parameter_count(),
        "inherited_parameter_fraction": inherited / model.parameter_count(),
        "parent_scheduler_timesteps": PARENT_SCHEDULER_TIMESTEPS,
        "refiner_reverse_start_timestep": OFFICIAL_REVERSE_START_TIMESTEP,
        "parent_run_type": PARENT_RUN_TYPE,
        "parent_hidden_dim": 512,
    }
    return model, report
