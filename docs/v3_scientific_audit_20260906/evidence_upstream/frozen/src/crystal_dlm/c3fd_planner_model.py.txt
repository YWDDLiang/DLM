"""One-model wrapper for the C³FD semantic composition planner."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import torch
from torch import Tensor, nn

from crystal_dlm.semantic_composition_head import (
    SemanticCompositionHead,
    SemanticCompositionOutput,
    SemanticHeadFlags,
)


@dataclass(frozen=True)
class C3FDPlannerConfig:
    context_size: int
    semantic_size: int
    num_species: int
    physics_feature_size: int
    rich_soft_head_dims: dict[str, int]
    num_families: int | None = None
    max_arity: int = 7
    ledger_feature_size: int = 0
    max_atoms: int = 20
    max_count: int = 20
    decoder_layers: int = 2
    decoder_heads: int = 4
    decoder_dropout: float = 0.05
    max_sequence_length: int = 16

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class C3FDPlannerModel(nn.Module):
    """Frozen-backbone context adapter plus a typed causal semantic decoder."""

    def __init__(
        self,
        config: C3FDPlannerConfig,
        *,
        physics_features: Tensor,
    ) -> None:
        super().__init__()
        self.config = config
        if tuple(physics_features.shape) != (
            int(config.num_species),
            int(config.physics_feature_size),
        ):
            raise ValueError("physics feature matrix does not match C3FD config")
        self.context_adapter = nn.Sequential(
            nn.LayerNorm(int(config.context_size)),
            nn.Linear(int(config.context_size), int(config.semantic_size)),
            nn.GELU(),
            nn.Linear(int(config.semantic_size), int(config.semantic_size)),
        )
        self.head = SemanticCompositionHead(
            hidden_size=int(config.semantic_size),
            num_species=int(config.num_species),
            max_atoms=int(config.max_atoms),
            max_count=int(config.max_count),
            physics_features=physics_features,
            rich_soft_head_dims=config.rich_soft_head_dims,
            num_families=config.num_families,
            max_arity=int(config.max_arity),
            ledger_feature_size=int(config.ledger_feature_size),
            decoder_layers=int(config.decoder_layers),
            decoder_heads=int(config.decoder_heads),
            decoder_dropout=float(config.decoder_dropout),
            max_sequence_length=int(config.max_sequence_length),
        )

    def context_sequence(self, context: Tensor, sequence_length: int) -> Tensor:
        if context.ndim != 2 or context.shape[-1] != int(self.config.context_size):
            raise ValueError("context must have shape [batch, context_size]")
        length = int(sequence_length)
        if length <= 0 or length > int(self.config.max_sequence_length):
            raise ValueError("invalid semantic sequence length")
        adapted = self.context_adapter(context)
        return adapted.unsqueeze(1).expand(-1, length, -1)

    def forward(
        self,
        context: Tensor,
        *,
        previous_species_indices: Tensor,
        previous_count_values: Tensor,
        previous_n_values: Tensor,
        ledger_features: Tensor | None = None,
        n_targets: Tensor | None = None,
        family_targets: Tensor | None = None,
        arity_targets: Tensor | None = None,
        species_targets: Tensor | None = None,
        count_targets: Tensor | None = None,
        rich_targets: Mapping[str, Tensor] | None = None,
        flags: SemanticHeadFlags | None = None,
        loss_weights: Mapping[str, float] | None = None,
    ) -> SemanticCompositionOutput:
        if previous_species_indices.shape != previous_count_values.shape:
            raise ValueError("semantic teacher inputs must share shape")
        hidden = self.context_sequence(context, previous_species_indices.shape[1])
        return self.head(
            hidden,
            previous_species_indices=previous_species_indices,
            previous_count_values=previous_count_values,
            previous_n_values=previous_n_values,
            ledger_features=ledger_features,
            n_targets=n_targets,
            family_targets=family_targets,
            arity_targets=arity_targets,
            species_targets=species_targets,
            count_targets=count_targets,
            rich_targets=rich_targets,
            flags=flags,
            loss_weights=loss_weights,
        )


__all__ = ["C3FDPlannerConfig", "C3FDPlannerModel"]
