"""Zero-effect wrappers used to prove the CrysLLMGen parent implementation."""

from __future__ import annotations

import dataclasses
from typing import Any


@dataclasses.dataclass(frozen=True, slots=True)
class DisabledExtensionConfig:
    representation: str = "atom"
    wyckoff_wrapper_enabled: bool = False
    topology_feedback_enabled: bool = False
    attempt_replacement_enabled: bool = False

    def __post_init__(self) -> None:
        if self.representation != "atom":
            raise ValueError("disabled extension must retain atom representation")
        if (
            self.wyckoff_wrapper_enabled
            or self.topology_feedback_enabled
            or self.attempt_replacement_enabled
        ):
            raise ValueError("disabled-extension config cannot enable new behavior")


class DisabledExtensionRefiner:
    """Transparent adapter around an instantiated upstream ``CSPDiffusion``."""

    def __init__(self, model: Any, config: DisabledExtensionConfig | None = None) -> None:
        self.model = model
        self.config = config or DisabledExtensionConfig()

    def decoder_step(
        self,
        time_embedding: Any,
        atom_types: Any,
        frac_coords: Any,
        lattices: Any,
        num_atoms: Any,
        node_to_graph: Any,
    ) -> Any:
        return self.model.decoder(
            time_embedding,
            atom_types,
            frac_coords,
            lattices,
            num_atoms,
            node_to_graph,
        )

    def sample(self, batch: Any, *, step_lr: float, diff_steps: int) -> Any:
        return self.model.sample(batch, step_lr=step_lr, diff_steps=diff_steps)
