"""Frozen terminal diffusion contract; implementation remains in the audited wrapper."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TerminalDiffusionContract:
    checkpoint_sha256: str = (
        "573e9b10af64b266b7c6cde4d0f8bdd8a7388fa98d36e2e82db341af3e511e7e"
    )
    steps: int = 800
    seed: int = 101117
    seed_by_sample_index: bool = True

    def validate(self) -> None:
        if self.steps != 800:
            raise ValueError("paper refinement is fixed to tau800")
        if not self.seed_by_sample_index:
            raise ValueError("refiner identity must follow sample_idx")


__all__ = ["TerminalDiffusionContract"]
