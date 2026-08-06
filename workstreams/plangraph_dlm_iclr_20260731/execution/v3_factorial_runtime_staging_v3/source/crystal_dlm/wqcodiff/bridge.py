"""Target-stratum chart catalogue and single-shot birth bridge."""

from __future__ import annotations

import dataclasses
import math
import random
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence

from .state import OrbitState, StratifiedState


@dataclasses.dataclass(frozen=True, slots=True)
class ChartSpec:
    space_group: int
    wyckoff_type: int
    letter: str
    multiplicity: int
    dimension: int
    primitive_multiplicity: int | None = None

    def __post_init__(self) -> None:
        if not 1 <= self.space_group <= 230:
            raise ValueError("invalid space group")
        if self.wyckoff_type < 0 or not self.letter:
            raise ValueError("invalid Wyckoff identifier")
        if self.multiplicity <= 0 or self.dimension not in {0, 1, 2, 3}:
            raise ValueError("invalid chart multiplicity/dimension")
        if self.primitive_multiplicity is None:
            object.__setattr__(self, "primitive_multiplicity", self.multiplicity)
        if int(self.primitive_multiplicity) <= 0 or self.multiplicity % int(self.primitive_multiplicity) != 0:
            raise ValueError("invalid primitive multiplicity")


class ChartCatalog(ABC):
    @abstractmethod
    def get(self, space_group: int, wyckoff_type: int) -> ChartSpec:
        raise NotImplementedError

    @abstractmethod
    def types(self, space_group: int) -> Sequence[int]:
        raise NotImplementedError


@dataclasses.dataclass(frozen=True, slots=True)
class BridgeResult:
    success: bool
    orbit: OrbitState | None
    base_coordinate: tuple[float, ...]
    residual: tuple[float, ...]
    reason: str = ""


BridgeResidual = Callable[[StratifiedState, ChartSpec, int, tuple[float, ...]], Sequence[float]]


class TargetStratumBridge:
    """Conditional base plus optional learned residual, evaluated once.

    The bridge never retries.  A non-finite output is returned as a terminal
    bridge failure by the caller.
    """

    def __init__(self, catalog: ChartCatalog, residual_model: BridgeResidual | None = None) -> None:
        self.catalog = catalog
        self.residual_model = residual_model

    def propose(
        self,
        *,
        state: StratifiedState,
        wyckoff_type: int,
        species: int,
        orbit_id: str,
        rng: random.Random,
    ) -> BridgeResult:
        try:
            spec = self.catalog.get(state.space_group, wyckoff_type)
        except Exception as exc:
            return BridgeResult(False, None, (), (), f"chart_lookup:{type(exc).__name__}:{exc}")
        base = tuple(rng.random() for _ in range(spec.dimension))
        try:
            raw_residual = (
                tuple(float(value) for value in self.residual_model(state, spec, species, base))
                if self.residual_model is not None
                else (0.0,) * spec.dimension
            )
        except Exception as exc:
            return BridgeResult(False, None, base, (), f"residual_model:{type(exc).__name__}:{exc}")
        if len(raw_residual) != spec.dimension or not all(math.isfinite(v) for v in raw_residual):
            return BridgeResult(False, None, base, raw_residual, "invalid_residual")
        coordinate = tuple((value + delta) % 1.0 for value, delta in zip(base, raw_residual))
        try:
            orbit = OrbitState(
                orbit_id=orbit_id,
                wyckoff_type=wyckoff_type,
                species=species,
                multiplicity=spec.multiplicity,
                chart_dimension=spec.dimension,
                free_coordinate=coordinate,
                primitive_multiplicity=spec.primitive_multiplicity,
            )
        except ValueError as exc:
            return BridgeResult(False, None, base, raw_residual, f"orbit_validation:{exc}")
        return BridgeResult(True, orbit, base, raw_residual)
