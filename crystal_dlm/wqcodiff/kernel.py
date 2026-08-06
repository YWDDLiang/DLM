"""Normalized legal-support event kernel and target-stratum transitions."""

from __future__ import annotations

import dataclasses
import math
import random
from collections.abc import Collection, Mapping, Sequence

from .bridge import ChartCatalog, TargetStratumBridge
from .events import TopologyEvent, TopologyEventType
from .state import OrbitState, StratifiedState


class TransitionError(RuntimeError):
    pass


def _softmax(logits: Sequence[float]) -> tuple[float, ...]:
    if not logits:
        raise ValueError("cannot normalize empty logits")
    if any(not math.isfinite(float(value)) for value in logits):
        raise ValueError("event logits must be finite")
    maximum = max(float(value) for value in logits)
    values = [math.exp(float(value) - maximum) for value in logits]
    total = math.fsum(values)
    if not math.isfinite(total) or total <= 0:
        raise ValueError("event normalization failed")
    probabilities = tuple(value / total for value in values)
    # Force an exact normalized final value up to floating point arithmetic.
    if len(probabilities) > 1:
        probabilities = probabilities[:-1] + (1.0 - math.fsum(probabilities[:-1]),)
    return probabilities


class TopologyEventKernel:
    def __init__(
        self,
        *,
        catalog: ChartCatalog,
        bridge: TargetStratumBridge,
        species: Sequence[int],
        max_atoms: int = 20,
    ) -> None:
        normalized_species = tuple(sorted(set(int(value) for value in species)))
        if not normalized_species or any(value <= 0 for value in normalized_species):
            raise ValueError("species vocabulary must contain positive atomic numbers")
        if max_atoms != 20:
            raise ValueError("the registered MP20 task requires max_atoms=20")
        self.catalog = catalog
        self.bridge = bridge
        self.species = normalized_species
        self.max_atoms = max_atoms

    def legal_events(
        self,
        state: StratifiedState,
        *,
        event_types: Collection[TopologyEventType] | None = None,
    ) -> tuple[TopologyEvent, ...]:
        """Return legal events, optionally restricted to selected event types.

        Filtering happens while the support is constructed, but the returned
        order is exactly the same as filtering the complete support afterward.
        This matters because registered recovery corruptions sample by index.
        """

        allowed = None if event_types is None else frozenset(event_types)

        def enabled(event_type: TopologyEventType) -> bool:
            return allowed is None or event_type in allowed

        events: list[TopologyEvent] = []
        if enabled(TopologyEventType.NONE):
            events.append(TopologyEvent(TopologyEventType.NONE))
        if enabled(TopologyEventType.BIRTH):
            birth_prefix = state.topology_hash()[:16]
            for wyckoff_type in self.catalog.types(state.space_group):
                spec = self.catalog.get(state.space_group, wyckoff_type)
                if state.atom_count + int(spec.primitive_multiplicity) <= self.max_atoms:
                    for species in self.species:
                        events.append(
                            TopologyEvent(
                                TopologyEventType.BIRTH,
                                target_wyckoff_type=wyckoff_type,
                                target_species=species,
                                new_orbit_id=f"o-{birth_prefix}-{wyckoff_type}-{species}",
                            )
                        )
        for orbit in state.orbits:
            if enabled(TopologyEventType.DEATH) and len(state.orbits) > 1:
                events.append(TopologyEvent(TopologyEventType.DEATH, orbit_id=orbit.orbit_id))
            if enabled(TopologyEventType.SPECIES_CHANGE):
                for species in self.species:
                    if species != orbit.species:
                        events.append(
                            TopologyEvent(
                                TopologyEventType.SPECIES_CHANGE,
                                orbit_id=orbit.orbit_id,
                                target_species=species,
                            )
                        )
            if enabled(TopologyEventType.WYCKOFF_CHANGE):
                for wyckoff_type in self.catalog.types(state.space_group):
                    if wyckoff_type == orbit.wyckoff_type:
                        continue
                    target = self.catalog.get(state.space_group, wyckoff_type)
                    new_count = (
                        state.atom_count
                        - int(orbit.primitive_multiplicity)
                        + int(target.primitive_multiplicity)
                    )
                    if 1 <= new_count <= self.max_atoms:
                        events.append(
                            TopologyEvent(
                                TopologyEventType.WYCKOFF_CHANGE,
                                orbit_id=orbit.orbit_id,
                                target_wyckoff_type=wyckoff_type,
                                new_orbit_id=orbit.orbit_id,
                            )
                        )
        return tuple(events)

    def probabilities(
        self,
        state: StratifiedState,
        logits: Mapping[TopologyEvent, float],
    ) -> tuple[tuple[TopologyEvent, float], ...]:
        events = self.legal_events(state)
        values = [float(logits.get(event, -30.0)) for event in events]
        return tuple(zip(events, _softmax(values)))

    def sample(
        self,
        state: StratifiedState,
        logits: Mapping[TopologyEvent, float],
        rng: random.Random,
    ) -> TopologyEvent:
        weighted = self.probabilities(state, logits)
        threshold = rng.random()
        cumulative = 0.0
        for event, probability in weighted:
            cumulative += probability
            if threshold <= cumulative:
                return event
        return weighted[-1][0]

    def apply(
        self,
        state: StratifiedState,
        event: TopologyEvent,
        rng: random.Random,
    ) -> StratifiedState:
        legal = set(self.legal_events(state, event_types=(event.event_type,)))
        if event not in legal:
            raise TransitionError(f"illegal transition: {event}")
        if event.event_type is TopologyEventType.NONE:
            return state

        orbits = list(state.orbits)
        index = {orbit.orbit_id: position for position, orbit in enumerate(orbits)}
        if event.event_type is TopologyEventType.DEATH:
            del orbits[index[event.orbit_id or ""]]
        elif event.event_type is TopologyEventType.SPECIES_CHANGE:
            position = index[event.orbit_id or ""]
            orbits[position] = dataclasses.replace(orbits[position], species=int(event.target_species))
        elif event.event_type is TopologyEventType.BIRTH:
            base_id = str(event.new_orbit_id)
            existing_ids = set(index)
            orbit_id = base_id
            suffix = 1
            while orbit_id in existing_ids:
                orbit_id = f"{base_id}-{suffix}"
                suffix += 1
            result = self.bridge.propose(
                state=state,
                wyckoff_type=int(event.target_wyckoff_type),
                species=int(event.target_species),
                orbit_id=orbit_id,
                rng=rng,
            )
            if not result.success or result.orbit is None:
                raise TransitionError(f"bridge_failure:{result.reason}")
            orbits.append(result.orbit)
        elif event.event_type is TopologyEventType.WYCKOFF_CHANGE:
            position = index[event.orbit_id or ""]
            old = orbits[position]
            # This is deliberately death followed by birth, never an implicit
            # resize of the old coordinate tensor.  The bridge may condition on
            # the pre-event state; death/birth is committed atomically so that
            # an otherwise valid one-orbit crystal never materializes as an
            # invalid empty semantic state.
            result = self.bridge.propose(
                state=state,
                wyckoff_type=int(event.target_wyckoff_type),
                species=old.species,
                orbit_id=old.orbit_id,
                rng=rng,
            )
            if not result.success or result.orbit is None:
                raise TransitionError(f"bridge_failure:{result.reason}")
            orbits[position] = result.orbit
        else:  # pragma: no cover - exhaustive enum guard
            raise TransitionError(f"unsupported event type: {event.event_type}")
        try:
            return state.replace_orbits(orbits)
        except ValueError as exc:
            raise TransitionError(f"target_stratum_validation:{exc}") from exc
