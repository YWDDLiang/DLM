"""Deterministic fixed-topology composition projection for WQ states.

The projector is a mechanism diagnostic.  It changes only complete-orbit
species labels, preserves the exact original element set, and never samples a
replacement candidate.  The default classifier is the frozen legacy SMACT
composition check; tests may inject a dependency-free classifier.
"""

from __future__ import annotations

import dataclasses
import hashlib
import itertools
import json
import math
from collections import Counter
from functools import reduce
from typing import Any, Callable, Mapping, Sequence

from crystal_dlm.composition_validity import formula_from_composition

from .state import OrbitState, StratifiedState


CompositionClassifier = Callable[
    [Sequence[int], Sequence[int]],
    Mapping[str, Any],
]


@dataclasses.dataclass(frozen=True, order=True, slots=True)
class ProjectionObjective:
    """Lexicographic objective frozen by the MLIP-free experiment plan."""

    changed_orbit_count: int
    affected_primitive_atom_count: int
    raw_composition_count_l1: int
    canonical_assignment_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True, slots=True)
class CompositionProjectionResult:
    """One attempt-preserving projection result."""

    status: str
    source_reason: str
    state: StratifiedState
    original_formula: str
    projected_formula: str
    original_raw_counts: tuple[tuple[int, int], ...]
    projected_raw_counts: tuple[tuple[int, int], ...]
    changed_orbit_ids: tuple[str, ...]
    candidate_assignments_considered: int
    classifier_evaluations: int
    max_changed_orbits: int
    max_candidate_assignments: int
    objective: ProjectionObjective | None = None
    error: str | None = None

    @property
    def projected(self) -> bool:
        return self.status == "projected"

    def to_dict(self, *, include_state: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": "wqcodiff_fixed_topology_composition_projection_v1",
            "status": self.status,
            "source_reason": self.source_reason,
            "attempt_id": self.state.attempt_id,
            "original_formula": self.original_formula,
            "projected_formula": self.projected_formula,
            "original_raw_counts": [
                {"atomic_number": atomic_number, "count": count}
                for atomic_number, count in self.original_raw_counts
            ],
            "projected_raw_counts": [
                {"atomic_number": atomic_number, "count": count}
                for atomic_number, count in self.projected_raw_counts
            ],
            "changed_orbit_ids": list(self.changed_orbit_ids),
            "candidate_assignments_considered": self.candidate_assignments_considered,
            "classifier_evaluations": self.classifier_evaluations,
            "max_changed_orbits": self.max_changed_orbits,
            "max_candidate_assignments": self.max_candidate_assignments,
            "objective": self.objective.to_dict() if self.objective else None,
            "error": self.error,
        }
        if include_state:
            payload["state"] = self.state.to_dict(canonical_storage=True)
        return payload


def legacy_smact_classifier(
    elems: Sequence[int],
    counts: Sequence[int],
) -> Mapping[str, Any]:
    """Run the frozen CrysLLMGen-compatible SMACT classifier."""

    from crystal_dlm.composition_validity import classify_smact_validity

    return classify_smact_validity(elems, counts)


def _raw_counts(
    orbits: Sequence[OrbitState],
    species_assignment: Sequence[int] | None = None,
) -> tuple[tuple[int, int], ...]:
    if species_assignment is not None and len(species_assignment) != len(orbits):
        raise ValueError("species assignment length does not match orbit count")
    counter: Counter[int] = Counter()
    for index, orbit in enumerate(orbits):
        species = (
            int(species_assignment[index])
            if species_assignment is not None
            else int(orbit.species)
        )
        counter[species] += int(orbit.primitive_multiplicity)
    return tuple((species, int(counter[species])) for species in sorted(counter))


def _reduced_counts(
    raw_counts: Sequence[tuple[int, int]],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if not raw_counts:
        return (), ()
    divisor = reduce(math.gcd, (int(count) for _, count in raw_counts))
    divisor = max(divisor, 1)
    return (
        tuple(int(species) for species, _ in raw_counts),
        tuple(int(count) // divisor for _, count in raw_counts),
    )


def _formula(raw_counts: Sequence[tuple[int, int]]) -> str:
    elems, counts = _reduced_counts(raw_counts)
    return formula_from_composition(elems, counts)


def _assignment_sha256(
    orbits: Sequence[OrbitState],
    assignment: Sequence[int],
) -> str:
    payload = [
        {"orbit_id": orbit.orbit_id, "species": int(species)}
        for orbit, species in zip(orbits, assignment)
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _classify(
    classifier: CompositionClassifier,
    raw_counts: Sequence[tuple[int, int]],
) -> Mapping[str, Any]:
    elems, counts = _reduced_counts(raw_counts)
    result = classifier(elems, counts)
    if not isinstance(result, Mapping):
        raise TypeError("composition classifier must return a mapping")
    if "valid" not in result or "reason" not in result:
        raise ValueError("composition classifier result needs valid and reason")
    return result


def _topology_invariants_hold(
    original: StratifiedState,
    projected: StratifiedState,
) -> bool:
    if (
        original.space_group != projected.space_group
        or original.lattice_system != projected.lattice_system
        or original.lattice_chart != projected.lattice_chart
        or original.attempt_id != projected.attempt_id
        or original.timestep != projected.timestep
        or original.space_group_committed != projected.space_group_committed
        or original.atom_count != projected.atom_count
    ):
        return False
    original_by_id = {orbit.orbit_id: orbit for orbit in original.orbits}
    projected_by_id = {orbit.orbit_id: orbit for orbit in projected.orbits}
    if set(original_by_id) != set(projected_by_id):
        return False
    for orbit_id, before in original_by_id.items():
        after = projected_by_id[orbit_id]
        if (
            before.wyckoff_type != after.wyckoff_type
            or before.multiplicity != after.multiplicity
            or before.primitive_multiplicity != after.primitive_multiplicity
            or before.chart_dimension != after.chart_dimension
            or before.free_coordinate != after.free_coordinate
        ):
            return False
    return {
        int(orbit.species) for orbit in original.orbits
    } == {int(orbit.species) for orbit in projected.orbits}


class FixedTopologyCompositionProjector:
    """Search objective-optimal whole-orbit species reassignments."""

    def __init__(
        self,
        *,
        classifier: CompositionClassifier = legacy_smact_classifier,
        applicable_source_reason: str = "charge_neutrality_fail",
        max_changed_orbits: int = 6,
        max_candidate_assignments: int = 100_000,
    ) -> None:
        if not applicable_source_reason:
            raise ValueError("applicable_source_reason is required")
        if not 1 <= int(max_changed_orbits) <= 20:
            raise ValueError("max_changed_orbits must be in [1,20]")
        if not 1 <= int(max_candidate_assignments) <= 1_000_000:
            raise ValueError("max_candidate_assignments must be in [1,1000000]")
        self.classifier = classifier
        self.applicable_source_reason = str(applicable_source_reason)
        self.max_changed_orbits = int(max_changed_orbits)
        self.max_candidate_assignments = int(max_candidate_assignments)

    def _result(
        self,
        *,
        status: str,
        source_reason: str,
        state: StratifiedState,
        original_raw_counts: tuple[tuple[int, int], ...],
        projected_raw_counts: tuple[tuple[int, int], ...] | None = None,
        changed_orbit_ids: Sequence[str] = (),
        candidate_assignments_considered: int = 0,
        classifier_evaluations: int = 0,
        objective: ProjectionObjective | None = None,
        error: str | None = None,
    ) -> CompositionProjectionResult:
        after = projected_raw_counts or original_raw_counts
        return CompositionProjectionResult(
            status=status,
            source_reason=source_reason,
            state=state,
            original_formula=_formula(original_raw_counts),
            projected_formula=_formula(after),
            original_raw_counts=original_raw_counts,
            projected_raw_counts=after,
            changed_orbit_ids=tuple(changed_orbit_ids),
            candidate_assignments_considered=int(candidate_assignments_considered),
            classifier_evaluations=int(classifier_evaluations),
            max_changed_orbits=self.max_changed_orbits,
            max_candidate_assignments=self.max_candidate_assignments,
            objective=objective,
            error=error,
        )

    def project(self, state: StratifiedState) -> CompositionProjectionResult:
        """Return one deterministic projection or the untouched input attempt."""

        stable_orbits = tuple(sorted(state.orbits, key=lambda orbit: orbit.orbit_id))
        original_assignment = tuple(int(orbit.species) for orbit in stable_orbits)
        original_raw = _raw_counts(stable_orbits, original_assignment)
        classifier_evaluations = 0
        try:
            source_classification = _classify(self.classifier, original_raw)
            classifier_evaluations += 1
        except Exception as exc:  # noqa: BLE001 - failure is retained and audited.
            return self._result(
                status="classifier_error",
                source_reason="classifier_error",
                state=state,
                original_raw_counts=original_raw,
                classifier_evaluations=classifier_evaluations,
                error=f"{type(exc).__name__}: {exc}",
            )

        source_reason = str(source_classification["reason"])
        if source_reason != self.applicable_source_reason:
            return self._result(
                status="identity_protected_reason",
                source_reason=source_reason,
                state=state,
                original_raw_counts=original_raw,
                classifier_evaluations=classifier_evaluations,
            )

        original_elements = tuple(sorted({int(value) for value in original_assignment}))
        if len(original_elements) < 2 or len(stable_orbits) < len(original_elements):
            return self._result(
                status="no_solution",
                source_reason=source_reason,
                state=state,
                original_raw_counts=original_raw,
                classifier_evaluations=classifier_evaluations,
            )

        candidate_assignments_considered = 0
        maximum_changes = min(self.max_changed_orbits, len(stable_orbits))
        original_counter = dict(original_raw)
        for changed_count in range(1, maximum_changes + 1):
            candidates: list[
                tuple[ProjectionObjective, tuple[int, ...], tuple[tuple[int, int], ...]]
            ] = []
            for changed_indices in itertools.combinations(
                range(len(stable_orbits)),
                changed_count,
            ):
                replacement_options = tuple(
                    tuple(
                        element
                        for element in original_elements
                        if element != original_assignment[index]
                    )
                    for index in changed_indices
                )
                for replacements in itertools.product(*replacement_options):
                    candidate_assignments_considered += 1
                    if (
                        candidate_assignments_considered
                        > self.max_candidate_assignments
                    ):
                        return self._result(
                            status="budget_exhausted",
                            source_reason=source_reason,
                            state=state,
                            original_raw_counts=original_raw,
                            candidate_assignments_considered=(
                                candidate_assignments_considered
                            ),
                            classifier_evaluations=classifier_evaluations,
                        )
                    assignment = list(original_assignment)
                    for index, species in zip(changed_indices, replacements):
                        assignment[index] = int(species)
                    assignment_tuple = tuple(assignment)
                    if set(assignment_tuple) != set(original_elements):
                        continue
                    raw_counts = _raw_counts(stable_orbits, assignment_tuple)
                    candidate_counter = dict(raw_counts)
                    count_l1 = sum(
                        abs(
                            int(candidate_counter.get(element, 0))
                            - int(original_counter.get(element, 0))
                        )
                        for element in original_elements
                    )
                    objective = ProjectionObjective(
                        changed_orbit_count=changed_count,
                        affected_primitive_atom_count=sum(
                            int(stable_orbits[index].primitive_multiplicity)
                            for index in changed_indices
                        ),
                        raw_composition_count_l1=count_l1,
                        canonical_assignment_sha256=_assignment_sha256(
                            stable_orbits,
                            assignment_tuple,
                        ),
                    )
                    candidates.append((objective, assignment_tuple, raw_counts))

            candidates.sort(key=lambda item: item[0])
            for objective, assignment, raw_counts in candidates:
                try:
                    classification = _classify(self.classifier, raw_counts)
                    classifier_evaluations += 1
                except Exception as exc:  # noqa: BLE001 - fail closed.
                    return self._result(
                        status="classifier_error",
                        source_reason=source_reason,
                        state=state,
                        original_raw_counts=original_raw,
                        candidate_assignments_considered=(
                            candidate_assignments_considered
                        ),
                        classifier_evaluations=classifier_evaluations,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                if classification.get("valid") is not True:
                    continue

                species_by_id = {
                    orbit.orbit_id: int(species)
                    for orbit, species in zip(stable_orbits, assignment)
                }
                projected_orbits = tuple(
                    dataclasses.replace(
                        orbit,
                        species=species_by_id[orbit.orbit_id],
                    )
                    for orbit in state.orbits
                )
                projected_state = state.replace_orbits(projected_orbits)
                if not _topology_invariants_hold(state, projected_state):
                    raise RuntimeError(
                        "fixed-topology composition projection violated an invariant"
                    )
                changed_ids = tuple(
                    orbit.orbit_id
                    for orbit, species in zip(stable_orbits, assignment)
                    if int(orbit.species) != int(species)
                )
                return self._result(
                    status="projected",
                    source_reason=source_reason,
                    state=projected_state,
                    original_raw_counts=original_raw,
                    projected_raw_counts=raw_counts,
                    changed_orbit_ids=changed_ids,
                    candidate_assignments_considered=(
                        candidate_assignments_considered
                    ),
                    classifier_evaluations=classifier_evaluations,
                    objective=objective,
                )

        return self._result(
            status="no_solution",
            source_reason=source_reason,
            state=state,
            original_raw_counts=original_raw,
            candidate_assignments_considered=candidate_assignments_considered,
            classifier_evaluations=classifier_evaluations,
        )


__all__ = [
    "CompositionClassifier",
    "CompositionProjectionResult",
    "FixedTopologyCompositionProjector",
    "ProjectionObjective",
    "legacy_smact_classifier",
]
