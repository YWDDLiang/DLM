from __future__ import annotations

import random
import unittest

from crystal_dlm.wqcodiff.bridge import ChartCatalog, ChartSpec, TargetStratumBridge
from crystal_dlm.wqcodiff.events import TopologyEventType
from crystal_dlm.wqcodiff.formal import (
    chart_roundtrip_error,
    permutation_posterior_difference,
    regularized_projector_error,
    run_synthetic_transition_audit,
)
from crystal_dlm.wqcodiff.kernel import TopologyEventKernel, TransitionError
from crystal_dlm.wqcodiff.state import OrbitState, StratifiedState


class FixtureCatalog(ChartCatalog):
    def __init__(self) -> None:
        self._specs = {
            (1, 0): ChartSpec(1, 0, "a", 1, 3),
            (1, 1): ChartSpec(1, 1, "b", 2, 1),
            (1, 2): ChartSpec(1, 2, "c", 4, 0),
        }

    def get(self, space_group: int, wyckoff_type: int) -> ChartSpec:
        return self._specs[(space_group, wyckoff_type)]

    def types(self, space_group: int):
        return tuple(key[1] for key in self._specs if key[0] == space_group)


def fixture_state() -> StratifiedState:
    return StratifiedState(
        space_group=1,
        lattice_system="triclinic",
        lattice_chart=(1.0, 1.0, 1.0, 0.0, 0.0, 0.0),
        orbits=(
            OrbitState("o0", 0, 6, 1, 3, (0.1, 0.2, 0.3)),
            OrbitState("o1", 1, 8, 2, 1, (0.4,)),
        ),
    )


class EventKernelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = FixtureCatalog()
        self.bridge = TargetStratumBridge(self.catalog)
        self.kernel = TopologyEventKernel(
            catalog=self.catalog,
            bridge=self.bridge,
            species=(6, 8, 14),
        )

    def test_support_contains_all_registered_event_kinds(self) -> None:
        kinds = {event.event_type for event in self.kernel.legal_events(fixture_state())}
        self.assertEqual(
            kinds,
            {
                TopologyEventType.NONE,
                TopologyEventType.BIRTH,
                TopologyEventType.DEATH,
                TopologyEventType.WYCKOFF_CHANGE,
                TopologyEventType.SPECIES_CHANGE,
            },
        )

    def test_filtered_support_is_exact_ordered_subsequence(self) -> None:
        state = fixture_state()
        full = self.kernel.legal_events(state)
        selections = (
            {TopologyEventType.NONE},
            {TopologyEventType.BIRTH},
            {TopologyEventType.DEATH},
            {TopologyEventType.WYCKOFF_CHANGE},
            {TopologyEventType.SPECIES_CHANGE},
            {TopologyEventType.BIRTH, TopologyEventType.DEATH},
            {
                TopologyEventType.BIRTH,
                TopologyEventType.DEATH,
                TopologyEventType.WYCKOFF_CHANGE,
                TopologyEventType.SPECIES_CHANGE,
            },
            set(),
        )
        for selected in selections:
            with self.subTest(selected=selected):
                expected = tuple(
                    event for event in full if event.event_type in selected
                )
                self.assertEqual(
                    self.kernel.legal_events(state, event_types=selected),
                    expected,
                )

    def test_all_legal_events_land_in_target_stratum(self) -> None:
        state = fixture_state()
        for index, event in enumerate(self.kernel.legal_events(state)):
            target = self.kernel.apply(state, event, random.Random(index))
            self.assertGreaterEqual(target.atom_count, 1)
            self.assertLessEqual(target.atom_count, 20)
            self.assertEqual(target.space_group, state.space_group)

    def test_illegal_event_is_rejected_without_retry(self) -> None:
        state = fixture_state()
        death = next(
            event
            for event in self.kernel.legal_events(state)
            if event.event_type is TopologyEventType.DEATH
        )
        once = self.kernel.apply(state, death, random.Random(0))
        with self.assertRaises(TransitionError):
            self.kernel.apply(once, death, random.Random(0))

    def test_kernel_normalizes_and_survives_dimension_changes(self) -> None:
        result = run_synthetic_transition_audit(
            self.kernel,
            fixture_state(),
            transitions=10_000,
            seed=7,
        )
        self.assertTrue(result.passed, result)
        self.assertLess(result.normalization_error_max, 1.0e-12)


class InvarianceTests(unittest.TestCase):
    def test_mp20_limit_uses_primitive_not_conventional_multiplicity(self) -> None:
        state = StratifiedState(
            space_group=225,
            lattice_system="cubic",
            lattice_chart=(1.0,),
            orbits=(
                OrbitState("o0", 0, 13, 4, 0, (), primitive_multiplicity=1),
                OrbitState("o1", 1, 8, 4, 0, (), primitive_multiplicity=1),
            ),
        )
        self.assertEqual(state.atom_count, 2)
        self.assertEqual(state.conventional_atom_count, 8)

    def test_storage_hash_and_set_posterior_are_permutation_invariant(self) -> None:
        state = fixture_state()

        def posterior(value: StratifiedState):
            return (
                float(value.atom_count),
                float(sum(orbit.species for orbit in value.orbits)),
                float(sum(orbit.wyckoff_type for orbit in value.orbits)),
            )

        self.assertLess(
            permutation_posterior_difference(state, posterior, trials=100),
            1.0e-12,
        )
        self.assertEqual(
            state.topology_hash(),
            state.permuted(random.Random(4)).topology_hash(),
        )

    def test_wrapped_chart_roundtrip(self) -> None:
        error = chart_roundtrip_error(
            (0.1, 0.9),
            lambda q: ((q[0] + 0.25) % 1.0, (q[1] - 0.125) % 1.0),
            lambda x: ((x[0] - 0.25) % 1.0, (x[1] + 0.125) % 1.0),
        )
        self.assertLess(error, 1.0e-12)

    def test_registered_projector_recovers_nonorthonormal_tangent(self) -> None:
        error = regularized_projector_error(
            [
                [[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
                [[-1.0, 0.0], [-1.0, -1.0], [0.0, -1.0]],
            ]
        )
        self.assertLess(error, 1.0e-6)


if __name__ == "__main__":
    unittest.main()
