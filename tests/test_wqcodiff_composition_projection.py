from __future__ import annotations

import copy
import unittest

from crystal_dlm.wqcodiff.composition_projection import (
    FixedTopologyCompositionProjector,
)
from crystal_dlm.wqcodiff.state import OrbitState, StratifiedState


LI = 3
OXYGEN = 8


def state_with_orbits(orbits: tuple[OrbitState, ...]) -> StratifiedState:
    return StratifiedState(
        space_group=1,
        lattice_system="triclinic",
        lattice_chart=(4.0, 4.1, 4.2, 90.0, 91.0, 92.0),
        orbits=orbits,
        attempt_id="mechanism-0001",
        timestep=0.0,
    )


def li2o_classifier(elems, counts):
    values = dict(zip((int(value) for value in elems), (int(value) for value in counts)))
    if values == {LI: 2, OXYGEN: 1}:
        return {
            "valid": True,
            "reason": "charge_neutral_pauling_valid",
        }
    return {"valid": False, "reason": "charge_neutrality_fail"}


class FixedTopologyCompositionProjectionTests(unittest.TestCase):
    def test_projects_one_complete_orbit_and_preserves_topology(self) -> None:
        state = state_with_orbits(
            (
                OrbitState("li", 0, LI, 1, 0, (), 1),
                OrbitState("o-a", 1, OXYGEN, 1, 1, (0.25,), 1),
                OrbitState("o-b", 2, OXYGEN, 1, 2, (0.25, 0.5), 1),
            )
        )
        before = copy.deepcopy(state.to_dict(canonical_storage=False))
        result = FixedTopologyCompositionProjector(
            classifier=li2o_classifier,
            max_changed_orbits=2,
        ).project(state)

        self.assertTrue(result.projected)
        self.assertEqual(result.original_formula, "LiO2")
        self.assertEqual(result.projected_formula, "Li2O")
        self.assertEqual(len(result.changed_orbit_ids), 1)
        self.assertEqual(result.objective.changed_orbit_count, 1)
        self.assertEqual(result.objective.affected_primitive_atom_count, 1)
        self.assertEqual(
            {orbit.species for orbit in result.state.orbits},
            {LI, OXYGEN},
        )
        self.assertEqual(result.state.atom_count, state.atom_count)
        self.assertEqual(
            state.to_dict(canonical_storage=False),
            before,
            "the immutable input state changed",
        )

        before_by_id = {orbit.orbit_id: orbit for orbit in state.orbits}
        after_by_id = {orbit.orbit_id: orbit for orbit in result.state.orbits}
        for orbit_id, original in before_by_id.items():
            projected = after_by_id[orbit_id]
            self.assertEqual(original.wyckoff_type, projected.wyckoff_type)
            self.assertEqual(original.multiplicity, projected.multiplicity)
            self.assertEqual(
                original.primitive_multiplicity,
                projected.primitive_multiplicity,
            )
            self.assertEqual(original.chart_dimension, projected.chart_dimension)
            self.assertEqual(original.free_coordinate, projected.free_coordinate)

    def test_is_deterministic_and_orbit_order_invariant(self) -> None:
        orbits = (
            OrbitState("li", 0, LI, 1, 0, (), 1),
            OrbitState("o-a", 1, OXYGEN, 1, 1, (0.25,), 1),
            OrbitState("o-b", 2, OXYGEN, 1, 2, (0.25, 0.5), 1),
        )
        projector = FixedTopologyCompositionProjector(
            classifier=li2o_classifier,
            max_changed_orbits=2,
        )
        first = projector.project(state_with_orbits(orbits))
        second = projector.project(state_with_orbits(tuple(reversed(orbits))))
        first_species = {
            orbit.orbit_id: orbit.species for orbit in first.state.orbits
        }
        second_species = {
            orbit.orbit_id: orbit.species for orbit in second.state.orbits
        }
        self.assertEqual(first_species, second_species)
        self.assertEqual(first.changed_orbit_ids, second.changed_orbit_ids)
        self.assertEqual(first.objective, second.objective)

    def test_prefers_fewer_affected_atoms(self) -> None:
        state = state_with_orbits(
            (
                OrbitState("li", 0, LI, 1, 0, (), 1),
                OrbitState("o-small", 1, OXYGEN, 1, 0, (), 1),
                OrbitState("o-large", 2, OXYGEN, 2, 0, (), 2),
            )
        )

        def any_changed_assignment_is_valid(elems, counts):
            values = dict(zip(elems, counts))
            if values == {LI: 1, OXYGEN: 3}:
                return {"valid": False, "reason": "charge_neutrality_fail"}
            return {"valid": True, "reason": "charge_neutral_pauling_valid"}

        result = FixedTopologyCompositionProjector(
            classifier=any_changed_assignment_is_valid
        ).project(state)
        self.assertTrue(result.projected)
        self.assertEqual(result.changed_orbit_ids, ("o-small",))
        self.assertEqual(result.objective.affected_primitive_atom_count, 1)

    def test_valid_and_pauling_cases_are_identity(self) -> None:
        state = state_with_orbits(
            (
                OrbitState("li", 0, LI, 2, 0, (), 2),
                OrbitState("o", 1, OXYGEN, 1, 0, (), 1),
            )
        )
        for reason, valid in (
            ("charge_neutral_pauling_valid", True),
            ("pauling_fail_or_ratio_rejected", False),
            ("all_metal_shortcut", True),
            ("single_element_shortcut", True),
        ):
            projector = FixedTopologyCompositionProjector(
                classifier=lambda elems, counts, reason=reason, valid=valid: {
                    "valid": valid,
                    "reason": reason,
                }
            )
            result = projector.project(state)
            self.assertEqual(result.status, "identity_protected_reason")
            self.assertIs(result.state, state)
            self.assertEqual(result.changed_orbit_ids, ())

    def test_exact_original_element_set_is_required(self) -> None:
        state = state_with_orbits(
            (
                OrbitState("li-only", 0, LI, 1, 0, (), 1),
                OrbitState("o-a", 1, OXYGEN, 1, 0, (), 1),
                OrbitState("o-b", 2, OXYGEN, 1, 0, (), 1),
            )
        )

        def invalid_unless_element_was_dropped(elems, counts):
            if tuple(elems) == (OXYGEN,):
                return {"valid": True, "reason": "single_element_shortcut"}
            return {"valid": False, "reason": "charge_neutrality_fail"}

        result = FixedTopologyCompositionProjector(
            classifier=invalid_unless_element_was_dropped,
            max_changed_orbits=1,
        ).project(state)
        self.assertEqual(result.status, "no_solution")
        self.assertIs(result.state, state)

    def test_search_budget_exhaustion_fails_closed(self) -> None:
        state = state_with_orbits(
            (
                OrbitState("li-a", 0, LI, 1, 0, (), 1),
                OrbitState("li-b", 1, LI, 1, 0, (), 1),
                OrbitState("o-a", 2, OXYGEN, 1, 0, (), 1),
                OrbitState("o-b", 3, OXYGEN, 1, 0, (), 1),
            )
        )
        projector = FixedTopologyCompositionProjector(
            classifier=lambda elems, counts: {
                "valid": False,
                "reason": "charge_neutrality_fail",
            },
            max_changed_orbits=2,
            max_candidate_assignments=1,
        )
        result = projector.project(state)
        self.assertEqual(result.status, "budget_exhausted")
        self.assertIs(result.state, state)
        self.assertGreater(result.candidate_assignments_considered, 1)

    def test_no_solution_keeps_original_attempt(self) -> None:
        state = state_with_orbits(
            (
                OrbitState("li", 0, LI, 1, 0, (), 1),
                OrbitState("o-a", 1, OXYGEN, 1, 0, (), 1),
                OrbitState("o-b", 2, OXYGEN, 1, 0, (), 1),
            )
        )
        result = FixedTopologyCompositionProjector(
            classifier=lambda elems, counts: {
                "valid": False,
                "reason": "charge_neutrality_fail",
            },
            max_changed_orbits=1,
        ).project(state)
        self.assertEqual(result.status, "no_solution")
        self.assertIs(result.state, state)
        self.assertEqual(result.original_formula, result.projected_formula)

    def test_classifier_error_is_a_retained_failure(self) -> None:
        state = state_with_orbits(
            (
                OrbitState("li", 0, LI, 1, 0, (), 1),
                OrbitState("o", 1, OXYGEN, 1, 0, (), 1),
            )
        )

        def broken_classifier(elems, counts):
            raise RuntimeError("classifier unavailable")

        result = FixedTopologyCompositionProjector(
            classifier=broken_classifier
        ).project(state)
        self.assertEqual(result.status, "classifier_error")
        self.assertIs(result.state, state)
        self.assertIn("classifier unavailable", result.error)

    def test_result_is_json_serializable(self) -> None:
        state = state_with_orbits(
            (
                OrbitState("li", 0, LI, 1, 0, (), 1),
                OrbitState("o-a", 1, OXYGEN, 1, 0, (), 1),
                OrbitState("o-b", 2, OXYGEN, 1, 0, (), 1),
            )
        )
        result = FixedTopologyCompositionProjector(
            classifier=li2o_classifier
        ).project(state)
        encoded = __import__("json").dumps(result.to_dict(), sort_keys=True)
        self.assertIn('"status": "projected"', encoded)


if __name__ == "__main__":
    unittest.main()
