from __future__ import annotations

import unittest

from crystal_dlm.wqcodiff.crysllmgen.wtb_identity_v2 import (
    audit_legacy_source_row,
    canonicalize_proposal_payload,
    composition_counts,
    composition_multiset_signature,
    ordered_atomic_numbers_signature,
    source_signature_v2,
)
from crystal_dlm.wqcodiff.state import OrbitState, StratifiedState


def _noncanonical_state() -> StratifiedState:
    return StratifiedState(
        space_group=1,
        lattice_system="triclinic",
        lattice_chart=(3.0, 4.0, 5.0, 90.0, 90.0, 90.0),
        orbits=(
            OrbitState(
                orbit_id="z-second",
                wyckoff_type=1,
                species=14,
                multiplicity=1,
                primitive_multiplicity=1,
                chart_dimension=3,
                free_coordinate=(0.2, 0.3, 0.4),
            ),
            OrbitState(
                orbit_id="a-first",
                wyckoff_type=0,
                species=8,
                multiplicity=1,
                primitive_multiplicity=1,
                chart_dimension=3,
                free_coordinate=(0.1, 0.2, 0.3),
            ),
        ),
        attempt_id="identity-v2-fixture",
    )


class WTBIdentityV2Tests(unittest.TestCase):
    def test_composition_signature_is_permutation_invariant(self) -> None:
        left = [14, 8, 8, 26]
        right = [8, 26, 14, 8]
        self.assertEqual(composition_counts(left), ((8, 2), (14, 1), (26, 1)))
        self.assertEqual(
            composition_multiset_signature(left),
            composition_multiset_signature(right),
        )
        self.assertNotEqual(
            ordered_atomic_numbers_signature(left),
            ordered_atomic_numbers_signature(right),
        )

    def test_canonical_payload_round_trip_is_stable(self) -> None:
        state = _noncanonical_state()
        original_orbits = [orbit.orbit_id for orbit in state.orbits]
        canonical_state, payload = canonicalize_proposal_payload(state.to_dict())
        canonical_orbits = [orbit.orbit_id for orbit in canonical_state.orbits]
        self.assertEqual(original_orbits, ["z-second", "a-first"])
        self.assertEqual(canonical_orbits, ["a-first", "z-second"])
        self.assertEqual(canonical_state.to_dict(), payload)
        self.assertEqual(canonical_state.topology_hash(), state.topology_hash())

    def test_legacy_order_mismatch_is_diagnostic_not_blocking(self) -> None:
        state = _noncanonical_state()
        payload = state.to_dict()
        topology = state.topology_hash()
        row = {
            "status": "succeeded",
            "proposal_state": payload,
            "proposal_topology_hash": topology,
            "atomic_numbers": [14, 8],
            "atom_count": 2,
        }
        _, canonical, audit = audit_legacy_source_row(
            row,
            canonical_atomic_numbers=[8, 14],
        )
        self.assertEqual(canonical, payload)
        self.assertTrue(audit.ok)
        self.assertTrue(audit.composition_multiset_identity)
        self.assertFalse(audit.legacy_order_identity)
        self.assertFalse(audit.legacy_order_mismatch_is_blocking)

    def test_true_composition_change_remains_fail_closed(self) -> None:
        state = _noncanonical_state()
        row = {
            "status": "succeeded",
            "proposal_state": state.to_dict(),
            "proposal_topology_hash": state.topology_hash(),
            "atomic_numbers": [14, 8],
            "atom_count": 2,
        }
        with self.assertRaisesRegex(
            ValueError,
            "composition_multiset_identity",
        ):
            audit_legacy_source_row(
                row,
                canonical_atomic_numbers=[8, 8],
            )

    def test_topology_change_remains_fail_closed(self) -> None:
        state = _noncanonical_state()
        row = {
            "status": "succeeded",
            "proposal_state": state.to_dict(),
            "proposal_topology_hash": "0" * 64,
            "atomic_numbers": [14, 8],
            "atom_count": 2,
        }
        with self.assertRaisesRegex(ValueError, "topology_identity"):
            audit_legacy_source_row(
                row,
                canonical_atomic_numbers=[8, 14],
            )

    def test_v2_source_signature_does_not_depend_on_atom_order(self) -> None:
        state = _noncanonical_state()
        payload = state.to_dict()
        first = source_signature_v2(
            proposal_state=payload,
            topology_hash=state.topology_hash(),
            atomic_numbers=[14, 8],
        )
        second = source_signature_v2(
            proposal_state=payload,
            topology_hash=state.topology_hash(),
            atomic_numbers=[8, 14],
        )
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
