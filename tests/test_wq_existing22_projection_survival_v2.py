from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from diagnostics.run_wq_existing22_projection_survival import SurvivalAuditError
from diagnostics.run_wq_existing22_projection_survival_v2 import (
    EXPECTED_REORDERED_ORDINALS,
    align_projected_state_to_source_order,
    validate_topology_invariants_by_id,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT
    / "configs"
    / "experiments"
    / "wyckoff_codiffusion"
    / "wq_existing22_projection_survival_v2_order_alignment.json"
)


def state() -> dict:
    return {
        "attempt_id": "a-order",
        "space_group": 1,
        "lattice_system": "triclinic",
        "lattice_chart": [1.0, 1.1, 1.2, 0.0, 0.0, 0.0],
        "orbits": [
            {
                "orbit_id": "o2",
                "wyckoff_type": 2,
                "species": 8,
                "multiplicity": 1,
                "primitive_multiplicity": 1,
                "chart_dimension": 0,
                "free_coordinate": [],
            },
            {
                "orbit_id": "o1",
                "wyckoff_type": 1,
                "species": 3,
                "multiplicity": 1,
                "primitive_multiplicity": 1,
                "chart_dimension": 0,
                "free_coordinate": [],
            },
        ],
        "timestep": 1.0,
        "space_group_committed": True,
    }


class OrderAlignmentTests(unittest.TestCase):
    def test_identity_only_reorder_is_accepted_and_restored(self) -> None:
        original = state()
        projected = copy.deepcopy(original)
        projected["orbits"][0]["species"] = 3
        projected["orbits"][1]["species"] = 8
        projected["orbits"].reverse()
        validate_topology_invariants_by_id(
            original,
            projected,
            ("o1", "o2"),
        )
        aligned, reordered = align_projected_state_to_source_order(
            original,
            projected,
        )
        self.assertTrue(reordered)
        self.assertEqual(
            [orbit["orbit_id"] for orbit in aligned["orbits"]],
            ["o2", "o1"],
        )
        self.assertEqual(
            {orbit["orbit_id"]: orbit["species"] for orbit in aligned["orbits"]},
            {"o2": 3, "o1": 8},
        )

    def test_per_id_geometry_change_still_fails_closed(self) -> None:
        original = state()
        projected = copy.deepcopy(original)
        projected["orbits"][0]["species"] = 3
        projected["orbits"][1]["species"] = 8
        projected["orbits"][0]["wyckoff_type"] = 9
        projected["orbits"].reverse()
        with self.assertRaises(SurvivalAuditError):
            validate_topology_invariants_by_id(
                original,
                projected,
                ("o1", "o2"),
            )

    def test_undeclared_species_change_still_fails_closed(self) -> None:
        original = state()
        projected = copy.deepcopy(original)
        projected["orbits"][0]["species"] = 3
        projected["orbits"][1]["species"] = 8
        projected["orbits"].reverse()
        with self.assertRaises(SurvivalAuditError):
            validate_topology_invariants_by_id(
                original,
                projected,
                ("o1",),
            )

    def test_contract_keeps_v1_denominator_and_thresholds(self) -> None:
        contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
        self.assertEqual(contract["denominator"]["attempts"], 22)
        self.assertEqual(
            contract["acceptance"]["composition_valid_count_exact"],
            22,
        )
        self.assertEqual(
            contract["acceptance"]["minimum_structural_valid_count"],
            20,
        )
        self.assertEqual(
            contract["acceptance"]["minimum_joint_valid_count"],
            20,
        )
        self.assertTrue(
            contract["threshold_rationale"]["unchanged_from_v1"]
        )
        self.assertEqual(
            tuple(
                contract["representation_alignment"][
                    "expected_reordered_ordinals"
                ]
            ),
            EXPECTED_REORDERED_ORDINALS,
        )
        self.assertFalse(contract["scope"]["slurm"])
        self.assertFalse(contract["scope"]["gpu"])
        self.assertFalse(contract["scope"]["chgnet_or_other_mlip"])
        self.assertFalse(contract["scope"]["sun"])


if __name__ == "__main__":
    unittest.main()
