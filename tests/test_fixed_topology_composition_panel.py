from __future__ import annotations

import unittest

from crystal_dlm.wqcodiff.composition_projection import (
    FixedTopologyCompositionProjector,
)
from crystal_dlm.wqcodiff.state import OrbitState, StratifiedState
from diagnostics.run_fixed_topology_composition_panel import (
    ProjectionPanelError,
    run_projection_panel,
)


LITHIUM = 3
OXYGEN = 8


def make_state(attempt_id: str, species: tuple[int, ...]) -> StratifiedState:
    return StratifiedState(
        space_group=1,
        lattice_system="triclinic",
        lattice_chart=(4.0, 4.1, 4.2, 90.0, 91.0, 92.0),
        orbits=tuple(
            OrbitState(f"o-{index}", index, value, 1, 0, (), 1)
            for index, value in enumerate(species)
        ),
        attempt_id=attempt_id,
    )


def classifier(elems, counts):
    values = dict(zip(elems, counts))
    if values == {LITHIUM: 2, OXYGEN: 1}:
        return {"valid": True, "reason": "charge_neutral_pauling_valid"}
    if values == {LITHIUM: 1, OXYGEN: 1}:
        return {"valid": False, "reason": "pauling_fail_or_ratio_rejected"}
    return {"valid": False, "reason": "charge_neutrality_fail"}


class FixedTopologyCompositionPanelTests(unittest.TestCase):
    def rows(self):
        no_neutral = make_state(
            "no-neutral",
            (LITHIUM, OXYGEN, OXYGEN),
        )
        pauling = make_state("pauling", (LITHIUM, OXYGEN))
        control = make_state(
            "control",
            (LITHIUM, LITHIUM, OXYGEN),
        )
        return [
            {
                "attempt_id": no_neutral.attempt_id,
                "panel_group": "no_neutral",
                "state": no_neutral.to_dict(canonical_storage=False),
            },
            {
                "attempt_id": pauling.attempt_id,
                "panel_group": "pauling_only",
                "state": pauling.to_dict(canonical_storage=False),
            },
            {
                "attempt_id": control.attempt_id,
                "panel_group": "valid_control",
                "state": control.to_dict(canonical_storage=False),
            },
        ]

    def test_small_mechanism_panel_passes(self) -> None:
        outputs, report = run_projection_panel(
            self.rows(),
            projector=FixedTopologyCompositionProjector(classifier=classifier),
            expected_group_counts={
                "no_neutral": 1,
                "pauling_only": 1,
                "valid_control": 1,
            },
            minimum_no_neutral_recovered=1,
        )
        self.assertTrue(report["ok"])
        self.assertEqual(report["no_neutral_recovered"], 1)
        self.assertEqual(report["valid_controls_identity"], 1)
        self.assertEqual(report["pauling_only_identity"], 1)
        self.assertEqual(len(outputs), 3)
        self.assertEqual(report["mlip_calls"], 0)
        self.assertEqual(report["llm_calls"], 0)
        self.assertFalse(report["retry_or_replacement_used"])

    def test_duplicate_attempt_id_fails_before_outputs(self) -> None:
        rows = self.rows()
        rows[1]["attempt_id"] = rows[0]["attempt_id"]
        rows[1]["state"]["attempt_id"] = rows[0]["attempt_id"]
        with self.assertRaisesRegex(ProjectionPanelError, "duplicate"):
            run_projection_panel(
                rows,
                projector=FixedTopologyCompositionProjector(classifier=classifier),
                expected_group_counts={
                    "no_neutral": 1,
                    "pauling_only": 1,
                    "valid_control": 1,
                },
                minimum_no_neutral_recovered=1,
            )

    def test_group_source_mismatch_fails_acceptance(self) -> None:
        rows = self.rows()
        rows[1]["panel_group"] = "valid_control"
        _, report = run_projection_panel(
            rows,
            projector=FixedTopologyCompositionProjector(classifier=classifier),
            expected_group_counts={
                "no_neutral": 1,
                "pauling_only": 0,
                "valid_control": 2,
            },
            minimum_no_neutral_recovered=1,
        )
        self.assertFalse(report["ok"])
        self.assertFalse(report["checks"]["source_reason_contract"])

    def test_state_attempt_id_must_match_outer_row(self) -> None:
        rows = self.rows()
        rows[0]["state"]["attempt_id"] = "different"
        with self.assertRaisesRegex(ProjectionPanelError, "differs"):
            run_projection_panel(
                rows,
                projector=FixedTopologyCompositionProjector(classifier=classifier),
                expected_group_counts={
                    "no_neutral": 1,
                    "pauling_only": 1,
                    "valid_control": 1,
                },
                minimum_no_neutral_recovered=1,
            )


if __name__ == "__main__":
    unittest.main()
