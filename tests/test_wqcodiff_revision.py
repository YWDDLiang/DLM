from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from crystal_dlm.wqcodiff.revision import (
    FieldRef,
    RevisionBudget,
    calibrate_revision_threshold_from_recovery,
    load_revision_threshold_lock,
    select_revision_threshold,
)


class RevisionBudgetTests(unittest.TestCase):
    def test_preview_does_not_consume_budget(self) -> None:
        budget = RevisionBudget(12)
        scores = {
            FieldRef("o0", "species"): 0.9,
            FieldRef("o1", "wyckoff_type"): 0.8,
        }
        preview = budget.preview(scores, threshold=0.5, current_field_count=12)
        self.assertEqual(len(preview.selected), 2)
        self.assertEqual(budget.total, 0)
        actual = budget.select(scores, threshold=0.5, current_field_count=12)
        self.assertEqual(actual.selected, preview.selected)
    def test_registered_churn_caps_are_hard(self) -> None:
        budget = RevisionBudget(initial_field_count=20)
        scores = {
            FieldRef(f"o{index}", "species"): 0.99 - 0.001 * index
            for index in range(20)
        }
        selected = []
        for _ in range(20):
            selected.extend(
                budget.select(scores, threshold=0.5, current_field_count=20).selected
            )
        self.assertEqual(len(selected), 10)
        self.assertEqual(budget.total, budget.total_limit)
        self.assertLessEqual(budget.churn, 0.5)
        self.assertTrue(all(budget.count(field) <= 2 for field in scores))

    def test_per_step_cap_is_ten_percent_rounded_up(self) -> None:
        budget = RevisionBudget(initial_field_count=100)
        scores = {
            FieldRef(f"o{index}", "wyckoff_type"): 0.9 for index in range(100)
        }
        decision = budget.select(scores, threshold=0.5, current_field_count=37)
        self.assertEqual(len(decision.selected), 4)


class ThresholdSelectionTests(unittest.TestCase):
    def test_threshold_respects_clean_false_remask_before_net_gain(self) -> None:
        result = select_revision_threshold(
            clean_scores=[0.55] * 10 + [0.1] * 190,
            wrong_scores=[0.95] * 80 + [0.65] * 20,
        )
        self.assertGreaterEqual(result.threshold, 0.6)
        self.assertLessEqual(result.clean_false_remask_rate, 0.05)

    def test_no_eligible_registered_threshold_is_gate_failure(self) -> None:
        with self.assertRaisesRegex(ValueError, "no registered threshold"):
            select_revision_threshold(
                clean_scores=[0.99] * 100,
                wrong_scores=[0.99] * 100,
            )

    def test_attempt_level_calibration_freezes_and_reloads_the_registered_winner(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = root / "recovery.jsonl"
            net = {0.5: 4, 0.6: 2, 0.7: 3, 0.8: 3, 0.9: 1}
            with artifacts.open("x", encoding="utf-8") as handle:
                for threshold in (0.5, 0.6, 0.7, 0.8, 0.9):
                    for operator, material_id in (("none", "clean"), ("joint", "wrong")):
                        selected = 1 if operator == "none" and threshold == 0.5 else 0
                        row = {
                            "schema": "wqcodiff_recovery_attempt_v1",
                            "attempt_id": f"{threshold}-{operator}",
                            "material_id": material_id,
                            "method": "M-WQ-STRAT-GEO",
                            "corruption_seed": 101,
                            "corruption_level": 0.7,
                            "operator": operator,
                            "geometry_condition": "clean" if operator == "none" else "noisy",
                            "schedule": "geometry-adaptive",
                            "control": "none",
                            "revision_threshold": threshold,
                            "status": "succeeded",
                            "initial_revisable_field_count": 10,
                            "mechanism": {
                                "revision_selected_actions": selected,
                                "net_correction": net[threshold] if operator == "joint" else 0,
                                "wrong_to_right": net[threshold] if operator == "joint" else 0,
                                "right_to_wrong": 0,
                            },
                        }
                        handle.write(json.dumps(row, sort_keys=True) + "\n")
            lock_path = root / "threshold_lock.json"
            result = calibrate_revision_threshold_from_recovery(
                [artifacts],
                output=lock_path,
                protocol_name="p",
                protocol_sha256="h" * 64,
            )
            self.assertEqual(result["selected_threshold"], 0.8)
            loaded = load_revision_threshold_lock(
                lock_path,
                protocol_name="p",
                protocol_sha256="h" * 64,
            )
            self.assertEqual(loaded["selected_threshold"], 0.8)


if __name__ == "__main__":
    unittest.main()
