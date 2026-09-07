"""Metadata mapping tests only; no Direct or scientific evaluator is called."""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "r03_direct_view_contract", ROOT / "operations/r03_c3fd_main_20260907/export_direct_view.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def row(index, *, success=True):
    return {
        "schema": MODULE.INPUT_SCHEMA, "purpose": "evaluation", "source_split": "evaluation",
        "method_id": "G", "endpoint": "tau800", "evaluation_ordinal": index,
        "attempt_id": f"request:{100 + index}", "trajectory_id": f"G:tau800:{100 + index}",
        "sample_idx": 100 + index, "success": success, "parseable": success,
        "source_attempt_status": "complete" if success else "controller_failure",
        "planner_seed": 17, "artifact_error": None if success else "controller failure retained",
        "structure": {"lattice": {"matrix": [[3.1234567890123456, 0, 0], [0, 3, 0], [0, 0, 3]]},
                      "sites": [{"species": [{"element": "Si", "occu": 1}], "abc": [0.1234567890123456, 0, 0]}]},
    }


class DirectMetadataViewTest(unittest.TestCase):
    def test_preserves_continuous_structure_and_every_request_without_mutating_source(self):
        source = [row(0), row(1, success=False)]
        before = copy.deepcopy(source)
        view, report = MODULE.direct_view(source, expected_denominator=2)
        self.assertEqual(source, before)
        self.assertEqual(len(view), 2)
        self.assertEqual([item["sample_idx"] for item in view], [100, 101])
        self.assertEqual([item["ordinal"] for item in view], [0, 1])
        self.assertIs(view[0]["structure"], source[0]["structure"])
        self.assertEqual(view[0]["structure"], before[0]["structure"])
        self.assertEqual(report["generation_succeeded"], 1)
        self.assertEqual(report["failed_requests_retained"], 1)

    def test_failed_row_with_stale_structure_preview_is_never_revived(self):
        source = [row(0, success=False)]
        source[0]["source_body_status"] = "succeeded"
        view, _report = MODULE.direct_view(source, expected_denominator=1)
        self.assertEqual(view[0]["status"], "failed")
        self.assertEqual(view[0]["reason"], "controller failure retained")
        self.assertIs(view[0]["structure"], source[0]["structure"])

    def test_success_claim_needs_actual_structure_and_cannot_hide_controller_failure(self):
        for change in ({"structure": None}, {"parseable": False}, {"source_attempt_status": "controller_failure"}):
            source = row(0)
            source.update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                MODULE.direct_view([source], expected_denominator=1)

    def test_denominator_order_and_global_identity_cannot_be_repaired(self):
        with self.assertRaises(ValueError):
            MODULE.direct_view([row(0)], expected_denominator=2)
        with self.assertRaises(ValueError):
            MODULE.direct_view([row(1), row(0)], expected_denominator=2)
        source = [row(0), row(1)]
        source[1]["sample_idx"] = source[0]["sample_idx"]
        with self.assertRaises(ValueError):
            MODULE.direct_view(source, expected_denominator=2)

    def test_mixed_endpoint_or_method_or_training_source_is_rejected(self):
        for change in ({"endpoint": "native"}, {"method_id": "R"}, {"purpose": "train"}):
            source = [row(0), row(1)]
            source[1].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                MODULE.direct_view(source, expected_denominator=2)


if __name__ == "__main__":
    unittest.main()
