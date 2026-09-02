from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts" / "pad_spad_body_to_planner_denominator.py"
SPEC = importlib.util.spec_from_file_location("pad_spad_body_tested", PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class PadPlannerFailureTests(unittest.TestCase):
    def test_failed_plan_becomes_failed_body_without_replacement(self) -> None:
        ledger = [
            {"sample_idx": 0, "planner_valid": True, "failure": None},
            {"sample_idx": 1, "planner_valid": False, "failure": "dead_end"},
        ]
        rows = MODULE.pad_rows(
            [{"sample_idx": 0, "parsed": True, "cif": "ok"}], ledger, denominator=2
        )
        self.assertEqual([row["sample_idx"] for row in rows], [0, 1])
        self.assertTrue(rows[0]["parsed"])
        self.assertFalse(rows[1]["parsed"])
        self.assertEqual(rows[1]["reason"], "planner:dead_end")


if __name__ == "__main__":
    unittest.main()
