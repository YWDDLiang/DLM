from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "freeze_spad_prospective_plans.py"
SPEC = importlib.util.spec_from_file_location("freeze_spad_prospective_plans_tested", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
freeze_requested = MODULE.freeze_requested


def successful(sample_idx: int, element: str) -> tuple[dict, dict]:
    record = {"sample_idx": sample_idx, "parsed": True, "comp_valid": True, "failure": None}
    plan = {
        "sample_idx": sample_idx,
        "plan_state": {"N": 2, "elements": [element], "counts": [2]},
        "species_program": [element],
        "species_program_indices": [0],
        "species_program_source": "planner_llama_pointer",
        "prompt_schema": "C3FD_NATIVE_PLAN_V2",
    }
    return record, plan


class FreezeRequestedPlanTests(unittest.TestCase):
    def test_retains_failed_request_and_does_not_replace_it(self) -> None:
        record0, plan0 = successful(0, "Li")
        record2, plan2 = successful(2, "Na")
        failed = {"sample_idx": 1, "parsed": False, "comp_valid": False, "failure": "dead_end"}
        plans, ledger, audit = freeze_requested(
            [record0, failed, record2], [plan0, plan2], requested=3
        )
        self.assertEqual([row["sample_idx"] for row in plans], [0, 2])
        self.assertEqual([row["sample_idx"] for row in ledger], [0, 1, 2])
        self.assertFalse(ledger[1]["planner_valid"])
        self.assertEqual(audit["planner_valid"], 2)
        self.assertEqual(audit["planner_invalid"], 1)

    def test_rejects_accounting_disagreement(self) -> None:
        record, _plan = successful(0, "Li")
        with self.assertRaisesRegex(ValueError, "accounting differs"):
            freeze_requested([record], [], requested=1)


if __name__ == "__main__":
    unittest.main()
