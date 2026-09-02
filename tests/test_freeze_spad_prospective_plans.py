from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "freeze_spad_prospective_plans.py"
SPEC = importlib.util.spec_from_file_location("freeze_spad_prospective_plans_tested", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
freeze_rows = MODULE.freeze_rows


def plan(sample_idx: int, element: str, count: int = 2) -> dict:
    return {
        "sample_idx": sample_idx,
        "plan_text": "same rendered schema",
        "plan_state": {"N": count, "elements": [element], "counts": [count]},
        "species_program": [element],
        "species_program_indices": [0],
        "species_program_source": "planner_llama_pointer",
    }


class FreezeActualPlanTests(unittest.TestCase):
    def test_freezes_actual_rows_without_resampling_and_reindexes(self) -> None:
        rows = [plan(10, "Li"), plan(11, "Na"), plan(12, "K")]
        selected, ledger, exclusions = freeze_rows(
            rows, blocked={"Na:2"}, count=2
        )
        self.assertEqual([row["sample_idx"] for row in selected], [0, 1])
        self.assertEqual([row["source_sample_idx"] for row in selected], [10, 12])
        self.assertEqual(
            [row["plan_state"]["elements"] for row in selected], [["Li"], ["K"]]
        )
        self.assertEqual(
            [row["exact_composition_identity"] for row in ledger], ["Li:2", "K:2"]
        )
        self.assertEqual(exclusions, {"blocked_exact": 1})

    def test_rejects_non_pointer_program(self) -> None:
        bad = plan(0, "Li")
        bad["species_program_source"] = "canonical"
        with self.assertRaisesRegex(RuntimeError, "only 0 eligible"):
            freeze_rows([bad], blocked=set(), count=1)


if __name__ == "__main__":
    unittest.main()
