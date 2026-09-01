import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "materialize_btrd_execution_plans",
    ROOT / "scripts" / "materialize_btrd_execution_plans.py",
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot import BTRD Plan materializer")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class BTRDExecutionPlanTest(unittest.TestCase):
    def test_teacher_prefix_is_reindexed_without_losing_mp20_identity(self) -> None:
        rows = [
            {
                "btrd_target_mode": "model494_tau200" if index < 3 else "mp20_anchor",
                "btrd_index": index,
                "btrd_source_row_idx": 100 + index,
                "btrd_exact_identity": f"id{index}",
                "prompt": "prompt",
                "plan_state": {"N": 1, "elements": ["Li"], "counts": [1]},
            }
            for index in range(4)
        ]
        plans = MODULE.materialize(rows, expected_teacher_rows=3)
        self.assertEqual([row["sample_idx"] for row in plans], [0, 1, 2])
        self.assertEqual([row["mp20_source_row_idx"] for row in plans], [100, 101, 102])
        self.assertTrue(all(row["teacher_steps"] == 200 for row in plans))
        self.assertTrue(all(row["outcomes_read"] is False for row in plans))

    def test_nonprefix_teacher_assignment_fails_closed(self) -> None:
        rows = [
            {
                "btrd_target_mode": "model494_tau200",
                "btrd_index": 1,
                "btrd_source_row_idx": 1,
                "btrd_exact_identity": "id",
                "prompt": "p",
                "plan_state": {},
            }
        ]
        with self.assertRaises(ValueError):
            MODULE.materialize(rows, expected_teacher_rows=1)


if __name__ == "__main__":
    unittest.main()
