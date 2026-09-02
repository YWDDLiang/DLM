import importlib.util
from pathlib import Path
import unittest

from crystal_dlm.dynamic_crystal import arrays_to_dynamic_answer


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_rollout_matched_pilot_plans",
    ROOT / "scripts" / "build_rollout_matched_pilot_plans.py",
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot import rollout pilot builder")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def source_row(index: int) -> dict:
    answer, _diagnostics = arrays_to_dynamic_answer(
        [5.0, 5.0, 5.0],
        [90.0, 90.0, 90.0],
        ["Li", "O"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    return {
        "source_row_idx": index,
        "reduced_composition_identity": f"Li{index + 1}O",
        "prompt": f"plan {index}",
        "plan_state": {"N": 2, "elements": ["Li", "O"], "counts": [1, 1]},
        "answer": answer,
    }


class RolloutPilotPlanBuilderTest(unittest.TestCase):
    def test_freezes_disjoint_train_and_holdout(self):
        selected = MODULE.select_rows(
            [source_row(index) for index in range(300)],
            count=256,
            seed=20260903,
        )
        train = MODULE.plan_rows(selected[::2], "train")
        holdout = MODULE.plan_rows(selected[1::2], "holdout")
        self.assertEqual(len(train), 128)
        self.assertEqual(len(holdout), 128)
        self.assertFalse(
            {row["reduced_composition_identity"] for row in train}
            & {row["reduced_composition_identity"] for row in holdout}
        )
        self.assertEqual([row["sample_idx"] for row in train], list(range(128)))


if __name__ == "__main__":
    unittest.main()
