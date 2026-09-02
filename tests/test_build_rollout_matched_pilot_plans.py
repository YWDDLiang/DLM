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
        "source_split": "train",
        "prompt_schema": "C3FD_NATIVE_PLAN_V2",
        "view": "teacher-native",
        "reduced_composition_identity": f"Li{index + 1}O",
        "prompt": f"plan {index}",
        "plan_state": {
            "N": 2,
            "elements": ["Li", "O"],
            "counts": [1, 1],
            "anion_framework": "oxide",
            "lattice_system": "cubic",
            "spacegroup_bucket": "sg_195_230",
            "volume_per_atom_bin": "volpa_010_014",
        },
        "answer": answer,
    }


class RolloutPilotPlanBuilderTest(unittest.TestCase):
    def test_freezes_disjoint_train_and_holdout(self):
        selected = MODULE.select_rows(
            [source_row(index) for index in range(300)],
            count=256,
            seed=20260903,
        )
        predictions = {
            index: {
                "predictions_by_checkpoint": {
                    "seed17": {
                        "lattice_system": {"prediction": "triclinic"},
                        "spacegroup_bucket": {"prediction": "sg_003_015"},
                        "volume_per_atom_bin": {"prediction": "volpa_015_019"},
                    }
                }
            }
            for index in range(300)
        }
        train = MODULE.plan_rows(
            selected[::2], "train", predictions, planner_seed="seed17"
        )
        holdout = MODULE.plan_rows(
            selected[1::2], "holdout", predictions, planner_seed="seed17"
        )
        self.assertEqual(len(train), 128)
        self.assertEqual(len(holdout), 128)
        self.assertFalse(
            {row["reduced_composition_identity"] for row in train}
            & {row["reduced_composition_identity"] for row in holdout}
        )
        self.assertEqual([row["sample_idx"] for row in train], list(range(128)))
        self.assertTrue(all("\"lattice_system\":\"triclinic\"" in row["prompt"] for row in train))


if __name__ == "__main__":
    unittest.main()
