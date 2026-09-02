import importlib.util
from pathlib import Path
import unittest

from crystal_dlm.dynamic_crystal import arrays_to_dynamic_answer
from crystal_dlm.fixed_slot import tokenize_answer_text


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_force_score_microstudent_data",
    ROOT / "scripts" / "build_force_score_microstudent_data.py",
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot import build_force_score_microstudent_data.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def answer(second_x: float) -> str:
    value, _diagnostics = arrays_to_dynamic_answer(
        [8.0, 8.0, 8.0],
        [90.0, 90.0, 90.0],
        ["O", "O"],
        [[0.0, 0.0, 0.0], [second_x, 0.0, 0.0]],
    )
    return value


class BuildForceScoreMicrostudentDataTest(unittest.TestCase):
    def test_builds_disjoint_base_split_and_paired_answers(self):
        source_answer = answer(0.10)
        target_answer = answer(0.20)
        target_tokens = tokenize_answer_text(target_answer)
        preflight = []
        results = []
        teacher = []
        for base_index in range(64):
            teacher.append(
                {
                    "source_row_idx": base_index,
                    "prompt": f"plan {base_index}",
                    "loss_profile": "fixed_slot",
                }
            )
            for offset in range(8):
                row_index = 8 * base_index + offset
                preflight.append(
                    {
                        "row_index": row_index,
                        "base_index": base_index,
                        "source_row_index": base_index,
                        "dynamic_answer": source_answer,
                        "num_atoms": 2,
                    }
                )
                results.append(
                    {
                        "row_index": row_index,
                        "base_index": base_index,
                        "teacher_mode": "force_projected",
                        "force_quantization": {"tokens": target_tokens},
                        "barrier_quantization": {"tokens": target_tokens},
                        "teacher_complete": True,
                        "selected_delta_eV_per_atom": -0.01,
                    }
                )
        train, holdout, manifest = MODULE.build_rows(preflight, results, teacher)
        self.assertEqual(len(train), 384)
        self.assertEqual(len(holdout), 128)
        self.assertEqual(manifest["base_overlap"], 0)
        self.assertEqual(manifest["train_base_structures"], 48)
        self.assertEqual(manifest["holdout_base_structures"], 16)
        self.assertTrue(all(row["source_answer"] == source_answer for row in train))
        self.assertTrue(all(row["answer"] == target_answer for row in holdout))
        self.assertTrue(all(row["changed_geometry_tokens"] == 1 for row in train))

    def test_rejects_element_change(self):
        source = answer(0.10)
        target, _diagnostics = arrays_to_dynamic_answer(
            [8.0, 8.0, 8.0],
            [90.0, 90.0, 90.0],
            ["O", "Si"],
            [[0.0, 0.0, 0.0], [0.10, 0.0, 0.0]],
        )
        with self.assertRaisesRegex(ValueError, "species order"):
            MODULE.assert_transition_contract(source, target)


if __name__ == "__main__":
    unittest.main()
