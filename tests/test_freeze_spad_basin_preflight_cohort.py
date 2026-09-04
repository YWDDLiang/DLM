import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "freeze_spad_basin_preflight_cohort.py"
SPEC = importlib.util.spec_from_file_location("freeze_spad_basin_preflight_test", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def row(index, n, elements, counts):
    return {
        "source_row_idx": index,
        "sample_idx": index,
        "outcomes_read": False,
        "prompt": "prompt",
        "teacher_answer": "answer",
        "plan_state": {"N": n, "elements": elements, "counts": counts},
        "species_program": list(reversed(elements)),
    }


class PreflightCohortTest(unittest.TestCase):
    def test_strata_cover_size_and_multiplicity(self):
        self.assertEqual(
            MODULE.stratum(row(0, 16, ["O", "Li", "P"], [8, 4, 4])),
            ("n13_20", "m06_plus", "species3plus"),
        )

    def test_round_robin_is_deterministic_and_unique(self):
        rows = [
            row(index, 4 + index % 16, ["O", "Li"], [2 + index % 3, 2 + index % 13])
            for index in range(200)
        ]
        for item in rows:
            item["plan_state"]["N"] = sum(item["plan_state"]["counts"])
        first = MODULE.select_round_robin(rows, count=128, seed=17)
        second = MODULE.select_round_robin(rows, count=128, seed=17)
        self.assertEqual(
            [value["source_row_idx"] for value in first],
            [value["source_row_idx"] for value in second],
        )
        self.assertEqual(len({value["source_row_idx"] for value in first}), 128)

    def test_round_robin_scales_to_4104_unique_train_rows(self):
        rows = [
            row(index, 13, ["Li", "O"], [7, 6])
            for index in range(5000)
        ]
        selected = MODULE.select_round_robin(rows, count=4104, seed=17)
        self.assertEqual(len(selected), 4104)
        self.assertEqual(
            len({value["source_row_idx"] for value in selected}), 4104
        )
        self.assertTrue(all(value["outcomes_read"] is False for value in selected))

    def test_outcome_read_is_rejected(self):
        value = row(0, 2, ["Li", "O"], [1, 1])
        value["outcomes_read"] = True
        with self.assertRaises(ValueError):
            MODULE.validate_source(value)

    def test_explicit_non_train_split_is_rejected(self):
        value = row(0, 2, ["Li", "O"], [1, 1])
        value["source_split"] = "val"
        with self.assertRaisesRegex(ValueError, "not MP20 train"):
            MODULE.validate_source(value)


if __name__ == "__main__":
    unittest.main()
