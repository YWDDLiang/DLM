import unittest

from crystal_dlm.sgtc_sampling import validate_sgtc_plan_rows_with_missing


class SGTCMissingPlansTest(unittest.TestCase):
    def test_fixed_ledger_preserves_explicit_planner_failure(self):
        rows = [
            {
                "sample_idx": 0,
                "reduced_composition_identity": "Na:1|Cl:1",
                "plan_state": {"N": 2},
                "prompt": "plan",
                "planner_failed": False,
            },
            {
                "sample_idx": 1,
                "reduced_composition_identity": "Li:2|O:1",
                "plan_state": None,
                "prompt": None,
                "planner_failed": True,
            },
        ]
        result = validate_sgtc_plan_rows_with_missing(rows, expected=2)
        self.assertEqual(result["plan_valid"], 1)
        self.assertEqual(result["plan_failed"], 1)

    def test_missing_row_cannot_hide_nonempty_prompt(self):
        rows = [
            {
                "sample_idx": 0,
                "reduced_composition_identity": "Na:1|Cl:1",
                "plan_state": None,
                "prompt": "replacement",
                "planner_failed": True,
            }
        ]
        with self.assertRaises(ValueError):
            validate_sgtc_plan_rows_with_missing(rows, expected=1)


if __name__ == "__main__":
    unittest.main()
