import unittest

from crystal_dlm.ordinal_rng import (
    derive_ordinal_seed,
    ordered_ordinal_records,
    sha256_text,
)


class OrdinalRngTests(unittest.TestCase):
    def test_seed_is_stable_and_rank_independent(self):
        first = derive_ordinal_seed(
            17,
            sample_idx=42,
            stage="planner_sampling",
            role="shared",
        )
        second = derive_ordinal_seed(
            17,
            sample_idx=42,
            stage="planner_sampling",
            role="shared",
        )
        self.assertEqual(first, second)
        self.assertEqual(first, 5664000075365521389)

    def test_seed_changes_by_registered_identity(self):
        base = derive_ordinal_seed(17, sample_idx=0, stage="body", role="shared")
        self.assertNotEqual(
            base,
            derive_ordinal_seed(17, sample_idx=1, stage="body", role="shared"),
        )
        self.assertNotEqual(
            base,
            derive_ordinal_seed(17, sample_idx=0, stage="refiner", role="shared"),
        )
        self.assertNotEqual(
            base,
            derive_ordinal_seed(17, sample_idx=0, stage="body", role="diagnostic"),
        )

    def test_seed_rejects_invalid_identity(self):
        with self.assertRaises(ValueError):
            derive_ordinal_seed(17, sample_idx=-1, stage="planner")
        with self.assertRaises(ValueError):
            derive_ordinal_seed(17, sample_idx=0, stage="")
        with self.assertRaises(ValueError):
            derive_ordinal_seed(17, sample_idx=0, stage="planner", role="")

    def test_ordered_records_are_strictly_sorted(self):
        records = [
            {"sample_idx": 2, "value": "c"},
            {"sample_idx": 0, "value": "a"},
            {"sample_idx": 1, "value": "b"},
        ]
        ordered = ordered_ordinal_records(
            records,
            expected_count=3,
            require_complete=True,
        )
        self.assertEqual([row["sample_idx"] for row in ordered], [0, 1, 2])

    def test_ordered_records_reject_duplicates_missing_and_out_of_range(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            ordered_ordinal_records(
                [{"sample_idx": 0}, {"sample_idx": 0}],
                expected_count=1,
            )
        with self.assertRaisesRegex(ValueError, "missing"):
            ordered_ordinal_records(
                [{"sample_idx": 0}, {"sample_idx": 2}],
                expected_count=3,
                require_complete=True,
            )
        with self.assertRaisesRegex(ValueError, "outside"):
            ordered_ordinal_records(
                [{"sample_idx": 3}],
                expected_count=3,
            )

    def test_sha256_text_does_not_normalize(self):
        self.assertNotEqual(sha256_text("formula: Li2O"), sha256_text("formula: Li2O\n"))


if __name__ == "__main__":
    unittest.main()
