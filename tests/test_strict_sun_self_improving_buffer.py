import unittest

from scripts.build_strict_sun_self_improving_buffer import accepted_tier_for_row, successful_original_indices


class StrictSunSelfImprovingBufferTest(unittest.TestCase):
    def test_successful_original_indices_remove_unsupported_and_relax_failures(self):
        summary = {
            "num_structures": 8,
            "unsupported_records": [{"index": 1}, {"index": 6}],
            "relax_failed_indices": [3],
        }
        self.assertEqual(successful_original_indices(summary, 5), [0, 2, 4, 5, 7])

    def test_fallback_when_summary_is_shorter_than_detailed(self):
        summary = {"num_structures": 1, "unsupported_records": [], "relax_failed_indices": []}
        self.assertEqual(successful_original_indices(summary, 3), [0, 1, 2])

    def test_accepts_strict_before_meta(self):
        self.assertEqual(
            accepted_tier_for_row(
                ehull=-0.01,
                novel_unique=True,
                strict_threshold=0.0,
                meta_threshold=0.1,
                accepted_tiers={"strict", "meta"},
            ),
            "strict",
        )

    def test_accepts_meta_when_configured(self):
        self.assertEqual(
            accepted_tier_for_row(
                ehull=0.05,
                novel_unique=True,
                strict_threshold=0.0,
                meta_threshold=0.1,
                accepted_tiers={"strict", "meta"},
            ),
            "meta",
        )

    def test_rejects_meta_when_not_configured(self):
        self.assertIsNone(
            accepted_tier_for_row(
                ehull=0.05,
                novel_unique=True,
                strict_threshold=0.0,
                meta_threshold=0.1,
                accepted_tiers={"strict"},
            )
        )


if __name__ == "__main__":
    unittest.main()
