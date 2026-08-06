import unittest

from scripts.build_mp20_ehull_weighted_sft_data import ehull_tier, generated_self_improving_tier, select_repeats


class MP20EhullWeightedDataTest(unittest.TestCase):
    def test_mp20_zero_and_generated_negative_share_high_tier(self):
        self.assertEqual(ehull_tier(0.0, eq0_tol=1e-12), "tier_high")
        self.assertEqual(ehull_tier(-0.01, eq0_tol=1e-12), "tier_high")
        self.assertEqual(ehull_tier(0.0005, eq0_tol=1e-12), "tier_mid_high")

    def test_formula_cap_limits_repeats_without_deleting_base_rows(self):
        rows = [
            {
                "sample_weight_tier": "tier_high",
                "sample_weight": 1.0,
                "formula": "Li2O",
                "chemsys": "Li-O",
                "source_row_idx": idx,
            }
            for idx in range(3)
        ]
        selected, summary = select_repeats(
            rows,
            target_count=5,
            tier_weights={"tier_high": 1.0},
            seed=7,
            max_formula_repeats=2,
            max_chemsys_repeats=10,
            mask_mix={"normal": 1.0},
            selection_role="ehull_weighted_repeat",
        )
        self.assertEqual(len(selected), 2)
        self.assertEqual(summary["selected_count"], 2)
        self.assertEqual({row["selection_role"] for row in selected}, {"ehull_weighted_repeat"})

    def test_generated_meta_self_improving_is_optional_mid_high(self):
        row = {"novel_unique": True, "meta_sun": True, "sample_weight_tier": "tier_mid_high"}
        self.assertIsNone(generated_self_improving_tier(row, ehull=0.05, include_meta=False))
        self.assertEqual(
            generated_self_improving_tier(row, ehull=0.05, include_meta=True),
            ("tier_mid_high", False, True),
        )

    def test_generated_strict_self_improving_stays_high(self):
        row = {"novel_unique": True, "strict_sun": True}
        self.assertEqual(
            generated_self_improving_tier(row, ehull=-0.02, include_meta=True),
            ("tier_high", True, True),
        )


if __name__ == "__main__":
    unittest.main()
