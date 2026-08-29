import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "finalize_sgtc_l7_official", ROOT / "scripts/finalize_sgtc_l7_official.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SGTCL7FinalizerTest(unittest.TestCase):
    def test_absolute_and_secondary_gates(self):
        absolute = MODULE.absolute_gate(
            {"strict_attempt_rate": 0.10, "meta_attempt_rate": 0.50}
        )
        self.assertTrue(absolute["eligible"])
        floors = MODULE.secondary_floor_gate(
            {
                "body_rate": -0.03,
                "direct_joint_rate": -0.03,
                "novel_rate": -0.05,
                "unique_rate": -0.05,
                "strict_retention": -0.10,
                "meta_retention": -0.10,
            }
        )
        self.assertTrue(floors["eligible"])

    def test_strict_stable_direction_is_predeclared(self):
        self.assertTrue(
            MODULE.direction_gate(
                {"strict_attempt_rate": -0.01, "meta_attempt_rate": 0.001}
            )["strict_or_meta_positive_other_ge_minus_1pp"]
        )
        self.assertFalse(
            MODULE.direction_gate(
                {"strict_attempt_rate": -0.011, "meta_attempt_rate": 0.001}
            )["strict_or_meta_positive_other_ge_minus_1pp"]
        )

    def test_paired_delta_interval(self):
        summary = MODULE.paired_delta_summary(
            [False, False, True, True], [False, True, True, False]
        )
        self.assertEqual(summary["known_both"], 4)
        self.assertEqual(summary["candidate_minus_control"], 0.0)
        self.assertLess(summary["wald95_lower"], 0.0)
        self.assertGreater(summary["wald95_upper"], 0.0)

    def test_continuous_distribution_reports_quantiles_and_ecdf(self):
        summary = MODULE.continuous_distribution([0.0, 0.01, 0.05, 0.20])
        self.assertEqual(summary["known"], 4)
        self.assertAlmostEqual(summary["quantiles"]["q50"], 0.03)
        self.assertEqual(summary["ecdf"]["le_0p05"]["count"], 3)
        self.assertAlmostEqual(summary["ecdf"]["le_0p05"]["rate"], 0.75)

    def test_continuous_pair_summary_is_candidate_minus_control(self):
        control = [
            {
                "ordinal": 0,
                "chemsys": "A-B",
                "official_hull_status": "known",
                "official_e_above_hull": 0.10,
            },
            {
                "ordinal": 1,
                "chemsys": "A-C",
                "official_hull_status": "known",
                "official_e_above_hull": 0.20,
            },
        ]
        candidate = [
            {
                "ordinal": 0,
                "chemsys": "A-B",
                "official_hull_status": "known",
                "official_e_above_hull": 0.05,
            },
            {
                "ordinal": 1,
                "chemsys": "A-C",
                "official_hull_status": "known",
                "official_e_above_hull": 0.25,
            },
        ]
        summary = MODULE.continuous_pair_summary(
            control,
            candidate,
            field="official_e_above_hull",
            require_official_known=True,
        )
        self.assertEqual(summary["known_both"], 2)
        self.assertAlmostEqual(summary["candidate_minus_control_mean"], 0.0)
        self.assertAlmostEqual(summary["fraction_lower"], 0.5)
        self.assertEqual(
            (summary["lower"], summary["higher"], summary["ties"]), (1, 1, 0)
        )

    def test_composition_identity_is_reduced_and_order_independent(self):
        left = MODULE.composition_identity(["O", "Li"], [2, 4])
        right = MODULE.composition_identity(["Li", "O"], [2, 1])
        self.assertEqual(left, "Li:2|O:1")
        self.assertEqual(left, right)


if __name__ == "__main__":
    unittest.main()
