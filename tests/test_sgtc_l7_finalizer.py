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


if __name__ == "__main__":
    unittest.main()
