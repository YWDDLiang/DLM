import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location("diagnose_programmed_basins", Path(__file__).resolve().parents[1] / "scripts/diagnose_programmed_basins.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def candidate(a, b, weight):
    return dict(raw_energy=a + b, terminal_energy=b, weight=weight, verified=True)


class TeacherDistributionDiagnosticTest(unittest.TestCase):
    def test_global_gain_can_coexist_with_a_worsening_condition(self):
        groups = [{"group_id": "gain", "candidates": [candidate(1, 1, .75), candidate(3, 3, .25)]},
                  {"group_id": "worse", "candidates": [candidate(1, 1, .4), candidate(3, 3, .6)]},
                  {"group_id": "single", "candidates": [candidate(2, 1, 1.)]},
                  {"group_id": "missing", "candidates": [{"verified": False, "weight": 0.}]}]
        summary = MODULE.teacher_reweighting_diagnostic(groups)["summary"]
        self.assertEqual(summary["verified_conditions"], 3)
        self.assertEqual(summary["unverified_conditions"], 1)
        self.assertEqual(summary["single_verified_path_conditions"], 1)
        self.assertEqual(summary["both_objectives_improve_conditions"], 1)
        self.assertEqual(summary["either_objective_worsens_conditions"], 1)
        self.assertAlmostEqual(summary["A"]["mean_delta_eV_atom"], -.1)
        self.assertEqual(summary["A"]["largest_positive_gain_share"], 1.)
        self.assertAlmostEqual(summary["mean_total_variation_from_uniform"], .35 / 3)

    def test_repeated_labels_are_not_counted_as_energy_contrast(self):
        summary = MODULE.teacher_reweighting_diagnostic([
            {"group_id": "same", "candidates": [candidate(2, 1, .5), candidate(2, 1, .5)]}])["summary"]
        self.assertEqual(summary["multiple_verified_path_conditions"], 1)
        self.assertEqual(summary["varying_energy_label_conditions"], 0)
        self.assertEqual(summary["reweighted_conditions"], 0)
        self.assertEqual(summary["A"]["mean_delta_eV_atom"], 0.)
        self.assertIsNone(summary["A"]["largest_positive_gain_share"])

    def test_unknown_labels_cannot_carry_weight(self):
        with self.assertRaises(ValueError):
            MODULE.teacher_reweighting_diagnostic([
                {"group_id": "bad", "candidates": [{"verified": False, "weight": 1.}]}])


if __name__ == "__main__":
    unittest.main()
