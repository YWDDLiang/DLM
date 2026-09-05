import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location("evaluate_programmed_paths", Path(__file__).resolve().parents[1] / "scripts/evaluate_programmed_paths.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class EvaluationMetricsTest(unittest.TestCase):
    def test_energy_definition_and_verified_subset_remain_distinct(self):
        result = MODULE.classify_stability(verified=False, energy=-2., hull_energy=-1.9, novel=True, unique=True)
        self.assertTrue(result["strict_sun"])
        self.assertFalse(result["verified_strict_sun"])

    def test_unknown_energy_or_hull_is_never_imputed_as_stable(self):
        for energy, hull in ((None, -1.), (-1., None), (float("nan"), -1.)):
            result = MODULE.classify_stability(verified=True, energy=energy, hull_energy=hull, novel=True, unique=True)
            self.assertIsNone(result["e_above_hull_eV_atom"])
            self.assertFalse(result["strict_sun"])
            self.assertFalse(result["meta_sun"])

    def test_stability_is_not_sun_without_novelty_and_uniqueness(self):
        result = MODULE.classify_stability(verified=True, energy=-2., hull_energy=-2., novel=False, unique=True)
        self.assertTrue(result["verified_strict_stable"])
        self.assertFalse(result["strict_sun"])


if __name__ == "__main__":
    unittest.main()
