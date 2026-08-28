from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crystal_dlm.sgtc_eval import paired_energy_stats, quantile, rate_delta_pp


class SGTCEvalTest(unittest.TestCase):
    def test_quantile_and_rate_delta(self):
        self.assertEqual(quantile([0.0, 10.0], 0.25), 2.5)
        self.assertEqual(rate_delta_pp(99, 100, 100), -1.0)

    def test_paired_energy_stats(self):
        control = {(17, 0): -1.0, (17, 1): -2.0, (18, 0): -3.0}
        candidate = {(17, 0): -1.2, (17, 1): -1.9, (18, 0): -3.1}
        result = paired_energy_stats(candidate, control, bootstrap_draws=20, seed=3)
        self.assertEqual(result["paired"], 3)
        self.assertAlmostEqual(result["fraction_candidate_lower"], 2 / 3)
        self.assertAlmostEqual(result["mean_delta"], (-0.2 + 0.1 - 0.1) / 3)


if __name__ == "__main__":
    unittest.main()
