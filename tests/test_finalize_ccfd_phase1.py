import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "finalize_ccfd_phase1", ROOT / "scripts" / "finalize_ccfd_phase1.py"
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load CCFD Phase-1 finalizer")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CCFDPhase1FinalizerTest(unittest.TestCase):
    def test_tvd(self):
        self.assertAlmostEqual(MODULE.tvd({"a": 5, "b": 5}, {"a": 10}), 0.5)

    def test_exact_mcnemar_direction(self):
        result = MODULE.exact_mcnemar([True, False, False], [False, True, True])
        self.assertEqual(result["f0_only"], 1)
        self.assertEqual(result["f1_only"], 2)

    def test_paired_ci_reports_delta(self):
        result = MODULE.paired_bootstrap_ci([False] * 20, [True] * 20, draws=100)
        self.assertEqual(result["delta"], 1.0)
        self.assertGreater(result["low"], 0.0)


if __name__ == "__main__":
    unittest.main()
