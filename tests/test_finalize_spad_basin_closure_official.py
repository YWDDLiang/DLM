import importlib.util
import os
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
os.environ["H1_ACTIVE_DENOMINATOR"] = "256"
for path in (ROOT / "eval_runtime",):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
SCRIPT = ROOT / "scripts" / "finalize_spad_basin_closure_official.py"
SPEC = importlib.util.spec_from_file_location("finalize_spad_basin_closure_official_test", SCRIPT)

try:
    MODULE = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(MODULE)
except ModuleNotFoundError:
    MODULE = None


class BasinClosureOfficialSourceTest(unittest.TestCase):
    def test_uncovered_cache_is_conservative_not_dropped(self):
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('hull_status = "official_cache_not_covered"', source)
        self.assertIn('"uncovered_cache_rows_count_as_not_stable": True', source)
        self.assertNotIn("official cache omitted", source)


@unittest.skipIf(MODULE is None, "pymatgen is unavailable in workstation Python")
class BasinClosureOfficialFinalizerTest(unittest.TestCase):
    def test_paired_binary_uses_candidate_minus_baseline_direction(self):
        baseline = [False] * 256
        candidate = [False] * 256
        candidate[3] = True
        result = MODULE.paired_binary(candidate, baseline)
        self.assertEqual(result["wins"], 1)
        self.assertEqual(result["losses"], 0)
        self.assertEqual(result["candidate_only"], 1)

    def test_describe_excludes_nonfinite_values(self):
        result = MODULE.describe([1.0, 3.0, float("nan")])
        self.assertEqual(result["count"], 2)
        self.assertEqual(result["median"], 2.0)


if __name__ == "__main__":
    unittest.main()
