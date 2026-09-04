import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "evaluate_spad_basin_closure_raw.py"
SPEC = importlib.util.spec_from_file_location("evaluate_spad_basin_closure_raw_test", SCRIPT)

try:
    import numpy  # noqa: F401
    for path in (ROOT / "src", SCRIPT.parent):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    MODULE = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(MODULE)
except ModuleNotFoundError:
    MODULE = None


@unittest.skipIf(MODULE is None, "numpy is unavailable in the workstation Python")
class BasinClosureRawScreenTest(unittest.TestCase):
    def test_volume_bin_parser_is_conservative(self):
        self.assertEqual(MODULE.volume_bin_bounds("volpa_010_014"), (10.0, 15.0))
        self.assertIsNone(MODULE.volume_bin_bounds("unknown"))
        self.assertIsNone(MODULE.volume_bin_bounds("volpa_bad_015"))
        self.assertIsNone(MODULE.volume_bin_bounds("volpa_010_015"))

    def test_expected_composition_requires_exact_N(self):
        plan = {"N": 3, "elements": ["Li", "O"], "counts": [2, 1]}
        self.assertEqual(MODULE.expected_composition(plan), (("Li", 2), ("O", 1)))
        self.assertIsNone(MODULE.expected_composition({**plan, "N": 4}))

    def test_distance_tail_preserves_short_contact_counts(self):
        result = MODULE.distance_tail([0.6, 0.8, 1.2, 2.0])
        self.assertEqual(result["known"], 4)
        self.assertEqual(result["below_0p75_A"], 1)
        self.assertEqual(result["below_1p00_A"], 2)
        self.assertEqual(result["below_1p50_A"], 3)

    def test_minimum_distance_includes_nonzero_self_image(self):
        class Lattice:
            matrix = numpy.eye(3) * 0.4

        class SingleSite:
            lattice = Lattice()

            def __len__(self):
                return 1

        self.assertAlmostEqual(MODULE.minimum_distance(SingleSite()), 0.4)

    def test_paired_binary_retains_all_ordinals(self):
        baseline = {0: {"ok": True}, 1: {"ok": False}, 2: {"ok": True}}
        closure = {0: {"ok": True}, 1: {"ok": True}, 2: {"ok": False}}
        result = MODULE.paired_binary(closure, baseline, "ok")
        self.assertEqual(result, {"paired": 3, "wins": 1, "losses": 1, "both_true": 1, "both_false": 0})

    def test_paired_tristate_does_not_turn_unknown_into_false(self):
        baseline = {0: {"v": True}, 1: {"v": None}, 2: {"v": False}}
        closure = {0: {"v": None}, 1: {"v": None}, 2: {"v": True}}
        result = MODULE.paired_tristate(closure, baseline, "v")
        self.assertEqual(result["paired"], 3)
        self.assertEqual(result["paired_known"], 1)
        self.assertEqual(result["wins"], 1)
        self.assertEqual(result["closure_unknown"], 2)
        self.assertEqual(result["BS_unknown"], 1)
        self.assertEqual(result["both_unknown"], 1)

    def test_paired_delta_uses_closure_minus_baseline(self):
        baseline = {
            0: {"x": 2.0, "cluster": "a", "composition_cluster": "Li:1"},
            1: {"x": 5.0, "cluster": "b", "composition_cluster": "O:1"},
        }
        closure = {
            0: {"x": 1.0, "cluster": "a", "composition_cluster": "Li:1"},
            1: {"x": 7.0, "cluster": "b", "composition_cluster": "O:1"},
        }
        result = MODULE.paired_delta(closure, baseline, field="x")
        self.assertEqual(result["paired_known"], 2)
        self.assertEqual(result["composition_mismatch_excluded"], 0)
        self.assertEqual(result["lower_count"], 1)
        self.assertEqual(result["higher_count"], 1)
        self.assertEqual(result["delta"]["median"], 0.5)

    def test_paired_delta_retains_but_does_not_compare_wrong_composition(self):
        baseline = {0: {"x": 2.0, "composition_cluster": "Li:1"}}
        closure = {0: {"x": 1.0, "composition_cluster": "O:1"}}
        result = MODULE.paired_delta(closure, baseline, field="x")
        self.assertEqual(result["paired_known"], 0)
        self.assertEqual(result["composition_mismatch_excluded"], 1)


if __name__ == "__main__":
    unittest.main()
