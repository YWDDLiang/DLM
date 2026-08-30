import importlib.util
import math
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_si_lwa_intent_data",
    ROOT / "scripts" / "build_si_lwa_intent_data.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


try:
    import numpy  # noqa: F401
except ModuleNotFoundError:
    numpy = None


class SILWAIntentDataTest(unittest.TestCase):
    def test_cn_histogram_is_normalized_and_uses_terminal_bin(self):
        histogram = MODULE.cn_histogram([1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 18.0])
        self.assertEqual(histogram, [0.125] * 8)
        self.assertAlmostEqual(sum(histogram), 1.0)

    @unittest.skipIf(numpy is None, "numpy is unavailable")
    def test_vpa_quantiles_assign_all_eight_classes(self):
        values = [float(index) for index in range(80)]
        edges = MODULE.quantile_edges(values)
        classes = {MODULE.assign_quantile(value, edges) for value in values}
        self.assertEqual(classes, set(range(8)))
        self.assertEqual(edges, sorted(edges))

    @unittest.skipIf(numpy is None, "numpy is unavailable")
    def test_representative_medoids_are_observed_distinct_and_deterministic(self):
        rows = []
        for cluster in range(8):
            for offset in range(3):
                row = [0.0] * 8
                row[cluster] = 1.0 - 0.01 * offset
                row[(cluster + 1) % 8] = 0.01 * offset
                rows.append(row)
        first_indices, first = MODULE.fit_representative_medoids(rows, seed=82000)
        second_indices, second = MODULE.fit_representative_medoids(rows, seed=82000)
        self.assertEqual(first_indices, second_indices)
        self.assertEqual(first, second)
        self.assertEqual(len(set(first_indices)), 8)
        self.assertTrue(all(row in rows for row in first))
        self.assertTrue(all(math.isclose(sum(row), 1.0) for row in first))

    def test_cli_does_not_accept_test_or_holdout_sources(self):
        options = {
            option
            for action in MODULE.build_parser()._actions
            for option in action.option_strings
        }
        self.assertFalse(any("test" in option.lower() for option in options))
        self.assertFalse(any("holdout" in option.lower() for option in options))


if __name__ == "__main__":
    unittest.main()
