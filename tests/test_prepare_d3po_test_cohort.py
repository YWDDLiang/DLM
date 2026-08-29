import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "prepare_d3po_test_cohort", ROOT / "scripts/prepare_d3po_test_cohort.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PrepareD3POTestCohortTest(unittest.TestCase):
    def test_forbidden_outcomes_are_found_recursively(self):
        value = {"plan_state": {"N": 2}, "nested": [{"official_e_hull": 0.0}]}
        self.assertEqual(
            MODULE.forbidden_paths(value), ["nested[0].official_e_hull"]
        )

    def test_certificate_validity_is_not_mistaken_for_science_outcome(self):
        value = {
            "plan_state": {
                "ccfd_v2_certificate": {"benchmark_valid": True}
            }
        }
        self.assertEqual(MODULE.forbidden_paths(value), [])

    def test_execution_index_is_global_while_source_index_is_preserved(self):
        text = (ROOT / "scripts/prepare_d3po_test_cohort.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('converted["source_sample_idx"] = int(row["sample_idx"])', text)
        self.assertIn('converted["sample_idx"] = row_index', text)


if __name__ == "__main__":
    unittest.main()
