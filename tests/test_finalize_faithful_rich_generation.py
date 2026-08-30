import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "finalize_faithful_rich_generation",
    ROOT / "scripts" / "finalize_faithful_rich_generation.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FinalizeFaithfulRichGenerationTest(unittest.TestCase):
    def test_index_summary_retains_missing_attempts(self):
        result = MODULE.summarize_indices(
            parsed_indices=[0, 2, 3],
            refined_indices=[3, 2, 0],
        )
        self.assertEqual(result["parsed_count"], 3)
        self.assertEqual(result["refined_count"], 3)
        self.assertIn(1, result["parsed_missing"])
        self.assertTrue(result["refined_matches_parsed_exactly"])

    def test_contract_has_two_views_and_streams(self):
        self.assertEqual(MODULE.VIEWS, ("H0", "R0S"))
        self.assertEqual(MODULE.STREAMS, (17, 18))
        self.assertEqual(MODULE.DENOMINATOR, 256)


if __name__ == "__main__":
    unittest.main()
