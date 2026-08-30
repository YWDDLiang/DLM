import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "finalize_rich_recovery_generation",
    ROOT / "scripts" / "finalize_rich_recovery_generation.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FinalizeRichRecoveryGenerationTest(unittest.TestCase):
    def test_index_summary_preserves_missing_attempts(self):
        result = MODULE.summarize_indices(
            parsed_indices=[0, 2, 3],
            refined_indices=[3, 0, 2],
            denominator=4,
        )
        self.assertEqual(result["parsed_missing"], [1])
        self.assertEqual(result["refined_missing"], [1])
        self.assertTrue(result["refined_matches_parsed_exactly"])

    def test_index_mismatch_is_detected(self):
        result = MODULE.summarize_indices(
            parsed_indices=[0, 2],
            refined_indices=[0, 1],
            denominator=3,
        )
        self.assertFalse(result["refined_matches_parsed_exactly"])

    def test_parent_mapping_keeps_completed_cell(self):
        parent = Path("/parent")
        recovery = Path("/recovery")
        _body, refine = MODULE.cell_paths(parent, recovery, 17, "M0")
        self.assertEqual(refine, parent / "stream17" / "M0" / "refine")
        _body, refine = MODULE.cell_paths(parent, recovery, 18, "R0")
        self.assertEqual(refine, recovery / "stream18" / "R0")


if __name__ == "__main__":
    unittest.main()
