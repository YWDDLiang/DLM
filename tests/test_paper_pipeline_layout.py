from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class PaperPipelineLayoutTest(unittest.TestCase):
    def test_small_facades_and_method_index_exist(self) -> None:
        expected = [
            "src/crystal_dlm/paper_pipeline/planner.py",
            "src/crystal_dlm/paper_pipeline/representation.py",
            "src/crystal_dlm/paper_pipeline/periodic_residual.py",
            "src/crystal_dlm/paper_pipeline/refinement.py",
            "src/crystal_dlm/paper_pipeline/evaluation.py",
            "docs/paper/README.md",
            "docs/paper/METHOD_AT_A_GLANCE.md",
        ]
        for relative in expected:
            path = ROOT / relative
            self.assertTrue(path.is_file(), relative)
            self.assertGreater(path.stat().st_size, 0)
        for relative in expected[:5]:
            self.assertLess(
                len((ROOT / relative).read_text(encoding="utf-8").splitlines()),
                100,
                f"paper facade became a duplicate implementation: {relative}",
            )

    def test_historical_refinement_profile_remains_fixed(self) -> None:
        refinement = (
            ROOT / "src/crystal_dlm/paper_pipeline/refinement.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("tau900", refinement)
        self.assertIn("fixed to tau800", refinement)


if __name__ == "__main__":
    unittest.main()
