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

    def test_residual_explanation_distinguishes_visibility_from_salience(self) -> None:
        text = (ROOT / "docs/paper/METHOD_AT_A_GLANCE.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("Global visibility", text)
        self.assertIn("scientific-salience residual", text)
        self.assertIn("triclinic minimum-image", text)
        self.assertIn("zero-initialized", text)

    def test_reader_entrypoint_explains_every_module_and_evidence_profile(self) -> None:
        text = (ROOT / "README.md").read_text(encoding="utf-8")
        for phrase in (
            "Science-Constrained LLM Planner",
            "Plan-Conditioned Crystal Diffusion Language",
            "G2 Periodic-Relational Denoising",
            "Frozen model494 Terminal Diffusion",
            "105/1000 = 10.50%",
            "81/486",
        ):
            self.assertIn(phrase, text)
        self.assertNotIn("tau900", text)


if __name__ == "__main__":
    unittest.main()
