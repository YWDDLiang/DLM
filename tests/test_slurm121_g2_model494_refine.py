from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "slurm" / "121_g2_model494_refine.sbatch"


class G2Model494SlurmTest(unittest.TestCase):
    def test_reuses_body_and_freezes_refiner(self) -> None:
        text = SCRIPT.read_text()
        self.assertIn("source_body_job\t39112", text)
        self.assertIn("body_rerun\tfalse", text)
        self.assertIn("REFINER_SHA=573e9b10", text)
        self.assertIn("tau\t800", text)
        self.assertIn("refiner_seed\t101117", text)
        self.assertIn("--seed-by-sample-index", text)
        self.assertIn("--diff-steps 800", text)
        self.assertIn("--max-proposals 256", text)

    def test_no_selection_or_official_query(self) -> None:
        text = SCRIPT.read_text()
        self.assertIn("retry_rerank_replacement\tfalse", text)
        self.assertIn("official_query\tfalse", text)
        self.assertNotIn("best-of", text.lower())


if __name__ == "__main__":
    unittest.main()
