from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "slurm/108_c3fd_llama_prospective_eval.sbatch").read_text(
    encoding="utf-8"
)


class Slurm108StaticTest(unittest.TestCase):
    def test_raw_runs_before_refined_on_four_cells(self):
        self.assertLess(SOURCE.index("run_four raw_"), SOURCE.index('run_four ""'))
        for arm in ("F", "M", "raw_F", "raw_M"):
            self.assertIn(arm, SOURCE)

    def test_fixed_denominator_and_no_selection(self):
        self.assertIn("denominator\t256", SOURCE)
        self.assertIn("selection\tfalse", SOURCE)
        self.assertIn("official_query\tfalse", SOURCE)


if __name__ == "__main__":
    unittest.main()
