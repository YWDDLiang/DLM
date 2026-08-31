from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "slurm/104_c3fd_prospective_seed21_source.sbatch").read_text(
    encoding="utf-8"
)


class Slurm104StaticTest(unittest.TestCase):
    def test_cpu_only_seed21_source(self):
        self.assertNotIn("#SBATCH --gres", SOURCE)
        self.assertIn("sampling_seed\t21", SOURCE)
        self.assertIn("--seed 21", SOURCE)
        self.assertIn("requested\t1000", SOURCE)

    def test_source_is_outcome_blind_and_unselected(self):
        self.assertIn("selection_or_outcomes\tfalse", SOURCE)
        self.assertNotIn("e_above_hull", SOURCE)
        self.assertNotIn("best", SOURCE.lower())


if __name__ == "__main__":
    unittest.main()
