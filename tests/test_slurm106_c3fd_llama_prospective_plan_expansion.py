from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "slurm/106_c3fd_llama_prospective_plan_expansion.sbatch").read_text(
    encoding="utf-8"
)


class Slurm106StaticTest(unittest.TestCase):
    def test_balanced_two_seed_assignment_is_fixed(self):
        self.assertIn("indices0_127:seedA;indices128_255:seedB", SOURCE)
        for spec in ("F 84017 A", "F 84018 B", "M 84117 A", "M 84118 B"):
            self.assertIn(f"sample_one {spec}", SOURCE)

    def test_no_selection_retry_or_outcomes(self):
        self.assertIn("retry_replacement_rerank\tfalse", SOURCE)
        self.assertIn("outcomes_read\tfalse", SOURCE)
        self.assertIn('"selection":False', SOURCE)


if __name__ == "__main__":
    unittest.main()
