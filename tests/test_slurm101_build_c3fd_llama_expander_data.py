from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "slurm/101_build_c3fd_llama_expander_data.sbatch").read_text(
    encoding="utf-8"
)


class Slurm101StaticTest(unittest.TestCase):
    def test_cpu_only_matched_F_M_contract(self):
        self.assertNotIn("#SBATCH --gres", SOURCE)
        self.assertIn("F_formula_only,M_learned_soft_prefix", SOURCE)
        self.assertIn("feature_dim\t54", SOURCE)
        self.assertIn("predicted_checkpoints\tseed17,seed18_no_selection", SOURCE)

    def test_builder_keeps_outcomes_out(self):
        self.assertIn("outcomes_read\tfalse", SOURCE)
        self.assertIn("visible_answers_identical", SOURCE)


if __name__ == "__main__":
    unittest.main()
