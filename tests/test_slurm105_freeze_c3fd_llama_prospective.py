from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "slurm/105_freeze_c3fd_llama_prospective.sbatch").read_text(
    encoding="utf-8"
)


class Slurm105StaticTest(unittest.TestCase):
    def test_freeze_waits_for_canary_and_seed21_source(self):
        self.assertIn("C3FD_LLAMA_CANARY_RUN", SOURCE)
        self.assertIn("c3fd_prospective_seed21_38936", SOURCE)
        self.assertIn("--planner-sampling-seed 21", SOURCE)

    def test_conditions_replay_both_checkpoints_without_outcomes(self):
        self.assertIn('--checkpoint "seed17=${C3FD17}"', SOURCE)
        self.assertIn('--checkpoint "seed18=${C3FD18}"', SOURCE)
        self.assertIn("policy_or_DLM_outcomes_read\tfalse", SOURCE)
        self.assertNotIn("e_above_hull", SOURCE)


if __name__ == "__main__":
    unittest.main()
