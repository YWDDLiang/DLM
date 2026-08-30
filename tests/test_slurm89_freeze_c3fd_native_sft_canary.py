from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "slurm" / "89_freeze_c3fd_native_sft_canary.sbatch"


class Slurm89FreezeC3FDNativeSFTCanaryTest(unittest.TestCase):
    def test_cpu_only_outcome_blind_balanced_contract(self):
        text = WRAPPER.read_text(encoding="utf-8")
        self.assertNotIn("#SBATCH --gres", text)
        self.assertIn("#SBATCH --cpus-per-task=8", text)
        self.assertIn("--per-split 128", text)
        self.assertIn("freeze_seed=20260831", text)
        self.assertIn("planner_checkpoints=seed17,seed18", text)
        self.assertIn("checkpoint_selection=none", text)
        self.assertIn("policy_or_test_outcomes_read=false", text)
        self.assertIn("official_query\\tfalse", text)

    def test_immutable_inputs_and_outputs(self):
        text = WRAPPER.read_text(encoding="utf-8")
        self.assertIn("TEACHER_MANIFEST_SHA", text)
        self.assertIn("PREDICTED_MANIFEST_SHA", text)
        self.assertIn('[[ ! -e "${OUTPUT}" ]]', text)
        self.assertIn("sha256sum -c SHA256SUMS", text)
        self.assertIn("_SUCCESS", text)


if __name__ == "__main__":
    unittest.main()
