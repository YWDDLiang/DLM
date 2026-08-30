from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "slurm" / "92_freeze_c3fd_native_alignment_pool.sbatch"


class Slurm92FreezeC3FDNativeAlignmentPoolTest(unittest.TestCase):
    def test_cpu_only_fixed_pool_contract(self):
        text = WRAPPER.read_text(encoding="utf-8")
        self.assertNotIn("#SBATCH --gres", text)
        self.assertIn("#SBATCH --cpus-per-task=8", text)
        for value in (
            "source=MP20_train_only",
            "selection_seed=20260901",
            "compositions=32",
            "planner_checkpoints=seed17,seed18",
            "K=4",
            "groups=64",
            "rows=256",
            "outcomes_read=false",
        ):
            self.assertIn(value, text)
        self.assertIn("old_3614_used\\tfalse", text)

    def test_exclusion_and_hash_guards(self):
        text = WRAPPER.read_text(encoding="utf-8")
        self.assertIn("EXCLUSION_INPUTS_BEFORE.sha256", text)
        self.assertIn("TEACHER_MANIFEST_SHA", text)
        self.assertIn("PREDICTED_MANIFEST_SHA", text)
        self.assertIn("BUILDER_SHA", text)
        self.assertIn('[[ ! -e "${OUTPUT}" ]]', text)
        self.assertIn("sha256sum -c SHA256SUMS", text)


if __name__ == "__main__":
    unittest.main()
