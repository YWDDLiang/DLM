from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = (ROOT / "slurm/209_audit_spad_basin_preflight_model_replay.sbatch").read_text(encoding="utf-8")


class PreflightReplayWrapperTest(unittest.TestCase):
    def test_two_gpu_bit_exact_replay_contract(self):
        self.assertIn("#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:2", WRAPPER)
        self.assertIn("#SBATCH --cpus-per-task=8", WRAPPER)
        self.assertIn("--nproc_per_node=2", WRAPPER)
        self.assertIn("-m torch.distributed.run", WRAPPER)
        self.assertIn("attempts)==128", WRAPPER)
        self.assertIn("'mismatches':len(mismatch)", WRAPPER)
        self.assertIn("bit_exact_required_before_value_labels", WRAPPER)
        self.assertNotIn("chgnet", WRAPPER.lower())
        self.assertNotIn("model494", WRAPPER.lower())


if __name__ == "__main__":
    unittest.main()
