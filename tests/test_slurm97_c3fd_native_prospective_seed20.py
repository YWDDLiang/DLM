from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "slurm" / "97_c3fd_native_prospective_seed20.sbatch"


class Slurm97C3FDNativeProspectiveSeed20Test(unittest.TestCase):
    def test_cpu_only_independent_planner_source(self):
        text = WRAPPER.read_text(encoding="utf-8")
        self.assertNotIn("#SBATCH --gres", text)
        self.assertIn("#SBATCH --cpus-per-task=16", text)
        self.assertIn("sampling_seed\\t20", text)
        self.assertIn("requested\\t1000", text)
        self.assertIn("spacegroup_mode\\tindependent_head", text)
        self.assertIn("selection_or_outcomes\\tfalse", text)
        self.assertIn("gpu_jobs_used\\t0", text)

    def test_sampler_assets_are_hash_pinned(self):
        text = WRAPPER.read_text(encoding="utf-8")
        self.assertIn("CHECKPOINT_SHA", text)
        self.assertIn("VOCAB_SHA", text)
        self.assertIn("SAMPLER_SHA", text)
        self.assertIn("plans_for_dlm.jsonl", text)


if __name__ == "__main__":
    unittest.main()
