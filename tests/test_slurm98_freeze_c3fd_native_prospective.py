from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "slurm" / "98_freeze_c3fd_native_prospective.sbatch"


class Slurm98FreezeC3FDNativeProspectiveTest(unittest.TestCase):
    def test_cpu_only_outcome_blind_contract(self):
        text = WRAPPER.read_text(encoding="utf-8")
        self.assertNotIn("#SBATCH --gres", text)
        self.assertIn("#SBATCH --cpus-per-task=8", text)
        self.assertIn("planner_sampling_seed\\t20", text)
        self.assertIn("selected\\t256", text)
        self.assertIn("MP20_train_and_all_existing_cohorts_exact_identity", text)
        self.assertIn("outcomes_read\\tfalse", text)
        self.assertIn("policy_outcomes_exist_at_freeze\\tfalse", text)

    def test_source_and_builder_are_pinned(self):
        text = WRAPPER.read_text(encoding="utf-8")
        self.assertIn("SOURCE_SHA", text)
        self.assertIn("MP20_SHA", text)
        self.assertIn("BUILDER_SHA", text)
        self.assertIn('[[ ! -e "${OUTPUT}" ]]', text)
        self.assertIn("sha256sum -c SHA256SUMS", text)


if __name__ == "__main__":
    unittest.main()
