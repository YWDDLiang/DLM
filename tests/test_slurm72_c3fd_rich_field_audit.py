from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
WRAPPERS = (
    ROOT / "slurm" / "72_c3fd_rich_field_audit.sbatch",
    ROOT / "slurm" / "73_c3fd_rich_field_audit_v2.sbatch",
)


class C3FDRichFieldAuditSlurmTest(unittest.TestCase):
    def test_wrapper_is_cpu_only_and_resource_bounded(self):
        for wrapper in WRAPPERS:
            with self.subTest(wrapper=wrapper.name):
                text = wrapper.read_text(encoding="utf-8")
                self.assertIn("#SBATCH --cpus-per-task=16", text)
                self.assertNotIn("#SBATCH --gres", text)
                self.assertIn("CUDA_VISIBLE_DEVICES=", text)
                self.assertIn("gpu_jobs_used\\t0", text)

    def test_wrapper_freezes_inputs_and_prevents_duplicate_contract(self):
        for wrapper in WRAPPERS:
            with self.subTest(wrapper=wrapper.name):
                text = wrapper.read_text(encoding="utf-8")
                self.assertIn("readonly CONTRACT_SHA=", text)
                self.assertIn('mkdir "${CONTRACT_DIR}"', text)
                self.assertIn('mkdir "${PREPARING}"', text)
                self.assertIn('[[ ! -e "${OUTPUT}" ]]', text)
                for frozen_hash in (
                    "SEED17_SHA",
                    "SEED18_SHA",
                    "VAL_SHA",
                    "VOCAB_SHA",
                    "SCRIPT_SHA",
                ):
                    self.assertIn(frozen_hash, text)

    def test_wrapper_runs_both_checkpoints_without_outcomes(self):
        for wrapper in WRAPPERS:
            with self.subTest(wrapper=wrapper.name):
                text = wrapper.read_text(encoding="utf-8")
                self.assertIn('--checkpoint "seed17=${CHECKPOINT17}"', text)
                self.assertIn('--checkpoint "seed18=${CHECKPOINT18}"', text)
                self.assertIn("outcomes_read\\tfalse", text)


if __name__ == "__main__":
    unittest.main()
