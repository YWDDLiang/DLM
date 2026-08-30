from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "slurm" / "88_audit_c3fd_native_teacher_sft.sbatch"


class Slurm88TeacherAuditTest(unittest.TestCase):
    def test_cpu_only_and_all_inputs_are_pinned(self):
        text = WRAPPER.read_text(encoding="utf-8")
        self.assertNotIn("#SBATCH --gres", text)
        self.assertIn("#SBATCH --cpus-per-task=16", text)
        for value in (
            "SCRIPT_SHA",
            "CONFIG_SHA",
            "TOKENIZER_SHA",
            "DATA_MANIFEST_SHA",
            "DATA_TRAIN_SHA",
            "DATA_VAL_SHA",
        ):
            self.assertIn(value, text)

    def test_fails_closed_on_truncation_or_boundary_mismatch(self):
        text = WRAPPER.read_text(encoding="utf-8")
        self.assertIn("over_limit", text)
        self.assertIn("prompt_answer_suffix_mismatches", text)
        self.assertIn("_FAILED", text)
        self.assertIn("_SUCCESS", text)
        self.assertNotIn("nvidia-smi", text)


if __name__ == "__main__":
    unittest.main()
