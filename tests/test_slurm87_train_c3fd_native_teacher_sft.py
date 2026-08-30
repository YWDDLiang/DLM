from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "slurm" / "87_train_c3fd_native_teacher_sft.sbatch"


class Slurm87TeacherSFTTest(unittest.TestCase):
    def setUp(self):
        self.text = WRAPPER.read_text(encoding="utf-8")

    def test_resources_and_allocated_gpu_mapping(self):
        self.assertIn("#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:4", self.text)
        self.assertIn("#SBATCH --cpus-per-task=32", self.text)
        self.assertIn("allocated_gpus", self.text)
        self.assertNotIn("nvidia-smi", self.text)

    def test_fresh_two_seed_two_stage_contract(self):
        for value in (
            "82017",
            "82018",
            "--epochs 2",
            "--max-train-steps 3392",
            "--lr-stage-boundary 1696",
            "--lr-stage2 1e-5",
            "--save-steps 3392",
            "--eval-steps 1696",
            "--grad-accum 8",
            "--lora-dropout 0.05",
            "--fail-on-truncation",
        ):
            self.assertIn(value, self.text)
        self.assertNotIn("--checkpoint-path", self.text)
        self.assertIn("checkpoint_path\"] is None", self.text)

    def test_teacher_data_and_hashes_are_pinned(self):
        self.assertIn("c3fd_native_teacher_sft_v1_20260831", self.text)
        for value in (
            "MODEL_TREE_SHA",
            "DATA_MANIFEST_SHA",
            "DATA_TRAIN_SHA",
            "DATA_VAL_SHA",
            "TRAINER_SHA",
            "LR_HELPER_SHA",
        ):
            self.assertIn(value, self.text)
        self.assertIn("predicted_plans_in_sft\tfalse", self.text)

    def test_only_step3392_is_eligible_and_failures_are_marked(self):
        self.assertIn("ENGINEERING_FAILURE.tsv", self.text)
        self.assertIn("_FAILED", self.text)
        self.assertIn("_SUCCESS", self.text)
        self.assertIn("step-3392", self.text)
        self.assertIn("checkpoint_dirs", self.text)
        self.assertIn("step0_lora_equality_canary.json", self.text)
        self.assertIn("grad_norm", self.text)


if __name__ == "__main__":
    unittest.main()
