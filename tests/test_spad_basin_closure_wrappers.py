from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "slurm" / "195_build_spad_basin_closure_sft_data.sbatch"
TRAIN = ROOT / "slurm" / "196_train_spad_basin_closure_ce.sbatch"
CANARY = ROOT / "slurm" / "197_spad_basin_closure_ce_canary.sbatch"


class SPADBasinClosureWrapperTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.build = BUILD.read_text(encoding="utf-8")
        cls.train = TRAIN.read_text(encoding="utf-8")
        cls.canary = CANARY.read_text(encoding="utf-8")

    def test_build_uses_full_teacher_pointer_and_preserves_contract(self):
        self.assertIn("#SBATCH --cpus-per-task=16", self.build)
        self.assertNotIn("#SBATCH --gres", self.build)
        self.assertIn("c3fd_native_teacher_sft_canonical_v2_20260902", self.build)
        self.assertIn("spad_species_pointer_v1_20260903", self.build)
        self.assertIn("build_spad_basin_closure_sft_data.py", self.build)
        self.assertIn('[[ ! -e "${OUTPUT}" ]]', self.build)
        self.assertIn("27136", self.build)
        self.assertIn("9047", self.build)
        self.assertIn("rollout_matched_transition_v1", self.build)
        self.assertIn("cell_then_reverse_llama_species_blocks_v1", self.build)
        self.assertIn("exact_N_elements_never_masked", self.build)
        self.assertIn('manifest["outcomes_read"] is False', self.build)
        self.assertIn("protected", self.build)

    def test_formal_training_resources_initialization_and_hyperparameters(self):
        self.assertIn("#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:4", self.train)
        self.assertIn("#SBATCH --cpus-per-task=16", self.train)
        self.assertIn(
            "spad_schedule_dlm_39520/train/checkpoints/step-1696", self.train
        )
        for value in (
            "--checkpoint-path \"${BASE}\"",
            "--require-paired-source",
            "--require-rollout-masks",
            "--epochs 1",
            "--max-train-steps 1696",
            "--batch-size 1",
            "--grad-accum 4",
            "--lr 5e-6",
            "--lora-rank 8",
            "--lora-alpha 32",
            "--lora-dropout 0.05",
            "--save-steps 1696",
            "--skip-final-alias",
        ):
            self.assertIn(value, self.train)
        self.assertIn("effective_batch\t16", self.train)
        self.assertIn('checkpoints[0]}" == step-1696', self.train)

    def test_formal_capability_is_resolved_and_complete(self):
        for value in (
            "spad_basin_closure_capability.json",
            '"checkpoint_path": str(checkpoint)',
            '"spad_cell_closure_trained": True',
            '"spad_species_block_closure_trained": True',
            '"closure_schedule_version": "cell_then_reverse_llama_species_blocks_v1"',
            '"adapter_model_sha256": adapter_model_sha256',
            'adapter_model = checkpoint / "adapter_model.safetensors"',
            "adapter_model_sha256 = sha256_file(adapter_model)",
        ):
            self.assertIn(value, self.train)
        self.assertIn("resolve(strict=True)", self.train)
        self.assertIn('[[ ! -e "${OUTPUT}/final" ]]', self.train)
        self.assertIn("POLICY_PATH", self.train)

    def test_canary_covers_six_states_on_one_gpu_and_is_nonpublishable(self):
        self.assertIn("#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1", self.canary)
        self.assertIn("#SBATCH --cpus-per-task=4", self.canary)
        self.assertIn(
            "spad_schedule_dlm_39520/train/checkpoints/step-1696", self.canary
        )
        self.assertIn("--max-train-steps 6", self.canary)
        self.assertIn('readonly CANARY_DATA="${RUN}/canary_data"', self.canary)
        self.assertIn('"rows_per_split": len(categories)', self.canary)
        self.assertIn('start["rollout_masks"]["train_rows"] == 6', self.canary)
        for category in (
            "cell_first", "cell_last", "coordinate_x", "coordinate_y",
            "coordinate_z", "multi_site_block",
        ):
            self.assertIn(category, self.canary)
        self.assertIn("--require-paired-source", self.canary)
        self.assertIn("--require-rollout-masks", self.canary)
        self.assertIn("finite_loss_and_gradient", self.canary)
        self.assertIn('"checkpoint_publishable": False', self.canary)
        self.assertIn('"scientific_result": False', self.canary)
        self.assertIn("mask_classes", self.canary)
        self.assertNotIn("POLICY_PATH", self.canary)
        self.assertNotIn("CAPABILITY_PATH", self.canary)

    def test_wrappers_do_not_use_physical_evaluators(self):
        forbidden = ("model494", "chgnet", "hull", "mp-api", "mp_api")
        for wrapper in (self.build, self.train, self.canary):
            lowered = wrapper.lower()
            for term in forbidden:
                self.assertNotIn(term, lowered)

    def test_every_run_is_non_overwriting(self):
        for wrapper in (self.build, self.train, self.canary):
            self.assertIn('mkdir "${RUN}"', wrapper)
            self.assertNotIn("mkdir -p", wrapper)


if __name__ == "__main__":
    unittest.main()
