from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "slurm" / "195_build_spad_basin_closure_sft_data.sbatch"
TRAIN = ROOT / "slurm" / "196_train_spad_basin_closure_ce.sbatch"
CANARY = ROOT / "slurm" / "197_spad_basin_closure_ce_canary.sbatch"
NATIVE = ROOT / "slurm" / "198_spad_basin_closure_native_stream17.sbatch"
RAW_SCREEN = ROOT / "slurm" / "199_spad_basin_closure_native_raw_screen.sbatch"
COMMON_RELAX = ROOT / "slurm" / "200_spad_basin_closure_common_relax.sbatch"
COMMON_FINAL = ROOT / "slurm" / "201_spad_basin_closure_common_relax_finalize.sbatch"
PREFLIGHT_FREEZE = ROOT / "slurm" / "202_freeze_spad_basin_preflight_cohort.sbatch"
PREFLIGHT_GENERATE = ROOT / "slurm" / "203_generate_spad_basin_preflight_states.sbatch"
TAU200_BRIDGE = ROOT / "slurm" / "204_spad_basin_closure_tau200_bridge.sbatch"
TAU200_RELAX = ROOT / "slurm" / "205_spad_basin_closure_tau200_common_relax.sbatch"
TAU200_FINAL = ROOT / "slurm" / "206_spad_basin_closure_tau200_finalize.sbatch"
TAU_CALIBRATION = ROOT / "slurm" / "207_spad_basin_closure_tau_calibration.sbatch"
TAU_CALIBRATION_EVAL = ROOT / "slurm" / "208_spad_basin_closure_tau_calibration_eval.sbatch"
TAU800_BRIDGE = ROOT / "slurm" / "211_spad_basin_closure_tau800_bridge.sbatch"
TAU_EVAL2 = ROOT / "slurm" / "217_spad_tau_calibration_eval_2gpu.sbatch"


class SPADBasinClosureWrapperTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.build = BUILD.read_text(encoding="utf-8")
        cls.train = TRAIN.read_text(encoding="utf-8")
        cls.canary = CANARY.read_text(encoding="utf-8")
        cls.native = NATIVE.read_text(encoding="utf-8")
        cls.raw_screen = RAW_SCREEN.read_text(encoding="utf-8")
        cls.common_relax = COMMON_RELAX.read_text(encoding="utf-8")
        cls.common_final = COMMON_FINAL.read_text(encoding="utf-8")
        cls.preflight_freeze = PREFLIGHT_FREEZE.read_text(encoding="utf-8")
        cls.preflight_generate = PREFLIGHT_GENERATE.read_text(encoding="utf-8")
        cls.tau200_bridge = TAU200_BRIDGE.read_text(encoding="utf-8")
        cls.tau200_relax = TAU200_RELAX.read_text(encoding="utf-8")
        cls.tau200_final = TAU200_FINAL.read_text(encoding="utf-8")
        cls.tau_calibration = TAU_CALIBRATION.read_text(encoding="utf-8")
        cls.tau_calibration_eval = TAU_CALIBRATION_EVAL.read_text(encoding="utf-8")
        cls.tau800_bridge = TAU800_BRIDGE.read_text(encoding="utf-8")
        cls.tau_eval2 = TAU_EVAL2.read_text(encoding="utf-8")

    def test_build_uses_full_teacher_pointer_and_preserves_contract(self):
        self.assertIn("#SBATCH --cpus-per-task=4", self.build)
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

    def test_native_screen_generation_uses_only_registered_closure(self):
        self.assertIn("#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1", self.native)
        self.assertIn("#SBATCH --cpus-per-task=4", self.native)
        self.assertNotIn("--nproc_per_node", self.native)
        self.assertIn("--generation-schedule spad", self.native)
        self.assertIn("--spad-basin-closure", self.native)
        self.assertIn("--spad-basin-closure-capability-json", self.native)
        self.assertIn("--pbc-min-distance-mask", self.native)
        self.assertIn("--num-samples 256", self.native)
        self.assertIn("--seed \"${DLM_SEED}\"", self.native)
        self.assertIn('"one_plan_one_trajectory": True', self.native)
        self.assertIn('"model494_or_inference_critic": False', self.native)
        self.assertIn('"selection_retry_replacement": False', self.native)
        self.assertIn('"world_size": 1', self.native)
        self.assertIn('"rank_seed_rule": "seed + rank"', self.native)
        self.assertNotIn("--spad-backfill", self.native)
        self.assertNotIn("refine_dlm_with_crysllmgen.py", self.native)

    def test_raw_screen_is_paired_fast_physics_without_direct_or_refiner(self):
        self.assertIn("#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1", self.raw_screen)
        self.assertIn("#SBATCH --cpus-per-task=4", self.raw_screen)
        self.assertIn("evaluate_spad_basin_closure_raw.py", self.raw_screen)
        self.assertIn("--batch-size 16", self.raw_screen)
        self.assertIn("--denominator 256", self.raw_screen)
        self.assertIn("spad_prospective_generation_s17_39537", self.raw_screen)
        self.assertIn('config["world_size"] == 1', self.raw_screen)
        self.assertIn('config["batch_size"] == 8', self.raw_screen)
        self.assertIn('config["seed"] == 91117', self.raw_screen)
        self.assertIn('config["rank_seed_rule"] == "seed + rank"', self.raw_screen)
        self.assertIn('bs_report["frozen_plan_dir"]', self.raw_screen)
        self.assertNotIn("run_crysllmgen_validity", self.raw_screen)
        self.assertNotIn("refine_dlm_with_crysllmgen.py", self.raw_screen)
        self.assertNotIn("query_official", self.raw_screen)

    def test_common_relax_array_can_schedule_each_gpu_independently(self):
        self.assertIn("#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1", self.common_relax)
        self.assertIn("#SBATCH --cpus-per-task=4", self.common_relax)
        self.assertIn("#SBATCH --array=0-1", self.common_relax)
        self.assertIn("relax_spad_basin_closure_shard.py", self.common_relax)
        self.assertIn("a100_sun.prepare_a100_input", self.common_relax)
        self.assertIn('--generated-pt "${FROZEN_INPUT}/all_attempts.pt"', self.common_relax)
        self.assertIn('--input-manifest "${FROZEN_INPUT}/input_manifest.json"', self.common_relax)
        self.assertIn('--shard-count 2 --device cuda', self.common_relax)
        self.assertIn("SLURM_ARRAY_JOB_ID", self.common_relax)
        self.assertIn('touch "${ARRAY_RUN}/_SUCCESS_rank${RANK}"', self.common_relax)

    def test_common_relax_finalizer_reuses_frozen_evaluator_and_baseline(self):
        self.assertIn("#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1", self.common_final)
        self.assertIn("#SBATCH --cpus-per-task=4", self.common_final)
        self.assertIn("run_full_reconstructed_eval.py", self.common_final)
        self.assertIn("CACHE_COVERAGE.json", self.common_final)
        self.assertIn("assert not (required - cache)", self.common_final)
        self.assertIn('summary["working_relax_cache_sha256_before"]', self.common_final)
        self.assertIn('summary["working_relax_cache_sha256_after"]', self.common_final)
        self.assertIn('"frozen_evaluator_cache_misses": 0', self.common_final)
        self.assertIn('sum(report["assigned_occurrences"] for report in reports) == 256', self.common_final)
        self.assertIn("spad_prospective_offline_s17_39542", self.common_final)
        self.assertIn("--arm candidate --repeat 0", self.common_final)
        self.assertIn("H1_ACTIVE_DENOMINATOR=256", self.common_final)
        self.assertIn('summary["generation_succeeded"] == 256', self.common_final)
        self.assertIn('"chgnet_relaxation_unknown": 256 - summary["chgnet_relaxation_known"]', self.common_final)
        self.assertIn('"closure_ce_minus_frozen_BS_CHGNet_relaxed_eV_per_atom"', self.common_final)
        self.assertIn('left["chgnet_composition"] == right["chgnet_composition"]', self.common_final)
        self.assertIn('"direct_run": False', self.common_final)
        self.assertIn('"model494": False', self.common_final)

    def test_preflight_freeze_is_train_only_and_outcome_blind(self):
        self.assertNotIn("#SBATCH --gres", self.preflight_freeze)
        self.assertIn("#SBATCH --cpus-per-task=4", self.preflight_freeze)
        self.assertIn("full_mp20_reference_bodies_39658/programs/plans_for_dlm.jsonl", self.preflight_freeze)
        self.assertIn("freeze_spad_basin_preflight_cohort.py", self.preflight_freeze)
        self.assertIn('[[ ! -e "${OUTPUT}" ]]', self.preflight_freeze)
        self.assertIn("128", self.preflight_freeze)

    def test_preflight_generation_uses_registered_closure_only(self):
        self.assertIn("#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1", self.preflight_generate)
        self.assertIn("#SBATCH --cpus-per-task=4", self.preflight_generate)
        self.assertIn("--spad-basin-closure", self.preflight_generate)
        self.assertIn("--spad-basin-closure-capability-json", self.preflight_generate)
        self.assertIn("--num-samples 128", self.preflight_generate)
        self.assertIn("'outcomes_read':False", self.preflight_generate)
        self.assertIn("'model494_or_energy_used':False", self.preflight_generate)
        self.assertNotIn("--spad-backfill", self.preflight_generate)
        self.assertNotIn("refine_dlm_with_crysllmgen", self.preflight_generate)

    def test_tau200_bridge_is_fixed_and_selection_free(self):
        self.assertIn("#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1", self.tau200_bridge)
        self.assertIn("#SBATCH --cpus-per-task=4", self.tau200_bridge)
        self.assertIn("spad_basin_closure_native_s17_39704", self.tau200_bridge)
        self.assertIn("--diff-steps 200", self.tau200_bridge)
        self.assertIn("--seed-by-sample-index", self.tau200_bridge)
        self.assertIn("--max-proposals 256", self.tau200_bridge)
        self.assertIn("'selection_retry_replacement':False", self.tau200_bridge)
        self.assertNotIn("--diff-steps 800", self.tau200_bridge)

    def test_tau200_evaluation_reuses_two_shards_and_existing_official_cache(self):
        self.assertIn("#SBATCH --array=0-1", self.tau200_relax)
        self.assertIn("#SBATCH --cpus-per-task=4", self.tau200_relax)
        self.assertIn("relax_spad_basin_closure_shard.py", self.tau200_relax)
        self.assertIn("--shard-count 2", self.tau200_relax)
        self.assertIn("spad_basin_closure_official_20260904_v1", self.tau200_final)
        self.assertIn("--model494-tau 200", self.tau200_final)
        self.assertNotIn("query", self.tau200_final.lower())

    def test_low_noise_calibration_is_development_only_and_fully_parallel(self):
        self.assertIn("#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:2", self.tau_calibration)
        self.assertIn("run_tau 0 400", self.tau_calibration)
        self.assertIn("run_tau 1 600", self.tau_calibration)
        self.assertIn("'development_only':True", self.tau_calibration)
        self.assertIn("#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:4", self.tau_calibration_eval)
        self.assertIn("run_shard 3 600 1", self.tau_calibration_eval)
        self.assertIn("run_shard 1 800 1", self.tau_calibration_eval)
        self.assertNotIn("query", self.tau_calibration_eval.lower())

    def test_tau800_bridge_is_an_explicit_development_anchor(self):
        self.assertIn("#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1", self.tau800_bridge)
        self.assertIn("--diff-steps 800", self.tau800_bridge)
        self.assertIn("--seed-by-sample-index", self.tau800_bridge)
        self.assertNotIn("chgnet", self.tau800_bridge.lower())

    def test_resource-adaptive_tau_evaluator_runs_three_fixed_taus_without_query(self):
        self.assertIn("#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1", self.tau_eval2)
        self.assertIn("#SBATCH --cpus-per-task=4", self.tau_eval2)
        self.assertIn("for tau in 400 600 800", self.tau_eval2)
        self.assertIn("--shard-count 1", self.tau_eval2)
        self.assertIn("automatic_selection':False", self.tau_eval2)
        self.assertNotIn("query", self.tau_eval2.lower())

    def test_every_run_is_non_overwriting(self):
        for wrapper in (
            self.build, self.train, self.canary, self.native, self.raw_screen,
            self.common_final, self.preflight_freeze, self.preflight_generate,
            self.tau200_bridge,
            self.tau200_final,
        ):
            self.assertIn('mkdir "${RUN}"', wrapper)
        for wrapper in (self.build, self.train, self.canary):
            self.assertNotIn("mkdir -p", wrapper)


if __name__ == "__main__":
    unittest.main()
