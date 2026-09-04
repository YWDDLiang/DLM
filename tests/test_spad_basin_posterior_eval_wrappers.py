from pathlib import Path
import re
import shutil
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
GENERATION = (
    ROOT / "slurm" / "214_spad_basin_posterior_stream18_generation_refine.sbatch"
)
EVALUATION = ROOT / "slurm" / "215_spad_basin_posterior_stream18_eval.sbatch"


class SPADBasinPosteriorHeldoutWrapperTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.generation = GENERATION.read_text(encoding="utf-8")
        cls.evaluation = EVALUATION.read_text(encoding="utf-8")

    def test_generation_supports_primary_only_and_two_arm_modes(self):
        self.assertIn("#SBATCH --cpus-per-task=8", self.generation)
        self.assertIn(
            "#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:2", self.generation
        )
        self.assertIn('readonly PRIMARY_ONLY="${SPAD_PRIMARY_ONLY:-false}"', self.generation)
        self.assertIn('readonly PRIMARY_WORLD_SIZE="${SPAD_PRIMARY_WORLD_SIZE:-4}"', self.generation)
        self.assertIn('run_arm k10 "${gpu_csv}" "${expected_gpus}"', self.generation)
        self.assertIn('run_arm closure_ce "${allocated_gpus[0]}" 1', self.generation)
        self.assertIn('run_arm k10 "${allocated_gpus[1]}" 1', self.generation)
        self.assertIn('"--nproc_per_node=${world_size}"', self.generation)
        self.assertIn('CHILD_PIDS+=("$!")', self.generation)
        self.assertIn("k10_primary_only", self.generation)
        self.assertIn("closure_ce_k10_concurrent", self.generation)
        primary_branch = re.search(
            r'if \[\[ "\$\{PRIMARY_ONLY\}" == true \]\]; then\n'
            r'  gpu_csv=.*?\n(.*?)\nelse',
            self.generation,
            flags=re.DOTALL,
        )
        self.assertIsNotNone(primary_branch)
        self.assertIn("run_arm k10", primary_branch.group(1))
        self.assertNotIn("run_arm closure_ce", primary_branch.group(1))

    def test_generation_requires_frozen_training_routes_and_selected_tau(self):
        self.assertIn("SPAD_BASIN_POSTERIOR_TRAIN_RUN:?", self.generation)
        self.assertIn("SPAD_SELECTED_TAU:?", self.generation)
        self.assertIn("200|400|600|800", self.generation)
        self.assertIn('train_run / "POLICY_PATH"', self.generation)
        self.assertIn('train_run / "CAPABILITY_PATH"', self.generation)
        self.assertIn("spad_basin_closure_ce_39700", self.generation)
        self.assertIn("POLICY_PATH", self.generation)
        self.assertIn("CAPABILITY_PATH", self.generation)

    def test_plan_stream_and_seed_defaults_are_runtime_parameters(self):
        expected_defaults = (
            'readonly PLAN_DIR="${SPAD_PLAN_DIR:-${ROOT}/cohorts/'
            'spad_prospective_seed23_256_v1_20260903}"',
            'readonly PLANNER_SEED="${SPAD_PLANNER_SEED:-23}"',
            'readonly STREAM="${SPAD_EVAL_STREAM:-18}"',
            'readonly DLM_SEED="${SPAD_DLM_SEED:-92117}"',
            'readonly REFINER_SEED="${SPAD_REFINER_SEED:-102117}"',
        )
        for source in (self.generation, self.evaluation):
            for value in expected_defaults:
                self.assertIn(value, source)
            self.assertIn(
                "for numeric_name in PLANNER_SEED STREAM DLM_SEED REFINER_SEED",
                source,
            )

        self.assertIn(
            '"${PLANNER_SEED}" "${STREAM}" "${DLM_SEED}" "${REFINER_SEED}"',
            self.generation,
        )
        self.assertIn(
            '"${PLAN_DIR}" "${PLANNER_SEED}" "${STREAM}" "${DLM_SEED}"',
            self.evaluation,
        )
        self.assertIn(
            'assert int(manifest["planner_sampling_seed"]) == planner_seed',
            self.generation,
        )
        self.assertIn(
            'assert report["planner_sampling_seed"] == planner_seed',
            self.evaluation,
        )

    def test_generation_is_parameterized_spad_and_failure_preserving(self):
        for value in (
            'stream${STREAM}_generation_refine_${SLURM_JOB_ID}',
            'cell="${RUN}/${arm}/stream${STREAM}"',
            '${arm}_stream${STREAM}_body.out',
            '${arm}_stream${STREAM}_refine.out',
            "--num-samples 256",
            "--batch-size 8",
            "--temperature 0.7",
            '--seed "${DLM_SEED}"',
            "--generation-schedule spad",
            "--spad-basin-closure",
            "--spad-basin-closure-capability-json",
            "--pbc-min-distance-mask",
            "--seed-by-sample-index",
            '--diff-steps "${SELECTED_TAU}"',
            "--invalid-refined-as-failure",
            '"all_failures_retained_in_fixed_denominator": True',
            '"one_plan_one_trajectory": True',
            '"automatic_route_tau_seed_checkpoint_choice": False',
            '"retry_rerank_replacement": False',
        ):
            self.assertIn(value, self.generation)
        for hardcoded in (
            'root / arm / "stream18"',
            '${arm}_stream18_',
            '"stream": 18',
            '"dlm_seed": 92117',
            '"refiner_seed": 102117',
            'heldout_stream18_generation',
        ):
            self.assertNotIn(hardcoded, self.generation)
        self.assertNotIn("--spad-backfill", self.generation)
        self.assertNotIn('body_metrics["graph_success"] == 256', self.generation)
        self.assertNotIn('body_metrics["parse_success"] == 256', self.generation)

    def test_evaluation_supports_primary_only_and_four_cell_modes(self):
        self.assertIn("#SBATCH --cpus-per-task=16", self.evaluation)
        self.assertIn(
            "#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:4", self.evaluation
        )
        first_wave = re.search(
            r"run_wave_one\(\) \{(.*?)\n\}", self.evaluation, flags=re.DOTALL
        )
        self.assertIsNotNone(first_wave)
        self.assertEqual(first_wave.group(1).count("run_full_cell "), 4)
        self.assertIn("run_primary_endpoint_sharded raw", first_wave.group(1))
        self.assertIn("run_primary_endpoint_sharded refined", first_wave.group(1))
        self.assertIn(
            'run_primary_endpoint_sharded raw 0 "${expected_gpus}"',
            first_wave.group(1),
        )
        self.assertIn('readonly PRIMARY_ONLY="${SPAD_PRIMARY_ONLY:-false}"', self.evaluation)
        self.assertIn('readonly PRIMARY_WORLD_SIZE="${SPAD_PRIMARY_WORLD_SIZE:-4}"', self.evaluation)
        self.assertIn("k10_raw_refined", self.evaluation)
        self.assertIn("relax_spad_basin_closure_shard.py", self.evaluation)
        self.assertIn('--shard-count "${shard_count}"', self.evaluation)
        self.assertNotIn("run_wave_two", self.evaluation)
        self.assertIn("run_wave_one", self.evaluation)
        self.assertIn("run_full_reconstructed_eval.py", self.evaluation)
        self.assertIn("working_relax_cache.jsonl", self.evaluation)
        primary_branch = first_wave.group(1).split("else", maxsplit=1)[0]
        self.assertIn("run_primary_endpoint_sharded raw", primary_branch)
        self.assertIn("run_primary_endpoint_sharded refined", primary_branch)
        self.assertNotIn("run_full_cell", primary_branch)
        self.assertNotIn("closure_ce", primary_branch)
        self.assertIn(
            'stream${STREAM}/${endpoint}_generation/generation.jsonl',
            self.evaluation,
        )
        self.assertIn(
            '--endpoint "${arm}_${endpoint}_stream${STREAM}"', self.evaluation
        )
        for hardcoded in (
            'root / arm / "stream18"',
            '/stream18/',
            '"stream": 18',
            '"dlm_seed": 92117',
            '"refiner_seed": 102117',
            'heldout_stream18_final',
        ):
            self.assertNotIn(hardcoded, self.evaluation)

    def test_evaluation_reuses_existing_official_cache_without_external_access(self):
        self.assertIn(
            "spad_basin_closure_official_20260904_v1/official_mp_cache",
            self.evaluation,
        )
        self.assertIn("finalize_spad_basin_closure_official.py", self.evaluation)
        self.assertIn(
            "spad_basin_closure_official_final_20260904_v1/attempt_results_official.jsonl",
            self.evaluation,
        )
        self.assertIn(
            '"paired_s17_closure_comparison_is_diagnostic_only": True',
            self.evaluation,
        )
        lowered = self.evaluation.lower()
        for forbidden in (
            "run_crysllmgen_metrics.py",
            "run_direct",
            "query_official",
            "mp_api_key",
            "materialsproject",
            "requests.get",
            "curl ",
        ):
            self.assertNotIn(forbidden, lowered)

    def test_evaluation_finalizes_all_four_cells_with_exact_model_flags(self):
        self.assertIn("for arm in closure_ce k10", self.evaluation)
        self.assertIn('finalize_cell "${arm}" raw', self.evaluation)
        self.assertIn('finalize_cell "${arm}" refined', self.evaluation)
        self.assertIn('if [[ "${endpoint}" == refined ]]', self.evaluation)
        self.assertIn(
            'model_args=(--model494 --model494-tau "${SELECTED_TAU}")',
            self.evaluation,
        )
        self.assertIn('assert report["model494"] is False', self.evaluation)
        self.assertIn('assert report["model494"] is True', self.evaluation)
        self.assertIn('assert report["model494_tau"] == tau', self.evaluation)

    def test_primary_k10_target_and_fixed_denominator_are_exact(self):
        for value in (
            '"primary_cell": "k10_refined"',
            '"strict_sun_min_count": 26',
            '"strict_sun_min_rate": 0.10',
            '"meta_sun_min_count": 128',
            '"meta_sun_min_rate": 0.50',
            'primary["strict_sun"]["count"] >= 26',
            'primary["meta_sun"]["count"] >= 128',
            '"target_met": strict_met and meta_met',
            '"fixed_denominator": 256',
            '"automatic_route_tau_seed_checkpoint_choice": False',
            '"all_failures_retained_in_fixed_denominator": True',
            'f"HELDOUT_STREAM{stream}_FINAL.json"',
        ):
            self.assertIn(value, self.evaluation)

    def test_selected_tau_is_required_and_must_match_generation(self):
        self.assertIn("SPAD_SELECTED_TAU:?", self.evaluation)
        self.assertIn("200|400|600|800", self.evaluation)
        self.assertIn('assert report["selected_tau"] == tau', self.evaluation)
        self.assertIn('assert report["stream"] == stream', self.evaluation)
        self.assertIn('assert report["dlm_seed"] == dlm_seed', self.evaluation)
        self.assertIn(
            'assert report["refiner_seed"] == refiner_seed', self.evaluation
        )
        self.assertIn(
            '"selected_tau_source": "completed_stream17_development_calibration"',
            self.evaluation,
        )

    def test_embedded_python_is_syntactically_valid(self):
        for path, source in (
            (GENERATION, self.generation),
            (EVALUATION, self.evaluation),
        ):
            blocks = re.findall(r"<<'PY'\n(.*?)\nPY", source, flags=re.DOTALL)
            self.assertGreater(len(blocks), 0, path)
            for index, block in enumerate(blocks):
                compile(block, f"{path.name}:heredoc-{index}", "exec")

    @unittest.skipUnless(shutil.which("bash"), "bash is unavailable")
    def test_bash_syntax(self):
        for path, source in (
            (GENERATION, self.generation),
            (EVALUATION, self.evaluation),
        ):
            result = subprocess.run(
                [shutil.which("bash"), "-n"],
                input=source.encode("utf-8"),
                capture_output=True,
                check=False,
            )
            diagnostic = result.stderr.decode("utf-8", errors="replace")
            self.assertEqual(result.returncode, 0, f"{path}: {diagnostic}")


if __name__ == "__main__":
    unittest.main()
