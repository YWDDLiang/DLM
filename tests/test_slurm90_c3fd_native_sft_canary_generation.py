from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "slurm" / "90_c3fd_native_sft_canary_generation.sbatch"


class Slurm90C3FDNativeSFTCanaryGenerationTest(unittest.TestCase):
    def test_four_cells_preserve_both_policies_and_planners(self):
        text = WRAPPER.read_text(encoding="utf-8")
        self.assertIn("#SBATCH --cpus-per-task=32", text)
        self.assertIn("#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:4", text)
        for command in (
            'run_cell 0 17 82017 "${POLICY17}" "${COHORT}/planner_seed17.jsonl"',
            'run_cell 1 17 82018 "${POLICY18}" "${COHORT}/planner_seed17.jsonl"',
            'run_cell 2 18 82017 "${POLICY17}" "${COHORT}/planner_seed18.jsonl"',
            'run_cell 3 18 82018 "${POLICY18}" "${COHORT}/planner_seed18.jsonl"',
        ):
            self.assertIn(command, text)
        self.assertNotIn("run_cell 4", text)
        self.assertNotIn("policybase", text.lower())

    def test_science_and_hash_contract(self):
        text = WRAPPER.read_text(encoding="utf-8")
        for name in (
            "MANIFEST_SHA",
            "LEDGER_SHA",
            "PLAN17_SHA",
            "PLAN18_SHA",
            "POLICY17_SHA",
            "POLICY18_SHA",
            "POLICY_CONFIG_SHA",
            "REFINER_SHA",
            "SAMPLER_SHA",
            "REFINER_SCRIPT_SHA",
        ):
            self.assertIn(name, text)
        self.assertIn("split=train128,val128", text)
        self.assertIn("temperature\\t0.7", text)
        self.assertIn("schedule\\texact_axis", text)
        self.assertIn("refiner_tau\\t800", text)
        self.assertIn("attempts_per_plan\\t1", text)
        self.assertIn("seed_or_checkpoint_selection\\tfalse", text)
        self.assertIn("retry_rerank_replacement\\tfalse", text)
        self.assertIn("official_query\\tfalse", text)

    def test_allocated_gpu_and_refiner_markers_are_safe(self):
        text = WRAPPER.read_text(encoding="utf-8")
        self.assertIn("ALLOCATED_GPU_LIST", text)
        self.assertIn('device="${GPU_DEVICES[${slot}]}"', text)
        self.assertNotIn('CUDA_VISIBLE_DEVICES="${slot}"', text)
        self.assertIn("refinement_metrics.json", text)
        self.assertIn('["output_file"]', text)
        self.assertIn('touch "${output}/refine/_SUCCESS"', text)


if __name__ == "__main__":
    unittest.main()
