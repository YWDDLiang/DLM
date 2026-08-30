from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "slurm" / "79_rich_recovery_canary_generation.sbatch"


class RichRecoveryCanaryGenerationSlurmTest(unittest.TestCase):
    def test_resources_and_science_contract_are_frozen(self):
        text = WRAPPER.read_text(encoding="utf-8")
        self.assertIn("#SBATCH --cpus-per-task=48", text)
        self.assertIn("#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:6", text)
        self.assertIn("arms\\tM0,RCF,R0", text)
        self.assertIn("streams\\t17,18", text)
        self.assertIn("denominator\\t256", text)
        self.assertIn("temperature\\t0.7", text)
        self.assertIn("refiner_tau\\t800", text)
        self.assertIn("rcf_tuple_value_changed_min\\t250", text)
        self.assertIn("refined_missing_policy\\tpreserve_by_sample_idx", text)
        self.assertIn("rerank_or_replacement\\tfalse", text)
        self.assertIn("official_query\\tfalse", text)

    def test_six_cells_use_paired_views_and_checkpoints(self):
        text = WRAPPER.read_text(encoding="utf-8")
        expected = (
            'run_cell 0 17 M0 "${MINIMAL_DLM}" M0',
            'run_cell 1 17 RCF "${RICH_DLM}" RCF',
            'run_cell 2 17 R0 "${RICH_DLM}" R0',
            'run_cell 3 18 M0 "${MINIMAL_DLM}" M0',
            'run_cell 4 18 RCF "${RICH_DLM}" RCF',
            'run_cell 5 18 R0 "${RICH_DLM}" R0',
        )
        for command in expected:
            self.assertIn(command, text)
        self.assertIn("--seed-by-sample-index", text)
        self.assertIn("--diff-steps 800", text)
        self.assertIn("ALLOCATED_GPU_LIST", text)
        self.assertIn('device="${GPU_DEVICES[${gpu}]}"', text)
        self.assertNotIn('CUDA_VISIBLE_DEVICES="${gpu}"', text)

    def test_all_scientific_inputs_have_hash_guards(self):
        text = WRAPPER.read_text(encoding="utf-8")
        for value in (
            "MANIFEST_SHA",
            "M0_SHA",
            "R0_SHA",
            "RCF_SHA",
            "LEDGER_SHA",
            "RICH_ADAPTER_SHA",
            "MINIMAL_ADAPTER_SHA",
            "REFINER_SHA",
            "SAMPLER_SHA",
            "REFINER_SCRIPT_SHA",
            "LIVE_ASSET_MANIFEST_SHA",
            "LIVE_AUDITOR_SHA",
            "CRYS_CODE_SHA",
        ):
            self.assertIn(value, text)

    def test_contract_and_live_assets_are_recomputed(self):
        text = WRAPPER.read_text(encoding="utf-8")
        self.assertIn("contract_payload.txt", text)
        self.assertIn("audit_live_assets.py", text)
        self.assertIn("CRYS_CODE_FILES.sha256", text)
        self.assertIn('x["selection"]["rcf_tuple_value_changed"]', text)
        self.assertIn("ENGINEERING_FAILURE.${BASHPID}.tsv", text)


if __name__ == "__main__":
    unittest.main()
