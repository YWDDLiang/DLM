from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "slurm" / "83_faithful_rich_diagnostic_generation.sbatch"


class Slurm83FaithfulRichDiagnosticGenerationTest(unittest.TestCase):
    def test_resources_and_four_frozen_cells(self):
        text = WRAPPER.read_text(encoding="utf-8")
        self.assertIn("#SBATCH --cpus-per-task=32", text)
        self.assertIn("#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:4", text)
        self.assertIn("faithful_rich_diagnostic_h0_r0s_256_v2_20260830", text)
        for command in (
            "run_cell 0 17 H0",
            "run_cell 1 17 R0S",
            "run_cell 2 18 H0",
            "run_cell 3 18 R0S",
        ):
            self.assertIn(command, text)
        self.assertNotIn(" M0 &", text)
        self.assertNotIn(" R0 &", text)

    def test_science_contract_and_hash_guards(self):
        text = WRAPPER.read_text(encoding="utf-8")
        for value in (
            "H0_SHA",
            "R0S_SHA",
            "LEDGER_SHA",
            "MANIFEST_SHA",
            "RICH_ADAPTER_SHA",
            "REFINER_SHA",
            "SAMPLER_SHA",
            "REFINER_SCRIPT_SHA",
            "CRYS_CODE_SHA",
        ):
            self.assertIn(value, text)
        self.assertIn("temperature\\t0.7", text)
        self.assertIn("schedule\\texact_axis", text)
        self.assertIn("refiner_tau\\t800", text)
        self.assertIn("attempts_per_plan\\t1", text)
        self.assertIn("official_query\\tfalse", text)

    def test_refiner_marker_uses_metrics_output_file(self):
        text = WRAPPER.read_text(encoding="utf-8")
        self.assertIn("refinement_metrics.json", text)
        self.assertIn('["output_file"]', text)
        self.assertIn('touch "${output}/refine/_SUCCESS"', text)
        self.assertIn("ALLOCATED_GPU_LIST", text)
        self.assertIn('device="${GPU_DEVICES[${slot}]}"', text)
        self.assertNotIn('CUDA_VISIBLE_DEVICES="${slot}"', text)


if __name__ == "__main__":
    unittest.main()
