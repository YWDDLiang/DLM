from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "slurm" / "100_c3fd_h1a2_fusion_prospective_generation.sbatch"


class Slurm100C3FDH1A2FusionGenerationTest(unittest.TestCase):
    def test_six_matched_cells(self):
        text = WRAPPER.read_text(encoding="utf-8")
        self.assertIn("#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:6", text)
        for command in (
            "run_cell 0 17 h1a2_fusion",
            "run_cell 1 17 v2_seed82017",
            "run_cell 2 17 v2_seed82018",
            "run_cell 3 18 h1a2_fusion",
            "run_cell 4 18 v2_seed82017",
            "run_cell 5 18 v2_seed82018",
        ):
            self.assertIn(command, text)
        self.assertIn("alignment_training\\tfalse_stopped_by_user", text)

    def test_inputs_and_sampling_contract_are_pinned(self):
        text = WRAPPER.read_text(encoding="utf-8")
        for name in (
            "MANIFEST_SHA", "H1A2_PLANS_SHA", "V2_PLANS_SHA", "LEDGER_SHA",
            "H1A2_MODEL_SHA", "POLICY17_SHA", "POLICY18_SHA", "REFINER_SHA",
            "SAMPLER_SHA", "REFINER_SCRIPT_SHA",
        ):
            self.assertIn(name, text)
        self.assertIn("temperature\\t0.7", text)
        self.assertIn("schedule\\texact_axis", text)
        self.assertIn("refiner_tau\\t800", text)
        self.assertIn("retry_rerank_replacement\\tfalse", text)

    def test_allocated_gpu_and_refiner_marker(self):
        text = WRAPPER.read_text(encoding="utf-8")
        self.assertIn("ALLOCATED_GPU_LIST", text)
        self.assertIn('device="${GPU_DEVICES[${slot}]}"', text)
        self.assertIn("refinement_metrics.json", text)
        self.assertIn('["output_file"]', text)


if __name__ == "__main__":
    unittest.main()
