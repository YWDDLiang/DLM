from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "slurm" / "75_rich_dlm_live_asset_manifest.sbatch"


class RichDLMLiveAssetManifestSlurmTest(unittest.TestCase):
    def test_job_is_zero_gpu_and_contract_frozen(self):
        text = WRAPPER.read_text(encoding="utf-8")
        self.assertIn("#SBATCH --cpus-per-task=4", text)
        self.assertNotIn("#SBATCH --gres", text)
        self.assertIn("gpu_jobs_used\\t0", text)
        self.assertIn("readonly CONTRACT_SHA=", text)
        self.assertIn("readonly SCRIPT_SHA=", text)
        self.assertIn("readonly RICH_ADAPTER_SHA=", text)
        self.assertIn('mkdir "${CONTRACT_DIR}"', text)

    def test_all_required_assets_are_explicit(self):
        text = WRAPPER.read_text(encoding="utf-8")
        for asset in (
            "rich_dlm",
            "minimal_dlm",
            "dlm_base",
            "model494",
            "mp20",
            "c3fd_checkpoint17",
            "c3fd_checkpoint18",
            "c3fd_data",
            "body_sampler",
            "rich_parser",
            "c3fd_sampler",
            "refiner",
            "evaluator",
        ):
            self.assertIn(f'--asset "{asset}=', text)
        self.assertIn('--expected-file "rich_dlm:adapter_model.safetensors=', text)


if __name__ == "__main__":
    unittest.main()
