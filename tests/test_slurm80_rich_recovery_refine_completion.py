from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "slurm" / "80_rich_recovery_refine_completion.sbatch"


class RichRecoveryRefineCompletionSlurmTest(unittest.TestCase):
    def test_recovery_uses_five_gpus_and_never_reruns_body(self):
        text = WRAPPER.read_text(encoding="utf-8")
        self.assertIn("#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:5", text)
        self.assertIn("#SBATCH --cpus-per-task=40", text)
        self.assertIn("body_rerun\\tfalse", text)
        self.assertNotIn("sample_sgtc_l6.py", text)
        self.assertIn("completed_parent_cell\\tstream17/M0", text)

    def test_five_missing_cells_keep_original_seeds_and_tau(self):
        text = WRAPPER.read_text(encoding="utf-8")
        for command in (
            "run_refine 0 17 RCF 101117",
            "run_refine 1 17 R0 101117",
            "run_refine 2 18 M0 102117",
            "run_refine 3 18 RCF 102117",
            "run_refine 4 18 R0 102117",
        ):
            self.assertIn(command, text)
        self.assertIn("--diff-steps 800", text)
        self.assertIn("--seed-by-sample-index", text)

    def test_every_body_and_asset_is_hash_guarded(self):
        text = WRAPPER.read_text(encoding="utf-8")
        for value in (
            "S17_RCF_SHA",
            "S17_R0_SHA",
            "S18_M0_SHA",
            "S18_RCF_SHA",
            "S18_R0_SHA",
            "COMPLETED_S17_M0_SHA",
            "REFINER_SHA",
            "REFINER_SCRIPT_SHA",
        ):
            self.assertIn(value, text)


if __name__ == "__main__":
    unittest.main()
