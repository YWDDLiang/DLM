from pathlib import Path
import shutil
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "slurm/112_fused_planner_compact_v2_body_direct.sbatch").read_text()


class Slurm112Test(unittest.TestCase):
    def test_single_cell_speed_first_contract(self):
        self.assertIn("#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1", SOURCE)
        self.assertIn("#SBATCH --cpus-per-task=8", SOURCE)
        self.assertIn("seed82017", SOURCE)
        self.assertIn("stream\t17", SOURCE)
        self.assertIn("--seed 91117", SOURCE)
        self.assertIn("--num-samples 256", SOURCE)

    def test_raw_body_and_direct_only(self):
        self.assertIn("sample_sgtc_l6.py", SOURCE)
        self.assertIn("assemble_raw_body_repeat.py", SOURCE)
        self.assertIn("run_crysllmgen_metrics.py", SOURCE)
        self.assertIn("model494\tfalse", SOURCE)
        self.assertIn("chgnet\tfalse", SOURCE)
        self.assertNotIn("refine_dlm", SOURCE)
        self.assertNotIn("run_full_reconstructed_eval", SOURCE)
        self.assertNotIn("MP_API", SOURCE)

    def test_no_selection_or_recovery_loop(self):
        self.assertIn("trajectory_per_plan\t1", SOURCE)
        self.assertIn("retry_filter_replacement_rerank_best_of_n\tfalse", SOURCE)
        self.assertNotIn("while true", SOURCE.lower())
        self.assertNotIn("for attempt", SOURCE.lower())

    def test_bash_syntax(self):
        bash = shutil.which("bash")
        if bash is None:
            self.skipTest("bash unavailable")
        subprocess.run(
            [bash, "-n", "slurm/112_fused_planner_compact_v2_body_direct.sbatch"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        )


if __name__ == "__main__":
    unittest.main()
