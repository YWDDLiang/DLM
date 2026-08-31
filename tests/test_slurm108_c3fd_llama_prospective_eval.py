import hashlib
from pathlib import Path
import re
import shutil
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "slurm/108_c3fd_llama_prospective_eval.sbatch").read_text(
    encoding="utf-8"
)


class Slurm108StaticTest(unittest.TestCase):
    def test_local_evaluator_inputs_match_pins(self):
        for constant, relative in (
            ("FULL_EVAL_SHA", "eval_runtime/run_full_reconstructed_eval.py"),
            ("RAW_REUSE_SHA", "eval_runtime/exact_raw_reuse.py"),
        ):
            match = re.search(rf"^readonly {constant}=([0-9a-f]{{64}})$", SOURCE, re.M)
            self.assertIsNotNone(match)
            observed = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            self.assertEqual(match.group(1), observed)

    def test_raw_runs_before_refined_on_four_cells(self):
        self.assertLess(SOURCE.index("run_four raw_"), SOURCE.index('run_four ""'))
        for arm in ("F", "M", "raw_F", "raw_M"):
            self.assertIn(arm, SOURCE)

    def test_fixed_denominator_and_no_selection(self):
        self.assertIn("denominator\t256", SOURCE)
        self.assertIn("selection\tfalse", SOURCE)
        self.assertIn("official_query\tfalse", SOURCE)

    def test_exact_reuse_is_raw_pair_only(self):
        self.assertIn("--exact-raw-reuse-role", SOURCE)
        self.assertIn("raw_F", SOURCE)
        self.assertIn("raw_M", SOURCE)
        self.assertIn("RAW_REUSE_SHA", SOURCE)
        self.assertIn(
            'if [[ "${arm}" == raw_F || "${arm}" == raw_M ]]', SOURCE
        )
        self.assertIn("exact_raw_reuse_pair_scope\twithin_stream_F_M", SOURCE)
        self.assertIn("refined_exact_reuse\tfalse", SOURCE)

    def test_completed_processes_have_bounded_group_cleanup(self):
        self.assertIn("setsid env", SOURCE)
        self.assertIn("wait_for_completed_process", SOURCE)
        self.assertIn("PROCESS_EXIT_GRACE_SECONDS", SOURCE)
        self.assertIn('[[ -e "${completion_artifact}" ]]', SOURCE)
        self.assertIn('kill -TERM -- "-${pid}"', SOURCE)
        self.assertIn("process_shutdown", SOURCE)
        self.assertNotIn("timeout --", SOURCE)

    def test_bash_syntax(self):
        bash = shutil.which("bash")
        if bash is None:
            self.skipTest("bash is unavailable")
        subprocess.run(
            [bash, "-n", "slurm/108_c3fd_llama_prospective_eval.sbatch"],
            check=True,
            capture_output=True,
            cwd=ROOT,
        )


if __name__ == "__main__":
    unittest.main()
