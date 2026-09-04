from pathlib import Path
import re
import shutil
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "slurm" / "219_spad_basin_paper1000_eval.sbatch"
FINALIZER = ROOT / "scripts" / "finalize_spad_paper1000_official.py"


class Paper1000EvalWrapperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.wrapper = WRAPPER.read_text(encoding="utf-8")
        cls.finalizer = FINALIZER.read_text(encoding="utf-8")

    def test_all_gpus_raw_then_refined(self) -> None:
        self.assertLess(
            self.wrapper.index("run_endpoint raw"),
            self.wrapper.index("run_endpoint refined"),
        )
        self.assertIn('for ((rank=0; rank<WORLD_SIZE; rank++))', self.wrapper)
        self.assertIn("--denominator 1000", self.wrapper)

    def test_no_direct_or_new_query(self) -> None:
        self.assertIn("direct\\tfalse", self.wrapper)
        self.assertIn("new_official_query\\tfalse", self.wrapper)
        self.assertNotIn("run_direct", self.wrapper)

    def test_uncovered_cache_is_conservative(self) -> None:
        self.assertIn("official_cache_not_covered", self.finalizer)
        self.assertIn("uncovered_cache_rows_count_as_not_stable", self.finalizer)
        self.assertIn('DENOMINATOR = 1000', self.finalizer)

    def test_embedded_python_is_syntactically_valid(self) -> None:
        blocks = re.findall(r"<<'PY'\n(.*?)\nPY", self.wrapper, flags=re.DOTALL)
        self.assertGreater(len(blocks), 0)
        for index, block in enumerate(blocks):
            compile(block, f"{WRAPPER.name}:heredoc-{index}", "exec")

    @unittest.skipUnless(shutil.which("bash"), "bash is unavailable")
    def test_bash_syntax(self) -> None:
        result = subprocess.run(
            [shutil.which("bash"), "-n"],
            input=self.wrapper.encode("utf-8"),
            capture_output=True,
            check=False,
        )
        self.assertEqual(
            result.returncode, 0, result.stderr.decode("utf-8", errors="replace")
        )


if __name__ == "__main__":
    unittest.main()
