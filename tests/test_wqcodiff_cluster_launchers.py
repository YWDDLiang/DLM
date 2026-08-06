from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ClusterLauncherTests(unittest.TestCase):
    def test_cpu_launcher_preserves_activated_conda_environment(self) -> None:
        script = (ROOT / "scripts/a800/slurm_cpu_submit.sh").read_text(
            encoding="utf-8"
        )
        activation = script.index('conda activate "\\${ENV_NAME}"')
        execution = script.index('/usr/bin/time -v "\\${COMMAND_ARGV[@]}"')
        self.assertLess(activation, execution)
        self.assertNotIn("bash -lc", script)

    def test_every_launcher_hard_gates_threads_and_isolates_bytecode(self) -> None:
        for relative in (
            "scripts/a800/slurm_submit.sh",
            "scripts/a800/slurm_cpu_submit.sh",
        ):
            with self.subTest(relative=relative):
                script = (ROOT / relative).read_text(encoding="utf-8")
                for variable in (
                    "OPENBLAS_NUM_THREADS",
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                ):
                    self.assertIn(f"export {variable}=1", script)
                self.assertIn("export PYTHONDONTWRITEBYTECODE=1", script)
                self.assertIn(
                    'export PYTHONPYCACHEPREFIX="\\${RUN_DIR}/.pycache/\\${SLURM_JOB_ID}"',
                    script,
                )

    def test_every_launcher_pins_runtime_to_project_root(self) -> None:
        for relative in (
            "scripts/a800/slurm_submit.sh",
            "scripts/a800/slurm_cpu_submit.sh",
        ):
            with self.subTest(relative=relative):
                script = (ROOT / relative).read_text(encoding="utf-8")
                activation = script.index('conda activate "\\${ENV_NAME}"')
                pin = script.index('export PYTHONPATH="\\${PROJECT_ROOT}"')
                assertion = script.index("runtime source mismatch")
                self.assertLess(activation, pin)
                self.assertLess(pin, assertion)
                self.assertNotIn(
                    '\\${PROJECT_ROOT}\\${PYTHONPATH:+:\\${PYTHONPATH}}', script
                )

    def test_launchers_execute_an_argv_array_without_eval(self) -> None:
        for relative in (
            "scripts/a800/slurm_submit.sh",
            "scripts/a800/slurm_cpu_submit.sh",
        ):
            with self.subTest(relative=relative):
                script = (ROOT / relative).read_text(encoding="utf-8")
                self.assertIn("COMMAND_ARGV=(", script)
                self.assertIn('"\\${COMMAND_ARGV[@]}"', script)
                self.assertNotIn('eval "${COMMAND}"', script)

    def test_gpu_launcher_records_utilization(self) -> None:
        script = (ROOT / "scripts/a800/slurm_submit.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("utilization.gpu", script)
        self.assertIn('"mean_utilization_percent"', script)


if __name__ == "__main__":
    unittest.main()
