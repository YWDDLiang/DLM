from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


THREAD_VARIABLES = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


class NumericThreadEnvironmentTests(unittest.TestCase):
    def _run_isolated_import(self, environment: dict[str, str]) -> None:
        pythonpath = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            str(ROOT)
            if not pythonpath
            else str(ROOT) + os.pathsep + pythonpath
        )
        expected = repr({name: environment.get(name, "1") for name in THREAD_VARIABLES})
        subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import os; import crystal_dlm.wqcodiff; "
                    f"assert {{name: os.environ[name] for name in {THREAD_VARIABLES!r}}} "
                    f"== {expected}"
                ),
            ],
            cwd=ROOT,
            env=environment,
            check=True,
        )

    def test_package_import_defaults_missing_thread_counts_to_one(self) -> None:
        environment = os.environ.copy()
        for name in THREAD_VARIABLES:
            environment.pop(name, None)
        self._run_isolated_import(environment)

    def test_package_import_preserves_explicit_slurm_thread_counts(self) -> None:
        for value in ("4", "8", "16"):
            environment = os.environ.copy()
            environment.update({name: value for name in THREAD_VARIABLES})
            self._run_isolated_import(environment)


if __name__ == "__main__":
    unittest.main()
