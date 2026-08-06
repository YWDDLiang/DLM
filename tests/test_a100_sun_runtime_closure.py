from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STAGER = ROOT / "scripts/a800/stage_crysllmgen_a100_sun_runtime.py"


class A100SUNRuntimeClosureTests(unittest.TestCase):
    def _stage(self, runtime: Path) -> dict:
        completed = subprocess.run(
            [
                sys.executable,
                str(STAGER),
                "--project-root",
                str(ROOT),
                "--runtime-root",
                str(runtime),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(completed.stdout)

    def test_staged_runtime_executes_isolated_import_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runtime = Path(temporary) / "runtime"
            report = self._stage(runtime)
            self.assertEqual(report["status"], "pass")
            self.assertEqual(report["import_preflight"]["status"], "pass")
            origins = report["import_preflight"]["origins"]
            self.assertEqual(
                origins["crystal_dlm.wqcodiff.contracts"],
                "crystal_dlm/wqcodiff/contracts.py",
            )
            self.assertEqual(
                origins["crystal_dlm.wqcodiff.crysllmgen.a100_sun"],
                "crystal_dlm/wqcodiff/crysllmgen/a100_sun.py",
            )

    def test_missing_delayed_dependency_fails_before_scientific_work(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runtime = Path(temporary) / "runtime"
            self._stage(runtime)
            (runtime / "crystal_dlm/wqcodiff/contracts.py").unlink()
            completed = subprocess.run(
                [
                    sys.executable,
                    str(STAGER),
                    "--runtime-root",
                    str(runtime),
                    "--verify-only",
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("dependency closure is incomplete", completed.stderr)

    def test_nonisolated_module_origin_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runtime = Path(temporary) / "runtime"
            self._stage(runtime)
            staged_contracts = runtime / "crystal_dlm/wqcodiff/contracts.py"
            staged_contracts.unlink()
            staged_contracts.symlink_to(ROOT / "crystal_dlm/wqcodiff/contracts.py")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(STAGER),
                    "--runtime-root",
                    str(runtime),
                    "--verify-only",
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("runtime dependency escaped frozen root", completed.stderr)


if __name__ == "__main__":
    unittest.main()
