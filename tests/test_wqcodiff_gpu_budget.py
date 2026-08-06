from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/a800/gpu_budget.py"
SPEC = importlib.util.spec_from_file_location("wqcodiff_gpu_budget", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class GpuBudgetTests(unittest.TestCase):
    def test_completed_usage_and_proposed_wave_are_both_counted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            usage = root / "run/outputs/job-1.job_usage.json"
            usage.parent.mkdir(parents=True)
            usage.write_text(
                json.dumps(
                    {
                        "schema": "wqcodiff_slurm_usage_v1",
                        "slurm_job_id": "1",
                        "slurm_job_name": "smoke",
                        "run_id": "r",
                        "week": 1,
                        "gpu_hours": 100.0,
                    }
                ),
                encoding="utf-8",
            )
            passed = MODULE.audit_budget(
                root, current_week=1, proposed_gpu_hours=70.0
            )
            failed = MODULE.audit_budget(
                root, current_week=1, proposed_gpu_hours=81.0
            )
            self.assertTrue(passed["ok"])
            self.assertFalse(failed["ok"])
            self.assertFalse(
                failed["checks"]["current_week_cumulative_within_limit"]
            )

    def test_multiple_runtime_roots_are_summed_and_duplicate_job_ids_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            roots = (root / "old", root / "new")
            for index, value in enumerate((20.0, 30.0), start=1):
                path = roots[index - 1] / f"run/outputs/{index}.job_usage.json"
                path.parent.mkdir(parents=True)
                path.write_text(
                    json.dumps(
                        {
                            "schema": "wqcodiff_slurm_usage_v1",
                            "slurm_job_id": str(index),
                            "slurm_job_name": f"job-{index}",
                            "run_id": f"r-{index}",
                            "week": 1,
                            "gpu_hours": value,
                        }
                    ),
                    encoding="utf-8",
                )
            result = MODULE.audit_budget(
                roots, current_week=1, proposed_gpu_hours=10.0
            )
            self.assertTrue(result["ok"])
            self.assertEqual(result["consumed_total"], 50.0)
            self.assertEqual(result["projected_total"], 60.0)
            duplicate = roots[1] / "duplicate.job_usage.json"
            duplicate.write_text(
                (roots[0] / "run/outputs/1.job_usage.json").read_text(),
                encoding="utf-8",
            )
            failed = MODULE.audit_budget(roots, current_week=1)
            self.assertFalse(failed["ok"])
            self.assertFalse(failed["checks"]["usage_records_valid"])


if __name__ == "__main__":
    unittest.main()
