import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class H1A2FactorialAssemblyTests(unittest.TestCase):
    def test_cli_preserves_four_complete_all_attempt_denominators(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = {}
            for arm in ("M00", "M10", "M01", "M11"):
                rows = []
                for sample_idx in range(2):
                    planner_arm = "P0" if arm in ("M00", "M01") else "Pstar"
                    row = {
                        "sample_idx": sample_idx,
                        "evaluation_order": sample_idx,
                        "factorial_arm": arm,
                        "planner_arm": planner_arm,
                        "attempt_status": "complete",
                        "earliest_failure_stage": None,
                        "planner_sampling_seed": 100 + sample_idx,
                        "body_sampling_seed": 200 + sample_idx,
                        "refiner_sampling_seed": 300 + sample_idx,
                        "raw_plan_text_sha256": (
                            ("a" if planner_arm == "P0" else "b") * 64
                        ),
                        "plan_text_sha256": (
                            ("c" if planner_arm == "P0" else "d") * 64
                        ),
                        "body_prompt_sha256": (
                            ("e" if planner_arm == "P0" else "f") * 64
                        ),
                    }
                    if sample_idx == 1 and planner_arm == "P0":
                        row.update(
                            {
                                "attempt_status": "failed",
                                "earliest_failure_stage": "planner",
                            }
                        )
                        for key in (
                            "raw_plan_text_sha256",
                            "plan_text_sha256",
                            "body_prompt_sha256",
                        ):
                            row.pop(key)
                    rows.append(row)
                path = root / f"{arm}.jsonl"
                path.write_text(
                    "".join(json.dumps(row) + "\n" for row in rows),
                    encoding="utf-8",
                )
                paths[arm] = path
            output = root / "assembled"
            command = [
                sys.executable,
                str(PROJECT_ROOT / "scripts/assemble_h1a2_factorial_attempts.py"),
                "--m00-attempts",
                str(paths["M00"]),
                "--m10-attempts",
                str(paths["M10"]),
                "--m01-attempts",
                str(paths["M01"]),
                "--m11-attempts",
                str(paths["M11"]),
                "--num-samples",
                "2",
                "--output-dir",
                str(output),
            ]
            completed = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = json.loads(
                (output / "factorial_assembly_report.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(report["total_attempts"], 8)
            self.assertEqual(report["pairing"]["planner_failure_pairs"], 1)
            rows = [
                json.loads(line)
                for line in (output / "factorial_attempts.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(len(rows), 8)
            self.assertEqual(
                [(row["sample_idx"], row["factorial_arm"]) for row in rows[:4]],
                [(0, "M00"), (0, "M10"), (0, "M01"), (0, "M11")],
            )


if __name__ == "__main__":
    unittest.main()
