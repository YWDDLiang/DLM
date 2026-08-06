import json
import tempfile
import unittest
from pathlib import Path

from scripts.a800.adapt_wq_parent_probe_for_eval import (
    adapt_parent_probe_for_eval,
)


ADAPTER_PATCH = "1" * 64
EVALUATION_PATCH = "2" * 64
ROOT = Path(__file__).resolve().parents[1]
PLAN = (
    ROOT
    / "configs/experiments/wyckoff_codiffusion/"
    "wq_parent_csp_sun256_v1.json"
)
SBATCH = ROOT / "scripts/a800/wq_parent_csp_sun256_v1/pipeline.sbatch"
SUBMIT = ROOT / "scripts/a800/wq_parent_csp_sun256_v1/submit_once.sh"


def _row(ordinal: int, status: str = "succeeded") -> dict:
    row = {
        "schema": "wq_parent_csp_probe_attempt_v1",
        "attempt_id": f"attempt-{ordinal}",
        "method": "DIAG-WQ-PROPOSAL-PARENT-CSP32",
        "status": status,
        "reason": "" if status == "succeeded" else "synthetic failure",
        "ordinal": ordinal,
        "retry_or_replacement_used": False,
        "adapter_training_execution_patch_sha256": ADAPTER_PATCH,
        "diagnostic_execution_patch_sha256": EVALUATION_PATCH,
    }
    if status == "succeeded":
        row.update(
            {
                "final_structure": {
                    "@module": "pymatgen.core.structure",
                    "@class": "Structure",
                    "charge": 0,
                    "lattice": {
                        "matrix": [[3, 0, 0], [0, 3, 0], [0, 0, 3]],
                        "pbc": [True, True, True],
                        "a": 3,
                        "b": 3,
                        "c": 3,
                        "alpha": 90,
                        "beta": 90,
                        "gamma": 90,
                        "volume": 27,
                    },
                    "properties": {},
                    "sites": [],
                },
                "final_volume": 27.0,
                "atom_count": 1,
            }
        )
    return row


class ParentCSPSUN256AdapterTests(unittest.TestCase):
    def _write(self, path: Path, rows: list[dict]) -> None:
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )

    def test_maps_final_structure_without_changing_denominator(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.jsonl"
            output = root / "generation.jsonl"
            manifest = root / "manifest.json"
            self._write(source, [_row(256), _row(257, "failed")])
            result = adapt_parent_probe_for_eval(
                source_jsonl=source,
                output_jsonl=output,
                manifest_json=manifest,
                expected_attempts=2,
                expected_start_ordinal=256,
                adapter_training_execution_patch_sha256=ADAPTER_PATCH,
                evaluation_execution_patch_sha256=EVALUATION_PATCH,
            )
            rows = [
                json.loads(line)
                for line in output.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(result["attempts"], 2)
            self.assertEqual(result["succeeded"], 1)
            self.assertEqual(result["failed"], 1)
            self.assertEqual(rows[0]["schema"], "wqcodiff_generation_attempt_v1")
            self.assertEqual(rows[0]["structure"], rows[0]["final_structure"])
            self.assertIsNone(rows[1]["structure"])
            self.assertEqual(
                [row["attempt_id"] for row in rows],
                ["attempt-256", "attempt-257"],
            )

    def test_rejects_noncontiguous_ordinals(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.jsonl"
            self._write(source, [_row(256), _row(258)])
            with self.assertRaisesRegex(ValueError, "ordinal"):
                adapt_parent_probe_for_eval(
                    source_jsonl=source,
                    output_jsonl=root / "generation.jsonl",
                    manifest_json=root / "manifest.json",
                    expected_attempts=2,
                    expected_start_ordinal=256,
                    adapter_training_execution_patch_sha256=ADAPTER_PATCH,
                    evaluation_execution_patch_sha256=EVALUATION_PATCH,
                )

    def test_rejects_retry_or_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.jsonl"
            row = _row(256)
            row["retry_or_replacement_used"] = True
            self._write(source, [row])
            with self.assertRaisesRegex(ValueError, "retry/replacement"):
                adapt_parent_probe_for_eval(
                    source_jsonl=source,
                    output_jsonl=root / "generation.jsonl",
                    manifest_json=root / "manifest.json",
                    expected_attempts=1,
                    expected_start_ordinal=256,
                    adapter_training_execution_patch_sha256=ADAPTER_PATCH,
                    evaluation_execution_patch_sha256=EVALUATION_PATCH,
                )


class ParentCSPSUN256PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.plan = json.loads(PLAN.read_text(encoding="utf-8"))
        self.sbatch = SBATCH.read_text(encoding="utf-8")
        self.submit = SUBMIT.read_text(encoding="utf-8")

    def test_plan_freezes_disjoint_256_attempt_panel(self) -> None:
        generation = self.plan["generation"]
        self.assertEqual(generation["attempts"], 256)
        self.assertEqual(generation["start_ordinal"], 256)
        self.assertEqual(generation["end_ordinal_inclusive"], 511)
        self.assertFalse(generation["attempt_retry_or_replacement_used"])
        self.assertEqual(self.plan["sun"]["denominator"], 256)
        self.assertTrue(self.plan["crysllmgen_direct_metrics"]["enabled"])

    def test_single_job_runs_generation_metrics_and_sun(self) -> None:
        self.assertIn("#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1", self.sbatch)
        self.assertIn("#SBATCH --cpus-per-task=16", self.sbatch)
        self.assertIn("--attempts 256", self.sbatch)
        self.assertIn("--start-ordinal 256", self.sbatch)
        self.assertIn("run_crysllmgen_metrics.py", self.sbatch)
        self.assertIn("run_crysllmgen_a100_sun.py", self.sbatch)
        self.assertIn("--expected-attempts 256", self.sbatch)
        self.assertNotIn("train_crysllmgen", self.sbatch)

    def test_submit_records_queue_without_blocking_or_retrying(self) -> None:
        self.assertIn("preexisting_queue_rows", self.submit)
        self.assertNotIn("waits for zero", self.submit)
        self.assertNotIn("scancel", self.submit)
        self.assertIn("submission_failed_no_retry", self.submit)
        self.assertIn('"long_training_submitted": False', self.submit)

    def test_exact_authorization_is_registered_in_gate_and_installer(self) -> None:
        authorization = "user_wq_parent_csp_sun256_v1_2026-07-24"
        installer = (
            ROOT / "scripts/a800/install_authorized_patch.py"
        ).read_text(encoding="utf-8")
        gate = (
            ROOT / "crystal_dlm/wqcodiff/crysllmgen/gate.py"
        ).read_text(encoding="utf-8")
        self.assertIn(authorization, gate)
        self.assertIn(authorization, installer)


if __name__ == "__main__":
    unittest.main()
