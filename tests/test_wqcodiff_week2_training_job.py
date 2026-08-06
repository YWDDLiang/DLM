from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from crystal_dlm.wqcodiff.registry import (
    load_experiment_registry,
    materialize_week2_training_plan,
)
from scripts.a800.run_week2_training_job import execute_job, resolve_job


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "configs/experiments/wyckoff_codiffusion/experiment_registry_v1.yaml"


class Week2TrainingJobTests(unittest.TestCase):
    def _plan(self, root: Path) -> Path:
        path = root / "plan.json"
        materialize_week2_training_plan(
            load_experiment_registry(REGISTRY),
            run_id="unit-week2-run",
            discrete_engine="B-WQ-D3PM",
            source_bundle_sha256="a" * 64,
            output=path,
        )
        return path

    def test_screen_resolution_requires_full_shared_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = self._plan(root)
            with self.assertRaisesRegex(FileNotFoundError, "dependency checkpoint"):
                resolve_job(
                    plan_path=plan,
                    job_id="screen-best-discrete-engine",
                    project_root=root,
                )
            checkpoint = (
                root
                / "runs/unit-week2-run/outputs/training/shared-wyckoff/checkpoint_0060000.pt"
            )
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"shared-optimizer-state")
            resolved = resolve_job(
                plan_path=plan,
                job_id="screen-best-discrete-engine",
                project_root=root,
            )
            self.assertEqual(resolved["variant"], "B-WQ-D3PM")
            self.assertEqual(
                resolved["dependencies"]["shared-wyckoff"]["sha256"],
                __import__("hashlib").sha256(b"shared-optimizer-state").hexdigest(),
            )
            self.assertTrue(
                any(
                    value.endswith("checkpoint_0060000.pt")
                    for value in resolved["argv"]
                )
            )

    def test_execute_records_terminal_hashes_and_refuses_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = self._plan(root)
            resolved = resolve_job(
                plan_path=plan,
                job_id="shared-wyckoff",
                project_root=root,
            )
            output = Path(resolved["output_dir"])

            def fake_run(*_args: object, **_kwargs: object) -> mock.Mock:
                output.mkdir(parents=True)
                (output / "shared_training_complete.json").write_text(
                    json.dumps({"ok": True}), encoding="utf-8"
                )
                (output / "checkpoint_0060000.pt").write_bytes(b"checkpoint")
                return mock.Mock(returncode=0)

            manifest = root / "evidence/manifest.json"
            events = root / "evidence/events.jsonl"
            complete = root / "evidence/complete.json"
            with mock.patch(
                "scripts.a800.run_week2_training_job.subprocess.run",
                side_effect=fake_run,
            ):
                result = execute_job(
                    resolved,
                    manifest_path=manifest,
                    events_path=events,
                    complete_path=complete,
                    project_root=root,
                )
            self.assertTrue(result["ok"])
            self.assertEqual(len(events.read_text(encoding="utf-8").splitlines()), 2)
            self.assertEqual(
                result["continuation_checkpoint"]["sha256"],
                __import__("hashlib").sha256(b"checkpoint").hexdigest(),
            )
            self.assertTrue(
                result["training_evidence"]["path"].endswith(
                    "shared_training_complete.json"
                )
            )
            with self.assertRaises(FileExistsError):
                execute_job(
                    resolved,
                    manifest_path=manifest,
                    events_path=events,
                    complete_path=complete,
                    project_root=root,
                )

    def test_screen_uses_partial_training_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = self._plan(root)
            shared = (
                root
                / "runs/unit-week2-run/outputs/training/shared-wyckoff/checkpoint_0060000.pt"
            )
            shared.parent.mkdir(parents=True)
            shared.write_bytes(b"shared")
            resolved = resolve_job(
                plan_path=plan,
                job_id="screen-best-discrete-engine",
                project_root=root,
            )
            output = Path(resolved["output_dir"])

            def fake_run(*_args: object, **_kwargs: object) -> mock.Mock:
                output.mkdir(parents=True)
                (output / "partial_training_complete.json").write_text(
                    json.dumps({"ok": True}), encoding="utf-8"
                )
                (output / "checkpoint_0085000.pt").write_bytes(b"checkpoint")
                (output / "model_ema_final.pt").write_bytes(b"ema")
                return mock.Mock(returncode=0)

            with mock.patch(
                "scripts.a800.run_week2_training_job.subprocess.run",
                side_effect=fake_run,
            ):
                result = execute_job(
                    resolved,
                    manifest_path=root / "evidence/manifest.json",
                    events_path=root / "evidence/events.jsonl",
                    complete_path=root / "evidence/complete.json",
                    project_root=root,
                )
            self.assertTrue(result["ok"])
            self.assertTrue(
                result["training_evidence"]["path"].endswith(
                    "partial_training_complete.json"
                )
            )
            self.assertIsNotNone(result["validation_ema"])


if __name__ == "__main__":
    unittest.main()
