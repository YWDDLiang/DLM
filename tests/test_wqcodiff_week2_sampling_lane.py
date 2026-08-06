from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from crystal_dlm.wqcodiff.registry import (
    load_experiment_registry,
    materialize_week2_sampling_plan,
    materialize_week2_training_plan,
)
from scripts.a800.run_week2_sampling_lane import execute_lane, resolve_lane


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "configs/experiments/wyckoff_codiffusion/experiment_registry_v1.yaml"


class Week2SamplingLaneTests(unittest.TestCase):
    def _plan(self, root: Path) -> tuple[Path, dict[str, Path]]:
        registry = load_experiment_registry(REGISTRY)
        training_path = root / "training-plan.json"
        training = materialize_week2_training_plan(
            registry,
            run_id="unit-week2-sampling-lane",
            discrete_engine="B-WQ-D3PM",
            source_bundle_sha256="c" * 64,
            output=training_path,
        )
        checkpoints: dict[str, Path] = {}
        for job in training["jobs"]:
            if job["phase"] != "screen-60000-to-85000":
                continue
            checkpoint = root / job["validation_ema"]
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(job["route"].encode("utf-8"))
            checkpoints[job["route"]] = checkpoint
        lock = root / "revision-threshold-lock.json"
        lock.write_text(
            json.dumps(
                {
                    "schema": "wqcodiff_revision_threshold_lock_v1",
                    "protocol_name": registry.protocol.name,
                    "protocol_sha256": registry.protocol.sha256,
                    "selected_threshold": 0.8,
                }
            ),
            encoding="utf-8",
        )
        plan = root / "sampling-plan.json"
        materialize_week2_sampling_plan(
            registry,
            run_id="unit-week2-sampling-lane",
            training_plan=training_path,
            revision_lock=lock,
            project_root=root,
            output=plan,
        )
        return plan, checkpoints

    def test_resolve_lane_verifies_artifacts_and_modulo_partition(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, checkpoints = self._plan(root)
            lane = resolve_lane(
                plan_path=plan,
                phase="preflight",
                lane_index=0,
                lane_count=4,
                project_root=root,
            )
            self.assertEqual(lane["cells"], 2)
            self.assertEqual(lane["attempts"], 512)
            self.assertEqual(
                [job["phase_ordinal"] for job in lane["jobs"]],
                [0, 4],
            )
            checkpoints["best-discrete-engine"].write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "frozen identity"):
                resolve_lane(
                    plan_path=plan,
                    phase="preflight",
                    lane_index=0,
                    lane_count=4,
                    project_root=root,
                )

    def test_execute_lane_records_hashes_and_refuses_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, _ = self._plan(root)
            lane = resolve_lane(
                plan_path=plan,
                phase="preflight",
                lane_index=1,
                lane_count=4,
                project_root=root,
            )

            def fake_run(argv: list[str], **_kwargs: object) -> mock.Mock:
                output = Path(argv[argv.index("--output") + 1])
                ledger = Path(argv[argv.index("--ledger") + 1])
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text('{"status":"succeeded"}\n', encoding="utf-8")
                output.with_suffix(".summary.json").write_text(
                    '{"ok":true}\n', encoding="utf-8"
                )
                ledger.write_text(
                    '{"status":"submitted"}\n{"status":"succeeded"}\n',
                    encoding="utf-8",
                )
                return mock.Mock(returncode=0)

            manifest = root / "evidence/manifest.json"
            events = root / "evidence/events.jsonl"
            complete = root / "evidence/complete.json"
            with mock.patch(
                "scripts.a800.run_week2_sampling_lane.subprocess.run",
                side_effect=fake_run,
            ):
                result = execute_lane(
                    lane,
                    manifest_path=manifest,
                    events_path=events,
                    complete_path=complete,
                    project_root=root,
                )
            self.assertTrue(result["ok"])
            self.assertEqual(len(events.read_text(encoding="utf-8").splitlines()), 4)
            with self.assertRaises(FileExistsError):
                execute_lane(
                    lane,
                    manifest_path=manifest,
                    events_path=events,
                    complete_path=complete,
                    project_root=root,
                )


if __name__ == "__main__":
    unittest.main()
