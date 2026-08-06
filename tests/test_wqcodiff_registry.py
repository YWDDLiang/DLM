from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from crystal_dlm.wqcodiff.registry import (
    load_experiment_registry,
    materialize_day7_plan,
    materialize_week2_sampling_plan,
    materialize_week2_training_plan,
)


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = (
    ROOT
    / "configs/experiments/wyckoff_codiffusion/experiment_registry_v1.yaml"
)


class ExperimentRegistryTests(unittest.TestCase):
    def test_registered_day7_matrix_expands_to_the_preregistered_counts(self) -> None:
        registry = load_experiment_registry(REGISTRY)
        with tempfile.TemporaryDirectory() as directory:
            result = materialize_day7_plan(
                registry,
                run_id="unit-day7",
                source_bundle_sha256="a" * 64,
                output=Path(directory) / "plan.json",
            )
        self.assertEqual(result["source_bundle_sha256"], "a" * 64)
        self.assertEqual(len(result["jobs"]), 297)
        self.assertEqual(result["phase_summary"]["day7-primary"]["jobs"], 180)
        self.assertEqual(
            result["phase_summary"]["day7-primary"]["attempts"], 737_280
        )
        self.assertEqual(result["phase_summary"]["day7-intervention"]["jobs"], 72)
        self.assertEqual(
            result["phase_summary"]["threshold-calibration"]["jobs"], 45
        )

    def test_threshold_pairing_is_threshold_independent_and_later_jobs_require_lock_value(self) -> None:
        registry = load_experiment_registry(REGISTRY)
        with tempfile.TemporaryDirectory() as directory:
            result = materialize_day7_plan(
                registry,
                run_id="unit-pairs",
                source_bundle_sha256="b" * 64,
                output=Path(directory) / "plan.json",
            )
        calibration = [
            job for job in result["jobs"] if job["phase"] == "threshold-calibration"
        ]
        grouped: dict[str, set[str]] = {}
        for job in calibration:
            argv = job["argv"]
            threshold = argv[argv.index("--revision-threshold") + 1]
            grouped.setdefault(job["pairing_id"], set()).add(threshold)
        self.assertTrue(grouped)
        self.assertTrue(all(len(values) == 5 for values in grouped.values()))
        self.assertTrue(
            all(
                job["argv"][job["argv"].index("--runtime-source-bundle-sha256") + 1]
                == "b" * 64
                for job in calibration
            )
        )
        primary = next(job for job in result["jobs"] if job["phase"] == "day7-primary")
        index = primary["argv"].index("--revision-threshold") + 1
        self.assertEqual(primary["argv"][index], "${REVISION_THRESHOLD}")

    def test_week2_plan_binds_two_shared_stages_and_five_screens(self) -> None:
        registry = load_experiment_registry(REGISTRY)
        with tempfile.TemporaryDirectory() as directory:
            result = materialize_week2_training_plan(
                registry,
                run_id="unit-week2",
                discrete_engine="B-WQ-D3PM",
                source_bundle_sha256="a" * 64,
                output=Path(directory) / "week2.json",
            )
        self.assertEqual(result["summary"], {"jobs": 7, "shared": 2, "screening": 5})
        self.assertEqual(len(result["datasets"]), 8)
        shared = [job for job in result["jobs"] if job["phase"] == "shared-60000"]
        screens = [
            job
            for job in result["jobs"]
            if job["phase"] == "screen-60000-to-85000"
        ]
        self.assertEqual(len(shared), 2)
        self.assertTrue(all("--stop-after-shared" in job["argv"] for job in shared))
        self.assertEqual(len(screens), 5)
        self.assertTrue(all(job["target_update"] == 85000 for job in screens))
        self.assertTrue(all("--shared-checkpoint" in job["argv"] for job in screens))
        self.assertTrue(all("--stop-after-update" in job["argv"] for job in screens))
        discrete = next(job for job in screens if job["route"] == "best-discrete-engine")
        self.assertEqual(discrete["variant"], "B-WQ-D3PM")
        self.assertEqual(discrete["depends_on"], ["shared-wyckoff"])
        atom = next(job for job in screens if job["route"] == "atom-joint")
        self.assertEqual(atom["depends_on"], ["shared-atom"])
        self.assertTrue(result["optimizer_checkpoint_required_for_continuation"])

    def test_week2_plan_rejects_unfrozen_engine_and_bad_source_hash(self) -> None:
        registry = load_experiment_registry(REGISTRY)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "bad.json"
            with self.assertRaisesRegex(ValueError, "Day-7 AR/D3PM/DLM"):
                materialize_week2_training_plan(
                    registry,
                    run_id="unit-week2",
                    discrete_engine="M-WQ-STRAT-GEO",
                    source_bundle_sha256="a" * 64,
                    output=output,
                )
            with self.assertRaisesRegex(ValueError, "source-bundle SHA256"):
                materialize_week2_training_plan(
                    registry,
                    run_id="unit-week2",
                    discrete_engine="B-WQ-AR",
                    source_bundle_sha256="not-a-sha",
                    output=output,
                )

    def test_week2_sampling_plan_binds_checkpoints_threshold_and_tau_grid(self) -> None:
        registry = load_experiment_registry(REGISTRY)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training_path = root / "training-plan.json"
            training = materialize_week2_training_plan(
                registry,
                run_id="unit-week2-sampling",
                discrete_engine="B-WQ-D3PM",
                source_bundle_sha256="b" * 64,
                output=training_path,
            )
            for job in training["jobs"]:
                if job["phase"] != "screen-60000-to-85000":
                    continue
                checkpoint = root / job["validation_ema"]
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                checkpoint.write_bytes(job["route"].encode("utf-8"))
            lock_path = root / "revision-threshold-lock.json"
            lock_path.write_text(
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
            result = materialize_week2_sampling_plan(
                registry,
                run_id="unit-week2-sampling",
                training_plan=training_path,
                revision_lock=lock_path,
                project_root=root,
                output=root / "sampling-plan.json",
            )
        self.assertEqual(result["configuration_count"], 8)
        self.assertEqual(
            result["phase_summary"]["preflight"],
            {"jobs": 8, "attempts": 2048, "backbone_calls": 131072},
        )
        self.assertEqual(
            result["phase_summary"]["development"],
            {"jobs": 24, "attempts": 24000, "backbone_calls": 1536000},
        )
        development = [job for job in result["jobs"] if job["phase"] == "development"]
        self.assertEqual({job["pairing_id"] for job in development}, {"week2-matched-screen-v1"})
        disc_once = [job for job in development if job["route"] == "disc-once"]
        self.assertEqual(
            {job["disc_once_tau"] for job in disc_once},
            {0.25, 0.5, 0.75, 1.0},
        )
        self.assertTrue(all("--revision-lock" in job["argv"] for job in result["jobs"]))


if __name__ == "__main__":
    unittest.main()
