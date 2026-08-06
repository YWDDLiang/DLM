from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from crystal_dlm.wqcodiff.registry import (
    load_experiment_registry,
    materialize_week2_sampling_plan,
    materialize_week2_training_plan,
)
from crystal_dlm.wqcodiff.screening import freeze_week2_champion


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "configs/experiments/wyckoff_codiffusion/experiment_registry_v1.yaml"


class Week2ScreeningTests(unittest.TestCase):
    def _artifacts(self, root: Path) -> tuple[Path, dict[str, Path]]:
        registry = load_experiment_registry(REGISTRY)
        training_path = root / "training-plan.json"
        training = materialize_week2_training_plan(
            registry,
            run_id="unit-week2-screen",
            discrete_engine="B-WQ-D3PM",
            source_bundle_sha256="d" * 64,
            output=training_path,
        )
        for job in training["jobs"]:
            if job["phase"] != "screen-60000-to-85000":
                continue
            for key in ("validation_ema", "continuation_checkpoint"):
                path = root / job[key]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(f"{job['route']}:{key}".encode("utf-8"))
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
        sampling_path = root / "sampling-plan.json"
        sampling = materialize_week2_sampling_plan(
            registry,
            run_id="unit-week2-screen",
            training_plan=training_path,
            revision_lock=lock,
            project_root=root,
            output=sampling_path,
        )
        specs: dict[str, dict[str, object]] = {}
        for job in sampling["jobs"]:
            if job["phase"] == "development":
                specs.setdefault(
                    job["configuration_id"],
                    {
                        "variant": job["variant"],
                        "tau": job["disc_once_tau"],
                        "pairing_id": job["pairing_id"],
                    },
                )
        sun_counts = {
            "best-discrete-engine": 2,
            "joint-no-revision": 3,
            "disc-once-tau-0p25": 1,
            "disc-once-tau-0p5": 1,
            "disc-once-tau-0p75": 2,
            "disc-once-tau-1p0": 1,
            "atom-joint": 3,
            "stratified-geometry": 3,
        }
        evaluation_paths: dict[str, Path] = {}
        for configuration_id, spec in specs.items():
            path = root / f"{configuration_id}.jsonl"
            rows = []
            for ordinal, sampling_seed in enumerate((101, 202, 303)):
                rows.append(
                    {
                        "schema": "wqcodiff_mlip_sun_attempt_v1",
                        "attempt_id": f"{configuration_id}:{sampling_seed}",
                        "pair_id": f"pair:{sampling_seed}",
                        "pairing_id": spec["pairing_id"],
                        "method": spec["variant"],
                        "disc_once_tau": spec["tau"],
                        "training_seed": 11,
                        "sampling_seed": sampling_seed,
                        "status": "succeeded",
                        "evaluator": "mattersim",
                        "stage": "raw",
                        "contract_hash": "contract",
                        "hull_sha256": "hull",
                        "novelty_reference_sha256": "novelty",
                        "matcher_contract_sha256": "matcher",
                        "subset_hash": "subset",
                        "mlip_sun_at_0p1": ordinal < sun_counts[configuration_id],
                        "novel_unique_standard": ordinal < sun_counts[configuration_id],
                        "mlip_sun_at_0p0": ordinal == 0,
                        "generation_calls": {"joint": 64, "prior": 2},
                    }
                )
            path.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                encoding="utf-8",
            )
            evaluation_paths[configuration_id] = path
        return sampling_path, evaluation_paths

    def test_lock_selects_tau_then_excludes_atom_and_proposed_routes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sampling, evaluations = self._artifacts(root)
            result = freeze_week2_champion(
                sampling_plan_path=sampling,
                evaluation_paths=evaluations,
                output=root / "champion-lock.json",
                expected_attempts_per_configuration=3,
                expected_attempts_per_sampling_seed=1,
                allow_nonpaper_attempts=True,
            )
        self.assertEqual(result["selected_disc_once_configuration"], "disc-once-tau-0p75")
        self.assertEqual(result["selected_disc_once_tau"], 0.75)
        self.assertEqual(
            result["selected_champion"]["configuration_id"],
            "joint-no-revision",
        )
        self.assertFalse(result["atom_baseline_is_champion_eligible"])
        self.assertFalse(result["stratified_geometry_is_champion_eligible"])
        expected = hashlib.sha256(
            b"joint-no-revision:continuation_checkpoint"
        ).hexdigest()
        self.assertEqual(
            result["selected_champion_continuation_checkpoint"]["sha256"],
            expected,
        )

    def test_mismatched_pair_set_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sampling, evaluations = self._artifacts(root)
            path = evaluations["atom-joint"]
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            rows[0]["pair_id"] = "unmatched"
            path.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "identical matched pairs"):
                freeze_week2_champion(
                    sampling_plan_path=sampling,
                    evaluation_paths=evaluations,
                    output=root / "champion-lock.json",
                    expected_attempts_per_configuration=3,
                    expected_attempts_per_sampling_seed=1,
                    allow_nonpaper_attempts=True,
                )


if __name__ == "__main__":
    unittest.main()
