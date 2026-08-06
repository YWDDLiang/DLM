from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from crystal_dlm.wqcodiff.contracts import (
    ArtifactLedger,
    AttemptLedger,
    AttemptRecord,
    AttemptStatus,
    SeedDeriver,
)
from crystal_dlm.wqcodiff.protocol import ACTIVE_PROTOCOL_NAME
from scripts.a800.reconcile_day7_terminal_timeout import (
    _hash_fixed_records,
    _load_timeout_evidence,
    reconcile_terminal_lane_timeout,
)


def _dataset_row(material_id: str) -> dict:
    return {
        "material_id": material_id,
        "decompositions": {
            "symprec_1e-02": {
                "state": {
                    "space_group": 1,
                    "lattice_system": "triclinic",
                    "lattice_chart": [1.0] * 6,
                    "orbits": [
                        {
                            "orbit_id": "o0",
                            "wyckoff_type": 0,
                            "species": 1,
                            "multiplicity": 1,
                            "primitive_multiplicity": 1,
                            "chart_dimension": 3,
                            "free_coordinate": [0.0, 0.0, 0.0],
                        }
                    ],
                    "space_group_committed": True,
                }
            }
        },
    }


class Day7TimeoutReconcileTests(unittest.TestCase):
    def test_missing_attempts_become_terminal_timeouts_without_retry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "val.jsonl"
            protocol = root / "protocol.yaml"
            protocol.write_text("protocol: unit\n", encoding="utf-8")
            with dataset.open("x", encoding="utf-8") as handle:
                for material_id in ("mp-a", "mp-b", "mp-c"):
                    handle.write(json.dumps(_dataset_row(material_id)) + "\n")
            selected, subset_hash = _hash_fixed_records((dataset,), 3)
            output = root / "cell.jsonl"
            ledger_path = root / "cell.attempts.jsonl"
            experiment_id = "threshold-unit"
            pairing_id = "threshold-pairs"
            deriver = SeedDeriver(ACTIVE_PROTOCOL_NAME, experiment_id)
            artifacts = ArtifactLedger(output)
            attempts = AttemptLedger(ledger_path)

            first_id = deriver.attempt_id(
                training_seed=11,
                sampling_seed=101,
                ordinal=0,
                method="M-WQ-STRAT-GEO",
            )
            first_seed = deriver.derive(
                training_seed=11,
                sampling_seed=101,
                attempt_id=first_id,
                stage="recovery",
            )
            attempts.append(
                AttemptRecord(
                    attempt_id=first_id,
                    method="M-WQ-STRAT-GEO",
                    training_seed=11,
                    sampling_seed=101,
                    stage="recovery",
                    status=AttemptStatus.SUBMITTED,
                    seed=first_seed,
                )
            )
            digest = artifacts.append(
                {
                    "schema": "wqcodiff_recovery_attempt_v1",
                    "attempt_id": first_id,
                    "material_id": selected[0]["material_id"],
                    "method": "M-WQ-STRAT-GEO",
                    "training_seed": 11,
                    "corruption_seed": 101,
                    "ordinal": 0,
                    "corruption_level": 0.9,
                    "operator": "joint",
                    "geometry_condition": "noisy",
                    "schedule": "geometry-adaptive",
                    "control": "none",
                    "revision_threshold": 0.9,
                    "subset_hash": subset_hash,
                    "checkpoint_sha256": "c" * 64,
                    "source_bundle_sha256": "s" * 64,
                    "runtime_source_bundle_sha256": "d" * 64,
                    "status": "succeeded",
                    "applicable": True,
                    "initial_revisable_field_count": 3,
                    "exact_full_protostructure_recovery": True,
                    "mechanism": {
                        "revision_selected_actions": 0,
                        "wrong_to_right": 1,
                        "right_to_wrong": 0,
                        "net_correction": 1,
                    },
                }
            )
            attempts.append(
                AttemptRecord(
                    attempt_id=first_id,
                    method="M-WQ-STRAT-GEO",
                    training_seed=11,
                    sampling_seed=101,
                    stage="recovery",
                    status=AttemptStatus.SUCCEEDED,
                    artifact_hash=digest,
                    seed=first_seed,
                )
            )

            second_id = deriver.attempt_id(
                training_seed=11,
                sampling_seed=101,
                ordinal=1,
                method="M-WQ-STRAT-GEO",
            )
            attempts.append(
                AttemptRecord(
                    attempt_id=second_id,
                    method="M-WQ-STRAT-GEO",
                    training_seed=11,
                    sampling_seed=101,
                    stage="recovery",
                    status=AttemptStatus.SUBMITTED,
                    seed=deriver.derive(
                        training_seed=11,
                        sampling_seed=101,
                        attempt_id=second_id,
                        stage="recovery",
                    ),
                )
            )

            argv = [
                "python",
                "-m",
                "crystal_dlm.wqcodiff",
                "--protocol",
                str(protocol),
                "recovery",
                "--dataset",
                str(dataset),
                "--experiment-id",
                experiment_id,
                "--pairing-id",
                pairing_id,
                "--runtime-source-bundle-sha256",
                "d" * 64,
                "--variant",
                "M-WQ-STRAT-GEO",
                "--training-seed",
                "11",
                "--corruption-seed",
                "101",
                "--structures",
                "3",
                "--corruption-level",
                "0.9",
                "--operator",
                "joint",
                "--geometry-condition",
                "noisy",
                "--schedule",
                "geometry-adaptive",
                "--control",
                "none",
                "--revision-threshold",
                "0.9",
                "--calls",
                "16",
                "--inference-batch-size",
                "64",
                "--output",
                str(output),
                "--ledger",
                str(ledger_path),
            ]
            job = {
                "cell_id": "timed-final-cell",
                "phase_ordinal": 44,
                "argv": argv,
                "output": str(output),
                "ledger": str(ledger_path),
            }
            manifest = root / "lane.manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema": "wqcodiff_day7_lane_plan_v1",
                        "phase": "threshold-calibration",
                        "lane_index": 0,
                        "cells": 1,
                        "attempts": 3,
                        "backbone_calls": 48,
                        "jobs": [job],
                    }
                ),
                encoding="utf-8",
            )
            events = root / "lane.events.jsonl"
            events.write_text(
                json.dumps(
                    {
                        "schema": "wqcodiff_day7_lane_event_v1",
                        "event": "started",
                        "cell_id": "timed-final-cell",
                        "phase_ordinal": 44,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            evidence = root / "timeout-evidence.json"
            evidence.write_text(
                json.dumps(
                    {
                        "schema": "wqcodiff_slurm_terminal_evidence_v1",
                        "job_id": "123",
                        "state": "TIMEOUT",
                        "reason": "three-hour allocation exhausted",
                    }
                ),
                encoding="utf-8",
            )
            complete = root / "lane.complete.json"
            result = reconcile_terminal_lane_timeout(
                manifest_path=manifest,
                events_path=events,
                complete_path=complete,
                timeout_evidence_path=evidence,
                slurm_job_id="123",
            )
            self.assertTrue(result["ok"])
            self.assertTrue(result["no_retry"])
            self.assertEqual(result["timeout_attempts"], 2)
            rows = ArtifactLedger(output).records()
            self.assertEqual(len(rows), 3)
            self.assertEqual(
                [row["status"] for row in rows], ["succeeded", "timeout", "timeout"]
            )
            self.assertTrue(all(row["initial_revisable_field_count"] == 3 for row in rows))
            summary = json.loads(output.with_suffix(".summary.json").read_text())
            self.assertEqual(summary["failed"], 2)
            self.assertTrue(summary["timeout_reconciliation"]["no_model_invocation"])
            terminal_event = json.loads(events.read_text().splitlines()[-1])
            self.assertEqual(terminal_event["returncode"], 124)
            self.assertEqual(terminal_event["terminal_status"], "timeout")

    def test_non_timeout_evidence_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = root / "evidence.json"
            evidence.write_text(
                json.dumps(
                    {
                        "schema": "wqcodiff_slurm_terminal_evidence_v1",
                        "job_id": "7",
                        "state": "FAILED",
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "does not record a TIMEOUT"):
                _load_timeout_evidence(evidence, "7")


if __name__ == "__main__":
    unittest.main()
