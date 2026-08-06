from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import crystal_dlm.wqcodiff.contracts as contracts
from crystal_dlm.wqcodiff.contracts import (
    ArtifactLedger,
    AttemptLedger,
    AttemptRecord,
    AttemptStatus,
    SeedDeriver,
)


class SeedContractTests(unittest.TestCase):
    def test_attempt_and_stage_seed_do_not_depend_on_world_size(self) -> None:
        deriver = SeedDeriver("sun_iclr_stratified_wyckoff_v3", "unit")
        attempt = deriver.attempt_id(
            training_seed=11,
            sampling_seed=101,
            ordinal=7,
            method="M-WQ-STRAT-GEO",
        )
        first = deriver.derive(
            training_seed=11,
            sampling_seed=101,
            attempt_id=attempt,
            stage="sample",
        )
        second = deriver.derive(
            training_seed=11,
            sampling_seed=101,
            attempt_id=attempt,
            stage="sample",
        )
        self.assertEqual(first, second)
        self.assertNotEqual(
            first,
            deriver.derive(
                training_seed=11,
                sampling_seed=101,
                attempt_id=attempt,
                stage="sun",
            ),
        )

    def test_attempt_id_changes_with_ordinal(self) -> None:
        deriver = SeedDeriver("p", "e")
        ids = {
            deriver.attempt_id(training_seed=11, sampling_seed=101, ordinal=i, method="m")
            for i in range(1000)
        }
        self.assertEqual(len(ids), 1000)

    def test_pair_seed_is_method_independent_but_ordinal_specific(self) -> None:
        deriver = SeedDeriver("p", "paired-final")
        first = deriver.paired_derive(
            training_seed=11,
            sampling_seed=101,
            ordinal=5,
            stage="initial_noise",
        )
        second = deriver.paired_derive(
            training_seed=11,
            sampling_seed=101,
            ordinal=5,
            stage="initial_noise",
        )
        self.assertEqual(first, second)
        self.assertNotEqual(
            first,
            deriver.paired_derive(
                training_seed=11,
                sampling_seed=101,
                ordinal=6,
                stage="initial_noise",
            ),
        )
        self.assertTrue(
            deriver.pair_id(training_seed=11, sampling_seed=101, ordinal=5).startswith(
                "p-"
            )
        )


class LedgerTests(unittest.TestCase):
    def test_terminal_record_is_append_only_and_unique(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "attempts.jsonl"
            ledger = AttemptLedger(path)
            deriver = SeedDeriver("p", "e")
            attempt = deriver.attempt_id(
                training_seed=11, sampling_seed=101, ordinal=0, method="m"
            )
            seed = deriver.derive(
                training_seed=11,
                sampling_seed=101,
                attempt_id=attempt,
                stage="sun",
            )
            record = AttemptRecord(
                attempt_id=attempt,
                method="m",
                training_seed=11,
                sampling_seed=101,
                stage="sun",
                status=AttemptStatus.FAILED,
                reason="graph",
                seed=seed,
                calls={"joint": 64},
            )
            ledger.append(record)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                ledger.append(record)
            with self.assertRaisesRegex(ValueError, "terminal"):
                ledger.append(
                    AttemptRecord(
                        attempt_id=attempt,
                        method="m",
                        training_seed=11,
                        sampling_seed=101,
                        stage="sun",
                        status=AttemptStatus.SUCCEEDED,
                        seed=seed,
                    )
                )
            audit = ledger.audit(
                seed_deriver=deriver,
                expected_attempt_ids=[attempt],
            )
            self.assertTrue(audit.ok)
            self.assertEqual(audit.attempts, 1)

    def test_failure_requires_reason(self) -> None:
        with self.assertRaisesRegex(ValueError, "reason"):
            AttemptRecord(
                attempt_id="a-x",
                method="m",
                training_seed=11,
                sampling_seed=101,
                stage="sun",
                status=AttemptStatus.FAILED,
            )

    def test_artifact_ledger_refuses_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ledger = ArtifactLedger(Path(directory) / "artifacts.jsonl")
            digest = ledger.append({"attempt_id": "a-1", "value": 1})
            self.assertEqual(len(digest), 64)
            with self.assertRaisesRegex(ValueError, "duplicate immutable"):
                ledger.append({"attempt_id": "a-1", "value": 2})

    def test_attempt_cache_observes_another_cooperative_writer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "attempts.jsonl"
            first = AttemptLedger(path)
            second = AttemptLedger(path)
            base = {
                "method": "m",
                "training_seed": 11,
                "sampling_seed": 101,
                "stage": "recovery",
            }
            first.append(
                AttemptRecord(
                    attempt_id="a-1",
                    status=AttemptStatus.SUBMITTED,
                    **base,
                )
            )
            second.append(
                AttemptRecord(
                    attempt_id="a-2",
                    status=AttemptStatus.SUBMITTED,
                    **base,
                )
            )
            with self.assertRaisesRegex(ValueError, "duplicate"):
                first.append(
                    AttemptRecord(
                        attempt_id="a-2",
                        status=AttemptStatus.SUBMITTED,
                        **base,
                    )
                )
            self.assertEqual(len(first.records()), 2)

    def test_artifact_cache_observes_another_cooperative_writer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifacts.jsonl"
            first = ArtifactLedger(path)
            second = ArtifactLedger(path)
            first.append({"attempt_id": "a-1", "value": 1})
            second.append({"attempt_id": "a-2", "value": 2})
            with self.assertRaisesRegex(ValueError, "duplicate immutable"):
                first.append({"attempt_id": "a-2", "value": 3})
            self.assertEqual(len(first.records()), 2)

    def test_append_cache_parses_each_preexisting_artifact_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifacts.jsonl"
            first = ArtifactLedger(path)
            for ordinal in range(128):
                first.append({"attempt_id": f"a-{ordinal}", "value": ordinal})
            second = ArtifactLedger(path)
            with mock.patch.object(
                contracts.json, "loads", wraps=json.loads
            ) as loads:
                second.append({"attempt_id": "a-new-0", "value": 0})
                first_count = loads.call_count
                for ordinal in range(1, 33):
                    second.append(
                        {"attempt_id": f"a-new-{ordinal}", "value": ordinal}
                    )
                self.assertEqual(first_count, 128)
                self.assertEqual(loads.call_count, 128)

    def test_partial_jsonl_tail_is_rejected_before_append(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifacts.jsonl"
            path.write_text('{"attempt_id":"broken"}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "partial trailing"):
                ArtifactLedger(path).append({"attempt_id": "a-1"})

    def test_incremental_append_preserves_canonical_bytes_and_digest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            attempt_path = root / "attempts.jsonl"
            artifact_path = root / "artifacts.jsonl"
            record = AttemptRecord(
                attempt_id="a-byte-parity",
                method="m",
                training_seed=11,
                sampling_seed=101,
                stage="recovery",
                status=AttemptStatus.SUBMITTED,
                calls={"projection": 16, "joint": 16},
                metadata={"z": 2, "a": 1},
            )
            AttemptLedger(attempt_path).append(record)
            expected_attempt = (
                contracts._canonical_json(record.to_dict()) + "\n"
            ).encode("utf-8")
            self.assertEqual(attempt_path.read_bytes(), expected_attempt)

            artifact = {"value": {"z": 2, "a": 1}, "attempt_id": "a-byte-parity"}
            digest = ArtifactLedger(artifact_path).append(artifact)
            expected_artifact = (
                contracts._canonical_json(artifact) + "\n"
            ).encode("utf-8")
            self.assertEqual(artifact_path.read_bytes(), expected_artifact)
            self.assertEqual(digest, hashlib.sha256(expected_artifact).hexdigest())


if __name__ == "__main__":
    unittest.main()
