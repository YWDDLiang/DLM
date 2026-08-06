from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.audit_h1a2_dlm_fixed_panel import audit_panel


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FixedPanelAuditTest(unittest.TestCase):
    def test_two_rank_panel_is_frozen_without_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            ledger = root / "val.jsonl"
            with ledger.open("w", encoding="utf-8") as handle:
                for ordinal in range(9):
                    digest = hashlib.sha256(str(ordinal).encode()).hexdigest()
                    handle.write(
                        json.dumps(
                            {
                                "ordinal": ordinal,
                                "model_visible_sha256": digest,
                                "plan_condition_sha256": digest,
                                "training_pair_sha256": digest,
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "dataset_version": "h1a2_r5c_plangraph_sidecar_v2",
                        "prompt_answer_byte_identity": True,
                        "plangraph_visibility": "collator_sidecar_only",
                        "output_file_sha256": {
                            "row_ledger/val.jsonl": sha256_file(ledger)
                        },
                        "fixed_validation_panel": {"row_count": 9},
                    }
                ),
                encoding="utf-8",
            )

            report, panel = audit_panel(
                dataset_manifest_path=manifest,
                validation_ledger_path=ledger,
                world_size=2,
                rows_per_rank=2,
            )

        self.assertTrue(report["gate_passed"])
        self.assertEqual(report["global_panel_row_count"], 4)
        self.assertEqual(
            [row["validation_ordinal"] for row in panel],
            [0, 2, 1, 3],
        )
        self.assertEqual(
            sorted(row["validation_ordinal"] for row in panel),
            [0, 1, 2, 3],
        )

    def test_ledger_sha_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            ledger = root / "val.jsonl"
            digest = hashlib.sha256(b"row").hexdigest()
            ledger.write_text(
                json.dumps(
                    {
                        "ordinal": 0,
                        "model_visible_sha256": digest,
                        "plan_condition_sha256": digest,
                        "training_pair_sha256": digest,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "dataset_version": "h1a2_r5c_plangraph_sidecar_v2",
                        "prompt_answer_byte_identity": True,
                        "plangraph_visibility": "collator_sidecar_only",
                        "output_file_sha256": {
                            "row_ledger/val.jsonl": "0" * 64
                        },
                        "fixed_validation_panel": {"row_count": 1},
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "ledger SHA mismatch"):
                audit_panel(
                    dataset_manifest_path=manifest,
                    validation_ledger_path=ledger,
                    world_size=1,
                    rows_per_rank=1,
                )


if __name__ == "__main__":
    unittest.main()
