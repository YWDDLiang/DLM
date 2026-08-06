from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from crystal_dlm.wqcodiff.crysllmgen.formal_execution import (
    AUTHORIZATION_RECORD,
    AUTHORIZATION_SHA256,
    CACHE_MANIFEST_SHA256,
    load_formal_execution_selection,
)


RUN_ID = "20260720_0401-crysllmgen-wq-final-v3"
SOURCE_SHA = "6eeb48310454199a37d622e9dade6ccbe0f5c280b14999a6ca5c26c7e11e5908"
PROTOCOL_SHA = "22efb63d3b7e37353206022d3ce9e43f0c19ed02dbd5fa75a71a053339707af0"
GATE_SHA = "534dbd55caf3599bad233bfd671bf304fe28891854101d2a201f1d1feaaa4bf5"
PATCH_SHA = "1" * 64


def _payload(*, flash_eligible: bool) -> dict[str, object]:
    return {
        "schema": "crysllmgen_formal_execution_selection_v1",
        "status": "selected_for_formal_training",
        "scientific_attempt": False,
        "run_id": RUN_ID,
        "source_bundle_sha256": SOURCE_SHA,
        "protocol_v4_sha256": PROTOCOL_SHA,
        "gate_a_lock_sha256": GATE_SHA,
        "execution_patch_sha256": PATCH_SHA,
        "authorization": {
            "record": AUTHORIZATION_RECORD,
            "sha256": AUTHORIZATION_SHA256,
        },
        "selected_attention_implementation": (
            "flash_attention_2" if flash_eligible else "sdpa"
        ),
        "selected_profile_id": (
            "cache4_mb8_acc4_gc_off_flash2"
            if flash_eligible
            else "cache4_mb8_acc4_gc_off_sdpa"
        ),
        "execution": {
            "world_size": 2,
            "per_device_microbatch": 8,
            "gradient_accumulation": 4,
            "global_effective_batch": 64,
            "data_mode": "pretokenized_memmap",
            "dataloader_num_workers_per_rank": 4,
            "pretokenized_cache_manifest_sha256": CACHE_MANIFEST_SHA256,
            "gradient_checkpointing": False,
            "gradient_checkpointing_use_reentrant": None,
        },
        "sdpa_reference": {
            "profile_id": "cache4_mb8_acc4_gc_off_sdpa",
            "seconds_per_update": 4.16302,
            "report_sha256": "d3d1948088439f8ea0da0d89cbee7551a734308aa51748cfbca88b9de6d57960",
            "summary_sha256": "c7505653b717d0e072f4d5394347aa9b6b8fc8e1280c376d8fccffe4212170f1",
        },
        "flash_candidate": {
            "profile_id": "cache4_mb8_acc4_gc_off_flash2",
            "profile_matrix_sha256": "2" * 64,
            "minimum_relative_speedup": 0.02,
            "maximum_seconds_per_update": 4.0797596,
            "eligible": flash_eligible,
        },
        "retry_or_replacement_used": False,
    }


class FormalExecutionSelectionTests(unittest.TestCase):
    def _load(self, payload: dict[str, object]):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selection.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            return load_formal_execution_selection(
                path,
                run_id=RUN_ID,
                base_source_bundle_sha256=SOURCE_SHA,
                protocol_v4_sha256=PROTOCOL_SHA,
                gate_a_lock_sha256=GATE_SHA,
                execution_patch_sha256=PATCH_SHA,
            )

    def test_flash_and_fallback_records_are_both_registered(self) -> None:
        flash = self._load(_payload(flash_eligible=True))
        self.assertEqual(flash.attention_implementation, "flash_attention_2")
        self.assertFalse(flash.gradient_checkpointing)
        fallback = self._load(_payload(flash_eligible=False))
        self.assertEqual(fallback.attention_implementation, "sdpa")

    def test_selection_cannot_disagree_with_eligibility(self) -> None:
        changed = copy.deepcopy(_payload(flash_eligible=False))
        changed["selected_attention_implementation"] = "flash_attention_2"
        changed["selected_profile_id"] = "cache4_mb8_acc4_gc_off_flash2"
        with self.assertRaisesRegex(ValueError, "ineligible"):
            self._load(changed)

    def test_execution_drift_is_rejected(self) -> None:
        changed = copy.deepcopy(_payload(flash_eligible=True))
        changed["execution"]["gradient_checkpointing"] = True
        with self.assertRaisesRegex(ValueError, "settings"):
            self._load(changed)


if __name__ == "__main__":
    unittest.main()
