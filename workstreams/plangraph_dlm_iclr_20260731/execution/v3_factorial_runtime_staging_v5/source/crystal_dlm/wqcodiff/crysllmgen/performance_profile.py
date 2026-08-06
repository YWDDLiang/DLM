"""Strict execution-only profile contracts for CrysLLMGen LoRA training."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping


SHA256 = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclasses.dataclass(frozen=True, slots=True)
class LoRAPerformanceVariant:
    profile_id: str
    world_size: int
    global_effective_batch: int
    optimizer_updates: int
    per_device_microbatch: int
    gradient_accumulation: int
    data_mode: str
    dataloader_num_workers: int
    gradient_checkpointing: bool
    gradient_checkpointing_use_reentrant: bool | None
    attention_implementation: str


@dataclasses.dataclass(frozen=True, slots=True)
class LoRADDPProfileMatrix:
    path: Path
    sha256: str
    data: Mapping[str, Any]
    authorization_record: str
    authorization_sha256: str
    variants: Mapping[str, LoRAPerformanceVariant]

    def select(self, profile_id: str) -> LoRAPerformanceVariant:
        try:
            return self.variants[str(profile_id)]
        except KeyError as exc:
            raise ValueError(f"unregistered DDP performance profile: {profile_id}") from exc


_EXPECTED_VARIANTS = {
    "lazy_mb4_acc8_gc_nonreentrant_sdpa": {
        "per_device_microbatch": 4,
        "gradient_accumulation": 8,
        "data_mode": "lazy_jsonl",
        "dataloader_num_workers": 0,
        "gradient_checkpointing": True,
        "gradient_checkpointing_use_reentrant": False,
    },
    "cache4_mb4_acc8_gc_nonreentrant_sdpa": {
        "per_device_microbatch": 4,
        "gradient_accumulation": 8,
        "data_mode": "pretokenized_memmap",
        "dataloader_num_workers": 4,
        "gradient_checkpointing": True,
        "gradient_checkpointing_use_reentrant": False,
    },
    "cache4_mb4_acc8_gc_reentrant_sdpa": {
        "per_device_microbatch": 4,
        "gradient_accumulation": 8,
        "data_mode": "pretokenized_memmap",
        "dataloader_num_workers": 4,
        "gradient_checkpointing": True,
        "gradient_checkpointing_use_reentrant": True,
    },
    "cache4_mb4_acc8_gc_off_sdpa": {
        "per_device_microbatch": 4,
        "gradient_accumulation": 8,
        "data_mode": "pretokenized_memmap",
        "dataloader_num_workers": 4,
        "gradient_checkpointing": False,
        "gradient_checkpointing_use_reentrant": None,
    },
    "cache4_mb8_acc4_gc_reentrant_sdpa": {
        "per_device_microbatch": 8,
        "gradient_accumulation": 4,
        "data_mode": "pretokenized_memmap",
        "dataloader_num_workers": 4,
        "gradient_checkpointing": True,
        "gradient_checkpointing_use_reentrant": True,
    },
    "cache4_mb8_acc4_gc_off_sdpa": {
        "per_device_microbatch": 8,
        "gradient_accumulation": 4,
        "data_mode": "pretokenized_memmap",
        "dataloader_num_workers": 4,
        "gradient_checkpointing": False,
        "gradient_checkpointing_use_reentrant": None,
    },
}

_FLASH_SCHEMA = "crysllmgen_lora_ddp_flash_followup_v1"
_FLASH_AUTHORIZATION_RECORD = (
    "runs/remote_audit/"
    "20260723_user_authorization_flash_profile_and_formal_progression_v1.json"
)
_FLASH_AUTHORIZATION_SHA256 = (
    "912479c51ed022f13d28754ec520da7417747ce19d655738f3f8140eba89cbe0"
)
_FLASH_PROFILE_ID = "cache4_mb8_acc4_gc_off_flash2"
_FLASH_EXPECTED = {
    "per_device_microbatch": 8,
    "gradient_accumulation": 4,
    "data_mode": "pretokenized_memmap",
    "dataloader_num_workers": 4,
    "gradient_checkpointing": False,
    "gradient_checkpointing_use_reentrant": None,
}


def _load_flash_followup(
    *,
    location: Path,
    payload: Mapping[str, Any],
    base_source_bundle_sha256: str,
    protocol_v4_sha256: str,
) -> LoRADDPProfileMatrix:
    if (
        payload.get("status") != "active_user_authorized_execution_profile"
        or payload.get("scientific_attempt") is not False
        or payload.get("base_source_bundle_sha256")
        != base_source_bundle_sha256
        or payload.get("protocol_v4_sha256") != protocol_v4_sha256
    ):
        raise ValueError("FlashAttention profile identity changed")
    authorization = payload.get("authorization") or {}
    if authorization != {
        "record": _FLASH_AUTHORIZATION_RECORD,
        "sha256": _FLASH_AUTHORIZATION_SHA256,
    }:
        raise ValueError("FlashAttention profile authorization identity changed")
    invariants = payload.get("frozen_invariants") or {}
    if invariants != {
        "training_stage": "mixed_edit",
        "representation": "wyckoff",
        "training_seed": 11,
        "world_size": 2,
        "global_effective_batch": 64,
        "optimizer_updates": 50,
        "attention_implementation": "flash_attention_2",
        "learning_rate_scheduler_epochs_and_example_order": "inherited_unchanged",
    }:
        raise ValueError("FlashAttention profile frozen invariants changed")
    entries = payload.get("variants")
    if not isinstance(entries, list) or len(entries) != 1:
        raise ValueError("FlashAttention profile denominator changed")
    entry = entries[0]
    if not isinstance(entry, Mapping) or entry.get("profile_id") != _FLASH_PROFILE_ID:
        raise ValueError("FlashAttention profile identity changed")
    if {key: entry.get(key) for key in _FLASH_EXPECTED} != _FLASH_EXPECTED:
        raise ValueError("FlashAttention execution settings changed")
    if 2 * int(entry["per_device_microbatch"]) * int(
        entry["gradient_accumulation"]
    ) != 64:
        raise ValueError("FlashAttention profile changed the frozen global batch")
    selection = payload.get("selection") or {}
    reference = selection.get("sdpa_reference") or {}
    if (
        selection.get("primary")
        != "lowest_train_runtime_seconds_per_optimizer_update"
        or float(selection.get("flash_minimum_relative_speedup", -1.0)) != 0.02
        or float(selection.get("flash_maximum_seconds_per_update", -1.0))
        != 4.0797596
        or selection.get("fallback") != "cache4_mb8_acc4_gc_off_sdpa"
        or reference.get("profile_id") != "cache4_mb8_acc4_gc_off_sdpa"
        or float(reference.get("seconds_per_update", -1.0)) != 4.16302
        or reference.get("report_sha256")
        != "d3d1948088439f8ea0da0d89cbee7551a734308aa51748cfbca88b9de6d57960"
        or reference.get("summary_sha256")
        != "c7505653b717d0e072f4d5394347aa9b6b8fc8e1280c376d8fccffe4212170f1"
        or selection.get("failed_or_ineligible_flash_policy")
        != "retain_failure_and_select_frozen_sdpa_without_retry"
        or selection.get("formal_training_submission")
        != "allowed_only_after_exclusive_execution_selection_record"
    ):
        raise ValueError("FlashAttention selection contract changed")
    variant = LoRAPerformanceVariant(
        profile_id=_FLASH_PROFILE_ID,
        world_size=2,
        global_effective_batch=64,
        optimizer_updates=50,
        per_device_microbatch=8,
        gradient_accumulation=4,
        data_mode="pretokenized_memmap",
        dataloader_num_workers=4,
        gradient_checkpointing=False,
        gradient_checkpointing_use_reentrant=None,
        attention_implementation="flash_attention_2",
    )
    return LoRADDPProfileMatrix(
        path=location,
        sha256=sha256_file(location),
        data=payload,
        authorization_record=_FLASH_AUTHORIZATION_RECORD,
        authorization_sha256=_FLASH_AUTHORIZATION_SHA256,
        variants={_FLASH_PROFILE_ID: variant},
    )


def load_lora_ddp_profile_matrix(
    path: str | Path,
    *,
    base_source_bundle_sha256: str,
    protocol_v4_sha256: str,
) -> LoRADDPProfileMatrix:
    """Load the exact user-authorized, non-scientific DDP profile matrix."""

    location = Path(path).resolve()
    payload = json.loads(location.read_text(encoding="utf-8"))
    if payload.get("schema") == _FLASH_SCHEMA:
        return _load_flash_followup(
            location=location,
            payload=payload,
            base_source_bundle_sha256=base_source_bundle_sha256,
            protocol_v4_sha256=protocol_v4_sha256,
        )
    if (
        payload.get("schema") != "crysllmgen_lora_ddp_profile_matrix_v1"
        or payload.get("status") != "active_user_authorized_execution_profile"
        or payload.get("scientific_attempt") is not False
    ):
        raise ValueError("DDP profile matrix is not active and non-scientific")
    if (
        payload.get("base_source_bundle_sha256") != base_source_bundle_sha256
        or payload.get("protocol_v4_sha256") != protocol_v4_sha256
    ):
        raise ValueError("DDP profile matrix source/protocol identity mismatch")
    authorization = payload.get("authorization") or {}
    authorization_record = str(authorization.get("record", ""))
    authorization_sha = str(authorization.get("sha256", ""))
    if (
        authorization_record
        != "runs/remote_audit/20260722_llamafactory_aligned_ddp_profile_authorization_v1.json"
        or SHA256.fullmatch(authorization_sha) is None
    ):
        raise ValueError("DDP profile matrix authorization identity changed")
    invariants = payload.get("frozen_invariants") or {}
    if invariants != {
        "training_stage": "mixed_edit",
        "representation": "wyckoff",
        "training_seed": 11,
        "world_size": 2,
        "global_effective_batch": 64,
        "optimizer_updates": 50,
        "attention_implementation": "sdpa",
        "learning_rate_scheduler_epochs_and_example_order": "inherited_unchanged",
    }:
        raise ValueError("DDP profile frozen invariants changed")
    entries = payload.get("variants")
    if not isinstance(entries, list) or len(entries) != len(_EXPECTED_VARIANTS):
        raise ValueError("DDP profile variant denominator changed")
    variants: dict[str, LoRAPerformanceVariant] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("malformed DDP profile variant")
        profile_id = str(entry.get("profile_id", ""))
        expected = _EXPECTED_VARIANTS.get(profile_id)
        if expected is None or profile_id in variants:
            raise ValueError(f"unexpected or duplicate DDP profile: {profile_id}")
        observed = {key: entry.get(key) for key in expected}
        if observed != expected:
            raise ValueError(f"DDP profile execution settings changed: {profile_id}")
        microbatch = int(entry["per_device_microbatch"])
        accumulation = int(entry["gradient_accumulation"])
        if 2 * microbatch * accumulation != 64:
            raise ValueError("DDP profile changed the frozen global batch")
        variants[profile_id] = LoRAPerformanceVariant(
            profile_id=profile_id,
            world_size=2,
            global_effective_batch=64,
            optimizer_updates=50,
            per_device_microbatch=microbatch,
            gradient_accumulation=accumulation,
            data_mode=str(entry["data_mode"]),
            dataloader_num_workers=int(entry["dataloader_num_workers"]),
            gradient_checkpointing=bool(entry["gradient_checkpointing"]),
            gradient_checkpointing_use_reentrant=entry[
                "gradient_checkpointing_use_reentrant"
            ],
            attention_implementation="sdpa",
        )
    if set(variants) != set(_EXPECTED_VARIANTS):
        raise ValueError("DDP profile variant identities changed")
    selection = payload.get("selection") or {}
    references = selection.get("references_seconds_per_update") or {}
    if (
        selection.get("primary")
        != "lowest_train_runtime_seconds_per_optimizer_update"
        or selection.get("formal_training_submission")
        != "forbidden_until_independent_profile_audit"
        or float(references.get("historical_one_a800_mb4_acc16", -1.0)) != 12.69
        or float(references.get("audited_two_a800_mb4_acc8", -1.0))
        != 14.094601
    ):
        raise ValueError("DDP profile selection contract changed")
    return LoRADDPProfileMatrix(
        path=location,
        sha256=sha256_file(location),
        data=payload,
        authorization_record=authorization_record,
        authorization_sha256=authorization_sha,
        variants=variants,
    )
