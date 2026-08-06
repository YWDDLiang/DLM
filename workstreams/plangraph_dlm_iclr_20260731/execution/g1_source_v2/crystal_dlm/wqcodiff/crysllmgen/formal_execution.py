"""Immutable execution selection for the formal three-epoch LoRA run."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping


SHA256 = re.compile(r"^[0-9a-f]{64}$")
AUTHORIZATION_RECORD = (
    "runs/remote_audit/"
    "20260723_user_authorization_flash_profile_and_formal_progression_v1.json"
)
AUTHORIZATION_SHA256 = (
    "912479c51ed022f13d28754ec520da7417747ce19d655738f3f8140eba89cbe0"
)
FLASH_PROFILE_ID = "cache4_mb8_acc4_gc_off_flash2"
SDPA_PROFILE_ID = "cache4_mb8_acc4_gc_off_sdpa"
CACHE_MANIFEST_SHA256 = (
    "afd954868f1f9c431068e92b444061079d741d6b23e8cdddccb1e9bedd9ddb23"
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclasses.dataclass(frozen=True, slots=True)
class FormalExecutionSelection:
    path: Path
    sha256: str
    data: Mapping[str, Any]
    attention_implementation: str
    selected_profile_id: str
    per_device_microbatch: int
    gradient_accumulation: int
    data_mode: str
    dataloader_num_workers: int
    gradient_checkpointing: bool
    gradient_checkpointing_use_reentrant: bool | None
    pretokenized_cache_manifest_sha256: str


def load_formal_execution_selection(
    path: str | Path,
    *,
    run_id: str,
    base_source_bundle_sha256: str,
    protocol_v4_sha256: str,
    gate_a_lock_sha256: str,
    execution_patch_sha256: str,
) -> FormalExecutionSelection:
    """Load the one-way FlashAttention-or-SDPA execution choice."""

    location = Path(path).resolve()
    payload = json.loads(location.read_text(encoding="utf-8"))
    if (
        payload.get("schema") != "crysllmgen_formal_execution_selection_v1"
        or payload.get("status") != "selected_for_formal_training"
        or payload.get("scientific_attempt") is not False
        or payload.get("run_id") != run_id
        or payload.get("source_bundle_sha256") != base_source_bundle_sha256
        or payload.get("protocol_v4_sha256") != protocol_v4_sha256
        or payload.get("gate_a_lock_sha256") != gate_a_lock_sha256
        or payload.get("execution_patch_sha256") != execution_patch_sha256
        or payload.get("retry_or_replacement_used") is not False
    ):
        raise ValueError("formal execution selection identity changed")
    if payload.get("authorization") != {
        "record": AUTHORIZATION_RECORD,
        "sha256": AUTHORIZATION_SHA256,
    }:
        raise ValueError("formal execution authorization identity changed")
    selected = str(payload.get("selected_attention_implementation", ""))
    selected_profile = str(payload.get("selected_profile_id", ""))
    if (selected, selected_profile) not in {
        ("sdpa", SDPA_PROFILE_ID),
        ("flash_attention_2", FLASH_PROFILE_ID),
    }:
        raise ValueError("unregistered formal attention selection")
    execution = payload.get("execution") or {}
    expected_execution = {
        "world_size": 2,
        "per_device_microbatch": 8,
        "gradient_accumulation": 4,
        "global_effective_batch": 64,
        "data_mode": "pretokenized_memmap",
        "dataloader_num_workers_per_rank": 4,
        "pretokenized_cache_manifest_sha256": CACHE_MANIFEST_SHA256,
        "gradient_checkpointing": False,
        "gradient_checkpointing_use_reentrant": None,
    }
    if execution != expected_execution:
        raise ValueError("formal execution settings changed")
    reference = payload.get("sdpa_reference") or {}
    if (
        reference.get("profile_id") != SDPA_PROFILE_ID
        or float(reference.get("seconds_per_update", -1.0)) != 4.16302
        or reference.get("report_sha256")
        != "d3d1948088439f8ea0da0d89cbee7551a734308aa51748cfbca88b9de6d57960"
        or reference.get("summary_sha256")
        != "c7505653b717d0e072f4d5394347aa9b6b8fc8e1280c376d8fccffe4212170f1"
    ):
        raise ValueError("formal SDPA reference changed")
    flash = payload.get("flash_candidate") or {}
    if (
        flash.get("profile_id") != FLASH_PROFILE_ID
        or flash.get("profile_matrix_sha256") is None
        or SHA256.fullmatch(str(flash["profile_matrix_sha256"])) is None
        or float(flash.get("minimum_relative_speedup", -1.0)) != 0.02
        or float(flash.get("maximum_seconds_per_update", -1.0)) != 4.0797596
        or not isinstance(flash.get("eligible"), bool)
    ):
        raise ValueError("formal FlashAttention candidate evidence changed")
    if selected == "flash_attention_2" and flash.get("eligible") is not True:
        raise ValueError("ineligible FlashAttention was selected")
    if selected == "sdpa" and flash.get("eligible") is True:
        raise ValueError("eligible FlashAttention was not selected")
    return FormalExecutionSelection(
        path=location,
        sha256=sha256_file(location),
        data=payload,
        attention_implementation=selected,
        selected_profile_id=selected_profile,
        per_device_microbatch=8,
        gradient_accumulation=4,
        data_mode="pretokenized_memmap",
        dataloader_num_workers=4,
        gradient_checkpointing=False,
        gradient_checkpointing_use_reentrant=None,
        pretokenized_cache_manifest_sha256=CACHE_MANIFEST_SHA256,
    )
