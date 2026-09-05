"""Compact checkpoint I/O for the inference-time PMTR repair head."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from crystal_dlm.manifold_repair_head import (
    ManifoldRepairConfig,
    ManifoldRepairHead,
)
from crystal_dlm.pmtr_runtime import PMTRLogitTransform, PMTRRuntimeConfig


PMTR_CHECKPOINT_SCHEMA = "pmtr_repair_head_v1"


@dataclass(frozen=True)
class LoadedPMTRCheckpoint:
    """A frozen inference transform and its compact provenance metadata."""

    transform: PMTRLogitTransform
    metadata: Mapping[str, Any]


def save_pmtr_checkpoint(
    path: str | Path,
    *,
    repair_head: ManifoldRepairHead,
    runtime_config: PMTRRuntimeConfig = PMTRRuntimeConfig(),
) -> Path:
    """Write config and CPU state tensors to one portable Torch file."""

    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": PMTR_CHECKPOINT_SCHEMA,
        "head_config": asdict(repair_head.config),
        "runtime_config": asdict(runtime_config),
        "repair_head_state": {
            key: value.detach().cpu()
            for key, value in repair_head.state_dict().items()
        },
    }
    torch.save(payload, destination)
    return destination


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"PMTR checkpoint {name} must be a mapping")
    return value


def load_pmtr_checkpoint(
    path: str | Path,
    *,
    tokenizer: Any,
    device: torch.device | str,
    dtype: torch.dtype,
    expected_hidden_size: int,
) -> LoadedPMTRCheckpoint:
    """Load a strict, frozen repair head and compile its logit transform."""

    checkpoint_path = Path(path).expanduser().resolve(strict=True)
    if not checkpoint_path.is_file():
        raise ValueError("PMTR checkpoint must be a file")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    payload = _mapping(payload, name="payload")
    if payload.get("schema") != PMTR_CHECKPOINT_SCHEMA:
        raise ValueError(
            f"PMTR checkpoint schema must equal {PMTR_CHECKPOINT_SCHEMA!r}"
        )
    head_config = ManifoldRepairConfig(
        **dict(_mapping(payload.get("head_config"), name="head_config"))
    )
    if int(head_config.hidden_size) != int(expected_hidden_size):
        raise ValueError(
            "PMTR repair-head hidden size does not match the DLM output hidden size"
        )
    runtime_config = PMTRRuntimeConfig(
        **dict(_mapping(payload.get("runtime_config"), name="runtime_config"))
    )
    state = _mapping(payload.get("repair_head_state"), name="repair_head_state")
    repair_head = ManifoldRepairHead(head_config)
    repair_head.load_state_dict(dict(state), strict=True)
    repair_head.to(device=device, dtype=dtype)
    repair_head.requires_grad_(False)
    repair_head.eval()
    transform = PMTRLogitTransform(
        repair_head=repair_head,
        tokenizer=tokenizer,
        config=runtime_config,
    )
    transform.eval()
    return LoadedPMTRCheckpoint(
        transform=transform,
        metadata={
            "schema": PMTR_CHECKPOINT_SCHEMA,
            "checkpoint_path": str(checkpoint_path),
            "head_config": asdict(head_config),
            "runtime_config": asdict(runtime_config),
        },
    )


__all__ = [
    "LoadedPMTRCheckpoint",
    "PMTR_CHECKPOINT_SCHEMA",
    "load_pmtr_checkpoint",
    "save_pmtr_checkpoint",
]
