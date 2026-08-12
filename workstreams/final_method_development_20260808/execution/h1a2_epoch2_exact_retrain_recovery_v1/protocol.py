"""Fail-closed helpers for the immutable H1-A2 epoch-2 retraining run."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping


HEX_SHA = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected one JSON object")
    return value


def write_json_exclusive(path: str | Path, value: Mapping[str, Any]) -> None:
    location = Path(path)
    location.parent.mkdir(parents=True, exist_ok=True)
    with location.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def require_hex_sha(value: str, label: str) -> str:
    observed = str(value).strip().lower()
    if HEX_SHA.fullmatch(observed) is None:
        raise ValueError(f"{label} must be one lowercase SHA-256")
    return observed


def require_file(path: str | Path, expected_sha256: str, label: str) -> Path:
    location = Path(path).resolve()
    if not location.is_file():
        raise FileNotFoundError(location)
    expected = require_hex_sha(expected_sha256, label)
    observed = sha256_file(location)
    if observed != expected:
        raise ValueError(f"{label} changed: expected={expected} observed={observed}")
    return location


def require_source_manifest(source_dir: str | Path, expected_manifest_sha256: str) -> Path:
    source = Path(source_dir).resolve()
    manifest = require_file(
        source / "SOURCE_SHA256.txt", expected_manifest_sha256, "execution source manifest"
    )
    listed: set[str] = set()
    for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        pieces = line.split("  ", 1)
        if len(pieces) != 2:
            raise ValueError(f"manifest line {line_number} is malformed")
        expected, relative = pieces
        require_hex_sha(expected, f"manifest line {line_number}")
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"manifest line {line_number} has unsafe path")
        listed.add(relative_path.as_posix())
        if sha256_file(source / relative_path) != expected:
            raise ValueError(f"source file changed: {relative}")
    observed = {
        path.relative_to(source).as_posix()
        for path in source.rglob("*")
        if path.is_file()
        and path.name != "SOURCE_SHA256.txt"
        and "__pycache__" not in path.parts
        and path.suffix not in {".pyc", ".pyo"}
    }
    if listed != observed:
        raise ValueError(
            f"source file set changed: missing={sorted(listed-observed)}, extra={sorted(observed-listed)}"
        )
    return manifest


def validate_config(config: Mapping[str, Any]) -> None:
    training = config.get("training") or {}
    resources = config.get("resource_policy") or {}
    auth = config.get("authorization") or {}
    lora = training.get("lora") or {}
    shards = (config.get("base_model") or {}).get("model_shards") or {}
    if (
        config.get("schema") != "h1a2_epoch2_exact_retrain_recovery_config_v1"
        or int(training.get("seed", -1)) != 17
        or training.get("command_model_path") != "/public/home/jiaosz/ywliang/models/Meta-Llama-3-8B/"
        or training.get("command_checkpoint_path") != "runs/20260602_182700-h1a2-rich-l3base-256/outputs/h1a2_llama_rich_sft/final"
        or training.get("command_data_dir") != "data/dlm_sft/mp_20_h1a2_rich_planner_noid_l3base"
        or int(training.get("max_length", -1)) != 768
        or float(training.get("epochs", -1)) != 1.0
        or int(training.get("batch_size", -1)) != 1
        or int(training.get("gradient_accumulation", -1)) != 8
        or int(training.get("expected_updates", -1)) != 3392
        or float(training.get("learning_rate", -1)) != 2e-5
        or float(training.get("weight_decay", -1)) != 0.0
        or int(training.get("warmup_steps", -1)) != 100
        or int(training.get("eval_steps", -1)) != 500
        or int(training.get("eval_max_batches", -1)) != 50
        or training.get("precision") != "bf16"
        or training.get("gradient_checkpointing") is not True
        or training.get("single_process") is not True
        or int(lora.get("rank", -1)) != 16
        or int(lora.get("alpha", -1)) != 32
        or float(lora.get("dropout", -1)) != 0.05
        or resources.get("partition") != "gpu"
        or int(resources.get("gpus", -1)) != 1
        or int(resources.get("cpus", -1)) > 8
        or set(shards) != {
            "model-00001-of-00004.safetensors",
            "model-00002-of-00004.safetensors",
            "model-00003-of-00004.safetensors",
            "model-00004-of-00004.safetensors",
        }
        or any(int((spec or {}).get("bytes", -1)) <= 0 for spec in shards.values())
        or auth.get("training") is not True
        or any(
            auth.get(key) is not False
            for key in (
                "sampling", "body_generation", "refinement", "sun",
                "materials_project_query", "retry", "replacement", "repair",
                "filter", "rerank", "rl"
            )
        )
    ):
        raise ValueError("exact H1-A2 retraining contract changed")
