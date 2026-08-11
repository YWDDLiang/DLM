#!/usr/bin/env python3
"""Shared fail-closed protocol helpers for the environment-repair V2 replay."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping


class ContractError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def line_set_sha256(values: Iterable[str]) -> str:
    ordered = sorted(set(values))
    payload = "" if not ordered else "\n".join(ordered) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ContractError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            if not isinstance(row, dict):
                raise ContractError(f"expected JSON object at {path}:{line_number}")
            rows.append(row)
    return rows


def write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_jsonl_exclusive(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": sha256_file(resolved)}


def require_file(path: Path, expected_sha256: str, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_file() or sha256_file(resolved) != expected_sha256:
        raise ContractError(f"{label} identity changed: {resolved}")
    return resolved


def require_source_manifest(source: Path, expected_manifest_sha256: str) -> None:
    root = source.resolve()
    manifest = root / "SOURCE_SHA256.txt"
    if sha256_file(manifest) != expected_manifest_sha256:
        raise ContractError("source manifest identity changed")
    expected: dict[str, str] = {}
    for raw in manifest.read_text(encoding="utf-8").splitlines():
        digest, sep, relative = raw.partition("  ")
        if not sep or len(digest) != 64 or relative.startswith(("/", "../")):
            raise ContractError("invalid source manifest row")
        expected[relative] = digest
    observed = {
        str(path.relative_to(root)).replace(os.sep, "/")
        for path in root.rglob("*")
        if path.is_file() and path.name != "SOURCE_SHA256.txt" and "__pycache__" not in path.parts
    }
    if set(expected) != observed:
        raise ContractError("source file set changed")
    for relative, digest in expected.items():
        if sha256_file(root / relative) != digest:
            raise ContractError(f"source file changed: {relative}")


def historical_paths(config: Mapping[str, Any], repeat: int) -> tuple[Path, Path, Path]:
    r03e = Path(str(config["historical"]["r03e_run_root"]))
    r03g = Path(str(config["historical"]["r03g_run_root"]))
    generation = r03e / f"repeats/{repeat}/arms/candidate/generation/generation.jsonl"
    relax_cache = r03e / f"repeats/{repeat}/arms/candidate/evaluation/r5c_a100_sun/working_chgnet_relax_cache.jsonl"
    old_attempts = r03g / f"repeats/{repeat}/arms/candidate/attempt_results.jsonl"
    return generation, relax_cache, old_attempts


def repeat_spec(config: Mapping[str, Any], repeat: int) -> Mapping[str, Any]:
    matches = [row for row in config["historical"]["repeats"] if int(row["repeat"]) == repeat]
    if len(matches) != 1:
        raise ContractError(f"missing repeat contract: {repeat}")
    return matches[0]
