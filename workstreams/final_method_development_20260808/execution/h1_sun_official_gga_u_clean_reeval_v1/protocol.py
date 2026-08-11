#!/usr/bin/env python3
"""Fail-closed helpers for the official-MP clean S.U.N. re-evaluation."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping


class ContractError(RuntimeError):
    """Raised when an immutable input or evaluation contract is violated."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def finite_float(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ContractError(f"{label} is not finite: {value!r}")
    return result


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
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ContractError(
                    f"expected JSON object at {path}:{line_number}"
                )
            rows.append(value)
    return rows


def identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(canonical_json(value) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_jsonl_exclusive(
    path: Path, rows: Iterable[Mapping[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def require_hex_sha(value: str, label: str) -> str:
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ContractError(f"{label} must be one lowercase SHA256")
    return value


def require_source_manifest(source: Path, expected_sha256: str) -> None:
    root = source.resolve()
    manifest = root / "SOURCE_SHA256.txt"
    if sha256_file(manifest) != require_hex_sha(
        expected_sha256, "source manifest SHA256"
    ):
        raise ContractError("source manifest identity changed")
    expected: dict[str, str] = {}
    for raw in manifest.read_text(encoding="utf-8").splitlines():
        digest, separator, relative = raw.partition("  ")
        if (
            not separator
            or len(digest) != 64
            or relative.startswith(("/", "../"))
            or "\\" in relative
        ):
            raise ContractError("invalid source manifest row")
        expected[relative] = digest
    observed = {
        str(path.relative_to(root)).replace(os.sep, "/")
        for path in root.rglob("*")
        if path.is_file()
        and path.name != "SOURCE_SHA256.txt"
        and "__pycache__" not in path.parts
    }
    if set(expected) != observed:
        raise ContractError("source file set changed")
    for relative, digest in expected.items():
        if sha256_file(root / relative) != digest:
            raise ContractError(f"source file changed: {relative}")


def normalized_chemsys(composition: Any) -> str:
    symbols = sorted({element.symbol for element in composition.elements})
    if not symbols:
        raise ContractError("empty composition")
    return "-".join(symbols)


def cell_specs(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for panel in config["panels"]:
        panel_name = str(panel["name"])
        root = Path(str(panel["root"]))
        template = str(panel["evaluation_dir_template"])
        success_relative = Path(
            str(panel["success_marker_relative_to_evaluation"])
        )
        if success_relative.is_absolute():
            raise ContractError("success marker must be relative to evaluation")
        expected_attempts = int(panel["expected_attempts"])
        for arm in panel["arms"]:
            for repeat in panel["repeats"]:
                for stage in panel["stages"]:
                    fields = {
                        "arm": str(arm),
                        "repeat": int(repeat),
                        "stage": str(stage),
                    }
                    relative = template.format(**fields)
                    evaluation_dir = (root / relative).resolve()
                    success_marker = (evaluation_dir / success_relative).resolve()
                    if not success_marker.is_relative_to(root.resolve()):
                        raise ContractError("success marker escapes configured panel root")
                    cell_id = (
                        f"{panel_name}__{fields['arm']}__r{fields['repeat']}__"
                        f"{fields['stage']}"
                    )
                    cells.append(
                        {
                            "cell_index": len(cells),
                            "cell_id": cell_id,
                            "panel": panel_name,
                            "arm": fields["arm"],
                            "repeat": fields["repeat"],
                            "stage": fields["stage"],
                            "expected_attempts": expected_attempts,
                            "evaluation_dir": str(evaluation_dir),
                            "success_marker": str(success_marker),
                        }
                    )
    identifiers = [cell["cell_id"] for cell in cells]
    if len(identifiers) != len(set(identifiers)):
        raise ContractError("duplicate configured cell identifiers")
    return cells
