#!/usr/bin/env python3
"""Fail-closed helpers for the nine-cell post-refine official S.U.N. panel."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping


DENOMINATOR = 256
STAGES = ("post_model494",)


class ContractError(RuntimeError):
    pass


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_hex_sha(value: str, label: str) -> str:
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ContractError(f"{label} must be one lowercase SHA256")
    return value


def require_file(path: Path, expected_sha256: str, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    if sha256_file(resolved) != require_hex_sha(expected_sha256, label):
        raise ContractError(f"{label} identity changed")
    return resolved


def require_source_manifest(source: Path, expected_sha256: str) -> None:
    root = source.resolve()
    manifest = require_file(
        root / "SOURCE_SHA256.txt", expected_sha256, "source manifest"
    )
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
                raise ContractError(f"expected object at {path}:{line_number}")
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


def finite_float(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ContractError(f"{label} is not finite")
    return result


def normalized_chemsys(composition: Any) -> str:
    symbols = sorted({element.symbol for element in composition.elements})
    if not symbols:
        raise ContractError("empty composition")
    return "-".join(symbols)


def cell_specs(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    upstream = config["upstream_generation"]
    expected = list(upstream["expected_cells"])
    if len(expected) != 9:
        raise ContractError("post-only evaluation cell count changed")
    cells: list[dict[str, Any]] = []
    for index, raw in enumerate(expected):
        cell = dict(raw)
        cell.update(
            {
                "cell_index": index,
                "expected_attempts": int(upstream["expected_attempts"]),
            }
        )
        cells.append(cell)
    if [cell["cell_id"] for cell in cells] != [
        "fresh_0",
        "fresh_1",
        "fresh_2",
        "fresh_3",
        "topology_repeat_0",
        "topology_repeat_1",
        "topology_repeat_2",
        "topology_repeat_3",
        "h1a2_b0_d1_once",
    ]:
        raise ContractError("post-only evaluation cell order changed")
    if any(cell["stage"] != "post_model494" for cell in cells):
        raise ContractError("pre-refine evaluation is forbidden")
    if len({str(cell["cell_id"]) for cell in cells}) != len(cells):
        raise ContractError("duplicate cell identifier")
    return cells


def load_upstream_cells(config: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    spec = config["upstream_generation"]
    root = Path(spec["run_root"]).resolve()
    marker = root / spec["required_marker"]
    if not marker.is_file():
        raise ContractError("upstream generation assembly is incomplete")
    terminal_path = root / spec["terminal_report"]
    terminal = read_json(terminal_path)
    registry_path = root / spec["evaluation_registry"]
    registered = read_jsonl(registry_path)
    if (
        terminal.get("status") != "complete"
        or terminal.get("ok") is not True
        or terminal.get("source_manifest_sha256") != spec["source_manifest_sha256"]
        or terminal.get("schema")
        != "h1_plan_recovery_official_sun_input_manifest_v3"
        or terminal.get("evaluated_stage") != "post_model494_only"
        or terminal.get("pre_refine_role") != "intermediate_only_not_scored"
        or int(terminal.get("evaluation_cell_count", -1)) != 9
        or int(terminal.get("attempts_per_cell", -1)) != DENOMINATOR
        or canonical_sha256(registered) != terminal.get("evaluation_cells_sha256")
    ):
        raise ContractError("upstream post-only terminal contract changed")
    expected = cell_specs(config)
    if len(registered) != len(expected):
        raise ContractError("upstream post-only registry count changed")
    keys = (
        "cell_id",
        "panel",
        "cohort_id",
        "cohort_index",
        "process_repeat",
        "stage",
        "body",
        "schedule",
    )
    merged: list[dict[str, Any]] = []
    for expected_cell, observed in zip(expected, registered):
        if any(observed.get(key) != expected_cell.get(key) for key in keys):
            raise ContractError(
                f"upstream cell metadata changed: {expected_cell['cell_id']}"
            )
        if int(observed.get("attempts", -1)) != DENOMINATOR:
            raise ContractError("upstream cell denominator changed")
        merged.append({**expected_cell, **observed})
    return terminal, merged
