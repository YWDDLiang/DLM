"""Fail-closed helpers for the R03 raw-Plan P0+B3 repeat panel."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, Mapping


CELLS = ("M00", "M10", "M01", "M11")
DENOMINATOR = 256
HEX_SHA = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_sha256(value: Any) -> str:
    return sha256_text(
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )


def read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected one JSON object")
    return value


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected one object")
            rows.append(value)
    return rows


def write_json_exclusive(path: str | Path, value: Mapping[str, Any]) -> None:
    location = Path(path)
    location.parent.mkdir(parents=True, exist_ok=True)
    with location.open("x", encoding="utf-8") as handle:
        json.dump(
            value,
            handle,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_jsonl_exclusive(
    path: str | Path, rows: Iterable[Mapping[str, Any]]
) -> None:
    location = Path(path)
    location.parent.mkdir(parents=True, exist_ok=True)
    with location.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    allow_nan=False,
                )
                + "\n"
            )
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
        raise ValueError(
            f"{label} changed: expected={expected} observed={observed}"
        )
    return location


def ordered_rows(
    rows: Iterable[Mapping[str, Any]], *, ordinal_field: str
) -> list[dict[str, Any]]:
    ordered = sorted(
        (dict(row) for row in rows), key=lambda row: int(row[ordinal_field])
    )
    if (
        len(ordered) != DENOMINATOR
        or [int(row.get(ordinal_field, -1)) for row in ordered]
        != list(range(DENOMINATOR))
    ):
        raise ValueError(f"{ordinal_field} coverage changed")
    return ordered


def validate_cell(value: str) -> str:
    cell = str(value)
    if cell not in CELLS:
        raise ValueError(f"cell must be one of {CELLS}")
    return cell


def attempt_id(cell: str, ordinal: int) -> str:
    return f"h1-ef-fourcell-{validate_cell(cell).lower()}-{int(ordinal):04d}"


def validate_config(config: Mapping[str, Any]) -> None:
    body = config.get("body") or {}
    schedule = config.get("schedule") or {}
    refiner = config.get("refiner") or {}
    sun = config.get("sun") or {}
    analysis = config.get("analysis") or {}
    firewall = config.get("decision_firewall") or {}
    planner_sources = config.get("planner_sources") or {}
    expected_cells = {
        "M00": {
            "planner": "P0",
            "body": "B3",
            "role": "r03_process_repeat_0",
            "method": "R03-RAW-P0-B3-SAFEAXIS-REFINE800-DIRECT-SUN-R0",
        },
        "M10": {
            "planner": "P0",
            "body": "B3",
            "role": "r03_process_repeat_1",
            "method": "R03-RAW-P0-B3-SAFEAXIS-REFINE800-DIRECT-SUN-R1",
        },
        "M01": {
            "planner": "P0",
            "body": "B3",
            "role": "r03_process_repeat_2",
            "method": "R03-RAW-P0-B3-SAFEAXIS-REFINE800-DIRECT-SUN-R2",
        },
        "M11": {
            "planner": "P0",
            "body": "B3",
            "role": "r03_process_repeat_3",
            "method": "R03-RAW-P0-B3-SAFEAXIS-REFINE800-DIRECT-SUN-R3",
        },
    }
    if (
        config.get("schema")
        != "h1_r03_raw_plan_b3_repeats4_config_v4"
        or config.get("status")
        != "user_authorized_complete_repeat_panel_evaluation"
        or int(config.get("denominator", -1)) != DENOMINATOR
        or config.get("ordinals") != "0..255"
        or dict(config.get("cells") or {}) != expected_cells
        or set(planner_sources) != {"P0", "SFT-v2"}
        or planner_sources.get("P0", {}).get("raw_generations_sha256")
        != "c29857e33bd89e94e2257d3b752a0c15bbe6953ac1b7d9f11e575056c6114f79"
        or planner_sources.get("SFT-v2", {}).get("raw_generations_sha256")
        != planner_sources.get("P0", {}).get("raw_generations_sha256")
        or int(planner_sources.get("P0", {}).get("expected_parsed", -1)) != 254
        or int(planner_sources.get("SFT-v2", {}).get("expected_parsed", -1)) != 254
        or planner_sources.get("SFT-v2", {}).get("unused_schema_slot") is not True
        or set((config.get("source_dependencies") or {}).keys())
        != {"r03d", "r03e", "r03f_mpcomplete"}
        or set((body.get("models") or {}).keys()) != {"B0", "B3"}
        or (body.get("models") or {}).get("B3", {}).get("adapter_sha256")
        != "ab4f3b82dfcafdcd0d111bc7ee424ff08ea0932a1a1466beaf91539917922bc7"
        or body.get("exact_length_generation") is not True
        or body.get("answer_token_count_formula") != "7+4N"
        or float(body.get("temperature", -1.0)) != 0.7
        or float(body.get("cfg_scale", -1.0)) != 0.0
        or int(body.get("max_batch_size", -1)) != 8
        or schedule.get("policy") != "d2_safe_axis"
        or schedule.get("all_xy_must_precede_all_z") is not True
        or schedule.get("mixed_axis_coordinate_groups_allowed") is not False
        or int(schedule.get("required_z_before_xy_count", -1)) != 0
        or int(refiner.get("diffusion_steps", -1)) != 800
        or int(refiner.get("effective_batch_size", -1)) != 1
        or sun.get("mp_api_enabled") is not False
        or sun.get("mp_api_completion_before_sbatch") is not True
        or sun.get("frozen_cache_only") is not True
        or int(sun.get("wanted_chemsys_count", -1)) != 227
        or int(sun.get("missing_chemsys_count", -1)) != 107
        or sun.get("r03f_snapshot_sha256")
        != "56f91774c798854d253c0726773593c415456a8b5361f31802c44d8e1bbad917"
        or (config.get("direct") or {})
        .get("gcd_before_comp_valid", {})
        .get("required_order")
        != "gcd_then_smact_validity"
        or int(analysis.get("bootstrap_draws", -1)) != 50000
        or int(analysis.get("bootstrap_seed", -1)) != 20260810
        or analysis.get("pooled_1024_is_descriptive_only") is not True
        or len((analysis.get("historical_r03_b0") or {}).get("attempt_results") or [])
        != 4
        or any(config.get(key) is not False for key in (
            "retry",
            "replacement",
            "repair",
            "filter",
            "rerank",
        ))
        or firewall.get("complete_repeat_panel_reporting_required") is not True
        or firewall.get("historical_summary_may_replace_current_repeat") is not False
        or firewall.get("formal_promotion") is not False
        or firewall.get("automatic_checkpoint_reselection") is not False
        or firewall.get("automatic_training") is not False
        or firewall.get("automatic_downstream") is not False
        or firewall.get("automatic_rl") is not False
    ):
        raise ValueError("R03 raw-Plan B3 repeat protocol or firewall changed")


def require_source_manifest(
    source_dir: str | Path, expected_manifest_sha256: str
) -> Path:
    source = Path(source_dir).resolve()
    manifest = require_file(
        source / "SOURCE_SHA256.txt",
        expected_manifest_sha256,
        "execution source manifest",
    )
    listed: set[str] = set()
    for line_number, line in enumerate(
        manifest.read_text(encoding="utf-8").splitlines(), start=1
    ):
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
            f"source file set changed: missing={sorted(listed-observed)}, "
            f"extra={sorted(observed-listed)}"
        )
    return manifest
