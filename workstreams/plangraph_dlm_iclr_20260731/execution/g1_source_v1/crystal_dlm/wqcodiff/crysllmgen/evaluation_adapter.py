"""Attempt-preserving adapters around the frozen R5-C SUN executor."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..contracts import ArtifactLedger, AttemptLedger, AttemptStatus, write_json_exclusive
from .gate import sha256_file


R5C_SCRIPT_SHA256 = "510bcf297247dfab7a77ff7aa564072806f49b0c212fe670d3221d1788ef305b"
GENERATION_SCHEMAS = {
    "wqcodiff_generation_attempt_v1",
    "crysllmgen_atom_generation_attempt_v1",
    "crysllmgen_wq_generation_attempt_v1",
}


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number}: record is not a mapping")
            result.append(payload)
    return result


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _stable_structure_hash(structure: Any) -> str:
    return hashlib.sha256(
        _canonical_json(structure.as_dict()).encode("utf-8")
    ).hexdigest()


def prepare_r5c_input(
    *,
    generation_jsonl: str | Path,
    generation_ledger: str | Path,
    structures_path: str | Path,
    manifest_path: str | Path,
    method: str,
    structure_stage: str,
) -> dict[str, Any]:
    """Write only materialized structures while retaining the full denominator."""

    if structure_stage not in {"raw", "common_refiner", "mlip_relaxed"}:
        raise ValueError("unregistered structure stage")
    structures_output = Path(structures_path).resolve()
    manifest_output = Path(manifest_path).resolve()
    for path in (structures_output, manifest_output):
        if path.exists():
            raise FileExistsError(f"R5-C preparation output is immutable: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    rows = _read_jsonl(generation_jsonl)
    if not rows:
        raise ValueError("generation artifact is empty")
    ids = [str(row.get("attempt_id", "")) for row in rows]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("generation artifact attempt IDs are missing or duplicated")
    if any(row.get("schema") not in GENERATION_SCHEMAS for row in rows):
        raise ValueError("unsupported generation artifact schema")
    if any(row.get("method") != method for row in rows):
        raise ValueError("generation artifact method mismatch")

    ledger = AttemptLedger(generation_ledger)
    audit = ledger.audit(terminal_stage="generation", expected_attempt_ids=ids)
    if not audit.ok:
        raise ValueError("generation attempt ledger failed before R5-C preparation")
    terminal = {
        record.attempt_id: record
        for record in ledger.records()
        if record.stage == "generation" and record.status.terminal
    }
    if set(terminal) != set(ids):
        raise ValueError("generation artifact/ledger denominators differ")

    from pymatgen.core import Structure
    from pymatgen.io.ase import AseAtomsAdaptor
    import ase.io

    atoms = []
    index_mapping: list[dict[str, Any]] = []
    generation_failures: list[dict[str, Any]] = []
    for ordinal, row in enumerate(rows):
        attempt_id = ids[ordinal]
        record = terminal[attempt_id]
        row_status = str(row.get("status", ""))
        if row_status != record.status.value:
            raise ValueError("generation artifact/ledger status mismatch")
        structure_payload = row.get("structure")
        if record.status is not AttemptStatus.SUCCEEDED or not isinstance(
            structure_payload, Mapping
        ):
            generation_failures.append(
                {
                    "attempt_id": attempt_id,
                    "generation_ordinal": ordinal,
                    "status": record.status.value,
                    "reason": record.reason or str(row.get("reason", "")),
                }
            )
            continue
        structure = Structure.from_dict(dict(structure_payload))
        ase_atoms = AseAtomsAdaptor.get_atoms(structure)
        extxyz_index = len(atoms)
        ase_atoms.info["attempt_id"] = attempt_id
        ase_atoms.info["generation_ordinal"] = ordinal
        ase_atoms.info["structure_stage"] = structure_stage
        atoms.append(ase_atoms)
        index_mapping.append(
            {
                "extxyz_index": extxyz_index,
                "attempt_id": attempt_id,
                "generation_ordinal": ordinal,
                "structure_hash": _stable_structure_hash(structure),
            }
        )
    if not atoms:
        raise ValueError("no generated structure reached the R5-C evaluator")
    ase.io.write(structures_output, atoms, format="extxyz")
    manifest = {
        "schema": "crysllmgen_r5c_input_manifest_v1",
        "method": method,
        "structure_stage": structure_stage,
        "total_attempts": len(rows),
        "materialized_structures": len(atoms),
        "generation_failures": generation_failures,
        "index_mapping": index_mapping,
        "generation_jsonl": str(Path(generation_jsonl).resolve()),
        "generation_jsonl_sha256": sha256_file(generation_jsonl),
        "generation_ledger": str(Path(generation_ledger).resolve()),
        "generation_ledger_sha256": sha256_file(generation_ledger),
        "structures_path": str(structures_output),
        "structures_sha256": sha256_file(structures_output),
        "generation_audit": dataclasses.asdict(audit),
    }
    write_json_exclusive(manifest_output, manifest)
    return manifest


def _metric_columns(path: str | Path) -> dict[str, list[Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("R5-C detailed metrics root is not a mapping")
    columns = {
        str(key): list(value)
        for key, value in payload.items()
        if isinstance(value, list)
    }
    if "energy_above_hull" not in columns and "energy_above_hull_per_atom" in columns:
        columns["energy_above_hull"] = columns["energy_above_hull_per_atom"]
    for required in ("energy_above_hull", "novel_unique"):
        if required not in columns:
            raise ValueError(f"R5-C detailed metrics is missing {required}")
    lengths = {len(value) for value in columns.values()}
    if len(lengths) != 1:
        raise ValueError("R5-C detailed metric columns have unequal lengths")
    return columns


def _load_index_set(path: str | Path, *, key: str) -> set[int]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    values = payload.get(key, ())
    if not isinstance(values, list):
        raise ValueError(f"invalid R5-C failure index payload: {key}")
    return {int(value) for value in values}


def aggregate_r5c_output(
    *,
    input_manifest_path: str | Path,
    r5c_summary_path: str | Path,
    detailed_metrics_path: str | Path,
    unsupported_path: str | Path,
    relax_failures_path: str | Path,
    output_jsonl: str | Path,
    output_summary: str | Path,
    evaluator: str,
    evaluator_checkpoint: str | Path,
    r5c_script: str | Path,
) -> dict[str, Any]:
    """Map R5-C survivor rows back to every originally submitted attempt."""

    if evaluator != "MatterSim-v1.0.0-5M":
        raise ValueError("the frozen R5-C primary adapter requires MatterSim 5M")
    if sha256_file(r5c_script) != R5C_SCRIPT_SHA256:
        raise ValueError("R5-C executor changed")
    manifest = json.loads(Path(input_manifest_path).read_text(encoding="utf-8"))
    if manifest.get("schema") != "crysllmgen_r5c_input_manifest_v1":
        raise ValueError("invalid R5-C input manifest")
    if sha256_file(manifest["structures_path"]) != manifest["structures_sha256"]:
        raise ValueError("R5-C input structures changed")
    r5c_summary = json.loads(Path(r5c_summary_path).read_text(encoding="utf-8"))
    materialized = int(manifest["materialized_structures"])
    if int(r5c_summary.get("num_structures", -1)) != materialized:
        raise ValueError("R5-C input/output structure denominators differ")
    unsupported_payload = json.loads(Path(unsupported_path).read_text(encoding="utf-8"))
    unsupported_records = unsupported_payload.get("unsupported_records", ())
    if not isinstance(unsupported_records, list):
        raise ValueError("invalid unsupported failure artifact")
    unsupported = {int(value["index"]) for value in unsupported_records}
    relax_failed = _load_index_set(
        relax_failures_path,
        key="relax_failed_indices",
    )
    if unsupported & relax_failed:
        raise ValueError("R5-C unsupported and relaxation failures overlap")
    successful_extxyz = [
        index
        for index in range(materialized)
        if index not in unsupported and index not in relax_failed
    ]
    columns = _metric_columns(detailed_metrics_path)
    metric_rows = len(columns["energy_above_hull"])
    if metric_rows != len(successful_extxyz):
        raise ValueError("R5-C detailed rows do not map to successful input indices")

    by_extxyz = {
        int(value["extxyz_index"]): value for value in manifest["index_mapping"]
    }
    if set(by_extxyz) != set(range(materialized)):
        raise ValueError("R5-C index mapping is incomplete")
    evaluated: dict[str, dict[str, Any]] = {}
    for metric_index, extxyz_index in enumerate(successful_extxyz):
        mapping = by_extxyz[extxyz_index]
        evaluated[str(mapping["attempt_id"])] = {
            key: values[metric_index] for key, values in columns.items()
        }

    output_path = Path(output_jsonl).resolve()
    summary_path = Path(output_summary).resolve()
    for path in (output_path, summary_path):
        if path.exists():
            raise FileExistsError(f"R5-C aggregation output is immutable: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    ledger = ArtifactLedger(output_path)
    counts = {
        "total": int(manifest["total_attempts"]),
        "generation_failed": 0,
        "unsupported": 0,
        "nonconverged": 0,
        "evaluated": 0,
        "stable_0p0": 0,
        "stable_0p1": 0,
        "novel_unique": 0,
        "sun_0p0": 0,
        "sun_0p1": 0,
    }
    generation_failures = {
        str(value["attempt_id"]): value for value in manifest["generation_failures"]
    }
    all_ids = [
        str(value["attempt_id"])
        for value in sorted(
            [*manifest["index_mapping"], *manifest["generation_failures"]],
            key=lambda value: int(value["generation_ordinal"]),
        )
    ]
    unsupported_ids = {str(by_extxyz[index]["attempt_id"]) for index in unsupported}
    nonconverged_ids = {
        str(by_extxyz[index]["attempt_id"]) for index in relax_failed
    }
    for attempt_id in all_ids:
        status = AttemptStatus.SUCCEEDED
        reason = ""
        metrics: dict[str, Any] = {}
        if attempt_id in generation_failures:
            status = AttemptStatus.FAILED
            reason = "generation:" + str(generation_failures[attempt_id].get("reason", ""))
            counts["generation_failed"] += 1
        elif attempt_id in unsupported_ids:
            status = AttemptStatus.UNSUPPORTED_ELEMENT
            reason = "r5c_unsupported_element"
            counts["unsupported"] += 1
        elif attempt_id in nonconverged_ids:
            status = AttemptStatus.NONCONVERGED
            reason = "r5c_relaxation_nonconverged"
            counts["nonconverged"] += 1
        else:
            metrics = evaluated[attempt_id]
            ehull = float(metrics["energy_above_hull"])
            if not math.isfinite(ehull):
                raise ValueError("R5-C returned non-finite energy above hull")
            novel_unique = bool(metrics["novel_unique"])
            stable_0p0 = ehull < 0.0
            stable_0p1 = ehull < 0.1
            sun_0p0 = stable_0p0 and novel_unique
            sun_0p1 = stable_0p1 and novel_unique
            metrics.update(
                {
                    "stable_0p0": stable_0p0,
                    "stable_0p1": stable_0p1,
                    "sun_0p0": sun_0p0,
                    "sun_0p1": sun_0p1,
                }
            )
            counts["evaluated"] += 1
            counts["stable_0p0"] += int(stable_0p0)
            counts["stable_0p1"] += int(stable_0p1)
            counts["novel_unique"] += int(novel_unique)
            counts["sun_0p0"] += int(sun_0p0)
            counts["sun_0p1"] += int(sun_0p1)
        ledger.append(
            {
                "schema": "crysllmgen_r5c_attempt_result_v1",
                "attempt_id": attempt_id,
                "method": manifest["method"],
                "structure_stage": manifest["structure_stage"],
                "evaluator": evaluator,
                "status": status.value,
                "reason": reason,
                "metrics": metrics,
            }
        )
    if sum(
        counts[key]
        for key in ("generation_failed", "unsupported", "nonconverged", "evaluated")
    ) != counts["total"]:
        raise RuntimeError("R5-C terminal accounting does not close")
    rates = {
        key: counts[key] / counts["total"]
        for key in (
            "evaluated",
            "stable_0p0",
            "stable_0p1",
            "novel_unique",
            "sun_0p0",
            "sun_0p1",
        )
    }
    summary = {
        "schema": "crysllmgen_r5c_attempt_summary_v1",
        "ok": True,
        "method": manifest["method"],
        "structure_stage": manifest["structure_stage"],
        "evaluator": evaluator,
        "denominator": "all_generation_attempts",
        "counts": counts,
        "rates": rates,
        "r5c_threshold_semantics": {"sun_0p0": "ehull_lt_0p0", "sun_0p1": "ehull_lt_0p1"},
        "input_manifest": str(Path(input_manifest_path).resolve()),
        "input_manifest_sha256": sha256_file(input_manifest_path),
        "r5c_summary": str(Path(r5c_summary_path).resolve()),
        "r5c_summary_sha256": sha256_file(r5c_summary_path),
        "detailed_metrics_sha256": sha256_file(detailed_metrics_path),
        "attempt_results": str(output_path),
        "attempt_results_sha256": sha256_file(output_path),
        "evaluator_checkpoint": str(Path(evaluator_checkpoint).resolve()),
        "evaluator_checkpoint_sha256": sha256_file(evaluator_checkpoint),
        "r5c_script": str(Path(r5c_script).resolve()),
        "r5c_script_sha256": R5C_SCRIPT_SHA256,
    }
    write_json_exclusive(summary_path, summary)
    return summary


def write_terminal_evaluator_failure(
    *,
    input_manifest_path: str | Path,
    output_jsonl: str | Path,
    output_summary: str | Path,
    reason: str,
    evaluator: str,
) -> dict[str, Any]:
    """Close the denominator if the frozen evaluator process itself fails."""

    manifest = json.loads(Path(input_manifest_path).read_text(encoding="utf-8"))
    records: list[Mapping[str, Any]] = sorted(
        [*manifest["index_mapping"], *manifest["generation_failures"]],
        key=lambda value: int(value["generation_ordinal"]),
    )
    ledger = ArtifactLedger(output_jsonl)
    for value in records:
        ledger.append(
            {
                "schema": "crysllmgen_r5c_attempt_result_v1",
                "attempt_id": str(value["attempt_id"]),
                "method": manifest["method"],
                "structure_stage": manifest["structure_stage"],
                "evaluator": evaluator,
                "status": AttemptStatus.FAILED.value,
                "reason": f"evaluator_process:{reason}",
                "metrics": {},
            }
        )
    summary = {
        "schema": "crysllmgen_r5c_attempt_summary_v1",
        "ok": False,
        "method": manifest["method"],
        "structure_stage": manifest["structure_stage"],
        "evaluator": evaluator,
        "denominator": "all_generation_attempts",
        "counts": {"total": len(records), "evaluator_failed": len(records)},
        "rates": {"sun_0p0": 0.0, "sun_0p1": 0.0},
        "reason": reason,
        "attempt_results": str(Path(output_jsonl).resolve()),
        "attempt_results_sha256": sha256_file(output_jsonl),
    }
    write_json_exclusive(output_summary, summary)
    return summary
