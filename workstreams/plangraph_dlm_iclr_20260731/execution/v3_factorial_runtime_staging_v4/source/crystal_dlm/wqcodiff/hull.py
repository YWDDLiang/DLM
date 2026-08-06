"""Evaluator-specific reference-energy closure and convex-hull artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import write_json_exclusive


def _read_energy_records(paths: Sequence[str | Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for raw_path in paths:
        path = Path(raw_path)
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                if payload.get("schema") != "wqcodiff_reference_energy_v1":
                    raise ValueError(f"{path}:{line_number}: invalid reference-energy schema")
                key = (str(payload["reference_id"]), str(payload["stage"]))
                if key in seen:
                    raise ValueError(f"duplicate immutable reference energy {key}")
                seen.add(key)
                records.append(payload)
    if not records:
        raise ValueError("no evaluator-specific reference energies supplied")
    return records


def _phase_diagram(entries_payload: Sequence[Mapping[str, Any]]) -> Any:
    from pymatgen.analysis.phase_diagram import PhaseDiagram
    from pymatgen.entries.computed_entries import ComputedEntry

    entries = [
        ComputedEntry(
            payload["composition"],
            float(payload["energy_total_ev"]),
            entry_id=str(payload["reference_id"]),
        )
        for payload in entries_payload
    ]
    return PhaseDiagram(entries)


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_hull_closure_step(
    energy_paths: Sequence[str | Path],
    *,
    output_path: str | Path,
    round_index: int,
    threshold_ev_per_atom: float = 0.2,
    max_rounds: int = 3,
    expected_reference_count: int = 45_229,
    allow_nonpaper_reference_count: bool = False,
) -> dict[str, Any]:
    if threshold_ev_per_atom != 0.2 or max_rounds != 3:
        raise ValueError("hull closure threshold/round cap differs from protocol")
    if not 0 <= round_index <= max_rounds:
        raise ValueError("invalid hull closure round")
    records = _read_energy_records(energy_paths)
    evaluators = {str(record["evaluator"]) for record in records}
    contracts = {str(record["contract_hash"]) for record in records}
    if len(evaluators) != 1 or len(contracts) != 1:
        raise ValueError("reference hull cannot mix evaluators or contract hashes")
    raw_records = [record for record in records if record["stage"] == "raw"]
    relaxed_records = [record for record in records if record["stage"] == "relaxed"]
    if (
        len(raw_records) != expected_reference_count
        and not allow_nonpaper_reference_count
    ):
        raise ValueError(
            f"final hull requires {expected_reference_count} raw reference records, "
            f"found {len(raw_records)}"
        )
    failed_records = [
        record
        for record in (*raw_records, *relaxed_records)
        if record.get("status") != "succeeded"
    ]
    if failed_records:
        result = {
            "schema": "wqcodiff_evaluator_hull_v1",
            "evaluator": next(iter(evaluators)),
            "contract_hash": next(iter(contracts)),
            "round_index": round_index,
            "threshold_ev_per_atom": threshold_ev_per_atom,
            "max_rounds": max_rounds,
            "reference_count": len(raw_records),
            "relaxed_count": len(relaxed_records),
            "pending_relaxation_ids": [],
            "pending_count": 0,
            "closed": False,
            "gate_passed": False,
            "failure_reason": "reference_energy_or_relaxation_failure",
            "failure_count": len(failed_records),
            "failed_reference_ids": sorted(
                str(record["reference_id"]) for record in failed_records
            ),
            "entries": [],
            "hull_sha256": "",
        }
        location = Path(output_path)
        write_json_exclusive(location, result)
        return result

    raw: dict[str, Mapping[str, Any]] = {}
    relaxed: dict[str, Mapping[str, Any]] = {}
    for record in records:
        if record.get("status") != "succeeded":
            continue
        target = relaxed if record["stage"] == "relaxed" else raw
        target[str(record["reference_id"])] = record
    if not raw:
        raise ValueError("all-reference raw single-point energies are required")
    chosen = {
        reference_id: relaxed.get(reference_id, record)
        for reference_id, record in raw.items()
    }
    phase_diagram = _phase_diagram(list(chosen.values()))
    from pymatgen.entries.computed_entries import ComputedEntry

    pending: list[str] = []
    ehull: dict[str, float | None] = {}
    for reference_id, record in sorted(chosen.items()):
        entry = ComputedEntry(
            record["composition"],
            float(record["energy_total_ev"]),
            entry_id=reference_id,
        )
        try:
            value = float(phase_diagram.get_e_above_hull(entry))
        except Exception:
            value = None
        ehull[reference_id] = value
        element_count = len(entry.composition.elements)
        if reference_id not in relaxed and (
            element_count == 1
            or (value is not None and value <= threshold_ev_per_atom)
        ):
            pending.append(reference_id)
    closed = not pending
    gate_passed = closed and round_index <= max_rounds
    if pending and round_index >= max_rounds:
        gate_passed = False
    entries_payload = [
        {
            "reference_id": reference_id,
            "structure_hash": record["structure_hash"],
            "composition": record["composition"],
            "energy_total_ev": float(record["energy_total_ev"]),
            "energy_per_atom_ev": float(record["energy_per_atom_ev"]),
            "energy_stage": str(record["stage"]),
            "e_above_hull_ev_per_atom": ehull[reference_id],
        }
        for reference_id, record in sorted(chosen.items())
    ]
    hash_payload = {
        "evaluator": next(iter(evaluators)),
        "contract_hash": next(iter(contracts)),
        "entries": entries_payload,
    }
    result = {
        "schema": "wqcodiff_evaluator_hull_v1",
        "evaluator": next(iter(evaluators)),
        "contract_hash": next(iter(contracts)),
        "round_index": round_index,
        "threshold_ev_per_atom": threshold_ev_per_atom,
        "max_rounds": max_rounds,
        "reference_count": len(raw),
        "relaxed_count": len(relaxed),
        "pending_relaxation_ids": pending,
        "pending_count": len(pending),
        "closed": closed,
        "gate_passed": gate_passed,
        "failure_reason": (
            "closure_not_reached_after_three_rounds"
            if pending and round_index >= max_rounds
            else ""
        ),
        "entries": entries_payload,
        "hull_sha256": _canonical_hash(hash_payload),
    }
    location = Path(output_path)
    write_json_exclusive(location, result)
    return result


def load_frozen_hull(path: str | Path) -> tuple[Mapping[str, Any], Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != "wqcodiff_evaluator_hull_v1":
        raise ValueError("invalid evaluator hull schema")
    if not payload.get("closed") or not payload.get("gate_passed"):
        raise ValueError("evaluator hull is not closed and frozen")
    expected = _canonical_hash(
        {
            "evaluator": payload["evaluator"],
            "contract_hash": payload["contract_hash"],
            "entries": payload["entries"],
        }
    )
    if expected != payload.get("hull_sha256"):
        raise ValueError("evaluator hull hash mismatch")
    return payload, _phase_diagram(payload["entries"])
