"""Read-only field-coverage audit for PlanGraph v1 source JSONL."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Dict

from crystal_dlm.plangraph_v1 import PlanGraphError, plangraph_from_record


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _failure_category(exc: Exception) -> str:
    message = str(exc).lower()
    if "representation" in message:
        return "unsupported_representation"
    if "missing plan_state" in message:
        return "missing_plan_state"
    if "missing dynamic-v1 answer" in message:
        return "missing_answer"
    if "answer does not match plan_state" in message:
        return "answer_plan_mismatch"
    if isinstance(exc, PlanGraphError):
        return "plangraph_contract"
    return type(exc).__name__


def audit_plangraph_jsonl(
    input_path: str | Path,
    *,
    max_failure_examples: int = 20,
) -> Dict[str, Any]:
    """Audit every non-empty row; no failed row is replaced or omitted."""

    path = Path(input_path).expanduser().resolve()
    counters: dict[str, Counter[Any]] = {
        "num_atoms": Counter(),
        "arity": Counter(),
        "lattice_system": Counter(),
        "spacegroup_bucket": Counter(),
        "charge_bucket": Counter(),
        "site_group_count": Counter(),
        "failure_category": Counter(),
    }
    field_coverage = Counter(
        {
            "oxidation_candidates_nonempty": 0,
            "spacegroup_known": 0,
            "volume_per_atom_known": 0,
            "lattice_system_known": 0,
            "charge_bucket_known": 0,
        }
    )
    total = 0
    converted = 0
    failure_examples: list[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            total += 1
            try:
                record = json.loads(raw_line)
                if not isinstance(record, dict):
                    raise PlanGraphError("source row must be a JSON object")
                graph = plangraph_from_record(record)
            except json.JSONDecodeError as exc:
                category = "json_decode"
                error = f"{type(exc).__name__}: {exc.msg}"
            except Exception as exc:  # noqa: BLE001 - audit must count every failure.
                category = _failure_category(exc)
                error = f"{type(exc).__name__}: {exc}"
            else:
                converted += 1
                composition = graph["composition"]
                symmetry = graph["symmetry"]
                lattice = graph["lattice"]
                counters["num_atoms"][int(composition["N"])] += 1
                counters["arity"][len(composition["elements"])] += 1
                counters["lattice_system"][symmetry["lattice_system"]] += 1
                counters["spacegroup_bucket"][symmetry["spacegroup_bucket"]] += 1
                counters["charge_bucket"][composition["charge_bucket"]] += 1
                counters["site_group_count"][len(graph["site_groups"])] += 1
                field_coverage["oxidation_candidates_nonempty"] += int(
                    bool(composition["oxidation_candidates"])
                )
                field_coverage["spacegroup_known"] += int(
                    symmetry["spacegroup_bucket"] != "sg_unknown"
                )
                field_coverage["volume_per_atom_known"] += int(
                    lattice["volume_per_atom_bin"] != "volpa_unknown"
                )
                field_coverage["lattice_system_known"] += int(
                    symmetry["lattice_system"] != "unknown"
                )
                field_coverage["charge_bucket_known"] += int(
                    composition["charge_bucket"] != "validator_unavailable"
                )
                continue
            counters["failure_category"][category] += 1
            if len(failure_examples) < int(max_failure_examples):
                failure_examples.append(
                    {
                        "line_number": line_number,
                        "category": category,
                        "error": error,
                    }
                )

    def sorted_counter(counter: Counter[Any]) -> Dict[str, int]:
        return {
            str(key): int(counter[key])
            for key in sorted(counter, key=lambda item: str(item))
        }

    coverage = {
        key: {
            "count": int(value),
            "fraction_of_converted": (float(value) / converted if converted else 0.0),
        }
        for key, value in sorted(field_coverage.items())
    }
    return {
        "audit_version": "plangraph_v1_field_coverage_v1",
        "input_path": str(path),
        "input_sha256": _sha256_file(path),
        "total_rows": total,
        "converted_rows": converted,
        "failed_rows": total - converted,
        "conversion_rate": float(converted) / total if total else 0.0,
        "all_rows_converted": total > 0 and converted == total,
        "distributions": {
            key: sorted_counter(counter)
            for key, counter in counters.items()
            if key != "failure_category"
        },
        "field_coverage": coverage,
        "failure_categories": sorted_counter(counters["failure_category"]),
        "failure_examples": failure_examples,
    }


__all__ = ["audit_plangraph_jsonl"]
