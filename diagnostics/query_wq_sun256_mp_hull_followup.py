#!/usr/bin/env python3
"""Resolve frozen WQ S.U.N. hull-unknown attempts with the official MP client.

This is a post-hoc sensitivity analysis.  It never modifies the frozen
generation, CrysLLMGen, or S.U.N. ledgers, and it does not alter the original
all-attempt headline metrics.  The Materials Project API key is read only from
``MP_API_KEY`` and is never serialized.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import pymatgen.entries.computed_entries as _computed_entries
import pymatgen.entries.compatibility as _compatibility

# MP's serialized payloads still use these historical module paths. Current
# pymatgen moved the objects without retaining the import aliases.
sys.modules.setdefault("pymatgen.core.entries", _computed_entries)
sys.modules.setdefault("pymatgen.analysis.compatibility", _compatibility)

from mp_api.client import MPRester
from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.core import Composition
from pymatgen.entries.computed_entries import ComputedEntry


THERMO_TYPE = "GGA_GGA+U"
THERMO_CRITERIA = {"thermo_types": [THERMO_TYPE]}
FROZEN_A100_EVAL_SUN_SHA256 = (
    "564b4490f01464012277653951f8a55b5c1575bc78091f5a06db25ca9339852b"
)
STRICT_THRESHOLD = 0.0
META_THRESHOLD = 0.1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sanitized_error(exc: BaseException) -> dict[str, str]:
    message = str(exc)
    for separator in ("Content:", "Response:", "{\"data\":"):
        message = message.split(separator, 1)[0]
    message = " ".join(message.split())[:500]
    return {"type": type(exc).__name__, "message": message}


def finite_float(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite: {value!r}")
    return result


def classify_hull(e_above_hull: float) -> str:
    if e_above_hull <= STRICT_THRESHOLD:
        return "strict_stable"
    if e_above_hull <= META_THRESHOLD:
        return "meta_only_stable"
    return "unstable"


def installed_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def exclusive_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def append_jsonl(handle: Any, value: Any) -> None:
    handle.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def slim_mp_entries(entries: list[Any]) -> list[dict[str, Any]]:
    """Freeze the exact fields used by the registered A100 phase diagram."""

    result = []
    for entry in entries:
        energy = finite_float(entry.energy, "reference entry energy")
        result.append(
            {
                "entry_id": (
                    None
                    if getattr(entry, "entry_id", None) is None
                    else str(entry.entry_id)
                ),
                "composition": entry.composition.as_dict(),
                "energy": energy,
            }
        )
    result.sort(
        key=lambda row: (
            "" if row["entry_id"] is None else row["entry_id"],
            json.dumps(row["composition"], sort_keys=True),
            row["energy"],
        )
    )
    return result


def phase_diagram_from_slim(entries: list[dict[str, Any]]) -> PhaseDiagram:
    return PhaseDiagram(
        [
            ComputedEntry(
                row["composition"],
                float(row["energy"]),
                entry_id=row["entry_id"],
            )
            for row in entries
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--taxonomy-json", type=Path, required=True)
    parser.add_argument("--frozen-a100-eval-sun-py", type=Path, required=True)
    parser.add_argument("--cache-jsonl", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    args = parser.parse_args()

    api_key = os.environ.get("MP_API_KEY", "")
    if not api_key:
        raise RuntimeError("MP_API_KEY is required")

    taxonomy_path = args.taxonomy_json.resolve()
    frozen_eval_path = args.frozen_a100_eval_sun_py.resolve()
    cache_path = args.cache_jsonl.resolve()
    report_path = args.report_json.resolve()
    if cache_path.exists() or report_path.exists():
        raise FileExistsError("cache/report output already exists")
    if sha256_file(frozen_eval_path) != FROZEN_A100_EVAL_SUN_SHA256:
        raise ValueError("frozen A100 eval_sun.py identity changed")

    with taxonomy_path.open("r", encoding="utf-8") as handle:
        taxonomy = json.load(handle)
    manifest_key = (
        "unknown_followup_manifest"
        if "unknown_followup_manifest" in taxonomy
        else "unknown_manifest"
    )
    manifest = list(taxonomy[manifest_key])
    if len(manifest) != 95:
        raise ValueError(f"expected 95 unknown attempts, found {len(manifest)}")
    if len({str(row["attempt_id"]) for row in manifest}) != len(manifest):
        raise ValueError("duplicate attempt IDs in unknown manifest")
    if len({str(row["chemsys"]) for row in manifest}) != len(manifest):
        raise ValueError("expected one unique chemsys per unknown attempt")
    if any(
        row.get("unknown_type", row.get("category")) != "hull_unknown"
        for row in manifest
    ):
        raise ValueError("manifest includes a non-hull unknown")

    package_versions = {
        package: installed_version(package)
        for package in ("mp-api", "pymatgen", "emmet-core")
    }
    created_at = datetime.now(timezone.utc).isoformat()
    results: list[dict[str, Any]] = []

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("x", encoding="utf-8") as cache_handle:
        with MPRester(api_key) as mpr:
            try:
                database_version = mpr.get_database_version()
            except Exception as exc:
                database_version = {
                    "status": "unavailable",
                    "error": sanitized_error(exc),
                }

            for index, source in enumerate(manifest, start=1):
                composition = Composition(str(source["formula"]))
                energy_per_atom = finite_float(
                    source["energy_per_atom"], "energy_per_atom"
                )
                record: dict[str, Any] = {
                    "schema": "wq_parent_csp_sun256_mpapi_cache_v2",
                    "query_index": index,
                    "query_total": len(manifest),
                    "attempt_id": str(source["attempt_id"]),
                    "source_ordinal": int(
                        source["source_ordinal"]
                        if "source_ordinal" in source
                        else source["ordinal"]
                    ),
                    "formula": str(source["formula"]),
                    "chemsys": str(source["chemsys"]),
                    "energy_per_atom": energy_per_atom,
                    "structure_sha256": source.get("structure_sha256"),
                    "thermo_type": THERMO_TYPE,
                    "thermo_criteria": THERMO_CRITERIA,
                    "query_method": (
                        "mp_api.client.MPRester.get_entries_in_chemsys"
                    ),
                    "compatible_only": True,
                    "a100_hull_semantics": (
                        "PhaseDiagram(reference_entries) + "
                        "PDEntry(CHGNet_relaxed_energy)"
                    ),
                    "retry_or_replacement_used": False,
                }
                try:
                    reference_entries = mpr.get_entries_in_chemsys(
                        sorted(element.symbol for element in composition.elements),
                        compatible_only=True,
                        additional_criteria=THERMO_CRITERIA,
                    )
                    slim_entries = slim_mp_entries(reference_entries)
                    if not slim_entries:
                        raise RuntimeError(
                            "get_entries_in_chemsys returned no reference entries"
                        )
                    phase_diagram = phase_diagram_from_slim(slim_entries)
                    _, raw_e_above_hull = (
                        phase_diagram.get_decomp_and_e_above_hull(
                            PDEntry(
                                composition,
                                energy_per_atom * composition.num_atoms,
                            ),
                            allow_negative=True,
                        )
                    )
                    e_above_hull = max(
                        finite_float(raw_e_above_hull, "e_above_hull"), 0.0
                    )
                    record.update(
                        {
                            "query_status": "resolved",
                            "reference_entry_count": len(slim_entries),
                            "reference_entries_sha256": canonical_sha256(
                                slim_entries
                            ),
                            "reference_entries": slim_entries,
                            "phase_diagram_constructed": True,
                            "e_above_hull": e_above_hull,
                            "posthoc_category": classify_hull(e_above_hull),
                        }
                    )
                except Exception as exc:
                    record.update(
                        {
                            "query_status": "query_error",
                            "reference_entry_count": None,
                            "reference_entries_sha256": None,
                            "reference_entries": None,
                            "phase_diagram_constructed": False,
                            "e_above_hull": None,
                            "posthoc_category": "hull_unknown",
                            "error": sanitized_error(exc),
                        }
                    )
                results.append(record)
                append_jsonl(cache_handle, record)
                print(
                    json.dumps(
                        {
                            "index": index,
                            "total": len(manifest),
                            "attempt_id": record["attempt_id"],
                            "chemsys": record["chemsys"],
                            "query_status": record["query_status"],
                            "posthoc_category": record["posthoc_category"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    category_counts = Counter(row["posthoc_category"] for row in results)
    query_counts = Counter(row["query_status"] for row in results)
    resolved = int(query_counts["resolved"])
    report = {
        "schema": "wq_parent_csp_sun256_mpapi_followup_v2",
        "created_at_utc": created_at,
        "status": (
            "complete_all_resolved"
            if resolved == len(results)
            else "complete_with_query_errors"
        ),
        "contract": {
            "posthoc_sensitivity_only": True,
            "frozen_job27480_outputs_modified": False,
            "original_all_attempt_denominator_modified": False,
            "retry_or_replacement_used": False,
            "api_key_serialized": False,
            "taxonomy_manifest_key": manifest_key,
            "legacy_monty_entry_module_alias": (
                "pymatgen.core.entries -> pymatgen.entries.computed_entries"
            ),
            "legacy_monty_compatibility_module_alias": (
                "pymatgen.analysis.compatibility -> "
                "pymatgen.entries.compatibility"
            ),
            "thermo_type": THERMO_TYPE,
            "thermo_criteria": THERMO_CRITERIA,
            "query_method": "MPRester.get_entries_in_chemsys",
            "compatible_only": True,
            "a100_hull_semantics_reproduced": True,
            "strict_threshold_ev_per_atom": STRICT_THRESHOLD,
            "meta_threshold_ev_per_atom": META_THRESHOLD,
        },
        "client": {
            "package_versions": package_versions,
            "database_version": database_version,
        },
        "inputs": {
            str(taxonomy_path): {
                "bytes": taxonomy_path.stat().st_size,
                "sha256": sha256_file(taxonomy_path),
                "hull_unknown_attempts": len(manifest),
            },
            str(frozen_eval_path): {
                "bytes": frozen_eval_path.stat().st_size,
                "sha256": sha256_file(frozen_eval_path),
            },
            str(Path(__file__).resolve()): {
                "bytes": Path(__file__).resolve().stat().st_size,
                "sha256": sha256_file(Path(__file__).resolve()),
            },
        },
        "cache": {
            "path": str(cache_path),
            "bytes": cache_path.stat().st_size,
            "sha256": sha256_file(cache_path),
            "rows": len(results),
        },
        "counts": {
            "query_status": dict(query_counts),
            "posthoc_category": dict(category_counts),
            "reference_entries_total": sum(
                int(row["reference_entry_count"])
                for row in results
                if row["reference_entry_count"] is not None
            ),
        },
        "scientific_interpretation": {
            "attempts_reclassified_in_original_report": 0,
            "resolved_sensitivity_counts": {
                category: int(category_counts.get(category, 0))
                for category in (
                    "strict_stable",
                    "meta_only_stable",
                    "unstable",
                    "hull_unknown",
                )
            },
            "original_job27480_strict_all_attempt_count": 11,
            "original_job27480_meta_all_attempt_count": 70,
            "original_job27480_hull_unknown_count": 95,
            "posthoc_sensitivity_strict_all_attempt_count": (
                11 + int(category_counts.get("strict_stable", 0))
            ),
            "posthoc_sensitivity_meta_all_attempt_count": (
                70
                + int(category_counts.get("strict_stable", 0))
                + int(category_counts.get("meta_only_stable", 0))
            ),
            "posthoc_sensitivity_all_attempt_denominator": 256,
        },
    }
    exclusive_json(report_path, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "cache_sha256": report["cache"]["sha256"],
                "report": str(report_path),
                "report_sha256": sha256_file(report_path),
                "counts": report["counts"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
