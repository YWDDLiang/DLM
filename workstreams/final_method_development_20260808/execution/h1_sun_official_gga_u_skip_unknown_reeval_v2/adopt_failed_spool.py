#!/usr/bin/env python3
"""Adopt one complete fresh official-query spool with explicit unknown rows."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from protocol import (
    ContractError,
    canonical_json,
    canonical_sha256,
    identity,
    read_json,
    read_jsonl,
    require_source_manifest,
    sha256_file,
    write_json_exclusive,
)


def read_fragment(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        value = json.loads(handle.read())
    if not isinstance(value, dict):
        raise ContractError(f"invalid fragment: {path}")
    return value


def output_identity(path: Path, final: Path) -> dict[str, Any]:
    result = identity(path)
    result["path"] = str((final / path.name).resolve())
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()

    source = args.source_dir.resolve()
    require_source_manifest(source, args.source_manifest_sha256)
    config = read_json(args.config.resolve())
    adoption = config["cache_adoption"]
    if int(adoption["new_mp_queries"]) != 0:
        raise ContractError("cache adoption must not issue MP queries")

    run_root = args.run_root.resolve()
    inputs = run_root / "inputs"
    if not (inputs / "inputs_SUCCESS").is_file():
        raise ContractError("frozen input manifest is incomplete")
    wanted = read_jsonl(inputs / "wanted_chemsys.jsonl")
    input_manifest = read_json(inputs / "input_manifest.json")
    expected_total = int(adoption["expected_query_count"])
    if len(wanted) != expected_total:
        raise ContractError("wanted chemsys count changed")
    if canonical_sha256(wanted) != input_manifest["wanted_chemsys_sha256"]:
        raise ContractError("wanted chemsys identity changed")

    failed_cache = Path(adoption["source_failed_cache"]).resolve()
    spool = failed_cache / "spool"
    failure_path = failed_cache / "query_failures.json"
    if sha256_file(failure_path) != adoption["source_query_failures_sha256"]:
        raise ContractError("source query-failure identity changed")
    source_run = failed_cache.parent
    source_wanted = source_run / "inputs/wanted_chemsys.jsonl"
    if sha256_file(source_wanted) != sha256_file(inputs / "wanted_chemsys.jsonl"):
        raise ContractError("adopted and repair wanted-chemsys ledgers differ")

    expected_names = {f"{index:05d}.json.gz" for index in range(expected_total)}
    observed_names = {path.name for path in spool.iterdir() if path.is_file()}
    if observed_names != expected_names:
        raise ContractError("source spool is not exactly contiguous and complete")

    resolved: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    retries = 0
    expected_error_type = str(adoption["accepted_unresolved_error_type"])
    expected_error_message = str(adoption["accepted_unresolved_error_message"])
    required_symbol = str(adoption["require_every_unresolved_chemsys_to_contain"])
    for index, spec in enumerate(wanted):
        row = read_fragment(spool / f"{index:05d}.json.gz")
        if (
            int(row.get("query_index", -1)) != index
            or int(row.get("query_total", -1)) != expected_total
            or row.get("chemsys") != spec["chemsys"]
            or row.get("elements") != spec["elements"]
            or row.get("query_method")
            != "mp_api.client.MPRester.get_entries_in_chemsys"
            or row.get("thermo_type") != "GGA_GGA+U"
            or row.get("additional_criteria") != {"thermo_types": ["GGA_GGA+U"]}
            or row.get("compatible_only") is not True
            or row.get("local_compatibility_reprocessing") is not False
        ):
            raise ContractError(f"official fragment contract changed at index {index}")
        attempts = int(row.get("manual_transport_attempts", 0))
        if attempts < 1 or attempts > 5:
            raise ContractError(f"invalid transport-attempt count at index {index}")
        retries += attempts - 1
        status = row.get("query_status")
        if status == "resolved":
            slim = row.get("slim_entries")
            full = row.get("full_entries_mson")
            if (
                not isinstance(slim, list)
                or not slim
                or not isinstance(full, list)
                or not full
                or row.get("error") is not None
                or row.get("phase_diagram_constructed") is not True
                or sorted(row.get("unary_reference_elements") or [])
                != sorted(spec["elements"])
                or canonical_sha256(slim) != row.get("slim_entries_sha256")
                or canonical_sha256(full) != row.get("full_entries_mson_sha256")
            ):
                raise ContractError(f"invalid resolved fragment at index {index}")
            resolved.append(row)
        elif status == "query_error":
            error = row.get("error") or {}
            if (
                row.get("entry_count") is not None
                or row.get("slim_entries") is not None
                or row.get("full_entries_mson") is not None
                or error.get("type") != expected_error_type
                or error.get("message") != expected_error_message
                or error.get("http_status") is not None
                or required_symbol not in spec["elements"]
            ):
                raise ContractError(f"unapproved unresolved fragment at index {index}")
            unresolved.append(
                {
                    "query_index": index,
                    "chemsys": spec["chemsys"],
                    "elements": spec["elements"],
                    "reason": "official_gga_gga_u_missing_yb_unary_reference",
                    "source_error": error,
                }
            )
        else:
            raise ContractError(f"unexpected query status at index {index}")
        rows.append(row)

    if len(resolved) != int(adoption["expected_resolved_count"]):
        raise ContractError("resolved query count changed")
    if len(unresolved) != int(adoption["expected_unresolved_count"]):
        raise ContractError("unresolved query count changed")
    source_failures = read_json(failure_path)
    derived_failures = [
        {"query_index": row["query_index"], "chemsys": row["chemsys"], "error": row["error"]}
        for row in rows
        if row["query_status"] == "query_error"
    ]
    if (
        int(source_failures.get("count", -1)) != len(unresolved)
        or source_failures.get("failures") != derived_failures
    ):
        raise ContractError("source query-failure ledger disagrees with spool")

    final = run_root / "official_mp_cache"
    if final.exists():
        raise FileExistsError(final)
    preparing = run_root / f".official_mp_cache.preparing.{os.getpid()}"
    failed = run_root / f".official_mp_cache.FAILED.{os.getpid()}"
    preparing.mkdir(parents=True, exist_ok=False)
    try:
        full_path = preparing / "official_full_entries.jsonl.gz"
        audit_path = preparing / "query_audit.jsonl"
        slim_path = preparing / "official_slim_cache.jsonl"
        unknown_path = preparing / "unresolved_chemsys.jsonl"
        with gzip.open(full_path, "xt", encoding="utf-8", newline="\n", compresslevel=6) as full_handle, audit_path.open("x", encoding="utf-8", newline="\n") as audit_handle, slim_path.open("x", encoding="utf-8", newline="\n") as slim_handle, unknown_path.open("x", encoding="utf-8", newline="\n") as unknown_handle:
            unresolved_by_index = {int(row["query_index"]): row for row in unresolved}
            for row in rows:
                full_handle.write(canonical_json({key: value for key, value in row.items() if key != "slim_entries"}) + "\n")
                audit_handle.write(canonical_json({key: value for key, value in row.items() if key not in ("slim_entries", "full_entries_mson")}) + "\n")
                if row["query_status"] == "resolved":
                    slim_handle.write(canonical_json({"chemsys": row["chemsys"], "entries": row["slim_entries"]}) + "\n")
                else:
                    unknown_handle.write(canonical_json(unresolved_by_index[int(row["query_index"])]) + "\n")
            for handle in (audit_handle, slim_handle, unknown_handle):
                handle.flush()
                os.fsync(handle.fileno())

        manifest = {
            "schema": "h1_official_mp_gga_u_skip_unknown_cache_manifest_v2",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "query_status": "complete_with_explicit_hull_unknown",
            "query_count": expected_total,
            "resolved_query_count": len(resolved),
            "unresolved_query_count": len(unresolved),
            "unresolved_policy": "explicit_hull_unknown_excluded_from_skip_unknown_denominators",
            "query_method": "mp_api.client.MPRester.get_entries_in_chemsys",
            "compatible_only": True,
            "thermo_type": "GGA_GGA+U",
            "additional_criteria": {"thermo_types": ["GGA_GGA+U"]},
            "local_compatibility_reprocessing": False,
            "source_query_was_fresh_empty_cache": True,
            "historical_cache_rows_reused": 0,
            "august_cache_rows_reused": 0,
            "fresh_official_spool_rows_adopted": expected_total,
            "new_mp_queries": 0,
            "resolved_phase_diagrams_constructed": True,
            "unresolved_reason": "official GGA_GGA+U response lacks the Yb unary reference",
            "manual_transport_retries": retries,
            "inputs": {
                "wanted_chemsys": identity(inputs / "wanted_chemsys.jsonl"),
                "input_manifest": identity(inputs / "input_manifest.json"),
                "source_wanted_chemsys": identity(source_wanted),
                "source_query_failures": identity(failure_path),
            },
            "outputs": {
                "full_mson_snapshot": output_identity(full_path, final),
                "query_audit": output_identity(audit_path, final),
                "slim_evaluation_cache": output_identity(slim_path, final),
                "unresolved_chemsys": output_identity(unknown_path, final),
            },
        }
        write_json_exclusive(preparing / "completion_manifest.json", manifest)
        (preparing / "completion_SUCCESS").touch(exist_ok=False)
        preparing.rename(final)
    except Exception:
        if preparing.exists():
            shutil.move(str(preparing), str(failed))
        raise
    print(canonical_json({"cache_adoption": "complete", "resolved": len(resolved), "hull_unknown": len(unresolved), "new_mp_queries": 0}))


if __name__ == "__main__":
    main()
