#!/usr/bin/env python3
"""Validate and atomically adopt the login-node official cache superset."""

from __future__ import annotations

import argparse
from pathlib import Path

from protocol import (
    ContractError,
    identity,
    read_json,
    read_jsonl,
    require_source_manifest,
    sha256_file,
    write_json_exclusive,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()

    source = args.source_dir.resolve()
    require_source_manifest(source, args.source_manifest_sha256)
    config = read_json(source / "CONFIG.json")
    spec = config["official_mp"]
    prequery = config["prequery_all_reconstructed"]
    run_root = args.run_root.resolve()
    if not (run_root / "status/preliminary_assembly_SUCCESS").is_file():
        raise ContractError("preliminary assembly is incomplete")
    if not (run_root / "status/precompleted_official_cache_SUCCESS").is_file():
        raise ContractError("login-node official cache precompletion is incomplete")

    incoming = run_root / "precompleted_official_mp_cache"
    final = run_root / "official_mp_cache"
    if final.exists() or not (incoming / "completion_SUCCESS").is_file():
        raise ContractError("precompleted official cache state is invalid")
    manifest_path = incoming / "completion_manifest.json"
    slim_path = incoming / "official_slim_cache.jsonl"
    unresolved_path = incoming / "unresolved_chemsys.jsonl"
    manifest = read_json(manifest_path)
    if (
        manifest.get("schema")
        != "h1_a2_r03_official_incremental_cache_manifest_v1"
        or manifest.get("query_status")
        != "complete_with_explicit_hull_unknown"
        or manifest.get("query_method") != spec["query_method"]
        or manifest.get("compatible_only") is not True
        or manifest.get("thermo_type") != "GGA_GGA+U"
        or manifest.get("unresolved_policy") != spec["unresolved_policy"]
        or manifest.get("historical_or_august_polluted_cache_rows_reused") != 0
        or int(manifest.get("wanted_query_count", -1)) != int(prequery["count"])
        or sha256_file(slim_path)
        != manifest["outputs"]["slim_evaluation_cache"]["sha256"]
        or sha256_file(unresolved_path)
        != manifest["outputs"]["unresolved_chemsys"]["sha256"]
    ):
        raise ContractError("precompleted official cache contract changed")

    resolved_rows = read_jsonl(slim_path)
    unresolved_rows = read_jsonl(unresolved_path)
    resolved = {str(row["chemsys"]) for row in resolved_rows}
    unresolved = {str(row["chemsys"]) for row in unresolved_rows}
    wanted = {
        str(row["chemsys"])
        for row in read_jsonl(run_root / "inputs/wanted_chemsys.jsonl")
    }
    if (
        len(resolved) != len(resolved_rows)
        or len(unresolved) != len(unresolved_rows)
        or resolved & unresolved
        or wanted - resolved - unresolved
    ):
        raise ContractError("official precompleted cache does not cover final N+U systems")

    incoming.rename(final)
    report = {
        "schema": "h1a2_v8_official_cache_adoption_v1",
        "status": "complete",
        "source_manifest_sha256": args.source_manifest_sha256,
        "actual_wanted_count": len(wanted),
        "prequery_superset_count": int(prequery["count"]),
        "actual_covered_resolved_count": len(wanted & resolved),
        "actual_covered_unresolved_count": len(wanted & unresolved),
        "missing_after_adoption": 0,
        "unresolved_policy": spec["unresolved_policy"],
        "cache_manifest": identity(final / "completion_manifest.json"),
        "slim_cache": identity(final / "official_slim_cache.jsonl"),
        "unresolved": identity(final / "unresolved_chemsys.jsonl"),
    }
    write_json_exclusive(run_root / "status/official_cache_adoption_report.json", report)
    (run_root / "status/official_cache_adoption_SUCCESS").touch(exist_ok=False)
    (run_root / "status/official_cache_completion_SUCCESS").touch(exist_ok=False)
    print(
        {
            "official_cache_adoption": "PASS",
            "wanted": len(wanted),
            "resolved": len(wanted & resolved),
            "unresolved": len(wanted & unresolved),
        }
    )


if __name__ == "__main__":
    main()
