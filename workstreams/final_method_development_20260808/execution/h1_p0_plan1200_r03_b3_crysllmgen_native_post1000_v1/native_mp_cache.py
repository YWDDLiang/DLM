#!/usr/bin/env python3
"""Audit and complete the all-parse-success planner union for native1000."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import parallel_complete as parallel

from native_protocol import (
    NATIVE_DENOMINATOR,
    canonical_sha256,
    identity,
    ordered_candidate_rows,
    read_json,
    read_jsonl,
    sha256_file,
    validate_repeat,
    write_json_exclusive,
    write_jsonl_exclusive,
)
from protocol import require_file, validate_config


def candidate_union(run_root: Path) -> tuple[set[str], list[dict[str, Any]]]:
    wanted: set[str] = set()
    reports: list[dict[str, Any]] = []
    pool_hashes: list[str] = []
    for repeat in range(3):
        validate_repeat(repeat)
        root = run_root / "repeats" / str(repeat) / "crysllmgen_native_candidates"
        pool_path = root / "candidate_pool.jsonl"
        manifest_path = root / "candidate_pool_manifest.json"
        rows = ordered_candidate_rows(read_jsonl(pool_path))
        manifest = read_json(manifest_path)
        observed_sha = sha256_file(pool_path)
        recorded = (manifest.get("artifacts") or {}).get("candidate_pool") or {}
        if (
            not (root / "_SUCCESS").is_file()
            or manifest.get("status") != "complete"
            or int(manifest.get("repeat", -1)) != repeat
            or int(manifest.get("parse_successes", -1)) != len(rows)
            or int(manifest.get("v3_prefix_count", -1)) != NATIVE_DENOMINATOR
            or manifest.get("R03_B3_shared_candidate_pool") is not True
            or recorded.get("sha256") != observed_sha
        ):
            raise ValueError(f"native candidate pool {repeat} changed")
        systems: set[str] = set()
        for rank, row in enumerate(rows):
            state = row.get("plan_state")
            if not isinstance(state, Mapping):
                raise ValueError(f"repeat {repeat} candidate {rank} lacks plan_state")
            elements = state.get("elements")
            counts = state.get("counts")
            if (
                not isinstance(elements, list)
                or not elements
                or len(set(map(str, elements))) != len(elements)
                or not isinstance(counts, list)
                or len(counts) != len(elements)
                or any(int(value) <= 0 for value in counts)
            ):
                raise ValueError(f"repeat {repeat} candidate {rank} composition changed")
            systems.add("-".join(sorted(map(str, elements))))
        wanted.update(systems)
        pool_hashes.append(observed_sha)
        reports.append(
            {
                "repeat": repeat,
                "candidate_count": len(rows),
                "candidate_pool": identity(pool_path),
                "candidate_pool_manifest": identity(manifest_path),
                "distinct_chemsys": len(systems),
                "chemsys_sha256": parallel.line_set_sha256(systems),
            }
        )
    if len(set(pool_hashes)) != 3:
        raise ValueError("three native candidate pools are not byte-distinct")
    return wanted, reports


def load_current_cache(
    run_root: Path, module: Any, wanted: set[str]
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    manifest_path = run_root / "mp_cache/completion_manifest.json"
    manifest = read_json(manifest_path)
    specification = manifest.get("completed_mp_hull_cache") or {}
    path = Path(str(specification.get("path", ""))).resolve()
    if (
        manifest.get("status") != "complete_all_wanted_chemsys_resolved"
        or specification.get("all_rows_populated") is not True
        or not path.is_file()
        or sha256_file(path) != specification.get("sha256")
    ):
        raise ValueError("V3 completed MP cache changed")
    relevant, all_systems = module.load_relevant_slim_cache(path, wanted)
    cached = {system: entries for system, entries in relevant.items() if entries}
    return cached, {
        "role": "completed_v3_first1000_union",
        "artifact": identity(path),
        "completion_manifest": identity(manifest_path),
        "distinct_chemsys": len(all_systems),
        "wanted_rows_contributed": len(cached),
    }


def context(args: argparse.Namespace) -> tuple[dict[str, Any], Any, set[str], list[dict[str, Any]], dict[str, list[dict[str, Any]]], dict[str, Any]]:
    parallel.require_login_node()
    native_source = args.native_source_dir.resolve()
    body_source = args.body_source_dir.resolve()
    parallel.require_source_manifest(native_source, args.native_source_manifest_sha256)
    sys.path.insert(0, str(body_source))
    config = read_json(args.body_config.resolve())
    validate_config(config)
    run_root = Path(config["run_root"]).resolve()
    if args.run_root.resolve() != run_root:
        raise ValueError("run root changed")
    sun = config["sun"]
    completion_path = require_file(
        sun["completion_module"], sun["completion_module_sha256"], "completion module"
    )
    module = parallel.load_module("h1_native1000_mp_completion", completion_path)
    wanted, planner_reports = candidate_union(run_root)
    cached, source_report = load_current_cache(run_root, module, wanted)
    return config, module, wanted, planner_reports, cached, source_report


def audit(args: argparse.Namespace) -> dict[str, Any]:
    config, _, wanted, planner_reports, cached, source_report = context(args)
    missing = wanted - set(cached)
    report = {
        "schema": "h1_plan1200_crysllmgen_native_mp_cache_audit_v1",
        "status": "complete",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "run_id": config["run_id"],
        "native_source_manifest_sha256": args.native_source_manifest_sha256,
        "candidate_pools": planner_reports,
        "cache_source": source_report,
        "wanted_chemsys_count": len(wanted),
        "wanted_chemsys_sha256": parallel.line_set_sha256(wanted),
        "wanted_chemsys": sorted(wanted),
        "cached_chemsys_count": len(cached),
        "missing_chemsys_count": len(missing),
        "missing_chemsys_sha256": parallel.line_set_sha256(missing),
        "missing_chemsys": sorted(missing),
        "mp_query_performed": False,
        "slurm_used": False,
        "gpu_used": False,
        "api_key_serialized": False,
    }
    write_json_exclusive(args.output.resolve(), report)
    print(json.dumps(report, sort_keys=True))
    return report


def complete(args: argparse.Namespace) -> dict[str, Any]:
    config, module, wanted, planner_reports, cached, source_report = context(args)
    run_root = Path(config["run_root"]).resolve()
    audit_report = read_json(args.audit.resolve())
    missing = sorted(wanted - set(cached))
    if (
        audit_report.get("status") != "complete"
        or audit_report.get("run_id") != config["run_id"]
        or audit_report.get("native_source_manifest_sha256")
        != args.native_source_manifest_sha256
        or int(audit_report.get("wanted_chemsys_count", -1)) != len(wanted)
        or audit_report.get("wanted_chemsys_sha256")
        != parallel.line_set_sha256(wanted)
        or int(audit_report.get("missing_chemsys_count", -1)) != len(missing)
        or audit_report.get("missing_chemsys_sha256")
        != parallel.line_set_sha256(missing)
        or audit_report.get("mp_query_performed") is not False
    ):
        raise ValueError("native cache audit contract changed")
    final_root = run_root / "native_mp_cache"
    if final_root.exists():
        raise FileExistsError(final_root)
    lock_path = run_root / "status/native_mp_parallel_completion.lock"
    write_json_exclusive(
        lock_path,
        {
            "schema": "h1_plan1200_native_mp_parallel_completion_lock_v1",
            "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "pid": os.getpid(),
            "workers": parallel.WORKERS,
            "max_requests_per_second": parallel.MAX_REQUESTS_PER_SECOND,
            "missing_count": len(missing),
            "native_source_manifest_sha256": args.native_source_manifest_sha256,
            "api_key_serialized": False,
        },
    )
    preparing = run_root / f".native_mp_cache.preparing.{os.getpid()}"
    failed = run_root / f".native_mp_cache.FAILED.{os.getpid()}"
    preparing.mkdir()
    spool = preparing / "spool"
    spool.mkdir()
    key = ""
    try:
        queried: dict[str, list[dict[str, Any]]] = {}
        progress: list[dict[str, Any]] = []
        if missing:
            if args.key_file is None or not args.key_file.is_file():
                raise RuntimeError(
                    f"{len(missing)} native candidate chemsys remain; one-time key required"
                )
            key = module.read_and_destroy_api_key(args.key_file.resolve())
            if args.key_file.exists():
                raise RuntimeError("one-time native key carrier was not destroyed")
            queried, progress = parallel.query_remaining(
                module=module,
                api_key=key,
                remaining=list(enumerate(missing, start=1)),
                total=len(missing),
                maximum_attempts=int(config["sun"]["maximum_transport_attempts_per_chemsys"]),
                spool=spool,
            )
            key = ""
            if (
                set(queried) != set(missing)
                or len(progress) != len(missing)
                or any(row.get("status") != "resolved" for row in progress)
            ):
                raise RuntimeError("not every native missing chemsys resolved")
        elif args.key_file is not None:
            raise ValueError("key file supplied although native cache has no missing systems")

        cached.update(queried)
        if set(cached) != wanted or any(not cached[system] for system in wanted):
            raise RuntimeError("native completed cache is not total")
        fragment = preparing / "mp_query_fragment.jsonl"
        progress_path = preparing / "mp_query_progress.jsonl"
        write_jsonl_exclusive(
            fragment,
            ({"chemsys": system, "entries": queried[system]} for system in missing),
        )
        write_jsonl_exclusive(progress_path, progress)
        completed = preparing / "completed_mp_hull_cache.jsonl"
        write_jsonl_exclusive(
            completed,
            ({"chemsys": system, "entries": cached[system]} for system in sorted(wanted)),
        )
        completed_sha = sha256_file(completed)
        parallel.write_sha(
            preparing / "completed_mp_hull_cache.sha256",
            completed_sha,
            completed.name,
        )
        status_counts = Counter(str(row["status"]) for row in progress)
        manifest = {
            "schema": "h1_plan1200_crysllmgen_native_mp_cache_completion_v1",
            "status": "complete_all_wanted_chemsys_resolved",
            "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "run_id": config["run_id"],
            "native_source_manifest_sha256": args.native_source_manifest_sha256,
            "candidate_pools": planner_reports,
            "cache_source": source_report,
            "audit": identity(args.audit),
            "wanted_chemsys_count": len(wanted),
            "wanted_chemsys_sha256": parallel.line_set_sha256(wanted),
            "base_cache_rows_reused": len(wanted) - len(missing),
            "missing_chemsys_count": len(missing),
            "missing_chemsys_sha256": parallel.line_set_sha256(missing),
            "query_status_counts": dict(sorted(status_counts.items())),
            "logical_queries": len(progress),
            "transport_retries": sum(
                int(row.get("transport_retries", 0)) for row in progress
            ),
            "workers": parallel.WORKERS,
            "max_requests_per_second": parallel.MAX_REQUESTS_PER_SECOND,
            "completed_mp_hull_cache": {
                "path": str(final_root / "completed_mp_hull_cache.jsonl"),
                "bytes": completed.stat().st_size,
                "sha256": completed_sha,
                "rows": len(wanted),
                "all_rows_populated": True,
            },
            "query_fragment": parallel.relocated_identity(
                fragment, final_root / "mp_query_fragment.jsonl"
            ),
            "query_progress": parallel.relocated_identity(
                progress_path, final_root / "mp_query_progress.jsonl"
            ),
            "execution_location": "A800_login_node",
            "slurm_used": False,
            "gpu_used": False,
            "api_key_serialized": False,
            "credential_environment_used": False,
            "one_time_key_carrier_destroyed": bool(missing),
            "mp_query_inside_slurm": False,
            "sample_retry_or_replacement_used": False,
            "transport_retry_is_not_sample_retry": True,
            "automatic_training": False,
            "automatic_rl": False,
        }
        write_json_exclusive(preparing / "completion_manifest.json", manifest)
        with (preparing / "completion_SUCCESS").open("x", encoding="ascii") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        preparing.rename(final_root)
    except Exception:
        key = ""
        if preparing.exists():
            preparing.rename(failed)
        raise
    print(json.dumps(manifest, sort_keys=True))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("audit", "complete"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--body-config", type=Path, required=True)
        sub.add_argument("--body-source-dir", type=Path, required=True)
        sub.add_argument("--native-source-dir", type=Path, required=True)
        sub.add_argument("--native-source-manifest-sha256", required=True)
        sub.add_argument("--run-root", type=Path, required=True)
        if command == "audit":
            sub.add_argument("--output", type=Path, required=True)
        else:
            sub.add_argument("--audit", type=Path, required=True)
            sub.add_argument("--key-file", type=Path)
    args = parser.parse_args()
    try:
        audit(args) if args.command == "audit" else complete(args)
    except Exception as exc:  # noqa: BLE001
        print(
            json.dumps(
                {
                    "schema": "h1_plan1200_crysllmgen_native_mp_cache_failure_v1",
                    "status": "failed_closed",
                    "command": args.command,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "api_key_serialized": False,
                    "automatic_retry": False,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
