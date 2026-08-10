#!/usr/bin/env python3
"""Audit and, when authorized, complete the three-cohort MP hull cache."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
import sys
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Mapping

from protocol import (
    DENOMINATOR,
    canonical_sha256,
    ordered_rows,
    read_json,
    read_jsonl,
    require_file,
    require_source_manifest,
    sha256_file,
    validate_config,
    write_json_exclusive,
    write_jsonl_exclusive,
)


def line_set_sha256(values: Iterable[str]) -> str:
    ordered = sorted(set(values))
    payload = "" if not ordered else "\n".join(ordered) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def identity(path: Path) -> dict[str, Any]:
    location = path.resolve()
    if not location.is_file():
        raise FileNotFoundError(location)
    return {
        "path": str(location),
        "bytes": location.stat().st_size,
        "sha256": sha256_file(location),
    }


def relocated_identity(source: Path, destination: Path) -> dict[str, Any]:
    """Hash a staged file while recording its immutable post-rename location."""

    location = source.resolve()
    if not location.is_file():
        raise FileNotFoundError(location)
    return {
        "path": str(destination.resolve()),
        "bytes": location.stat().st_size,
        "sha256": sha256_file(location),
    }


def load_module(path: Path) -> ModuleType:
    specification = importlib.util.spec_from_file_location(
        "h1_plan1200_frozen_mp_completion", path
    )
    if specification is None or specification.loader is None:
        raise ImportError(f"cannot load frozen completion module: {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def require_login_node() -> None:
    if os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOB_NAME"):
        raise RuntimeError("MP cache audit/completion is login-node-only")
    if any(os.environ.get(name) for name in ("MP_API_KEY", "PMG_MAPI_KEY", "MAPI_KEY")):
        raise RuntimeError("MP credentials must arrive only through a one-time key file")


def planner_union(run_root: Path) -> tuple[set[str], list[dict[str, Any]]]:
    terminal = read_json(run_root / "planner_terminal_report.json")
    if (
        terminal.get("status") != "complete"
        or terminal.get("three_independent_plan_batches") is not True
        or terminal.get("shared_between_R03_and_B3_within_repeat") is not True
        or int(terminal.get("repeat_count", -1)) != 3
        or not (run_root / "status/planner_assembly_SUCCESS").is_file()
    ):
        raise ValueError("planner terminal evidence is not complete")

    wanted: set[str] = set()
    reports: list[dict[str, Any]] = []
    cohort_hashes: list[str] = []
    for repeat in range(3):
        cohort_root = run_root / "repeats" / str(repeat) / "cohort"
        cohort_path = cohort_root / "cohort1000.jsonl"
        manifest_path = cohort_root / "cohort_manifest.json"
        manifest = read_json(manifest_path)
        rows = ordered_rows(read_jsonl(cohort_path), ordinal_field="cohort_ordinal")
        recorded = (manifest.get("artifacts") or {}).get("cohort1000") or {}
        observed_sha = sha256_file(cohort_path)
        if (
            not (cohort_root / "_SUCCESS").is_file()
            or manifest.get("status") != "complete"
            or int(manifest.get("repeat", -1)) != repeat
            or int(manifest.get("raw_attempts", -1)) != 1200
            or int(manifest.get("selected_attempts", -1)) != DENOMINATOR
            or manifest.get("shared_between_R03_and_B3") is not True
            or manifest.get("arm_outcome_dependent_replacement") is not False
            or recorded.get("sha256") != observed_sha
        ):
            raise ValueError(f"planner cohort {repeat} contract changed")
        systems: set[str] = set()
        for ordinal, row in enumerate(rows):
            state = row.get("plan_state")
            if not isinstance(state, Mapping):
                raise ValueError(f"repeat {repeat} ordinal {ordinal} lacks plan_state")
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
                raise ValueError(f"repeat {repeat} ordinal {ordinal} composition changed")
            systems.add("-".join(sorted(map(str, elements))))
        wanted.update(systems)
        cohort_hashes.append(observed_sha)
        reports.append(
            {
                "repeat": repeat,
                "cohort1000": identity(cohort_path),
                "cohort_manifest": identity(manifest_path),
                "distinct_chemsys": len(systems),
                "chemsys_sha256": line_set_sha256(systems),
            }
        )
    if len(set(cohort_hashes)) != 3:
        raise ValueError("three planner cohorts are not byte-distinct")
    return wanted, reports


def cache_sources(
    config: Mapping[str, Any], module: ModuleType, wanted: set[str]
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    sun = config["sun"]
    run_root = Path(config["run_root"]).resolve()
    project = run_root.parents[1]
    specifications = [
        {
            "role": "frozen_base",
            "path": str(project / sun["base_mp_hull_cache"]),
            "sha256": sun["base_mp_hull_cache_sha256"],
            "bytes": int(sun["base_mp_hull_cache_bytes"]),
            "rows": int(sun["base_mp_hull_cache_rows"]),
        },
        *list(sun.get("supplemental_completed_caches") or []),
    ]
    merged: dict[str, list[dict[str, Any]]] = {}
    reports: list[dict[str, Any]] = []
    for specification in specifications:
        path = require_file(
            specification["path"], specification["sha256"], specification["role"]
        )
        if "bytes" in specification and path.stat().st_size != int(specification["bytes"]):
            raise ValueError(f"{specification['role']} cache byte count changed")
        if "rows" in specification:
            physical_rows = sum(1 for line in path.open(encoding="utf-8") if line.strip())
            if physical_rows != int(specification["rows"]):
                raise ValueError(f"{specification['role']} cache row count changed")
        relevant, all_systems = module.load_relevant_slim_cache(path, wanted)
        contributed = 0
        for system, entries in relevant.items():
            if not entries:
                continue
            if system in merged and canonical_sha256(merged[system]) != canonical_sha256(entries):
                raise ValueError(f"cache sources disagree for {system}")
            if system not in merged:
                contributed += 1
                merged[system] = entries
        reports.append(
            {
                "role": specification["role"],
                "artifact": identity(path),
                "distinct_chemsys": len(all_systems),
                "wanted_rows_contributed": contributed,
            }
        )
    return merged, reports


def audit(args: argparse.Namespace) -> dict[str, Any]:
    require_login_node()
    source = args.source_dir.resolve()
    require_source_manifest(source, args.source_manifest_sha256)
    config = read_json(args.config.resolve())
    validate_config(config)
    run_root = Path(config["run_root"]).resolve()
    if args.run_root.resolve() != run_root:
        raise ValueError("run root changed")
    sun = config["sun"]
    completion_path = require_file(
        sun["completion_module"], sun["completion_module_sha256"], "completion module"
    )
    module = load_module(completion_path)
    wanted, planner_reports = planner_union(run_root)
    cached, source_reports = cache_sources(config, module, wanted)
    missing = wanted - set(cached)
    report = {
        "schema": "h1_plan1200_mp_cache_audit_v3",
        "status": "complete",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "run_id": config["run_id"],
        "source_manifest_sha256": args.source_manifest_sha256,
        "planner_repeats": planner_reports,
        "cache_sources": source_reports,
        "wanted_chemsys_count": len(wanted),
        "wanted_chemsys_sha256": line_set_sha256(wanted),
        "wanted_chemsys": sorted(wanted),
        "cached_chemsys_count": len(cached),
        "missing_chemsys_count": len(missing),
        "missing_chemsys_sha256": line_set_sha256(missing),
        "missing_chemsys": sorted(missing),
        "mp_query_performed": False,
        "slurm_used": False,
        "gpu_used": False,
        "api_key_serialized": False,
        "retry_replacement_repair_filter_rerank": False,
    }
    write_json_exclusive(args.output.resolve(), report)
    print(json.dumps(report, sort_keys=True))
    return report


def write_sha(path: Path, digest: str, filename: str) -> None:
    with path.open("x", encoding="ascii", newline="\n") as handle:
        handle.write(f"{digest}  {filename}\n")
        handle.flush()
        os.fsync(handle.fileno())


def complete(args: argparse.Namespace) -> dict[str, Any]:
    require_login_node()
    source = args.source_dir.resolve()
    require_source_manifest(source, args.source_manifest_sha256)
    config = read_json(args.config.resolve())
    validate_config(config)
    run_root = Path(config["run_root"]).resolve()
    if args.run_root.resolve() != run_root:
        raise ValueError("run root changed")
    audit_report = read_json(args.audit.resolve())
    if (
        audit_report.get("status") != "complete"
        or audit_report.get("run_id") != config["run_id"]
        or audit_report.get("source_manifest_sha256") != args.source_manifest_sha256
        or audit_report.get("mp_query_performed") is not False
    ):
        raise ValueError("cache audit contract changed")
    wanted, planner_reports = planner_union(run_root)
    if (
        int(audit_report.get("wanted_chemsys_count", -1)) != len(wanted)
        or audit_report.get("wanted_chemsys_sha256") != line_set_sha256(wanted)
    ):
        raise ValueError("cache audit wanted set changed")

    sun = config["sun"]
    completion_path = require_file(
        sun["completion_module"], sun["completion_module_sha256"], "completion module"
    )
    module = load_module(completion_path)
    cached, source_reports = cache_sources(config, module, wanted)
    missing = wanted - set(cached)
    if (
        int(audit_report.get("missing_chemsys_count", -1)) != len(missing)
        or audit_report.get("missing_chemsys_sha256") != line_set_sha256(missing)
    ):
        raise ValueError("cache audit missing set changed")
    if (run_root / "mp_cache").exists():
        raise FileExistsError(run_root / "mp_cache")
    mp_root = run_root / f".mp_cache.preparing.{os.getpid()}"
    mp_root.mkdir()
    try:
        queried: dict[str, list[dict[str, Any]]] = {}
        progress: list[dict[str, Any]] = []
        fragment = mp_root / "mp_query_fragment.jsonl"
        progress_path = mp_root / "mp_query_progress.jsonl"
        if missing:
            if args.key_file is None or not args.key_file.is_file():
                raise RuntimeError(
                    f"{len(missing)} MP chemsys remain; a one-time key file is required"
                )
            key = module.read_and_destroy_api_key(args.key_file.resolve())
            try:
                queried, progress = module.query_missing_chemsys(
                    api_key=key,
                    missing=sorted(missing),
                    completed_cache_path=fragment,
                    progress_path=progress_path,
                    maximum_attempts=int(sun["maximum_transport_attempts_per_chemsys"]),
                )
            finally:
                key = ""
            if (
                set(queried) != missing
                or len(progress) != len(missing)
                or any(row.get("status") != "resolved" for row in progress)
                or any(not queried.get(system) for system in missing)
            ):
                raise RuntimeError("not every frozen missing chemsys resolved")
        else:
            write_jsonl_exclusive(fragment, [])
            write_jsonl_exclusive(progress_path, [])

        cached.update(queried)
        if set(cached) != wanted or any(not cached[system] for system in wanted):
            raise RuntimeError("completed planner-union cache is not total")
        completed = mp_root / "completed_mp_hull_cache.jsonl"
        write_jsonl_exclusive(
            completed,
            ({"chemsys": system, "entries": cached[system]} for system in sorted(wanted)),
        )
        completed_sha = sha256_file(completed)
        write_sha(mp_root / "completed_mp_hull_cache.sha256", completed_sha, completed.name)
        status_counts = Counter(str(row["status"]) for row in progress)
        final_cache = run_root / "mp_cache/completed_mp_hull_cache.jsonl"
        manifest = {
            "schema": "h1_plan1200_mp_cache_completion_manifest_v3",
            "status": "complete_all_wanted_chemsys_resolved",
            "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "run_id": config["run_id"],
            "source_manifest_sha256": args.source_manifest_sha256,
            "planner_repeats": planner_reports,
            "cache_sources": source_reports,
            "audit": identity(args.audit.resolve()),
            "frozen_completion_module": identity(completion_path),
            "wanted_chemsys_count": len(wanted),
            "wanted_chemsys_sha256": line_set_sha256(wanted),
            "base_and_supplemental_populated": len(wanted) - len(missing),
            "missing_chemsys_count": len(missing),
            "missing_chemsys_sha256": line_set_sha256(missing),
            "query_status_counts": dict(sorted(status_counts.items())),
            "logical_queries": len(progress),
            "transport_retries": sum(int(row.get("transport_retries", 0)) for row in progress),
            "completed_mp_hull_cache": {
                "path": str(final_cache),
                "bytes": completed.stat().st_size,
                "sha256": completed_sha,
                "rows": len(wanted),
                "all_rows_populated": True,
            },
            "query_fragment": relocated_identity(
                fragment, run_root / "mp_cache/mp_query_fragment.jsonl"
            ),
            "query_progress": relocated_identity(
                progress_path, run_root / "mp_cache/mp_query_progress.jsonl"
            ),
            "execution_location": "A800_login_node",
            "slurm_used": False,
            "gpu_used": False,
            "api_key_serialized": False,
            "credential_environment_used": False,
            "mp_query_inside_slurm": False,
            "sample_retry_or_replacement_used": False,
            "transport_retry_is_not_sample_retry": True,
            "automatic_training": False,
            "automatic_promotion": False,
            "automatic_rl": False,
        }
        write_json_exclusive(mp_root / "completion_manifest.json", manifest)
        with (mp_root / "completion_SUCCESS").open("x", encoding="ascii") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        mp_root.rename(run_root / "mp_cache")
    except Exception:
        failed = run_root / f".mp_cache.FAILED.{os.getpid()}"
        if mp_root.exists():
            mp_root.rename(failed)
        raise
    print(json.dumps(manifest, sort_keys=True))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("audit", "complete"):
        sub = subparsers.add_parser(name)
        sub.add_argument("--config", type=Path, required=True)
        sub.add_argument("--source-dir", type=Path, required=True)
        sub.add_argument("--source-manifest-sha256", required=True)
        sub.add_argument("--run-root", type=Path, required=True)
        if name == "audit":
            sub.add_argument("--output", type=Path, required=True)
        else:
            sub.add_argument("--audit", type=Path, required=True)
            sub.add_argument("--key-file", type=Path)
    args = parser.parse_args()
    try:
        audit(args) if args.command == "audit" else complete(args)
    except Exception as exc:  # noqa: BLE001 - preserve fail-closed evidence.
        print(
            json.dumps(
                {
                    "schema": "h1_plan1200_mp_cache_failure_v3",
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
