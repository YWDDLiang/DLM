#!/usr/bin/env python3
"""Complete only missing Planner-union MP hull cache rows before Slurm."""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import os
import sys
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping

from protocol import (
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


def _line_set_sha256(values: set[str]) -> str:
    payload = "" if not values else "\n".join(sorted(values)) + "\n"
    return __import__("hashlib").sha256(payload.encode("utf-8")).hexdigest()


def _identity(path: Path) -> dict[str, Any]:
    location = path.resolve()
    if not location.is_file():
        raise FileNotFoundError(location)
    return {
        "path": str(location),
        "bytes": location.stat().st_size,
        "sha256": sha256_file(location),
    }


def _load_frozen_completion_module(path: Path) -> ModuleType:
    specification = importlib.util.spec_from_file_location(
        "h1_r03f_frozen_mp_completion", path
    )
    if specification is None or specification.loader is None:
        raise ImportError(f"cannot load frozen MP completion module: {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def _planner_chemsys(
    config: Mapping[str, Any],
) -> tuple[set[str], dict[str, dict[str, Any]]]:
    union: set[str] = set()
    reports: dict[str, dict[str, Any]] = {}
    for label, specification in config["planner_sources"].items():
        path = require_file(
            specification["raw_generations"],
            specification["raw_generations_sha256"],
            f"{label} raw256",
        )
        rows = ordered_rows(read_jsonl(path), ordinal_field="sample_idx")
        systems: set[str] = set()
        parsed = 0
        for row in rows:
            if row.get("parsed") is not True:
                continue
            parsed += 1
            state = row.get("plan_state") or row.get("parsed_plan")
            if not isinstance(state, dict):
                raise ValueError(f"{label} parsed row lacks plan state")
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
                raise ValueError(f"{label} parsed composition contract changed")
            systems.add("-".join(sorted(map(str, elements))))
        if parsed != int(specification["expected_parsed"]):
            raise ValueError(f"{label} parsed count changed")
        union.update(systems)
        reports[label] = {
            "raw_generations": _identity(path),
            "parsed": parsed,
            "distinct_chemsys": len(systems),
            "chemsys_sha256": _line_set_sha256(systems),
        }
    return union, reports


def _write_sha256_file(path: Path, digest: str, filename: str) -> None:
    with path.open("x", encoding="ascii", newline="\n") as handle:
        handle.write(f"{digest}  {filename}\n")
        handle.flush()
        os.fsync(handle.fileno())


def complete(args: argparse.Namespace) -> dict[str, Any]:
    if os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOB_NAME"):
        raise RuntimeError("MP cache completion is A800 login-node-only")
    if any(
        os.environ.get(name)
        for name in ("MP_API_KEY", "PMG_MAPI_KEY", "MAPI_KEY")
    ):
        raise RuntimeError("MP credentials must arrive only through the key file")

    source = args.source_dir.resolve()
    require_source_manifest(source, args.source_manifest_sha256)
    config = read_json(args.config.resolve())
    validate_config(config)

    prepared_root = args.prepared_root.resolve()
    final_root = Path(config["run_root"]).resolve()
    if (
        prepared_root.parent != final_root.parent
        or not prepared_root.name.startswith(f".{final_root.name}.preparing.")
        or final_root.exists()
    ):
        raise ValueError("prepared/final run identity changed")

    r03f = config["source_dependencies"]["r03f_mpcomplete"]
    require_source_manifest(
        r03f["source_dir"], r03f["source_manifest_sha256"]
    )
    sun = config["sun"]
    completion_module_path = require_file(
        sun["completion_module"],
        sun["completion_module_sha256"],
        "frozen R03F MP completion module",
    )
    completion_module = _load_frozen_completion_module(
        completion_module_path
    )

    project_root = final_root.parents[1]
    base_cache = require_file(
        project_root / sun["base_mp_hull_cache"],
        sun["base_mp_hull_cache_sha256"],
        "frozen base MP hull cache",
    )
    if base_cache.stat().st_size != int(sun["base_mp_hull_cache_bytes"]):
        raise ValueError("frozen base MP hull cache byte count changed")

    wanted, planner_reports = _planner_chemsys(config)
    if (
        len(wanted) != int(sun["wanted_chemsys_count"])
        or _line_set_sha256(wanted) != sun["wanted_chemsys_sha256"]
    ):
        raise ValueError("Planner-union chemsys contract changed")
    cached, all_cached_chemsys = completion_module.load_relevant_slim_cache(
        base_cache, wanted
    )
    if len(all_cached_chemsys) != int(sun["base_mp_hull_cache_rows"]):
        raise ValueError("frozen base MP hull cache row identity changed")
    missing = {system for system in wanted if not cached.get(system)}
    if (
        len(missing) != int(sun["missing_chemsys_count"])
        or _line_set_sha256(missing) != sun["missing_chemsys_sha256"]
    ):
        raise ValueError("Planner-union MP cache gap changed")

    mp_root = prepared_root / "mp_cache"
    mp_root.mkdir()
    claim = {
        "schema": "h1_ef_fourcell_mp_cache_claim_v1",
        "status": "claimed_before_external_query",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "run_id": config["run_id"],
        "wanted_chemsys_count": len(wanted),
        "wanted_chemsys_sha256": _line_set_sha256(wanted),
        "missing_chemsys_count": len(missing),
        "missing_chemsys_sha256": _line_set_sha256(missing),
        "api_key_present": True,
        "api_key_serialized": False,
        "slurm_used": False,
        "gpu_used": False,
        "sample_retry_or_replacement_used": False,
        "automatic_training": False,
        "automatic_downstream": False,
        "automatic_rl": False,
    }
    write_json_exclusive(mp_root / "claim.json", claim)

    key = completion_module.read_and_destroy_api_key(args.key_file)
    try:
        queried, progress = completion_module.query_missing_chemsys(
            api_key=key,
            missing=sorted(missing),
            completed_cache_path=mp_root / "mp_query_fragment.jsonl",
            progress_path=mp_root / "mp_query_progress.jsonl",
            maximum_attempts=int(
                sun["maximum_transport_attempts_per_chemsys"]
            ),
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

    cached.update(queried)
    if set(cached) != wanted or any(not cached[system] for system in wanted):
        raise RuntimeError("completed Planner-union cache is not total")
    completed_cache = mp_root / "planner_union_completed_mp_hull_cache.jsonl"
    write_jsonl_exclusive(
        completed_cache,
        (
            {"chemsys": system, "entries": cached[system]}
            for system in sorted(wanted)
        ),
    )
    completed_sha = sha256_file(completed_cache)
    _write_sha256_file(
        mp_root / "planner_union_completed_mp_hull_cache.sha256",
        completed_sha,
        completed_cache.name,
    )

    status_counts = Counter(str(row["status"]) for row in progress)
    manifest = {
        "schema": "h1_ef_fourcell_mp_cache_completion_manifest_v1",
        "status": "complete_all_missing_resolved",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "run_id": config["run_id"],
        "source_manifest_sha256": args.source_manifest_sha256,
        "planner_sources": planner_reports,
        "base_mp_hull_cache": _identity(base_cache),
        "frozen_completion_module": _identity(completion_module_path),
        "wanted_chemsys_count": len(wanted),
        "wanted_chemsys_sha256": _line_set_sha256(wanted),
        "base_cache_populated_for_wanted": len(wanted) - len(missing),
        "missing_chemsys_count": len(missing),
        "missing_chemsys_sha256": _line_set_sha256(missing),
        "query_status_counts": dict(sorted(status_counts.items())),
        "logical_queries": len(progress),
        "transport_retries": sum(
            int(row["transport_retries"]) for row in progress
        ),
        "completed_mp_hull_cache": {
            "path": sun["completed_mp_hull_cache"],
            "bytes": completed_cache.stat().st_size,
            "sha256": completed_sha,
            "rows": len(wanted),
            "all_rows_populated": True,
        },
        "query_fragment": {
            "path": "mp_cache/mp_query_fragment.jsonl",
            "bytes": (mp_root / "mp_query_fragment.jsonl").stat().st_size,
            "sha256": sha256_file(mp_root / "mp_query_fragment.jsonl"),
        },
        "query_progress": {
            "path": "mp_cache/mp_query_progress.jsonl",
            "bytes": (mp_root / "mp_query_progress.jsonl").stat().st_size,
            "sha256": sha256_file(mp_root / "mp_query_progress.jsonl"),
        },
        "execution_location": "A800_login_node",
        "slurm_used": False,
        "gpu_used": False,
        "api_key_serialized": False,
        "credential_environment_used": False,
        "sample_retry_or_replacement_used": False,
        "transport_retry_is_not_sample_retry": True,
        "automatic_training": False,
        "automatic_downstream": False,
        "automatic_rl": False,
        "contract_sha256": canonical_sha256(
            {
                "wanted": _line_set_sha256(wanted),
                "missing": _line_set_sha256(missing),
                "base_cache": sun["base_mp_hull_cache_sha256"],
                "completion_module": sun["completion_module_sha256"],
            }
        ),
    }
    write_json_exclusive(mp_root / "completion_manifest.json", manifest)
    with (mp_root / "completion_SUCCESS").open("x", encoding="ascii") as handle:
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps(manifest, sort_keys=True))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--key-file", type=Path, required=True)
    args = parser.parse_args()
    try:
        complete(args)
    except Exception as exc:
        print(
            json.dumps(
                {
                    "schema": "h1_ef_fourcell_mp_cache_failure_v1",
                    "status": "failed_closed",
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
