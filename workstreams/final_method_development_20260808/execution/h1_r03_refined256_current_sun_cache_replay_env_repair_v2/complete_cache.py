#!/usr/bin/env python3
"""Build the cohort-complete MP cache in the repaired diff_meets_diff environment."""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import importlib.util
import json
import os
import shutil
import sys
import threading
import time
from collections import Counter, deque
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping

from protocol import (
    ContractError,
    historical_paths,
    identity,
    line_set_sha256,
    read_json,
    read_jsonl,
    repeat_spec,
    require_file,
    require_source_manifest,
    sha256_file,
    write_json_exclusive,
    write_jsonl_exclusive,
)


class SlidingWindowLimiter:
    def __init__(self, maximum: int) -> None:
        self.maximum = maximum
        self.events: deque[float] = deque()
        self.lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self.lock:
                now = time.monotonic()
                while self.events and self.events[0] <= now - 1.0:
                    self.events.popleft()
                if len(self.events) < self.maximum:
                    self.events.append(now)
                    return
                wait = max(0.001, 1.001 - (now - self.events[0]))
            time.sleep(wait)


def load_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("h1_current_mp_completion", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def retry_after_seconds(exc: BaseException) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    raw = None if headers is None else headers.get("Retry-After")
    try:
        return None if raw is None else max(0.0, min(300.0, float(raw)))
    except (TypeError, ValueError):
        return None


def wanted_chemsys(config: Mapping[str, Any]) -> set[str]:
    wanted: set[str] = set()
    for repeat in range(4):
        generation, relax_cache, old_attempts = historical_paths(config, repeat)
        spec = repeat_spec(config, repeat)
        require_file(generation, str(spec["generation_sha256"]), f"repeat {repeat} generation")
        require_file(relax_cache, str(spec["frozen_relax_cache_sha256"]), f"repeat {repeat} relax cache")
        require_file(old_attempts, str(spec["old_attempt_results_sha256"]), f"repeat {repeat} old results")
        generation_rows = read_jsonl(generation)
        if len(generation_rows) != 256 or sum(row.get("status") == "succeeded" for row in generation_rows) != 248:
            raise ContractError(f"repeat {repeat} frozen refined256 contract changed")
        ordinals = [int(row.get("ordinal", -1)) for row in generation_rows]
        if ordinals != list(range(256)) or any(bool(row.get("retry_or_replacement_used")) for row in generation_rows):
            raise ContractError(f"repeat {repeat} generation ledger changed")
        for row in read_jsonl(old_attempts):
            hull = row.get("hull_recompute") or {}
            if hull.get("applicable"):
                wanted.add(str(hull["chemsys"]))
    expected = int(config["current_mp"]["wanted_chemsys_count"])
    if len(wanted) != expected:
        raise ContractError(f"historical wanted chemsys changed: {len(wanted)} != {expected}")
    return wanted


def load_existing(config: Mapping[str, Any], wanted: set[str]) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    merged: dict[str, list[dict[str, Any]]] = {}
    reports: list[dict[str, Any]] = []
    for source in config["current_mp"]["sources_oldest_to_newest"]:
        path = require_file(Path(source["path"]), str(source["sha256"]), str(source["role"]))
        rows = read_jsonl(path)
        source_map = {
            str(row["chemsys"]): row.get("entries")
            for row in rows
            if str(row.get("chemsys", "")) in wanted and isinstance(row.get("entries"), list) and row["entries"]
        }
        overwritten = sum(system in merged for system in source_map)
        merged.update(source_map)
        reports.append(
            {
                "role": source["role"],
                "artifact": identity(path),
                "wanted_resolved": len(source_map),
                "new_contribution": len(source_map) - overwritten,
                "overwrote_older_snapshot": overwritten,
            }
        )
    return merged, reports


def query_missing(
    *, module: ModuleType, api_key: str, missing: list[str], workers: int,
    maximum_rps: int, maximum_attempts: int, spool: Path,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    limiter = SlidingWindowLimiter(maximum_rps)
    local = threading.local()
    clients: list[Any] = []
    clients_lock = threading.Lock()

    def get_client() -> Any:
        client = getattr(local, "client", None)
        if client is None:
            client = module.CurrentMPThermoClient(api_key)
            original_get = client.session.get

            def limited_get(*args: Any, **kwargs: Any) -> Any:
                limiter.acquire()
                return original_get(*args, **kwargs)

            client.session.get = limited_get
            local.client = client
            with clients_lock:
                clients.append(client)
        return client

    def one(item: tuple[int, str]) -> tuple[int, dict[str, Any], dict[str, Any]]:
        query_index, chemsys = item
        started = time.monotonic()
        entries = None
        response_audit = None
        final_error = None
        attempts_used = 0
        client = get_client()
        for transport_attempt in range(1, maximum_attempts + 1):
            attempts_used = transport_attempt
            try:
                raw_entries, response_audit = client.get_entries_in_chemsys(chemsys)
                entries = module.slim_entries(raw_entries)
                final_error = None
                break
            except Exception as exc:  # noqa: BLE001
                if isinstance(exc, module.CompletionError):
                    raise ContractError(f"MP response contract failed at query {query_index}") from None
                final_error = module.sanitized_query_error(exc)
                if final_error.get("http_status") in {401, 403}:
                    raise ContractError("MP authorization rejected") from None
                if transport_attempt < maximum_attempts:
                    backoff = min(16.0, float(2 ** (transport_attempt - 1)))
                    time.sleep(max(backoff, retry_after_seconds(exc) or 0.0))
        if not isinstance(entries, list) or not entries:
            raise ContractError(f"MP query {query_index}/{len(missing)} did not resolve")
        fragment = {"chemsys": chemsys, "entries": entries}
        progress = {
            "schema": "h1_r03_refined256_current_mp_query_v1",
            "query_index": query_index,
            "query_total": len(missing),
            "chemsys": chemsys,
            "status": "resolved",
            "entry_count": len(entries),
            "transport_attempts": attempts_used,
            "transport_retries": max(0, attempts_used - 1),
            "elapsed_seconds": time.monotonic() - started,
            "response_audit": response_audit,
            "error": final_error,
            "api_key_serialized": False,
            "sample_retry_or_replacement_used": False,
        }
        write_json_exclusive(spool / f"query_{query_index:04d}.json", {"fragment": fragment, "progress": progress})
        return query_index, fragment, progress

    results: dict[str, list[dict[str, Any]]] = {}
    progress: list[dict[str, Any]] = []
    failed = False
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
    futures = {executor.submit(one, item): item for item in enumerate(missing, start=1)}
    try:
        for future in concurrent.futures.as_completed(futures):
            query_index, fragment, row = future.result()
            results[str(fragment["chemsys"])] = list(fragment["entries"])
            progress.append(row)
            print(json.dumps({"query_index": query_index, "query_total": len(missing), "chemsys": row["chemsys"], "status": "resolved", "entry_count": row["entry_count"], "transport_attempts": row["transport_attempts"], "workers": workers}, sort_keys=True), flush=True)
    except BaseException:
        failed = True
        for future in futures:
            future.cancel()
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=failed)
        for client in clients:
            client.session.close()
    progress.sort(key=lambda row: int(row["query_index"]))
    return results, progress


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--key-file", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("SLURM_JOB_ID"):
        raise ContractError("MP completion must run on the login node")
    if any(os.environ.get(name) for name in ("MP_API_KEY", "PMG_MAPI_KEY", "MAPI_KEY")):
        raise ContractError("credential environment variables are forbidden")
    source = args.source_dir.resolve()
    require_source_manifest(source, args.source_manifest_sha256)
    config = read_json(source / "CONFIG.json")
    run_root = args.run_root.resolve()
    if Path(config["run_root"]).resolve() != run_root:
        raise ContractError("run root changed")
    status = run_root / "status"
    lock = status / "cache_completion.lock"
    if lock.exists() or (run_root / "mp_cache").exists():
        raise FileExistsError("cache completion was already invoked")
    write_json_exclusive(lock, {"schema": "h1_r03_refined256_current_mp_completion_lock_v1", "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "pid": os.getpid(), "source_manifest_sha256": args.source_manifest_sha256, "api_key_serialized": False})

    wanted = wanted_chemsys(config)
    cached, source_reports = load_existing(config, wanted)
    missing = sorted(wanted - set(cached))
    mp = config["current_mp"]
    if len(cached) != int(mp["existing_resolved_count"]) or len(missing) != int(mp["missing_chemsys_count"]) or line_set_sha256(missing) != mp["missing_chemsys_sha256"]:
        raise ContractError("current-cache historical coverage changed")
    module_path = require_file(Path(mp["completion_module"]), str(mp["completion_module_sha256"]), "current MP completion module")
    module = load_module(module_path)
    preparing = run_root / f".mp_cache.preparing.{os.getpid()}"
    failed_path = run_root / f".mp_cache.FAILED.{os.getpid()}"
    preparing.mkdir()
    spool = preparing / "spool"
    spool.mkdir()
    key = ""
    try:
        key = module.read_and_destroy_api_key(args.key_file.resolve())
        if args.key_file.exists():
            raise ContractError("one-time key carrier was not destroyed")
        queried, progress = query_missing(
            module=module,
            api_key=key,
            missing=missing,
            workers=int(mp["workers"]),
            maximum_rps=int(mp["max_requests_per_second"]),
            maximum_attempts=int(mp["maximum_transport_attempts_per_chemsys"]),
            spool=spool,
        )
        key = ""
        if set(queried) != set(missing) or any(row.get("status") != "resolved" for row in progress):
            raise ContractError("not every missing historical chemsys resolved")
        cached.update(queried)
        if set(cached) != wanted or any(not cached[system] for system in wanted):
            raise ContractError("completed historical cache is not total")
        query_fragment = preparing / "mp_query_fragment.jsonl"
        query_progress = preparing / "mp_query_progress.jsonl"
        completed = preparing / "completed_mp_hull_cache.jsonl"
        write_jsonl_exclusive(query_fragment, ({"chemsys": system, "entries": queried[system]} for system in missing))
        write_jsonl_exclusive(query_progress, progress)
        write_jsonl_exclusive(completed, ({"chemsys": system, "entries": cached[system]} for system in sorted(wanted)))
        completed_sha = sha256_file(completed)
        completed_identity = identity(completed)
        completed_identity["path"] = str(
            (run_root / "mp_cache/completed_mp_hull_cache.jsonl").resolve()
        )
        manifest = {
            "schema": "h1_r03_refined256_current_mp_completion_manifest_v1",
            "status": "complete_all_historical_chemsys_resolved",
            "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "run_id": config["run_id"],
            "source_manifest_sha256": args.source_manifest_sha256,
            "wanted_chemsys_count": len(wanted),
            "wanted_chemsys_sha256": line_set_sha256(wanted),
            "existing_resolved_count": len(wanted) - len(missing),
            "missing_chemsys_count": len(missing),
            "missing_chemsys_sha256": line_set_sha256(missing),
            "cache_sources": source_reports,
            "completion_module": identity(module_path),
            "workers": int(mp["workers"]),
            "max_requests_per_second": int(mp["max_requests_per_second"]),
            "query_status_counts": dict(Counter(str(row["status"]) for row in progress)),
            "transport_retries": sum(int(row["transport_retries"]) for row in progress),
            "completed_mp_hull_cache": {
                **completed_identity,
                "rows": len(wanted),
                "all_rows_populated": True,
            },
            "query_fragment": identity(query_fragment),
            "query_progress": identity(query_progress),
            "execution_location": "A800_login_node",
            "slurm_used": False,
            "gpu_used": False,
            "api_key_serialized": False,
            "credential_environment_used": False,
            "one_time_key_carrier_destroyed": True,
            "sample_retry_or_replacement_used": False,
        }
        write_json_exclusive(preparing / "completion_manifest.json", manifest)
        (preparing / "completion_SUCCESS").open("x").close()
        shutil.rmtree(spool)
        preparing.rename(run_root / "mp_cache")
        print(json.dumps(manifest, sort_keys=True), flush=True)
    except BaseException as exc:
        key = ""
        args.key_file.unlink(missing_ok=True)
        if preparing.exists():
            write_json_exclusive(preparing / "failure.json", {"schema": "h1_r03_refined256_current_mp_completion_failure_v1", "status": "failed_closed", "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "error_type": type(exc).__name__, "error_message_serialized": False, "api_key_serialized": False, "automatic_retry": False})
            preparing.rename(failed_path)
        raise


if __name__ == "__main__":
    main()
