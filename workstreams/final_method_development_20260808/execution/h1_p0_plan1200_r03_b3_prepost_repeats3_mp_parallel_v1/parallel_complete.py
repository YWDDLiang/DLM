#!/usr/bin/env python3
"""Resume a frozen serial MP-cache prefix with bounded parallel queries."""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
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
from typing import Any, Callable, Iterable, Mapping

WORKERS = 6
MAX_REQUESTS_PER_SECOND = 10
SERIAL_SCHEMA = "h1a2c_mp_chemsys_query_v1"
BODY_SOURCE_MANIFEST_SHA256 = (
    "080db87fc12319b02000e121b306b7d26eed194b26afe2921276a1647ccc7ed8"
)


class ParallelCompletionError(RuntimeError):
    """A fail-closed parallel completion contract violation."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    location = source.resolve()
    if not location.is_file():
        raise FileNotFoundError(location)
    return {
        "path": str(destination.resolve()),
        "bytes": location.stat().st_size,
        "sha256": sha256_file(location),
    }


def load_module(name: str, path: Path) -> ModuleType:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ImportError(f"cannot load frozen module: {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def require_source_manifest(source: Path, expected_manifest_sha256: str) -> None:
    root = source.resolve()
    manifest = root / "SOURCE_SHA256.txt"
    if sha256_file(manifest) != expected_manifest_sha256:
        raise ParallelCompletionError(f"source manifest changed: {root}")
    entries: list[tuple[str, str]] = []
    for raw in manifest.read_text(encoding="utf-8").splitlines():
        digest, separator, relative = raw.partition("  ")
        if not separator or len(digest) != 64 or relative.startswith(("/", "../")):
            raise ParallelCompletionError("source manifest row is invalid")
        entries.append((digest, relative))
    listed = {relative for _, relative in entries}
    observed = {
        str(path.relative_to(root)).replace(os.sep, "/")
        for path in root.rglob("*")
        if path.is_file()
        and path.name != "SOURCE_SHA256.txt"
        and "__pycache__" not in path.parts
        and not path.name.endswith((".pyc", ".pyo"))
    }
    if listed != observed:
        raise ParallelCompletionError("parallel source file set changed")
    for digest, relative in entries:
        if sha256_file(root / relative) != digest:
            raise ParallelCompletionError(f"parallel source file changed: {relative}")


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ParallelCompletionError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            if not isinstance(row, dict):
                raise ParallelCompletionError(
                    f"expected JSON object at {path}:{line_number}"
                )
            rows.append(row)
    return rows


def write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_jsonl_exclusive(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    allow_nan=False,
                )
                + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())


def write_sha(path: Path, digest: str, filename: str) -> None:
    with path.open("x", encoding="ascii", newline="\n") as handle:
        handle.write(f"{digest}  {filename}\n")
        handle.flush()
        os.fsync(handle.fileno())


class SlidingWindowLimiter:
    """A process-wide strict one-second request window."""

    def __init__(self, maximum: int) -> None:
        self.maximum = maximum
        self._events: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            wait_seconds = 0.0
            with self._lock:
                now = time.monotonic()
                while self._events and self._events[0] <= now - 1.0:
                    self._events.popleft()
                if len(self._events) < self.maximum:
                    self._events.append(now)
                    return
                wait_seconds = max(0.001, 1.001 - (now - self._events[0]))
            time.sleep(wait_seconds)


def retry_after_seconds(exc: BaseException) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    raw = headers.get("Retry-After")
    if raw is None:
        return None
    try:
        return max(0.0, min(300.0, float(raw)))
    except (TypeError, ValueError):
        return None


def validate_serial_prefix(
    checkpoint: Path, missing: list[str]
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]], dict[str, Any]]:
    fragment_path = checkpoint / "mp_query_fragment.jsonl"
    progress_path = checkpoint / "mp_query_progress.jsonl"
    fragments = read_jsonl(fragment_path)
    progress = read_jsonl(progress_path)
    if not progress or len(fragments) not in {len(progress), len(progress) + 1}:
        raise ParallelCompletionError("serial checkpoint rows are incomplete")
    if len(progress) >= len(missing):
        raise ParallelCompletionError("serial checkpoint is already total")
    uncommitted_trailing_fragment: dict[str, Any] | None = None
    if len(fragments) == len(progress) + 1:
        candidate = fragments[-1]
        expected = missing[len(progress)]
        entries = candidate.get("entries")
        if (
            candidate.get("chemsys") != expected
            or not isinstance(entries, list)
            or not entries
        ):
            raise ParallelCompletionError(
                "serial checkpoint trailing fragment is not the next frozen query"
            )
        uncommitted_trailing_fragment = {
            "query_index": len(progress) + 1,
            "chemsys": expected,
            "entry_count": len(entries),
            "reused": False,
            "reason": "fragment fsynced but progress row absent at operator-approved interrupt",
        }
    queried: dict[str, list[dict[str, Any]]] = {}
    for offset, (fragment, row) in enumerate(
        zip(fragments[: len(progress)], progress), start=1
    ):
        expected = missing[offset - 1]
        entries = fragment.get("entries")
        if (
            fragment.get("chemsys") != expected
            or row.get("schema") != SERIAL_SCHEMA
            or int(row.get("query_index", -1)) != offset
            or int(row.get("query_total", -1)) != len(missing)
            or row.get("chemsys") != expected
            or row.get("status") != "resolved"
            or not isinstance(entries, list)
            or not entries
            or int(row.get("entry_count", -1)) != len(entries)
            or int(row.get("transport_attempts", -1)) < 1
        ):
            raise ParallelCompletionError(
                f"serial checkpoint is not a resolved prefix at index {offset}"
            )
        queried[expected] = entries
    return queried, progress, {
        "directory": str(checkpoint.resolve()),
        "resolved_prefix_count": len(progress),
        "fragment": identity(fragment_path),
        "progress": identity(progress_path),
        "physical_fragment_rows": len(fragments),
        "physical_progress_rows": len(progress),
        "uncommitted_trailing_fragment": uncommitted_trailing_fragment,
    }


def require_login_node() -> None:
    if os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOB_NAME"):
        raise ParallelCompletionError("parallel MP completion is login-node-only")
    if any(os.environ.get(name) for name in ("MP_API_KEY", "PMG_MAPI_KEY", "MAPI_KEY")):
        raise ParallelCompletionError("credential environment variables are forbidden")


def query_remaining(
    *,
    module: ModuleType,
    api_key: str,
    remaining: list[tuple[int, str]],
    total: int,
    maximum_attempts: int,
    spool: Path,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    limiter = SlidingWindowLimiter(MAX_REQUESTS_PER_SECOND)
    thread_local = threading.local()
    clients: list[Any] = []
    clients_lock = threading.Lock()

    def client_for_thread() -> Any:
        client = getattr(thread_local, "client", None)
        if client is None:
            client = module.CurrentMPThermoClient(api_key)
            original_get = client.session.get

            def limited_get(*args: Any, **kwargs: Any) -> Any:
                limiter.acquire()
                return original_get(*args, **kwargs)

            client.session.get = limited_get
            thread_local.client = client
            with clients_lock:
                clients.append(client)
        return client

    def query_one(item: tuple[int, str]) -> tuple[int, dict[str, Any], dict[str, Any]]:
        query_index, chemsys = item
        started = time.monotonic()
        entries: list[dict[str, Any]] | None = None
        response_audit: dict[str, Any] | None = None
        final_error: dict[str, Any] | None = None
        attempts_used = 0
        client = client_for_thread()
        for transport_attempt in range(1, maximum_attempts + 1):
            attempts_used = transport_attempt
            try:
                raw_entries, response_audit = client.get_entries_in_chemsys(chemsys)
                entries = module.slim_entries(raw_entries)
                final_error = None
                break
            except Exception as exc:  # noqa: BLE001 - sanitize and fail closed.
                if isinstance(exc, module.CompletionError):
                    raise ParallelCompletionError(
                        f"frozen response contract failed at query {query_index}"
                    ) from None
                final_error = module.sanitized_query_error(exc)
                status_code = final_error.get("http_status")
                if status_code in {401, 403}:
                    raise ParallelCompletionError(
                        f"Materials Project authorization rejected at query {query_index}"
                    ) from None
                if transport_attempt < maximum_attempts:
                    requested_wait = retry_after_seconds(exc)
                    backoff = min(16.0, float(2 ** (transport_attempt - 1)))
                    time.sleep(max(backoff, requested_wait or 0.0))
        status = "resolved" if entries else ("empty" if entries == [] else "query_error")
        if status != "resolved" or entries is None:
            raise ParallelCompletionError(
                f"Materials Project query did not resolve index {query_index}: {status}"
            )
        fragment = {"chemsys": chemsys, "entries": entries}
        row = {
            "schema": SERIAL_SCHEMA,
            "query_index": query_index,
            "query_total": total,
            "chemsys": chemsys,
            "status": status,
            "entry_count": len(entries),
            "transport_attempts": attempts_used,
            "transport_retries": max(0, attempts_used - 1),
            "sample_retry_or_replacement_used": False,
            "elapsed_seconds": time.monotonic() - started,
            "error": final_error,
            "response_audit": response_audit,
            "api_key_serialized": False,
            "completion_mode": "bounded_parallel_continuation",
        }
        write_json_exclusive(
            spool / f"query_{query_index:05d}.json",
            {"fragment": fragment, "progress": row},
        )
        return query_index, fragment, row

    queried: dict[str, list[dict[str, Any]]] = {}
    progress: list[dict[str, Any]] = []
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS)
    futures = {executor.submit(query_one, item): item for item in remaining}
    failed = False
    try:
        for future in concurrent.futures.as_completed(futures):
            query_index, fragment, row = future.result()
            queried[str(fragment["chemsys"])] = list(fragment["entries"])
            progress.append(row)
            print(
                json.dumps(
                    {
                        "query_index": query_index,
                        "query_total": total,
                        "chemsys": row["chemsys"],
                        "status": row["status"],
                        "entry_count": row["entry_count"],
                        "transport_attempts": row["transport_attempts"],
                        "workers": WORKERS,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    except BaseException:
        failed = True
        for future in futures:
            future.cancel()
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=failed)
        for client in clients:
            client.session.close()
        clients.clear()
    progress.sort(key=lambda row: int(row["query_index"]))
    return queried, progress


def complete(args: argparse.Namespace) -> dict[str, Any]:
    require_login_node()
    body_source = args.body_source_dir.resolve()
    parallel_source = args.parallel_source_dir.resolve()
    run_root = args.run_root.resolve()
    if args.body_source_manifest_sha256 != BODY_SOURCE_MANIFEST_SHA256:
        raise ParallelCompletionError("frozen V3 body source identity changed")
    require_source_manifest(parallel_source, args.parallel_source_manifest_sha256)
    sys.path.insert(0, str(body_source))
    body = load_module("h1_plan1200_body_mp_cache_v3", body_source / "mp_cache.py")
    body.require_source_manifest(body_source, args.body_source_manifest_sha256)
    config = body.read_json(args.config.resolve())
    body.validate_config(config)
    if Path(config["run_root"]).resolve() != run_root:
        raise ParallelCompletionError("run root changed")

    audit = body.read_json(args.audit.resolve())
    if (
        audit.get("status") != "complete"
        or audit.get("run_id") != config["run_id"]
        or audit.get("source_manifest_sha256") != args.body_source_manifest_sha256
        or audit.get("mp_query_performed") is not False
    ):
        raise ParallelCompletionError("cache audit contract changed")

    sun = config["sun"]
    completion_module_path = body.require_file(
        sun["completion_module"],
        sun["completion_module_sha256"],
        "completion module",
    )
    module = body.load_module(completion_module_path)
    wanted, planner_reports = body.planner_union(run_root)
    cached, source_reports = body.cache_sources(config, module, wanted)
    missing = sorted(wanted - set(cached))
    if (
        int(audit.get("wanted_chemsys_count", -1)) != len(wanted)
        or audit.get("wanted_chemsys_sha256") != line_set_sha256(wanted)
        or int(audit.get("missing_chemsys_count", -1)) != len(missing)
        or audit.get("missing_chemsys_sha256") != line_set_sha256(missing)
    ):
        raise ParallelCompletionError("frozen wanted/missing set changed")
    if (run_root / "mp_cache").exists():
        raise FileExistsError(run_root / "mp_cache")

    serial_queried, serial_progress, serial_report = validate_serial_prefix(
        args.serial_checkpoint_dir.resolve(), missing
    )
    remaining = [
        (index, chemsys)
        for index, chemsys in enumerate(missing, start=1)
        if index > len(serial_progress)
    ]
    if not remaining:
        raise ParallelCompletionError("parallel continuation has no remaining work")

    lock_path = run_root / "status/mp_parallel_completion.lock"
    lock = {
        "schema": "h1_plan1200_mp_parallel_completion_lock_v1",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "pid": os.getpid(),
        "workers": WORKERS,
        "max_requests_per_second": MAX_REQUESTS_PER_SECOND,
        "serial_prefix_count": len(serial_progress),
        "remaining_count": len(remaining),
        "parallel_source_manifest_sha256": args.parallel_source_manifest_sha256,
        "api_key_serialized": False,
    }
    write_json_exclusive(lock_path, lock)

    mp_root = run_root / f".mp_cache.parallel.preparing.{os.getpid()}"
    failed_root = run_root / f".mp_cache.parallel.FAILED.{os.getpid()}"
    mp_root.mkdir()
    spool = mp_root / "spool"
    spool.mkdir()
    key = ""
    try:
        if not args.key_file.is_file():
            raise ParallelCompletionError("one-time key carrier is required")
        key = module.read_and_destroy_api_key(args.key_file.resolve())
        if args.key_file.exists():
            raise ParallelCompletionError("one-time key carrier was not destroyed")
        parallel_queried, parallel_progress = query_remaining(
            module=module,
            api_key=key,
            remaining=remaining,
            total=len(missing),
            maximum_attempts=int(sun["maximum_transport_attempts_per_chemsys"]),
            spool=spool,
        )
        key = ""

        queried = {**serial_queried, **parallel_queried}
        progress = sorted(
            [*serial_progress, *parallel_progress],
            key=lambda row: int(row["query_index"]),
        )
        if (
            set(queried) != set(missing)
            or len(progress) != len(missing)
            or [int(row["query_index"]) for row in progress]
            != list(range(1, len(missing) + 1))
            or [str(row["chemsys"]) for row in progress] != missing
            or any(row.get("status") != "resolved" for row in progress)
            or any(not queried.get(chemsys) for chemsys in missing)
        ):
            raise ParallelCompletionError("not every frozen missing chemsys resolved")

        fragment = mp_root / "mp_query_fragment.jsonl"
        progress_path = mp_root / "mp_query_progress.jsonl"
        write_jsonl_exclusive(
            fragment,
            ({"chemsys": chemsys, "entries": queried[chemsys]} for chemsys in missing),
        )
        write_jsonl_exclusive(progress_path, progress)

        cached.update(queried)
        if set(cached) != wanted or any(not cached[system] for system in wanted):
            raise ParallelCompletionError("completed planner-union cache is not total")
        completed = mp_root / "completed_mp_hull_cache.jsonl"
        write_jsonl_exclusive(
            completed,
            ({"chemsys": system, "entries": cached[system]} for system in sorted(wanted)),
        )
        completed_sha = sha256_file(completed)
        write_sha(
            mp_root / "completed_mp_hull_cache.sha256",
            completed_sha,
            completed.name,
        )

        status_counts = Counter(str(row["status"]) for row in progress)
        final_root = run_root / "mp_cache"
        manifest = {
            "schema": "h1_plan1200_mp_cache_completion_manifest_v3",
            "status": "complete_all_wanted_chemsys_resolved",
            "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "run_id": config["run_id"],
            "source_manifest_sha256": args.body_source_manifest_sha256,
            "planner_repeats": planner_reports,
            "cache_sources": source_reports,
            "audit": identity(args.audit.resolve()),
            "frozen_completion_module": identity(completion_module_path),
            "parallel_completion_driver": identity(Path(__file__).resolve()),
            "parallel_source_manifest": identity(parallel_source / "SOURCE_SHA256.txt"),
            "parallel_source_manifest_sha256": args.parallel_source_manifest_sha256,
            "completion_mode": "audited_serial_prefix_plus_bounded_parallel_continuation",
            "serial_checkpoint": serial_report,
            "workers": WORKERS,
            "max_requests_per_second": MAX_REQUESTS_PER_SECOND,
            "serial_resolved_queries": len(serial_progress),
            "parallel_resolved_queries": len(parallel_progress),
            "wanted_chemsys_count": len(wanted),
            "wanted_chemsys_sha256": line_set_sha256(wanted),
            "base_and_supplemental_populated": len(wanted) - len(missing),
            "missing_chemsys_count": len(missing),
            "missing_chemsys_sha256": line_set_sha256(missing),
            "query_status_counts": dict(sorted(status_counts.items())),
            "logical_queries": len(progress),
            "transport_retries": sum(
                int(row.get("transport_retries", 0)) for row in progress
            ),
            "completed_mp_hull_cache": {
                "path": str(final_root / "completed_mp_hull_cache.jsonl"),
                "bytes": completed.stat().st_size,
                "sha256": completed_sha,
                "rows": len(wanted),
                "all_rows_populated": True,
            },
            "query_fragment": relocated_identity(
                fragment, final_root / "mp_query_fragment.jsonl"
            ),
            "query_progress": relocated_identity(
                progress_path, final_root / "mp_query_progress.jsonl"
            ),
            "execution_location": "A800_login_node",
            "slurm_used": False,
            "gpu_used": False,
            "api_key_serialized": False,
            "credential_environment_used": False,
            "one_time_key_carrier_destroyed": True,
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
        shutil.rmtree(spool)
        mp_root.rename(final_root)
        print(json.dumps(manifest, sort_keys=True), flush=True)
        return manifest
    except BaseException as exc:
        key = ""
        if args.key_file.exists():
            args.key_file.unlink(missing_ok=True)
        failure = {
            "schema": "h1_plan1200_mp_parallel_completion_failure_v1",
            "status": "failed_closed",
            "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "error_type": type(exc).__name__,
            "error_message_serialized": False,
            "api_key_serialized": False,
            "automatic_retry": False,
        }
        if mp_root.exists():
            write_json_exclusive(mp_root / "failure.json", failure)
            mp_root.rename(failed_root)
        print(json.dumps(failure, sort_keys=True), file=sys.stderr, flush=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--body-source-dir", type=Path, required=True)
    parser.add_argument("--body-source-manifest-sha256", required=True)
    parser.add_argument("--parallel-source-dir", type=Path, required=True)
    parser.add_argument("--parallel-source-manifest-sha256", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--serial-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--key-file", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    try:
        complete(parse_args())
    except BaseException:
        raise SystemExit(1) from None
