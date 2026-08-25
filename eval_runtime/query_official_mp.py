#!/usr/bin/env python3
"""Build a fresh, fixed-thermo MP cache through the official high-level API."""

from __future__ import annotations

import argparse
import gzip
import importlib.metadata
import json
import os
import platform
import shutil
import stat
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

import pymatgen.entries.compatibility as _compatibility
import pymatgen.entries.computed_entries as _computed_entries

# Old MP MSON payloads can still contain these import paths.
sys.modules.setdefault("pymatgen.core.entries", _computed_entries)
sys.modules.setdefault("pymatgen.analysis.compatibility", _compatibility)

from mp_api.client import MPRester
from pymatgen.analysis.phase_diagram import PhaseDiagram

from protocol import (
    ContractError,
    canonical_json,
    canonical_sha256,
    finite_float,
    identity,
    read_json,
    read_jsonl,
    require_source_manifest,
    write_json_exclusive,
)


THERMO_TYPE = "GGA_GGA+U"
THERMO_CRITERIA = {"thermo_types": [THERMO_TYPE]}


class SlidingWindowLimiter:
    """Process-wide strict one-second HTTP request window."""

    def __init__(self, maximum: int) -> None:
        if maximum <= 0:
            raise ValueError("maximum requests per second must be positive")
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
                delay = max(0.001, 1.001 - (now - self.events[0]))
            time.sleep(delay)


def package_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for name in ("mp-api", "pymatgen", "emmet-core", "monty"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def require_official_runtime(config: dict[str, Any]) -> None:
    runtime = config.get("runtime") or {}
    expected_python = str(runtime.get("official_mp_python") or "")
    if sys.executable != expected_python:
        raise ContractError(
            f"official MP interpreter changed: {sys.executable!r} != {expected_python!r}"
        )
    expected_python_version = str(runtime.get("official_mp_python_version") or "")
    if platform.python_version() != expected_python_version:
        raise ContractError(
            "official MP Python version changed: "
            f"{platform.python_version()!r} != {expected_python_version!r}"
        )
    expected_packages = runtime.get("official_mp_packages") or {}
    observed = package_versions()
    for package, expected in expected_packages.items():
        if observed.get(str(package)) != str(expected):
            raise ContractError(
                f"official MP package changed: {package}={observed.get(str(package))!r} "
                f"!= {expected!r}"
            )


def read_destroy_key(path: Path) -> str:
    location = path.expanduser()
    details = location.lstat()
    raw = b""
    try:
        mode = stat.S_IMODE(details.st_mode)
        if not stat.S_ISREG(details.st_mode):
            raise ContractError("one-time key carrier must be a regular file")
        if details.st_uid != os.getuid():
            raise ContractError("one-time key carrier is not owned by this user")
        if mode != 0o600:
            raise ContractError(
                f"one-time key carrier mode is {oct(mode)}, expected 0o600"
            )
        if details.st_size <= 0 or details.st_size > 256:
            raise ContractError("one-time key carrier size is invalid")
        raw = location.read_bytes()
    finally:
        location.unlink(missing_ok=True)
    key = raw.decode("ascii").strip()
    if len(key) != 32 or any(character.isspace() for character in key):
        raise ContractError("one-time key carrier is malformed")
    return key


def sanitized_error(exc: BaseException, secret: str) -> dict[str, Any]:
    message = str(exc).replace(secret, "[REDACTED]")
    for separator in ("Content:", "Response:", "{\"data\":"):
        message = message.split(separator, 1)[0]
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    return {
        "type": type(exc).__name__,
        "http_status": None if status_code is None else int(status_code),
        "message": " ".join(message.split())[:500],
    }


def retry_after_seconds(exc: BaseException) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    raw = headers.get("Retry-After")
    if raw is None:
        return None
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        try:
            target = parsedate_to_datetime(str(raw))
            if target.tzinfo is None:
                target = target.replace(tzinfo=timezone.utc)
            return max(0.0, (target - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return None


def entry_sort_key(entry: Any) -> tuple[str, str, float]:
    return (
        "" if getattr(entry, "entry_id", None) is None else str(entry.entry_id),
        canonical_json(entry.composition.as_dict()),
        finite_float(entry.energy, "entry energy"),
    )


def slim_entries(entries: list[Any]) -> list[dict[str, Any]]:
    rows = [
        {
            "entry_id": (
                None
                if getattr(entry, "entry_id", None) is None
                else str(entry.entry_id)
            ),
            "composition": entry.composition.as_dict(),
            "energy": finite_float(entry.energy, "entry energy"),
        }
        for entry in sorted(entries, key=entry_sort_key)
    ]
    if len(rows) != len(entries):
        raise AssertionError("entry serialization changed row count")
    return rows


def full_entries(entries: list[Any]) -> list[dict[str, Any]]:
    rows = [entry.as_dict() for entry in sorted(entries, key=entry_sort_key)]
    if any(not isinstance(row, dict) for row in rows):
        raise ContractError("official MP entry did not serialize to MSON object")
    return rows


def validate_reference_set(entries: list[Any], elements: list[str]) -> dict[str, Any]:
    if not entries:
        raise ContractError("official get_entries_in_chemsys returned no entries")
    requested = set(elements)
    outside: set[str] = set()
    unary: set[str] = set()
    for entry in entries:
        symbols = {element.symbol for element in entry.composition.elements}
        outside.update(symbols - requested)
        if len(symbols) == 1:
            unary.update(symbols)
    if outside:
        raise ContractError(f"reference entries contain outside elements: {outside}")
    missing_unary = sorted(requested - unary)
    if missing_unary:
        raise ContractError(f"missing unary references: {missing_unary}")
    phase_diagram = PhaseDiagram(entries)
    pd_elements = {element.symbol for element in phase_diagram.elements}
    if pd_elements != requested:
        raise ContractError(
            f"phase diagram elements {sorted(pd_elements)} != {sorted(requested)}"
        )
    return {
        "unary_reference_elements": sorted(unary),
        "phase_diagram_constructed": True,
    }


def write_fragment(path: Path, record: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with gzip.open(
        temporary, "xt", encoding="utf-8", newline="\n", compresslevel=6
    ) as handle:
        handle.write(canonical_json(record) + "\n")
    os.replace(temporary, path)


def read_fragment(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        value = json.loads(handle.read())
    if not isinstance(value, dict):
        raise ContractError(f"invalid fragment: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    key_source = parser.add_mutually_exclusive_group(required=True)
    key_source.add_argument("--key-file", type=Path)
    key_source.add_argument("--key-env")
    args = parser.parse_args()

    source = args.source_dir.resolve()
    require_source_manifest(source, args.source_manifest_sha256)
    config = read_json(args.config.resolve())
    require_official_runtime(config)
    thermo = config["thermo"]
    if (
        thermo.get("query_method")
        != "mp_api.client.MPRester.get_entries_in_chemsys"
        or thermo.get("thermo_type") != THERMO_TYPE
        or thermo.get("additional_criteria") != THERMO_CRITERIA
        or thermo.get("compatible_only") is not True
        or thermo.get("fresh_empty_cache") is not True
        or thermo.get("reuse_any_historical_or_august_cache") is not False
        or thermo.get("local_compatibility_reprocessing") is not False
    ):
        raise ContractError("official fixed-thermo contract changed")
    if any(os.environ.get(name) for name in ("MP_API_KEY", "PMG_MAPI_KEY", "MAPI_KEY")):
        raise ContractError("ambient MP credentials are forbidden")
    workers = int(config["query"]["workers"])
    maximum_rps = int(config["query"]["max_http_requests_per_second"])
    maximum_attempts = int(config["query"]["max_transport_attempts"])
    if workers != 6 or maximum_rps != 8 or maximum_attempts != 5:
        raise ContractError("bounded official-query execution contract changed")

    run_root = args.run_root.resolve()
    inputs = run_root / "inputs"
    if not (inputs / "inputs_SUCCESS").is_file():
        raise ContractError("frozen input manifest is incomplete")
    wanted = read_jsonl(inputs / "wanted_chemsys.jsonl")
    input_manifest = read_json(inputs / "input_manifest.json")
    if len(wanted) != int(input_manifest["wanted_chemsys_count"]):
        raise ContractError("wanted chemsys count changed")
    if canonical_sha256(wanted) != input_manifest["wanted_chemsys_sha256"]:
        raise ContractError("wanted chemsys identity changed")
    if [int(row["query_index"]) for row in wanted] != list(range(len(wanted))):
        raise ContractError("wanted query indexes are not contiguous")

    final = run_root / "official_mp_cache"
    if final.exists():
        raise FileExistsError(final)
    preparing = run_root / f".official_mp_cache.preparing.{os.getpid()}"
    failed = run_root / f".official_mp_cache.FAILED.{os.getpid()}"
    preparing.mkdir(parents=True, exist_ok=False)
    spool = preparing / "spool"
    spool.mkdir()
    api_key = ""
    try:
        if args.key_env:
            api_key = os.environ.pop(str(args.key_env), "").strip()
            if not api_key:
                raise ContractError("in-memory MP credential is missing")
        else:
            api_key = read_destroy_key(args.key_file)
    except Exception:
        if preparing.exists():
            shutil.move(str(preparing), str(failed))
        raise
    started = datetime.now(timezone.utc).isoformat()

    clients: list[MPRester] = []
    clients_lock = threading.Lock()
    local = threading.local()
    limiter = SlidingWindowLimiter(maximum_rps)

    def client_for_thread() -> MPRester:
        client = getattr(local, "client", None)
        if client is None:
            client = MPRester(api_key)
            original_get = client.session.get

            def limited_get(*request_args: Any, **request_kwargs: Any) -> Any:
                limiter.acquire()
                return original_get(*request_args, **request_kwargs)

            client.session.get = limited_get
            local.client = client
            with clients_lock:
                clients.append(client)
        return client

    def query_one(spec: dict[str, Any]) -> dict[str, Any]:
        index = int(spec["query_index"])
        chemsys = str(spec["chemsys"])
        elements = [str(value) for value in spec["elements"]]
        record: dict[str, Any] = {
            "schema": "h1_official_mp_gga_u_chemsys_v1",
            "query_index": index,
            "query_total": len(wanted),
            "chemsys": chemsys,
            "elements": elements,
            "thermo_type": THERMO_TYPE,
            "additional_criteria": THERMO_CRITERIA,
            "compatible_only": True,
            "query_method": "mp_api.client.MPRester.get_entries_in_chemsys",
            "local_compatibility_reprocessing": False,
            "manual_transport_attempts": 0,
        }
        for transport_attempt in range(1, maximum_attempts + 1):
            record["manual_transport_attempts"] = transport_attempt
            try:
                entries = client_for_thread().get_entries_in_chemsys(
                    elements,
                    compatible_only=True,
                    additional_criteria=THERMO_CRITERIA,
                )
                validation = validate_reference_set(entries, elements)
                slim = slim_entries(entries)
                full = full_entries(entries)
                record.update(
                    {
                        "query_status": "resolved",
                        "entry_count": len(entries),
                        "slim_entries_sha256": canonical_sha256(slim),
                        "full_entries_mson_sha256": canonical_sha256(full),
                        "slim_entries": slim,
                        "full_entries_mson": full,
                        "error": None,
                        **validation,
                    }
                )
                break
            except ContractError as exc:
                record["error"] = sanitized_error(exc, api_key)
                break
            except Exception as exc:
                error = sanitized_error(exc, api_key)
                record["error"] = error
                if error["http_status"] in {401, 403}:
                    break
                if transport_attempt < maximum_attempts:
                    requested = retry_after_seconds(exc) or 0.0
                    backoff = min(16.0, float(2 ** (transport_attempt - 1)))
                    time.sleep(max(requested, backoff))
        if record.get("query_status") != "resolved":
            record.update(
                {
                    "query_status": "query_error",
                    "entry_count": None,
                    "slim_entries_sha256": None,
                    "full_entries_mson_sha256": None,
                    "slim_entries": None,
                    "full_entries_mson": None,
                    "phase_diagram_constructed": False,
                }
            )
        path = spool / f"{index:05d}.json.gz"
        write_fragment(path, record)
        return {
            "query_index": index,
            "chemsys": chemsys,
            "query_status": record["query_status"],
            "entry_count": record["entry_count"],
            "manual_transport_attempts": record["manual_transport_attempts"],
        }

    try:
        limiter.acquire()
        with MPRester(api_key) as probe:
            database_version = probe.get_database_version()
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(query_one, spec) for spec in wanted]
            completed = 0
            for future in as_completed(futures):
                result = future.result()
                completed += 1
                print(
                    canonical_json({"completed": completed, "total": len(wanted), **result}),
                    flush=True,
                )
        for client in clients:
            try:
                client.session.close()
            except Exception:
                pass

        failures: list[dict[str, Any]] = []
        total_transport_retries = 0
        for index in range(len(wanted)):
            row = read_fragment(spool / f"{index:05d}.json.gz")
            total_transport_retries += max(
                0, int(row["manual_transport_attempts"]) - 1
            )
            if row["query_status"] != "resolved":
                failures.append(
                    {
                        key: row.get(key)
                        for key in ("query_index", "chemsys", "error")
                    }
                )
        authorization_failures = [
            row for row in failures
            if (row.get("error") or {}).get("http_status") in {401, 403}
        ]
        if authorization_failures:
            raise ContractError("official MP authorization failed")

        full_path = preparing / "official_full_entries.jsonl.gz"
        audit_path = preparing / "query_audit.jsonl"
        slim_path = preparing / "official_slim_cache.jsonl"
        unresolved_path = preparing / "unresolved_chemsys.jsonl"
        with gzip.open(
            full_path, "xt", encoding="utf-8", newline="\n", compresslevel=6
        ) as full_handle, audit_path.open(
            "x", encoding="utf-8", newline="\n"
        ) as audit_handle, slim_path.open(
            "x", encoding="utf-8", newline="\n"
        ) as slim_handle, unresolved_path.open(
            "x", encoding="utf-8", newline="\n"
        ) as unresolved_handle:
            for index in range(len(wanted)):
                row = read_fragment(spool / f"{index:05d}.json.gz")
                full_handle.write(
                    canonical_json(
                        {
                            key: value
                            for key, value in row.items()
                            if key != "slim_entries"
                        }
                    )
                    + "\n"
                )
                audit_handle.write(
                    canonical_json(
                        {
                            key: value
                            for key, value in row.items()
                            if key not in ("slim_entries", "full_entries_mson")
                        }
                    )
                    + "\n"
                )
                if row["query_status"] == "resolved":
                    slim_handle.write(
                        canonical_json(
                            {"chemsys": row["chemsys"], "entries": row["slim_entries"]}
                        )
                        + "\n"
                    )
                else:
                    unresolved_handle.write(
                        canonical_json(
                            {
                                "chemsys": row["chemsys"],
                                "elements": row["elements"],
                                "reason": "fresh_official_query_unresolved",
                                "error": row.get("error"),
                            }
                        )
                        + "\n"
                    )
            audit_handle.flush()
            os.fsync(audit_handle.fileno())
            slim_handle.flush()
            os.fsync(slim_handle.fileno())
            unresolved_handle.flush()
            os.fsync(unresolved_handle.fileno())

        full_identity = identity(full_path)
        audit_identity = identity(audit_path)
        slim_identity = identity(slim_path)
        unresolved_identity = identity(unresolved_path)
        full_identity["path"] = str((final / full_path.name).resolve())
        audit_identity["path"] = str((final / audit_path.name).resolve())
        slim_identity["path"] = str((final / slim_path.name).resolve())
        unresolved_identity["path"] = str((final / unresolved_path.name).resolve())
        manifest = {
            "schema": "h1_official_mp_gga_u_clean_cache_manifest_v1",
            "started_at_utc": started,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "query_status": "complete_with_explicit_unresolved",
            "query_count": len(wanted),
            "query_resolved": len(wanted) - len(failures),
            "query_unresolved": len(failures),
            "query_method": "mp_api.client.MPRester.get_entries_in_chemsys",
            "compatible_only": True,
            "thermo_type": THERMO_TYPE,
            "additional_criteria": THERMO_CRITERIA,
            "local_compatibility_reprocessing": False,
            "fresh_empty_cache": True,
            "historical_cache_rows_reused": 0,
            "august_cache_rows_reused": 0,
            "all_resolved_phase_diagrams_constructed": True,
            "all_resolved_elemental_references_present": True,
            "max_transport_attempts": maximum_attempts,
            "manual_transport_retries": total_transport_retries,
            "workers": workers,
            "max_http_requests_per_second": maximum_rps,
            "database_version": str(database_version),
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "package_versions": package_versions(),
            "inputs": {
                "wanted_chemsys": identity(inputs / "wanted_chemsys.jsonl"),
                "input_manifest": identity(inputs / "input_manifest.json"),
            },
            "outputs": {
                "full_mson_snapshot": full_identity,
                "query_audit": audit_identity,
                "slim_evaluation_cache": slim_identity,
                "unresolved_chemsys": unresolved_identity,
            },
            "api_key_serialized": False,
            "api_key_carrier_destroyed_before_first_query": True,
        }
        write_json_exclusive(preparing / "completion_manifest.json", manifest)
        (preparing / "completion_SUCCESS").touch(exist_ok=False)
        shutil.rmtree(spool)
        preparing.rename(final)
    except Exception:
        for client in clients:
            try:
                client.session.close()
            except Exception:
                pass
        if preparing.exists():
            shutil.move(str(preparing), str(failed))
        raise
    finally:
        api_key = ""
    print(canonical_json({"official_mp_cache": "complete", "queries": len(wanted)}))


if __name__ == "__main__":
    main()
