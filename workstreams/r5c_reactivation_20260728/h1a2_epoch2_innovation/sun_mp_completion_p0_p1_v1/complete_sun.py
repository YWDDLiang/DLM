#!/usr/bin/env python3
"""Complete missing MP hull references for a frozen paired S.U.N. run.

The source run is immutable.  This program does no generation, CHGNet work,
training, repair, filtering, or sample retry/replacement.  It queries each
missing chemical system once logically (with bounded transport retries), then
maps the resulting phase diagrams back to the original 256 attempts per arm.
The API key is read from a mode-0600 file, unlinked immediately, and never
serialized or printed.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import importlib.metadata
import itertools
import json
import math
import os
import random
import shutil
import stat
import sys
import time
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ARM_ORDER = ("P0", "P1")
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20260731
STRICT_SEED_OFFSET = 8
META_SEED_OFFSET = 9
CURRENT_MP_THERMO_ENDPOINT = "https://api.materialsproject.org/materials/thermo/"
CURRENT_MP_PAGE_LIMIT = 1000
OPTIONAL_MSON_MODULE_PREFIXES = ("emmet.", "mp_api.")
MSON_MODULE_REDIRECTS = {
    "pymatgen.analysis.compatibility": "pymatgen.entries.compatibility",
    "pymatgen.core.entries": "pymatgen.entries.computed_entries",
    "pymatgen.core.structure_matcher": "pymatgen.analysis.structure_matcher",
}


class CompletionError(RuntimeError):
    """Raised when the frozen completion contract cannot be honored."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CompletionError(f"{path}: expected one JSON object")
    return value


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise CompletionError(
                    f"{path}:{line_number}: expected one JSON object"
                )
            rows.append(value)
    return rows


def write_json_exclusive(path: str | Path, value: Mapping[str, Any]) -> None:
    location = Path(path)
    location.parent.mkdir(parents=True, exist_ok=True)
    with location.open("x", encoding="utf-8") as handle:
        json.dump(
            value,
            handle,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_jsonl_exclusive(
    path: str | Path, rows: Iterable[Mapping[str, Any]]
) -> None:
    location = Path(path)
    location.parent.mkdir(parents=True, exist_ok=True)
    with location.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    dict(row),
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    allow_nan=False,
                )
                + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())


def identity(path: str | Path) -> dict[str, Any]:
    location = Path(path).resolve()
    return {
        "path": str(location),
        "bytes": location.stat().st_size,
        "sha256": sha256_file(location),
    }


def finite_float(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise CompletionError(f"{label} is not finite")
    return result


def require_sha(path: str | Path, expected: str, label: str) -> Path:
    location = Path(path).resolve()
    if not location.is_file():
        raise FileNotFoundError(location)
    observed = sha256_file(location)
    if observed != str(expected):
        raise CompletionError(
            f"{label} changed: expected={expected}, observed={observed}"
        )
    return location


def require_hex_sha(value: str, label: str) -> str:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise CompletionError(f"{label} must be one lowercase SHA256")
    return value


def require_source_manifest(source_dir: Path, expected_manifest_sha256: str) -> Path:
    manifest = source_dir / "SOURCE_SHA256.txt"
    require_sha(manifest, expected_manifest_sha256, "completion source manifest")
    entries: list[tuple[str, str]] = []
    for line_number, line in enumerate(
        manifest.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        pieces = line.split("  ", 1)
        if len(pieces) != 2:
            raise CompletionError(f"{manifest}:{line_number}: malformed entry")
        expected, relative = pieces
        require_hex_sha(expected, f"{manifest}:{line_number}")
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise CompletionError(f"{manifest}:{line_number}: unsafe path")
        entries.append((expected, relative))
    listed = {relative for _, relative in entries}
    observed = {
        path.relative_to(source_dir).as_posix()
        for path in source_dir.rglob("*")
        if path.is_file()
        and path.name not in {"SOURCE_SHA256.txt"}
        and "__pycache__" not in path.parts
        and not path.name.endswith(".pyc")
    }
    if listed != observed:
        raise CompletionError(
            "completion source file set changed: "
            f"missing={sorted(listed - observed)}, extra={sorted(observed - listed)}"
        )
    for expected, relative in entries:
        require_sha(source_dir / relative, expected, f"source file {relative}")
    return manifest


def read_and_destroy_api_key(path: Path) -> str:
    """Read a private regular file and unlink it before returning the key."""

    location = path.resolve(strict=True)
    details = location.lstat()
    try:
        if not stat.S_ISREG(details.st_mode):
            raise CompletionError("API key carrier must be a regular file")
        if details.st_uid != os.getuid():
            raise CompletionError("API key carrier is not owned by this user")
        if stat.S_IMODE(details.st_mode) & 0o077:
            raise CompletionError("API key carrier permissions must be 0600 or stricter")
        if details.st_size <= 0 or details.st_size > 256:
            raise CompletionError("API key carrier size is invalid")
        key = location.read_text(encoding="ascii").strip()
    finally:
        location.unlink(missing_ok=True)
    if len(key) != 32 or any(character.isspace() for character in key):
        raise CompletionError("Materials Project API key shape is invalid")
    return key


def clean_composition(value: Mapping[str, Any]) -> dict[str, float]:
    result = {
        str(element): finite_float(amount, f"composition[{element}]")
        for element, amount in value.items()
        if not str(element).startswith("@")
    }
    if not result or any(amount <= 0.0 for amount in result.values()):
        raise CompletionError("composition must have positive element amounts")
    return dict(sorted(result.items()))


def chemsys_for_composition(composition: Mapping[str, Any]) -> str:
    return "-".join(sorted(clean_composition(composition)))


def _resolve_source_file(
    source_run: Path, specification: Mapping[str, Any], label: str
) -> Path:
    path = (source_run / str(specification["path"])).resolve()
    if source_run != path and source_run not in path.parents:
        raise CompletionError(f"{label} escapes the frozen source run")
    return require_sha(path, str(specification["sha256"]), label)


def collect_arm_context(
    *,
    arm: str,
    arm_config: Mapping[str, Any],
    source_run: Path,
) -> dict[str, Any]:
    attempts_path = _resolve_source_file(
        source_run, arm_config["attempt_results"], f"{arm} attempt results"
    )
    relax_path = _resolve_source_file(
        source_run, arm_config["relax_results"], f"{arm} relaxation results"
    )
    manifest_path = _resolve_source_file(
        source_run, arm_config["input_manifest"], f"{arm} input manifest"
    )
    attempts = read_jsonl(attempts_path)
    relax_rows = read_jsonl(relax_path)
    manifest = read_json(manifest_path)
    expected = arm_config["expected_counts"]
    method = str(arm_config["method"])
    if (
        len(attempts) != int(expected["total_attempts"])
        or [int(row.get("generation_ordinal", -1)) for row in attempts]
        != list(range(int(expected["total_attempts"])))
        or len({str(row.get("attempt_id")) for row in attempts}) != len(attempts)
        or {str(row.get("method")) for row in attempts} != {method}
        or any(row.get("retry_or_replacement_used") is not False for row in attempts)
        or manifest.get("retry_or_replacement_used") is not False
    ):
        raise CompletionError(f"{arm} frozen all-attempt mapping changed")

    novel_unique = [
        row for row in attempts if (row.get("metrics") or {}).get("novel_unique") is True
    ]
    if (
        len(novel_unique) != int(expected["novel_unique"])
        or len(relax_rows) != len(novel_unique)
    ):
        raise CompletionError(f"{arm} novel-unique relaxation mapping changed")

    targets: list[dict[str, Any]] = []
    for local_index, (attempt, relax) in enumerate(zip(novel_unique, relax_rows)):
        if int(relax.get("local_index", -1)) != local_index:
            raise CompletionError(f"{arm} relaxation local index changed")
        energy = finite_float(relax["energy_per_atom"], f"{arm} energy")
        source_energy = finite_float(
            attempt["metrics"]["energy_per_atom"], f"{arm} source energy"
        )
        if not math.isclose(energy, source_energy, rel_tol=0.0, abs_tol=1e-12):
            raise CompletionError(f"{arm} CHGNet energy mapping changed")
        composition = clean_composition(relax["composition"])
        targets.append(
            {
                "arm": arm,
                "local_index": local_index,
                "attempt_id": str(attempt["attempt_id"]),
                "generation_ordinal": int(attempt["generation_ordinal"]),
                "composition": composition,
                "chemsys": chemsys_for_composition(composition),
                "energy_per_atom": energy,
                "source_evaluation_status": str(attempt["evaluation_status"]),
                "source_e_above_hull": attempt["metrics"].get("e_above_hull"),
            }
        )

    statuses = Counter(target["source_evaluation_status"] for target in targets)
    unknown_chemsys = {
        target["chemsys"]
        for target in targets
        if target["source_evaluation_status"] == "relaxation_or_hull_unknown"
    }
    if (
        statuses["evaluated"] != int(expected["source_evaluated"])
        or statuses["relaxation_or_hull_unknown"]
        != int(expected["source_hull_unknown"])
        or len(unknown_chemsys) != int(expected["distinct_missing_chemsys"])
        or set(statuses) != {"evaluated", "relaxation_or_hull_unknown"}
    ):
        raise CompletionError(f"{arm} source MP coverage changed")
    return {
        "arm": arm,
        "method": method,
        "attempts": attempts,
        "targets": targets,
        "unknown_chemsys": unknown_chemsys,
        "manifest": manifest,
        "paths": {
            "attempt_results": attempts_path,
            "relax_results": relax_path,
            "input_manifest": manifest_path,
        },
    }


def load_relevant_slim_cache(
    path: Path, wanted_chemsys: set[str]
) -> tuple[dict[str, list[dict[str, Any]] | None], set[str]]:
    relevant: dict[str, list[dict[str, Any]] | None] = {}
    all_chemsys: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                chemsys = str(row["chemsys"])
                entries = row.get("entries")
                if entries is not None and not isinstance(entries, list):
                    raise TypeError("entries is not a list or null")
            except Exception as exc:
                raise CompletionError(
                    f"base MP cache line {line_number} is invalid: {type(exc).__name__}"
                ) from None
            all_chemsys.add(chemsys)
            if chemsys in wanted_chemsys:
                relevant[chemsys] = entries
    return relevant, all_chemsys


def audit_source_cache_coverage(
    *,
    cached: Mapping[str, list[dict[str, Any]] | None],
    all_cached_chemsys: set[str],
    wanted_chemsys: set[str],
    source_unknown_chemsys: set[str],
) -> dict[str, int]:
    """Validate reusable rows without treating stale unknown rows as resolved.

    A source attempt can be hull-unknown even when the append-only cache already
    contains a populated row for that chemsys (for example, an older conflicting
    row that did not yield a usable phase diagram).  Every source-unknown
    chemsys is therefore queried again and the fresh result overrides the old
    row.  Only source-evaluated chemsys are required to have a populated
    reusable cache row.
    """

    source_evaluated_chemsys = wanted_chemsys - source_unknown_chemsys
    unusable_evaluated = {
        chemsys
        for chemsys in source_evaluated_chemsys
        if not cached.get(chemsys)
    }
    if unusable_evaluated:
        raise CompletionError(
            "a source-evaluated chemsys lacks a populated reusable cache row"
        )
    return {
        "source_evaluated_chemsys": len(source_evaluated_chemsys),
        "source_unknown_chemsys": len(source_unknown_chemsys),
        "base_cache_records_for_source_unknown": len(
            source_unknown_chemsys & all_cached_chemsys
        ),
        "base_cache_populated_records_for_source_unknown": sum(
            bool(cached.get(chemsys)) for chemsys in source_unknown_chemsys
        ),
    }


def slim_entries(entries: Sequence[Any]) -> list[dict[str, Any]]:
    result = []
    for entry in entries:
        result.append(
            {
                "entry_id": (
                    str(entry.entry_id) if getattr(entry, "entry_id", None) is not None else None
                ),
                "composition": {
                    str(element): float(amount)
                    for element, amount in entry.composition.as_dict().items()
                },
                "energy": finite_float(entry.energy, "MP entry energy"),
            }
        )
    result.sort(
        key=lambda row: (
            str(row["entry_id"]),
            json.dumps(row["composition"], sort_keys=True, separators=(",", ":")),
            float(row["energy"]),
        )
    )
    return result


def chemsys_query_values(chemsys: str) -> list[str]:
    """Match MPRester.get_entries_in_chemsys by including every subsystem."""

    elements = chemsys.split("-")
    if not elements or any(not element for element in elements):
        raise CompletionError("invalid chemical system")
    return [
        "-".join(sorted(subsystem))
        for size in range(1, len(elements) + 1)
        for subsystem in itertools.combinations(elements, size)
    ]


def strip_unavailable_optional_mson_tags(
    value: Any,
    stripped_modules: set[str],
    redirected_modules: set[str] | None = None,
) -> Any:
    """Normalize moved classes and keep absent optional metadata as plain JSON."""

    if isinstance(value, list):
        return [
            strip_unavailable_optional_mson_tags(
                item, stripped_modules, redirected_modules
            )
            for item in value
        ]
    if not isinstance(value, dict):
        return value

    module = value.get("@module")
    if isinstance(module, str):
        redirected = MSON_MODULE_REDIRECTS.get(module, module)
        if redirected != module:
            if redirected_modules is not None:
                redirected_modules.add(f"{module}->{redirected}")
            value = dict(value)
            value["@module"] = redirected
            module = redirected
        try:
            importlib.import_module(module)
        except ModuleNotFoundError as exc:
            missing_name = str(exc.name or "")
            optional = module.startswith(
                OPTIONAL_MSON_MODULE_PREFIXES
            ) or missing_name.startswith(OPTIONAL_MSON_MODULE_PREFIXES)
            if not optional:
                raise
            stripped_modules.add(module)
            return {
                key: strip_unavailable_optional_mson_tags(
                    item, stripped_modules, redirected_modules
                )
                for key, item in value.items()
                if key not in {"@module", "@class", "@version"}
            }
    return {
        key: strip_unavailable_optional_mson_tags(
            item, stripped_modules, redirected_modules
        )
        for key, item in value.items()
    }


class CurrentMPThermoClient:
    """Current-API adapter preserving MPRester compatible-entry semantics."""

    def __init__(self, api_key: str) -> None:
        import requests

        self.session = requests.Session()
        self.session.headers = {
            "x-api-key": api_key,
            "user-agent": (
                "h1a2c-sun-mp-completion/1 "
                f"pymatgen/{importlib.metadata.version('pymatgen')}"
            ),
        }

    def __enter__(self) -> "CurrentMPThermoClient":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.session.close()

    def get_entries_in_chemsys(
        self, chemsys: str
    ) -> tuple[list[Any], dict[str, Any]]:
        from monty.json import MontyDecoder
        from pymatgen.entries.compatibility import (
            MaterialsProject2020Compatibility,
        )
        from pymatgen.entries.computed_entries import ComputedEntry

        query_values = chemsys_query_values(chemsys)
        documents: list[dict[str, Any]] = []
        skip = 0
        total_documents: int | None = None
        pages = 0
        while True:
            response = self.session.get(
                CURRENT_MP_THERMO_ENDPOINT,
                params={
                    "_fields": "entries",
                    "chemsys": ",".join(query_values),
                    "_limit": CURRENT_MP_PAGE_LIMIT,
                    "_skip": skip,
                },
                timeout=(10, 60),
            )
            response.raise_for_status()
            try:
                payload = response.json()
            except ValueError:
                raise CompletionError(
                    "Materials Project thermo response is not valid JSON"
                ) from None
            page = payload.get("data")
            metadata = payload.get("meta")
            if not isinstance(page, list) or not isinstance(metadata, dict):
                raise CompletionError(
                    "Materials Project thermo response schema changed"
                )
            if any(not isinstance(document, dict) for document in page):
                raise CompletionError(
                    "Materials Project thermo document schema changed"
                )
            pages += 1
            documents.extend(page)
            try:
                total_documents = int(metadata["total_doc"])
            except (KeyError, TypeError, ValueError):
                raise CompletionError(
                    "Materials Project thermo response lacks total_doc"
                ) from None
            skip += len(page)
            if skip >= total_documents:
                break
            if not page:
                raise CompletionError(
                    "Materials Project thermo pagination made no progress"
                )

        raw_entries: list[dict[str, Any]] = []
        for document in documents:
            variants = document.get("entries")
            if not isinstance(variants, dict):
                raise CompletionError(
                    "Materials Project thermo entries schema changed"
                )
            if any(not isinstance(entry, dict) for entry in variants.values()):
                raise CompletionError(
                    "Materials Project thermo entry schema changed"
                )
            raw_entries.extend(variants.values())

        stripped_modules: set[str] = set()
        redirected_modules: set[str] = set()
        decoded_entries: list[Any] = []
        decoder = MontyDecoder()
        for raw_entry in raw_entries:
            sanitized = strip_unavailable_optional_mson_tags(
                raw_entry, stripped_modules, redirected_modules
            )
            decoded = decoder.process_decoded(sanitized)
            if not isinstance(decoded, ComputedEntry):
                raise CompletionError(
                    "Materials Project thermo entry did not decode as ComputedEntry"
                )
            decoded_entries.append(decoded)

        compatibility = MaterialsProject2020Compatibility()
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Failed to guess oxidation states.*"
            )
            compatible_entries = compatibility.process_entries(
                decoded_entries, clean=True
            )
        deduplicated = list(set(compatible_entries))
        return deduplicated, {
            "api_documents": len(documents),
            "api_total_documents": total_documents,
            "api_pages": pages,
            "raw_entry_variants": len(raw_entries),
            "compatible_entries": len(deduplicated),
            "mson_module_redirects": sorted(redirected_modules),
            "optional_mson_modules_stripped": sorted(stripped_modules),
            "subsystem_chemsys_count": len(query_values),
        }


def sanitized_query_error(exc: BaseException) -> dict[str, Any]:
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    return {
        "type": type(exc).__name__,
        "http_status": int(status_code) if status_code is not None else None,
        "message_serialized": False,
    }


def query_missing_chemsys(
    *,
    api_key: str,
    missing: Sequence[str],
    completed_cache_path: Path,
    progress_path: Path,
    maximum_attempts: int,
) -> tuple[dict[str, list[dict[str, Any]] | None], list[dict[str, Any]]]:
    results: dict[str, list[dict[str, Any]] | None] = {}
    progress: list[dict[str, Any]] = []
    with completed_cache_path.open("a", encoding="utf-8") as cache_handle, progress_path.open(
        "x", encoding="utf-8"
    ) as progress_handle:
        with CurrentMPThermoClient(api_key) as client:
            for query_index, chemsys in enumerate(missing, start=1):
                started = time.monotonic()
                entries: list[dict[str, Any]] | None = None
                response_audit: dict[str, Any] | None = None
                final_error: dict[str, Any] | None = None
                attempts_used = 0
                for transport_attempt in range(1, maximum_attempts + 1):
                    attempts_used = transport_attempt
                    try:
                        raw_entries, response_audit = client.get_entries_in_chemsys(
                            chemsys
                        )
                        entries = slim_entries(raw_entries)
                        final_error = None
                        break
                    except Exception as exc:
                        if isinstance(exc, CompletionError):
                            raise
                        final_error = sanitized_query_error(exc)
                        if final_error["http_status"] in {401, 403}:
                            raise CompletionError(
                                "Materials Project authorization was rejected "
                                f"with HTTP {final_error['http_status']}"
                            ) from None
                        if transport_attempt < maximum_attempts:
                            time.sleep(min(16, 2 ** (transport_attempt - 1)))

                status = (
                    "resolved"
                    if entries
                    else ("empty" if entries == [] else "query_error")
                )
                cache_record = {"chemsys": chemsys, "entries": entries}
                cache_handle.write(
                    json.dumps(
                        cache_record,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=True,
                        allow_nan=False,
                    )
                    + "\n"
                )
                cache_handle.flush()
                os.fsync(cache_handle.fileno())
                row = {
                    "schema": "h1a2c_mp_chemsys_query_v1",
                    "query_index": query_index,
                    "query_total": len(missing),
                    "chemsys": chemsys,
                    "status": status,
                    "entry_count": len(entries) if entries is not None else None,
                    "transport_attempts": attempts_used,
                    "transport_retries": max(0, attempts_used - 1),
                    "sample_retry_or_replacement_used": False,
                    "elapsed_seconds": time.monotonic() - started,
                    "error": final_error,
                    "response_audit": response_audit,
                    "api_key_serialized": False,
                }
                progress_handle.write(
                    json.dumps(
                        row,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=True,
                        allow_nan=False,
                    )
                    + "\n"
                )
                progress_handle.flush()
                os.fsync(progress_handle.fileno())
                results[chemsys] = entries
                progress.append(row)
                print(
                    json.dumps(
                        {
                            "query_index": query_index,
                            "query_total": len(missing),
                            "chemsys": chemsys,
                            "status": status,
                            "entry_count": row["entry_count"],
                            "transport_attempts": attempts_used,
                        },
                        sort_keys=True,
                    ),
                        flush=True,
                    )
    return results, progress


def compute_hulls(
    targets: Sequence[Mapping[str, Any]],
    slim_cache: Mapping[str, list[dict[str, Any]] | None],
) -> dict[tuple[str, str], float | None]:
    from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry
    from pymatgen.entries.computed_entries import ComputedEntry
    from pymatgen.core import Composition

    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for target in targets:
        grouped.setdefault(str(target["chemsys"]), []).append(target)
    results: dict[tuple[str, str], float | None] = {}
    for chemsys, items in grouped.items():
        slim = slim_cache.get(chemsys)
        phase_diagram = None
        if slim:
            try:
                phase_diagram = PhaseDiagram(
                    [
                        ComputedEntry(
                            row["composition"],
                            finite_float(row["energy"], "cached MP energy"),
                            entry_id=row.get("entry_id"),
                        )
                        for row in slim
                    ]
                )
            except Exception:
                phase_diagram = None
        for target in items:
            key = (str(target["arm"]), str(target["attempt_id"]))
            if phase_diagram is None:
                results[key] = None
                continue
            composition = Composition(target["composition"])
            entry = PDEntry(
                composition,
                finite_float(target["energy_per_atom"], "CHGNet energy")
                * composition.num_atoms,
            )
            try:
                _, raw = phase_diagram.get_decomp_and_e_above_hull(
                    entry, allow_negative=True
                )
                value = max(finite_float(raw, "e_above_hull"), 0.0)
            except Exception:
                value = None
            results[key] = value
    return results


def complete_arm_attempts(
    *,
    context: Mapping[str, Any],
    hulls: Mapping[tuple[str, str], float | None],
    strict_threshold: float,
    meta_threshold: float,
    execution_manifest_sha256: str,
    queried_chemsys: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    targets_by_attempt = {
        str(target["attempt_id"]): target for target in context["targets"]
    }
    completed: list[dict[str, Any]] = []
    source_parity = 0
    source_unknown_resolved = 0
    for source in context["attempts"]:
        row = copy.deepcopy(source)
        row["schema"] = "crysllmgen_r5c_a100_sun_mp_completed_attempt_v1"
        row["source_attempt_sha256"] = canonical_sha256(source)
        row["mp_completion_execution_manifest_sha256"] = execution_manifest_sha256
        attempt_id = str(source["attempt_id"])
        target = targets_by_attempt.get(attempt_id)
        if target is None:
            row["mp_completion"] = {
                "applicable": False,
                "reason": "not_novel_unique",
                "generation_rerun": False,
                "chgnet_rerun": False,
                "sample_retry_or_replacement_used": False,
            }
            completed.append(row)
            continue

        e_hull = hulls[(str(context["arm"]), attempt_id)]
        source_value = target["source_e_above_hull"]
        if source_value is not None:
            if e_hull is None or not math.isclose(
                finite_float(source_value, "source e_above_hull"),
                e_hull,
                rel_tol=0.0,
                abs_tol=1e-10,
            ):
                raise CompletionError(
                    f"{context['arm']} cached source e_hull parity failed"
                )
            source_parity += 1
        elif e_hull is not None:
            source_unknown_resolved += 1

        row["metrics"]["e_above_hull"] = e_hull
        row["metrics"]["strict_full_sun"] = (
            e_hull is not None and e_hull <= strict_threshold
        )
        row["metrics"]["meta_full_sun"] = (
            e_hull is not None and e_hull <= meta_threshold
        )
        row["evaluation_status"] = (
            "evaluated" if e_hull is not None else "relaxation_or_hull_unknown"
        )
        row["mp_completion"] = {
            "applicable": True,
            "chemsys": target["chemsys"],
            "reference_source": (
                "materials_project_api_completion"
                if target["chemsys"] in queried_chemsys
                else "frozen_mp_hull_cache"
            ),
            "source_evaluation_status": target["source_evaluation_status"],
            "source_e_above_hull": source_value,
            "completed_e_above_hull": e_hull,
            "generation_rerun": False,
            "chgnet_rerun": False,
            "chgnet_energy_reused": True,
            "sample_retry_or_replacement_used": False,
        }
        completed.append(row)

    counts = {
        "total_attempts": len(completed),
        "reconstructed": sum(
            row.get("generation_status") == "succeeded" for row in completed
        ),
        "novel": sum(
            (row.get("metrics") or {}).get("novel") is True for row in completed
        ),
        "unique": sum(
            (row.get("metrics") or {}).get("unique_representative") is True
            for row in completed
        ),
        "novel_unique": sum(
            (row.get("metrics") or {}).get("novel_unique") is True
            for row in completed
        ),
        "strict_full_sun": sum(
            (row.get("metrics") or {}).get("strict_full_sun") is True
            for row in completed
        ),
        "meta_full_sun": sum(
            (row.get("metrics") or {}).get("meta_full_sun") is True
            for row in completed
        ),
        "relaxation_or_hull_unknown": sum(
            row.get("evaluation_status") == "relaxation_or_hull_unknown"
            for row in completed
        ),
        "source_evaluated_parity": source_parity,
        "source_unknown_resolved": source_unknown_resolved,
    }
    if (
        counts["total_attempts"] != 256
        or counts["strict_full_sun"] > counts["meta_full_sun"]
        or counts["novel_unique"]
        != len(context["targets"])
    ):
        raise CompletionError(f"{context['arm']} completed count invariants failed")
    return completed, counts


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * float(probability)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return (
        float(sorted_values[lower]) * (1.0 - fraction)
        + float(sorted_values[upper]) * fraction
    )


def exact_mcnemar(
    candidate: Sequence[bool], baseline: Sequence[bool]
) -> dict[str, Any]:
    candidate_only = sum(
        bool(left) and not bool(right) for left, right in zip(candidate, baseline)
    )
    baseline_only = sum(
        not bool(left) and bool(right) for left, right in zip(candidate, baseline)
    )
    discordant = candidate_only + baseline_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(
            math.comb(discordant, value)
            for value in range(min(candidate_only, baseline_only) + 1)
        ) / (2**discordant)
        p_value = min(1.0, 2.0 * tail)
    return {
        "candidate_only": candidate_only,
        "baseline_only": baseline_only,
        "discordant": discordant,
        "two_sided_exact_p_value": p_value,
    }


def paired_effect(
    candidate: Sequence[bool],
    baseline: Sequence[bool],
    *,
    seed_offset: int,
) -> dict[str, Any]:
    if len(candidate) != 256 or len(baseline) != 256:
        raise CompletionError("paired effects require 256 registered pairs")
    differences = [
        float(bool(left)) - float(bool(right))
        for left, right in zip(candidate, baseline)
    ]
    rng = random.Random(BOOTSTRAP_SEED + seed_offset)
    samples = [
        100.0 * sum(differences[rng.randrange(256)] for _ in range(256)) / 256
        for _ in range(BOOTSTRAP_DRAWS)
    ]
    samples.sort()
    candidate_count = sum(bool(value) for value in candidate)
    baseline_count = sum(bool(value) for value in baseline)
    return {
        "attempts": 256,
        "candidate_arm": "P1",
        "baseline_arm": "P0",
        "candidate_count": candidate_count,
        "baseline_count": baseline_count,
        "candidate_rate": candidate_count / 256,
        "baseline_rate": baseline_count / 256,
        "difference_percentage_points": 100.0
        * (candidate_count - baseline_count)
        / 256,
        "bootstrap": {
            "draws": BOOTSTRAP_DRAWS,
            "seed": BOOTSTRAP_SEED + seed_offset,
            "ci95_lower_percentage_points": _quantile(samples, 0.025),
            "ci95_upper_percentage_points": _quantile(samples, 0.975),
        },
        "exact_mcnemar": exact_mcnemar(candidate, baseline),
    }


def _gate_passed(gate: Mapping[str, Any]) -> bool:
    observed = float(gate["observed_pp"])
    threshold = float(gate["threshold_pp"])
    if gate["operator"] == ">=":
        return observed >= threshold
    if gate["operator"] == "<=":
        return observed <= threshold
    raise CompletionError(f"unsupported screening operator: {gate['operator']}")


def execute(args: argparse.Namespace) -> dict[str, Any]:
    if os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOB_NAME"):
        raise CompletionError("MP completion is login-node-only and forbids Slurm")
    execution_sha = require_hex_sha(
        args.execution_manifest_sha256, "completion execution manifest"
    )
    project_root = args.project_root.resolve()
    source_dir = args.source_dir.resolve()
    require_source_manifest(source_dir, execution_sha)
    config = read_json(args.config.resolve())
    if (
        config.get("status") != "user_authorized_manual_mp_completion"
        or config["execution"]["slurm_allowed"] is not False
        or config["execution"]["gpu_allowed"] is not False
        or config["authorization"]["new_generation_authorized"] is not False
        or config["authorization"]["chgnet_rerun_authorized"] is not False
    ):
        raise CompletionError("manual completion authorization contract changed")
    observed_pymatgen = importlib.metadata.version("pymatgen")
    if observed_pymatgen != config["materials_project"]["pymatgen_version"]:
        raise CompletionError(
            f"pymatgen version changed: {observed_pymatgen}"
        )

    source_run = (
        project_root / str(config["source_run"]["run_root"])
    ).resolve()
    terminal_path = _resolve_source_file(
        source_run,
        config["source_run"]["terminal_report"],
        "source terminal report",
    )
    ledger_path = _resolve_source_file(
        source_run,
        config["source_run"]["attempt_ledger"],
        "source attempt ledger",
    )
    base_cache_path = _resolve_source_file(
        source_run,
        config["source_run"]["base_mp_hull_cache"],
        "source MP hull cache",
    )
    source_terminal = read_json(terminal_path)
    if (
        source_terminal.get("execution_manifest_sha256")
        != config["source_run"]["execution_manifest_sha256"]
        or source_terminal.get("attempts_per_arm") != 256
        or source_terminal.get("retry_or_replacement_used") is not False
    ):
        raise CompletionError("source terminal identity changed")
    if len(read_jsonl(ledger_path)) != 256:
        raise CompletionError("source paired ledger denominator changed")

    contexts = {
        arm: collect_arm_context(
            arm=arm,
            arm_config=config["source_run"]["arms"][arm],
            source_run=source_run,
        )
        for arm in ARM_ORDER
    }
    overlap = contexts["P0"]["unknown_chemsys"] & contexts["P1"]["unknown_chemsys"]
    missing_union = contexts["P0"]["unknown_chemsys"] | contexts["P1"]["unknown_chemsys"]
    if (
        len(overlap)
        != int(config["source_run"]["expected_distinct_missing_chemsys_overlap"])
        or len(missing_union)
        != int(config["source_run"]["expected_distinct_missing_chemsys_union"])
        or len(missing_union) > int(config["materials_project"]["maximum_queries"])
    ):
        raise CompletionError("deduplicated missing-chemsys scope changed")

    all_targets = [
        target for arm in ARM_ORDER for target in contexts[arm]["targets"]
    ]
    wanted_chemsys = {str(target["chemsys"]) for target in all_targets}
    cached, all_cached_chemsys = load_relevant_slim_cache(
        base_cache_path, wanted_chemsys
    )
    source_cache_audit = audit_source_cache_coverage(
        cached=cached,
        all_cached_chemsys=all_cached_chemsys,
        wanted_chemsys=wanted_chemsys,
        source_unknown_chemsys=missing_union,
    )

    run_root = args.run_root.resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    existing_top_level = {path.name for path in run_root.iterdir()}
    if existing_top_level - {"logs"}:
        raise CompletionError("completion run root is not empty")
    created_at = datetime.now(timezone.utc).isoformat()
    claim = {
        "schema": "h1a2c_p0_p1_sun256_mp_completion_claim_v1",
        "status": "claimed_before_external_queries",
        "created_at_utc": created_at,
        "run_id": config["run_id"],
        "execution_manifest_sha256": execution_sha,
        "source_terminal": identity(terminal_path),
        "attempts_per_arm": 256,
        "distinct_missing_chemsys": len(missing_union),
        "logical_mp_queries": len(missing_union),
        "source_cache_audit": source_cache_audit,
        "api_key_present": True,
        "api_key_serialized": False,
        "slurm_used": False,
        "gpu_used": False,
        "new_generation": False,
        "chgnet_calls": 0,
        "sample_retry_or_replacement_used": False,
        "automatic_downstream_authorized": False,
    }
    write_json_exclusive(run_root / "claim.json", claim)

    api_key = read_and_destroy_api_key(args.key_file)
    cache_output = run_root / "mp_hull_cache_completed.jsonl"
    shutil.copyfile(base_cache_path, cache_output)
    query_results, query_progress = query_missing_chemsys(
        api_key=api_key,
        missing=sorted(missing_union),
        completed_cache_path=cache_output,
        progress_path=run_root / "mp_query_progress.jsonl",
        maximum_attempts=int(
            config["materials_project"]["maximum_transport_attempts_per_chemsys"]
        ),
    )
    api_key = ""
    if (
        set(query_results) != missing_union
        or len(query_progress) != len(missing_union)
    ):
        raise CompletionError("Materials Project query coverage is incomplete")
    cached.update(query_results)
    hulls = compute_hulls(all_targets, cached)

    strict_threshold = float(config["sun"]["strict_threshold_ev_per_atom"])
    meta_threshold = float(config["sun"]["meta_threshold_ev_per_atom"])
    completed_by_arm: dict[str, list[dict[str, Any]]] = {}
    counts_by_arm: dict[str, dict[str, Any]] = {}
    for arm in ARM_ORDER:
        completed, counts = complete_arm_attempts(
            context=contexts[arm],
            hulls=hulls,
            strict_threshold=strict_threshold,
            meta_threshold=meta_threshold,
            execution_manifest_sha256=execution_sha,
            queried_chemsys=missing_union,
        )
        arm_dir = run_root / "arms" / arm
        attempts_output = arm_dir / "attempt_results.jsonl"
        write_jsonl_exclusive(attempts_output, completed)
        summary = {
            "schema": "h1a2c_p0_p1_sun256_mp_completed_arm_v1",
            "ok": True,
            "arm": arm,
            "method": contexts[arm]["method"],
            "counts": counts,
            "rates_all_attempts": {
                "strict_full_sun": counts["strict_full_sun"] / 256,
                "meta_full_sun": counts["meta_full_sun"] / 256,
                "novel_unique": counts["novel_unique"] / 256,
                "remaining_hull_unknown": counts["relaxation_or_hull_unknown"]
                / 256,
            },
            "denominator": "all_registered_attempts",
            "attempt_results": identity(attempts_output),
            "source_inputs": {
                name: identity(path) for name, path in contexts[arm]["paths"].items()
            },
            "generation_rerun": False,
            "chgnet_rerun": False,
            "sample_retry_or_replacement_used": False,
            "api_key_serialized": False,
        }
        write_json_exclusive(arm_dir / "summary.json", summary)
        completed_by_arm[arm] = completed
        counts_by_arm[arm] = counts

    vectors: dict[str, dict[str, list[bool]]] = {}
    for arm in ARM_ORDER:
        vectors[arm] = {
            "sun_strict": [
                row["metrics"]["strict_full_sun"]
                for row in completed_by_arm[arm]
            ],
            "sun_meta": [
                row["metrics"]["meta_full_sun"]
                for row in completed_by_arm[arm]
            ],
        }
    effects = {
        "sun_strict": paired_effect(
            vectors["P1"]["sun_strict"],
            vectors["P0"]["sun_strict"],
            seed_offset=STRICT_SEED_OFFSET,
        ),
        "sun_meta": paired_effect(
            vectors["P1"]["sun_meta"],
            vectors["P0"]["sun_meta"],
            seed_offset=META_SEED_OFFSET,
        ),
    }
    gates = copy.deepcopy(source_terminal["screening_gates"])
    gates["strict_sun_noninferiority"]["observed_pp"] = effects["sun_strict"][
        "difference_percentage_points"
    ]
    gates["strict_sun_noninferiority"]["passed"] = _gate_passed(
        gates["strict_sun_noninferiority"]
    )
    gates["meta_sun_gain"]["observed_pp"] = effects["sun_meta"][
        "difference_percentage_points"
    ]
    gates["meta_sun_gain"]["passed"] = _gate_passed(gates["meta_sun_gain"])
    screening_passed = all(bool(value["passed"]) for value in gates.values())
    failed_gates = [
        name for name, value in gates.items() if not bool(value["passed"])
    ]
    decision = (
        "exploratory_support_only_formal_promotion_ineligible"
        if screening_passed
        else "stop_exploratory_screen"
    )
    query_counts = Counter(row["status"] for row in query_progress)
    remaining_unknown = sum(
        counts_by_arm[arm]["relaxation_or_hull_unknown"] for arm in ARM_ORDER
    )
    completion_status = (
        "complete_all_missing_hulls_resolved"
        if remaining_unknown == 0
        else "complete_with_remaining_hull_unknowns"
    )
    terminal = {
        "schema": "h1a2c_p0_p1_sun256_mp_completion_terminal_v1",
        "ok": True,
        "status": completion_status,
        "created_at_utc": created_at,
        "run_id": config["run_id"],
        "source_run_id": source_terminal["run_id"],
        "source_terminal": identity(terminal_path),
        "attempts_per_arm": 256,
        "denominator": "all_registered_attempts",
        "arms": {
            arm: {
                "method": contexts[arm]["method"],
                "counts": counts_by_arm[arm],
                "rates_all_attempts": {
                    "strict_full_sun": counts_by_arm[arm]["strict_full_sun"] / 256,
                    "meta_full_sun": counts_by_arm[arm]["meta_full_sun"] / 256,
                    "remaining_hull_unknown": counts_by_arm[arm][
                        "relaxation_or_hull_unknown"
                    ]
                    / 256,
                },
                "attempt_results": identity(
                    run_root / "arms" / arm / "attempt_results.jsonl"
                ),
                "summary": identity(run_root / "arms" / arm / "summary.json"),
            }
            for arm in ARM_ORDER
        },
        "mp_completion": {
            "client": config["materials_project"]["client"],
            "method": config["materials_project"]["method"],
            "compatible_only": True,
            "pymatgen_version": importlib.metadata.version("pymatgen"),
            "distinct_missing_chemsys": len(missing_union),
            "distinct_overlap_between_arms": len(overlap),
            "source_cache_audit": source_cache_audit,
            "logical_queries_submitted": len(query_progress),
            "query_counts": dict(query_counts),
            "transport_attempts_total": sum(
                int(row["transport_attempts"]) for row in query_progress
            ),
            "transport_retries_total": sum(
                int(row["transport_retries"]) for row in query_progress
            ),
            "query_progress": identity(run_root / "mp_query_progress.jsonl"),
            "completed_cache": identity(cache_output),
            "api_key_serialized": False,
        },
        "paired_effects_P1_minus_P0": effects,
        "screening_gates": gates,
        "screening_passed": screening_passed,
        "failed_screening_gates": failed_gates,
        "decision": decision,
        "formal_promotion_eligible": False,
        "formal_ineligibility_reasons": source_terminal[
            "formal_ineligibility_reasons"
        ],
        "execution": {
            "location": config["execution"]["location"],
            "slurm_used": False,
            "gpu_used": False,
            "new_generation": False,
            "chgnet_calls": 0,
            "chgnet_energy_reused": True,
            "sample_retry_or_replacement_used": False,
        },
        "manual_mp_reference_completion_authorized": True,
        "automatic_crystal_evaluation_authorized": False,
        "automatic_promotion_authorized": False,
        "automatic_downstream_authorized": False,
        "recommended_next_step": (
            "Review the exact all-attempt paired result before authorizing any new experiment."
        ),
    }
    final_dir = run_root / "final"
    terminal_path_output = final_dir / "terminal_report.json"
    write_json_exclusive(terminal_path_output, terminal)
    decision_record = {
        "schema": "h1a2c_p0_p1_sun256_mp_completion_decision_v1",
        "status": completion_status,
        "decision": decision,
        "screening_passed": screening_passed,
        "failed_screening_gates": failed_gates,
        "formal_promotion_eligible": False,
        "terminal_report_sha256": sha256_file(terminal_path_output),
        "automatic_crystal_evaluation_authorized": False,
        "automatic_downstream_authorized": False,
    }
    write_json_exclusive(final_dir / "decision.json", decision_record)
    with (final_dir / "_SUCCESS").open("x", encoding="ascii") as handle:
        handle.flush()
        os.fsync(handle.fileno())
    return decision_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--execution-manifest-sha256", required=True)
    return parser.parse_args()


def main() -> None:
    result = execute(parse_args())
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
