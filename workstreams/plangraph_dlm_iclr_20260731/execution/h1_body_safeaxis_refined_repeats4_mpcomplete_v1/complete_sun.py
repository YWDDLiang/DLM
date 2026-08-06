#!/usr/bin/env python3
"""Complete missing MP hull references for frozen R03E repeated S.U.N.

The R03E source is immutable.  This program performs no generation, refinement,
CHGNet relaxation, direct metric, novelty, filtering, ranking, training, or
sample retry/replacement work.  It queries the union of missing chemical
systems once logically (bounded transport retries only), builds one common
thermodynamic snapshot, and maps that snapshot back to all 4 repeats x 2 arms x
256 original attempts.  The API key is read from a user-owned mode-0600 file,
unlinked immediately, and never serialized or printed.
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
import stat
import time
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPEATS = (0, 1, 2, 3)
ARMS = ("control", "candidate")
ARM_KEYS = tuple(
    f"r{repeat}_{arm}" for repeat in REPEATS for arm in ARMS
)
ENDPOINTS = ("strict_full_sun", "meta_full_sun")
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
        and ".pytest_cache" not in path.parts
        and not path.name.endswith((".pyc", ".pyo"))
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


def _resolve_frozen_file(
    source_run: Path, specification: Mapping[str, Any], label: str
) -> Path:
    configured = Path(str(specification["path"]))
    path = (
        configured.resolve()
        if configured.is_absolute()
        else (source_run / configured).resolve()
    )
    result = require_sha(path, str(specification["sha256"]), label)
    expected_bytes = specification.get("bytes")
    if expected_bytes is not None and result.stat().st_size != int(expected_bytes):
        raise CompletionError(f"{label} byte count changed")
    return result


def collect_arm_context(
    *,
    arm_key: str,
    arm_config: Mapping[str, Any],
    source_run: Path,
    attempt_ledger_sha256: str,
) -> dict[str, Any]:
    repeat = int(arm_config["repeat"])
    arm = str(arm_config["arm"])
    if arm_key != f"r{repeat}_{arm}" or repeat not in REPEATS or arm not in ARMS:
        raise CompletionError(f"{arm_key} frozen arm identity changed")
    attempts_path = _resolve_source_file(
        source_run, arm_config["attempt_results"], f"{arm_key} attempt results"
    )
    summary_path = _resolve_source_file(
        source_run, arm_config["attempt_summary"], f"{arm_key} attempt summary"
    )
    manifest_path = _resolve_source_file(
        source_run, arm_config["input_manifest"], f"{arm_key} input manifest"
    )
    strict_relax_path = _resolve_source_file(
        source_run,
        arm_config["strict_relax_results"],
        f"{arm_key} strict relaxation results",
    )
    meta_relax_path = _resolve_source_file(
        source_run,
        arm_config["meta_relax_results"],
        f"{arm_key} meta relaxation results",
    )
    evaluation_path = _resolve_source_file(
        source_run,
        arm_config["evaluation_report"],
        f"{arm_key} evaluation report",
    )
    preflight_path = _resolve_source_file(
        source_run,
        arm_config["preflight_report"],
        f"{arm_key} repeat preflight report",
    )
    attempts = read_jsonl(attempts_path)
    if (
        sha256_file(strict_relax_path) != sha256_file(meta_relax_path)
        or strict_relax_path.read_bytes() != meta_relax_path.read_bytes()
    ):
        raise CompletionError(
            f"{arm_key} strict/meta relaxation evidence is not identical"
        )
    relax_rows = read_jsonl(strict_relax_path)
    manifest = read_json(manifest_path)
    source_summary = read_json(summary_path)
    evaluation = read_json(evaluation_path)
    preflight = read_json(preflight_path)
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
        raise CompletionError(f"{arm_key} frozen all-attempt mapping changed")
    if (
        source_summary.get("ok") is not True
        or source_summary.get("method") != method
        or (source_summary.get("counts") or {}).get("total_attempts") != 256
        or source_summary.get("retry_or_replacement_used") is not False
        or evaluation.get("schema") != "h1_r03e_arm_evaluation_v1"
        or evaluation.get("status") != "complete"
        or evaluation.get("ok") is not True
        or int(evaluation.get("repeat", -1)) != repeat
        or evaluation.get("arm") != arm
        or evaluation.get("method") != method
        or int(evaluation.get("attempts", -1)) != 256
        or evaluation.get("all_generation_successes_diffusion_refined") is not True
        or int(evaluation.get("diffusion_steps", -1)) != 800
        or any(
            evaluation.get(name) is not False
            for name in (
                "formal_g3",
                "automatic_promotion",
                "automatic_training",
                "automatic_downstream",
            )
        )
        or preflight.get("schema") != "h1_r03e_repeat_preflight_v1"
        or preflight.get("status") != "pass"
        or int(preflight.get("repeat", -1)) != repeat
        or int(preflight.get("attempts_per_arm", -1)) != 256
        or preflight.get("attempt_ledger_sha256") != attempt_ledger_sha256
        or preflight.get("mp_api_enabled") is not False
        or preflight.get("generation_rerun") is not False
        or preflight.get("body_rerun") is not False
        or preflight.get("new_scientific_seed_per_repeat") is not False
    ):
        raise CompletionError(f"{arm_key} frozen reports changed")

    novel_unique = [
        row for row in attempts if (row.get("metrics") or {}).get("novel_unique") is True
    ]
    if (
        len(novel_unique) != int(expected["novel_unique"])
        or len(relax_rows) != len(novel_unique)
    ):
        raise CompletionError(f"{arm_key} novel-unique relaxation mapping changed")

    targets: list[dict[str, Any]] = []
    for local_index, (attempt, relax) in enumerate(zip(novel_unique, relax_rows)):
        if int(relax.get("local_index", -1)) != local_index:
            raise CompletionError(f"{arm_key} relaxation local index changed")
        energy = finite_float(relax["energy_per_atom"], f"{arm_key} energy")
        source_energy = finite_float(
            attempt["metrics"]["energy_per_atom"], f"{arm_key} source energy"
        )
        if not math.isclose(energy, source_energy, rel_tol=0.0, abs_tol=1e-12):
            raise CompletionError(f"{arm_key} CHGNet energy mapping changed")
        status = str(attempt["evaluation_status"])
        source_e_hull = attempt["metrics"].get("e_above_hull")
        if status == "evaluated":
            finite_float(source_e_hull, f"{arm_key} source e_above_hull")
        elif status == "relaxation_or_hull_unknown":
            if (
                source_e_hull is not None
                or attempt["metrics"].get("strict_full_sun") is not False
                or attempt["metrics"].get("meta_full_sun") is not False
            ):
                raise CompletionError(f"{arm_key} unknown hull semantics changed")
        else:
            raise CompletionError(f"{arm_key} source evaluation status changed")
        composition = clean_composition(relax["composition"])
        targets.append(
            {
                "arm_key": arm_key,
                "arm": arm,
                "repeat": repeat,
                "local_index": local_index,
                "attempt_id": str(attempt["attempt_id"]),
                "generation_ordinal": int(attempt["generation_ordinal"]),
                "composition": composition,
                "chemsys": chemsys_for_composition(composition),
                "energy_per_atom": energy,
                "source_evaluation_status": status,
                "source_e_above_hull": source_e_hull,
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
        raise CompletionError(f"{arm_key} source MP coverage changed")
    return {
        "arm_key": arm_key,
        "arm": arm,
        "repeat": repeat,
        "method": method,
        "attempts": attempts,
        "targets": targets,
        "unknown_chemsys": unknown_chemsys,
        "manifest": manifest,
        "paths": {
            "attempt_results": attempts_path,
            "attempt_summary": summary_path,
            "input_manifest": manifest_path,
            "strict_relax_results": strict_relax_path,
            "meta_relax_results": meta_relax_path,
            "evaluation_report": evaluation_path,
            "preflight_report": preflight_path,
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
    with completed_cache_path.open("x", encoding="utf-8") as cache_handle, progress_path.open(
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
                if status != "resolved":
                    raise CompletionError(
                        f"Materials Project query did not resolve {chemsys}: {status}"
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
            key = (str(target["arm_key"]), str(target["attempt_id"]))
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
                "refinement_rerun": False,
                "chgnet_rerun": False,
                "direct_metrics_rerun": False,
                "novelty_rerun": False,
                "sample_retry_or_replacement_used": False,
            }
            completed.append(row)
            continue

        e_hull = hulls[(str(context["arm_key"]), attempt_id)]
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
            "refinement_rerun": False,
            "chgnet_rerun": False,
            "chgnet_energy_reused": True,
            "direct_metrics_rerun": False,
            "novelty_rerun": False,
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
        raise CompletionError(
            f"{context['arm_key']} completed count invariants failed"
        )
    return completed, counts


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


def hierarchical_paired_bootstrap(
    differences: Sequence[Sequence[Sequence[float]]],
    *,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    import numpy as np

    matrix = np.asarray(differences, dtype=np.float64)
    if matrix.shape != (4, 256, 2):
        raise CompletionError("hierarchical bootstrap matrix shape changed")
    rng = np.random.default_rng(seed)
    samples = np.empty((replicates, 2), dtype=np.float64)
    for start in range(0, replicates, 500):
        stop = min(replicates, start + 500)
        size = stop - start
        repeat_draw = rng.integers(0, 4, size=(size, 4))
        ordinal_draw = rng.integers(0, 256, size=(size, 4, 256))
        values = matrix[repeat_draw[:, :, None], ordinal_draw, :]
        samples[start:stop] = values.mean(axis=(1, 2))
    lower, upper = np.quantile(samples, [0.025, 0.975], axis=0)
    return {
        endpoint: {
            "mean_delta": float(matrix[:, :, index].mean()),
            "hierarchical_paired_bootstrap_95ci": [
                float(lower[index]),
                float(upper[index]),
            ],
        }
        for index, endpoint in enumerate(ENDPOINTS)
    }


def _validate_authorization(config: Mapping[str, Any]) -> None:
    authorization = config["authorization"]
    if (
        config.get("status") != "user_authorized_evaluation_coverage_completion"
        or config["execution"]["slurm_allowed"] is not False
        or config["execution"]["gpu_allowed"] is not False
        or authorization["manual_mp_reference_completion_authorized"] is not True
        or any(
            authorization[name] is not False
            for name in (
                "new_generation_authorized",
                "refinement_rerun_authorized",
                "chgnet_rerun_authorized",
                "direct_metrics_rerun_authorized",
                "novelty_rerun_authorized",
                "automatic_crystal_evaluation_authorized",
                "checkpoint_reselection_authorized",
                "automatic_training_authorized",
                "automatic_promotion_authorized",
                "automatic_downstream_authorized",
            )
        )
        or any(config["decision_firewall"].values())
    ):
        raise CompletionError("R03F authorization firewall changed")


def _source_artifact_contract(
    terminal_path: Path,
    ledger_path: Path,
    base_cache_path: Path,
    base_cache_sha256: str,
    contexts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "source_terminal": identity(terminal_path),
        "attempt_ledger": identity(ledger_path),
        "base_mp_hull_cache": {
            "path": str(base_cache_path.resolve()),
            "bytes": base_cache_path.stat().st_size,
            "sha256": base_cache_sha256,
        },
        "arms": {
            arm_key: {
                name: identity(path)
                for name, path in context["paths"].items()
            }
            for arm_key, context in contexts.items()
        },
    }


def prepare_frozen_state(args: argparse.Namespace) -> dict[str, Any]:
    if os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOB_NAME"):
        raise CompletionError("MP completion is login-node-only and forbids Slurm")
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") not in {"", "-1"}:
        raise CompletionError("MP completion requires CUDA_VISIBLE_DEVICES empty")
    execution_sha = require_hex_sha(
        args.execution_manifest_sha256, "R03F source manifest"
    )
    source_dir = args.source_dir.resolve()
    require_source_manifest(source_dir, execution_sha)
    config = read_json(args.config.resolve())
    _validate_authorization(config)
    observed_pymatgen = importlib.metadata.version("pymatgen")
    if observed_pymatgen != config["materials_project"]["pymatgen_version"]:
        raise CompletionError(f"pymatgen version changed: {observed_pymatgen}")

    source_run = Path(str(config["source_run"]["run_root"])).resolve()
    if not source_run.is_dir():
        raise FileNotFoundError(source_run)
    terminal_path = _resolve_source_file(
        source_run,
        config["source_run"]["terminal_report"],
        "R03E terminal report",
    )
    ledger_path = _resolve_frozen_file(
        source_run,
        config["source_run"]["attempt_ledger"],
        "frozen H1 attempt ledger",
    )
    base_cache_path = _resolve_source_file(
        source_run,
        config["source_run"]["base_mp_hull_cache"],
        "frozen base MP hull cache",
    )
    source_terminal = read_json(terminal_path)
    if (
        source_terminal.get("schema")
        != "h1_r03e_refined_repeats4_terminal_report_v1"
        or source_terminal.get("status") != "complete"
        or int(source_terminal.get("repeat_count", -1)) != 4
        or int(source_terminal.get("attempts_per_arm_per_repeat", -1)) != 256
        or source_terminal.get("source_manifest_sha256")
        != config["source_run"]["source_manifest_sha256"]
        or any(
            source_terminal.get(name) is not False
            for name in (
                "formal_g3",
                "automatic_promotion",
                "automatic_training",
                "automatic_downstream",
            )
        )
    ):
        raise CompletionError("R03E terminal identity changed")
    ledger = read_jsonl(ledger_path)
    if len(ledger) != 256:
        raise CompletionError("frozen H1 attempt ledger denominator changed")

    configured_arms = config["source_run"]["arms"]
    if set(configured_arms) != set(ARM_KEYS):
        raise CompletionError("R03F eight-arm source mapping changed")
    contexts = {
        arm_key: collect_arm_context(
            arm_key=arm_key,
            arm_config=configured_arms[arm_key],
            source_run=source_run,
            attempt_ledger_sha256=sha256_file(ledger_path),
        )
        for arm_key in ARM_KEYS
    }
    missing_sets = [
        contexts[arm_key]["unknown_chemsys"] for arm_key in ARM_KEYS
    ]
    missing_union = set().union(*missing_sets)
    missing_union_sha = hashlib.sha256(
        "\n".join(sorted(missing_union)).encode("utf-8")
    ).hexdigest()
    systems_in_multiple_arms = sum(
        sum(system in values for values in missing_sets) > 1
        for system in missing_union
    )
    max_membership = max(
        sum(system in values for values in missing_sets)
        for system in missing_union
    )
    all_targets = [
        target
        for arm_key in ARM_KEYS
        for target in contexts[arm_key]["targets"]
    ]
    wanted_chemsys = {str(target["chemsys"]) for target in all_targets}
    if (
        len(missing_union)
        != int(config["source_run"]["expected_distinct_missing_chemsys_union"])
        or missing_union_sha
        != config["source_run"]["expected_distinct_missing_chemsys_union_sha256"]
        or len(wanted_chemsys)
        != int(config["source_run"]["expected_distinct_wanted_chemsys_union"])
        or systems_in_multiple_arms
        != int(config["source_run"]["expected_missing_systems_in_multiple_arms"])
        or max_membership
        != int(config["source_run"]["expected_max_missing_chemsys_membership"])
        or len(missing_union)
        > int(config["materials_project"]["maximum_queries"])
    ):
        raise CompletionError("deduplicated eight-arm missing-chemsys scope changed")
    cached, all_cached_chemsys = load_relevant_slim_cache(
        base_cache_path, wanted_chemsys
    )
    source_cache_audit = audit_source_cache_coverage(
        cached=cached,
        all_cached_chemsys=all_cached_chemsys,
        wanted_chemsys=wanted_chemsys,
        source_unknown_chemsys=missing_union,
    )
    configured_run_root = Path(str(config["run_root"])).resolve()
    if args.run_root.resolve() != configured_run_root:
        raise CompletionError("fixed R03F run root changed")
    if configured_run_root.exists() and args.preflight_only:
        raise CompletionError("fixed R03F run root already exists")
    if configured_run_root.exists() and not args.preflight_only:
        existing = {path.name for path in configured_run_root.iterdir()}
        if existing - {"logs"}:
            raise CompletionError("fixed R03F run root is not empty")
    input_contract = {
        "schema": "h1_r03f_frozen_input_contract_v1",
        "status": "pass",
        "run_id": config["run_id"],
        "execution_manifest_sha256": execution_sha,
        "source_artifacts": _source_artifact_contract(
            terminal_path,
            ledger_path,
            base_cache_path,
            config["source_run"]["base_mp_hull_cache"]["sha256"],
            contexts,
        ),
        "repeat_count": 4,
        "arm_count": 8,
        "attempts_per_arm": 256,
        "raw_attempts_total": 2048,
        "missing_chemsys_union": len(missing_union),
        "missing_chemsys_union_sha256": missing_union_sha,
        "wanted_chemsys_union": len(wanted_chemsys),
        "systems_in_multiple_arms": systems_in_multiple_arms,
        "max_missing_chemsys_membership": max_membership,
        "source_cache_audit": source_cache_audit,
        "pymatgen_version": observed_pymatgen,
        "generation_rerun": False,
        "refinement_rerun": False,
        "chgnet_rerun": False,
        "direct_metrics_rerun": False,
        "novelty_rerun": False,
        "sample_retry_or_replacement_used": False,
        "filter_or_rerank_used": False,
        "slurm_used": False,
        "gpu_used": False,
        "api_key_read": False,
        "api_key_serialized": False,
    }
    return {
        "execution_sha": execution_sha,
        "config": config,
        "source_run": source_run,
        "source_terminal": source_terminal,
        "terminal_path": terminal_path,
        "ledger_path": ledger_path,
        "base_cache_path": base_cache_path,
        "base_cache_stat": {
            "bytes": base_cache_path.stat().st_size,
            "mtime_ns": base_cache_path.stat().st_mtime_ns,
        },
        "contexts": contexts,
        "missing_union": missing_union,
        "wanted_chemsys": wanted_chemsys,
        "all_targets": all_targets,
        "cached": cached,
        "input_contract": input_contract,
    }


def _verify_source_evidence_unchanged(state: Mapping[str, Any]) -> None:
    config = state["config"]
    source_run = state["source_run"]
    _resolve_source_file(
        source_run, config["source_run"]["terminal_report"], "R03E terminal report"
    )
    _resolve_frozen_file(
        source_run, config["source_run"]["attempt_ledger"], "frozen attempt ledger"
    )
    for arm_key, arm_config in config["source_run"]["arms"].items():
        for field in (
            "attempt_results",
            "attempt_summary",
            "input_manifest",
            "strict_relax_results",
            "meta_relax_results",
            "evaluation_report",
            "preflight_report",
        ):
            _resolve_source_file(
                source_run,
                arm_config[field],
                f"{arm_key} {field}",
            )
    cache = state["base_cache_path"]
    if (
        cache.stat().st_size != state["base_cache_stat"]["bytes"]
        or cache.stat().st_mtime_ns != state["base_cache_stat"]["mtime_ns"]
    ):
        raise CompletionError("frozen base MP hull cache changed during R03F")


def execute(args: argparse.Namespace) -> dict[str, Any]:
    state = prepare_frozen_state(args)
    if args.preflight_only:
        return {
            "schema": "h1_r03f_non_network_preflight_v1",
            "status": "pass",
            "input_contract": state["input_contract"],
            "run_root_exists": args.run_root.resolve().exists(),
            "network_used": False,
            "api_key_read": False,
        }

    config = state["config"]
    run_root = args.run_root.resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    existing_top_level = {path.name for path in run_root.iterdir()}
    if existing_top_level - {"logs"}:
        raise CompletionError("fixed R03F run root is not empty")
    created_at = datetime.now(timezone.utc).isoformat()
    write_json_exclusive(run_root / "input_contract.json", state["input_contract"])
    submission_record = {
        "schema": "h1_r03f_login_node_submission_record_v1",
        "status": "started",
        "created_at_utc": created_at,
        "run_id": config["run_id"],
        "execution_manifest_sha256": state["execution_sha"],
        "location": config["execution"]["location"],
        "slurm_used": False,
        "gpu_used": False,
        "cuda_visible_devices": "",
        "api_key_carrier": "user_owned_mode_0600_temporary_file",
        "api_key_serialized": False,
        "generation_rerun": False,
        "refinement_rerun": False,
        "chgnet_rerun": False,
        "direct_metrics_rerun": False,
        "novelty_rerun": False,
        "automatic_downstream": False,
    }
    write_json_exclusive(
        run_root / "status" / "submission_record.json", submission_record
    )
    claim = {
        "schema": "h1_r03f_mp_completion_claim_v1",
        "status": "claimed_before_external_queries",
        "created_at_utc": created_at,
        "run_id": config["run_id"],
        "execution_manifest_sha256": state["execution_sha"],
        "logical_mp_queries": len(state["missing_union"]),
        "common_snapshot_for_all_eight_arms": True,
        "api_key_present": True,
        "api_key_serialized": False,
        "sample_retry_or_replacement_used": False,
        "automatic_downstream": False,
    }
    write_json_exclusive(run_root / "claim.json", claim)

    api_key = read_and_destroy_api_key(args.key_file)
    query_fragment = run_root / "mp_query_entries.jsonl"
    query_results, query_progress = query_missing_chemsys(
        api_key=api_key,
        missing=sorted(state["missing_union"]),
        completed_cache_path=query_fragment,
        progress_path=run_root / "mp_query_progress.jsonl",
        maximum_attempts=int(
            config["materials_project"]["maximum_transport_attempts_per_chemsys"]
        ),
    )
    api_key = ""
    if (
        set(query_results) != state["missing_union"]
        or len(query_progress) != len(state["missing_union"])
        or any(row["status"] != "resolved" for row in query_progress)
        or any(not query_results[system] for system in state["missing_union"])
    ):
        raise CompletionError("Materials Project query coverage is incomplete")
    state["cached"].update(query_results)
    if any(not state["cached"].get(system) for system in state["wanted_chemsys"]):
        raise CompletionError("common completed snapshot has an empty chemical system")
    snapshot_path = run_root / "common_mp_thermo_snapshot.jsonl"
    write_jsonl_exclusive(
        snapshot_path,
        (
            {"chemsys": system, "entries": state["cached"][system]}
            for system in sorted(state["wanted_chemsys"])
        ),
    )
    hulls = compute_hulls(state["all_targets"], state["cached"])

    strict_threshold = float(config["sun"]["strict_threshold_ev_per_atom"])
    meta_threshold = float(config["sun"]["meta_threshold_ev_per_atom"])
    completed_by_key: dict[str, list[dict[str, Any]]] = {}
    counts_by_key: dict[str, dict[str, Any]] = {}
    vectors: dict[str, dict[str, list[bool]]] = {}
    source_counts_by_key: dict[str, dict[str, int]] = {}
    for arm_key in ARM_KEYS:
        context = state["contexts"][arm_key]
        completed, counts = complete_arm_attempts(
            context=context,
            hulls=hulls,
            strict_threshold=strict_threshold,
            meta_threshold=meta_threshold,
            execution_manifest_sha256=state["execution_sha"],
            queried_chemsys=state["missing_union"],
        )
        expected = config["source_run"]["arms"][arm_key]["expected_counts"]
        if (
            counts["relaxation_or_hull_unknown"] != 0
            or counts["source_evaluated_parity"] != int(expected["source_evaluated"])
            or counts["source_unknown_resolved"]
            != int(expected["source_hull_unknown"])
        ):
            raise CompletionError(f"{arm_key} completed hull hard gate failed")
        arm_dir = (
            run_root
            / "repeats"
            / str(context["repeat"])
            / "arms"
            / context["arm"]
        )
        attempts_output = arm_dir / "attempt_results.jsonl"
        write_jsonl_exclusive(attempts_output, completed)
        vectors[arm_key] = {
            endpoint: [
                bool((row.get("metrics") or {}).get(endpoint)) for row in completed
            ]
            for endpoint in ENDPOINTS
        }
        vector_payload = {
            "schema": "h1_r03f_all_attempt_sun_vectors_v1",
            "repeat": context["repeat"],
            "arm": context["arm"],
            "arm_key": arm_key,
            "ordinals": list(range(256)),
            **vectors[arm_key],
        }
        vectors_output = arm_dir / "sun_vectors.json"
        write_json_exclusive(vectors_output, vector_payload)
        source_counts = {
            endpoint: sum(
                bool((row.get("metrics") or {}).get(endpoint))
                for row in context["attempts"]
            )
            for endpoint in ENDPOINTS
        }
        source_counts["e_hull_unknown"] = int(expected["source_hull_unknown"])
        source_counts_by_key[arm_key] = source_counts
        summary = {
            "schema": "h1_r03f_mp_completed_arm_v1",
            "status": "complete",
            "ok": True,
            "repeat": context["repeat"],
            "arm": context["arm"],
            "arm_key": arm_key,
            "method": context["method"],
            "counts": counts,
            "source_frozen_cache_counts": source_counts,
            "coverage_delta": {
                "e_hull_unknown": -source_counts["e_hull_unknown"],
                "strict_full_sun": counts["strict_full_sun"]
                - source_counts["strict_full_sun"],
                "meta_full_sun": counts["meta_full_sun"]
                - source_counts["meta_full_sun"],
            },
            "rates_raw_all_attempts": {
                "strict_full_sun": counts["strict_full_sun"] / 256,
                "meta_full_sun": counts["meta_full_sun"] / 256,
                "novel_unique": counts["novel_unique"] / 256,
                "remaining_hull_unknown": 0.0,
            },
            "attempt_results": identity(attempts_output),
            "sun_vectors": identity(vectors_output),
            "source_inputs": {
                name: identity(path) for name, path in context["paths"].items()
            },
            "common_snapshot": identity(snapshot_path),
            "generation_rerun": False,
            "refinement_rerun": False,
            "chgnet_rerun": False,
            "direct_metrics_rerun": False,
            "novelty_rerun": False,
            "sample_retry_or_replacement_used": False,
            "filter_or_rerank_used": False,
            "api_key_serialized": False,
        }
        summary_output = arm_dir / "summary.json"
        write_json_exclusive(summary_output, summary)
        completed_by_key[arm_key] = completed
        counts_by_key[arm_key] = counts

    repeat_reports = []
    differences: list[list[list[float]]] = []
    for repeat in REPEATS:
        control_key = f"r{repeat}_control"
        candidate_key = f"r{repeat}_candidate"
        repeat_difference = []
        effects = {}
        for ordinal in range(256):
            repeat_difference.append(
                [
                    float(vectors[candidate_key][endpoint][ordinal])
                    - float(vectors[control_key][endpoint][ordinal])
                    for endpoint in ENDPOINTS
                ]
            )
        differences.append(repeat_difference)
        for endpoint in ENDPOINTS:
            control = vectors[control_key][endpoint]
            candidate = vectors[candidate_key][endpoint]
            effects[endpoint] = {
                "control_count": sum(control),
                "candidate_count": sum(candidate),
                "delta_count": sum(candidate) - sum(control),
                "delta_rate": (sum(candidate) - sum(control)) / 256,
                "mcnemar": exact_mcnemar(candidate, control),
            }
        repeat_reports.append(
            {
                "repeat": repeat,
                "arms": {
                    "control": {
                        "source_frozen_cache_counts": source_counts_by_key[control_key],
                        "completed_counts": counts_by_key[control_key],
                    },
                    "candidate": {
                        "source_frozen_cache_counts": source_counts_by_key[
                            candidate_key
                        ],
                        "completed_counts": counts_by_key[candidate_key],
                    },
                },
                "candidate_minus_control": effects,
            }
        )
    bootstrap = hierarchical_paired_bootstrap(
        differences,
        seed=int(config["analysis"]["bootstrap_seed"]),
        replicates=int(config["analysis"]["bootstrap_replicates"]),
    )
    pooled = {}
    sign_stability = {}
    for endpoint_index, endpoint in enumerate(ENDPOINTS):
        control = [
            value
            for repeat in REPEATS
            for value in vectors[f"r{repeat}_control"][endpoint]
        ]
        candidate = [
            value
            for repeat in REPEATS
            for value in vectors[f"r{repeat}_candidate"][endpoint]
        ]
        repeat_deltas = [
            sum(differences[repeat][ordinal][endpoint_index] for ordinal in range(256))
            / 256
            for repeat in REPEATS
        ]
        pooled[endpoint] = {
            "control_count": sum(control),
            "candidate_count": sum(candidate),
            "descriptive_denominator_per_arm": 1024,
            "candidate_minus_control_count": sum(candidate) - sum(control),
            "candidate_minus_control_rate": (sum(candidate) - sum(control)) / 1024,
            "descriptive_pooled_mcnemar": exact_mcnemar(candidate, control),
            **bootstrap[endpoint],
        }
        sign_stability[endpoint] = {
            "repeat_deltas": repeat_deltas,
            "positive_repeats": sum(value > 0.0 for value in repeat_deltas),
            "nonnegative_repeats": sum(value >= 0.0 for value in repeat_deltas),
            "negative_repeats": sum(value < 0.0 for value in repeat_deltas),
        }

    _verify_source_evidence_unchanged(state)
    query_counts = Counter(row["status"] for row in query_progress)
    total_unknown_before = sum(
        values["e_hull_unknown"] for values in source_counts_by_key.values()
    )
    total_unknown_after = sum(
        values["relaxation_or_hull_unknown"] for values in counts_by_key.values()
    )
    total_existing_finite = sum(
        int(config["source_run"]["arms"][arm_key]["expected_counts"]["source_evaluated"])
        for arm_key in ARM_KEYS
    )
    total_existing_parity = sum(
        values["source_evaluated_parity"] for values in counts_by_key.values()
    )
    if total_unknown_after != 0 or total_existing_parity != total_existing_finite:
        raise CompletionError("R03F global E_hull hard gate failed")

    terminal = {
        "schema": "h1_r03f_refined_repeats4_mp_completion_terminal_v1",
        "status": "complete_all_missing_hulls_resolved",
        "ok": True,
        "created_at_utc": created_at,
        "run_id": config["run_id"],
        "execution_manifest_sha256": state["execution_sha"],
        "source_r03e": {
            "terminal": identity(state["terminal_path"]),
            "decision_preserved": state["source_terminal"]["decision"],
            "terminal_rewritten": False,
            "evidence_unchanged": True,
        },
        "repeat_count": 4,
        "arm_count": 8,
        "attempts_per_arm_per_repeat": 256,
        "raw_attempts_total": 2048,
        "denominator": "raw_all_attempts",
        "arms": {
            arm_key: {
                "repeat": state["contexts"][arm_key]["repeat"],
                "arm": state["contexts"][arm_key]["arm"],
                "method": state["contexts"][arm_key]["method"],
                "source_frozen_cache_counts": source_counts_by_key[arm_key],
                "completed_counts": counts_by_key[arm_key],
                "attempt_results": identity(
                    run_root
                    / "repeats"
                    / str(state["contexts"][arm_key]["repeat"])
                    / "arms"
                    / state["contexts"][arm_key]["arm"]
                    / "attempt_results.jsonl"
                ),
                "sun_vectors": identity(
                    run_root
                    / "repeats"
                    / str(state["contexts"][arm_key]["repeat"])
                    / "arms"
                    / state["contexts"][arm_key]["arm"]
                    / "sun_vectors.json"
                ),
            }
            for arm_key in ARM_KEYS
        },
        "repeat_reports": repeat_reports,
        "pooled_candidate_minus_control": pooled,
        "sign_stability": sign_stability,
        "bootstrap": {
            "method": "hierarchical_paired_repeat_block_and_ordinal",
            "seed": int(config["analysis"]["bootstrap_seed"]),
            "replicates": int(config["analysis"]["bootstrap_replicates"]),
            "pooled_1024_independence_assumed": False,
        },
        "coverage": {
            "applicable_novel_unique_total": sum(
                values["novel_unique"] for values in counts_by_key.values()
            ),
            "e_hull_unknown_before": total_unknown_before,
            "e_hull_unknown_after": total_unknown_after,
            "existing_finite_e_hull": total_existing_finite,
            "existing_finite_e_hull_parity": total_existing_parity,
            "existing_finite_e_hull_parity_rate": 1.0,
        },
        "mp_completion": {
            "client": config["materials_project"]["client"],
            "method": config["materials_project"]["method"],
            "compatible_only": True,
            "pymatgen_version": importlib.metadata.version("pymatgen"),
            "common_snapshot_for_all_eight_arms": True,
            "distinct_missing_chemsys": len(state["missing_union"]),
            "logical_queries_submitted": len(query_progress),
            "query_counts": dict(query_counts),
            "transport_attempts_total": sum(
                int(row["transport_attempts"]) for row in query_progress
            ),
            "transport_retries_total": sum(
                int(row["transport_retries"]) for row in query_progress
            ),
            "query_progress": identity(run_root / "mp_query_progress.jsonl"),
            "query_fragment": identity(query_fragment),
            "common_snapshot": identity(snapshot_path),
            "api_key_serialized": False,
        },
        "execution": {
            "location": config["execution"]["location"],
            "slurm_used": False,
            "gpu_used": False,
            "generation_rerun": False,
            "refinement_rerun": False,
            "chgnet_rerun": False,
            "direct_metrics_rerun": False,
            "novelty_rerun": False,
            "sample_retry_or_replacement_used": False,
            "filter_or_rerank_used": False,
        },
        "formal_g3": False,
        "automatic_promotion": False,
        "automatic_training": False,
        "checkpoint_reselection": False,
        "automatic_downstream": False,
        "decision": "coverage_completed_report_only_no_automatic_promotion",
        "recommended_next_step": (
            "Review completed-cache strict/meta paired effects, then authorize at "
            "most one minimal H1-based scientific change."
        ),
    }
    final_dir = run_root / "final"
    terminal_path_output = final_dir / "terminal_report.json"
    write_json_exclusive(terminal_path_output, terminal)
    decision_record = {
        "schema": "h1_r03f_mp_completion_decision_v1",
        "status": terminal["status"],
        "decision": terminal["decision"],
        "terminal_report_sha256": sha256_file(terminal_path_output),
        "formal_g3": False,
        "automatic_promotion": False,
        "automatic_training": False,
        "checkpoint_reselection": False,
        "automatic_downstream": False,
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
    parser.add_argument("--key-file", type=Path)
    parser.add_argument("--execution-manifest-sha256", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.preflight_only and args.key_file is None:
        raise CompletionError("--key-file is required outside preflight-only mode")
    try:
        result = execute(args)
    except Exception as exc:
        run_root = args.run_root.resolve()
        if not args.preflight_only and run_root.is_dir():
            failure_path = run_root / "status" / "failure_report.json"
            if not failure_path.exists():
                write_json_exclusive(
                    failure_path,
                    {
                        "schema": "h1_r03f_failure_report_v1",
                        "status": "failed_no_retry",
                        "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                        "exception_type": type(exc).__name__,
                        "reason": str(exc),
                        "api_key_serialized": False,
                        "sample_retry_or_replacement_used": False,
                        "automatic_downstream": False,
                    },
                )
        raise
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
