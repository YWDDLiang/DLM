#!/usr/bin/env python3
"""Complete official E_hull for the frozen archived H1-A2/R03 first256 pair."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import stat
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import pymatgen.entries.compatibility as _compatibility
import pymatgen.entries.computed_entries as _computed_entries

sys.modules.setdefault("pymatgen.core.entries", _computed_entries)
sys.modules.setdefault("pymatgen.analysis.compatibility", _compatibility)

from mp_api.client import MPRester
from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.core import Composition
from pymatgen.entries.computed_entries import ComputedEntry


class ContractError(RuntimeError):
    pass


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ContractError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ContractError(f"expected JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(canonical_json(value) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def require_file(path: Path, expected_sha256: str, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    observed = sha256_file(resolved)
    if observed != expected_sha256:
        raise ContractError(f"{label} identity changed")
    return resolved


def require_source(source: Path, expected_manifest_sha256: str) -> None:
    manifest = require_file(
        source / "SOURCE_SHA256.txt",
        expected_manifest_sha256,
        "source manifest",
    )
    expected: dict[str, str] = {}
    for raw in manifest.read_text(encoding="utf-8").splitlines():
        digest, separator, relative = raw.partition("  ")
        if not separator or len(digest) != 64 or relative.startswith(("/", "../")):
            raise ContractError("invalid source manifest row")
        expected[relative] = digest
    observed = {
        str(path.relative_to(source)).replace(os.sep, "/")
        for path in source.rglob("*")
        if path.is_file()
        and path.name != "SOURCE_SHA256.txt"
        and "__pycache__" not in path.parts
    }
    if observed != set(expected):
        raise ContractError("source file set changed")
    for relative, digest in expected.items():
        if sha256_file(source / relative) != digest:
            raise ContractError(f"source file changed: {relative}")


def finite_float(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ContractError(f"{label} is not finite")
    return result


def normalized_chemsys(composition: Composition) -> str:
    symbols = sorted(element.symbol for element in composition.elements)
    if not symbols:
        raise ContractError("empty composition")
    return "-".join(symbols)


def package_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for name in ("mp-api", "emmet-core", "pymatgen"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def read_destroy_key(path: Path) -> str:
    location = path.expanduser()
    details = location.lstat()
    raw = b""
    try:
        mode = stat.S_IMODE(details.st_mode)
        if (
            not stat.S_ISREG(details.st_mode)
            or details.st_uid != os.getuid()
            or mode != 0o600
        ):
            raise ContractError("one-time key carrier ownership or mode is invalid")
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
    for separator in ("Content:", "Response:", '{"data":'):
        message = message.split(separator, 1)[0]
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    return {
        "type": type(exc).__name__,
        "http_status": None if status is None else int(status),
        "message": " ".join(message.split())[:500],
    }


def slim_entries(entries: list[Any]) -> list[dict[str, Any]]:
    def key(entry: Any) -> tuple[str, str, float]:
        return (
            "" if getattr(entry, "entry_id", None) is None else str(entry.entry_id),
            canonical_json(entry.composition.as_dict()),
            finite_float(entry.energy, "reference energy"),
        )

    return [
        {
            "entry_id": (
                None if getattr(entry, "entry_id", None) is None else str(entry.entry_id)
            ),
            "composition": entry.composition.as_dict(),
            "energy": finite_float(entry.energy, "reference energy"),
        }
        for entry in sorted(entries, key=key)
    ]


def validate_reference_set(entries: list[Any], elements: list[str]) -> None:
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
        raise ContractError(f"reference entries contain outside elements: {sorted(outside)}")
    missing = sorted(requested - unary)
    if missing:
        raise ContractError(f"missing unary references: {missing}")
    diagram = PhaseDiagram(entries)
    if {element.symbol for element in diagram.elements} != requested:
        raise ContractError("phase diagram element set changed")


def phase_diagram(rows: list[dict[str, Any]], chemsys: str) -> PhaseDiagram:
    entries = [
        ComputedEntry(
            row["composition"],
            finite_float(row["energy"], "reference energy"),
            entry_id=row.get("entry_id"),
        )
        for row in rows
    ]
    diagram = PhaseDiagram(entries)
    if {element.symbol for element in diagram.elements} != set(chemsys.split("-")):
        raise ContractError(f"phase diagram elements changed: {chemsys}")
    return diagram


def exact_hull(diagram: PhaseDiagram, composition: Composition, energy: float) -> float:
    _, raw = diagram.get_decomp_and_e_above_hull(
        PDEntry(composition, energy * composition.num_atoms),
        allow_negative=True,
    )
    return max(finite_float(raw, "official e_above_hull"), 0.0)


def exact_mcnemar(left: list[bool], right: list[bool]) -> dict[str, Any]:
    if len(left) != len(right):
        raise ContractError("paired vectors differ in length")
    left_only = sum(a and not b for a, b in zip(left, right))
    right_only = sum(b and not a for a, b in zip(left, right))
    discordant = left_only + right_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(
            math.comb(discordant, index)
            for index in range(min(left_only, right_only) + 1)
        ) / (2**discordant)
        p_value = min(1.0, 2.0 * tail)
    return {
        "control_only": left_only,
        "candidate_only": right_only,
        "discordant": discordant,
        "two_sided_exact_p": p_value,
    }


def load_arm(
    upstream: Path,
    arm: str,
    spec: Mapping[str, Any],
    expected_attempts: int,
) -> dict[str, Any]:
    arm_root = upstream / "arms" / arm
    evaluation = arm_root / "evaluation/r5c_a100_sun"
    paths = {
        "generation": require_file(
            arm_root / "generation/generation.jsonl",
            str(spec["generation_sha256"]),
            f"{arm} generation",
        ),
        "attempts": require_file(
            evaluation / "attempt_results.jsonl",
            str(spec["attempt_results_sha256"]),
            f"{arm} attempt results",
        ),
        "summary": require_file(
            evaluation / "attempt_summary.json",
            str(spec["attempt_summary_sha256"]),
            f"{arm} attempt summary",
        ),
        "manifest": require_file(
            evaluation / "input_manifest.json",
            str(spec["input_manifest_sha256"]),
            f"{arm} input manifest",
        ),
        "strict_relax": require_file(
            evaluation / "exact_strict/relax_results.jsonl",
            str(spec["relax_results_sha256"]),
            f"{arm} strict relax results",
        ),
        "meta_relax": require_file(
            evaluation / "exact_meta_like/relax_results.jsonl",
            str(spec["relax_results_sha256"]),
            f"{arm} meta relax results",
        ),
        "direct": require_file(
            arm_root / "evaluation/crysllmgen_metrics/report.json",
            str(spec["direct_report_sha256"]),
            f"{arm} Direct report",
        ),
    }
    for marker in (
        arm_root / "generation/_SUCCESS",
        arm_root / "evaluation/_SUCCESS",
    ):
        if not marker.is_file():
            raise ContractError(f"missing upstream marker: {marker}")

    generation = read_jsonl(paths["generation"])
    attempts = read_jsonl(paths["attempts"])
    summary = read_json(paths["summary"])
    manifest = read_json(paths["manifest"])
    strict_relax = read_jsonl(paths["strict_relax"])
    meta_relax = read_jsonl(paths["meta_relax"])
    if strict_relax != meta_relax:
        raise ContractError(f"{arm} strict/meta relax ledgers differ")
    if len(generation) != expected_attempts or len(attempts) != expected_attempts:
        raise ContractError(f"{arm} denominator changed")
    if [int(row["ordinal"]) for row in generation] != list(range(expected_attempts)):
        raise ContractError(f"{arm} generation ordinals changed")
    if [int(row["generation_ordinal"]) for row in attempts] != list(range(expected_attempts)):
        raise ContractError(f"{arm} evaluation ordinals changed")
    if [str(row["attempt_id"]) for row in generation] != [
        str(row["attempt_id"]) for row in attempts
    ]:
        raise ContractError(f"{arm} generation/evaluation alignment changed")
    if any(row.get("retry_or_replacement_used") is not False for row in generation):
        raise ContractError(f"{arm} generation used retry/replacement")
    if any(row.get("retry_or_replacement_used") is not False for row in attempts):
        raise ContractError(f"{arm} evaluation used retry/replacement")
    if int(summary["counts"]["total_attempts"]) != expected_attempts:
        raise ContractError(f"{arm} summary denominator changed")

    manifest_by_id = {
        str(row["attempt_id"]): row for row in manifest["attempt_records"]
    }
    if len(manifest_by_id) != expected_attempts:
        raise ContractError(f"{arm} input manifest mapping changed")
    novel_unique = [
        row for row in attempts if bool((row.get("metrics") or {}).get("novel_unique"))
    ]
    if len(novel_unique) != int(spec["expected_novel_unique"]):
        raise ContractError(f"{arm} novel-unique count changed")
    novel_unique.sort(
        key=lambda row: int(manifest_by_id[str(row["attempt_id"])]["reconstructed_index"])
    )
    if len(novel_unique) != len(strict_relax):
        raise ContractError(f"{arm} relaxed-energy alignment length changed")
    relax_by_local = {int(row["local_index"]): row for row in strict_relax}
    if set(relax_by_local) != set(range(len(novel_unique))):
        raise ContractError(f"{arm} relaxed-energy local indices changed")

    alignment: dict[str, dict[str, Any]] = {}
    for local_index, attempt in enumerate(novel_unique):
        attempt_id = str(attempt["attempt_id"])
        row = relax_by_local[local_index]
        composition = Composition(row["composition"])
        old_energy = attempt["metrics"].get("energy_per_atom")
        energy = row.get("energy_per_atom")
        if old_energy is None or energy is None or not math.isclose(
            float(old_energy), float(energy), rel_tol=0.0, abs_tol=1e-12
        ):
            raise ContractError(f"{arm} frozen relaxed energy changed: {attempt_id}")
        alignment[attempt_id] = {
            "composition": composition.as_dict(),
            "chemsys": normalized_chemsys(composition),
            "energy_per_atom": finite_float(energy, "relaxed energy"),
        }
    return {
        "attempts": attempts,
        "summary": summary,
        "alignment": alignment,
        "paths": {name: identity(path) for name, path in paths.items()},
        "direct": read_json(paths["direct"]),
    }


def evaluate_arm(
    arm: str,
    loaded: Mapping[str, Any],
    resolved: Mapping[str, list[dict[str, Any]]],
    unresolved: Mapping[str, Mapping[str, Any]],
    strict_threshold: float,
    meta_threshold: float,
    historical: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    diagrams: dict[str, PhaseDiagram] = {}
    new_attempts: list[dict[str, Any]] = []
    old_unknown = newly_resolved = evaluated = hull_unknown = 0
    strict_01 = strict_10 = meta_01 = meta_10 = 0
    for old in loaded["attempts"]:
        new = copy.deepcopy(old)
        new["schema"] = "crysllmgen_r5c_a100_sun_attempt_official_ehull_completion_v1"
        attempt_id = str(old["attempt_id"])
        metrics = new["metrics"]
        old_metrics = old["metrics"]
        repair: dict[str, Any] = {
            "old_e_above_hull": old_metrics.get("e_above_hull"),
            "old_evaluation_status": str(old["evaluation_status"]),
            "query_method": "MPRester.get_entries_in_chemsys",
            "thermo_type": "GGA_GGA+U",
        }
        if bool(metrics.get("novel_unique")):
            aligned = loaded["alignment"].get(attempt_id)
            if aligned is None:
                raise ContractError(f"{arm} missing relaxed-energy alignment: {attempt_id}")
            if old_metrics.get("e_above_hull") is None:
                old_unknown += 1
            composition = Composition(aligned["composition"])
            chemsys = str(aligned["chemsys"])
            energy = finite_float(aligned["energy_per_atom"], "relaxed energy")
            if chemsys in unresolved:
                new_hull = None
                new["evaluation_status"] = "hull_unknown"
                repair["hull_unknown_reason"] = unresolved[chemsys]["reason"]
                hull_unknown += 1
            else:
                if chemsys not in resolved:
                    raise ContractError(f"missing official cache row: {chemsys}")
                if chemsys not in diagrams:
                    diagrams[chemsys] = phase_diagram(resolved[chemsys], chemsys)
                new_hull = exact_hull(diagrams[chemsys], composition, energy)
                new["evaluation_status"] = "evaluated"
                evaluated += 1
                if old_metrics.get("e_above_hull") is None:
                    newly_resolved += 1
            metrics["energy_per_atom"] = energy
            metrics["e_above_hull"] = new_hull
            metrics["strict_full_sun"] = (
                new_hull is not None and new_hull <= strict_threshold
            )
            metrics["meta_full_sun"] = (
                new_hull is not None and new_hull <= meta_threshold
            )
        else:
            if attempt_id in loaded["alignment"]:
                raise ContractError(f"{arm} alignment contains non-N+U attempt")
            if metrics.get("e_above_hull") is not None:
                raise ContractError(f"{arm} non-N+U attempt carries E_hull")
        old_strict = bool(old_metrics.get("strict_full_sun"))
        new_strict = bool(metrics.get("strict_full_sun"))
        old_meta = bool(old_metrics.get("meta_full_sun"))
        new_meta = bool(metrics.get("meta_full_sun"))
        strict_01 += int(not old_strict and new_strict)
        strict_10 += int(old_strict and not new_strict)
        meta_01 += int(not old_meta and new_meta)
        meta_10 += int(old_meta and not new_meta)
        repair.update(
            {
                "official_e_above_hull": metrics.get("e_above_hull"),
                "official_evaluation_status": str(new["evaluation_status"]),
            }
        )
        new["stability_completion"] = repair
        new_attempts.append(new)

    summary = loaded["summary"]
    total = int(summary["counts"]["total_attempts"])
    reconstructed = int(summary["counts"]["reconstructed"])
    novel_unique = int(summary["counts"]["novel_unique"])
    strict = sum(bool(row["metrics"].get("strict_full_sun")) for row in new_attempts)
    meta = sum(bool(row["metrics"].get("meta_full_sun")) for row in new_attempts)
    if evaluated + hull_unknown != novel_unique or meta < strict:
        raise ContractError(f"{arm} official stability accounting is incomplete")
    report = {
        "arm": arm,
        "label": historical["label"],
        "denominators": {
            "all_attempts": total,
            "reconstructed_exact_legacy": reconstructed,
            "novel_unique": novel_unique,
            "hull_evaluated_novel_unique": evaluated,
            "hull_unknown": hull_unknown,
            "all_attempts_skip_hull_unknown": total - hull_unknown,
            "reconstructed_skip_hull_unknown": reconstructed - hull_unknown,
        },
        "frozen_non_stability_counts": {
            "reconstructed": reconstructed,
            "novel": int(summary["counts"]["novel"]),
            "unique": int(summary["counts"]["unique"]),
            "novel_unique": novel_unique,
        },
        "archived_frozen_cache": {
            "strict_full_sun": int(summary["counts"]["strict_full_sun"]),
            "meta_full_sun": int(summary["counts"]["meta_full_sun"]),
            "hull_unknown": old_unknown,
        },
        "official": {
            "strict_full_sun": strict,
            "meta_full_sun": meta,
            "strict_rate_all_attempts": strict / total,
            "meta_rate_all_attempts": meta / total,
            "strict_rate_reconstructed": strict / reconstructed,
            "meta_rate_reconstructed": meta / reconstructed,
            "strict_rate_hull_evaluated_novel_unique": strict / evaluated,
            "meta_rate_hull_evaluated_novel_unique": meta / evaluated,
        },
        "historical_expected": {
            "strict_full_sun": int(historical["historical_strict"]),
            "meta_full_sun": int(historical["historical_meta"]),
            "strict_match": strict == int(historical["historical_strict"]),
            "meta_match": meta == int(historical["historical_meta"]),
        },
        "archived_to_official": {
            "old_unknown_now_resolved": newly_resolved,
            "strict_0_to_1": strict_01,
            "strict_1_to_0": strict_10,
            "meta_0_to_1": meta_01,
            "meta_1_to_0": meta_10,
        },
        "direct_report": loaded["direct"],
        "input_identities": loaded["paths"],
    }
    return new_attempts, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--key-file", type=Path, required=True)
    args = parser.parse_args()

    source = args.source_dir.resolve()
    require_source(source, args.source_manifest_sha256)
    config = read_json(source / "CONFIG.json")
    if config.get("schema") != "h1_r03_h1a2_archived_first256_official_ehull_completion_config_v1":
        raise ContractError("unexpected config schema")
    run_root = args.run_root.resolve()
    if run_root != Path(config["run_root"]).resolve():
        raise ContractError("run root changed")
    output = run_root / "official_results"
    if output.exists():
        raise FileExistsError(output)
    preparing = run_root / f".official_results.preparing.{os.getpid()}"
    failed = run_root / f".official_results.FAILED.{os.getpid()}"
    preparing.mkdir(parents=True, exist_ok=False)
    secret = ""
    try:
        runtime = config["official_mp"]
        if Path(sys.executable).resolve() != Path(runtime["python"]).resolve():
            raise ContractError("official MP Python runtime changed")
        if platform.python_version() != runtime["python_version"]:
            raise ContractError("official MP Python version changed")
        versions = package_versions()
        if any(versions.get(name) != value for name, value in runtime["packages"].items()):
            raise ContractError("official MP package versions changed")
        if any(os.environ.get(name) for name in ("MP_API_KEY", "PMG_MAPI_KEY", "MAPI_KEY")):
            raise ContractError("ambient MP credentials are forbidden")

        upstream = Path(config["upstream"]["run_root"]).resolve()
        require_file(
            upstream / "terminal_report.json",
            config["upstream"]["terminal_report_sha256"],
            "upstream terminal report",
        )
        for marker in (upstream / "_SUCCESS", upstream / "status/PIPELINE_SUCCESS"):
            if not marker.is_file():
                raise ContractError(f"upstream marker is absent: {marker}")
        expected_attempts = int(config["upstream"]["attempts_per_arm"])
        loaded = {
            arm: load_arm(upstream, arm, spec, expected_attempts)
            for arm, spec in config["upstream"]["arms"].items()
        }
        wanted = {
            row["chemsys"]
            for arm in loaded.values()
            for row in arm["alignment"].values()
        }

        base_spec = config["base_official_cache"]
        base = Path(base_spec["root"]).resolve()
        base_manifest = require_file(
            base / "completion_manifest.json",
            base_spec["completion_manifest_sha256"],
            "base official completion manifest",
        )
        slim_path = require_file(
            base / "official_slim_cache.jsonl",
            base_spec["slim_cache_sha256"],
            "base official slim cache",
        )
        unresolved_path = require_file(
            base / "unresolved_chemsys.jsonl",
            base_spec["unresolved_sha256"],
            "base official unresolved ledger",
        )
        slim_rows = read_jsonl(slim_path)
        unresolved_rows = read_jsonl(unresolved_path)
        resolved_all = {str(row["chemsys"]): row["entries"] for row in slim_rows}
        unresolved_all = {str(row["chemsys"]): row for row in unresolved_rows}
        if (
            len(resolved_all) != int(base_spec["resolved_count"])
            or len(unresolved_all) != int(base_spec["unresolved_count"])
            or set(resolved_all) & set(unresolved_all)
        ):
            raise ContractError("base official cache accounting changed")
        base_resolved = wanted & set(resolved_all)
        base_unresolved = wanted & set(unresolved_all)
        genuinely_missing = wanted - set(resolved_all) - set(unresolved_all)
        query_targets = sorted(wanted - set(resolved_all))
        observed_counts = (
            len(wanted),
            len(base_resolved),
            len(base_unresolved),
            len(genuinely_missing),
            len(query_targets),
        )
        expected_counts = (
            int(runtime["expected_wanted_chemsys"]),
            int(runtime["expected_base_resolved"]),
            int(runtime["expected_base_unresolved"]),
            int(runtime["expected_genuinely_missing"]),
            int(runtime["expected_query_targets"]),
        )
        if observed_counts != expected_counts:
            raise ContractError(
                f"official query target audit changed: {observed_counts} != {expected_counts}"
            )

        secret = read_destroy_key(args.key_file)
        (run_root / "status/KEY_CARRIER_DESTROYED").touch(exist_ok=False)
        query_audit: list[dict[str, Any]] = []
        database_version: str | None = None
        with MPRester(secret) as client:
            database_version = str(client.get_database_version())
            for query_index, chemsys in enumerate(query_targets):
                elements = chemsys.split("-")
                record: dict[str, Any] = {
                    "query_index": query_index,
                    "chemsys": chemsys,
                    "elements": elements,
                    "query_method": runtime["query_method"],
                    "compatible_only": True,
                    "thermo_type": "GGA_GGA+U",
                    "transport_attempts": 0,
                }
                entries: list[Any] | None = None
                for transport_attempt in range(1, int(runtime["max_transport_attempts"]) + 1):
                    record["transport_attempts"] = transport_attempt
                    try:
                        entries = client.get_entries_in_chemsys(
                            elements,
                            compatible_only=True,
                            additional_criteria={"thermo_types": ["GGA_GGA+U"]},
                        )
                        validate_reference_set(entries, elements)
                        break
                    except ContractError as exc:
                        record.update(
                            {
                                "query_status": "unresolved",
                                "reason": "official_reference_contract_unresolved",
                                "error": sanitized_error(exc, secret),
                            }
                        )
                        entries = None
                        break
                    except Exception as exc:
                        error = sanitized_error(exc, secret)
                        record["error"] = error
                        if error["http_status"] in (401, 403):
                            raise ContractError("official MP authorization failed") from exc
                        if transport_attempt == int(runtime["max_transport_attempts"]):
                            record.update(
                                {
                                    "query_status": "unresolved",
                                    "reason": "official_transport_unresolved_after_bounded_attempts",
                                }
                            )
                            entries = None
                            break
                        time.sleep(float(2 ** (transport_attempt - 1)))
                if entries is not None:
                    slim = slim_entries(entries)
                    resolved_all[chemsys] = slim
                    unresolved_all.pop(chemsys, None)
                    record.update(
                        {
                            "query_status": "resolved",
                            "entry_count": len(slim),
                            "entries_sha256": hashlib.sha256(
                                canonical_json(slim).encode("utf-8")
                            ).hexdigest(),
                            "reason": None,
                            "error": None,
                        }
                    )
                else:
                    unresolved_all[chemsys] = {
                        "chemsys": chemsys,
                        "elements": elements,
                        "reason": record["reason"],
                        "source": "fresh_official_requery_20260813",
                        "source_error": record.get("error"),
                    }
                query_audit.append(record)
                print(
                    canonical_json(
                        {
                            "completed": query_index + 1,
                            "total": len(query_targets),
                            "chemsys": chemsys,
                            "status": record["query_status"],
                            "transport_attempts": record["transport_attempts"],
                        }
                    ),
                    flush=True,
                )
                time.sleep(float(runtime["request_interval_seconds"]))

        secret = ""
        resolved = {chemsys: resolved_all[chemsys] for chemsys in sorted(wanted & set(resolved_all))}
        unresolved = {
            chemsys: unresolved_all[chemsys]
            for chemsys in sorted(wanted & set(unresolved_all))
        }
        if set(resolved) & set(unresolved) or set(resolved) | set(unresolved) != wanted:
            raise ContractError("post-query official cache coverage is incomplete")
        cache_dir = preparing / "official_mp_cache"
        write_jsonl(
            cache_dir / "official_slim_cache.jsonl",
            ({"chemsys": chemsys, "entries": resolved[chemsys]} for chemsys in sorted(resolved)),
        )
        write_jsonl(
            cache_dir / "unresolved_chemsys.jsonl",
            (unresolved[chemsys] for chemsys in sorted(unresolved)),
        )
        write_jsonl(cache_dir / "query_audit.jsonl", query_audit)

        arm_attempts: dict[str, list[dict[str, Any]]] = {}
        arm_reports: dict[str, dict[str, Any]] = {}
        for arm, historical in config["upstream"]["arms"].items():
            attempts, report = evaluate_arm(
                arm,
                loaded[arm],
                resolved,
                unresolved,
                float(runtime["strict_threshold_ev_per_atom"]),
                float(runtime["meta_threshold_ev_per_atom"]),
                historical,
            )
            arm_attempts[arm] = attempts
            arm_reports[arm] = report
            arm_dir = preparing / "arms" / arm
            write_jsonl(arm_dir / "attempt_results_official.jsonl", attempts)
            write_json(arm_dir / "report.json", report)
            (arm_dir / "_SUCCESS").touch(exist_ok=False)

        paired: dict[str, Any] = {}
        for metric in ("strict_full_sun", "meta_full_sun"):
            control = [bool(row["metrics"].get(metric)) for row in arm_attempts["control"]]
            candidate = [bool(row["metrics"].get(metric)) for row in arm_attempts["candidate"]]
            paired[metric] = exact_mcnemar(control, candidate)

        query_manifest = {
            "schema": "h1_r03_h1a2_archived_first256_official_query_manifest_v1",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "database_version": database_version,
            "query_method": runtime["query_method"],
            "compatible_only": True,
            "thermo_type": "GGA_GGA+U",
            "additional_criteria": {"thermo_types": ["GGA_GGA+U"]},
            "wanted_chemsys": len(wanted),
            "base_resolved": len(base_resolved),
            "base_unresolved_requeried": len(base_unresolved),
            "genuinely_missing_queried": len(genuinely_missing),
            "query_targets": len(query_targets),
            "query_resolved": sum(row["query_status"] == "resolved" for row in query_audit),
            "query_unresolved": sum(row["query_status"] == "unresolved" for row in query_audit),
            "final_resolved": len(resolved),
            "final_unresolved": len(unresolved),
            "credential_serialized": False,
            "key_carrier_destroyed_before_first_http_request": True,
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "package_versions": versions,
            "base_cache": {
                "completion_manifest": identity(base_manifest),
                "slim_cache": identity(slim_path),
                "unresolved": identity(unresolved_path),
            },
        }
        write_json(cache_dir / "completion_manifest.json", query_manifest)
        (cache_dir / "completion_SUCCESS").touch(exist_ok=False)

        terminal = {
            "schema": "h1_r03_h1a2_archived_first256_official_ehull_terminal_v1",
            "status": "complete",
            "ok": True,
            "source_manifest_sha256": args.source_manifest_sha256,
            "scientific_contract": {
                "attempts_per_arm": expected_attempts,
                "generation_refine_energy_novelty_uniqueness_rerun": False,
                "official_query_method": runtime["query_method"],
                "compatible_only": True,
                "thermo_type": "GGA_GGA+U",
                "strict_threshold_ev_per_atom": float(runtime["strict_threshold_ev_per_atom"]),
                "meta_threshold_ev_per_atom": float(runtime["meta_threshold_ev_per_atom"]),
                "unresolved_policy": runtime["unresolved_policy"],
            },
            "query": query_manifest,
            "arms": arm_reports,
            "paired_exact_mcnemar": paired,
        }
        write_json(preparing / "terminal_report.json", terminal)

        lines = [
            "# Archived H1-A2 vs R03 first256 — official E_hull completion",
            "",
            "Only stability was recomputed. Generation, model-494 refine800, Direct, U, N, and CHGNet relaxed energies are byte-frozen.",
            "",
            f"- Official query: `{runtime['query_method']}`, `compatible_only=True`, `GGA_GGA+U`.",
            f"- Chemical systems: {len(wanted)} total; {len(base_resolved)} reused resolved; {len(query_targets)} freshly queried; {len(unresolved)} remain explicit hull-unknown.",
            "",
            "| Arm | Generated | Joint valid | Novel+unique | Hull evaluated | Hull unknown | Strict S.U.N. | Meta-S.U.N. | Historical expected |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for arm in ("control", "candidate"):
            report = arm_reports[arm]
            direct = report["direct_report"]
            den = report["denominators"]
            off = report["official"]
            hist = report["historical_expected"]
            lines.append(
                f"| {report['label']} | {direct['generation_succeeded']}/256 | "
                f"{direct['valid_count']}/256 | {den['novel_unique']}/256 | "
                f"{den['hull_evaluated_novel_unique']} | {den['hull_unknown']} | "
                f"{off['strict_full_sun']}/256 ({100*off['strict_rate_all_attempts']:.2f}%) | "
                f"{off['meta_full_sun']}/256 ({100*off['meta_rate_all_attempts']:.2f}%) | "
                f"{hist['strict_full_sun']} strict / {hist['meta_full_sun']} meta |"
            )
        lines.extend(["", "## Exact paired McNemar", "", "| Endpoint | H1-A2 only | R03 only | Discordant | Two-sided exact p |", "|---|---:|---:|---:|---:|"])
        for metric, label in (("strict_full_sun", "strict S.U.N."), ("meta_full_sun", "meta-S.U.N.")):
            row = paired[metric]
            lines.append(
                f"| {label} | {row['control_only']} | {row['candidate_only']} | "
                f"{row['discordant']} | {row['two_sided_exact_p']:.8g} |"
            )
        lines.extend(
            [
                "",
                "Any remaining official-reference failure is preserved as `hull_unknown` and is not silently counted as an evaluated unstable structure.",
                "",
            ]
        )
        markdown = preparing / "RESULTS_COMPLETE.md"
        with markdown.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write("\n".join(lines))
            handle.flush()
            os.fsync(handle.fileno())
        (preparing / "_SUCCESS").touch(exist_ok=False)
        preparing.rename(output)
        os.link(output / "terminal_report.json", run_root / "terminal_report.json")
        os.link(output / "RESULTS_COMPLETE.md", run_root / "RESULTS_COMPLETE.md")
        (run_root / "status/OFFICIAL_QUERY_SUCCESS").touch(exist_ok=False)
        (run_root / "status/STABILITY_REEVALUATION_SUCCESS").touch(exist_ok=False)
        (run_root / "status/RESULTS_COMPLETE").touch(exist_ok=False)
        print(
            canonical_json(
                {
                    "status": "complete",
                    "query_resolved": query_manifest["query_resolved"],
                    "query_unresolved": query_manifest["query_unresolved"],
                    "control_strict": arm_reports["control"]["official"]["strict_full_sun"],
                    "control_meta": arm_reports["control"]["official"]["meta_full_sun"],
                    "candidate_strict": arm_reports["candidate"]["official"]["strict_full_sun"],
                    "candidate_meta": arm_reports["candidate"]["official"]["meta_full_sun"],
                }
            ),
            flush=True,
        )
    except Exception as exc:
        safe_message = str(exc).replace(secret, "[REDACTED]") if secret else str(exc)
        secret = ""
        if args.key_file.exists():
            args.key_file.unlink(missing_ok=True)
        if preparing.exists():
            write_json(
                preparing / "failure.json",
                {
                    "schema": "h1_r03_h1a2_archived_first256_official_ehull_failure_v1",
                    "error_type": type(exc).__name__,
                    "error_message": " ".join(safe_message.split())[:500],
                    "credential_serialized": False,
                },
            )
            shutil.move(str(preparing), str(failed))
        raise ContractError(
            "official E_hull completion failed; see sanitized failure evidence"
        ) from None
    finally:
        secret = ""


if __name__ == "__main__":
    main()
