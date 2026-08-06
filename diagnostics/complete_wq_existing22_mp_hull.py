#!/usr/bin/env python3
"""Complete the eight frozen job27631 MP hull unknowns on a login node.

The original job27631 artifacts are immutable inputs.  This program performs
no CHGNet work, generation, repair, retry, replacement, training, Slurm, or
GPU work.  The Materials Project key is read only from ``MP_API_KEY`` and is
never included in an artifact or log message.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Mapping, Sequence


CONTRACT_SCHEMA = "wqcodiff_existing22_mp_completion_contract_v1"
CACHE_SCHEMA = "wqcodiff_existing22_mp_query_cache_v1"
ATTEMPT_SCHEMA = "wqcodiff_existing22_mp_completed_attempt_v1"
REPORT_SCHEMA = "wqcodiff_existing22_mp_completion_report_v1"
TERMINAL_SCHEMA = "wqcodiff_existing22_mp_completion_terminal_v1"
CLAIM_SCHEMA = "wqcodiff_existing22_mp_completion_claim_v1"
THERMO_TYPE = "GGA_GGA+U"
THERMO_CRITERIA = {"thermo_types": [THERMO_TYPE]}
STRICT_THRESHOLD = 0.0
META_THRESHOLD = 0.1


class CompletionError(RuntimeError):
    """Raised when the frozen completion contract cannot be honored."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CompletionError(f"expected a JSON object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise CompletionError(
                    f"expected a JSON object at {path}:{number}"
                )
            rows.append(value)
    return rows


def write_json_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_jsonl_exclusive(
    path: Path, rows: Sequence[Mapping[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    dict(row),
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())


def append_jsonl(handle: Any, value: Mapping[str, Any]) -> None:
    handle.write(
        json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    )
    handle.flush()
    os.fsync(handle.fileno())


def finite_float(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise CompletionError(f"{label} is not finite")
    return result


def normalized_composition(value: Mapping[str, Any]) -> dict[str, float]:
    result = {
        str(element): finite_float(amount, f"composition[{element}]")
        for element, amount in value.items()
    }
    if not result or any(amount <= 0.0 for amount in result.values()):
        raise CompletionError("composition must have positive amounts")
    return dict(sorted(result.items()))


def scientific_decision(
    *,
    strict_count: int,
    meta_count: int,
    unknown_count: int,
    minimum_strict: int,
    minimum_meta: int,
) -> str:
    if strict_count >= minimum_strict and meta_count >= minimum_meta:
        return "PASS"
    if (
        strict_count + unknown_count < minimum_strict
        or meta_count + unknown_count < minimum_meta
    ):
        return "FAIL"
    return "INCONCLUSIVE_MP_COVERAGE"


def classify_hull(e_above_hull: float) -> str:
    if e_above_hull <= STRICT_THRESHOLD:
        return "strict_stable"
    if e_above_hull <= META_THRESHOLD:
        return "meta_only_stable"
    return "unstable"


def sanitized_error(exc: BaseException) -> dict[str, str]:
    message = str(exc)
    for separator in (
        "Content:",
        "Response:",
        "{\"data\":",
        "x-api-key",
        "api_key",
    ):
        message = message.split(separator, 1)[0]
    message = " ".join(message.split())[:400]
    return {"type": type(exc).__name__, "message": message}


def installed_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def validate_lowercase_sha256(value: str, label: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise CompletionError(f"{label} is not a lowercase SHA256")


def resolve_file(
    *,
    root: Path,
    base: Path,
    specification: Mapping[str, Any],
    label: str,
) -> Path:
    path = (base / str(specification["path"])).resolve()
    if root != path and root not in path.parents:
        raise CompletionError(f"{label} escapes the project root")
    if not path.is_file():
        raise CompletionError(f"{label} is missing: {path}")
    expected = str(specification["sha256"])
    validate_lowercase_sha256(expected, f"{label} sha256")
    if sha256_file(path) != expected:
        raise CompletionError(f"{label} SHA256 changed")
    return path


def validate_contract(
    *,
    project_root: Path,
    contract_path: Path,
) -> tuple[dict[str, Any], dict[str, Path]]:
    root = project_root.resolve()
    contract_path = contract_path.resolve()
    if root != contract_path and root not in contract_path.parents:
        raise CompletionError("contract escapes the project root")
    contract = load_json(contract_path)
    if contract.get("schema") != CONTRACT_SCHEMA:
        raise CompletionError("invalid completion contract schema")
    if contract.get("run_id") != "20260720_0401-crysllmgen-wq-final-v3":
        raise CompletionError("run identity changed")
    if contract["execution"] != {
        "location": "A800_login_node",
        "slurm_allowed": False,
        "gpu_allowed": False,
        "original_outputs_modified": False,
        "scientific_call_limit": 1,
        "overwrite": False,
    }:
        raise CompletionError("login-node execution scope changed")
    scope = contract["scope"]
    if (
        int(scope.get("fixed_queries", -1)) != 8
        or scope.get("all22_recomputation") is not True
        or int(scope.get("chgnet_calls", -1)) != 0
        or int(scope.get("geometry_changes", -1)) != 0
        or int(scope.get("slurm_jobs", -1)) != 0
        or int(scope.get("gpu_jobs", -1)) != 0
        or scope.get("new_generation") is not False
        or scope.get("training") is not False
        or scope.get("sample_retry_or_replacement") is not False
        or scope.get("mp_query_retry_or_replacement") is not False
    ):
        raise CompletionError("scientific scope changed")
    mp = contract["materials_project"]
    if (
        mp.get("client") != "mp_api.client.MPRester"
        or mp.get("method") != "get_entries_in_chemsys"
        or mp.get("thermo_type") != THERMO_TYPE
        or mp.get("additional_criteria") != THERMO_CRITERIA
        or mp.get("compatible_only") is not True
        or float(mp.get("strict_threshold_ev_per_atom", 1.0))
        != STRICT_THRESHOLD
        or float(mp.get("meta_threshold_ev_per_atom", 1.0))
        != META_THRESHOLD
        or int(mp.get("maximum_queries", -1)) != 8
        or mp.get("api_key_source") != "MP_API_KEY process environment only"
        or mp.get("api_key_serialized") is not False
        or mp.get("query_retry_or_replacement") is not False
    ):
        raise CompletionError("Materials Project query contract changed")

    paths: dict[str, Path] = {"contract": contract_path}
    authorization = contract["authorization"]
    paths["authorization"] = resolve_file(
        root=root,
        base=root,
        specification=authorization,
        label="authorization record",
    )
    source = contract["source_output"]
    source_directory = (root / str(source["directory"])).resolve()
    if root not in source_directory.parents or not source_directory.is_dir():
        raise CompletionError("source output directory is missing or unsafe")
    paths["source_directory"] = source_directory
    paths["original_contract"] = resolve_file(
        root=root,
        base=root,
        specification=source["original_contract"],
        label="original contract",
    )
    paths["frozen_a100_eval_sun"] = resolve_file(
        root=root,
        base=root,
        specification=source["frozen_a100_eval_sun"],
        label="frozen A100 evaluator",
    )
    for key in (
        "terminal_acceptance",
        "adapter_manifest",
        "input_manifest",
        "attempt_results",
        "attempt_summary",
        "sun_run_contract",
        "strict_relax_results",
        "meta_relax_results",
    ):
        paths[key] = resolve_file(
            root=root,
            base=source_directory,
            specification=source[key],
            label=key,
        )
    return contract, paths


def validate_frozen_inputs(
    *,
    contract: Mapping[str, Any],
    paths: Mapping[str, Path],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    terminal = load_json(paths["terminal_acceptance"])
    adapter = load_json(paths["adapter_manifest"])
    input_manifest = load_json(paths["input_manifest"])
    attempts = load_jsonl(paths["attempt_results"])
    summary = load_json(paths["attempt_summary"])
    sun_run_contract = load_json(paths["sun_run_contract"])
    strict_relax = load_jsonl(paths["strict_relax_results"])
    meta_relax = load_jsonl(paths["meta_relax_results"])

    denominator = contract["denominator"]
    if (
        terminal.get("schema")
        != "wqcodiff_existing22_chgnet_sun_terminal_acceptance_v1"
        or terminal.get("ok") is not True
        or terminal.get("scientific_decision") != "INCONCLUSIVE_MP_COVERAGE"
        or int(terminal.get("attempts", -1)) != 22
        or int(terminal.get("reconstructed_structures", -1)) != 17
        or int(terminal.get("frozen_structural_failures", -1)) != 5
        or terminal.get("new_generation") is not False
        or terminal.get("training") is not False
        or terminal.get("retry_or_replacement_used") is not False
        or terminal.get("observed_counts")
        != {
            "strict_full_sun": 0,
            "meta_full_sun": 3,
            "relaxation_or_hull_unknown": 8,
        }
    ):
        raise CompletionError("job27631 terminal identity changed")
    if (
        adapter.get("schema")
        != "wqcodiff_existing22_chgnet_sun_adapter_manifest_v1"
        or int(adapter.get("attempts", -1)) != 22
        or int(adapter.get("reconstructed_structures", -1)) != 17
        or int(adapter.get("failed_placeholders", -1)) != 5
        or adapter.get("geometry_repair_or_rescue") is not False
        or adapter.get("retry_or_replacement_used") is not False
    ):
        raise CompletionError("adapter manifest identity changed")
    if (
        input_manifest.get("schema")
        != "crysllmgen_r5c_a100_input_manifest_v1"
        or int(input_manifest.get("total_attempts", -1)) != 22
        or int(input_manifest.get("reconstructed_structures", -1)) != 17
        or input_manifest.get("retry_or_replacement_used") is not False
    ):
        raise CompletionError("input manifest identity changed")
    if (
        summary.get("schema") != "crysllmgen_r5c_a100_sun_summary_v1"
        or summary.get("ok") is not True
        or summary.get("denominator") != "all_generation_attempts"
        or summary.get("retry_or_replacement_used") is not False
        or summary.get("counts")
        != {
            "total_attempts": 22,
            "reconstructed": 17,
            "novel": 17,
            "unique": 17,
            "novel_unique": 17,
            "strict_full_sun": 0,
            "meta_full_sun": 3,
            "relaxation_or_hull_unknown": 8,
        }
    ):
        raise CompletionError("attempt summary identity changed")
    if (
        sun_run_contract.get("expected_attempts") != 22
        or sun_run_contract.get("offline") is not True
        or sun_run_contract.get("retry_or_replacement_used") is not False
    ):
        raise CompletionError("job27631 run contract identity changed")
    if (
        len(attempts) != int(denominator["attempts"])
        or len(strict_relax) != int(denominator["reconstructed_structures"])
        or strict_relax != meta_relax
    ):
        raise CompletionError("frozen result row counts changed")

    adapter_by_generation = {
        int(row["generation_ordinal"]): row
        for row in adapter["attempt_records"]
    }
    input_by_generation = {
        int(row["generation_ordinal"]): row
        for row in input_manifest["attempt_records"]
    }
    relax_by_index = {
        int(row["local_index"]): row for row in strict_relax
    }
    attempt_by_generation = {
        int(row["generation_ordinal"]): row for row in attempts
    }
    if (
        len(adapter_by_generation) != 22
        or len(input_by_generation) != 22
        or len(attempt_by_generation) != 22
        or len(relax_by_index) != 17
    ):
        raise CompletionError("duplicate or missing frozen identities")

    expected_records = list(contract["unknown_records"])
    if len(expected_records) != 8:
        raise CompletionError("contract must contain exactly eight unknowns")
    if len({str(row["attempt_id"]) for row in expected_records}) != 8:
        raise CompletionError("duplicate expected attempt IDs")
    if len({str(row["chemsys"]) for row in expected_records}) != 8:
        raise CompletionError("duplicate expected chemsys")
    observed_unknowns = {
        str(row["attempt_id"])
        for row in attempts
        if row.get("evaluation_status") == "relaxation_or_hull_unknown"
    }
    expected_unknowns = {
        str(row["attempt_id"]) for row in expected_records
    }
    if observed_unknowns != expected_unknowns:
        raise CompletionError("frozen unknown attempt set changed")

    frozen: list[dict[str, Any]] = []
    for expected in expected_records:
        generation = int(expected["generation_ordinal"])
        source_ordinal = int(expected["source_ordinal"])
        reconstructed_index = int(expected["reconstructed_index"])
        attempt_id = str(expected["attempt_id"])
        composition = normalized_composition(expected["composition"])
        chemsys = "-".join(sorted(composition))
        energy = finite_float(
            expected["energy_per_atom"], "expected energy_per_atom"
        )
        adapter_row = adapter_by_generation[generation]
        input_row = input_by_generation[generation]
        relax_row = relax_by_index[reconstructed_index]
        attempt_row = attempt_by_generation[generation]
        if (
            str(adapter_row.get("attempt_id")) != attempt_id
            or int(adapter_row.get("ordinal", -1)) != source_ordinal
            or adapter_row.get("status") != "succeeded"
            or str(input_row.get("attempt_id")) != attempt_id
            or input_row.get("status") != "succeeded"
            or int(input_row.get("reconstructed_index", -1))
            != reconstructed_index
            or str(input_row.get("structure_sha256"))
            != str(expected["input_structure_sha256"])
            or int(relax_row.get("local_index", -1)) != reconstructed_index
            or normalized_composition(relax_row.get("composition", {}))
            != composition
            or finite_float(
                relax_row.get("energy_per_atom"), "relax energy_per_atom"
            )
            != energy
            or str(attempt_row.get("attempt_id")) != attempt_id
            or attempt_row.get("generation_status") != "succeeded"
            or attempt_row.get("evaluation_status")
            != "relaxation_or_hull_unknown"
            or attempt_row.get("retry_or_replacement_used") is not False
            or finite_float(
                attempt_row.get("metrics", {}).get("energy_per_atom"),
                "attempt energy_per_atom",
            )
            != energy
            or attempt_row.get("metrics", {}).get("e_above_hull")
            is not None
            or attempt_row.get("metrics", {}).get("novel_unique") is not True
            or chemsys != str(expected["chemsys"])
        ):
            raise CompletionError(
                f"frozen identity mismatch for attempt {attempt_id}"
            )
        frozen.append(
            {
                "generation_ordinal": generation,
                "source_ordinal": source_ordinal,
                "reconstructed_index": reconstructed_index,
                "attempt_id": attempt_id,
                "input_structure_sha256": str(
                    expected["input_structure_sha256"]
                ),
                "composition": composition,
                "chemsys": chemsys,
                "energy_per_atom": energy,
            }
        )
    return attempts, frozen, attempt_by_generation


def slim_mp_entries(entries: Sequence[Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for entry in entries:
        result.append(
            {
                "entry_id": (
                    None
                    if getattr(entry, "entry_id", None) is None
                    else str(entry.entry_id)
                ),
                "composition": normalized_composition(
                    entry.composition.as_dict()
                ),
                "energy": finite_float(
                    entry.energy, "reference entry energy"
                ),
            }
        )
    result.sort(
        key=lambda row: (
            "" if row["entry_id"] is None else row["entry_id"],
            json.dumps(row["composition"], sort_keys=True),
            row["energy"],
        )
    )
    return result


def load_mp_dependencies() -> dict[str, Any]:
    import pymatgen.entries.compatibility as compatibility
    import pymatgen.entries.computed_entries as computed_entries

    sys.modules.setdefault("pymatgen.core.entries", computed_entries)
    sys.modules.setdefault(
        "pymatgen.analysis.compatibility", compatibility
    )
    from mp_api.client import MPRester
    from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
    from pymatgen.core import Composition
    from pymatgen.entries.computed_entries import ComputedEntry

    return {
        "MPRester": MPRester,
        "PDEntry": PDEntry,
        "PhaseDiagram": PhaseDiagram,
        "Composition": Composition,
        "ComputedEntry": ComputedEntry,
    }


def phase_diagram_from_slim(
    entries: Sequence[Mapping[str, Any]], dependencies: Mapping[str, Any]
) -> Any:
    computed_entry = dependencies["ComputedEntry"]
    return dependencies["PhaseDiagram"](
        [
            computed_entry(
                row["composition"],
                float(row["energy"]),
                entry_id=row["entry_id"],
            )
            for row in entries
        ]
    )


def query_frozen_records(
    *,
    records: Sequence[Mapping[str, Any]],
    cache_path: Path,
    api_key: str,
    dependencies: Mapping[str, Any],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    mprester = dependencies["MPRester"]
    composition_class = dependencies["Composition"]
    pd_entry = dependencies["PDEntry"]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("x", encoding="utf-8") as cache_handle:
        with mprester(api_key) as client:
            for index, source in enumerate(records, start=1):
                composition = composition_class(source["composition"])
                energy_per_atom = finite_float(
                    source["energy_per_atom"], "energy_per_atom"
                )
                record: dict[str, Any] = {
                    "schema": CACHE_SCHEMA,
                    "query_index": index,
                    "query_total": len(records),
                    "attempt_id": str(source["attempt_id"]),
                    "generation_ordinal": int(
                        source["generation_ordinal"]
                    ),
                    "source_ordinal": int(source["source_ordinal"]),
                    "reconstructed_index": int(
                        source["reconstructed_index"]
                    ),
                    "input_structure_sha256": str(
                        source["input_structure_sha256"]
                    ),
                    "composition": normalized_composition(
                        source["composition"]
                    ),
                    "formula": composition.reduced_formula,
                    "chemsys": str(source["chemsys"]),
                    "energy_per_atom": energy_per_atom,
                    "thermo_type": THERMO_TYPE,
                    "thermo_criteria": THERMO_CRITERIA,
                    "query_method": (
                        "mp_api.client.MPRester.get_entries_in_chemsys"
                    ),
                    "compatible_only": True,
                    "a100_hull_semantics": (
                        "PhaseDiagram(reference_entries) + "
                        "PDEntry(CHGNet_relaxed_energy)"
                    ),
                    "manual_query_attempts": 1,
                    "retry_or_replacement_used": False,
                }
                try:
                    reference_entries = client.get_entries_in_chemsys(
                        sorted(
                            element.symbol
                            for element in composition.elements
                        ),
                        compatible_only=True,
                        additional_criteria=THERMO_CRITERIA,
                    )
                    slim_entries = slim_mp_entries(reference_entries)
                    if not slim_entries:
                        raise CompletionError(
                            "get_entries_in_chemsys returned no entries"
                        )
                    phase_diagram = phase_diagram_from_slim(
                        slim_entries, dependencies
                    )
                    _, raw_e_above_hull = (
                        phase_diagram.get_decomp_and_e_above_hull(
                            pd_entry(
                                composition,
                                energy_per_atom * composition.num_atoms,
                            ),
                            allow_negative=True,
                        )
                    )
                    e_above_hull = max(
                        finite_float(raw_e_above_hull, "e_above_hull"),
                        0.0,
                    )
                    record.update(
                        {
                            "query_status": "resolved",
                            "reference_entry_count": len(slim_entries),
                            "reference_entries_sha256": canonical_sha256(
                                slim_entries
                            ),
                            "reference_entries": slim_entries,
                            "phase_diagram_constructed": True,
                            "e_above_hull": e_above_hull,
                            "posthoc_category": classify_hull(
                                e_above_hull
                            ),
                        }
                    )
                except Exception as exc:
                    record.update(
                        {
                            "query_status": "query_error",
                            "reference_entry_count": None,
                            "reference_entries_sha256": None,
                            "reference_entries": None,
                            "phase_diagram_constructed": False,
                            "e_above_hull": None,
                            "posthoc_category": "hull_unknown",
                            "error": sanitized_error(exc),
                        }
                    )
                results.append(record)
                append_jsonl(cache_handle, record)
                print(
                    json.dumps(
                        {
                            "query_index": index,
                            "query_total": len(records),
                            "source_ordinal": record["source_ordinal"],
                            "attempt_id": record["attempt_id"],
                            "chemsys": record["chemsys"],
                            "query_status": record["query_status"],
                            "posthoc_category": record[
                                "posthoc_category"
                            ],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    return results


def recompute_completed_attempts(
    *,
    original_attempts: Sequence[Mapping[str, Any]],
    query_results: Sequence[Mapping[str, Any]],
    completion_execution_patch_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    query_by_attempt = {
        str(row["attempt_id"]): row for row in query_results
    }
    if len(query_by_attempt) != len(query_results):
        raise CompletionError("duplicate query result attempt ID")
    completed: list[dict[str, Any]] = []
    for source in original_attempts:
        row = copy.deepcopy(dict(source))
        row["schema"] = ATTEMPT_SCHEMA
        row["job27631_attempt_sha256"] = canonical_sha256(source)
        row["job27631_execution_patch_sha256"] = source.get(
            "execution_patch_sha256"
        )
        row[
            "mp_completion_execution_patch_sha256"
        ] = completion_execution_patch_sha256
        attempt_id = str(source["attempt_id"])
        result = query_by_attempt.get(attempt_id)
        if result is None:
            row["mp_completion"] = {
                "applicable": False,
                "reason": "not_one_of_the_eight_frozen_hull_unknowns",
                "original_classification_preserved": True,
                "retry_or_replacement_used": False,
            }
        else:
            metrics = row["metrics"]
            resolved = result["query_status"] == "resolved"
            if resolved:
                e_above_hull = finite_float(
                    result["e_above_hull"], "completed e_above_hull"
                )
                strict = e_above_hull <= STRICT_THRESHOLD
                meta = e_above_hull <= META_THRESHOLD
                row["evaluation_status"] = "evaluated"
                metrics["e_above_hull"] = e_above_hull
                metrics["strict_full_sun"] = strict
                metrics["meta_full_sun"] = meta
            else:
                row[
                    "evaluation_status"
                ] = "relaxation_or_hull_unknown"
                metrics["e_above_hull"] = None
                metrics["strict_full_sun"] = False
                metrics["meta_full_sun"] = False
            row["mp_completion"] = {
                "applicable": True,
                "query_status": result["query_status"],
                "posthoc_category": result["posthoc_category"],
                "e_above_hull": result["e_above_hull"],
                "query_cache_record_sha256": canonical_sha256(result),
                "original_job27631_artifact_modified": False,
                "chgnet_rerun": False,
                "retry_or_replacement_used": False,
            }
        row["retry_or_replacement_used"] = False
        completed.append(row)

    if len(completed) != 22:
        raise CompletionError("completed denominator is not 22")
    if len({str(row["attempt_id"]) for row in completed}) != 22:
        raise CompletionError("completed attempt IDs are not unique")
    counts = {
        "total_attempts": len(completed),
        "reconstructed": sum(
            row.get("generation_status") == "succeeded"
            for row in completed
        ),
        "novel_unique": sum(
            row.get("metrics", {}).get("novel_unique") is True
            for row in completed
        ),
        "strict_full_sun": sum(
            row.get("metrics", {}).get("strict_full_sun") is True
            for row in completed
        ),
        "meta_full_sun": sum(
            row.get("metrics", {}).get("meta_full_sun") is True
            for row in completed
        ),
        "relaxation_or_hull_unknown": sum(
            row.get("evaluation_status")
            == "relaxation_or_hull_unknown"
            for row in completed
        ),
        "frozen_structural_failures": sum(
            row.get("generation_status") == "failed"
            for row in completed
        ),
    }
    if (
        counts["reconstructed"] != 17
        or counts["novel_unique"] != 17
        or counts["frozen_structural_failures"] != 5
        or counts["strict_full_sun"] > counts["meta_full_sun"]
    ):
        raise CompletionError("completed count invariants failed")
    return completed, counts


def execute(
    *,
    project_root: Path,
    contract_path: Path,
    execution_patch_sha256: str,
) -> dict[str, Any]:
    if os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOB_NAME"):
        raise CompletionError("this completion is forbidden inside Slurm")
    validate_lowercase_sha256(
        execution_patch_sha256, "execution patch SHA256"
    )
    api_key = os.environ.get("MP_API_KEY", "")
    if not api_key:
        raise CompletionError("MP_API_KEY is required")

    root = project_root.resolve()
    contract, paths = validate_contract(
        project_root=root, contract_path=contract_path
    )
    original_attempts, frozen, _ = validate_frozen_inputs(
        contract=contract, paths=paths
    )
    packages = {
        package: installed_version(package)
        for package in ("mp-api", "pymatgen", "emmet-core")
    }
    expected_versions = {
        "mp-api": contract["materials_project"]["mp_api_version"],
        "pymatgen": contract["materials_project"]["pymatgen_version"],
        "emmet-core": contract["materials_project"]["emmet_core_version"],
    }
    if packages != expected_versions:
        raise CompletionError(
            f"MP sidecar versions changed: {packages!r}"
        )
    dependencies = load_mp_dependencies()

    output = contract["output"]
    claim_path = (root / str(output["claim"])).resolve()
    output_directory = (root / str(output["directory"])).resolve()
    if (
        root not in claim_path.parents
        or root not in output_directory.parents
        or claim_path.exists()
        or output_directory.exists()
    ):
        raise CompletionError("claim or output identity already exists")
    cache_path = output_directory / str(output["query_cache"])
    completed_path = output_directory / str(
        output["completed_attempt_results"]
    )
    report_path = output_directory / str(output["completion_report"])
    terminal_path = output_directory / str(output["terminal_acceptance"])
    for path in (cache_path, completed_path, report_path, terminal_path):
        if path.exists():
            raise CompletionError(f"output already exists: {path}")

    created_at = datetime.now(timezone.utc).isoformat()
    claim = {
        "schema": CLAIM_SCHEMA,
        "created_at_utc": created_at,
        "status": "claimed_before_external_query",
        "run_id": contract["run_id"],
        "contract": str(paths["contract"]),
        "contract_sha256": sha256_file(paths["contract"]),
        "authorization_record_sha256": sha256_file(
            paths["authorization"]
        ),
        "execution_patch_sha256": execution_patch_sha256,
        "execution_location": "A800_login_node",
        "slurm_used": False,
        "gpu_used": False,
        "fixed_queries": 8,
        "fixed_attempt_ids": [row["attempt_id"] for row in frozen],
        "api_key_present": True,
        "api_key_serialized": False,
        "original_job27631_outputs_modified": False,
        "new_generation": False,
        "chgnet_calls": 0,
        "training": False,
        "retry_or_replacement_used": False,
    }
    write_json_exclusive(claim_path, claim)
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    output_directory.mkdir()

    query_results = query_frozen_records(
        records=frozen,
        cache_path=cache_path,
        api_key=api_key,
        dependencies=dependencies,
    )
    completed, counts = recompute_completed_attempts(
        original_attempts=original_attempts,
        query_results=query_results,
        completion_execution_patch_sha256=execution_patch_sha256,
    )
    write_jsonl_exclusive(completed_path, completed)

    rule = contract["decision_rule"]
    decision = scientific_decision(
        strict_count=counts["strict_full_sun"],
        meta_count=counts["meta_full_sun"],
        unknown_count=counts["relaxation_or_hull_unknown"],
        minimum_strict=int(rule["minimum_strict_full_sun_count"]),
        minimum_meta=int(rule["minimum_meta_full_sun_count"]),
    )
    consequence_key = {
        "PASS": "pass_consequence",
        "FAIL": "fail_consequence",
        "INCONCLUSIVE_MP_COVERAGE": "inconclusive_consequence",
    }[decision]
    query_counts = Counter(
        str(row["query_status"]) for row in query_results
    )
    category_counts = Counter(
        str(row["posthoc_category"]) for row in query_results
    )
    report = {
        "schema": REPORT_SCHEMA,
        "created_at_utc": created_at,
        "status": (
            "complete_all_eight_resolved"
            if int(query_counts["resolved"]) == 8
            else "complete_with_structured_query_errors"
        ),
        "run_id": contract["run_id"],
        "execution": {
            "location": "A800_login_node",
            "slurm_used": False,
            "gpu_used": False,
            "manual_query_calls": 8,
            "query_retry_or_replacement_used": False,
            "sample_retry_or_replacement_used": False,
            "api_key_serialized": False,
            "materials_project_database_version": (
                "not_queried_to_preserve_exact_eight_record_scope"
            ),
        },
        "client": {
            "python": (
                f"{sys.version_info.major}.{sys.version_info.minor}."
                f"{sys.version_info.micro}"
            ),
            "package_versions": packages,
            "thermo_type": THERMO_TYPE,
            "thermo_criteria": THERMO_CRITERIA,
            "compatible_only": True,
            "query_method": "MPRester.get_entries_in_chemsys",
            "a100_hull_semantics_reproduced": True,
        },
        "frozen_inputs": {
            label: {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for label, path in paths.items()
            if label != "source_directory"
        },
        "outputs": {
            "claim": {
                "path": str(claim_path),
                "bytes": claim_path.stat().st_size,
                "sha256": sha256_file(claim_path),
            },
            "query_cache": {
                "path": str(cache_path),
                "bytes": cache_path.stat().st_size,
                "sha256": sha256_file(cache_path),
                "rows": len(query_results),
            },
            "completed_attempt_results": {
                "path": str(completed_path),
                "bytes": completed_path.stat().st_size,
                "sha256": sha256_file(completed_path),
                "rows": len(completed),
            },
        },
        "query_counts": dict(query_counts),
        "query_category_counts": dict(category_counts),
        "all22_counts": counts,
        "all22_rates": {
            "novel_unique": counts["novel_unique"] / 22,
            "strict_full_sun": counts["strict_full_sun"] / 22,
            "meta_full_sun": counts["meta_full_sun"] / 22,
            "remaining_unknown": counts[
                "relaxation_or_hull_unknown"
            ]
            / 22,
        },
        "minimum_counts": {
            "strict_full_sun": int(
                rule["minimum_strict_full_sun_count"]
            ),
            "meta_full_sun": int(
                rule["minimum_meta_full_sun_count"]
            ),
        },
        "optimistic_upper_counts": {
            "strict_full_sun": counts["strict_full_sun"]
            + counts["relaxation_or_hull_unknown"],
            "meta_full_sun": counts["meta_full_sun"]
            + counts["relaxation_or_hull_unknown"],
        },
        "scientific_decision": decision,
        "decision_consequence": rule[consequence_key],
        "historical_formal_survival_result": (
            contract["frozen_history"]["formal_survival_result"]
        ),
        "historical_formal_survival_result_rewritten": False,
        "original_job27631_outputs_modified": False,
        "new_generation": False,
        "chgnet_calls": 0,
        "geometry_changes": 0,
        "training": False,
        "retry_or_replacement_used": False,
    }
    write_json_exclusive(report_path, report)

    terminal = {
        "schema": TERMINAL_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "ok": True,
        "evaluation_integrity": "PASS",
        "status": report["status"],
        "run_id": contract["run_id"],
        "contract_sha256": sha256_file(paths["contract"]),
        "authorization_record_sha256": sha256_file(
            paths["authorization"]
        ),
        "execution_patch_sha256": execution_patch_sha256,
        "claim_sha256": sha256_file(claim_path),
        "query_cache_sha256": sha256_file(cache_path),
        "completed_attempt_results_sha256": sha256_file(completed_path),
        "completion_report_sha256": sha256_file(report_path),
        "fixed_queries": 8,
        "resolved_queries": int(query_counts["resolved"]),
        "structured_query_errors": int(query_counts["query_error"]),
        "all22_counts": counts,
        "all22_rates": report["all22_rates"],
        "minimum_counts": report["minimum_counts"],
        "optimistic_upper_counts": report["optimistic_upper_counts"],
        "scientific_decision": decision,
        "decision_consequence": rule[consequence_key],
        "execution_location": "A800_login_node",
        "slurm_used": False,
        "gpu_used": False,
        "api_key_serialized": False,
        "original_job27631_outputs_modified": False,
        "formal_survival_result_rewritten": False,
        "new_generation": False,
        "chgnet_calls": 0,
        "geometry_changes": 0,
        "training": False,
        "retry_or_replacement_used": False,
    }
    write_json_exclusive(terminal_path, terminal)
    print("WQ_EXISTING22_MP_COMPLETION_INTEGRITY=PASS")
    print(f"WQ_EXISTING22_MP_COMPLETION_DECISION={decision}")
    print(
        json.dumps(
            {
                "status": terminal["status"],
                "resolved_queries": terminal["resolved_queries"],
                "structured_query_errors": terminal[
                    "structured_query_errors"
                ],
                "all22_counts": counts,
                "scientific_decision": decision,
                "terminal_acceptance": str(terminal_path),
                "terminal_acceptance_sha256": sha256_file(terminal_path),
            },
            sort_keys=True,
        )
    )
    return terminal


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--execution-patch-sha256", required=True)
    args = parser.parse_args()
    try:
        execute(
            project_root=args.project_root,
            contract_path=args.contract,
            execution_patch_sha256=args.execution_patch_sha256,
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "error": sanitized_error(exc),
                    "api_key_serialized": False,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
