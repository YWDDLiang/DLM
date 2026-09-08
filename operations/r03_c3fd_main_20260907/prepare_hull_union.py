#!/usr/bin/env python3
"""Prepare/finalize one shared, provenance-preserving official hull cache.

No API client, credentials, GPU, energies of generated structures, or SUN
labels are used here.  The unchanged official query script receives only the
missing chemsys list in a fresh directory.  A completed query's transport
failures are NOT sufficient to publish an evaluation-ready union cache.

Input manifest: {"schema":"r03_hull_union_inputs_v1", "purpose":"evaluation",
"phase":"pilot256", "inputs":[{"cell_id":"R_seed17", "arm":"R", "seed":17,
"type":"planner", "path":".../plans_for_dlm.jsonl", "expected_requests":256}]}.
Types are planner (body_eligible compositions as an upper bound), body, or
eval_paths. Counts are explicitly declared, without a hard-coded cohort size.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import sys
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
from crystal_dlm.fixed_slot import SYMBOL_TO_Z


INPUT_SCHEMA = "r03_hull_union_inputs_v1"
PREPARE_SCHEMA = "r03_hull_union_preparation_v1"
UNION_SCHEMA = "r03_official_hull_union_cache_v1"
CLEAN_SCHEMA = "h1_official_mp_gga_u_clean_cache_manifest_v1"
THERMO = {
    "query_method": "mp_api.client.MPRester.get_entries_in_chemsys",
    "compatible_only": True, "thermo_type": "GGA_GGA+U",
    "additional_criteria": {"thermo_types": ["GGA_GGA+U"]},
    "local_compatibility_reprocessing": False,
}
CACHE_OUTPUTS = {
    "slim_evaluation_cache": "official_slim_cache.jsonl",
    "query_audit": "query_audit.jsonl",
    "unresolved_chemsys": "unresolved_chemsys.jsonl",
}


class HullUnionError(ValueError):
    pass


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def identity(path: Path) -> dict[str, Any]:
    path = path.resolve()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def read_json(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise HullUnionError(f"expected JSON object: {path}")
    return result


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise HullUnionError(f"non-object row at {path}:{number}")
            rows.append(value)
    return rows


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False) + "\n")


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")


def verify_identity(path: Path, expected: Mapping[str, Any]) -> dict[str, Any]:
    actual = identity(path)
    if expected.get("bytes") != actual["bytes"] or expected.get("sha256") != actual["sha256"]:
        raise HullUnionError(f"frozen file bytes/SHA changed: {path}")
    return {**actual, "recorded_path": expected.get("path")}


def positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise HullUnionError(f"{name} must be a positive integer")
    return value


def canonical_chemsys(elements: Sequence[str]) -> str:
    if not isinstance(elements, (list, tuple)) or not elements:
        raise HullUnionError("composition elements must be a nonempty sequence")
    if any(not isinstance(symbol, str) or symbol not in SYMBOL_TO_Z for symbol in elements):
        raise HullUnionError("composition contains an unknown element symbol")
    return "-".join(sorted(set(elements)))


def validate_chemsys(name: Any) -> str:
    if not isinstance(name, str) or canonical_chemsys(name.split("-")) != name:
        raise HullUnionError(f"chemsys is not canonical: {name!r}")
    return name


def plan_chemsys(plan: Mapping[str, Any]) -> str:
    elements, counts = plan.get("elements"), plan.get("counts")
    if (not isinstance(elements, list) or not isinstance(counts, list) or len(elements) != len(counts)
            or len(set(elements)) != len(elements)):
        raise HullUnionError("Planner composition arrays are not aligned unique elements/counts")
    values = [positive_integer(value, "Plan count") for value in counts]
    if not values or sum(values) != positive_integer(plan.get("N"), "Plan N") or sum(values) > 20:
        raise HullUnionError("Planner MP20 count conservation changed")
    return canonical_chemsys(elements)


def dynamic_body_chemsys(text: str) -> str:
    """Read composition only, without ranking/filtering by geometry or energy."""
    elements = re.findall(r"<E_([A-Z][a-z]?)>", str(text))
    count = re.search(r"<N_(\d{3})>", str(text))
    if count is None or len(elements) != int(count.group(1)) or not 1 <= len(elements) <= 20:
        raise HullUnionError("native body does not contain an exact MP20 element canvas")
    return canonical_chemsys(elements)


def structure_chemsys(structure: Mapping[str, Any]) -> str:
    sites = structure.get("sites")
    if not isinstance(sites, list) or not sites:
        raise HullUnionError("evaluation structure lacks MSON sites")
    elements = []
    for site in sites:
        species = site.get("species") if isinstance(site, Mapping) else None
        if not isinstance(species, list) or not species:
            raise HullUnionError("evaluation site lacks species")
        for item in species:
            occupancy = item.get("occu", 1.0)
            if isinstance(occupancy, bool) or not isinstance(occupancy, (int, float)) or not math.isfinite(occupancy) or occupancy <= 0:
                raise HullUnionError("evaluation site occupancy is not positive finite")
            elements.append(item.get("element"))
    return canonical_chemsys(elements)


def extract_chemsys(row: Mapping[str, Any], input_type: str, *, purpose="evaluation") -> tuple[str | None, str | None]:
    if purpose not in ('evaluation','training_feedback'):
        raise HullUnionError('unknown reference-cache purpose')
    if purpose == 'training_feedback' and row.get('source_split') != 'train':
        raise HullUnionError('training hull references require explicit TRAIN-only conditions')
    if purpose == 'evaluation' and row.get("source_split") == "train":
        raise HullUnionError("hull union inputs are evaluation-only; training rows are forbidden")
    if input_type == "planner":
        if type(row.get("body_eligible")) is not bool:
            raise HullUnionError("native Planner input requires an explicit body_eligible boolean")
        if not row["body_eligible"]:
            return None, "planner_body_ineligible"
        if row.get("parsed") is False or row.get("attempt_status", "complete") != "complete":
            raise HullUnionError("Planner eligible flag conflicts with its failure status")
        plan = row.get("plan_state")
        if not isinstance(plan, Mapping):
            raise HullUnionError("eligible native Planner row lacks plan_state")
        return plan_chemsys(plan), None
    if input_type == "body":
        arrays = row.get("arrays")
        if isinstance(arrays, Mapping) and isinstance(arrays.get("species"), list):
            return canonical_chemsys(arrays["species"]), None
        for key in ("raw_body_text", "text", "body"):
            if isinstance(row.get(key), str) and row[key]:
                try:
                    return dynamic_body_chemsys(row[key]), None
                except HullUnionError:
                    pass
        if row.get("body_generation_complete") is False or row.get("parsed") is False or row.get("body_eligible") is False:
            return None, "body_composition_unavailable"
        raise HullUnionError("body input lacks usable composition and an explicit failure status")
    if input_type == "eval_paths":
        if row.get("source_split") != "evaluation":
            raise HullUnionError("eval_paths inputs require source_split=evaluation")
        if row.get("parseable") is False:
            return None, "recorded_endpoint_parse_failure"
        if isinstance(row.get("structure"), Mapping):
            return structure_chemsys(row["structure"]), None
        if row.get("endpoint") == "tau800":
            return None, "refined_structure_missing_no_native_substitution"
        if isinstance(row.get("body"), str):
            try:
                return dynamic_body_chemsys(row["body"]), None
            except HullUnionError:
                return None, "native_endpoint_composition_unavailable"
        return None, "endpoint_structure_unavailable"
    raise HullUnionError(f"unsupported explicitly declared input type {input_type!r}")


def collect_inputs(manifest_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = read_json(manifest_path)
    if manifest.get("schema") != INPUT_SCHEMA or manifest.get("purpose") not in ("evaluation", "training_feedback"):
        raise HullUnionError("an explicit evaluation/training_feedback hull input manifest is required")
    if not isinstance(manifest.get("phase"), str) or not manifest["phase"]:
        raise HullUnionError("declare the common experimental phase")
    inputs = manifest.get("inputs")
    if not isinstance(inputs, list) or not inputs:
        raise HullUnionError("declare all arm/seed input sources")
    systems: dict[str, list[str]] = {}
    summaries = []
    seen_cells = set()
    for spec in inputs:
        cell_id = spec.get("cell_id")
        if not isinstance(cell_id, str) or not cell_id or cell_id in seen_cells:
            raise HullUnionError("each input source needs a unique nonempty cell_id")
        seen_cells.add(cell_id)
        if not isinstance(spec.get('arm'),str) or not re.fullmatch(r'[A-Za-z0-9_.:-]{1,128}',spec['arm']) or 'seed' not in spec:
            raise HullUnionError('input must declare its stable arm identity and seed')
        kind = str(spec.get("type"))
        path = Path(spec["path"])
        path = path.resolve() if path.is_absolute() else (manifest_path.parent / path).resolve()
        file_id = identity(path)
        if spec.get("sha256") is not None and spec["sha256"] != file_id["sha256"]:
            raise HullUnionError(f"declared input SHA changed for {cell_id}")
        rows = read_jsonl(path)
        if manifest['purpose'] == 'training_feedback' and any(row.get('source_split') != 'train' for row in rows):
            raise HullUnionError('training hull references require explicit TRAIN-only conditions')
        expected = positive_integer(spec.get("expected_requests"), "expected_requests")
        if len(rows) != expected:
            raise HullUnionError(f"all-request denominator changed for {cell_id}: {len(rows)} != {expected}")
        sample_ids = [row.get("sample_idx") for row in rows]
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in sample_ids) or len(set(sample_ids)) != len(rows):
            raise HullUnionError(f"request IDs are absent/duplicated in {cell_id}")
        counts = Counter()
        cell_systems = set()
        for row in rows:
            name, reason = extract_chemsys(row, kind, purpose=manifest['purpose'])
            if name is None:
                counts[str(reason)] += 1
            else:
                cell_systems.add(name)
                counts["composition_available"] += 1
        for name in sorted(cell_systems):
            systems.setdefault(name, []).append(cell_id)
        summaries.append({
            "cell_id": cell_id, "arm": spec["arm"], "seed": spec["seed"], "type": kind,
            "endpoint": spec.get("endpoint"), "expected_requests": expected, "observed_requests": len(rows),
            "source": file_id, "sample_ids_sha256": canonical_sha256(sample_ids),
            "composition_counts": dict(counts), "chemsys": sorted(cell_systems),
            "scope": "model_generated_composition_upper_bound" if kind == "planner" else "actual_endpoint_compositions",
        })
    wanted = [{"query_index": index, "chemsys": name, "elements": name.split("-")}
              for index, name in enumerate(sorted(systems))]
    report = {"schema": INPUT_SCHEMA, "purpose": manifest['purpose'], "phase": manifest["phase"],
              "manifest": identity(manifest_path), "cells": summaries,
              "wanted_chemsys_count": len(wanted), "wanted_chemsys_sha256": canonical_sha256(wanted),
              "chemsys_sources": systems, "planner_upper_bound_present": any(row["type"] == "planner" for row in summaries),
              "generated_energy_or_SUN_used_for_selection": False, "training_use": manifest['purpose'] == 'training_feedback'}
    return wanted, report


def require_thermo(record: Mapping[str, Any], label: str) -> None:
    if any(record.get(key) != value for key, value in THERMO.items()):
        raise HullUnionError(f"{label} is not the fixed official GGA_GGA+U/compatible-only/no-local-reprocessing protocol")


def official_unresolved_kind(error: Any, chemsys: str) -> str | None:
    """Whitelist actual reference-set absence, never HTTP/transport failures."""
    if not isinstance(error, Mapping) or error.get("type") != "ContractError" or error.get("http_status") is not None:
        return None
    message = error.get("message")
    if message == "official get_entries_in_chemsys returned no entries":
        return "official_empty_reference_set"
    if not isinstance(message, str) or not message.startswith("missing unary references: "):
        return None
    try:
        missing = ast.literal_eval(message[len("missing unary references: "):])
    except (ValueError, SyntaxError):
        return None
    if (not isinstance(missing, list) or not missing or any(symbol not in chemsys.split("-") for symbol in missing)
            or len(set(missing)) != len(missing)):
        return None
    return "official_missing_elemental_references"


def validate_entries(entries: Any, chemsys: str) -> None:
    if not isinstance(entries, list) or not entries:
        raise HullUnionError(f"resolved cache has no entries for {chemsys}")
    requested = set(chemsys.split("-"))
    unary = set()
    for entry in entries:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("composition"), Mapping):
            raise HullUnionError(f"invalid slim entry composition for {chemsys}")
        composition = entry["composition"]
        if not composition or not set(composition).issubset(requested):
            raise HullUnionError(f"outside/empty slim entry elements for {chemsys}")
        values = list(composition.values())
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0 for value in values):
            raise HullUnionError(f"nonpositive/nonfinite reference stoichiometry for {chemsys}")
        energy = entry.get("energy")
        if isinstance(energy, bool) or not isinstance(energy, (int, float)) or not math.isfinite(energy):
            raise HullUnionError(f"nonfinite reference energy for {chemsys}")
        if len(composition) == 1:
            unary.update(composition)
    if unary != requested:
        raise HullUnionError(f"resolved cache lacks elemental references for {chemsys}")


def index_chemsys(rows: Sequence[Mapping[str, Any]], label: str) -> dict[str, Mapping[str, Any]]:
    result = {}
    for row in rows:
        name = validate_chemsys(row.get("chemsys"))
        if name in result:
            raise HullUnionError(f"duplicate {label} chemsys {name}")
        result[name] = row
    return result


def load_cache(cache: Path) -> dict[str, Any]:
    cache = cache.resolve()
    if not (cache / "completion_SUCCESS").is_file():
        raise HullUnionError(f"cache completion marker absent: {cache}")
    manifest_path = cache / "completion_manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("schema") not in (CLEAN_SCHEMA, UNION_SCHEMA):
        raise HullUnionError(f"cache lacks the supported complete official audit envelope: {cache}; schema={manifest.get('schema')}")
    require_thermo(manifest, "cache manifest")
    if not isinstance(manifest.get("database_version"), str) or not manifest["database_version"]:
        raise HullUnionError("cache must declare its official MP database_version")
    if manifest.get("all_resolved_phase_diagrams_constructed") is not True or manifest.get("all_resolved_elemental_references_present") is not True:
        raise HullUnionError("cache did not validate every resolved phase diagram and elemental reference")
    pins = {key: verify_identity(cache / filename, manifest.get("outputs", {}).get(key, {}))
            for key, filename in CACHE_OUTPUTS.items()}
    resolved = index_chemsys(read_jsonl(cache / "official_slim_cache.jsonl"), "resolved")
    unknown = index_chemsys(read_jsonl(cache / "unresolved_chemsys.jsonl"), "unresolved")
    audit_rows = read_jsonl(cache / "query_audit.jsonl")
    audits = index_chemsys(audit_rows, "query audit")
    if set(resolved) & set(unknown) or set(audits) != set(resolved) | set(unknown):
        raise HullUnionError("cache resolved/unresolved/query-audit accounting disagrees")
    if (manifest.get("query_count") != len(audits) or manifest.get("query_resolved") != len(resolved)
            or manifest.get("query_unresolved") != len(unknown)
            or sorted(row.get("query_index") for row in audit_rows) != list(range(len(audit_rows)))):
        raise HullUnionError("cache query counts/indices are incomplete")
    official_unknown, query_errors = {}, {}
    for name, audit in audits.items():
        require_thermo(audit, f"query audit {name}")
        if audit.get("elements") != name.split("-") or audit.get("query_total") != len(audits):
            raise HullUnionError(f"query-audit chemical identity changed: {name}")
        if name in resolved:
            entries = resolved[name].get("entries")
            validate_entries(entries, name)
            if (audit.get("query_status") != "resolved" or audit.get("entry_count") != len(entries)
                    or audit.get("slim_entries_sha256") != canonical_sha256(entries)
                    or audit.get("phase_diagram_constructed") is not True
                    or audit.get("unary_reference_elements") != name.split("-")):
                raise HullUnionError(f"resolved slim record is not bound to its official validation: {name}")
        else:
            record = unknown[name]
            error = record.get("error", record.get("source_error"))
            if record.get("elements") != name.split("-") or error != audit.get("error") or audit.get("query_status") != "query_error":
                raise HullUnionError(f"unresolved record is not bound to its query audit: {name}")
            kind = official_unresolved_kind(error, name)
            if kind:
                official_unknown[name] = {**record, "official_reference_kind": kind}
            else:
                query_errors[name] = {**record, "classification": "query_or_transport_error_not_official_reference_absence"}
    return {
        "resolved": resolved, "official_unresolved": official_unknown, "query_errors": query_errors, "audits": audits,
        "original_record_sha256": {name: canonical_sha256(row) for name, row in {**resolved, **unknown}.items()},
        "identity": {"directory": str(cache), "completion_manifest": identity(manifest_path),
                     "completion_marker": identity(cache / "completion_SUCCESS"), "outputs_verified": pins,
                     "other_recorded_outputs": {key: value for key, value in manifest.get("outputs", {}).items() if key not in CACHE_OUTPUTS},
                     "database_version": manifest.get("database_version"), "package_versions": manifest.get("package_versions"),
                     "fresh_empty_cache": manifest.get("fresh_empty_cache"), "schema": manifest["schema"]},
        "manifest": manifest,
    }


def choose_known(caches: Sequence[Mapping[str, Any]], wanted: set[str]) -> tuple[dict, dict, dict, list]:
    resolved, unknown, provenance, errors = {}, {}, {}, []
    for cache in caches:
        for name, row in cache["query_errors"].items():
            if name in wanted:
                errors.append({"chemsys": name, "cache": cache["identity"], "record": row,
                               "accepted_as_official_unresolved": False})
        for kind, records in (("resolved", cache["resolved"]), ("official_unresolved", cache["official_unresolved"])):
            for name, row in records.items():
                if name not in wanted:
                    continue
                source = {"cache": cache["identity"], "classification": kind,
                          "source_record_sha256": cache["original_record_sha256"][name],
                          "source_query_audit_sha256": canonical_sha256(cache["audits"][name]),
                          "source_query_audit": cache["audits"][name]}
                if kind == "resolved":
                    if name in resolved and canonical_sha256(resolved[name]) != canonical_sha256(row):
                        raise HullUnionError(f"conflicting verified reference sets for {name}; choose one declared cache snapshot")
                    resolved[name] = row
                    unknown.pop(name, None)
                    provenance[name] = source
                elif name not in resolved:
                    if name in unknown and canonical_sha256(unknown[name]) != canonical_sha256(row):
                        # Either official absence still means unknown; source
                        # precedence is explicit input order, never an energy.
                        continue
                    unknown[name] = row
                    provenance[name] = source
    return resolved, unknown, provenance, errors


def load_legacy_resolved_supplement(cache: Path, wanted: set[str]) -> dict[str, Any]:
    """Read only missing resolved systems from the explicit August 13 schema.

    The original full-file and per-entry digests remain bound to the original
    audit. Phase diagrams are revalidated locally; no query or compatibility
    processing takes place, and legacy unknown rows are never imported.
    """
    from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry
    from pymatgen.core import Composition
    cache = cache.resolve()
    manifest_path = cache/'completion_manifest.json'
    manifest = read_json(manifest_path)
    if (not (cache/'completion_SUCCESS').is_file()
            or manifest.get('schema') != 'h1_r03_best_fresh_official_mp_query_v1'
            or manifest.get('query_method') != 'MPRester.get_entries_in_chemsys'
            or manifest.get('compatible_only') is not True or manifest.get('thermo_type') != 'GGA_GGA+U'
            or manifest.get('additional_criteria') != {'thermo_types':['GGA_GGA+U']}
            or manifest.get('fresh_empty_cache') is not True
            or not isinstance(manifest.get('database_version'),str) or not manifest['database_version']):
        raise HullUnionError('legacy resolved supplement lacks its original official query contract')
    names = {'resolved':'official_slim_cache.jsonl','unresolved':'unresolved_chemsys.jsonl','query_audit':'query_audit.jsonl'}
    pins = {key:verify_identity(cache/name,manifest.get('cache_identities',{}).get(key,{})) for key,name in names.items()}
    resolved = index_chemsys(read_jsonl(cache/names['resolved']),'legacy resolved')
    unknown = index_chemsys(read_jsonl(cache/names['unresolved']),'legacy unresolved')
    audits = index_chemsys(read_jsonl(cache/names['query_audit']),'legacy audit')
    if (set(resolved)&set(unknown) or set(audits) != set(resolved)|set(unknown)
            or manifest.get('wanted_chemsys') != len(audits)
            or manifest.get('resolved_chemsys') != len(resolved) or manifest.get('unresolved_chemsys') != len(unknown)
            or sorted(row.get('query_index') for row in audits.values()) != list(range(len(audits)))):
        raise HullUnionError('legacy cache full-file accounting is incomplete')
    selected, adapted = {}, {}
    for name in sorted(wanted & set(resolved)):
        row, audit = resolved[name], audits[name]
        entries = row.get('entries')
        validate_entries(entries,name)
        original_sha = hashlib.sha256(json.dumps(entries,ensure_ascii=False,sort_keys=True,
            separators=(',',':'),allow_nan=False).encode()).hexdigest()
        if (audit.get('query_status') != 'resolved' or audit.get('error') is not None
                or audit.get('query_method') != 'MPRester.get_entries_in_chemsys'
                or audit.get('compatible_only') is not True or audit.get('thermo_type') != 'GGA_GGA+U'
                or audit.get('elements') != name.split('-') or audit.get('entry_count') != len(entries)
                or audit.get('entries_sha256') != original_sha):
            raise HullUnionError('legacy resolved entries differ from their recorded official query')
        diagram = PhaseDiagram([PDEntry(Composition(entry['composition']),entry['energy']) for entry in entries])
        if {element.symbol for element in diagram.elements} != set(name.split('-')):
            raise HullUnionError('legacy supplement phase diagram changed its element set')
        selected[name] = row
        adapted[name] = {**audit,**THERMO,'query_total':len(audits),
            'slim_entries_sha256':canonical_sha256(entries),'phase_diagram_constructed':True,
            'unary_reference_elements':name.split('-'),'legacy_original_query_audit':audit,
            'legacy_original_query_audit_sha256':canonical_sha256(audit),
            'validation_origin':'local_revalidation_of_unchanged_official_cached_entries', 'new_query':False}
    return {'resolved':selected,'official_unresolved':{},'query_errors':{},'audits':adapted,
        'original_record_sha256':{name:canonical_sha256(row) for name,row in selected.items()},
        'identity':{'directory':str(cache),'completion_manifest':identity(manifest_path),
            'completion_marker':identity(cache/'completion_SUCCESS'),'outputs_verified':pins,
            'database_version':manifest['database_version'],'package_versions':manifest.get('package_versions'),
            'schema':manifest['schema'],'selected_missing_resolved_systems':sorted(selected),
            'legacy_unknowns_imported':False}, 'manifest':manifest}


def write_source_manifest(directory: Path, names: Sequence[str]) -> str:
    with (directory / "SOURCE_SHA256.txt").open("x", encoding="utf-8", newline="\n") as handle:
        for name in sorted(names):
            if Path(name).name != name or name == "SOURCE_SHA256.txt":
                raise HullUnionError("source-manifest entries must be simple relative filenames")
            handle.write(identity(directory / name)["sha256"] + "  " + name + "\n")
    return identity(directory / "SOURCE_SHA256.txt")["sha256"]


def prepare(*, inputs_manifest: Path, known_caches: Sequence[Path], query_config: Path,
            query_source: Path, run_root: Path, legacy_resolved_caches: Sequence[Path] = ()) -> dict[str, Any]:
    if run_root.exists():
        raise FileExistsError(run_root)
    wanted_rows, inputs_report = collect_inputs(inputs_manifest)
    wanted = {row["chemsys"] for row in wanted_rows}
    loaded = [load_cache(path) for path in known_caches]
    resolved, unknown, provenance, excluded = choose_known(loaded, wanted)
    for path in legacy_resolved_caches:
        supplement = load_legacy_resolved_supplement(path,wanted-set(resolved)-set(unknown))
        new_resolved, _, new_provenance, _ = choose_known([supplement],wanted)
        resolved.update(new_resolved)
        provenance.update(new_provenance)
        loaded.append(supplement)
    reused_versions = {source["cache"]["database_version"] for source in provenance.values()}
    if len(reused_versions) > 1:
        raise HullUnionError("known rows span MP database versions; choose one version before querying the union")
    missing = wanted - set(resolved) - set(unknown)
    config = read_json(query_config)
    require_thermo(config.get("thermo", {}), "query configuration")
    if (config["thermo"].get("fresh_empty_cache") is not True
            or config["thermo"].get("reuse_any_historical_or_august_cache") is not False):
        raise HullUnionError("unchanged official query config must create a fresh missing-only cache")
    if not config.get("runtime", {}).get("official_mp_python"):
        raise HullUnionError("query config lacks the registered official interpreter")
    run_root.mkdir(parents=True, exist_ok=False)
    write_json(run_root / "INPUT_SOURCES.json", inputs_report)
    write_jsonl(run_root / "union_wanted_chemsys.jsonl", wanted_rows)
    write_jsonl(run_root / "known_resolved.jsonl", [resolved[name] for name in sorted(resolved)])
    write_jsonl(run_root / "known_official_unresolved.jsonl", [unknown[name] for name in sorted(unknown)])
    write_json(run_root / "known_provenance.json", provenance)
    write_jsonl(run_root / "excluded_query_errors.jsonl", excluded)
    query_root = run_root / "missing_query"
    input_dir = query_root / "inputs"
    input_dir.mkdir(parents=True)
    missing_rows = [{"query_index": index, "chemsys": name, "elements": name.split("-")}
                    for index, name in enumerate(sorted(missing))]
    write_jsonl(input_dir / "wanted_chemsys.jsonl", missing_rows)
    write_json(input_dir / "input_manifest.json", {
        "schema": "r03_hull_union_missing_query_inputs_v1", "wanted_chemsys_count": len(missing_rows),
        "wanted_chemsys_sha256": canonical_sha256(missing_rows), "selection": "common_composition_union_minus_verified_known",
        "full_union": identity(run_root / "union_wanted_chemsys.jsonl"),
        "input_sources": identity(run_root / "INPUT_SOURCES.json"),
        "fresh_empty_cache_scope": "new_missing_subset_only", "training_use": False,
    })
    (input_dir / "inputs_SUCCESS").touch()
    write_source_manifest(input_dir, ("wanted_chemsys.jsonl", "input_manifest.json", "inputs_SUCCESS"))
    source_dir = run_root / "query_source"
    source_dir.mkdir()
    for name in ("protocol.py", "query_official_mp.py"):
        shutil.copyfile(query_source / name, source_dir / name)
    shutil.copyfile(query_config, source_dir / "CONFIG.json")
    source_sha = write_source_manifest(source_dir, ("protocol.py", "query_official_mp.py", "CONFIG.json"))
    command = {
        "needed": bool(missing), "source_manifest_sha256": source_sha,
        "environment": {"H1_ACTIVE_DENOMINATOR": "256"},
        "legacy_protocol_import_note": "256 enables the unchanged protocol import only; actual cohort and query counts remain in their input manifests and are not changed",
        "argv_without_credential": [config["runtime"]["official_mp_python"], str((source_dir / "query_official_mp.py").resolve()),
            "--config", str((source_dir / "CONFIG.json").resolve()), "--source-dir", str(source_dir.resolve()),
            "--source-manifest-sha256", source_sha, "--run-root", str(query_root.resolve())],
        "credential_argument_supplied_by_root": "--key-env NAME or --key-file PATH; this adapter never reads credentials",
        "query_script_or_evaluator_modified": False,
    }
    write_json(run_root / "QUERY_COMMAND.json", command)
    frozen_files = {str(path.relative_to(run_root)): identity(path) for path in run_root.rglob("*") if path.is_file()}
    report = {
        "schema": PREPARE_SCHEMA, "status": "prepared", "purpose": inputs_report['purpose'], "phase": inputs_report["phase"],
        "input_sources": inputs_report, "wanted_chemsys": len(wanted), "known_resolved": len(resolved),
        "known_official_unresolved": len(unknown), "missing_chemsys": sorted(missing),
        "historical_query_error_records_excluded": len(excluded),
        "known_cache_sources": [cache["identity"] for cache in loaded],
        "reused_database_version": next(iter(reused_versions)) if reused_versions else None,
        "frozen_files": frozen_files, "query_command": command,
        "actual_endpoint_subset_check_required": inputs_report["planner_upper_bound_present"],
        "new_api_calls": 0, "training_use": inputs_report['training_use'],
    }
    write_json(run_root / "PREPARE_FINAL.json", report)
    (run_root / "PREPARE_SUCCESS").touch()
    return report


def verify_preparation(run_root: Path) -> dict[str, Any]:
    if not (run_root / "PREPARE_SUCCESS").is_file():
        raise HullUnionError("union preparation is not complete")
    report = read_json(run_root / "PREPARE_FINAL.json")
    if report.get("schema") != PREPARE_SCHEMA or report.get("purpose") not in ("evaluation", "training_feedback"):
        raise HullUnionError("wrong union preparation schema/purpose")
    for relative, expected in report["frozen_files"].items():
        path = (run_root / relative).resolve()
        if run_root.resolve() not in path.parents:
            raise HullUnionError("preparation file escaped its run root")
        verify_identity(path, expected)
    for cell in report["input_sources"]["cells"]:
        verify_identity(Path(cell["source"]["path"]), cell["source"])
    return report


def endpoint_subset_report(actual_inputs: Path, wanted: set[str]) -> dict[str, Any]:
    actual_rows, sources = collect_inputs(actual_inputs)
    if any(cell["type"] == "planner" for cell in sources["cells"]):
        raise HullUnionError("endpoint subset verification requires body/eval_paths, not another Planner upper bound")
    actual = {row["chemsys"] for row in actual_rows}
    outside = sorted(actual - wanted)
    return {"schema": "r03_hull_union_endpoint_subset_v1", "sources": sources,
            "actual_chemsys": len(actual), "outside_registered_union": outside,
            "is_subset": not outside, "not_covered": len(outside), "training_use": sources['training_use']}


def finalize(*, run_root: Path, fresh_cache: Path | None = None, actual_inputs: Path | None = None) -> dict[str, Any]:
    preparation = verify_preparation(run_root)
    final = run_root / "official_mp_cache"
    if final.exists():
        raise FileExistsError(final)
    wanted = {row["chemsys"] for row in read_jsonl(run_root / "union_wanted_chemsys.jsonl")}
    missing = set(preparation["missing_chemsys"])
    resolved = dict(index_chemsys(read_jsonl(run_root / "known_resolved.jsonl"), "prepared known resolved"))
    unknown = dict(index_chemsys(read_jsonl(run_root / "known_official_unresolved.jsonl"), "prepared official unresolved"))
    provenance = read_json(run_root / "known_provenance.json")
    fresh_identity = None
    database_version = preparation.get("reused_database_version")
    version_conflict = None
    errors = {}
    if missing:
        location = fresh_cache or run_root / "missing_query" / "official_mp_cache"
        fresh = load_cache(location)
        if fresh["manifest"].get("schema") != CLEAN_SCHEMA or fresh["manifest"].get("fresh_empty_cache") is not True:
            raise HullUnionError("new missing subset must come from the unchanged fresh official-query cache")
        fresh_version = fresh["manifest"]["database_version"]
        if database_version is not None and fresh_version != database_version:
            version_conflict = {"known_database_version": database_version, "fresh_database_version": fresh_version,
                                "required_action": "requery_the_entire_registered_union_at_one_MP_database_version"}
        if database_version is None:
            database_version = fresh_version
        query_inputs = fresh["manifest"].get("inputs", {})
        verify_identity(run_root / "missing_query/inputs/wanted_chemsys.jsonl", query_inputs.get("wanted_chemsys", {}))
        verify_identity(run_root / "missing_query/inputs/input_manifest.json", query_inputs.get("input_manifest", {}))
        if set(fresh["audits"]) != missing:
            raise HullUnionError("fresh query covered a different set than the frozen missing subset")
        new_resolved, new_unknown, new_provenance, _ = choose_known([fresh], missing)
        resolved.update(new_resolved)
        unknown.update(new_unknown)
        provenance.update(new_provenance)
        errors = dict(fresh["query_errors"])
        fresh_identity = fresh["identity"]
    elif fresh_cache is not None:
        raise HullUnionError("no missing systems were registered; an unrelated fresh cache cannot be substituted")
    uncovered = sorted(wanted - set(resolved) - set(unknown))
    endpoints = endpoint_subset_report(actual_inputs, wanted) if actual_inputs else None
    if endpoints is not None and endpoints["sources"]["phase"] != preparation["phase"]:
        raise HullUnionError("actual endpoint sources belong to a different experimental phase")
    if uncovered or version_conflict is not None or (endpoints is not None and not endpoints["is_subset"]):
        actual_outside = endpoints["outside_registered_union"] if endpoints is not None else []
        all_uncovered = sorted(set(uncovered) | set(actual_outside))
        blocked = {"schema": UNION_SCHEMA, "status": "blocked", "coverage_accounted": False,
                   "not_covered": len(all_uncovered), "uncovered_chemsys": all_uncovered,
                   "registered_union_uncovered_chemsys": uncovered,
                   "query_or_transport_errors": errors, "actual_endpoint_check": endpoints,
                   "database_version_conflict": version_conflict,
                   "known_cache_sources": preparation["known_cache_sources"], "fresh_query_source": fresh_identity,
                   "transport_errors_promoted_to_official_unresolved": 0,
                   "evaluation_cache_published": False, "training_use": preparation['training_use']}
        write_json(run_root / "FINALIZE_BLOCKED.json", blocked)
        return blocked
    if set(resolved) & set(unknown) or set(resolved) | set(unknown) != wanted:
        raise HullUnionError("final resolved/unresolved union accounting changed")
    final.mkdir()
    slim_rows = [resolved[name] for name in sorted(resolved)]
    unresolved_rows = []
    audits = []
    provenance_rows = []
    for index, name in enumerate(sorted(wanted)):
        source = provenance[name]
        audit = dict(source["source_query_audit"])
        audit.update(query_index=index, query_total=len(wanted), union_source={
            "cache_manifest": source["cache"]["completion_manifest"],
            "source_record_sha256": source["source_record_sha256"],
            "source_query_audit_sha256": source["source_query_audit_sha256"],
            "original_query_index": source["source_query_audit"]["query_index"],
        })
        audits.append(audit)
        provenance_rows.append({"chemsys": name, **source})
        if name in unknown:
            record = unknown[name]
            unresolved_rows.append({"chemsys": name, "elements": name.split("-"),
                "reason": "official_reference_contract_unresolved", "official_reference_kind": record["official_reference_kind"],
                "error": record.get("error", record.get("source_error")),
                "source_record_sha256": source["source_record_sha256"]})
    write_jsonl(final / "official_slim_cache.jsonl", slim_rows)
    write_jsonl(final / "unresolved_chemsys.jsonl", unresolved_rows)
    write_jsonl(final / "query_audit.jsonl", audits)
    write_jsonl(final / "cache_provenance.jsonl", provenance_rows)
    manifest = {
        "schema": UNION_SCHEMA, "query_status": "coverage_accounted", "coverage_accounted": True,
        "database_version": database_version if wanted else "not_applicable_empty_union",
        "database_version_scope": "all_reference_rows" if wanted else "no_reference_rows",
        "not_covered": 0, "query_count": len(wanted), "query_resolved": len(resolved), "query_unresolved": len(unknown),
        **THERMO, "fresh_empty_cache": False, "assembly_method": "verified_common_union_of_registered_sources",
        "historical_cache_rows_reused": preparation["known_resolved"] + preparation["known_official_unresolved"],
        "new_official_query_count": len(missing), "known_cache_sources": preparation["known_cache_sources"],
        "fresh_query_source": fresh_identity, "preparation": identity(run_root / "PREPARE_FINAL.json"),
        "all_resolved_phase_diagrams_constructed": True, "all_resolved_elemental_references_present": True,
        "transport_errors_promoted_to_official_unresolved": 0, "training_use": preparation['training_use'],
        "purpose": preparation['purpose'],
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "actual_endpoint_check": endpoints,
        "actual_endpoint_subset_check_required": preparation["actual_endpoint_subset_check_required"] and endpoints is None,
        "outputs": {key: identity(final / filename) for key, filename in CACHE_OUTPUTS.items()},
        "cache_provenance": identity(final / "cache_provenance.jsonl"),
    }
    write_json(final / "completion_manifest.json", manifest)
    (final / "completion_SUCCESS").touch()
    return manifest


def verify_endpoints(*, run_root: Path, actual_inputs: Path, output_report: Path) -> dict[str, Any]:
    preparation = verify_preparation(run_root)
    cache = load_cache(run_root / "official_mp_cache")
    wanted = {row["chemsys"] for row in read_jsonl(run_root / "union_wanted_chemsys.jsonl")}
    report = endpoint_subset_report(actual_inputs, wanted)
    if report["sources"]["phase"] != preparation["phase"]:
        raise HullUnionError("actual endpoint sources belong to a different experimental phase")
    actual = {name for cell in report["sources"]["cells"] for name in cell["chemsys"]}
    not_covered = actual - set(cache["resolved"]) - set(cache["official_unresolved"])
    report.update(coverage_accounted=not not_covered and report["is_subset"], not_covered=len(not_covered),
                  actual_uncovered_chemsys=sorted(not_covered), cache_identity=cache["identity"])
    write_json(output_report, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--inputs-manifest", type=Path, required=True)
    prep.add_argument("--known-cache", type=Path, action="append", default=[])
    prep.add_argument('--legacy-resolved-cache', type=Path, action='append', default=[],
                      help='Explicit old official schema used only for missing resolved systems')
    prep.add_argument("--query-config", type=Path, required=True)
    prep.add_argument("--query-source-dir", type=Path, required=True)
    prep.add_argument("--run-root", type=Path, required=True)
    finish = commands.add_parser("finalize")
    finish.add_argument("--run-root", type=Path, required=True)
    finish.add_argument("--fresh-cache", type=Path)
    finish.add_argument("--actual-inputs-manifest", type=Path)
    verify = commands.add_parser("verify-endpoints")
    verify.add_argument("--run-root", type=Path, required=True)
    verify.add_argument("--inputs-manifest", type=Path, required=True)
    verify.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        report = prepare(inputs_manifest=args.inputs_manifest, known_caches=args.known_cache,
                         query_config=args.query_config, query_source=args.query_source_dir, run_root=args.run_root,
                         legacy_resolved_caches=args.legacy_resolved_cache)
        print(canonical_json({"status": report["status"], "wanted_chemsys": report["wanted_chemsys"],
                              "known_resolved": report["known_resolved"], "known_official_unresolved": report["known_official_unresolved"],
                              "missing_chemsys": report["missing_chemsys"], "query_command": report["query_command"]}))
    elif args.command == "finalize":
        report = finalize(run_root=args.run_root, fresh_cache=args.fresh_cache, actual_inputs=args.actual_inputs_manifest)
        print(canonical_json({key: report.get(key) for key in ("status", "query_status", "coverage_accounted", "not_covered",
                                                               "query_resolved", "query_unresolved", "actual_endpoint_subset_check_required")}))
        if report.get("coverage_accounted") is not True:
            raise SystemExit(2)
    else:
        report = verify_endpoints(run_root=args.run_root, actual_inputs=args.inputs_manifest, output_report=args.output_report)
        print(canonical_json({key: report[key] for key in ("is_subset", "coverage_accounted", "not_covered")}))
        if not report["coverage_accounted"]:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
