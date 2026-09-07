#!/usr/bin/env python3
"""CPU-only all-request scoring of a completed R03 canary or pilot.

This invokes only hull-cache coverage verification and the existing frozen
N/U/hull evaluator. It never queries references, generates labels, launches
Slurm, or selects a policy from the small engineering canary.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Mapping

SOURCE = Path(__file__).resolve().parents[2]
SCHEMA = "r03_trial_evaluation_manifest_v1"
RESULT_SCHEMA = "r03_trial_evaluation_final_v1"
ENDPOINTS = ("native", "tau800")
ROLE_ORDER = ("R", "I", "G", "P")
ENGINEERING_ERROR_WORDS = (
    "out of memory", "cuda error", "cudaerror", "cuda oom", "brokenprocesspool",
    "modulenotfounderror", "no module named", "importerror", "device-side assert",
    "dll load failed", "undefined symbol", "memoryerror", "cublas_status_alloc_failed",
    "cudnn_status_alloc_failed", "cannot allocate memory",
)


def read_json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"expected JSONL objects: {path}")
    return rows


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def file_identity(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def resolved(value, base):
    if not isinstance(value, str) or not value:
        raise ValueError("explicit source paths are required")
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def require_success(directory):
    if not (directory / "_SUCCESS").is_file() or (directory / "_FAILED").exists():
        raise ValueError(f"component/stage is incomplete or failed: {directory}")


def validate_manifest(manifest, manifest_path):
    if manifest.get("schema") != SCHEMA or not isinstance(manifest.get("phase"), str) or not manifest["phase"]:
        raise ValueError("explicit trial schema and phase are required")
    count = manifest.get("expected_requests")
    formal = manifest.get("cohort_role") == "independent_main"
    preview_roles = manifest.get('preview_roles')
    preview = preview_roles is not None
    if preview and (formal or count != 256 or not isinstance(preview_roles, list)
                    or not 0 < len(preview_roles) < 4 or len(set(preview_roles)) != len(preview_roles)
                    or not set(preview_roles).issubset(ROLE_ORDER)):
        raise ValueError('preview must declare a subset of completed 256-request pilot methods')
    if type(count) is not int or count not in ((256, 500) if formal else (16, 256)):
        raise ValueError("registered trial counts are 16 canary or 256 pilot requests per method")
    if formal and (manifest.get("selected_role") not in ("G", "P") or not manifest.get("method_freeze")):
        raise ValueError("formal components require an explicit frozen candidate")
    matched_interface = manifest.get("include_matched_interface_reference", False)
    if type(matched_interface) is not bool or (count != 16 and matched_interface):
        raise ValueError("include_matched_interface_reference is an optional canary-only boolean")
    construction_geometry = manifest.get("registered_construction_geometry", count == 256 or formal)
    if type(construction_geometry) is not bool or ((count == 256 or formal) and not construction_geometry):
        raise ValueError("the registered pilot requires the new G/P construction geometry")
    validity_artifact = manifest.get("validity_artifact", "legacy_existing_direct_read_only")
    if validity_artifact not in ("legacy_existing_direct_read_only", "basic_comp_struct_only"):
        raise ValueError("validity artifact must be the registered existing or comp/struct-only form")
    if (count == 256 or formal) and validity_artifact != "basic_comp_struct_only":
        raise ValueError("the 256 pilot reports only comp_valid and struct_valid")
    methods = manifest.get("methods")
    if not isinstance(methods, list) or any(not isinstance(item, dict) for item in methods):
        raise ValueError("declare the complete method set")
    expected_roles = ({"I", "G", "P"} if matched_interface else {"G", "P"}) if count == 16 else set(ROLE_ORDER)
    if formal:
        expected_roles = {"R", manifest['selected_role']}
    elif preview:
        expected_roles = set(preview_roles)
    roles = [item.get("role") for item in methods]
    ids = [item.get("method_id") for item in methods]
    if len(roles) != len(expected_roles) or set(roles) != expected_roles or len(set(ids)) != len(ids):
        raise ValueError("methods are missing, duplicated, or outside the registered trial; no partial denominator scoring")
    if any(not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", value) for value in ids):
        raise ValueError("stable, unique method IDs are required")
    seeds = [item.get("planner_seed") for item in methods]
    if any(type(seed) is not int or seed < 0 for seed in seeds) or len(set(seeds)) != 1:
        raise ValueError("trial methods must use their declared common Planner seed")
    base = Path(manifest_path).resolve().parent
    normalized = []
    for item in sorted(methods, key=lambda row: ROLE_ORDER.index(row["role"])):
        normalized.append({**item, "component_dir": resolved(item.get("component_dir"), base)})
    if len({item["component_dir"] for item in normalized}) != len(normalized):
        raise ValueError("one component cannot be counted as two methods")
    return {"phase": manifest["phase"], "scope": "formal" if formal else ('preview' if preview else ("canary" if count == 16 else "pilot")),
            "cohort_role": "independent_main" if formal else "fixed_development",
            "selected_role": manifest.get("selected_role"),
            "expected_requests": count, "methods": normalized,
            "include_matched_interface_reference": matched_interface,
            "registered_construction_geometry": construction_geometry,
            "validity_artifact": validity_artifact,
            "method_freeze": resolved(manifest["method_freeze"], base) if manifest.get("method_freeze") else None,
            "reuse_report": resolved(manifest['reuse_report'], base) if manifest.get('reuse_report') else None,
            "frozen_config": resolved(manifest.get("frozen_config"), base),
            "hull_run_root": resolved(manifest.get("hull_run_root"), base)}


def endpoint_cache_key(row):
    if not row["success"]:
        return row["trajectory_id"]
    text = json.dumps(row["structure"], sort_keys=True) if row.get("structure") is not None else row.get("body")
    if not isinstance(text, str):
        raise ValueError("successful endpoint has no actual labelled geometry")
    return hashlib.sha256(text.encode()).hexdigest()


def construction_evidence(component, method, final, trial, expected_ids):
    """Bind actual construction outputs, not only the reused G/P method names."""
    body_dir = component / "body"
    require_success(body_dir)
    config_path = body_dir / "run_config.json"
    config = read_json(config_path)
    geometry = trial["registered_construction_geometry"] and method["role"] in ("G", "P")
    legacy = trial["scope"] == "canary" and not trial["registered_construction_geometry"]
    observed = config.get("construction_geometry_enabled")
    if observed is None and legacy and config.get("geometry_support_scope") in (
            "post_construction_repair_only", "original_constructor_masks_only"):
        observed = False  # Explicit old scope, not missing-value equality for the new method.
    recorded = final.get("construction_geometry", False if legacy else None)
    if observed is not geometry or recorded is not geometry:
        raise ValueError("actual construction geometry differs from this registered method generation")
    singleton = trial["scope"] == "pilot" or trial["registered_construction_geometry"]
    if singleton and (config.get("max_batch_size") != 1 or final.get("body_batch_size") != 1):
        raise ValueError("all new-geometry trial construction must use registered body batch1")
    evidence = {"geometry_enabled": geometry, "body_batch_size": config.get("max_batch_size"),
                "new_construction_checked": not legacy, "source_files": [file_identity(config_path)]}
    if legacy:
        return evidence
    raw_path = body_dir / "raw_generations.jsonl"
    rows = read_rows(raw_path)
    if [r.get("sample_idx") for r in rows] != expected_ids:
        raise ValueError("actual construction ledger lost or reordered trial request IDs")
    partition_path = body_dir / "batch_partition.json"
    partition = json.loads(partition_path.read_text(encoding="utf-8"))
    eligible = [r["sample_idx"] for r in rows if r.get("body_eligible") is True]
    if (not isinstance(partition, list) or any(not isinstance(batch, list) or len(batch) != 1 for batch in partition)
            or sorted(value for batch in partition for value in batch) != sorted(eligible)):
        raise ValueError("actual construction partition is not complete singleton accounting")
    evidence["source_files"].extend([file_identity(raw_path), file_identity(partition_path)])
    if not geometry:
        if any((r.get("construction_geometry") or {}).get("enabled") is True for r in rows):
            raise ValueError("R/I records unexpectedly used construction geometry")
        return evidence
    construction_path = body_dir / "construction_raw_generations.jsonl"
    originals = read_rows(construction_path)
    if [r.get("sample_idx") for r in originals] != expected_ids:
        raise ValueError("pre-repair construction snapshot is not the same all-request ledger")
    evidence["source_files"].append(file_identity(construction_path))
    snapshots = []
    for row, original in zip(rows, originals):
        for key in ("sample_idx", "attempt_id", "body_noise_seed", "plan_state", "body_eligible", "body_generation_complete"):
            if row.get(key) != original.get(key):
                raise ValueError(f"repair changed the saved construction identity: {key}")
        monitor = original.get("construction_geometry") or {}
        if original.get("body_eligible") is False:
            if original.get("attempt_status") != "planner_failure" or original.get("body_generation_complete") is not False:
                raise ValueError("unattempted construction lacks its explicit upstream failure")
            snapshot = {"state": "upstream_failure", "reason": original.get("reason")}
        elif monitor.get("enabled") is not True:
            raise ValueError("G/P attempted construction without the registered geometry bridge")
        elif original.get("body_generation_complete") is True:
            tokens = row.get("construction_raw_body_token_ids")
            if (not isinstance(tokens, list) or not tokens or any(type(v) is not int for v in tokens)
                    or tokens != original.get("raw_body_token_ids")):
                raise ValueError("actual construction_raw_body_token_ids differ from the pre-repair snapshot")
            snapshot = {"state": "complete", "tokens": tokens, "construction_status": row.get("construction_status")}
        else:
            failure = monitor.get("failure")
            if (original.get("attempt_status") != "construction_constraint_failure" or monitor.get("status") != "no_legal_support"
                    or not isinstance(failure, dict) or failure.get("failure_class") != "construction_constraint_no_legal_support"
                    or not isinstance(failure.get("partial_body_token_ids"), list) or not failure["partial_body_token_ids"]
                    or row.get("repair_skip_reason") != "construction_incomplete" or row.get("repair_used") is not False
                    or row.get("construction_raw_body_token_ids") is not None or row.get("raw_body_token_ids") is not None):
                raise ValueError("no-support construction must remain an explicit un-repaired failure")
            if row.get("construction_geometry") != monitor:
                raise ValueError("repair replaced the recorded no-support construction evidence")
            snapshot = {"state": "no_legal_support", "failure": failure}
        snapshots.append({"sample_idx": row["sample_idx"], "attempt_id": row["attempt_id"],
                          "body_noise_seed": row.get("body_noise_seed"), **snapshot})
    evidence["snapshots"] = snapshots
    return evidence


def validate_endpoint(component, method, endpoint, count, expected_ids, trial):
    directory, label_dir = component / endpoint, component / (endpoint + "_labels")
    legacy = trial["validity_artifact"] == "legacy_existing_direct_read_only"
    validity_dir = component / (endpoint + ("_direct" if legacy else "_validity"))
    for path in (directory, label_dir, validity_dir):
        require_success(path)
    paths_file, labels_file = directory / "paths.jsonl", label_dir / "labels.jsonl"
    rows, labels = read_rows(paths_file), read_rows(labels_file)
    if any(type(row.get(key)) is not int for row in rows for key in ("sample_idx", "evaluation_ordinal")):
        raise ValueError("request IDs/ordinals must remain actual integers, not coercible values")
    if len(rows) != count or len(labels) != count or [row.get("evaluation_ordinal") for row in rows] != list(range(count)):
        raise ValueError("endpoint or label request denominator/order changed")
    if [row.get("sample_idx") for row in rows] != expected_ids:
        raise ValueError("endpoint global request IDs differ from its component")
    indexed = {row.get("trajectory_id"): row for row in rows}
    labelled = {row.get("trajectory_id"): row for row in labels}
    attempts = [row.get("attempt_id") for row in rows]
    if (len(indexed) != count or None in indexed or indexed.keys() != labelled.keys() or len(labelled) != count
            or len(set(attempts)) != count or any(not isinstance(a, str) or not a for a in attempts)):
        raise ValueError("endpoint/label/original attempt identities are incomplete or duplicated")
    report = read_json(label_dir / "LABEL_FINAL.json")
    if report.get("purpose") != "evaluation" or report.get("requested") != count or report.get("completed") != count:
        raise ValueError("label purpose or completion count is not the registered evaluation")
    actual_statuses = Counter(row.get("status") for row in labels)
    if {k: v for k, v in report.get("statuses", {}).items() if v} != dict(actual_statuses):
        raise ValueError("label status receipt differs from actual label rows")
    if actual_statuses.get("worker_error", 0):
        raise ValueError("worker failures require engineering recovery before scientific scoring")
    for row in rows:
        label = labelled[row["trajectory_id"]]
        if (row.get("source_split") != "evaluation" or row.get("purpose") != "evaluation"
                or row.get("endpoint") != endpoint or row.get("method_id") != method["method_id"]
                or type(row.get("success")) is not bool or type(row.get("parseable")) is not bool):
            raise ValueError("input endpoint role, method, or explicit execution state changed")
        seed = row.get("planner_seed", (row.get("planner_record") or {}).get("seed"))
        if seed != method["planner_seed"]:
            raise ValueError("saved request Planner seed differs from the trial manifest")
        if label.get("source_split") != "evaluation" or label.get("endpoint") != endpoint:
            raise ValueError("training or wrong-endpoint labels cannot enter this trial")
        for field in ("group_id", "source_row_idx"):
            if str(label.get(field)) != str(row.get(field)):
                raise ValueError(f"label/request identity mismatch: {field}")
        if label.get("endpoint_cache_key") != endpoint_cache_key(row):
            raise ValueError("label does not belong to this exact endpoint geometry")
        error = str(label.get("error") or "").lower()
        if any(word in error for word in ENGINEERING_ERROR_WORDS):
            raise ValueError("label environment/OOM failure requires engineering recovery")
        if not row["success"] and (label.get("status") != "generation_failure" or label.get("verified") is not False):
            raise ValueError("a failed endpoint acquired a physical success label")
    validity = read_json(validity_dir / "report.json")
    validity_rows = read_rows(validity_dir / "attempt_metrics.jsonl")
    if validity.get("attempts") != count or len(validity_rows) != count:
        raise ValueError("basic validity denominator differs from the actual request count")
    if [(r.get("ordinal"), r.get("attempt_id")) for r in validity_rows] != list(enumerate(attempts)):
        raise ValueError("basic validity rows do not preserve the original request order/IDs")
    if any(type(row.get("ordinal")) is not int for row in validity_rows):
        raise ValueError("basic validity ordinals must remain exact integers")
    for field, key in (("comp_valid", "comp_valid_count"), ("struct_valid", "struct_valid_count")):
        if validity.get(key) != sum(row.get(field) is True for row in validity_rows):
            raise ValueError("basic validity counts differ from the attempt ledger")
    if validity.get("generation_succeeded") != sum(row["success"] for row in rows):
        raise ValueError("basic validity source success count differs from endpoint availability")
    if not legacy and (validity.get("schema") != "crysllmgen_basic_validity_v1"
                       or validity.get("reported_metrics") != ["comp_valid", "struct_valid"]
                       or "joint_valid" not in validity.get("omitted_metrics", [])
                       or any("valid" in row for row in validity_rows)):
        raise ValueError("new pilot validity contains metrics beyond comp_valid and struct_valid")
    files = [paths_file, labels_file, label_dir / "LABEL_FINAL.json", validity_dir / "report.json", validity_dir / "attempt_metrics.jsonl"]
    return {"endpoint": endpoint, "paths": paths_file, "labels": labels_file,
            "rows": rows, "label_report": report, "validity": validity,
            "source_files": [file_identity(path) for path in files]}


def preflight_components(trial):
    count = trial["expected_requests"]
    verified, offsets, body_seeds, refiner_seeds = [], set(), set(), set()
    for method in trial["methods"]:
        component = method["component_dir"]
        require_success(component)
        final = read_json(component / "COMPONENT_FINAL.json")
        for field, expected in (("role", method["role"]), ("method_id", method["method_id"]), ("requests", count),
                                ("planner_seed", method["planner_seed"]), ("labels_purpose", "evaluation"), ("pooled_NU_scored_here", False)):
            if final.get(field) != expected:
                raise ValueError(f"component does not match its registered {field}")
        if trial["validity_artifact"] == "basic_comp_struct_only" and (
                final.get("validity_metrics") != ["comp_valid", "struct_valid"]
                or final.get("joint_valid_reported") is not False
                or final.get("direct_suite_run") is not False):
            raise ValueError("pilot component did not preserve the comp/struct-only validity request")
        offset = final.get("sample_index_offset")
        if type(offset) is not int or offset < 0:
            raise ValueError("component global offset is not explicit")
        offsets.add(offset)
        body_seeds.add(final.get("body_seed"))
        refiner_seeds.add(final.get("refiner_seed"))
        cells = {endpoint: validate_endpoint(component, method, endpoint, count, list(range(offset, offset + count)), trial)
                 for endpoint in ENDPOINTS}
        construction = construction_evidence(component, method, final, trial, list(range(offset, offset + count)))
        native, refined = cells["native"]["rows"], cells["tau800"]["rows"]
        for a, b in zip(native, refined):
            for field in ("sample_idx", "attempt_id", "group_id", "plan_state", "body_noise_seed"):
                if a.get(field) != b.get(field):
                    raise ValueError(f"native/refined request identity differs: {field}")
        sources = [file_identity(component / "COMPONENT_FINAL.json")]
        for path in (component / "COMPONENT_CONFIG.json", component / "body/run_config.json"):
            if path.is_file():
                sources.append(file_identity(path))
        sources.extend(construction["source_files"])
        verified.append({**method, "component_final": final, "cells": cells, "source_files": sources,
                         "construction_evidence": construction})
    if len(offsets) != 1 or len(body_seeds) != 1 or len(refiner_seeds) != 1 or None in body_seeds or None in refiner_seeds:
        raise ValueError("trial methods changed registered global offsets or common body/refiner seeds")
    candidate = {item["role"]: item for item in verified if item["role"] != "R"}
    if trial.get("scope") == "formal":
        if set(candidate) != {trial["selected_role"]}:
            raise ValueError("formal components differ from the declared frozen candidate")
        return verified
    if trial["registered_construction_geometry"] and {'G', 'P'}.issubset(candidate):
        if candidate["G"]["construction_evidence"]["snapshots"] != candidate["P"]["construction_evidence"]["snapshots"]:
            raise ValueError("G/P actual construction outputs or common no-support failures differ")
    if not candidate:
        return verified
    anchor = candidate.get('G', next(iter(candidate.values())))["cells"]["native"]["rows"]
    for role, item in candidate.items():
        for a, b in zip(anchor, item["cells"]["native"]["rows"]):
            for field in ("sample_idx", "attempt_id", "plan_state", "prompt", "body_noise_seed"):
                if a.get(field) != b.get(field):
                    raise ValueError(f"I/G/P do not share the registered Planner request: {role}:{field}")
            if role == "P":
                for field in ("species_program", "species_program_indices", "species_program_source", "r03_control"):
                    if (a.get("planner_record") or {}).get(field) != (b.get("planner_record") or {}).get(field):
                        raise ValueError("G/P program conditions differ")
    return verified


def actual_hull_manifest(trial, components):
    return {"schema": "r03_hull_union_inputs_v1", "purpose": "evaluation", "phase": trial["phase"],
            "inputs": [{"cell_id": f"{item['method_id']}:{endpoint}", "arm": item["role"], "seed": item["planner_seed"],
                        "type": "eval_paths", "endpoint": endpoint, "path": str(item["cells"][endpoint]["paths"]),
                        "sha256": file_identity(item["cells"][endpoint]["paths"])["sha256"],
                        "expected_requests": trial["expected_requests"]}
                       for item in components for endpoint in ENDPOINTS]}


def run_command(command, name, output_dir):
    """Only CPU verification/scoring entrypoints; inherit genuine Slurm context."""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    write_json(output_dir / f"{name}.command.json", {"command": command, "started_utc": datetime.now(timezone.utc).isoformat()})
    with (output_dir / f"{name}.out").open("x") as stdout, (output_dir / f"{name}.err").open("x") as stderr:
        completed = subprocess.run(command, env=env, stdout=stdout, stderr=stderr, check=False)
    if completed.returncode:
        raise RuntimeError(f"CPU {name} failed with exit {completed.returncode}; inspect preserved logs")


def finite(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(float(value))


def wilson95_percent(successes, count):
    z = 1.959963984540054
    fraction, correction = successes / count, z * z / count
    center = (fraction + correction / 2) / (1 + correction)
    radius = z * math.sqrt(fraction * (1 - fraction) / count + z * z / (4 * count * count)) / (1 + correction)
    return [0. if successes == 0 else 100 * max(0., center - radius),
            100. if successes == count else 100 * min(1., center + radius)]


def summarize_evaluation(directory, cell, method, trial):
    require_success(directory)
    report = read_json(directory / "EVALUATION_FINAL.json")
    rows = read_rows(directory / "attempt_results.jsonl")
    count, endpoint = trial["expected_requests"], cell["endpoint"]
    if (report.get("counts", {}).get("requests") != count or len(rows) != count or report.get("endpoint") != endpoint
            or report.get("cohort_role") != trial.get("cohort_role", "fixed_development")
            or report.get("policy_stage") != ("reference" if method["role"] == "R" else "final")
            or "conditional_1000" in report):
        raise ValueError("scorer changed the registered trial endpoint/denominator/selection")
    frozen = read_json(trial["frozen_config"])
    if (report.get("frozen_nu_source_sha256") != frozen.get("frozen_code", {}).get("eval_sun_sha256")
            or report.get("terminal_protocol") != cell["label_report"].get("protocol")
            or report.get("verification_protocol") != cell["label_report"].get("verification_protocol")
            or Path(report.get("official_cache", "")).resolve() != (trial["hull_run_root"] / "official_mp_cache").resolve()):
        raise ValueError("scorer did not preserve the frozen N/U, label, or common-cache identity")
    for result, source in zip(rows, cell["rows"]):
        for field in ("sample_idx", "trajectory_id", "group_id"):
            if result.get(field) != source.get(field):
                raise ValueError("scored result identity differs from the accepted endpoint input")
    for key in ("strict_sun", "meta_sun", "verified_strict_sun", "verified_meta_sun", "terminal_verified", "reconstructed"):
        if report["counts"].get(key) != sum(row.get(key) is True for row in rows):
            raise ValueError("scorer headline does not match its actual attempt results")
    statuses = Counter(row.get("official_hull_status") for row in rows)
    if statuses.get("official_cache_not_covered", 0):
        raise ValueError("scorer found an uncovered hull system after mandatory coverage verification")
    return {"method_id": method["method_id"], "role": method["role"], "endpoint": endpoint, "denominator": count,
            "novelty_uniqueness": {key: report['counts'][key] for key in ('novel', 'unique_representative', 'novel_unique')},
            "validity": {key: cell["validity"][key] for key in ("comp_valid_count", "struct_valid_count")},
            "headline": {"strict_sun": report["counts"]["strict_sun"], "meta_sun": report["counts"]["meta_sun"],
                         "strict_percent": 100 * report["counts"]["strict_sun"] / count,
                         "meta_percent": 100 * report["counts"]["meta_sun"] / count,
                         "strict_wilson95_percent": wilson95_percent(report['counts']['strict_sun'], count),
                         "meta_wilson95_percent": wilson95_percent(report['counts']['meta_sun'], count),
                         "interval_scope": "descriptive binomial approximation; pooled uniqueness dependence is not modeled"},
            "verified": {"terminals": report["counts"]["terminal_verified"], "strict_sun": report["counts"]["verified_strict_sun"],
                         "meta_sun": report["counts"]["verified_meta_sun"]},
            "unknown": {"hull_statuses": dict(statuses),
                        "official_unresolved": statuses.get("official_cache_unresolved", 0),
                        "not_reconstructed": statuses.get("input_not_reconstructed", 0),
                        "terminal_energy_missing": sum(not finite(row.get("terminal_energy_eV_atom")) for row in rows),
                        "e_hull_missing": sum(not finite(row.get("e_above_hull_eV_atom")) for row in rows)},
            "evaluation_directory": str(directory), "evaluation_source_files": [file_identity(directory / name)
                                                                                  for name in ("EVALUATION_FINAL.json", "attempt_results.jsonl")]}


def adoption_rule(scope, summaries, frozen=None):
    if scope == 'preview':
        return {'enabled': False, 'selected_role': None, 'reason': 'Completed pilot methods only; full four-arm trial is pending.'}
    if scope == "canary":
        return {"enabled": False, "selected_role": None, "reason": "engineering canary only; no policy selection from 16 requests"}
    if scope == "formal":
        if frozen is None:
            raise ValueError("formal scoring requires the existing method freeze")
        return {"enabled": False, "selected_role": frozen['selected_role'], "method_freeze": frozen,
                "reason": "Independent confirmation; no selection from formal scores."}
    lookup = {(row["role"], row["endpoint"]): row for row in summaries}
    g_tau, p_tau = lookup["G", "tau800"], lookup["P", "tau800"]
    gates = {"tau800_strict_sun_nondecreasing": p_tau["headline"]["strict_sun"] >= g_tau["headline"]["strict_sun"],
             "tau800_meta_sun_nondecreasing": p_tau["headline"]["meta_sun"] >= g_tau["headline"]["meta_sun"],
             "at_least_one_tau800_sun_count_increases": any(p_tau["headline"][key] > g_tau["headline"][key]
                                                            for key in ("strict_sun", "meta_sun"))}
    selected = "P" if all(gates.values()) else "G"
    if frozen is not None:
        return {"enabled": False, "selected_role": frozen["selected_role"], "method_freeze": frozen,
                "late_development_rule_role": selected, "gates": gates,
                "reason": "Method already frozen; late pilot scores cannot revise that decision.",
                "statistical_significance_claimed": False, "fresh_formal_requests_still_required": True}
    return {"enabled": True, "selected_role": selected, "gates": gates,
            "rule": "tau800 Strict and Meta SUN >= G; at least one tau800 SUN count increases",
            "diagnostics_not_additional_gates": ["native SUN", "comp_valid", "struct_valid", "verified SUN"],
            "development_selection_only": True, "statistical_significance_claimed": False,
            "fresh_formal_requests_still_required": True}


def render_summary(report):
    lines = [f"# R03 {report['phase']} evaluation", "", f"{report['expected_requests']} requests per method and endpoint.", "",
             "| Method | Endpoint | Requests | comp_valid/struct_valid | N/U | Strict/Meta SUN | Verified Strict/Meta | Official unresolved | Missing terminal energy |",
             "|---|---|---:|---|---|---|---|---:|---:|"]
    for row in report["methods"]:
        validity, headline, verified, unknown = row["validity"], row["headline"], row["verified"], row["unknown"]
        lines.append(f"| {row['method_id']} | {row['endpoint']} | {row['denominator']} | "
                     f"{validity['comp_valid_count']}/{validity['struct_valid_count']} | "
                     f"{row['novelty_uniqueness']['novel']}/{row['novelty_uniqueness']['unique_representative']} | "
                     f"{headline['strict_sun']}/{headline['meta_sun']} | {verified['strict_sun']}/{verified['meta_sun']} | "
                     f"{unknown['official_unresolved']} | {unknown['terminal_energy_missing']} |")
    lines += ["", "All requests remain in each denominator; verified results are reported separately.", ""]
    if report["adoption"]["enabled"]:
        lines += [f"Development rule selects **{report['adoption']['selected_role']}**.",
                  report["adoption"]["rule"] + ".", "This is development selection, not evidence of statistical significance."]
    elif report["adoption"].get("method_freeze"):
        lines.append(f"Frozen candidate remains **{report['adoption']['selected_role']}**.")
        if 'late_development_rule_role' in report['adoption']:
            lines.append(f"The late development counts satisfy the rule for **{report['adoption']['late_development_rule_role']}**; this does not revise the timed freeze.")
        else:
            lines.append("Independent confirmation; no selection from formal scores.")
    elif report['scope'] == 'preview':
        lines.append('Completed pilot methods only; full four-arm comparison is pending. No method selection is made.')
    else:
        lines.append("Engineering canary only: no P/G selection is made.")
    return "\n".join(lines) + "\n"


def evaluate_trial(manifest_path, output_dir, *, command_runner=run_command):
    manifest_path, output_dir = Path(manifest_path).resolve(), Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError("trial evaluation output must be new")
    manifest = read_json(manifest_path)
    trial = validate_manifest(manifest, manifest_path)
    components = preflight_components(trial)
    config_pin = file_identity(trial["frozen_config"])
    output_dir.mkdir(parents=True, exist_ok=False)
    source_pins = [file_identity(manifest_path), config_pin]
    frozen = None
    if trial["method_freeze"]:
        frozen = read_json(trial["method_freeze"])
        if (frozen.get("schema") != "r03_method_freeze_v1" or frozen.get("selected_role") not in ("G", "P")
                or datetime.fromisoformat(frozen["recorded_utc"].replace("Z", "+00:00"))
                > datetime.fromisoformat(frozen["freeze_deadline_utc"].replace("Z", "+00:00"))):
            raise ValueError("invalid or late method freeze receipt")
        source_pins.append(file_identity(trial["method_freeze"]))
        if trial['scope'] == 'formal' and trial['selected_role'] != frozen['selected_role']:
            raise ValueError('formal candidate differs from the timed method freeze')
    for item in components:
        source_pins.extend(item["source_files"])
        for cell in item["cells"].values():
            source_pins.extend(cell["source_files"])
    reusable = {}
    if trial['reuse_report']:
        prior = read_json(trial['reuse_report'])
        if (trial['scope'] != 'pilot' or prior.get('scope') != 'preview'
                or prior['expected_requests'] != trial['expected_requests']
                or prior['scoring_source']['sha256'] != file_identity(SOURCE / 'scripts/evaluate_programmed_paths.py')['sha256']):
            raise ValueError('only matching completed pilot preview scores can be reused')
        require_success(trial['reuse_report'].parent)
        prior_pins = prior['source_files'] + [pin for row in prior['methods'] for pin in row['evaluation_source_files']]
        for pin in prior_pins:
            if file_identity(pin['path'])['sha256'] != pin['sha256']:
                raise ValueError('a completed preview source or score changed')
        source_pins.extend(prior_pins + [file_identity(trial['reuse_report'])])
        reusable = {(row['role'], row['endpoint']): row for row in prior['methods']}
    try:
        actual = output_dir / "ACTUAL_HULL_INPUTS.json"
        write_json(actual, actual_hull_manifest(trial, components))
        coverage_path = output_dir / "HULL_ENDPOINT_COVERAGE.json"
        command_runner([sys.executable, str(SOURCE / "operations/r03_c3fd_main_20260907/prepare_hull_union.py"),
                        "verify-endpoints", "--run-root", str(trial["hull_run_root"]),
                        "--inputs-manifest", str(actual), "--output-report", str(coverage_path)], "coverage", output_dir)
        coverage = read_json(coverage_path)
        if coverage.get("coverage_accounted") is not True or coverage.get("not_covered") != 0 or coverage.get("is_subset") is not True:
            raise ValueError("actual endpoint hull coverage is not completely accounted; no scores may be produced")
        summaries = []
        for item in components:
            parent = output_dir / "evaluations" / item["role"]
            parent.mkdir(parents=True, exist_ok=False)
            for endpoint in ENDPOINTS:
                cell, destination = item["cells"][endpoint], parent / endpoint
                prior_cell = reusable.get((item['role'], endpoint))
                if prior_cell is not None:
                    current_pins = {pin['path']: pin['sha256'] for pin in cell['source_files']}
                    earlier_pins = {pin['path']: pin['sha256'] for pin in prior['source_files']}
                    if any(earlier_pins.get(path) != digest for path, digest in current_pins.items()):
                        raise ValueError('preview inputs differ from the current completed method')
                    destination = Path(prior_cell['evaluation_directory'])
                else:
                    command_runner([sys.executable, str(SOURCE / "scripts/evaluate_programmed_paths.py"),
                                "--paths-jsonl", str(cell["paths"]), "--labels-jsonl", str(cell["labels"]),
                                "--frozen-config", str(trial["frozen_config"]), "--official-cache", str(trial["hull_run_root"] / "official_mp_cache"),
                                "--output-dir", str(destination), "--expected-requests", str(trial["expected_requests"]),
                                "--endpoint", endpoint, "--cohort-role", trial['cohort_role'], "--policy-stage",
                                    "reference" if item["role"] == "R" else "final"], f"{item['role']}_{endpoint}", output_dir)
                summaries.append(summarize_evaluation(destination, cell, item, trial))
        for pin in source_pins:
            if file_identity(pin["path"])["sha256"] != pin["sha256"]:
                raise ValueError("a registered source changed while trial scoring was in progress")
        report = {"schema": RESULT_SCHEMA, "phase": trial["phase"], "scope": trial["scope"],
                  "expected_requests": trial["expected_requests"], "methods": summaries,
                  "coverage": coverage, "coverage_source": file_identity(coverage_path), "source_files": source_pins,
                  "scoring_source": file_identity(SOURCE / "scripts/evaluate_programmed_paths.py"),
                  "adoption": adoption_rule(trial["scope"], summaries, frozen),
                  "labels_created": False, "new_official_query": False, "GPU_calls": 0,
                  "selection_json_used": False, "cross_method_NU_pooling": False,
                  "complete_four_arm_trial": trial['scope'] == 'pilot',
                  "reused_cells": [f'{role}:{endpoint}' for role, endpoint in reusable],
                  "registered_construction_geometry": trial["registered_construction_geometry"],
                  "include_matched_interface_reference": trial["include_matched_interface_reference"],
                  "construction_checks": [{"role": item["role"], "geometry_enabled": item["construction_evidence"]["geometry_enabled"],
                                            "body_batch_size": item["construction_evidence"]["body_batch_size"],
                                            "new_construction_checked": item["construction_evidence"]["new_construction_checked"],
                                            "states": dict(Counter(row["state"] for row in item["construction_evidence"].get("snapshots", [])))}
                                           for item in components],
                  "G_P_construction_exact_match_checked": trial["registered_construction_geometry"] and {'G', 'P'}.issubset(item['role'] for item in components),
                  "completed_utc": datetime.now(timezone.utc).isoformat()}
        write_json(output_dir / "TRIAL_EVALUATION_FINAL.json", report)
        (output_dir / "summary.md").write_text(render_summary(report), encoding="utf-8")
        (output_dir / "_SUCCESS").touch()
        return report
    except BaseException as error:
        write_json(output_dir / "FAILURE.json", {"type": type(error).__name__, "message": str(error),
                                                "partial_scores_are_a_complete_trial": False})
        (output_dir / "_FAILED").touch()
        raise


def evaluate_formal(run_root):
    """Reuse seed verification/scoring, then export and score each whole arm once."""
    root = Path(run_root).resolve()
    registration_path, freeze_path = root / 'FORMAL_REGISTRATION.json', root / 'METHOD_FREEZE.json'
    registration, frozen = read_json(registration_path), read_json(freeze_path)
    count, seeds, selected = registration['requests_per_seed'], registration['planner_seeds'], frozen['selected_role']
    producer = Path(registration['producer_source'])
    if (count not in (256, 500) or len(set(seeds)) != 2 or selected != registration['selected_role']
            or producer.name != frozen['producer_commit']):
        raise ValueError('formal registration differs from the frozen candidate')
    total, formal_root = 2 * count, root / f'formal_{2 * count}'
    require_success(formal_root)
    destination = root / 'evaluation_formal'
    destination.mkdir(exist_ok=False)
    pins = [file_identity(registration_path), file_identity(freeze_path)]
    config = root / 'evaluation_pilot256' / 'TRIAL_EVALUATION_FINAL.json'
    pilot = read_json(config)
    config_pin = next(pin for pin in pilot['source_files'] if Path(pin['path']).name == 'CONFIG.json')
    pins.extend([file_identity(config), config_pin])
    common = {'schema': SCHEMA, 'expected_requests': count, 'registered_construction_geometry': True,
              'validity_artifact': 'basic_comp_struct_only', 'cohort_role': 'independent_main',
              'selected_role': selected, 'method_freeze': str(freeze_path),
              'frozen_config': config_pin['path'], 'hull_run_root': str(root / 'hull_formal')}
    try:
        seed_reports = []
        for seed in seeds:
            manifest = {**common, 'phase': f'formal_seed_{seed}',
                        'methods': [{'method_id': 'R03_' + role, 'role': role, 'planner_seed': seed,
                                     'component_dir': str(formal_root / role / f'seed_{seed}')} for role in ('R', selected)]}
            path = destination / f'INPUTS_seed_{seed}.json'
            write_json(path, manifest)
            report = evaluate_trial(path, destination / f'seed_{seed}')
            seed_reports.append(report)
            pins.extend(report['source_files'])
        cells, hull_inputs = [], []
        for role in ('R', selected):
            method = {'role': role, 'method_id': 'R03_' + role}
            for endpoint in ENDPOINTS:
                target = destination / 'pooled' / role / endpoint
                target.mkdir(parents=True, exist_ok=False)
                descriptors, label_paths, validity = [], [], Counter()
                for index, seed in enumerate(seeds):
                    component = formal_root / role / f'seed_{seed}'
                    final = read_json(component / 'COMPONENT_FINAL.json')
                    if final['sample_index_offset'] != index * count:
                        raise ValueError('formal global request indices differ from the registration')
                    descriptor = {'component_id': f'seed_{seed}', 'method_id': method['method_id'],
                                  'expected_requests': count, 'sample_idx_start': index * count, 'planner_seed': seed,
                                  'body_dir': str(component / 'body'),
                                  'planner_run_config': file_identity(component / 'planner/run_config.json')}
                    if endpoint == 'tau800':
                        descriptor.update(refined_pt=final['refined_pt'],
                                          refiner_identity=file_identity(component / 'REFINER_IDENTITY.json'))
                    descriptors.append(descriptor)
                    label_paths.append(component / (endpoint + '_labels') / 'labels.jsonl')
                    validity.update(next(row['validity'] for row in seed_reports[index]['methods']
                                         if row['role'] == role and row['endpoint'] == endpoint))
                merged = {'schema': 'r03_evaluation_components_v1', 'method_id': method['method_id'],
                          'endpoint': endpoint, 'expected_requests': total, 'components': descriptors}
                manifest_path = target / 'COMPONENTS.json'
                write_json(manifest_path, merged)
                run_command([sys.executable, str(producer / 'src/scripts/export_r03_evaluation_inputs.py'),
                             '--input-manifest', str(manifest_path), '--endpoint', endpoint,
                             '--expected-requests', str(total), '--method-id', method['method_id'],
                             '--output-dir', str(target / 'inputs')], 'export', target)
                paths = target / 'inputs/paths.jsonl'
                cell = {'endpoint': endpoint, 'paths': paths, 'rows': read_rows(paths),
                        'label_paths': label_paths, 'label_report': read_json(label_paths[0].parent / 'LABEL_FINAL.json'),
                        'validity': dict(validity), 'destination': target}
                cells.append((method, cell))
                pins.extend([file_identity(manifest_path), file_identity(paths)])
                hull_inputs.append({'cell_id': role + '_' + endpoint, 'arm': role, 'seed': seeds,
                                    'type': 'eval_paths', 'endpoint': endpoint, 'path': str(paths),
                                    'sha256': file_identity(paths)['sha256'], 'expected_requests': total})
        hull_manifest = destination / 'ACTUAL_POOLED_HULL_INPUTS.json'
        write_json(hull_manifest, {'schema': 'r03_hull_union_inputs_v1', 'purpose': 'evaluation',
                                  'phase': 'formal_pooled', 'inputs': hull_inputs})
        coverage_path = destination / 'HULL_ENDPOINT_COVERAGE.json'
        run_command([sys.executable, str(SOURCE / 'operations/r03_c3fd_main_20260907/prepare_hull_union.py'),
                     'verify-endpoints', '--run-root', str(root / 'hull_formal'),
                     '--inputs-manifest', str(hull_manifest), '--output-report', str(coverage_path)], 'coverage', destination)
        coverage = read_json(coverage_path)
        if coverage.get('coverage_accounted') is not True or coverage.get('not_covered') != 0 or coverage.get('is_subset') is not True:
            raise ValueError('formal pooled endpoint coverage is incomplete')
        summaries = []
        for method, cell in cells:
            target = cell['destination']
            run_command([sys.executable, str(SOURCE / 'scripts/evaluate_programmed_paths.py'),
                         '--paths-jsonl', str(cell['paths']), '--labels-jsonl', *map(str, cell['label_paths']),
                         '--frozen-config', common['frozen_config'], '--official-cache', str(root / 'hull_formal/official_mp_cache'),
                         '--output-dir', str(target / 'scores'), '--expected-requests', str(total),
                         '--endpoint', cell['endpoint'], '--cohort-role', 'independent_main',
                         '--policy-stage', 'reference' if method['role'] == 'R' else 'final'], 'score', target)
            summaries.append(summarize_evaluation(target / 'scores', cell, method, {**common, 'expected_requests': total,
                                  'frozen_config': Path(common['frozen_config']), 'hull_run_root': root / 'hull_formal'}))
        for pin in pins:
            if file_identity(pin['path'])['sha256'] != pin['sha256']:
                raise ValueError('a formal source changed during pooled evaluation')
        report = {'schema': 'r03_formal_evaluation_final_v1', 'phase': 'formal', 'scope': 'formal',
                  'expected_requests': total, 'planner_seeds': seeds, 'methods': summaries,
                  'per_seed': seed_reports, 'adoption': adoption_rule('formal', summaries, frozen),
                  'coverage': coverage, 'source_files': pins, 'per_seed_unique_counts_summed': False,
                  'new_labels': False, 'new_official_query': False,
                  'completed_utc': datetime.now(timezone.utc).isoformat()}
        write_json(destination / 'FORMAL_EVALUATION_FINAL.json', report)
        (destination / 'summary.md').write_text(render_summary(report), encoding='utf-8')
        (destination / '_SUCCESS').touch()
        return report
    except BaseException as error:
        write_json(destination / 'FAILURE.json', {'type': type(error).__name__, 'message': str(error)})
        (destination / '_FAILED').touch()
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    report = evaluate_trial(args.manifest, args.output_dir)
    print(json.dumps({"phase": report["phase"], "requests_per_method": report["expected_requests"],
                      "scored_cells": len(report["methods"]), "adoption": report["adoption"]}), flush=True)


if __name__ == "__main__":
    main()
