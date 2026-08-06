"""Immutable Gate-B/C selection for the CrysLLMGen/Wyckoff protocol."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..contracts import write_json_exclusive
from .gate import GateALock, sha256_file
from .protocol import load_protocol_v4


HANDOFF_TAUS = (0.25, 0.5, 0.75, 1.0)
GATE_B_ATTEMPTS = 256
GATE_C_ATTEMPTS = 3000
GATE_C_SEED_COUNTS = {101: 1000, 202: 1000, 303: 1000}


def _json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON evidence is not a mapping: {path}")
    return value


def _jsonl(paths: str | Path | Sequence[str | Path]) -> list[dict[str, Any]]:
    inputs: Iterable[str | Path]
    if isinstance(paths, (str, Path)):
        inputs = (paths,)
    else:
        inputs = paths
    result: list[dict[str, Any]] = []
    for raw in inputs:
        with Path(raw).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{raw}:{line_number}: row is not a mapping")
                result.append(value)
    return result


def _identity(paths: str | Path | Sequence[str | Path]) -> list[dict[str, Any]]:
    values = (paths,) if isinstance(paths, (str, Path)) else paths
    return [
        {
            "path": str(Path(value).resolve()),
            "bytes": Path(value).stat().st_size,
            "sha256": sha256_file(value),
        }
        for value in values
    ]


def _metric_value(report: Mapping[str, Any], key: str) -> float:
    metrics = report.get("metrics_unchanged_upstream", {})
    if key in metrics:
        return float(metrics[key])
    attempts = int(report["attempts"])
    count_key = {
        "comp_valid": "comp_valid_count",
        "struct_valid": "struct_valid_count",
        "valid": "valid_count",
    }.get(key)
    if count_key is None:
        raise ValueError(f"CrysLLMGen metric is missing {key}")
    return 100.0 * int(report[count_key]) / attempts


def _configuration(entry: Mapping[str, Any], *, expected_attempts: int) -> dict[str, Any]:
    required = {
        "configuration_id",
        "method",
        "generation_jsonl",
        "r5c_attempt_jsonl",
        "crysllmgen_metrics_report",
    }
    if not required <= set(entry):
        raise ValueError("screening configuration lacks registered evidence paths")
    configuration_id = str(entry["configuration_id"])
    method = str(entry["method"])
    generation_paths = entry["generation_jsonl"]
    if not isinstance(generation_paths, (str, Path)):
        raise ValueError(
            f"{configuration_id}: generation shards must be merged before screening"
        )
    r5c_paths = entry["r5c_attempt_jsonl"]
    generation = _jsonl(generation_paths)
    if len(generation) != expected_attempts:
        raise ValueError(f"{configuration_id}: generation denominator changed")
    attempt_ids = [str(row.get("attempt_id", "")) for row in generation]
    pair_ids = [str(row.get("pair_id", "")) for row in generation]
    if (
        any(not value for value in attempt_ids + pair_ids)
        or len(set(attempt_ids)) != expected_attempts
        or len(set(pair_ids)) != expected_attempts
    ):
        raise ValueError(f"{configuration_id}: attempt/pair identity is invalid")
    if {row.get("schema") for row in generation} != {
        "wqcodiff_generation_attempt_v1"
    }:
        raise ValueError(f"{configuration_id}: generation schema changed")
    if {str(row.get("method")) for row in generation} != {method}:
        raise ValueError(f"{configuration_id}: method mismatch")
    if {int(row.get("training_seed", -1)) for row in generation} != {11}:
        raise ValueError(f"{configuration_id}: screening must use training seed 11")
    if any(bool(row.get("retry_or_replacement_used")) for row in generation):
        raise ValueError(f"{configuration_id}: retry/replacement evidence detected")
    terminal = {"succeeded", "failed"}
    if {str(row.get("status")) for row in generation} - terminal:
        raise ValueError(f"{configuration_id}: nonterminal generation status")

    tau = entry.get("handoff_tau")
    if method.startswith("C-WQ-"):
        if tau is None or float(tau) not in HANDOFF_TAUS:
            raise ValueError(f"{configuration_id}: WQ handoff tau is unregistered")
        if {float(row.get("handoff_tau", -1.0)) for row in generation} != {
            float(tau)
        }:
            raise ValueError(f"{configuration_id}: row handoff tau mismatch")
    elif tau is not None:
        raise ValueError(f"{configuration_id}: atom method cannot carry handoff tau")

    r5c = _jsonl(r5c_paths)
    if len(r5c) != expected_attempts:
        raise ValueError(f"{configuration_id}: R5-C denominator changed")
    by_attempt: dict[str, dict[str, Any]] = {}
    for row in r5c:
        if row.get("schema") != "crysllmgen_r5c_attempt_result_v1":
            raise ValueError(f"{configuration_id}: R5-C schema changed")
        attempt_id = str(row.get("attempt_id", ""))
        if not attempt_id or attempt_id in by_attempt:
            raise ValueError(f"{configuration_id}: duplicate R5-C attempt")
        if str(row.get("method")) != method:
            raise ValueError(f"{configuration_id}: R5-C method mismatch")
        by_attempt[attempt_id] = row
    if set(by_attempt) != set(attempt_ids):
        raise ValueError(f"{configuration_id}: generation/R5-C identities differ")

    metric_path = Path(str(entry["crysllmgen_metrics_report"])).resolve()
    metric_report = _json(metric_path)
    if (
        metric_report.get("schema") != "crysllmgen_generation_metrics_report_v1"
        or not metric_report.get("ok")
        or str(metric_report.get("method")) != method
        or int(metric_report.get("attempts", -1)) != expected_attempts
        or metric_report.get("denominator") != "all_generation_attempts"
        or metric_report.get("generation_jsonl_sha256")
        != sha256_file(generation_paths)
        or bool(metric_report.get("retry_or_replacement_used"))
    ):
        raise ValueError(f"{configuration_id}: invalid CrysLLMGen metric evidence")

    pairs: dict[str, dict[str, Any]] = {}
    successes = 0
    total_calls = 0.0
    total_flops = 0.0
    sun = novel_unique = 0
    seed_counts: dict[int, int] = {}
    for generated in generation:
        pair_id = str(generated["pair_id"])
        evaluated = by_attempt[str(generated["attempt_id"])]
        metrics = evaluated.get("metrics") or {}
        success = generated.get("status") == "succeeded"
        successes += int(success)
        calls = sum(float(value) for value in (generated.get("calls") or {}).values())
        flops = float(generated.get("generation_flops_lower_bound") or 0.0)
        total_calls += calls
        total_flops += flops
        is_sun = bool(metrics.get("sun_0p1", False))
        is_novel_unique = bool(metrics.get("novel_unique", False))
        sun += int(is_sun)
        novel_unique += int(is_novel_unique)
        sampling_seed = int(generated["sampling_seed"])
        seed_counts[sampling_seed] = seed_counts.get(sampling_seed, 0) + 1
        pairs[pair_id] = {
            "attempt_id": str(generated["attempt_id"]),
            "sampling_seed": sampling_seed,
            "generation_succeeded": success,
            "sun_0p1": is_sun,
            "novel_unique": is_novel_unique,
            "calls": calls,
            "flops": flops,
            "revision_total": int(generated.get("revision_total") or 0),
        }
    denominator = len(generation)
    return {
        "configuration_id": configuration_id,
        "method": method,
        "handoff_tau": None if tau is None else float(tau),
        "attempts": denominator,
        "success_rate": successes / denominator,
        "sun_0p1": sun / denominator,
        "novel_unique": novel_unique / denominator,
        "mean_calls": total_calls / denominator,
        "mean_flops": total_flops / denominator,
        "sampling_seed_counts": seed_counts,
        "crysllmgen": {
            key: _metric_value(metric_report, key)
            for key in (
                "comp_valid",
                "struct_valid",
                "valid",
                "cov_recall",
                "cov_precision",
                "wdist_density",
                "wdist_num_elems",
            )
        },
        "pairs": pairs,
        "artifacts": {
            "generation": _identity(generation_paths),
            "r5c": _identity(r5c_paths),
            "crysllmgen_metrics": {
                "path": str(metric_path),
                "sha256": sha256_file(metric_path),
            },
        },
    }


def _public_configuration(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != "pairs"}


def _ranking(value: Mapping[str, Any]) -> tuple[float, float, float, float, float]:
    return (
        -float(value["sun_0p1"]),
        -float(value["novel_unique"]),
        -float(value["success_rate"]),
        float(value["mean_flops"] or math.inf),
        -float(value["handoff_tau"] or 0.0),
    )


def freeze_gate_b(
    *,
    protocol_path: str | Path,
    gate_a_lock_path: str | Path,
    evidence_manifest_path: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    """Freeze the matched atom baseline and best WQ handoff tau after smoke."""

    protocol = load_protocol_v4(protocol_path)
    project_root = Path(protocol_path).resolve().parents[3]
    gate = GateALock.load(
        gate_a_lock_path,
        project_root=project_root,
        protocol_path=protocol_path,
    )
    manifest_path = Path(evidence_manifest_path).resolve()
    manifest = _json(manifest_path)
    if manifest.get("schema") != "crysllmgen_gate_b_evidence_manifest_v1":
        raise ValueError("invalid Gate-B evidence manifest")
    configurations = [
        _configuration(value, expected_attempts=GATE_B_ATTEMPTS)
        for value in manifest.get("configurations", ())
    ]
    by_id = {value["configuration_id"]: value for value in configurations}
    expected = {
        "C-ATOM-OFFICIAL",
        "C-ATOM-MATCHED",
        *(f"C-WQ-HANDOFF-tau-{str(value).replace('.', 'p')}" for value in HANDOFF_TAUS),
    }
    if set(by_id) != expected or len(by_id) != len(configurations):
        raise ValueError("Gate-B requires official/matched atom and four handoff taus")

    nll_reports: dict[str, dict[str, Any]] = {}
    for role, raw in dict(manifest.get("nll_reports") or {}).items():
        path = Path(str(raw)).resolve()
        report = _json(path)
        if (
            report.get("schema") != "crysllmgen_lora_validation_nll_v1"
            or not report.get("ok")
            or float(report.get("nll_improvement", 0.0)) <= 0.0
            or bool(report.get("retry_or_replacement_used"))
        ):
            raise ValueError(f"Gate-B NLL failed for {role}")
        nll_reports[str(role)] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "nll_improvement": float(report["nll_improvement"]),
        }
    if set(nll_reports) != {"atom_coarse", "wq_coarse", "wq_mixed_edit"}:
        raise ValueError("Gate-B requires the three registered NLL comparisons")

    handoffs = [value for value in configurations if value["method"] == "C-WQ-HANDOFF"]
    selected = min(handoffs, key=_ranking)
    atom = by_id["C-ATOM-MATCHED"]
    official = by_id["C-ATOM-OFFICIAL"]
    official_reference = {
        "comp_valid": 89.2,
        "struct_valid": 99.9,
        "cov_recall": 94.1079,
        "cov_precision": 99.5,
    }
    checks = {
        "all_nll_better_than_backbone": True,
        "official_comp_valid_within_15pp": abs(
            official["crysllmgen"]["comp_valid"] - official_reference["comp_valid"]
        ) <= 15.0,
        "official_struct_valid_at_least_95pct": official["crysllmgen"]["struct_valid"] >= 95.0,
        "official_cov_recall_at_least_80pct": official["crysllmgen"]["cov_recall"] >= 80.0,
        "official_cov_precision_at_least_90pct": official["crysllmgen"]["cov_precision"] >= 90.0,
        "selected_wq_roundtrip_at_least_99pct": selected["success_rate"] >= 0.99,
        "selected_wq_not_worse_than_atom_topology_validity": selected["success_rate"]
        >= atom["success_rate"],
        "selected_wq_cov_recall_at_least_80pct": selected["crysllmgen"]["cov_recall"] >= 80.0,
        "selected_wq_cov_precision_at_least_90pct": selected["crysllmgen"]["cov_precision"] >= 90.0,
        "selected_wq_valid_at_least_70pct": selected["crysllmgen"]["valid"] >= 70.0,
        "selected_wq_density_wdist_at_most_1p5": selected["crysllmgen"]["wdist_density"] <= 1.5,
        "selected_wq_element_count_wdist_at_most_0p5": selected["crysllmgen"]["wdist_num_elems"] <= 0.5,
    }
    result = {
        "schema": "crysllmgen_gate_b_lock_v1",
        "ok": all(checks.values()),
        "checks": checks,
        "selected_handoff_configuration": selected["configuration_id"],
        "selected_handoff_tau": selected["handoff_tau"],
        "matched_atom_baseline": _public_configuration(atom),
        "wq_handoff_champion": _public_configuration(selected),
        "official_reproduction": _public_configuration(official),
        "official_historical_reference": official_reference,
        "all_configurations": [_public_configuration(value) for value in configurations],
        "nll_reports": nll_reports,
        "protocol_sha256": protocol.sha256,
        "gate_a_lock_sha256": gate.sha256,
        "source_bundle_sha256": gate.source_bundle_sha256,
        "evidence_manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
        },
        "retry_or_replacement_used": False,
    }
    write_json_exclusive(output, result)
    return result


def _paired_effect(
    candidate: Mapping[str, Any], baseline: Mapping[str, Any]
) -> dict[str, Any]:
    candidate_pairs = candidate["pairs"]
    baseline_pairs = baseline["pairs"]
    if set(candidate_pairs) != set(baseline_pairs):
        raise ValueError("Gate-C configurations do not share identical pair IDs")
    by_seed: dict[int, dict[str, int]] = {}
    improved = worsened = 0
    for pair_id in sorted(candidate_pairs):
        left = candidate_pairs[pair_id]
        right = baseline_pairs[pair_id]
        if int(left["sampling_seed"]) != int(right["sampling_seed"]):
            raise ValueError("paired Gate-C rows disagree on sampling seed")
        seed = int(left["sampling_seed"])
        cell = by_seed.setdefault(seed, {"candidate_sun": 0, "baseline_sun": 0, "attempts": 0})
        cell["candidate_sun"] += int(left["sun_0p1"])
        cell["baseline_sun"] += int(right["sun_0p1"])
        cell["attempts"] += 1
        improved += int(left["sun_0p1"] and not right["sun_0p1"])
        worsened += int(right["sun_0p1"] and not left["sun_0p1"])
    seed_gain_pp = {
        str(seed): 100.0 * (value["candidate_sun"] - value["baseline_sun"]) / value["attempts"]
        for seed, value in sorted(by_seed.items())
    }
    changed = improved + worsened
    return {
        "gain_pp": 100.0 * (candidate["sun_0p1"] - baseline["sun_0p1"]),
        "novel_unique_drop_pp": 100.0 * (baseline["novel_unique"] - candidate["novel_unique"]),
        "sampling_seed_gain_pp": seed_gain_pp,
        "wrong_to_right_pairs": improved,
        "right_to_wrong_pairs": worsened,
        "outcome_revision_precision": improved / changed if changed else 0.0,
    }


def freeze_gate_c(
    *,
    protocol_path: str | Path,
    gate_a_lock_path: str | Path,
    gate_b_lock_path: str | Path,
    evidence_manifest_path: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    """Apply the preregistered 3x1000 causal revision promotion gate."""

    protocol = load_protocol_v4(protocol_path)
    project_root = Path(protocol_path).resolve().parents[3]
    gate_a = GateALock.load(
        gate_a_lock_path,
        project_root=project_root,
        protocol_path=protocol_path,
    )
    gate_b_path = Path(gate_b_lock_path).resolve()
    gate_b = _json(gate_b_path)
    if (
        gate_b.get("schema") != "crysllmgen_gate_b_lock_v1"
        or not gate_b.get("ok")
        or gate_b.get("gate_a_lock_sha256") != gate_a.sha256
        or gate_b.get("source_bundle_sha256") != gate_a.source_bundle_sha256
    ):
        raise ValueError("Gate-C requires a passing, source-matched Gate-B lock")
    tau = float(gate_b["selected_handoff_tau"])
    manifest_path = Path(evidence_manifest_path).resolve()
    manifest = _json(manifest_path)
    if manifest.get("schema") != "crysllmgen_gate_c_evidence_manifest_v1":
        raise ValueError("invalid Gate-C evidence manifest")
    configurations = [
        _configuration(value, expected_attempts=GATE_C_ATTEMPTS)
        for value in manifest.get("configurations", ())
    ]
    by_method = {value["method"]: value for value in configurations}
    expected_methods = {
        "C-WQ-HANDOFF",
        "C-WQ-CONFEDIT",
        "C-WQ-GEOREV",
        "C-WQ-BIRTH-DEATH-ONLY",
        "C-WQ-RANDOM-MATCHED-COUNT",
        "C-WQ-SHUFFLED-GEOMETRY",
        "C-WQ-EXTRA-CALL-IGNORED",
    }
    if set(by_method) != expected_methods or len(by_method) != len(configurations):
        raise ValueError("Gate-C configuration inventory changed")
    for value in configurations:
        if value["handoff_tau"] != tau:
            raise ValueError("Gate-C uses a handoff tau different from Gate-B")
        if value["sampling_seed_counts"] != GATE_C_SEED_COUNTS:
            raise ValueError("Gate-C requires exactly 1000 attempts for each sampling seed")

    baseline = by_method["C-WQ-HANDOFF"]
    final = by_method["C-WQ-GEOREV"]
    effect = _paired_effect(final, baseline)
    alternatives = {
        method: _paired_effect(value, baseline)
        for method, value in by_method.items()
        if method not in {"C-WQ-HANDOFF", "C-WQ-GEOREV"}
    }
    compute_ratio = (
        final["mean_flops"] / baseline["mean_flops"]
        if baseline["mean_flops"] > 0.0
        else final["mean_calls"] / baseline["mean_calls"]
    )
    controls_fail_to_reproduce = all(
        by_method[method]["sun_0p1"] < final["sun_0p1"]
        and value["gain_pp"] < effect["gain_pp"]
        for method, value in alternatives.items()
    )
    checks = {
        "matter_sim_sun_gain_at_least_2pp": effect["gain_pp"] >= 2.0,
        "all_sampling_seed_directions_positive": all(
            value > 0.0 for value in effect["sampling_seed_gain_pp"].values()
        ),
        "novel_unique_drop_at_most_2pp": effect["novel_unique_drop_pp"] <= 2.0,
        "wrong_to_right_exceeds_right_to_wrong": effect["wrong_to_right_pairs"]
        > effect["right_to_wrong_pairs"],
        "outcome_revision_precision_above_half": effect["outcome_revision_precision"] > 0.5,
        "causal_controls_cannot_reproduce_gain": controls_fail_to_reproduce,
        "compute_ratio_at_most_2": compute_ratio <= 2.0,
    }
    result = {
        "schema": "crysllmgen_gate_c_lock_v1",
        "ok": all(checks.values()),
        "checks": checks,
        "selected_handoff_tau": tau,
        "baseline": _public_configuration(baseline),
        "final_candidate": _public_configuration(final),
        "paired_effect": effect,
        "alternative_effects": alternatives,
        "compute_ratio": compute_ratio,
        "frozen_final_method": "C-WQ-GEOREV" if all(checks.values()) else "C-WQ-HANDOFF",
        "configurations": [_public_configuration(value) for value in configurations],
        "protocol_sha256": protocol.sha256,
        "gate_a_lock_sha256": gate_a.sha256,
        "gate_b_lock": {"path": str(gate_b_path), "sha256": sha256_file(gate_b_path)},
        "source_bundle_sha256": gate_a.source_bundle_sha256,
        "evidence_manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
        "retry_or_replacement_used": False,
    }
    write_json_exclusive(output, result)
    return result
