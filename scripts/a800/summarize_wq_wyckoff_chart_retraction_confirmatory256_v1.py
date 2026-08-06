#!/usr/bin/env python3
"""Create the immutable WTB-256 paired result and promotion decision.

The summary consumes already frozen R/U/T generation, CrysLLMGen direct
metrics, and exact R5-C A100-on-A800 S.U.N. outputs.  It performs no model
inference, MLIP evaluation, API access, training, retry, replacement, or
outcome-dependent filtering.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crystal_dlm.wqcodiff.contracts import write_json_exclusive  # noqa: E402
from crystal_dlm.wqcodiff.crysllmgen.gate import sha256_file  # noqa: E402
from crystal_dlm.wqcodiff.crysllmgen.wtb_confirmatory import (  # noqa: E402
    ARM_METHODS,
    ATTEMPTS,
    BOOTSTRAP_SEED,
    END_ORDINAL_INCLUSIVE,
    IDENTITY,
    SAMPLING_SEED,
    START_ORDINAL,
    TRAINING_SEED,
    build_confirmatory_cells,
    paired_binary_effect,
    paired_numeric_effect,
)


CONTRACT_SCHEMA = "wq_wyckoff_chart_retraction_confirmatory256_contract_v1"
ARMS_REPORT_SCHEMA = "wq_wyckoff_chart_retraction_arms_report_v1"
GENERATION_SCHEMA = "wqcodiff_generation_attempt_v1"
MECHANICS_SCHEMA = "wq_wyckoff_chart_retraction_arm_mechanics_v1"
DIRECT_REPORT_SCHEMA = "crysllmgen_generation_metrics_report_v1"
DIRECT_ATTEMPT_SCHEMA = "crysllmgen_metric_attempt_v1"
SUN_SUMMARY_SCHEMA = "crysllmgen_r5c_a100_sun_summary_v1"
SUN_ATTEMPT_SCHEMA = "crysllmgen_r5c_a100_sun_attempt_v1"
SUMMARY_SCHEMA = "wq_wyckoff_chart_retraction_confirmatory256_summary_v1"
LOCK_SCHEMA = "wq_wyckoff_chart_retraction_confirmatory256_promotion_lock_v1"


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not one JSON object")
    return payload


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            rows.append(row)
    return rows


def _identity(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _load_contract(path: Path) -> tuple[dict[str, Any], str]:
    payload = _json(path)
    if payload.get("schema") != CONTRACT_SCHEMA or payload.get("identity") != IDENTITY:
        raise ValueError("unexpected WTB-256 contract identity")
    if payload.get("status") != "local_built_remote_execution_not_authorized":
        raise ValueError("WTB-256 contract status changed")
    return payload, sha256_file(path)


def _load_generation(
    path: Path,
    *,
    arm: str,
    contract_sha256: str,
    execution_patch_sha256: str,
) -> list[dict[str, Any]]:
    rows = _jsonl(path)
    cells = build_confirmatory_cells()
    if len(rows) != ATTEMPTS:
        raise ValueError(f"{arm}: generation denominator changed")
    for row, cell in zip(rows, cells, strict=True):
        if (
            row.get("schema") != GENERATION_SCHEMA
            or row.get("identity") != IDENTITY
            or row.get("arm") != arm
            or row.get("method") != ARM_METHODS[arm]
            or row.get("attempt_id") != cell.arm_attempt_ids[arm]
            or row.get("source_attempt_id") != cell.source_attempt_id
            or row.get("pair_id") != cell.pair_id
            or int(row.get("ordinal", -1)) != cell.ordinal
            or int(row.get("forward_noise_seed", -1))
            != cell.forward_noise_seed
            or int(row.get("reverse_noise_seed", -1))
            != cell.reverse_noise_seed
            or int(row.get("training_seed", -1)) != TRAINING_SEED
            or int(row.get("sampling_seed", -1)) != SAMPLING_SEED
            or row.get("contract_sha256") != contract_sha256
            or row.get("execution_patch_sha256") != execution_patch_sha256
            or row.get("retry_or_replacement_used") is not False
            or row.get("best_of_or_rerank_used") is not False
            or row.get("status") not in {"succeeded", "failed"}
        ):
            raise ValueError(f"{arm}: generation identity changed at {cell.ordinal}")
        descriptor_names = (
            "minimum_pair_distance_angstrom",
            "volume_per_atom_angstrom3",
            "density_g_cm3",
        )
        if row["status"] == "succeeded":
            descriptors = [row.get(name) for name in descriptor_names]
            if (
                any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    for value in descriptors
                )
                or not isinstance(
                    row.get("collision_free_at_0p5_angstrom"),
                    bool,
                )
                or row["collision_free_at_0p5_angstrom"]
                is not (
                    float(row["minimum_pair_distance_angstrom"]) >= 0.5
                    and float(row.get("volume", 0.0)) >= 0.1
                )
            ):
                raise ValueError(
                    f"{arm}: invalid geometry descriptors at {cell.ordinal}"
                )
        elif (
            any(row.get(name) is not None for name in descriptor_names)
            or row.get("collision_free_at_0p5_angstrom") is not False
        ):
            raise ValueError(
                f"{arm}: failed generation has geometry evidence at {cell.ordinal}"
            )
    return rows


def _load_direct(
    directory: Path,
    *,
    generation_path: Path,
    generation_rows: list[dict[str, Any]],
    method: str,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    report_path = directory / "report.json"
    attempts_path = directory / "attempt_metrics.jsonl"
    report = _json(report_path)
    rows = _jsonl(attempts_path)
    if (
        report.get("schema") != DIRECT_REPORT_SCHEMA
        or report.get("ok") is not True
        or report.get("method") != method
        or int(report.get("attempts", -1)) != ATTEMPTS
        or report.get("denominator") != "all_generation_attempts"
        or report.get("generation_jsonl_sha256") != sha256_file(generation_path)
        or report.get("attempt_metrics_sha256") != sha256_file(attempts_path)
        or report.get("retry_or_replacement_used") is not False
        or len(rows) != ATTEMPTS
    ):
        raise ValueError(f"{method}: invalid direct-metric evidence")
    by_attempt = {str(row.get("attempt_id", "")): row for row in rows}
    if len(by_attempt) != ATTEMPTS:
        raise ValueError(f"{method}: duplicate direct-metric attempts")
    expected = {str(row["attempt_id"]) for row in generation_rows}
    if set(by_attempt) != expected:
        raise ValueError(f"{method}: generation/direct attempt mismatch")
    for attempt_id, row in by_attempt.items():
        if (
            row.get("schema") != DIRECT_ATTEMPT_SCHEMA
            or row.get("method") != method
            or not isinstance(row.get("comp_valid"), bool)
            or not isinstance(row.get("struct_valid"), bool)
            or not isinstance(row.get("valid"), bool)
        ):
            raise ValueError(f"{method}: invalid direct attempt {attempt_id}")
    return report, by_attempt


def _load_sun(
    directory: Path,
    *,
    generation_rows: list[dict[str, Any]],
    method: str,
    execution_patch_sha256: str,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    summary_path = directory / "attempt_summary.json"
    attempts_path = directory / "attempt_results.jsonl"
    summary = _json(summary_path)
    rows = _jsonl(attempts_path)
    if (
        summary.get("schema") != SUN_SUMMARY_SCHEMA
        or summary.get("ok") is not True
        or summary.get("method") != method
        or summary.get("denominator") != "all_generation_attempts"
        or int((summary.get("counts") or {}).get("total_attempts", -1))
        != ATTEMPTS
        or summary.get("execution_patch_sha256") != execution_patch_sha256
        or summary.get("coverage_adjusted_selection_role")
        != "report_only_never_checkpoint_selection"
        or summary.get("retry_or_replacement_used") is not False
        or len(rows) != ATTEMPTS
    ):
        raise ValueError(f"{method}: invalid S.U.N. summary")
    by_attempt = {str(row.get("attempt_id", "")): row for row in rows}
    if len(by_attempt) != ATTEMPTS:
        raise ValueError(f"{method}: duplicate S.U.N. attempts")
    expected = {str(row["attempt_id"]) for row in generation_rows}
    if set(by_attempt) != expected:
        raise ValueError(f"{method}: generation/S.U.N. attempt mismatch")
    for attempt_id, row in by_attempt.items():
        metrics = row.get("metrics")
        if (
            row.get("schema") != SUN_ATTEMPT_SCHEMA
            or row.get("method") != method
            or row.get("execution_patch_sha256") != execution_patch_sha256
            or row.get("retry_or_replacement_used") is not False
            or not isinstance(metrics, Mapping)
            or any(
                not isinstance(metrics.get(key), bool)
                for key in (
                    "novel",
                    "unique_representative",
                    "novel_unique",
                    "strict_full_sun",
                    "meta_full_sun",
                )
            )
        ):
            raise ValueError(f"{method}: invalid S.U.N. attempt {attempt_id}")
    return summary, by_attempt


def _vectors(
    generation: list[dict[str, Any]],
    direct: Mapping[str, Mapping[str, Any]],
    sun: Mapping[str, Mapping[str, Any]],
) -> dict[str, list[bool]]:
    result = {
        "generation_success": [],
        "collision_free": [],
        "comp_valid": [],
        "struct_valid": [],
        "joint_valid": [],
        "novel_unique": [],
        "strict_sun": [],
        "meta_sun": [],
    }
    for row in generation:
        attempt_id = str(row["attempt_id"])
        direct_row = direct[attempt_id]
        sun_metrics = sun[attempt_id]["metrics"]
        result["generation_success"].append(row["status"] == "succeeded")
        result["collision_free"].append(
            row["status"] == "succeeded"
            and row["collision_free_at_0p5_angstrom"] is True
        )
        result["comp_valid"].append(bool(direct_row["comp_valid"]))
        result["struct_valid"].append(bool(direct_row["struct_valid"]))
        result["joint_valid"].append(bool(direct_row["valid"]))
        result["novel_unique"].append(bool(sun_metrics["novel_unique"]))
        result["strict_sun"].append(bool(sun_metrics["strict_full_sun"]))
        result["meta_sun"].append(bool(sun_metrics["meta_full_sun"]))
    return result


def _geometry_vectors(
    generation: list[dict[str, Any]],
) -> dict[str, list[float | None]]:
    names = (
        "minimum_pair_distance_angstrom",
        "volume_per_atom_angstrom3",
        "density_g_cm3",
    )
    return {
        name: [
            (
                float(row[name])
                if row["status"] == "succeeded" and row.get(name) is not None
                else None
            )
            for row in generation
        ]
        for name in names
    }


def _reason_class(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "unspecified"
    if text.startswith("upstream_generation:"):
        return "upstream_generation"
    return text.split(":", 1)[0]


def _failure_taxonomy(
    generation: list[dict[str, Any]],
    direct: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    categories: Counter[str] = Counter()
    generation_reasons: Counter[str] = Counter()
    direct_reasons: Counter[str] = Counter()
    for row in generation:
        attempt_id = str(row["attempt_id"])
        metric = direct[attempt_id]
        if row["status"] != "succeeded":
            categories["generation_failed"] += 1
            generation_reasons[_reason_class(row.get("reason"))] += 1
            continue
        comp = bool(metric["comp_valid"])
        struct = bool(metric["struct_valid"])
        valid = bool(metric["valid"])
        if valid:
            categories["joint_valid"] += 1
        elif not comp and not struct:
            categories["composition_and_structure_invalid"] += 1
        elif not comp and struct:
            categories["composition_invalid_only"] += 1
        elif comp and not struct:
            categories["structure_invalid_only"] += 1
        else:
            categories["post_validity_or_fingerprint_invalid"] += 1
        if not valid:
            direct_reasons[_reason_class(metric.get("reason"))] += 1
    if sum(categories.values()) != ATTEMPTS:
        raise ValueError("WTB-256 failure taxonomy denominator changed")
    return {
        "attempts": ATTEMPTS,
        "collision_cutoff_angstrom": 0.5,
        "terminal_categories": dict(sorted(categories.items())),
        "generation_failure_reason_classes": dict(
            sorted(generation_reasons.items())
        ),
        "direct_invalid_reason_classes": dict(sorted(direct_reasons.items())),
        "collision_free_count": sum(
            row.get("collision_free_at_0p5_angstrom") is True
            for row in generation
        ),
        "collision_or_failed_count": sum(
            row.get("collision_free_at_0p5_angstrom") is not True
            for row in generation
        ),
    }


def _rate(values: list[bool]) -> dict[str, Any]:
    count = sum(bool(value) for value in values)
    return {
        "count": count,
        "attempts": len(values),
        "rate": count / len(values),
        "percentage": 100.0 * count / len(values),
    }


def _run(args: argparse.Namespace, output: Path) -> dict[str, Any]:
    contract, contract_sha256 = _load_contract(args.contract)
    arms_report = _json(args.arms_dir / "arms_report.json")
    execution_patch_sha256 = str(arms_report.get("execution_patch_sha256", ""))
    if (
        arms_report.get("schema") != ARMS_REPORT_SCHEMA
        or arms_report.get("ok") is not True
        or arms_report.get("acceptance") != "PASS"
        or arms_report.get("contract_sha256") != contract_sha256
        or int(arms_report.get("attempts_per_arm", -1)) != ATTEMPTS
        or arms_report.get("retry_or_replacement_used") is not False
        or len(execution_patch_sha256) != 64
    ):
        raise ValueError("WTB-256 arms report is not an immutable PASS")

    generation: dict[str, list[dict[str, Any]]] = {}
    direct_reports: dict[str, dict[str, Any]] = {}
    direct_rows: dict[str, dict[str, dict[str, Any]]] = {}
    sun_summaries: dict[str, dict[str, Any]] = {}
    sun_rows: dict[str, dict[str, dict[str, Any]]] = {}
    evidence: dict[str, Any] = {
        "contract": _identity(args.contract),
        "arms_report": _identity(args.arms_dir / "arms_report.json"),
    }
    for arm in ARM_METHODS:
        generation_path = args.arms_dir / f"{arm.lower()}_generation.jsonl"
        generation[arm] = _load_generation(
            generation_path,
            arm=arm,
            contract_sha256=contract_sha256,
            execution_patch_sha256=execution_patch_sha256,
        )
        direct_directory = args.evaluation_dir / arm / "crysllmgen_metrics"
        sun_directory = args.evaluation_dir / arm / "r5c_a100_sun"
        direct_reports[arm], direct_rows[arm] = _load_direct(
            direct_directory,
            generation_path=generation_path,
            generation_rows=generation[arm],
            method=ARM_METHODS[arm],
        )
        sun_summaries[arm], sun_rows[arm] = _load_sun(
            sun_directory,
            generation_rows=generation[arm],
            method=ARM_METHODS[arm],
            execution_patch_sha256=execution_patch_sha256,
        )
        evidence[arm] = {
            "generation": _identity(generation_path),
            "direct_report": _identity(direct_directory / "report.json"),
            "direct_attempts": _identity(
                direct_directory / "attempt_metrics.jsonl"
            ),
            "sun_summary": _identity(sun_directory / "attempt_summary.json"),
            "sun_attempts": _identity(sun_directory / "attempt_results.jsonl"),
        }

    mechanics_path = args.arms_dir / "arm_mechanics.jsonl"
    mechanics = _jsonl(mechanics_path)
    if len(mechanics) != 3 * ATTEMPTS:
        raise ValueError("WTB-256 mechanics denominator changed")
    mechanics_by_attempt = {
        str(row.get("attempt_id", "")): row for row in mechanics
    }
    if len(mechanics_by_attempt) != 3 * ATTEMPTS:
        raise ValueError("WTB-256 mechanics attempts are duplicated")
    evidence["mechanics"] = _identity(mechanics_path)
    evidence["t_trajectory_evidence"] = _identity(
        args.arms_dir / "t_trajectory_evidence.jsonl"
    )

    vectors = {
        arm: _vectors(generation[arm], direct_rows[arm], sun_rows[arm])
        for arm in ARM_METHODS
    }
    geometry_vectors = {
        arm: _geometry_vectors(generation[arm])
        for arm in ARM_METHODS
    }
    failure_taxonomy = {
        arm: _failure_taxonomy(generation[arm], direct_rows[arm])
        for arm in ARM_METHODS
    }
    arm_results = {
        arm: {
            metric: _rate(values)
            for metric, values in vectors[arm].items()
        }
        for arm in ARM_METHODS
    }
    comparisons: dict[str, Any] = {}
    for left, right in (("T", "U"), ("T", "R"), ("U", "R")):
        label = f"{left}_vs_{right}"
        comparisons[label] = {
            metric: paired_binary_effect(
                vectors[left][metric],
                vectors[right][metric],
                seed=BOOTSTRAP_SEED
                + int.from_bytes(
                    hashlib.sha256(f"{label}:{metric}".encode("ascii")).digest()[:2],
                    "big",
                ),
            )
            for metric in (
                "generation_success",
                "collision_free",
                "comp_valid",
                "struct_valid",
                "joint_valid",
                "strict_sun",
                "meta_sun",
            )
        }
        # Novel+unique is a frozen evaluator point label.  Its paired point
        # effect is valid; a CI is intentionally omitted because true
        # replicate-wise uniqueness would require recomputing equivalence
        # classes inside every bootstrap sample.
        left_nu = vectors[left]["novel_unique"]
        right_nu = vectors[right]["novel_unique"]
        comparisons[label]["novel_unique"] = {
            "attempts": ATTEMPTS,
            "left_count": sum(left_nu),
            "right_count": sum(right_nu),
            "difference_percentage_points": 100.0
            * (sum(left_nu) - sum(right_nu))
            / ATTEMPTS,
            "inference": "point_estimate_only",
            "reason": (
                "replicate-wise uniqueness is not approximated from frozen "
                "representative labels"
            ),
        }

    paired_geometry: dict[str, Any] = {}
    for left, right in (("T", "U"), ("T", "R"), ("U", "R")):
        label = f"{left}_vs_{right}"
        paired_geometry[label] = {}
        for metric in geometry_vectors[left]:
            paired_values = [
                (left_value, right_value)
                for left_value, right_value in zip(
                    geometry_vectors[left][metric],
                    geometry_vectors[right][metric],
                    strict=True,
                )
                if left_value is not None and right_value is not None
            ]
            if not paired_values:
                paired_geometry[label][metric] = {
                    "registered_attempts": ATTEMPTS,
                    "common_observed_pairs": 0,
                    "coverage": 0.0,
                    "selection_role": (
                        "descriptive_common_observed_pairs_only"
                    ),
                    "bootstrap": None,
                }
                continue
            left_values = [value[0] for value in paired_values]
            right_values = [value[1] for value in paired_values]
            effect = paired_numeric_effect(
                left_values,
                right_values,
                seed=BOOTSTRAP_SEED
                + int.from_bytes(
                    hashlib.sha256(
                        f"{label}:{metric}".encode("ascii")
                    ).digest()[:2],
                    "big",
                ),
            )
            paired_geometry[label][metric] = {
                "registered_attempts": ATTEMPTS,
                "coverage": len(paired_values) / ATTEMPTS,
                **effect,
            }

    cells = build_confirmatory_cells()
    exact_pairing = True
    composition_identity = True
    no_retry = True
    for index, cell in enumerate(cells):
        rows = {arm: generation[arm][index] for arm in ARM_METHODS}
        exact_pairing = exact_pairing and all(
            row["pair_id"] == cell.pair_id
            and row["source_attempt_id"] == cell.source_attempt_id
            and row["forward_noise_seed"] == cell.forward_noise_seed
            and row["reverse_noise_seed"] == cell.reverse_noise_seed
            for row in rows.values()
        )
        composition_identity = composition_identity and (
            rows["R"].get("composition_signature")
            == rows["T"].get("composition_signature")
        )
        no_retry = no_retry and all(
            row.get("retry_or_replacement_used") is False
            and row.get("best_of_or_rerank_used") is False
            for row in rows.values()
        )

    t_topology = []
    t_projection = []
    for row in generation["T"]:
        mechanics_row = mechanics_by_attempt[str(row["attempt_id"])]
        details = mechanics_row.get("details") or {}
        mechanics_detail = details.get("mechanics") or {}
        succeeded = (
            row["status"] == "succeeded"
            and mechanics_row.get("status") == "succeeded"
        )
        t_topology.append(
            succeeded
            and details.get("topology_hash_unchanged") is True
            and mechanics_detail.get("topology_hash_unchanged") is True
        )
        t_projection.append(
            succeeded
            and int(mechanics_row.get("parent_decoder_calls", -1)) == 64
            and int(mechanics_row.get("projection_calls", -1)) == 64
            and mechanics_detail.get("lattice_projection_methods")
            == ["global_chart_retraction_v1"]
            and mechanics_detail.get(
                "all_chart_retraction_audit_values_finite"
            )
            is True
        )

    integrity_gates = {
        "registered_attempts_R_256": len(generation["R"]) == ATTEMPTS,
        "registered_attempts_U_256": len(generation["U"]) == ATTEMPTS,
        "registered_attempts_T_256": len(generation["T"]) == ATTEMPTS,
        "ordinal_range_512_767": [
            int(row["ordinal"]) for row in generation["T"]
        ]
        == list(range(START_ORDINAL, END_ORDINAL_INCLUSIVE + 1)),
        "paired_source_and_noise_identity": exact_pairing,
        "T_composition_byte_identity_to_R_source": composition_identity,
        "T_exact_topology_retention_256_of_256": all(t_topology),
        "T_every_step_global_chart_retraction_256_of_256": all(t_projection),
        "T_generation_success_not_below_U": (
            arm_results["T"]["generation_success"]["count"]
            >= arm_results["U"]["generation_success"]["count"]
        ),
        "all_attempt_denominator_256": all(
            int(direct_reports[arm]["attempts"]) == ATTEMPTS
            and int(sun_summaries[arm]["counts"]["total_attempts"]) == ATTEMPTS
            for arm in ARM_METHODS
        ),
        "no_retry_replacement_best_of_or_rerank": no_retry,
        "coverage_adjusted_not_used_for_selection": all(
            sun_summaries[arm]["coverage_adjusted_selection_role"]
            == "report_only_never_checkpoint_selection"
            for arm in ARM_METHODS
        ),
    }
    thresholds = contract["promotion_gates"]
    scientific_gates = {
        "T_joint_valid_vs_R_at_least_plus_3pp": (
            comparisons["T_vs_R"]["joint_valid"][
                "difference_percentage_points"
            ]
            >= float(thresholds["T_joint_valid_vs_R_minimum_percentage_points"])
        ),
        "T_strict_SUN_at_least_9pct": (
            arm_results["T"]["strict_sun"]["percentage"]
            >= float(thresholds["T_strict_sun_minimum_percent"])
        ),
        "T_meta_SUN_at_least_46p1pct": (
            arm_results["T"]["meta_sun"]["percentage"]
            >= float(thresholds["T_meta_sun_minimum_percent"])
        ),
        "paired_T_minus_U_joint_valid_nonnegative": (
            comparisons["T_vs_U"]["joint_valid"][
                "difference_percentage_points"
            ]
            >= 0.0
        ),
        "paired_T_minus_U_strict_SUN_nonnegative": (
            comparisons["T_vs_U"]["strict_sun"][
                "difference_percentage_points"
            ]
            >= 0.0
        ),
        "T_novel_unique_drop_vs_U_at_most_2pp": (
            comparisons["T_vs_U"]["novel_unique"][
                "difference_percentage_points"
            ]
            >= -float(thresholds["T_novel_unique_maximum_drop_percentage_points"])
        ),
    }
    integrity_pass = all(integrity_gates.values())
    scientific_pass = all(scientific_gates.values())
    positive_residual_signal = (
        comparisons["T_vs_U"]["joint_valid"]["difference_percentage_points"]
        > 0.0
        or comparisons["T_vs_U"]["strict_sun"][
            "difference_percentage_points"
        ]
        > 0.0
    )
    if not integrity_pass:
        decision = "invalid_integrity_stop_no_retry"
    elif scientific_pass:
        decision = "training_free_promotion_to_l3"
    elif positive_residual_signal:
        decision = "positive_below_gate_requires_new_adapter_contract"
    else:
        decision = "stop_tangent_bridge_no_training"

    output.mkdir(parents=True, exist_ok=False)
    summary = {
        "schema": SUMMARY_SCHEMA,
        "ok": integrity_pass,
        "acceptance": "PASS" if integrity_pass else "FAIL",
        "identity": IDENTITY,
        "contract_sha256": contract_sha256,
        "execution_patch_sha256": execution_patch_sha256,
        "panel": {
            "training_seed": TRAINING_SEED,
            "sampling_seed": SAMPLING_SEED,
            "start_ordinal": START_ORDINAL,
            "end_ordinal_inclusive": END_ORDINAL_INCLUSIVE,
            "attempts_per_arm": ATTEMPTS,
        },
        "arm_results": arm_results,
        "failure_taxonomy": failure_taxonomy,
        "paired_comparisons": comparisons,
        "paired_geometry_descriptors": paired_geometry,
        "integrity_gates": integrity_gates,
        "integrity_pass": integrity_pass,
        "scientific_gates": scientific_gates,
        "scientific_promotion_pass": scientific_pass,
        "decision": decision,
        "evidence": evidence,
        "retry_or_replacement_used": False,
        "training_performed": False,
        "selection_uses_chgnet_for_training_or_checkpointing": False,
        "coverage_adjusted_selection_role": "report_only",
        "dft_used": False,
    }
    summary_path = output / "summary.json"
    write_json_exclusive(summary_path, summary)
    lock = {
        "schema": LOCK_SCHEMA,
        "identity": IDENTITY,
        "contract_sha256": contract_sha256,
        "execution_patch_sha256": execution_patch_sha256,
        "summary_sha256": sha256_file(summary_path),
        "integrity_pass": integrity_pass,
        "scientific_promotion_pass": scientific_pass,
        "decision": decision,
        "allowed_next_action": {
            "invalid_integrity_stop_no_retry": "stop_preserve_evidence",
            "training_free_promotion_to_l3": (
                "prepare_separate_multiseed_1000_contract"
            ),
            "positive_below_gate_requires_new_adapter_contract": (
                "design_separate_mlip_free_adapter_gate_and_request_authorization"
            ),
            "stop_tangent_bridge_no_training": (
                "stop_bridge_return_to_wq_chemistry_baseline"
            ),
        }[decision],
        "automatic_training_authorized": False,
        "retry_or_replacement_allowed": False,
    }
    write_json_exclusive(output / "promotion_lock.json", lock)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--arms-dir", type=Path, required=True)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for name in ("contract", "arms_dir", "evaluation_dir", "output_dir"):
        setattr(args, name, getattr(args, name).resolve())
    result = _run(args, args.output_dir)
    print(json.dumps(result, sort_keys=True))
    if not result["integrity_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
