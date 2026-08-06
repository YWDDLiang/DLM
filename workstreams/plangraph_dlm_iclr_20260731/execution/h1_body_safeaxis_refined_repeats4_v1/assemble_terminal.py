#!/usr/bin/env python3
"""Assemble the preregistered four-repeat paired R03E result."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from protocol import (
    ARMS,
    DENOMINATOR,
    REPEATS,
    read_json,
    read_jsonl,
    require_source_manifest,
    validate_config,
    write_json_exclusive,
)


ENDPOINTS = (
    "generation_complete",
    "composition_valid",
    "structure_valid",
    "joint_valid",
    "novel",
    "unique_representative",
    "novel_unique",
    "strict_full_sun",
    "meta_full_sun",
)


def _mcnemar(control: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    control_only = int(np.sum(control & ~candidate))
    candidate_only = int(np.sum(candidate & ~control))
    discordant = control_only + candidate_only
    if discordant == 0:
        p_value = 1.0
    else:
        lower = min(control_only, candidate_only)
        tail = sum(
            math.comb(discordant, index) for index in range(lower + 1)
        ) / (2**discordant)
        p_value = min(1.0, 2.0 * tail)
    return {
        "control_only": control_only,
        "candidate_only": candidate_only,
        "discordant": discordant,
        "exact_two_sided_p": p_value,
    }


def _endpoint_vectors(
    generation: list[dict[str, Any]],
    direct: list[dict[str, Any]],
    sun: list[dict[str, Any]],
) -> dict[str, np.ndarray]:
    if not (
        len(generation) == len(direct) == len(sun) == DENOMINATOR
    ):
        raise ValueError("attempt vectors changed denominator")
    return {
        "generation_complete": np.asarray(
            [row.get("status") == "succeeded" for row in generation], dtype=bool
        ),
        "composition_valid": np.asarray(
            [bool(row.get("comp_valid")) for row in direct], dtype=bool
        ),
        "structure_valid": np.asarray(
            [bool(row.get("struct_valid")) for row in direct], dtype=bool
        ),
        "joint_valid": np.asarray(
            [bool(row.get("valid")) for row in direct], dtype=bool
        ),
        "novel": np.asarray(
            [bool((row.get("metrics") or {}).get("novel")) for row in sun],
            dtype=bool,
        ),
        "unique_representative": np.asarray(
            [
                bool((row.get("metrics") or {}).get("unique_representative"))
                for row in sun
            ],
            dtype=bool,
        ),
        "novel_unique": np.asarray(
            [bool((row.get("metrics") or {}).get("novel_unique")) for row in sun],
            dtype=bool,
        ),
        "strict_full_sun": np.asarray(
            [
                bool((row.get("metrics") or {}).get("strict_full_sun"))
                for row in sun
            ],
            dtype=bool,
        ),
        "meta_full_sun": np.asarray(
            [
                bool((row.get("metrics") or {}).get("meta_full_sun"))
                for row in sun
            ],
            dtype=bool,
        ),
    }


def _hierarchical_bootstrap(
    differences: np.ndarray,
    *,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    if differences.shape != (len(REPEATS), DENOMINATOR, len(ENDPOINTS)):
        raise ValueError("hierarchical bootstrap matrix shape changed")
    rng = np.random.default_rng(seed)
    samples = np.empty((replicates, len(ENDPOINTS)), dtype=np.float64)
    batch_size = 500
    for start in range(0, replicates, batch_size):
        stop = min(replicates, start + batch_size)
        size = stop - start
        repeat_draw = rng.integers(
            0, len(REPEATS), size=(size, len(REPEATS))
        )
        ordinal_draw = rng.integers(
            0,
            DENOMINATOR,
            size=(size, len(REPEATS), DENOMINATOR),
        )
        values = differences[
            repeat_draw[:, :, None],
            ordinal_draw,
            :,
        ]
        samples[start:stop] = values.mean(axis=(1, 2))
    lower, upper = np.quantile(samples, [0.025, 0.975], axis=0)
    return {
        endpoint: {
            "mean_delta": float(differences[:, :, index].mean()),
            "hierarchical_paired_bootstrap_95ci": [
                float(lower[index]),
                float(upper[index]),
            ],
        }
        for index, endpoint in enumerate(ENDPOINTS)
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    args = parser.parse_args()

    source = args.source_dir.resolve()
    require_source_manifest(source, args.source_manifest_sha256)
    config = read_json(args.config.resolve())
    validate_config(config)
    run_root = args.run_root.resolve()

    vectors: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    repeat_reports: list[dict[str, Any]] = []
    all_body_success_refined = True
    candidate_refiner_failure_classes: set[str] = set()
    control_refiner_failure_classes: set[str] = set()
    for repeat in REPEATS:
        repeat_root = run_root / f"repeats/{repeat}"
        preflight = read_json(repeat_root / "preflight_report.json")
        if (
            preflight.get("status") != "pass"
            or int(preflight.get("repeat", -1)) != repeat
            or preflight.get("new_scientific_seed_per_repeat") is not False
            or preflight.get("arm_order")
            != config["protocol"]["arm_order"][repeat]
        ):
            raise ValueError(f"repeat {repeat} preflight contract changed")
        vectors[repeat] = {}
        arm_report: dict[str, Any] = {}
        for arm in ARMS:
            arm_root = repeat_root / "arms" / arm
            if not (arm_root / "evaluation/_SUCCESS").is_file():
                raise FileNotFoundError(arm_root / "evaluation/_SUCCESS")
            evaluation = read_json(arm_root / "evaluation/evaluation_report.json")
            refinement = read_json(arm_root / "refinement/refinement_metrics.json")
            generation = read_jsonl(arm_root / "generation/generation.jsonl")
            direct = read_jsonl(
                arm_root
                / "evaluation/crysllmgen_metrics/attempt_metrics.jsonl"
            )
            sun = read_jsonl(
                arm_root / "evaluation/r5c_a100_sun/attempt_results.jsonl"
            )
            if (
                evaluation.get("ok") is not True
                or int(evaluation.get("repeat", -1)) != repeat
                or evaluation.get("arm") != arm
                or int(evaluation.get("attempts", -1)) != DENOMINATOR
                or int(refinement.get("all_attempt_denominator", -1))
                != DENOMINATOR
                or int(refinement.get("body_complete", -1))
                != int(config["arms"][arm]["expected_body_success"])
            ):
                raise ValueError(f"repeat {repeat} {arm} report changed")
            body_complete = int(refinement["body_complete"])
            refiner_complete = int(refinement["refiner_complete"])
            all_body_success_refined &= body_complete == refiner_complete
            refiner_classes = {
                str(key)
                for key, count in (refinement.get("failures") or {}).items()
                if int(count) > 0 and str(key).startswith("refiner:")
            }
            if arm == "control":
                control_refiner_failure_classes.update(refiner_classes)
            else:
                candidate_refiner_failure_classes.update(refiner_classes)
            vectors[repeat][arm] = _endpoint_vectors(generation, direct, sun)
            arm_report[arm] = {
                "body_complete": body_complete,
                "refiner_complete": refiner_complete,
                "counts": {
                    endpoint: int(vectors[repeat][arm][endpoint].sum())
                    for endpoint in ENDPOINTS
                },
                "rates": {
                    endpoint: float(vectors[repeat][arm][endpoint].mean())
                    for endpoint in ENDPOINTS
                },
                "refiner_failures": refinement.get("failures") or {},
            }

        endpoint_effects: dict[str, Any] = {}
        for endpoint in ENDPOINTS:
            control = vectors[repeat]["control"][endpoint]
            candidate = vectors[repeat]["candidate"][endpoint]
            endpoint_effects[endpoint] = {
                "delta_count": int(candidate.sum() - control.sum()),
                "delta_rate": float(candidate.mean() - control.mean()),
                "mcnemar": _mcnemar(control, candidate),
            }
        repeat_reports.append(
            {
                "repeat": repeat,
                "arm_order": config["protocol"]["arm_order"][repeat],
                "arms": arm_report,
                "candidate_minus_control": endpoint_effects,
            }
        )

    differences = np.empty(
        (len(REPEATS), DENOMINATOR, len(ENDPOINTS)),
        dtype=np.float64,
    )
    for repeat_index, repeat in enumerate(REPEATS):
        for endpoint_index, endpoint in enumerate(ENDPOINTS):
            differences[repeat_index, :, endpoint_index] = (
                vectors[repeat]["candidate"][endpoint].astype(np.float64)
                - vectors[repeat]["control"][endpoint].astype(np.float64)
            )
    bootstrap = _hierarchical_bootstrap(
        differences,
        seed=int(config["analysis"]["bootstrap_seed"]),
        replicates=int(config["analysis"]["bootstrap_replicates"]),
    )

    pooled: dict[str, Any] = {}
    sign_stability: dict[str, Any] = {}
    for endpoint_index, endpoint in enumerate(ENDPOINTS):
        control = np.concatenate(
            [vectors[repeat]["control"][endpoint] for repeat in REPEATS]
        )
        candidate = np.concatenate(
            [vectors[repeat]["candidate"][endpoint] for repeat in REPEATS]
        )
        repeat_deltas = [
            float(differences[index, :, endpoint_index].mean())
            for index in range(len(REPEATS))
        ]
        pooled[endpoint] = {
            "control_count": int(control.sum()),
            "candidate_count": int(candidate.sum()),
            "descriptive_denominator_per_arm": len(control),
            "candidate_minus_control_count": int(
                candidate.sum() - control.sum()
            ),
            "candidate_minus_control_rate": float(
                candidate.mean() - control.mean()
            ),
            "descriptive_pooled_mcnemar": _mcnemar(control, candidate),
            **bootstrap[endpoint],
        }
        sign_stability[endpoint] = {
            "repeat_deltas": repeat_deltas,
            "positive_repeats": sum(value > 0.0 for value in repeat_deltas),
            "nonnegative_repeats": sum(value >= 0.0 for value in repeat_deltas),
            "negative_repeats": sum(value < 0.0 for value in repeat_deltas),
        }

    analysis = config["analysis"]
    new_candidate_refiner_classes = sorted(
        candidate_refiner_failure_classes - control_refiner_failure_classes
    )
    gate_checks = {
        "all_four_repeats_complete": len(repeat_reports) == 4,
        "all_body_successes_refined": all_body_success_refined,
        "joint_positive_repeats": (
            sign_stability["joint_valid"]["positive_repeats"]
            >= int(analysis["joint_positive_repeats_min"])
        ),
        "joint_mean_positive": pooled["joint_valid"]["mean_delta"] > 0.0,
        "meta_nonnegative_repeats": (
            sign_stability["meta_full_sun"]["nonnegative_repeats"]
            >= int(analysis["meta_nonnegative_repeats_min"])
        ),
        "meta_mean_positive": pooled["meta_full_sun"]["mean_delta"] > 0.0,
        "strict_mean_not_materially_worse": (
            pooled["strict_full_sun"]["mean_delta"]
            >= float(analysis["strict_mean_delta_min"])
        ),
        "structure_mean_not_materially_worse": (
            pooled["structure_valid"]["mean_delta"]
            >= float(analysis["structure_mean_delta_min"])
        ),
        "no_new_post_body_failure_class": not new_candidate_refiner_classes,
    }
    passed = all(gate_checks.values())
    report = {
        "schema": "h1_r03e_refined_repeats4_terminal_report_v1",
        "status": "complete",
        "decision": (
            "safe_axis_refined_signal_passed"
            if passed
            else "safe_axis_refined_signal_stopped"
        ),
        "safe_axis_refined_signal_passed": passed,
        "repeat_count": 4,
        "attempts_per_arm_per_repeat": DENOMINATOR,
        "pooled_attempts_per_arm_descriptive": 4 * DENOMINATOR,
        "repeat_seed_role": "same_frozen_h1_seed_ledger_process_realizations",
        "new_scientific_seed_per_repeat": False,
        "repeat_reports": repeat_reports,
        "pooled": pooled,
        "sign_stability": sign_stability,
        "bootstrap": {
            "method": "hierarchical_paired_repeat_block_and_ordinal",
            "seed": int(analysis["bootstrap_seed"]),
            "replicates": int(analysis["bootstrap_replicates"]),
            "pooled_1024_independence_assumed": False,
        },
        "gate_checks": gate_checks,
        "new_candidate_post_body_failure_classes": new_candidate_refiner_classes,
        "formal_g3": False,
        "automatic_promotion": False,
        "automatic_training": False,
        "automatic_downstream": False,
        "source_manifest_sha256": args.source_manifest_sha256,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    output = run_root / "terminal_report.json"
    write_json_exclusive(output, report)
    with (run_root / "_SUCCESS").open("x", encoding="ascii") as handle:
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
