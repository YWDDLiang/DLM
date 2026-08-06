#!/usr/bin/env python3
"""Assemble frozen direct metrics plus repaired S.U.N. without model selection."""

from __future__ import annotations

import argparse
import json
import math
import random
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

from protocol import (
    ARM_ORDER,
    DENOMINATOR,
    read_json,
    read_jsonl,
    require_source_manifest,
    sha256_file,
    verify_frozen_arm,
    write_json_exclusive,
)


BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20260802


def quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * float(probability)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def mcnemar(candidate: Sequence[bool], baseline: Sequence[bool]) -> dict[str, Any]:
    candidate_only = sum(
        bool(left) and not bool(right)
        for left, right in zip(candidate, baseline)
    )
    baseline_only = sum(
        not bool(left) and bool(right)
        for left, right in zip(candidate, baseline)
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
    candidate_arm: str,
    baseline_arm: str,
    seed_offset: int,
) -> dict[str, Any]:
    if len(candidate) != DENOMINATOR or len(baseline) != DENOMINATOR:
        raise ValueError("paired effect lost the all-attempt denominator")
    differences = [
        float(bool(left)) - float(bool(right))
        for left, right in zip(candidate, baseline)
    ]
    rng = random.Random(BOOTSTRAP_SEED + seed_offset)
    draws = [
        100.0
        * sum(differences[rng.randrange(DENOMINATOR)] for _ in range(DENOMINATOR))
        / DENOMINATOR
        for _ in range(BOOTSTRAP_DRAWS)
    ]
    candidate_count = sum(bool(value) for value in candidate)
    baseline_count = sum(bool(value) for value in baseline)
    return {
        "candidate_arm": candidate_arm,
        "baseline_arm": baseline_arm,
        "attempts": DENOMINATOR,
        "candidate_count": candidate_count,
        "baseline_count": baseline_count,
        "candidate_rate": candidate_count / DENOMINATOR,
        "baseline_rate": baseline_count / DENOMINATOR,
        "difference_percentage_points": (
            100.0 * (candidate_count - baseline_count) / DENOMINATOR
        ),
        "bootstrap": {
            "draws": BOOTSTRAP_DRAWS,
            "seed": BOOTSTRAP_SEED + seed_offset,
            "ci95_lower_percentage_points": quantile(draws, 0.025),
            "ci95_upper_percentage_points": quantile(draws, 0.975),
        },
        "exact_mcnemar": mcnemar(candidate, baseline),
    }


def interaction(
    vectors: Mapping[str, Sequence[bool]], *, seed_offset: int
) -> dict[str, Any]:
    per_ordinal = [
        100.0
        * (
            float(vectors["M11"][index])
            - float(vectors["M10"][index])
            - float(vectors["M01"][index])
            + float(vectors["M00"][index])
        )
        for index in range(DENOMINATOR)
    ]
    rng = random.Random(BOOTSTRAP_SEED + seed_offset)
    draws = [
        sum(
            per_ordinal[rng.randrange(DENOMINATOR)]
            for _ in range(DENOMINATOR)
        )
        / DENOMINATOR
        for _ in range(BOOTSTRAP_DRAWS)
    ]
    return {
        "definition": "M11-M10-M01+M00",
        "interaction_percentage_points": sum(per_ordinal) / DENOMINATOR,
        "bootstrap": {
            "draws": BOOTSTRAP_DRAWS,
            "seed": BOOTSTRAP_SEED + seed_offset,
            "ci95_lower_percentage_points": quantile(draws, 0.025),
            "ci95_upper_percentage_points": quantile(draws, 0.975),
        },
    }


def arm_evidence(
    arm: str, input_manifest: Path, arms_root: Path
) -> dict[str, Any]:
    frozen = verify_frozen_arm(input_manifest, arm)
    output = arms_root / arm
    if not (output / "_SUCCESS").is_file():
        raise FileNotFoundError(output / "_SUCCESS")
    evaluation_report = read_json(output / "evaluation_report.json")
    sun_attempts = read_jsonl(output / "r5c_a100_sun/attempt_results.jsonl")
    if (
        evaluation_report.get("status") != "complete"
        or evaluation_report.get("ok") is not True
        or evaluation_report.get("arm") != arm
        or len(sun_attempts) != DENOMINATOR
        or [str(row.get("attempt_id")) for row in sun_attempts]
        != frozen["attempt_ids"]
    ):
        raise ValueError(f"{arm} evaluation evidence changed")
    direct = frozen["direct_attempts"]
    vectors = {
        "generation_succeeded": [
            row.get("status") == "succeeded" for row in frozen["generation"]
        ],
        "composition_valid": [bool(row["comp_valid"]) for row in direct],
        "structure_valid": [bool(row["struct_valid"]) for row in direct],
        "joint_valid": [bool(row["valid"]) for row in direct],
        "novel": [bool(row["metrics"]["novel"]) for row in sun_attempts],
        "unique": [
            bool(row["metrics"]["unique_representative"]) for row in sun_attempts
        ],
        "novel_unique": [
            bool(row["metrics"]["novel_unique"]) for row in sun_attempts
        ],
        "strict_sun": [
            bool(row["metrics"]["strict_full_sun"]) for row in sun_attempts
        ],
        "meta_sun": [
            bool(row["metrics"]["meta_full_sun"]) for row in sun_attempts
        ],
    }
    return {
        "vectors": vectors,
        "summary": {
            "attempts": DENOMINATOR,
            "counts": {key: sum(values) for key, values in vectors.items()},
            "rates": {
                key: sum(values) / DENOMINATOR for key, values in vectors.items()
            },
            "frozen_v7_generation_jsonl_sha256": frozen["specification"][
                "generation_jsonl"
            ]["sha256"],
            "frozen_v7_direct_attempts_sha256": frozen["specification"][
                "direct_attempt_metrics"
            ]["sha256"],
            "v8_evaluation_report_sha256": sha256_file(
                output / "evaluation_report.json"
            ),
            "generation_or_refinement_rerun": False,
            "direct_metrics_rerun": False,
        },
    }


def assemble(args: argparse.Namespace) -> dict[str, Any]:
    source = args.source_dir.resolve()
    require_source_manifest(source, args.source_manifest_sha256)
    config = read_json(args.config.resolve())
    if (
        config["decision_firewall"].get(
            "diagnostic_only_after_phase2_scientific_stops"
        )
        is not True
        or config["decision_firewall"].get("formal_g3") is not False
        or config["decision_firewall"].get("automatic_promotion") is not False
        or config["decision_firewall"].get("automatic_downstream") is not False
    ):
        raise ValueError("decision firewall changed")
    evidence = {
        arm: arm_evidence(
            arm, args.input_manifest.resolve(), args.arms_root.resolve()
        )
        for arm in ARM_ORDER
    }
    metric_names = tuple(evidence["M00"]["vectors"])
    effects: dict[str, Any] = {}
    for metric_index, metric in enumerate(metric_names):
        vectors = {arm: evidence[arm]["vectors"][metric] for arm in ARM_ORDER}
        offset = metric_index * 10
        effects[metric] = {
            "planner_at_B0": paired_effect(
                vectors["M10"],
                vectors["M00"],
                candidate_arm="M10",
                baseline_arm="M00",
                seed_offset=offset,
            ),
            "planner_at_B2": paired_effect(
                vectors["M11"],
                vectors["M01"],
                candidate_arm="M11",
                baseline_arm="M01",
                seed_offset=offset + 1,
            ),
            "body_at_P0": paired_effect(
                vectors["M01"],
                vectors["M00"],
                candidate_arm="M01",
                baseline_arm="M00",
                seed_offset=offset + 2,
            ),
            "body_at_Pstar": paired_effect(
                vectors["M11"],
                vectors["M10"],
                candidate_arm="M11",
                baseline_arm="M10",
                seed_offset=offset + 3,
            ),
            "joint_M11_vs_M00": paired_effect(
                vectors["M11"],
                vectors["M00"],
                candidate_arm="M11",
                baseline_arm="M00",
                seed_offset=offset + 4,
            ),
            "factorial_interaction": interaction(
                vectors, seed_offset=offset + 5
            ),
        }
    return {
        "schema": "h1a2_v3_poststop_sun256_evaluation_repair_terminal_v1",
        "status": "complete",
        "decision": "diagnostic_only_retain_phase2_scientific_stops",
        "run_id": config["run_id"],
        "arms": {arm: evidence[arm]["summary"] for arm in ARM_ORDER},
        "paired_effects": effects,
        "all_attempt_denominator_per_arm": DENOMINATOR,
        "input_reuse": {
            "source_run": config["source_v7"]["run_root"],
            "source_manifest_sha256": config["source_v7"][
                "source_manifest_sha256"
            ],
            "generation_or_refinement_rerun": False,
            "direct_metrics_rerun": False,
            "all_successful_inputs_refined_steps": 800,
        },
        "sun": {
            "strict_threshold_ev_per_atom": 0.0,
            "meta_threshold_ev_per_atom": 0.1,
            "mp_api_enabled": False,
            "frozen_cache_only": True,
            "denominator": "raw_all_attempt",
        },
        "phase2_planner_gate_passed": False,
        "phase2_body_gate_passed": False,
        "formal_g3": False,
        "checkpoint_reselection": False,
        "automatic_promotion": False,
        "automatic_downstream": False,
        "source_manifest_sha256": args.source_manifest_sha256,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--arms-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    args = parser.parse_args()

    output = args.output.resolve()
    try:
        report = assemble(args)
    except Exception as exc:
        failure = {
            "schema": "h1a2_v3_poststop_sun256_evaluation_repair_terminal_v1",
            "status": "failed",
            "decision": "diagnostic_evaluation_repair_failed_no_reselection",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
            "formal_g3": False,
            "automatic_promotion": False,
            "automatic_downstream": False,
        }
        write_json_exclusive(output, failure)
        print(json.dumps(failure, sort_keys=True))
        raise
    write_json_exclusive(output, report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
