#!/usr/bin/env python3
"""Assemble the four-arm diagnostic without selecting or promoting a model."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[3]
for location in (PROJECT_ROOT, HERE):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

from protocol import (  # noqa: E402
    ARM_ORDER,
    read_json,
    read_jsonl,
    require_hex_sha,
    require_runtime_manifest,
    require_source_manifest,
    rows_by_attempt,
    sha256_file,
    write_json_exclusive,
)


DENOMINATOR = 256
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20260801


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * float(probability)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _mcnemar(candidate: Sequence[bool], baseline: Sequence[bool]) -> dict[str, Any]:
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


def _paired_effect(
    candidate: Sequence[bool],
    baseline: Sequence[bool],
    *,
    candidate_arm: str,
    baseline_arm: str,
    seed_offset: int,
) -> dict[str, Any]:
    if len(candidate) != DENOMINATOR or len(baseline) != DENOMINATOR:
        raise ValueError("paired effect lost the 256-attempt denominator")
    differences = [
        float(bool(left)) - float(bool(right))
        for left, right in zip(candidate, baseline)
    ]
    rng = random.Random(BOOTSTRAP_SEED + int(seed_offset))
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
        "difference_percentage_points": 100.0
        * (candidate_count - baseline_count)
        / DENOMINATOR,
        "bootstrap": {
            "draws": BOOTSTRAP_DRAWS,
            "seed": BOOTSTRAP_SEED + int(seed_offset),
            "ci95_lower_percentage_points": _quantile(draws, 0.025),
            "ci95_upper_percentage_points": _quantile(draws, 0.975),
        },
        "exact_mcnemar": _mcnemar(candidate, baseline),
    }


def _interaction(
    vectors: Mapping[str, Sequence[bool]],
    *,
    seed_offset: int,
) -> dict[str, Any]:
    if any(len(vectors[arm]) != DENOMINATOR for arm in ARM_ORDER):
        raise ValueError("factorial interaction lost an ordinal")
    per_ordinal = [
        100.0
        * (
            float(vectors["M11"][idx])
            - float(vectors["M10"][idx])
            - float(vectors["M01"][idx])
            + float(vectors["M00"][idx])
        )
        for idx in range(DENOMINATOR)
    ]
    rng = random.Random(BOOTSTRAP_SEED + int(seed_offset))
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
            "seed": BOOTSTRAP_SEED + int(seed_offset),
            "ci95_lower_percentage_points": _quantile(draws, 0.025),
            "ci95_upper_percentage_points": _quantile(draws, 0.975),
        },
    }


def _arm_vectors(arm: str, arm_root: Path) -> dict[str, Any]:
    generation_dir = arm_root / "generation"
    evaluation_dir = arm_root / "evaluation"
    if not (generation_dir / "_SUCCESS").is_file():
        raise FileNotFoundError(generation_dir / "_SUCCESS")
    if not (evaluation_dir / "_SUCCESS").is_file():
        raise FileNotFoundError(evaluation_dir / "_SUCCESS")
    generation_report = read_json(generation_dir / "generation_report.json")
    evaluation_report = read_json(evaluation_dir / "evaluation_report.json")
    generation = read_jsonl(generation_dir / "generation.jsonl")
    direct = rows_by_attempt(
        evaluation_dir / "crysllmgen_metrics" / "attempt_metrics.jsonl",
        "crysllmgen_metric_attempt_v1",
    )
    sun = rows_by_attempt(
        evaluation_dir / "r5c_a100_sun" / "attempt_results.jsonl",
        "crysllmgen_r5c_a100_sun_attempt_v1",
    )
    expected_ids = [str(row["attempt_id"]) for row in generation]
    if (
        generation_report.get("ok") is not True
        or evaluation_report.get("ok") is not True
        or generation_report.get("all_successes_diffusion_refined") is not True
        or int(generation_report.get("diffusion_steps", -1)) != 800
        or len(generation) != DENOMINATOR
        or list(direct) != expected_ids
        or list(sun) != expected_ids
    ):
        raise ValueError(f"{arm} evidence mapping changed")
    vectors = {
        "generation_succeeded": [
            row.get("status") == "succeeded" for row in generation
        ],
        "composition_valid": [
            bool(direct[attempt]["comp_valid"]) for attempt in expected_ids
        ],
        "structure_valid": [
            bool(direct[attempt]["struct_valid"]) for attempt in expected_ids
        ],
        "joint_valid": [
            bool(direct[attempt]["valid"]) for attempt in expected_ids
        ],
        "novel": [
            bool(sun[attempt]["metrics"]["novel"]) for attempt in expected_ids
        ],
        "unique": [
            bool(sun[attempt]["metrics"]["unique_representative"])
            for attempt in expected_ids
        ],
        "novel_unique": [
            bool(sun[attempt]["metrics"]["novel_unique"])
            for attempt in expected_ids
        ],
        "strict_sun": [
            bool(sun[attempt]["metrics"]["strict_full_sun"])
            for attempt in expected_ids
        ],
        "meta_sun": [
            bool(sun[attempt]["metrics"]["meta_full_sun"])
            for attempt in expected_ids
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
            "diffusion_refinement": {
                "checkpoint": generation_report["diffusion_refiner"],
                "reverse_steps": 800,
                "applied_to_every_success": True,
            },
            "generation_report_sha256": sha256_file(
                generation_dir / "generation_report.json"
            ),
            "evaluation_report_sha256": sha256_file(
                evaluation_dir / "evaluation_report.json"
            ),
        },
    }


def _assemble(args: argparse.Namespace) -> dict[str, Any]:
    execution_sha = require_hex_sha(
        args.execution_manifest_sha256,
        "execution source manifest",
    )
    source = args.source_dir.resolve()
    require_source_manifest(source, execution_sha)
    require_runtime_manifest(args.project_root.resolve(), source)
    config = read_json(args.config.resolve())
    if (
        config["decision_firewall"].get("diagnostic_only") is not True
        or config["decision_firewall"].get("formal_g3") is not False
        or config["decision_firewall"].get("automatic_promotion") is not False
        or config["decision_firewall"].get("automatic_downstream") is not False
    ):
        raise ValueError("terminal decision firewall changed")

    arm_evidence = {
        arm: _arm_vectors(arm, args.arms_root.resolve() / arm)
        for arm in ARM_ORDER
    }
    metric_names = tuple(arm_evidence["M00"]["vectors"])
    effects: dict[str, Any] = {}
    for metric_index, metric in enumerate(metric_names):
        vectors = {
            arm: arm_evidence[arm]["vectors"][metric] for arm in ARM_ORDER
        }
        offset = metric_index * 10
        effects[metric] = {
            "planner_at_B0": _paired_effect(
                vectors["M10"],
                vectors["M00"],
                candidate_arm="M10",
                baseline_arm="M00",
                seed_offset=offset,
            ),
            "planner_at_B2": _paired_effect(
                vectors["M11"],
                vectors["M01"],
                candidate_arm="M11",
                baseline_arm="M01",
                seed_offset=offset + 1,
            ),
            "body_at_P0": _paired_effect(
                vectors["M01"],
                vectors["M00"],
                candidate_arm="M01",
                baseline_arm="M00",
                seed_offset=offset + 2,
            ),
            "body_at_Pstar": _paired_effect(
                vectors["M11"],
                vectors["M10"],
                candidate_arm="M11",
                baseline_arm="M10",
                seed_offset=offset + 3,
            ),
            "joint_M11_vs_M00": _paired_effect(
                vectors["M11"],
                vectors["M00"],
                candidate_arm="M11",
                baseline_arm="M00",
                seed_offset=offset + 4,
            ),
            "factorial_interaction": _interaction(
                vectors,
                seed_offset=offset + 5,
            ),
        }

    return {
        "schema": "h1a2_v3_poststop_sun256_terminal_v1",
        "status": "complete",
        "decision": "diagnostic_only_retain_phase2_scientific_stops",
        "run_id": config["run_id"],
        "arms": {
            arm: arm_evidence[arm]["summary"] for arm in ARM_ORDER
        },
        "paired_effects": effects,
        "all_attempt_denominator_per_arm": DENOMINATOR,
        "planner_source": {
            "reused_frozen_planner512_ordinals": "0..255",
            "P0_step": 2,
            "Pstar_step": 400,
        },
        "body_source": {
            "B0": "frozen_R5C",
            "B2_step": 1696,
        },
        "diffusion_refinement": {
            "required_for_all_arms": True,
            "checkpoint": config["refiner"]["name"],
            "checkpoint_sha256": config["refiner"]["checkpoint_sha256"],
            "reverse_steps": 800,
            "same_refiner_and_paired_noise_all_arms": True,
        },
        "sun": {
            "strict_threshold_ev_per_atom": 0.0,
            "meta_threshold_ev_per_atom": 0.1,
            "mp_api_enabled": False,
            "denominator": "raw_all_attempt",
        },
        "phase2_planner_gate_passed": False,
        "phase2_body_gate_passed": False,
        "formal_g3": False,
        "checkpoint_reselection_from_generation_or_sun": False,
        "automatic_promotion": False,
        "automatic_downstream": False,
        "execution_manifest_sha256": execution_sha,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--arms-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execution-manifest-sha256", required=True)
    args = parser.parse_args()

    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    try:
        report = _assemble(args)
    except Exception as exc:
        failure = {
            "schema": "h1a2_v3_poststop_sun256_terminal_v1",
            "status": "failed",
            "decision": "diagnostic_execution_failed_no_reselection",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
            "formal_g3": False,
            "automatic_promotion": False,
            "automatic_downstream": False,
        }
        write_json_exclusive(output / "terminal_report.json", failure)
        print(json.dumps(failure, sort_keys=True))
        raise
    write_json_exclusive(output / "terminal_report.json", report)
    with (output / "_SUCCESS").open("x", encoding="ascii") as handle:
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
