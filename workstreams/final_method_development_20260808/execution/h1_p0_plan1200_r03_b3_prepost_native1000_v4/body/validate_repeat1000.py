#!/usr/bin/env python3
"""Validate one arm/repeat across pre- and post-model_494 stages."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from protocol import (
    DENOMINATOR,
    PAIRED_SEED_NAMESPACE,
    ordered_rows,
    paired_seed,
    read_json,
    read_jsonl,
    require_source_manifest,
    sha256_file,
    validate_arm,
    validate_config,
    validate_repeat,
    write_json_exclusive,
)


def identity(path: Path) -> dict[str, Any]:
    location = path.resolve()
    if not location.is_file():
        raise FileNotFoundError(location)
    return {"path": str(location), "bytes": location.stat().st_size, "sha256": sha256_file(location)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True)
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--repeat-root", type=Path, required=True)
    args = parser.parse_args()

    arm = validate_arm(args.arm)
    repeat = validate_repeat(args.repeat)
    require_source_manifest(args.source_dir.resolve(), args.source_manifest_sha256)
    config = read_json(args.config.resolve())
    validate_config(config)
    root = args.repeat_root.resolve()
    if root.exists() is False:
        raise FileNotFoundError(root)

    body = ordered_rows(read_jsonl(root / "body/body_attempts.jsonl"), ordinal_field="ordinal")
    refinement = ordered_rows(
        read_jsonl(root / "refinement/refinement_attempts.jsonl"), ordinal_field="ordinal"
    )
    pre = ordered_rows(
        read_jsonl(root / "stages/pre_model494/generation/generation.jsonl"),
        ordinal_field="ordinal",
    )
    post = ordered_rows(
        read_jsonl(root / "stages/post_model494/generation/generation.jsonl"),
        ordinal_field="ordinal",
    )
    for ordinal, (body_row, refinement_row, pre_row, post_row) in enumerate(
        zip(body, refinement, pre, post, strict=True)
    ):
        body_seed = paired_seed(repeat, ordinal, "body")
        refiner_seed = paired_seed(repeat, ordinal, "refiner")
        if (
            str(body_row.get("arm")) != arm
            or int(body_row.get("repeat", -1)) != repeat
            or int(body_row.get("body_noise_seed", -1)) != body_seed
            or int(body_row.get("refiner_noise_seed", -1)) != refiner_seed
            or int(refinement_row.get("body_noise_seed", -1)) != body_seed
            or int(refinement_row.get("refiner_sampling_seed", -1)) != refiner_seed
            or int(pre_row.get("body_noise_seed", -1)) != body_seed
            or int(pre_row.get("refiner_noise_seed", -1)) != refiner_seed
            or int(post_row.get("body_noise_seed", -1)) != body_seed
            or int(post_row.get("refiner_noise_seed", -1)) != refiner_seed
            or pre_row.get("source_plan_state_sha256") != body_row.get("plan_state_sha256")
            or post_row.get("source_plan_state_sha256") != body_row.get("plan_state_sha256")
            or pre_row.get("pair_id") != post_row.get("pair_id")
            or pre_row.get("diffusion_refinement_applied") is not False
            or (
                post_row.get("status") == "succeeded"
                and post_row.get("diffusion_refinement_applied") is not True
            )
            or pre_row.get("retry_or_replacement_used") is not False
            or post_row.get("retry_or_replacement_used") is not False
        ):
            raise ValueError(f"paired pre/post identity changed at ordinal {ordinal}")

    stages: dict[str, Any] = {}
    for stage in ("pre_model494", "post_model494"):
        stage_root = root / "stages" / stage
        report_path = stage_root / "evaluation/stage_report.json"
        report = read_json(report_path)
        if (
            not (stage_root / "evaluation/_SUCCESS").is_file()
            or report.get("status") != "complete"
            or report.get("ok") is not True
            or report.get("arm") != arm
            or int(report.get("repeat", -1)) != repeat
            or report.get("stage") != stage
            or int(report.get("attempts", -1)) != DENOMINATOR
            or report.get("source_manifest_sha256") != args.source_manifest_sha256
        ):
            raise ValueError(f"{stage} validation evidence changed")
        stages[stage] = report

    body_report = read_json(root / "body/generation_report.json")
    refinement_report = read_json(root / "refinement/refinement_metrics.json")
    report = {
        "schema": "h1_plan1200_arm_repeat_report_v1",
        "status": "complete",
        "ok": True,
        "arm": arm,
        "repeat": repeat,
        "attempts": DENOMINATOR,
        "body_succeeded": int(body_report["succeeded"]),
        "body_failed": int(body_report["failed"]),
        "refiner_complete": int(refinement_report["refiner_complete"]),
        "stages": stages,
        "paired_seed_mode": "paired_sha256_repeat_ordinal_v1",
        "paired_seed_namespace": PAIRED_SEED_NAMESPACE,
        "same_frozen_plans_between_arms": True,
        "same_ordinals_pre_post": True,
        "artifacts": {
            "body_attempts": identity(root / "body/body_attempts.jsonl"),
            "body_report": identity(root / "body/generation_report.json"),
            "refinement_attempts": identity(root / "refinement/refinement_attempts.jsonl"),
            "refinement_report": identity(root / "refinement/refinement_metrics.json"),
            "pre_generation": identity(
                root / "stages/pre_model494/generation/generation.jsonl"
            ),
            "post_generation": identity(
                root / "stages/post_model494/generation/generation.jsonl"
            ),
            "pre_stage_report": identity(
                root / "stages/pre_model494/evaluation/stage_report.json"
            ),
            "post_stage_report": identity(
                root / "stages/post_model494/evaluation/stage_report.json"
            ),
        },
        "source_manifest_sha256": args.source_manifest_sha256,
        "retry_replacement_repair_filter_rerank": False,
        "automatic_training": False,
        "automatic_promotion": False,
        "automatic_rl": False,
    }
    write_json_exclusive(root / "repeat_report.json", report)
    with (root / "_SUCCESS").open("x", encoding="ascii") as handle:
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
