#!/usr/bin/env python3
"""Merge reused prefix and new reserve refinements into 1,000 structures."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch

from finalize_post1000 import _structure
from native_protocol import (
    NATIVE_DENOMINATOR,
    PREFIX_COUNT,
    candidate_seed,
    identity,
    read_json,
    read_jsonl,
    sha256_file,
    validate_arm,
    validate_repeat,
    write_json_exclusive,
    write_jsonl_exclusive,
)
from protocol import ordered_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True)
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument("--selected-body-dir", type=Path, required=True)
    parser.add_argument("--main-post-generation-dir", type=Path, required=True)
    parser.add_argument("--reserve-refinement-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    arm = validate_arm(args.arm)
    repeat = validate_repeat(args.repeat)
    selected_root = args.selected_body_dir.resolve()
    main_post_root = args.main_post_generation_dir.resolve()
    reserve_root = args.reserve_refinement_dir.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)

    selected = ordered_rows(
        read_jsonl(selected_root / "body_attempts.jsonl"), ordinal_field="ordinal"
    )
    selection_report = read_json(selected_root / "selection_report.json")
    main_post = ordered_rows(
        read_jsonl(main_post_root / "generation.jsonl"), ordinal_field="ordinal"
    )
    main_report = read_json(main_post_root / "generation_report.json")
    reserve_metrics = read_json(reserve_root / "refinement_metrics.json")
    if (
        int(selection_report.get("selected_body_successes", -1)) != NATIVE_DENOMINATOR
        or int(main_report.get("attempts", -1)) != NATIVE_DENOMINATOR
        or main_report.get("stage") != "post_model494"
        or str(main_report.get("arm")) != arm
        or int(main_report.get("repeat", -1)) != repeat
        or reserve_metrics.get("status") != "complete"
        or reserve_metrics.get("all_selected_candidates_refined_after_merge") is not True
    ):
        raise ValueError("native selection or source refinement contract changed")

    payload_paths = sorted(reserve_root.glob("dlm_refined_mp_*.pt"))
    if len(payload_paths) != 1:
        raise ValueError("expected one reserve refined payload")
    payload = torch.load(payload_paths[0], map_location="cpu")
    reserve_ordinals = [
        int(value) for value in payload["sample_idx"].detach().cpu().tolist()
    ]
    expected_reserve_ordinals = [
        int(row["ordinal"])
        for row in selected
        if int(row["source_candidate_rank"]) >= PREFIX_COUNT
    ]
    if reserve_ordinals != expected_reserve_ordinals:
        raise ValueError("reserve refined payload order changed")
    reserve_structures: dict[int, dict[str, Any]] = {}
    atom_offset = 0
    for success_index, ordinal in enumerate(reserve_ordinals):
        structure, atom_offset = _structure(
            payload, success_index=success_index, atom_offset=atom_offset
        )
        reserve_structures[ordinal] = structure
    if atom_offset != int(payload["atom_types"].shape[1]):
        raise ValueError("reserve refined payload contains trailing sites")

    rows: list[dict[str, Any]] = []
    prefix_reused = 0
    reserve_new = 0
    for native_ordinal, body_row in enumerate(selected):
        rank = int(body_row["source_candidate_rank"])
        expected_seed = candidate_seed(repeat, rank, "refiner")
        if (
            body_row.get("status") != "succeeded"
            or int(body_row.get("source_refiner_noise_seed", -1)) != expected_seed
        ):
            raise ValueError(f"selected body identity changed at native ordinal {native_ordinal}")
        if rank < PREFIX_COUNT:
            source = main_post[rank]
            structure = source.get("structure")
            if (
                source.get("status") != "succeeded"
                or not isinstance(structure, dict)
                or source.get("diffusion_refinement_applied") is not True
                or int(source.get("refiner_noise_seed", -1)) != expected_seed
            ):
                raise RuntimeError(
                    f"selected prefix candidate {rank} lacks a reusable refinement"
                )
            prefix_reused += 1
            refinement_source = "v3_prefix_reused"
        else:
            structure = reserve_structures.get(native_ordinal)
            if not isinstance(structure, dict):
                raise RuntimeError(
                    f"selected reserve candidate {rank} lacks a completed refinement"
                )
            reserve_new += 1
            refinement_source = "native_reserve_new"
        rows.append(
            {
                "schema": "wqcodiff_generation_attempt_v1",
                "attempt_id": (
                    f"h1-plan1200-native-{arm.lower()}-r{repeat}-"
                    f"post-model494-{native_ordinal:04d}"
                ),
                "method": f"P0-{arm}-SAFEAXIS-post_model494",
                "ordinal": native_ordinal,
                "sample_idx": native_ordinal,
                "pair_id": f"h1-plan1200-native-r{repeat}:{native_ordinal:04d}",
                "repeat": repeat,
                "arm": arm,
                "planner_arm": "P0",
                "body_arm": arm,
                "evaluation_stage": "post_model494",
                "schedule_arm": "D2_SAFE_AXIS",
                "status": "succeeded",
                "reason": "",
                "structure": structure,
                "body_noise_seed": int(body_row["source_body_noise_seed"]),
                "refiner_noise_seed": expected_seed,
                "source_plan_state_sha256": body_row.get("plan_state_sha256"),
                "source_candidate_rank": rank,
                "source_planner_candidate_ordinal": int(
                    body_row["planner_candidate_ordinal"]
                ),
                "refinement_source": refinement_source,
                "sampling_contract": "crysllmgen_native_first_1000_body_successes",
                "diffusion_refinement_applied": True,
                "diffusion_refinement_steps": 800,
                "retry_or_replacement_used": False,
            }
        )
    if prefix_reused + reserve_new != NATIVE_DENOMINATOR:
        raise AssertionError("native post-refine denominator changed")

    output.mkdir(parents=True)
    generation_path = output / "generation.jsonl"
    write_jsonl_exclusive(generation_path, rows)
    report = {
        "schema": "h1_plan1200_crysllmgen_native_generation_report_v1",
        "status": "complete",
        "ok": True,
        "arm": arm,
        "repeat": repeat,
        "stage": "post_model494",
        "planner": "P0",
        "body": arm,
        "method": f"P0-{arm}-SAFEAXIS-post_model494",
        "attempts": NATIVE_DENOMINATOR,
        "generation_succeeded": NATIVE_DENOMINATOR,
        "generation_failed": 0,
        "refiner_complete": NATIVE_DENOMINATOR,
        "prefix_refinements_reused": prefix_reused,
        "reserve_refinements_new": reserve_new,
        "sampling_contract": "crysllmgen_native_first_1000_body_successes",
        "all_1000_diffusion_refined": True,
        "diffusion_steps": 800,
        "generation_jsonl_sha256": sha256_file(generation_path),
        "same_plan_retry": False,
        "stochastic_replacement": False,
        "retry_replacement_repair_filter_rerank": False,
        "artifacts": {
            "selection_report": identity(selected_root / "selection_report.json"),
            "main_post_generation": identity(main_post_root / "generation.jsonl"),
            "reserve_refinement_report": identity(
                reserve_root / "refinement_metrics.json"
            ),
            "reserve_refined_payload": identity(payload_paths[0]),
        },
    }
    write_json_exclusive(output / "generation_report.json", report)
    with (output / "_SUCCESS").open("x", encoding="ascii") as handle:
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
