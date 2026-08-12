#!/usr/bin/env python3
"""Freeze the first 256 records of an exact historical raw-1200 world2 sample."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from crystal_dlm.r5_plan_state import build_body_prompt

from protocol import (
    DENOMINATOR,
    PLANNER_RAW_ATTEMPTS,
    canonical_sha256,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_text,
    validate_config,
    write_json_exclusive,
    write_jsonl_exclusive,
)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return dict(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--planner-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--checkpoint-config-sha256", required=True)
    args = parser.parse_args()

    planner_dir = args.planner_dir.resolve()
    config = read_json(args.config.resolve())
    validate_config(config)
    planner_config = config["planner"]
    registered = [
        *planner_config["cohorts"],
        config["topology_match"]["planner"],
    ]
    matching = [
        spec
        for spec in registered
        if spec["cohort_id"] == args.cohort_id and int(spec["seed"]) == args.seed
    ]
    if len(matching) != 1:
        raise ValueError("cohort id/seed is not registered")
    if args.checkpoint_sha256 != config["training_upstream"]["adapter_sha256"]:
        raise ValueError("planner checkpoint identity changed")
    if args.checkpoint_config_sha256 != config["training_upstream"]["adapter_config_sha256"]:
        raise ValueError("planner checkpoint config identity changed")
    run_config = read_json(planner_dir / "run_config.json")
    expected_run_config = {
        "model_path": planner_config["base_model"],
        "checkpoint_path": config["training_upstream"]["adapter_path"],
        "output_dir": str(planner_dir),
        "num_samples": PLANNER_RAW_ATTEMPTS,
        "batch_size": 4,
        "max_new_tokens": 96,
        "temperature": 0.9,
        "top_p": 0.95,
        "top_k": 50,
        "max_atoms": 20,
        "prompt_style": "h1_rich_plan_v1",
        "include_sample_id": False,
        "seed": args.seed,
        "do_sample": True,
        "stop_after_plan_marker": True,
        "truncate_after_plan_marker": True,
        "distributed": True,
        "world_size": 2,
        "method": "h1_llm_formula_planner",
        "prompt_version": "h1_llm_formula_planner_v1",
        "effective_do_sample": True,
        "rich_field_required": True,
    }
    for key, expected in expected_run_config.items():
        if run_config.get(key) != expected:
            raise ValueError(
                f"planner run_config {key} changed: "
                f"expected={expected!r} observed={run_config.get(key)!r}"
            )
    raw_path = planner_dir / "raw_generations.jsonl"
    rows = read_jsonl(raw_path)
    if len(rows) != PLANNER_RAW_ATTEMPTS:
        raise ValueError("planner raw-attempt count changed")
    metrics = read_json(planner_dir / "sample_metrics.json")
    if (
        int(metrics.get("requested_samples", -1)) != PLANNER_RAW_ATTEMPTS
        or int(metrics.get("decoded_samples", -1)) != PLANNER_RAW_ATTEMPTS
        or int(metrics.get("plan_parse_success", -1))
        != sum(row.get("parsed") is True for row in rows)
        or metrics.get("distributed") is not True
        or int(metrics.get("world_size", -1)) != 2
    ):
        raise ValueError("planner sample metrics contract changed")
    sample_indexes = [int(row.get("sample_idx", -1)) for row in rows]
    if sorted(sample_indexes) != list(range(PLANNER_RAW_ATTEMPTS)):
        raise ValueError("planner sample_idx coverage changed")
    expected_rank_concatenation = list(range(0, PLANNER_RAW_ATTEMPTS, 2)) + list(
        range(1, PLANNER_RAW_ATTEMPTS, 2)
    )
    if sample_indexes != expected_rank_concatenation:
        raise ValueError("historical world-size-2 rank-concatenated order changed")
    concatenated_rank_rows: list[dict[str, Any]] = []
    for rank in (0, 1):
        rank_rows = read_jsonl(planner_dir / f"raw_generations.rank{rank}.jsonl")
        if (
            len(rank_rows) != PLANNER_RAW_ATTEMPTS // 2
            or [int(row.get("sample_idx", -1)) for row in rank_rows]
            != list(range(rank, PLANNER_RAW_ATTEMPTS, 2))
        ):
            raise ValueError(f"planner rank{rank} raw order changed")
        rank_metrics = read_json(
            planner_dir / f"sample_metrics.rank{rank}.json"
        )
        if (
            int(rank_metrics.get("rank", -1)) != rank
            or int(rank_metrics.get("world_size", -1)) != 2
            or int(rank_metrics.get("requested_samples", -1))
            != PLANNER_RAW_ATTEMPTS // 2
            or int(rank_metrics.get("decoded_samples", -1))
            != PLANNER_RAW_ATTEMPTS // 2
            or int(rank_metrics.get("plan_parse_success", -1))
            != sum(row.get("parsed") is True for row in rank_rows)
        ):
            raise ValueError(f"planner rank{rank} metrics changed")
        concatenated_rank_rows.extend(rank_rows)
    if [canonical_sha256(row) for row in rows] != [
        canonical_sha256(row) for row in concatenated_rank_rows
    ]:
        raise ValueError("merged raw generations differ from rank concatenation")
    if any(
        row.get("planner_model_path") != planner_config["base_model"]
        or row.get("planner_checkpoint_path")
        != config["training_upstream"]["adapter_path"]
        or row.get("prompt_style") != "h1_rich_plan_v1"
        or row.get("prompt_version") != "h1_llm_formula_planner_v1"
        or "seed_mode" in row
        or "planner_sampling_seed" in row
        or "formula_constraint_mode" in row
        for row in rows
    ):
        raise ValueError("planner raw record provenance changed")

    selected_rows = rows[:DENOMINATOR]
    if [int(row["sample_idx"]) for row in selected_rows] != (
        expected_rank_concatenation[:DENOMINATOR]
    ):
        raise ValueError("historical raw1200-to-first256 selection changed")

    frozen: list[dict[str, Any]] = []
    formula_hist: Counter[str] = Counter()
    element_count_hist: Counter[str] = Counter()
    atom_count_hist: Counter[str] = Counter()
    charge_hist: Counter[str] = Counter()
    parse_failures: Counter[str] = Counter()
    for ordinal, raw in enumerate(selected_rows):
        parsed = raw.get("parsed") is True
        record: dict[str, Any] = {
            "schema": "h1a2_retrained_world2_cohort_attempt_v1",
            "cohort_id": args.cohort_id,
            "cohort_ordinal": ordinal,
            "planner_sample_idx": int(raw["sample_idx"]),
            "planner_sampling_seed": args.seed,
            "planner_rank": int(raw["sample_idx"]) % 2,
            "planner_effective_rank_seed": args.seed + int(raw["sample_idx"]) % 2,
            "planner_world_size": 2,
            "planner_batch_size_per_rank": 4,
            "planner_rng": "stateful_torch_seed_plus_rank",
            "planner_merge_order": "rank_concatenated_file_order",
            "parsed": parsed,
            "raw_record_sha256": canonical_sha256(raw),
            "raw_plan_text_sha256": sha256_text(str(raw.get("raw_plan_text") or "")),
            "raw_rich_seven_line_forwarded": False,
            "body_prompt_contract": "historical_r5c_plan_state_json_exact_length",
            "retry": False,
            "replacement": False,
            "repair": False,
            "filter": False,
            "rerank": False,
        }
        if not parsed:
            reason = str(raw.get("reason") or "planner_parse_failure")
            parse_failures[reason] += 1
            record.update({"body_eligible": False, "ineligible_reason": f"planner:{reason}", "plan_state": None, "plan_state_sha256": None, "body_prompt": None, "body_prompt_sha256": None, "canonical_charge_bucket_visible": False})
        else:
            plan = _mapping(raw.get("plan_state"), f"plan_state[{ordinal}]")
            prompt = build_body_prompt(plan)
            if not prompt.endswith("dynamic_crystal_body:") or '"charge_bucket"' not in prompt:
                raise ValueError(f"historical body prompt changed at ordinal {ordinal}")
            elements = tuple(str(value) for value in (plan.get("elements") or []))
            if not elements or int(plan.get("N", 0)) <= 0:
                raise ValueError(f"parsed plan is incomplete at ordinal {ordinal}")
            formula_hist[str(plan.get("formula") or "")] += 1
            element_count_hist[str(len(elements))] += 1
            atom_count_hist[str(int(plan["N"]))] += 1
            charge_hist[str(plan.get("charge_bucket"))] += 1
            record.update({"body_eligible": True, "ineligible_reason": None, "plan_state": plan, "plan_state_sha256": canonical_sha256(plan), "body_prompt": prompt, "body_prompt_sha256": sha256_text(prompt), "canonical_charge_bucket_visible": True})
        frozen.append(record)

    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    cohort_path = output / "cohort256.jsonl"
    write_jsonl_exclusive(cohort_path, frozen)
    report = {
        "schema": "h1a2_retrained_world2_cohort_report_v2",
        "cohort_id": args.cohort_id,
        "seed": args.seed,
        "world_size": 2,
        "batch_size_per_rank": 4,
        "num_samples": PLANNER_RAW_ATTEMPTS,
        "raw_attempts": PLANNER_RAW_ATTEMPTS,
        "raw_parsed": sum(row.get("parsed") is True for row in rows),
        "raw_planner_failed": sum(row.get("parsed") is not True for row in rows),
        "raw_generations_sha256": sha256_file(raw_path),
        "rank_raw_sha256": {
            str(rank): sha256_file(planner_dir / f"raw_generations.rank{rank}.jsonl")
            for rank in (0, 1)
        },
        "run_config_sha256": sha256_file(planner_dir / "run_config.json"),
        "sample_metrics_sha256": sha256_file(planner_dir / "sample_metrics.json"),
        "checkpoint_sha256": args.checkpoint_sha256,
        "checkpoint_config_sha256": args.checkpoint_config_sha256,
        "effective_rank_seeds": [args.seed, args.seed + 1],
        "sampler_contract": "historical_d38743_implicit_stateful_rank_rng",
        "seed_mode": "legacy_rank_implicit_in_frozen_sampler",
        "formula_constraint_mode": "not_implemented_in_frozen_sampler_equivalent_to_off",
        "selection": "first_256_raw_records_in_merged_file_order_with_failures_preserved",
        "frozen_attempts": DENOMINATOR,
        "attempts": DENOMINATOR,
        "parsed": sum(row["parsed"] is True for row in frozen),
        "planner_failed": sum(row["parsed"] is not True for row in frozen),
        "planner_sample_idx_sequence_sha256": canonical_sha256([row["planner_sample_idx"] for row in frozen]),
        "selected_rank_counts": {
            "0": sum(int(row["planner_rank"]) == 0 for row in frozen),
            "1": sum(int(row["planner_rank"]) == 1 for row in frozen),
        },
        "cohort256_sha256": sha256_file(cohort_path),
        "distribution": {
            "formula_top30": formula_hist.most_common(30),
            "element_count": dict(sorted(element_count_hist.items())),
            "atom_count": dict(sorted(atom_count_hist.items(), key=lambda item: int(item[0]))),
            "charge_bucket": dict(sorted(charge_hist.items())),
            "parse_failures": dict(sorted(parse_failures.items())),
        },
        "retry_replacement_repair_filter_rerank": False,
    }
    write_json_exclusive(output / "cohort_report.json", report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
