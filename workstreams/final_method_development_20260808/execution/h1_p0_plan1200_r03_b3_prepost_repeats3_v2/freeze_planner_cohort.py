#!/usr/bin/env python3
"""Audit one P0 Plan1200 draw and freeze its shared full-1000 cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from crystal_dlm.r5_plan_state import build_body_prompt


RAW_DENOMINATOR = 1200
COHORT_DENOMINATOR = 1000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected one JSON object")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected one object")
            rows.append(value)
    return rows


def write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_jsonl_exclusive(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument("--expected-seed", type=int, required=True)
    parser.add_argument("--planner-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.repeat not in {0, 1, 2}:
        raise ValueError("repeat must be 0, 1, or 2")
    planner_dir = args.planner_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)

    raw_path = planner_dir / "raw_generations.jsonl"
    plans_path = planner_dir / "plans_for_dlm.jsonl"
    metrics_path = planner_dir / "sample_metrics.json"
    config_path = planner_dir / "run_config.json"
    raw = sorted(read_jsonl(raw_path), key=lambda row: int(row["sample_idx"]))
    plans = sorted(read_jsonl(plans_path), key=lambda row: int(row["sample_idx"]))
    metrics = read_json(metrics_path)
    run_config = read_json(config_path)

    raw_ordinals = [int(row.get("sample_idx", -1)) for row in raw]
    plan_ordinals = [int(row.get("sample_idx", -1)) for row in plans]
    expected_config = {
        "num_samples": RAW_DENOMINATOR,
        "seed": int(args.expected_seed),
        "seed_mode": "stateless_ordinal_v1",
        "prompt_style": "h1_rich_plan_v1",
        "include_sample_id": False,
        "temperature": 0.9,
        "top_p": 0.95,
        "top_k": 50,
        "max_new_tokens": 96,
        "max_atoms": 20,
        "formula_constraint_mode": "off",
    }
    observed_config = {key: run_config.get(key) for key in expected_config}
    if observed_config != expected_config:
        raise ValueError(
            f"planner run contract changed: expected={expected_config} observed={observed_config}"
        )
    if (
        len(raw) != RAW_DENOMINATOR
        or raw_ordinals != list(range(RAW_DENOMINATOR))
        or len(set(plan_ordinals)) != len(plan_ordinals)
        or len(plans) < COHORT_DENOMINATOR
        or int(metrics.get("requested_samples", -1)) != RAW_DENOMINATOR
        or int(metrics.get("decoded_samples", -1)) != RAW_DENOMINATOR
        or int(metrics.get("plan_parse_success", -1)) != len(plans)
    ):
        raise ValueError("Plan1200 all-attempt or parse-success denominator changed")

    raw_by_ordinal = {int(row["sample_idx"]): row for row in raw}
    selected = plans[:COHORT_DENOMINATOR]
    selected_ordinals = [int(row["sample_idx"]) for row in selected]
    reserve = plans[COHORT_DENOMINATOR:]
    if selected_ordinals != sorted(selected_ordinals):
        raise AssertionError("cohort selection is not planner-ordinal stable")

    cohort_rows: list[dict[str, Any]] = []
    prompt_hashes: list[str] = []
    for cohort_ordinal, plan_record in enumerate(selected):
        plan_state = plan_record.get("plan_state")
        if not isinstance(plan_state, dict):
            raise ValueError(f"selected planner row {cohort_ordinal} lacks plan_state")
        prompt = build_body_prompt(plan_state)
        if "plan_state:" not in prompt or '"charge_bucket"' not in prompt:
            raise ValueError("historical R5-C body prompt lost canonical charge_bucket")
        planner_ordinal = int(plan_record["sample_idx"])
        raw_record = raw_by_ordinal[planner_ordinal]
        raw_text = str(raw_record.get("raw_plan_text") or "")
        if "charge:" not in raw_text.lower():
            raise ValueError("P0 rich seven-line plan lost its raw charge field")
        prompt_sha = sha256_text(prompt)
        prompt_hashes.append(prompt_sha)
        cohort_rows.append(
            {
                **plan_record,
                "repeat": int(args.repeat),
                "cohort_ordinal": int(cohort_ordinal),
                "planner_candidate_ordinal": planner_ordinal,
                "attempt_id": (
                    f"p0-plan1200-r{int(args.repeat)}-{int(cohort_ordinal):04d}"
                ),
                "body_prompt": prompt,
                "body_prompt_sha256": prompt_sha,
                "body_prompt_contract": "historical_r5c_plan_state_json_exact_length",
                "raw_rich_seven_line_forwarded": False,
                "canonical_charge_bucket_visible": True,
            }
        )

    output_dir.mkdir(parents=True)
    cohort_path = output_dir / "cohort1000.jsonl"
    write_jsonl_exclusive(cohort_path, cohort_rows)
    failure_reasons = Counter(
        str(row.get("reason") or "parsed") if not bool(row.get("parsed")) else "parsed"
        for row in raw
    )
    manifest = {
        "schema": "h1_p0_plan1200_frozen_cohort1000_v1",
        "status": "complete",
        "repeat": int(args.repeat),
        "planner_seed": int(args.expected_seed),
        "seed_mode": "stateless_ordinal_v1",
        "raw_attempts": RAW_DENOMINATOR,
        "parse_successes": len(plans),
        "parse_failures": RAW_DENOMINATOR - len(plans),
        "selected_attempts": COHORT_DENOMINATOR,
        "selection": "first_1000_parse_successes_by_planner_ordinal",
        "selected_planner_ordinals": selected_ordinals,
        "reserve_parse_success_count": len(reserve),
        "reserve_parse_success_ordinals": [int(row["sample_idx"]) for row in reserve],
        "raw_status_counts": dict(sorted(failure_reasons.items())),
        "shared_between_R03_and_B3": True,
        "arm_outcome_dependent_replacement": False,
        "raw_rich_seven_line_forwarded": False,
        "model_visible_prompt_contract": "historical_r5c_plan_state_json_exact_length",
        "canonical_charge_bucket_visible": True,
        "unique_body_prompt_sha256": len(set(prompt_hashes)),
        "artifacts": {
            "raw_generations": {
                "path": str(raw_path),
                "bytes": raw_path.stat().st_size,
                "sha256": sha256_file(raw_path),
            },
            "plans_for_dlm": {
                "path": str(plans_path),
                "bytes": plans_path.stat().st_size,
                "sha256": sha256_file(plans_path),
            },
            "sample_metrics": {
                "path": str(metrics_path),
                "bytes": metrics_path.stat().st_size,
                "sha256": sha256_file(metrics_path),
            },
            "run_config": {
                "path": str(config_path),
                "bytes": config_path.stat().st_size,
                "sha256": sha256_file(config_path),
            },
            "cohort1000": {
                "path": str(cohort_path),
                "bytes": cohort_path.stat().st_size,
                "sha256": sha256_file(cohort_path),
            },
        },
        "automatic_body_submission": False,
        "retry": False,
        "repair": False,
        "filter": False,
        "rerank": False,
    }
    write_json_exclusive(output_dir / "cohort_manifest.json", manifest)
    with (output_dir / "_SUCCESS").open("x", encoding="ascii") as handle:
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
