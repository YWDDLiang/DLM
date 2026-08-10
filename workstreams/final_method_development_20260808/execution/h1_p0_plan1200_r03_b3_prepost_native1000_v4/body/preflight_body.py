#!/usr/bin/env python3
"""Fail-closed preflight for both three-repeat body arrays."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from protocol import (
    DENOMINATOR,
    PAIRED_SEED_NAMESPACE,
    paired_seed,
    ordered_rows,
    read_json,
    read_jsonl,
    require_file,
    require_source_manifest,
    sha256_file,
    validate_config,
    validate_frozen_cohort_row,
    write_json_exclusive,
)


def identity(path: Path) -> dict[str, Any]:
    location = path.resolve()
    if not location.is_file():
        raise FileNotFoundError(location)
    return {
        "path": str(location),
        "bytes": location.stat().st_size,
        "sha256": sha256_file(location),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("body preflight must precede Slurm submission")
    if any(os.environ.get(name) for name in ("MP_API_KEY", "PMG_MAPI_KEY", "MAPI_KEY")):
        raise RuntimeError("MP credentials must be absent during body preflight")
    source = args.source_dir.resolve()
    require_source_manifest(source, args.source_manifest_sha256)
    config = read_json(args.config.resolve())
    validate_config(config)
    run_root = args.run_root.resolve()
    if run_root != Path(config["run_root"]).resolve() or source != run_root / "body_source":
        raise ValueError("body source/run root identity changed")
    if args.output.resolve() != run_root / "status/body_preflight_report.json":
        raise ValueError("body preflight output identity changed")

    import_contract = read_json(run_root / "INPUT_IMPORT_CONTRACT.json")
    import_report = read_json(run_root / "status/v4_input_import_report.json")
    if (
        import_contract.get("schema") != "h1_plan1200_v4_input_import_contract_v1"
        or import_report.get("status") != "complete"
        or import_report.get("contract_sha256")
        != sha256_file(run_root / "INPUT_IMPORT_CONTRACT.json")
        or not (run_root / "status/v4_input_import_SUCCESS").is_file()
    ):
        raise ValueError("V4 imported-input provenance is incomplete")

    planner = read_json(run_root / "planner_terminal_report.json")
    if (
        planner.get("status") != "complete"
        or planner.get("three_independent_plan_batches") is not True
        or int(planner.get("repeat_count", -1)) != 3
        or int(planner.get("frozen_cohort_attempts_per_repeat", -1)) != DENOMINATOR
        or not (run_root / "status/planner_assembly_SUCCESS").is_file()
        or sha256_file(run_root / "planner_terminal_report.json")
        != import_contract["planner"]["terminal_report_sha256"]
    ):
        raise ValueError("planner terminal evidence is incomplete")

    runtime = config["runtime"]
    manifests = {
        "v4": identity(
            require_source_manifest(
                runtime["v4_source"], runtime["v4_source_manifest_sha256"]
            )
        ),
        "r03d": identity(
            require_source_manifest(runtime["r03d"], runtime["r03d_source_manifest_sha256"])
        ),
        "r03e": identity(
            require_source_manifest(runtime["r03e"], runtime["r03e_source_manifest_sha256"])
        ),
    }

    direct = config["direct"]
    direct_runner = require_file(direct["runner"], direct["runner_sha256"], "Direct runner")
    compute = require_file(
        direct["upstream_compute_metrics"],
        direct["upstream_compute_metrics_sha256"],
        "upstream compute_metrics",
    )
    compute_text = compute.read_text(encoding="utf-8")
    gcd_statement = "counts = counts / np.gcd.reduce(counts)"
    validity_statement = "self.comp_valid = smact_validity(self.elems, self.comps)"
    if (
        compute_text.count(gcd_statement) != 1
        or compute_text.count(validity_statement) != 1
        or compute_text.index(gcd_statement) >= compute_text.index(validity_statement)
    ):
        raise ValueError("GCD-before-comp_valid implementation changed")

    body = config["body"]
    models: dict[str, Any] = {}
    for arm, specification in body["models"].items():
        checkpoint = Path(specification["checkpoint"]).resolve()
        adapter = require_file(
            checkpoint / body["adapter_file"], specification["adapter_sha256"], f"{arm} adapter"
        )
        if adapter.stat().st_size != int(body["adapter_expected_bytes"]):
            raise ValueError(f"{arm} adapter byte count changed")
        tokenizer = require_file(
            checkpoint / "tokenizer.json", body["tokenizer_json_sha256"], f"{arm} tokenizer"
        )
        tokenizer_config = require_file(
            checkpoint / "tokenizer_config.json",
            body["tokenizer_config_sha256"],
            f"{arm} tokenizer config",
        )
        models[arm] = {
            "adapter": identity(adapter),
            "tokenizer": identity(tokenizer),
            "tokenizer_config": identity(tokenizer_config),
        }
    refiner = require_file(
        config["refiner"]["checkpoint"],
        config["refiner"]["checkpoint_sha256"],
        "model_494",
    )

    cohorts: list[dict[str, Any]] = []
    cohort_hashes: list[str] = []
    for repeat in range(3):
        cohort_root = run_root / "repeats" / str(repeat) / "cohort"
        path = cohort_root / "cohort1000.jsonl"
        rows = ordered_rows(read_jsonl(path), ordinal_field="cohort_ordinal")
        for ordinal, row in enumerate(rows):
            validate_frozen_cohort_row(row, repeat=repeat, ordinal=ordinal)
        expected_cohort = import_contract["planner"]["repeats"][repeat]
        if (
            sha256_file(path) != expected_cohort["cohort1000_sha256"]
            or sha256_file(cohort_root / "cohort_manifest.json")
            != expected_cohort["cohort_manifest_sha256"]
        ):
            raise ValueError(f"repeat {repeat} imported cohort identity changed")
        for ordinal in range(DENOMINATOR):
            if paired_seed(repeat, ordinal, "body") == paired_seed(repeat, ordinal, "refiner"):
                raise ValueError("body/refiner channels unexpectedly share a seed")
        cohort_hashes.append(sha256_file(path))
        cohorts.append(
            {
                "repeat": repeat,
                "cohort1000": identity(path),
                "manifest": identity(cohort_root / "cohort_manifest.json"),
                "model_visible_prompt": "historical_r5c_plan_state_json_exact_length",
                "raw_rich_seven_line_forwarded": False,
                "canonical_charge_bucket_visible": True,
            }
        )
    if len(set(cohort_hashes)) != 3:
        raise ValueError("planner cohorts are not byte-distinct")

    sun = config["sun"]
    completion_path = run_root / sun["completion_manifest"]
    completion = read_json(completion_path)
    completed = completion.get("completed_mp_hull_cache") or {}
    cache_path = Path(str(completed.get("path", ""))).resolve()
    if (
        completion.get("status") != "complete_all_wanted_chemsys_resolved"
        or completion.get("source_manifest_sha256")
        != import_contract["main_mp_cache"]["origin_body_source_manifest_sha256"]
        or completion.get("api_key_serialized") is not False
        or completion.get("mp_query_inside_slurm") is not False
        or completion.get("sample_retry_or_replacement_used") is not False
        or completed.get("all_rows_populated") is not True
        or int(completed.get("rows", -1)) != int(completion.get("wanted_chemsys_count", -2))
        or completed.get("sha256")
        != import_contract["main_mp_cache"]["completed_cache_sha256"]
        or int(completed.get("rows", -1))
        != int(import_contract["main_mp_cache"]["rows"])
        or sha256_file(completion_path)
        != import_contract["main_mp_cache"]["completion_manifest_sha256"]
        or not (run_root / sun["completed_cache"]).is_file()
        or sha256_file(run_root / sun["completed_cache"])
        != completed.get("sha256")
        or not (run_root / sun["completion_success_marker"]).is_file()
    ):
        raise ValueError("completed planner-union MP cache contract changed")

    report = {
        "schema": "h1_plan1200_body_preflight_v4",
        "status": "pass",
        "run_id": config["run_id"],
        "paired_seed_namespace": PAIRED_SEED_NAMESPACE,
        "source_manifest_sha256": args.source_manifest_sha256,
        "planner_terminal": identity(run_root / "planner_terminal_report.json"),
        "input_import_contract": identity(run_root / "INPUT_IMPORT_CONTRACT.json"),
        "input_import_report": identity(
            run_root / "status/v4_input_import_report.json"
        ),
        "cohorts": cohorts,
        "models": models,
        "refiner": identity(refiner),
        "runtime_manifests": manifests,
        "direct": {
            "runner": identity(direct_runner),
            "upstream_compute_metrics": identity(compute),
            "required_order": "gcd_then_smact_validity",
            "gcd_statement_offset": compute_text.index(gcd_statement),
            "validity_statement_offset": compute_text.index(validity_statement),
        },
        "sun": {
            "completion_manifest": identity(completion_path),
            "completed_cache": identity(run_root / sun["completed_cache"]),
            "origin_completion_cache_path": str(cache_path),
            "wanted_chemsys_count": int(completion["wanted_chemsys_count"]),
            "missing_chemsys_resolved": int(completion["missing_chemsys_count"]),
            "mp_query_inside_slurm": False,
        },
        "separate_arm_arrays_required": True,
        "attempts_per_arm_repeat": DENOMINATOR,
        "process_repeats": 3,
        "stages": ["pre_model494", "post_model494"],
        "headline_sun_denominator": "reconstructed_structures_exact_legacy",
        "secondary_sun_denominator": "all_1000_attempts",
        "mp_credentials_present": False,
        "retry_replacement_repair_filter_rerank": False,
        "automatic_training": False,
        "automatic_promotion": False,
        "automatic_rl": False,
    }
    write_json_exclusive(args.output.resolve(), report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
