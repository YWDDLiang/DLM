#!/usr/bin/env python3
"""Fail-closed preflight for the CrysLLMGen-native post-refine arrays."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from native_protocol import (
    NATIVE_DENOMINATOR,
    ordered_candidate_rows,
    identity,
    read_json,
    read_jsonl,
    sha256_file,
    validate_frozen_candidate_row,
    write_json_exclusive,
)
from protocol import require_file, require_source_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--body-source-dir", type=Path, required=True)
    parser.add_argument("--native-source-dir", type=Path, required=True)
    parser.add_argument("--native-source-manifest-sha256", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("native preflight must precede Slurm submission")
    if any(os.environ.get(name) for name in ("MP_API_KEY", "PMG_MAPI_KEY", "MAPI_KEY")):
        raise RuntimeError("MP credentials must be absent during native preflight")
    native_source = args.native_source_dir.resolve()
    body_source = args.body_source_dir.resolve()
    require_source_manifest(native_source, args.native_source_manifest_sha256)
    config = read_json(args.config.resolve())
    run_root = args.run_root.resolve()
    if (
        config.get("schema")
        != "h1_p0_plan1200_r03_b3_crysllmgen_native_post1000_config_v2"
        or run_root != Path(str(config.get("run_root"))).resolve()
        or native_source != run_root / "native1000_source"
        or body_source != run_root / "body_source"
        or int(config.get("native_refined_denominator", -1)) != NATIVE_DENOMINATOR
        or config.get("selection")
        != "first_1000_body_successes_by_frozen_candidate_order_per_arm_repeat"
        or any(config.get(key) is not False for key in ("same_plan_retry", "stochastic_replacement", "repair", "filter", "rerank"))
    ):
        raise ValueError("native configuration contract changed")
    body_manifest_sha = sha256_file(body_source / "SOURCE_SHA256.txt")
    import_contract = read_json(run_root / "INPUT_IMPORT_CONTRACT.json")
    import_report = read_json(run_root / "status/v4_input_import_report.json")
    if (
        import_contract.get("schema") != "h1_plan1200_v4_input_import_contract_v1"
        or import_report.get("status") != "complete"
        or import_report.get("body_source_manifest", {}).get("sha256")
        != body_manifest_sha
        or config.get("body_source_manifest_binding")
        != "status/v4_input_import_report.json"
        or not (run_root / "status/v4_input_import_SUCCESS").is_file()
    ):
        raise ValueError("V4 source/import binding changed")

    submission = read_json(run_root / "status/body_submission_record.json")
    if (
        submission.get("status") != "complete"
        or submission.get("separate_arm_arrays") is not True
        or not all(isinstance(submission.get(key), int) for key in ("R03_array_job", "B3_array_job", "assembly_job"))
    ):
        raise ValueError("main V4 body submission evidence is incomplete")

    pools: list[dict[str, Any]] = []
    pool_hashes: list[str] = []
    for repeat in range(3):
        root = run_root / "repeats" / str(repeat) / "crysllmgen_native_candidates"
        rows = ordered_candidate_rows(read_jsonl(root / "candidate_pool.jsonl"))
        for candidate_rank, row in enumerate(rows):
            validate_frozen_candidate_row(
                row, repeat=repeat, candidate_rank=candidate_rank
            )
        manifest = read_json(root / "candidate_pool_manifest.json")
        expected = import_contract["planner"]["repeats"][repeat]
        if (
            not (root / "_SUCCESS").is_file()
            or manifest.get("status") != "complete"
            or int(manifest.get("repeat", -1)) != repeat
            or int(manifest.get("parse_successes", -1)) != len(rows)
            or len(rows) < NATIVE_DENOMINATOR
            or manifest.get("v3_prefix_byte_identity") is not True
            or manifest.get("R03_B3_shared_candidate_pool") is not True
            or sha256_file(root / "candidate_pool.jsonl")
            != expected["candidate_pool_sha256"]
            or sha256_file(root / "candidate_pool_manifest.json")
            != expected["candidate_pool_manifest_sha256"]
        ):
            raise ValueError(f"native candidate pool {repeat} changed")
        pool_hashes.append(sha256_file(root / "candidate_pool.jsonl"))
        pools.append(
            {
                "repeat": repeat,
                "candidate_count": len(rows),
                "candidate_pool": identity(root / "candidate_pool.jsonl"),
                "candidate_pool_manifest": identity(root / "candidate_pool_manifest.json"),
            }
        )
    if len(set(pool_hashes)) != 3:
        raise ValueError("three native candidate pools are not byte-distinct")

    completion_path = run_root / "native_mp_cache/completion_manifest.json"
    completion = read_json(completion_path)
    cache = completion.get("completed_mp_hull_cache") or {}
    cache_path = Path(str(cache.get("path", ""))).resolve()
    if (
        completion.get("status") != "complete_all_wanted_chemsys_resolved"
        or completion.get("native_source_manifest_sha256")
        != import_contract["native_mp_cache"]["origin_native_source_manifest_sha256"]
        or completion.get("api_key_serialized") is not False
        or completion.get("mp_query_inside_slurm") is not False
        or cache.get("all_rows_populated") is not True
        or int(cache.get("rows", -1)) != int(completion.get("wanted_chemsys_count", -2))
        or cache.get("sha256")
        != import_contract["native_mp_cache"]["completed_cache_sha256"]
        or int(cache.get("rows", -1))
        != int(import_contract["native_mp_cache"]["rows"])
        or sha256_file(completion_path)
        != import_contract["native_mp_cache"]["completion_manifest_sha256"]
        or not (run_root / "native_mp_cache/completed_mp_hull_cache.jsonl").is_file()
        or sha256_file(run_root / "native_mp_cache/completed_mp_hull_cache.jsonl")
        != cache.get("sha256")
        or not (run_root / "native_mp_cache/completion_SUCCESS").is_file()
    ):
        raise ValueError("native MP cache is incomplete")

    refiner = config["refiner"]
    refiner_path = require_file(
        refiner["checkpoint"], refiner["checkpoint_sha256"], "model_494"
    )
    upstream = config["upstream_sampling_contract"]
    upstream_path = require_file(
        upstream["source"], upstream["source_sha256"], "upstream crysllmgen_sample.py"
    )
    source_text = upstream_path.read_text(encoding="utf-8")
    required = [
        "idx = 0",
        "while idx < args.num_samples:",
        "idx += len(data_dicts)",
        "dataset = SampleDataset(collected_data)",
        "outputs, _ = diffusion_model.sample(batch, diff_steps=args.diff_steps)",
    ]
    offsets = [source_text.index(statement) for statement in required]
    if offsets != sorted(offsets):
        raise ValueError("upstream CrysLLMGen success-counter/refiner order changed")

    report = {
        "schema": "h1_plan1200_crysllmgen_native_preflight_v2",
        "status": "pass",
        "run_id": config["run_id"],
        "native_source_manifest_sha256": args.native_source_manifest_sha256,
        "body_source_manifest_sha256": body_manifest_sha,
        "input_import_contract": identity(run_root / "INPUT_IMPORT_CONTRACT.json"),
        "input_import_report": identity(
            run_root / "status/v4_input_import_report.json"
        ),
        "main_body_submission": identity(run_root / "status/body_submission_record.json"),
        "main_jobs": {
            "R03_array_job": submission["R03_array_job"],
            "B3_array_job": submission["B3_array_job"],
            "assembly_job": submission["assembly_job"],
        },
        "candidate_pools": pools,
        "native_mp_completion": identity(completion_path),
        "native_mp_cache": identity(
            run_root / "native_mp_cache/completed_mp_hull_cache.jsonl"
        ),
        "origin_completion_cache_path": str(cache_path),
        "refiner": identity(refiner_path),
        "upstream_crysllmgen_sample": identity(upstream_path),
        "upstream_control_flow_offsets": dict(zip(required, offsets)),
        "upstream_semantics": "1000 process_one-success candidates all enter diffusion",
        "native_refined_denominator_per_arm_repeat": NATIVE_DENOMINATOR,
        "separate_arm_arrays_required": True,
        "gpu_partition_only": True,
        "mp_credentials_present": False,
        "retry_replacement_repair_filter_rerank": False,
        "automatic_training": False,
        "automatic_rl": False,
    }
    write_json_exclusive(args.output.resolve(), report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
