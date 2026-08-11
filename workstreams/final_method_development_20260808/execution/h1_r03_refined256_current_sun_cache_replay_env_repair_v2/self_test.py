#!/usr/bin/env python3
"""Cheap V2 source, repair-provenance, and authorization checks."""

from __future__ import annotations

import argparse
import json
import py_compile
from pathlib import Path

from protocol import ContractError, read_json, require_source_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    args = parser.parse_args()
    source = args.source_dir.resolve()
    require_source_manifest(source, args.source_manifest_sha256)
    config = read_json(source / "CONFIG.json")
    repair = config["repair_provenance"]
    authorization = config["authorization"]
    forbidden = (
        "generation_rerun",
        "planner_rerun",
        "body_rerun",
        "refinement_rerun",
        "chgnet_rerun",
        "training",
        "rl",
        "retry_replacement_repair_filter_rerank",
    )
    if any(authorization[name] is not False for name in forbidden):
        raise ContractError("evaluation-only authorization changed")
    if (
        config["prompt_audit"]["active_seven_line_branch_byte_equal"] is not True
        or config["current_mp"]["wanted_chemsys_count"] != 224
        or config["current_mp"]["existing_resolved_count"] != 132
        or config["current_mp"]["missing_chemsys_count"] != 92
        or len(config["historical"]["repeats"]) != 4
        or repair["failed_before_http_queries"] is not True
        or repair["failed_before_slurm_submission"] is not True
        or repair["one_time_key_destroyed"] is not True
        or repair["only_code_change"]
        != "activate_diff_meets_diff_before_login_node_mp_completion"
    ):
        raise ContractError("frozen replay contract changed")
    for path in source.glob("*.py"):
        py_compile.compile(str(path), doraise=True)
    print(json.dumps({"schema": "h1_r03_refined256_current_sun_cache_replay_self_test_v1", "status": "pass", "source_manifest_sha256": args.source_manifest_sha256}, sort_keys=True))


if __name__ == "__main__":
    main()
