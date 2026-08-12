#!/usr/bin/env python3
"""Validate fresh and historical-topology world-size-2 planner cohorts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from analyze_planner_distributions import (
    build_audit,
    compare_summaries,
    summarize_rows,
)
from protocol import (
    DENOMINATOR,
    PLANNER_RAW_ATTEMPTS,
    canonical_sha256,
    read_json,
    read_jsonl,
    require_file,
    require_source_manifest,
    sha256_file,
    validate_config,
    write_json_exclusive,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    source = args.source_dir.resolve()
    require_source_manifest(source, args.source_manifest_sha256)
    config = read_json(source / "CONFIG.json")
    validate_config(config)
    specs = list(config["planner"]["cohorts"])
    if len(specs) != 4:
        raise ValueError("exactly four planner cohorts are required")
    if [str(spec["cohort_id"]) for spec in specs] != list(config["downstream_cohorts"]):
        raise ValueError("planner/downstream cohort order changed")
    seeds = [int(spec["seed"]) for spec in specs]
    if len(set(seeds)) != 4:
        raise ValueError("planner seeds are not independent")

    reports: list[dict[str, Any]] = []
    raw_hashes: list[str] = []
    frozen_hashes: list[str] = []
    for index, spec in enumerate(specs):
        cohort_id = str(spec["cohort_id"])
        marker = run_root / "status" / f"planner_{index}_SUCCESS"
        exit_code = run_root / "status" / f"planner_{index}_exit_code.txt"
        report_path = run_root / "planner" / cohort_id / "frozen" / "cohort_report.json"
        cohort_path = run_root / "planner" / cohort_id / "frozen" / "cohort256.jsonl"
        if not marker.is_file() or not exit_code.is_file() or exit_code.read_text().strip() != "0":
            raise RuntimeError(f"planner cohort is not terminal-success: {cohort_id}")
        if not report_path.is_file() or not cohort_path.is_file():
            raise FileNotFoundError(f"planner evidence missing: {cohort_id}")
        report = read_json(report_path)
        if (
            report.get("cohort_id") != cohort_id
            or int(report.get("seed", -1)) != int(spec["seed"])
            or int(report.get("world_size", -1)) != 2
            or int(report.get("batch_size_per_rank", -1)) != 4
            or int(report.get("num_samples", -1)) != PLANNER_RAW_ATTEMPTS
            or int(report.get("raw_attempts", -1)) != PLANNER_RAW_ATTEMPTS
            or int(report.get("raw_parsed", -1))
            + int(report.get("raw_planner_failed", -1))
            != PLANNER_RAW_ATTEMPTS
            or int(report.get("frozen_attempts", -1)) != DENOMINATOR
            or int(report.get("attempts", -1)) != DENOMINATOR
            or int(report.get("parsed", -1)) + int(report.get("planner_failed", -1)) != DENOMINATOR
            or report.get("selection")
            != "first_256_raw_records_in_merged_file_order_with_failures_preserved"
            or report.get("retry_replacement_repair_filter_rerank") is not False
            or report.get("checkpoint_sha256")
            != config["training_upstream"]["adapter_sha256"]
            or report.get("checkpoint_config_sha256")
            != config["training_upstream"]["adapter_config_sha256"]
            or report.get("effective_rank_seeds")
            != [int(spec["seed"]), int(spec["seed"]) + 1]
            or report.get("selected_rank_counts") != {"0": DENOMINATOR, "1": 0}
            or report.get("cohort256_sha256") != sha256_file(cohort_path)
        ):
            raise ValueError(f"planner contract changed: {cohort_id}")
        raw_hashes.append(str(report["raw_generations_sha256"]))
        frozen_hashes.append(str(report["cohort256_sha256"]))
        reports.append(report)
    if len(set(raw_hashes)) != 4 or len(set(frozen_hashes)) != 4:
        raise ValueError("fresh planner cohorts are not distinct")

    topology_spec = config["topology_match"]["planner"]
    topology_index = 4
    topology_id = str(topology_spec["cohort_id"])
    topology_marker = run_root / "status" / f"planner_{topology_index}_SUCCESS"
    topology_exit = run_root / "status" / f"planner_{topology_index}_exit_code.txt"
    topology_report_path = (
        run_root / "planner" / topology_id / "frozen" / "cohort_report.json"
    )
    topology_path = (
        run_root / "planner" / topology_id / "frozen" / "cohort256.jsonl"
    )
    if (
        not topology_marker.is_file()
        or not topology_exit.is_file()
        or topology_exit.read_text().strip() != "0"
    ):
        raise RuntimeError("topology-match planner cohort is not terminal-success")
    if not topology_report_path.is_file() or not topology_path.is_file():
        raise FileNotFoundError("topology-match planner evidence missing")
    topology_report = read_json(topology_report_path)
    if (
        topology_report.get("cohort_id") != topology_id
        or int(topology_report.get("seed", -1)) != int(topology_spec["seed"])
        or int(topology_report.get("world_size", -1)) != 2
        or int(topology_report.get("batch_size_per_rank", -1)) != 4
        or int(topology_report.get("num_samples", -1)) != PLANNER_RAW_ATTEMPTS
        or int(topology_report.get("raw_attempts", -1)) != PLANNER_RAW_ATTEMPTS
        or int(topology_report.get("raw_parsed", -1))
        + int(topology_report.get("raw_planner_failed", -1))
        != PLANNER_RAW_ATTEMPTS
        or int(topology_report.get("frozen_attempts", -1)) != DENOMINATOR
        or int(topology_report.get("attempts", -1)) != DENOMINATOR
        or int(topology_report.get("parsed", -1))
        + int(topology_report.get("planner_failed", -1))
        != DENOMINATOR
        or topology_report.get("selection")
        != "first_256_raw_records_in_merged_file_order_with_failures_preserved"
        or topology_report.get("retry_replacement_repair_filter_rerank") is not False
        or topology_report.get("checkpoint_sha256")
        != config["training_upstream"]["adapter_sha256"]
        or topology_report.get("checkpoint_config_sha256")
        != config["training_upstream"]["adapter_config_sha256"]
        or topology_report.get("effective_rank_seeds") != [17, 18]
        or topology_report.get("selected_rank_counts")
        != {"0": DENOMINATOR, "1": 0}
        or topology_report.get("cohort256_sha256") != sha256_file(topology_path)
    ):
        raise ValueError("topology-match planner contract changed")
    if (
        str(topology_report["raw_generations_sha256"]) in set(raw_hashes)
        or str(topology_report["cohort256_sha256"]) in set(frozen_hashes)
    ):
        raise ValueError("topology-match cohort unexpectedly duplicates a fresh cohort")

    deep_audit_path = run_root / "planner_distribution_deep_audit.json"
    deep_audit = build_audit(
        run_root=run_root,
        config=config,
        output_path=deep_audit_path,
    )

    reference_spec = config["historical_planner_reference"]
    reference_path = require_file(
        reference_spec["cohort256"],
        reference_spec["cohort256_sha256"],
        "historical H1-A2 seed17/world2 cohort",
    )
    reference_rows = read_jsonl(reference_path)
    topology_rows = read_jsonl(topology_path)
    if len(reference_rows) != DENOMINATOR or len(topology_rows) != DENOMINATOR:
        raise ValueError("topology-match planner denominator changed")
    reference_summary = summarize_rows(
        reference_rows, cohort_id=str(reference_spec["cohort_id"])
    )
    topology_summary = summarize_rows(topology_rows, cohort_id=topology_id)
    topology_audit_path = run_root / "planner_topology_match_audit.json"
    topology_audit = {
        "schema": "h1a2_retrained_seed17_world2_topology_planner_audit_v1",
        "status": "complete",
        "historical_identity": {
            "path": str(reference_path),
            "bytes": reference_path.stat().st_size,
            "sha256": sha256_file(reference_path),
        },
        "retrained_identity": {
            "path": str(topology_path),
            "bytes": topology_path.stat().st_size,
            "sha256": sha256_file(topology_path),
        },
        "byte_identical": reference_path.read_bytes() == topology_path.read_bytes(),
        "historical_summary": reference_summary,
        "retrained_summary": topology_summary,
        "comparison": compare_summaries(reference_summary, topology_summary),
    }
    topology_audit["audit_payload_sha256"] = canonical_sha256(topology_audit)
    write_json_exclusive(topology_audit_path, topology_audit)

    terminal = {
        "schema": "h1a2_retrained_world2_planner_terminal_v1",
        "status": "complete",
        "ok": True,
        "cohort_count": 5,
        "fresh_cohort_count": 4,
        "topology_match_cohort_count": 1,
        "denominator_per_cohort": DENOMINATOR,
        "raw_planner_attempts_per_cohort": PLANNER_RAW_ATTEMPTS,
        "planner_selection": "first_256_raw_records_in_merged_file_order_with_failures_preserved",
        "world_size": 2,
        "batch_size_per_rank": 4,
        "seeds": seeds,
        "seeds_sha256": canonical_sha256(seeds),
        "all_raw_cohorts_distinct": True,
        "all_frozen_cohorts_distinct": True,
        "deep_distribution_audit": {
            "path": str(deep_audit_path),
            "bytes": deep_audit_path.stat().st_size,
            "sha256": sha256_file(deep_audit_path),
            "audit_payload_sha256": deep_audit["audit_payload_sha256"],
            "reference_cohort_id": deep_audit["reference"]["summary"][
                "cohort_id"
            ],
            "fresh_cohort_count": len(
                deep_audit["fresh_cohort_summaries"]
            ),
        },
        "topology_match_audit": {
            "path": str(topology_audit_path),
            "bytes": topology_audit_path.stat().st_size,
            "sha256": sha256_file(topology_audit_path),
            "audit_payload_sha256": topology_audit["audit_payload_sha256"],
            "historical_cohort_id": reference_summary["cohort_id"],
            "retrained_cohort_id": topology_summary["cohort_id"],
            "byte_identical": topology_audit["byte_identical"],
        },
        "reports": reports,
        "topology_match_report": topology_report,
        "source_manifest_sha256": args.source_manifest_sha256,
    }
    write_json_exclusive(run_root / "planner_terminal_report.json", terminal)
    (run_root / "status" / "planner_assembly_SUCCESS").touch(exist_ok=False)
    print(json.dumps(terminal, sort_keys=True))


if __name__ == "__main__":
    main()
