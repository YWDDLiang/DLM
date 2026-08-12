#!/usr/bin/env python3
"""Audit all immutable post-refine generation and evaluator inputs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from protocol import (
    ContractError,
    DENOMINATOR,
    identity,
    load_upstream_cells,
    read_json,
    read_jsonl,
    require_file,
    require_source_manifest,
    sha256_file,
    write_json_exclusive,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    source = args.source_dir.resolve()
    require_source_manifest(source, args.source_manifest_sha256)
    config = read_json(source / "CONFIG.json")
    if (
        config.get("schema")
        != "h1a2_retrained_postonly_sun256_official_recovery_config_v1"
    ):
        raise ContractError("unexpected post-only config schema")
    upstream_spec = config["upstream_generation"]
    if (
        upstream_spec.get("evaluated_stage") != "post_model494_only"
        or upstream_spec.get("pre_refine_role")
        != "intermediate_only_not_scored"
    ):
        raise ContractError("pre-refine evaluation is forbidden")
    upstream = Path(upstream_spec["run_root"]).resolve()
    require_file(
        upstream / "source/SOURCE_SHA256.txt",
        upstream_spec["source_manifest_sha256"],
        "upstream source manifest",
    )
    terminal, cells = load_upstream_cells(config)
    terminal_path = upstream / upstream_spec["terminal_report"]
    registry_path = upstream / upstream_spec["evaluation_registry"]
    upstream_cell_reports: list[dict[str, Any]] = []
    for cell in cells:
        generation_path = Path(cell["generation_jsonl"]).resolve()
        direct_report_path = Path(cell["direct_report"]).resolve()
        if (
            not generation_path.is_relative_to(upstream)
            or not direct_report_path.is_relative_to(upstream)
        ):
            raise ContractError("upstream cell path escapes generation run")
        generation_report_path = generation_path.parent / "generation_report.json"
        direct_attempts_path = direct_report_path.parent / "attempt_metrics.jsonl"
        rows = read_jsonl(generation_path)
        generation_report = read_json(generation_report_path)
        direct_report = read_json(direct_report_path)
        direct_rows = read_jsonl(direct_attempts_path)
        statuses = {str(row.get("status")) for row in rows}
        if (
            len(rows) != DENOMINATOR
            or [int(row.get("ordinal", -1)) for row in rows]
            != list(range(DENOMINATOR))
            or len({str(row.get("attempt_id")) for row in rows}) != DENOMINATOR
            or not statuses.issubset({"succeeded", "failed"})
            or any(row.get("retry_or_replacement_used") is not False for row in rows)
            or any(
                row.get("status") == "succeeded"
                and not isinstance(row.get("structure"), dict)
                for row in rows
            )
            or int(generation_report.get("attempts", -1)) != DENOMINATOR
            or generation_report.get("stage") != "post_model494"
            or generation_report.get("retry_replacement_repair_filter_rerank")
            is not False
            or int(direct_report.get("attempts", -1)) != DENOMINATOR
            or len(direct_rows) != DENOMINATOR
            or sha256_file(generation_path) != cell["generation_sha256"]
            or sha256_file(direct_report_path) != cell["direct_report_sha256"]
        ):
            raise ContractError(f"upstream cell contract changed: {cell['cell_id']}")
        upstream_cell_reports.append(
            {
                **{
                    key: cell.get(key)
                    for key in (
                        "cell_index",
                        "cell_id",
                        "panel",
                        "cohort_id",
                        "cohort_index",
                        "process_repeat",
                        "stage",
                        "body",
                        "schedule",
                    )
                },
                "generation": identity(generation_path),
                "generation_report": identity(generation_report_path),
                "direct_report": identity(direct_report_path),
                "direct_attempts": identity(direct_attempts_path),
                "generation_succeeded": sum(
                    row.get("status") == "succeeded" for row in rows
                ),
            }
        )

    preliminary = config["preliminary_sun"]
    runtime = Path(preliminary["runtime_root"]).resolve()
    bundle = Path(preliminary["source_bundle_root"]).resolve()
    assets = {
        "upstream_terminal": identity(terminal_path),
        "upstream_registry": identity(registry_path),
        "upstream_marker": identity(upstream / upstream_spec["required_marker"]),
        "r03e_source_manifest": identity(
            require_file(
                bundle / "SOURCE_SHA256.txt",
                preliminary["source_bundle_manifest_sha256"],
                "R03E source manifest",
            )
        ),
        "runner": identity(
            require_file(
                runtime / "scripts/a800/run_crysllmgen_a100_sun.py",
                preliminary["runner_sha256"],
                "R03E S.U.N. runner",
            )
        ),
        "a100_sun_module": identity(
            require_file(
                runtime / "crystal_dlm/wqcodiff/crysllmgen/a100_sun.py",
                preliminary["a100_sun_module_sha256"],
                "R03E a100_sun module",
            )
        ),
        "legacy_mp_cache": identity(
            require_file(
                Path(preliminary["legacy_mp_cache"]),
                preliminary["legacy_mp_cache_sha256"],
                "preliminary offline MP cache",
            )
        ),
        "chgnet_relax_cache": identity(
            require_file(
                Path(preliminary["chgnet_relax_cache"]),
                preliminary["chgnet_relax_cache_sha256"],
                "CHGNet relax cache",
            )
        ),
        "chgnet_model": identity(
            require_file(
                Path(preliminary["chgnet_model_asset"]),
                preliminary["chgnet_model_sha256"],
                "CHGNet model asset",
            )
        ),
        "chgnet_runtime_checkpoint": identity(
            require_file(
                Path(preliminary["chgnet_runtime_checkpoint"]),
                preliminary["chgnet_model_sha256"],
                "CHGNet runtime checkpoint",
            )
        ),
    }
    official = config["official_mp"]
    base = Path(official["base_cache_root"]).resolve()
    assets.update(
        {
            "official_base_manifest": identity(
                require_file(
                    base / "completion_manifest.json",
                    official["base_completion_manifest_sha256"],
                    "official base cache manifest",
                )
            ),
            "official_base_slim": identity(
                require_file(
                    base / "official_slim_cache.jsonl",
                    official["base_slim_cache_sha256"],
                    "official base slim cache",
                )
            ),
            "official_base_unresolved": identity(
                require_file(
                    base / "unresolved_chemsys.jsonl",
                    official["base_unresolved_sha256"],
                    "official base unresolved ledger",
                )
            ),
        }
    )

    benchmark = config["historical_official_refined256_benchmark"]
    benchmark_root = Path(benchmark["run_root"]).resolve()
    benchmark_terminal_path = require_file(
        benchmark_root / "terminal_report.json",
        benchmark["terminal_report_sha256"],
        "historical clean-official refined256 terminal",
    )
    benchmark_results_path = require_file(
        benchmark_root / "RESULTS_COMPLETE.md",
        benchmark["results_markdown_sha256"],
        "historical clean-official refined256 markdown",
    )
    benchmark_terminal = read_json(benchmark_terminal_path)
    historical_cells = sorted(
        [
            row
            for row in benchmark_terminal.get("cell_reports", [])
            if (row.get("cell") or {}).get("panel") == benchmark["source_panel"]
            and (row.get("cell") or {}).get("arm") == "R03"
            and (row.get("cell") or {}).get("stage") == "post_model494"
        ],
        key=lambda row: int(row["cell"]["repeat"]),
    )
    if (
        len(historical_cells) != 4
        or [int(row["clean"]["strict_full_sun"]) for row in historical_cells]
        != benchmark["strict_fixed_all_counts"]
        or [int(row["clean"]["meta_full_sun"]) for row in historical_cells]
        != benchmark["meta_fixed_all_counts"]
    ):
        raise ContractError("historical clean-official refined256 benchmark changed")
    assets.update(
        {
            "historical_official_refined256_terminal": identity(
                benchmark_terminal_path
            ),
            "historical_official_refined256_markdown": identity(
                benchmark_results_path
            ),
        }
    )

    reference = config["historical_h1a2_planner_reference"]
    reference_path = require_file(
        Path(reference["terminal_report"]),
        reference["terminal_report_sha256"],
        "historical H1-A2 planner terminal",
    )
    reference_terminal = read_json(reference_path)
    matches = [
        row
        for row in reference_terminal.get("planner_distributions", [])
        if row.get("cohort_id") == reference["cohort_id"]
    ]
    if (
        len(matches) != 1
        or matches[0].get("cohort256_sha256") != reference["cohort256_sha256"]
    ):
        raise ContractError("historical H1-A2 planner reference changed")
    assets["historical_h1a2_planner_reference"] = identity(reference_path)

    result = {
        "schema": "h1a2_retrained_postonly_sun256_preflight_v1",
        "status": "pass",
        "source_manifest_sha256": args.source_manifest_sha256,
        "evaluated_stage": "post_model494_only",
        "pre_refine_evaluated": False,
        "cells": upstream_cell_reports,
        "assets": assets,
        "upstream_terminal_identity": identity(terminal_path),
        "no_generation_or_refinement_rerun": True,
        "preliminary_hull_is_not_scientific_headline": True,
    }
    if args.output is not None:
        write_json_exclusive(args.output.resolve(), result)
    print({"preflight": "PASS", "cells": len(upstream_cell_reports)})


if __name__ == "__main__":
    main()
