#!/usr/bin/env python3
"""Focused identity preflight for the four-cell Direct/S.U.N. execution."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from protocol import (
    DENOMINATOR,
    ordered_rows,
    read_json,
    read_jsonl,
    require_file,
    require_source_manifest,
    sha256_file,
    validate_config,
    write_json_exclusive,
)


def _identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _verified_identity(path: Path, sha256: str) -> dict[str, Any]:
    """Record a path already verified by ``require_file`` without rehashing."""

    resolved = path.resolve()
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = args.source_dir.resolve()
    require_source_manifest(source, args.source_manifest_sha256)
    config = read_json(args.config.resolve())
    validate_config(config)
    prepared_root = args.prepared_root.resolve()
    final_root = Path(config["run_root"]).resolve()
    if (
        prepared_root.parent != final_root.parent
        or not prepared_root.name.startswith(f".{final_root.name}.preparing.")
        or final_root.exists()
    ):
        raise ValueError("prepared/final run identity changed")

    dependency_reports: dict[str, Any] = {}
    for name, specification in config["source_dependencies"].items():
        manifest = require_source_manifest(
            specification["source_dir"],
            specification["source_manifest_sha256"],
        )
        dependency_reports[name] = _identity(manifest)

    gcd_contract = config["direct"]["gcd_before_comp_valid"]
    direct_runner = require_file(
        gcd_contract["runner"],
        gcd_contract["runner_sha256"],
        "R03E Direct metric runner",
    )
    compute_metrics = require_file(
        gcd_contract["upstream_compute_metrics"],
        gcd_contract["upstream_compute_metrics_sha256"],
        "R03E upstream compute_metrics.py",
    )
    compute_source = compute_metrics.read_text(encoding="utf-8")
    gcd_statement = gcd_contract["gcd_statement"]
    validity_statement = gcd_contract["validity_statement"]
    if (
        compute_source.count(gcd_statement) != 1
        or compute_source.count(validity_statement) != 1
        or compute_source.index(gcd_statement)
        >= compute_source.index(validity_statement)
    ):
        raise ValueError("R03E GCD-before-comp_valid source order changed")
    gcd_report = {
        "status": "pass",
        "required_order": "gcd_then_smact_validity",
        "runner": _verified_identity(
            direct_runner, gcd_contract["runner_sha256"]
        ),
        "upstream_compute_metrics": _verified_identity(
            compute_metrics,
            gcd_contract["upstream_compute_metrics_sha256"],
        ),
        "gcd_statement_offset": compute_source.index(gcd_statement),
        "validity_statement_offset": compute_source.index(validity_statement),
    }

    planner_reports: dict[str, Any] = {}
    for name, specification in config["planner_sources"].items():
        path = require_file(
            specification["raw_generations"],
            specification["raw_generations_sha256"],
            f"{name} raw256",
        )
        rows = ordered_rows(read_jsonl(path), ordinal_field="sample_idx")
        parsed = sum(row.get("parsed") is True for row in rows)
        if parsed != int(specification["expected_parsed"]):
            raise ValueError(f"{name} parsed count changed")
        planner_reports[name] = {
            **_identity(path),
            "rows": len(rows),
            "parsed": parsed,
        }

    ledger_spec = config["attempt_ledger"]
    ledger_path = require_file(
        ledger_spec["path"], ledger_spec["sha256"], "common H1 seed ledger"
    )
    ledger = ordered_rows(read_jsonl(ledger_path), ordinal_field="sample_idx")
    if any(
        int(row.get("ordinal", -1)) != ordinal
        or not isinstance(row.get(ledger_spec["body_seed_field"]), int)
        or not isinstance(row.get(ledger_spec["refiner_seed_field"]), int)
        for ordinal, row in enumerate(ledger)
    ):
        raise ValueError("common H1 seed ledger fields changed")

    body = config["body"]
    model_reports: dict[str, Any] = {}
    for name, specification in body["models"].items():
        checkpoint = Path(specification["checkpoint"]).resolve()
        adapter = require_file(
            checkpoint / body["adapter_file"],
            specification["adapter_sha256"],
            f"{name} body adapter",
        )
        if adapter.stat().st_size != int(body["adapter_expected_bytes"]):
            raise ValueError(f"{name} body adapter byte count changed")
        tokenizer = require_file(
            checkpoint / "tokenizer.json",
            body["tokenizer_json_sha256"],
            f"{name} tokenizer.json",
        )
        tokenizer_config = require_file(
            checkpoint / "tokenizer_config.json",
            body["tokenizer_config_sha256"],
            f"{name} tokenizer_config.json",
        )
        model_reports[name] = {
            "checkpoint": str(checkpoint),
            "adapter": _verified_identity(
                adapter, specification["adapter_sha256"]
            ),
            "tokenizer_json": _verified_identity(
                tokenizer, body["tokenizer_json_sha256"]
            ),
            "tokenizer_config_json": _verified_identity(
                tokenizer_config, body["tokenizer_config_sha256"]
            ),
        }
        terminal = specification.get("training_terminal_report")
        if terminal:
            verified_terminal = require_file(
                terminal,
                specification["training_terminal_report_sha256"],
                f"{name} terminal training report",
            )
            model_reports[name]["training_terminal_report"] = _verified_identity(
                verified_terminal,
                specification["training_terminal_report_sha256"],
            )

    refiner = config["refiner"]
    refiner_path = require_file(
        refiner["checkpoint"],
        refiner["checkpoint_sha256"],
        "model_494 refiner",
    )
    project = final_root.parents[1]
    sun = config["sun"]
    required_relative = [
        config["direct"]["gt_csv"],
        sun["eval_sun_py"],
        sun["eval_sun_resumable_py"],
        sun["train_csv"],
        sun["training_index_cache"],
        sun["base_mp_hull_cache"],
        sun["chgnet_relax_cache"],
    ]
    required_absolute = [
        sun["chgnet_model_asset"],
        sun["chgnet_runtime_checkpoint"],
        sun["completion_module"],
        sun["r03e_sun_runner"],
        sun["r03e_a100_sun_module"],
    ]
    required_assets = [
        _identity(project / relative) for relative in required_relative
    ] + [_identity(Path(path)) for path in required_absolute]
    base_cache = require_file(
        project / sun["base_mp_hull_cache"],
        sun["base_mp_hull_cache_sha256"],
        "frozen base MP hull cache",
    )
    if base_cache.stat().st_size != int(sun["base_mp_hull_cache_bytes"]):
        raise ValueError("frozen base MP hull cache byte count changed")
    require_file(
        sun["completion_module"],
        sun["completion_module_sha256"],
        "frozen R03F MP completion module",
    )
    require_file(
        sun["r03e_sun_runner"],
        sun["r03e_sun_runner_sha256"],
        "byte-frozen R03E S.U.N. runner",
    )
    require_file(
        sun["r03e_a100_sun_module"],
        sun["r03e_a100_sun_module_sha256"],
        "byte-frozen R03E a100_sun module",
    )

    completion_manifest_path = prepared_root / sun["completion_manifest"]
    completion_manifest = read_json(completion_manifest_path)
    completed_spec = completion_manifest.get("completed_mp_hull_cache") or {}
    completed_cache = require_file(
        prepared_root / sun["completed_mp_hull_cache"],
        completed_spec.get("sha256", ""),
        "completed Planner-union MP hull cache",
    )
    completed_sha_file = prepared_root / sun[
        "completed_mp_hull_cache_sha256_file"
    ]
    expected_sha_line = (
        f"{completed_spec.get('sha256', '')}  {completed_cache.name}\n"
    )
    if (
        not completed_sha_file.is_file()
        or completed_sha_file.read_text(encoding="ascii") != expected_sha_line
    ):
        raise ValueError("completed MP cache SHA file changed")
    completion_success = prepared_root / sun["completion_success_marker"]
    if (
        not completion_success.is_file()
        or completion_manifest.get("schema")
        != "h1_ef_fourcell_mp_cache_completion_manifest_v1"
        or completion_manifest.get("status")
        != "complete_all_missing_resolved"
        or completion_manifest.get("run_id") != config["run_id"]
        or completion_manifest.get("source_manifest_sha256")
        != args.source_manifest_sha256
        or int(completion_manifest.get("wanted_chemsys_count", -1))
        != int(sun["wanted_chemsys_count"])
        or completion_manifest.get("wanted_chemsys_sha256")
        != sun["wanted_chemsys_sha256"]
        or int(completion_manifest.get("missing_chemsys_count", -1))
        != int(sun["missing_chemsys_count"])
        or completion_manifest.get("missing_chemsys_sha256")
        != sun["missing_chemsys_sha256"]
        or int(completion_manifest.get("base_mp_hull_cache_rows", -1))
        != int(sun["base_mp_hull_cache_rows"])
        or int(
            completion_manifest.get(
                "base_mp_hull_cache_distinct_chemsys", -1
            )
        )
        != int(sun["base_mp_hull_cache_distinct_chemsys"])
        or completion_manifest.get("query_status_counts")
        != {"resolved": int(sun["missing_chemsys_count"])}
        or completed_spec.get("path") != sun["completed_mp_hull_cache"]
        or int(completed_spec.get("rows", -1))
        != int(sun["wanted_chemsys_count"])
        or completed_spec.get("all_rows_populated") is not True
        or completion_manifest.get("api_key_serialized") is not False
        or completion_manifest.get("sample_retry_or_replacement_used")
        is not False
    ):
        raise ValueError("completed MP cache contract changed")
    if any(os.environ.get(name) for name in ("MP_API_KEY", "PMG_MAPI_KEY", "MAPI_KEY")):
        raise RuntimeError("MP API credentials must be absent")

    report = {
        "schema": "h1_ef_fourcell_preflight_report_v1",
        "status": "pass",
        "run_id": config["run_id"],
        "cells": list(config["cells"]),
        "attempts_per_cell": DENOMINATOR,
        "source_manifest_sha256": args.source_manifest_sha256,
        "source_dependencies": dependency_reports,
        "gcd_before_comp_valid": gcd_report,
        "planner_sources": planner_reports,
        "attempt_ledger": _identity(ledger_path),
        "body_models": model_reports,
        "refiner": _verified_identity(
            refiner_path, refiner["checkpoint_sha256"]
        ),
        "required_evaluation_assets": required_assets,
        "completed_mp_cache": {
            "manifest": _identity(completion_manifest_path),
            "cache": _verified_identity(completed_cache, completed_spec["sha256"]),
            "cache_sha256_file": _identity(completed_sha_file),
            "success_marker": str(completion_success.resolve()),
            "wanted_chemsys": int(sun["wanted_chemsys_count"]),
            "missing_chemsys_resolved": int(sun["missing_chemsys_count"]),
        },
        "mp_credentials_present": False,
        "mp_api_enabled": False,
        "retry_replacement_repair_filter_rerank": False,
        "automatic_training": False,
        "automatic_downstream": False,
        "automatic_rl": False,
    }
    write_json_exclusive(args.output.resolve(), report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
