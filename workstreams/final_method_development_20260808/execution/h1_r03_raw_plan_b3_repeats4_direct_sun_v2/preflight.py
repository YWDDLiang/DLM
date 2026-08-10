#!/usr/bin/env python3
"""Fail-closed identity preflight for R03 raw Plan × B3 repeats."""

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


def identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def verified(path: Path, sha256: str) -> dict[str, Any]:
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
    prepared = args.prepared_root.resolve()
    final_root = Path(config["run_root"]).resolve()
    if (
        prepared.parent != final_root.parent
        or not prepared.name.startswith(f".{final_root.name}.preparing.")
        or final_root.exists()
    ):
        raise ValueError("prepared/final run identity changed")

    runtime_spec = config["analysis"]["reused_v4_runtime"]
    runtime_manifest = require_source_manifest(
        runtime_spec["source_dir"], runtime_spec["source_manifest_sha256"]
    )

    dependency_reports: dict[str, Any] = {}
    for name, spec in config["source_dependencies"].items():
        manifest = require_source_manifest(
            spec["source_dir"], spec["source_manifest_sha256"]
        )
        dependency_reports[name] = identity(manifest)

    raw512_spec = config["analysis"]["r03_raw_plan512"]
    raw512 = require_file(
        raw512_spec["path"], raw512_spec["sha256"], "R03 raw Plan512"
    )
    if sum(1 for line in raw512.open(encoding="utf-8") if line.strip()) != int(
        raw512_spec["rows"]
    ):
        raise ValueError("R03 raw Plan512 row count changed")

    raw256 = prepared / "inputs/r03_p0_raw_plan_first256.jsonl"
    raw256_sha = config["planner_sources"]["P0"]["raw_generations_sha256"]
    require_file(raw256, raw256_sha, "R03 raw Plan first256")
    raw_rows = ordered_rows(read_jsonl(raw256), ordinal_field="sample_idx")
    parsed = sum(row.get("parsed") is True for row in raw_rows)
    if parsed != 254:
        raise ValueError(f"R03 raw Plan parsed count changed: {parsed}")
    expected_final_raw = final_root / "inputs/r03_p0_raw_plan_first256.jsonl"
    for name, spec in config["planner_sources"].items():
        if (
            Path(spec["raw_generations"]).resolve() != expected_final_raw
            or spec["raw_generations_sha256"] != raw256_sha
            or int(spec["expected_parsed"]) != 254
        ):
            raise ValueError(f"{name} schema slot diverged from frozen R03 raw256")

    ledger_spec = config["attempt_ledger"]
    ledger_path = require_file(
        ledger_spec["path"], ledger_spec["sha256"], "common H1 seed ledger"
    )
    ledger = ordered_rows(read_jsonl(ledger_path), ordinal_field="sample_idx")
    for ordinal, row in enumerate(ledger):
        if (
            int(row.get("ordinal", -1)) != ordinal
            or not isinstance(row.get(ledger_spec["body_seed_field"]), int)
            or not isinstance(row.get(ledger_spec["refiner_seed_field"]), int)
        ):
            raise ValueError(f"seed ledger changed at ordinal {ordinal}")

    gcd = config["direct"]["gcd_before_comp_valid"]
    direct_runner = require_file(
        gcd["runner"], gcd["runner_sha256"], "R03 Direct runner"
    )
    compute_metrics = require_file(
        gcd["upstream_compute_metrics"],
        gcd["upstream_compute_metrics_sha256"],
        "R03 upstream compute_metrics.py",
    )
    compute_source = compute_metrics.read_text(encoding="utf-8")
    if (
        compute_source.count(gcd["gcd_statement"]) != 1
        or compute_source.count(gcd["validity_statement"]) != 1
        or compute_source.index(gcd["gcd_statement"])
        >= compute_source.index(gcd["validity_statement"])
    ):
        raise ValueError("GCD-before-comp_valid order changed")

    body = config["body"]
    b3 = body["models"]["B3"]
    checkpoint = Path(b3["checkpoint"]).resolve()
    adapter = require_file(
        checkpoint / body["adapter_file"], b3["adapter_sha256"], "B3 adapter"
    )
    if adapter.stat().st_size != int(body["adapter_expected_bytes"]):
        raise ValueError("B3 adapter byte count changed")
    tokenizer = require_file(
        checkpoint / "tokenizer.json",
        body["tokenizer_json_sha256"],
        "B3 tokenizer.json",
    )
    tokenizer_config = require_file(
        checkpoint / "tokenizer_config.json",
        body["tokenizer_config_sha256"],
        "B3 tokenizer_config.json",
    )
    training_terminal = require_file(
        b3["training_terminal_report"],
        b3["training_terminal_report_sha256"],
        "B3 training terminal report",
    )
    refiner = require_file(
        config["refiner"]["checkpoint"],
        config["refiner"]["checkpoint_sha256"],
        "model_494 refiner",
    )

    sun = config["sun"]
    cache = prepared / sun["completed_mp_hull_cache"]
    require_file(cache, sun["r03f_snapshot_sha256"], "R03F snapshot")
    cache_rows = read_jsonl(cache)
    if len(cache_rows) != int(sun["wanted_chemsys_count"]):
        raise ValueError("R03F snapshot row count changed")
    cache_sha_file = prepared / sun["completed_mp_hull_cache_sha256_file"]
    expected_sha_line = f"{sun['r03f_snapshot_sha256']}  {cache.name}\n"
    if cache_sha_file.read_text(encoding="ascii") != expected_sha_line:
        raise ValueError("R03F snapshot SHA file changed")
    completion_manifest_path = prepared / sun["completion_manifest"]
    completion = read_json(completion_manifest_path)
    completed_spec = completion.get("completed_mp_hull_cache") or {}
    if (
        completion.get("schema")
        != "h1_ef_fourcell_mp_cache_completion_manifest_v1"
        or completion.get("status") != "complete_all_missing_resolved"
        or completion.get("run_id") != config["run_id"]
        or completion.get("source_manifest_sha256")
        != args.source_manifest_sha256
        or int(completion.get("wanted_chemsys_count", -1)) != 227
        or int(completion.get("missing_chemsys_count", -1)) != 107
        or completion.get("external_query_performed") is not False
        or completion.get("api_key_serialized") is not False
        or completion.get("sample_retry_or_replacement_used") is not False
        or completed_spec.get("path") != sun["completed_mp_hull_cache"]
        or int(completed_spec.get("rows", -1)) != 227
        or completed_spec.get("sha256") != sun["r03f_snapshot_sha256"]
        or completed_spec.get("all_rows_populated") is not True
        or not (prepared / sun["completion_success_marker"]).is_file()
    ):
        raise ValueError("R03F frozen-cache completion contract changed")

    historical = config["analysis"]["historical_r03_b0"]
    historical_terminal = require_file(
        historical["terminal_report"],
        historical["terminal_report_sha256"],
        "R03G terminal report",
    )
    historical_attempts: list[dict[str, Any]] = []
    for repeat, spec in enumerate(historical["attempt_results"]):
        path = require_file(
            spec["path"], spec["sha256"], f"R03G repeat {repeat} attempts"
        )
        rows = ordered_rows(read_jsonl(path), ordinal_field="generation_ordinal")
        if any(
            row.get("schema") != "crysllmgen_r5c_a100_sun_attempt_v1"
            or row.get("retry_or_replacement_used") is not False
            for row in rows
        ):
            raise ValueError(f"R03G repeat {repeat} attempt contract changed")
        historical_attempts.append(verified(path, spec["sha256"]))

    if any(
        os.environ.get(name) for name in ("MP_API_KEY", "PMG_MAPI_KEY", "MAPI_KEY")
    ):
        raise RuntimeError("MP credentials must be absent")

    report = {
        "schema": "h1_r03_raw_plan_b3_repeats4_preflight_v1",
        "status": "pass",
        "run_id": config["run_id"],
        "source_manifest_sha256": args.source_manifest_sha256,
        "reused_v4_runtime_manifest": identity(runtime_manifest),
        "source_dependencies": dependency_reports,
        "r03_raw_plan512": verified(raw512, raw512_spec["sha256"]),
        "r03_raw_plan_first256": {
            **verified(raw256, raw256_sha),
            "rows": DENOMINATOR,
            "parsed": parsed,
        },
        "attempt_ledger": verified(ledger_path, ledger_spec["sha256"]),
        "gcd_before_comp_valid": {
            "status": "pass",
            "required_order": "gcd_then_smact_validity",
            "runner": verified(direct_runner, gcd["runner_sha256"]),
            "upstream_compute_metrics": verified(
                compute_metrics, gcd["upstream_compute_metrics_sha256"]
            ),
            "gcd_statement_offset": compute_source.index(gcd["gcd_statement"]),
            "validity_statement_offset": compute_source.index(
                gcd["validity_statement"]
            ),
        },
        "b3": {
            "adapter": verified(adapter, b3["adapter_sha256"]),
            "tokenizer_json": verified(tokenizer, body["tokenizer_json_sha256"]),
            "tokenizer_config_json": verified(
                tokenizer_config, body["tokenizer_config_sha256"]
            ),
            "training_terminal": verified(
                training_terminal, b3["training_terminal_report_sha256"]
            ),
        },
        "refiner": verified(refiner, config["refiner"]["checkpoint_sha256"]),
        "r03f_cache": {
            **verified(cache, sun["r03f_snapshot_sha256"]),
            "rows": len(cache_rows),
            "external_query_performed": False,
        },
        "historical_r03_b0": {
            "terminal_report": verified(
                historical_terminal, historical["terminal_report_sha256"]
            ),
            "attempt_results": historical_attempts,
        },
        "process_repeats": 4,
        "attempts_per_repeat": DENOMINATOR,
        "pooled_1024_is_descriptive_only": True,
        "mp_credentials_present": False,
        "retry_replacement_repair_filter_rerank": False,
        "automatic_training": False,
        "automatic_downstream": False,
        "automatic_rl": False,
    }
    write_json_exclusive(args.output.resolve(), report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
