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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = args.source_dir.resolve()
    require_source_manifest(source, args.source_manifest_sha256)
    config = read_json(args.config.resolve())
    validate_config(config)

    dependency_reports: dict[str, Any] = {}
    for name, specification in config["source_dependencies"].items():
        manifest = require_source_manifest(
            specification["source_dir"],
            specification["source_manifest_sha256"],
        )
        dependency_reports[name] = _identity(manifest)

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
    project = Path(config["run_root"]).parents[1]
    required_relative = [
        config["direct"]["gt_csv"],
        config["sun"]["eval_sun_py"],
        config["sun"]["eval_sun_resumable_py"],
        config["sun"]["train_csv"],
        config["sun"]["training_index_cache"],
        config["sun"]["mp_hull_cache"],
        config["sun"]["chgnet_relax_cache"],
    ]
    required_absolute = [
        config["sun"]["chgnet_model_asset"],
        config["sun"]["chgnet_runtime_checkpoint"],
    ]
    required_assets = [
        _identity(project / relative) for relative in required_relative
    ] + [_identity(Path(path)) for path in required_absolute]
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
        "planner_sources": planner_reports,
        "attempt_ledger": _identity(ledger_path),
        "body_models": model_reports,
        "refiner": _verified_identity(
            refiner_path, refiner["checkpoint_sha256"]
        ),
        "required_evaluation_assets": required_assets,
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
