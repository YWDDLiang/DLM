#!/usr/bin/env python3
"""Verify one R03E repeat before any frozen refiner computation."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

from protocol import (
    ARMS,
    DENOMINATOR,
    ordered_rows,
    read_json,
    read_jsonl,
    require_identity,
    require_source_manifest,
    sha256_file,
    validate_config,
    validate_repeat,
    write_json_exclusive,
)


REQUIRED_ORIGINS = {
    "runner": "scripts/a800/run_crysllmgen_a100_sun.py",
    "crystal_dlm": "crystal_dlm/__init__.py",
    "crystal_dlm.wqcodiff": "crystal_dlm/wqcodiff/__init__.py",
    "crystal_dlm.wqcodiff.contracts": "crystal_dlm/wqcodiff/contracts.py",
    "crystal_dlm.wqcodiff.crysllmgen": (
        "crystal_dlm/wqcodiff/crysllmgen/__init__.py"
    ),
    "crystal_dlm.wqcodiff.crysllmgen.epoch_training": (
        "crystal_dlm/wqcodiff/crysllmgen/epoch_training.py"
    ),
    "crystal_dlm.wqcodiff.crysllmgen.a100_sun": (
        "crystal_dlm/wqcodiff/crysllmgen/a100_sun.py"
    ),
}


def isolated_runtime_preflight(runtime_root: Path) -> dict[str, Any]:
    runtime = runtime_root.resolve()
    runner = runtime / "scripts/a800/run_crysllmgen_a100_sun.py"
    bootstrap = """
import runpy
import sys
from pathlib import Path
runtime = Path(sys.argv[1]).resolve()
runner = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(runtime))
sys.argv = [str(runner), "--preflight-runtime-root", str(runtime)]
runpy.run_path(str(runner), run_name="__main__")
"""
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            bootstrap,
            str(runtime),
            str(runner),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "isolated S.U.N. runtime preflight failed\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError(f"unexpected runtime preflight output: {completed.stdout!r}")
    report = json.loads(lines[0])
    if report.get("status") != "pass" or report.get("origins") != REQUIRED_ORIGINS:
        raise RuntimeError(f"S.U.N. runtime import origins changed: {report}")
    return report


def _verify_graphs(
    path: Path,
    *,
    arm: str,
    body_success_ordinals: set[int],
) -> dict[str, Any]:
    loaded = torch.load(path, map_location="cpu")
    if not isinstance(loaded, list):
        raise TypeError(f"{arm} proposal graph payload is not a list")
    observed: dict[int, dict[str, Any]] = {}
    expected_schedule = "D1" if arm == "control" else "D2_SAFE_AXIS"
    for record in loaded:
        if not isinstance(record, dict) or not isinstance(record.get("graph"), dict):
            raise TypeError(f"{arm} proposal graph record is malformed")
        ordinal = int(record.get("ordinal", -1))
        graph = record["graph"]
        metadata = graph.get("h1_safeaxis256_metadata")
        if (
            ordinal in observed
            or not isinstance(metadata, dict)
            or int(metadata.get("ordinal", -1)) != ordinal
            or str(metadata.get("schedule_arm")) != expected_schedule
        ):
            raise ValueError(f"{arm} proposal graph identity changed")
        observed[ordinal] = record
    if set(observed) != body_success_ordinals:
        raise ValueError(f"{arm} proposal graphs no longer match body successes")
    return {
        "graphs": len(observed),
        "ordinals": sorted(observed),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    args = parser.parse_args()

    repeat = validate_repeat(args.repeat)
    source = args.source_dir.resolve()
    require_source_manifest(source, args.source_manifest_sha256)
    config = read_json(args.config.resolve())
    validate_config(config)
    protocol_path = source.parents[3] / config["protocol"]["path"]
    if (
        not protocol_path.is_file()
        or sha256_file(protocol_path) != config["protocol"]["sha256"]
    ):
        raise ValueError("preregistered R03E repeat protocol changed")
    expected_order = config["protocol"]["arm_order"][repeat]
    if sorted(expected_order) != sorted(ARMS):
        raise ValueError("balanced repeat arm order changed")

    source_body = config["source_body_run"]
    source_manifest = (
        Path(source_body["source_dir"]).resolve() / "SOURCE_SHA256.txt"
    )
    if (
        not source_manifest.is_file()
        or sha256_file(source_manifest)
        != source_body["source_manifest_sha256"]
    ):
        raise ValueError("frozen R03D source manifest changed")
    run_root = Path(source_body["run_root"]).resolve()
    frozen_reports = {
        "terminal_report": (
            run_root / "terminal_report.json",
            source_body["terminal_report_sha256"],
        ),
        "generation_report": (
            run_root / "screen/generation_report.json",
            source_body["generation_report_sha256"],
        ),
        "schedule_invariants": (
            run_root / "screen/schedule_invariants.json",
            source_body["schedule_invariants_sha256"],
        ),
    }
    for label, (path, expected_sha) in frozen_reports.items():
        if not path.is_file() or sha256_file(path) != expected_sha:
            raise ValueError(f"frozen R03D {label} changed")

    ledger_path = Path(config["attempt_ledger"]["path"]).resolve()
    if (
        not ledger_path.is_file()
        or sha256_file(ledger_path) != config["attempt_ledger"]["sha256"]
    ):
        raise ValueError("frozen H1 attempt ledger changed")
    ledger = ordered_rows(
        read_jsonl(ledger_path),
        ordinal_field="sample_idx",
    )
    if any(
        int(row.get("ordinal", -1)) != ordinal
        or int(row.get("evaluation_ordinal", -1)) != ordinal
        or not isinstance(row.get("refiner_noise_seed"), int)
        for ordinal, row in enumerate(ledger)
    ):
        raise ValueError("frozen H1 ordinal/refiner seed ledger changed")

    arm_reports: dict[str, Any] = {}
    for arm in ARMS:
        specification = config["arms"][arm]
        attempt_path = require_identity(
            specification["body_attempts"], f"{arm} body attempts"
        )
        graph_path = require_identity(
            specification["proposal_graphs"], f"{arm} proposal graphs"
        )
        attempts = ordered_rows(
            read_jsonl(attempt_path),
            ordinal_field="ordinal",
        )
        success_ordinals = {
            ordinal
            for ordinal, row in enumerate(attempts)
            if row.get("status") == "succeeded"
        }
        if (
            len(attempts) != DENOMINATOR
            or len(success_ordinals) != int(specification["expected_body_success"])
            or {
                ordinal
                for ordinal, row in enumerate(attempts)
                if row.get("earliest_failure_stage") == "planner"
            }
            != set(source_body["planner_ineligible_ordinals"])
            or any(
                int(row.get("sample_idx", -1)) != ordinal
                or str(row.get("planner_arm")) != "P0"
                or str(row.get("body_checkpoint_arm")) != "B0"
                or row.get("retry_used") is not False
                or row.get("replacement_used") is not False
                or row.get("repair_used") is not False
                or row.get("filter_used") is not False
                or row.get("rerank_used") is not False
                or (
                    row.get("body_eligible") is True
                    and int(row.get("body_noise_seed", -1))
                    != int(ledger[ordinal]["body_noise_seed"])
                )
                for ordinal, row in enumerate(attempts)
            )
        ):
            raise ValueError(f"{arm} frozen body attempt contract changed")
        graph_report = _verify_graphs(
            graph_path,
            arm=arm,
            body_success_ordinals=success_ordinals,
        )
        arm_reports[arm] = {
            "body_attempts": len(attempts),
            "body_success": len(success_ordinals),
            "proposal_graphs": graph_report["graphs"],
            "body_attempts_sha256": sha256_file(attempt_path),
            "proposal_graphs_sha256": sha256_file(graph_path),
        }

    present_credentials = [
        name
        for name in ("MP_API_KEY", "PMG_MAPI_KEY", "MAPI_KEY")
        if bool(os.environ.get(name))
    ]
    if present_credentials:
        raise RuntimeError(f"MP credentials must be absent: {present_credentials}")
    refiner = Path(config["refiner"]["checkpoint"]).resolve()
    if not refiner.is_file():
        raise FileNotFoundError(refiner)

    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    report = {
        "schema": "h1_r03e_repeat_preflight_v1",
        "status": "pass",
        "repeat": repeat,
        "arm_order": expected_order,
        "attempts_per_arm": DENOMINATOR,
        "attempt_ledger_sha256": sha256_file(ledger_path),
        "protocol_sha256": sha256_file(protocol_path),
        "refiner_checkpoint": str(refiner),
        "refiner_checkpoint_sha256_recorded": config["refiner"][
            "checkpoint_sha256"
        ],
        "refiner_checkpoint_rehashed": False,
        "arms": arm_reports,
        "runtime_import_preflight": isolated_runtime_preflight(source / "runtime"),
        "mp_credentials_present": False,
        "mp_api_enabled": False,
        "source_manifest_sha256": args.source_manifest_sha256,
        "generation_rerun": False,
        "body_rerun": False,
        "new_scientific_seed_per_repeat": False,
    }
    write_json_exclusive(output / "preflight_report.json", report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
