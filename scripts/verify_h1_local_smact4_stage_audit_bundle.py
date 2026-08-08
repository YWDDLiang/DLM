#!/usr/bin/env python3
"""Verify local-only SMACT4 stage audits without importing SMACT4 on A800."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from crystal_dlm.h1_local_smact4_ledger import (  # noqa: E402
    EXPECTED_SMACT4_CONTRACT_SHA256,
    STAGE_AUDIT_MANIFEST_SCHEMA,
    read_object,
    sha256_file,
)
from scripts.evaluate_h1_nocharge_sft_planner_gate import (  # noqa: E402
    load_smact4_audit,
)


ALLOWED_ARMS = {"p0", "sft_v2", "sft_v2_c"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, choices=(64, 256), required=True)
    parser.add_argument("--arms", required=True)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--source-inventory-sha256", required=True)
    parser.add_argument("--science-ledger", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    arms = [value for value in args.arms.split(",") if value]
    if (
        not arms
        or arms[0] != "p0"
        or len(arms) != len(set(arms))
        or not set(arms) <= ALLOWED_ARMS
    ):
        raise SystemExit("invalid exact-SMACT4 audit arm list")

    root = args.audit_root.resolve()
    manifest_path = root / "MANIFEST.json"
    success_path = root / "_SUCCESS"
    manifest = read_object(manifest_path)
    success = read_object(success_path)
    manifest_sha = sha256_file(manifest_path)
    if (
        manifest_sha != args.expected_manifest_sha256
        or manifest.get("schema") != STAGE_AUDIT_MANIFEST_SCHEMA
        or manifest.get("status") != "pass"
        or manifest.get("execution_location") != "local_windows_only"
        or manifest.get("a800_smact4_execution") is not False
        or int(manifest.get("stage", -1)) != int(args.stage)
        or int(manifest.get("denominator", -1)) != int(args.stage)
        or manifest.get("arms") != arms
        or manifest.get("source_inventory_sha256")
        != args.source_inventory_sha256
        or manifest.get("science_ledger_sha256") != sha256_file(args.science_ledger)
        or manifest.get("smact4_contract_sha256")
        != EXPECTED_SMACT4_CONTRACT_SHA256
        or success.get("schema") != STAGE_AUDIT_MANIFEST_SCHEMA
        or success.get("complete") is not True
        or success.get("manifest_sha256") != manifest_sha
    ):
        raise SystemExit("local exact-SMACT4 stage manifest identity mismatch")
    reports = manifest.get("reports")
    if not isinstance(reports, dict) or set(reports) != set(arms):
        raise SystemExit("local exact-SMACT4 stage report set mismatch")
    output_reports: dict[str, dict[str, object]] = {}
    for arm in arms:
        item = reports[arm]
        if not isinstance(item, dict):
            raise SystemExit(f"{arm} local audit manifest row is invalid")
        expected_name = f"{arm}_smact4.json"
        if item.get("file") != expected_name:
            raise SystemExit(f"{arm} local audit filename is not frozen")
        report_path = root / expected_name
        raw_path = args.raw_root / arm / "raw_generations.jsonl"
        if (
            not report_path.is_file()
            or sha256_file(report_path) != item.get("sha256")
            or report_path.stat().st_size != int(item.get("bytes", -1))
            or sha256_file(raw_path) != item.get("raw_generations_sha256")
        ):
            raise SystemExit(f"{arm} local audit file/raw identity mismatch")
        _rows, contract = load_smact4_audit(
            report_path,
            raw_path=raw_path,
            arm=arm,
            denominator=int(args.stage),
        )
        if contract.get("contract_sha256") != EXPECTED_SMACT4_CONTRACT_SHA256:
            raise SystemExit(f"{arm} local audit contract SHA mismatch")
        output_reports[arm] = {
            "path": str(report_path),
            "sha256": sha256_file(report_path),
            "raw_sha256": sha256_file(raw_path),
        }
    observed_files = {
        path.name for path in root.iterdir() if path.is_file()
    }
    expected_files = {"MANIFEST.json", "_SUCCESS"} | {
        str(reports[arm]["file"]) for arm in arms
    }
    if observed_files != expected_files:
        raise SystemExit("local audit bundle has missing or extra files")
    result = {
        "schema": "h1_local_smact4_stage_audit_import_v1",
        "status": "pass",
        "stage": int(args.stage),
        "arms": arms,
        "manifest_sha256": manifest_sha,
        "source_inventory_sha256": args.source_inventory_sha256,
        "science_ledger_sha256": sha256_file(args.science_ledger),
        "reports": output_reports,
        "smact4_executed_on_a800": False,
    }
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
