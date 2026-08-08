#!/usr/bin/env python3
"""Build complete raw64/raw256 SMACT4 audit bundles on the local machine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from crystal_dlm.h1_local_smact4_ledger import (  # noqa: E402
    STAGE_AUDIT_MANIFEST_SCHEMA,
    sha256_file,
)
from crystal_dlm.h1_nocharge_ion_aux import (  # noqa: E402
    SMACT4_VERSION,
    load_smact4_icsd24_oxidation_map,
)
from scripts.audit_h1_nocharge_planner_smact4 import (  # noqa: E402
    SCHEMA,
    audit_rows,
    read_raw_rows,
)


ALLOWED_ARMS = {"p0", "sft_v2", "sft_v2_c"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, choices=(64, 256), required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--arms", required=True)
    parser.add_argument("--source-inventory-sha256", required=True)
    parser.add_argument("--science-ledger-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    arms = [value for value in args.arms.split(",") if value]
    if (
        not arms
        or arms[0] != "p0"
        or len(arms) != len(set(arms))
        or not set(arms) <= ALLOWED_ARMS
    ):
        raise SystemExit("invalid local SMACT4 stage arm list")

    oxidation_map, contract = load_smact4_icsd24_oxidation_map()
    if contract.get("smact_version") != SMACT4_VERSION:
        raise SystemExit(f"exact local SMACT {SMACT4_VERSION} is required")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    reports: dict[str, dict[str, object]] = {}
    for arm in arms:
        raw_path = args.raw_root / arm / "raw_generations.jsonl"
        rows = read_raw_rows(raw_path, int(args.stage))
        attempts, summary = audit_rows(rows, oxidation_map)
        if not summary["official_witness_parity_100pct"]:
            raise SystemExit(f"{arm} exact-SMACT4 witness parity failed")
        report = {
            "schema": SCHEMA,
            "status": "pass",
            "arm": arm,
            "denominator": int(args.stage),
            "raw_generations_sha256": sha256_file(raw_path),
            "smact4_contract": contract,
            "summary": summary,
            "attempts": attempts,
        }
        report_path = output / f"{arm}_smact4.json"
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        reports[arm] = {
            "file": report_path.name,
            "sha256": sha256_file(report_path),
            "bytes": report_path.stat().st_size,
            "raw_generations_sha256": report["raw_generations_sha256"],
        }
    manifest = {
        "schema": STAGE_AUDIT_MANIFEST_SCHEMA,
        "status": "pass",
        "execution_location": "local_windows_only",
        "a800_smact4_execution": False,
        "stage": int(args.stage),
        "denominator": int(args.stage),
        "arms": arms,
        "source_inventory_sha256": args.source_inventory_sha256,
        "science_ledger_sha256": args.science_ledger_sha256,
        "smact4_contract_sha256": contract["contract_sha256"],
        "reports": reports,
    }
    manifest_path = output / "MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest_sha = sha256_file(manifest_path)
    (output / "_SUCCESS").write_text(
        json.dumps(
            {
                "schema": STAGE_AUDIT_MANIFEST_SCHEMA,
                "complete": True,
                "manifest_sha256": manifest_sha,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({**manifest, "manifest_sha256": manifest_sha}, sort_keys=True))


if __name__ == "__main__":
    main()
