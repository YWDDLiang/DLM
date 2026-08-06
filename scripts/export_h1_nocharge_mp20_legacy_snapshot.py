#!/usr/bin/env python3
"""Freeze MP20 Planner metadata under the legacy SMACT 3.1.0 evaluator."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from importlib.metadata import version as package_version
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from crystal_dlm.h1_nocharge_ion_aux import canonical_json_sha256  # noqa: E402
from scripts.build_h1_nocharge_ion_aux_sft_data import load_source_split  # noqa: E402


SCHEMA = "h1_nocharge_mp20_legacy_snapshot_v1"
LEGACY_SMACT_VERSION = "3.1.0"
LEGACY_EVALUATOR_SHA256 = "ca1c94f583e0c97a172b5c9b7ba96505257fd74dedfc618b584c34486ac1f178"
FROZEN_COUNTS = {"train": 27136, "val": 9047, "test": 9046}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--allow-nonfrozen-fixture", action="store_true")
    args = parser.parse_args()

    import smact
    import smact.screening

    evaluator_path = PROJECT_ROOT / "crystal_dlm" / "composition_validity.py"
    evaluator_sha = sha256_file(evaluator_path)
    installed_version = str(
        getattr(smact, "__version__", None) or package_version("SMACT")
    )
    if installed_version != LEGACY_SMACT_VERSION:
        raise RuntimeError(
            f"legacy snapshot requires SMACT {LEGACY_SMACT_VERSION}, found {installed_version!r}"
        )
    if evaluator_sha != LEGACY_EVALUATOR_SHA256:
        raise RuntimeError(
            f"legacy evaluator source changed: {evaluator_sha} != {LEGACY_EVALUATOR_SHA256}"
        )
    splits = tuple(value.strip() for value in str(args.splits).split(",") if value.strip())
    if not splits or any(split not in FROZEN_COUNTS for split in splits):
        raise ValueError(f"invalid split list {splits}")
    if not args.allow_nonfrozen_fixture and set(splits) != set(FROZEN_COUNTS):
        raise RuntimeError("frozen legacy snapshot requires train,val,test")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "pass",
        "fixture_only": bool(args.allow_nonfrozen_fixture),
        "legacy_smact_version": LEGACY_SMACT_VERSION,
        "legacy_evaluator_sha256": evaluator_sha,
        "installed_smact_init_sha256": sha256_file(Path(smact.__file__)),
        "installed_smact_screening_sha256": sha256_file(Path(smact.screening.__file__)),
        "splits": {},
    }
    for split in splits:
        csv_path = args.input_dir / f"{split}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)
        rows = load_source_split(csv_path, split=split)
        if not args.allow_nonfrozen_fixture and len(rows) != FROZEN_COUNTS[split]:
            raise RuntimeError(
                f"frozen MP20 {split} count is {len(rows)}, expected {FROZEN_COUNTS[split]}"
            )
        output_path = args.output_dir / f"{split}.jsonl"
        write_jsonl(output_path, rows)
        reasons = Counter(str(row["legacy"].get("reason") or "unknown") for row in rows)
        primary = sum(
            row["legacy"].get("valid") is True
            and row["legacy"].get("reason") == "charge_neutral_pauling_valid"
            for row in rows
        )
        report["splits"][split] = {
            "row_count": len(rows),
            "source_csv_sha256": sha256_file(csv_path),
            "snapshot_jsonl_sha256": sha256_file(output_path),
            "legacy_primary_count": primary,
            "legacy_reason_counts": dict(sorted(reasons.items())),
        }
    report["contract_sha256"] = canonical_json_sha256(report)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "_SUCCESS").write_text(
        json.dumps(
            {
                "schema": SCHEMA,
                "contract_sha256": report["contract_sha256"],
                "complete": True,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
