#!/usr/bin/env python3
"""Merge frozen 256-chunk Direct/CHGNet outputs into a global CTV label ledger."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def row_ordinal(row: dict[str, Any]) -> int:
    if row.get("ordinal") is not None:
        return int(row["ordinal"])
    attempt_id = str(row.get("attempt_id") or "")
    if not attempt_id:
        raise ValueError("CTV evaluation row has no ordinal or attempt id")
    return int(attempt_id.rsplit("-", 1)[-1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation-root", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-branches", type=int, required=True)
    args = parser.parse_args()

    expected = int(args.expected_branches)
    metadata = read_jsonl(args.generation_root / "branch_metadata.jsonl")
    metadata_by_global = {
        int(row["global_branch_ordinal"]): row for row in metadata
    }
    if len(metadata_by_global) != expected or set(metadata_by_global) != set(
        range(expected)
    ):
        raise ValueError("CTV branch metadata does not cover the global denominator")

    rows: list[dict[str, Any]] = []
    failures: Counter[str] = Counter()
    for global_ordinal in range(expected):
        meta = metadata_by_global[global_ordinal]
        chunk_index = int(meta["chunk_index"])
        local_ordinal = int(meta["local_ordinal"])
        chunk = args.evaluation_root / f"chunk{chunk_index:02d}"
        labels = {
            int(row["ordinal"]): row
            for row in read_jsonl(
                chunk / "full_reconstructed/attempt_labels_preofficial.jsonl"
            )
        }
        direct = {
            row_ordinal(row): row
            for row in read_jsonl(chunk / "direct/attempt_metrics.jsonl")
        }
        if set(labels) != set(range(256)) or set(direct) != set(range(256)):
            raise ValueError(f"CTV evaluation chunk {chunk_index} is incomplete")
        label = labels[local_ordinal]
        metric = direct[local_ordinal]
        energy = label.get("chgnet_energy_per_atom")
        known = label.get("chgnet_relaxation_known") is True and energy is not None
        if not known:
            failures["chgnet_unknown"] += 1
        if metric.get("valid") is not True:
            failures["direct_invalid"] += 1
        rows.append(
            {
                "schema": "h1a2_ctv_branch_terminal_label_v1",
                **meta,
                "direct_valid": metric.get("valid") is True,
                "reconstructed": label.get("reconstructed") is True,
                "chgnet_relaxation_known": known,
                "chgnet_energy_per_atom": None if energy is None else float(energy),
                "chgnet_composition": label.get("chgnet_composition"),
                "novel": label.get("novel") is True,
                "unique_representative": label.get("unique_representative") is True,
                "novel_unique": label.get("novel_unique") is True,
                "unknown_is_negative": False,
            }
        )

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    label_path = output / "CTV_BRANCH_TERMINAL_LABELS.jsonl"
    with label_path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    report = {
        "schema": "h1a2_ctv_branch_terminal_labels_manifest_v1",
        "branches": len(rows),
        "direct_valid": sum(row["direct_valid"] for row in rows),
        "reconstructed": sum(row["reconstructed"] for row in rows),
        "chgnet_known": sum(row["chgnet_relaxation_known"] for row in rows),
        "failures": dict(failures.most_common()),
        "unknown_policy": "missing; never negative or high energy",
        "labels_sha256": hashlib.sha256(label_path.read_bytes()).hexdigest(),
    }
    (output / "CTV_BRANCH_TERMINAL_LABELS_MANIFEST.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "_SUCCESS").touch()
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
