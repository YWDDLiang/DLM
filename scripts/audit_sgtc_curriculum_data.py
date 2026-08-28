#!/usr/bin/env python3
"""Fail-closed audit for SGTC-DLM-v1 curriculum datasets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from crystal_dlm.stable_geometry_curriculum import (  # noqa: E402
    forbidden_training_paths,
    source_ehull,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-data", type=Path, required=True)
    parser.add_argument("--curriculum-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    source = args.source_data.resolve()
    curriculum = args.curriculum_root.resolve()
    manifest = json.loads(
        (curriculum / "SGTC_DATA_MANIFEST.json").read_text(encoding="utf-8")
    )
    threshold = float(manifest["strict_threshold_eV_per_atom"])
    ledger_rows = list(
        csv.DictReader((curriculum / "SGTC_SELECTION_LEDGER.csv").open(encoding="utf-8"))
    )
    ledger_by_split: dict[str, list[dict[str, str]]] = {}
    for row in ledger_rows:
        ledger_by_split.setdefault(str(row["split"]), []).append(row)

    report_splits = {}
    leakage_paths = []
    roundtrip_mismatches = 0
    for split in ("train", "val", "test"):
        source_rows = read_jsonl(source / f"{split}.jsonl")
        g0_rows = read_jsonl(curriculum / "g0_all" / f"{split}.jsonl")
        g1_rows = read_jsonl(curriculum / "g1_strict" / f"{split}.jsonl")
        expected_g1 = [row for row in source_rows if source_ehull(row) <= threshold]
        if len(g0_rows) != len(source_rows) or len(g1_rows) != len(expected_g1):
            raise ValueError(f"SGTC {split} selection count changed")
        if len(ledger_by_split.get(split, [])) != len(source_rows):
            raise ValueError(f"SGTC {split} ledger denominator changed")
        for source_row, clean_row in zip(source_rows, g0_rows):
            leakage_paths.extend(forbidden_training_paths(clean_row))
            for key in (
                "prompt",
                "answer",
                "reduced_composition_identity",
                "source_prompt_sha256",
            ):
                if clean_row.get(key) != source_row.get(key):
                    roundtrip_mismatches += 1
        expected_cursor = 0
        for source_row in source_rows:
            if source_ehull(source_row) > threshold:
                continue
            clean_row = g1_rows[expected_cursor]
            expected_cursor += 1
            leakage_paths.extend(forbidden_training_paths(clean_row))
            for key in (
                "prompt",
                "answer",
                "reduced_composition_identity",
                "source_prompt_sha256",
            ):
                if clean_row.get(key) != source_row.get(key):
                    roundtrip_mismatches += 1
        report_splits[split] = {
            "g0_all": len(g0_rows),
            "g1_strict": len(g1_rows),
            "strict_fraction": len(g1_rows) / len(g0_rows),
        }

    vocab_source = sha256(source / "vocab_tokens.txt")
    vocab_g0 = sha256(curriculum / "g0_all/vocab_tokens.txt")
    vocab_g1 = sha256(curriculum / "g1_strict/vocab_tokens.txt")
    gates = {
        "selection_exact": True,
        "outcome_fields_absent": not leakage_paths,
        "prompt_answer_roundtrip": roundtrip_mismatches == 0,
        "vocab_hash_equal": vocab_source == vocab_g0 == vocab_g1,
        "strict_train_sufficient": report_splits["train"]["g1_strict"] >= 5000,
        "strict_val_sufficient": report_splits["val"]["g1_strict"] >= 1000,
    }
    authorized = all(gates.values())
    report = {
        "schema": "h1a2_sgtc_curriculum_audit_v1",
        "splits": report_splits,
        "strict_threshold_eV_per_atom": threshold,
        "leakage_paths": leakage_paths[:20],
        "leakage_count": len(leakage_paths),
        "roundtrip_mismatches": roundtrip_mismatches,
        "vocab_sha256": vocab_source,
        "gates": gates,
        "sgtc_data_authorized": authorized,
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "SGTC_DATA_AUDIT.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# SGTC-DLM-v1 data audit",
        "",
        "The stable selection is offline training metadata and is absent from model-visible rows.",
        "",
        f"- G0 train/val/test: `{report_splits['train']['g0_all']}/"
        f"{report_splits['val']['g0_all']}/{report_splits['test']['g0_all']}`",
        f"- G1 strict train/val/test: `{report_splits['train']['g1_strict']}/"
        f"{report_splits['val']['g1_strict']}/{report_splits['test']['g1_strict']}`",
        f"- Outcome-field leakage: `{len(leakage_paths)}`",
        f"- Prompt/answer mismatches: `{roundtrip_mismatches}`",
        f"- Authorized: `{authorized}`",
    ]
    (output / "SGTC_DATA_AUDIT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    if authorized:
        (output / "_SUCCESS").touch()
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if not authorized:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
