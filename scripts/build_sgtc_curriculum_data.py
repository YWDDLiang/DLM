#!/usr/bin/env python3
"""Build outcome-sanitized all-MP20 and strict-stable SGTC datasets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from crystal_dlm.stable_geometry_curriculum import (  # noqa: E402
    source_ehull,
    strip_training_outcomes,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-data", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--strict-threshold", type=float, default=1e-8)
    args = parser.parse_args()

    source = args.source_data.resolve()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=False)
    arms = {
        "g0_all": output / "g0_all",
        "g1_strict": output / "g1_strict",
    }
    for path in arms.values():
        path.mkdir()
        shutil.copy2(source / "vocab_tokens.txt", path / "vocab_tokens.txt")

    manifest: dict[str, Any] = {
        "schema": "h1a2_sgtc_curriculum_data_v1",
        "source_data": str(source),
        "strict_threshold_eV_per_atom": float(args.strict_threshold),
        "selection_field_visible_to_model": False,
        "stable_or_energy_prompt_token": False,
        "splits": {},
    }
    ledger_path = output / "SGTC_SELECTION_LEDGER.csv"
    with ledger_path.open("x", newline="", encoding="utf-8") as ledger_handle:
        writer = csv.DictWriter(
            ledger_handle,
            fieldnames=(
                "split",
                "source_ordinal",
                "material_id",
                "reduced_composition_identity",
                "source_e_above_hull",
                "selected_g1_strict",
            ),
        )
        writer.writeheader()
        for split in ("train", "val", "test"):
            source_path = source / f"{split}.jsonl"
            rows = read_jsonl(source_path)
            all_rows = []
            strict_rows = []
            for ordinal, row in enumerate(rows):
                ehull = source_ehull(row)
                sanitized = strip_training_outcomes(row)
                all_rows.append(sanitized)
                selected = ehull <= float(args.strict_threshold)
                if selected:
                    strict_rows.append(sanitized)
                metadata = row.get("metadata") or {}
                writer.writerow(
                    {
                        "split": split,
                        "source_ordinal": ordinal,
                        "material_id": metadata.get("material_id", ""),
                        "reduced_composition_identity": row.get(
                            "reduced_composition_identity", ""
                        ),
                        "source_e_above_hull": f"{ehull:.12g}",
                        "selected_g1_strict": int(selected),
                    }
                )
            write_jsonl(arms["g0_all"] / f"{split}.jsonl", all_rows)
            write_jsonl(arms["g1_strict"] / f"{split}.jsonl", strict_rows)
            manifest["splits"][split] = {
                "source_rows": len(rows),
                "g0_all_rows": len(all_rows),
                "g1_strict_rows": len(strict_rows),
                "source_sha256": sha256(source_path),
                "g0_all_sha256": sha256(arms["g0_all"] / f"{split}.jsonl"),
                "g1_strict_sha256": sha256(
                    arms["g1_strict"] / f"{split}.jsonl"
                ),
            }
    manifest["vocab_sha256"] = sha256(source / "vocab_tokens.txt")
    manifest["selection_ledger_sha256"] = sha256(ledger_path)
    (output / "SGTC_DATA_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "_SUCCESS").touch()
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
