#!/usr/bin/env python3
"""Build a reusable MP-20 composition and PBC-geometry reason cache."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from crystal_dlm.composition_validity import composition_record
from crystal_dlm.fixed_slot import parse_fixed_slot_answer, write_json


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def pbc_duplicate_record(frac_coords: Iterable[Iterable[float]]) -> dict[str, Any]:
    exact_counts: Counter[tuple[int, int, int]] = Counter()
    pbc_counts: Counter[tuple[int, int, int]] = Counter()
    for coord in frac_coords:
        bins = tuple(int(round(float(value) * 100.0)) for value in coord)
        exact_counts[bins] += 1
        pbc_counts[tuple(value % 100 for value in bins)] += 1
    exact_duplicate_sites = sum(count - 1 for count in exact_counts.values() if count > 1)
    pbc_duplicate_sites = sum(count - 1 for count in pbc_counts.values() if count > 1)
    return {
        "exact_duplicate_site_count": exact_duplicate_sites,
        "pbc_equivalent_duplicate_site_count": pbc_duplicate_sites,
        "pbc_only_duplicate_site_count": max(0, pbc_duplicate_sites - exact_duplicate_sites),
        "has_exact_duplicate": exact_duplicate_sites > 0,
        "has_pbc_equivalent_duplicate": pbc_duplicate_sites > 0,
    }


def shortcut_type(reason: str) -> str:
    if reason == "single_element_shortcut":
        return "single_element"
    if reason == "all_metal_shortcut":
        return "all_metal"
    return "none"


def cache_record(
    *,
    source: str,
    split: str,
    sample_id: Any,
    atom_types: list[int],
    frac_coords: list[list[float]] | None = None,
    answer: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    comp = composition_record(atom_types)
    pbc = pbc_duplicate_record(frac_coords or [])
    reason = str(comp["reason"])
    return {
        "source": source,
        "split": split,
        "sample_id": sample_id,
        "answer": answer,
        "formula": comp["formula"],
        "atom_types": [int(value) for value in atom_types],
        "num_atoms": comp["num_atoms"],
        "num_elements": comp["num_elements"],
        "reason": reason,
        "comp_valid": bool(comp["comp_valid"]),
        "strict_valid": reason == "charge_neutral_pauling_valid",
        "shortcut_type": shortcut_type(reason),
        "symbols": comp["symbols"],
        "counts": comp["counts"],
        **pbc,
        "metadata": dict(metadata or {}),
    }


def records_from_sft_jsonl(path: Path, *, source: str, split: str) -> Iterable[dict[str, Any]]:
    for row_idx, row in enumerate(read_jsonl(path)):
        answer = str(row.get("answer") or row.get("text") or row.get("response") or "")
        try:
            arrays = parse_fixed_slot_answer(answer)
            yield cache_record(
                source=source,
                split=split,
                sample_id=row.get("sample_idx", row_idx),
                atom_types=[int(value) for value in arrays["atom_types"]],
                frac_coords=[[float(value) for value in coord] for coord in arrays["frac_coords"]],
                answer=answer,
                metadata=row.get("metadata") or {},
            )
        except Exception as exc:
            yield {
                "source": source,
                "split": split,
                "sample_id": row.get("sample_idx", row_idx),
                "answer": answer,
                "reason": f"parse_error:{type(exc).__name__}",
                "comp_valid": False,
                "strict_valid": False,
                "shortcut_type": "none",
                "metadata": {"message": str(exc)},
            }


def records_from_csv(path: Path, *, source: str, split: str) -> Iterable[dict[str, Any]]:
    from pymatgen.core import Structure

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row_idx, row in enumerate(reader):
            try:
                structure = Structure.from_str(row["cif"], fmt="cif")
                atom_types = [int(site.specie.Z) for site in structure.sites]
                frac_coords = [[float(value) for value in coord] for coord in structure.frac_coords]
                yield cache_record(
                    source=source,
                    split=split,
                    sample_id=row.get("material_id") or row_idx,
                    atom_types=atom_types,
                    frac_coords=frac_coords,
                    metadata={key: row.get(key) for key in ("material_id", "formula") if row.get(key)},
                )
            except Exception as exc:
                yield {
                    "source": source,
                    "split": split,
                    "sample_id": row.get("material_id") or row_idx,
                    "reason": f"parse_error:{type(exc).__name__}",
                    "comp_valid": False,
                    "strict_valid": False,
                    "shortcut_type": "none",
                    "metadata": {"message": str(exc)},
                }


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    reason_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    strict = comp_valid = single = all_metal = pbc_dup = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            count += 1
            reason = str(row.get("reason", "unknown"))
            reason_counts[reason] += 1
            source_counts[str(row.get("source", "unknown"))] += 1
            strict += int(bool(row.get("strict_valid")))
            comp_valid += int(bool(row.get("comp_valid")))
            single += int(row.get("shortcut_type") == "single_element")
            all_metal += int(row.get("shortcut_type") == "all_metal")
            pbc_dup += int(bool(row.get("has_pbc_equivalent_duplicate")))
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {
        "count": count,
        "comp_valid_rate": comp_valid / max(1, count),
        "strict_valid_rate": strict / max(1, count),
        "single_element_rate": single / max(1, count),
        "all_metal_rate": all_metal / max(1, count),
        "pbc_equivalent_duplicate_rate": pbc_dup / max(1, count),
        "reason_counts": dict(reason_counts.most_common()),
        "source_counts": dict(source_counts.most_common()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-dir", type=Path, default=PROJECT_ROOT / "data/dlm_sft/mp_20")
    parser.add_argument("--reference-csv-dir", type=Path, default=PROJECT_ROOT / "reference/crysllmgen/data/mp_20")
    parser.add_argument("--raw-jsonl", action="append", default=[], help="Extra source as name=path.")
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--include-reference-csv", action="store_true")
    args = parser.parse_args()

    streams: list[Iterable[dict[str, Any]]] = []
    if args.sft_dir.exists():
        for split in ("train", "val", "test"):
            path = args.sft_dir / f"{split}.jsonl"
            if path.exists():
                streams.append(records_from_sft_jsonl(path, source=f"sft_{split}", split=split))
    if args.include_reference_csv and args.reference_csv_dir.exists():
        for split in ("train", "val", "test"):
            path = args.reference_csv_dir / f"{split}.csv"
            if path.exists():
                streams.append(records_from_csv(path, source=f"reference_{split}", split=split))
    for spec in args.raw_jsonl:
        if "=" not in spec:
            raise ValueError("--raw-jsonl must be name=path")
        name, raw_path = spec.split("=", 1)
        streams.append(records_from_sft_jsonl(Path(raw_path), source=name, split="raw"))

    def iter_all() -> Iterable[dict[str, Any]]:
        for stream in streams:
            yield from stream

    summary = write_jsonl(args.output_jsonl, iter_all())
    summary["output_jsonl"] = str(args.output_jsonl)
    if args.summary_json:
        write_json(str(args.summary_json), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
