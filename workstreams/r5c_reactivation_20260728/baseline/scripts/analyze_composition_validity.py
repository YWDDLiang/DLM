#!/usr/bin/env python3
"""Analyze MP-20 composition validity with the same SMACT rule as CrysLLMGen."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from crystal_dlm.composition_validity import composition_record, pbc_duplicate_record
from crystal_dlm.fixed_slot import parse_fixed_slot_answer, write_json
from crystal_dlm.fixed_slot_compressed import (
    CompressedFixedSlotConfig,
    parse_compressed_fixed_slot_answer,
)
from crystal_dlm.dynamic_crystal import parse_dynamic_answer
from crystal_dlm.cif_lite import parse_cif_lite_answer
from crystal_dlm.crysllmgen_text import parse_crysllmgen_text
from crystal_dlm.fixed_plain import parse_plain_answer


def load_pt_arrays(path: Path, eval_index: int = 0) -> List[Dict[str, Any]]:
    import torch

    payload = torch.load(path, map_location="cpu")
    frac_coords = payload["frac_coords"]
    atom_types = payload["atom_types"]
    lengths = payload["lengths"]
    angles = payload["angles"]
    num_atoms = payload["num_atoms"]

    if frac_coords.dim() == 3:
        frac_coords = frac_coords[eval_index]
    if atom_types.dim() == 2:
        atom_types = atom_types[eval_index]
    if lengths.dim() == 3:
        lengths = lengths[eval_index]
    if angles.dim() == 3:
        angles = angles[eval_index]
    if num_atoms.dim() == 2:
        num_atoms = num_atoms[eval_index]

    rows: List[Dict[str, Any]] = []
    start = 0
    for idx, n_atom in enumerate(num_atoms.tolist()):
        n_atom = int(n_atom)
        cur_frac = frac_coords[start : start + n_atom].tolist()
        cur_atom = [int(value) for value in atom_types[start : start + n_atom].tolist()]
        rows.append(
            {
                "sample_idx": idx,
                "frac_coords": cur_frac,
                "atom_types": cur_atom,
                "lengths": [float(value) for value in lengths[idx].tolist()],
                "angles": [float(value) for value in angles[idx].tolist()],
                "num_atoms": n_atom,
            }
        )
        start += n_atom
    return rows


def load_raw_jsonl(
    path: Path,
    text_key: str = "text",
    representation: str = "fixed_slot",
    compressed_config: CompressedFixedSlotConfig | None = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for row_idx, line in enumerate(handle):
            if not line.strip():
                continue
            record = json.loads(line)
            text = record.get(text_key) or record.get("answer") or record.get("text")
            if not text and representation == "cif_lite_modular":
                if record.get("composition_text") and record.get("lattice_text") and record.get("sites_text"):
                    text = (
                        str(record["composition_text"]).rstrip()
                        + "\n\n"
                        + str(record["lattice_text"]).rstrip()
                        + "\n\n"
                        + str(record["sites_text"]).rstrip()
                    )
            if not text:
                continue
            try:
                if representation == "dynamic_v1":
                    arrays = parse_dynamic_answer(str(text))
                elif representation == "cif_lite_modular":
                    arrays = parse_cif_lite_answer(str(text), require_no_pbc_duplicate=False)
                elif representation == "crysllmgen_text":
                    arrays = parse_crysllmgen_text(str(text), require_no_pbc_duplicate=False)
                elif representation == "fixed_plain":
                    arrays = parse_plain_answer(str(text))
                elif representation == "fixed_slot_compressed_v1":
                    arrays = parse_compressed_fixed_slot_answer(
                        str(text),
                        config=compressed_config or CompressedFixedSlotConfig(),
                    )
                else:
                    arrays = parse_fixed_slot_answer(str(text))
                rows.append(
                    {
                        "sample_idx": int(record.get("sample_idx", row_idx)),
                        "atom_types": list(arrays["atom_types"]),
                        "frac_coords": list(arrays["frac_coords"]),
                        "num_atoms": int(arrays["num_atoms"]),
                        "parsed": bool(record.get("parsed", True)),
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "sample_idx": int(record.get("sample_idx", row_idx)),
                        "parse_error": type(exc).__name__,
                        "message": str(exc),
                        "parsed": False,
                    }
                )
    return rows


def load_reference_split(csv_path: Path) -> List[Dict[str, Any]]:
    from pymatgen.core import Structure

    rows: List[Dict[str, Any]] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader):
            structure = Structure.from_str(row["cif"], fmt="cif")
            atom_types = [int(site.specie.Z) for site in structure.sites]
            rows.append({"sample_idx": idx, "atom_types": atom_types, "num_atoms": len(atom_types)})
    return rows


def summary_for_arrays(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    reason_counts: Counter[str] = Counter()
    formula_counts: Counter[str] = Counter()
    n_hist: Counter[str] = Counter()
    num_elements_hist: Counter[str] = Counter()
    reduced_count_hist: Counter[str] = Counter()
    valid_count = 0
    shortcut_count = 0
    pbc_duplicate_count = 0
    records: List[Dict[str, Any]] = []
    for row_idx, row in enumerate(rows):
        if "atom_types" not in row:
            reason_counts[str(row.get("parse_error", "missing_atom_types"))] += 1
            continue
        comp = composition_record(row["atom_types"])
        pbc = pbc_duplicate_record(row.get("frac_coords", []))
        valid_count += int(comp["comp_valid"])
        reason_counts[comp["reason"]] += 1
        formula_counts[comp["formula"]] += 1
        n_hist[str(comp["num_atoms"])] += 1
        num_elements_hist[str(comp["num_elements"])] += 1
        reduced_count_hist[",".join(map(str, comp["counts"]))] += 1
        shortcut_count += int(comp["reason"] in {"single_element_shortcut", "all_metal_shortcut"})
        pbc_duplicate_count += int(bool(pbc["has_pbc_equivalent_duplicate"]))
        records.append(
            {
                "row_idx": row_idx,
                "sample_idx": row.get("sample_idx", row_idx),
                **comp,
                **pbc,
            }
        )
    total = len(records)
    return {
        "count": total,
        "comp_valid_count": valid_count,
        "comp_valid_rate": valid_count / max(1, total),
        "shortcut_count": shortcut_count,
        "shortcut_fraction": shortcut_count / max(1, total),
        "pbc_equivalent_duplicate_count": pbc_duplicate_count,
        "pbc_equivalent_duplicate_fraction": pbc_duplicate_count / max(1, total),
        "reason_counts": dict(reason_counts.most_common()),
        "formula_top30": dict(formula_counts.most_common(30)),
        "num_atoms_histogram": dict(n_hist.most_common()),
        "num_elements_histogram": dict(num_elements_hist.most_common()),
        "reduced_count_histogram_top30": dict(reduced_count_hist.most_common(30)),
        "records": records,
    }


def reorder_for_distributed_refinement(rows: Sequence[Mapping[str, Any]], world_size: int) -> List[Mapping[str, Any]]:
    if world_size <= 1:
        return list(rows)
    reordered: List[Mapping[str, Any]] = []
    for rank in range(world_size):
        reordered.extend(rows[rank::world_size])
    return reordered


def compare_raw_refined(
    raw_rows: Sequence[Mapping[str, Any]],
    refined_rows: Sequence[Mapping[str, Any]],
    world_size: int,
) -> Dict[str, Any]:
    # Refinement usually runs with --max-proposals=1000 even when target-valid sampling
    # overshoots by a few proposals. Truncate before distributed rank-slice reordering;
    # otherwise extra tail proposals shift rank1+ alignment and create false atom-type
    # mismatches.
    raw_base = list(raw_rows)[: len(refined_rows)]
    raw_ordered = reorder_for_distributed_refinement(raw_base, world_size)
    compare_count = min(len(raw_ordered), len(refined_rows))
    mismatches: List[Dict[str, Any]] = []
    multiset_mismatches: List[Dict[str, Any]] = []
    comp_changes: Counter[str] = Counter()
    order_mismatch_count = 0
    multiset_mismatch_count = 0
    for idx in range(compare_count):
        raw_atom_types = list(map(int, raw_ordered[idx].get("atom_types", [])))
        refined_atom_types = list(map(int, refined_rows[idx].get("atom_types", [])))
        raw_comp = composition_record(raw_atom_types) if raw_atom_types else {"comp_valid": False}
        refined_comp = composition_record(refined_atom_types) if refined_atom_types else {"comp_valid": False}
        comp_changes[f"{int(raw_comp['comp_valid'])}->{int(refined_comp['comp_valid'])}"] += 1
        order_changed = raw_atom_types != refined_atom_types
        multiset_changed = sorted(raw_atom_types) != sorted(refined_atom_types)
        order_mismatch_count += int(order_changed)
        multiset_mismatch_count += int(multiset_changed)
        if order_changed and len(mismatches) < 20:
            mismatches.append(
                {
                    "aligned_idx": idx,
                    "raw_sample_idx": raw_ordered[idx].get("sample_idx"),
                    "raw_atom_types": raw_atom_types,
                    "refined_atom_types": refined_atom_types,
                    "same_atom_type_multiset": not multiset_changed,
                }
            )
        if multiset_changed and len(multiset_mismatches) < 20:
            multiset_mismatches.append(
                {
                    "aligned_idx": idx,
                    "raw_sample_idx": raw_ordered[idx].get("sample_idx"),
                    "raw_atom_types": raw_atom_types,
                    "refined_atom_types": refined_atom_types,
                }
            )
    return {
        "world_size_assumed": world_size,
        "raw_count": len(raw_rows),
        "raw_count_used_for_alignment": len(raw_base),
        "refined_count": len(refined_rows),
        "compare_count": compare_count,
        "atom_type_order_mismatch_count": order_mismatch_count,
        "atom_type_multiset_mismatch_count": multiset_mismatch_count,
        "atom_type_mismatch_count": order_mismatch_count,
        "atom_type_mismatch_examples": mismatches,
        "atom_type_multiset_mismatch_examples": multiset_mismatches,
        "comp_valid_transition_counts": dict(comp_changes.most_common()),
    }


def reference_summary(reference_csv_dir: Path) -> Dict[str, Any]:
    summaries = {}
    for split in ("train", "val", "test"):
        csv_path = reference_csv_dir / f"{split}.csv"
        if csv_path.exists():
            summaries[split] = summary_for_arrays(load_reference_split(csv_path))
            summaries[split].pop("records", None)
    return summaries


def write_markdown(payload: Mapping[str, Any], path: Path) -> None:
    lines = [
        "# Composition Validity Diagnosis",
        "",
        "## 诊断摘要",
        "",
    ]
    for name in ("raw_pt", "refined_pt", "raw_jsonl"):
        summary = payload.get(name)
        if summary:
            lines.extend(
                [
                    f"### {name}",
                    f"- 样本数：{summary['count']}",
                    f"- comp_valid：{summary['comp_valid_rate'] * 100:.4f}",
                    f"- shortcut fraction：{summary['shortcut_fraction'] * 100:.4f}",
                    "- reason counts:",
                    "```json",
                    json.dumps(summary["reason_counts"], ensure_ascii=False, indent=2),
                    "```",
                    "",
                ]
            )
    if payload.get("raw_refined_alignment"):
        lines.extend(
            [
                "## raw/refined atom_types 对齐",
                "",
                "```json",
                json.dumps(payload["raw_refined_alignment"], ensure_ascii=False, indent=2),
                "```",
                "",
            ]
        )
    if payload.get("reference"):
        ref_small = {
            split: {
                "count": item["count"],
                "comp_valid_rate": item["comp_valid_rate"],
                "reason_counts": item["reason_counts"],
                "num_atoms_histogram": item["num_atoms_histogram"],
                "num_elements_histogram": item["num_elements_histogram"],
            }
            for split, item in payload["reference"].items()
        }
        lines.extend(
            [
                "## MP-20 reference composition",
                "",
                "```json",
                json.dumps(ref_small, ensure_ascii=False, indent=2),
                "```",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-pt", type=Path, default=None)
    parser.add_argument("--refined-pt", type=Path, default=None)
    parser.add_argument("--raw-generations-jsonl", type=Path, default=None)
    parser.add_argument("--text-key", default="text")
    parser.add_argument(
        "--representation",
        choices=["fixed_slot", "fixed_slot_compressed_v1", "dynamic_v1", "cif_lite_modular", "crysllmgen_text", "fixed_plain"],
        default="fixed_slot",
    )
    parser.add_argument("--compressed-token-config", type=Path, default=None)
    parser.add_argument("--reference-csv-dir", type=Path, default=PROJECT_ROOT / "reference/crysllmgen/data/mp_20")
    parser.add_argument("--eval-index", type=int, default=0)
    parser.add_argument("--refined-world-size", type=int, default=4)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    compressed_config = (
        CompressedFixedSlotConfig.from_path(args.compressed_token_config)
        if args.compressed_token_config is not None
        else None
    )

    payload: Dict[str, Any] = {
        "raw_pt_path": None if args.raw_pt is None else str(args.raw_pt),
        "refined_pt_path": None if args.refined_pt is None else str(args.refined_pt),
        "raw_generations_jsonl": None if args.raw_generations_jsonl is None else str(args.raw_generations_jsonl),
        "reference_csv_dir": str(args.reference_csv_dir),
    }
    raw_rows: List[Dict[str, Any]] = []
    refined_rows: List[Dict[str, Any]] = []
    if args.raw_pt:
        raw_rows = load_pt_arrays(args.raw_pt, eval_index=args.eval_index)
        payload["raw_pt"] = summary_for_arrays(raw_rows)
    if args.refined_pt:
        refined_rows = load_pt_arrays(args.refined_pt, eval_index=args.eval_index)
        payload["refined_pt"] = summary_for_arrays(refined_rows)
    if args.raw_generations_jsonl:
        payload["raw_jsonl"] = summary_for_arrays(
            load_raw_jsonl(
                args.raw_generations_jsonl,
                text_key=args.text_key,
                representation=args.representation,
                compressed_config=compressed_config,
            )
        )
    if raw_rows and refined_rows:
        payload["raw_refined_alignment"] = compare_raw_refined(raw_rows, refined_rows, args.refined_world_size)
    payload["reference"] = reference_summary(args.reference_csv_dir)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    write_json(str(args.output_json), payload)
    write_markdown(payload, args.output_md)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
