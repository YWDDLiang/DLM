#!/usr/bin/env python3
"""Build a SEAL-style SFT buffer from diffusion-refined DLM samples.

Raw DLM generations are useful as diagnostics and as RL negatives, but SFT
positive examples should pass through the same diffusion-refinement path used
by final evaluation. This script converts a CrysLLMGen refined ``.pt`` payload
back into fixed-slot answers and keeps only composition-valid, non-duplicate
structures under light shortcut/formula caps.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from crystal_dlm.composition_validity import composition_record, pbc_duplicate_record
from crystal_dlm.fixed_slot import (
    CANONICAL_PROMPT,
    Z_TO_SYMBOL,
    arrays_to_answer,
    parse_fixed_slot_answer,
    write_json,
)
from scripts.analyze_composition_validity import load_pt_arrays, load_raw_jsonl, reorder_for_distributed_refinement
from scripts.sample_llada_crystals import graph_from_arrays, import_process_one


ACCEPTED_REASONS = {
    "charge_neutral_pauling_valid",
    "all_metal_shortcut",
    "single_element_shortcut",
}


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            count += 1
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return count


def canonical_answer_id(answer: str) -> str:
    arrays = parse_fixed_slot_answer(answer, strict=True)
    pbc_coords = []
    for coord in arrays["frac_coords"]:
        pbc_coords.append(tuple(int(round(float(value) * 100.0)) % 100 for value in coord))
    species_coords = sorted(zip(arrays["species"], pbc_coords))
    return json.dumps(
        {
            "n": arrays["num_atoms"],
            "species_coords": species_coords,
            "lengths": [round(float(value), 3) for value in arrays["lengths"]],
            "angles": [round(float(value), 3) for value in arrays["angles"]],
        },
        sort_keys=True,
    )


def lattice_is_physical(lengths: Sequence[float], angles: Sequence[float]) -> bool:
    if len(lengths) != 3 or len(angles) != 3:
        return False
    if any(float(value) <= 0.05 for value in lengths):
        return False
    alpha, beta, gamma = [math.radians(float(value)) for value in angles]
    cos_a, cos_b, cos_g = math.cos(alpha), math.cos(beta), math.cos(gamma)
    volume_factor_sq = 1.0 + 2.0 * cos_a * cos_b * cos_g - cos_a**2 - cos_b**2 - cos_g**2
    return volume_factor_sq > 1e-6


def row_to_answer(row: Mapping[str, Any]) -> tuple[str, Mapping[str, Any]]:
    atom_types = [int(value) for value in row["atom_types"]]
    species = [Z_TO_SYMBOL[value] for value in atom_types]
    answer, diagnostics = arrays_to_answer(
        lengths=row["lengths"],
        angles=row["angles"],
        species=species,
        frac_coords=row["frac_coords"],
    )
    arrays = parse_fixed_slot_answer(answer, strict=True)
    return answer, {
        "arrays": arrays,
        "encode_diagnostics": diagnostics.to_dict(),
        "comp": composition_record(arrays["atom_types"]),
        "pbc": pbc_duplicate_record(arrays["frac_coords"]),
    }


def maybe_validate_graph(arrays: Mapping[str, Any], process_one: Any | None) -> bool:
    if process_one is None:
        return True
    graph_from_arrays(arrays, process_one)
    return True


def fraction_cap_exceeded(
    *,
    accepted_count: int,
    reason: str,
    reason_counts: Counter[str],
    max_single_fraction: float,
    max_all_metal_fraction: float,
) -> bool:
    next_total = accepted_count + 1
    if reason == "single_element_shortcut":
        return (reason_counts[reason] + 1) / max(1, next_total) > max_single_fraction
    if reason == "all_metal_shortcut":
        return (reason_counts[reason] + 1) / max(1, next_total) > max_all_metal_fraction
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refined-pt", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--raw-generations-jsonl", type=Path, default=None)
    parser.add_argument("--refined-world-size", type=int, default=2)
    parser.add_argument("--prompt", default=CANONICAL_PROMPT)
    parser.add_argument("--max-accepted", type=int, default=None)
    parser.add_argument("--max-formula-repeats", type=int, default=8)
    parser.add_argument("--max-single-fraction", type=float, default=0.10)
    parser.add_argument("--max-all-metal-fraction", type=float, default=0.60)
    parser.add_argument("--min-strict-fraction", type=float, default=0.0)
    parser.add_argument("--crysllmgen-dir", type=Path, default=PROJECT_ROOT / "reference/crysllmgen")
    parser.add_argument("--skip-graph-validation", action="store_true")
    args = parser.parse_args()

    refined_rows = load_pt_arrays(args.refined_pt)
    raw_rows: list[Mapping[str, Any]] = []
    if args.raw_generations_jsonl:
        raw_rows = reorder_for_distributed_refinement(
            load_raw_jsonl(args.raw_generations_jsonl),
            args.refined_world_size,
        )[: len(refined_rows)]

    process_one = None if args.skip_graph_validation else import_process_one(args.crysllmgen_dir)
    accepted: list[dict[str, Any]] = []
    failure_counts: Counter[str] = Counter()
    accepted_reason_counts: Counter[str] = Counter()
    formula_counts: Counter[str] = Counter()
    answer_ids: set[str] = set()
    raw_reason_counts: Counter[str] = Counter()

    for idx, row in enumerate(refined_rows):
        if args.max_accepted is not None and len(accepted) >= int(args.max_accepted):
            break
        raw_row = raw_rows[idx] if idx < len(raw_rows) else {}
        raw_comp = composition_record(raw_row.get("atom_types", [])) if raw_row.get("atom_types") else None
        if raw_comp:
            raw_reason_counts[str(raw_comp["reason"])] += 1
        try:
            answer, diag = row_to_answer(row)
            arrays = diag["arrays"]
            comp = diag["comp"]
            pbc = diag["pbc"]
        except Exception as exc:
            failure_counts[f"serialize:{type(exc).__name__}"] += 1
            continue

        reason = str(comp["reason"])
        if reason not in ACCEPTED_REASONS:
            failure_counts[f"composition:{reason}"] += 1
            continue
        if not lattice_is_physical(arrays["lengths"], arrays["angles"]):
            failure_counts["invalid_lattice_volume"] += 1
            continue
        if pbc["has_pbc_equivalent_duplicate"]:
            failure_counts["pbc_equivalent_duplicate"] += 1
            continue
        if fraction_cap_exceeded(
            accepted_count=len(accepted),
            reason=reason,
            reason_counts=accepted_reason_counts,
            max_single_fraction=float(args.max_single_fraction),
            max_all_metal_fraction=float(args.max_all_metal_fraction),
        ):
            failure_counts[f"fraction_cap:{reason}"] += 1
            continue
        formula = str(comp["formula"])
        if formula_counts[formula] >= int(args.max_formula_repeats):
            failure_counts["formula_repeat_cap"] += 1
            continue
        answer_id = canonical_answer_id(answer)
        if answer_id in answer_ids:
            failure_counts["canonical_duplicate"] += 1
            continue
        try:
            maybe_validate_graph(arrays, process_one)
        except Exception as exc:
            failure_counts[f"graph:{type(exc).__name__}"] += 1
            continue

        formula_counts[formula] += 1
        accepted_reason_counts[reason] += 1
        answer_ids.add(answer_id)
        mask_policy = "n_active_element" if reason == "charge_neutral_pauling_valid" else "active_element"
        accepted.append(
            {
                "task": "unconditional",
                "prompt": str(args.prompt),
                "answer": answer,
                "text": str(args.prompt).rstrip() + "\n" + answer,
                "mask_policy": mask_policy,
                "seal_buffer": True,
                "seal_source_kind": "diffusion_refined_success",
                "source_refined_idx": idx,
                "source_raw_sample_idx": raw_row.get("sample_idx", idx),
                "raw_reason": None if raw_comp is None else raw_comp["reason"],
                "reason_after": reason,
                "formula_after": formula,
                "composition_valid_after": bool(comp["comp_valid"]),
                "strict_valid_after": reason == "charge_neutral_pauling_valid",
                "encode_diagnostics": diag["encode_diagnostics"],
            }
        )

    accepted_total = len(accepted)
    if accepted_total and float(args.min_strict_fraction) > 0:
        strict_count = accepted_reason_counts["charge_neutral_pauling_valid"]
        strict_fraction = strict_count / accepted_total
        if strict_fraction < float(args.min_strict_fraction):
            failure_counts["warning:min_strict_fraction_not_met"] += 1

    write_jsonl(args.output_jsonl, accepted)
    summary = {
        "refined_pt": str(args.refined_pt),
        "raw_generations_jsonl": None if args.raw_generations_jsonl is None else str(args.raw_generations_jsonl),
        "refined_count": len(refined_rows),
        "accepted_count": accepted_total,
        "accepted_fraction": accepted_total / max(1, len(refined_rows)),
        "accepted_reason_counts": dict(accepted_reason_counts.most_common()),
        "accepted_formula_top30": dict(formula_counts.most_common(30)),
        "raw_reason_counts_aligned": dict(raw_reason_counts.most_common()),
        "failure_counts": dict(failure_counts.most_common()),
        "max_formula_repeats": args.max_formula_repeats,
        "max_single_fraction": args.max_single_fraction,
        "max_all_metal_fraction": args.max_all_metal_fraction,
        "min_strict_fraction": args.min_strict_fraction,
    }
    write_json(args.summary_json, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
