#!/usr/bin/env python3
"""Build a fixed-slot SFT buffer from relaxed self-improving S.U.N. structures.

The original version accepted only strict S.U.N. rows (novel + unique +
E_hull < 0).  That pool is usually too small at 1000-sample scale, so this
builder now supports a conservative meta-S.U.N. tier as well:

* strict: novel + unique + E_hull < 0, high tier
* meta: novel + unique + E_hull < 0.1, mid-high tier

Rows are still encoded from the relaxed structure, not the raw/refined proposal.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from crystal_dlm.composition_validity import composition_record, pbc_duplicate_record
from crystal_dlm.fixed_slot import CANONICAL_PROMPT, structure_to_answer, write_json
from crystal_dlm.fixed_slot import parse_fixed_slot_answer


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            count += 1
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return count


def finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except Exception:
        return None
    return result if math.isfinite(result) else None


def detailed_column(detailed: Mapping[str, Any], *names: str) -> list[Any]:
    for name in names:
        value = detailed.get(name)
        if isinstance(value, list):
            return value
    return []


def successful_original_indices(summary: Mapping[str, Any], detailed_len: int) -> list[int]:
    total = int(summary.get("num_structures") or detailed_len)
    unsupported = {
        int(row["index"])
        for row in summary.get("unsupported_records", [])
        if isinstance(row, Mapping) and "index" in row
    }
    relax_failed = {int(value) for value in summary.get("relax_failed_indices", [])}
    indices = [idx for idx in range(total) if idx not in unsupported and idx not in relax_failed]
    if len(indices) < detailed_len:
        return list(range(detailed_len))
    return indices[:detailed_len]


def formula_chemsys(comp: Mapping[str, Any]) -> tuple[str, str]:
    formula = str(comp.get("formula") or "")
    chemsys = "-".join(sorted(str(symbol) for symbol in comp.get("symbols", []) if str(symbol)))
    return formula, chemsys


def parse_set(text: str) -> set[str]:
    return {item.strip() for item in str(text or "").split(",") if item.strip()}


def accepted_tier_for_row(
    *,
    ehull: float,
    novel_unique: bool,
    strict_threshold: float,
    meta_threshold: float,
    accepted_tiers: set[str],
) -> str | None:
    if not novel_unique:
        return None
    if "strict" in accepted_tiers and ehull < strict_threshold:
        return "strict"
    if "meta" in accepted_tiers and ehull < meta_threshold:
        return "meta"
    return None


def load_relaxed_structures(path: Path):
    import ase.io
    from pymatgen.io.ase import AseAtomsAdaptor

    atoms_list = ase.io.read(path, ":")
    if not isinstance(atoms_list, list):
        atoms_list = [atoms_list]
    adaptor = AseAtomsAdaptor()
    return [adaptor.get_structure(atoms) for atoms in atoms_list]


def maybe_validate_graph(answer: str, process_one: Any | None) -> None:
    if process_one is None:
        return
    from scripts.sample_llada_crystals import graph_from_arrays

    arrays = parse_fixed_slot_answer(answer, strict=True)
    graph_from_arrays(arrays, process_one)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--relaxed-extxyz", type=Path, required=True)
    parser.add_argument("--mattergen-summary-json", type=Path, required=True)
    parser.add_argument("--mattergen-detailed-json", type=Path, required=True)
    parser.add_argument("--refined-pt", type=Path, required=True)
    parser.add_argument("--raw-generations-jsonl", type=Path, default=None)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--prompt", default=CANONICAL_PROMPT)
    parser.add_argument("--max-accepted", type=int, default=None)
    parser.add_argument("--max-formula-repeats", type=int, default=8)
    parser.add_argument("--max-chemsys-repeats", type=int, default=64)
    parser.add_argument("--accepted-tiers", default="strict", help="Comma-separated tiers: strict,meta.")
    parser.add_argument("--strict-ehull-threshold", type=float, default=0.0)
    parser.add_argument("--meta-ehull-threshold", type=float, default=0.1)
    parser.add_argument("--strict-sample-weight-tier", default="tier_high")
    parser.add_argument("--meta-sample-weight-tier", default="tier_mid_high")
    parser.add_argument(
        "--accepted-composition-reasons",
        default="charge_neutral_pauling_valid",
        help="Comma-separated composition reasons accepted into the S.U.N. SFT buffer.",
    )
    parser.add_argument("--crysllmgen-dir", type=Path, default=PROJECT_ROOT / "reference/crysllmgen")
    parser.add_argument("--skip-graph-validation", action="store_true")
    args = parser.parse_args()

    if not args.refined_pt.exists():
        raise FileNotFoundError(f"refined source is required: {args.refined_pt}")
    accepted_tiers = parse_set(args.accepted_tiers)
    accepted_reasons = parse_set(args.accepted_composition_reasons)
    summary = read_json(args.mattergen_summary_json)
    detailed = read_json(args.mattergen_detailed_json)
    ehulls = detailed_column(detailed, "energy_above_hull", "energy_above_hull_per_atom")
    novel_unique = detailed_column(detailed, "novel_unique")
    novel = detailed_column(detailed, "novel")
    unique = detailed_column(detailed, "unique")
    comp_valid_detail = detailed_column(detailed, "comp_validity")
    n = min(len(ehulls), len(novel_unique))
    structures = load_relaxed_structures(args.relaxed_extxyz)
    if len(structures) < n:
        n = len(structures)
    raw_rows = read_jsonl(args.raw_generations_jsonl)
    original_indices = successful_original_indices(summary, n)
    if args.skip_graph_validation:
        process_one = None
    else:
        from scripts.sample_llada_crystals import import_process_one

        process_one = import_process_one(args.crysllmgen_dir)

    accepted: list[dict[str, Any]] = []
    failure_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    formula_counts: Counter[str] = Counter()
    chemsys_counts: Counter[str] = Counter()
    tier_counts: Counter[str] = Counter()
    strict_sun_formula_counts: Counter[str] = Counter()
    meta_sun_formula_counts: Counter[str] = Counter()
    for detail_idx in range(n):
        original_idx = int(original_indices[detail_idx]) if detail_idx < len(original_indices) else detail_idx
        ehull = finite_float(ehulls[detail_idx])
        is_novel_unique = bool(novel_unique[detail_idx])
        if ehull is None:
            failure_counts["missing_ehull"] += 1
            continue
        tier = accepted_tier_for_row(
            ehull=ehull,
            novel_unique=is_novel_unique,
            strict_threshold=float(args.strict_ehull_threshold),
            meta_threshold=float(args.meta_ehull_threshold),
            accepted_tiers=accepted_tiers,
        )
        if tier is None and not is_novel_unique:
            failure_counts["not_novel_unique"] += 1
            continue
        if tier is None:
            if ehull >= float(args.meta_ehull_threshold):
                failure_counts["not_meta_stable_ehull_lt_threshold"] += 1
            elif ehull >= float(args.strict_ehull_threshold):
                failure_counts["not_strict_stable_ehull_lt_threshold"] += 1
            else:
                failure_counts["tier_not_accepted"] += 1
            continue
        if comp_valid_detail and detail_idx < len(comp_valid_detail) and not bool(comp_valid_detail[detail_idx]):
            failure_counts["mattergen_comp_invalid"] += 1
            continue
        try:
            answer, encode_diag = structure_to_answer(structures[detail_idx], separator="")
            arrays = parse_fixed_slot_answer(answer, strict=True)
            comp = composition_record(arrays["atom_types"])
            pbc = pbc_duplicate_record(arrays["frac_coords"])
        except Exception as exc:
            failure_counts[f"encode:{type(exc).__name__}"] += 1
            continue
        reason = str(comp.get("reason"))
        if accepted_reasons and reason not in accepted_reasons:
            failure_counts[f"composition_reason:{reason}"] += 1
            continue
        if pbc.get("has_pbc_equivalent_duplicate"):
            failure_counts["pbc_equivalent_duplicate"] += 1
            continue
        formula, chemsys = formula_chemsys(comp)
        if args.max_formula_repeats > 0 and formula_counts[formula] >= args.max_formula_repeats:
            failure_counts["formula_cap"] += 1
            continue
        if args.max_chemsys_repeats > 0 and chemsys_counts[chemsys] >= args.max_chemsys_repeats:
            failure_counts["chemsys_cap"] += 1
            continue
        try:
            maybe_validate_graph(answer, process_one)
        except Exception as exc:
            failure_counts[f"graph:{type(exc).__name__}"] += 1
            continue
        raw_row = raw_rows[original_idx] if original_idx < len(raw_rows) else {}
        prompt = str(args.prompt)
        row = {
            "task": "unconditional",
            "prompt": prompt,
            "answer": answer,
            "text": prompt.rstrip() + "\n" + answer,
            "mask_policy": "n_active_element",
            "selection_role": f"{tier}_sun_self_improving",
            "source_kind": "relaxed_sun_after_crysllmgen_refinement",
            "source_refined_pt": str(args.refined_pt),
            "source_refined_idx": original_idx,
            "source_detail_idx": detail_idx,
            "source_raw_sample_idx": raw_row.get("sample_idx", original_idx),
            "e_above_hull": ehull,
            "strict_sun": bool(ehull < float(args.strict_ehull_threshold)),
            "meta_sun": bool(ehull < float(args.meta_ehull_threshold)),
            "novel": bool(novel[detail_idx]) if detail_idx < len(novel) else None,
            "unique": bool(unique[detail_idx]) if detail_idx < len(unique) else None,
            "novel_unique": True,
            "self_improving_tier": tier,
            "sample_weight_tier": (
                str(args.strict_sample_weight_tier)
                if tier == "strict"
                else str(args.meta_sample_weight_tier)
            ),
            "composition_reason": reason,
            "composition_bucket": "strict",
            "comp_valid": bool(comp.get("comp_valid")),
            "strict_valid": reason == "charge_neutral_pauling_valid",
            "formula": formula,
            "chemsys": chemsys,
            "num_atoms": comp.get("num_atoms"),
            "num_elements": comp.get("num_elements"),
            "pbc_equivalent_duplicate": False,
            "encode_diagnostics": encode_diag.to_dict(),
            "buffer_structure_source": "relaxed_extxyz",
        }
        accepted.append(row)
        reason_counts[reason] += 1
        tier_counts[tier] += 1
        formula_counts[formula] += 1
        chemsys_counts[chemsys] += 1
        if tier == "strict":
            strict_sun_formula_counts[formula] += 1
        elif tier == "meta":
            meta_sun_formula_counts[formula] += 1
        if args.max_accepted is not None and len(accepted) >= int(args.max_accepted):
            break

    write_jsonl(args.output_jsonl, accepted)
    payload = {
        "relaxed_extxyz": str(args.relaxed_extxyz),
        "mattergen_summary_json": str(args.mattergen_summary_json),
        "mattergen_detailed_json": str(args.mattergen_detailed_json),
        "refined_pt": str(args.refined_pt),
        "raw_generations_jsonl": None if args.raw_generations_jsonl is None else str(args.raw_generations_jsonl),
        "detailed_rows_seen": n,
        "accepted_count": len(accepted),
        "accepted_fraction_of_detailed": len(accepted) / max(1, n),
        "accepted_tiers": sorted(accepted_tiers),
        "tier_counts": dict(tier_counts.most_common()),
        "strict_ehull_threshold": float(args.strict_ehull_threshold),
        "meta_ehull_threshold": float(args.meta_ehull_threshold),
        "accepted_composition_reasons": sorted(accepted_reasons),
        "reason_counts": dict(reason_counts.most_common()),
        "failure_counts": dict(failure_counts.most_common()),
        "formula_top30": dict(formula_counts.most_common(30)),
        "chemsys_top30": dict(chemsys_counts.most_common(30)),
        "strict_sun_formula_top30": dict(strict_sun_formula_counts.most_common(30)),
        "meta_sun_formula_top30": dict(meta_sun_formula_counts.most_common(30)),
        "buffer_structure_source": "relaxed_extxyz",
        "policy": "Only relaxed structures that came from CrysLLMGen-refined proposals and satisfy configured strict/meta S.U.N. tiers are accepted.",
    }
    write_json(str(args.summary_json), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
