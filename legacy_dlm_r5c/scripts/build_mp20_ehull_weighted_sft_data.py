#!/usr/bin/env python3
"""Build an E-hull-aware fixed-slot MP-20 SFT dataset.

The builder preserves every original MP-20 row and adds a controlled number of
weighted repeat rows.  It is intentionally conservative: generated strict-SUN
rows with negative E-hull are placed in the same high-weight tier as MP-20
E_hull=0 rows, never above them.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from crystal_dlm.composition_validity import composition_record, pbc_duplicate_record
from crystal_dlm.fixed_slot import parse_fixed_slot_answer, write_json


DEFAULT_TIER_WEIGHTS = {
    "tier_high": 1.0,
    "tier_mid_high": 0.7,
    "tier_mid": 0.45,
    "tier_low_mid": 0.25,
    "tier_low_retained": 0.1,
}
DEFAULT_MASK_POLICY_MIX = {
    "normal": 0.40,
    "active_element": 0.30,
    "n_active_element": 0.20,
    "active_element_empty": 0.10,
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            count += 1
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return count


def parse_mix(spec: str, defaults: dict[str, float]) -> dict[str, float]:
    if not spec:
        return dict(defaults)
    result: dict[str, float] = {}
    for item in spec.split(","):
        if not item.strip():
            continue
        if "=" in item:
            key, value = item.split("=", 1)
        else:
            key, value = item.split(":", 1)
        result[key.strip()] = float(value)
    total = sum(result.values())
    if total <= 0:
        raise ValueError(f"Invalid mix: {spec!r}")
    return {key: value / total for key, value in result.items()}


def weighted_choice(rng: random.Random, weights: dict[str, float]) -> str:
    keys = list(weights)
    values = [weights[key] for key in keys]
    return str(rng.choices(keys, weights=values, k=1)[0])


def load_csv_metadata(csv_path: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    by_material_id: dict[str, dict[str, Any]] = {}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader):
            payload = dict(row)
            payload["_csv_row_idx"] = idx
            rows.append(payload)
            material_id = str(row.get("material_id") or "")
            if material_id:
                by_material_id[material_id] = payload
    return rows, by_material_id


def float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except Exception:
        return None


def ehull_tier(value: float | None, *, eq0_tol: float) -> str:
    if value is None:
        return "tier_low_retained"
    if abs(value) <= eq0_tol:
        return "tier_high"
    if value < 0:
        return "tier_high"
    if value <= 0.001:
        return "tier_mid_high"
    if value <= 0.01:
        return "tier_mid"
    if value <= 0.05:
        return "tier_low_mid"
    return "tier_low_retained"


def chemsys_from_symbols(symbols: Iterable[str]) -> str:
    return "-".join(sorted({str(symbol) for symbol in symbols if str(symbol)}))


def row_answer(row: dict[str, Any]) -> str:
    return str(row.get("answer") or row.get("response") or row.get("text") or "")


def annotate_base_row(
    row: dict[str, Any],
    *,
    split: str,
    row_idx: int,
    csv_rows: list[dict[str, Any]],
    csv_by_material_id: dict[str, dict[str, Any]],
    tier_weights: dict[str, float],
    eq0_tol: float,
    mask_policy: str,
    selection_role: str,
) -> dict[str, Any]:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    material_id = str(metadata.get("material_id") or "")
    csv_row = csv_by_material_id.get(material_id) if material_id else None
    if csv_row is None and row_idx < len(csv_rows):
        csv_row = csv_rows[row_idx]
    csv_row = csv_row or {}
    ehull = float_or_none(metadata.get("e_above_hull", csv_row.get("e_above_hull")))
    formation = float_or_none(metadata.get("formation_energy_per_atom", csv_row.get("formation_energy_per_atom")))
    try:
        arrays = parse_fixed_slot_answer(row_answer(row), strict=True)
        comp = composition_record(arrays["atom_types"])
        pbc = pbc_duplicate_record(arrays["frac_coords"])
    except Exception as exc:
        comp = {
            "formula": metadata.get("pretty_formula") or csv_row.get("pretty_formula") or None,
            "symbols": [],
            "comp_valid": False,
            "reason": f"parse_error:{type(exc).__name__}",
            "num_atoms": None,
            "num_elements": None,
        }
        pbc = {"has_pbc_equivalent_duplicate": None}
    tier = ehull_tier(ehull, eq0_tol=eq0_tol)
    clean = dict(row)
    clean["metadata"] = dict(metadata)
    if material_id or csv_row.get("material_id"):
        clean["metadata"]["material_id"] = material_id or csv_row.get("material_id")
    clean.update(
        {
            "selection_role": selection_role,
            "source_kind": "mp20",
            "source_split": split,
            "source_row_idx": row_idx,
            "csv_row_idx": csv_row.get("_csv_row_idx"),
            "e_above_hull": ehull,
            "formation_energy_per_atom": formation,
            "sample_weight_tier": tier,
            "sample_weight": float(tier_weights.get(tier, 0.1)),
            "composition_reason": comp.get("reason"),
            "composition_bucket": composition_bucket(str(comp.get("reason"))),
            "comp_valid": bool(comp.get("comp_valid")),
            "strict_valid": comp.get("reason") == "charge_neutral_pauling_valid",
            "formula": comp.get("formula") or metadata.get("pretty_formula") or csv_row.get("pretty_formula"),
            "chemsys": chemsys_from_symbols(comp.get("symbols", [])) or str(csv_row.get("elements") or ""),
            "num_atoms": comp.get("num_atoms"),
            "num_elements": comp.get("num_elements"),
            "pbc_equivalent_duplicate": pbc.get("has_pbc_equivalent_duplicate"),
            "mask_policy": mask_policy,
        }
    )
    clean["cap_keys"] = {
        "formula": clean.get("formula"),
        "chemsys": clean.get("chemsys"),
    }
    return clean


def composition_bucket(reason: str) -> str:
    if reason == "charge_neutral_pauling_valid":
        return "strict"
    if reason == "all_metal_shortcut":
        return "all_metal"
    if reason == "single_element_shortcut":
        return "single_element"
    return "invalid"


def generated_self_improving_tier(
    row: dict[str, Any],
    *,
    ehull: float | None,
    include_meta: bool,
) -> tuple[str, bool, bool] | None:
    if ehull is None:
        return None
    novel_unique = bool(row.get("novel_unique"))
    strict_sun = bool(row.get("strict_sun")) or (novel_unique and ehull < 0.0)
    meta_sun = bool(row.get("meta_sun")) or (novel_unique and ehull < 0.1)
    if strict_sun and ehull < 0.0:
        return str(row.get("sample_weight_tier") or "tier_high"), True, True
    if include_meta and meta_sun and ehull < 0.1:
        return str(row.get("sample_weight_tier") or "tier_mid_high"), False, True
    return None


def annotate_generated_row(
    row: dict[str, Any],
    *,
    row_idx: int,
    tier_weights: dict[str, float],
    mask_policy: str,
    include_meta: bool,
) -> dict[str, Any] | None:
    ehull = float_or_none(row.get("e_above_hull", row.get("energy_above_hull_per_atom")))
    tier_payload = generated_self_improving_tier(row, ehull=ehull, include_meta=include_meta)
    if tier_payload is None:
        return None
    tier, strict_sun, meta_sun = tier_payload
    try:
        arrays = parse_fixed_slot_answer(row_answer(row), strict=True)
        comp = composition_record(arrays["atom_types"])
        pbc = pbc_duplicate_record(arrays["frac_coords"])
    except Exception:
        return None
    clean = dict(row)
    clean.update(
        {
            "selection_role": "self_improving_base",
            "source_kind": "generated_self_improving",
            "source_row_idx": row_idx,
            "e_above_hull": ehull,
            "strict_sun": bool(strict_sun),
            "meta_sun": bool(meta_sun),
            "sample_weight_tier": tier,
            "sample_weight": float(tier_weights.get(tier, row.get("sample_weight", 0.5) or 0.5)),
            "composition_reason": comp.get("reason"),
            "composition_bucket": composition_bucket(str(comp.get("reason"))),
            "comp_valid": bool(comp.get("comp_valid")),
            "strict_valid": comp.get("reason") == "charge_neutral_pauling_valid",
            "formula": comp.get("formula"),
            "chemsys": chemsys_from_symbols(comp.get("symbols", [])),
            "num_atoms": comp.get("num_atoms"),
            "num_elements": comp.get("num_elements"),
            "pbc_equivalent_duplicate": pbc.get("has_pbc_equivalent_duplicate"),
            "mask_policy": mask_policy,
        }
    )
    clean["cap_keys"] = {
        "formula": clean.get("formula"),
        "chemsys": clean.get("chemsys"),
    }
    return clean


def select_repeats(
    rows: list[dict[str, Any]],
    *,
    target_count: int,
    tier_weights: dict[str, float],
    seed: int,
    max_formula_repeats: int,
    max_chemsys_repeats: int,
    mask_mix: dict[str, float],
    selection_role: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    pool = [row for row in rows if float(row.get("sample_weight", 0.0)) > 0.0]
    if not pool or target_count <= 0:
        return [], {"target_count": target_count, "selected_count": 0}
    formula_counts: Counter[str] = Counter()
    chemsys_counts: Counter[str] = Counter()
    selected: list[dict[str, Any]] = []
    skip_counts: Counter[str] = Counter()
    weights = [float(tier_weights.get(str(row.get("sample_weight_tier")), row.get("sample_weight", 0.1))) for row in pool]
    max_attempts = max(1000, target_count * 100)
    attempts = 0
    while len(selected) < target_count and attempts < max_attempts:
        attempts += 1
        base = rng.choices(pool, weights=weights, k=1)[0]
        formula = str(base.get("formula") or "")
        chemsys = str(base.get("chemsys") or "")
        if max_formula_repeats > 0 and formula_counts[formula] >= max_formula_repeats:
            skip_counts["formula_cap"] += 1
            continue
        if max_chemsys_repeats > 0 and chemsys_counts[chemsys] >= max_chemsys_repeats:
            skip_counts["chemsys_cap"] += 1
            continue
        row = dict(base)
        row["selection_role"] = selection_role
        row["repeat_source_row_idx"] = base.get("source_row_idx")
        row["repeat_idx"] = len(selected)
        row["mask_policy"] = weighted_choice(rng, mask_mix)
        selected.append(row)
        formula_counts[formula] += 1
        chemsys_counts[chemsys] += 1
    return selected, {
        "target_count": target_count,
        "selected_count": len(selected),
        "attempts": attempts,
        "skip_counts": dict(skip_counts.most_common()),
        "formula_top30": dict(formula_counts.most_common(30)),
        "chemsys_top30": dict(chemsys_counts.most_common(30)),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    tiers = Counter(str(row.get("sample_weight_tier")) for row in rows)
    roles = Counter(str(row.get("selection_role")) for row in rows)
    reasons = Counter(str(row.get("composition_reason")) for row in rows)
    buckets = Counter(str(row.get("composition_bucket")) for row in rows)
    formulas = Counter(str(row.get("formula")) for row in rows)
    chemsys = Counter(str(row.get("chemsys")) for row in rows)
    return {
        "count": total,
        "tier_counts": dict(tiers.most_common()),
        "selection_role_counts": dict(roles.most_common()),
        "reason_counts": dict(reasons.most_common()),
        "bucket_counts": dict(buckets.most_common()),
        "comp_valid_rate": (
            buckets.get("strict", 0) + buckets.get("all_metal", 0) + buckets.get("single_element", 0)
        )
        / max(1, total),
        "strict_valid_rate": buckets.get("strict", 0) / max(1, total),
        "all_metal_rate": buckets.get("all_metal", 0) / max(1, total),
        "single_element_rate": buckets.get("single_element", 0) / max(1, total),
        "top_formula_counts": dict(formulas.most_common(30)),
        "top_chemsys_counts": dict(chemsys.most_common(30)),
    }


def copy_sidecars(base_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("vocab_tokens.txt", "vocab.json", "stats.json", "tokenizer_report.json", "prompt_pool.json"):
        src = base_dir / name
        if src.exists():
            shutil.copy2(src, output_dir / name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=Path, default=PROJECT_ROOT / "data/dlm_sft/mp_20")
    parser.add_argument("--csv-dir", type=Path, default=PROJECT_ROOT / "reference/crysllmgen/data/mp_20")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--self-improving-jsonl", type=Path, action="append", default=[])
    parser.add_argument("--extra-fraction", type=float, default=0.15)
    parser.add_argument("--self-improving-fraction", type=float, default=0.05)
    parser.add_argument("--include-meta-self-improving", action="store_true")
    parser.add_argument(
        "--self-improving-repeat-with-replacement",
        action="store_true",
        help="Allow generated self-improving rows to be repeated up to formula/chemsys caps.",
    )
    parser.add_argument("--max-formula-repeats", type=int, default=8)
    parser.add_argument("--max-chemsys-repeats", type=int, default=64)
    parser.add_argument("--tier-weights", default="")
    parser.add_argument("--mask-policy-mix", default="")
    parser.add_argument("--eq0-tol", type=float, default=1e-12)
    parser.add_argument("--seed", type=int, default=20260526)
    args = parser.parse_args()

    tier_weights = parse_mix(args.tier_weights, DEFAULT_TIER_WEIGHTS)
    mask_mix = parse_mix(args.mask_policy_mix, DEFAULT_MASK_POLICY_MIX)
    rng = random.Random(args.seed)

    split_summaries: dict[str, Any] = {}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        base_rows = read_jsonl(args.base_dir / f"{split}.jsonl")
        csv_rows, csv_by_material_id = load_csv_metadata(args.csv_dir / f"{split}.csv")
        annotated = [
            annotate_base_row(
                row,
                split=split,
                row_idx=idx,
                csv_rows=csv_rows,
                csv_by_material_id=csv_by_material_id,
                tier_weights=tier_weights,
                eq0_tol=float(args.eq0_tol),
                mask_policy=weighted_choice(rng, mask_mix),
                selection_role="base_preserved",
            )
            for idx, row in enumerate(base_rows)
        ]
        if split == "train":
            extra_target = int(round(len(annotated) * float(args.extra_fraction)))
            repeats, repeat_summary = select_repeats(
                annotated,
                target_count=extra_target,
                tier_weights=tier_weights,
                seed=args.seed + 17,
                max_formula_repeats=args.max_formula_repeats,
                max_chemsys_repeats=args.max_chemsys_repeats,
                mask_mix=mask_mix,
                selection_role="ehull_weighted_repeat",
            )
            generated_rows: list[dict[str, Any]] = []
            for path in args.self_improving_jsonl:
                for idx, row in enumerate(read_jsonl(path)):
                    clean = annotate_generated_row(
                        row,
                        row_idx=idx,
                        tier_weights=tier_weights,
                        mask_policy=weighted_choice(rng, mask_mix),
                        include_meta=bool(args.include_meta_self_improving),
                    )
                    if clean is not None:
                        generated_rows.append(clean)
            generated_target = int(round(len(annotated) * float(args.self_improving_fraction)))
            generated_target_for_selection = (
                generated_target
                if args.self_improving_repeat_with_replacement
                else min(generated_target, len(generated_rows))
            )
            generated_selected, generated_summary = select_repeats(
                generated_rows,
                target_count=generated_target_for_selection,
                tier_weights=tier_weights,
                seed=args.seed + 31,
                max_formula_repeats=args.max_formula_repeats,
                max_chemsys_repeats=args.max_chemsys_repeats,
                mask_mix=mask_mix,
                selection_role="self_improving_repeat",
            )
            output_rows = annotated + repeats + generated_selected
            rng.shuffle(output_rows)
            split_summaries[split] = {
                "base": summarize(annotated),
                "repeats": repeat_summary,
                "self_improving_pool_count": len(generated_rows),
                "self_improving_target_count": generated_target,
                "self_improving_repeat_with_replacement": bool(args.self_improving_repeat_with_replacement),
                "self_improving_selected": generated_summary,
                "output": summarize(output_rows),
            }
        else:
            output_rows = annotated
            split_summaries[split] = {"output": summarize(output_rows)}
        write_jsonl(args.output_dir / f"{split}.jsonl", output_rows)
        write_jsonl(args.output_dir / f"{split}_metadata_cache.jsonl", output_rows)

    copy_sidecars(args.base_dir, args.output_dir)
    summary = {
        "base_dir": str(args.base_dir),
        "csv_dir": str(args.csv_dir),
        "output_dir": str(args.output_dir),
        "self_improving_jsonl": [str(path) for path in args.self_improving_jsonl],
        "extra_fraction": args.extra_fraction,
        "self_improving_fraction": args.self_improving_fraction,
        "include_meta_self_improving": bool(args.include_meta_self_improving),
        "self_improving_repeat_with_replacement": bool(args.self_improving_repeat_with_replacement),
        "tier_weights": tier_weights,
        "mask_policy_mix": mask_mix,
        "max_formula_repeats": args.max_formula_repeats,
        "max_chemsys_repeats": args.max_chemsys_repeats,
        "splits": split_summaries,
        "policy": (
            "MP-20 E_hull=0 and generated strict S.U.N./E_hull<0 are tier_high; "
            "generated meta S.U.N./E_hull<0.1 can be included as a mid-high self-improving tier."
        ),
    }
    write_json(str(args.output_dir / "ehull_weight_summary.json"), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
