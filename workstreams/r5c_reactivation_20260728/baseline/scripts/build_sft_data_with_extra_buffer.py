#!/usr/bin/env python3
"""Append selected SEAL-style buffer rows to an existing fixed-slot SFT dataset."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from crystal_dlm.composition_validity import composition_record
from crystal_dlm.fixed_slot import arrays_to_answer, parse_fixed_slot_answer, write_json


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            count += 1
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return count


def classify_row(row: dict[str, Any]) -> dict[str, Any]:
    answer = str(row.get("answer") or row.get("response") or row.get("text") or "")
    arrays = parse_fixed_slot_answer(answer)
    comp = composition_record(arrays["atom_types"])
    reason = str(comp["reason"])
    if reason == "charge_neutral_pauling_valid":
        bucket = "strict"
    elif reason == "all_metal_shortcut":
        bucket = "all_metal"
    elif reason == "single_element_shortcut":
        bucket = "single_element"
    else:
        bucket = "invalid"
    return {
        "composition_reason": reason,
        "composition_bucket": bucket,
        "comp_valid": bool(comp["comp_valid"]),
        "strict_valid": reason == "charge_neutral_pauling_valid",
        "formula": comp["formula"],
        "num_atoms": comp["num_atoms"],
        "num_elements": comp["num_elements"],
    }


def normalized_row(row: dict[str, Any], *, selection_role: str | None = None) -> dict[str, Any]:
    clean = dict(row)
    comp = classify_row(clean)
    clean.update(comp)
    if selection_role is not None:
        clean["selection_role"] = selection_role
    if not clean.get("mask_policy"):
        clean["mask_policy"] = "n_active_element" if clean.get("strict_valid") else "active_element"
    return clean


def physically_equivalent_copy(
    row: dict[str, Any],
    *,
    rng: random.Random,
    origin_shift: bool,
    site_permutation: bool,
) -> dict[str, Any]:
    if not origin_shift and not site_permutation:
        return dict(row)
    arrays = parse_fixed_slot_answer(str(row["answer"]), strict=True)
    species = list(arrays["species"])
    coords = [list(coord) for coord in arrays["frac_coords"]]
    if site_permutation and len(species) > 1:
        order = list(range(len(species)))
        rng.shuffle(order)
        species = [species[idx] for idx in order]
        coords = [coords[idx] for idx in order]
    shift = [0.0, 0.0, 0.0]
    if origin_shift:
        shift = [rng.random(), rng.random(), rng.random()]
        coords = [[(float(value) + shift[axis]) % 1.0 for axis, value in enumerate(coord)] for coord in coords]
    answer, _ = arrays_to_answer(arrays["lengths"], arrays["angles"], species, coords)
    clean = dict(row)
    clean["answer"] = answer
    prompt = str(clean.get("prompt") or "")
    if prompt:
        clean["text"] = prompt.rstrip() + "\n" + answer
    clean["physical_augmentation"] = {
        "origin_shift": shift if origin_shift else None,
        "site_permutation": bool(site_permutation),
    }
    return clean


def select_extra_rows(
    rows: list[dict[str, Any]],
    *,
    target_count: int,
    seed: int,
    max_formula_repeats: int | None,
    accepted_buckets: set[str],
    repeat_to_target: bool,
    augmentation_mask_policies: list[str],
    augment_origin_shift: bool,
    augment_site_permutation: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    selected_unique: list[dict[str, Any]] = []
    formula_counts: Counter[str] = Counter()
    skip_counts: Counter[str] = Counter()
    for row in shuffled:
        if len(selected_unique) >= target_count:
            break
        try:
            clean = normalized_row(row, selection_role="seal_extra")
        except Exception as exc:
            skip_counts[f"parse:{type(exc).__name__}"] += 1
            continue
        bucket = str(clean.get("composition_bucket"))
        if bucket not in accepted_buckets:
            skip_counts[f"bucket_rejected:{bucket}:{clean.get('composition_reason')}"] += 1
            continue
        formula = str(clean.get("formula") or "")
        if max_formula_repeats is not None and max_formula_repeats > 0:
            if formula_counts[formula] >= max_formula_repeats:
                skip_counts["formula_repeat_cap"] += 1
                continue
        formula_counts[formula] += 1
        selected_unique.append(clean)

    selected: list[dict[str, Any]] = list(selected_unique)
    augmented_count = 0
    if repeat_to_target and selected_unique and len(selected_unique) < target_count:
        policies = [policy for policy in augmentation_mask_policies if policy]
        if not policies:
            policies = ["n_active_element", "active_element", "active_element_empty", "normal"]
        selected = []
        for repeat_idx in range(target_count):
            base = physically_equivalent_copy(
                selected_unique[repeat_idx % len(selected_unique)],
                rng=rng,
                origin_shift=augment_origin_shift and repeat_idx >= len(selected_unique),
                site_permutation=augment_site_permutation and repeat_idx >= len(selected_unique),
            )
            base["selection_role"] = "seal_extra" if repeat_idx < len(selected_unique) else "seal_extra_augmented"
            base["mask_policy"] = policies[repeat_idx % len(policies)]
            base["augmentation_copy_idx"] = repeat_idx // len(selected_unique)
            selected.append(base)
        augmented_count = max(0, len(selected) - len(selected_unique))
    return selected, {
        "target_count": target_count,
        "selected_count": len(selected),
        "selected_unique_count": len(selected_unique),
        "augmented_count": augmented_count,
        "selected_formula_top30": dict(formula_counts.most_common(30)),
        "skip_counts": dict(skip_counts.most_common()),
        "accepted_buckets": sorted(accepted_buckets),
        "repeat_to_target": bool(repeat_to_target),
        "augmentation_mask_policies": augmentation_mask_policies,
        "augment_origin_shift": bool(augment_origin_shift),
        "augment_site_permutation": bool(augment_site_permutation),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    reasons = Counter(str(row.get("composition_reason")) for row in rows)
    buckets = Counter(str(row.get("composition_bucket")) for row in rows)
    roles = Counter(str(row.get("selection_role", "unknown")) for row in rows)
    masks = Counter(str(row.get("mask_policy", "normal")) for row in rows)
    total = len(rows)
    return {
        "count": total,
        "reason_counts": dict(reasons.most_common()),
        "bucket_counts": dict(buckets.most_common()),
        "selection_role_counts": dict(roles.most_common()),
        "mask_policy_counts": dict(masks.most_common()),
        "comp_valid_rate": (
            buckets.get("strict", 0) + buckets.get("all_metal", 0) + buckets.get("single_element", 0)
        )
        / max(1, total),
        "strict_valid_rate": buckets.get("strict", 0) / max(1, total),
        "single_element_rate": buckets.get("single_element", 0) / max(1, total),
        "all_metal_rate": buckets.get("all_metal", 0) / max(1, total),
        "invalid_rate": buckets.get("invalid", 0) / max(1, total),
    }


def copy_sidecars(base_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("vocab_tokens.txt", "vocab.json", "stats.json", "tokenizer_report.json", "prompt_pool.json"):
        src = base_dir / name
        if src.exists():
            shutil.copy2(src, output_dir / name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--extra-jsonl", type=Path, action="append", required=True)
    parser.add_argument("--extra-fraction", type=float, default=0.15)
    parser.add_argument("--max-extra", type=int, default=None)
    parser.add_argument("--max-formula-repeats", type=int, default=8)
    parser.add_argument(
        "--repeat-extra-to-target",
        action="store_true",
        help=(
            "Repeat selected extra rows with different mask policies until extra_fraction is reached. "
            "Use only with refined/accepted buffers; raw buffers remain guarded separately."
        ),
    )
    parser.add_argument(
        "--augmentation-mask-policies",
        default="n_active_element,active_element,active_element_empty,normal",
        help="Comma-separated mask-policy cycle used when --repeat-extra-to-target is set.",
    )
    parser.add_argument(
        "--augment-extra-origin-shift",
        action="store_true",
        help="For repeated refined-buffer rows, apply a global PBC origin shift before fixed-slot encoding.",
    )
    parser.add_argument(
        "--augment-extra-site-permutation",
        action="store_true",
        help="For repeated refined-buffer rows, permute active sites before fixed-slot encoding.",
    )
    parser.add_argument(
        "--accepted-buckets",
        default="strict",
        help="Comma-separated extra-buffer composition buckets to mix: strict,all_metal,single_element.",
    )
    parser.add_argument("--seed", type=int, default=20260521)
    args = parser.parse_args()

    train_rows = [normalized_row(row, selection_role=row.get("selection_role") or "base_preserved") for row in read_jsonl(args.base_dir / "train.jsonl")]
    extra_rows: list[dict[str, Any]] = []
    for path in args.extra_jsonl:
        extra_rows.extend(read_jsonl(path))
    target_count = int(round(len(train_rows) * float(args.extra_fraction)))
    if args.max_extra is not None:
        target_count = min(target_count, int(args.max_extra))
    selected_extra, extra_summary = select_extra_rows(
        extra_rows,
        target_count=target_count,
        seed=args.seed,
        max_formula_repeats=args.max_formula_repeats,
        accepted_buckets={item.strip() for item in args.accepted_buckets.split(",") if item.strip()},
        repeat_to_target=bool(args.repeat_extra_to_target),
        augmentation_mask_policies=[
            item.strip() for item in args.augmentation_mask_policies.split(",") if item.strip()
        ],
        augment_origin_shift=bool(args.augment_extra_origin_shift),
        augment_site_permutation=bool(args.augment_extra_site_permutation),
    )
    combined = train_rows + selected_extra
    random.Random(args.seed).shuffle(combined)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "train.jsonl", combined)
    for split in ("val", "test"):
        src = args.base_dir / f"{split}.jsonl"
        dst = args.output_dir / f"{split}.jsonl"
        if src.exists():
            shutil.copy2(src, dst)
    copy_sidecars(args.base_dir, args.output_dir)
    summary = {
        "base_dir": str(args.base_dir),
        "output_dir": str(args.output_dir),
        "extra_jsonl": [str(path) for path in args.extra_jsonl],
        "extra_fraction": args.extra_fraction,
        "max_extra": args.max_extra,
        "max_formula_repeats": args.max_formula_repeats,
        "accepted_buckets": args.accepted_buckets,
        "train": summarize(combined),
        "base_train": summarize(train_rows),
        "extra_selection": extra_summary,
    }
    write_json(str(args.output_dir / "seal_mix_summary.json"), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
