#!/usr/bin/env python3
"""Build reason-balanced MP-20 SFT data for composition-aware training."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from crystal_dlm.composition_validity import composition_record
from crystal_dlm.fixed_slot import parse_fixed_slot_answer, write_json


DEFAULT_REASON_FRACTIONS = {
    "strict": 0.55,
    "all_metal": 0.35,
    "single_element": 0.02,
    "invalid": 0.08,
}
DEFAULT_MASK_POLICY_MIX = {
    "normal": 0.40,
    "active_element": 0.30,
    "n_active_element": 0.20,
    "active_element_empty": 0.10,
}


def parse_mix(spec: str, defaults: dict[str, float]) -> dict[str, float]:
    if not spec:
        return dict(defaults)
    result: dict[str, float] = {}
    for item in spec.split(","):
        if not item.strip():
            continue
        key, value = item.split("=", 1) if "=" in item else item.split(":", 1)
        result[key.strip()] = float(value)
    total = sum(result.values())
    if total <= 0:
        raise ValueError(f"Invalid mix: {spec}")
    return {key: value / total for key, value in result.items()}


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for row_idx, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            row["_source_idx"] = row_idx
            rows.append(row)
    return rows


def classify_row(row: dict[str, Any]) -> dict[str, Any]:
    try:
        arrays = parse_fixed_slot_answer(str(row.get("answer") or row.get("text") or ""))
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
            "bucket": bucket,
            "reason": reason,
            "comp_valid": bool(comp["comp_valid"]),
            "strict_valid": reason == "charge_neutral_pauling_valid",
            "formula": comp["formula"],
            "num_atoms": comp["num_atoms"],
            "num_elements": comp["num_elements"],
        }
    except Exception as exc:
        return {
            "bucket": "invalid",
            "reason": f"parse_error:{type(exc).__name__}",
            "comp_valid": False,
            "strict_valid": False,
            "formula": None,
            "num_atoms": None,
            "num_elements": None,
            "message": str(exc),
        }


def annotate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    annotated = []
    for row in rows:
        new_row = dict(row)
        comp = classify_row(new_row)
        new_row["_comp"] = comp
        annotated.append(new_row)
    return annotated


def weighted_choice(rng: random.Random, weights: dict[str, float]) -> str:
    keys = list(weights)
    values = [weights[key] for key in keys]
    return str(rng.choices(keys, weights=values, k=1)[0])


def sample_bucket(
    rows: list[dict[str, Any]],
    count: int,
    rng: random.Random,
    *,
    formula_counts: Counter[str] | None = None,
    max_formula_repeats: int | None = None,
) -> list[dict[str, Any]]:
    if count <= 0 or not rows:
        return []
    if count <= len(rows):
        candidates = rng.sample(rows, len(rows))
    else:
        candidates = list(rows)

    if max_formula_repeats is None or max_formula_repeats <= 0:
        if count <= len(rows):
            return candidates[:count]
        return [rng.choice(rows) for _ in range(count)]

    formula_counts = formula_counts if formula_counts is not None else Counter()
    selected: list[dict[str, Any]] = []
    attempts = 0
    max_attempts = max(1000, count * max(20, len(rows)))
    while len(selected) < count and attempts < max_attempts:
        attempts += 1
        row = candidates[(attempts - 1) % len(candidates)] if attempts <= len(candidates) else rng.choice(rows)
        formula = str(row.get("_comp", {}).get("formula") or row.get("formula") or "")
        if formula_counts[formula] >= max_formula_repeats:
            continue
        selected.append(row)
        formula_counts[formula] += 1

    # If a bucket is too small for the requested cap, fill the tail explicitly
    # and report the realized repeats in the summary instead of silently changing
    # the requested dataset size.
    while len(selected) < count:
        row = rng.choice(rows)
        formula = str(row.get("_comp", {}).get("formula") or row.get("formula") or "")
        selected.append(row)
        formula_counts[formula] += 1
    return selected


def target_counts(size: int, fractions: dict[str, float]) -> dict[str, int]:
    counts = {bucket: int(round(size * fractions.get(bucket, 0.0))) for bucket in DEFAULT_REASON_FRACTIONS}
    delta = size - sum(counts.values())
    counts["strict"] += delta
    return counts


def clean_row(
    row: dict[str, Any],
    *,
    mask_policy: str,
    repeat_idx: int | None = None,
    selection_role: str | None = None,
) -> dict[str, Any]:
    clean = {key: value for key, value in row.items() if not key.startswith("_")}
    comp = dict(row.get("_comp") or {})
    clean["composition_reason"] = comp.get("reason")
    clean["composition_bucket"] = comp.get("bucket")
    clean["comp_valid"] = comp.get("comp_valid")
    clean["strict_valid"] = comp.get("strict_valid")
    clean["formula"] = comp.get("formula")
    clean["mask_policy"] = mask_policy
    clean["balance_source_idx"] = row.get("_source_idx")
    if selection_role is not None:
        clean["selection_role"] = selection_role
    if repeat_idx is not None:
        clean["balance_repeat_idx"] = repeat_idx
    return clean


def build_train_split(
    rows: list[dict[str, Any]],
    *,
    output_size: int,
    fractions: dict[str, float],
    mask_mix: dict[str, float],
    seed: int,
    max_formula_repeats: int | None = None,
    preserve_input_train: bool = False,
    extra_train_fraction: float = 0.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    buckets: dict[str, list[dict[str, Any]]] = {key: [] for key in DEFAULT_REASON_FRACTIONS}
    for row in rows:
        buckets[str(row["_comp"]["bucket"])].append(row)
    fallback_rows = buckets["strict"] or rows
    base_output: list[dict[str, Any]] = []
    if preserve_input_train:
        base_rows = list(rows)
        rng.shuffle(base_rows)
        base_output = [
            clean_row(
                row,
                mask_policy=weighted_choice(rng, mask_mix),
                repeat_idx=idx,
                selection_role="base_preserved",
            )
            for idx, row in enumerate(base_rows)
        ]
        extra_size = max(0, int(round(len(rows) * float(extra_train_fraction))))
        counts = target_counts(extra_size, fractions)
    else:
        counts = target_counts(output_size, fractions)
    selected: list[dict[str, Any]] = []
    selected_formula_counts: Counter[str] = Counter()
    for bucket, count in counts.items():
        selected.extend(
            sample_bucket(
                buckets[bucket] or fallback_rows,
                count,
                rng,
                formula_counts=selected_formula_counts,
                max_formula_repeats=max_formula_repeats,
            )
        )
    rng.shuffle(selected)
    selected_output = [
        clean_row(
            row,
            mask_policy=weighted_choice(rng, mask_mix),
            repeat_idx=idx,
            selection_role="selected_extra" if preserve_input_train else "balanced_sample",
        )
        for idx, row in enumerate(selected)
    ]
    output = base_output + selected_output
    rng.shuffle(output)
    summary = summarize_rows(output)
    summary.update(
        {
            "requested_output_size": output_size,
            "actual_output_size": len(output),
            "preserve_input_train": preserve_input_train,
            "preserved_base_count": len(base_output),
            "selected_extra_count": len(selected_output) if preserve_input_train else 0,
            "extra_train_fraction": extra_train_fraction if preserve_input_train else 0.0,
            "target_reason_fractions": fractions,
            "target_reason_counts": counts,
            "source_bucket_counts": {key: len(value) for key, value in buckets.items()},
            "mask_policy_mix": mask_mix,
            "max_formula_repeats": max_formula_repeats,
            "selected_formula_top30": dict(selected_formula_counts.most_common(30)),
            "selected_formula_over_cap_count": (
                sum(
                    1
                    for value in selected_formula_counts.values()
                    if max_formula_repeats is not None
                    and max_formula_repeats > 0
                    and value > max_formula_repeats
                )
            ),
        }
    )
    return output, summary


def annotate_eval_split(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output = [clean_row(row, mask_policy="normal") for row in rows]
    return output, summarize_rows(output)


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    reason_counts = Counter(str(row.get("composition_reason")) for row in rows)
    bucket_counts = Counter(str(row.get("composition_bucket")) for row in rows)
    mask_counts = Counter(str(row.get("mask_policy", "normal")) for row in rows)
    selection_counts = Counter(str(row.get("selection_role", "unknown")) for row in rows)
    total = len(rows)
    return {
        "count": total,
        "bucket_counts": dict(bucket_counts.most_common()),
        "reason_counts": dict(reason_counts.most_common()),
        "mask_policy_counts": dict(mask_counts.most_common()),
        "selection_role_counts": dict(selection_counts.most_common()),
        "strict_valid_rate": bucket_counts.get("strict", 0) / max(1, total),
        "single_element_rate": bucket_counts.get("single_element", 0) / max(1, total),
        "all_metal_rate": bucket_counts.get("all_metal", 0) / max(1, total),
        "invalid_rate": bucket_counts.get("invalid", 0) / max(1, total),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def copy_sidecars(input_dir: Path, output_dir: Path) -> None:
    for name in ("vocab_tokens.txt", "vocab.json", "stats.json", "tokenizer_report.json", "prompt_pool.json"):
        src = input_dir / name
        if src.exists():
            shutil.copy2(src, output_dir / name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=PROJECT_ROOT / "data/dlm_sft/mp_20")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-train-size", type=int, default=None)
    parser.add_argument("--reason-fractions", default="")
    parser.add_argument("--mask-policy-mix", default="")
    parser.add_argument("--max-formula-repeats", type=int, default=None)
    parser.add_argument(
        "--preserve-input-train",
        action="store_true",
        help="Keep every original train row once, then append selected better examples.",
    )
    parser.add_argument(
        "--extra-train-fraction",
        type=float,
        default=0.0,
        help="When preserving input train, append this fraction of selected extra rows.",
    )
    parser.add_argument("--seed", type=int, default=20260521)
    args = parser.parse_args()

    fractions = parse_mix(args.reason_fractions, DEFAULT_REASON_FRACTIONS)
    mask_mix = parse_mix(args.mask_policy_mix, DEFAULT_MASK_POLICY_MIX)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "seed": args.seed,
        "splits": {},
    }
    for split in ("train", "val", "test"):
        rows = annotate_rows(load_rows(args.input_dir / f"{split}.jsonl"))
        if split == "train":
            target_size = int(args.target_train_size or len(rows))
            output, split_summary = build_train_split(
                rows,
                output_size=target_size,
                fractions=fractions,
                mask_mix=mask_mix,
                seed=args.seed,
                max_formula_repeats=args.max_formula_repeats,
                preserve_input_train=args.preserve_input_train,
                extra_train_fraction=args.extra_train_fraction,
            )
        else:
            output, split_summary = annotate_eval_split(rows)
        write_jsonl(args.output_dir / f"{split}.jsonl", output)
        summary["splits"][split] = split_summary
    copy_sidecars(args.input_dir, args.output_dir)
    write_json(str(args.output_dir / "reason_balance_summary.json"), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
