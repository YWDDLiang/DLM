#!/usr/bin/env python3
"""Build a compact doping dataset with hidden high-quality holdout rows."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from crystal_dlm.doping import (
    CANONICAL_BSITE_INDICES,
    DIRECTED_PROMPT,
    compact_answer,
    compact_special_tokens,
    combo_name,
    normalize_combo,
    parse_compact_answer,
    prompt_for_properties,
    property_bins_from_rows,
    read_jsonl,
    write_json,
    write_jsonl,
)


def is_target_bandgap(row: Mapping[str, Any]) -> bool:
    band_gap = float(row["properties"]["band_gap"])
    return 1.7 <= band_gap <= 2.7


def validate_compact_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    if len(rows) != 225:
        raise RuntimeError(f"Expected 225 compact rows, got {len(rows)}")
    site_patterns = Counter(tuple(row["dopant_site_indices"]) for row in rows)
    if set(site_patterns) != {CANONICAL_BSITE_INDICES}:
        raise RuntimeError(f"Unexpected B-site patterns: {dict(site_patterns)}")
    for row in rows:
        parsed = parse_compact_answer(row["answer"])
        if parsed["dopants"] != list(normalize_combo(row["dopants"])):
            raise RuntimeError(f"Dopant mismatch for {row.get('name')}")


def stratified_split(rows: List[Dict[str, Any]], seed: int) -> Dict[str, List[Dict[str, Any]]]:
    rng = random.Random(seed)
    by_label = {True: [], False: []}
    for row in rows:
        by_label[bool(row["target_band_gap"])].append(row)
    splits = {"train": [], "val": [], "test": []}
    for label_rows in by_label.values():
        rng.shuffle(label_rows)
        n_rows = len(label_rows)
        n_val = round(n_rows * 0.15)
        n_test = round(n_rows * 0.15)
        splits["val"].extend(label_rows[:n_val])
        splits["test"].extend(label_rows[n_val : n_val + n_test])
        splits["train"].extend(label_rows[n_val + n_test :])
    for split_rows in splits.values():
        rng.shuffle(split_rows)
    return splits


def minmax(value: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return (value - lo) / (hi - lo)


def good_score(row: Mapping[str, Any], target_rows: Sequence[Mapping[str, Any]]) -> float:
    properties = row["properties"]
    fe_values = [float(item["properties"]["formation_energy"]) for item in target_rows]
    dfe_values = [float(item["properties"]["defect_formation_energy"]) for item in target_rows]
    bg_values = [float(item["properties"]["band_gap"]) for item in target_rows]
    fe_score = 1.0 - minmax(float(properties["formation_energy"]), min(fe_values), max(fe_values))
    dfe_score = minmax(float(properties["defect_formation_energy"]), min(dfe_values), max(dfe_values))
    bg_center_span = max(0.5, max(abs(value - 2.2) for value in bg_values))
    bg_score = 1.0 - min(1.0, abs(float(properties["band_gap"]) - 2.2) / bg_center_span)
    return 0.45 * fe_score + 0.45 * dfe_score + 0.10 * bg_score


def select_good_rows(
    rows: Sequence[Dict[str, Any]],
    num_holdout: int,
    num_train_anchors: int,
) -> Dict[str, Any]:
    target_rows = [row for row in rows if is_target_bandgap(row)]
    fe_median = statistics.median(float(row["properties"]["formation_energy"]) for row in target_rows)
    dfe_median = statistics.median(float(row["properties"]["defect_formation_energy"]) for row in target_rows)
    good_pool = [
        row
        for row in target_rows
        if float(row["properties"]["formation_energy"]) <= fe_median
        and float(row["properties"]["defect_formation_energy"]) >= dfe_median
    ]
    good_pool.sort(key=lambda row: good_score(row, target_rows), reverse=True)
    if len(good_pool) < num_holdout + num_train_anchors:
        raise RuntimeError(
            f"Need at least {num_holdout + num_train_anchors} balanced-good rows, got {len(good_pool)}"
        )

    selected = good_pool[: num_holdout + num_train_anchors]
    holdout_rows = selected[0::2][:num_holdout]
    holdout_names = {row["name"] for row in holdout_rows}
    anchor_rows = [row for row in selected if row["name"] not in holdout_names]
    if len(anchor_rows) < num_train_anchors:
        anchor_rows.extend(row for row in good_pool if row["name"] not in holdout_names and row not in anchor_rows)
    anchor_rows = anchor_rows[:num_train_anchors]

    return {
        "target_count": len(target_rows),
        "good_pool": good_pool,
        "formation_energy_target_median": fe_median,
        "defect_formation_energy_target_median": dfe_median,
        "hidden_good_holdout": holdout_rows,
        "train_good_anchors": anchor_rows,
    }


def make_sft_row(
    row: Mapping[str, Any],
    prompt: str,
    split: str,
    copy_index: int,
    reason: str,
    role: str,
) -> Dict[str, Any]:
    parsed = parse_compact_answer(row["answer"])
    return {
        "prompt": prompt,
        "answer": compact_answer(parsed["dopants"], parsed["dopant_site_indices"]),
        "task": "doping_compact",
        "split": split,
        "copy_index": copy_index,
        "oversample_reason": reason,
        "metadata": {
            "name": combo_name(parsed["dopants"]),
            "dopants": parsed["dopants"],
            "dopant_site_indices": parsed["dopant_site_indices"],
            "properties": row["properties"],
            "target_band_gap": row["target_band_gap"],
            "source_cif": row.get("source_cif"),
            "good_holdout_role": role,
        },
    }


def expand_train_rows(
    rows: Iterable[Dict[str, Any]],
    bins,
    anchor_names: set[str],
    anchor_directed_copies: int,
) -> List[Dict[str, Any]]:
    rows = list(rows)
    target_top20_names = {
        row["name"]
        for row in sorted(
            [item for item in rows if item["target_band_gap"]],
            key=lambda item: (
                float(item["properties"]["formation_energy"]),
                -float(item["properties"]["defect_formation_energy"]),
            ),
        )[:20]
    }
    expanded: List[Dict[str, Any]] = []
    for row in rows:
        role = "train_good_anchor" if row["name"] in anchor_names else "train_regular"
        prompt = prompt_for_properties(row["properties"], bins)
        expanded.append(make_sft_row(row, prompt, "train", len(expanded), "base", role))
        if row["target_band_gap"]:
            expanded.append(make_sft_row(row, prompt, "train", len(expanded), "target_band_gap_x1", role))
            expanded.append(make_sft_row(row, prompt, "train", len(expanded), "target_band_gap_x2", role))
        if row["name"] in target_top20_names:
            expanded.append(make_sft_row(row, DIRECTED_PROMPT, "train", len(expanded), "top_objective_directed", role))
        if row["name"] in anchor_names:
            for copy_idx in range(anchor_directed_copies):
                expanded.append(
                    make_sft_row(
                        row,
                        DIRECTED_PROMPT,
                        "train",
                        len(expanded),
                        f"good_anchor_directed_x{copy_idx + 1}",
                        role,
                    )
                )
    return expanded


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "name": row["name"],
            "dopants": row["dopants"],
            "band_gap": row["properties"]["band_gap"],
            "formation_energy": row["properties"]["formation_energy"],
            "defect_formation_energy": row["properties"]["defect_formation_energy"],
        }
        for row in rows
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, default=PROJECT_ROOT / "data/doping_crystal/compact_records.jsonl")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data/doping_compact_good_holdout")
    parser.add_argument("--seed", type=int, default=20260519)
    parser.add_argument("--num-holdout-good", type=int, default=6)
    parser.add_argument("--num-train-good-anchors", type=int, default=8)
    parser.add_argument("--good-anchor-directed-copies", type=int, default=3)
    args = parser.parse_args()

    rows = read_jsonl(args.input_jsonl)
    validate_compact_rows(rows)
    bins = property_bins_from_rows(rows)
    good_selection = select_good_rows(rows, args.num_holdout_good, args.num_train_good_anchors)
    holdout_names = {row["name"] for row in good_selection["hidden_good_holdout"]}
    anchor_names = {row["name"] for row in good_selection["train_good_anchors"]}

    rows_by_name = {row["name"]: row for row in rows}
    remaining_rows = [row for row in rows if row["name"] not in holdout_names and row["name"] not in anchor_names]
    splits = stratified_split(remaining_rows, args.seed)
    splits["train"].extend(rows_by_name[name] for name in sorted(anchor_names))
    splits["test"].extend(rows_by_name[name] for name in sorted(holdout_names))
    rng = random.Random(args.seed)
    rng.shuffle(splits["train"])
    rng.shuffle(splits["test"])

    train_rows = expand_train_rows(
        splits["train"],
        bins,
        anchor_names=anchor_names,
        anchor_directed_copies=args.good_anchor_directed_copies,
    )
    val_rows = [
        make_sft_row(row, prompt_for_properties(row["properties"], bins), "val", idx, "base", "val_regular")
        for idx, row in enumerate(splits["val"])
    ]
    test_rows = [
        make_sft_row(
            row,
            prompt_for_properties(row["properties"], bins),
            "test",
            idx,
            "base",
            "hidden_good_holdout" if row["name"] in holdout_names else "test_regular",
        )
        for idx, row in enumerate(splits["test"])
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "train.jsonl", train_rows)
    write_jsonl(args.output_dir / "val.jsonl", val_rows)
    write_jsonl(args.output_dir / "test.jsonl", test_rows)
    with (args.output_dir / "vocab_tokens.txt").open("w", encoding="utf-8") as handle:
        for token in compact_special_tokens():
            handle.write(token + "\n")

    stats = {
        "input_jsonl": str(args.input_jsonl),
        "seed": args.seed,
        "rows": len(rows),
        "candidate_space": 286,
        "answer_token_count": 6,
        "vocab_token_count": len(compact_special_tokens()),
        "canonical_bsite_indices": list(CANONICAL_BSITE_INDICES),
        "directed_prompt": DIRECTED_PROMPT,
        "property_bins": bins.to_dict(),
        "good_definition": {
            "band_gap_range": [1.7, 2.7],
            "formation_energy_max": good_selection["formation_energy_target_median"],
            "defect_formation_energy_min": good_selection["defect_formation_energy_target_median"],
            "description": "target bandgap, formation_energy <= target median, defect_formation_energy >= target median",
        },
        "target_count": good_selection["target_count"],
        "good_pool_count": len(good_selection["good_pool"]),
        "hidden_good_holdout_count": len(holdout_names),
        "train_good_anchor_count": len(anchor_names),
        "hidden_good_holdout": summarize_rows(good_selection["hidden_good_holdout"]),
        "train_good_anchors": summarize_rows(good_selection["train_good_anchors"]),
        "split_counts_base": {key: len(value) for key, value in splits.items()},
        "split_target_counts_base": {
            key: sum(bool(row["target_band_gap"]) for row in value)
            for key, value in splits.items()
        },
        "train_count_after_oversampling": len(train_rows),
        "good_anchor_directed_copies": args.good_anchor_directed_copies,
    }
    write_json(args.output_dir / "stats.json", stats)
    print(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
