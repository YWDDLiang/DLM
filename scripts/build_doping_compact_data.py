#!/usr/bin/env python3
"""Build compact conditional-doping SFT data."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

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


def make_sft_row(
    row: Mapping[str, Any],
    prompt: str,
    split: str,
    copy_index: int,
    reason: str,
) -> Dict[str, Any]:
    parsed = parse_compact_answer(row["answer"])
    answer = compact_answer(parsed["dopants"], parsed["dopant_site_indices"])
    return {
        "prompt": prompt,
        "answer": answer,
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
        },
    }


def expand_train_rows(rows: Iterable[Dict[str, Any]], bins) -> List[Dict[str, Any]]:
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
        prompt = prompt_for_properties(row["properties"], bins)
        expanded.append(make_sft_row(row, prompt, "train", len(expanded), "base"))
        if row["target_band_gap"]:
            expanded.append(make_sft_row(row, prompt, "train", len(expanded), "target_band_gap_x1"))
            expanded.append(make_sft_row(row, prompt, "train", len(expanded), "target_band_gap_x2"))
        if row["name"] in target_top20_names:
            expanded.append(make_sft_row(row, DIRECTED_PROMPT, "train", len(expanded), "top_objective_directed"))
    return expanded


def validate_compact_rows(rows: List[Dict[str, Any]]) -> None:
    if len(rows) != 225:
        raise RuntimeError(f"Expected 225 compact rows, got {len(rows)}")
    site_patterns = Counter(tuple(row["dopant_site_indices"]) for row in rows)
    if set(site_patterns) != {CANONICAL_BSITE_INDICES}:
        raise RuntimeError(f"Unexpected B-site patterns: {dict(site_patterns)}")
    for row in rows:
        parsed = parse_compact_answer(row["answer"])
        if parsed["dopants"] != list(normalize_combo(row["dopants"])):
            raise RuntimeError(f"Dopant mismatch for {row.get('name')}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, default=PROJECT_ROOT / "data/doping_crystal/compact_records.jsonl")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data/doping_compact")
    parser.add_argument("--seed", type=int, default=20260519)
    args = parser.parse_args()

    rows = read_jsonl(args.input_jsonl)
    validate_compact_rows(rows)
    bins = property_bins_from_rows(rows)
    splits = stratified_split(rows, args.seed)

    train_rows = expand_train_rows(splits["train"], bins)
    val_rows = [
        make_sft_row(row, prompt_for_properties(row["properties"], bins), "val", idx, "base")
        for idx, row in enumerate(splits["val"])
    ]
    test_rows = [
        make_sft_row(row, prompt_for_properties(row["properties"], bins), "test", idx, "base")
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
        "canonical_bsite_indices": list(CANONICAL_BSITE_INDICES),
        "directed_prompt": DIRECTED_PROMPT,
        "property_bins": bins.to_dict(),
        "split_counts_base": {key: len(value) for key, value in splits.items()},
        "split_target_counts_base": {
            key: sum(bool(row["target_band_gap"]) for row in value)
            for key, value in splits.items()
        },
        "train_count_after_oversampling": len(train_rows),
        "answer_token_count": 6,
        "vocab_token_count": len(compact_special_tokens()),
    }
    write_json(args.output_dir / "stats.json", stats)
    print(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
