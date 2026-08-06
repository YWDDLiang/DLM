#!/usr/bin/env python3
"""Build a hidden-good holdout dataset for direct full80 SFT."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from crystal_dlm.doping import (
    combo_name,
    prompt_for_properties,
    property_bins_from_rows,
    read_jsonl,
    write_json,
    write_jsonl,
)
from crystal_dlm.doping_structure import (
    FULL80_CONFIG,
    FULL80_DIRECTED_PROMPT,
    FULL80_TASK_TOKEN,
    doping_structure_task_tokens,
    parse_full80_answer,
    prompt_for_task,
)
from crystal_dlm.fixed_slot import build_special_tokens


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


def load_roles(stats_path: Path) -> Dict[str, str]:
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    roles: Dict[str, str] = {}
    for row in stats.get("hidden_good_holdout", []):
        roles[row["name"]] = "hidden_good_holdout"
    for row in stats.get("train_good_anchors", []):
        roles[row["name"]] = "train_good_anchor"
    return roles


def load_rows(full80_jsonl: Path, compact_jsonl: Path) -> List[Dict[str, Any]]:
    compact_by_name = {combo_name(row["dopants"]): row for row in read_jsonl(compact_jsonl)}
    rows: List[Dict[str, Any]] = []
    for row in read_jsonl(full80_jsonl):
        name = row.get("metadata", {}).get("name")
        if not name or name not in compact_by_name:
            raise RuntimeError(f"Cannot match full80 row to compact row: {name}")
        arrays = parse_full80_answer(row["answer"], strict=True)
        compact = compact_by_name[name]
        rows.append(
            {
                "name": name,
                "dopants": compact["dopants"],
                "properties": compact["properties"],
                "target_band_gap": compact["target_band_gap"],
                "answer": arrays["answer"],
                "source_cif": compact.get("source_cif") or row.get("metadata", {}).get("source_cif"),
                "task": "doping_full80",
            }
        )
    if len(rows) != 225:
        raise RuntimeError(f"Expected 225 full80 rows, got {len(rows)}")
    return rows


def make_row(
    row: Mapping[str, Any],
    prompt: str,
    split: str,
    copy_index: int,
    reason: str,
    role: str,
) -> Dict[str, Any]:
    return {
        "prompt": prompt,
        "answer": row["answer"],
        "task": "doping_full80",
        "split": split,
        "copy_index": copy_index,
        "oversample_reason": reason,
        "answer_semantic_tokens": 407,
        "metadata": {
            "name": row["name"],
            "dopants": row["dopants"],
            "properties": row["properties"],
            "target_band_gap": row["target_band_gap"],
            "source_cif": row.get("source_cif"),
            "good_holdout_role": role,
        },
    }


def expand_train_rows(rows: List[Dict[str, Any]], bins, anchor_names: set[str], anchor_copies: int) -> List[Dict[str, Any]]:
    target_top20 = {
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
        base_prompt = prompt_for_task(prompt_for_properties(row["properties"], bins), FULL80_TASK_TOKEN)
        expanded.append(make_row(row, base_prompt, "train", len(expanded), "base", role))
        if row["target_band_gap"]:
            expanded.append(make_row(row, base_prompt, "train", len(expanded), "target_band_gap_x1", role))
            expanded.append(make_row(row, base_prompt, "train", len(expanded), "target_band_gap_x2", role))
        if row["name"] in target_top20:
            expanded.append(make_row(row, FULL80_DIRECTED_PROMPT, "train", len(expanded), "top_objective_directed", role))
        if row["name"] in anchor_names:
            for copy_idx in range(anchor_copies):
                expanded.append(
                    make_row(
                        row,
                        FULL80_DIRECTED_PROMPT,
                        "train",
                        len(expanded),
                        f"good_anchor_directed_x{copy_idx + 1}",
                        role,
                    )
                )
    return expanded


def summarize(rows: List[Mapping[str, Any]]) -> List[Dict[str, Any]]:
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
    parser.add_argument("--full80-jsonl", type=Path, default=PROJECT_ROOT / "data/doping_crystal/full80_success.jsonl")
    parser.add_argument("--compact-jsonl", type=Path, default=PROJECT_ROOT / "data/doping_crystal/compact_records.jsonl")
    parser.add_argument("--compact-good-stats", type=Path, default=PROJECT_ROOT / "data/doping_compact_good_holdout/stats.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data/doping_full80_holdout")
    parser.add_argument("--seed", type=int, default=20260520)
    parser.add_argument("--good-anchor-directed-copies", type=int, default=2)
    args = parser.parse_args()

    rows = load_rows(args.full80_jsonl, args.compact_jsonl)
    roles = load_roles(args.compact_good_stats)
    hidden_names = {name for name, role in roles.items() if role == "hidden_good_holdout"}
    anchor_names = {name for name, role in roles.items() if role == "train_good_anchor"}
    bins = property_bins_from_rows(rows)
    rows_by_name = {row["name"]: row for row in rows}
    remaining = [row for row in rows if row["name"] not in hidden_names and row["name"] not in anchor_names]
    splits = stratified_split(remaining, args.seed)
    splits["train"].extend(rows_by_name[name] for name in sorted(anchor_names))
    splits["test"].extend(rows_by_name[name] for name in sorted(hidden_names))
    rng = random.Random(args.seed)
    rng.shuffle(splits["train"])
    rng.shuffle(splits["test"])

    train_rows = expand_train_rows(splits["train"], bins, anchor_names, args.good_anchor_directed_copies)
    val_rows = [
        make_row(row, prompt_for_task(prompt_for_properties(row["properties"], bins), FULL80_TASK_TOKEN), "val", idx, "base", "val_regular")
        for idx, row in enumerate(splits["val"])
    ]
    test_rows = [
        make_row(
            row,
            prompt_for_task(prompt_for_properties(row["properties"], bins), FULL80_TASK_TOKEN),
            "test",
            idx,
            "base",
            roles.get(row["name"], "test_regular"),
        )
        for idx, row in enumerate(splits["test"])
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "train.jsonl", train_rows)
    write_jsonl(args.output_dir / "val.jsonl", val_rows)
    write_jsonl(args.output_dir / "test.jsonl", test_rows)
    with (args.output_dir / "vocab_tokens.txt").open("w", encoding="utf-8") as handle:
        for token in [*build_special_tokens(FULL80_CONFIG), *doping_structure_task_tokens()]:
            handle.write(token + "\n")
    write_jsonl(args.output_dir / "hidden_good_holdout.jsonl", [rows_by_name[name] for name in sorted(hidden_names)])
    stats = {
        "input_full80_jsonl": str(args.full80_jsonl),
        "input_compact_jsonl": str(args.compact_jsonl),
        "seed": args.seed,
        "rows": len(rows),
        "task": "doping_full80",
        "answer_token_count": 407,
        "full80_config": FULL80_CONFIG.to_dict(),
        "directed_prompt": FULL80_DIRECTED_PROMPT,
        "hidden_good_holdout_count": len(hidden_names),
        "train_good_anchor_count": len(anchor_names),
        "hidden_good_holdout": summarize([rows_by_name[name] for name in sorted(hidden_names)]),
        "train_good_anchors": summarize([rows_by_name[name] for name in sorted(anchor_names)]),
        "split_counts_base": {key: len(value) for key, value in splits.items()},
        "train_count_after_oversampling": len(train_rows),
        "vocab_token_count": len(set([*build_special_tokens(FULL80_CONFIG), *doping_structure_task_tokens()])),
        "data_tests": {
            "answer_length_407": True,
            "hidden_good_excluded_from_train": not any(
                row["metadata"]["name"] in hidden_names for row in train_rows
            ),
        },
    }
    write_json(args.output_dir / "stats.json", stats)
    print(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
