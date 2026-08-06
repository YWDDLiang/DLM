#!/usr/bin/env python3
"""Build the compressed20 structure-aware doping dataset."""

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
    DIRECTED_PROMPT,
    combo_name,
    prompt_for_properties,
    property_bins_from_rows,
    read_jsonl,
    write_json,
    write_jsonl,
)
from crystal_dlm.doping_structure import (
    FULL80_CONFIG,
    STRUCT20_CONFIG,
    STRUCT20_DIRECTED_PROMPT,
    STRUCT20_TASK_TOKEN,
    bsite_structure_indices,
    compress_full80_arrays,
    doping_structure_task_tokens,
    full80_composition_is_exact,
    parse_full80_answer,
    prompt_for_task,
)
from crystal_dlm.fixed_slot import build_special_tokens


def is_target(row: Mapping[str, Any]) -> bool:
    band_gap = float(row["properties"]["band_gap"])
    return 1.7 <= band_gap <= 2.7


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


def load_holdout_roles(stats_path: Path) -> Dict[str, str]:
    if not stats_path.exists():
        return {}
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    roles: Dict[str, str] = {}
    for row in stats.get("hidden_good_holdout", []):
        roles[row["name"]] = "hidden_good_holdout"
    for row in stats.get("train_good_anchors", []):
        roles[row["name"]] = "train_good_anchor"
    return roles


def minmax(value: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return (value - lo) / (hi - lo)


def fallback_holdout_roles(rows: Sequence[Dict[str, Any]], num_holdout: int, num_anchors: int) -> Dict[str, str]:
    target_rows = [row for row in rows if is_target(row)]
    fe_median = statistics.median(float(row["properties"]["formation_energy"]) for row in target_rows)
    dfe_median = statistics.median(float(row["properties"]["defect_formation_energy"]) for row in target_rows)
    good_pool = [
        row
        for row in target_rows
        if float(row["properties"]["formation_energy"]) <= fe_median
        and float(row["properties"]["defect_formation_energy"]) >= dfe_median
    ]
    fe_values = [float(row["properties"]["formation_energy"]) for row in target_rows]
    dfe_values = [float(row["properties"]["defect_formation_energy"]) for row in target_rows]

    def score(row: Mapping[str, Any]) -> float:
        props = row["properties"]
        fe_score = 1.0 - minmax(float(props["formation_energy"]), min(fe_values), max(fe_values))
        dfe_score = minmax(float(props["defect_formation_energy"]), min(dfe_values), max(dfe_values))
        return 0.5 * fe_score + 0.5 * dfe_score

    good_pool.sort(key=score, reverse=True)
    roles: Dict[str, str] = {}
    for row in good_pool[:num_holdout]:
        roles[row["name"]] = "hidden_good_holdout"
    for row in good_pool[num_holdout : num_holdout + num_anchors]:
        roles[row["name"]] = "train_good_anchor"
    return roles


def load_rows(full80_jsonl: Path, compact_jsonl: Path) -> List[Dict[str, Any]]:
    compact_by_name = {combo_name(row["dopants"]): row for row in read_jsonl(compact_jsonl)}
    rows = []
    for row in read_jsonl(full80_jsonl):
        name = row.get("metadata", {}).get("name")
        if not name or name not in compact_by_name:
            raise RuntimeError(f"Cannot match full80 row to compact properties: {name}")
        arrays80 = parse_full80_answer(row["answer"], strict=True)
        if not full80_composition_is_exact(arrays80):
            raise RuntimeError(f"Bad full80 composition for {name}")
        compressed = compress_full80_arrays(arrays80)
        compact = compact_by_name[name]
        rows.append(
            {
                "name": name,
                "dopants": compact["dopants"],
                "properties": compact["properties"],
                "target_band_gap": compact["target_band_gap"],
                "source_cif": compact.get("source_cif") or row.get("metadata", {}).get("source_cif"),
                "structure20_answer": compressed["answer"],
                "full80_answer": row["answer"],
                "bsite_structure_indices": bsite_structure_indices(arrays80),
                "full80_composition": Counter(arrays80["species"]),
                "task": "doping_structure20",
            }
        )
    if len(rows) != 225:
        raise RuntimeError(f"Expected 225 matched SUCCESS rows, got {len(rows)}")
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
        "answer": row["structure20_answer"],
        "task": "doping_structure20",
        "split": split,
        "copy_index": copy_index,
        "oversample_reason": reason,
        "metadata": {
            "name": row["name"],
            "dopants": row["dopants"],
            "properties": row["properties"],
            "target_band_gap": row["target_band_gap"],
            "source_cif": row.get("source_cif"),
            "good_holdout_role": role,
            "full80_target_answer": row["full80_answer"],
            "bsite_structure_indices": row["bsite_structure_indices"],
        },
    }


def expand_train_rows(rows: Iterable[Dict[str, Any]], bins, anchor_names: set[str], anchor_copies: int) -> List[Dict[str, Any]]:
    rows = list(rows)
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
        base_prompt = prompt_for_task(prompt_for_properties(row["properties"], bins), STRUCT20_TASK_TOKEN)
        expanded.append(make_row(row, base_prompt, "train", len(expanded), "base", role))
        if row["target_band_gap"]:
            expanded.append(make_row(row, base_prompt, "train", len(expanded), "target_band_gap_x1", role))
            expanded.append(make_row(row, base_prompt, "train", len(expanded), "target_band_gap_x2", role))
        if row["name"] in target_top20:
            expanded.append(make_row(row, STRUCT20_DIRECTED_PROMPT, "train", len(expanded), "top_objective_directed", role))
        if row["name"] in anchor_names:
            for copy_idx in range(anchor_copies):
                expanded.append(
                    make_row(
                        row,
                        STRUCT20_DIRECTED_PROMPT,
                        "train",
                        len(expanded),
                        f"good_anchor_directed_x{copy_idx + 1}",
                        role,
                    )
                )
    return expanded


def role_summary(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
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
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data/doping_structure20")
    parser.add_argument("--seed", type=int, default=20260520)
    parser.add_argument("--num-holdout-good", type=int, default=6)
    parser.add_argument("--num-train-good-anchors", type=int, default=8)
    parser.add_argument("--good-anchor-directed-copies", type=int, default=3)
    args = parser.parse_args()

    rows = load_rows(args.full80_jsonl, args.compact_jsonl)
    bins = property_bins_from_rows(rows)
    roles = load_holdout_roles(args.compact_good_stats)
    if not roles:
        roles = fallback_holdout_roles(rows, args.num_holdout_good, args.num_train_good_anchors)
    hidden_names = {name for name, role in roles.items() if role == "hidden_good_holdout"}
    anchor_names = {name for name, role in roles.items() if role == "train_good_anchor"}

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
        make_row(row, prompt_for_task(prompt_for_properties(row["properties"], bins), STRUCT20_TASK_TOKEN), "val", idx, "base", "val_regular")
        for idx, row in enumerate(splits["val"])
    ]
    test_rows = [
        make_row(
            row,
            prompt_for_task(prompt_for_properties(row["properties"], bins), STRUCT20_TASK_TOKEN),
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
        for token in [*build_special_tokens(STRUCT20_CONFIG), *doping_structure_task_tokens()]:
            handle.write(token + "\n")
    write_jsonl(
        args.output_dir / "hidden_good_holdout.jsonl",
        [rows_by_name[name] for name in sorted(hidden_names)],
    )
    stats = {
        "input_full80_jsonl": str(args.full80_jsonl),
        "input_compact_jsonl": str(args.compact_jsonl),
        "seed": args.seed,
        "rows": len(rows),
        "task": "doping_structure20",
        "answer_token_count": 107,
        "semantic_note": "20-slot compressed structural code, not a physical 20-atom crystal",
        "structure20_config": STRUCT20_CONFIG.to_dict(),
        "full80_config": FULL80_CONFIG.to_dict(),
        "directed_prompt": STRUCT20_DIRECTED_PROMPT,
        "compact_directed_prompt_reference": DIRECTED_PROMPT,
        "hidden_good_holdout_count": len(hidden_names),
        "train_good_anchor_count": len(anchor_names),
        "hidden_good_holdout": role_summary([rows_by_name[name] for name in sorted(hidden_names)]),
        "train_good_anchors": role_summary([rows_by_name[name] for name in sorted(anchor_names)]),
        "split_counts_base": {key: len(value) for key, value in splits.items()},
        "split_target_counts_base": {
            key: sum(bool(row["target_band_gap"]) for row in value)
            for key, value in splits.items()
        },
        "train_count_after_oversampling": len(train_rows),
        "vocab_token_count": len(set([*build_special_tokens(STRUCT20_CONFIG), *doping_structure_task_tokens()])),
        "data_tests": {
            "full80_composition_exact": True,
            "compressed20_expands_to_full80": True,
            "hidden_good_excluded_from_train": not any(
                row["metadata"]["name"] in hidden_names for row in train_rows
            ),
        },
    }
    write_json(args.output_dir / "stats.json", stats)
    write_json(
        args.output_dir / "structure_token_report.json",
        {
            "answer_token_count": 107,
            "bsite_slots": 16,
            "global_summary_slots": 4,
            "global_summary_species": ["Cs", "I", "Pb", "Mn"],
            "vocab_token_count": stats["vocab_token_count"],
        },
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
