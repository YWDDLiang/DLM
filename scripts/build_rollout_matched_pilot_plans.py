#!/usr/bin/env python3
"""Freeze 128 train and 128 holdout Plans for rollout-matched DLM pilot."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random

from crystal_dlm.dynamic_crystal import parse_dynamic_answer


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def n_bin(value: int) -> str:
    if value <= 5:
        return "N02_05"
    if value <= 10:
        return "N06_10"
    if value <= 15:
        return "N11_15"
    return "N16_20"


def arity_bin(value: int) -> str:
    return str(value) if value <= 3 else "4plus"


def select_rows(rows: list[dict], *, count: int, seed: int) -> list[dict]:
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    seen_compositions: set[str] = set()
    for index in indices:
        row = rows[index]
        identity = str(row["reduced_composition_identity"])
        if identity in seen_compositions:
            continue
        plan = dict(row["plan_state"])
        count_atoms = int(plan["N"])
        arity = len(plan["elements"])
        if not 2 <= count_atoms <= 20:
            continue
        parse_dynamic_answer(str(row["answer"]), strict=True)
        seen_compositions.add(identity)
        groups[(n_bin(count_atoms), arity_bin(arity))].append(row)
    selected: list[dict] = []
    positions = defaultdict(int)
    keys = sorted(groups)
    while len(selected) < count:
        progressed = False
        for key in keys:
            position = positions[key]
            if position < len(groups[key]):
                selected.append(groups[key][position])
                positions[key] += 1
                progressed = True
                if len(selected) == count:
                    break
        if not progressed:
            raise RuntimeError("insufficient unique stratified pilot rows")
    return selected


def plan_rows(rows: list[dict], split: str) -> list[dict]:
    result = []
    for sample_idx, source in enumerate(rows):
        result.append(
            {
                "schema": "rollout_matched_pilot_plan_v1",
                "sample_idx": sample_idx,
                "source_sample_idx": int(source["source_row_idx"]),
                "source_row_idx": int(source["source_row_idx"]),
                "split": split,
                "prompt": str(source["prompt"]),
                "plan_state": dict(source["plan_state"]),
                "reduced_composition_identity": str(
                    source["reduced_composition_identity"]
                ),
                "target_answer": str(source["answer"]),
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-sft-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260903)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    source_rows = read_jsonl(args.teacher_sft_jsonl.resolve())
    selected = select_rows(source_rows, count=256, seed=int(args.seed))
    train = plan_rows(selected[::2], "train")
    holdout = plan_rows(selected[1::2], "holdout")
    train_ids = {row["reduced_composition_identity"] for row in train}
    holdout_ids = {row["reduced_composition_identity"] for row in holdout}
    if len(train) != 128 or len(holdout) != 128 or train_ids & holdout_ids:
        raise RuntimeError("pilot split count or composition isolation changed")
    output.mkdir(parents=True)
    for split, rows in (("train", train), ("holdout", holdout)):
        (output / f"{split}_plans.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
        )
    manifest = {
        "schema": "rollout_matched_pilot_plans_v1",
        "status": "complete",
        "selection_seed": int(args.seed),
        "train_rows": 128,
        "holdout_rows": 128,
        "exact_composition_overlap": 0,
        "outcomes_read": False,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    (output / "_SUCCESS").touch()


if __name__ == "__main__":
    main()
