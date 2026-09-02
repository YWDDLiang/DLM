#!/usr/bin/env python3
"""Freeze 128 train and 128 holdout Plans for rollout-matched DLM pilot."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random

from crystal_dlm.c3fd_native_plan import build_native_inference_prompt
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
        if (
            str(row.get("source_split")) != "train"
            or str(row.get("prompt_schema")) != "C3FD_NATIVE_PLAN_V2"
            or str(row.get("view")) != "teacher-native"
        ):
            raise ValueError("pilot source must be MP20-train teacher-native V2")
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


def plan_rows(
    rows: list[dict],
    split: str,
    predictions: dict[int, dict],
    *,
    planner_seed: str,
) -> list[dict]:
    result = []
    for sample_idx, source in enumerate(rows):
        source_index = int(source["source_row_idx"])
        predicted = predictions[source_index]["predictions_by_checkpoint"][
            str(planner_seed)
        ]
        predicted_soft = {
            field: str(predicted[field]["prediction"])
            for field in (
                "lattice_system",
                "spacegroup_bucket",
                "volume_per_atom_bin",
            )
        }
        result.append(
            {
                "schema": "rollout_matched_pilot_plan_v1",
                "sample_idx": sample_idx,
                "source_sample_idx": source_index,
                "source_row_idx": source_index,
                "split": split,
                "prompt": build_native_inference_prompt(
                    dict(source["plan_state"]), predicted_soft
                ),
                "plan_state": dict(source["plan_state"]),
                "predicted_structural_plan": predicted_soft,
                "planner_prediction_seed": str(planner_seed),
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
    parser.add_argument("--predicted-soft-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--planner-seed", default="seed17")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    source_rows = read_jsonl(args.teacher_sft_jsonl.resolve())
    prediction_rows = read_jsonl(args.predicted_soft_jsonl.resolve())
    predictions = {int(row["source_row_idx"]): row for row in prediction_rows}
    if len(predictions) != len(prediction_rows):
        raise ValueError("predicted-soft source_row_idx is not unique")
    selected = select_rows(source_rows, count=256, seed=int(args.seed))
    train = plan_rows(
        selected[::2], "train", predictions, planner_seed=str(args.planner_seed)
    )
    holdout = plan_rows(
        selected[1::2], "holdout", predictions, planner_seed=str(args.planner_seed)
    )
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
        "planner_prediction_seed": str(args.planner_seed),
        "prompt_mode": "C3FD-predicted-native-V2",
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
