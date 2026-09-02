#!/usr/bin/env python3
"""Build paired q0→q1 data for the projected-force residual microstudent."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

from crystal_dlm.dynamic_crystal import parse_dynamic_answer
from crystal_dlm.fixed_slot import tokenize_answer_text


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def selected_tokens(result: dict[str, Any]) -> list[str]:
    mode = str(result["teacher_mode"])
    payload = (
        result["force_quantization"]
        if mode == "force_projected"
        else result["barrier_quantization"]
    )
    tokens = [str(token) for token in payload["tokens"]]
    if not tokens:
        raise ValueError("selected teacher has no dynamic tokens")
    return tokens


def assert_transition_contract(source_answer: str, target_answer: str) -> int:
    source_tokens = tokenize_answer_text(source_answer)
    target_tokens = tokenize_answer_text(target_answer)
    if len(source_tokens) != len(target_tokens):
        raise ValueError("source/target dynamic lengths differ")
    source = parse_dynamic_answer(source_answer, strict=True)
    target = parse_dynamic_answer(target_answer, strict=True)
    if source["num_atoms"] != target["num_atoms"]:
        raise ValueError("source/target atom counts differ")
    if source["species"] != target["species"]:
        raise ValueError("source/target species order differs")
    count = int(source["num_atoms"])
    if source_tokens[:7] != target_tokens[:7]:
        raise ValueError("projected teacher unexpectedly changed N or lattice tokens")
    for site in range(count):
        element_position = 7 + 4 * site
        if source_tokens[element_position] != target_tokens[element_position]:
            raise ValueError("projected teacher unexpectedly changed an element token")
    changed = sum(left != right for left, right in zip(source_tokens, target_tokens))
    return int(changed)


def build_rows(
    preflight_rows: list[dict[str, Any]],
    result_rows: list[dict[str, Any]],
    teacher_sft_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if len(preflight_rows) != 512 or len(result_rows) != 512:
        raise ValueError("microstudent builder requires 512 paired preflight rows")
    teacher_by_source: dict[int, dict[str, Any]] = {}
    for row in teacher_sft_rows:
        source_index = int(row["source_row_idx"])
        if source_index in teacher_by_source:
            raise ValueError("teacher SFT source_row_idx is not unique")
        teacher_by_source[source_index] = row
    result_by_index = {int(row["row_index"]): row for row in result_rows}
    if len(result_by_index) != 512:
        raise ValueError("projected teacher row_index is not unique")

    train: list[dict[str, Any]] = []
    holdout: list[dict[str, Any]] = []
    modes = Counter()
    changed_positions = Counter()
    for source in preflight_rows:
        row_index = int(source["row_index"])
        result = result_by_index[row_index]
        if int(source["base_index"]) != int(result["base_index"]):
            raise ValueError("preflight/result base_index mismatch")
        teacher = teacher_by_source[int(source["source_row_index"])]
        source_answer = str(source["dynamic_answer"])
        target_answer = "".join(selected_tokens(result))
        changed = assert_transition_contract(source_answer, target_answer)
        split = "holdout" if int(source["base_index"]) % 4 == 0 else "train"
        output = {
            "schema": "projected_force_microstudent_pair_v1",
            "prompt": str(teacher["prompt"]),
            "source_answer": source_answer,
            "answer": target_answer,
            "num_atoms": int(source["num_atoms"]),
            "loss_profile": str(teacher.get("loss_profile") or "fixed_slot"),
            "sample_weight": 0.125,
            "source_row_idx": int(source["source_row_index"]),
            "base_index": int(source["base_index"]),
            "preflight_row_index": row_index,
            "transition_mode": str(result["teacher_mode"]),
            "changed_geometry_tokens": changed,
            "energy_teacher_known": bool(result.get("teacher_complete")),
            "selected_delta_eV_per_atom": result.get("selected_delta_eV_per_atom"),
            "split": split,
        }
        modes[output["transition_mode"]] += 1
        changed_positions[changed] += 1
        (holdout if split == "holdout" else train).append(output)
    if len(train) != 384 or len(holdout) != 128:
        raise RuntimeError("base-structure split changed")
    manifest = {
        "schema": "projected_force_microstudent_data_v1",
        "status": "complete",
        "rows": 512,
        "train_rows": len(train),
        "holdout_rows": len(holdout),
        "train_base_structures": len({row["base_index"] for row in train}),
        "holdout_base_structures": len({row["base_index"] for row in holdout}),
        "base_overlap": len(
            {row["base_index"] for row in train}
            & {row["base_index"] for row in holdout}
        ),
        "transition_modes": dict(sorted(modes.items())),
        "changed_geometry_token_histogram": {
            str(key): value for key, value in sorted(changed_positions.items())
        },
        "sample_weight_per_base": 1.0,
        "outcomes_read": False,
    }
    return train, holdout, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-jsonl", type=Path, required=True)
    parser.add_argument("--teacher-rows-jsonl", type=Path, required=True)
    parser.add_argument("--teacher-sft-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    train, holdout, manifest = build_rows(
        read_jsonl(args.preflight_jsonl.resolve()),
        read_jsonl(args.teacher_rows_jsonl.resolve()),
        read_jsonl(args.teacher_sft_jsonl.resolve()),
    )
    output.mkdir(parents=True)
    (output / "train.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in train)
    )
    (output / "val.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in holdout)
    )
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    (output / "_SUCCESS").touch()
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
