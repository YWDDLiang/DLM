#!/usr/bin/env python3
"""Canonicalize full MP20 teacher bodies to the inference element-slot order."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

from crystal_dlm.canonical_site_order import canonicalize_dynamic_answer_to_plan
from crystal_dlm.dynamic_crystal import parse_dynamic_answer


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def convert_split(path: Path, split: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source = read_jsonl(path)
    output: list[dict[str, Any]] = []
    changed = 0
    mismatched_slots: list[int] = []
    for ordinal, row in enumerate(source):
        if str(row.get("source_split")) != split:
            raise ValueError(f"{split} row {ordinal} has wrong source_split")
        if str(row.get("view")) != "teacher-native":
            raise ValueError(f"{split} row {ordinal} is not teacher-native")
        if str(row.get("prompt_schema")) != "C3FD_NATIVE_PLAN_V2":
            raise ValueError(f"{split} row {ordinal} has wrong prompt schema")
        plan = dict(row["plan_state"])
        canonical, diagnostics = canonicalize_dynamic_answer_to_plan(
            str(row["answer"]), plan
        )
        parsed = parse_dynamic_answer(canonical, strict=True)
        expected: list[str] = []
        for element, count in zip(plan["elements"], plan["counts"], strict=True):
            expected.extend([str(element)] * int(count))
        if parsed["species"] != expected:
            raise RuntimeError("canonical body does not match inference element order")
        changed += int(diagnostics["changed"])
        mismatched_slots.append(int(diagnostics["mismatched_element_slots"]))
        converted = dict(row)
        converted["answer"] = canonical
        if "answer_sha256" in converted:
            converted["answer_sha256"] = hashlib.sha256(
                canonical.encode("utf-8")
            ).hexdigest()
        converted["site_order"] = "plan_expanded_stable_v1"
        converted["canonicalization_changed"] = bool(diagnostics["changed"])
        output.append(converted)
    if [int(row["source_row_idx"]) for row in output] != [
        int(row["source_row_idx"]) for row in source
    ]:
        raise RuntimeError("canonicalization changed source row order")
    return output, {
        "rows": len(source),
        "changed_rows": changed,
        "unchanged_rows": len(source) - changed,
        "mismatched_slots_total": sum(mismatched_slots),
        "mismatched_slots_histogram": dict(sorted(Counter(mismatched_slots).items())),
        "dropped_rows": 0,
        "prompt_changed_rows": 0,
        "physical_operation": "site_permutation_only",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--val-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    train, train_report = convert_split(args.train_jsonl, "train")
    val, val_report = convert_split(args.val_jsonl, "val")
    if len(train) != 27136 or len(val) != 9047:
        raise ValueError("full MP20 teacher row counts changed")

    args.output_dir.mkdir(parents=True)
    write_jsonl(args.output_dir / "train.jsonl", train)
    write_jsonl(args.output_dir / "val.jsonl", val)
    manifest = {
        "schema": "c3fd_native_teacher_sft_canonical_site_v1",
        "status": "complete",
        "teacher": "original_MP20_only",
        "site_order": "plan_expanded_stable_v1",
        "train": train_report,
        "val": val_report,
        "outcomes_read": False,
        "energy_or_hull_fields_added": False,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "_SUCCESS").touch()
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
