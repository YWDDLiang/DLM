#!/usr/bin/env python3
"""Materialize globally indexed tau200 execution Plans from a BTRD subset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def materialize(rows, *, expected_teacher_rows: int):
    teachers = [row for row in rows if row.get("btrd_target_mode") == "model494_tau200"]
    if len(teachers) != expected_teacher_rows:
        raise ValueError("BTRD tau200 teacher count changed")
    if [int(row["btrd_index"]) for row in teachers] != list(range(expected_teacher_rows)):
        raise ValueError("BTRD teacher rows must be the frozen prefix")
    output = []
    for index, source in enumerate(teachers):
        row = {
            "schema": "btrd_execution_plan_v1",
            "sample_idx": index,
            "source_sample_idx": index,
            "btrd_index": index,
            "mp20_source_row_idx": int(source["btrd_source_row_idx"]),
            "reduced_composition_identity": str(source["btrd_exact_identity"]),
            "prompt": str(source["prompt"]),
            "prompt_schema": str(source.get("prompt_schema") or "C3FD_NATIVE_PLAN_V2"),
            "plan_state": dict(source["plan_state"]),
            "teacher_steps": 200,
            "outcomes_read": False,
        }
        output.append(row)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-teacher-rows", type=int, default=6144)
    args = parser.parse_args()
    source = args.subset_jsonl.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    rows = read_jsonl(source)
    plans = materialize(rows, expected_teacher_rows=args.expected_teacher_rows)
    output.mkdir(parents=True)
    plan_path = output / "plans_for_dlm.jsonl"
    plan_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in plans),
        encoding="utf-8",
    )
    manifest = {
        "schema": "btrd_execution_plans_v1",
        "status": "complete",
        "rows": len(plans),
        "teacher_steps": 200,
        "source_sha256": sha256_file(source),
        "plans_sha256": sha256_file(plan_path),
        "outcomes_read": False,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "SHA256SUMS").write_text(
        f"{sha256_file(plan_path)}  plans_for_dlm.jsonl\n"
        f"{sha256_file(output / 'manifest.json')}  manifest.json\n",
        encoding="utf-8",
    )
    (output / "_SUCCESS").touch()
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
