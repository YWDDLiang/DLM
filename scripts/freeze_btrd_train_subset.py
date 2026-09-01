#!/usr/bin/env python3
"""Freeze the train-only, evaluation-disjoint BTRD teacher subset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from crystal_dlm.composition_identity import identity_from_plan_state, identity_text


SCHEMA = "btrd_train_subset_v1"


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number} is not an object")
            yield value


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact_identity(row: Mapping[str, Any]) -> str:
    state = row.get("plan_state")
    if not isinstance(state, Mapping):
        raise ValueError("row lacks plan_state")
    return identity_text(identity_from_plan_state(state))


def freeze_rows(
    train_rows,
    evaluation_rows,
    *,
    count: int,
    teacher_count: int,
    selection_seed: int,
):
    if not 0 < teacher_count <= count:
        raise ValueError("teacher_count must be in (0, count]")
    evaluation_identities = {exact_identity(row) for row in evaluation_rows}
    candidates = []
    seen_source_idx: set[int] = set()
    for ordinal, row in enumerate(train_rows):
        source_idx = int(row.get("source_row_idx", ordinal))
        if source_idx in seen_source_idx:
            raise ValueError("duplicate train source_row_idx")
        seen_source_idx.add(source_idx)
        identity = exact_identity(row)
        if identity in evaluation_identities:
            continue
        answer_sha = str(row.get("answer_sha256") or "")
        key = hashlib.sha256(
            f"{selection_seed}\0{source_idx}\0{identity}\0{answer_sha}".encode()
        ).hexdigest()
        candidates.append((key, source_idx, identity, row))
    candidates.sort(key=lambda item: (item[0], item[1]))
    if len(candidates) < count:
        raise ValueError("insufficient evaluation-disjoint BTRD train rows")
    selected = []
    for index, (_key, source_idx, identity, source) in enumerate(candidates[:count]):
        selected.append(
            {
                **dict(source),
                "btrd_schema": SCHEMA,
                "btrd_index": index,
                "btrd_source_row_idx": source_idx,
                "btrd_exact_identity": identity,
                "btrd_target_mode": "model494_tau200" if index < teacher_count else "mp20_anchor",
                "btrd_selection_outcomes_read": False,
            }
        )
    return selected, {
        "evaluation_exact_identities": len(evaluation_identities),
        "eligible_train_rows": len(candidates),
        "excluded_exact_overlap": len(train_rows) - len(candidates),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--evaluation-plans", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=8192)
    parser.add_argument("--teacher-count", type=int, default=6144)
    parser.add_argument("--selection-seed", type=int, default=20260901)
    args = parser.parse_args()

    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    train_path = args.train_jsonl.resolve()
    eval_path = args.evaluation_plans.resolve()
    train_sha = sha256_file(train_path)
    eval_sha = sha256_file(eval_path)
    train_rows = list(iter_jsonl(train_path))
    evaluation_rows = list(iter_jsonl(eval_path))
    if sha256_file(train_path) != train_sha or sha256_file(eval_path) != eval_sha:
        raise RuntimeError("BTRD frozen source changed while reading")
    selected, audit = freeze_rows(
        train_rows,
        evaluation_rows,
        count=args.count,
        teacher_count=args.teacher_count,
        selection_seed=args.selection_seed,
    )
    output.mkdir(parents=True)
    selected_path = output / "selected.jsonl"
    write_jsonl(selected_path, selected)
    manifest = {
        "schema": SCHEMA,
        "status": "complete",
        "selected": len(selected),
        "model494_tau200_rows": sum(
            row["btrd_target_mode"] == "model494_tau200" for row in selected
        ),
        "mp20_anchor_rows": sum(
            row["btrd_target_mode"] == "mp20_anchor" for row in selected
        ),
        "selection_seed": int(args.selection_seed),
        "selection": "content-hash order before any teacher outcome",
        "outcomes_read": False,
        "train_source": {"path": str(train_path), "sha256": train_sha, "rows": len(train_rows)},
        "evaluation_source": {"path": str(eval_path), "sha256": eval_sha, "rows": len(evaluation_rows)},
        "audit": audit,
        "selected_sha256": sha256_file(selected_path),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "SHA256SUMS").write_text(
        f"{sha256_file(selected_path)}  selected.jsonl\n"
        f"{sha256_file(output / 'manifest.json')}  manifest.json\n",
        encoding="utf-8",
    )
    (output / "_SUCCESS").touch()
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
