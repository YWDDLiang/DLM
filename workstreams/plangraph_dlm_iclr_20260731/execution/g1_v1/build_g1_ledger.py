#!/usr/bin/env python3
"""Build the frozen G1 seed ledger and PG-shuffle training control."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

from g1_protocol import (
    G1_ATTEMPTS,
    G1_SEED_BASE,
    build_seed_ledger,
    canonical_json,
    ledger_sha256,
    sha256_file,
    shuffle_dependency_links,
    write_json,
)


EXPECTED_FIELDS = {
    "task",
    "representation",
    "prompt",
    "answer",
    "text",
    "loss_profile",
    "sample_weight",
    "training_pair_sha256",
}


def transformed_identity(prompt: str, answer: str) -> str:
    return hashlib.sha256(
        f"plangraph_pg_shuffle_training_pair_v1\0{prompt}\0{answer}".encode(
            "utf-8"
        )
    ).hexdigest()


def transform_split(
    source: Path,
    target: Path,
    *,
    expected_sha256: str,
    expected_rows: int,
    seed: int,
) -> dict:
    observed_source_sha = sha256_file(source)
    if observed_source_sha != str(expected_sha256):
        raise ValueError(
            f"{source} SHA mismatch: {observed_source_sha} != {expected_sha256}"
        )
    rows = 0
    changed = 0
    composition_mismatches = 0
    with source.open(encoding="utf-8") as input_handle, target.open(
        "x", encoding="utf-8"
    ) as output_handle:
        for line in input_handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if set(row) != EXPECTED_FIELDS:
                raise ValueError(f"planner record fields changed at row {rows}")
            graph = json.loads(str(row["answer"]))
            identity = str(row["training_pair_sha256"])
            shuffled = shuffle_dependency_links(
                graph,
                identity=identity,
                seed=int(seed),
            )
            if shuffled["composition"] != graph["composition"]:
                composition_mismatches += 1
            answer = canonical_json(shuffled)
            if answer != str(row["answer"]):
                changed += 1
            transformed = dict(row)
            transformed["answer"] = answer
            transformed["text"] = str(row["prompt"]) + answer
            transformed["training_pair_sha256"] = transformed_identity(
                str(row["prompt"]), answer
            )
            output_handle.write(canonical_json(transformed) + "\n")
            rows += 1
    if rows != int(expected_rows):
        raise ValueError(f"{source} rows {rows} != frozen {expected_rows}")
    if changed != rows:
        raise ValueError(f"{source} has {rows - changed} identity shuffles")
    if composition_mismatches:
        raise ValueError(
            f"{source} PG-shuffle changed {composition_mismatches} compositions"
        )
    return {
        "source_path": str(source),
        "source_sha256": observed_source_sha,
        "output_path": target.name,
        "output_sha256": sha256_file(target),
        "rows": rows,
        "changed_dependency_rows": changed,
        "composition_mismatches": composition_mismatches,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--train-sha256", required=True)
    parser.add_argument("--val-sha256", required=True)
    parser.add_argument("--train-rows", type=int, default=27136)
    parser.add_argument("--val-rows", type=int, default=9047)
    parser.add_argument("--attempts", type=int, default=G1_ATTEMPTS)
    parser.add_argument("--seed-base", type=int, default=G1_SEED_BASE)
    args = parser.parse_args()

    if args.output_root.exists():
        raise FileExistsError(f"refusing existing output root {args.output_root}")
    args.output_root.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(
            prefix=f".{args.output_root.name}.stage.",
            dir=str(args.output_root.parent),
        )
    )
    try:
        ledger_dir = stage / "ledger"
        shuffle_dir = stage / "PG-shuffle"
        ledger_dir.mkdir()
        shuffle_dir.mkdir()
        seed_rows = build_seed_ledger(
            attempts=int(args.attempts),
            seed_base=int(args.seed_base),
        )
        ledger_path = ledger_dir / "seed_ledger.jsonl"
        with ledger_path.open("x", encoding="utf-8") as handle:
            for row in seed_rows:
                handle.write(canonical_json(row) + "\n")
        seed_ledger_sha = ledger_sha256(seed_rows)
        if sha256_file(ledger_path) != seed_ledger_sha:
            raise AssertionError("seed ledger byte SHA disagrees with protocol SHA")

        split_reports = {
            "train": transform_split(
                args.dataset_root / "planner" / "train.jsonl",
                shuffle_dir / "train.jsonl",
                expected_sha256=str(args.train_sha256),
                expected_rows=int(args.train_rows),
                seed=int(args.seed_base),
            ),
            "val": transform_split(
                args.dataset_root / "planner" / "val.jsonl",
                shuffle_dir / "val.jsonl",
                expected_sha256=str(args.val_sha256),
                expected_rows=int(args.val_rows),
                seed=int(args.seed_base),
            ),
        }
        report = {
            "schema": "plangraph-dlm-g1-ledger@1",
            "status": "complete",
            "attempts": G1_ATTEMPTS,
            "ordinal_start": 0,
            "ordinal_end_inclusive": G1_ATTEMPTS - 1,
            "seed_base": int(args.seed_base),
            "seed_rule": "seed_base_plus_ordinal",
            "seed_ledger_path": "ledger/seed_ledger.jsonl",
            "seed_ledger_sha256": seed_ledger_sha,
            "sample_id_in_prompt": False,
            "retry": False,
            "replacement": False,
            "repair": False,
            "filter": False,
            "rerank": False,
            "shuffle_control": {
                "name": "content_keyed_dependency_link_shuffle_v1",
                "key": "training_pair_sha256",
                "seed": int(args.seed_base),
                "composition_changed": False,
                "rows_filtered": 0,
            },
            "splits": split_reports,
        }
        write_json(stage / "ledger_report.json", report)
        stage.rename(args.output_root)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
