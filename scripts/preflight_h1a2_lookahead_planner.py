#!/usr/bin/env python3
"""CPU/tokenizer preflight for a frozen H1-A2 P-control/P* data stream."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from crystal_dlm.h1a2_planner_batch import (  # noqa: E402
    format_training_prompt,
    prepare_planner_example,
)
from crystal_dlm.h1a2_planner_objective import (  # noqa: E402
    FIELD_GROUP_IDS,
    LOOKAHEAD_FIELDS,
)


PREFLIGHT_SCHEMA = "h1a2_lookahead_planner_preflight_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise TypeError(f"{path}:{line_number} is not a JSON object")
            yield payload


def scan_split(
    path: Path,
    tokenizer: Any,
    *,
    max_length: int,
    vocabs: dict[str, tuple[str, ...]],
) -> dict[str, Any]:
    count = 0
    minimum_tokens: int | None = None
    maximum_tokens = 0
    maximum_prompt_tokens = 0
    maximum_answer_tokens = 0
    source_identities: list[str] = []
    prompt_digest = hashlib.sha256()
    for row in iter_jsonl(path):
        example = prepare_planner_example(
            row,
            tokenizer,
            max_length=max_length,
            lookahead_vocabs=vocabs,
        )
        count += 1
        token_count = len(example.input_ids)
        minimum_tokens = (
            token_count
            if minimum_tokens is None
            else min(minimum_tokens, token_count)
        )
        maximum_tokens = max(maximum_tokens, token_count)
        maximum_prompt_tokens = max(maximum_prompt_tokens, example.prompt_tokens)
        maximum_answer_tokens = max(maximum_answer_tokens, example.answer_tokens)
        if example.formula_boundary >= example.lattice_boundary:
            raise ValueError("formula/lattice causal boundary order changed")
        observed_groups = {
            value for value in example.field_group_ids if value >= 0
        }
        if observed_groups != set(FIELD_GROUP_IDS.values()):
            raise ValueError("one or more field groups are absent from a row")
        if set(example.lookahead_labels) != set(LOOKAHEAD_FIELDS):
            raise ValueError("look-ahead label coverage changed")
        if example.source_line_sha256 is None:
            raise ValueError("source-line identity is absent")
        source_identities.append(example.source_line_sha256)
        prompt = format_training_prompt(tokenizer, row)
        prompt_digest.update(len(prompt.encode("utf-8")).to_bytes(8, "big"))
        prompt_digest.update(prompt.encode("utf-8"))
    if not count:
        raise ValueError(f"{path} contains no rows")
    if len(source_identities) != len(set(source_identities)):
        raise ValueError(f"{path} contains duplicate source-line identities")
    return {
        "rows": count,
        "minimum_total_tokens": minimum_tokens,
        "maximum_total_tokens": maximum_tokens,
        "maximum_prompt_tokens_after_left_truncation": maximum_prompt_tokens,
        "maximum_answer_tokens_with_eos": maximum_answer_tokens,
        "ordered_prompt_bytes_sha256": prompt_digest.hexdigest(),
        "unique_source_line_identities": len(set(source_identities)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-train-rows", type=int, required=True)
    parser.add_argument("--expected-val-rows", type=int, required=True)
    parser.add_argument("--max-length", type=int, default=768)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    manifest_path = args.data_dir / "manifest.json"
    success_path = args.data_dir / "_SUCCESS"
    vocab_path = args.data_dir / "lookahead_vocabs.json"
    for path in (
        manifest_path,
        success_path,
        vocab_path,
        args.data_dir / "train.jsonl",
        args.data_dir / "val.jsonl",
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    success = json.loads(success_path.read_text(encoding="utf-8"))
    if success.get("status") != "complete":
        raise ValueError("data _SUCCESS status is not complete")
    if success.get("manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("data manifest SHA does not match _SUCCESS")
    raw_vocabs = json.loads(vocab_path.read_text(encoding="utf-8"))
    vocabs = {
        field: tuple(str(value) for value in raw_vocabs.get(field, ()))
        for field in LOOKAHEAD_FIELDS
    }
    if any(
        not values or values != tuple(sorted(set(values)))
        for values in vocabs.values()
    ):
        raise ValueError("look-ahead vocabularies are not unique and sorted")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path,
        trust_remote_code=True,
        use_fast=True,
    )
    if not tokenizer.is_fast:
        raise ValueError("real-tokenizer preflight requires a fast tokenizer")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    splits = {
        "train": scan_split(
            args.data_dir / "train.jsonl",
            tokenizer,
            max_length=int(args.max_length),
            vocabs=vocabs,
        ),
        "val": scan_split(
            args.data_dir / "val.jsonl",
            tokenizer,
            max_length=int(args.max_length),
            vocabs=vocabs,
        ),
    }
    if splits["train"]["rows"] != int(args.expected_train_rows):
        raise ValueError("train row count differs from the registered count")
    if splits["val"]["rows"] != int(args.expected_val_rows):
        raise ValueError("validation row count differs from the registered count")
    tokenizer_inventory = []
    for filename in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ):
        path = args.tokenizer_path / filename
        if path.is_file():
            tokenizer_inventory.append(
                {
                    "path": filename,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    if not tokenizer_inventory:
        raise ValueError("tokenizer identity files are absent")
    report = {
        "schema": PREFLIGHT_SCHEMA,
        "status": "complete",
        "data_dir": str(args.data_dir),
        "data_manifest_sha256": sha256_file(manifest_path),
        "tokenizer_path": str(args.tokenizer_path),
        "tokenizer_inventory": tokenizer_inventory,
        "tokenizer_is_fast": True,
        "max_length": int(args.max_length),
        "additive_answer_eos_parity": True,
        "field_mapping_complete": True,
        "lookahead_boundary_order_valid": True,
        "splits": splits,
    }
    args.output_dir.mkdir(parents=True)
    report_path = args.output_dir / "preflight_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "_SUCCESS").write_text(
        json.dumps(
            {
                "schema": PREFLIGHT_SCHEMA,
                "status": "complete",
                "report_sha256": sha256_file(report_path),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
