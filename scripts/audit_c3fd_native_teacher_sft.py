#!/usr/bin/env python3
"""Audit formal teacher-rich SFT tokenization without loading model weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
from typing import Any, Iterable


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"non-object JSONL row at {path}:{line_number}")
            yield row


def quantile(values: list[int], fraction: float) -> int:
    if not values:
        raise ValueError("cannot summarize empty values")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return int(ordered[index])


def audit_split(path: Path, tokenizer, *, max_length: int) -> dict[str, Any]:
    total_lengths: list[int] = []
    prompt_lengths: list[int] = []
    answer_lengths: list[int] = []
    suffix_mismatches = 0
    over_limit = 0
    rows = 0
    for row in iter_jsonl(path):
        if row.get("view") != "teacher-native":
            raise ValueError(f"formal SFT contains non-teacher view in {path}")
        if row.get("prompt_schema") != "C3FD_NATIVE_PLAN_V2":
            raise ValueError(f"formal SFT prompt schema changed in {path}")
        if "prediction_checkpoint" in row:
            raise ValueError(f"formal SFT contains predicted Planner fields in {path}")
        prompt = str(row["prompt"]).rstrip() + "\n"
        answer = str(row["answer"])
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(prompt + answer, add_special_tokens=False)["input_ids"]
        suffix_mismatches += int(full_ids[len(prompt_ids) :] != answer_ids)
        over_limit += int(len(full_ids) > int(max_length))
        prompt_lengths.append(len(prompt_ids))
        answer_lengths.append(len(answer_ids))
        total_lengths.append(len(full_ids))
        rows += 1
    if rows == 0:
        raise ValueError(f"empty SFT split {path}")
    return {
        "rows": rows,
        "prompt_tokens": {
            "min": min(prompt_lengths),
            "median": quantile(prompt_lengths, 0.5),
            "p95": quantile(prompt_lengths, 0.95),
            "max": max(prompt_lengths),
        },
        "answer_tokens": {
            "min": min(answer_lengths),
            "median": quantile(answer_lengths, 0.5),
            "p95": quantile(answer_lengths, 0.95),
            "max": max(answer_lengths),
        },
        "total_tokens": {
            "min": min(total_lengths),
            "median": quantile(total_lengths, 0.5),
            "p95": quantile(total_lengths, 0.95),
            "max": max(total_lengths),
        },
        "max_length": int(max_length),
        "over_limit": over_limit,
        "prompt_answer_suffix_mismatches": suffix_mismatches,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=382)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    vocab_tokens = [
        line.strip()
        for line in (args.data_dir / "vocab_tokens.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    added = tokenizer.add_special_tokens({"additional_special_tokens": vocab_tokens})
    staging = args.output_dir.with_name(f".{args.output_dir.name}.preparing")
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir(parents=True)
    try:
        splits = {
            split: audit_split(
                args.data_dir / f"{split}.jsonl",
                tokenizer,
                max_length=args.max_length,
            )
            for split in ("train", "val")
        }
        gates = {
            "teacher_only": True,
            "no_truncation": all(row["over_limit"] == 0 for row in splits.values()),
            "prompt_answer_boundary_exact": all(
                row["prompt_answer_suffix_mismatches"] == 0 for row in splits.values()
            ),
        }
        if not all(gates.values()):
            raise ValueError(f"teacher SFT tokenization audit failed: {gates}")
        report = {
            "schema": "c3fd_native_teacher_sft_token_audit_v1",
            "model_path": str(args.model_path.resolve()),
            "data_dir": str(args.data_dir.resolve()),
            "model_config_sha256": sha256_file(args.model_path / "config.json"),
            "tokenizer_json_sha256": sha256_file(args.model_path / "tokenizer.json"),
            "data_manifest_sha256": sha256_file(args.data_dir / "manifest.json"),
            "vocab_tokens_sha256": sha256_file(args.data_dir / "vocab_tokens.txt"),
            "vocab_tokens": len(vocab_tokens),
            "new_tokens_added": int(added),
            "tokenizer_size_after_add": len(tokenizer),
            "splits": splits,
            "gate": gates,
            "outcomes_read": False,
        }
        report_path = staging / "TOKENIZATION_AUDIT.json"
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (staging / "SHA256SUMS").write_text(
            f"{sha256_file(report_path)}  TOKENIZATION_AUDIT.json\n",
            encoding="utf-8",
        )
        (staging / "_SUCCESS").touch()
        staging.rename(args.output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
