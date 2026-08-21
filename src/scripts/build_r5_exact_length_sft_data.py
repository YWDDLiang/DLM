#!/usr/bin/env python3
"""Build R5 exact-length dynamic-body SFT data from MP-20 CSV splits."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from crystal_dlm.dynamic_crystal import (  # noqa: E402
    build_special_tokens,
    parse_dynamic_answer,
    structure_to_dynamic_answer,
)
from crystal_dlm.fixed_slot import FixedSlotConfig, metadata_from_csv_row, write_json  # noqa: E402
from crystal_dlm.r5_dynamic_length import build_exact_length_record  # noqa: E402
from crystal_dlm.r5_plan_state import PLAN_STATE_VERSION, plan_state_from_arrays  # noqa: E402


def load_tokenizer(tokenizer_path: str | None):
    if not tokenizer_path:
        return None
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    tokenizer.add_special_tokens({"additional_special_tokens": build_special_tokens()})
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def token_len(tokenizer, text: str) -> int | None:
    if tokenizer is None:
        return None
    return int(len(tokenizer(text, add_special_tokens=False)["input_ids"]))


def read_rows(csv_path: Path, limit: int | None = None) -> Iterable[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader):
            if limit is not None and idx >= limit:
                break
            yield row


def histogram_add(histogram: Dict[str, int], key: Any, count: int = 1) -> None:
    histogram[str(key)] = int(histogram.get(str(key), 0)) + int(count)


def structure_row_to_arrays(row: Mapping[str, str]) -> Dict[str, Any]:
    from pymatgen.core import Structure

    structure = Structure.from_str(str(row["cif"]), fmt="cif")
    answer, _ = structure_to_dynamic_answer(structure)
    return parse_dynamic_answer(answer, strict=True)


def build_split(
    *,
    split: str,
    input_csv: Path,
    output_jsonl: Path,
    tokenizer,
    limit: int | None,
    progress_every: int,
    answer_separator: str,
) -> Dict[str, Any]:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    stats: Dict[str, Any] = {
        "split": split,
        "input_csv": str(input_csv),
        "rows_seen": 0,
        "rows_written": 0,
        "failures": 0,
        "representation": "dynamic_v1",
        "r5_representation": "r5_exact_dynamic_v1",
        "answer_semantic_lengths": {},
        "answer_tokenizer_lengths": {},
        "prompt_tokenizer_lengths": {},
        "atom_count_histogram": {},
        "charge_bucket_histogram": {},
        "formula_match_plan": 0,
    }
    failure_path = output_jsonl.with_suffix(".failure_cases.jsonl")
    with output_jsonl.open("w", encoding="utf-8") as out, failure_path.open("w", encoding="utf-8") as failures:
        for row_idx, row in enumerate(read_rows(input_csv, limit=limit)):
            stats["rows_seen"] += 1
            try:
                metadata = metadata_from_csv_row(row)
                arrays = structure_row_to_arrays(row)
                plan = plan_state_from_arrays(arrays, metadata=metadata)
                record = build_exact_length_record(
                    plan_state=plan,
                    arrays=arrays,
                    metadata=metadata,
                    answer_separator=answer_separator,
                )
                record["prompt_length"] = token_len(tokenizer, record["prompt"].rstrip() + "\n")
                record["answer_model_length"] = token_len(tokenizer, record["answer"])
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                stats["rows_written"] += 1
                stats["formula_match_plan"] += 1
                histogram_add(stats["answer_semantic_lengths"], record["answer_semantic_length"])
                histogram_add(stats["atom_count_histogram"], record["num_atoms"])
                histogram_add(stats["charge_bucket_histogram"], plan["charge_bucket"])
                if record["answer_model_length"] is not None:
                    histogram_add(stats["answer_tokenizer_lengths"], record["answer_model_length"])
                if record["prompt_length"] is not None:
                    histogram_add(stats["prompt_tokenizer_lengths"], record["prompt_length"])
            except Exception as exc:  # noqa: BLE001
                stats["failures"] += 1
                failures.write(
                    json.dumps(
                        {
                            "split": split,
                            "row_idx": row_idx,
                            "material_id": row.get("material_id"),
                            "reason": type(exc).__name__,
                            "message": str(exc),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            if progress_every > 0 and stats["rows_seen"] % progress_every == 0:
                print(
                    json.dumps(
                        {
                            "event": "r5_exact_length_builder_progress",
                            "split": split,
                            "rows_seen": stats["rows_seen"],
                            "rows_written": stats["rows_written"],
                            "failures": stats["failures"],
                        }
                    ),
                    flush=True,
                )
    stats["formula_match_plan_rate"] = stats["formula_match_plan"] / max(1, stats["rows_written"])
    return stats


def max_hist_key(histogram: Dict[str, int]) -> int:
    if not histogram:
        return 0
    return max(int(float(key)) for key in histogram)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--answer-separator", default="")
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.tokenizer_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    splits: Dict[str, Any] = {}
    for split in ("train", "val", "test"):
        splits[split] = build_split(
            split=split,
            input_csv=args.input_dir / f"{split}.csv",
            output_jsonl=args.output_dir / f"{split}.jsonl",
            tokenizer=tokenizer,
            limit=args.limit,
            progress_every=args.progress_every,
            answer_separator=args.answer_separator,
        )
    vocab_tokens = build_special_tokens(FixedSlotConfig())
    (args.output_dir / "vocab_tokens.txt").write_text("\n".join(vocab_tokens) + "\n", encoding="utf-8")
    max_answer = max(max_hist_key(item["answer_tokenizer_lengths"]) for item in splits.values())
    max_prompt = max(max_hist_key(item["prompt_tokenizer_lengths"]) for item in splits.values())
    max_semantic = max(max_hist_key(item["answer_semantic_lengths"]) for item in splits.values())
    summary = {
        "representation": "dynamic_v1",
        "r5_representation": "r5_exact_dynamic_v1",
        "plan_state_version": PLAN_STATE_VERSION,
        "splits": splits,
        "max_answer_token_count": max_semantic,
        "answer_token_count": max_semantic,
        "max_answer_model_length": max_answer,
        "max_prompt_model_length": max_prompt,
        "max_length_recommended": max_prompt + max_answer + 16,
        "special_token_count": len(vocab_tokens),
    }
    write_json(str(args.output_dir / "stats.json"), summary)
    write_json(
        str(args.output_dir / "_SUCCESS"),
        {
            "representation": "dynamic_v1",
            "r5_representation": "r5_exact_dynamic_v1",
            "complete": True,
            "splits": {
                split: {
                    "rows_seen": split_stats["rows_seen"],
                    "rows_written": split_stats["rows_written"],
                    "failures": split_stats["failures"],
                    "formula_match_plan_rate": split_stats["formula_match_plan_rate"],
                }
                for split, split_stats in splits.items()
            },
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
