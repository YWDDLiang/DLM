#!/usr/bin/env python3
"""Build mixed chemical-plan + fixed-slot SFT data.

The training set contains two row types:

* ``loss_profile=text`` rows teach the model to write a short natural-language
  chemistry plan ending in ``crystal tokens:``.
* ``loss_profile=fixed_slot`` rows condition on a gold plan and supervise only
  the 107 fixed-slot crystal tokens.

Validation/test rows keep only the fixed-slot structure task so checkpoint
selection remains close to the downstream sampler.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from crystal_dlm.chemical_plan import (  # noqa: E402
    PLAN_PROMPT,
    build_plan_conditioned_prompt,
    chemical_plan_from_fixed_arrays,
)
from crystal_dlm.fixed_slot import ANSWER_TOKEN_COUNT, parse_fixed_slot_answer, write_json  # noqa: E402


def load_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def token_len(tokenizer, text: str) -> int | None:
    if tokenizer is None:
        return None
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def load_tokenizer(tokenizer_path: str | None, vocab_file: Path):
    if not tokenizer_path:
        return None
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if vocab_file.exists():
        with vocab_file.open(encoding="utf-8") as handle:
            tokens = [line.strip() for line in handle if line.strip()]
        tokenizer.add_special_tokens({"additional_special_tokens": tokens})
    return tokenizer


def copy_metadata_files(input_dir: Path, output_dir: Path) -> None:
    for filename in ("vocab_tokens.txt", "prompt_pool.json"):
        src = input_dir / filename
        if src.exists():
            shutil.copy2(src, output_dir / filename)


def composition_reason_from_row(row: Dict[str, Any], arrays: Dict[str, Any]) -> str:
    if row.get("composition_reason"):
        return str(row["composition_reason"])
    metadata = row.get("metadata") or {}
    if metadata.get("composition_reason"):
        return str(metadata["composition_reason"])
    try:
        from crystal_dlm.composition_validity import composition_record

        return str(composition_record(arrays["atom_types"]).get("reason") or "unknown")
    except Exception:
        return "unknown"


def make_plan_row(row: Dict[str, Any], plan: str, tokenizer) -> Dict[str, Any]:
    prompt = PLAN_PROMPT
    prompt_text = prompt.rstrip() + "\n"
    answer = plan.rstrip() + "\n"
    metadata = dict(row.get("metadata") or {})
    metadata["source_task"] = row.get("task", "unconditional")
    sample_weight = float(row.get("sample_weight", 1.0) or 1.0)
    return {
        "task": "chemical_plan",
        "prompt": prompt,
        "answer": answer,
        "text": prompt_text + answer,
        "prompt_length": token_len(tokenizer, prompt_text),
        "answer_model_length": token_len(tokenizer, answer),
        "loss_profile": "text",
        "sample_weight": sample_weight,
        "selection_role": "chemical_plan",
        "metadata": metadata,
    }


def make_structure_row(row: Dict[str, Any], plan: str, tokenizer) -> Dict[str, Any]:
    prompt = build_plan_conditioned_prompt(plan)
    prompt_text = prompt.rstrip() + "\n"
    answer = str(row["answer"])
    out = dict(row)
    out.update(
        {
            "task": "chemical_plan_conditioned_fixed_slot",
            "prompt": prompt,
            "answer": answer,
            "text": prompt_text + answer,
            "prompt_length": token_len(tokenizer, prompt_text),
            "answer_model_length": token_len(tokenizer, answer),
            "loss_profile": "fixed_slot",
            "selection_role": str(row.get("selection_role") or "base_preserved"),
        }
    )
    return out


def build_split(
    *,
    split: str,
    input_path: Path,
    output_path: Path,
    tokenizer,
    rng: random.Random,
    plan_row_fraction: float,
    train_only_plan_rows: bool,
    limit: int | None,
) -> Dict[str, Any]:
    stats: Dict[str, Any] = {
        "split": split,
        "input_path": str(input_path),
        "rows_seen": 0,
        "rows_written": 0,
        "fixed_slot_rows": 0,
        "chemical_plan_rows": 0,
        "failures": 0,
        "answer_token_count": ANSWER_TOKEN_COUNT,
        "reason_counts": {},
        "plan_answer_tokenizer_lengths": {},
        "structure_prompt_tokenizer_lengths": {},
    }
    failure_path = output_path.with_suffix(".failure_cases.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out, failure_path.open("w", encoding="utf-8") as failures:
        for idx, row in enumerate(load_jsonl(input_path)):
            if limit is not None and idx >= limit:
                break
            stats["rows_seen"] += 1
            try:
                arrays = parse_fixed_slot_answer(str(row["answer"]), strict=True)
                reason = composition_reason_from_row(row, arrays)
                plan_payload = chemical_plan_from_fixed_arrays(
                    arrays,
                    metadata=row.get("metadata") or {},
                    composition_reason=reason,
                )
                reason = str(plan_payload.get("reason") or reason)
                stats["reason_counts"][reason] = stats["reason_counts"].get(reason, 0) + 1
                plan = str(plan_payload["plan"])
                structure_row = make_structure_row(row, plan, tokenizer)
                out.write(json.dumps(structure_row, ensure_ascii=False) + "\n")
                stats["rows_written"] += 1
                stats["fixed_slot_rows"] += 1
                if structure_row.get("prompt_length") is not None:
                    key = str(structure_row["prompt_length"])
                    stats["structure_prompt_tokenizer_lengths"][key] = (
                        stats["structure_prompt_tokenizer_lengths"].get(key, 0) + 1
                    )
                add_plan = split == "train" or not train_only_plan_rows
                if add_plan and rng.random() < float(plan_row_fraction):
                    plan_row = make_plan_row(row, plan, tokenizer)
                    out.write(json.dumps(plan_row, ensure_ascii=False) + "\n")
                    stats["rows_written"] += 1
                    stats["chemical_plan_rows"] += 1
                    if plan_row.get("answer_model_length") is not None:
                        key = str(plan_row["answer_model_length"])
                        stats["plan_answer_tokenizer_lengths"][key] = (
                            stats["plan_answer_tokenizer_lengths"].get(key, 0) + 1
                        )
            except Exception as exc:  # noqa: BLE001
                stats["failures"] += 1
                failures.write(
                    json.dumps(
                        {
                            "row_idx": idx,
                            "reason": type(exc).__name__,
                            "message": str(exc),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--plan-row-fraction", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=20260528)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--train-only-plan-rows", action="store_true", default=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    copy_metadata_files(args.input_dir, args.output_dir)
    tokenizer = load_tokenizer(args.tokenizer_path, args.output_dir / "vocab_tokens.txt")
    run_config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    run_config["representation"] = "fixed_slot"
    run_config["mixed_loss_profiles"] = ["fixed_slot", "text"]
    write_json(str(args.output_dir / "run_config.json"), run_config)

    split_stats: Dict[str, Any] = {}
    split_seed_offsets = {"train": 0, "val": 10_000, "test": 20_000}
    for split in ("train", "val", "test"):
        split_stats[split] = build_split(
            split=split,
            input_path=args.input_dir / f"{split}.jsonl",
            output_path=args.output_dir / f"{split}.jsonl",
            tokenizer=tokenizer,
            rng=random.Random(args.seed + split_seed_offsets[split]),
            plan_row_fraction=args.plan_row_fraction,
            train_only_plan_rows=args.train_only_plan_rows,
            limit=args.limit,
        )
    aggregate = {
        "representation": "fixed_slot",
        "answer_token_count": ANSWER_TOKEN_COUNT,
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "plan_prompt": PLAN_PROMPT,
        "splits": split_stats,
    }
    write_json(str(args.output_dir / "stats.json"), aggregate)
    reason_counts: Counter[str] = Counter()
    for stats in split_stats.values():
        reason_counts.update(stats.get("reason_counts", {}))
    write_json(
        str(args.output_dir / "chemical_plan_summary.json"),
        {
            "total_rows_written": sum(int(stats["rows_written"]) for stats in split_stats.values()),
            "fixed_slot_rows": sum(int(stats["fixed_slot_rows"]) for stats in split_stats.values()),
            "chemical_plan_rows": sum(int(stats["chemical_plan_rows"]) for stats in split_stats.values()),
            "reason_counts": dict(reason_counts.most_common()),
        },
    )


if __name__ == "__main__":
    main()
