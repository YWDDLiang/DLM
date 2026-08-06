#!/usr/bin/env python3
"""Build fixed-slot + physical-header SFT data from existing fixed-slot JSONL."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from crystal_dlm.fixed_slot import (  # noqa: E402
    ANSWER_TOKEN_COUNT,
    CANONICAL_PROMPT,
    PROMPT_POOL,
    FixedSlotConfig,
    build_special_tokens,
    parse_fixed_slot_answer,
    write_json,
)
from crystal_dlm.physical_header import (  # noqa: E402
    PHYSICAL_HEADER_ANSWER_TOKEN_COUNT,
    PHYSICAL_HEADER_CANONICAL_PROMPT,
    PHYSICAL_HEADER_PROMPT_POOL,
    build_physical_header_special_tokens,
    prepend_physical_header_to_answer,
)


def load_tokenizer(tokenizer_path: Optional[str]):
    if not tokenizer_path:
        return None
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    special_tokens = build_special_tokens() + build_physical_header_special_tokens()
    tokenizer.add_special_tokens({"additional_special_tokens": list(dict.fromkeys(special_tokens))})
    return tokenizer


def read_jsonl(path: Path, limit: Optional[int] = None) -> Iterable[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if limit is not None and idx >= limit:
                break
            if line.strip():
                yield json.loads(line)


def compute_prompt_length(tokenizer, prompt: str) -> Optional[int]:
    if tokenizer is None:
        return None
    encoded = tokenizer(prompt.rstrip() + "\n", add_special_tokens=False)
    return len(encoded["input_ids"])


def compute_answer_length(tokenizer, answer: str) -> Optional[int]:
    if tokenizer is None:
        return None
    encoded = tokenizer(answer, add_special_tokens=False)
    return len(encoded["input_ids"])


def prompt_for_split(split: str, rng: random.Random) -> str:
    if split == "train":
        return rng.choice(PHYSICAL_HEADER_PROMPT_POOL)
    return PHYSICAL_HEADER_CANONICAL_PROMPT


def update_counter(counter: Dict[str, int], key: Any, inc: int = 1) -> None:
    text = str(key)
    counter[text] = counter.get(text, 0) + int(inc)


def build_split(
    *,
    split: str,
    input_jsonl: Path,
    output_jsonl: Path,
    tokenizer,
    rng: random.Random,
    limit: Optional[int],
    answer_separator: str,
) -> Dict[str, Any]:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    stats: Dict[str, Any] = {
        "split": split,
        "input_jsonl": str(input_jsonl),
        "rows_seen": 0,
        "rows_written": 0,
        "failures": 0,
        "fixed_body_answer_token_count": ANSWER_TOKEN_COUNT,
        "answer_token_count": PHYSICAL_HEADER_ANSWER_TOKEN_COUNT,
        "answer_tokenizer_lengths": {},
        "num_atoms_histogram": {},
        "header_token_histogram": {},
        "composition_token_histogram": {},
        "lattice_system_token_histogram": {},
        "high_symmetry_token_histogram": {},
    }
    failure_path = output_jsonl.with_suffix(".failure_cases.jsonl")
    with output_jsonl.open("w", encoding="utf-8") as out, failure_path.open("w", encoding="utf-8") as failures:
        for row_idx, row in enumerate(read_jsonl(input_jsonl, limit=limit)):
            stats["rows_seen"] += 1
            try:
                fixed_answer = str(row["answer"])
                arrays = parse_fixed_slot_answer(fixed_answer, config=FixedSlotConfig(), strict=True)
                answer, header_labels = prepend_physical_header_to_answer(
                    fixed_answer,
                    separator=answer_separator,
                    config=FixedSlotConfig(),
                )
                prompt = prompt_for_split(split, rng)
                prompt_text = prompt.rstrip() + "\n"
                answer_model_length = compute_answer_length(tokenizer, answer)
                record = dict(row)
                metadata = dict(row.get("metadata") or {})
                metadata["physical_header"] = header_labels
                metadata["source_prompt"] = row.get("prompt")
                record.update(
                    {
                        "task": "physical_header_unconditional",
                        "representation": "fixed_slot_physical_header",
                        "prompt": prompt,
                        "answer": answer,
                        "fixed_slot_answer": fixed_answer,
                        "text": prompt_text + answer,
                        "prompt_length": compute_prompt_length(tokenizer, prompt),
                        "answer_model_length": answer_model_length,
                        "answer_token_count": PHYSICAL_HEADER_ANSWER_TOKEN_COUNT,
                        "loss_profile": "fixed_slot",
                        "metadata": metadata,
                    }
                )
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                stats["rows_written"] += 1
                if answer_model_length is not None:
                    update_counter(stats["answer_tokenizer_lengths"], answer_model_length)
                update_counter(stats["num_atoms_histogram"], arrays["num_atoms"])
                for token in header_labels.get("tokens", []):
                    update_counter(stats["header_token_histogram"], token)
                update_counter(stats["composition_token_histogram"], header_labels.get("composition_token"))
                update_counter(stats["lattice_system_token_histogram"], header_labels.get("lattice_system_token"))
                update_counter(stats["high_symmetry_token_histogram"], header_labels.get("high_symmetry_token"))
            except Exception as exc:
                stats["failures"] += 1
                failures.write(
                    json.dumps(
                        {
                            "split": split,
                            "row_idx": row_idx,
                            "reason": type(exc).__name__,
                            "message": str(exc),
                            "source_task": row.get("task"),
                            "source_loss_profile": row.get("loss_profile"),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    return stats


def write_markdown(output_dir: Path, stats: Dict[str, Any]) -> None:
    total_seen = sum(item["rows_seen"] for item in stats["splits"].values())
    total_written = sum(item["rows_written"] for item in stats["splits"].values())
    total_failures = sum(item["failures"] for item in stats["splits"].values())
    lines = [
        "# Physical Header SFT Data",
        "",
        f"- representation: `fixed_slot_physical_header`",
        f"- total_seen: `{total_seen}`",
        f"- total_written: `{total_written}`",
        f"- total_failures: `{total_failures}`",
        f"- answer_token_count: `{PHYSICAL_HEADER_ANSWER_TOKEN_COUNT}`",
        f"- header_token_count: `{PHYSICAL_HEADER_ANSWER_TOKEN_COUNT - ANSWER_TOKEN_COUNT}`",
        "",
        "## Split Summary",
        "",
        "| split | seen | written | failures |",
        "| --- | ---: | ---: | ---: |",
    ]
    for split, payload in stats["splits"].items():
        lines.append(
            f"| {split} | {payload['rows_seen']} | {payload['rows_written']} | {payload['failures']} |"
        )
    lines.extend(["", "## Train Header Counts", ""])
    train_stats = stats["splits"].get("train", {})
    for key in ("composition_token_histogram", "lattice_system_token_histogram", "high_symmetry_token_histogram"):
        lines.append(f"### {key}")
        counter = Counter(train_stats.get(key, {}))
        for token, count in counter.most_common(20):
            lines.append(f"- `{token}`: {count}")
        lines.append("")
    (output_dir / "result.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=PROJECT_ROOT / "data/dlm_sft/mp_20")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data/dlm_sft/mp_20_physical_header_v0")
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--seed", type=int, default=20260528)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--answer-separator",
        default="",
        help="Separator between header/body special tokens. Empty keeps one semantic token per tokenizer token.",
    )
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.tokenizer_path)
    rng = random.Random(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    run_config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    run_config.update(
        {
            "representation": "fixed_slot_physical_header",
            "answer_token_count": PHYSICAL_HEADER_ANSWER_TOKEN_COUNT,
            "body_answer_token_count": ANSWER_TOKEN_COUNT,
            "source_prompt_pool": PROMPT_POOL,
            "source_canonical_prompt": CANONICAL_PROMPT,
        }
    )
    write_json(str(args.output_dir / "run_config.json"), run_config)

    special_tokens = list(dict.fromkeys(build_special_tokens() + build_physical_header_special_tokens()))
    with (args.output_dir / "vocab_tokens.txt").open("w", encoding="utf-8") as handle:
        for token in special_tokens:
            handle.write(token + "\n")
    write_json(
        str(args.output_dir / "prompt_pool.json"),
        {
            "canonical": PHYSICAL_HEADER_CANONICAL_PROMPT,
            "pool": PHYSICAL_HEADER_PROMPT_POOL,
            "source_fixed_slot_canonical": CANONICAL_PROMPT,
        },
    )
    write_json(
        str(args.output_dir / "physical_header_config.json"),
        {
            "representation": "fixed_slot_physical_header",
            "answer_token_count": PHYSICAL_HEADER_ANSWER_TOKEN_COUNT,
            "body_answer_token_count": ANSWER_TOKEN_COUNT,
            "header_tokens": build_physical_header_special_tokens(),
        },
    )

    all_stats: Dict[str, Any] = {
        "representation": "fixed_slot_physical_header",
        "answer_token_count": PHYSICAL_HEADER_ANSWER_TOKEN_COUNT,
        "body_answer_token_count": ANSWER_TOKEN_COUNT,
        "special_token_count": len(special_tokens),
        "new_header_token_count": len(build_physical_header_special_tokens()),
        "tokenizer_path": args.tokenizer_path,
        "answer_separator": args.answer_separator,
        "prompt_lengths_computed": tokenizer is not None,
        "splits": {},
    }
    for split in ("train", "val", "test"):
        stats = build_split(
            split=split,
            input_jsonl=args.input_dir / f"{split}.jsonl",
            output_jsonl=args.output_dir / f"{split}.jsonl",
            tokenizer=tokenizer,
            rng=rng,
            limit=args.limit,
            answer_separator=args.answer_separator,
        )
        all_stats["splits"][split] = stats

    combined_failure_path = args.output_dir / "failure_cases.jsonl"
    with combined_failure_path.open("w", encoding="utf-8") as combined:
        for split in ("train", "val", "test"):
            split_failure_path = args.output_dir / f"{split}.failure_cases.jsonl"
            if split_failure_path.exists():
                combined.write(split_failure_path.read_text(encoding="utf-8"))

    tokenizer_report = {
        "tokenizer_path": args.tokenizer_path,
        "prompt_lengths_computed": tokenizer is not None,
        "special_token_count": len(special_tokens),
        "new_header_token_count": len(build_physical_header_special_tokens()),
        "answer_separator": args.answer_separator,
        "pad_token_id": None if tokenizer is None else tokenizer.pad_token_id,
        "mask_token_id": 126336,
        "pad_token_equals_mask": None if tokenizer is None else tokenizer.pad_token_id == 126336,
    }
    write_json(str(args.output_dir / "stats.json"), all_stats)
    write_json(str(args.output_dir / "tokenizer_report.json"), tokenizer_report)
    write_markdown(args.output_dir, all_stats)

    total_failures = sum(item["failures"] for item in all_stats["splits"].values())
    write_json(
        str(args.output_dir / "next_run_suggestion.json"),
        {
            "status": "CONTINUE" if total_failures == 0 else "WATCH",
            "suggestions": [
                "Run 32-row Slurm SFT smoke before any full physical-header SFT.",
            ],
        },
    )
    print(json.dumps({"output_dir": str(args.output_dir), "failures": total_failures}, indent=2))


if __name__ == "__main__":
    main()
