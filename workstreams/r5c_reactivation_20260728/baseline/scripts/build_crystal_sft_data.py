#!/usr/bin/env python3
"""Build MP-20 fixed-slot SFT JSONL for LLaDA experiments."""

from __future__ import annotations

import argparse
import csv
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
    FixedSlotError,
    build_special_tokens,
    metadata_from_csv_row,
    parse_fixed_slot_answer,
    structure_to_answer,
    write_json,
)


def load_tokenizer(tokenizer_path: Optional[str]):
    if not tokenizer_path:
        return None
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    special_tokens = build_special_tokens()
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    return tokenizer


def import_crysllmgen_process_one(crysllmgen_dir: Path):
    sys.path.insert(0, str(crysllmgen_dir.resolve()))
    from data_utils import process_one

    return process_one


def read_rows(csv_path: Path, limit: Optional[int] = None) -> Iterable[Dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader):
            if limit is not None and idx >= limit:
                break
            yield row


def prompt_for_split(split: str, rng: random.Random) -> str:
    if split == "train":
        return rng.choice(PROMPT_POOL)
    return CANONICAL_PROMPT


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


def validate_graph(answer: str, process_one) -> None:
    arrays = parse_fixed_slot_answer(answer)
    from crystal_dlm.fixed_slot import arrays_to_structure

    structure = arrays_to_structure(arrays)
    process_one(structure.to(fmt="cif"), True, False, "crystalnn", False, 0.01)


def build_split(
    split: str,
    input_csv: Path,
    output_jsonl: Path,
    tokenizer,
    process_one,
    config: FixedSlotConfig,
    rng: random.Random,
    limit: Optional[int],
    skip_graph_validation: bool,
    progress_every: int,
    answer_separator: str,
) -> Dict[str, Any]:
    from pymatgen.core import Structure

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    stats: Dict[str, Any] = {
        "split": split,
        "input_csv": str(input_csv),
        "rows_seen": 0,
        "rows_written": 0,
        "failures": 0,
        "answer_token_count": ANSWER_TOKEN_COUNT,
        "length_clips": 0,
        "angle_clips": 0,
        "coord_clips": 0,
        "coord_wraps": 0,
        "answer_tokenizer_lengths": {},
        "atom_count_histogram": {},
        "element_histogram": {},
    }
    failure_path = output_jsonl.with_suffix(".failure_cases.jsonl")
    with output_jsonl.open("w", encoding="utf-8") as out, failure_path.open(
        "w", encoding="utf-8"
    ) as failures:
        for row_idx, row in enumerate(read_rows(input_csv, limit=limit)):
            stats["rows_seen"] += 1
            try:
                cif = row["cif"]
                structure = Structure.from_str(cif, fmt="cif")
                answer, diagnostics = structure_to_answer(
                    structure,
                    config=config,
                    separator=answer_separator,
                )
                if not skip_graph_validation:
                    validate_graph(answer, process_one)
                prompt = prompt_for_split(split, rng)
                prompt_text = prompt.rstrip() + "\n"
                answer_model_length = compute_answer_length(tokenizer, answer)
                record = {
                    "task": "unconditional",
                    "prompt": prompt,
                    "answer": answer,
                    "text": prompt_text + answer,
                    "prompt_length": compute_prompt_length(tokenizer, prompt),
                    "answer_model_length": answer_model_length,
                    "metadata": metadata_from_csv_row(row),
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                stats["rows_written"] += 1
                stats["length_clips"] += diagnostics.length_clips
                stats["angle_clips"] += diagnostics.angle_clips
                stats["coord_clips"] += diagnostics.coord_clips
                stats["coord_wraps"] += diagnostics.coord_wraps
                if answer_model_length is not None:
                    key = str(answer_model_length)
                    stats["answer_tokenizer_lengths"][key] = (
                        stats["answer_tokenizer_lengths"].get(key, 0) + 1
                    )
                atom_count = len(structure)
                stats["atom_count_histogram"][str(atom_count)] = (
                    stats["atom_count_histogram"].get(str(atom_count), 0) + 1
                )
                for site in structure.sites:
                    symbol = site.specie.symbol
                    stats["element_histogram"][symbol] = (
                        stats["element_histogram"].get(symbol, 0) + 1
                    )
            except Exception as exc:  # Keep data failures auditable.
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
                            "event": "data_builder_progress",
                            "split": split,
                            "rows_seen": stats["rows_seen"],
                            "rows_written": stats["rows_written"],
                            "failures": stats["failures"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=PROJECT_ROOT / "reference/crysllmgen/data/mp_20",
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data/dlm_sft/mp_20")
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument(
        "--crysllmgen-dir",
        type=Path,
        default=PROJECT_ROOT / "reference/crysllmgen",
    )
    parser.add_argument("--seed", type=int, default=20260514)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--skip-graph-validation", action="store_true")
    parser.add_argument(
        "--answer-separator",
        default="",
        help="Separator between fixed-slot schema tokens. Empty keeps 107 semantic tokens at 107 tokenizer tokens.",
    )
    args = parser.parse_args()

    config = FixedSlotConfig()
    tokenizer = load_tokenizer(args.tokenizer_path)
    process_one = import_crysllmgen_process_one(args.crysllmgen_dir)
    rng = random.Random(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    write_json(str(args.output_dir / "run_config.json"), run_config)

    special_tokens = build_special_tokens(config)
    with (args.output_dir / "vocab_tokens.txt").open("w", encoding="utf-8") as handle:
        for token in special_tokens:
            handle.write(token + "\n")
    with (args.output_dir / "prompt_pool.json").open("w", encoding="utf-8") as handle:
        json.dump({"canonical": CANONICAL_PROMPT, "pool": PROMPT_POOL}, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    all_stats: Dict[str, Any] = {
        "schema": config.to_dict(),
        "special_token_count": len(special_tokens),
        "tokenizer_path": args.tokenizer_path,
        "answer_separator": args.answer_separator,
        "prompt_lengths_computed": tokenizer is not None,
        "graph_validation": not args.skip_graph_validation,
        "splits": {},
    }
    for split in ("train", "val", "test"):
        stats = build_split(
            split=split,
            input_csv=args.input_dir / f"{split}.csv",
            output_jsonl=args.output_dir / f"{split}.jsonl",
            tokenizer=tokenizer,
            process_one=process_one,
            config=config,
            rng=rng,
            limit=args.limit,
            skip_graph_validation=args.skip_graph_validation,
            progress_every=args.progress_every,
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
        "answer_separator": args.answer_separator,
        "pad_token_id": None if tokenizer is None else tokenizer.pad_token_id,
        "mask_token_id": 126336,
        "pad_token_equals_mask": None if tokenizer is None else tokenizer.pad_token_id == 126336,
    }
    write_json(str(args.output_dir / "stats.json"), all_stats)
    write_json(str(args.output_dir / "tokenizer_report.json"), tokenizer_report)
    total_seen = sum(item["rows_seen"] for item in all_stats["splits"].values())
    total_written = sum(item["rows_written"] for item in all_stats["splits"].values())
    total_failures = sum(item["failures"] for item in all_stats["splits"].values())
    result_md = "\n".join(
        [
            "# result",
            "",
            "## 数据构建结果",
            "",
            f"- 总读取行数：{total_seen}",
            f"- 总写出行数：{total_written}",
            f"- 总失败数：{total_failures}",
            f"- 固定 answer token 数：{ANSWER_TOKEN_COUNT}",
            f"- answer separator：`{args.answer_separator}`",
            f"- tokenizer 路径：`{args.tokenizer_path}`",
            f"- graph validation：{not args.skip_graph_validation}",
            "",
        ]
    )
    (args.output_dir / "result.md").write_text(result_md, encoding="utf-8")
    write_json(
        str(args.output_dir / "next_run_suggestion.json"),
        {
            "status": "CONTINUE" if total_failures == 0 else "WATCH",
            "suggestions": [
                "若全量数据构建 failure 为 0，进入 32/256 sample SFT smoke。",
            ],
        },
    )


if __name__ == "__main__":
    main()
