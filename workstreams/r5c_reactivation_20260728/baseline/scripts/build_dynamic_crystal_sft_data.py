#!/usr/bin/env python3
"""Build MP-20 dynamic-v1 SFT JSONL for LLaDA experiments."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from crystal_dlm.dynamic_crystal import (  # noqa: E402
    CANONICAL_DYNAMIC_PROMPT,
    DYNAMIC_PROMPT_POOL,
    build_special_tokens,
    dynamic_max_answer_token_count,
    metadata_from_csv_row,
    parse_dynamic_answer,
    structure_to_dynamic_answer,
    write_json,
)
from crystal_dlm.fixed_slot import FixedSlotConfig  # noqa: E402


def load_tokenizer(tokenizer_path: Optional[str]):
    if not tokenizer_path:
        return None
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    tokenizer.add_special_tokens({"additional_special_tokens": build_special_tokens()})
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
    return rng.choice(DYNAMIC_PROMPT_POOL) if split == "train" else CANONICAL_DYNAMIC_PROMPT


def compute_length(tokenizer, text: str) -> Optional[int]:
    if tokenizer is None:
        return None
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def validate_graph(answer: str, process_one) -> None:
    from crystal_dlm.dynamic_crystal import arrays_to_structure

    arrays = parse_dynamic_answer(answer, strict=True)
    structure = arrays_to_structure(arrays)
    process_one(structure.to(fmt="cif"), True, False, "crystalnn", False, 0.01)


def build_split(
    *,
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
        "representation": "dynamic_v1",
        "max_answer_token_count": dynamic_max_answer_token_count(config),
        "length_clips": 0,
        "angle_clips": 0,
        "coord_clips": 0,
        "coord_wraps": 0,
        "answer_semantic_lengths": {},
        "answer_tokenizer_lengths": {},
        "atom_count_histogram": {},
        "element_histogram": {},
    }
    failure_path = output_jsonl.with_suffix(".failure_cases.jsonl")
    with output_jsonl.open("w", encoding="utf-8") as out, failure_path.open("w", encoding="utf-8") as failures:
        for row_idx, row in enumerate(read_rows(input_csv, limit=limit)):
            stats["rows_seen"] += 1
            try:
                structure = Structure.from_str(row["cif"], fmt="cif")
                answer, diagnostics = structure_to_dynamic_answer(
                    structure,
                    config=config,
                    separator=answer_separator,
                )
                if not skip_graph_validation:
                    validate_graph(answer, process_one)
                prompt = prompt_for_split(split, rng)
                prompt_text = prompt.rstrip() + "\n"
                answer_model_length = compute_length(tokenizer, answer)
                arrays = parse_dynamic_answer(answer, strict=True)
                record = {
                    "task": "unconditional",
                    "representation": "dynamic_v1",
                    "prompt": prompt,
                    "answer": answer,
                    "text": prompt_text + answer,
                    "prompt_length": compute_length(tokenizer, prompt_text),
                    "answer_model_length": answer_model_length,
                    "answer_semantic_length": len(arrays["tokens"]),
                    "metadata": metadata_from_csv_row(row),
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                stats["rows_written"] += 1
                stats["length_clips"] += diagnostics.length_clips
                stats["angle_clips"] += diagnostics.angle_clips
                stats["coord_clips"] += diagnostics.coord_clips
                stats["coord_wraps"] += diagnostics.coord_wraps
                semantic_key = str(len(arrays["tokens"]))
                stats["answer_semantic_lengths"][semantic_key] = stats["answer_semantic_lengths"].get(semantic_key, 0) + 1
                if answer_model_length is not None:
                    key = str(answer_model_length)
                    stats["answer_tokenizer_lengths"][key] = stats["answer_tokenizer_lengths"].get(key, 0) + 1
                atom_count = int(arrays["num_atoms"])
                stats["atom_count_histogram"][str(atom_count)] = stats["atom_count_histogram"].get(str(atom_count), 0) + 1
                for symbol in arrays["species"]:
                    stats["element_histogram"][symbol] = stats["element_histogram"].get(symbol, 0) + 1
            except Exception as exc:
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
                            "event": "dynamic_data_builder_progress",
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
    parser.add_argument("--input-dir", type=Path, default=PROJECT_ROOT / "reference/crysllmgen/data/mp_20")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data/dlm_sft/mp_20_dynamic_v1")
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--crysllmgen-dir", type=Path, default=PROJECT_ROOT / "reference/crysllmgen")
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--skip-graph-validation", action="store_true")
    parser.add_argument("--answer-separator", default="")
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.tokenizer_path)
    process_one = None if args.skip_graph_validation else import_crysllmgen_process_one(args.crysllmgen_dir)
    config = FixedSlotConfig()
    rng = random.Random(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    splits: Dict[str, Any] = {}
    for split in ("train", "val", "test"):
        splits[split] = build_split(
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

    vocab_tokens = build_special_tokens(config)
    (args.output_dir / "vocab_tokens.txt").write_text("\n".join(vocab_tokens) + "\n", encoding="utf-8")
    summary = {
        "representation": "dynamic_v1",
        "max_answer_token_count": dynamic_max_answer_token_count(config),
        "answer_token_count": dynamic_max_answer_token_count(config),
        "splits": splits,
        "prompt_pool": DYNAMIC_PROMPT_POOL,
        "special_token_count": len(vocab_tokens),
    }
    write_json(str(args.output_dir / "stats.json"), summary)
    write_json(
        str(args.output_dir / "_SUCCESS"),
        {
            "representation": "dynamic_v1",
            "complete": True,
            "splits": {
                split: {
                    "rows_seen": split_stats["rows_seen"],
                    "rows_written": split_stats["rows_written"],
                    "failures": split_stats["failures"],
                }
                for split, split_stats in splits.items()
            },
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
