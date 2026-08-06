#!/usr/bin/env python3
"""Build MP-20 CIF-lite modular SFT JSONL for LLaDA experiments."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from crystal_dlm.cif_lite import (  # noqa: E402
    CIF_LITE_PROMPT_POOL,
    CIF_LITE_PROMPT_VERSION,
    build_module_prompt,
    metadata_from_csv_row,
    module_record,
    parse_cif_lite_modules,
    structure_to_cif_lite_modules,
    write_json,
)


def load_tokenizer(tokenizer_path: Optional[str]):
    if not tokenizer_path:
        return None
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
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


def compute_length(tokenizer, text: str) -> Optional[int]:
    if tokenizer is None:
        return None
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def validate_graph(modules: Dict[str, Any], process_one) -> None:
    arrays = parse_cif_lite_modules(
        modules["composition"],
        modules["lattice"],
        modules["sites"],
    )
    from crystal_dlm.cif_lite import arrays_to_structure

    process_one(arrays_to_structure(arrays).to(fmt="cif"), True, False, "crystalnn", False, 0.01)


def choose_prompt(split: str, module: str, rng: random.Random, modules: Dict[str, Any]) -> str:
    if split == "train":
        prefix = rng.choice(CIF_LITE_PROMPT_POOL[module])
        if module == "composition":
            return prefix
        if module == "lattice":
            return prefix + "\n\n" + modules["composition"].rstrip() + "\n"
        if module == "sites":
            return prefix + "\n\n" + modules["composition"].rstrip() + "\n\n" + modules["lattice"].rstrip() + "\n"
    if module == "composition":
        return build_module_prompt("composition")
    if module == "lattice":
        return build_module_prompt("lattice", composition_block=modules["composition"])
    if module == "sites":
        return build_module_prompt(
            "sites",
            composition_block=modules["composition"],
            lattice_block=modules["lattice"],
        )
    raise ValueError(f"Unknown module {module!r}")


def histogram_add(histogram: Dict[str, int], key: Any, count: int = 1) -> None:
    text_key = str(key)
    histogram[text_key] = int(histogram.get(text_key, 0)) + int(count)


def build_split(
    *,
    split: str,
    input_csv: Path,
    output_jsonl: Path,
    tokenizer,
    process_one,
    seed: int,
    limit: Optional[int],
    skip_graph_validation: bool,
    progress_every: int,
    train_origin_shift: bool,
    train_permute_sites: bool,
) -> Dict[str, Any]:
    from pymatgen.core import Structure

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    stats: Dict[str, Any] = {
        "split": split,
        "input_csv": str(input_csv),
        "rows_seen": 0,
        "structures_written": 0,
        "rows_written": 0,
        "failures": 0,
        "representation": "cif_lite_modular",
        "module_counts": {},
        "answer_tokenizer_lengths": {},
        "prompt_tokenizer_lengths": {},
        "atom_count_histogram": {},
        "num_elements_histogram": {},
        "element_histogram": {},
    }
    failure_path = output_jsonl.with_suffix(".failure_cases.jsonl")
    with output_jsonl.open("w", encoding="utf-8") as out, failure_path.open("w", encoding="utf-8") as failures:
        for row_idx, row in enumerate(read_rows(input_csv, limit=limit)):
            stats["rows_seen"] += 1
            try:
                structure = Structure.from_str(row["cif"], fmt="cif")
                split_offset = {"train": 0, "val": 1, "test": 2}[split]
                row_rng = random.Random(int(seed) + split_offset * 1_000_003 + row_idx)
                modules = structure_to_cif_lite_modules(
                    structure,
                    rng=row_rng,
                    origin_shift=split == "train" and train_origin_shift,
                    permute_sites=split == "train" and train_permute_sites,
                )
                if not skip_graph_validation:
                    validate_graph(modules, process_one)
                metadata = metadata_from_csv_row(row)
                for module in ("composition", "lattice", "sites"):
                    prompt = choose_prompt(split, module, row_rng, modules)
                    record = module_record(
                        module=module,
                        modules=modules,
                        metadata=metadata,
                        prompt=prompt,
                        tokenizer=tokenizer,
                    )
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    stats["rows_written"] += 1
                    histogram_add(stats["module_counts"], module)
                    if record["answer_model_length"] is not None:
                        histogram_add(stats["answer_tokenizer_lengths"], record["answer_model_length"])
                    if record["prompt_length"] is not None:
                        histogram_add(stats["prompt_tokenizer_lengths"], record["prompt_length"])
                arrays = modules["arrays"]
                stats["structures_written"] += 1
                histogram_add(stats["atom_count_histogram"], arrays["num_atoms"])
                histogram_add(stats["num_elements_histogram"], len(set(arrays["species"])))
                for symbol, count in Counter(arrays["species"]).items():
                    histogram_add(stats["element_histogram"], symbol, count)
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
                            "event": "cif_lite_data_builder_progress",
                            "split": split,
                            "rows_seen": stats["rows_seen"],
                            "structures_written": stats["structures_written"],
                            "rows_written": stats["rows_written"],
                            "failures": stats["failures"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    return stats


def max_hist_key(histogram: Dict[str, int]) -> int:
    if not histogram:
        return 0
    return max(int(float(key)) for key in histogram)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=PROJECT_ROOT / "reference/crysllmgen/data/mp_20")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data/dlm_sft/mp_20_cif_lite_modular")
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--crysllmgen-dir", type=Path, default=PROJECT_ROOT / "reference/crysllmgen")
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--skip-graph-validation", action="store_true")
    parser.add_argument("--no-train-origin-shift", dest="train_origin_shift", action="store_false")
    parser.add_argument("--train-origin-shift", dest="train_origin_shift", action="store_true", default=True)
    parser.add_argument("--train-permute-sites", action="store_true")
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.tokenizer_path)
    process_one = None if args.skip_graph_validation else import_crysllmgen_process_one(args.crysllmgen_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    splits: Dict[str, Any] = {}
    for split in ("train", "val", "test"):
        splits[split] = build_split(
            split=split,
            input_csv=args.input_dir / f"{split}.csv",
            output_jsonl=args.output_dir / f"{split}.jsonl",
            tokenizer=tokenizer,
            process_one=process_one,
            seed=args.seed,
            limit=args.limit,
            skip_graph_validation=args.skip_graph_validation,
            progress_every=args.progress_every,
            train_origin_shift=args.train_origin_shift,
            train_permute_sites=args.train_permute_sites,
        )

    max_answer = max(max_hist_key(item["answer_tokenizer_lengths"]) for item in splits.values())
    max_prompt = max(max_hist_key(item["prompt_tokenizer_lengths"]) for item in splits.values())
    summary = {
        "representation": "cif_lite_modular",
        "splits": splits,
        "prompt_pool": CIF_LITE_PROMPT_POOL,
        "prompt_version": CIF_LITE_PROMPT_VERSION,
        "answer_token_count": max_answer + 8,
        "max_answer_model_length": max_answer,
        "max_prompt_model_length": max_prompt,
        "max_length_recommended": max_prompt + max_answer + 16,
        "special_token_count": 0,
        "train_origin_shift": bool(args.train_origin_shift),
        "train_permute_sites": bool(args.train_permute_sites),
    }
    write_json(str(args.output_dir / "stats.json"), summary)
    write_json(
        str(args.output_dir / "_SUCCESS"),
        {
            "representation": "cif_lite_modular",
            "prompt_version": CIF_LITE_PROMPT_VERSION,
            "complete": True,
            "splits": {
                split: {
                    "rows_seen": split_stats["rows_seen"],
                    "structures_written": split_stats["structures_written"],
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
