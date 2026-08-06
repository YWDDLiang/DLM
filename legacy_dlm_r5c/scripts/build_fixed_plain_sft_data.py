#!/usr/bin/env python3
"""Build MP-20 fixed-schema plain-text staged SFT JSONL."""

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

from crystal_dlm.fixed_plain import (  # noqa: E402
    FIXED_PLAIN_MODULE_TO_ID,
    FIXED_PLAIN_PROMPT_VERSION,
    coords_prompt,
    count_prompt,
    elements_prompt,
    lattice_prompt,
    metadata_from_csv_row,
    split_plain_tokens,
    structure_to_plain_answer,
    tokenize_plain_answer,
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


def token_len(tokenizer, text: str) -> Optional[int]:
    if tokenizer is None:
        return None
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def validate_graph(arrays: Dict[str, Any], process_one) -> None:
    from crystal_dlm.fixed_plain import arrays_to_structure

    process_one(arrays_to_structure(arrays).to(fmt="cif"), True, False, "crystalnn", False, 0.01)


def hist_add(hist: Dict[str, int], key: Any, count: int = 1) -> None:
    hist[str(key)] = int(hist.get(str(key), 0)) + int(count)


def module_record(
    *,
    module: str,
    prompt: str,
    answer: str,
    tokenizer,
    arrays: Dict[str, Any],
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    prompt_text = prompt.rstrip() + "\n"
    answer_text = answer.rstrip()
    return {
        "task": module,
        "module": module,
        "module_id": int(FIXED_PLAIN_MODULE_TO_ID[module]),
        "representation": "fixed_plain",
        "prompt_version": FIXED_PLAIN_PROMPT_VERSION,
        "prompt": prompt.rstrip(),
        "answer": answer_text,
        "text": prompt_text + answer_text,
        "prompt_length": token_len(tokenizer, prompt_text),
        "answer_model_length": token_len(tokenizer, answer_text),
        "num_atoms": int(arrays["num_atoms"]),
        "metadata": dict(metadata),
    }


def write_record(out, record: Dict[str, Any], stats: Dict[str, Any]) -> None:
    out.write(json.dumps(record, ensure_ascii=False) + "\n")
    stats["rows_written"] += 1
    hist_add(stats["module_counts"], record["module"])
    if record["answer_model_length"] is not None:
        hist_add(stats["answer_tokenizer_lengths"], record["answer_model_length"])
    if record["prompt_length"] is not None:
        hist_add(stats["prompt_tokenizer_lengths"], record["prompt_length"])


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
        "representation": "fixed_plain",
        "prompt_version": FIXED_PLAIN_PROMPT_VERSION,
        "module_counts": {},
        "answer_tokenizer_lengths": {},
        "prompt_tokenizer_lengths": {},
        "atom_count_histogram": {},
        "num_elements_histogram": {},
        "element_histogram": {},
    }
    rng = random.Random(int(seed) + {"train": 0, "val": 1, "test": 2}[split] * 1_000_003)
    failure_path = output_jsonl.with_suffix(".failure_cases.jsonl")
    with output_jsonl.open("w", encoding="utf-8") as out, failure_path.open("w", encoding="utf-8") as failures:
        for row_idx, row in enumerate(read_rows(input_csv, limit=limit)):
            stats["rows_seen"] += 1
            try:
                structure = Structure.from_str(row["cif"], fmt="cif")
                # Keep the conservative first version aligned with fixed-slot:
                # no origin shift and no site permutation.
                answer, _ = structure_to_plain_answer(structure, separator=" ")
                tokens = tokenize_plain_answer(answer)
                modules = split_plain_tokens(tokens)
                arrays = modules["arrays"]
                if not skip_graph_validation:
                    validate_graph(arrays, process_one)
                metadata = metadata_from_csv_row(row)
                count_text = modules["count_text"]
                lattice_text = modules["lattice_text"]
                elements_text = modules["elements_text"]
                coords_text = modules["coords_text"]
                module_rows = (
                    ("count", count_prompt(), count_text),
                    ("lattice", lattice_prompt(count_text), lattice_text),
                    ("elements", elements_prompt(count_text, lattice_text), elements_text),
                    ("coords", coords_prompt(count_text, lattice_text, elements_text), coords_text),
                )
                for module, prompt, module_answer in module_rows:
                    write_record(
                        out,
                        module_record(
                            module=module,
                            prompt=prompt,
                            answer=module_answer,
                            tokenizer=tokenizer,
                            arrays=arrays,
                            metadata=metadata,
                        ),
                        stats,
                    )
                stats["structures_written"] += 1
                hist_add(stats["atom_count_histogram"], arrays["num_atoms"])
                hist_add(stats["num_elements_histogram"], len(set(arrays["species"])))
                for symbol in arrays["species"]:
                    hist_add(stats["element_histogram"], symbol)
                if split == "train":
                    rng.random()
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
                print(json.dumps({"event": "fixed_plain_builder_progress", **stats}, ensure_ascii=False), flush=True)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=PROJECT_ROOT / "reference/crysllmgen/data/mp_20")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data/dlm_sft/mp_20_fixed_plain")
    parser.add_argument("--tokenizer-path", default="/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct/")
    parser.add_argument("--crysllmgen-dir", type=Path, default=PROJECT_ROOT / "reference/crysllmgen")
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--skip-graph-validation", action="store_true")
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.tokenizer_path)
    process_one = None if args.skip_graph_validation else import_crysllmgen_process_one(args.crysllmgen_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(str(args.output_dir / "run_config.json"), {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()})

    all_stats: Dict[str, Any] = {
        "representation": "fixed_plain",
        "prompt_version": FIXED_PLAIN_PROMPT_VERSION,
        "tokenizer_path": args.tokenizer_path,
        "graph_validation": not args.skip_graph_validation,
        "splits": {},
    }
    max_answer_model_length = 0
    max_prompt_model_length = 0
    for split in ("train", "val", "test"):
        stats = build_split(
            split=split,
            input_csv=args.input_dir / f"{split}.csv",
            output_jsonl=args.output_dir / f"{split}.jsonl",
            tokenizer=tokenizer,
            process_one=process_one,
            seed=args.seed,
            limit=args.limit,
            skip_graph_validation=args.skip_graph_validation,
            progress_every=args.progress_every,
        )
        all_stats["splits"][split] = stats
        if stats["answer_tokenizer_lengths"]:
            max_answer_model_length = max(max_answer_model_length, max(map(int, stats["answer_tokenizer_lengths"].keys())))
        if stats["prompt_tokenizer_lengths"]:
            max_prompt_model_length = max(max_prompt_model_length, max(map(int, stats["prompt_tokenizer_lengths"].keys())))
    all_stats["max_answer_model_length"] = max_answer_model_length
    all_stats["max_prompt_model_length"] = max_prompt_model_length
    all_stats["answer_token_count"] = max_answer_model_length + 8
    all_stats["max_length_recommended"] = max_answer_model_length + max_prompt_model_length + 16
    write_json(str(args.output_dir / "stats.json"), all_stats)
    (args.output_dir / "_SUCCESS").write_text("ok\n", encoding="utf-8")


if __name__ == "__main__":
    main()
