#!/usr/bin/env python3
"""Build compressed-token fixed-slot SFT JSONL from existing MP-20 JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from crystal_dlm.fixed_slot import (  # noqa: E402
    ANSWER_TOKEN_COUNT,
    FixedSlotError,
    arrays_to_structure,
    parse_fixed_slot_answer,
    tokenize_answer_text,
    write_json,
)
from crystal_dlm.fixed_slot_compressed import (  # noqa: E402
    CompressedFixedSlotConfig,
    arrays_to_compressed_answer,
    build_compressed_special_tokens,
    compressed_token_sources,
    parse_compressed_fixed_slot_answer,
)


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL") from exc


def load_tokenizer(tokenizer_path: Optional[str], special_tokens: list[str]):
    if not tokenizer_path:
        return None
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    return tokenizer


def import_crysllmgen_process_one(crysllmgen_dir: Path):
    sys.path.insert(0, str(crysllmgen_dir.resolve()))
    from data_utils import process_one

    return process_one


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


def validate_graph(answer: str, process_one, config: CompressedFixedSlotConfig) -> None:
    arrays = parse_compressed_fixed_slot_answer(answer, config=config)
    structure = arrays_to_structure(arrays)
    process_one(structure.to(fmt="cif"), True, False, "crystalnn", False, 0.01)


def build_config(args: argparse.Namespace) -> CompressedFixedSlotConfig:
    variant = args.variant
    share_coordinates = True
    share_lengths = variant in {"abl2", "abl3"} or args.share_lengths
    share_angles = variant == "abl3" or args.share_angles
    return CompressedFixedSlotConfig(
        share_coordinates=share_coordinates,
        share_lengths=share_lengths,
        share_angles=share_angles,
    )


def convert_record(
    record: Dict[str, Any],
    config: CompressedFixedSlotConfig,
    tokenizer,
    answer_separator: str,
) -> Dict[str, Any]:
    arrays = parse_fixed_slot_answer(str(record["answer"]), strict=False)
    answer, diagnostics = arrays_to_compressed_answer(
        lengths=arrays["lengths"],
        angles=arrays["angles"],
        species=arrays["species"],
        frac_coords=arrays["frac_coords"],
        config=config,
        separator=answer_separator,
    )
    prompt = str(record.get("prompt", "")).rstrip()
    converted = dict(record)
    converted["representation"] = config.representation
    converted["compression"] = {
        "share_coordinates": config.share_coordinates,
        "share_lengths": config.share_lengths,
        "share_angles": config.share_angles,
    }
    converted["answer"] = answer
    converted["text"] = prompt + "\n" + answer
    converted["prompt_length"] = compute_prompt_length(tokenizer, prompt)
    converted["answer_model_length"] = compute_answer_length(tokenizer, answer)
    converted["answer_semantic_token_count"] = len(tokenize_answer_text(str(record["answer"])))
    converted["compressed_answer_semantic_token_count"] = len(
        parse_compressed_fixed_slot_answer(answer, config=config, strict=True)["tokens"]
    )
    metadata = dict(converted.get("metadata") or {})
    metadata["compressed_encode_diagnostics"] = diagnostics.to_dict()
    converted["metadata"] = metadata
    return converted


def convert_split(
    split: str,
    input_jsonl: Path,
    output_jsonl: Path,
    tokenizer,
    config: CompressedFixedSlotConfig,
    process_one,
    answer_separator: str,
    skip_graph_validation: bool,
) -> Dict[str, Any]:
    stats: Dict[str, Any] = {
        "split": split,
        "input_jsonl": str(input_jsonl),
        "rows_seen": 0,
        "rows_written": 0,
        "failures": 0,
        "answer_token_count": ANSWER_TOKEN_COUNT,
        "answer_tokenizer_lengths": {},
        "atom_count_histogram": {},
        "element_histogram": {},
        "length_clips": 0,
        "angle_clips": 0,
        "coord_clips": 0,
        "coord_wraps": 0,
    }
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    failure_path = output_jsonl.with_suffix(".failure_cases.jsonl")
    with output_jsonl.open("w", encoding="utf-8") as out, failure_path.open(
        "w", encoding="utf-8"
    ) as failures:
        for row_idx, record in enumerate(iter_jsonl(input_jsonl)):
            stats["rows_seen"] += 1
            try:
                converted = convert_record(
                    record=record,
                    config=config,
                    tokenizer=tokenizer,
                    answer_separator=answer_separator,
                )
                if not skip_graph_validation:
                    validate_graph(converted["answer"], process_one, config=config)
                arrays = parse_compressed_fixed_slot_answer(
                    converted["answer"], config=config, strict=True
                )
                out.write(json.dumps(converted, ensure_ascii=False) + "\n")
                stats["rows_written"] += 1
                diag = converted["metadata"]["compressed_encode_diagnostics"]
                stats["length_clips"] += int(diag.get("length_clips", 0))
                stats["angle_clips"] += int(diag.get("angle_clips", 0))
                stats["coord_clips"] += int(diag.get("coord_clips", 0))
                stats["coord_wraps"] += int(diag.get("coord_wraps", 0))
                if converted.get("answer_model_length") is not None:
                    key = str(converted["answer_model_length"])
                    stats["answer_tokenizer_lengths"][key] = (
                        stats["answer_tokenizer_lengths"].get(key, 0) + 1
                    )
                atom_count = int(arrays["num_atoms"])
                stats["atom_count_histogram"][str(atom_count)] = (
                    stats["atom_count_histogram"].get(str(atom_count), 0) + 1
                )
                for symbol in arrays["species"]:
                    stats["element_histogram"][symbol] = (
                        stats["element_histogram"].get(symbol, 0) + 1
                    )
            except Exception as exc:
                stats["failures"] += 1
                failures.write(
                    json.dumps(
                        {
                            "split": split,
                            "row_idx": row_idx,
                            "reason": type(exc).__name__,
                            "message": str(exc),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    return stats


def summarize_token_counts(config: CompressedFixedSlotConfig) -> Dict[str, Any]:
    tokens = build_compressed_special_tokens(config)
    families = Counter()
    for token in tokens:
        if token.startswith("<N_"):
            families["count"] += 1
        elif token.startswith("<L_") or token.startswith("<LA_") or token.startswith("<LB_") or token.startswith("<LC_"):
            families["length"] += 1
        elif token.startswith("<A_") or token.startswith("<AA_") or token.startswith("<AB_") or token.startswith("<AG_"):
            families["angle"] += 1
        elif token.startswith("<S"):
            families["slot"] += 1
        elif token.startswith("<E_"):
            families["element"] += 1
        elif token.startswith("<C_") or token.startswith("<X_") or token.startswith("<Y_") or token.startswith("<Z_"):
            families["coord"] += 1
        elif token == "<EMPTY>":
            families["empty"] += 1
    return {"special_token_count": len(tokens), "families": dict(families)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=PROJECT_ROOT / "data/dlm_sft/mp_20")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data/dlm_sft/mp_20_fixed_slot_compressed_abl1",
    )
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument(
        "--crysllmgen-dir",
        type=Path,
        default=PROJECT_ROOT / "reference/crysllmgen",
    )
    parser.add_argument("--variant", choices=["abl1", "abl2", "abl3"], default="abl1")
    parser.add_argument("--share-lengths", action="store_true")
    parser.add_argument("--share-angles", action="store_true")
    parser.add_argument("--answer-separator", default="")
    parser.add_argument("--skip-graph-validation", action="store_true")
    args = parser.parse_args()

    config = build_config(args)
    special_tokens = build_compressed_special_tokens(config)
    tokenizer = load_tokenizer(args.tokenizer_path, special_tokens)
    process_one = None if args.skip_graph_validation else import_crysllmgen_process_one(args.crysllmgen_dir)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    write_json(str(args.output_dir / "run_config.json"), run_config)
    write_json(str(args.output_dir / "compressed_token_config.json"), config.to_dict())
    write_json(str(args.output_dir / "token_map.json"), compressed_token_sources(config))
    write_json(str(args.output_dir / "token_count_summary.json"), summarize_token_counts(config))
    with (args.output_dir / "vocab_tokens.txt").open("w", encoding="utf-8") as handle:
        for token in special_tokens:
            handle.write(token + "\n")

    all_stats: Dict[str, Any] = {
        "schema": config.to_dict(),
        "token_summary": summarize_token_counts(config),
        "tokenizer_path": args.tokenizer_path,
        "answer_separator": args.answer_separator,
        "graph_validation": not args.skip_graph_validation,
        "splits": {},
    }
    for split in ("train", "val", "test"):
        all_stats["splits"][split] = convert_split(
            split=split,
            input_jsonl=args.input_dir / f"{split}.jsonl",
            output_jsonl=args.output_dir / f"{split}.jsonl",
            tokenizer=tokenizer,
            config=config,
            process_one=process_one,
            answer_separator=args.answer_separator,
            skip_graph_validation=args.skip_graph_validation,
        )

    combined_failure_path = args.output_dir / "failure_cases.jsonl"
    with combined_failure_path.open("w", encoding="utf-8") as combined:
        for split in ("train", "val", "test"):
            split_failure_path = args.output_dir / f"{split}.failure_cases.jsonl"
            if split_failure_path.exists():
                combined.write(split_failure_path.read_text(encoding="utf-8"))

    total_seen = sum(item["rows_seen"] for item in all_stats["splits"].values())
    total_written = sum(item["rows_written"] for item in all_stats["splits"].values())
    total_failures = sum(item["failures"] for item in all_stats["splits"].values())
    write_json(str(args.output_dir / "stats.json"), all_stats)
    result_md = "\n".join(
        [
            "# fixed-slot compressed data build",
            "",
            f"- variant: `{args.variant}`",
            f"- rows seen: {total_seen}",
            f"- rows written: {total_written}",
            f"- failures: {total_failures}",
            f"- answer semantic positions: {ANSWER_TOKEN_COUNT}",
            f"- special token count: {len(special_tokens)}",
            f"- share coordinates: {config.share_coordinates}",
            f"- share lengths: {config.share_lengths}",
            f"- share angles: {config.share_angles}",
            "",
        ]
    )
    (args.output_dir / "result.md").write_text(result_md, encoding="utf-8")


if __name__ == "__main__":
    main()
