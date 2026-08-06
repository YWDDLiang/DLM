#!/usr/bin/env python3
"""Build R5 atom-sequence plan-state SFT data from MP-20 CSV splits."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from crystal_dlm.dynamic_crystal import parse_dynamic_answer, structure_to_dynamic_answer  # noqa: E402
from crystal_dlm.fixed_slot import metadata_from_csv_row, write_json  # noqa: E402
from crystal_dlm.r5_plan_state import (  # noqa: E402
    PLAN_STATE_VERSION,
    build_atomfields_plan_prompt,
    build_atomseq_plan_prompt,
    build_atomslots_plan_prompt,
    build_countfields_plan_prompt,
    build_countvalence_plan_prompt,
    parse_atomfields_plan_state,
    parse_atomseq_plan_state,
    parse_atomslots_plan_state,
    parse_countfields_plan_state,
    parse_countvalence_plan_state,
    plan_state_from_arrays,
    plan_state_to_atomfields,
    plan_state_to_atomseq,
    plan_state_to_atomslots,
    plan_state_to_countfields,
    plan_state_to_countvalencefields,
    validate_plan_state,
)


def load_tokenizer(tokenizer_path: str | None):
    if not tokenizer_path:
        return None
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
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


def structure_row_to_plan(row: Mapping[str, str]) -> Dict[str, Any]:
    from pymatgen.core import Structure

    structure = Structure.from_str(str(row["cif"]), fmt="cif")
    answer, _ = structure_to_dynamic_answer(structure)
    arrays = parse_dynamic_answer(answer, strict=True)
    return plan_state_from_arrays(arrays, metadata=metadata_from_csv_row(row))


def keep_plan(plan: Mapping[str, Any], composition_filter: str) -> bool:
    if composition_filter == "all":
        return True
    raise ValueError(
        f"Unsupported composition filter {composition_filter!r}. "
        "R5 de novo plan-state training must keep the full MP-20 distribution."
    )


def sample_weight_for_plan(plan: Mapping[str, Any], profile: str) -> float:
    if profile == "uniform":
        return 1.0
    raise ValueError(f"Unsupported sample weight profile {profile!r}")


def plan_format_helpers(plan_format: str):
    if plan_format == "atomseq":
        return {
            "prompt": build_atomseq_plan_prompt(),
            "to_text": plan_state_to_atomseq,
            "parse": parse_atomseq_plan_state,
            "task": "r5_plan_state_atomseq_generation",
            "representation": "r5_plan_state_atomseq",
            "answer_key": "atomseq_plan_state",
            "roundtrip_key": "atomseq_roundtrip_valid",
            "event": "r5_plan_atomseq_builder_progress",
        }
    if plan_format == "atomslots":
        return {
            "prompt": build_atomslots_plan_prompt(),
            "to_text": plan_state_to_atomslots,
            "parse": parse_atomslots_plan_state,
            "task": "r5_plan_state_atomslots_generation",
            "representation": "r5_plan_state_atomslots",
            "answer_key": "atomslots_plan_state",
            "roundtrip_key": "atomslots_roundtrip_valid",
            "event": "r5_plan_atomslots_builder_progress",
        }
    if plan_format == "atomfields":
        return {
            "prompt": build_atomfields_plan_prompt(),
            "to_text": plan_state_to_atomfields,
            "parse": parse_atomfields_plan_state,
            "task": "r5_plan_state_atomfields_generation",
            "representation": "r5_plan_state_atomfields",
            "answer_key": "atomfields_plan_state",
            "roundtrip_key": "atomfields_roundtrip_valid",
            "event": "r5_plan_atomfields_builder_progress",
        }
    if plan_format == "countfields":
        return {
            "prompt": build_countfields_plan_prompt(),
            "to_text": plan_state_to_countfields,
            "parse": parse_countfields_plan_state,
            "task": "r5_plan_state_countfields_generation",
            "representation": "r5_plan_state_countfields",
            "answer_key": "countfields_plan_state",
            "roundtrip_key": "countfields_roundtrip_valid",
            "event": "r5_plan_countfields_builder_progress",
        }
    if plan_format == "countvalence":
        return {
            "prompt": build_countvalence_plan_prompt(),
            "to_text": plan_state_to_countvalencefields,
            "parse": parse_countvalence_plan_state,
            "task": "r5_plan_state_countvalence_generation",
            "representation": "r5_plan_state_countvalence",
            "answer_key": "countvalence_plan_state",
            "roundtrip_key": "countvalence_roundtrip_valid",
            "event": "r5_plan_countvalence_builder_progress",
        }
    raise ValueError(f"Unsupported plan format {plan_format!r}")


def build_split(
    *,
    split: str,
    input_csv: Path,
    output_jsonl: Path,
    tokenizer,
    limit: int | None,
    progress_every: int,
    composition_filter: str,
    plan_format: str,
    sample_weight_profile: str,
) -> Dict[str, Any]:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    helpers = plan_format_helpers(plan_format)
    prompt = helpers["prompt"]
    prompt_text = prompt.rstrip() + "\n"
    roundtrip_key = str(helpers["roundtrip_key"])
    stats: Dict[str, Any] = {
        "split": split,
        "input_csv": str(input_csv),
        "rows_seen": 0,
        "rows_written": 0,
        "rows_filtered": 0,
        "failures": 0,
        roundtrip_key: 0,
        "strict_valid": 0,
        "comp_valid": 0,
        "atom_count_histogram": {},
        "charge_bucket_histogram": {},
        "anion_framework_histogram": {},
        "lattice_system_histogram": {},
        "num_elements_histogram": {},
        "formula_histogram_top": {},
        "sample_weight_histogram": {},
        "answer_tokenizer_lengths": {},
        "prompt_tokenizer_lengths": {},
    }
    formula_counter: Counter[str] = Counter()
    failure_path = output_jsonl.with_suffix(".failure_cases.jsonl")
    with output_jsonl.open("w", encoding="utf-8") as out, failure_path.open("w", encoding="utf-8") as failures:
        for row_idx, row in enumerate(read_rows(input_csv, limit=limit)):
            stats["rows_seen"] += 1
            try:
                plan = structure_row_to_plan(row)
                if not keep_plan(plan, composition_filter):
                    stats["rows_filtered"] += 1
                    continue
                answer = helpers["to_text"](plan)
                roundtrip = helpers["parse"](answer)
                validation = validate_plan_state(roundtrip)
                if not validation.valid:
                    raise ValueError(f"{plan_format} plan roundtrip invalid: {validation.to_dict()}")
                metadata = metadata_from_csv_row(row)
                validator = plan.get("validator") or {}
                composition_reason = str(validator.get("reason") or "unknown")
                comp_valid = bool(validator.get("valid"))
                sample_weight = sample_weight_for_plan(plan, sample_weight_profile)
                record = {
                    "task": helpers["task"],
                    "representation": helpers["representation"],
                    "prompt": prompt,
                    "answer": answer,
                    "text": prompt_text + answer,
                    "prompt_length": token_len(tokenizer, prompt_text),
                    "answer_model_length": token_len(tokenizer, answer),
                    "plan_state": plan,
                    helpers["answer_key"]: answer,
                    "plan_validation": validation.to_dict(),
                    "metadata": metadata,
                    "composition_bucket": plan["charge_bucket"],
                    "composition_reason": composition_reason,
                    "strict_valid": composition_reason == "charge_neutral_pauling_valid",
                    "comp_valid": comp_valid,
                    "num_elements": len(plan.get("elements") or []),
                    "num_elements_bucket": f"k{len(plan.get('elements') or [])}",
                    "loss_profile": "text",
                    "sample_weight": sample_weight,
                    "sample_weight_profile": sample_weight_profile,
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                stats["rows_written"] += 1
                stats[roundtrip_key] += int(validation.valid)
                stats["strict_valid"] += int(record["strict_valid"])
                stats["comp_valid"] += int(comp_valid)
                histogram_add(stats["atom_count_histogram"], plan["N"])
                histogram_add(stats["num_elements_histogram"], len(plan.get("elements") or []))
                histogram_add(stats["charge_bucket_histogram"], plan["charge_bucket"])
                histogram_add(stats["anion_framework_histogram"], plan["anion_framework"])
                histogram_add(stats["lattice_system_histogram"], plan["lattice_system"])
                formula_counter[str(plan["formula"])] += 1
                if record["answer_model_length"] is not None:
                    histogram_add(stats["answer_tokenizer_lengths"], record["answer_model_length"])
                if record["prompt_length"] is not None:
                    histogram_add(stats["prompt_tokenizer_lengths"], record["prompt_length"])
                histogram_add(stats["sample_weight_histogram"], f"{sample_weight:.2f}")
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
                            "event": helpers["event"],
                            "split": split,
                            "rows_seen": stats["rows_seen"],
                            "rows_written": stats["rows_written"],
                            "rows_filtered": stats["rows_filtered"],
                            "failures": stats["failures"],
                        }
                    ),
                    flush=True,
                )
    stats["formula_histogram_top"] = dict(formula_counter.most_common(100))
    stats["roundtrip_key"] = roundtrip_key
    stats[f"{roundtrip_key}_rate"] = stats[roundtrip_key] / max(1, stats["rows_written"])
    stats["strict_valid_rate"] = stats["strict_valid"] / max(1, stats["rows_written"])
    stats["comp_valid_rate"] = stats["comp_valid"] / max(1, stats["rows_written"])
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=PROJECT_ROOT / "reference/crysllmgen/data/mp_20")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data/dlm_sft/mp_20_r5_plan_atomseq")
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument(
        "--composition-filter",
        choices=["all"],
        default="all",
        help="R5 de novo plan-state training keeps the full MP-20 distribution.",
    )
    parser.add_argument(
        "--sample-weight-profile",
        choices=["uniform"],
        default="uniform",
        help="Loss-weight profile written into JSONL. R5 de novo runs use uniform weights only.",
    )
    parser.add_argument(
        "--plan-format",
        choices=["atomseq", "atomslots", "atomfields", "countfields", "countvalence"],
        default="atomseq",
    )
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.tokenizer_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    helpers = plan_format_helpers(args.plan_format)
    splits: Dict[str, Any] = {}
    for split in ("train", "val", "test"):
        splits[split] = build_split(
            split=split,
            input_csv=args.input_dir / f"{split}.csv",
            output_jsonl=args.output_dir / f"{split}.jsonl",
            tokenizer=tokenizer,
            limit=args.limit,
            progress_every=args.progress_every,
            composition_filter=args.composition_filter,
            plan_format=args.plan_format,
            sample_weight_profile=args.sample_weight_profile,
        )
    max_answer = max(
        (max((int(float(key)) for key in item["answer_tokenizer_lengths"]), default=0) for item in splits.values()),
        default=0,
    )
    max_prompt = max(
        (max((int(float(key)) for key in item["prompt_tokenizer_lengths"]), default=0) for item in splits.values()),
        default=0,
    )
    summary = {
        "representation": helpers["representation"],
        "plan_format": args.plan_format,
        "composition_filter": args.composition_filter,
        "sample_weight_profile": args.sample_weight_profile,
        "plan_state_version": PLAN_STATE_VERSION,
        "splits": splits,
        "answer_token_count": max_answer + 8,
        "max_answer_model_length": max_answer,
        "max_prompt_model_length": max_prompt,
        "max_length_recommended": max_prompt + max_answer + 16,
        "special_token_count": 0,
        "prompt": helpers["prompt"],
    }
    write_json(str(args.output_dir / "stats.json"), summary)
    write_json(
        str(args.output_dir / "_SUCCESS"),
        {
            "representation": helpers["representation"],
            "complete": True,
            "plan_format": args.plan_format,
            "composition_filter": args.composition_filter,
            "sample_weight_profile": args.sample_weight_profile,
            "splits": {
                split: {
                    "rows_seen": split_stats["rows_seen"],
                    "rows_written": split_stats["rows_written"],
                    "rows_filtered": split_stats["rows_filtered"],
                    "failures": split_stats["failures"],
                    "roundtrip_valid_rate": split_stats[f"{split_stats['roundtrip_key']}_rate"],
                    "strict_valid_rate": split_stats["strict_valid_rate"],
                    "comp_valid_rate": split_stats["comp_valid_rate"],
                }
                for split, split_stats in splits.items()
            },
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
