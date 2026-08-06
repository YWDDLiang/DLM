#!/usr/bin/env python3
"""Build R5 plan-state JSON SFT data from MP-20 CSV splits."""

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
    build_plan_prompt,
    plan_state_from_arrays,
    plan_state_to_json,
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


NUM_ELEMENT_WEIGHT_PROFILE = {
    1: 0.05,
    2: 2.20,
    3: 3.00,
    4: 1.60,
    5: 0.70,
    6: 0.25,
    7: 0.10,
}


def num_element_weight(plan: Mapping[str, Any], profile: str) -> float:
    if profile in {"", "default", "composition_only"}:
        return 1.0
    if profile != "mp20_num_elements":
        raise ValueError(f"Unsupported plan weight profile {profile!r}")
    num_elements = len(plan.get("elements") or [])
    return float(NUM_ELEMENT_WEIGHT_PROFILE.get(int(num_elements), 0.05))


def plan_sample_weight(plan: Mapping[str, Any], *, weight_profile: str = "default") -> float:
    validator = plan.get("validator") or {}
    reason = str(validator.get("reason") or "")
    if reason == "charge_neutral_pauling_valid":
        composition_weight = 1.4
    elif reason == "all_metal_shortcut":
        composition_weight = 0.55
    elif reason == "single_element_shortcut":
        composition_weight = 0.05
    elif reason in {"charge_neutrality_fail", "pauling_fail_or_ratio_rejected", "oxidation_state_missing"}:
        composition_weight = 0.35
    else:
        composition_weight = 1.0
    return composition_weight * num_element_weight(plan, weight_profile)


def structure_row_to_plan(row: Mapping[str, str]) -> Dict[str, Any]:
    from pymatgen.core import Structure

    structure = Structure.from_str(str(row["cif"]), fmt="cif")
    answer, _ = structure_to_dynamic_answer(structure)
    arrays = parse_dynamic_answer(answer, strict=True)
    return plan_state_from_arrays(arrays, metadata=metadata_from_csv_row(row))


def build_split(
    *,
    split: str,
    input_csv: Path,
    output_jsonl: Path,
    tokenizer,
    limit: int | None,
    progress_every: int,
) -> Dict[str, Any]:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    prompt = build_plan_prompt()
    prompt_text = prompt.rstrip() + "\n"
    stats: Dict[str, Any] = {
        "split": split,
        "input_csv": str(input_csv),
        "rows_seen": 0,
        "rows_written": 0,
        "failures": 0,
        "valid_N": 0,
        "valid_formula": 0,
        "valid_plan": 0,
        "atom_count_histogram": {},
        "charge_bucket_histogram": {},
        "anion_framework_histogram": {},
        "lattice_system_histogram": {},
        "formula_histogram_top": {},
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
                validation = validate_plan_state(plan)
                answer = plan_state_to_json(plan)
                metadata = metadata_from_csv_row(row)
                validator = plan.get("validator") or {}
                composition_reason = str(validator.get("reason") or "unknown")
                comp_valid = bool(validator.get("valid"))
                record = {
                    "task": "r5_plan_state_generation",
                    "representation": "r5_plan_state",
                    "prompt": prompt,
                    "answer": answer,
                    "text": prompt_text + answer,
                    "prompt_length": token_len(tokenizer, prompt_text),
                    "answer_model_length": token_len(tokenizer, answer),
                    "plan_state": plan,
                    "plan_validation": validation.to_dict(),
                    "metadata": metadata,
                    "composition_bucket": plan["charge_bucket"],
                    "composition_reason": composition_reason,
                    "strict_valid": composition_reason == "charge_neutral_pauling_valid",
                    "comp_valid": comp_valid,
                    "loss_profile": "text",
                    "num_elements": len(plan.get("elements") or []),
                    "num_elements_bucket": f"k{len(plan.get('elements') or [])}",
                    "sample_weight": plan_sample_weight(plan),
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                stats["rows_written"] += 1
                stats["valid_N"] += int(validation.valid_N)
                stats["valid_formula"] += int(validation.valid_formula)
                stats["valid_plan"] += int(validation.valid)
                histogram_add(stats["atom_count_histogram"], plan["N"])
                histogram_add(stats["charge_bucket_histogram"], plan["charge_bucket"])
                histogram_add(stats["anion_framework_histogram"], plan["anion_framework"])
                histogram_add(stats["lattice_system_histogram"], plan["lattice_system"])
                formula_counter[str(plan["formula"])] += 1
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
                            "event": "r5_plan_state_builder_progress",
                            "split": split,
                            "rows_seen": stats["rows_seen"],
                            "rows_written": stats["rows_written"],
                            "failures": stats["failures"],
                        }
                    ),
                    flush=True,
                )
    stats["formula_histogram_top"] = dict(formula_counter.most_common(100))
    stats["valid_N_rate"] = stats["valid_N"] / max(1, stats["rows_written"])
    stats["valid_formula_rate"] = stats["valid_formula"] / max(1, stats["rows_written"])
    stats["valid_plan_rate"] = stats["valid_plan"] / max(1, stats["rows_written"])
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=PROJECT_ROOT / "reference/crysllmgen/data/mp_20")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data/dlm_sft/mp_20_r5_plan_state")
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=1000)
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
        "representation": "r5_plan_state",
        "plan_state_version": PLAN_STATE_VERSION,
        "splits": splits,
        "answer_token_count": max_answer + 8,
        "max_answer_model_length": max_answer,
        "max_prompt_model_length": max_prompt,
        "max_length_recommended": max_prompt + max_answer + 16,
        "special_token_count": 0,
        "prompt": build_plan_prompt(),
    }
    write_json(str(args.output_dir / "stats.json"), summary)
    write_json(
        str(args.output_dir / "_SUCCESS"),
        {
            "representation": "r5_plan_state",
            "complete": True,
            "splits": {
                split: {
                    "rows_seen": split_stats["rows_seen"],
                    "rows_written": split_stats["rows_written"],
                    "failures": split_stats["failures"],
                    "valid_formula_rate": split_stats["valid_formula_rate"],
                    "valid_N_rate": split_stats["valid_N_rate"],
                }
                for split, split_stats in splits.items()
            },
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
