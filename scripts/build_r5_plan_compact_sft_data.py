#!/usr/bin/env python3
"""Build R5 compact plan-state SFT data from MP-20 CSV splits."""

from __future__ import annotations

import argparse
import csv
import json
import random
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
    build_compact_plan_prompt,
    build_compact_plan_repair_prompt,
    normalize_compact_plan_for_repair_target,
    parse_compact_plan_state,
    plan_state_from_arrays,
    plan_state_to_compact,
    validate_plan_state,
)
from scripts.build_r5_plan_state_sft_data import plan_sample_weight  # noqa: E402


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


REPAIR_EXTRA_ELEMENT_POOL = [
    "Li",
    "O",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "K",
    "Ca",
    "Ti",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Sr",
    "Y",
    "Zr",
    "Ba",
    "La",
]


def _compact_line(elements: list[str], counts: list[int], *, n_value: int, plan: Mapping[str, Any]) -> str:
    pairs = ",".join(f"{symbol}:{int(count)}" for symbol, count in zip(elements, counts))
    return (
        f"N={int(n_value)};"
        f"E={pairs};"
        f"LS={plan.get('lattice_system', 'triclinic')};"
        f"SG={plan.get('spacegroup_bucket', 'sg_001_002')};"
        f"VP={plan.get('volume_per_atom_bin', 'volpa_unknown')}"
    )


def _normalized_repair_target(visible_plan: str, fallback_answer: str) -> str:
    try:
        target = normalize_compact_plan_for_repair_target(visible_plan)
        validation = validate_plan_state(parse_compact_plan_state(target))
        if validation.valid:
            return target
    except Exception:
        pass
    return fallback_answer


def compact_plan_repair_corruptions(
    plan: Mapping[str, Any],
    compact_answer: str,
    *,
    rng: random.Random,
    limit: int,
) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    elements = [str(symbol) for symbol in (plan.get("elements") or [])]
    counts = [int(value) for value in (plan.get("counts") or [])]
    n_value = int(plan.get("N") or sum(counts))
    mandatory: list[dict[str, Any]] = []
    optional: list[dict[str, Any]] = []

    over_elements = list(elements)
    over_counts = list(counts)
    while sum(over_counts) <= 20:
        symbol = rng.choice(REPAIR_EXTRA_ELEMENT_POOL)
        count = rng.randint(1, 4)
        over_elements.append(symbol)
        over_counts.append(count)
    visible_overrun = _compact_line(over_elements, over_counts, n_value=min(20, max(1, n_value)), plan=plan)
    mandatory.append(
        {
            "visible_plan": visible_overrun,
            "target_answer": compact_answer,
            "labels": ["atom_count_out_of_range", "generated_N_count_mismatch"],
            "corruption": "atom_count_overrun",
            "target_policy": "prune_to_training_plan",
        }
    )

    mismatch_n = max(1, n_value - 1) if n_value > 1 else 2
    visible_mismatch = _compact_line(elements, counts, n_value=mismatch_n, plan=plan)
    mandatory.append(
        {
            "visible_plan": visible_mismatch,
            "target_answer": _normalized_repair_target(visible_mismatch, compact_answer),
            "labels": ["generated_N_count_mismatch"],
            "corruption": "generated_n_mismatch",
            "target_policy": "normalize_visible_plan",
        }
    )

    if elements:
        single_count = min(20, max(1, n_value))
        optional.append(
            {
                "visible_plan": _compact_line([elements[0]], [single_count], n_value=single_count, plan=plan),
                "target_answer": compact_answer,
                "labels": ["single_element"],
                "corruption": "single_element_collapse",
                "target_policy": "recover_training_plan",
            }
        )

    if len(elements) >= 2:
        charge_elements = [elements[0], elements[-1]]
        charge_counts = [max(1, min(10, counts[0] + 1)), max(1, min(10, counts[-1] + 3))]
        optional.append(
            {
                "visible_plan": _compact_line(
                    charge_elements,
                    charge_counts,
                    n_value=sum(charge_counts),
                    plan=plan,
                ),
                "target_answer": compact_answer,
                "labels": ["charge_fail"],
                "corruption": "composition_charge_shift",
                "target_policy": "recover_training_plan",
            }
        )

    visible_bucket = compact_answer.replace("SG=sg_", "SG=sg_000", 1).replace(";VP=", ";volpa=", 1)
    optional.append(
        {
            "visible_plan": visible_bucket,
            "target_answer": _normalized_repair_target(visible_bucket, compact_answer),
            "labels": ["syntax_drift", "spacegroup_bucket_malformed"],
            "corruption": "bucket_syntax_drift",
            "target_policy": "normalize_visible_plan",
        }
    )
    rng.shuffle(optional)
    return (mandatory + optional)[:limit]


def build_split(
    *,
    split: str,
    input_csv: Path,
    output_jsonl: Path,
    tokenizer,
    limit: int | None,
    progress_every: int,
    weight_profile: str,
    repair_augmentations_per_row: int,
    repair_sample_weight: float,
    repair_seed: int,
) -> Dict[str, Any]:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    prompt = build_compact_plan_prompt()
    prompt_text = prompt.rstrip() + "\n"
    stats: Dict[str, Any] = {
        "split": split,
        "input_csv": str(input_csv),
        "rows_seen": 0,
        "rows_written": 0,
        "base_rows_written": 0,
        "repair_rows_written": 0,
        "failures": 0,
        "compact_roundtrip_valid": 0,
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
                answer = plan_state_to_compact(plan)
                roundtrip = parse_compact_plan_state(answer)
                validation = validate_plan_state(roundtrip)
                if not validation.valid:
                    raise ValueError(f"compact plan roundtrip invalid: {validation.to_dict()}")
                metadata = metadata_from_csv_row(row)
                validator = plan.get("validator") or {}
                composition_reason = str(validator.get("reason") or "unknown")
                comp_valid = bool(validator.get("valid"))
                record = {
                    "task": "r5_plan_state_compact_generation",
                    "representation": "r5_plan_state_compact",
                    "prompt": prompt,
                    "answer": answer,
                    "text": prompt_text + answer,
                    "prompt_length": token_len(tokenizer, prompt_text),
                    "answer_model_length": token_len(tokenizer, answer),
                    "plan_state": plan,
                    "compact_plan_state": answer,
                    "plan_validation": validation.to_dict(),
                    "metadata": metadata,
                    "composition_bucket": plan["charge_bucket"],
                    "composition_reason": composition_reason,
                    "strict_valid": composition_reason == "charge_neutral_pauling_valid",
                    "comp_valid": comp_valid,
                    "num_elements": len(plan.get("elements") or []),
                    "num_elements_bucket": f"k{len(plan.get('elements') or [])}",
                    "loss_profile": "text",
                    "sample_weight": plan_sample_weight(plan, weight_profile=weight_profile),
                    "sample_weight_profile": weight_profile,
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                stats["rows_written"] += 1
                stats["base_rows_written"] += 1
                stats["compact_roundtrip_valid"] += int(validation.valid)
                stats["strict_valid"] += int(record["strict_valid"])
                stats["comp_valid"] += int(comp_valid)
                histogram_add(stats["atom_count_histogram"], plan["N"])
                histogram_add(stats["num_elements_histogram"], len(plan.get("elements") or []))
                histogram_add(stats["charge_bucket_histogram"], plan["charge_bucket"])
                histogram_add(stats["anion_framework_histogram"], plan["anion_framework"])
                histogram_add(stats["lattice_system_histogram"], plan["lattice_system"])
                formula_counter[str(plan["formula"])] += 1
                histogram_add(stats["sample_weight_histogram"], record["sample_weight"])
                if record["answer_model_length"] is not None:
                    histogram_add(stats["answer_tokenizer_lengths"], record["answer_model_length"])
                if record["prompt_length"] is not None:
                    histogram_add(stats["prompt_tokenizer_lengths"], record["prompt_length"])
                if split == "train" and repair_augmentations_per_row > 0:
                    row_rng = random.Random(int(repair_seed) + row_idx * 1009)
                    for repair_idx, corruption in enumerate(
                        compact_plan_repair_corruptions(
                            plan,
                            answer,
                            rng=row_rng,
                            limit=int(repair_augmentations_per_row),
                        )
                    ):
                        target_answer = str(corruption.get("target_answer") or answer)
                        target_plan = parse_compact_plan_state(target_answer)
                        target_validation = validate_plan_state(target_plan)
                        repair_prompt = build_compact_plan_repair_prompt(
                            visible_plan=str(corruption["visible_plan"]),
                            violation_labels=list(corruption["labels"]),
                        )
                        repair_prompt_text = repair_prompt.rstrip() + "\n"
                        repair_record = {
                            "task": "r5_plan_state_compact_repair",
                            "representation": "r5_plan_state_compact",
                            "prompt": repair_prompt,
                            "answer": target_answer,
                            "text": repair_prompt_text + target_answer,
                            "prompt_length": token_len(tokenizer, repair_prompt_text),
                            "answer_model_length": token_len(tokenizer, target_answer),
                            "plan_state": target_plan,
                            "compact_plan_state": target_answer,
                            "visible_compact_plan": str(corruption["visible_plan"]),
                            "violation_labels": list(corruption["labels"]),
                            "repair_corruption": str(corruption["corruption"]),
                            "repair_target_policy": str(corruption.get("target_policy") or "recover_training_plan"),
                            "repair_source_row_idx": row_idx,
                            "repair_idx": repair_idx,
                            "plan_validation": target_validation.to_dict(),
                            "metadata": metadata,
                            "composition_bucket": target_plan["charge_bucket"],
                            "composition_reason": str((target_plan.get("validator") or {}).get("reason") or "unknown"),
                            "strict_valid": str((target_plan.get("validator") or {}).get("reason") or "")
                            == "charge_neutral_pauling_valid",
                            "comp_valid": (target_plan.get("validator") or {}).get("valid") is True,
                            "num_elements": len(target_plan.get("elements") or []),
                            "num_elements_bucket": f"k{len(target_plan.get('elements') or [])}",
                            "loss_profile": "text",
                            "sample_weight": float(repair_sample_weight)
                            * plan_sample_weight(plan, weight_profile=weight_profile),
                            "sample_weight_profile": f"{weight_profile}+repair",
                        }
                        out.write(json.dumps(repair_record, ensure_ascii=False) + "\n")
                        stats["rows_written"] += 1
                        stats["repair_rows_written"] += 1
                        histogram_add(stats["sample_weight_histogram"], repair_record["sample_weight"])
                        if repair_record["answer_model_length"] is not None:
                            histogram_add(stats["answer_tokenizer_lengths"], repair_record["answer_model_length"])
                        if repair_record["prompt_length"] is not None:
                            histogram_add(stats["prompt_tokenizer_lengths"], repair_record["prompt_length"])
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
                            "event": "r5_plan_compact_builder_progress",
                            "split": split,
                            "rows_seen": stats["rows_seen"],
                            "rows_written": stats["rows_written"],
                            "failures": stats["failures"],
                        }
                    ),
                    flush=True,
                )
    stats["formula_histogram_top"] = dict(formula_counter.most_common(100))
    stats["compact_roundtrip_valid_rate"] = stats["compact_roundtrip_valid"] / max(1, stats["base_rows_written"])
    stats["strict_valid_rate"] = stats["strict_valid"] / max(1, stats["base_rows_written"])
    stats["comp_valid_rate"] = stats["comp_valid"] / max(1, stats["base_rows_written"])
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=PROJECT_ROOT / "reference/crysllmgen/data/mp_20")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data/dlm_sft/mp_20_r5_plan_compact")
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--weight-profile", choices=["default", "composition_only", "mp20_num_elements"], default="default")
    parser.add_argument("--repair-augmentations-per-row", type=int, default=0)
    parser.add_argument("--repair-sample-weight", type=float, default=2.0)
    parser.add_argument("--repair-seed", type=int, default=20260530)
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
            weight_profile=args.weight_profile,
            repair_augmentations_per_row=args.repair_augmentations_per_row,
            repair_sample_weight=args.repair_sample_weight,
            repair_seed=args.repair_seed,
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
        "representation": "r5_plan_state_compact",
        "sample_weight_profile": args.weight_profile,
        "repair_augmentations_per_row": args.repair_augmentations_per_row,
        "repair_sample_weight": args.repair_sample_weight,
        "repair_seed": args.repair_seed,
        "plan_state_version": PLAN_STATE_VERSION,
        "splits": splits,
        "answer_token_count": max_answer + 8,
        "max_answer_model_length": max_answer,
        "max_prompt_model_length": max_prompt,
        "max_length_recommended": max_prompt + max_answer + 16,
        "special_token_count": 0,
        "prompt": build_compact_plan_prompt(),
    }
    write_json(str(args.output_dir / "stats.json"), summary)
    write_json(
        str(args.output_dir / "_SUCCESS"),
        {
            "representation": "r5_plan_state_compact",
            "complete": True,
            "splits": {
                split: {
                    "rows_seen": split_stats["rows_seen"],
                    "rows_written": split_stats["rows_written"],
                    "base_rows_written": split_stats["base_rows_written"],
                    "repair_rows_written": split_stats["repair_rows_written"],
                    "failures": split_stats["failures"],
                    "compact_roundtrip_valid_rate": split_stats["compact_roundtrip_valid_rate"],
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
