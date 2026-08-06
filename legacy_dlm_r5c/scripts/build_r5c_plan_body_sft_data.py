#!/usr/bin/env python3
"""Build R5-C single-pass plan-to-body SFT data from MP-20 CSV splits."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from crystal_dlm.crysllmgen_text import CRYSLLMGEN_TEXT_PROMPT  # noqa: E402
from crystal_dlm.dynamic_crystal import build_special_tokens, parse_dynamic_answer, structure_to_dynamic_answer  # noqa: E402
from crystal_dlm.fixed_slot import FixedSlotConfig, metadata_from_csv_row, write_json  # noqa: E402
from crystal_dlm.r5_plan_body import (  # noqa: E402
    R5C_PLAN_FORMAT,
    R5C_PLAN_BODY_PROMPT_VERSION,
    R5C_PLAN_BODY_REPRESENTATION,
    R5C_PLAN_STYLES,
    R5C_SEMANTIC_PLAN_FORMAT,
    build_body_replay_record,
    build_plan_only_record,
    build_plan_body_record,
    normalize_plan_style,
    representation_for_plan_style,
)
from crystal_dlm.r5_plan_state import PLAN_STATE_VERSION, plan_state_from_arrays  # noqa: E402


def load_tokenizer(tokenizer_path: str | None):
    if not tokenizer_path:
        return None
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    tokenizer.add_special_tokens({"additional_special_tokens": build_special_tokens()})
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def read_rows(csv_path: Path, limit: int | None = None) -> Iterable[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader):
            if limit is not None and idx >= limit:
                break
            yield row


def histogram_add(histogram: Dict[str, int], key: Any, count: int = 1) -> None:
    histogram[str(key)] = int(histogram.get(str(key), 0)) + int(count)


def structure_row_to_arrays(row: Mapping[str, str], *, answer_separator: str = "") -> Dict[str, Any]:
    from pymatgen.core import Structure

    structure = Structure.from_str(str(row["cif"]), fmt="cif")
    answer, _ = structure_to_dynamic_answer(structure, separator=answer_separator)
    return parse_dynamic_answer(answer, strict=True)


def max_hist_key(histogram: Dict[str, int]) -> int:
    if not histogram:
        return 0
    return max(int(float(key)) for key in histogram)


def split_names(value: str | Sequence[str]) -> list[str]:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item).strip() for item in value if str(item).strip()]


def parse_mixture(value: str | Sequence[str]) -> list[str]:
    allowed = {"joint", "plan_only", "body_replay"}
    names = split_names(value)
    unknown = [name for name in names if name not in allowed]
    if unknown:
        raise ValueError(f"Unknown mixture task(s): {','.join(unknown)}")
    if not names:
        raise ValueError("At least one mixture task is required")
    return names


def build_split(
    *,
    split: str,
    input_csv: Path,
    output_jsonl: Path,
    tokenizer,
    limit: int | None,
    progress_every: int,
    answer_separator: str,
    mixture: Sequence[str],
    joint_weight: float,
    plan_only_weight: float,
    body_replay_weight: float,
    plan_style: str,
) -> Dict[str, Any]:
    plan_style = normalize_plan_style(plan_style)
    r5_representation = representation_for_plan_style(plan_style)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    stats: Dict[str, Any] = {
        "split": split,
        "input_csv": str(input_csv),
        "rows_seen": 0,
        "rows_written": 0,
        "failures": 0,
        "representation": "dynamic_v1",
        "r5_representation": r5_representation,
        "prompt_version": R5C_PLAN_BODY_PROMPT_VERSION,
        "plan_format": plan_style,
        "mixture": list(mixture),
        "answer_tokenizer_lengths": {},
        "prompt_tokenizer_lengths": {},
        "plan_tokenizer_lengths": {},
        "body_tokenizer_lengths": {},
        "body_semantic_lengths": {},
        "task_histogram": {},
        "atom_count_histogram": {},
        "num_elements_histogram": {},
        "family_histogram": {},
        "arity_histogram": {},
        "size_histogram": {},
        "charge_bucket_histogram": {},
        "lattice_system_histogram": {},
        "spacegroup_bucket_histogram": {},
        "formula_match_plan": 0,
    }
    failure_path = output_jsonl.with_suffix(".failure_cases.jsonl")
    with output_jsonl.open("w", encoding="utf-8") as out, failure_path.open("w", encoding="utf-8") as failures:
        for row_idx, row in enumerate(read_rows(input_csv, limit=limit)):
            stats["rows_seen"] += 1
            try:
                metadata = metadata_from_csv_row(row)
                arrays = structure_row_to_arrays(row, answer_separator=answer_separator)
                plan = plan_state_from_arrays(arrays, metadata=metadata)
                records = []
                if "joint" in mixture:
                    records.append(
                        build_plan_body_record(
                            plan_state=plan,
                            arrays=arrays,
                            metadata=metadata,
                            tokenizer=tokenizer,
                            prompt=CRYSLLMGEN_TEXT_PROMPT,
                            answer_separator=answer_separator,
                            sample_weight=joint_weight,
                            plan_style=plan_style,
                        )
                    )
                if "plan_only" in mixture:
                    records.append(
                        build_plan_only_record(
                            plan_state=plan,
                            metadata=metadata,
                            tokenizer=tokenizer,
                            prompt=CRYSLLMGEN_TEXT_PROMPT,
                            sample_weight=plan_only_weight,
                            plan_style=plan_style,
                        )
                    )
                if "body_replay" in mixture:
                    records.append(
                        build_body_replay_record(
                            plan_state=plan,
                            arrays=arrays,
                            metadata=metadata,
                            tokenizer=tokenizer,
                            prompt=CRYSLLMGEN_TEXT_PROMPT,
                            answer_separator=answer_separator,
                            sample_weight=body_replay_weight,
                            plan_style=plan_style,
                        )
                    )
                for record in records:
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    stats["rows_written"] += 1
                    stats["formula_match_plan"] += 1
                    histogram_add(stats["task_histogram"], record["task"])
                    histogram_add(stats["body_semantic_lengths"], record["body_semantic_length"])
                    histogram_add(stats["atom_count_histogram"], record["num_atoms"])
                    histogram_add(stats["num_elements_histogram"], len(record["plan_state"]["elements"]))
                    histogram_add(stats["family_histogram"], record["plan_state"].get("family", "unknown"))
                    histogram_add(stats["arity_histogram"], record["plan_state"].get("arity", "unknown"))
                    histogram_add(stats["size_histogram"], record["plan_state"].get("size", "unknown"))
                    histogram_add(stats["charge_bucket_histogram"], record["plan_state"].get("charge_bucket", "unknown"))
                    histogram_add(stats["lattice_system_histogram"], record["plan_state"].get("lattice_system", "unknown"))
                    histogram_add(stats["spacegroup_bucket_histogram"], record["plan_state"].get("spacegroup_bucket", "sg_unknown"))
                    if record["answer_model_length"] is not None:
                        histogram_add(stats["answer_tokenizer_lengths"], record["answer_model_length"])
                    if record["prompt_length"] is not None:
                        histogram_add(stats["prompt_tokenizer_lengths"], record["prompt_length"])
                    if record["plan_model_length"] is not None:
                        histogram_add(stats["plan_tokenizer_lengths"], record["plan_model_length"])
                    if record["body_model_length"] is not None:
                        histogram_add(stats["body_tokenizer_lengths"], record["body_model_length"])
            except Exception as exc:  # noqa: BLE001 - keep bad rows auditable.
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
                            "event": "r5c_plan_body_builder_progress",
                            "split": split,
                            "rows_seen": stats["rows_seen"],
                            "rows_written": stats["rows_written"],
                            "failures": stats["failures"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    stats["formula_match_plan_rate"] = stats["formula_match_plan"] / max(1, stats["rows_written"])
    return stats


def recommended_answer_token_count(*, max_answer_model_length: int, max_body_semantic_length: int) -> int:
    if int(max_answer_model_length) > 0:
        return int(max_answer_model_length) + 8
    return max(512, int(max_body_semantic_length) + 256)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=PROJECT_ROOT / "reference/crysllmgen/data/mp_20")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data/dlm_sft/mp_20_r5c_plan_body")
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument(
        "--mixture",
        default="joint,plan_only,body_replay",
        help="Comma-separated training rows to emit: joint,plan_only,body_replay.",
    )
    parser.add_argument("--joint-weight", type=float, default=1.0)
    parser.add_argument("--plan-only-weight", type=float, default=1.5)
    parser.add_argument("--body-replay-weight", type=float, default=0.25)
    parser.add_argument(
        "--plan-style",
        choices=list(R5C_PLAN_STYLES),
        default=R5C_PLAN_FORMAT,
        help="Text plan style to emit. DN5 uses formula_end_v1; DN4 uses semantic_formula_v1.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--answer-separator", default="")
    parser.add_argument(
        "--allow-missing-splits",
        action="store_true",
        help="Skip split CSVs that are absent instead of failing.",
    )
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.tokenizer_path)
    plan_style = normalize_plan_style(args.plan_style)
    r5_representation = representation_for_plan_style(plan_style)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    requested_splits = split_names(args.splits)
    mixture = parse_mixture(args.mixture)
    splits: Dict[str, Any] = {}
    for split in requested_splits:
        input_csv = args.input_dir / f"{split}.csv"
        if not input_csv.exists():
            if args.allow_missing_splits:
                splits[split] = {
                    "split": split,
                    "input_csv": str(input_csv),
                    "missing": True,
                    "rows_seen": 0,
                    "rows_written": 0,
                    "failures": 0,
                }
                continue
            raise FileNotFoundError(f"Requested split CSV does not exist: {input_csv}")
        splits[split] = build_split(
            split=split,
            input_csv=input_csv,
            output_jsonl=args.output_dir / f"{split}.jsonl",
            tokenizer=tokenizer,
            limit=args.limit,
            progress_every=args.progress_every,
            answer_separator=args.answer_separator,
            mixture=mixture,
            joint_weight=args.joint_weight,
            plan_only_weight=args.plan_only_weight,
            body_replay_weight=args.body_replay_weight,
            plan_style=plan_style,
        )

    vocab_tokens = build_special_tokens(FixedSlotConfig())
    (args.output_dir / "vocab_tokens.txt").write_text("\n".join(vocab_tokens) + "\n", encoding="utf-8")
    written_splits = [item for item in splits.values() if not item.get("missing")]
    empty_written = [
        item["split"]
        for item in written_splits
        if int(item.get("rows_seen", 0)) > 0 and int(item.get("rows_written", 0)) == 0
    ]
    if empty_written:
        raise RuntimeError(
            "No records were written for split(s) "
            + ",".join(empty_written)
            + "; inspect *.failure_cases.jsonl for dependency or data errors."
        )
    max_answer = max((max_hist_key(item["answer_tokenizer_lengths"]) for item in written_splits), default=0)
    max_prompt = max((max_hist_key(item["prompt_tokenizer_lengths"]) for item in written_splits), default=0)
    max_body = max((max_hist_key(item["body_semantic_lengths"]) for item in written_splits), default=0)
    answer_token_count = recommended_answer_token_count(
        max_answer_model_length=max_answer,
        max_body_semantic_length=max_body,
    )
    summary = {
        "representation": "dynamic_v1",
        "r5_representation": r5_representation,
        "prompt_version": R5C_PLAN_BODY_PROMPT_VERSION,
        "prompt": CRYSLLMGEN_TEXT_PROMPT,
        "plan_state_version": PLAN_STATE_VERSION,
        "plan_format": plan_style,
        "answer_layout": "plan block followed by body block in one answer trajectory",
        "mixture": mixture,
        "sample_weights": {
            "joint": float(args.joint_weight),
            "plan_only": float(args.plan_only_weight),
            "body_replay": float(args.body_replay_weight),
        },
        "splits": splits,
        "max_answer_token_count": answer_token_count,
        "answer_token_count": answer_token_count,
        "max_answer_model_length": max_answer,
        "max_prompt_model_length": max_prompt,
        "max_body_semantic_length": max_body,
        "max_length_recommended": max_prompt + answer_token_count + 16,
        "special_token_count": len(vocab_tokens),
        "loss_profile": "text",
        "answer_separator": args.answer_separator,
    }
    write_json(str(args.output_dir / "stats.json"), summary)
    write_json(
        str(args.output_dir / "_SUCCESS"),
        {
            "representation": "dynamic_v1",
            "r5_representation": r5_representation,
            "prompt_version": R5C_PLAN_BODY_PROMPT_VERSION,
            "plan_format": plan_style,
            "mixture": mixture,
            "complete": True,
            "splits": {
                split: {
                    "missing": bool(split_stats.get("missing", False)),
                    "rows_seen": split_stats["rows_seen"],
                    "rows_written": split_stats["rows_written"],
                    "failures": split_stats["failures"],
                    "formula_match_plan_rate": split_stats.get("formula_match_plan_rate"),
                }
                for split, split_stats in splits.items()
            },
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
