#!/usr/bin/env python3
"""Build H1-G1 robust exact-body SFT data with multiple condition views."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from crystal_dlm.dynamic_crystal import build_special_tokens  # noqa: E402
from crystal_dlm.fixed_slot import FixedSlotConfig, metadata_from_csv_row, write_json  # noqa: E402
from crystal_dlm.h1_formula_only_body import (  # noqa: E402
    H1G1_CONDITION_VIEWS,
    H1G1_ROBUST_BODY_PROMPT_VERSION,
    H1G1_ROBUST_BODY_REPRESENTATION,
    build_condition_view_body_record,
)
from crystal_dlm.r5_plan_state import PLAN_STATE_VERSION, plan_state_from_arrays  # noqa: E402
from scripts.build_r5c_plan_body_sft_data import (  # noqa: E402
    histogram_add,
    max_hist_key,
    read_rows,
    split_names,
    structure_row_to_arrays,
)


def load_tokenizer(tokenizer_path: str | None):
    if not tokenizer_path:
        return None
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    tokenizer.add_special_tokens({"additional_special_tokens": build_special_tokens()})
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def token_len(tokenizer: Any, text: str) -> int | None:
    if tokenizer is None:
        return None
    return int(len(tokenizer(text, add_special_tokens=False)["input_ids"]))


def parse_views(value: str | Sequence[str]) -> list[str]:
    views = split_names(value)
    if not views:
        raise ValueError("at least one condition view is required")
    unknown = [view for view in views if view not in H1G1_CONDITION_VIEWS]
    if unknown:
        raise ValueError(f"unknown condition view(s) {unknown}; expected {H1G1_CONDITION_VIEWS}")
    return views


def view_weight(view: str, weights: Mapping[str, float]) -> float:
    return float(weights.get(view, 1.0))


def build_split(
    *,
    split: str,
    input_csv: Path,
    output_jsonl: Path,
    tokenizer: Any,
    limit: int | None,
    progress_every: int,
    answer_separator: str,
    condition_views: Sequence[str],
    weights: Mapping[str, float],
) -> Dict[str, Any]:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    stats: Dict[str, Any] = {
        "split": split,
        "input_csv": str(input_csv),
        "rows_seen": 0,
        "rows_written": 0,
        "failures": 0,
        "representation": "dynamic_v1",
        "r5_representation": H1G1_ROBUST_BODY_REPRESENTATION,
        "prompt_version": H1G1_ROBUST_BODY_PROMPT_VERSION,
        "condition_views": list(condition_views),
        "condition_view_histogram": {},
        "condition_view_weights": dict(weights),
        "answer_tokenizer_lengths": {},
        "prompt_tokenizer_lengths": {},
        "body_semantic_lengths": {},
        "atom_count_histogram": {},
        "num_elements_histogram": {},
        "formula_histogram": {},
        "anion_framework_histogram": {},
        "charge_bucket_histogram": {},
        "lattice_system_histogram": {},
        "spacegroup_bucket_histogram": {},
        "volume_per_atom_bin_histogram": {},
    }
    failure_path = output_jsonl.with_suffix(".failure_cases.jsonl")
    with output_jsonl.open("w", encoding="utf-8") as out, failure_path.open("w", encoding="utf-8") as failures:
        for row_idx, row in enumerate(read_rows(input_csv, limit=limit)):
            stats["rows_seen"] += 1
            try:
                metadata = metadata_from_csv_row(row)
                arrays = structure_row_to_arrays(row, answer_separator=answer_separator)
                plan = plan_state_from_arrays(arrays, metadata=metadata)
                for view in condition_views:
                    record = build_condition_view_body_record(
                        plan_state=plan,
                        arrays=arrays,
                        metadata=metadata,
                        answer_separator=answer_separator,
                        condition_view=view,
                        sample_weight=view_weight(view, weights),
                    )
                    record["prompt_length"] = token_len(tokenizer, record["prompt"].rstrip() + "\n")
                    record["answer_model_length"] = token_len(tokenizer, record["answer"])
                    record["body_model_length"] = record["answer_model_length"]
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    stats["rows_written"] += 1
                    histogram_add(stats["condition_view_histogram"], view)
                    histogram_add(stats["body_semantic_lengths"], record["answer_semantic_length"])
                    histogram_add(stats["atom_count_histogram"], record["num_atoms"])
                    histogram_add(stats["num_elements_histogram"], len(record["plan_state"]["elements"]))
                    histogram_add(stats["formula_histogram"], record["plan_state"].get("formula", "unknown"))
                    histogram_add(stats["anion_framework_histogram"], record["plan_state"].get("anion_framework", "unknown"))
                    histogram_add(stats["charge_bucket_histogram"], record["plan_state"].get("charge_bucket", "unknown"))
                    histogram_add(stats["lattice_system_histogram"], record["plan_state"].get("lattice_system", "unknown"))
                    histogram_add(stats["spacegroup_bucket_histogram"], record["plan_state"].get("spacegroup_bucket", "sg_unknown"))
                    histogram_add(stats["volume_per_atom_bin_histogram"], record["plan_state"].get("volume_per_atom_bin", "volpa_unknown"))
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
                            "event": "h1g1_robust_exact_dlm_builder_progress",
                            "split": split,
                            "rows_seen": stats["rows_seen"],
                            "rows_written": stats["rows_written"],
                            "failures": stats["failures"],
                        }
                    ),
                    flush=True,
                )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=PROJECT_ROOT / "reference/crysllmgen/data/mp_20")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data/dlm_sft/mp_20_h1g1_robust_exact_dlm")
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--condition-views", default="full-rich,condition-dropout,formula-volume-sg,formula-volume-only")
    parser.add_argument("--full-rich-weight", type=float, default=1.0)
    parser.add_argument("--condition-dropout-weight", type=float, default=0.5)
    parser.add_argument("--formula-volume-sg-weight", type=float, default=0.5)
    parser.add_argument("--formula-volume-only-weight", type=float, default=0.25)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--answer-separator", default="")
    parser.add_argument("--allow-missing-splits", action="store_true")
    args = parser.parse_args()

    condition_views = parse_views(args.condition_views)
    weights = {
        "full-rich": float(args.full_rich_weight),
        "condition-dropout": float(args.condition_dropout_weight),
        "formula-volume-sg": float(args.formula_volume_sg_weight),
        "formula-volume-only": float(args.formula_volume_only_weight),
    }
    tokenizer = load_tokenizer(args.tokenizer_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    splits: Dict[str, Any] = {}
    for split in split_names(args.splits):
        input_csv = args.input_dir / f"{split}.csv"
        if not input_csv.exists():
            if args.allow_missing_splits:
                splits[split] = {"split": split, "input_csv": str(input_csv), "missing": True}
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
            condition_views=condition_views,
            weights=weights,
        )

    vocab_tokens = build_special_tokens(FixedSlotConfig())
    (args.output_dir / "vocab_tokens.txt").write_text("\n".join(vocab_tokens) + "\n", encoding="utf-8")
    written = [item for item in splits.values() if not item.get("missing")]
    max_answer = max((max_hist_key(item["answer_tokenizer_lengths"]) for item in written), default=0)
    max_prompt = max((max_hist_key(item["prompt_tokenizer_lengths"]) for item in written), default=0)
    max_body = max((max_hist_key(item["body_semantic_lengths"]) for item in written), default=0)
    summary = {
        "representation": "dynamic_v1",
        "r5_representation": H1G1_ROBUST_BODY_REPRESENTATION,
        "prompt_version": H1G1_ROBUST_BODY_PROMPT_VERSION,
        "plan_state_version": PLAN_STATE_VERSION,
        "condition_views": condition_views,
        "condition_view_weights": weights,
        "answer_token_count": int(max_body),
        "max_answer_model_length": int(max_answer),
        "max_prompt_model_length": int(max_prompt),
        "max_length_recommended": int(max_prompt + max_answer + 16),
        "splits": splits,
    }
    write_json(str(args.output_dir / "stats.json"), summary)
    write_json(
        str(args.output_dir / "_SUCCESS"),
        {
            "representation": "dynamic_v1",
            "r5_representation": H1G1_ROBUST_BODY_REPRESENTATION,
            "complete": True,
            "splits": {
                split: {
                    "missing": bool(stats.get("missing", False)),
                    "rows_seen": stats.get("rows_seen", 0),
                    "rows_written": stats.get("rows_written", 0),
                    "failures": stats.get("failures", 0),
                }
                for split, stats in splits.items()
            },
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
