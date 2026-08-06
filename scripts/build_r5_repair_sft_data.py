#!/usr/bin/env python3
"""Build R5 corrective-repair SFT rows from labeled failure examples."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from crystal_dlm.r5_repair import make_repair_record, normalize_violation_labels  # noqa: E402
from crystal_dlm.r5_plan_state import parse_plan_state_json  # noqa: E402
from crystal_dlm.fixed_slot import write_json  # noqa: E402


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


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


def plan_from_row(row: Mapping[str, Any]) -> Dict[str, Any] | None:
    plan = row.get("plan_state") or row.get("r5_plan_state")
    if isinstance(plan, dict):
        return dict(plan)
    for key in ("conditioning_prompt", "prompt", "visible_proposal"):
        value = row.get(key)
        if value:
            try:
                return parse_plan_state_json(str(value))
            except Exception:
                pass
    return None


def target_from_row(row: Mapping[str, Any], target_field: str) -> str | None:
    for key in (target_field, "target", "target_block", "corrected_block", "reference_block", "answer"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return None


def visible_from_row(row: Mapping[str, Any]) -> str | None:
    for key in ("visible_proposal", "source_proposal", "failed_proposal", "text", "answer"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return None


def labels_from_row(row: Mapping[str, Any]) -> list[str]:
    labels = row.get("violation_labels") or row.get("labels") or row.get("failure_labels")
    if isinstance(labels, str):
        try:
            decoded = json.loads(labels)
            labels = decoded
        except Exception:
            labels = [item.strip() for item in labels.split(",")]
    if not isinstance(labels, list):
        reason = row.get("reason") or row.get("failure_reason")
        labels = [] if reason is None else [str(reason)]
    return normalize_violation_labels(labels)


def build_split(
    *,
    input_jsonl: Path,
    output_jsonl: Path,
    tokenizer,
    target_field: str,
    limit: int | None,
) -> Dict[str, Any]:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    stats: Dict[str, Any] = {
        "input_jsonl": str(input_jsonl),
        "rows_seen": 0,
        "rows_written": 0,
        "skipped_no_plan": 0,
        "skipped_no_visible": 0,
        "skipped_no_target": 0,
        "label_histogram": {},
        "masked_block_histogram": {},
        "answer_tokenizer_lengths": {},
        "prompt_tokenizer_lengths": {},
    }
    label_counter: Counter[str] = Counter()
    block_counter: Counter[str] = Counter()
    with output_jsonl.open("w", encoding="utf-8") as out:
        for row_idx, row in enumerate(read_jsonl(input_jsonl)):
            if limit is not None and row_idx >= limit:
                break
            stats["rows_seen"] += 1
            plan = plan_from_row(row)
            if plan is None:
                stats["skipped_no_plan"] += 1
                continue
            visible = visible_from_row(row)
            if visible is None:
                stats["skipped_no_visible"] += 1
                continue
            target = target_from_row(row, target_field)
            if target is None:
                stats["skipped_no_target"] += 1
                continue
            labels = labels_from_row(row)
            record = make_repair_record(
                plan_state=plan,
                visible_proposal=visible,
                target=target,
                violation_labels=labels,
                masked_block=row.get("masked_block"),
                metadata=row.get("metadata") or {"source_row_idx": row_idx},
                sample_weight=float(row.get("sample_weight", 1.0) or 1.0),
            )
            record["prompt_length"] = token_len(tokenizer, record["prompt"].rstrip() + "\n")
            record["answer_model_length"] = token_len(tokenizer, record["answer"])
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            stats["rows_written"] += 1
            for label in record["violation_labels"]:
                label_counter[label] += 1
            block_counter[str(record["masked_block"])] += 1
            if record["answer_model_length"] is not None:
                key = str(record["answer_model_length"])
                stats["answer_tokenizer_lengths"][key] = stats["answer_tokenizer_lengths"].get(key, 0) + 1
            if record["prompt_length"] is not None:
                key = str(record["prompt_length"])
                stats["prompt_tokenizer_lengths"][key] = stats["prompt_tokenizer_lengths"].get(key, 0) + 1
    stats["label_histogram"] = dict(label_counter.most_common())
    stats["masked_block_histogram"] = dict(block_counter.most_common())
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data/dlm_sft/mp_20_r5_repair")
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--target-field", default="target")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--mirror-train-to-val-test",
        action="store_true",
        default=True,
        help="Write val/test JSONL mirrors so llada_sft has smoke eval splits. They are not independent holdouts.",
    )
    parser.add_argument("--no-mirror-train-to-val-test", dest="mirror_train_to_val_test", action="store_false")
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.tokenizer_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stats = build_split(
        input_jsonl=args.input_jsonl,
        output_jsonl=args.output_dir / "train.jsonl",
        tokenizer=tokenizer,
        target_field=args.target_field,
        limit=args.limit,
    )
    max_answer = max((int(float(key)) for key in stats["answer_tokenizer_lengths"]), default=0)
    max_prompt = max((int(float(key)) for key in stats["prompt_tokenizer_lengths"]), default=0)
    split_stats = {"train": stats}
    if args.mirror_train_to_val_test:
        train_text = (args.output_dir / "train.jsonl").read_text(encoding="utf-8")
        for split in ("val", "test"):
            (args.output_dir / f"{split}.jsonl").write_text(train_text, encoding="utf-8")
            split_stats[split] = {
                **stats,
                "input_jsonl": str(args.input_jsonl),
                "mirrored_from_train": True,
            }
    summary = {
        "representation": "r5_repair_text",
        "splits": split_stats,
        "source": str(args.input_jsonl),
        "answer_token_count": max_answer + 8,
        "max_answer_model_length": max_answer,
        "max_prompt_model_length": max_prompt,
        "max_length_recommended": max_prompt + max_answer + 16,
        "special_token_count": 0,
    }
    write_json(str(args.output_dir / "stats.json"), summary)
    write_json(str(args.output_dir / "_SUCCESS"), {"representation": "r5_repair_text", "complete": True, "splits": split_stats})
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
