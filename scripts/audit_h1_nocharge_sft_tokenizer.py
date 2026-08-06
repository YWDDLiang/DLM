#!/usr/bin/env python3
"""Audit the frozen no-charge C0/C1 ledgers with the Planner tokenizer."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from crystal_dlm.h1_nocharge_ion_aux import (  # noqa: E402
    H1_NOCHARGE_ION_AUX_SCHEMA,
    canonical_json_sha256,
)


SCHEMA = "h1_nocharge_ion_aux_tokenizer_audit_v1"
EXPECTED = {"train": 3200, "val": 640}
NONAUX_TASKS = {
    "direct_nocharge_plan",
    "conditional_mp20_anchor",
    "p0_kl_anchor",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path, expected: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path} contains a non-object row")
                rows.append(value)
    if len(rows) != expected:
        raise ValueError(f"{path} has {len(rows)} records, expected {expected}")
    if [int(row.get("ledger_ordinal", -1)) for row in rows] != list(range(expected)):
        raise ValueError(f"{path} ledger ordinals are not exact and ordered")
    return rows


def formatted_prompt(tokenizer, row: Mapping[str, Any]) -> str:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError(f"record {row.get('record_id')} has no valid messages")
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        return str(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )
    return f"System: {messages[0]['content']}\n\nUser: {messages[1]['content']}\n\nAssistant:"


def answer_encoding(
    tokenizer,
    answer: str,
    weighted_spans: Sequence[Mapping[str, Any]],
) -> tuple[list[int], list[float]]:
    clean = str(answer).strip()
    encoded = tokenizer(
        clean + (tokenizer.eos_token or ""),
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    if "offset_mapping" not in encoded:
        raise RuntimeError("frozen no-charge SFT requires a fast tokenizer")
    token_ids = [int(value) for value in encoded["input_ids"]]
    weights = [1.0] * len(token_ids)
    for span in weighted_spans:
        start = int(span["start"])
        end = int(span["end"])
        weight = float(span["weight"])
        if not (0 <= start < end <= len(clean)):
            raise ValueError(f"invalid weighted span in answer {clean!r}: {span}")
        covered = False
        for idx, (token_start, token_end) in enumerate(encoded["offset_mapping"]):
            if int(token_end) <= start or int(token_start) >= end:
                continue
            weights[idx] = max(weights[idx], weight)
            covered = True
        if not covered:
            raise ValueError(f"weighted span covers no answer token: {span}")
    return token_ids, weights


def audit_record(tokenizer, row: Mapping[str, Any], *, max_length: int) -> dict[str, Any]:
    if row.get("schema") != H1_NOCHARGE_ION_AUX_SCHEMA:
        raise ValueError(f"record {row.get('record_id')} has the wrong schema")
    prompt = formatted_prompt(tokenizer, row)
    prompt_ids = [
        int(value)
        for value in tokenizer(prompt, add_special_tokens=False)["input_ids"]
    ]
    answer = str(row.get("answer") or "").strip()
    if not answer:
        raise ValueError(f"record {row.get('record_id')} has an empty answer")
    answer_ids, weights = answer_encoding(
        tokenizer,
        answer,
        row.get("weighted_answer_spans") or [],
    )
    if len(answer_ids) >= int(max_length):
        raise ValueError(
            f"record {row.get('record_id')} answer has {len(answer_ids)} tokens for max_length={max_length}"
        )
    task = str(row.get("task"))
    formula = str(row.get("formula") or "")
    lower_answer = answer.lower()
    if task == "direct_nocharge_plan" and "charge:" in lower_answer:
        raise ValueError(f"record {row.get('record_id')} leaks generated charge")
    if task in {"conditional_mp20_anchor", "p0_kl_anchor"}:
        if formula and formula in answer:
            raise ValueError(f"record {row.get('record_id')} targets an input-only formula")
        if "charge:" in lower_answer:
            raise ValueError(f"record {row.get('record_id')} replays generated charge")
    return {
        "record_id": str(row["record_id"]),
        "task": task,
        "source_row_idx": int(row["source_row_idx"]),
        "infill_cursor": row.get("infill_cursor"),
        "prompt_token_count": len(prompt_ids),
        "answer_token_count": len(answer_ids),
        "total_before_prompt_left_truncation": len(prompt_ids) + len(answer_ids),
        "prompt_left_truncation_count": max(0, len(prompt_ids) + len(answer_ids) - int(max_length)),
        "answer_ids_sha256": canonical_json_sha256(answer_ids),
        "prompt_ids_sha256": canonical_json_sha256(prompt_ids),
        "weight_vector_sha256": canonical_json_sha256(weights),
        "weight_policy": [
            {
                "weight": float(span["weight"]),
                "label": str(span.get("label") or ""),
            }
            for span in (row.get("weighted_answer_spans") or [])
        ],
        "loss_mode": str(row.get("loss_mode") or "sft"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "pass",
        "tokenizer_path": args.tokenizer_path,
        "max_length": int(args.max_length),
        "splits": {},
        "input_sha256": {},
    }
    for split, expected in EXPECTED.items():
        raw = {
            arm: read_jsonl(args.data_root / arm / f"{split}.jsonl", expected)
            for arm in ("c0", "c1")
        }
        audited = {
            arm: [
                audit_record(tokenizer, row, max_length=int(args.max_length))
                for row in raw[arm]
            ]
            for arm in ("c0", "c1")
        }
        pair_failures: list[str] = []
        nonaux_token_identity = 0
        weighted_policy_identity = 0
        for c0, c1 in zip(audited["c0"], audited["c1"]):
            for field in ("record_id", "task", "source_row_idx", "infill_cursor", "loss_mode"):
                if c0[field] != c1[field]:
                    pair_failures.append(f"{field}:{c0['record_id']}")
            if c0["weight_policy"] != c1["weight_policy"]:
                pair_failures.append(f"weight_policy:{c0['record_id']}")
            else:
                weighted_policy_identity += 1
            if c0["task"] in NONAUX_TASKS:
                if any(
                    c0[field] != c1[field]
                    for field in (
                        "prompt_ids_sha256",
                        "answer_ids_sha256",
                        "weight_vector_sha256",
                    )
                ):
                    pair_failures.append(f"nonaux_token_identity:{c0['record_id']}")
                else:
                    nonaux_token_identity += 1
        if pair_failures:
            raise RuntimeError(f"{split} tokenizer pair audit failed: {pair_failures[:16]}")
        all_rows = audited["c0"] + audited["c1"]
        report["splits"][split] = {
            "rows_per_arm": expected,
            "pair_failures": pair_failures,
            "weighted_policy_identity_count": weighted_policy_identity,
            "nonaux_token_identity_count": nonaux_token_identity,
            "max_prompt_tokens": max(row["prompt_token_count"] for row in all_rows),
            "max_answer_tokens": max(row["answer_token_count"] for row in all_rows),
            "max_total_before_prompt_left_truncation": max(
                row["total_before_prompt_left_truncation"] for row in all_rows
            ),
            "max_prompt_left_truncation_count": max(
                row["prompt_left_truncation_count"] for row in all_rows
            ),
        }
        for arm in ("c0", "c1"):
            path = args.data_root / arm / f"{split}.jsonl"
            report["input_sha256"][f"{arm}_{split}"] = sha256_file(path)
    report["tokenizer"] = {
        "vocab_size": len(tokenizer),
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "padding_side": tokenizer.padding_side,
    }
    report["report_contract_sha256"] = canonical_json_sha256(report)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": report["status"], "splits": report["splits"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
