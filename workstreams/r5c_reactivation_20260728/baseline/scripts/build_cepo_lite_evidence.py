#!/usr/bin/env python3
"""Attach CEPO-lite token credit weights to scored TraceRL rollouts."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from crystal_dlm.fixed_slot import tokenize_answer_text, write_json


NEGATIVE_REASONS = {
    "charge_neutrality_fail",
    "pauling_fail_or_ratio_rejected",
    "oxidation_state_missing",
}

COMP_VALID_REASONS = {
    "charge_neutral_pauling_valid",
    "all_metal_shortcut",
    "single_element_shortcut",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def reason(row: dict[str, Any]) -> str:
    diagnostics = row.get("diagnostics") or {}
    return str(diagnostics.get("smact_reason") or diagnostics.get("reason") or row.get("reason") or "unknown")


def is_positive(row: dict[str, Any], *, include_all_comp_valid: bool = False) -> bool:
    row_reason = reason(row)
    if include_all_comp_valid and row_reason in COMP_VALID_REASONS:
        return True
    return row_reason == "charge_neutral_pauling_valid" or bool(row.get("seal_buffer"))


def is_negative(
    row: dict[str, Any],
    *,
    include_all_metal: bool = False,
    include_single_element: bool = False,
) -> bool:
    diagnostics = row.get("diagnostics") or {}
    row_reason = reason(row)
    if include_all_metal and row_reason == "all_metal_shortcut":
        return True
    if include_single_element and row_reason == "single_element_shortcut":
        return True
    return row_reason in NEGATIVE_REASONS or bool(diagnostics.get("has_pbc_equivalent_duplicate"))


def row_score(row: dict[str, Any]) -> float:
    return float(row.get("reward", 0.0))


def group_key(row: dict[str, Any]) -> str:
    if "prompt_id" in row:
        return f"prompt_id:{row['prompt_id']}"
    return str(row.get("prompt") or "default_prompt")


def atom_count_from_tokens(tokens: list[str]) -> int:
    if not tokens or not tokens[0].startswith("<N_"):
        return 0
    try:
        return int(tokens[0][3:6])
    except Exception:
        return 0


def credit_positions(tokens: list[str]) -> set[int]:
    n_atom = max(0, min(20, atom_count_from_tokens(tokens)))
    positions = {0}
    positions.update(8 + 5 * idx for idx in range(n_atom))
    if n_atom < 20:
        positions.add(8 + 5 * n_atom)
    return {pos for pos in positions if 0 <= pos < len(tokens)}


def role(
    row: dict[str, Any],
    pos: dict[str, Any] | None,
    neg: dict[str, Any] | None,
    *,
    include_all_comp_valid_positive: bool = False,
    include_all_metal_negative: bool = False,
    include_single_element_negative: bool = False,
) -> str:
    sample_idx = row.get("sample_idx")
    if pos is not None and sample_idx == pos.get("sample_idx"):
        return "positive"
    if neg is not None and sample_idx == neg.get("sample_idx"):
        return "negative"
    if is_positive(row, include_all_comp_valid=include_all_comp_valid_positive):
        return "positive_like"
    if is_negative(
        row,
        include_all_metal=include_all_metal_negative,
        include_single_element=include_single_element_negative,
    ):
        return "negative_like"
    return "other"


def token_weights(
    row: dict[str, Any],
    pos: dict[str, Any],
    neg: dict[str, Any],
    *,
    lambda_weight: float,
    clip_eps: float,
    zero_non_credit: bool = False,
) -> list[float]:
    tokens = tokenize_answer_text(str(row.get("response") or row.get("answer") or row.get("text") or ""))
    pos_tokens = tokenize_answer_text(str(pos.get("response") or pos.get("answer") or pos.get("text") or ""))
    neg_tokens = tokenize_answer_text(str(neg.get("response") or neg.get("answer") or neg.get("text") or ""))
    weights = [0.0 if zero_non_credit else 1.0] * len(tokens)
    lo = 1.0 - float(clip_eps)
    hi = 1.0 + float(clip_eps)
    for pos_idx in credit_positions(tokens):
        weights[pos_idx] = 1.0
        if pos_idx >= len(pos_tokens) or pos_idx >= len(neg_tokens):
            continue
        current = tokens[pos_idx]
        positive = pos_tokens[pos_idx]
        negative = neg_tokens[pos_idx]
        if positive == negative:
            continue
        if current == positive:
            weights[pos_idx] = min(hi, 1.0 + float(lambda_weight))
        elif current == negative:
            weights[pos_idx] = max(lo, 1.0 - float(lambda_weight))
    return weights


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--lambda-weight", type=float, default=0.3)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument(
        "--positive-all-comp-valid",
        action="store_true",
        help="Treat strict, all-metal, and single-element comp-valid rollouts as CEPO positive evidence.",
    )
    parser.add_argument(
        "--include-all-metal-negative",
        action="store_true",
        help="Treat all-metal shortcuts as CEPO negative evidence. Off by default because all-metal remains comp-valid.",
    )
    parser.add_argument(
        "--include-single-element-negative",
        action="store_true",
        help="Treat single-element shortcuts as CEPO negative evidence. Off by default for comp-valid-priority RL.",
    )
    parser.add_argument(
        "--zero-non-credit",
        action="store_true",
        help="Set non-composition token weights to zero so TraceRL updates only N/active-element/boundary tokens.",
    )
    args = parser.parse_args()

    rows = read_jsonl(args.input_jsonl)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[group_key(row)].append(row)

    output: list[dict[str, Any]] = []
    group_has_pos_neg = 0
    role_counts: Counter[str] = Counter()
    weight_values: list[float] = []
    for _, group in grouped.items():
        positives = [
            row
            for row in group
            if is_positive(row, include_all_comp_valid=args.positive_all_comp_valid)
        ]
        negatives = [
            row
            for row in group
            if is_negative(
                row,
                include_all_metal=args.include_all_metal_negative,
                include_single_element=args.include_single_element_negative,
            )
        ]
        pos = max(positives, key=row_score) if positives else None
        neg = min(negatives, key=row_score) if negatives else None
        has_pair = pos is not None and neg is not None
        group_has_pos_neg += int(has_pair)
        for row in group:
            new_row = dict(row)
            row_role = role(
                row,
                pos,
                neg,
                include_all_comp_valid_positive=args.positive_all_comp_valid,
                include_all_metal_negative=args.include_all_metal_negative,
                include_single_element_negative=args.include_single_element_negative,
            )
            role_counts[row_role] += 1
            if has_pair:
                weights = token_weights(
                    row,
                    pos,  # type: ignore[arg-type]
                    neg,  # type: ignore[arg-type]
                    lambda_weight=args.lambda_weight,
                    clip_eps=args.clip_eps,
                    zero_non_credit=args.zero_non_credit,
                )
                new_row["cepo_positive_sample_idx"] = pos.get("sample_idx")
                new_row["cepo_negative_sample_idx"] = neg.get("sample_idx")
            else:
                tokens = tokenize_answer_text(str(row.get("response") or row.get("answer") or row.get("text") or ""))
                if args.zero_non_credit:
                    credit = credit_positions(tokens)
                    weights = [1.0 if idx in credit else 0.0 for idx in range(len(tokens))]
                else:
                    weights = [1.0] * len(tokens)
                new_row["cepo_positive_sample_idx"] = None
                new_row["cepo_negative_sample_idx"] = None
            new_row["cepo_group_has_pos_neg"] = has_pair
            new_row["cepo_role"] = row_role
            new_row["cepo_token_weights"] = weights
            weight_values.extend(weights)
            output.append(new_row)

    write_jsonl(args.output_jsonl, output)
    clipped = sum(1 for value in weight_values if abs(value - 1.0) >= float(args.clip_eps) - 1e-9)
    summary = {
        "input_jsonl": str(args.input_jsonl),
        "output_jsonl": str(args.output_jsonl),
        "count": len(output),
        "groups": len(grouped),
        "group_has_pos_neg_rate": group_has_pos_neg / max(1, len(grouped)),
        "cepo_role_counts": dict(role_counts.most_common()),
        "token_weight_mean": sum(weight_values) / max(1, len(weight_values)),
        "credit_clip_ratio": clipped / max(1, len(weight_values)),
        "lambda_weight": args.lambda_weight,
        "clip_eps": args.clip_eps,
        "positive_all_comp_valid": bool(args.positive_all_comp_valid),
        "include_all_metal_negative": bool(args.include_all_metal_negative),
        "include_single_element_negative": bool(args.include_single_element_negative),
        "zero_non_credit": bool(args.zero_non_credit),
    }
    write_json(str(args.summary_json), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
