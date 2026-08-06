#!/usr/bin/env python3
"""Build formula-semantic fixed-slot MP-20 SFT data.

This builder keeps the successful fixed-slot answer format unchanged.  It adds
training rows whose prompt explicitly ties natural formula text to the matching
special-token element formula, while the answer remains the same 107-token
fixed-slot structure.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from crystal_dlm.chemical_plan import classify_composition_for_plan  # noqa: E402
from crystal_dlm.composition_validity import (  # noqa: E402
    element_symbols,
    formula_from_composition,
    reduced_composition,
)
from crystal_dlm.fixed_slot import (  # noqa: E402
    ANSWER_TOKEN_COUNT,
    Z_TO_SYMBOL,
    parse_fixed_slot_answer,
    write_json,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            count += 1
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return count


def maybe_load_tokenizer(tokenizer_path: str | None, vocab_file: Path):
    if not tokenizer_path:
        return None
    try:
        from transformers import AutoTokenizer
    except Exception:
        return None
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if vocab_file.exists():
        with vocab_file.open(encoding="utf-8") as handle:
            tokens = [line.strip() for line in handle if line.strip()]
        tokenizer.add_special_tokens({"additional_special_tokens": tokens})
    return tokenizer


def token_len(tokenizer, text: str) -> int | None:
    if tokenizer is None:
        return None
    return int(len(tokenizer(text, add_special_tokens=False)["input_ids"]))


def row_answer(row: dict[str, Any]) -> str:
    return str(row.get("answer") or row.get("response") or row.get("text") or "")


def formula_from_counter(counter: Counter[int]) -> tuple[list[int], list[int], str]:
    elems = sorted(int(elem) for elem in counter)
    counts = [int(counter[elem]) for elem in elems]
    return elems, counts, formula_from_composition(elems, counts)


def reason_bucket(reason: str) -> str:
    reason = str(reason or "").strip()
    if reason == "charge_neutral_pauling_valid":
        return "strict"
    if reason == "all_metal_shortcut":
        return "all_metal"
    if reason == "single_element_shortcut":
        return "single_element"
    if reason:
        return "invalid"
    return "unknown"


def signed_state(value: int) -> str:
    value = int(value)
    return f"+{value}" if value > 0 else str(value)


def stability_hint(row: dict[str, Any]) -> str:
    raw = row.get("e_above_hull")
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    if raw in (None, ""):
        raw = metadata.get("e_above_hull")
    try:
        value = float(raw)
    except Exception:
        return "unknown"
    if value <= 0.0:
        return "stable_ehull_0"
    if value < 0.1:
        return "metastable_ehull_lt_0.1"
    return "higher_ehull"


def build_formula_payload(row: dict[str, Any], arrays: dict[str, Any]) -> dict[str, Any]:
    atom_counter: Counter[int] = Counter(int(value) for value in arrays["atom_types"])
    elems, full_counts, full_formula = formula_from_counter(atom_counter)
    reduced_elems, reduced_counts = reduced_composition(arrays["atom_types"])
    reduced_formula = formula_from_composition(reduced_elems, reduced_counts)
    symbols = list(element_symbols(elems))
    special_formula = " + ".join(
        f"<E_{symbol}> x{count}" for symbol, count in zip(symbols, full_counts)
    )
    equivalence = "; ".join(f"{symbol} == <E_{symbol}>" for symbol in symbols)
    classification = classify_composition_for_plan(list(reduced_elems), list(reduced_counts))
    reason = str(row.get("composition_reason") or classification.get("reason") or "unknown")
    oxidation_states = classification.get("oxidation_states")
    if oxidation_states:
        ox_symbols = list(element_symbols(reduced_elems))
        oxidation_hint = ", ".join(
            f"{symbol}{signed_state(int(state))}"
            for symbol, state in zip(ox_symbols, oxidation_states)
        )
    elif reason == "all_metal_shortcut":
        oxidation_hint = "all-metal/intermetallic bonding; avoid collapsing to one element"
    elif reason == "single_element_shortcut":
        oxidation_hint = "single-element shortcut; not preferred for diverse discovery"
    elif reason == "charge_neutrality_fail":
        oxidation_hint = "charge neutrality not found; prefer nearby balanced ratios"
    elif reason == "pauling_fail_or_ratio_rejected":
        oxidation_hint = "Pauling or ratio screen rejects common charge assignment"
    elif reason == "oxidation_state_missing":
        oxidation_hint = "one or more oxidation states are unsupported"
    else:
        oxidation_hint = "prefer charge-balanced inorganic chemistry"
    return {
        "num_atoms": int(arrays["num_atoms"]),
        "atom_count_token": f"<N_{int(arrays['num_atoms']):03d}>",
        "full_formula": full_formula,
        "reduced_formula": reduced_formula,
        "special_formula": special_formula,
        "element_equivalence": equivalence,
        "symbols": symbols,
        "full_counts": full_counts,
        "num_elements": len(symbols),
        "composition_reason": reason,
        "composition_bucket": reason_bucket(reason),
        "oxidation_hint": oxidation_hint,
        "stability_hint": stability_hint(row),
    }


def build_formula_prompt(payload: dict[str, Any]) -> str:
    return (
        "Generate only the 107 fixed-slot crystal tokens for one MP-20-like inorganic crystal.\n"
        "The natural formula, special-token formula, atom count token, and output structure must describe the same atom multiset.\n"
        f"full_formula: {payload['full_formula']}\n"
        f"reduced_formula: {payload['reduced_formula']}\n"
        f"special_formula: {payload['special_formula']}\n"
        f"element_equivalence: {payload['element_equivalence']}\n"
        f"atom_count_token: {payload['atom_count_token']}\n"
        f"composition_reason: {payload['composition_reason']}\n"
        f"oxidation_hint: {payload['oxidation_hint']}\n"
        f"stability_hint: {payload['stability_hint']}\n"
        "Use the given formula exactly, keep lattice volume non-zero, and avoid exact or PBC-equivalent duplicate coordinates.\n"
        "fixed_slot_crystal_tokens:"
    )


def make_semantic_row(
    row: dict[str, Any],
    *,
    tokenizer,
    semantic_weight_multiplier: float,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if payload is None:
        arrays = parse_fixed_slot_answer(row_answer(row), strict=True)
        payload = build_formula_payload(row, arrays)
        if int(payload["num_atoms"]) != int(arrays["num_atoms"]):
            raise ValueError("formula payload atom count mismatch")
    answer = str(row["answer"])
    prompt = build_formula_prompt(payload)
    prompt_text = prompt.rstrip() + "\n"
    out = dict(row)
    metadata = dict(row.get("metadata") or {})
    metadata["source_task"] = row.get("task", "unconditional")
    metadata["formula_semantic"] = payload
    out.update(
        {
            "task": "formula_semantic_fixed_slot",
            "prompt": prompt,
            "answer": answer,
            "text": prompt_text + answer,
            "prompt_length": token_len(tokenizer, prompt_text),
            "answer_model_length": token_len(tokenizer, answer),
            "loss_profile": "fixed_slot",
            "composition_reason": payload["composition_reason"],
            "composition_bucket": payload["composition_bucket"],
            "full_formula": payload["full_formula"],
            "reduced_formula": payload["reduced_formula"],
            "num_atoms": int(payload["num_atoms"]),
            "num_elements": int(payload["num_elements"]),
            "selection_role": "formula_semantic_conditioned",
            "source_selection_role": row.get("selection_role"),
            "formula_semantic": payload,
            "semantic_conditioned": True,
            "metadata": metadata,
            "sample_weight": float(row.get("sample_weight", 1.0) or 1.0)
            * float(semantic_weight_multiplier),
        }
    )
    return out


def annotate_base_row(row: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    metadata = dict(row.get("metadata") or {})
    metadata.setdefault("formula_semantic_reference", payload)
    out.update(
        {
            "composition_reason": payload["composition_reason"],
            "composition_bucket": payload["composition_bucket"],
            "full_formula": payload["full_formula"],
            "reduced_formula": payload["reduced_formula"],
            "num_atoms": int(payload["num_atoms"]),
            "num_elements": int(payload["num_elements"]),
            "metadata": metadata,
        }
    )
    return out


def copy_sidecars(input_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in (
        "vocab_tokens.txt",
        "vocab.json",
        "stats.json",
        "tokenizer_report.json",
        "prompt_pool.json",
        "ehull_weight_summary.json",
    ):
        src = input_dir / filename
        if src.exists():
            shutil.copy2(src, output_dir / filename)


def histogram(values: Iterable[int | None]) -> dict[str, int]:
    counter = Counter("none" if value is None else str(value) for value in values)
    return dict(counter.most_common())


def parse_name_set(text: str | None) -> set[str]:
    if not text:
        return set()
    values: set[str] = set()
    for item in str(text).split(","):
        item = item.strip()
        if item:
            values.add(item)
    return values


def semantic_allowed(
    payload: dict[str, Any],
    *,
    include_reasons: set[str],
    exclude_reasons: set[str],
    min_elements: int,
    max_elements: int,
) -> bool:
    reason = str(payload.get("composition_reason") or "")
    bucket = str(payload.get("composition_bucket") or "")
    num_elements = int(payload.get("num_elements") or 0)
    names = {reason, bucket}
    if include_reasons and names.isdisjoint(include_reasons):
        return False
    if exclude_reasons and not names.isdisjoint(exclude_reasons):
        return False
    if min_elements and num_elements < min_elements:
        return False
    if max_elements and num_elements > max_elements:
        return False
    return True


def build_split(
    *,
    split: str,
    input_path: Path,
    output_path: Path,
    tokenizer,
    semantic_fraction: float,
    semantic_train_only: bool,
    semantic_weight_multiplier: float,
    semantic_include_reasons: set[str],
    semantic_exclude_reasons: set[str],
    semantic_min_elements: int,
    semantic_max_elements: int,
    annotate_base_composition: bool,
    seed: int,
    limit: int | None,
) -> dict[str, Any]:
    rng = random.Random(seed)
    input_rows = read_jsonl(input_path)
    if limit is not None:
        input_rows = input_rows[: int(limit)]
    output_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    semantic_rows = 0
    semantic_prompt_lengths: list[int | None] = []
    answer_lengths: list[int | None] = []
    base_reason_counts: Counter[str] = Counter()
    semantic_reason_counts: Counter[str] = Counter()

    add_semantic_for_split = (split == "train") or not semantic_train_only
    for idx, row in enumerate(input_rows):
        payload = None
        base_row = row
        needs_payload = (
            annotate_base_composition
            or add_semantic_for_split
            or bool(semantic_include_reasons)
            or bool(semantic_exclude_reasons)
            or bool(semantic_min_elements)
            or bool(semantic_max_elements)
        )
        if needs_payload:
            try:
                arrays = parse_fixed_slot_answer(row_answer(row), strict=True)
                payload = build_formula_payload(row, arrays)
            except Exception as exc:  # noqa: BLE001
                failures.append({"row_idx": idx, "stage": "base_payload", "reason": type(exc).__name__, "message": str(exc)})
        if payload is not None:
            base_reason_counts[str(payload["composition_reason"])] += 1
            if annotate_base_composition:
                base_row = annotate_base_row(row, payload)
        output_rows.append(base_row)
        if not add_semantic_for_split or rng.random() > float(semantic_fraction):
            continue
        if payload is None:
            continue
        if not semantic_allowed(
            payload,
            include_reasons=semantic_include_reasons,
            exclude_reasons=semantic_exclude_reasons,
            min_elements=semantic_min_elements,
            max_elements=semantic_max_elements,
        ):
            continue
        try:
            semantic = make_semantic_row(
                row,
                tokenizer=tokenizer,
                semantic_weight_multiplier=semantic_weight_multiplier,
                payload=payload,
            )
        except Exception as exc:  # noqa: BLE001
            failures.append({"row_idx": idx, "stage": "semantic_row", "reason": type(exc).__name__, "message": str(exc)})
            continue
        output_rows.append(semantic)
        semantic_rows += 1
        semantic_reason_counts[str(payload["composition_reason"])] += 1
        semantic_prompt_lengths.append(semantic.get("prompt_length"))
        answer_lengths.append(semantic.get("answer_model_length"))

    write_jsonl(output_path, output_rows)
    failure_path = output_path.with_suffix(".failure_cases.jsonl")
    write_jsonl(failure_path, failures)
    return {
        "split": split,
        "input_rows": len(input_rows),
        "output_rows": len(output_rows),
        "base_rows": len(input_rows),
        "semantic_rows": semantic_rows,
        "failures": len(failures),
        "semantic_fraction": semantic_fraction if add_semantic_for_split else 0.0,
        "semantic_include_reasons": sorted(semantic_include_reasons),
        "semantic_exclude_reasons": sorted(semantic_exclude_reasons),
        "semantic_min_elements": int(semantic_min_elements),
        "semantic_max_elements": int(semantic_max_elements),
        "annotate_base_composition": bool(annotate_base_composition),
        "base_reason_counts": dict(base_reason_counts.most_common()),
        "semantic_reason_counts": dict(semantic_reason_counts.most_common()),
        "semantic_prompt_length_histogram": histogram(semantic_prompt_lengths),
        "answer_model_length_histogram": histogram(answer_lengths),
        "max_prompt_length": max([value for value in semantic_prompt_lengths if value is not None], default=None),
        "max_answer_model_length": max([value for value in answer_lengths if value is not None], default=None),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--semantic-fraction", type=float, default=1.0)
    parser.add_argument("--semantic-weight-multiplier", type=float, default=1.0)
    parser.add_argument("--semantic-train-only", action="store_true", default=True)
    parser.add_argument(
        "--semantic-include-reasons",
        default="",
        help="Comma-separated reason/bucket allowlist for semantic rows. Empty means allow all.",
    )
    parser.add_argument(
        "--semantic-exclude-reasons",
        default="",
        help="Comma-separated reason/bucket denylist for semantic rows.",
    )
    parser.add_argument("--semantic-min-elements", type=int, default=0)
    parser.add_argument("--semantic-max-elements", type=int, default=0)
    parser.add_argument(
        "--no-annotate-base-composition",
        action="store_false",
        dest="annotate_base_composition",
        help="Do not add top-level composition_reason/composition_bucket metadata to base rows.",
    )
    parser.set_defaults(annotate_base_composition=True)
    parser.add_argument("--seed", type=int, default=20260528)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    copy_sidecars(args.input_dir, args.output_dir)
    tokenizer = maybe_load_tokenizer(args.tokenizer_path, args.output_dir / "vocab_tokens.txt")
    semantic_include_reasons = parse_name_set(args.semantic_include_reasons)
    semantic_exclude_reasons = parse_name_set(args.semantic_exclude_reasons)
    split_stats: dict[str, Any] = {}
    for offset, split in enumerate(("train", "val", "test")):
        split_stats[split] = build_split(
            split=split,
            input_path=args.input_dir / f"{split}.jsonl",
            output_path=args.output_dir / f"{split}.jsonl",
            tokenizer=tokenizer,
            semantic_fraction=args.semantic_fraction,
            semantic_train_only=bool(args.semantic_train_only),
            semantic_weight_multiplier=float(args.semantic_weight_multiplier),
            semantic_include_reasons=semantic_include_reasons,
            semantic_exclude_reasons=semantic_exclude_reasons,
            semantic_min_elements=int(args.semantic_min_elements),
            semantic_max_elements=int(args.semantic_max_elements),
            annotate_base_composition=bool(args.annotate_base_composition),
            seed=int(args.seed) + offset * 10_000,
            limit=args.limit,
        )

    max_total = 0
    for stats in split_stats.values():
        prompt = stats.get("max_prompt_length")
        answer = stats.get("max_answer_model_length")
        if prompt is not None and answer is not None:
            max_total = max(max_total, int(prompt) + int(answer))
    summary = {
        "representation": "fixed_slot",
        "answer_token_count": ANSWER_TOKEN_COUNT,
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "semantic_policy": "formula_conditioned_fixed_slot_rows_train_only",
        "semantic_fraction": args.semantic_fraction,
        "semantic_weight_multiplier": args.semantic_weight_multiplier,
        "semantic_include_reasons": sorted(semantic_include_reasons),
        "semantic_exclude_reasons": sorted(semantic_exclude_reasons),
        "semantic_min_elements": int(args.semantic_min_elements),
        "semantic_max_elements": int(args.semantic_max_elements),
        "annotate_base_composition": bool(args.annotate_base_composition),
        "semantic_train_only": bool(args.semantic_train_only),
        "max_length_recommended": max(256, max_total + 16),
        "splits": split_stats,
    }
    write_json(str(args.output_dir / "formula_semantic_summary.json"), summary)
    write_json(str(args.output_dir / "stats.json"), summary)
    write_json(
        str(args.output_dir / "prompt_pool.json"),
        {
            "prompt_pool": ["formula_semantic_fixed_slot", "original_fixed_slot"],
            "canonical_prompt": "formula_semantic_fixed_slot",
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
