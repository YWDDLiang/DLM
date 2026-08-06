#!/usr/bin/env python3
"""Diagnose why raw crystal generations miss MP-20 composition validity."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from crystal_dlm.composition_validity import composition_record, pbc_duplicate_record
from crystal_dlm.fixed_slot import parse_fixed_slot_answer


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def load_raw_jsonl(path: Path, text_key: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for row_idx, line in enumerate(handle):
            if not line.strip():
                continue
            record = json.loads(line)
            text = record.get(text_key) or record.get("answer") or record.get("text")
            rows.append(
                {
                    "row_idx": row_idx,
                    "sample_idx": int(record.get("sample_idx", row_idx)),
                    "text": str(text or ""),
                    "parsed": record.get("parsed"),
                    "raw": record,
                }
            )
    return rows


def compact_counter(counter: Counter[str], limit: int) -> dict[str, int]:
    return dict(counter.most_common(limit))


def add_example(
    examples: dict[str, list[dict[str, Any]]],
    key: str,
    payload: dict[str, Any],
    limit: int,
) -> None:
    if len(examples[key]) < limit:
        examples[key].append(payload)


def analyze(path: Path, *, text_key: str, example_limit: int, top_k: int) -> dict[str, Any]:
    rows = load_raw_jsonl(path, text_key)
    reason_counts: Counter[str] = Counter()
    invalid_reason_counts: Counter[str] = Counter()
    shortcut_reason_counts: Counter[str] = Counter()
    formula_by_reason: dict[str, Counter[str]] = defaultdict(Counter)
    counts_by_reason: dict[str, Counter[str]] = defaultdict(Counter)
    elements_by_reason: dict[str, Counter[str]] = defaultdict(Counter)
    atom_count_by_reason: dict[str, Counter[str]] = defaultdict(Counter)
    num_elements_by_reason: dict[str, Counter[str]] = defaultdict(Counter)
    pbc_by_reason: dict[str, Counter[str]] = defaultdict(Counter)
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    total = 0
    parsed = 0
    comp_valid = 0
    strict = 0
    pbc_duplicate = 0

    for row in rows:
        total += 1
        try:
            arrays = parse_fixed_slot_answer(row["text"])
        except Exception as exc:
            reason = f"parse_error:{type(exc).__name__}"
            reason_counts[reason] += 1
            add_example(
                examples,
                reason,
                {
                    "sample_idx": row["sample_idx"],
                    "message": str(exc),
                    "text_prefix": row["text"][:240],
                },
                example_limit,
            )
            continue
        parsed += 1
        comp = composition_record(arrays["atom_types"])
        pbc = pbc_duplicate_record(arrays["frac_coords"])
        reason = str(comp["reason"])
        comp_valid += int(bool(comp["comp_valid"]))
        strict += int(reason == "charge_neutral_pauling_valid")
        pbc_duplicate += int(bool(pbc["has_pbc_equivalent_duplicate"]))
        reason_counts[reason] += 1
        if not comp["comp_valid"]:
            invalid_reason_counts[reason] += 1
        if reason in {"single_element_shortcut", "all_metal_shortcut"}:
            shortcut_reason_counts[reason] += 1
        formula = str(comp["formula"])
        formula_by_reason[reason][formula] += 1
        counts_by_reason[reason][",".join(map(str, comp["counts"]))] += 1
        atom_count_by_reason[reason][str(comp["num_atoms"])] += 1
        num_elements_by_reason[reason][str(comp["num_elements"])] += 1
        pbc_by_reason[reason]["pbc_dup" if pbc["has_pbc_equivalent_duplicate"] else "no_pbc_dup"] += 1
        for symbol in comp["symbols"]:
            elements_by_reason[reason][str(symbol)] += 1
        add_example(
            examples,
            reason,
            {
                "sample_idx": row["sample_idx"],
                "formula": formula,
                "symbols": comp["symbols"],
                "counts": comp["counts"],
                "num_atoms": comp["num_atoms"],
                "num_elements": comp["num_elements"],
                "pbc_equivalent_duplicate_site_count": pbc["pbc_equivalent_duplicate_site_count"],
                "text_prefix": row["text"][:260],
            },
            example_limit,
        )

    invalid_total = sum(invalid_reason_counts.values())
    shortcut_total = sum(shortcut_reason_counts.values())
    reason_details: dict[str, Any] = {}
    for reason, count in reason_counts.most_common():
        reason_details[reason] = {
            "count": count,
            "fraction": count / max(1, parsed),
            "formula_top": compact_counter(formula_by_reason[reason], top_k),
            "reduced_count_top": compact_counter(counts_by_reason[reason], top_k),
            "atom_count_histogram": compact_counter(atom_count_by_reason[reason], top_k),
            "num_elements_histogram": compact_counter(num_elements_by_reason[reason], top_k),
            "element_top": compact_counter(elements_by_reason[reason], top_k),
            "pbc_counts": dict(pbc_by_reason[reason]),
            "examples": examples.get(reason, []),
        }

    return {
        "input_jsonl": str(path),
        "total_rows": total,
        "parsed_count": parsed,
        "parse_rate": parsed / max(1, total),
        "comp_valid_count": comp_valid,
        "comp_valid_rate": comp_valid / max(1, parsed),
        "strict_valid_count": strict,
        "strict_valid_rate": strict / max(1, parsed),
        "shortcut_count": shortcut_total,
        "shortcut_fraction": shortcut_total / max(1, parsed),
        "invalid_count": invalid_total,
        "invalid_fraction": invalid_total / max(1, parsed),
        "pbc_equivalent_duplicate_count": pbc_duplicate,
        "pbc_equivalent_duplicate_fraction": pbc_duplicate / max(1, parsed),
        "reason_counts": dict(reason_counts.most_common()),
        "invalid_reason_counts": dict(invalid_reason_counts.most_common()),
        "invalid_reason_fractions_within_invalid": {
            key: value / max(1, invalid_total)
            for key, value in invalid_reason_counts.most_common()
        },
        "shortcut_reason_counts": dict(shortcut_reason_counts.most_common()),
        "reason_details": reason_details,
        "headline": build_headline(
            parsed=parsed,
            comp_valid=comp_valid,
            strict=strict,
            invalid_reason_counts=invalid_reason_counts,
            shortcut_reason_counts=shortcut_reason_counts,
        ),
    }


def build_headline(
    *,
    parsed: int,
    comp_valid: int,
    strict: int,
    invalid_reason_counts: Counter[str],
    shortcut_reason_counts: Counter[str],
) -> list[str]:
    misses = max(0, parsed - comp_valid)
    notes = [
        f"comp_valid_gap_to_90 = {max(0.0, 0.90 - comp_valid / max(1, parsed)):.4f}",
        f"strict_valid_rate = {strict / max(1, parsed):.4f}",
    ]
    if misses:
        top_invalid = invalid_reason_counts.most_common(3)
        notes.append(
            "invalid contributors: "
            + ", ".join(f"{reason}={count / max(1, parsed):.4f}" for reason, count in top_invalid)
        )
    if shortcut_reason_counts:
        notes.append(
            "shortcut contributors: "
            + ", ".join(
                f"{reason}={count / max(1, parsed):.4f}"
                for reason, count in shortcut_reason_counts.most_common()
            )
        )
    return notes


def write_markdown(path: Path, payload: Mapping[str, Any]) -> None:
    lines = [
        "# Composition Failure Modes",
        "",
        "## Headline",
    ]
    for note in payload["headline"]:
        lines.append(f"- {note}")
    lines.extend(
        [
            "",
            "## Summary",
            f"- parse_rate: {payload['parse_rate']:.4f}",
            f"- comp_valid_rate: {payload['comp_valid_rate']:.4f}",
            f"- strict_valid_rate: {payload['strict_valid_rate']:.4f}",
            f"- shortcut_fraction: {payload['shortcut_fraction']:.4f}",
            f"- invalid_fraction: {payload['invalid_fraction']:.4f}",
            f"- pbc_duplicate_fraction: {payload['pbc_equivalent_duplicate_fraction']:.4f}",
            "",
            "## Reason Counts",
            "```json",
            json.dumps(payload["reason_counts"], ensure_ascii=False, indent=2),
            "```",
            "",
            "## Invalid Reasons",
            "```json",
            json.dumps(payload["invalid_reason_counts"], ensure_ascii=False, indent=2),
            "```",
            "",
        ]
    )
    for reason, details in payload["reason_details"].items():
        lines.extend(
            [
                f"## {reason}",
                f"- count: {details['count']}",
                f"- fraction: {details['fraction']:.4f}",
                f"- formula_top: {details['formula_top']}",
                f"- reduced_count_top: {details['reduced_count_top']}",
                f"- num_elements_histogram: {details['num_elements_histogram']}",
                f"- atom_count_histogram: {details['atom_count_histogram']}",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-jsonl", type=Path, required=True)
    parser.add_argument("--text-key", default="text")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--example-limit", type=int, default=12)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    payload = analyze(
        args.raw_jsonl,
        text_key=args.text_key,
        example_limit=args.example_limit,
        top_k=args.top_k,
    )
    write_json(args.output_json, payload)
    write_markdown(args.output_md, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
