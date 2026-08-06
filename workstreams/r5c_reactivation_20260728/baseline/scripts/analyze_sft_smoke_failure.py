#!/usr/bin/env python3
"""Summarize failed MP-20 SFT smoke candidates and shortcut failure modes."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def load_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def find_composition_block(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    if key in payload and isinstance(payload[key], Mapping):
        return payload[key]
    for value in payload.values():
        if isinstance(value, Mapping) and "records" in value and "reason_counts" in value:
            return value
    if "records" in payload and "reason_counts" in payload:
        return payload
    raise KeyError(f"Could not find composition block {key!r}")


def discover_candidates(run_dir: Path) -> list[tuple[str, Path, Path | None]]:
    candidates: list[tuple[str, Path, Path | None]] = []
    for comp_path in sorted((run_dir / "notes").glob("*/sample256_composition.json")):
        name = comp_path.parent.name
        raw_candidates = [
            run_dir / "outputs" / name / "raw_generations.jsonl",
            run_dir / "outputs" / name / "sample256" / "raw_generations.jsonl",
            run_dir / "outputs" / f"{name}_sample256" / "raw_generations.jsonl",
        ]
        raw_path = next((path for path in raw_candidates if path.exists()), None)
        candidates.append((name, comp_path, raw_path))
    if candidates:
        return candidates
    comp_path = run_dir / "notes" / "sample256_composition.json"
    if comp_path.exists():
        raw_path = run_dir / "outputs" / "sample256" / "raw_generations.jsonl"
        candidates.append(("candidate", comp_path, raw_path if raw_path.exists() else None))
    return candidates


def compact_counter(mapping: Mapping[str, Any], limit: int = 20) -> dict[str, Any]:
    return dict(list(mapping.items())[:limit])


def record_examples(
    records: list[Mapping[str, Any]],
    raw_rows: list[Mapping[str, Any]],
    *,
    reason: str,
    limit: int,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    by_sample_idx = {
        int(row.get("sample_idx", idx)): row
        for idx, row in enumerate(raw_rows)
        if isinstance(row, Mapping)
    }
    for record in records:
        if str(record.get("reason")) != reason:
            continue
        sample_idx = int(record.get("sample_idx", record.get("row_idx", len(examples))))
        raw = by_sample_idx.get(sample_idx, {})
        text = str(raw.get("text") or raw.get("answer") or "")
        examples.append(
            {
                "sample_idx": sample_idx,
                "formula": record.get("formula"),
                "symbols": record.get("symbols"),
                "counts": record.get("counts"),
                "num_atoms": record.get("num_atoms"),
                "num_elements": record.get("num_elements"),
                "pbc_equivalent_duplicate_site_count": record.get("pbc_equivalent_duplicate_site_count"),
                "text_prefix": text[:240],
            }
        )
        if len(examples) >= limit:
            break
    return examples


def summarize_candidate(
    name: str,
    comp_path: Path,
    raw_path: Path | None,
    *,
    composition_key: str,
    example_limit: int,
) -> dict[str, Any]:
    payload = read_json(comp_path)
    comp = find_composition_block(payload, composition_key)
    records = list(comp.get("records") or [])
    raw_rows = load_jsonl(raw_path)
    count = int(comp.get("count") or len(records) or 0)
    reason_counts = dict(comp.get("reason_counts") or {})
    single_count = int(reason_counts.get("single_element_shortcut", 0))
    all_metal_count = int(reason_counts.get("all_metal_shortcut", 0))
    strict_count = int(reason_counts.get("charge_neutral_pauling_valid", 0))
    pbc_fraction = float(comp.get("pbc_equivalent_duplicate_fraction", 0.0))

    num_elements_hist = dict(comp.get("num_elements_histogram") or {})
    formula_top = dict(comp.get("formula_top30") or {})
    top_single_formula = None
    if records:
        single_formulas = Counter(
            str(record.get("formula"))
            for record in records
            if str(record.get("reason")) == "single_element_shortcut"
        )
        if single_formulas:
            top_single_formula = dict(single_formulas.most_common(20))

    return {
        "name": name,
        "composition_json": str(comp_path),
        "raw_jsonl": str(raw_path) if raw_path else None,
        "count": count,
        "comp_valid_rate": float(comp.get("comp_valid_rate", 0.0)),
        "strict_valid_rate": strict_count / max(1, count),
        "single_element_rate": single_count / max(1, count),
        "all_metal_rate": all_metal_count / max(1, count),
        "shortcut_fraction": float(comp.get("shortcut_fraction", 0.0)),
        "pbc_equivalent_duplicate_fraction": pbc_fraction,
        "reason_counts": reason_counts,
        "formula_top30": compact_counter(formula_top, 30),
        "single_element_formula_top20": top_single_formula or {},
        "num_atoms_histogram": compact_counter(dict(comp.get("num_atoms_histogram") or {}), 30),
        "num_elements_histogram": compact_counter(num_elements_hist, 20),
        "reduced_count_histogram_top30": compact_counter(
            dict(comp.get("reduced_count_histogram_top30") or {}),
            30,
        ),
        "examples": {
            "single_element_shortcut": record_examples(
                records,
                raw_rows,
                reason="single_element_shortcut",
                limit=example_limit,
            ),
            "charge_neutrality_fail": record_examples(
                records,
                raw_rows,
                reason="charge_neutrality_fail",
                limit=example_limit,
            ),
            "pauling_fail_or_ratio_rejected": record_examples(
                records,
                raw_rows,
                reason="pauling_fail_or_ratio_rejected",
                limit=example_limit,
            ),
        },
    }


def load_train_reference(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    payload = read_json(path)
    train = payload.get("splits", {}).get("train", {})
    return {
        "path": str(path),
        "strict_valid_rate": train.get("strict_valid_rate"),
        "single_element_rate": train.get("single_element_rate"),
        "all_metal_rate": train.get("all_metal_rate"),
        "invalid_rate": train.get("invalid_rate"),
        "bucket_counts": train.get("bucket_counts"),
        "mask_policy_counts": train.get("mask_policy_counts"),
        "source_bucket_counts": train.get("source_bucket_counts"),
    }


def diagnose(candidates: list[dict[str, Any]], train_ref: Mapping[str, Any]) -> list[str]:
    notes: list[str] = []
    train_single = train_ref.get("single_element_rate")
    for candidate in candidates:
        name = candidate["name"]
        single = float(candidate["single_element_rate"])
        strict = float(candidate["strict_valid_rate"])
        all_metal = float(candidate["all_metal_rate"])
        pbc = float(candidate["pbc_equivalent_duplicate_fraction"])
        if single > 0.02:
            notes.append(
                f"{name}: single_element_rate={single:.4f} exceeds gate 0.02; stop scaling."
            )
        if train_single is not None and single > max(0.02, float(train_single) * 4.0):
            notes.append(
                f"{name}: single-element shortcut amplified from train {float(train_single):.4f} to {single:.4f}."
            )
        if strict < 0.35:
            notes.append(
                f"{name}: strict_valid_rate={strict:.4f} is below smoke gate 0.35 despite parse/graph success."
            )
        if all_metal > 0.40:
            notes.append(f"{name}: all_metal_rate={all_metal:.4f} exceeds final target 0.40.")
        if pbc > 0:
            notes.append(f"{name}: PBC duplicate rate={pbc:.4f}; inspect coordinate credit/masks.")
    if not notes:
        notes.append("No obvious shortcut amplification found in composition smoke summaries.")
    return notes


def write_markdown(path: Path, payload: Mapping[str, Any]) -> None:
    lines: list[str] = []
    lines.append(f"# {payload['run_dir']} SFT smoke failure analysis")
    lines.append("")
    lines.append("## Diagnosis")
    for note in payload["diagnosis"]:
        lines.append(f"- {note}")
    lines.append("")
    train_ref = payload.get("train_reference") or {}
    if train_ref:
        lines.append("## Train reference")
        lines.append(
            "- rates: "
            f"strict={train_ref.get('strict_valid_rate')}, "
            f"single={train_ref.get('single_element_rate')}, "
            f"all_metal={train_ref.get('all_metal_rate')}, "
            f"invalid={train_ref.get('invalid_rate')}"
        )
        lines.append(f"- mask_policy_counts: {train_ref.get('mask_policy_counts')}")
        lines.append("")
    for candidate in payload["candidates"]:
        lines.append(f"## Candidate {candidate['name']}")
        lines.append(
            "- rates: "
            f"comp={candidate['comp_valid_rate']:.4f}, "
            f"strict={candidate['strict_valid_rate']:.4f}, "
            f"single={candidate['single_element_rate']:.4f}, "
            f"all_metal={candidate['all_metal_rate']:.4f}, "
            f"shortcut={candidate['shortcut_fraction']:.4f}, "
            f"pbc_dup={candidate['pbc_equivalent_duplicate_fraction']:.4f}"
        )
        lines.append(f"- reason_counts: {candidate['reason_counts']}")
        lines.append(f"- num_elements_histogram: {candidate['num_elements_histogram']}")
        lines.append(f"- single_element_formula_top20: {candidate['single_element_formula_top20']}")
        examples = candidate["examples"]["single_element_shortcut"][:5]
        if examples:
            lines.append("- single examples:")
            for example in examples:
                lines.append(
                    "  - "
                    f"sample={example['sample_idx']} "
                    f"formula={example['formula']} "
                    f"num_atoms={example['num_atoms']} "
                    f"symbols={example['symbols']} "
                    f"counts={example['counts']}"
                )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--train-summary-json", type=Path, default=None)
    parser.add_argument("--composition-key", default="raw_jsonl")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--example-limit", type=int, default=12)
    args = parser.parse_args()

    candidates = discover_candidates(args.run_dir)
    if not candidates:
        raise FileNotFoundError(f"No sample256_composition.json candidates under {args.run_dir}")
    summaries = [
        summarize_candidate(
            name,
            comp_path,
            raw_path,
            composition_key=args.composition_key,
            example_limit=args.example_limit,
        )
        for name, comp_path, raw_path in candidates
    ]
    train_ref = load_train_reference(args.train_summary_json)
    payload = {
        "run_dir": str(args.run_dir),
        "train_reference": train_ref,
        "candidates": summaries,
        "diagnosis": diagnose(summaries, train_ref),
    }
    write_json(args.output_json, payload)
    write_markdown(args.output_md, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
