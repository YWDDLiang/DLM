#!/usr/bin/env python3
"""Freeze the first and last 500 valid records of one raw rich-Plan stream."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


RAW_ATTEMPTS = 1000
ROUND_ATTEMPTS = 500


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        for line_index, line in enumerate(handle):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"JSONL row {line_index} is not an object")
            yield line_index, value


def normalized_plan(row: Mapping[str, Any]) -> dict[str, Any] | None:
    if row.get("parsed") is False or row.get("planner_parsed") is False:
        return None
    plan = row.get("plan_state") or row.get("r5_plan_state")
    if not isinstance(plan, Mapping):
        return None
    try:
        num_atoms = int(plan["N"])
        elements = [str(value) for value in plan["elements"]]
        counts = [int(value) for value in plan["counts"]]
    except (KeyError, TypeError, ValueError):
        return None
    formula = str(plan.get("formula") or "").strip()
    if not formula or not 1 <= num_atoms <= 20:
        return None
    if not elements or len(elements) != len(counts):
        return None
    if any(count <= 0 for count in counts) or sum(counts) != num_atoms:
        return None
    result = dict(plan)
    result["N"] = num_atoms
    result["elements"] = elements
    result["counts"] = counts
    result["formula"] = formula
    return result


def freeze(input_path: Path, output_dir: Path) -> dict[str, Any]:
    valid: list[dict[str, Any]] = []
    raw_rows = 0
    invalid_rows = 0
    for line_index, row in read_jsonl(input_path):
        raw_rows += 1
        plan = normalized_plan(row)
        if plan is None:
            invalid_rows += 1
            continue
        source_sample_idx = row.get("sample_idx", row.get("ordinal", line_index))
        valid.append(
            {
                "source_line_index": int(line_index),
                "source_sample_idx": int(source_sample_idx),
                "plan_state": plan,
                "raw_plan_text": str(row.get("raw_plan_text") or row.get("plan_text") or ""),
                "source_record_sha256": canonical_sha256(row),
                "plan_state_sha256": canonical_sha256(plan),
            }
        )
    if len(valid) < RAW_ATTEMPTS:
        raise ValueError(f"need at least {RAW_ATTEMPTS} valid Plans, found {len(valid)}")

    cohort = valid[:RAW_ATTEMPTS]
    for sample_idx, record in enumerate(cohort):
        record["sample_idx"] = sample_idx
        record["valid_ordinal"] = sample_idx
    rounds = {
        "round1_first500": cohort[:ROUND_ATTEMPTS],
        "round2_last500": cohort[ROUND_ATTEMPTS:],
    }
    if {row["sample_idx"] for row in rounds["round1_first500"]} & {
        row["sample_idx"] for row in rounds["round2_last500"]
    }:
        raise RuntimeError("the two frozen rounds overlap")

    output_dir.mkdir(parents=True, exist_ok=False)
    outputs: dict[str, dict[str, Any]] = {}
    payloads = {"raw1000": cohort, **rounds}
    for name, rows in payloads.items():
        path = output_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(canonical_json(row) + "\n")
        outputs[name] = {
            "path": str(path),
            "rows": len(rows),
            "sample_idx_min": min(int(row["sample_idx"]) for row in rows),
            "sample_idx_max": max(int(row["sample_idx"]) for row in rows),
            "sha256": canonical_sha256(rows),
        }

    formulas = Counter(str(row["plan_state"]["formula"]) for row in cohort)
    manifest = {
        "schema": "h1a2_raw1000_two_valid500_v1",
        "source": str(input_path.resolve()),
        "cohort_rule": "first_1000_valid_in_source_order_split_first500_last500",
        "raw_rows_scanned": raw_rows,
        "invalid_rows_skipped": invalid_rows,
        "valid_rows_available": len(valid),
        "frozen_rows": len(cohort),
        "unique_plan_states": len({row["plan_state_sha256"] for row in cohort}),
        "unique_formulas": len(formulas),
        "duplicate_formula_instances": sum(count - 1 for count in formulas.values()),
        "outputs": outputs,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "_SUCCESS").touch()
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(freeze(args.input, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
