#!/usr/bin/env python3
"""Freeze outcome-blind train-only rich Plans for energy-pair mining."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crystal_dlm.r5_plan_state import build_body_prompt, validate_plan_state  # noqa: E402


SALT = "h1a2-energy-pair-plan-v1"


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError(f"non-object JSONL row in {path}")
                yield value


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def n_bin(value: int) -> str:
    if value <= 4:
        return "01_04"
    if value <= 8:
        return "05_08"
    if value <= 12:
        return "09_12"
    return "13_20"


def plan_from_row(row: Mapping[str, Any]) -> Mapping[str, Any] | None:
    value = row.get("plan_state") or row.get("r5_plan_state")
    return value if isinstance(value, Mapping) else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--excluded-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-plans", type=int, default=256)
    parser.add_argument("--validation-plans", type=int, default=64)
    args = parser.parse_args()
    if args.validation_plans <= 0 or args.validation_plans >= args.num_plans:
        raise ValueError("validation-plans must lie strictly inside num-plans")

    excluded_formulas: set[str] = set()
    excluded_plan_hashes: set[str] = set()
    for row in read_jsonl(args.excluded_jsonl):
        plan = plan_from_row(row)
        if plan is None:
            continue
        excluded_formulas.add(str(plan.get("formula") or ""))
        excluded_plan_hashes.add(sha256(plan))

    candidates: dict[str, dict[str, Any]] = {}
    seen_formulas: set[str] = set()
    source_rows = 0
    rejected: Counter[str] = Counter()
    for source_idx, row in enumerate(read_jsonl(args.train_jsonl)):
        source_rows += 1
        plan = plan_from_row(row)
        if plan is None:
            rejected["missing_plan_state"] += 1
            continue
        validation = validate_plan_state(plan)
        if not validation.valid:
            rejected["invalid_plan_state"] += 1
            continue
        formula = str(plan["formula"])
        plan_hash = sha256(plan)
        if formula in excluded_formulas or plan_hash in excluded_plan_hashes:
            rejected["excluded_raw1000_identity"] += 1
            continue
        if formula in seen_formulas:
            rejected["duplicate_formula"] += 1
            continue
        seen_formulas.add(formula)
        selection_hash = hashlib.sha256(
            f"{SALT}|select|{plan_hash}".encode("utf-8")
        ).hexdigest()
        candidates[selection_hash] = {
            "source_row_idx": source_idx,
            "plan_state": dict(plan),
            "plan_state_sha256": plan_hash,
            "formula": formula,
        }

    ordered = [candidates[key] for key in sorted(candidates)]
    selected = ordered[: int(args.num_plans)]
    if len(selected) != int(args.num_plans):
        raise ValueError(f"only {len(selected)} eligible unique Plans")
    validation_rank = sorted(
        range(len(selected)),
        key=lambda idx: hashlib.sha256(
            f"{SALT}|split|{selected[idx]['plan_state_sha256']}".encode("utf-8")
        ).hexdigest(),
    )
    validation_indices = set(validation_rank[: int(args.validation_plans)])

    args.output_dir.mkdir(parents=True, exist_ok=False)
    output_rows: list[dict[str, Any]] = []
    strata: Counter[str] = Counter()
    for sample_idx, value in enumerate(selected):
        plan = value["plan_state"]
        split = "validation" if sample_idx in validation_indices else "train"
        stratum = "|".join(
            (
                str(plan.get("anion_framework") or "other"),
                f"arity{len(plan.get('elements') or ())}",
                f"N{n_bin(int(plan['N']))}",
            )
        )
        strata[f"{split}:{stratum}"] += 1
        output_rows.append(
            {
                "sample_idx": sample_idx,
                "pair_split": split,
                "selection_rule": "salted_plan_hash_without_outcome_labels",
                "source_row_idx": value["source_row_idx"],
                "source_plan_state_sha256": value["plan_state_sha256"],
                "plan_state": plan,
                "prompt": build_body_prompt(plan).rstrip() + "\n",
            }
        )
    with (args.output_dir / "plans_for_dlm.jsonl").open("x", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {
        "schema": "h1a2_energy_pair_plan_cohort_v1",
        "salt": SALT,
        "source": str(args.train_jsonl.resolve()),
        "excluded": str(args.excluded_jsonl.resolve()),
        "source_rows": source_rows,
        "eligible_unique_formula_plans": len(ordered),
        "selected": len(output_rows),
        "train": sum(row["pair_split"] == "train" for row in output_rows),
        "validation": sum(row["pair_split"] == "validation" for row in output_rows),
        "selected_rows_sha256": sha256(output_rows),
        "strata": dict(sorted(strata.items())),
        "rejected": dict(sorted(rejected.items())),
        "outcome_labels_used_for_selection": False,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "_SUCCESS").touch()
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
