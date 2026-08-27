#!/usr/bin/env python3
"""Audit count-valence Planner serialization and rich-property round trips."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crystal_dlm.r5_plan_state import (  # noqa: E402
    parse_countvalence_plan_state,
    plan_state_to_countvalencefields,
)
from crystal_dlm.valence_assignment import annotate_plan_with_valence  # noqa: E402


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError(f"non-object row in {path}")
                yield value


def plan_from_row(row: Mapping[str, Any]) -> Mapping[str, Any] | None:
    value = row.get("plan_state") or row.get("r5_plan_state")
    return value if isinstance(value, Mapping) else None


def canonical_composition(plan: Mapping[str, Any]) -> tuple[str, int, tuple[str, ...], tuple[int, ...]]:
    return (
        str(plan.get("formula") or ""),
        int(plan.get("N") or 0),
        tuple(str(value) for value in (plan.get("elements") or ())),
        tuple(int(value) for value in (plan.get("counts") or ())),
    )


def audit_dataset(name: str, path: Path, *, max_species: int = 7) -> dict[str, Any]:
    rows = 0
    plans = 0
    serialized = 0
    parsed = 0
    exact_composition = 0
    exact_soft_fields = 0
    source_oxidation_known = 0
    generated_charge_known = 0
    generated_charge_neutral = 0
    charge_bucket_match = 0
    valence_assignment_known = 0
    mixed_valence = 0
    failures: Counter[str] = Counter()
    assignment_modes: Counter[str] = Counter()
    assignment_failures: Counter[str] = Counter()
    pair_vocab: Counter[str] = Counter()
    counts: Counter[int] = Counter()
    oxidation_values: Counter[str] = Counter()
    charge_buckets: Counter[str] = Counter()

    for row in iter_jsonl(path):
        rows += 1
        source_plan = plan_from_row(row)
        if source_plan is None:
            failures["missing_plan_state"] += 1
            continue
        plans += 1
        oxidation = source_plan.get("oxidation_candidates")
        if isinstance(oxidation, (list, tuple)) and len(oxidation) == len(source_plan.get("elements") or ()):
            if all(value not in (None, "unknown") for value in oxidation):
                source_oxidation_known += 1
        try:
            plan = annotate_plan_with_valence(source_plan, max_species=max_species)
        except Exception as exc:  # noqa: BLE001
            assignment_failures[type(exc).__name__] += 1
            plan = dict(source_plan)
        assignment = plan.get("valence_assignment") or {}
        if assignment.get("assigned") is True:
            valence_assignment_known += 1
            assignment_modes[str(assignment.get("mode") or "unknown")] += 1
            if assignment.get("mode") == "ionic_mixed":
                mixed_valence += 1
        else:
            assignment_failures[str(assignment.get("reason") or "missing_assignment")] += 1
        try:
            text = plan_state_to_countvalencefields(plan)
            serialized += 1
            rebuilt = parse_countvalence_plan_state(text)
            parsed += 1
        except Exception as exc:  # noqa: BLE001
            failures[type(exc).__name__] += 1
            continue
        if canonical_composition(plan) == canonical_composition(rebuilt):
            exact_composition += 1
        else:
            failures["composition_roundtrip_mismatch"] += 1
        source_soft = (
            str(source_plan.get("lattice_system")),
            str(source_plan.get("spacegroup_bucket")),
            str(source_plan.get("volume_per_atom_bin")),
        )
        rebuilt_soft = (
            str(rebuilt.get("lattice_system")),
            str(rebuilt.get("spacegroup_bucket")),
            str(rebuilt.get("volume_per_atom_bin")),
        )
        if source_soft == rebuilt_soft:
            exact_soft_fields += 1
        else:
            failures["soft_field_roundtrip_mismatch"] += 1
        if rebuilt.get("generated_charge_sum_known") is True:
            generated_charge_known += 1
            if int(rebuilt.get("generated_charge_sum") or 0) == 0:
                generated_charge_neutral += 1
        if rebuilt.get("charge_bucket_match") is True:
            charge_bucket_match += 1
        charge_buckets[str(plan.get("charge_bucket") or "unknown")] += 1
        for value in plan.get("valence_species") or ():
            if not isinstance(value, Mapping):
                continue
            element = str(value.get("element") or "unknown")
            count = int(value.get("count") or 0)
            ox = value.get("oxidation_state")
            token = f"{element}|{ox if ox not in (None, 'unknown') else 'unknown'}|{count}"
            pair_vocab[token] += 1
            counts[int(count)] += 1
            oxidation_values[str(ox if ox not in (None, "unknown") else "unknown")] += 1

    def rate(value: int, denominator: int = plans) -> float:
        return 0.0 if denominator == 0 else value / denominator

    return {
        "name": name,
        "path": str(path.resolve()),
        "rows": rows,
        "plans": plans,
        "serialized": serialized,
        "parsed": parsed,
        "rates": {
            "serialization": rate(serialized),
            "parse": rate(parsed),
            "exact_composition_roundtrip": rate(exact_composition),
            "exact_soft_field_roundtrip": rate(exact_soft_fields),
            "source_oxidation_known": rate(source_oxidation_known),
            "valence_assignment_known": rate(valence_assignment_known),
            "mixed_valence": rate(mixed_valence),
            "generated_charge_known": rate(generated_charge_known),
            "generated_charge_neutral": rate(generated_charge_neutral),
            "charge_bucket_match": rate(charge_bucket_match),
        },
        "counts": {
            "exact_composition_roundtrip": exact_composition,
            "exact_soft_field_roundtrip": exact_soft_fields,
            "source_oxidation_known": source_oxidation_known,
            "valence_assignment_known": valence_assignment_known,
            "mixed_valence": mixed_valence,
            "generated_charge_known": generated_charge_known,
            "generated_charge_neutral": generated_charge_neutral,
            "charge_bucket_match": charge_bucket_match,
        },
        "vocabulary": {
            "element_oxidation_count_triples": len(pair_vocab),
            "count_min": min(counts) if counts else None,
            "count_max": max(counts) if counts else None,
            "oxidation_values": dict(sorted(oxidation_values.items())),
        },
        "charge_buckets": dict(sorted(charge_buckets.items())),
        "assignment_modes": dict(sorted(assignment_modes.items())),
        "assignment_failures": dict(sorted(assignment_failures.items())),
        "failures": dict(sorted(failures.items())),
    }


def training_gate(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_name = {str(row["name"]): row for row in results}
    train_like = [by_name[name] for name in ("train", "val") if name in by_name]
    required_train_val_present = len(train_like) == 2
    roundtrip_gate = required_train_val_present and all(
        row["rates"]["serialization"] >= 0.95
        and row["rates"]["parse"] >= 0.95
        and row["rates"]["exact_composition_roundtrip"] >= 0.95
        and row["rates"]["exact_soft_field_roundtrip"] >= 0.95
        for row in train_like
    )
    coverage_gate = required_train_val_present and all(
        row["rates"]["valence_assignment_known"] >= 0.95
        and row["rates"]["generated_charge_neutral"] >= 0.95
        for row in train_like
    )
    train = by_name.get("train")
    raw = by_name.get("raw1000")
    raw_gap_gate = bool(
        train
        and raw
        and raw["rates"]["valence_assignment_known"]
        >= train["rates"]["valence_assignment_known"] - 0.03
        and raw["rates"]["generated_charge_neutral"]
        >= train["rates"]["generated_charge_neutral"] - 0.03
    )
    authorized = bool(roundtrip_gate and coverage_gate and raw_gap_gate)
    return {
        "required_train_val_present": required_train_val_present,
        "train_val_roundtrip_95pct": bool(roundtrip_gate),
        "train_val_valence_and_neutral_95pct": bool(coverage_gate),
        "raw1000_within_3pp_of_train": raw_gap_gate,
        "candidate_training_authorized": authorized,
    }


def write_report(results: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    gate = training_gate(results)
    payload = {
        "schema": "h1a2_planner_countvalence_coverage_audit_v2",
        "datasets": results,
        "gate": gate,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "PLANNER_COUNTVALENCE_AUDIT.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Planner count-valence coverage audit",
        "",
        f"Training authorized: **{gate['candidate_training_authorized']}**",
        "",
        "| Dataset | Plans | Serialize | Parse | Composition RT | Soft-field RT | Source ox. | Assigned | Mixed | Charge neutral |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        rates = row["rates"]
        lines.append(
            f"| {row['name']} | {row['plans']} | {rates['serialization']:.2%} | "
            f"{rates['parse']:.2%} | {rates['exact_composition_roundtrip']:.2%} | "
            f"{rates['exact_soft_field_roundtrip']:.2%} | "
            f"{rates['source_oxidation_known']:.2%} | "
            f"{rates['valence_assignment_known']:.2%} | {rates['mixed_valence']:.2%} | "
            f"{rates['generated_charge_neutral']:.2%} |"
        )
    (output_dir / "PLANNER_COUNTVALENCE_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output_dir / "_SUCCESS").touch()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="append", required=True, help="NAME=PATH; repeatable")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-species", type=int, default=7)
    args = parser.parse_args()
    results = []
    for item in args.dataset:
        if "=" not in item:
            parser.error(f"invalid --dataset {item!r}; expected NAME=PATH")
        name, path = item.split("=", 1)
        results.append(
            audit_dataset(name.strip(), Path(path), max_species=int(args.max_species))
        )
    print(json.dumps(write_report(results, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
