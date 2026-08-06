#!/usr/bin/env python3
"""Plan-only JointChem evaluation with historical GCD-reduced SMACT semantics."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any, Sequence


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from evaluate_jointchem_plans import (  # noqa: E402
    compare_stats,
    extract_generated_plan,
    iter_jsonl,
    load_generated,
    load_teacher,
    sha256_file,
    summarize,
)
from crystal_dlm.composition_validity import (  # noqa: E402
    classify_smact_validity,
    reduced_composition,
)
from crystal_dlm.fixed_slot import SYMBOL_TO_Z  # noqa: E402
from crystal_dlm.r5_plan_state import (  # noqa: E402
    anion_framework_from_symbols,
    charge_bucket_from_classification,
)


NORMALIZATION = "gcd_reduced_atom_multiset_v1"


def normalized_plan_composition(
    symbols: Sequence[str],
    counts: Sequence[int],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Convert a unit-cell formula to the reduced composition used historically."""

    if len(symbols) != len(counts) or not symbols:
        raise ValueError("elements/counts must be non-empty and have equal lengths")
    atom_types: list[int] = []
    for symbol, raw_count in zip(symbols, counts):
        count = int(raw_count)
        if count <= 0:
            raise ValueError(f"composition count must be positive, observed {count}")
        atom_types.extend([int(SYMBOL_TO_Z[str(symbol)])] * count)
    return reduced_composition(atom_types)


def composition_audit(path: Path, *, denominator: int) -> dict[str, Any]:
    reason_counts: Counter[str] = Counter()
    parse_count = 0
    comp_valid_count = 0
    anion_match_count = 0
    charge_match_count = 0
    all_metal_count = 0
    single_element_count = 0
    seen_sample_ids: set[int] = set()
    failures: list[dict[str, Any]] = []

    for row_number, record in enumerate(iter_jsonl(path)):
        sample_idx = int(record.get("sample_idx", row_number))
        if sample_idx in seen_sample_ids:
            raise ValueError(f"duplicate sample_idx={sample_idx}")
        seen_sample_ids.add(sample_idx)
        plan, failure = extract_generated_plan(record)
        if plan is None:
            reason_counts["parse_failure"] += 1
            if len(failures) < 20:
                failures.append(
                    {
                        "sample_idx": sample_idx,
                        "stage": "parse",
                        "reason": failure,
                    }
                )
            continue
        parse_count += 1
        symbols = [str(value) for value in plan["elements"]]
        counts = [int(value) for value in plan["counts"]]
        elems, reduced_counts = normalized_plan_composition(symbols, counts)
        classification = dict(classify_smact_validity(elems, reduced_counts))
        reason = str(classification.get("reason", "unknown"))
        reason_counts[reason] += 1
        if classification.get("valid") is True:
            comp_valid_count += 1
        elif len(failures) < 20:
            failures.append(
                {
                    "sample_idx": sample_idx,
                    "stage": "composition",
                    "formula": plan["formula"],
                    "reason": reason,
                }
            )
        if reason == "all_metal_shortcut":
            all_metal_count += 1
        if reason == "single_element_shortcut":
            single_element_count += 1
        if str(plan.get("anion_framework")) == anion_framework_from_symbols(symbols):
            anion_match_count += 1
        if str(plan.get("charge_bucket")) == charge_bucket_from_classification(classification):
            charge_match_count += 1

    if len(seen_sample_ids) != int(denominator):
        raise ValueError(
            f"all-attempt denominator mismatch: expected {denominator}, "
            f"observed {len(seen_sample_ids)}"
        )
    expected_ids = set(range(int(denominator)))
    if seen_sample_ids != expected_ids:
        missing = sorted(expected_ids - seen_sample_ids)[:20]
        extra = sorted(seen_sample_ids - expected_ids)[:20]
        raise ValueError(f"sample_idx ledger mismatch; missing={missing}, extra={extra}")
    return {
        "denominator": int(denominator),
        "count_normalization": NORMALIZATION,
        "parse_count": parse_count,
        "parse_rate": parse_count / int(denominator),
        "composition_valid_count": comp_valid_count,
        "composition_valid_rate": comp_valid_count / int(denominator),
        "anion_match_count": anion_match_count,
        "anion_match_rate": anion_match_count / int(denominator),
        "charge_match_count": charge_match_count,
        "charge_match_rate": charge_match_count / int(denominator),
        "all_metal_count": all_metal_count,
        "all_metal_rate": all_metal_count / int(denominator),
        "single_element_count": single_element_count,
        "single_element_rate": single_element_count / int(denominator),
        "reason_counts": dict(sorted(reason_counts.items())),
        "failure_examples": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--identity", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--raw-generations", type=Path, required=True)
    parser.add_argument("--teacher-jsonl", type=Path, required=True)
    parser.add_argument("--sample-metrics", type=Path, required=True)
    parser.add_argument("--denominator", type=int, default=512)
    parser.add_argument("--execution-manifest-sha256", required=True)
    parser.add_argument("--checkpoint-identity-sha256", required=True)
    parser.add_argument("--initial-adapter-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if int(args.denominator) != 512:
        raise ValueError("JointChem Plan-only arm screen freezes denominator=512")
    sample_metrics = json.loads(args.sample_metrics.read_text(encoding="utf-8"))
    decoded = int(sample_metrics.get("decoded_samples", -1))
    if decoded != int(args.denominator):
        raise ValueError(f"decoded_samples mismatch: expected {args.denominator}, observed {decoded}")

    audit = composition_audit(args.raw_generations, denominator=int(args.denominator))
    teacher_raw = load_teacher(args.teacher_jsonl, dedupe=True)
    generated_raw = load_generated(args.raw_generations)
    report = {
        "schema": "h1a2_jointchem_plan_report_v2",
        "corrects_schema": "h1a2_jointchem_plan_report_v1",
        "identity": args.identity,
        "arm": args.arm,
        "step": int(args.step),
        "denominator": int(args.denominator),
        "composition_count_normalization": NORMALIZATION,
        "execution_manifest_sha256": str(args.execution_manifest_sha256),
        "checkpoint_identity_sha256": str(args.checkpoint_identity_sha256),
        "initial_adapter_sha256": str(args.initial_adapter_sha256),
        "raw_generations_sha256": sha256_file(args.raw_generations),
        "sample_metrics_sha256": sha256_file(args.sample_metrics),
        "teacher_jsonl_sha256": sha256_file(args.teacher_jsonl),
        "composition": audit,
        "generated_distribution": summarize(generated_raw, top_k=30),
        "teacher_distribution": summarize(teacher_raw, top_k=30),
        "distribution_comparison": compare_stats(teacher_raw, generated_raw),
        "energy_or_stability_label_used": False,
        "sun_or_mlip_used": False,
        "retry_or_replacement_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
