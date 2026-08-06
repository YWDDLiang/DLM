#!/usr/bin/env python3
"""Evaluate one frozen H1-A2 V3 Planner arm on 512 raw attempts."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from crystal_dlm.composition_validity import (
    classify_smact_validity,
    reduced_composition,
)
from crystal_dlm.fixed_slot import SYMBOL_TO_Z
from scripts.analyze_r5c_plan_distribution import (
    compare_stats,
    extract_generated_plan,
    load_generated,
    load_teacher,
    summarize,
)


DENOMINATOR = 512
NORMALIZATION = "gcd_reduced_atom_multiset_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path} contains a non-object row")
                yield value


def normalized_plan_composition(
    symbols: Sequence[str],
    counts: Sequence[int],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if len(symbols) != len(counts) or not symbols:
        raise ValueError("elements/counts must be non-empty and aligned")
    atom_types: list[int] = []
    for symbol, raw_count in zip(symbols, counts):
        count = int(raw_count)
        if count <= 0:
            raise ValueError("composition counts must be positive")
        atom_types.extend([int(SYMBOL_TO_Z[str(symbol)])] * count)
    return reduced_composition(atom_types)


def audit_attempts(path: Path) -> dict[str, Any]:
    rows = list(iter_jsonl(path))
    ordinals = [int(row.get("sample_idx", -1)) for row in rows]
    if len(rows) != DENOMINATOR or ordinals != list(range(DENOMINATOR)):
        raise ValueError(
            "Planner raw all-attempt ledger must be exactly ordered ordinals 0..511"
        )

    reason_counts: Counter[str] = Counter()
    formula_counts: Counter[str] = Counter()
    parse_count = 0
    completion_count = 0
    comp_valid_count = 0
    all_metal_count = 0
    single_element_count = 0
    n_sum = 0
    failures: list[dict[str, Any]] = []

    for row in rows:
        sample_idx = int(row["sample_idx"])
        if row.get("plan_end_marker_present") is True:
            completion_count += 1
        plan, failure = extract_generated_plan(row)
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
        elements, reduced_counts = normalized_plan_composition(symbols, counts)
        classification = dict(classify_smact_validity(elements, reduced_counts))
        reason = str(classification.get("reason", "unknown"))
        reason_counts[reason] += 1
        if classification.get("valid") is True:
            comp_valid_count += 1
        elif len(failures) < 20:
            failures.append(
                {
                    "sample_idx": sample_idx,
                    "stage": "composition",
                    "formula": str(plan["formula"]),
                    "reason": reason,
                }
            )
        if reason == "all_metal_shortcut":
            all_metal_count += 1
        if reason == "single_element_shortcut":
            single_element_count += 1
        formula_counts[str(plan["formula"])] += 1
        n_sum += int(plan["N"])

    return {
        "denominator": DENOMINATOR,
        "composition_count_normalization": NORMALIZATION,
        "parse_count": parse_count,
        "parse_rate": parse_count / DENOMINATOR,
        "completion_count": completion_count,
        "completion_rate": completion_count / DENOMINATOR,
        "composition_valid_count": comp_valid_count,
        "composition_valid_rate": comp_valid_count / DENOMINATOR,
        "unique_formula_count": len(formula_counts),
        "unique_formula_rate": len(formula_counts) / DENOMINATOR,
        "mean_N": n_sum / max(1, parse_count),
        "all_metal_count": all_metal_count,
        "all_metal_rate": all_metal_count / DENOMINATOR,
        "single_element_count": single_element_count,
        "single_element_rate": single_element_count / DENOMINATOR,
        "reason_counts": dict(sorted(reason_counts.items())),
        "failure_examples": failures,
    }


def validate_sampler_config(config: Mapping[str, Any]) -> None:
    expected = {
        "num_samples": DENOMINATOR,
        "max_new_tokens": 96,
        "temperature": 0.9,
        "top_p": 0.95,
        "top_k": 50,
        "max_atoms": 20,
        "prompt_style": "h1_rich_plan_v1",
        "include_sample_id": False,
        "seed": 17029,
        "seed_mode": "stateless_ordinal_v1",
        "rank_independent_sampling": True,
        "effective_generation_batch_size": 1,
    }
    mismatches = {
        key: {"expected": value, "observed": config.get(key)}
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise ValueError(f"frozen Planner sampler config changed: {mismatches}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("P0", "P-control", "P-star"), required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--raw-generations", type=Path, required=True)
    parser.add_argument("--sample-metrics", type=Path, required=True)
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument("--teacher-jsonl", type=Path, required=True)
    parser.add_argument("--execution-manifest-sha256", required=True)
    parser.add_argument("--checkpoint-identity-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sample_metrics = json.loads(args.sample_metrics.read_text(encoding="utf-8"))
    if int(sample_metrics.get("decoded_samples", -1)) != DENOMINATOR:
        raise ValueError("decoded sample count is not the frozen 512 denominator")
    run_config = json.loads(args.run_config.read_text(encoding="utf-8"))
    validate_sampler_config(run_config)

    audit = audit_attempts(args.raw_generations)
    teacher_raw = load_teacher(args.teacher_jsonl, dedupe=True)
    generated_raw = load_generated(args.raw_generations)
    report = {
        "schema": "h1a2_v3_planner512_arm_report_v1",
        "status": "complete",
        "arm": args.arm,
        "step": int(args.step),
        "denominator": DENOMINATOR,
        "checkpoint_identity_sha256": args.checkpoint_identity_sha256,
        "execution_manifest_sha256": args.execution_manifest_sha256,
        "raw_generations_sha256": sha256_file(args.raw_generations),
        "sample_metrics_sha256": sha256_file(args.sample_metrics),
        "run_config_sha256": sha256_file(args.run_config),
        "teacher_jsonl_sha256": sha256_file(args.teacher_jsonl),
        "attempt_audit": audit,
        "generated_distribution": summarize(generated_raw, top_k=30),
        "teacher_distribution": summarize(teacher_raw, top_k=30),
        "distribution_comparison": compare_stats(teacher_raw, generated_raw),
        "model_proposed_plan": True,
        "sample_id_used": False,
        "retry_replacement_repair_filter_or_rerank_used": False,
        "generation_sun_energy_or_hull_used_for_selection": False,
        "automatic_promotion": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
