#!/usr/bin/env python3
"""CPU-only audit for the CCFD-v2 semantic composition compiler."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from functools import reduce
import json
from math import gcd
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crystal_dlm.ccfd_v2 import (  # noqa: E402
    CERTIFICATE_BENCHMARK,
    CERTIFICATE_EXTENDED_ONLY,
    compile_plan_actions,
    render_rich_plan,
    replay_actions,
)
from crystal_dlm.composition_validity import classify_smact_validity  # noqa: E402
from crystal_dlm.fixed_slot import SYMBOL_TO_Z  # noqa: E402
from crystal_dlm.r5_plan_body import H1_RICH_PLAN_FORMAT, parse_composition_plan  # noqa: E402


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"non-object row in {path}")
            yield row


def plan_from_row(row: Mapping[str, Any]) -> Mapping[str, Any] | None:
    value = row.get("plan_state") or row.get("r5_plan_state")
    if isinstance(value, Mapping):
        return value
    # Frozen raw Planner cohorts are themselves plan objects.
    if all(key in row for key in ("N", "elements", "counts")):
        return row
    return None


def independent_classification(plan: Mapping[str, Any]) -> dict[str, Any]:
    symbols = [str(value) for value in (plan.get("elements") or ())]
    counts = [int(value) for value in (plan.get("counts") or ())]
    if len(symbols) != len(counts) or not symbols:
        return {"valid": False, "reason": "malformed_plan_composition"}
    if any(symbol not in SYMBOL_TO_Z for symbol in symbols):
        return {"valid": False, "reason": "unsupported_element"}
    divisor = reduce(gcd, counts) if counts else 1
    divisor = max(1, int(divisor))
    reduced = [int(value) // divisor for value in counts]
    return classify_smact_validity(
        [int(SYMBOL_TO_Z[symbol]) for symbol in symbols], reduced
    )


def audit_dataset(name: str, path: Path, *, max_atoms: int, max_species: int) -> dict[str, Any]:
    rows = plans = legacy_valid = compiled = ended = 0
    n_conserved = charge_conserved = composition_roundtrip = rich_roundtrip = 0
    benchmark_certificates = extended_only = benchmark_unknown = invalid_certificates = 0
    certificate_agreement = extended_not_positive = benchmark_legacy_recall = 0
    failures: Counter[str] = Counter()
    assignment_sources: Counter[str] = Counter()
    certificate_classes: Counter[str] = Counter()
    benchmark_reasons: Counter[str] = Counter()
    branches: Counter[str] = Counter()
    n_distribution: Counter[int] = Counter()

    for row in iter_jsonl(path):
        rows += 1
        plan = plan_from_row(row)
        if plan is None:
            failures["missing_plan_state"] += 1
            continue
        plans += 1
        independent = independent_classification(plan)
        independent_valid = independent.get("valid") is True
        legacy_valid += int(independent_valid)
        benchmark_reasons[str(independent.get("reason") or "unknown")] += 1
        try:
            actions, metadata = compile_plan_actions(
                plan,
                max_atoms=max_atoms,
                max_species=max_species,
            )
            compiled += 1
            assignment_sources[str(metadata.get("assignment_source") or "unknown")] += 1
            state = replay_actions(actions, max_atoms=max_atoms, max_species=max_species)
        except Exception as exc:  # noqa: BLE001 - every failure is counted, never repaired.
            failures[f"compile:{type(exc).__name__}:{str(exc)}"] += 1
            continue

        ended += int(state.ended)
        source_n = int(plan.get("N") or 0)
        n_distribution[source_n] += 1
        n_conserved += int(
            state.target_atoms == source_n
            and state.emitted_atoms == source_n
            and state.remaining_atoms == 0
        )
        charge_conserved += int(state.net_charge == 0)
        branches[str(state.branch or "unknown")] += 1
        certificate = state.certificate()
        certificate_classes[certificate.certificate_class] += 1
        benchmark_certificates += int(certificate.certificate_class == CERTIFICATE_BENCHMARK)
        extended_only += int(certificate.certificate_class == CERTIFICATE_EXTENDED_ONLY)
        benchmark_unknown += int(certificate.certificate_class == "benchmark_unknown")
        invalid_certificates += int(
            certificate.certificate_class not in {CERTIFICATE_BENCHMARK, CERTIFICATE_EXTENDED_ONLY}
        )
        certificate_agreement += int(
            certificate.benchmark_compatible == independent_valid
        )
        extended_not_positive += int(
            certificate.certificate_class != CERTIFICATE_EXTENDED_ONLY
            or not certificate.usable_as_positive
        )
        benchmark_legacy_recall += int(independent_valid and certificate.benchmark_compatible)

        expected = tuple(
            sorted(
                (int(SYMBOL_TO_Z[str(symbol)]), int(count))
                for symbol, count in zip(plan.get("elements") or (), plan.get("counts") or ())
            )
        )
        observed = tuple(zip(certificate.elements, certificate.counts))
        composition_roundtrip += int(expected == observed)
        if expected != observed:
            failures["semantic_composition_roundtrip_mismatch"] += 1

        try:
            rich = render_rich_plan(state, soft_fields=plan)
            rebuilt = parse_composition_plan(rich, plan_style=H1_RICH_PLAN_FORMAT)
            rebuilt_composition = tuple(
                sorted(
                    (int(SYMBOL_TO_Z[str(symbol)]), int(count))
                    for symbol, count in zip(rebuilt["elements"], rebuilt["counts"])
                )
            )
            rich_roundtrip += int(
                int(rebuilt["N"]) == source_n and rebuilt_composition == expected
            )
        except Exception as exc:  # noqa: BLE001
            failures[f"rich_render:{type(exc).__name__}:{str(exc)}"] += 1

    def rate(value: int, denominator: int = plans) -> float:
        return 0.0 if denominator == 0 else float(value) / float(denominator)

    return {
        "name": name,
        "path": str(path.resolve()),
        "rows": rows,
        "plans": plans,
        "counts": {
            "legacy_valid": legacy_valid,
            "compiled": compiled,
            "ended": ended,
            "n_conserved": n_conserved,
            "charge_conserved": charge_conserved,
            "composition_roundtrip": composition_roundtrip,
            "rich_roundtrip": rich_roundtrip,
            "benchmark_certificates": benchmark_certificates,
            "extended_only": extended_only,
            "benchmark_unknown": benchmark_unknown,
            "invalid_certificates": invalid_certificates,
            "certificate_agreement": certificate_agreement,
            "extended_not_positive": extended_not_positive,
            "benchmark_legacy_recall": benchmark_legacy_recall,
        },
        "rates": {
            "legacy_comp_valid": rate(legacy_valid),
            "semantic_compile": rate(compiled),
            "N_conservation": rate(n_conserved, compiled),
            "charge_conservation": rate(charge_conserved, compiled),
            "composition_roundtrip": rate(composition_roundtrip, compiled),
            "rich_roundtrip": rate(rich_roundtrip, compiled),
            "benchmark_certificate": rate(benchmark_certificates),
            "extended_only_unknown": rate(extended_only),
            "certificate_agreement": rate(certificate_agreement, compiled),
            "extended_never_positive": rate(extended_not_positive, compiled),
            "benchmark_recall_among_legacy_valid": rate(
                benchmark_legacy_recall, legacy_valid
            ),
        },
        "assignment_sources": dict(sorted(assignment_sources.items())),
        "certificate_classes": dict(sorted(certificate_classes.items())),
        "benchmark_reasons": dict(sorted(benchmark_reasons.items())),
        "branches": dict(sorted(branches.items())),
        "N": {str(key): value for key, value in sorted(n_distribution.items())},
        "failures": dict(failures.most_common()),
    }


def build_gate(results: list[dict[str, Any]]) -> dict[str, bool]:
    by_name = {str(row["name"]): row for row in results}
    train = by_name.get("train")
    val = by_name.get("val")
    raw = by_name.get("raw1000")
    required = train is not None and val is not None
    strict_rows = [row for row in (train, val) if row is not None]
    invariant_gate = required and all(
        row["rates"]["N_conservation"] == 1.0
        and row["rates"]["charge_conservation"] == 1.0
        and row["rates"]["composition_roundtrip"] == 1.0
        and row["rates"]["rich_roundtrip"] == 1.0
        for row in strict_rows
    )
    certificate_gate = required and all(
        row["rates"]["certificate_agreement"] == 1.0
        and row["rates"]["extended_never_positive"] == 1.0
        and row["rates"]["benchmark_recall_among_legacy_valid"] == 1.0
        for row in strict_rows
    )
    coverage_gate = required and all(
        row["rates"]["semantic_compile"] >= row["rates"]["legacy_comp_valid"]
        for row in strict_rows
    )
    raw_gate = bool(
        train
        and raw
        and raw["rates"]["semantic_compile"]
        >= train["rates"]["semantic_compile"] - 0.03
    )
    gate = {
        "train_val_present": required,
        "train_val_exact_invariants": bool(invariant_gate),
        "dual_certificate_alignment": bool(certificate_gate),
        "semantic_coverage_at_least_legacy": bool(coverage_gate),
        "raw1000_within_3pp_of_train": raw_gate,
    }
    gate["semantic_compiler_authorized"] = all(gate.values())
    # A successful CPU compiler audit authorizes only tokenizer/interface work,
    # never a Planner GPU claim by itself.
    gate["planner_gpu_training_authorized"] = False
    return gate


def write_report(results: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    gate = build_gate(results)
    payload = {
        "schema": "h1a2_ccfd_v2_semantic_audit_v1",
        "claim_boundary": "composition compiler only; contribution1 and stability claims unchanged",
        "contract": {
            "action_order": "N then canonical element-valence-count actions then explicit end",
            "N_policy": "hard locked; no repair or replacement",
            "terminal": "remaining_atoms=0 and net_charge=0",
            "certificates": {
                "benchmark_compatible": "independent CrysLLMGen/SMACT pass; positive allowed",
                "extended_only": "exact broader SMACT-state ledger but independent failure; unknown, never positive",
            },
            "rich_plan": "render unchanged seven-line h1_rich_plan_v1 after semantic completion",
        },
        "datasets": results,
        "gate": gate,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    stem = "CCFD_V2_SEMANTIC_AUDIT"
    (output_dir / f"{stem}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    fields = [
        "name",
        "plans",
        "legacy_comp_valid",
        "semantic_compile",
        "N_conservation",
        "charge_conservation",
        "composition_roundtrip",
        "rich_roundtrip",
        "benchmark_certificate",
        "extended_only_unknown",
        "certificate_agreement",
        "benchmark_recall_among_legacy_valid",
    ]
    with (output_dir / f"{stem}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in results:
            writer.writerow(
                {
                    "name": row["name"],
                    "plans": row["plans"],
                    **{key: row["rates"][key] for key in fields if key in row["rates"]},
                }
            )
    lines = [
        "# CCFD-v2 semantic compiler audit",
        "",
        f"Semantic compiler authorized: **{gate['semantic_compiler_authorized']}**",
        "",
        "| Dataset | Plans | Legacy valid | Semantic compile | N/charge | Composition RT | Rich RT | Benchmark cert | Extended-only | Cert agreement |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        rates = row["rates"]
        lines.append(
            f"| {row['name']} | {row['plans']} | {rates['legacy_comp_valid']:.2%} | "
            f"{rates['semantic_compile']:.2%} | "
            f"{rates['N_conservation']:.2%}/{rates['charge_conservation']:.2%} | "
            f"{rates['composition_roundtrip']:.2%} | {rates['rich_roundtrip']:.2%} | "
            f"{rates['benchmark_certificate']:.2%} | {rates['extended_only_unknown']:.2%} | "
            f"{rates['certificate_agreement']:.2%} |"
        )
    lines.extend(["", "## Frozen gates", ""])
    lines.extend(f"- {key}: `{value}`" for key, value in gate.items())
    lines.extend(
        [
            "",
            "`extended_only` rows remain unknown and cannot count as comp-valid. "
            "This audit does not authorize Planner GPU training by itself.",
        ]
    )
    (output_dir / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output_dir / "_SUCCESS").touch()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="append", required=True, help="NAME=PATH")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-atoms", type=int, default=20)
    parser.add_argument("--max-species", type=int, default=7)
    args = parser.parse_args()
    results = []
    for item in args.dataset:
        if "=" not in item:
            parser.error(f"invalid --dataset {item!r}; expected NAME=PATH")
        name, raw_path = item.split("=", 1)
        results.append(
            audit_dataset(
                name.strip(),
                Path(raw_path),
                max_atoms=int(args.max_atoms),
                max_species=int(args.max_species),
            )
        )
    payload = write_report(results, args.output_dir)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
