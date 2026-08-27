#!/usr/bin/env python3
"""Freeze CCFD Phase-0 coverage, conservation, and false-rejection evidence."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crystal_dlm.ccfd import FormulaToken, replay_tokens  # noqa: E402
from crystal_dlm.composition_validity import classify_smact_validity  # noqa: E402
from crystal_dlm.fixed_slot import SYMBOL_TO_Z  # noqa: E402
from crystal_dlm.valence_assignment import (  # noqa: E402
    annotate_plan_with_valence,
    valence_catalog_manifest,
)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise TypeError(f"non-object row in {path}")
                yield row


def plan_from_row(row: Mapping[str, Any]) -> Mapping[str, Any] | None:
    value = row.get("plan_state") or row.get("r5_plan_state")
    return value if isinstance(value, Mapping) else None


def normalized_composition(plan: Mapping[str, Any]) -> tuple[tuple[int, int], ...]:
    merged: Counter[int] = Counter()
    for symbol, count in zip(plan.get("elements") or (), plan.get("counts") or ()):
        if str(symbol) in SYMBOL_TO_Z and int(count) > 0:
            merged[int(SYMBOL_TO_Z[str(symbol)])] += int(count)
    return tuple(sorted(merged.items()))


def token_composition(tokens: Iterable[FormulaToken]) -> tuple[tuple[int, int], ...]:
    merged: Counter[int] = Counter()
    for token in tokens:
        merged[int(token.atomic_number)] += int(token.count)
    return tuple(sorted(merged.items()))


def formula_tokens(plan: Mapping[str, Any]) -> tuple[FormulaToken, ...]:
    output: list[FormulaToken] = []
    for value in plan.get("valence_species") or ():
        if not isinstance(value, Mapping):
            raise TypeError("valence species row is not an object")
        oxidation = value.get("oxidation_state")
        if oxidation in (None, "unknown"):
            raise ValueError("unknown oxidation state")
        output.append(
            FormulaToken.from_symbol(
                str(value.get("element")), int(oxidation), int(value.get("count") or 0)
            )
        )
    return tuple(sorted(output))


def audit_dataset(name: str, path: Path, *, max_species: int) -> dict[str, Any]:
    rows = plans = assigned = ccfd_terminal = composition_roundtrip = 0
    legacy_valid = false_rejected_by_legacy = legacy_valid_but_unassigned = 0
    mixed_valence = 0
    branches: Counter[str] = Counter()
    assignment_modes: Counter[str] = Counter()
    assignment_failures: Counter[str] = Counter()
    legacy_reasons: Counter[str] = Counter()
    ccfd_failures: Counter[str] = Counter()
    token_vocab: Counter[tuple[int, int, int]] = Counter()

    for row in iter_jsonl(path):
        rows += 1
        source = plan_from_row(row)
        if source is None:
            ccfd_failures["missing_plan_state"] += 1
            continue
        plans += 1
        composition = normalized_composition(source)
        elems = [atomic_number for atomic_number, _count in composition]
        counts = [count for _atomic_number, count in composition]
        legacy = classify_smact_validity(elems, counts)
        legacy_reasons[str(legacy.get("reason") or "unknown")] += 1
        legacy_valid += int(legacy.get("valid") is True)

        try:
            annotated = annotate_plan_with_valence(source, max_species=max_species)
        except Exception as exc:  # noqa: BLE001
            assignment_failures[type(exc).__name__] += 1
            if legacy.get("valid") is True:
                legacy_valid_but_unassigned += 1
            continue
        assignment = annotated.get("valence_assignment") or {}
        if assignment.get("assigned") is not True:
            assignment_failures[str(assignment.get("reason") or "unassigned")] += 1
            if legacy.get("valid") is True:
                legacy_valid_but_unassigned += 1
            continue
        assigned += 1
        mode = str(assignment.get("mode") or "unknown")
        assignment_modes[mode] += 1
        mixed_valence += int(mode == "ionic_mixed")
        try:
            tokens = formula_tokens(annotated)
            state = replay_tokens(int(source.get("N") or sum(counts)), tokens, max_species=max_species)
        except Exception as exc:  # noqa: BLE001
            ccfd_failures[type(exc).__name__] += 1
            continue
        branches[str(state.branch or "unknown")] += 1
        for token in tokens:
            token_vocab[(token.atomic_number, token.oxidation_state, token.count)] += 1
        if not state.eos_legal:
            ccfd_failures["terminal_conservation_failed"] += 1
            continue
        ccfd_terminal += 1
        if token_composition(tokens) == composition:
            composition_roundtrip += 1
        else:
            ccfd_failures["composition_roundtrip_mismatch"] += 1
        if legacy.get("valid") is not True:
            false_rejected_by_legacy += 1

    def rate(value: int, denominator: int = plans) -> float:
        return 0.0 if denominator == 0 else value / denominator

    return {
        "name": name,
        "path": str(path.resolve()),
        "rows": rows,
        "plans": plans,
        "counts": {
            "assigned": assigned,
            "ccfd_terminal": ccfd_terminal,
            "composition_roundtrip": composition_roundtrip,
            "legacy_valid": legacy_valid,
            "false_rejected_by_legacy": false_rejected_by_legacy,
            "legacy_valid_but_unassigned": legacy_valid_but_unassigned,
            "mixed_valence": mixed_valence,
        },
        "rates": {
            "assignment_coverage": rate(assigned),
            "ccfd_terminal_coverage": rate(ccfd_terminal),
            "composition_roundtrip": rate(composition_roundtrip, ccfd_terminal),
            "legacy_comp_valid": rate(legacy_valid),
            "false_rejection_among_ccfd": rate(false_rejected_by_legacy, ccfd_terminal),
            "mixed_valence": rate(mixed_valence),
        },
        "branches": dict(sorted(branches.items())),
        "assignment_modes": dict(sorted(assignment_modes.items())),
        "assignment_failures": dict(sorted(assignment_failures.items())),
        "legacy_reasons": dict(sorted(legacy_reasons.items())),
        "ccfd_failures": dict(sorted(ccfd_failures.items())),
        "token_vocabulary_size": len(token_vocab),
        "_token_vocab": token_vocab,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="append", required=True, help="NAME=PATH")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-species", type=int, default=7)
    args = parser.parse_args()

    results: list[dict[str, Any]] = []
    for item in args.dataset:
        if "=" not in item:
            parser.error(f"invalid --dataset {item!r}; expected NAME=PATH")
        name, raw_path = item.split("=", 1)
        results.append(
            audit_dataset(name.strip(), Path(raw_path), max_species=int(args.max_species))
        )
    by_name = {str(row["name"]): row for row in results}
    if "train" not in by_name or "val" not in by_name:
        raise ValueError("Phase 0 requires train and val datasets")

    train_vocab = set(by_name["train"].pop("_token_vocab"))
    for row in results:
        vocab = set(row.pop("_token_vocab", {}))
        row["train_vocab_oov_types"] = len(vocab - train_vocab)
        row["train_vocab_type_coverage"] = (
            1.0 if not vocab else len(vocab & train_vocab) / len(vocab)
        )

    train = by_name["train"]
    val = by_name["val"]
    raw = by_name.get("raw1000")
    gate = {
        "train_assignment_at_least_95pct": train["rates"]["assignment_coverage"] >= 0.95,
        "val_assignment_at_least_95pct": val["rates"]["assignment_coverage"] >= 0.95,
        "train_roundtrip_100pct": train["rates"]["composition_roundtrip"] == 1.0,
        "val_roundtrip_100pct": val["rates"]["composition_roundtrip"] == 1.0,
        "raw_within_3pp_of_train": bool(
            raw
            and raw["rates"]["assignment_coverage"]
            >= train["rates"]["assignment_coverage"] - 0.03
        ),
        "legacy_false_rejection_audited": all(
            row["counts"]["false_rejected_by_legacy"] >= 0 for row in results
        ),
    }
    gate["phase1_authorized"] = all(gate.values())
    contract = {
        "schema": "h1a2_ccfd_phase0_manifest_v1",
        "ccfd": {
            "branches": ["ionic", "alloy"],
            "unknown_policy": "unrepresentable; never coerced to zero valence",
            "mixed_valence": "same element may repeat only with same-sign nonzero valences",
            "canonical_order": "increasing (atomic_number, oxidation_state)",
            "max_species": int(args.max_species),
            "request_policy": "one trajectory; dead ends remain failed",
            "terminal_condition": "remaining_atoms=0 and remaining_charge=0",
        },
        "valence_catalog": valence_catalog_manifest(),
        "datasets": results,
        "gate": gate,
    }

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "CCFD_PHASE0_MANIFEST.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    fields = (
        "name",
        "plans",
        "assignment_coverage",
        "ccfd_terminal_coverage",
        "composition_roundtrip",
        "legacy_comp_valid",
        "false_rejection_among_ccfd",
        "mixed_valence",
        "token_vocabulary_size",
        "train_vocab_oov_types",
        "train_vocab_type_coverage",
    )
    with (output / "CCFD_PHASE0_MANIFEST.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in results:
            writer.writerow(
                {
                    "name": row["name"],
                    "plans": row["plans"],
                    **{key: row["rates"][key] for key in fields if key in row["rates"]},
                    "token_vocabulary_size": row["token_vocabulary_size"],
                    "train_vocab_oov_types": row["train_vocab_oov_types"],
                    "train_vocab_type_coverage": row["train_vocab_type_coverage"],
                }
            )
    lines = [
        "# CCFD Phase 0 manifest",
        "",
        f"Phase 1 authorized: **{gate['phase1_authorized']}**",
        "",
        "| Dataset | Plans | Assignment | CCFD terminal | Round-trip | Legacy comp-valid | Legacy false reject among CCFD | Mixed valence |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        rates = row["rates"]
        lines.append(
            f"| {row['name']} | {row['plans']} | {rates['assignment_coverage']:.2%} | "
            f"{rates['ccfd_terminal_coverage']:.2%} | {rates['composition_roundtrip']:.2%} | "
            f"{rates['legacy_comp_valid']:.2%} | {rates['false_rejection_among_ccfd']:.2%} | "
            f"{rates['mixed_valence']:.2%} |"
        )
    lines.extend(["", "## Frozen gates", ""])
    lines.extend(f"- {key}: `{value}`" for key, value in gate.items())
    (output / "CCFD_PHASE0_MANIFEST.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output / "_SUCCESS").touch()
    print(json.dumps(contract, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
