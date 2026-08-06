#!/usr/bin/env python3
"""Exact-tokenizer CR-Plan V2 policy, support-parity, and CPU micro audit."""

from __future__ import annotations

import argparse
from collections import Counter
from functools import reduce
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Iterable

from crystal_dlm.composition_validity import classify_smact_validity
from crystal_dlm.fixed_slot import SYMBOL_TO_Z
from crystal_dlm.h1_crplan import (
    CRPLAN_SCHEMA,
    CRPlanTokenVocabulary,
    FormulaGrammarError,
    OxidationReachability,
    PlanFormulaCursor,
    TerminalChargeError,
    load_frozen_smact_table,
)


EXPECTED_MISSING = ("He", "Ne", "Ar", "Pm", "At", "Rn", "Fr", "Ra")
CURSOR_FIXTURES = (
    ("terminal_only", ""),
    ("terminal_only", "for"),
    ("terminal_only", "formula: "),
    ("terminal_only", "formula: Fe2"),
    ("terminal_only", "formula: Fe2O3"),
    ("full_prefix", "formula: "),
    ("full_prefix", "formula: Fe"),
    ("full_prefix", "formula: Fe2"),
    ("full_prefix", "formula: Fe2O"),
    ("full_prefix", "formula: Fe2O3"),
    ("full_prefix", "formula: NaCl"),
    ("grammar_only", "formula: Na"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def direct_counts(counts: Iterable[int]) -> tuple[int, ...]:
    values = tuple(int(value) for value in counts)
    divisor = reduce(math.gcd, values)
    return tuple(int(value // max(1, divisor)) for value in values)


def clone_reachability(base: OxidationReachability) -> OxidationReachability:
    return OxidationReachability(
        base.oxidation_states,
        metals=base.metals,
        max_atoms=base.max_atoms,
        table_source=base.table_source,
        table_version=base.table_version,
        missing_state_policy=base.missing_state_policy,
    )


def terminal_certificate_digest(
    vocabulary: CRPlanTokenVocabulary,
    cursor: PlanFormulaCursor,
    token_ids: Iterable[int],
) -> tuple[int, str]:
    rows: dict[str, dict[str, Any]] = {}
    for token_id in token_ids:
        updated = cursor.feed(vocabulary.fragments[int(token_id)])
        if (
            updated.phase != "after_formula"
            or updated.value is None
            or updated.value.certificate is None
        ):
            raise AssertionError(
                f"terminal token {token_id} did not produce a certificate"
            )
        rows[str(int(token_id))] = updated.value.certificate.to_dict()
    payload = json.dumps(
        rows,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return len(rows), hashlib.sha256(payload).hexdigest()


def support_row(
    vocabulary: CRPlanTokenVocabulary,
    base: OxidationReachability,
    *,
    mode: str,
    text: str,
) -> dict[str, Any]:
    trie_reachability = clone_reachability(base)
    scalar_reachability = clone_reachability(base)
    trie_cursor = PlanFormulaCursor.from_text(
        text,
        mode=mode,
        reachability=trie_reachability,
    )
    scalar_cursor = PlanFormulaCursor.from_text(
        text,
        mode=mode,
        reachability=scalar_reachability,
    )
    trie_before = trie_reachability.diagnostics.snapshot()
    started = time.perf_counter()
    trie_support = vocabulary.support(trie_cursor)
    trie_seconds = time.perf_counter() - started
    trie_after = trie_reachability.diagnostics.snapshot()
    scalar_before = scalar_reachability.diagnostics.snapshot()
    started = time.perf_counter()
    scalar_support = vocabulary.support_scalar_reference(scalar_cursor)
    scalar_seconds = time.perf_counter() - started
    scalar_after = scalar_reachability.diagnostics.snapshot()
    trie_certificate_count, trie_certificate_sha256 = (
        terminal_certificate_digest(
            vocabulary,
            trie_cursor,
            trie_support.terminal_token_ids,
        )
    )
    scalar_certificate_count, scalar_certificate_sha256 = (
        terminal_certificate_digest(
            vocabulary,
            scalar_cursor,
            scalar_support.terminal_token_ids,
        )
    )
    fields_equal = {
        "token_ids": trie_support.token_ids == scalar_support.token_ids,
        "terminal_token_ids": (
            trie_support.terminal_token_ids
            == scalar_support.terminal_token_ids
        ),
        "rejection_counts": (
            trie_support.rejection_counts
            == scalar_support.rejection_counts
        ),
        "terminal_certificates": (
            trie_certificate_count == scalar_certificate_count
            and trie_certificate_sha256 == scalar_certificate_sha256
        ),
    }
    return {
        "mode": mode,
        "text": text,
        "cursor_signature_equal": (
            trie_cursor.signature() == scalar_cursor.signature()
        ),
        "fields_equal": fields_equal,
        "all_equal": all(fields_equal.values()),
        "legal_support_size": len(trie_support.token_ids),
        "terminal_support_size": len(trie_support.terminal_token_ids),
        "terminal_certificate_count": trie_certificate_count,
        "terminal_certificate_sha256": trie_certificate_sha256,
        "rejection_counts": trie_support.rejection_dict(),
        "trie_seconds": trie_seconds,
        "scalar_seconds": scalar_seconds,
        "speedup": (
            scalar_seconds / trie_seconds
            if trie_seconds > 0
            else None
        ),
        "trie_dp_delta": {
            key: int(trie_after[key]) - int(trie_before[key])
            for key in trie_before
        },
        "scalar_dp_delta": {
            key: int(scalar_after[key]) - int(scalar_before[key])
            for key in scalar_before
        },
    }


def missing_policy_rows(
    reachability: OxidationReachability,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    missing = tuple(
        reachability.table_report()["elements_without_states"]
    )
    for symbol in missing:
        nonshortcut_symbols = ("O", symbol)
        nonshortcut_counts = (1, 1)
        direct = dict(
            classify_smact_validity(
                tuple(SYMBOL_TO_Z[value] for value in nonshortcut_symbols),
                direct_counts(nonshortcut_counts),
            )
        )
        certificate = reachability.terminal_certificate(
            zip(nonshortcut_symbols, nonshortcut_counts)
        )
        unary_direct = dict(
            classify_smact_validity(
                (SYMBOL_TO_Z[symbol],),
                (1,),
            )
        )
        unary_certificate = reachability.terminal_certificate({symbol: 1})
        rows.append(
            {
                "element": symbol,
                "nonshortcut_direct": direct,
                "nonshortcut_certificate": certificate.to_dict(),
                "nonshortcut_aligned": (
                    direct.get("valid") is False
                    and direct.get("reason") == "oxidation_state_missing"
                    and certificate.terminal_allowed is False
                    and certificate.stratum
                    == "charge_applicable_oxidation_state_missing"
                ),
                "unary_direct": unary_direct,
                "unary_certificate": unary_certificate.to_dict(),
                "unary_shortcut_aligned": (
                    unary_direct.get("valid") is True
                    and unary_direct.get("reason") == "single_element_shortcut"
                    and unary_certificate.terminal_allowed is True
                    and unary_certificate.stratum
                    == "charge_not_applicable_unary"
                ),
            }
        )
    if missing != EXPECTED_MISSING:
        rows.append(
            {
                "element": "__missing_set_mismatch__",
                "observed": list(missing),
                "expected": list(EXPECTED_MISSING),
                "nonshortcut_aligned": False,
                "unary_shortcut_aligned": False,
            }
        )
    return rows


def all_metal_rows(
    reachability: OxidationReachability,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for symbols in (("Fe", "Cu"), ("Na", "Fr"), ("Ba", "Ra")):
        direct = dict(
            classify_smact_validity(
                tuple(SYMBOL_TO_Z[value] for value in symbols),
                (1, 1),
            )
        )
        certificate = reachability.terminal_certificate(
            {value: 1 for value in symbols}
        )
        rows.append(
            {
                "symbols": list(symbols),
                "direct": direct,
                "certificate": certificate.to_dict(),
                "aligned": (
                    direct.get("valid") is True
                    and direct.get("reason") == "all_metal_shortcut"
                    and certificate.terminal_allowed is True
                    and certificate.stratum
                    == "charge_not_applicable_all_metal"
                ),
            }
        )
    return rows


def missing_vs_all_metal_precedence_rows(
    reachability: OxidationReachability,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for symbols in (("Fe", "Pm"),):
        direct = dict(
            classify_smact_validity(
                tuple(SYMBOL_TO_Z[value] for value in symbols),
                (1, 1),
            )
        )
        certificate = reachability.terminal_certificate(
            {value: 1 for value in symbols}
        )
        rows.append(
            {
                "symbols": list(symbols),
                "direct": direct,
                "certificate": certificate.to_dict(),
                "aligned": (
                    direct.get("valid") is False
                    and direct.get("reason") == "oxidation_state_missing"
                    and certificate.terminal_allowed is False
                    and certificate.stratum
                    == "charge_applicable_oxidation_state_missing"
                ),
                "interpretation": (
                    "Pm is absent from the frozen Direct metal set, so "
                    "Fe-Pm is not an all-metal shortcut."
                ),
            }
        )
    return rows


def endpoint_rows(
    reachability: OxidationReachability,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for formula in (
        "Fe2O3",
        "NaCl",
        "Fe3O4",
        "NaO",
        "OHe",
        "He",
        "FePm",
        "NaFr",
        "BaRa",
    ):
        row: dict[str, Any] = {"formula": formula}
        for mode in ("terminal_only", "full_prefix"):
            try:
                cursor = PlanFormulaCursor.from_text(
                    f"formula: {formula}\n",
                    mode=mode,
                    reachability=reachability,
                )
                row[mode] = {
                    "accepted": True,
                    "certificate": cursor.value.certificate.to_dict(),
                }
            except (FormulaGrammarError, TerminalChargeError) as exc:
                row[mode] = {
                    "accepted": False,
                    "error": type(exc).__name__,
                    "message": str(exc),
                }
        terminal_certificate = reachability.terminal_certificate(
            _parse_formula_counts(formula)
        )
        row["recomputed_certificate"] = terminal_certificate.to_dict()
        row["endpoint_semantics_consistent"] = (
            row["terminal_only"]["accepted"]
            == terminal_certificate.terminal_allowed
            and (
                not row["terminal_only"]["accepted"]
                or row["terminal_only"]["certificate"]
                == terminal_certificate.to_dict()
            )
            and row["full_prefix"]["accepted"]
            == terminal_certificate.terminal_allowed
            and (
                not row["full_prefix"]["accepted"]
                or row["full_prefix"]["certificate"]
                == terminal_certificate.to_dict()
            )
        )
        rows.append(row)
    return rows


def _parse_formula_counts(formula: str) -> dict[str, int]:
    import re

    counts: Counter[str] = Counter()
    for symbol, digits in re.findall(r"([A-Z][a-z]?)(\d*)", formula):
        counts[symbol] += int(digits) if digits else 1
    return dict(counts)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.tokenizer),
        local_files_only=True,
    )
    tokenizer_seconds = time.perf_counter() - started
    reachability = load_frozen_smact_table(
        max_atoms=20,
        missing_state_policy="fail_closed",
    )
    started = time.perf_counter()
    vocabulary = CRPlanTokenVocabulary.from_tokenizer(tokenizer)
    vocabulary_seconds = time.perf_counter() - started

    support_rows = [
        support_row(
            vocabulary,
            reachability,
            mode=mode,
            text=text,
        )
        for mode, text in CURSOR_FIXTURES
    ]
    missing_rows = missing_policy_rows(reachability)
    alloy_rows = all_metal_rows(reachability)
    precedence_rows = missing_vs_all_metal_precedence_rows(reachability)
    endpoints = endpoint_rows(reachability)
    failures: list[str] = []
    if not all(row["all_equal"] for row in support_rows):
        failures.append("trie_scalar_support_parity")
    if not all(
        row["nonshortcut_aligned"] and row["unary_shortcut_aligned"]
        for row in missing_rows
    ):
        failures.append("missing_state_direct_alignment")
    if not all(row["aligned"] for row in alloy_rows):
        failures.append("all_metal_direct_alignment")
    if not all(row["aligned"] for row in precedence_rows):
        failures.append("missing_vs_all_metal_precedence_alignment")
    if not all(row["endpoint_semantics_consistent"] for row in endpoints):
        failures.append("terminal_full_endpoint_consistency")
    report = {
        "schema": "h1_crplan_fourarm512_exact_tokenizer_preflight_repair_v2",
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "crplan_schema": CRPLAN_SCHEMA,
        "source_manifest_sha256": sha256_file(args.source_manifest),
        "tokenizer_path": str(args.tokenizer),
        "tokenizer_seconds": tokenizer_seconds,
        "vocab_size": len(tokenizer),
        "token_fragment_sha256": vocabulary.fragment_sha256,
        "vocabulary_build_seconds": vocabulary_seconds,
        "oxidation_table": reachability.table_report(),
        "support_rows": support_rows,
        "support_parity_all": all(row["all_equal"] for row in support_rows),
        "support_trie_seconds_total": sum(
            row["trie_seconds"] for row in support_rows
        ),
        "support_scalar_seconds_total": sum(
            row["scalar_seconds"] for row in support_rows
        ),
        "support_trie_dp_states_max": max(
            row["trie_dp_delta"]["states_created"]
            for row in support_rows
        ),
        "missing_policy_rows": missing_rows,
        "all_metal_rows": alloy_rows,
        "missing_vs_all_metal_precedence_rows": precedence_rows,
        "endpoint_rows": endpoints,
        "network_used": False,
        "model_loaded": False,
        "gpu_used": False,
        "generation_run": False,
        "retry_replacement_repair_filter_or_rerank_used": False,
        "automatic_downstream": False,
    }
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, sort_keys=True))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
