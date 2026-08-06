#!/usr/bin/env python3
"""Exact-tokenizer CR-Plan V3 optimized-support and CPU parity audit."""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from functools import reduce
import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Iterable

from crystal_dlm.composition_validity import classify_smact_validity
from crystal_dlm.fixed_slot import SYMBOL_TO_Z
from crystal_dlm.h1_crplan import (
    CRPLAN_SCHEMA,
    CRPLAN_MODES,
    CRPlanTokenVocabulary,
    FORMULA_LABEL,
    FormulaGrammarError,
    FormulaValueCursor,
    OxidationReachability,
    PlanFormulaCursor,
    TerminalChargeError,
    load_frozen_smact_table,
)


EXPECTED_MISSING = ("He", "Ne", "Ar", "Pm", "At", "Rn", "Fr", "Ra")
CURSOR_TEXTS = (
    *(FORMULA_LABEL[:width] for width in range(len(FORMULA_LABEL))),
    "xxfor",
    "formula: ",
    "formula: F",
    "formula: Fe",
    "formula: Fe2",
    "formula: Fe2O",
    "formula: Fe2O3",
    "formula: Na",
    "formula: NaCl",
    "formula: Na19",
    "formula: Xe",
)
CURSOR_FIXTURES = tuple(
    (mode, text)
    for text in CURSOR_TEXTS
    for mode in CRPLAN_MODES
) + (
    ("grammar_only", "formula: FePm"),
    ("terminal_only", "formula: FePm"),
    ("grammar_only", "formula: OHe"),
    ("terminal_only", "formula: OHe"),
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
    vocabulary._support_cache.clear()
    vocabulary._support_bundle_cache.clear()
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


def load_real_formula_value_signatures(
    raw_jsonl: Path,
) -> tuple[tuple[Any, ...], ...]:
    signatures: set[tuple[Any, ...]] = set()
    with raw_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            diagnostics = record.get("crplan_diagnostics")
            if not isinstance(diagnostics, dict):
                continue
            for step in diagnostics.get("steps") or ():
                if (
                    not isinstance(step, dict)
                    or step.get("phase") != "formula_value"
                    or not isinstance(step.get("formula_signature"), str)
                ):
                    continue
                signature = ast.literal_eval(step["formula_signature"])
                if not isinstance(signature, tuple) or len(signature) != 7:
                    raise AssertionError(
                        "unexpected real-model formula cursor signature"
                    )
                if bool(signature[5]):
                    continue
                signatures.add(signature)
    return tuple(sorted(signatures, key=repr))


def cursor_from_value_signature(
    signature: tuple[Any, ...],
    *,
    mode: str,
    reachability: OxidationReachability,
) -> PlanFormulaCursor:
    (
        max_atoms,
        committed_counts,
        pending_symbol_prefix,
        count_digits,
        seen_element,
        done,
        certificate_stratum,
    ) = signature
    if bool(done) or certificate_stratum is not None:
        raise AssertionError(
            "preterminal trace unexpectedly carried a certificate"
        )
    value = FormulaValueCursor(
        max_atoms=int(max_atoms),
        committed_counts=tuple(
            (str(symbol), int(count))
            for symbol, count in committed_counts
        ),
        pending_symbol_prefix=str(pending_symbol_prefix),
        count_digits=str(count_digits),
        seen_element=bool(seen_element),
        done=False,
        certificate=None,
    )
    return PlanFormulaCursor(
        mode=mode,
        reachability=reachability,
        phase="formula_value",
        value=value,
    )


def real_cursor_support_row(
    vocabulary: CRPlanTokenVocabulary,
    base: OxidationReachability,
    *,
    signature: tuple[Any, ...],
    mode: str,
) -> dict[str, Any]:
    optimized_reachability = clone_reachability(base)
    scalar_reachability = clone_reachability(base)
    optimized_cursor = cursor_from_value_signature(
        signature,
        mode=mode,
        reachability=optimized_reachability,
    )
    scalar_cursor = cursor_from_value_signature(
        signature,
        mode=mode,
        reachability=scalar_reachability,
    )
    vocabulary._support_cache.clear()
    vocabulary._support_bundle_cache.clear()
    before = optimized_reachability.diagnostics.snapshot()
    started = time.perf_counter()
    optimized = vocabulary.support(optimized_cursor)
    optimized_seconds = time.perf_counter() - started
    after = optimized_reachability.diagnostics.snapshot()
    scalar = vocabulary.support_scalar_reference(scalar_cursor)
    fields_equal = {
        "token_ids": optimized.token_ids == scalar.token_ids,
        "terminal_token_ids": (
            optimized.terminal_token_ids
            == scalar.terminal_token_ids
        ),
        "rejection_counts": (
            optimized.rejection_counts == scalar.rejection_counts
        ),
    }
    return {
        "mode": mode,
        "formula_signature": repr(signature),
        "all_equal": all(fields_equal.values()),
        "fields_equal": fields_equal,
        "optimized_seconds": optimized_seconds,
        "optimized_dp_delta": {
            key: int(after[key]) - int(before[key])
            for key in before
        },
        "legal_support_size": len(optimized.token_ids),
        "terminal_support_size": len(optimized.terminal_token_ids),
        "rejection_counts": optimized.rejection_dict(),
    }


def logical_terminal_fields(
    certificate: Any,
) -> tuple[Any, ...]:
    return (
        certificate.counts,
        certificate.total_atoms,
        certificate.stratum,
        certificate.terminal_allowed,
        certificate.charge_applicable,
        certificate.primary_charge_witness,
        certificate.missing_elements,
        certificate.missing_state_policy,
    )


def deterministic_composition_rows(
    reachability: OxidationReachability,
) -> tuple[tuple[tuple[str, int], ...], ...]:
    rng = random.Random(20260804)
    active = tuple(
        symbol
        for symbol, states in reachability.oxidation_states.items()
        if states
    )
    selected = tuple(
        symbol
        for symbol in (
            "Li",
            "Na",
            "Fe",
            "Cu",
            "Si",
            "O",
            "Cl",
            "S",
            "N",
        )
        if symbol in active
    )
    rows: set[tuple[tuple[str, int], ...]] = set()
    for left in selected:
        for right in selected:
            if left == right:
                continue
            for left_count in range(1, 5):
                for right_count in range(1, 5):
                    if left_count + right_count <= 8:
                        rows.add(
                            reachability.terminal_decision(
                                {
                                    left: left_count,
                                    right: right_count,
                                }
                            ).counts
                        )
    for _ in range(160):
        arity = rng.randint(1, 4)
        symbols = rng.sample(list(active), arity)
        remaining = reachability.max_atoms
        counts: dict[str, int] = {}
        for position, symbol in enumerate(symbols):
            maximum = remaining - (arity - position - 1)
            count = rng.randint(1, maximum)
            counts[symbol] = count
            remaining -= count
        rows.add(reachability.terminal_decision(counts).counts)
    return tuple(sorted(rows, key=repr))


def bitset_parity_report(
    base: OxidationReachability,
) -> dict[str, Any]:
    reachability = clone_reachability(base)
    rows = deterministic_composition_rows(reachability)
    mismatches: list[dict[str, Any]] = []
    decision_mismatches: list[dict[str, Any]] = []
    prefix_mismatches: list[dict[str, Any]] = []
    for counts in rows:
        mask = reachability._mixed_charge_mask(counts)
        observed = {
            index - reachability._charge_offset
            for index in range(2 * reachability._charge_offset + 1)
            if mask & (1 << index)
        }
        expected = set(reachability._mixed_charge_set(counts))
        if observed != expected:
            mismatches.append(
                {
                    "counts": list(counts),
                    "observed": sorted(observed),
                    "expected": sorted(expected),
                }
            )
        decision = reachability.terminal_decision(counts)
        certificate = reachability.terminal_certificate(counts)
        if logical_terminal_fields(decision) != logical_terminal_fields(
            certificate
        ):
            decision_mismatches.append(
                {
                    "counts": list(counts),
                    "decision": decision.to_dict(),
                    "certificate": certificate.to_dict(),
                }
            )
        total = sum(count for _, count in counts)
        if not 1 <= total <= reachability.max_atoms:
            expected_prefix = False
        elif len(counts) == 1:
            expected_prefix = True
        elif all(
            symbol in reachability.metals
            for symbol, _ in counts
        ):
            expected_prefix = True
        elif any(
            not reachability.oxidation_states[symbol]
            for symbol, _ in counts
        ):
            expected_prefix = (
                reachability.missing_state_policy
                == "allow_non_applicable"
            )
        else:
            remaining = reachability.max_atoms - total
            current = reachability._mixed_charge_set(counts)
            expected_prefix = (
                0 in current
                or any(
                    -charge
                    in reachability._suffix_exact_charge_sets[
                        future_atoms
                    ]
                    for charge in current
                    for future_atoms in range(1, remaining + 1)
                )
            )
        observed_prefix = (
            reachability.materialized_prefix_reachable(counts)
        )
        if observed_prefix != expected_prefix:
            prefix_mismatches.append(
                {
                    "counts": list(counts),
                    "observed": observed_prefix,
                    "expected": expected_prefix,
                }
            )
    return {
        "row_count": len(rows),
        "mixed_charge_mask_set_parity": not mismatches,
        "mixed_charge_mismatches": mismatches,
        "terminal_decision_certificate_parity": not decision_mismatches,
        "terminal_decision_mismatches": decision_mismatches,
        "prefix_bitset_set_parity": not prefix_mismatches,
        "prefix_mismatches": prefix_mismatches,
    }


def _top_k_top_p_distribution(
    logits: tuple[float, ...],
    token_ids: Iterable[int],
    *,
    top_k: int = 50,
    top_p: float = 0.95,
) -> tuple[tuple[int, float], ...]:
    ids = tuple(sorted(set(int(value) for value in token_ids)))
    if not ids:
        return ()
    ranked = sorted(
        ids,
        key=lambda token_id: (-float(logits[token_id]), token_id),
    )
    threshold_position = min(int(top_k), len(ranked)) - 1
    threshold = float(logits[ranked[threshold_position]])
    top_k_ids = tuple(
        token_id
        for token_id in ranked
        if float(logits[token_id]) >= threshold
    )
    maximum = max(float(logits[token_id]) for token_id in top_k_ids)
    weights = tuple(
        math.exp(float(logits[token_id]) - maximum)
        for token_id in top_k_ids
    )
    normalizer = sum(weights)
    cumulative = 0.0
    keep_count = len(top_k_ids)
    for index, weight in enumerate(weights):
        cumulative += weight / normalizer
        if cumulative > float(top_p):
            keep_count = index + 1
            break
    kept_ids = top_k_ids[:keep_count]
    kept_weights = weights[:keep_count]
    kept_normalizer = sum(kept_weights)
    return tuple(
        (token_id, float(weight / kept_normalizer))
        for token_id, weight in zip(kept_ids, kept_weights)
    )


def mask_probability_parity_report(
    vocabulary: CRPlanTokenVocabulary,
    base: OxidationReachability,
) -> dict[str, Any]:
    optimized_reachability = clone_reachability(base)
    scalar_reachability = clone_reachability(base)
    optimized_cursor = PlanFormulaCursor.from_text(
        "formula: Fe2",
        mode="full_prefix",
        reachability=optimized_reachability,
    )
    vocabulary._support_cache.clear()
    vocabulary._support_bundle_cache.clear()
    optimized_bundle = vocabulary.support_bundle(optimized_cursor)
    scalar_supports = {
        mode: vocabulary.support_scalar_reference(
            PlanFormulaCursor.from_text(
                "formula: Fe2",
                mode=mode,
                reachability=scalar_reachability,
            )
        )
        for mode in CRPLAN_MODES
    }
    full_ids = optimized_bundle.full_prefix.token_ids
    full_id_set = set(full_ids)
    illegal_ids = tuple(
        token_id
        for token_id in range(vocabulary.vocab_size)
        if token_id not in full_id_set
    )
    rng = random.Random(20260804)
    fixtures: list[
        tuple[str, tuple[float, ...], tuple[int, ...], tuple[int, ...]]
    ] = []
    random_logits = tuple(
        rng.gauss(0.0, 1.0)
        for _ in range(vocabulary.vocab_size)
    )
    fixtures.append(
        (
            "random_logits",
            random_logits,
            full_ids,
            scalar_supports["full_prefix"].token_ids,
        )
    )
    denominator = max(1, vocabulary.vocab_size - 1)
    backfill_values = [
        -2.0 + 4.0 * index / denominator
        for index in range(vocabulary.vocab_size)
    ]
    for rank, token_id in enumerate(illegal_ids[:64]):
        backfill_values[int(token_id)] = 100.0 - float(rank)
    backfill = tuple(backfill_values)
    fixtures.append(
        (
            "illegal_global_top50_backfill",
            backfill,
            full_ids,
            scalar_supports["full_prefix"].token_ids,
        )
    )
    tie_values = [-5.0 for _ in range(vocabulary.vocab_size)]
    for token_id in full_ids[:80]:
        tie_values[int(token_id)] = 1.0
    ties = tuple(tie_values)
    fixtures.append(
        (
            "kth_legal_tie",
            ties,
            full_ids,
            scalar_supports["full_prefix"].token_ids,
        )
    )
    boundary = tuple(
        1.0 - 9.0 * index / denominator
        for index in range(vocabulary.vocab_size)
    )
    fixtures.append(
        (
            "top_p_boundary",
            boundary,
            full_ids,
            scalar_supports["full_prefix"].token_ids,
        )
    )
    small = tuple(full_ids[:17])
    fixtures.append(
        (
            "fewer_than_50_legal",
            random_logits,
            small,
            small,
        )
    )
    fixtures.append(
        (
            "empty_support",
            random_logits,
            (),
            (),
        )
    )
    rows: list[dict[str, Any]] = []
    for name, logits, optimized_ids, reference_ids in fixtures:
        optimized_mask_ids = tuple(sorted(set(optimized_ids)))
        reference_mask_ids = tuple(sorted(set(reference_ids)))
        masks_equal = optimized_mask_ids == reference_mask_ids
        optimized_distribution = _top_k_top_p_distribution(
            logits,
            optimized_mask_ids,
        )
        reference_distribution = _top_k_top_p_distribution(
            logits,
            reference_mask_ids,
        )
        distributions_equal = (
            optimized_distribution == reference_distribution
        )
        rows.append(
            {
                "fixture": name,
                "masks_equal": masks_equal,
                "distributions_equal": distributions_equal,
                "finite_support_size": len(optimized_mask_ids),
                "nonzero_probability_size": len(
                    optimized_distribution
                ),
            }
        )
    return {
        "rows": rows,
        "all_equal": all(
            row["masks_equal"] and row["distributions_equal"]
            for row in rows
        ),
        "lazy_top_k_legality_used": False,
        "full_online_mask_used": True,
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
    parser.add_argument(
        "--candidate-raw-jsonl",
        type=Path,
        required=True,
    )
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
    real_signatures = load_real_formula_value_signatures(
        args.candidate_raw_jsonl
    )
    real_cursor_rows = [
        real_cursor_support_row(
            vocabulary,
            reachability,
            signature=signature,
            mode=mode,
        )
        for signature in real_signatures
        for mode in CRPLAN_MODES
    ]
    bitset_report = bitset_parity_report(reachability)
    mask_report = mask_probability_parity_report(
        vocabulary,
        reachability,
    )
    missing_rows = missing_policy_rows(reachability)
    alloy_rows = all_metal_rows(reachability)
    precedence_rows = missing_vs_all_metal_precedence_rows(reachability)
    endpoints = endpoint_rows(reachability)
    failures: list[str] = []
    if not all(row["all_equal"] for row in support_rows):
        failures.append("optimized_scalar_fixture_support_parity")
    if not real_signatures:
        failures.append("real_model_cursor_trace_missing")
    if not all(row["all_equal"] for row in real_cursor_rows):
        failures.append("optimized_scalar_real_cursor_support_parity")
    if not bitset_report["mixed_charge_mask_set_parity"]:
        failures.append("mixed_charge_bitset_set_parity")
    if not bitset_report["prefix_bitset_set_parity"]:
        failures.append("prefix_bitset_set_parity")
    if not bitset_report["terminal_decision_certificate_parity"]:
        failures.append("terminal_decision_certificate_parity")
    if not mask_report["all_equal"]:
        failures.append("mask_probability_pipeline_parity")
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
        "schema": "h1_crplan_fourarm512_exact_support_optimization_preflight_v3",
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "crplan_schema": CRPLAN_SCHEMA,
        "source_manifest_sha256": sha256_file(args.source_manifest),
        "tokenizer_path": str(args.tokenizer),
        "candidate_raw_jsonl": str(args.candidate_raw_jsonl),
        "candidate_raw_jsonl_sha256": sha256_file(
            args.candidate_raw_jsonl
        ),
        "tokenizer_seconds": tokenizer_seconds,
        "vocab_size": len(tokenizer),
        "token_fragment_sha256": vocabulary.fragment_sha256,
        "vocabulary_build_seconds": vocabulary_seconds,
        "oxidation_table": reachability.table_report(),
        "support_rows": support_rows,
        "support_parity_all": all(row["all_equal"] for row in support_rows),
        "support_optimized_seconds_total": sum(
            row["trie_seconds"] for row in support_rows
        ),
        "support_scalar_seconds_total": sum(
            row["scalar_seconds"] for row in support_rows
        ),
        "support_optimized_dp_states_max": max(
            row["trie_dp_delta"]["states_created"]
            for row in support_rows
        ),
        "real_model_formula_cursor_signature_count": len(
            real_signatures
        ),
        "real_model_support_row_count": len(real_cursor_rows),
        "real_model_support_rows": real_cursor_rows,
        "real_model_support_parity_all": all(
            row["all_equal"] for row in real_cursor_rows
        ),
        "real_model_optimized_seconds_total": sum(
            row["optimized_seconds"] for row in real_cursor_rows
        ),
        "real_model_optimized_dp_states_max": max(
            (
                row["optimized_dp_delta"]["states_created"]
                for row in real_cursor_rows
            ),
            default=0,
        ),
        "bitset_parity": bitset_report,
        "mask_probability_parity": mask_report,
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
