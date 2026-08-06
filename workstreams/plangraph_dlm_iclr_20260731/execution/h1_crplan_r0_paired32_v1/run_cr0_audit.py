#!/usr/bin/env python3
"""Run the frozen EVAL-0/CR-0 audit before any CR-Plan GPU sampling."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from itertools import combinations, product
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from transformers import AutoTokenizer

from crystal_dlm.composition_validity import classify_smact_validity
from crystal_dlm.fixed_slot import SYMBOL_TO_Z
from crystal_dlm.h1_crplan import (
    CRPLAN_MODES,
    CRPlanTokenVocabulary,
    FORMULA_SYMBOLS,
    OxidationReachability,
    PlanFormulaCursor,
    load_frozen_smact_table,
)
from crystal_dlm.ordinal_rng import derive_ordinal_seed
from crystal_dlm.r5_plan_body import symbol_counts_from_formula


FIXTURE_FORMULAS = (
    "Li2O",
    "NaCl",
    "Al2O3",
    "SiO2",
    "BN",
    "SiC",
    "Fe2O3",
    "Fe3O4",
    "CuO",
    "Cu2O",
    "CaCO3",
    "Na2SO4",
    "BaTiO3",
    "SrTiO3",
    "LiFePO4",
    "Na20",
    "FeOFeO2",
    "NaO",
    "Fe",
    "FeNi",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_counts(formula: str) -> tuple[list[str], list[int]]:
    symbols, counts = symbol_counts_from_formula(formula)
    return [str(value) for value in symbols], [int(value) for value in counts]


def evaluator_alignment(
    reachability: OxidationReachability,
    formula: str,
) -> dict[str, Any]:
    symbols, counts = normalized_counts(formula)
    certificate = reachability.terminal_certificate(zip(symbols, counts))
    elements = [int(SYMBOL_TO_Z[value]) for value in symbols]
    evaluator = dict(classify_smact_validity(elements, counts))
    reason = str(evaluator["reason"])
    if certificate.stratum == "charge_applicable_uniform_neutral":
        aligned = reason in {
            "charge_neutral_pauling_valid",
            "pauling_fail_or_ratio_rejected",
        }
    elif certificate.stratum == "charge_applicable_mixed_valence_only":
        aligned = reason == "charge_neutrality_fail"
    elif certificate.stratum == "charge_applicable_no_neutral_witness":
        aligned = reason == "charge_neutrality_fail"
    elif certificate.stratum == "charge_not_applicable_unary":
        aligned = reason == "single_element_shortcut"
    elif certificate.stratum == "charge_not_applicable_all_metal":
        aligned = reason == "all_metal_shortcut"
    elif certificate.stratum == "charge_not_applicable_table_missing":
        aligned = reason == "oxidation_state_missing"
    else:
        aligned = False
    return {
        "formula": formula,
        "symbols": symbols,
        "counts": counts,
        "certificate": certificate.to_dict(),
        "frozen_evaluator": evaluator,
        "aligned": bool(aligned),
    }


def recompute_witness(row: Mapping[str, Any]) -> bool:
    certificate = row["certificate"]
    counts = {
        str(value["element"]): int(value["count"])
        for value in certificate["counts"]
    }
    uniform = certificate["uniform_oxidation_witness"]
    if uniform:
        return (
            set(uniform) == set(counts)
            and sum(counts[symbol] * int(uniform[symbol]) for symbol in counts)
            == 0
        )
    mixed = certificate["mixed_valence_witness"]
    if mixed:
        if set(mixed) != set(counts):
            return False
        atom_count_ok = all(
            sum(int(value) for value in mixed[symbol].values())
            == counts[symbol]
            for symbol in counts
        )
        charge = sum(
            int(oxidation) * int(number)
            for values in mixed.values()
            for oxidation, number in values.items()
        )
        return atom_count_ok and charge == 0
    return not certificate["primary_charge_witness"]


def brute_force_charge_set(states: Sequence[int], count: int) -> set[int]:
    return {
        sum(values)
        for values in product(tuple(int(value) for value in states), repeat=count)
    }


def tokenizer_audit(
    tokenizer: Any,
    vocabulary: CRPlanTokenVocabulary,
    reachability: OxidationReachability,
    fixture_formulas: Sequence[str],
) -> dict[str, Any]:
    plans = tuple(
        "\n".join(
            (
                f"formula: {formula}",
                "anion: other",
                "charge: neutral_plausible",
                "lattice: triclinic",
                "spacegroup: sg_001_002",
                "volume: volpa_016_020",
                "end: plan",
            )
        )
        for formula in fixture_formulas
    )
    parity_rows: list[dict[str, Any]] = []
    for text in plans:
        token_ids = [
            int(value)
            for value in tokenizer.encode(text, add_special_tokens=False)
        ]
        whole = tokenizer.decode(
            token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        fragments = "".join(vocabulary.fragments[value] for value in token_ids)
        cursor_ok = True
        cursor_error = None
        try:
            cursor = PlanFormulaCursor.from_text(
                fragments,
                mode="grammar_only",
                reachability=reachability,
            )
            cursor_ok = cursor.phase == "after_formula"
        except Exception as exc:  # noqa: BLE001
            cursor_ok = False
            cursor_error = f"{type(exc).__name__}: {exc}"
        parity_rows.append(
            {
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "token_count": len(token_ids),
                "whole_decode_equals_fragment_concat": whole == fragments,
                "whole_decode_equals_source": whole == text,
                "cursor_ok": cursor_ok,
                "cursor_error": cursor_error,
            }
        )
    relevant_characters = frozenset(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789: \t\r\n"
    )
    relevant_fragments = [
        (token_id, fragment)
        for token_id, fragment in enumerate(vocabulary.fragments)
        if fragment and all(value in relevant_characters for value in fragment)
    ]
    transition_prefixes = (
        "formula: ",
        "formula: F",
        "formula: Fe",
        "formula: Fe2",
        "formula: Fe2O",
        "formula: Na19",
        "formula: Li2O",
        "formula: Fe3O4",
    )
    transition_rows: list[dict[str, Any]] = []
    transition_mismatches: list[dict[str, Any]] = []
    for prefix in transition_prefixes:
        prefix_ids = [
            int(value)
            for value in tokenizer.encode(prefix, add_special_tokens=False)
        ]
        decoded_prefix = tokenizer.decode(
            prefix_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        cursor = PlanFormulaCursor.from_text(
            decoded_prefix,
            mode="grammar_only",
            reachability=reachability,
        )
        support = vocabulary.support(cursor)
        mismatch_count = 0
        for token_id in support.token_ids:
            combined = tokenizer.decode(
                [*prefix_ids, int(token_id)],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            incremental = decoded_prefix + vocabulary.fragments[token_id]
            if combined != incremental:
                mismatch_count += 1
                if len(transition_mismatches) < 20:
                    transition_mismatches.append(
                        {
                            "prefix": prefix,
                            "token_id": int(token_id),
                            "fragment_repr": repr(
                                vocabulary.fragments[token_id]
                            ),
                            "combined_suffix_repr": repr(
                                combined[len(decoded_prefix) :]
                            ),
                        }
                    )
        transition_rows.append(
            {
                "prefix": prefix,
                "prefix_roundtrip": decoded_prefix == prefix,
                "legal_support_size": len(support.token_ids),
                "incremental_decode_mismatch_count": mismatch_count,
            }
        )
    return {
        "schema": "h1_crplan_tokenizer_audit_v1",
        "vocab_size": len(tokenizer),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "padding_side": tokenizer.padding_side,
        "fragment_sha256": vocabulary.fragment_sha256,
        "formula_relevant_ascii_fragment_count": len(relevant_fragments),
        "fixture_rows": parity_rows,
        "all_fixture_whole_decode_fragment_parity": all(
            value["whole_decode_equals_fragment_concat"]
            for value in parity_rows
        ),
        "all_fixture_source_roundtrip": all(
            value["whole_decode_equals_source"] for value in parity_rows
        ),
        "all_fixture_cursor_ok": all(
            value["cursor_ok"] for value in parity_rows
        ),
        "transition_rows": transition_rows,
        "transition_decode_mismatches": transition_mismatches,
        "all_legal_transition_decode_parity": all(
            value["prefix_roundtrip"]
            and value["incremental_decode_mismatch_count"] == 0
            for value in transition_rows
        ),
    }


def probability_and_empty_support_audit(
    tokenizer: Any,
    vocabulary: CRPlanTokenVocabulary,
    reachability: OxidationReachability,
) -> dict[str, Any]:
    import torch

    seek = PlanFormulaCursor(
        mode="full_prefix",
        reachability=reachability,
    )
    after = PlanFormulaCursor.from_text(
        "formula: Fe2O3\n",
        mode="full_prefix",
        reachability=reachability,
    )
    after_support = vocabulary.support(after)
    full_support = tuple(range(len(tokenizer)))
    scores = torch.linspace(-3.0, 3.0, steps=len(tokenizer), dtype=torch.float64)
    direct = torch.softmax(scores, dim=-1)
    allowed = torch.tensor(after_support.token_ids, dtype=torch.long)
    restricted = torch.full_like(scores, -torch.inf)
    restricted[allowed] = scores[allowed]
    renormalized = torch.softmax(restricted, dim=-1)
    same_support_probability_parity = bool(
        after_support.token_ids == full_support
        and torch.equal(direct, renormalized)
    )
    fake_empty = CRPlanTokenVocabulary(("",), eos_token_id=0)
    formula_value = seek.feed("formula: ")
    empty_support = fake_empty.support(formula_value)
    return {
        "schema": "h1_crplan_probability_empty_support_audit_v1",
        "same_full_support_probability_parity": same_support_probability_parity,
        "after_formula_support_is_full_vocab": (
            after_support.token_ids == full_support
        ),
        "empty_support_detected": len(empty_support.token_ids) == 0,
        "silent_fallback_on_empty_support": False,
    }


def ordinal_rng_audit() -> dict[str, Any]:
    baseline = {
        ordinal: derive_ordinal_seed(
            17029,
            sample_idx=ordinal,
            stage="planner_sampling",
            role="shared",
        )
        for ordinal in range(32)
    }
    resumed = {
        ordinal: derive_ordinal_seed(
            17029,
            sample_idx=ordinal,
            stage="planner_sampling",
            role="shared",
        )
        for ordinal in reversed(range(32))
    }
    return {
        "schema": "h1_crplan_ordinal_rng_audit_v1",
        "ordinal_count": 32,
        "unique_seed_count": len(set(baseline.values())),
        "order_independent": baseline == resumed,
        "arm_role": "shared",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    tokenizer_source = (
        args.checkpoint_path
        if (args.checkpoint_path / "tokenizer_config.json").exists()
        else args.model_path
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        trust_remote_code=True,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    reachability = load_frozen_smact_table(max_atoms=20)
    vocabulary = CRPlanTokenVocabulary.from_tokenizer(tokenizer)

    fixture_formulas = list(FIXTURE_FORMULAS)
    missing_elements = [
        symbol
        for symbol in FORMULA_SYMBOLS
        if not reachability.oxidation_states[symbol]
    ]
    table_missing_fixture = None
    if missing_elements:
        missing = missing_elements[0]
        partner = "O" if missing != "O" else "Na"
        table_missing_fixture = f"{missing}{partner}"
        fixture_formulas.append(table_missing_fixture)

    alignments = [
        evaluator_alignment(reachability, formula)
        for formula in fixture_formulas
    ]
    exhaustive_formulas = [
        (
            f"{left}{'' if left_count == 1 else left_count}"
            f"{right}{'' if right_count == 1 else right_count}"
        )
        for left, right in combinations(
            ("Li", "Na", "Mg", "Al", "Fe", "O", "Cl"),
            2,
        )
        for left_count in range(1, 4)
        for right_count in range(1, 4)
    ]
    exhaustive_alignments = [
        evaluator_alignment(reachability, formula)
        for formula in exhaustive_formulas
    ]
    witness_rows = [
        {
            "formula": row["formula"],
            "recomputes": recompute_witness(row),
        }
        for row in alignments
    ]
    false_exclusions: list[dict[str, Any]] = []
    for row in (*alignments, *exhaustive_alignments):
        if row["frozen_evaluator"]["valid"] is not True:
            continue
        formula = str(row["formula"])
        try:
            cursor = PlanFormulaCursor.from_text(
                f"formula: {formula}\n",
                mode="full_prefix",
                reachability=reachability,
            )
            reachable = cursor.phase == "after_formula"
            error = None
        except Exception as exc:  # noqa: BLE001
            reachable = False
            error = f"{type(exc).__name__}: {exc}"
        if not reachable:
            false_exclusions.append({"formula": formula, "error": error})

    dp_rows: list[dict[str, Any]] = []
    for symbol in ("Fe", "Cu", "O"):
        states = reachability.oxidation_states.get(symbol, ())
        if not states:
            continue
        for count in range(1, 5):
            observed = set(
                reachability.element_charge_allocations(symbol, count)
            )
            expected = brute_force_charge_set(states, count)
            dp_rows.append(
                {
                    "element": symbol,
                    "count": count,
                    "state_count": len(states),
                    "parity": observed == expected,
                    "observed_charge_count": len(observed),
                }
            )

    tokenizer_report = tokenizer_audit(
        tokenizer,
        vocabulary,
        reachability,
        fixture_formulas,
    )
    probability_report = probability_and_empty_support_audit(
        tokenizer,
        vocabulary,
        reachability,
    )
    rng_report = ordinal_rng_audit()
    failures: list[str] = []
    if not all(value["aligned"] for value in alignments):
        failures.append("frozen_evaluator_alignment")
    if not all(value["aligned"] for value in exhaustive_alignments):
        failures.append("exhaustive_frozen_evaluator_alignment")
    if not all(value["recomputes"] for value in witness_rows):
        failures.append("terminal_witness_recomputation")
    if false_exclusions:
        failures.append("valid_fixture_false_exclusion")
    if not all(value["parity"] for value in dp_rows):
        failures.append("dp_bruteforce_parity")
    if not tokenizer_report["all_fixture_whole_decode_fragment_parity"]:
        failures.append("tokenizer_fragment_concatenation")
    if not tokenizer_report["all_fixture_source_roundtrip"]:
        failures.append("tokenizer_source_roundtrip")
    if not tokenizer_report["all_fixture_cursor_ok"]:
        failures.append("tokenizer_cursor_recovery")
    if not tokenizer_report["all_legal_transition_decode_parity"]:
        failures.append("tokenizer_incremental_transition_decode")
    if not probability_report["same_full_support_probability_parity"]:
        failures.append("same_support_probability_parity")
    if not probability_report["empty_support_detected"]:
        failures.append("empty_support_fail_close")
    if not rng_report["order_independent"] or rng_report["unique_seed_count"] != 32:
        failures.append("ordinal_rng")
    if reachability.diagnostics.states_created > 100_000:
        failures.append("dp_state_budget")
    uniform_pauling_fail_count = sum(
        int(
            value["certificate"]["primary_charge_witness"] is True
            and value["frozen_evaluator"]["reason"]
            == "pauling_fail_or_ratio_rejected"
        )
        for value in (*alignments, *exhaustive_alignments)
    )
    if uniform_pauling_fail_count == 0:
        failures.append("pauling_non_hard_constraint_fixture_missing")

    report = {
        "schema": "h1_crplan_cr0_terminal_report_v1",
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "source_manifest_sha256": sha256_file(args.source_manifest),
        "tokenizer_source": str(tokenizer_source),
        "oxidation_table": reachability.table_report(),
        "tokenizer": tokenizer_report,
        "evaluator_alignment": alignments,
        "exhaustive_evaluator_alignment": {
            "formula_count": len(exhaustive_alignments),
            "all_aligned": all(
                value["aligned"] for value in exhaustive_alignments
            ),
            "misaligned": [
                value
                for value in exhaustive_alignments
                if not value["aligned"]
            ],
        },
        "uniform_neutral_but_pauling_fail_fixture_count": (
            uniform_pauling_fail_count
        ),
        "table_missing_fixture": table_missing_fixture,
        "prefix_semantics": {
            "elements_without_states": missing_elements,
            "full_prefix_structurally_degenerates_toward_terminal_only": (
                bool(missing_elements)
            ),
            "reason": (
                "any prefix with at least one remaining atom may append a "
                "table-missing element that the frozen terminal contract "
                "allows as non-applicable"
                if missing_elements
                else None
            ),
            "prefix_control_gain_attribution_allowed": (
                not bool(missing_elements)
            ),
        },
        "terminal_witness_recomputation": witness_rows,
        "valid_evaluator_fixture_count": sum(
            int(value["frozen_evaluator"]["valid"] is True)
            for value in alignments
        ),
        "valid_fixture_false_exclusions": false_exclusions,
        "dp_bruteforce_rows": dp_rows,
        "dp": reachability.diagnostics.snapshot(),
        "probability_and_empty_support": probability_report,
        "ordinal_rng": rng_report,
        "modes": list(CRPLAN_MODES),
        "mixed_valence_only_is_primary_gain": False,
        "pauling_is_hard_constraint": False,
        "retry_replacement_repair_filter_or_rerank_used": False,
        "automatic_downstream": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, sort_keys=True))
    return 0 if not failures else 2


if __name__ == "__main__":
    sys.exit(main())
