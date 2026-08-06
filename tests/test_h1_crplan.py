from __future__ import annotations

from itertools import product
import unittest

from crystal_dlm.h1_crplan import (
    CRPlanDeadEndError,
    CRPlanIdentityError,
    CRPlanTokenVocabulary,
    FormulaGrammarError,
    FormulaValueCursor,
    OxidationReachability,
    PlanFormulaCursor,
    TerminalChargeError,
    validate_crplan_parsed_identity,
)
from workstreams.plangraph_dlm_iclr_20260731.execution.h1_crplan_r0_paired32_script_package_repair_v5.evaluate_paired32 import (
    classify_attempts,
)


FAKE_STATES = {
    "Li": (1,),
    "Na": (1,),
    "Fe": (2, 3),
    "O": (-2,),
    "Cl": (-1,),
    "Si": (4,),
    "Cu": (1, 2),
}
FAKE_METALS = {"Li", "Na", "Fe", "Cu"}


def policy() -> OxidationReachability:
    return OxidationReachability(
        FAKE_STATES,
        metals=FAKE_METALS,
        max_atoms=20,
        table_source="unit_test",
        table_version="1",
    )


def fail_closed_policy() -> OxidationReachability:
    return OxidationReachability(
        FAKE_STATES,
        metals=FAKE_METALS,
        max_atoms=20,
        table_source="unit_test",
        table_version="1",
        missing_state_policy="fail_closed",
    )


def feed_formula(
    text: str,
    *,
    mode: str = "full_prefix",
) -> PlanFormulaCursor:
    return PlanFormulaCursor.from_text(
        text,
        mode=mode,
        reachability=policy(),
    )


def rich_plan(formula: str, *, newline: str = "\n") -> str:
    return newline.join(
        (
            f"formula: {formula}",
            "anion: oxide",
            "charge: neutral_plausible",
            "lattice: triclinic",
            "spacegroup: sg_001_002",
            "volume: volpa_016_020",
            "end: plan",
        )
    )


def diagnostics_for_formula(formula: str) -> dict:
    cursor = feed_formula(f"formula: {formula}\n")
    return {
        "dead_end": None,
        "final_cursor_phase": cursor.phase,
        "final_cursor_error": None,
        "terminal_certificate": cursor.value.certificate.to_dict(),
    }


class TerminalCertificateTests(unittest.TestCase):
    def test_uniform_witness_matches_evaluator_abstraction(self) -> None:
        certificate = policy().terminal_certificate({"Fe": 2, "O": 3})
        self.assertTrue(certificate.terminal_allowed)
        self.assertTrue(certificate.charge_applicable)
        self.assertTrue(certificate.primary_charge_witness)
        self.assertEqual(
            dict(certificate.uniform_oxidation_witness),
            {"Fe": 3, "O": -2},
        )

    def test_mixed_valence_is_preserved_but_not_primary(self) -> None:
        certificate = policy().terminal_certificate({"Fe": 3, "O": 4})
        self.assertTrue(certificate.terminal_allowed)
        self.assertTrue(certificate.charge_applicable)
        self.assertFalse(certificate.primary_charge_witness)
        self.assertEqual(
            certificate.stratum,
            "charge_applicable_mixed_valence_only",
        )
        witness = {
            symbol: dict(values)
            for symbol, values in certificate.mixed_valence_witness
        }
        self.assertEqual(witness["Fe"], {2: 1, 3: 2})
        self.assertEqual(witness["O"], {-2: 4})

    def test_invalid_charge_has_no_terminal_path(self) -> None:
        certificate = policy().terminal_certificate({"Na": 1, "O": 1})
        self.assertFalse(certificate.terminal_allowed)
        self.assertEqual(
            certificate.stratum,
            "charge_applicable_no_neutral_witness",
        )

    def test_non_applicable_strata_are_explicit(self) -> None:
        reachability = policy()
        self.assertEqual(
            reachability.terminal_certificate({"O": 2}).stratum,
            "charge_not_applicable_unary",
        )
        self.assertEqual(
            reachability.terminal_certificate({"Fe": 1, "Cu": 1}).stratum,
            "charge_not_applicable_all_metal",
        )
        self.assertEqual(
            reachability.terminal_certificate({"Na": 1, "Xe": 1}).stratum,
            "charge_not_applicable_table_missing",
        )

    def test_table_hash_is_order_independent(self) -> None:
        left = OxidationReachability(
            {"O": (-2,), "Fe": (3, 2)},
            metals=("Fe",),
            table_source="x",
            table_version="1",
        )
        right = OxidationReachability(
            {"Fe": (2, 3), "O": (-2,)},
            metals=("Fe",),
            table_source="x",
            table_version="1",
        )
        self.assertEqual(left.table_sha256, right.table_sha256)

    def test_missing_state_policy_preserves_table_hash_but_changes_contract(self) -> None:
        conservative = policy()
        fail_closed = fail_closed_policy()
        self.assertEqual(conservative.table_sha256, fail_closed.table_sha256)
        self.assertNotEqual(
            conservative.constraint_contract_sha256,
            fail_closed.constraint_contract_sha256,
        )
        certificate = fail_closed.terminal_certificate({"Na": 1, "Xe": 1})
        self.assertFalse(certificate.terminal_allowed)
        self.assertTrue(certificate.charge_applicable)
        self.assertEqual(
            certificate.stratum,
            "charge_applicable_oxidation_state_missing",
        )
        self.assertEqual(certificate.missing_state_policy, "fail_closed")

    def test_fail_closed_policy_preserves_direct_shortcuts(self) -> None:
        reachability = fail_closed_policy()
        self.assertTrue(
            reachability.terminal_certificate({"Xe": 2}).terminal_allowed
        )
        self.assertTrue(
            reachability.terminal_certificate(
                {"Fe": 1, "Cu": 1}
            ).terminal_allowed
        )

    def test_fast_terminal_decision_matches_full_certificate(self) -> None:
        rows = (
            {"Fe": 2, "O": 3},
            {"Fe": 3, "O": 4},
            {"Na": 1, "O": 1},
            {"O": 2},
            {"Fe": 1, "Cu": 1},
            {"Na": 1, "Xe": 1},
        )
        for reachability in (policy(), fail_closed_policy()):
            for counts in rows:
                with self.subTest(
                    policy=reachability.missing_state_policy,
                    counts=counts,
                ):
                    decision = reachability.terminal_decision(counts)
                    certificate = reachability.terminal_certificate(counts)
                    self.assertEqual(
                        (
                            decision.counts,
                            decision.total_atoms,
                            decision.stratum,
                            decision.terminal_allowed,
                            decision.charge_applicable,
                            decision.primary_charge_witness,
                            decision.missing_elements,
                            decision.missing_state_policy,
                        ),
                        (
                            certificate.counts,
                            certificate.total_atoms,
                            certificate.stratum,
                            certificate.terminal_allowed,
                            certificate.charge_applicable,
                            certificate.primary_charge_witness,
                            certificate.missing_elements,
                            certificate.missing_state_policy,
                        ),
                    )


class FormulaCursorTests(unittest.TestCase):
    def test_rich_plan_formula_is_constrained_and_remainder_is_free(self) -> None:
        cursor = feed_formula(
            "formula: Fe2O3\nanion: oxide\ncharge: neutral_plausible"
        )
        self.assertEqual(cursor.phase, "after_formula")
        self.assertIsNotNone(cursor.value)
        self.assertEqual(
            cursor.value.certificate.stratum,
            "charge_applicable_uniform_neutral",
        )

    def test_label_can_cross_token_fragments(self) -> None:
        cursor = PlanFormulaCursor(
            mode="full_prefix",
            reachability=policy(),
        )
        for fragment in ("for", "mula", ": Fe", "2O", "3\nanion:"):
            cursor = cursor.feed(fragment)
        self.assertEqual(cursor.phase, "after_formula")
        self.assertTrue(cursor.value.certificate.primary_charge_witness)

    def test_terminal_only_blocks_invalid_newline(self) -> None:
        with self.assertRaises(TerminalChargeError):
            feed_formula("formula: NaO\n", mode="terminal_only")

    def test_grammar_only_does_not_claim_chemistry(self) -> None:
        cursor = feed_formula("formula: NaO\n", mode="grammar_only")
        self.assertEqual(cursor.phase, "after_formula")
        self.assertFalse(cursor.value.certificate.terminal_allowed)

    def test_full_prefix_blocks_unreachable_second_element(self) -> None:
        cursor = FormulaValueCursor(max_atoms=20).feed(
            "Na19",
            mode="full_prefix",
            reachability=policy(),
        )
        with self.assertRaisesRegex(
            FormulaGrammarError,
            "no neutral completion",
        ):
            cursor.feed(
                "Cl",
                mode="full_prefix",
                reachability=policy(),
            )

    def test_count_cannot_be_followed_by_lowercase(self) -> None:
        with self.assertRaises(FormulaGrammarError):
            FormulaValueCursor(max_atoms=20).feed(
                "C2a",
                mode="grammar_only",
                reachability=policy(),
            )

    def test_atom_budget_and_zero_count_fail_closed(self) -> None:
        for formula in ("Na0", "Na21"):
            with self.subTest(formula=formula):
                with self.assertRaises(FormulaGrammarError):
                    FormulaValueCursor(max_atoms=20).feed(
                        formula,
                        mode="grammar_only",
                        reachability=policy(),
                    )

    def test_repeated_elements_are_not_silently_removed(self) -> None:
        cursor = feed_formula("formula: FeOFeO2\n")
        self.assertEqual(
            cursor.value.certificate.counts,
            (("O", 3), ("Fe", 2)),
        )


class ReachabilityParityTests(unittest.TestCase):
    @staticmethod
    def brute_force_element(states: tuple[int, ...], count: int) -> set[int]:
        return {
            sum(values)
            for values in product(states, repeat=count)
        }

    def test_element_dp_matches_brute_force(self) -> None:
        reachability = policy()
        for symbol in ("Fe", "Cu", "O"):
            for count in range(1, 6):
                with self.subTest(symbol=symbol, count=count):
                    self.assertEqual(
                        set(
                            reachability.element_charge_allocations(
                                symbol,
                                count,
                            )
                        ),
                        self.brute_force_element(
                            reachability.oxidation_states[symbol],
                            count,
                        ),
                    )

    def test_bitset_mixed_charge_mask_matches_set_oracle(self) -> None:
        rows = (
            {"Fe": 2, "O": 3},
            {"Fe": 3, "O": 4},
            {"Cu": 4, "O": 3},
            {"Na": 3, "Cl": 2, "O": 1},
        )
        reachability = fail_closed_policy()
        for counts in rows:
            with self.subTest(counts=counts):
                # Let the public canonicalizer define element ordering.
                canonical = reachability.terminal_decision(counts).counts
                mask = reachability._mixed_charge_mask(canonical)
                observed = {
                    index - reachability._charge_offset
                    for index in range(2 * reachability._charge_offset + 1)
                    if mask & (1 << index)
                }
                self.assertEqual(
                    observed,
                    set(reachability._mixed_charge_set(canonical)),
                )

    @staticmethod
    def brute_force_prefix_reachable(
        reachability: OxidationReachability,
        counts: dict[str, int],
    ) -> bool:
        certificate = reachability.terminal_certificate(counts)
        if certificate.terminal_allowed:
            return True
        total_atoms = sum(counts.values())
        remaining = reachability.max_atoms - total_atoms
        if (
            remaining >= 1
            and reachability.missing_state_policy == "allow_non_applicable"
            and any(
                not states
                for states in reachability.oxidation_states.values()
            )
        ):
            return True
        current = {0}
        for symbol, count in counts.items():
            states = reachability.oxidation_states[symbol]
            if not states:
                current = set()
                break
            current = {
                left + sum(allocation)
                for left in current
                for allocation in product(states, repeat=count)
            }
        atom_states = {
            state
            for states in reachability.oxidation_states.values()
            for state in states
        }
        suffix = {0}
        suffix_by_atoms = [{0}]
        for _ in range(remaining):
            suffix = {
                left + state
                for left in suffix
                for state in atom_states
            }
            suffix_by_atoms.append(suffix)
        return any(
            -charge in suffix_by_atoms[future_atoms]
            for charge in current
            for future_atoms in range(1, remaining + 1)
        )

    def test_fast_prefix_reachability_matches_brute_force_contract(self) -> None:
        rows = (
            {"Fe": 2, "O": 3},
            {"Fe": 3, "O": 4},
            {"Na": 1, "O": 1},
            {"Na": 18, "O": 1},
            {"Na": 1, "Xe": 1},
            {"Fe": 1, "Cu": 1},
        )
        for reachability in (policy(), fail_closed_policy()):
            for counts in rows:
                with self.subTest(
                    policy=reachability.missing_state_policy,
                    counts=counts,
                ):
                    self.assertEqual(
                        reachability.materialized_prefix_reachable(counts),
                        self.brute_force_prefix_reachable(
                            reachability,
                            counts,
                        ),
                    )

    def test_fail_closed_full_prefix_preserves_shared_shortcuts(self) -> None:
        reachability = OxidationReachability(
            FAKE_STATES,
            metals=(*FAKE_METALS, "Pm"),
            max_atoms=20,
            table_source="unit_test",
            table_version="1",
            missing_state_policy="fail_closed",
        )
        for formula, stratum in (
            ("Xe", "charge_not_applicable_unary"),
            ("FePm", "charge_not_applicable_all_metal"),
        ):
            with self.subTest(formula=formula):
                cursor = PlanFormulaCursor.from_text(
                    f"formula: {formula}\n",
                    mode="full_prefix",
                    reachability=reachability,
                )
                self.assertEqual(cursor.phase, "after_formula")
                self.assertEqual(cursor.value.certificate.stratum, stratum)

    def test_prefix_cache_is_used(self) -> None:
        reachability = policy()
        self.assertTrue(
            reachability.materialized_prefix_reachable({"Fe": 2, "O": 3})
        )
        before = reachability.diagnostics.cache_hits
        self.assertTrue(
            reachability.materialized_prefix_reachable({"O": 3, "Fe": 2})
        )
        self.assertGreater(reachability.diagnostics.cache_hits, before)

    def test_table_missing_suffix_is_not_falsely_excluded(self) -> None:
        reachability = policy()
        self.assertFalse(
            reachability.terminal_certificate({"Na": 18, "O": 1}).terminal_allowed
        )
        self.assertTrue(
            reachability.materialized_prefix_reachable({"Na": 18, "O": 1})
        )
        terminal = reachability.terminal_certificate(
            {"Na": 18, "O": 1, "Xe": 1}
        )
        self.assertTrue(terminal.terminal_allowed)
        self.assertEqual(
            terminal.stratum,
            "charge_not_applicable_table_missing",
        )

    def test_fail_closed_policy_does_not_use_missing_element_as_suffix(self) -> None:
        reachability = fail_closed_policy()
        self.assertFalse(
            reachability.materialized_prefix_reachable(
                {"Na": 18, "O": 1}
            )
        )
        with self.assertRaises(TerminalChargeError):
            PlanFormulaCursor.from_text(
                "formula: NaXe\n",
                mode="terminal_only",
                reachability=reachability,
            )
        with self.assertRaisesRegex(
            FormulaGrammarError,
            "no neutral completion",
        ):
            PlanFormulaCursor.from_text(
                "formula: NaXe",
                mode="full_prefix",
                reachability=reachability,
            )
        grammar = PlanFormulaCursor.from_text(
            "formula: NaXe\n",
            mode="grammar_only",
            reachability=reachability,
        )
        self.assertEqual(grammar.phase, "after_formula")
        self.assertFalse(grammar.value.certificate.terminal_allowed)

    def test_every_small_allowed_formula_has_reachable_character_prefixes(self) -> None:
        reachability = policy()
        symbols = ("Li", "Na", "Fe", "O", "Cl", "Xe")
        formulas: list[str] = []
        for left in symbols:
            for right in symbols:
                if left == right:
                    continue
                for left_count in range(1, 4):
                    for right_count in range(1, 4):
                        formula = (
                            f"{left}{'' if left_count == 1 else left_count}"
                            f"{right}{'' if right_count == 1 else right_count}"
                        )
                        certificate = reachability.terminal_certificate(
                            {left: left_count, right: right_count}
                        )
                        if certificate.terminal_allowed:
                            formulas.append(formula)
        for formula in formulas:
            with self.subTest(formula=formula):
                cursor = PlanFormulaCursor(
                    mode="full_prefix",
                    reachability=reachability,
                )
                for character in f"formula: {formula}\n":
                    cursor = cursor.feed(character)
                self.assertEqual(cursor.phase, "after_formula")


class ParserFSMIdentityTests(unittest.TestCase):
    def test_exact_rich_plan_identity_and_crlf(self) -> None:
        identity = validate_crplan_parsed_identity(
            raw_model_text=rich_plan("FeOFeO2", newline="\r\n"),
            prompt_style="h1_rich_plan_v1",
            parsed_symbols=("O", "Fe"),
            parsed_counts=(3, 2),
            diagnostics=diagnostics_for_formula("FeOFeO2"),
            mode="full_prefix",
        )
        self.assertTrue(identity["verified"])
        self.assertEqual(identity["raw_repeated_elements"], ["Fe", "O"])

    def test_exact_nocharge_rich_plan_identity(self) -> None:
        text = (
            "formula: Fe2O3\n"
            "anion: oxide\n"
            "lattice: orthorhombic\n"
            "spacegroup: sg_016_074\n"
            "volume: volpa_002\n"
            "end: plan"
        )
        identity = validate_crplan_parsed_identity(
            raw_model_text=text,
            prompt_style="h1_rich_nocharge_plan_v1",
            parsed_symbols=("O", "Fe"),
            parsed_counts=(3, 2),
            diagnostics=diagnostics_for_formula("Fe2O3"),
            mode="full_prefix",
        )
        self.assertTrue(identity["verified"])

    def test_nocharge_rich_plan_rejects_generated_charge(self) -> None:
        text = (
            "formula: Fe2O3\n"
            "anion: oxide\n"
            "charge: neutral_plausible\n"
            "lattice: orthorhombic\n"
            "spacegroup: sg_016_074\n"
            "volume: volpa_002\n"
            "end: plan"
        )
        with self.assertRaises(CRPlanIdentityError):
            validate_crplan_parsed_identity(
                raw_model_text=text,
                prompt_style="h1_rich_nocharge_plan_v1",
                parsed_symbols=("O", "Fe"),
                parsed_counts=(3, 2),
                diagnostics=diagnostics_for_formula("Fe2O3"),
                mode="full_prefix",
            )

    def test_spaced_formula_colon_fails_closed(self) -> None:
        text = rich_plan("Fe2O3").replace("formula:", "formula :", 1)
        with self.assertRaises(CRPlanIdentityError):
            validate_crplan_parsed_identity(
                raw_model_text=text,
                prompt_style="h1_rich_plan_v1",
                parsed_symbols=("O", "Fe"),
                parsed_counts=(3, 2),
                diagnostics=diagnostics_for_formula("Fe2O3"),
                mode="full_prefix",
            )

    def test_duplicate_formula_field_fails_closed(self) -> None:
        text = rich_plan("Fe2O3").replace(
            "anion: oxide",
            "formula: NaCl\nanion: oxide",
        )
        with self.assertRaises(CRPlanIdentityError):
            validate_crplan_parsed_identity(
                raw_model_text=text,
                prompt_style="h1_rich_plan_v1",
                parsed_symbols=("O", "Fe"),
                parsed_counts=(3, 2),
                diagnostics=diagnostics_for_formula("Fe2O3"),
                mode="full_prefix",
            )

    def test_formula_prefill_has_no_generated_identity(self) -> None:
        with self.assertRaises(CRPlanIdentityError):
            validate_crplan_parsed_identity(
                raw_model_text="formula: Fe2O3\nend: plan",
                prompt_style="formula_prefill_v1",
                parsed_symbols=("O", "Fe"),
                parsed_counts=(3, 2),
                diagnostics=diagnostics_for_formula("Fe2O3"),
                mode="full_prefix",
            )

    def test_raw_formula_parser_mismatch_fails_closed(self) -> None:
        with self.assertRaises(CRPlanIdentityError):
            validate_crplan_parsed_identity(
                raw_model_text=rich_plan("Fe2O3"),
                prompt_style="h1_rich_plan_v1",
                parsed_symbols=("O", "Fe"),
                parsed_counts=(4, 3),
                diagnostics=diagnostics_for_formula("Fe2O3"),
                mode="full_prefix",
            )


class PairedEvaluatorGateTests(unittest.TestCase):
    def test_unparsed_identity_failure_cannot_hide_inside_parse_allowance(self) -> None:
        summary, _ = classify_attempts(
            (
                {
                    "sample_idx": 0,
                    "parsed": False,
                    "plan_end_marker_present": True,
                    "reason": "CRPlanIdentityError",
                    "fail_closed": True,
                    "crplan_diagnostics": {
                        "dead_end": None,
                        "legal_support_enforcement": "mask_or_raise",
                        "mask_application_count": 3,
                        "masked_step_count": 3,
                        "empty_support_error_raised": False,
                        "silent_fallback_used_by_decoder": False,
                        "retry_replacement_repair_filter_or_rerank_used": False,
                        "preterminal_support_difference_steps": 1,
                    },
                },
            ),
            reachability=policy(),
            require_crplan_identity=True,
        )
        self.assertEqual(summary["parse_count"], 0)
        self.assertEqual(summary["crplan_identity_failure_count"], 1)
        self.assertEqual(summary["dp_cache_telemetry_invalid_count"], 1)
        self.assertFalse(summary["certificate_and_no_fallback_parity"])


class TokenSupportTests(unittest.TestCase):
    def test_trie_support_matches_scalar_reference(self) -> None:
        vocabulary = CRPlanTokenVocabulary(
            (
                "",
                "formula: ",
                "for",
                "mula",
                ": ",
                "Fe",
                "Fe2",
                "Fe2O3\n",
                "NaO\nanion:",
                "NaCl\n",
                "Xe",
                "Xe2\n",
                "O",
                "2",
                "3",
                "\n",
                "\r\n",
                "unrelated",
                "formula: Fe2O3\nanion:",
                "formula: NaO\nanion:",
            ),
            eos_token_id=0,
        )
        cursor_rows = (
            ("terminal_only", ""),
            ("terminal_only", "formula: "),
            ("terminal_only", "formula: Fe"),
            ("full_prefix", "formula: "),
            ("full_prefix", "formula: Fe2"),
            ("grammar_only", "formula: Na"),
        )
        for mode, text in cursor_rows:
            with self.subTest(mode=mode, text=text):
                reachability = fail_closed_policy()
                cursor = PlanFormulaCursor.from_text(
                    text,
                    mode=mode,
                    reachability=reachability,
                )
                self.assertEqual(
                    vocabulary.support(cursor),
                    vocabulary.support_scalar_reference(cursor),
                )

    def test_token_crossing_newline_is_checked_character_by_character(self) -> None:
        vocabulary = CRPlanTokenVocabulary(
            (
                "formula: NaO\nanion:",
                "formula: Fe2O3\nanion:",
                "unrelated",
            ),
            eos_token_id=None,
        )
        cursor = PlanFormulaCursor(
            mode="terminal_only",
            reachability=policy(),
        )
        support = vocabulary.support(cursor)
        self.assertNotIn(0, support.token_ids)
        self.assertIn(1, support.token_ids)
        self.assertIn(2, support.token_ids)
        self.assertEqual(
            support.rejection_dict()["terminal_charge_block"],
            1,
        )

    def test_combined_bundle_is_nested_and_scalar_exact(self) -> None:
        fragments = (
            "",
            "f",
            "fo",
            "for",
            "formula:",
            "formula: ",
            "formula: Fe",
            "formula: NaO\n",
            "Fe",
            "Fe2",
            "Fe2O3\n",
            "Na",
            "NaCl\n",
            "NaO\nanion:",
            "Xe",
            "Pm",
            "2",
            "3",
            "O",
            "Cl",
            "\n",
            "\r\n",
            "anion:",
            "unrelated",
        )
        vocabulary = CRPlanTokenVocabulary(
            fragments,
            eos_token_id=0,
        )
        reachability = fail_closed_policy()
        cursor = PlanFormulaCursor.from_text(
            "formula: Fe2",
            mode="full_prefix",
            reachability=reachability,
        )
        bundle = vocabulary.support_bundle(cursor)
        grammar = set(bundle.grammar_only.token_ids)
        terminal = set(bundle.terminal_only.token_ids)
        full = set(bundle.full_prefix.token_ids)
        self.assertTrue(full <= terminal <= grammar)
        for mode in ("grammar_only", "terminal_only", "full_prefix"):
            with self.subTest(mode=mode):
                mode_cursor = PlanFormulaCursor.from_text(
                    "formula: Fe2",
                    mode=mode,
                    reachability=fail_closed_policy(),
                )
                self.assertEqual(
                    bundle.for_mode(mode),
                    vocabulary.support_scalar_reference(mode_cursor),
                )

    def test_eos_is_unchanged_before_formula_but_blocked_inside_value(self) -> None:
        vocabulary = CRPlanTokenVocabulary(
            ("", "formula: ", "Fe2O3\n"),
            eos_token_id=0,
        )
        seek = PlanFormulaCursor(
            mode="full_prefix",
            reachability=policy(),
        )
        self.assertIn(0, vocabulary.support(seek).token_ids)
        value = seek.feed("formula: ")
        self.assertNotIn(0, vocabulary.support(value).token_ids)

    def test_empty_support_is_explicit(self) -> None:
        vocabulary = CRPlanTokenVocabulary(("",), eos_token_id=0)
        cursor = PlanFormulaCursor(
            mode="full_prefix",
            reachability=policy(),
        ).feed("formula: ")
        support = vocabulary.support(cursor)
        self.assertEqual(support.token_ids, ())
        with self.assertRaises(CRPlanDeadEndError):
            if not support.token_ids:
                raise CRPlanDeadEndError("test fail closed")


if __name__ == "__main__":
    unittest.main()
