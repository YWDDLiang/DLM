from pathlib import Path
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crystal_dlm.ccfd import FormulaToken
from crystal_dlm.ccfd_v2 import (
    BenchmarkReachability,
    CERTIFICATE_BENCHMARK,
    CERTIFICATE_EXTENDED_ONLY,
    CCFDv2State,
    END_COMPOSITION,
    SetAtomCount,
    compile_plan_actions,
    legal_next_actions,
    render_rich_plan,
    replay_actions,
)


def benchmark_true(_elements, _counts):
    return {"valid": True, "reason": "charge_neutral_pauling_valid"}


def benchmark_false(_elements, _counts):
    return {"valid": False, "reason": "charge_neutrality_fail"}


class CCFDv2Test(unittest.TestCase):
    def test_N_is_first_and_locked(self):
        state = CCFDv2State.start()
        with self.assertRaisesRegex(ValueError, "before species"):
            state.append_species(FormulaToken.from_symbol("O", -2, 1))
        state = state.apply(SetAtomCount(2))
        with self.assertRaisesRegex(ValueError, "already locked"):
            state.apply(SetAtomCount(3))
        with self.assertRaisesRegex(ValueError, "outside"):
            CCFDv2State.start().apply(SetAtomCount(21))

    def test_exact_atom_charge_and_explicit_end(self):
        actions = (
            SetAtomCount(2),
            FormulaToken.from_symbol("O", -2, 1),
            FormulaToken.from_symbol("Fe", 2, 1),
            END_COMPOSITION,
        )
        state = replay_actions(actions)
        self.assertTrue(state.ended)
        self.assertEqual(state.remaining_atoms, 0)
        self.assertEqual(state.net_charge, 0)
        certificate = state.certificate(benchmark_validator=benchmark_true)
        self.assertEqual(certificate.certificate_class, CERTIFICATE_BENCHMARK)
        self.assertTrue(certificate.usable_as_positive)
        with self.assertRaisesRegex(ValueError, "already ended"):
            state.apply(FormulaToken.from_symbol("Ni", 2, 1))

    def test_premature_end_and_count_overflow_fail_closed(self):
        state = CCFDv2State.start().apply(SetAtomCount(2))
        with self.assertRaisesRegex(ValueError, "before exact"):
            state.apply(END_COMPOSITION)
        with self.assertRaisesRegex(ValueError, "exceeds"):
            state.apply(FormulaToken.from_symbol("O", -2, 3))

    def test_extended_only_is_unknown_not_positive(self):
        state = replay_actions(
            (
                SetAtomCount(2),
                FormulaToken.from_symbol("F", -1, 1),
                FormulaToken.from_symbol("Na", 1, 1),
                END_COMPOSITION,
            )
        )
        certificate = state.certificate(benchmark_validator=benchmark_false)
        self.assertEqual(certificate.certificate_class, CERTIFICATE_EXTENDED_ONLY)
        self.assertTrue(certificate.extended_valid)
        self.assertFalse(certificate.usable_as_positive)

    def test_extended_compiler_witness_stays_unknown(self):
        plan = {"N": 2, "elements": ["Na", "Cl"], "counts": [1, 1]}

        def extended(_symbols, _counts, _max_species):
            return (
                FormulaToken.from_symbol("Na", 1, 1),
                FormulaToken.from_symbol("Cl", -1, 1),
            )

        actions, metadata = compile_plan_actions(
            plan,
            benchmark_validator=benchmark_false,
            extended_assigner=extended,
        )
        state = replay_actions(actions)
        certificate = state.certificate(benchmark_validator=benchmark_false)
        self.assertEqual(certificate.certificate_class, CERTIFICATE_EXTENDED_ONLY)
        self.assertEqual(
            metadata["assignment_source"], "smact_adjacent_mixed_diagnostic"
        )

    def test_reachability_masks_dead_end_species_and_N(self):
        catalog = tuple(
            sorted(
                (
                    FormulaToken.from_symbol("O", -2, 1),
                    FormulaToken.from_symbol("O", -2, 2),
                    FormulaToken.from_symbol("Fe", 2, 1),
                )
            )
        )
        start_actions = legal_next_actions(
            CCFDv2State.start(),
            catalog,
            atom_count_choices=(1, 2),
            benchmark_validator=benchmark_true,
            max_species=2,
        )
        self.assertEqual(start_actions, (SetAtomCount(2),))
        state = CCFDv2State.start().apply(SetAtomCount(2))
        legal = legal_next_actions(
            state,
            catalog,
            benchmark_validator=benchmark_true,
            max_species=2,
        )
        self.assertIn(FormulaToken.from_symbol("O", -2, 1), legal)
        self.assertNotIn(FormulaToken.from_symbol("O", -2, 2), legal)

    def test_fast_typed_reachability_enforces_exact_charge_and_terminal_certificate(self):
        oxygen = FormulaToken.from_symbol("O", -2, 1)
        iron = FormulaToken.from_symbol("Fe", 2, 1)
        oracle = BenchmarkReachability(
            ((oxygen.atomic_number, oxygen.oxidation_state), (iron.atomic_number, iron.oxidation_state))
        )
        state = CCFDv2State.start().apply(SetAtomCount(2))
        legal = oracle.legal_species_counts(
            state,
            benchmark_validator=benchmark_true,
            max_species=2,
        )
        self.assertIn(oxygen, legal)
        self.assertNotIn(FormulaToken.from_symbol("O", -2, 2), legal)
        after_oxygen = state.apply(oxygen)
        self.assertTrue(oracle.can_complete(after_oxygen, max_species=2))
        self.assertEqual(
            oracle.legal_species_counts(
                after_oxygen,
                benchmark_validator=benchmark_true,
                max_species=2,
            ),
            (iron,),
        )

    def test_exact_arity_rejects_chemically_complete_shorter_path(self):
        oxygen = FormulaToken.from_symbol("O", -2, 2)
        iron = FormulaToken.from_symbol("Fe", 2, 2)
        oracle = BenchmarkReachability(
            ((oxygen.atomic_number, oxygen.oxidation_state), (iron.atomic_number, iron.oxidation_state))
        )
        state = CCFDv2State.start().apply(SetAtomCount(4)).apply(oxygen)
        self.assertIn(
            iron,
            oracle.legal_species_counts(
                state,
                benchmark_validator=benchmark_true,
                max_species=3,
            ),
        )
        self.assertNotIn(
            iron,
            oracle.legal_species_counts(
                state,
                benchmark_validator=benchmark_true,
                max_species=3,
                target_arity=3,
            ),
        )

    def test_plan_compiler_preserves_N_composition_and_rich_schema(self):
        plan = {
            "N": 2,
            "elements": ["Na", "Cl"],
            "counts": [1, 1],
            "anion_framework": "halide",
            "charge_bucket": "neutral_plausible",
            "lattice_system": "cubic",
            "spacegroup_bucket": "sg_195_230",
            "volume_per_atom_bin": "volpa_010_014",
        }

        def nacl_validator(_elements, _counts):
            return {
                "valid": True,
                "reason": "charge_neutral_pauling_valid",
                "oxidation_states": (1, -1),
            }

        actions, metadata = compile_plan_actions(plan, benchmark_validator=nacl_validator)
        state = replay_actions(actions)
        certificate = state.certificate(benchmark_validator=nacl_validator)
        self.assertEqual(certificate.formula, "NaCl")
        self.assertEqual(certificate.target_atoms, 2)
        self.assertEqual(metadata["assignment_source"], "benchmark_oxidation_witness")
        rich = render_rich_plan(
            state,
            soft_fields=plan,
            benchmark_validator=nacl_validator,
        )
        self.assertEqual(
            rich.splitlines(),
            [
                "formula: NaCl",
                "anion: halide",
                "charge: neutral_plausible",
                "lattice: cubic",
                "spacegroup: sg_195_230",
                "volume: volpa_010_014",
                "end: plan",
            ],
        )

    def test_plan_compiler_rejects_redundant_N_mismatch(self):
        plan = {"N": 3, "elements": ["Na", "Cl"], "counts": [1, 1]}
        with self.assertRaisesRegex(ValueError, "does not equal"):
            compile_plan_actions(plan, benchmark_validator=benchmark_false)


if __name__ == "__main__":
    unittest.main()
