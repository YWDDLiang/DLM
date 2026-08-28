from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crystal_dlm.ccfd import FormulaToken
from crystal_dlm.ccfd_v2 import BenchmarkReachability, CCFDv2State, SetAtomCount
from crystal_dlm.composition_pair_prior import ValenceNode
from crystal_dlm.family_reachability import (
    FamilyAwareBenchmarkReachability,
    family_prefix_reachable,
)


def benchmark_true(_elements, _counts):
    return {"valid": True, "reason": "synthetic_joint_reachability"}


class FamilyReachabilityTest(unittest.TestCase):
    def test_joint_oracle_removes_split_family_charge_false_positive(self):
        lithium = FormulaToken.from_symbol("Li", 1, 1)
        oxygen = FormulaToken.from_symbol("O", -2, 1)
        fluorine = FormulaToken.from_symbol("F", -1, 1)
        iron = FormulaToken.from_symbol("Fe", 2, 1)
        nodes = tuple(
            ValenceNode(token.atomic_number, token.oxidation_state)
            for token in (lithium, oxygen, fluorine, iron)
        )
        state = CCFDv2State.start().apply(SetAtomCount(2))

        generic = BenchmarkReachability(
            tuple((node.atomic_number, node.oxidation_state) for node in nodes)
        )
        self.assertIn(
            lithium,
            generic.legal_species_counts(
                state,
                benchmark_validator=benchmark_true,
                max_species=2,
                target_arity=2,
            ),
        )
        self.assertTrue(
            family_prefix_reachable(
                state.apply(lithium),
                family="oxide",
                target_arity=2,
                vocabulary_nodes=nodes,
            )
        )

        joint = FamilyAwareBenchmarkReachability(nodes)
        legal = joint.legal_species_counts(
            state,
            family="oxide",
            benchmark_validator=benchmark_true,
            max_species=2,
            target_arity=2,
        )
        self.assertNotIn(lithium, legal)
        self.assertIn(oxygen, legal)

    def test_every_returned_action_retains_a_terminal_witness(self):
        nodes = tuple(
            ValenceNode(token.atomic_number, token.oxidation_state)
            for token in (
                FormulaToken.from_symbol("Li", 1, 1),
                FormulaToken.from_symbol("O", -2, 1),
                FormulaToken.from_symbol("F", -1, 1),
                FormulaToken.from_symbol("Fe", 2, 1),
            )
        )
        oracle = FamilyAwareBenchmarkReachability(nodes)
        state = CCFDv2State.start().apply(SetAtomCount(2))
        legal = oracle.legal_species_counts(
            state,
            family="oxide",
            target_arity=2,
            benchmark_validator=benchmark_true,
            max_species=2,
        )
        self.assertTrue(legal)
        for token in legal:
            self.assertTrue(
                oracle.can_complete(
                    state.apply(token),
                    family="oxide",
                    target_arity=2,
                    benchmark_validator=benchmark_true,
                    max_species=2,
                )
            )


if __name__ == "__main__":
    unittest.main()
