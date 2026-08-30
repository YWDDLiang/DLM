import importlib.util
from pathlib import Path
import unittest

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover
    raise unittest.SkipTest("PyTorch is required for C3FD sampling tests") from exc

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "sample_c3fd_plans", ROOT / "scripts" / "sample_c3fd_plans.py"
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot import sample_c3fd_plans.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

from crystal_dlm.ccfd import FormulaToken
from crystal_dlm.ccfd_v2 import CCFDv2State, SetAtomCount


class SampleC3FDPlansTest(unittest.TestCase):
    @staticmethod
    def benchmark_true(_elements, _counts):
        return {"valid": True, "reason": "synthetic_joint_reachability"}

    def test_semantic_history_starts_with_locked_N(self):
        start = CCFDv2State.start().apply(SetAtomCount(5))
        after_first = start.apply(FormulaToken.from_symbol("O", -2, 2))
        after_second = after_first.apply(FormulaToken.from_symbol("Fe", 2, 2))
        species, counts, n_values, ledger, target = MODULE.semantic_inputs(
            5,
            [3, 4],
            [2, 2],
            state_history=[start, after_first, after_second],
            target_arity=3,
        )
        self.assertEqual(target, 3)
        self.assertTrue(torch.equal(species, torch.tensor([[-1, -1, 3, 4]])))
        self.assertTrue(torch.equal(counts, torch.tensor([[0, 0, 2, 2]])))
        self.assertTrue(torch.equal(n_values, torch.tensor([[0, 5, 0, 0]])))
        self.assertEqual(tuple(ledger.shape), (1, 4, 6))
        self.assertTrue(torch.equal(ledger[0, 0], torch.zeros(6)))
        self.assertAlmostEqual(float(ledger[0, 1, 0]), 5 / 20)
        self.assertAlmostEqual(float(ledger[0, -1, 2]), 1 / 7)

    def test_family_prefix_policy_rejects_higher_priority_anion(self):
        self.assertFalse(MODULE.element_allowed_for_family("O", "sulfide"))
        self.assertTrue(MODULE.element_allowed_for_family("S", "sulfide"))
        self.assertFalse(MODULE.element_allowed_for_family("F", "nitride"))
        self.assertTrue(MODULE.element_allowed_for_family("N", "nitride"))

    def test_sampler_never_selects_negative_infinity(self):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(7)
        logits = torch.tensor([float("-inf"), 0.0, float("-inf")])
        for _ in range(10):
            self.assertEqual(
                MODULE.sample_index(
                    logits,
                    rng=generator,
                    temperature=0.9,
                    top_p=0.95,
                    top_k=50,
                ),
                1,
            )

    def test_independent_spacegroup_draw_preserves_lattice_volume_rng_prefix(self):
        values = {
            "lattice_system": ["triclinic", "cubic", "<UNKNOWN>"],
            "spacegroup_bucket": ["sg_001_002", "sg_195_230", "<UNKNOWN>"],
            "volume_per_atom_bin": ["volpa_005_009", "volpa_010_014", "<UNKNOWN>"],
        }
        logits = {
            "lattice_system": torch.tensor([0.1, 0.9, -2.0]),
            "spacegroup_bucket": torch.tensor([0.8, 0.2, -2.0]),
            "volume_per_atom_bin": torch.tensor([0.7, 0.3, -2.0]),
        }
        legacy_rng = torch.Generator(device="cpu")
        legacy_rng.manual_seed(17001)
        legacy_lattice = MODULE.sample_soft_value(
            logits["lattice_system"],
            values["lattice_system"],
            rng=legacy_rng,
            temperature=0.7,
            top_p=0.95,
            top_k=0,
        )
        legacy_volume = MODULE.sample_soft_value(
            logits["volume_per_atom_bin"],
            values["volume_per_atom_bin"],
            rng=legacy_rng,
            temperature=0.7,
            top_p=0.95,
            top_k=0,
        )

        corrected_rng = torch.Generator(device="cpu")
        corrected_rng.manual_seed(17001)
        corrected = MODULE.sample_structural_soft_fields(
            logits,
            values,
            rng=corrected_rng,
            temperature=0.7,
            top_p=0.95,
            top_k=0,
            spacegroup_mode="independent_head",
        )
        self.assertEqual(corrected["lattice_system"], legacy_lattice)
        self.assertEqual(corrected["volume_per_atom_bin"], legacy_volume)
        self.assertIn(corrected["spacegroup_bucket"], values["spacegroup_bucket"][:-1])

    def test_legacy_compiler_mode_remains_exact(self):
        rng = torch.Generator(device="cpu")
        rng.manual_seed(9)
        result = MODULE.sample_structural_soft_fields(
            {
                "lattice_system": torch.tensor([float("-inf"), 0.0]),
                "spacegroup_bucket": torch.tensor([0.0, 0.0]),
                "volume_per_atom_bin": torch.tensor([0.0]),
            },
            {
                "lattice_system": ["triclinic", "cubic"],
                "spacegroup_bucket": ["sg_001_002", "sg_195_230"],
                "volume_per_atom_bin": ["volpa_005_009"],
            },
            rng=rng,
            temperature=0.7,
            top_p=0.95,
            top_k=0,
            spacegroup_mode="lattice_compiler",
        )
        self.assertEqual(result["lattice_system"], "cubic")
        self.assertEqual(result["spacegroup_bucket"], "sg_195_230")

    def test_family_aware_oracle_rejects_split_reachability_false_positive(self):
        lithium = FormulaToken.from_symbol("Li", 1, 1)
        oxygen = FormulaToken.from_symbol("O", -2, 1)
        fluorine = FormulaToken.from_symbol("F", -1, 1)
        iron = FormulaToken.from_symbol("Fe", 2, 1)
        nodes = tuple(
            MODULE.ValenceNode(token.atomic_number, token.oxidation_state)
            for token in (lithium, oxygen, fluorine, iron)
        )
        state = CCFDv2State.start().apply(SetAtomCount(2))

        # The old split checks admit Li: F can close charge, while O exists as
        # a separate family witness.  No single suffix can satisfy both.
        generic = MODULE.BenchmarkReachability(
            tuple((node.atomic_number, node.oxidation_state) for node in nodes)
        )
        self.assertIn(
            lithium,
            generic.legal_species_counts(
                state,
                benchmark_validator=self.benchmark_true,
                max_species=2,
                target_arity=2,
            ),
        )
        self.assertTrue(
            MODULE.family_prefix_reachable(
                state.apply(lithium),
                family="oxide",
                target_arity=2,
                vocabulary_nodes=nodes,
            )
        )

        joint = MODULE.FamilyAwareBenchmarkReachability(nodes)
        legal = joint.legal_species_counts(
            state,
            family="oxide",
            benchmark_validator=self.benchmark_true,
            max_species=2,
            target_arity=2,
        )
        self.assertNotIn(lithium, legal)
        self.assertIn(oxygen, legal)


if __name__ == "__main__":
    unittest.main()
