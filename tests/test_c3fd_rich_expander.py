import copy
import unittest

from crystal_dlm.c3fd_rich_expander import (
    FEATURE_DIM,
    SoftPrefixProjector,
    SoftPrefixProjectorConfig,
    assemble_expanded_plan,
    pack_soft_prefix_features,
    rich_suffix_from_plan_state,
)


class C3FDRichExpanderTest(unittest.TestCase):
    def plan(self):
        return {
            "N": 2,
            "elements": ["Na", "Cl"],
            "counts": [1, 1],
            "anion_framework": "halide",
            "charge_bucket": "neutral_plausible",
            "lattice_system": "cubic",
            "spacegroup_bucket": "sg_195_230",
            "volume_per_atom_bin": "volpa_020_024",
        }

    def vocabulary(self):
        return {
            "species": [
                {"id": 0, "atomic_number": 11, "oxidation_state": 1},
                {"id": 1, "atomic_number": 17, "oxidation_state": -1},
            ],
            "soft_vocabulary": {
                "lattice_system": ["cubic", "hexagonal"],
                "spacegroup_bucket": ["sg_195_230", "sg_168_194"],
                "volume_per_atom_bin": ["volpa_020_024", "volpa_025_029"],
            },
        }

    def semantic(self):
        return {
            "certificate_class": "benchmark_compatible",
            "composition_supervision": True,
            "proposal_supervision": True,
            "proposal_targets": {"N": 2, "arity": 2, "family": 1},
            "species_labels": [0, 1],
            "count_targets": [1, 1],
            "ledger_steps": [
                {"remaining_atoms": 2, "net_charge": 0, "branch": "unset"},
                {"remaining_atoms": 0, "net_charge": 0, "branch": "ionic"},
            ],
        }

    def predicted(self):
        return {
            "predictions_by_checkpoint": {
                checkpoint: {
                    "lattice_system": {
                        "prediction": "cubic",
                        "confidence": 0.8,
                    },
                    "spacegroup_bucket": {
                        "prediction": "sg_195_230",
                        "confidence": 0.7,
                    },
                    "volume_per_atom_bin": {
                        "prediction": "volpa_020_024",
                        "confidence": 0.6,
                    },
                }
                for checkpoint in ("seed17", "seed18")
            }
        }

    def test_suffix_roundtrip_preserves_formula(self):
        suffix = rich_suffix_from_plan_state(self.plan())
        result = assemble_expanded_plan(self.plan(), suffix)
        self.assertEqual(result["plan_state"]["formula"], "NaCl")
        self.assertEqual(len(result["plan_text"].splitlines()), 7)

    def test_missing_field_is_not_filled(self):
        suffix = rich_suffix_from_plan_state(self.plan())
        suffix = "\n".join(
            line for line in suffix.splitlines() if not line.startswith("volume:")
        )
        with self.assertRaises(ValueError):
            assemble_expanded_plan(self.plan(), suffix)

    def test_feature_vector_is_fixed_and_ignores_teacher_rich_values(self):
        first = pack_soft_prefix_features(
            self.semantic(), self.vocabulary(), self.predicted()
        )
        changed = copy.deepcopy(self.semantic())
        changed["plan_state"] = {
            "lattice_system": "triclinic",
            "spacegroup_bucket": "sg_001_002",
            "volume_per_atom_bin": "volpa_000_004",
        }
        second = pack_soft_prefix_features(
            changed, self.vocabulary(), self.predicted()
        )
        self.assertEqual(len(first), FEATURE_DIM)
        self.assertEqual(first, second)

    @unittest.skipIf(SoftPrefixProjector is None, "PyTorch unavailable")
    def test_projector_shape(self):
        import torch

        config = SoftPrefixProjectorConfig(
            prefix_length=3, model_hidden_dim=16, projector_hidden_dim=8
        )
        projector = SoftPrefixProjector(config)
        output = projector(torch.zeros((2, FEATURE_DIM)))
        self.assertEqual(tuple(output.shape), (2, 3, 16))


if __name__ == "__main__":
    unittest.main()
