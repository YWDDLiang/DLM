import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_c3fd_planner_data", ROOT / "scripts" / "build_c3fd_planner_data.py"
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot import build_c3fd_planner_data.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class BuildC3FDPlannerDataTest(unittest.TestCase):
    def vocabulary(self):
        return {
            "species": [
                {"id": 0, "atomic_number": 8, "oxidation_state": -2},
                {"id": 1, "atomic_number": 26, "oxidation_state": 2},
            ],
            "soft_vocabulary": {
                field: ["known", MODULE.UNKNOWN_SOFT] for field in MODULE.SOFT_FIELDS
            },
        }

    def row(self, nodes):
        return {
            "source_row_idx": 1,
            "sample_weight": 1.0,
            "plan_state": {},
            "composition_supervision": True,
            "certificate_class": "benchmark_compatible",
            "compile_error": None,
            "N": 2,
            "nodes": [
                {"atomic_number": atomic_number, "oxidation_state": oxidation}
                for atomic_number, oxidation in nodes
            ],
            "counts": [1] * len(nodes),
            "soft_values": {field: "known" for field in MODULE.SOFT_FIELDS},
        }

    def test_known_nodes_keep_composition_supervision(self):
        encoded, manifest = MODULE.encode_rows(
            [self.row(((8, -2), (26, 2)))], self.vocabulary()
        )
        self.assertTrue(encoded[0]["composition_supervision"])
        self.assertEqual(encoded[0]["species_labels"], [0, 1])
        self.assertEqual(encoded[0]["N_target"], 2)
        self.assertEqual(encoded[0]["count_targets"], [1, 1])
        self.assertEqual(
            encoded[0]["proposal_targets"],
            {"N": 2, "arity": 2, "family": 0},
        )
        self.assertEqual(
            encoded[0]["ledger_steps"],
            [
                {
                    "remaining_atoms": 2,
                    "net_charge": 0,
                    "remaining_species": 2,
                    "branch": "unset",
                },
                {
                    "remaining_atoms": 2,
                    "net_charge": 0,
                    "remaining_species": 2,
                    "branch": "unset",
                },
                {
                    "remaining_atoms": 1,
                    "net_charge": -2,
                    "remaining_species": 1,
                    "branch": "ionic",
                },
                {
                    "remaining_atoms": 0,
                    "net_charge": 0,
                    "remaining_species": 0,
                    "branch": "ionic",
                },
            ],
        )
        self.assertEqual(manifest["benchmark_in_vocab_rate"], 1.0)

    def test_oov_node_fails_closed_without_dropping_row(self):
        encoded, manifest = MODULE.encode_rows(
            [self.row(((8, -2), (27, 2)))], self.vocabulary()
        )
        self.assertEqual(len(encoded), 1)
        self.assertFalse(encoded[0]["composition_supervision"])
        self.assertEqual(manifest["benchmark_in_vocab_rate"], 0.0)
        self.assertEqual(manifest["oov_nodes"], {"27|2": 1})
        self.assertEqual(encoded[0]["proposal_targets"]["arity"], 2)

    def test_compile_failure_preserves_proposal_only_targets(self):
        # The redundant N mismatch deliberately prevents semantic compilation,
        # but family/N/arity remain valid proposal supervision.
        raw = {
            "plan_state": {
                "N": 3,
                "elements": ["O"],
                "counts": [2],
                "anion_framework": "known",
                "charge_bucket": "known",
                "lattice_system": "known",
                "spacegroup_bucket": "known",
                "volume_per_atom_bin": "known",
            }
        }
        compiled = MODULE.compile_row(raw, 0)
        self.assertFalse(compiled["composition_supervision"])
        self.assertEqual(compiled["N"], 3)
        encoded, _manifest = MODULE.encode_rows([compiled], self.vocabulary())
        self.assertTrue(encoded[0]["proposal_supervision"])
        self.assertEqual(
            encoded[0]["proposal_targets"],
            {"N": 3, "arity": 1, "family": 0},
        )


if __name__ == "__main__":
    unittest.main()
