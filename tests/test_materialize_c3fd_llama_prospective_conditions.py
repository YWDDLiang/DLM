import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "materialize_c3fd_llama_prospective_conditions",
    ROOT / "scripts/materialize_c3fd_llama_prospective_conditions.py",
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot import prospective condition materializer")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ProspectiveConditionMaterializerTest(unittest.TestCase):
    def vocabulary(self):
        return {
            "species": [
                {"id": 0, "atomic_number": 11, "oxidation_state": 1},
                {"id": 1, "atomic_number": 17, "oxidation_state": -1},
            ],
            "soft_vocabulary": {
                "anion_framework": ["oxide", "halide"],
                "lattice_system": ["cubic"],
                "spacegroup_bucket": ["sg_195_230"],
                "volume_per_atom_bin": ["volpa_020_024"],
            },
        }

    def record(self):
        return {
            "sample_idx": 9,
            "target_proposal": {"N": 2, "arity": 2, "family": "halide"},
            "semantic_trace": [
                {"action": "proposal", "N": 2, "arity": 2, "family": "halide"},
                {"action": "species", "atomic_number": 11, "oxidation_state": 1, "count": 1},
                {"action": "species", "atomic_number": 17, "oxidation_state": -1, "count": 1},
                {"action": "EOS"},
            ],
            "certificate": {"certificate_class": "benchmark_compatible"},
            "plan_state": {"N": 2, "elements": ["Na", "Cl"], "counts": [1, 1]},
        }

    def test_reconstructs_complete_semantic_state(self):
        semantic, history, species, counts = MODULE.state_from_record(
            self.record(), self.vocabulary()
        )
        self.assertEqual(species, [0, 1])
        self.assertEqual(counts, [1, 1])
        self.assertEqual(history[-1].net_charge, 0)
        self.assertEqual(semantic["proposal_targets"]["family"], 1)

    def test_source_contains_no_outcome_access(self):
        source = (
            ROOT / "scripts/materialize_c3fd_llama_prospective_conditions.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("e_above_hull", source)
        self.assertIn('"outcomes_read": False', source)


if __name__ == "__main__":
    unittest.main()
