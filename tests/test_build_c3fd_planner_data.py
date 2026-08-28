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
        self.assertEqual(manifest["benchmark_in_vocab_rate"], 1.0)

    def test_oov_node_fails_closed_without_dropping_row(self):
        encoded, manifest = MODULE.encode_rows(
            [self.row(((8, -2), (27, 2)))], self.vocabulary()
        )
        self.assertEqual(len(encoded), 1)
        self.assertFalse(encoded[0]["composition_supervision"])
        self.assertEqual(manifest["benchmark_in_vocab_rate"], 0.0)
        self.assertEqual(manifest["oov_nodes"], {"27|2": 1})


if __name__ == "__main__":
    unittest.main()
