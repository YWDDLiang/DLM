import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "eval_runtime" / "exact_raw_reuse.py"


def load_module():
    specification = importlib.util.spec_from_file_location(
        "exact_raw_reuse_test_module", MODULE_PATH
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


class ExactRawReuseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reuse = load_module()

    def test_canonical_identity_ignores_mapping_order_but_not_small_geometry_changes(self):
        structure = {
            "@module": "pymatgen.core.structure",
            "lattice": {
                "matrix": [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            },
            "sites": [
                {
                    "species": [{"element": "Li", "occu": 1}],
                    "abc": [0.1, 0.2, 0.3],
                }
            ],
        }
        reordered = {
            "sites": [
                {
                    "abc": [0.1, 0.2, 0.3],
                    "species": [{"occu": 1, "element": "Li"}],
                }
            ],
            "lattice": {
                "matrix": [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            },
            "@module": "pymatgen.core.structure",
        }
        near_but_not_equal = {
            **structure,
            "sites": [
                {
                    "species": [{"element": "Li", "occu": 1}],
                    "abc": [0.1, 0.2, 0.300000000001],
                }
            ],
        }

        identity = self.reuse.canonical_structure_sha256(structure)
        self.assertEqual(identity, self.reuse.canonical_structure_sha256(reordered))
        self.assertNotEqual(
            identity, self.reuse.canonical_structure_sha256(near_but_not_equal)
        )

    def test_pair_plan_counts_exact_hits_and_assigns_each_unique_identity_once(self):
        identities = [f"{value:064x}" for value in range(1, 5)]
        views = {
            "F": self.reuse.ManifestIdentities(
                total_attempts=4,
                reconstructed=(identities[0], identities[1], identities[0]),
            ),
            "M": self.reuse.ManifestIdentities(
                total_attempts=4,
                reconstructed=(
                    identities[1],
                    identities[2],
                    identities[1],
                    identities[3],
                ),
            ),
        }

        plan = self.reuse.build_pair_plan(views)

        self.assertEqual(plan.unique_identities, tuple(identities))
        self.assertEqual(plan.pair_attempts, 8)
        self.assertEqual(plan.pair_reconstructed, 7)
        self.assertEqual(plan.cache_misses, 4)
        self.assertEqual(plan.cache_hits, 3)
        owned = plan.owned_identities("F") + plan.owned_identities("M")
        self.assertCountEqual(owned, identities)
        self.assertEqual(len(owned), len(set(owned)))
        self.assertEqual(plan.owner_for(identities[0]), "F")
        self.assertEqual(plan.owner_for(identities[2]), "M")
        self.assertEqual(plan.owner_for(identities[3]), "M")

    def test_manifest_view_excludes_failed_attempts_without_changing_denominator(self):
        first = "1" * 64
        second = "2" * 64
        manifest = {
            "schema": "crysllmgen_r5c_a100_input_manifest_v1",
            "total_attempts": 4,
            "reconstructed_structures": 2,
            "attempt_records": [
                {
                    "generation_ordinal": 0,
                    "reconstructed_index": 0,
                    "status": "succeeded",
                    "structure_sha256": first,
                },
                {
                    "generation_ordinal": 1,
                    "reconstructed_index": None,
                    "status": "failed",
                },
                {
                    "generation_ordinal": 2,
                    "reconstructed_index": 1,
                    "status": "succeeded",
                    "structure_sha256": second,
                },
                {
                    "generation_ordinal": 3,
                    "reconstructed_index": None,
                    "status": "failed",
                },
            ],
        }

        view = self.reuse.manifest_identities(manifest)

        self.assertEqual(view.total_attempts, 4)
        self.assertEqual(view.reconstructed, (first, second))

    def test_results_map_back_to_every_original_reconstructed_row_in_order(self):
        first = "1" * 64
        second = "2" * 64
        expected_first = (None, {"Li": 1.0})
        expected_second = (-2.5, {"O": 2.0})

        mapped = self.reuse.map_results_to_identities(
            (first, second, first, first),
            {first: expected_first, second: expected_second},
        )

        self.assertEqual(
            mapped,
            [expected_first, expected_second, expected_first, expected_first],
        )


if __name__ == "__main__":
    unittest.main()
