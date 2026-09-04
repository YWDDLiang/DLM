import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "relax_spad_basin_closure_shard.py"
SPEC = importlib.util.spec_from_file_location("relax_spad_basin_closure_shard_test", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class RelaxShardTest(unittest.TestCase):
    def test_key_owner_is_deterministic_and_duplicate_safe(self):
        keys = [f"{index:064x}" for index in range(1, 257)]
        owners = [MODULE.owner_for_key(key, shard_count=4) for key in keys]
        self.assertTrue(all(0 <= owner < 4 for owner in owners))
        self.assertEqual(
            MODULE.owner_for_key(keys[17], shard_count=4),
            MODULE.owner_for_key(keys[17], shard_count=4),
        )

    def test_generation_validation_retains_exact_denominator(self):
        rows = [
            {
                "ordinal": index,
                "sample_idx": index,
                "status": "succeeded",
                "structure": {"index": index},
            }
            for index in range(4)
        ]
        MODULE.validate_generation(rows, denominator=4)
        with self.assertRaises(ValueError):
            MODULE.validate_generation(rows[:-1], denominator=4)

    def test_invalid_shard_is_rejected(self):
        with self.assertRaises(ValueError):
            MODULE.owner_for_key("0" * 64, shard_count=0)

    def test_manifest_maps_frozen_structures_back_to_ordinals(self):
        structures = ["second", "first"]
        manifest = {
            "total_attempts": 2,
            "reconstructed_structures": 2,
            "attempt_records": [
                {"generation_ordinal": 0, "reconstructed_index": 1},
                {"generation_ordinal": 1, "reconstructed_index": 0},
            ],
        }
        self.assertEqual(
            MODULE.map_structures_to_ordinals(structures, manifest, denominator=2),
            {0: "first", 1: "second"},
        )


if __name__ == "__main__":
    unittest.main()
