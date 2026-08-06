import unittest

from crystal_dlm.distributed_data import (
    audit_distributed_index_shards,
    strided_shard_indices,
)


class DistributedDataTests(unittest.TestCase):
    def test_odd_validation_panel_has_no_padding_duplicate(self):
        shards = [
            strided_shard_indices(9047, num_replicas=2, rank=rank)
            for rank in range(2)
        ]
        report = audit_distributed_index_shards(9047, shards)
        self.assertEqual([len(shard) for shard in shards], [4524, 4523])
        self.assertEqual(report["total_assigned"], 9047)
        self.assertEqual(report["unique_assigned"], 9047)
        self.assertEqual(report["duplicate_count"], 0)
        self.assertEqual(report["missing_count"], 0)
        self.assertTrue(report["gate_passed"])

    def test_even_smoke_panel_is_split_exactly(self):
        shards = [
            strided_shard_indices(32, num_replicas=2, rank=rank)
            for rank in range(2)
        ]
        report = audit_distributed_index_shards(32, shards)
        self.assertEqual([len(shard) for shard in shards], [16, 16])
        self.assertEqual(shards[0][:3], [0, 2, 4])
        self.assertEqual(shards[1][:3], [1, 3, 5])
        self.assertTrue(report["gate_passed"])

    def test_duplicate_or_missing_index_fails_closed(self):
        report = audit_distributed_index_shards(4, [[0, 2], [1, 2]])
        self.assertEqual(report["duplicate_indices"], [2])
        self.assertEqual(report["missing_indices"], [3])
        self.assertFalse(report["gate_passed"])

    def test_rank_order_drift_fails_even_with_exact_cover(self):
        report = audit_distributed_index_shards(4, [[2, 0], [1, 3]])
        self.assertTrue(report["exact_cover"])
        self.assertFalse(report["rank_mapping_exact"])
        self.assertFalse(report["gate_passed"])

    def test_invalid_rank_is_rejected(self):
        with self.assertRaises(ValueError):
            strided_shard_indices(4, num_replicas=2, rank=2)


if __name__ == "__main__":
    unittest.main()
