import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "shard_plan_jsonl", ROOT / "scripts" / "shard_plan_jsonl.py"
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot import Plan sharder")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PlanShardTest(unittest.TestCase):
    def test_parent_and_source_identity_survive_local_reindex(self) -> None:
        rows = [
            {"sample_idx": index, "source_sample_idx": index + 100, "plan_state": {}}
            for index in range(5)
        ]
        shards = MODULE.shard_rows(rows, shard_size=2)
        self.assertEqual([len(shard) for shard in shards], [2, 2, 1])
        self.assertEqual([row["sample_idx"] for row in shards[1]], [0, 1])
        self.assertEqual(
            [row["parent_execution_sample_idx"] for row in shards[1]], [2, 3]
        )
        self.assertEqual([row["source_sample_idx"] for row in shards[1]], [102, 103])

    def test_noncontiguous_input_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            MODULE.shard_rows([{"sample_idx": 1}], shard_size=2)


if __name__ == "__main__":
    unittest.main()
