import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover
    raise unittest.SkipTest("PyTorch is required") from exc


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "combine_btrd_proposal_graphs",
    ROOT / "scripts" / "combine_btrd_proposal_graphs.py",
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot import BTRD graph combiner")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class BTRDGraphCombineTest(unittest.TestCase):
    def test_global_indices_and_failures_are_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dirs = []
            for shard, indices in enumerate(((0, 1), (2, 3))):
                body = root / f"body{shard}"
                body.mkdir()
                rows = []
                graphs = []
                for local, index in enumerate(indices):
                    parsed = index != 1
                    rows.append(
                        {
                            "sample_idx": local,
                            "source_sample_idx": index,
                            "parsed": parsed,
                            "reason": None if parsed else "synthetic",
                        }
                    )
                    if parsed:
                        graphs.append({"value": index})
                (body / "raw_generations.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in rows)
                )
                torch.save(graphs, body / "proposal_graphs.pt")
                dirs.append(body)
            accounting, graphs = MODULE.combine(dirs, expected_requested=4)
            self.assertEqual([row["btrd_index"] for row in accounting], [0, 1, 2, 3])
            self.assertEqual([graph["sample_idx"] for graph in graphs], [0, 2, 3])
            self.assertFalse(accounting[1]["parsed"])


if __name__ == "__main__":
    unittest.main()
