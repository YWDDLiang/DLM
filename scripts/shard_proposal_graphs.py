#!/usr/bin/env python3
"""Split proposal graphs into deterministic interleaved refinement shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposal-graphs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    args = parser.parse_args()
    if args.shard_count <= 0:
        raise ValueError("shard-count must be positive")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    graphs = torch.load(args.proposal_graphs, map_location="cpu")
    if not isinstance(graphs, list) or not graphs:
        raise ValueError("proposal graph payload must be a nonempty list")
    sample_indices = [int(graph["sample_idx"]) for graph in graphs]
    if len(set(sample_indices)) != len(sample_indices):
        raise ValueError("proposal sample_idx values must be unique")
    args.output_dir.mkdir(parents=True)
    shards = []
    for rank in range(args.shard_count):
        values = graphs[rank :: args.shard_count]
        path = args.output_dir / f"shard-{rank:03d}.pt"
        torch.save(values, path)
        shards.append(
            {
                "rank": rank,
                "path": str(path),
                "count": len(values),
                "sample_indices": [int(graph["sample_idx"]) for graph in values],
            }
        )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "proposal_graph_refinement_shards_v1",
                "source": str(args.proposal_graphs.resolve()),
                "total": len(graphs),
                "shard_count": args.shard_count,
                "shards": shards,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
