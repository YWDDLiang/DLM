#!/usr/bin/env python3
"""Combine BTRD body shards into globally indexed model494 proposals."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def combine(body_dirs, *, expected_requested: int):
    accounting = []
    proposals = []
    seen: set[int] = set()
    for body_dir in body_dirs:
        rows = read_jsonl(body_dir / "raw_generations.jsonl")
        graphs = torch.load(body_dir / "proposal_graphs.pt", map_location="cpu")
        parsed_rows = [row for row in rows if row.get("parsed") is True]
        if len(parsed_rows) != len(graphs):
            raise ValueError(f"parsed/graph count mismatch: {body_dir}")
        graph_iter = iter(graphs)
        for row in rows:
            btrd_index = int(row["source_sample_idx"])
            if btrd_index in seen:
                raise ValueError("duplicate global BTRD index")
            seen.add(btrd_index)
            parsed = row.get("parsed") is True
            accounting.append(
                {
                    "btrd_index": btrd_index,
                    "parsed": parsed,
                    "reason": None if parsed else str(row.get("reason") or "body_failure"),
                    "body_dir": str(body_dir),
                }
            )
            if parsed:
                graph = dict(next(graph_iter))
                graph["sample_idx"] = btrd_index
                graph["source_sample_idx"] = btrd_index
                graph["btrd_index"] = btrd_index
                proposals.append((btrd_index, graph))
    if len(accounting) != expected_requested or seen != set(range(expected_requested)):
        raise ValueError("BTRD shard accounting does not cover the frozen denominator")
    proposals.sort(key=lambda item: item[0])
    accounting.sort(key=lambda row: row["btrd_index"])
    return accounting, [graph for _index, graph in proposals]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--body-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-requested", type=int, default=6144)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    body_dirs = [path.resolve() for path in args.body_dir]
    accounting, proposals = combine(
        body_dirs, expected_requested=args.expected_requested
    )
    output.mkdir(parents=True)
    graph_path = output / "proposal_graphs.pt"
    torch.save(proposals, graph_path)
    accounting_path = output / "all_requested_accounting.jsonl"
    accounting_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in accounting),
        encoding="utf-8",
    )
    manifest = {
        "schema": "btrd_combined_proposal_graphs_v1",
        "status": "complete",
        "requested": len(accounting),
        "parsed_graphs": len(proposals),
        "failed": len(accounting) - len(proposals),
        "body_dirs": [str(path) for path in body_dirs],
        "proposal_graphs_sha256": sha256_file(graph_path),
        "accounting_sha256": sha256_file(accounting_path),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "_SUCCESS").touch()
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
