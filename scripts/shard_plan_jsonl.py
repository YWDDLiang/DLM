#!/usr/bin/env python3
"""Shard a frozen Plan JSONL while preserving parent/source identities."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def shard_rows(rows, *, shard_size: int):
    if shard_size <= 0:
        raise ValueError("shard_size must be positive")
    if [int(row["sample_idx"]) for row in rows] != list(range(len(rows))):
        raise ValueError("input Plan sample_idx must be contiguous")
    shards = []
    for start in range(0, len(rows), shard_size):
        values = []
        for local_idx, source in enumerate(rows[start : start + shard_size]):
            row = dict(source)
            row["parent_execution_sample_idx"] = int(source["sample_idx"])
            row["sample_idx"] = local_idx
            row["shard_local_sample_idx"] = local_idx
            values.append(row)
        shards.append(values)
    return shards


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-size", type=int, required=True)
    args = parser.parse_args()
    source = args.input_jsonl.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    rows = read_jsonl(source)
    shards = shard_rows(rows, shard_size=args.shard_size)
    output.mkdir(parents=True)
    files = []
    manifest_shards = []
    for index, values in enumerate(shards):
        path = output / f"shard-{index:03d}.jsonl"
        path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in values),
            encoding="utf-8",
        )
        files.append(path)
        manifest_shards.append(
            {"index": index, "rows": len(values), "sha256": sha256_file(path)}
        )
    manifest = {
        "schema": "fixed_plan_shards_v1",
        "status": "complete",
        "source": str(source),
        "source_sha256": sha256_file(source),
        "rows": len(rows),
        "shard_size": int(args.shard_size),
        "shards": manifest_shards,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    files.append(manifest_path)
    (output / "SHA256SUMS").write_text(
        "".join(f"{sha256_file(path)}  {path.name}\n" for path in files),
        encoding="utf-8",
    )
    (output / "_SUCCESS").touch()
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
