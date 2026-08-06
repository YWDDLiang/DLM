#!/usr/bin/env python3
"""Merge immutable per-seed attempt shards without selection or replacement."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-total", type=int, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("attempt shard merging must run through Slurm CPU")
    output = args.output.resolve()
    manifest_path = args.manifest.resolve()
    for path in (output, manifest_path):
        if path.exists():
            raise FileExistsError(f"merge output is immutable: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    seen_attempts: set[str] = set()
    seen_pairs: set[str] = set()
    method: str | None = None
    rows = 0
    sources = []
    with output.open("xb") as destination:
        for input_path in args.input:
            source = input_path.resolve()
            source_rows = 0
            with source.open("rb") as handle:
                for line_number, raw in enumerate(handle, start=1):
                    if not raw.strip():
                        continue
                    row = json.loads(raw)
                    if row.get("schema") != "wqcodiff_generation_attempt_v1":
                        raise ValueError(f"{source}:{line_number}: invalid generation schema")
                    attempt_id = str(row.get("attempt_id", ""))
                    pair_id = str(row.get("pair_id", ""))
                    if not attempt_id or attempt_id in seen_attempts:
                        raise ValueError("missing/duplicate attempt ID across shards")
                    if not pair_id or pair_id in seen_pairs:
                        raise ValueError("missing/duplicate pair ID across shards")
                    observed_method = str(row.get("method", ""))
                    if method is None:
                        method = observed_method
                    elif observed_method != method:
                        raise ValueError("attempt shards mix methods")
                    if bool(row.get("retry_or_replacement_used")):
                        raise ValueError("attempt shard contains retry/replacement")
                    seen_attempts.add(attempt_id)
                    seen_pairs.add(pair_id)
                    destination.write(raw if raw.endswith(b"\n") else raw + b"\n")
                    rows += 1
                    source_rows += 1
            sources.append(
                {
                    "path": str(source),
                    "sha256": _sha(source),
                    "rows": source_rows,
                }
            )
        destination.flush()
        os.fsync(destination.fileno())
    if rows != args.expected_total:
        raise ValueError("merged attempt denominator differs from --expected-total")
    manifest = {
        "schema": "crysllmgen_attempt_shard_merge_v1",
        "method": method,
        "rows": rows,
        "output": str(output),
        "output_sha256": _sha(output),
        "sources": sources,
        "selection": "none_all_rows_in_argument_order",
        "retry_or_replacement_used": False,
    }
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
