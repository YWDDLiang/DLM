#!/usr/bin/env python3
"""Print a shell-safe descriptor for one frozen post-refine evaluation cell."""

from __future__ import annotations

import argparse
from pathlib import Path

from protocol import ContractError, load_upstream_cells, read_json, require_source_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--cell-index", type=int, required=True)
    args = parser.parse_args()
    source = args.source_dir.resolve()
    require_source_manifest(source, args.source_manifest_sha256)
    config = read_json(source / "CONFIG.json")
    _, cells = load_upstream_cells(config)
    index = int(args.cell_index)
    if index < 0 or index >= len(cells):
        raise ContractError("cell index out of range")
    cell = cells[index]
    values = [
        str(cell["cell_id"]),
        str(cell["panel"]),
        str(cell["cohort_id"]),
        str(cell["generation_jsonl"]),
    ]
    if any(not value or "\n" in value or "\r" in value for value in values):
        raise ContractError("unsafe cell descriptor")
    print("\n".join(values))


if __name__ == "__main__":
    main()
