#!/usr/bin/env python3
"""Collect every reconstructed chemical system for a fresh official MP query."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import protocol


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--cell", action="append", required=True, help="cell_id=attempt_labels_preofficial.jsonl")
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    cells: list[dict[str, object]] = []
    wanted: set[str] = set()
    for item in args.cell:
        cell_id, separator, raw_path = item.partition("=")
        if not separator or not cell_id:
            raise protocol.ContractError("--cell must be cell_id=path")
        path = Path(raw_path).resolve()
        rows = protocol.read_jsonl(path)
        chemsys = {
            str(row["chemsys"])
            for row in rows
            if row.get("reconstructed") is True and row.get("chemsys")
        }
        reconstructed = sum(row.get("reconstructed") is True for row in rows)
        if len(chemsys) == 0 or any(row.get("retry_or_replacement_used") is not False for row in rows):
            raise protocol.ContractError(f"preofficial cell is malformed: {cell_id}")
        wanted.update(chemsys)
        cells.append(
            {
                "cell_id": cell_id,
                "attempts": len(rows),
                "reconstructed": reconstructed,
                "unique_chemsys": len(chemsys),
                "labels": protocol.identity(path),
            }
        )
    final = run_root / "inputs"
    if final.exists():
        raise FileExistsError(final)
    preparing = run_root / ".inputs.preparing"
    failed = run_root / ".inputs.FAILED"
    preparing.mkdir(parents=True, exist_ok=False)
    try:
        wanted_rows = [
            {"query_index": index, "chemsys": chemsys, "elements": chemsys.split("-")}
            for index, chemsys in enumerate(sorted(wanted))
        ]
        protocol.write_jsonl_exclusive(preparing / "wanted_chemsys.jsonl", wanted_rows)
        manifest = {
            "schema": "h1a2_epoch2_exactplan_official_input_manifest_v1",
            "cell_count": len(cells),
            "cells": cells,
            "wanted_chemsys_count": len(wanted_rows),
            "wanted_chemsys_sha256": protocol.canonical_sha256(wanted_rows),
            "stability_scope": "all_reconstructed_before_NU_intersection",
            "fresh_official_query": True,
            "historical_cache_reuse": False,
        }
        protocol.write_json_exclusive(preparing / "input_manifest.json", manifest)
        (preparing / "inputs_SUCCESS").touch(exist_ok=False)
        preparing.rename(final)
    except Exception:
        if preparing.exists():
            shutil.move(str(preparing), str(failed))
        raise
    print(protocol.canonical_json({"cells": len(cells), "wanted_chemsys": len(wanted)}), flush=True)


if __name__ == "__main__":
    main()
