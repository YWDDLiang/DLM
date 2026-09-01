#!/usr/bin/env python3
"""Freeze an outcome-blind ordinal ledger for one fused-Planner sampling run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA = "fused_planner_request_ledger_v1"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def build_rows(*, requested: int, seed: int, purpose: str) -> list[dict[str, Any]]:
    if requested <= 0:
        raise ValueError("requested must be positive")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if not purpose.strip():
        raise ValueError("purpose must be nonempty")
    return [
        {
            "schema": SCHEMA,
            "ordinal": index,
            "sample_idx": index,
            "planner_sampling_seed": seed,
            "stability_goal": "meta_or_better",
            "purpose": purpose,
            "outcomes_read": False,
        }
        for index in range(requested)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--requested", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--purpose", required=True)
    args = parser.parse_args()

    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    rows = build_rows(requested=args.requested, seed=args.seed, purpose=args.purpose)
    output.mkdir(parents=True)
    ledger = output / "ledger.jsonl"
    write_jsonl(ledger, rows)
    manifest = {
        "schema": SCHEMA,
        "status": "complete",
        "requested": len(rows),
        "planner_sampling_seed": int(args.seed),
        "purpose": str(args.purpose),
        "outcomes_read": False,
        "selection": "none; one immutable ordinal per request",
        "ledger_sha256": sha256_file(ledger),
    }
    write_json(output / "manifest.json", manifest)
    (output / "SHA256SUMS").write_text(
        "".join(
            f"{sha256_file(path)}  {path.name}\n"
            for path in (ledger, output / "manifest.json")
        ),
        encoding="utf-8",
    )
    (output / "_SUCCESS").touch()
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
