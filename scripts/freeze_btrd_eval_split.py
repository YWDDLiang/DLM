#!/usr/bin/env python3
"""Freeze the BTRD first256 development and 903-row confirmation split."""

from __future__ import annotations

import argparse
import hashlib
import json
from math import gcd
from pathlib import Path


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def reduced_identity(row: dict) -> str:
    if row.get("reduced_composition_identity"):
        return str(row["reduced_composition_identity"])
    plan = row["plan_state"]
    pairs = sorted(
        (str(symbol), int(count))
        for symbol, count in zip(plan["elements"], plan["counts"], strict=True)
    )
    divisor = 0
    for _symbol, count in pairs:
        if count <= 0:
            raise ValueError("composition counts must be positive")
        divisor = gcd(divisor, count)
    return "|".join(f"{symbol}:{count // divisor}" for symbol, count in pairs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--known-splits", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--development-count", type=int, default=256)
    args = parser.parse_args()
    source = args.known_splits.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    main_path = source / "main1000/plans_for_dlm.jsonl"
    remainder_path = source / "remainder/plans_for_dlm.jsonl"
    main_rows = read_jsonl(main_path)
    remainder_rows = read_jsonl(remainder_path)
    if len(main_rows) != 1000 or len(remainder_rows) != 159:
        raise ValueError("official-known source split changed")
    rows = main_rows + remainder_rows
    if args.development_count != 256 or len(rows) != 1159:
        raise ValueError("BTRD evaluation denominator changed")
    identities = [reduced_identity(row) for row in rows]

    development = rows[: args.development_count]
    confirmation = rows[args.development_count :]
    output.mkdir(parents=True)
    (output / "development256").mkdir()
    (output / "confirmation903").mkdir()
    development_path = output / "development256/plans_for_dlm.jsonl"
    confirmation_path = output / "confirmation903/plans_for_dlm.jsonl"
    ledger_path = output / "ledger.jsonl"
    write_jsonl(development_path, development)
    write_jsonl(confirmation_path, confirmation)
    write_jsonl(
        ledger_path,
        [
            {
                "global_index": index,
                "role": "development" if index < args.development_count else "confirmation",
                "reduced_composition_identity": identities[index],
                "source_sample_idx": row.get("source_sample_idx"),
            }
            for index, row in enumerate(rows)
        ],
    )
    manifest = {
        "schema": "btrd_evaluation_split_v1",
        "status": "complete",
        "planner_requests": 1200,
        "official_known": 1159,
        "official_unavailable": 41,
        "development_rows": len(development),
        "confirmation_rows": len(confirmation),
        "unique_exact_identities": len(set(identities)),
        "selection_outcomes_read": False,
        "membership_rule": "first256 official-known order; remaining903 confirmation",
        "inputs": {
            "source_manifest_sha256": sha256_file(source / "manifest.json"),
            "main1000_sha256": sha256_file(main_path),
            "remainder159_sha256": sha256_file(remainder_path),
        },
        "outputs": {
            "development256_sha256": sha256_file(development_path),
            "confirmation903_sha256": sha256_file(confirmation_path),
            "ledger_sha256": sha256_file(ledger_path),
        },
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    files = sorted(path for path in output.rglob("*") if path.is_file())
    (output / "SHA256SUMS").write_text(
        "".join(
            f"{sha256_file(path)}  {path.relative_to(output).as_posix()}\n"
            for path in files
        ),
        encoding="utf-8",
    )
    (output / "_SUCCESS").touch()
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
