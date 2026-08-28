#!/usr/bin/env python3
"""Assemble deterministic per-ordinal C³FD sampling shards."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Iterable


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--requested", type=int, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    rows: dict[int, dict[str, Any]] = {}
    shard_metrics = []
    for shard in args.shard:
        if not (shard / "_SUCCESS").is_file():
            raise RuntimeError(f"incomplete C3FD shard {shard}")
        shard_metrics.append(json.loads((shard / "sample_metrics.json").read_text(encoding="utf-8")))
        for row in iter_jsonl(shard / "raw_generations.jsonl"):
            sample_idx = int(row["sample_idx"])
            if sample_idx in rows:
                raise RuntimeError(f"duplicate sample_idx {sample_idx}")
            rows[sample_idx] = row
    expected = list(range(int(args.requested)))
    if sorted(rows) != expected:
        raise RuntimeError("C3FD shard ordinals do not cover requested denominator")
    args.output_dir.mkdir(parents=True)
    failures: Counter[str] = Counter()
    n_counts: Counter[str] = Counter()
    parsed = 0
    with (args.output_dir / "raw_generations.jsonl").open("w", encoding="utf-8") as raw_handle, (
        args.output_dir / "plans_for_dlm.jsonl"
    ).open("w", encoding="utf-8") as plan_handle:
        for sample_idx in expected:
            row = rows[sample_idx]
            raw_handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            if row.get("parsed") is True:
                parsed += 1
                plan = row.get("plan_state") or {}
                n_counts[str(plan.get("N"))] += 1
                plan_handle.write(
                    json.dumps(
                        {
                            "sample_idx": sample_idx,
                            "plan_text": row.get("plan_text"),
                            "plan_state": plan,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
            else:
                failures[str(row.get("failure") or "unknown")] += 1
    metrics = {
        "schema": "h1a2_c3fd_sampling_assembled_v1",
        "requested_samples": int(args.requested),
        "parsed_samples": parsed,
        "all_request_benchmark_comp_valid": parsed,
        "parse_rate": parsed / int(args.requested),
        "formula_bpe": False,
        "repair": False,
        "replacement": False,
        "rerank": False,
        "rl": False,
        "N": dict(sorted(n_counts.items())),
        "failures": dict(failures.most_common()),
        "shards": shard_metrics,
    }
    (args.output_dir / "sample_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "_SUCCESS").touch()
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
