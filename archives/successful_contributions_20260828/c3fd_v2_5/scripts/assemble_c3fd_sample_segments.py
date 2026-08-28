#!/usr/bin/env python3
"""Assemble non-overlapping C3FD request segments without replacement."""

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
    parser.add_argument("--segment", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--requested", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--reachability-mode", default="pauling_bitset")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    raw_by_index: dict[int, dict[str, Any]] = {}
    plan_by_index: dict[int, dict[str, Any]] = {}
    segment_metrics = []
    for segment in args.segment:
        if not (segment / "_SUCCESS").is_file():
            raise FileNotFoundError(f"segment lacks _SUCCESS: {segment}")
        metrics = json.loads((segment / "sample_metrics.json").read_text(encoding="utf-8"))
        if int(metrics["seed"]) != int(args.seed):
            raise ValueError("segment seed mismatch")
        if str(metrics.get("reachability_mode")) != str(args.reachability_mode):
            raise ValueError("segment reachability mode mismatch")
        segment_metrics.append(metrics)
        for row in iter_jsonl(segment / "raw_generations.jsonl"):
            sample_idx = int(row["sample_idx"])
            if sample_idx in raw_by_index:
                raise ValueError(f"duplicate raw sample_idx {sample_idx}")
            raw_by_index[sample_idx] = row
        for row in iter_jsonl(segment / "plans_for_dlm.jsonl"):
            sample_idx = int(row["sample_idx"])
            if sample_idx in plan_by_index:
                raise ValueError(f"duplicate plan sample_idx {sample_idx}")
            plan_by_index[sample_idx] = row

    expected = set(range(int(args.requested)))
    if set(raw_by_index) != expected:
        missing = sorted(expected - set(raw_by_index))
        extra = sorted(set(raw_by_index) - expected)
        raise ValueError(f"global request ledger mismatch missing={missing[:5]} extra={extra[:5]}")
    parsed_indices = {
        sample_idx for sample_idx, row in raw_by_index.items() if row.get("parsed") is True
    }
    if set(plan_by_index) != parsed_indices:
        raise ValueError("plans_for_dlm indices do not equal parsed raw indices")

    failures: Counter[str] = Counter()
    n_counts: Counter[int] = Counter()
    arity_counts: Counter[int] = Counter()
    family_counts: Counter[str] = Counter()
    certificate_counts: Counter[str] = Counter()
    for sample_idx in sorted(raw_by_index):
        row = raw_by_index[sample_idx]
        if row.get("parsed") is not True:
            failures[str(row.get("failure") or "unknown_failure")] += 1
            continue
        proposal = row.get("target_proposal") or {}
        n_counts[int(proposal["N"])] += 1
        arity_counts[int(proposal["arity"])] += 1
        family_counts[str(proposal["family"])] += 1
        certificate = row.get("certificate") or {}
        certificate_counts[str(certificate.get("certificate_class"))] += 1

    args.output_dir.mkdir(parents=True)
    with (args.output_dir / "raw_generations.jsonl").open("w", encoding="utf-8") as handle:
        for sample_idx in sorted(raw_by_index):
            handle.write(json.dumps(raw_by_index[sample_idx], ensure_ascii=False, sort_keys=True) + "\n")
    with (args.output_dir / "plans_for_dlm.jsonl").open("w", encoding="utf-8") as handle:
        for sample_idx in sorted(plan_by_index):
            handle.write(json.dumps(plan_by_index[sample_idx], ensure_ascii=False, sort_keys=True) + "\n")
    metrics = {
        "schema": "h1a2_c3fd_segmented_sampling_metrics_v1",
        "requested_samples": int(args.requested),
        "start_index": 0,
        "end_index_exclusive": int(args.requested),
        "parsed_samples": len(parsed_indices),
        "all_request_benchmark_comp_valid": len(parsed_indices),
        "parse_rate": len(parsed_indices) / int(args.requested),
        "formula_bpe": False,
        "repair": False,
        "replacement": False,
        "rerank": False,
        "rl": False,
        "pair_prior_weight": 0.0,
        "species_top_k": 0,
        "seed": int(args.seed),
        "reachability_mode": str(args.reachability_mode),
        "elapsed_sec": sum(float(value.get("elapsed_sec", 0.0)) for value in segment_metrics),
        "N": {str(key): value for key, value in sorted(n_counts.items())},
        "arity": {str(key): value for key, value in sorted(arity_counts.items())},
        "family": dict(sorted(family_counts.items())),
        "certificate_classes": dict(sorted(certificate_counts.items())),
        "failures": dict(failures.most_common()),
        "segments": [str(path.resolve()) for path in args.segment],
        "segment_metrics": segment_metrics,
    }
    (args.output_dir / "sample_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "run_config.json").write_text(
        json.dumps(vars(args), default=str, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "_SUCCESS").touch()
    print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
