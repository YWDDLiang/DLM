#!/usr/bin/env python3
"""Merge deterministic independent model494 refinement workers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposal-graphs", type=Path, required=True)
    parser.add_argument("--worker-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--diff-steps", type=int, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    source = torch.load(args.proposal_graphs, map_location="cpu")
    expected = {int(graph["sample_idx"]) for graph in source}
    payloads = []
    wall_time = 0.0
    for worker_dir in args.worker_dir:
        metrics = json.loads(
            (worker_dir / "refinement_metrics.json").read_text(encoding="utf-8")
        )
        payload = torch.load(Path(metrics["output_file"]), map_location="cpu")
        payloads.append(payload)
        wall_time = max(wall_time, float(metrics.get("time_sec") or 0.0))
    actual_values = [
        int(value)
        for payload in payloads
        for value in payload["sample_indices"].view(-1).tolist()
    ]
    if len(actual_values) != len(expected) or set(actual_values) != expected:
        raise ValueError("refinement shards do not exactly cover source sample_idx")
    if len(set(actual_values)) != len(actual_values):
        raise ValueError("refinement shards contain duplicate sample_idx")
    merged = {
        "frac_coords": torch.cat([item["frac_coords"] for item in payloads], dim=1),
        "num_atoms": torch.cat([item["num_atoms"] for item in payloads], dim=1),
        "atom_types": torch.cat([item["atom_types"] for item in payloads], dim=1),
        "lengths": torch.cat([item["lengths"] for item in payloads], dim=1),
        "angles": torch.cat([item["angles"] for item in payloads], dim=1),
        "sample_indices": torch.cat(
            [item["sample_indices"].view(-1) for item in payloads], dim=0
        ),
        "time": wall_time,
    }
    args.output_dir.mkdir(parents=True)
    output_file = args.output_dir / f"dlm_refined_mp_{len(actual_values)}.pt"
    torch.save(merged, output_file)
    (args.output_dir / "refinement_metrics.json").write_text(
        json.dumps(
            {
                "schema": "independent_refinement_worker_merge_v1",
                "num_proposals": len(expected),
                "assigned_proposals": len(actual_values),
                "num_output_structures": len(actual_values),
                "output_file": str(output_file),
                "time_sec": wall_time,
                "diff_steps": args.diff_steps,
                "workers": len(payloads),
                "seed_by_sample_index": True,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
