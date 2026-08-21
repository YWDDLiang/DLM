#!/usr/bin/env python3
"""Prepare E2 graph, metadata, and failure-preserving ledgers from E1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from h1a2_repro.refine_panel import select_refinement_panel


def load_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposal-graphs", type=Path, required=True)
    parser.add_argument("--accepted-tasks", type=Path, required=True)
    parser.add_argument("--e1-tasks", type=Path, required=True)
    parser.add_argument("--e2-contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260822)
    args = parser.parse_args()

    import torch

    graphs = torch.load(args.proposal_graphs, map_location="cpu")
    accepted = load_jsonl(args.accepted_tasks)
    all_tasks = load_jsonl(args.e1_tasks)
    contract_ids = [str(row["task_id"]) for row in load_jsonl(args.e2_contract)]
    selected_graphs, metadata, ledger, report = select_refinement_panel(
        graphs,
        accepted,
        all_tasks,
        contract_ids,
        seed=args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(selected_graphs, args.output_dir / "e2_proposal_graphs.pt")
    write_jsonl(args.output_dir / "e2_selected_metadata.jsonl", metadata)
    write_jsonl(args.output_dir / "e2_attempt_ledger.jsonl", ledger)
    (args.output_dir / "e2_selection_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
