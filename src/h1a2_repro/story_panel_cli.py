"""CLI for preregistered E1/E2 story-panel ledgers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .story_panel import build_panels, load_plan_rows


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--learned-plans", type=Path, required=True)
    parser.add_argument("--gold-plans", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-pairs", type=int, default=24)
    parser.add_argument("--e2-pairs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260822)
    args = parser.parse_args()

    learned = load_plan_rows(args.learned_plans, "learned")
    gold = load_plan_rows(args.gold_plans, "gold")
    tasks, e2_task_ids, report = build_panels(
        learned,
        gold,
        num_pairs=args.num_pairs,
        e2_pairs=args.e2_pairs,
        seed=args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "e1_tasks.jsonl", tasks)
    write_jsonl(args.output_dir / "e2_contract.jsonl", ({"task_id": task_id} for task_id in e2_task_ids))
    (args.output_dir / "selection_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
