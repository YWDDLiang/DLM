#!/usr/bin/env python3
"""Convert generated rich plans into H1-G1 condition-view DLM prompt JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from crystal_dlm.h1_formula_only_body import H1G1_CONDITION_VIEWS, build_condition_view_body_prompt  # noqa: E402


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def plan_from_row(row: Mapping[str, Any], idx: int) -> dict[str, Any]:
    plan = row.get("plan_state") or row.get("r5_plan_state") or row.get("parsed_plan")
    if not isinstance(plan, dict):
        raise ValueError(f"row {idx} has no plan_state/r5_plan_state/parsed_plan")
    return dict(plan)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--condition-view", choices=H1G1_CONDITION_VIEWS, required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.output_jsonl.open("w", encoding="utf-8") as out:
        for idx, row in enumerate(iter_jsonl(args.input_jsonl)):
            if args.limit is not None and count >= int(args.limit):
                break
            plan = plan_from_row(row, idx)
            prompt = build_condition_view_body_prompt(plan, condition_view=args.condition_view).rstrip() + "\n"
            payload = {
                "sample_idx": row.get("sample_idx", count),
                "source_idx": idx,
                "condition_view": args.condition_view,
                "plan_state": plan,
                "r5_plan_state": plan,
                "prompt": prompt,
                "source_record": {key: value for key, value in row.items() if key not in {"prompt"}},
            }
            out.write(json.dumps(payload, ensure_ascii=False) + "\n")
            count += 1
    print(json.dumps({"input_jsonl": str(args.input_jsonl), "output_jsonl": str(args.output_jsonl), "condition_view": args.condition_view, "rows": count}, indent=2))


if __name__ == "__main__":
    main()
