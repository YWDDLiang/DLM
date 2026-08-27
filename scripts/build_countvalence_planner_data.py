#!/usr/bin/env python3
"""Convert existing rich-Plan SFT rows to one-model count-valence targets."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crystal_dlm.r5_plan_state import (  # noqa: E402
    build_countvalence_plan_prompt,
    plan_state_to_countvalencefields,
)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError(f"non-object row in {path}")
                yield value


def convert_split(source: Path, destination: Path) -> dict[str, Any]:
    rows = 0
    written = 0
    failures: Counter[str] = Counter()
    prompt = build_countvalence_plan_prompt()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as output:
        for row in iter_jsonl(source):
            rows += 1
            plan = row.get("plan_state") or row.get("r5_plan_state")
            if not isinstance(plan, Mapping):
                failures["missing_plan_state"] += 1
                continue
            try:
                answer = plan_state_to_countvalencefields(plan)
            except Exception as exc:  # noqa: BLE001
                failures[type(exc).__name__] += 1
                continue
            record = {
                "task": "h1_llm_countvalence_rich_plan",
                "representation": "h1_llm_countvalence_v1",
                "prompt": prompt,
                "answer": answer,
                "text": prompt.rstrip() + "\n" + answer,
                "plan_state": dict(plan),
                "source_row_idx": row.get("row_idx"),
                "sample_weight": float(row.get("sample_weight", 1.0) or 1.0),
            }
            output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            written += 1
    return {
        "source": str(source.resolve()),
        "destination": str(destination.resolve()),
        "rows": rows,
        "written": written,
        "coverage": 0.0 if rows == 0 else written / rows,
        "failures": dict(sorted(failures.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    splits = {}
    for split in ("train", "val", "test"):
        source = args.input_dir / f"{split}.jsonl"
        if not source.is_file():
            continue
        splits[split] = convert_split(source, args.output_dir / f"{split}.jsonl")
    if "train" not in splits or "val" not in splits:
        raise ValueError("count-valence Planner data requires train and val splits")
    manifest = {
        "schema": "h1a2_countvalence_planner_data_v1",
        "input_dir": str(args.input_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "splits": splits,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "_SUCCESS").touch()
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
