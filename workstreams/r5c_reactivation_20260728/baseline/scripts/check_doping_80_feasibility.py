#!/usr/bin/env python3
"""Check 80-slot doped-structure feasibility artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from crystal_dlm.fixed_slot import FixedSlotConfig, build_special_tokens, parse_fixed_slot_answer
from crystal_dlm.doping import read_jsonl, write_json, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full80-jsonl", type=Path, default=PROJECT_ROOT / "data/doping_crystal/full80_success.jsonl")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data/doping_full80")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    config = FixedSlotConfig(max_atoms=80)
    rows = read_jsonl(args.full80_jsonl)
    if args.limit:
        rows = rows[: args.limit]
    failures: List[Dict[str, Any]] = []
    parsed_rows = []
    for idx, row in enumerate(rows):
        try:
            arrays = parse_fixed_slot_answer(row["answer"], config=config, strict=True)
            if int(arrays["num_atoms"]) != 80:
                raise ValueError(f"num_atoms={arrays['num_atoms']}")
            if int(row.get("answer_semantic_tokens", 407)) != 407:
                raise ValueError(f"answer_semantic_tokens={row.get('answer_semantic_tokens')}")
            parsed_rows.append(
                {
                    "prompt": row.get(
                        "prompt",
                        "Generate the 80-slot doped CsPbI3 structure with target properties:",
                    ),
                    "answer": row["answer"],
                    "metadata": row.get("metadata", {}),
                    "answer_semantic_tokens": 407,
                }
            )
        except Exception as exc:
            failures.append({"index": idx, "reason": type(exc).__name__, "message": str(exc)})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    n = len(parsed_rows)
    train = parsed_rows[: max(1, int(n * 0.7))]
    val = parsed_rows[max(1, int(n * 0.7)) : max(2, int(n * 0.85))]
    test = parsed_rows[max(2, int(n * 0.85)) :]
    write_jsonl(args.output_dir / "train.jsonl", train)
    write_jsonl(args.output_dir / "val.jsonl", val or train[:1])
    write_jsonl(args.output_dir / "test.jsonl", test or val[:1] or train[:1])
    write_jsonl(args.output_dir / "failure_cases.jsonl", failures)
    with (args.output_dir / "vocab_tokens.txt").open("w", encoding="utf-8") as handle:
        for token in build_special_tokens(config):
            handle.write(token + "\n")
    metrics = {
        "input_rows": len(rows),
        "parsed_rows": len(parsed_rows),
        "failure_count": len(failures),
        "parse_rate": len(parsed_rows) / max(1, len(rows)),
        "answer_token_count": 407,
        "max_atoms": 80,
        "train_count": len(train),
        "val_count": len(val or train[:1]),
        "test_count": len(test or val[:1] or train[:1]),
    }
    write_json(args.output_dir / "feasibility_metrics.json", metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
