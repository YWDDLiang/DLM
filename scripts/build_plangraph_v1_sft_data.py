#!/usr/bin/env python3
"""Atomically build frozen PlanGraph Planner/body SFT splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from crystal_dlm.plangraph_dataset import (  # noqa: E402
    PlanGraphDatasetBuildError,
    build_plangraph_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing the frozen source <split>.jsonl files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New output directory; existing paths are always refused",
    )
    parser.add_argument(
        "--splits",
        default="train,val",
        help="Comma-separated source splits, in frozen publication order",
    )
    parser.add_argument(
        "--vocab-file",
        type=Path,
        help="Defaults to INPUT_DIR/vocab_tokens.txt",
    )
    parser.add_argument(
        "--minimum-conversion-rate",
        type=float,
        default=0.98,
        help="Registered lower coverage gate; publication still requires every row",
    )
    parser.add_argument("--max-failure-examples", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    split_names = [item.strip() for item in args.splits.split(",") if item.strip()]
    missing_required = sorted({"train", "val"} - set(split_names))
    if missing_required:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": (
                        "formal frozen publication requires train and val; "
                        f"missing={missing_required}"
                    ),
                },
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1
    input_dir = args.input_dir.expanduser().resolve()
    split_inputs = {
        split: input_dir / f"{split}.jsonl" for split in split_names
    }
    vocab_file = (
        args.vocab_file.expanduser().resolve()
        if args.vocab_file is not None
        else input_dir / "vocab_tokens.txt"
    )
    try:
        report = build_plangraph_dataset(
            split_inputs=split_inputs,
            output_dir=args.output_dir,
            vocab_file=vocab_file,
            project_root=PROJECT_ROOT,
            minimum_conversion_rate=args.minimum_conversion_rate,
            require_all_rows=True,
            fail_on_cross_split_overlap=True,
            max_failure_examples=args.max_failure_examples,
        )
    except PlanGraphDatasetBuildError as exc:
        payload = {
            "ok": False,
            "error": str(exc),
            "report": exc.report,
        }
        print(json.dumps(payload, indent=2, sort_keys=True), file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "ok": True,
                "output_dir": report["output_dir"],
                "manifest_sha256": report["manifest_sha256"],
                "total_rows": report["total_rows"],
                "fixed_validation_panel": report["fixed_validation_panel"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
