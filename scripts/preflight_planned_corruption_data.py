#!/usr/bin/env python3
"""Run the frozen-tokenizer and CPU-mask preflight for PlanGraph-DLM data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from transformers import AutoTokenizer  # noqa: E402

from crystal_dlm.h1_readonly_guard import assert_writable_output_path  # noqa: E402
from crystal_dlm.planned_preflight import (  # noqa: E402
    PlannedPreflightError,
    preflight_planned_data,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Published PlanGraph dataset body directory",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        required=True,
        help="Frozen local tokenizer/checkpoint path",
    )
    parser.add_argument(
        "--policy",
        choices=["d1", "d2"],
        required=True,
    )
    parser.add_argument("--max-length", type=int, required=True)
    parser.add_argument("--splits", default="train,val")
    parser.add_argument("--corruption-seed", type=int, default=20260731)
    parser.add_argument("--mask-smoke-rows", type=int, default=32)
    parser.add_argument("--mask-smoke-batch-size", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional new JSON report path; existing paths are refused",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tokenizer_path = args.tokenizer_path.expanduser().resolve()
    split_names = [
        item.strip() for item in args.splits.split(",") if item.strip()
    ]
    missing_required = sorted({"train", "val"} - set(split_names))
    if missing_required:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": (
                        "formal tokenizer preflight requires train and val; "
                        f"missing={missing_required}"
                    ),
                },
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1
    if not tokenizer_path.exists():
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": f"tokenizer path does not exist: {tokenizer_path}",
                },
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        report = preflight_planned_data(
            data_dir=args.data_dir,
            tokenizer=tokenizer,
            splits=split_names,
            max_length=args.max_length,
            policy=args.policy,
            corruption_seed=args.corruption_seed,
            mask_smoke_rows=args.mask_smoke_rows,
            mask_smoke_batch_size=args.mask_smoke_batch_size,
            verify_manifest=True,
        )
    except (PlannedPreflightError, OSError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                },
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1

    report["tokenizer"]["tokenizer_path"] = str(tokenizer_path)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is None:
        print(rendered)
    else:
        output = assert_writable_output_path(
            args.output,
            project_root=PROJECT_ROOT,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as handle:
            handle.write(rendered)
            handle.write("\n")
        print(
            json.dumps(
                {
                    "ok": report["preflight_gate_passed"],
                    "output": str(output),
                },
                sort_keys=True,
            )
        )
    return 0 if report["preflight_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
