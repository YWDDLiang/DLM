"""CLI for summarizing aligned pre/post-refiner JSONL rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .attribution_cli import load_jsonl
from .refiner_attribution import summarize_refiner_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize_refiner_rows(load_jsonl(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
