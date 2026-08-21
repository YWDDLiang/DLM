"""CLI for the Plan memorization and quality audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .attribution_cli import load_jsonl
from .plan_audit import audit_plans


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit_plans(load_jsonl(args.generated), load_jsonl(args.train))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
