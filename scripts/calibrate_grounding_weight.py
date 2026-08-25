#!/usr/bin/env python3
"""Compute the single grounding coefficient from training-only gradient norms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from h1a2_repro.counterfactual import calibrated_grounding_weight  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="JSON with ce_gradient_norm and grounding_gradient_norm")
    parser.add_argument("--target-ratio", type=float, default=0.1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    weight = calibrated_grounding_weight(
        ce_gradient_norm=float(payload["ce_gradient_norm"]),
        grounding_gradient_norm=float(payload["grounding_gradient_norm"]),
        target_ratio=args.target_ratio,
    )
    args.output.write_text(json.dumps({"grounding_weight": weight, "target_ratio": args.target_ratio}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

