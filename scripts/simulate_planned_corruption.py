#!/usr/bin/env python3
"""Run a CPU-only D1/D2 mask-distribution simulation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from crystal_dlm.h1_readonly_guard import assert_writable_output_path  # noqa: E402
from crystal_dlm.planned_corruption import (  # noqa: E402
    current_order_groups,
    plangraph_dependency_groups,
    simulate_planned_policy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=("d1", "d2"), required=True)
    parser.add_argument("--num-atoms", type=int)
    parser.add_argument(
        "--plangraph",
        type=Path,
        help="PlanGraph JSON input required for D2",
    )
    parser.add_argument("--trials", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional exclusive JSON output; stdout is always supported",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.policy == "d1":
        if args.num_atoms is None:
            raise SystemExit("--num-atoms is required for D1")
        groups = current_order_groups(args.num_atoms)
    else:
        if args.plangraph is None:
            raise SystemExit("--plangraph is required for D2")
        graph = json.loads(args.plangraph.read_text(encoding="utf-8"))
        groups = plangraph_dependency_groups(graph)
    summary = simulate_planned_policy(
        groups,
        trials=args.trials,
        seed=args.seed,
        policy_name=args.policy,
    )
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    if args.output is None:
        print(rendered)
    else:
        output = assert_writable_output_path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as handle:
            handle.write(rendered)
            handle.write("\n")
        print(json.dumps({"output": str(output), "status": "complete"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
