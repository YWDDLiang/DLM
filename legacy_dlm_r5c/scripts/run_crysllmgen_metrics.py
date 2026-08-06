#!/usr/bin/env python3
"""Run CrysLLMGen compute_metrics.py and save a JSON summary."""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-path", type=Path, required=True)
    parser.add_argument("--gt-file", type=Path, default=PROJECT_ROOT / "reference/crysllmgen/data/mp_20/test.csv")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--crysllmgen-dir", type=Path, default=PROJECT_ROOT / "reference/crysllmgen")
    parser.add_argument("--eval-model-name", default="mp20")
    parser.add_argument("--tasks", nargs="+", default=["gen"])
    args = parser.parse_args()
    root_path = args.root_path.resolve()
    if root_path.is_dir():
        candidates = sorted(
            root_path.glob("dlm_refined_mp_*.pt"),
            key=lambda path: (path.stat().st_size, path.name),
            reverse=True,
        )
        candidates = [path for path in candidates if ".rank" not in path.name]
        if candidates:
            root_path = candidates[0].resolve()
    gt_file = args.gt_file.resolve()

    command = [
        sys.executable,
        "compute_metrics.py",
        "--root_path",
        str(root_path),
        "--tasks",
        *args.tasks,
        "--eval_model_name",
        args.eval_model_name,
        "--gt_file",
        str(gt_file),
    ]
    result = subprocess.run(
        command,
        cwd=args.crysllmgen_dir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    metrics = {}
    for line in result.stdout.splitlines()[::-1]:
        stripped = line.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                metrics = ast.literal_eval(stripped)
                break
            except Exception:
                pass
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "command": command,
                "returncode": result.returncode,
                "metrics": metrics,
                "stdout_tail": result.stdout.splitlines()[-80:],
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
        handle.write("\n")
    if result.returncode != 0:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
