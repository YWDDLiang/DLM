#!/usr/bin/env python3
"""Publish R5-C-identical JSONL with a training-only PlanGraph sidecar."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from crystal_dlm.h1a2_sidecar_dataset import (  # noqa: E402
    H1A2SidecarBuildError,
    build_h1a2_sidecar_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Frozen mp_20_r5_exact_length directory.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
    )
    args = parser.parse_args()
    split_inputs = {
        split: args.source_dir / f"{split}.jsonl" for split in args.splits
    }
    try:
        manifest = build_h1a2_sidecar_dataset(
            split_inputs=split_inputs,
            output_dir=args.output_dir,
            vocab_file=args.source_dir / "vocab_tokens.txt",
            project_root=PROJECT_ROOT,
        )
    except H1A2SidecarBuildError as exc:
        payload = {
            "status": "failed",
            "error": str(exc),
            "report": exc.report,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        raise SystemExit(2) from exc
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
