#!/usr/bin/env python3
"""Static self-test for the exact retraining source bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from protocol import read_json, require_source_manifest, sha256_file, validate_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    args = parser.parse_args()
    source = args.source_dir.resolve()
    manifest = require_source_manifest(source, args.source_manifest_sha256)
    config = read_json(source / "CONFIG.json")
    validate_config(config)
    required = {
        "CONFIG.json", "README.md", "preflight.py", "protocol.py",
        "self_test.py", "train_epoch2.sbatch", "verify_training.py",
        "prepare_and_submit_once.sh", "SOURCE_SHA256.txt"
    }
    observed = {path.name for path in source.iterdir() if path.is_file()}
    if observed != required:
        raise ValueError(f"source file contract changed: {sorted(observed)}")
    for path in sorted(source.glob("*.py")):
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
    print(json.dumps({
        "self_test": "PASS",
        "source_manifest_sha256": sha256_file(manifest),
        "expected_updates": 3392,
        "materials_project_access": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
