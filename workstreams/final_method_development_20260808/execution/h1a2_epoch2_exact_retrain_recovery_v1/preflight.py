#!/usr/bin/env python3
"""Audit every scientific input to the exact H1-A2 epoch-2 retraining job."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
from pathlib import Path

from protocol import read_json, require_file, require_source_manifest, sha256_file, validate_config, write_json_exclusive


def count_rows(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--phase", choices=("prepare", "prepared", "job"), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    source = args.source_dir.resolve()
    manifest = require_source_manifest(source, args.source_manifest_sha256)
    config = read_json(source / "CONFIG.json")
    validate_config(config)
    project = Path(str(config["project_root"])).resolve()
    run_root = Path(str(config["run_root"])).resolve()
    if args.phase == "prepare" and run_root.exists():
        raise FileExistsError(f"immutable run already exists: {run_root}")
    if args.phase in {"prepared", "job"} and not run_root.is_dir():
        raise FileNotFoundError(run_root)

    checked: dict[str, dict[str, object]] = {}
    for relative, expected in dict(config["runtime_files"]).items():
        path = require_file(project / relative, str(expected), f"runtime {relative}")
        checked[f"runtime:{relative}"] = {"sha256": str(expected), "bytes": path.stat().st_size}

    base = Path(str(config["base_model"]["path"])).resolve()
    for relative, expected in dict(config["base_model"]["files"]).items():
        path = require_file(base / relative, str(expected), f"base model {relative}")
        checked[f"base:{relative}"] = {"sha256": str(expected), "bytes": path.stat().st_size}
    shards = dict(config["base_model"]["model_shards"])
    if not shards:
        raise ValueError("base-model shard ledger is empty")
    for relative, spec in shards.items():
        path = (base / relative).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        expected_bytes = int(spec["bytes"])
        if path.stat().st_size != expected_bytes:
            raise ValueError(f"base model shard size changed: {relative}")
        checked[f"base:{relative}"] = {
            "identity": "index_sha256_plus_exact_filename_and_bytes",
            "bytes": path.stat().st_size,
            "hash_verified": False,
        }

    epoch1 = Path(str(config["epoch1_adapter"]["path"])).resolve()
    for relative, expected in dict(config["epoch1_adapter"]["files"]).items():
        path = require_file(epoch1 / relative, str(expected), f"epoch1 {relative}")
        checked[f"epoch1:{relative}"] = {"sha256": str(expected), "bytes": path.stat().st_size}

    data = Path(str(config["data"]["path"])).resolve()
    rows: dict[str, int] = {}
    for relative, spec in dict(config["data"]["splits"]).items():
        path = require_file(data / relative, str(spec["sha256"]), f"data {relative}")
        observed_rows = count_rows(path)
        if observed_rows != int(spec["rows"]):
            raise ValueError(f"{relative} row count changed")
        rows[relative] = observed_rows
        checked[f"data:{relative}"] = {"sha256": str(spec["sha256"]), "bytes": path.stat().st_size}

    env = dict(config["environment"])
    observed_versions = {"python": platform.python_version()}
    if observed_versions["python"] != str(env["python"]):
        raise ValueError(f"Python version changed: {observed_versions['python']}")
    for package, expected in dict(env["packages"]).items():
        observed = importlib.metadata.version(package)
        observed_versions[package] = observed
        if observed != str(expected):
            raise ValueError(f"package {package} changed: expected={expected} observed={observed}")

    report = {
        "schema": "h1a2_epoch2_exact_retrain_preflight_v1",
        "phase": args.phase,
        "source_manifest_sha256": sha256_file(manifest),
        "checked": checked,
        "data_rows": rows,
        "versions": observed_versions,
        "historical_command_defaults_locked": True,
        "engineering_status": "PASS",
    }
    if args.output:
        write_json_exclusive(args.output, report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
