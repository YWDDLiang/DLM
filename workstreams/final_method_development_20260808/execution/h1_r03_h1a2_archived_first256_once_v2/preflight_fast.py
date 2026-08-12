#!/usr/bin/env python3
"""Fast preflight using registered checkpoint identity instead of repeated rehashing."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def require_hash(path: Path, expected: str, label: str) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} missing: {path}")
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"{label} SHA changed: {observed}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": observed}


def verify_manifest(source: Path, expected_sha: str) -> dict[str, Any]:
    source = source.resolve()
    manifest = source / "SOURCE_SHA256.txt"
    identity = require_hash(manifest, expected_sha, f"{source.name} manifest")
    entries = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(None, 1)
        relative = relative.strip().lstrip("*")
        if relative.startswith("./"):
            relative = relative[2:]
        require_hash(source / relative, expected, f"{source.name}/{relative}")
        entries.append(relative)
    identity["files"] = len(entries)
    return identity


def registered_stat(spec: dict[str, Any], label: str) -> dict[str, Any]:
    path = Path(spec["path"]).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} missing: {path}")
    stat = path.stat()
    if stat.st_size != int(spec["bytes"]) or int(stat.st_mtime) != int(spec["mtime_unix"]):
        raise ValueError(f"{label} registered size/mtime identity changed")
    return {
        "path": str(path),
        "bytes": stat.st_size,
        "mtime_unix": int(stat.st_mtime),
        "registered_sha256": spec["registered_sha256"],
        "rehash_for_this_run": False,
    }


def write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = read_json(args.config.resolve())
    source = args.source_dir.resolve()
    run = args.run_root.resolve()
    if (
        config.get("schema") != "h1_r03_h1a2_archived_first256_once_config_v2"
        or config.get("status") != "user_authorized_registered_checkpoint_identity"
        or int(config.get("denominator", -1)) != 256
        or int(config.get("repeat", -1)) != 0
        or config.get("arm_order") != ["control", "candidate"]
        or config["scientific_contract"].get("planner_resample") is not False
        or int(config["scientific_contract"].get("diffusion_steps", -1)) != 800
    ):
        raise ValueError("v2 one-shot contract changed")
    if source != Path(config["source_dir"]).resolve() or run != Path(config["run_root"]).resolve():
        raise ValueError("v2 frozen paths changed")
    if not run.is_dir() or sorted(path.name for path in run.iterdir()) != ["logs", "status"]:
        raise ValueError("v2 run root is not an empty prepared shell")
    archived = config["archived_wrapper_v1"]
    v1_source = Path(archived["source_dir"])
    report = {
        "schema": "h1_r03_h1a2_archived_first256_once_fast_preflight_v2",
        "status": "pass",
        "wrapper_source": verify_manifest(source, args.source_manifest_sha256),
        "archived_wrapper_v1": verify_manifest(v1_source, archived["source_manifest_sha256"]),
        "scientific_config": require_hash(
            v1_source / archived["scientific_config"],
            archived["scientific_config_sha256"],
            "v1 scientific config",
        ),
        "body_source": verify_manifest(
            Path(config["body_source"]["path"]), config["body_source"]["manifest_sha256"]
        ),
        "refiner_source": verify_manifest(
            Path(config["refiner_source"]["path"]), config["refiner_source"]["manifest_sha256"]
        ),
        "attempt_ledger": require_hash(
            Path(config["attempt_ledger"]["path"]), config["attempt_ledger"]["sha256"], "first256 ledger"
        ),
        "registered_checkpoints": {
            name: registered_stat(spec, name)
            for name, spec in config["registered_checkpoints"].items()
        },
        "checkpoint_rehash_performed": False,
        "mp_credentials_present": [
            name for name in ("MP_API_KEY", "PMG_MAPI_KEY", "MAPI_KEY") if os.environ.get(name)
        ],
        "planner_resample": False,
        "new_scientific_seed": False,
        "retry_replacement_repair_filter_rerank": False,
    }
    if report["mp_credentials_present"]:
        raise ValueError("archived evaluator must remain offline")
    body_config = read_json(Path(config["body_source"]["path"]) / "CONFIG.json")
    ref_config = read_json(Path(config["refiner_source"]["path"]) / "config.json")
    if (
        body_config["body"]["adapter_sha256"]
        != config["registered_checkpoints"]["body_adapter"]["registered_sha256"]
        or float(body_config["body"]["temperature"]) != 0.7
        or int(body_config["body"]["max_batch_size"]) != 8
        or ref_config["refiner"]["checkpoint_sha256"]
        != config["registered_checkpoints"]["model_494"]["registered_sha256"]
        or int(ref_config["refiner"]["diffusion_steps"]) != 800
    ):
        raise ValueError("archived scientific configuration changed")
    write_json_exclusive(args.output.resolve(), report)
    print(json.dumps({"status": "pass", "checkpoint_rehash_performed": False}, sort_keys=True))


if __name__ == "__main__":
    main()
