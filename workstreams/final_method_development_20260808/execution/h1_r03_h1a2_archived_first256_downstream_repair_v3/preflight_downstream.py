#!/usr/bin/env python3
"""Fail-closed preflight for the authorized downstream-only repair."""

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
    entries: list[str] = []
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


def validate_body_gate(config: dict[str, Any]) -> dict[str, Any]:
    upstream = Path(config["upstream_v2"]["run_root"]).resolve()
    status = upstream / "status"
    required = [status / "BODY_IDENTITY_SUCCESS", status / "PIPELINE_FAILURE"]
    if not all(path.is_file() for path in required):
        raise FileNotFoundError("upstream success/failure evidence is incomplete")
    if (status / "exit_code").read_text(encoding="utf-8").strip() != str(
        config["upstream_v2"]["required_exit_code_file"]
    ):
        raise ValueError("upstream failure exit-code evidence changed")
    gate_path = status / "body_identity_gate.json"
    gate = read_json(gate_path)
    if gate.get("status") != "pass" or gate.get("byte_identical_to_archived_success") is not True:
        raise ValueError("upstream body byte-identity gate did not pass")
    observed: dict[str, Any] = {}
    for name, expected in config["body_artifacts"].items():
        entry = gate.get("artifacts", {}).get(name)
        path = upstream / "body" / name
        if (
            not isinstance(entry, dict)
            or entry.get("matches_archived_success") is not True
            or entry.get("sha256") != expected["sha256"]
            or int(entry.get("bytes", -1)) != int(expected["bytes"])
            or Path(entry.get("path", "")).resolve() != path
            or not path.is_file()
            or path.stat().st_size != int(expected["bytes"])
        ):
            raise ValueError(f"upstream body identity changed: {name}")
        observed[name] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256_from_completed_byte_gate": entry["sha256"],
            "rehash_for_repair": False,
        }
    arms = upstream / "arms"
    if arms.exists() and any(path.is_file() for path in arms.rglob("*")):
        raise ValueError("failed upstream unexpectedly contains downstream science outputs")
    return {
        "gate": require_hash(gate_path, sha256_file(gate_path), "upstream body gate"),
        "artifacts": observed,
        "body_reused": True,
        "body_generation_rerun": False,
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
        config.get("schema") != "h1_r03_h1a2_archived_first256_downstream_repair_config_v3"
        or config.get("status") != "user_authorized_single_engineering_repair"
        or source != Path(config["source_dir"]).resolve()
        or run != Path(config["run_root"]).resolve()
        or int(config.get("denominator", -1)) != 256
        or int(config.get("repeat", -1)) != 0
        or config.get("arm_order") != ["control", "candidate"]
        or config["scientific_contract"].get("body_generation_rerun") is not False
        or config["engineering_repair"].get("scope") != "python_import_precedence_only"
    ):
        raise ValueError("downstream repair contract changed")
    if not run.is_dir() or sorted(path.name for path in run.iterdir()) != ["logs", "status"]:
        raise ValueError("repair run root is not an empty prepared shell")
    if any(os.environ.get(name) for name in ("MP_API_KEY", "PMG_MAPI_KEY", "MAPI_KEY")):
        raise ValueError("archived evaluator must remain offline")

    upstream_source = Path(config["upstream_v2"]["source_dir"])
    v1 = config["archived_wrapper_v1"]
    refiner = config["refiner_source"]
    model = config["registered_model_494"]
    model_path = Path(model["path"]).resolve()
    model_stat = model_path.stat()
    if model_stat.st_size != int(model["bytes"]) or int(model_stat.st_mtime) != int(model["mtime_unix"]):
        raise ValueError("model_494 registered size/mtime identity changed")
    required_module = Path(refiner["path"]) / refiner["required_module"]
    if not required_module.is_file():
        raise FileNotFoundError(required_module)

    report = {
        "schema": "h1_r03_h1a2_archived_first256_downstream_preflight_v3",
        "status": "pass",
        "repair_source": verify_manifest(source, args.source_manifest_sha256),
        "upstream_v2_source": verify_manifest(
            upstream_source, config["upstream_v2"]["source_manifest_sha256"]
        ),
        "archived_wrapper_v1": verify_manifest(
            Path(v1["source_dir"]), v1["source_manifest_sha256"]
        ),
        "scientific_config": require_hash(
            Path(v1["source_dir"]) / v1["scientific_config"],
            v1["scientific_config_sha256"],
            "archived scientific config",
        ),
        "refiner_source": verify_manifest(Path(refiner["path"]), refiner["manifest_sha256"]),
        "required_refiner_module": str(required_module.resolve()),
        "attempt_ledger": require_hash(
            Path(config["attempt_ledger"]["path"]),
            config["attempt_ledger"]["sha256"],
            "first256 attempt ledger",
        ),
        "upstream_body": validate_body_gate(config),
        "registered_model_494": {
            "path": str(model_path),
            "bytes": model_stat.st_size,
            "mtime_unix": int(model_stat.st_mtime),
            "registered_sha256": model["registered_sha256"],
            "rehash_for_this_run": False,
        },
        "body_generation_rerun": False,
        "checkpoint_rehash_performed": False,
        "scientific_configuration_changed": False,
        "retry_replacement_repair_filter_rerank": False,
    }
    write_json_exclusive(args.output.resolve(), report)
    print(json.dumps({"status": "pass", "body_reused": True, "checkpoint_rehash_performed": False}, sort_keys=True))


if __name__ == "__main__":
    main()
