#!/usr/bin/env python3
"""Fail-closed audit of the frozen V4 body runner's runtime config contract."""

from __future__ import annotations

import argparse
from pathlib import Path

from protocol import read_json, validate_config, write_json_exclusive


REQUIRED_BODY_KEYS = (
    "base_model",
    "checkpoint",
    "adapter_file",
    "adapter_expected_bytes",
    "adapter_sha256",
    "large_artifact_rehash",
    "tokenizer_vocab_sha256",
    "tokenizer_json_sha256",
    "tokenizer_config_sha256",
    "tokenizer_size",
    "models",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = read_json(args.config.resolve())
    validate_config(config)
    body = config["body"]
    missing = [key for key in REQUIRED_BODY_KEYS if key not in body]
    if missing:
        raise KeyError(f"body runtime config keys missing: {missing}")

    checkpoint = Path(body["checkpoint"]).resolve()
    adapter = checkpoint / str(body["adapter_file"])
    tokenizer_json = checkpoint / "tokenizer.json"
    tokenizer_config = checkpoint / "tokenizer_config.json"
    if (
        not adapter.is_file()
        or adapter.stat().st_size != int(body["adapter_expected_bytes"])
        or body["large_artifact_rehash"] is not False
    ):
        raise ValueError("body adapter path or byte size changed")
    if not tokenizer_json.is_file() or not tokenizer_config.is_file():
        raise FileNotFoundError("body tokenizer files are missing")

    report = {
        "schema": "h1a2_postplanner_body_runtime_contract_v2",
        "status": "pass",
        "required_body_keys": list(REQUIRED_BODY_KEYS),
        "checkpoint": str(checkpoint),
        "adapter_file": str(body["adapter_file"]),
        "adapter_bytes": adapter.stat().st_size,
        "adapter_sha256_registered": str(body["adapter_sha256"]),
        "tokenizer_sha256_registered": {
            "tokenizer_json": str(body["tokenizer_json_sha256"]),
            "tokenizer_config": str(body["tokenizer_config_sha256"]),
        },
        "large_artifact_rehashed": False,
        "identity_basis": "registered_sha256_plus_current_path_and_byte_size",
        "tokenizer_vocab_sha256": str(body["tokenizer_vocab_sha256"]),
        "tokenizer_size": int(body["tokenizer_size"]),
        "frozen_v4_failure_prevented": "KeyError: 'adapter_file'",
    }
    write_json_exclusive(args.output.resolve(), report)
    print("audit_body_contract: PASS")


if __name__ == "__main__":
    main()
