#!/usr/bin/env python3
"""Non-network contract and runtime preflight for four-arm 512."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

from crystal_dlm.h1_crplan import CRPLAN_SCHEMA, load_frozen_smact_table
from workstreams.plangraph_dlm_iclr_20260731.execution.h1_crplan_fourarm512_route_amendment_v1.evaluate_fourarm512 import (
    DENOMINATOR,
    IDENTITY,
    MODES,
)
from workstreams.plangraph_dlm_iclr_20260731.execution.h1_crplan_fourarm512_route_amendment_v1.generate_science_ledger import (
    build_ledger,
)
from workstreams.plangraph_dlm_iclr_20260731.execution.h1_crplan_fourarm512_route_amendment_v1.run_fourarm512_arm import (
    validate_contract,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = read_json(args.config)
    ledger = read_json(args.ledger)
    authorization = read_json(args.authorization)
    execution = read_json(args.execution_manifest)
    ledger_sha256 = sha256_file(args.ledger)
    expected_ledger = build_ledger()
    checks = {
        "python_ge_3p10": sys.version_info >= (3, 10),
        "identity": config.get("identity") == IDENTITY,
        "denominator": config.get("attempts_per_arm") == DENOMINATOR,
        "modes": tuple(config.get("modes") or ()) == MODES,
        "ledger_byte_semantics": ledger == expected_ledger,
        "ledger_sha_frozen_in_config": (
            config.get("science_ledger_sha256") == ledger_sha256
        ),
        "authorization_identity": (
            authorization.get("schema")
            == "h1_crplan_fourarm512_route_amendment_authorization_v1"
        ),
        "execution_identity": execution.get("identity") == IDENTITY,
        "missing_policy_fail_closed": (
            config.get("missing_state_policy") == "fail_closed"
        ),
        "state_gate_not_reused": (
            config["state_and_latency_policy"][
                "v4_100000_state_gate_reused"
            ]
            is False
        ),
        "automatic_downstream_false": (
            config.get("automatic_downstream") is False
            and execution.get("automatic_downstream") is False
        ),
        "crplan_schema_v2": CRPLAN_SCHEMA == "h1_crplan_v2",
    }
    for mode in MODES:
        validate_contract(
            config=config,
            ledger=ledger,
            ledger_sha256=ledger_sha256,
            mode=mode,
        )
    reachability = None
    if not args.defer_runtime_contract:
        reachability = load_frozen_smact_table(
            max_atoms=int(config["max_atoms"]),
            missing_state_policy="fail_closed",
        )
        checks.update(
            {
                "missing_policy_runtime": (
                    reachability.missing_state_policy == "fail_closed"
                ),
                "missing_nonshortcut_fails_closed": (
                    reachability.terminal_certificate(
                        {"Na": 1, "Xe": 1}
                    ).terminal_allowed
                    is False
                ),
                "unary_shortcut_preserved": (
                    reachability.terminal_certificate(
                        {"Xe": 2}
                    ).terminal_allowed
                    is True
                ),
                "all_metal_shortcut_preserved": (
                    reachability.terminal_certificate(
                        {"Fe": 1, "Cu": 1}
                    ).terminal_allowed
                    is True
                ),
            }
        )
    if args.source_manifest is not None:
        checks["source_manifest_exists"] = args.source_manifest.is_file()
        checks["source_manifest_sha_nonempty"] = bool(
            sha256_file(args.source_manifest)
        )
    if args.model_path is not None:
        checks["base_model_exists"] = args.model_path.is_dir()
    if args.checkpoint_path is not None:
        checks["checkpoint_exists"] = args.checkpoint_path.is_dir()
        adapter = args.checkpoint_path / "adapter_model.safetensors"
        checks["adapter_sha256"] = (
            adapter.is_file()
            and sha256_file(adapter) == config["adapter_model_sha256"]
        )
    if args.v4_terminal is not None:
        checks["v4_terminal_unchanged"] = (
            sha256_file(args.v4_terminal)
            == config["frozen_predecessors"]["v4_terminal_sha256"]
        )
    if args.e1_terminal is not None:
        checks["e1_terminal_unchanged"] = (
            sha256_file(args.e1_terminal)
            == config["frozen_predecessors"]["e1_terminal_sha256"]
        )
    passed = all(checks.values())
    report = {
        "schema": "h1_crplan_fourarm512_preflight_report_v1",
        "identity": IDENTITY,
        "status": "pass" if passed else "fail",
        "checks": checks,
        "science_ledger_sha256": ledger_sha256,
        "oxidation_table_sha256": (
            None if reachability is None else reachability.table_sha256
        ),
        "constraint_contract_sha256": (
            None
            if reachability is None
            else reachability.constraint_contract_sha256
        ),
        "runtime_contract_deferred_to_a800": bool(
            args.defer_runtime_contract
        ),
        "network": False,
        "gpu_used": False,
        "model_loaded": False,
        "generation": False,
        "body": False,
        "direct": False,
        "sun": False,
        "downstream": False,
    }
    if args.output is not None:
        if args.output.exists():
            raise FileExistsError(f"refusing to overwrite {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, sort_keys=True))
    if not passed:
        raise SystemExit(2)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--execution-manifest", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--v4-terminal", type=Path)
    parser.add_argument("--e1-terminal", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--defer-runtime-contract", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
