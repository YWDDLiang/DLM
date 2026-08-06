#!/usr/bin/env python3
"""Fail-closed local/remote preflight for the no-charge ion-aux SFT route."""

from __future__ import annotations

import argparse
import csv
import hashlib
from importlib.metadata import version as package_version
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping


IDENTITY = "h1_nocharge_ion_aux_sft_v1"
LEGACY_SMACT = "3.1.0"
SECONDARY_SMACT = "4.0.0"
EXPECTED_COUNTS = {"train": 27136, "val": 9047, "test": 9046}
EXPECTED_LEGACY_EVALUATOR_SHA256 = (
    "ca1c94f583e0c97a172b5c9b7ba96505257fd74dedfc618b584c34486ac1f178"
)
EXPECTED_SMACT4_CONTRACT_SHA256 = (
    "ad070f3ad8d025ec9d7918dc1aa84e3f8faaf4ee0a124fbe8602208d5609ca19"
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
        raise ValueError(f"{path} must contain a JSON object")
    return value


def csv_record_count(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def runtime_probe(
    python: Path,
    *,
    source_root: Path,
    expected_smact: str,
    secondary: bool,
) -> dict[str, Any]:
    if not python.is_file():
        raise FileNotFoundError(python)
    code = r'''
import json, sys
from importlib.metadata import version
payload = {
    "python": sys.version.split()[0],
    "smact": version("SMACT"),
}
if __import__("os").environ.get("H1_SECONDARY") == "1":
    from crystal_dlm.h1_nocharge_ion_aux import load_smact4_icsd24_oxidation_map
    mapping, contract = load_smact4_icsd24_oxidation_map()
    payload["contract_sha256"] = contract["contract_sha256"]
    payload["oxidation_elements"] = len(mapping)
else:
    import torch, transformers, peft
    payload.update({
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": getattr(peft, "__version__", "unknown"),
        "cuda_available": bool(torch.cuda.is_available()),
    })
print(json.dumps(payload, sort_keys=True))
'''
    env = dict(os.environ)
    env["PYTHONPATH"] = str(source_root)
    env["H1_SECONDARY"] = "1" if secondary else "0"
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    completed = subprocess.run(
        [str(python), "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"runtime probe returned no JSON: {python}")
    report = json.loads(lines[-1])
    if report.get("smact") != expected_smact:
        raise RuntimeError(
            f"{python} has SMACT {report.get('smact')!r}, expected {expected_smact}"
        )
    if secondary and report.get("contract_sha256") != EXPECTED_SMACT4_CONTRACT_SHA256:
        raise RuntimeError("SMACT4 oxidation contract mismatch")
    return report


def validate_ledger(path: Path, denominator: int, base_seed: int) -> dict[str, Any]:
    from crystal_dlm.ordinal_rng import derive_ordinal_seed

    value = read_json(path)
    rows = value.get("rows")
    if (
        value.get("schema") != "h1_nocharge_planner_science_ledger_v1"
        or value.get("identity") != IDENTITY
        or int(value.get("denominator", -1)) != denominator
        or int(value.get("base_seed", -1)) != base_seed
        or value.get("independent_from_training_and_other_stages") is not True
        or not isinstance(rows, list)
        or len(rows) != denominator
    ):
        raise RuntimeError(f"ledger contract mismatch: {path}")
    for ordinal, row in enumerate(rows):
        expected = derive_ordinal_seed(
            base_seed,
            sample_idx=ordinal,
            stage="planner_sampling",
            role="shared",
        )
        if (
            not isinstance(row, Mapping)
            or int(row.get("ordinal", -1)) != ordinal
            or int(row.get("planner_sampling_seed", -1)) != expected
            or row.get("role") != "shared"
        ):
            raise RuntimeError(f"ledger ordinal mismatch at {ordinal}: {path}")
    return {
        "denominator": denominator,
        "base_seed": base_seed,
        "sha256": sha256_file(path),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    source_root = args.source_root.resolve()
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    config = read_json(args.config)
    authorization = read_json(args.authorization)
    execution = read_json(args.config.parent / "EXECUTION_MANIFEST.json")
    checks: dict[str, bool] = {
        "identity": config.get("identity") == IDENTITY,
        "authorization": (
            authorization.get("identity") == IDENTITY
            and authorization.get("user_authorized") is True
        ),
        "execution_manifest": (
            execution.get("identity") == IDENTITY
            and execution.get("automatic_rl") is False
            and execution.get("automatic_downstream") is False
            and execution.get("dlm_model_changed") is False
        ),
        "dlm_frozen": config["downstream"]["body_dlm_changed"] is False,
        "automatic_rl_false": config.get("automatic_rl") is False,
        "automatic_downstream_false": config.get("automatic_downstream") is False,
        "generated_charge_false": (
            authorization["explicit_constraints"]["generated_charge_field"] is False
        ),
        "fixed_endpoint": config["training"]["fixed_endpoint"] == "checkpoint-0400",
        "optimizer_geometry": (
            config["training"]["batch_size"] == 1
            and config["training"]["gradient_accumulation"] == 8
            and config["training"]["updates"] == 400
            and config["training"]["batch_size"]
            * config["training"]["gradient_accumulation"]
            * config["training"]["updates"]
            == config["data"]["train_records_per_arm"]
        ),
        "dual_evaluator_contract": (
            config["legacy_evaluator"]["smact_version"] == LEGACY_SMACT
            and config["legacy_evaluator"]["composition_validity_sha256"]
            == EXPECTED_LEGACY_EVALUATOR_SHA256
            and config["secondary_evaluator"]["smact_version"] == SECONDARY_SMACT
            and config["secondary_evaluator"]["contract_sha256"]
            == EXPECTED_SMACT4_CONTRACT_SHA256
        ),
    }
    evaluator = source_root / "crystal_dlm/composition_validity.py"
    checks["legacy_evaluator_source_sha"] = (
        evaluator.is_file()
        and sha256_file(evaluator) == EXPECTED_LEGACY_EVALUATOR_SHA256
    )
    ledgers = {
        "planner64": validate_ledger(
            args.ledger64, 64, int(config["planner"]["stage64_base_seed"])
        ),
        "planner256": validate_ledger(
            args.ledger256, 256, int(config["planner"]["stage256_base_seed"])
        ),
    }
    checks["independent_stage_base_seeds"] = (
        ledgers["planner64"]["base_seed"] != ledgers["planner256"]["base_seed"]
        and ledgers["planner64"]["base_seed"] != config["training"]["seed"]
        and ledgers["planner256"]["base_seed"] != config["training"]["seed"]
    )

    runtimes: dict[str, Any] = {}
    assets: dict[str, Any] = {}
    if not args.defer_remote:
        legacy_python = args.legacy_python.resolve()
        smact4_python = args.smact4_python.resolve()
        if legacy_python == smact4_python:
            raise RuntimeError("legacy and SMACT4 evaluators must use different runtimes")
        runtimes["legacy"] = runtime_probe(
            legacy_python,
            source_root=source_root,
            expected_smact=LEGACY_SMACT,
            secondary=False,
        )
        runtimes["secondary"] = runtime_probe(
            smact4_python,
            source_root=source_root,
            expected_smact=SECONDARY_SMACT,
            secondary=True,
        )
        checks["runtime_firewall"] = (
            runtimes["legacy"]["smact"] == LEGACY_SMACT
            and runtimes["secondary"]["smact"] == SECONDARY_SMACT
        )
        for name, path in (
            ("model", args.model_path),
            ("p0_adapter", args.p0_adapter_path),
            ("mp20", args.mp20_dir),
        ):
            checks[f"asset_{name}"] = path is not None and path.is_dir()
        adapter_weight = args.p0_adapter_path / "adapter_model.safetensors"
        checks["p0_adapter_sha"] = (
            adapter_weight.is_file()
            and sha256_file(adapter_weight) == config["p0_adapter_weight_sha256"]
        )
        for split, expected in EXPECTED_COUNTS.items():
            csv_path = args.mp20_dir / f"{split}.csv"
            observed = csv_record_count(csv_path)
            checks[f"mp20_{split}_count"] = observed == expected
            assets[f"mp20_{split}"] = {
                "path": str(csv_path),
                "count": observed,
                "sha256": sha256_file(csv_path),
            }
        inventory = source_root / "SOURCE_SHA256.txt"
        checks["source_inventory_exists"] = inventory.is_file()
        if args.expected_source_inventory_sha256:
            checks["source_inventory_identity"] = (
                sha256_file(inventory) == args.expected_source_inventory_sha256
            )

    passed = all(checks.values())
    report = {
        "schema": "h1_nocharge_ion_aux_preflight_v1",
        "identity": IDENTITY,
        "status": "pass" if passed else "fail",
        "checks": checks,
        "ledgers": ledgers,
        "runtimes": runtimes,
        "assets": assets,
        "remote_checks_deferred": bool(args.defer_remote),
        "network": False,
        "generation": False,
        "training": False,
        "body_or_downstream": False,
    }
    if args.output:
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(report, sort_keys=True))
    if not passed:
        raise SystemExit(2)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--ledger64", type=Path, required=True)
    parser.add_argument("--ledger256", type=Path, required=True)
    parser.add_argument("--legacy-python", type=Path)
    parser.add_argument("--smact4-python", type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--p0-adapter-path", type=Path)
    parser.add_argument("--mp20-dir", type=Path)
    parser.add_argument("--expected-source-inventory-sha256")
    parser.add_argument("--defer-remote", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.defer_remote and any(
        value is None
        for value in (
            args.legacy_python,
            args.smact4_python,
            args.model_path,
            args.p0_adapter_path,
            args.mp20_dir,
        )
    ):
        parser.error("remote preflight requires both runtimes and all asset paths")
    return args


if __name__ == "__main__":
    run(parse_args())
