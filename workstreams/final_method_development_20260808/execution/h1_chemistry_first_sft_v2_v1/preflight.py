#!/usr/bin/env python3
"""Fail-closed local and A800 preflight for chemistry-first Planner SFT."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping


IDENTITY = "h1_chemistry_first_sft_v2_smact_split_v2"
LEDGER_SCHEMA = "h1_chemistry_first_planner_science_ledger_v1"
EXPECTED_COUNTS = {"train": 27136, "val": 9047, "test": 9046}
EXPECTED_LEGACY_SMACT = "3.1.0"
EXPECTED_SECONDARY_SMACT = "4.0.0"
EXPECTED_SMACT4_WHEEL_SHA256 = (
    "e3eb968da92d47a8ef9a4af42af5589a6de61cccbca9d329937e1f4e402f0551"
)
EXPECTED_LEGACY_EVALUATOR_SHA256 = (
    "ca1c94f583e0c97a172b5c9b7ba96505257fd74dedfc618b584c34486ac1f178"
)
EXPECTED_SMACT4_CONTRACT_SHA256 = (
    "ad070f3ad8d025ec9d7918dc1aa84e3f8faaf4ee0a124fbe8602208d5609ca19"
)
EXPECTED_PROMPT_SHA256 = (
    "b6231c94cda80ee9762ea247827ef98c69c4226d8130ea85be9fee0fc2f53bcd"
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


def validate_ledger(path: Path, denominator: int, base_seed: int) -> dict[str, Any]:
    from crystal_dlm.ordinal_rng import derive_ordinal_seed

    value = read_json(path)
    rows = value.get("rows")
    if (
        value.get("schema") != LEDGER_SCHEMA
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


def runtime_probe(
    python: Path,
    *,
    source_root: Path,
    expected_smact: str,
) -> dict[str, Any]:
    if not python.is_file():
        raise FileNotFoundError(python)
    code = r'''
import json, os, sys
from importlib.metadata import version
import peft, torch, transformers
payload = {
    "python": sys.version.split()[0],
    "smact": version("SMACT"),
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "peft": getattr(peft, "__version__", "unknown"),
    "cuda_available": bool(torch.cuda.is_available()),
}
print(json.dumps(payload, sort_keys=True))
'''
    env = dict(os.environ)
    env.update(
        {
            "PYTHONPATH": str(source_root),
            "CUDA_VISIBLE_DEVICES": "",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "USE_TORCH": "1",
            "USE_TF": "0",
            "USE_FLAX": "0",
        }
    )
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
    return report


def focused_tests(python: Path, source_root: Path) -> dict[str, Any]:
    modules = (
        "tests.test_h1_chemistry_first_sft",
        "tests.test_h1_chemistry_first_optimizer_audit",
        "tests.test_h1_nocharge_sft_tokenizer_audit",
        "tests.test_h1_nocharge_planner_gate",
        "tests.test_h1_nocharge_ion_aux",
        "tests.test_peft_adapter_identity",
    )
    env = dict(os.environ)
    env.update(
        {
            "PYTHONPATH": str(source_root),
            "CUDA_VISIBLE_DEVICES": "",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
        }
    )
    completed = subprocess.run(
        [str(python), "-m", "unittest", *modules],
        cwd=source_root,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return {
        "modules": list(modules),
        "passed": True,
        "stdout_tail": completed.stdout.splitlines()[-5:],
        "stderr_tail": completed.stderr.splitlines()[-8:],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    source_root = args.source_root.resolve()
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from crystal_dlm.h1_chemistry_first_sft import (
        H1_CHEMISTRY_FIRST_INFERENCE_MESSAGES_SHA256,
        H1_CHEMISTRY_FIRST_SFT_SEED,
    )

    config = read_json(args.config)
    authorization = read_json(args.authorization)
    execution = read_json(args.config.parent / "EXECUTION_MANIFEST.json")
    checks: dict[str, bool] = {
        "identity": config.get("identity") == IDENTITY,
        "authorization": authorization.get("identity") == IDENTITY
        and authorization.get("user_authorized") is True,
        "always_execute_both": authorization.get("always_execute_candidates")
        == ["sft_v2", "sft_v2_c"],
        "execution_manifest": execution.get("identity") == IDENTITY,
        "no_automatic_rl": config.get("automatic_rl") is False
        and execution.get("automatic_rl") is False,
        "no_automatic_downstream": config.get("automatic_downstream") is False
        and execution.get("automatic_downstream") is False,
        "dlm_frozen": config.get("body_dlm_changed") is False
        and execution.get("dlm_model_changed") is False,
        "six_line_nocharge_prompt": config["prompt"]["generated_charge_field"]
        is False
        and config["prompt"]["inference_messages_sha256"] == EXPECTED_PROMPT_SHA256
        and H1_CHEMISTRY_FIRST_INFERENCE_MESSAGES_SHA256 == EXPECTED_PROMPT_SHA256,
        "training_seed": config["training"]["seed"] == H1_CHEMISTRY_FIRST_SFT_SEED,
        "optimizer_contract": config["training"]["batch_size"] == 1
        and config["training"]["gradient_accumulation"] == 8
        and config["training"]["learning_rate"] == 0.000002
        and config["training"]["weight_decay"] == 0.0
        and config["training"]["epochs"] == 1
        and config["training"]["drop_last"] is False
        and config["training"]["repeat_records"] is False,
        "dual_evaluator_contract": config["evaluators"]["legacy"]["smact_version"]
        == EXPECTED_LEGACY_SMACT
        and config["evaluators"]["legacy"]["composition_validity_sha256"]
        == EXPECTED_LEGACY_EVALUATOR_SHA256
        and config["evaluators"]["secondary"]["smact_version"]
        == EXPECTED_SECONDARY_SMACT
        and config["evaluators"]["secondary"]["contract_sha256"]
        == EXPECTED_SMACT4_CONTRACT_SHA256,
        "a800_smact3_only": config.get("a800_smact_version")
        == EXPECTED_LEGACY_SMACT
        and config.get("a800_smact4_execution") is False
        and execution.get("a800_smact4_execution") is False,
        "local_smact4_ledger_contract": config["local_smact4_ledger"][
            "execution_location"
        ]
        == "local_windows_only"
        and config["local_smact4_ledger"]["a800_execution"] is False
        and config["local_smact4_ledger"]["wheel_sha256"]
        == EXPECTED_SMACT4_WHEEL_SHA256
        and config["local_smact4_ledger"]["contract_sha256"]
        == EXPECTED_SMACT4_CONTRACT_SHA256,
    }
    evaluator = source_root / "crystal_dlm/composition_validity.py"
    checks["legacy_evaluator_source_sha"] = evaluator.is_file() and sha256_file(
        evaluator
    ) == EXPECTED_LEGACY_EVALUATOR_SHA256
    ledgers = {
        "planner64": validate_ledger(
            args.ledger64, 64, int(config["planner"]["stage64_base_seed"])
        ),
        "planner256": validate_ledger(
            args.ledger256, 256, int(config["planner"]["stage256_base_seed"])
        ),
    }
    checks["independent_seeds"] = (
        ledgers["planner64"]["base_seed"] != ledgers["planner256"]["base_seed"]
        and config["training"]["seed"]
        not in {ledgers["planner64"]["base_seed"], ledgers["planner256"]["base_seed"]}
    )

    runtimes: dict[str, Any] = {}
    local_smact4_contract: dict[str, Any] = {
        "execution_location": "local_windows_only",
        "a800_execution": False,
        "wheel_sha256": EXPECTED_SMACT4_WHEEL_SHA256,
        "contract_sha256": EXPECTED_SMACT4_CONTRACT_SHA256,
    }
    assets: dict[str, Any] = {}
    tests: dict[str, Any] = {}
    if not args.defer_remote:
        legacy_python = args.legacy_python.resolve()
        runtimes["legacy"] = runtime_probe(
            legacy_python,
            source_root=source_root,
            expected_smact=EXPECTED_LEGACY_SMACT,
        )
        checks["runtime_firewall"] = runtimes["legacy"]["smact"] == EXPECTED_LEGACY_SMACT
        checks["no_smact4_runtime_probe_on_a800"] = True
        tests = focused_tests(legacy_python, source_root)
        checks["focused_tests"] = tests["passed"]
        for name, path in (
            ("model", args.model_path),
            ("p0_adapter", args.p0_adapter_path),
            ("mp20", args.mp20_dir),
        ):
            checks[f"asset_{name}"] = path is not None and path.is_dir()
        adapter_weight = args.p0_adapter_path / "adapter_model.safetensors"
        checks["p0_adapter_sha"] = adapter_weight.is_file() and sha256_file(
            adapter_weight
        ) == config["p0_adapter_weight_sha256"]
        adapter_config = args.p0_adapter_path / "adapter_config.json"
        checks["p0_adapter_config_sha"] = adapter_config.is_file() and sha256_file(
            adapter_config
        ) == config["p0_adapter_config_sha256"]
        for split, expected in EXPECTED_COUNTS.items():
            path = args.mp20_dir / f"{split}.csv"
            observed = csv_record_count(path)
            checks[f"mp20_{split}_count"] = observed == expected
            assets[f"mp20_{split}"] = {
                "path": str(path),
                "count": observed,
                "sha256": sha256_file(path),
            }
        inventory = source_root / "SOURCE_SHA256.txt"
        checks["source_inventory_exists"] = inventory.is_file()
        if args.expected_source_inventory_sha256:
            checks["source_inventory_identity"] = sha256_file(
                inventory
            ) == args.expected_source_inventory_sha256

    passed = all(checks.values())
    report = {
        "schema": "h1_chemistry_first_preflight_v1",
        "identity": IDENTITY,
        "status": "pass" if passed else "fail",
        "checks": checks,
        "ledgers": ledgers,
        "runtimes": runtimes,
        "local_smact4_ledger_contract": local_smact4_contract,
        "assets": assets,
        "focused_tests": tests,
        "remote_checks_deferred": bool(args.defer_remote),
        "network": False,
        "generation": False,
        "training": False,
        "body_or_downstream": False,
        "smact4_executed_on_a800": False,
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
            args.model_path,
            args.p0_adapter_path,
            args.mp20_dir,
        )
    ):
        parser.error("remote preflight requires the legacy runtime and all asset paths")
    return args


if __name__ == "__main__":
    run(parse_args())
