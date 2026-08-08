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


IDENTITY = "h1_chemistry_first_sft_v2_v1"
LEDGER_SCHEMA = "h1_chemistry_first_planner_science_ledger_v1"
EXPECTED_COUNTS = {"train": 27136, "val": 9047, "test": 9046}
EXPECTED_LEGACY_SMACT = "3.1.0"
EXPECTED_SECONDARY_SMACT = "4.0.0"
EXPECTED_SMACT4_RUNTIME_IDENTITY = "smact4_400_runtime_v1"
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
    secondary: bool,
    tokenizer_path: Path,
) -> dict[str, Any]:
    if not python.is_file():
        raise FileNotFoundError(python)
    code = r'''
import json, os, sys
from importlib.metadata import version
payload = {"python": sys.version.split()[0], "smact": version("SMACT")}
if os.environ["H1_SECONDARY"] == "1":
    import transformers
    from transformers import AutoTokenizer
    from crystal_dlm.h1_nocharge_ion_aux import load_smact4_icsd24_oxidation_map
    mapping, contract = load_smact4_icsd24_oxidation_map()
    tokenizer = AutoTokenizer.from_pretrained(
        os.environ["H1_TOKENIZER_PATH"], trust_remote_code=True, use_fast=True
    )
    payload.update({
        "transformers": transformers.__version__,
        "contract_sha256": contract["contract_sha256"],
        "oxidation_elements": len(mapping),
        "tokenizer_vocab": len(tokenizer),
    })
else:
    import peft, torch, transformers
    payload.update({
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": getattr(peft, "__version__", "unknown"),
        "cuda_available": bool(torch.cuda.is_available()),
    })
print(json.dumps(payload, sort_keys=True))
'''
    env = dict(os.environ)
    env.update(
        {
            "PYTHONPATH": str(source_root),
            "H1_SECONDARY": "1" if secondary else "0",
            "H1_TOKENIZER_PATH": str(tokenizer_path),
            "CUDA_VISIBLE_DEVICES": "",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "USE_TORCH": "0" if secondary else "1",
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
    if secondary and report.get("contract_sha256") != EXPECTED_SMACT4_CONTRACT_SHA256:
        raise RuntimeError("SMACT4 oxidation contract mismatch")
    return report


def validate_shared_smact4_runtime(
    python: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    runtime_config = config.get("shared_smact4_runtime")
    if not isinstance(runtime_config, Mapping):
        raise RuntimeError("shared SMACT4 runtime config is missing")
    runtime_root = python.parent.resolve()
    expected_root = Path(str(runtime_config.get("run_root", ""))).resolve()
    if python.name != "python" or runtime_root != expected_root:
        raise RuntimeError("shared SMACT4 wrapper path is not the frozen runtime")
    terminal_path = runtime_root / "terminal_report.json"
    terminal_sha_path = runtime_root / "terminal_report.sha256"
    success_path = runtime_root / "_SUCCESS"
    copied_wheel = runtime_root / str(runtime_config.get("wheel", ""))
    base_python_path_file = runtime_root / "base_python_path.txt"
    site_inventory = runtime_root / "SITE_SHA256.txt"
    for path in (
        terminal_path,
        terminal_sha_path,
        success_path,
        copied_wheel,
        base_python_path_file,
        site_inventory,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    terminal = read_json(terminal_path)
    success = read_json(success_path)
    terminal_sha256 = sha256_file(terminal_path)
    recorded_sha = terminal_sha_path.read_text(encoding="utf-8").split()[0]
    base_python_relative = Path(
        base_python_path_file.read_text(encoding="utf-8").strip()
    )
    if base_python_relative.is_absolute() or ".." in base_python_relative.parts:
        raise RuntimeError("shared SMACT4 base Python path is unsafe")
    base_python = (runtime_root / base_python_relative).resolve()
    if runtime_root not in base_python.parents:
        raise RuntimeError("shared SMACT4 base Python escapes runtime root")
    checks = {
        "schema": terminal.get("schema") == "smact4_400_runtime_manifest_v2",
        "identity": terminal.get("identity") == EXPECTED_SMACT4_RUNTIME_IDENTITY,
        "status": terminal.get("status") == "pass",
        "wheel_sha": terminal.get("wheel_sha256") == EXPECTED_SMACT4_WHEEL_SHA256
        and sha256_file(copied_wheel) == EXPECTED_SMACT4_WHEEL_SHA256,
        "wrapper_sha": terminal.get("wrapper_sha256") == sha256_file(python),
        "bundle_identity": terminal.get("bundle_archive")
        == runtime_config.get("bundle_archive")
        and terminal.get("bundle_archive_sha256")
        == runtime_config.get("bundle_archive_sha256")
        and terminal.get("bundle_manifest_sha256")
        == runtime_config.get("bundle_manifest_sha256")
        and terminal.get("bundle_validation", {}).get("wheel_count")
        == runtime_config.get("wheel_count"),
        "probe": terminal.get("probe", {}).get("smact") == EXPECTED_SECONDARY_SMACT
        and terminal.get("probe", {}).get("contract_sha256")
        == EXPECTED_SMACT4_CONTRACT_SHA256,
        "supported_python": [3, 11, 0]
        <= list(terminal.get("base_python_version") or [])
        < [3, 14, 0],
        "base_python_identity": base_python_relative.as_posix()
        == terminal.get("base_python_relative")
        and base_python.is_file()
        and sha256_file(base_python) == terminal.get("base_python_sha256"),
        "site_inventory": terminal.get("site_inventory_sha256")
        == sha256_file(site_inventory)
        and int(terminal.get("site_file_count", 0)) > 0,
        "terminal_sha": recorded_sha == terminal_sha256
        and success.get("terminal_sha256") == terminal_sha256,
        "success_identity": success.get("identity")
        == EXPECTED_SMACT4_RUNTIME_IDENTITY,
        "offline": terminal.get("network") is False
        and terminal.get("global_environment_mutation") is False
        and terminal.get("user_site_isolation") is True
        and terminal.get("atomic_publish") is True,
    }
    if not all(checks.values()):
        failed = sorted(name for name, value in checks.items() if not value)
        raise RuntimeError(f"shared SMACT4 runtime terminal mismatch: {failed}")
    return {
        "root": str(runtime_root),
        "terminal_sha256": terminal_sha256,
        "wheel_sha256": EXPECTED_SMACT4_WHEEL_SHA256,
        "bundle_archive_sha256": runtime_config.get("bundle_archive_sha256"),
        "bundle_manifest_sha256": runtime_config.get("bundle_manifest_sha256"),
        "checks": checks,
    }


def focused_tests(python: Path, source_root: Path) -> dict[str, Any]:
    modules = (
        "tests.test_h1_chemistry_first_sft",
        "tests.test_h1_nocharge_sft_tokenizer_audit",
        "tests.test_h1_nocharge_planner_gate",
        "tests.test_h1_nocharge_ion_aux",
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
    shared_runtime: dict[str, Any] = {}
    assets: dict[str, Any] = {}
    tests: dict[str, Any] = {}
    if not args.defer_remote:
        legacy_python = args.legacy_python.resolve()
        smact4_python = args.smact4_python.resolve()
        if legacy_python == smact4_python:
            raise RuntimeError("legacy and exact SMACT4 runtimes must differ")
        shared_runtime = validate_shared_smact4_runtime(smact4_python, config)
        checks["shared_smact4_runtime_terminal"] = all(
            shared_runtime["checks"].values()
        )
        runtimes["legacy"] = runtime_probe(
            legacy_python,
            source_root=source_root,
            expected_smact=EXPECTED_LEGACY_SMACT,
            secondary=False,
            tokenizer_path=args.model_path,
        )
        runtimes["secondary"] = runtime_probe(
            smact4_python,
            source_root=source_root,
            expected_smact=EXPECTED_SECONDARY_SMACT,
            secondary=True,
            tokenizer_path=args.model_path,
        )
        checks["runtime_firewall"] = (
            runtimes["legacy"]["smact"] == EXPECTED_LEGACY_SMACT
            and runtimes["secondary"]["smact"] == EXPECTED_SECONDARY_SMACT
            and runtimes["secondary"]["tokenizer_vocab"] > 100000
        )
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
        "shared_smact4_runtime": shared_runtime,
        "assets": assets,
        "focused_tests": tests,
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
