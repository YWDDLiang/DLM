#!/usr/bin/env python3
"""Fail closed unless the recovery run matches the historical best refiner."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_hash(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"{label} SHA256 changed: {observed}")
    return observed


def compact_sha256(value: Any) -> str:
    payload = json.dumps(value, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def int_value(node: ast.AST, constants: dict[str, int]) -> int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    if isinstance(node, ast.Name):
        return constants.get(node.id)
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, int)
    ):
        return -node.operand.value
    return None


def wrapper_contract(path: Path) -> dict[str, Any]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    constants: dict[str, int] = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            value = int_value(node.value, constants)
            if value is not None:
                constants[node.targets[0].id] = value

    model_timesteps: list[int] = []
    sample_steps: list[int] = []
    loader_batches: list[tuple[int | None, bool | None]] = []
    called: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id == "CSPDiffusion":
            if node.args:
                value = int_value(node.args[0], constants)
                if value is not None:
                    model_timesteps.append(value)
        if isinstance(node.func, ast.Attribute):
            called.add(node.func.attr)
            if node.func.attr == "sample":
                for keyword in node.keywords:
                    if keyword.arg == "diff_steps":
                        value = int_value(keyword.value, constants)
                        if value is not None:
                            sample_steps.append(value)
        if isinstance(node.func, ast.Name) and node.func.id == "DataLoader":
            batch_size: int | None = None
            shuffle: bool | None = None
            for keyword in node.keywords:
                if keyword.arg == "batch_size":
                    batch_size = int_value(keyword.value, constants)
                elif keyword.arg == "shuffle" and isinstance(
                    keyword.value, ast.Constant
                ):
                    shuffle = bool(keyword.value.value)
            loader_batches.append((batch_size, shuffle))

    required_seed_calls = {"seed", "manual_seed", "manual_seed_all"}
    if model_timesteps != [1000] or sample_steps != [800]:
        raise ValueError(
            f"refiner wrapper schedule changed: model={model_timesteps}, "
            f"sample={sample_steps}"
        )
    if loader_batches != [(1, False)]:
        raise ValueError(f"refiner wrapper batch contract changed: {loader_batches}")
    if not required_seed_calls.issubset(called):
        raise ValueError(f"refiner wrapper seed calls changed: {sorted(called)}")
    if "load_state_dict" not in called:
        raise ValueError("refiner wrapper no longer loads the frozen checkpoint")
    return {
        "model_timesteps": model_timesteps[0],
        "reverse_updates": sample_steps[0],
        "effective_batch_size": loader_batches[0][0],
        "shuffle": loader_batches[0][1],
        "seed_calls": sorted(required_seed_calls),
    }


def diffusion_kernel_contract(path: Path) -> dict[str, Any]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    sample = next(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "sample"
        ),
        None,
    )
    if sample is None:
        raise ValueError("CSPDiffusion.sample is missing")
    starts_from_argument = any(
        isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "time_start"
            for target in node.targets
        )
        and isinstance(node.value, ast.Name)
        and node.value.id == "diff_steps"
        for node in ast.walk(sample)
    )
    reverse_loop_wrappers: list[str] = []
    for node in ast.walk(sample):
        if not isinstance(node, ast.For):
            continue
        iterator = node.iter
        wrappers: list[str] = []
        if (
            isinstance(iterator, ast.Call)
            and isinstance(iterator.func, ast.Name)
            and iterator.func.id == "tqdm"
            and len(iterator.args) == 1
        ):
            wrappers.append("tqdm")
            iterator = iterator.args[0]
        if (
            isinstance(iterator, ast.Call)
            and isinstance(iterator.func, ast.Name)
            and iterator.func.id == "range"
            and len(iterator.args) == 3
            and isinstance(iterator.args[0], ast.Name)
            and iterator.args[0].id == "time_start"
            and int_value(iterator.args[1], {}) == 0
            and int_value(iterator.args[2], {}) == -1
        ):
            reverse_loop_wrappers.append("(".join(wrappers) or "direct")
    if not starts_from_argument or reverse_loop_wrappers != ["tqdm"]:
        raise ValueError("diffusion kernel no longer performs diff_steps..1 updates")
    return {
        "time_start_is_diff_steps": True,
        "loop": "tqdm(range(diff_steps, 0, -1))",
        "exact_reverse_updates_when_diff_steps_800": 800,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    refiner = config["refiner"]
    provenance = refiner["historical_best_provenance"]
    if (
        int(refiner["timesteps"]) != 1000
        or int(refiner["diffusion_steps"]) != 800
        or int(refiner["num_evals"]) != 1
        or int(refiner["effective_batch_size"]) != 1
    ):
        raise ValueError("configured refiner schedule differs from historical best")

    historical_wrapper = Path(provenance["historical_wrapper"])
    current_wrapper = Path(provenance["current_wrapper"])
    runtime_loader = Path(provenance["runtime_loader"])
    diffusion_kernel = Path(provenance["diffusion_kernel"])
    observed_hashes = {
        "historical_wrapper": require_hash(
            historical_wrapper,
            provenance["historical_wrapper_sha256"],
            "historical best wrapper",
        ),
        "current_wrapper": require_hash(
            current_wrapper,
            provenance["current_wrapper_sha256"],
            "current frozen wrapper",
        ),
        "runtime_loader": require_hash(
            runtime_loader,
            provenance["runtime_loader_sha256"],
            "shared refiner runtime loader",
        ),
        "diffusion_kernel": require_hash(
            diffusion_kernel,
            provenance["diffusion_kernel_sha256"],
            "shared CrysLLMGen diffusion kernel",
        ),
        "checkpoint": require_hash(
            args.checkpoint,
            refiner["checkpoint_sha256"],
            "model_494 checkpoint",
        ),
    }
    historical_contract = wrapper_contract(historical_wrapper)
    current_contract = wrapper_contract(current_wrapper)
    if historical_contract != current_contract:
        raise ValueError(
            "current and historical wrapper scientific contracts differ: "
            f"{historical_contract} != {current_contract}"
        )
    kernel_contract = diffusion_kernel_contract(diffusion_kernel)

    ledger_rows = [
        json.loads(line)
        for line in args.ledger.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if (
        len(ledger_rows) != 256
        or [int(row.get("ordinal", -1)) for row in ledger_rows] != list(range(256))
        or any(not isinstance(row.get("refiner_noise_seed"), int) for row in ledger_rows)
    ):
        raise ValueError("historical refiner seed ledger changed")
    seed_vector = [int(row["refiner_noise_seed"]) for row in ledger_rows]
    seed_vector_sha256 = compact_sha256(seed_vector)
    if seed_vector_sha256 != provenance["refiner_seed_vector_sha256"]:
        raise ValueError("historical refiner seed vector changed")

    report = {
        "schema": "h1a2_historical_best_refiner_contract_audit_v1",
        "status": "pass",
        "historical_best_run": provenance["historical_best_run"],
        "historical_best_repeat_semantics": provenance["repeat_semantics"],
        "historical_clean_strict_sun_counts": provenance[
            "historical_clean_strict_sun_counts"
        ],
        "historical_clean_meta_sun_counts": provenance[
            "historical_clean_meta_sun_counts"
        ],
        "historical_refined_counts": provenance["historical_refined_counts"],
        "scientific_wrapper_contract": current_contract,
        "diffusion_kernel_contract": kernel_contract,
        "refiner_seed_vector_sha256": seed_vector_sha256,
        "same_seed_vector_across_process_repeats": True,
        "checkpoint_rehashed": True,
        "observed_sha256": observed_hashes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
