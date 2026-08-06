#!/usr/bin/env python3
"""Run the immutable 4x8 schedule-correct parent bridge preflight.

This is an evaluation-only diagnostic.  It reuses eight hash-fixed clean WQ
proposals, strict-loads the released CrysLLMGen parent, creates the exact
training-time forward-noised state at four timesteps, and executes one
32-step reverse trajectory per cell.  It never generates proposals, trains,
relaxes, queries an API, or uses an MLIP.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crystal_dlm.wqcodiff.crysllmgen.atom_sampling import (  # noqa: E402
    _output_structure,
    expanded_state_to_parent_batch,
    load_registered_csp,
)
from crystal_dlm.wqcodiff.crysllmgen.bridge_parity import (  # noqa: E402
    BRIDGE_CELL_COUNT,
    CleanProposalCondition,
    build_bridge_cells,
    build_numpy_parent_schedules,
    forward_noise_numpy,
    forward_noise_torch,
    reconstruction_errors,
    run_parent_reverse_from_noisy_state,
)
from crystal_dlm.wqcodiff.crysllmgen.schedules import (  # noqa: E402
    build_beta_tables,
    build_coordinate_sigmas,
)
from crystal_dlm.wqcodiff.contracts import write_json_exclusive  # noqa: E402
from crystal_dlm.wqcodiff.crysllmgen.gate import sha256_file  # noqa: E402
from crystal_dlm.wqcodiff.runtime import expand_state  # noqa: E402
from crystal_dlm.wqcodiff.charts import PyXtalChartCatalog  # noqa: E402
from crystal_dlm.wqcodiff.state import StratifiedState  # noqa: E402


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _require_sha256(value: str, *, name: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be one lowercase SHA256")
    return value


def _load_contract(path: Path) -> tuple[dict[str, Any], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "wq_schedule_correct_bridge_parity_contract_v1":
        raise ValueError("unexpected bridge-parity contract schema")
    if payload["matrix"]["total_cells"] != BRIDGE_CELL_COUNT:
        raise ValueError("bridge-parity contract no longer contains 32 cells")
    if payload["model_selection"]["training_performed"]:
        raise ValueError("bridge-parity contract unexpectedly enables training")
    if not payload["model_selection"]["mlip_free"]:
        raise ValueError("bridge-parity contract is not MLIP-free")
    return payload, sha256_file(path)


def _load_source_rows(path: Path, source: Mapping[str, Any]) -> list[dict[str, Any]]:
    if sha256_file(path) != source["generation_jsonl_sha256"]:
        raise ValueError("source generation SHA256 differs from the frozen contract")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except Exception as exc:
                raise ValueError(f"invalid source JSONL row {line_number}") from exc
            rows.append(row)
    if len(rows) != int(source["rows"]):
        raise ValueError("source generation row count differs from the frozen contract")
    attempt_ids = [str(row.get("attempt_id", "")) for row in rows]
    if not all(attempt_ids) or len(set(attempt_ids)) != len(attempt_ids):
        raise ValueError("source generation attempt identities are missing or duplicated")
    for row in rows:
        if row.get("schema") != source["required_schema"]:
            raise ValueError("source generation schema differs from the frozen contract")
        if row.get("status") != source["required_status"]:
            raise ValueError("source panel contains a non-success row")
        if not isinstance(row.get(source["proposal_field"]), Mapping):
            raise ValueError("source panel row has no clean proposal state")
    return rows


def _select_rows(
    rows: list[dict[str, Any]], source: Mapping[str, Any]
) -> list[dict[str, Any]]:
    salt = str(source["selection_salt"])
    ranked = sorted(
        rows,
        key=lambda row: (
            hashlib.sha256(
                f"{salt}:{row['attempt_id']}".encode("utf-8")
            ).hexdigest(),
            row["attempt_id"],
        ),
    )
    selected = ranked[: int(source["selection_count"])]
    if len(selected) != 8:
        raise ValueError("bridge-parity selection did not produce exactly eight rows")
    return selected


def _module_state_sha256(module: Any) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _max_abs(first: Any, second: Any) -> float:
    left = first.detach().double().cpu()
    right = second.detach().double().cpu()
    if left.shape != right.shape:
        raise ValueError(f"schedule shapes differ: {left.shape} versus {right.shape}")
    return float(torch.max(torch.abs(left - right)).item())


def _schedule_audit(model: Any, tolerance: float) -> dict[str, Any]:
    derived_beta = build_beta_tables(1000)
    derived_coordinate = build_coordinate_sigmas(1000)
    comparisons = {
        "betas": _max_abs(model.beta_scheduler.betas, derived_beta["betas"]),
        "alphas": _max_abs(model.beta_scheduler.alphas, derived_beta["alphas"]),
        "alphas_cumprod": _max_abs(
            model.beta_scheduler.alphas_cumprod,
            derived_beta["alphas_cumprod"],
        ),
        "posterior_sigmas": _max_abs(
            model.beta_scheduler.sigmas,
            derived_beta["posterior_sigmas"],
        ),
        "coordinate_sigmas": _max_abs(
            model.sigma_scheduler.sigmas,
            derived_coordinate,
        ),
    }
    maximum = max(comparisons.values())
    return {
        "ok": maximum <= tolerance,
        "max_absolute_error": maximum,
        "absolute_errors": comparisons,
        "absolute_tolerance": tolerance,
        "relative_tolerance": 0.0,
        "schedule_tensor_length": len(model.beta_scheduler.alphas_cumprod),
        "scheduler_timesteps": int(model.beta_scheduler.timesteps),
    }


def _torch_reconstruction_errors(bridge_state: Mapping[str, Any]) -> dict[str, float]:
    clean_x = bridge_state["condition_frac_coords"]
    noisy_x = bridge_state["state_frac_coords"]
    coordinate_noise = bridge_state["coordinate_noise"]
    coordinate_sigma = bridge_state["coordinate_sigma"]
    reconstructed_x = (noisy_x - coordinate_sigma * coordinate_noise) % 1.0
    delta = torch.abs(reconstructed_x - clean_x)
    periodic_delta = torch.minimum(delta, 1.0 - delta)

    clean_l = bridge_state["condition_lattice"]
    noisy_l = bridge_state["state_lattice"]
    lattice_noise = bridge_state["lattice_noise"]
    alpha_bar = bridge_state["alpha_bar"]
    reconstructed_l = (
        noisy_l - torch.sqrt(1.0 - alpha_bar) * lattice_noise
    ) / torch.sqrt(alpha_bar)
    return {
        "coordinate_periodic_max_abs_error": float(periodic_delta.max().item()),
        "lattice_max_abs_error": float(
            torch.max(torch.abs(reconstructed_l - clean_l)).item()
        ),
    }


def _write_jsonl_exclusive(path: Path, rows: list[Mapping[str, Any]]) -> str:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    allow_nan=False,
                )
            )
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return sha256_file(path)


def _run(args: argparse.Namespace, output: Path) -> dict[str, Any]:
    started = time.time()
    contract, contract_sha256 = _load_contract(args.contract)
    _require_sha256(args.execution_patch_sha256, name="execution patch")
    if args.device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("bridge parity requires one CUDA A800 allocation")
    device_name = torch.cuda.get_device_name(args.device)
    if "A800" not in device_name:
        raise RuntimeError(f"bridge parity requires A800, observed {device_name}")

    source = contract["source_panel"]
    rows = _load_source_rows(args.source_generation, source)
    selected = _select_rows(rows, source)
    selection_payload = {
        "schema": "wq_schedule_correct_bridge_selection_v1",
        "contract_sha256": contract_sha256,
        "source_generation_sha256": source["generation_jsonl_sha256"],
        "selection_rule": source["selection_rule"],
        "selection_salt": source["selection_salt"],
        "selection_count": len(selected),
        "selection_uses_scientific_outcomes": False,
        "attempt_ids": [row["attempt_id"] for row in selected],
        "proposal_state_sha256": [
            hashlib.sha256(_canonical_json(row["proposal_state"])).hexdigest()
            for row in selected
        ],
    }
    selection_path = output / "selection_manifest.json"
    write_json_exclusive(selection_path, selection_payload)

    if sha256_file(args.checkpoint) != contract["parent"]["checkpoint_sha256"]:
        raise ValueError("parent checkpoint SHA256 differs from the frozen contract")
    model, parent_identity = load_registered_csp(
        snapshot_root=ROOT / "crystal_dlm/wqcodiff/crysllmgen/upstream",
        checkpoint=args.checkpoint,
        device=args.device,
    )
    schedule = _schedule_audit(
        model,
        float(contract["gates"]["derived_vs_strict_loaded_schedule_max_abs_error"]),
    )
    if not schedule["ok"]:
        raise RuntimeError("strict-loaded parent schedule parity failed")
    strict_load = {
        "ok": True,
        "parent_identity": parent_identity,
        "decoder_state_sha256": _module_state_sha256(model.decoder),
        "time_embedding_state_sha256": _module_state_sha256(model.time_embedding),
        "beta_scheduler_state_sha256": _module_state_sha256(model.beta_scheduler),
        "sigma_scheduler_state_sha256": _module_state_sha256(model.sigma_scheduler),
    }

    cells = build_bridge_cells(base_seed=int(contract["matrix"]["base_noise_seed"]))
    numpy_schedules = build_numpy_parent_schedules()
    catalog = PyXtalChartCatalog()
    attempt_rows: list[dict[str, Any]] = []
    successful = 0
    first_step_invalid = 0
    nonfinite = 0
    max_numpy_coordinate_reconstruction_error = 0.0
    max_numpy_lattice_reconstruction_error = 0.0
    max_torch_coordinate_reconstruction_error = 0.0
    max_torch_lattice_reconstruction_error = 0.0

    for cell in cells:
        source_row = selected[cell.panel_index]
        row_started = time.time()
        base = {
            "schema": "wq_schedule_correct_bridge_cell_v1",
            "cell_id": cell.cell_id,
            "timestep": cell.timestep,
            "panel_index": cell.panel_index,
            "source_attempt_id": source_row["attempt_id"],
            "forward_noise_seed": cell.forward_noise_seed,
            "reverse_noise_seed": cell.reverse_noise_seed,
            "retry_or_replacement_used": False,
            "clean_condition_used_as_reverse_state": False,
            "execution_patch_sha256": args.execution_patch_sha256,
        }
        try:
            proposal = StratifiedState.from_dict(dict(source_row["proposal_state"]))
            expanded = expand_state(proposal, catalog)
            batch = expanded_state_to_parent_batch(expanded, args.device)
            data_utils = __import__(
                "models_ddpm.data_utils",
                fromlist=["lattice_params_to_matrix_torch"],
            )
            clean_lattice = data_utils.lattice_params_to_matrix_torch(
                batch.lengths, batch.angles
            )
            clean_condition = CleanProposalCondition(
                frac_coords=batch.frac_coords.detach().double().cpu().numpy() % 1.0,
                lattice=clean_lattice[0].detach().double().cpu().numpy(),
            )
            numpy_input = forward_noise_numpy(
                clean_condition,
                schedules=numpy_schedules,
                cell=cell,
            )
            numpy_errors = reconstruction_errors(numpy_input)
            max_numpy_coordinate_reconstruction_error = max(
                max_numpy_coordinate_reconstruction_error,
                numpy_errors["coordinate_periodic_max_abs_error"],
            )
            max_numpy_lattice_reconstruction_error = max(
                max_numpy_lattice_reconstruction_error,
                numpy_errors["lattice_max_abs_error"],
            )
            coordinate_noise = torch.tensor(
                np.array(numpy_input.state.coordinate_noise, copy=True),
                dtype=batch.frac_coords.dtype,
                device=args.device,
            )
            lattice_noise = torch.tensor(
                np.array(numpy_input.state.lattice_noise, copy=True)[None, :, :],
                dtype=clean_lattice.dtype,
                device=args.device,
            )
            torch_state = forward_noise_torch(
                clean_frac_coords=batch.frac_coords,
                clean_lattice=clean_lattice,
                coordinate_noise=coordinate_noise,
                lattice_noise=lattice_noise,
                timestep=cell.timestep,
                alphas_cumprod=model.beta_scheduler.alphas_cumprod,
                coordinate_sigmas=model.sigma_scheduler.sigmas,
            )
            torch_errors = _torch_reconstruction_errors(torch_state)
            max_torch_coordinate_reconstruction_error = max(
                max_torch_coordinate_reconstruction_error,
                torch_errors["coordinate_periodic_max_abs_error"],
            )
            max_torch_lattice_reconstruction_error = max(
                max_torch_lattice_reconstruction_error,
                torch_errors["lattice_max_abs_error"],
            )
            random.seed(cell.reverse_noise_seed)
            np.random.seed(cell.reverse_noise_seed % (1 << 32))
            torch.manual_seed(cell.reverse_noise_seed)
            torch.cuda.manual_seed_all(cell.reverse_noise_seed)
            output_state, trajectory = run_parent_reverse_from_noisy_state(
                model=model,
                batch=batch,
                bridge_state=torch_state,
                reverse_steps=int(contract["matrix"]["reverse_steps_per_cell"]),
            )
            if not trajectory["first_reverse_lattice_valid"]:
                first_step_invalid += 1
            structure = _output_structure(output_state)
            final_volume = float(structure.volume)
            if not math.isfinite(final_volume) or final_volume <= 0.0:
                raise ValueError("non-finite or non-positive final volume")
            successful += 1
            attempt_rows.append(
                {
                    **base,
                    "status": "succeeded",
                    "reason": "",
                    "condition_sha256": clean_condition.sha256,
                    "numpy_reconstruction": numpy_errors,
                    "torch_reconstruction": torch_errors,
                    "trajectory": trajectory,
                    "final_structure": structure.as_dict(),
                    "final_volume": final_volume,
                    "atom_count": len(structure),
                    "walltime_s": time.time() - row_started,
                }
            )
        except Exception as exc:
            reason = f"{type(exc).__name__}:{exc}"
            if "non-finite" in reason.lower():
                nonfinite += 1
            attempt_rows.append(
                {
                    **base,
                    "status": "failed",
                    "reason": reason,
                    "traceback": traceback.format_exc(),
                    "walltime_s": time.time() - row_started,
                }
            )

    attempts_path = output / "attempts.jsonl"
    attempts_sha256 = _write_jsonl_exclusive(attempts_path, attempt_rows)
    gates = contract["gates"]
    terminal = len(attempt_rows)
    observed = {
        "terminal_cells": terminal,
        "successful_positive_volume_outputs": successful,
        "failed_cells": terminal - successful,
        "first_reverse_step_invalid_lattice_count": first_step_invalid,
        "nonfinite_trajectory_count": nonfinite,
        "max_numpy_coordinate_reconstruction_error": (
            max_numpy_coordinate_reconstruction_error
        ),
        "max_numpy_lattice_reconstruction_error": (
            max_numpy_lattice_reconstruction_error
        ),
        "max_torch_coordinate_reconstruction_error": (
            max_torch_coordinate_reconstruction_error
        ),
        "max_torch_lattice_reconstruction_error": (
            max_torch_lattice_reconstruction_error
        ),
        "retry_or_replacement_used": False,
    }
    gate_checks = {
        "schedule_parity": schedule["ok"],
        "terminal_cells": terminal == int(gates["terminal_cells"]),
        "raw_success": successful
        == int(gates["successful_positive_volume_outputs"]),
        "first_reverse_lattice": first_step_invalid
        == int(gates["first_reverse_step_invalid_lattice_count"]),
        "finite_trajectories": nonfinite == int(gates["nonfinite_trajectory_count"]),
        "numpy_coordinate_reconstruction": (
            max_numpy_coordinate_reconstruction_error
            <= float(gates["known_noise_coordinate_reconstruction_max_abs_error"])
        ),
        "numpy_lattice_reconstruction": (
            max_numpy_lattice_reconstruction_error
            <= float(gates["known_noise_lattice_reconstruction_max_abs_error"])
        ),
        "torch_coordinate_reconstruction": (
            max_torch_coordinate_reconstruction_error
            <= float(gates["known_noise_coordinate_reconstruction_max_abs_error"])
        ),
        "torch_lattice_reconstruction": (
            max_torch_lattice_reconstruction_error
            <= float(gates["known_noise_lattice_reconstruction_max_abs_error"])
        ),
        "no_retry_or_replacement": True,
    }
    ok = all(gate_checks.values())
    report = {
        "schema": "wq_schedule_correct_bridge_parity_terminal_v1",
        "ok": ok,
        "acceptance": "PASS" if ok else "FAIL",
        "run_id": contract["run_id"],
        "contract_sha256": contract_sha256,
        "authorization_record_sha256": contract["authorization"]["record_sha256"],
        "execution_patch_sha256": args.execution_patch_sha256,
        "source_generation_sha256": source["generation_jsonl_sha256"],
        "selection_manifest_sha256": sha256_file(selection_path),
        "attempts_jsonl_sha256": attempts_sha256,
        "strict_load": strict_load,
        "schedule_audit": schedule,
        "observed": observed,
        "gate_checks": gate_checks,
        "training_performed": False,
        "new_generation_performed": False,
        "mlip_used": False,
        "external_api_used": False,
        "walltime_s": time.time() - started,
        "gpu": {
            "name": device_name,
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated(args.device)),
        },
    }
    write_json_exclusive(output / "terminal_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT
        / "configs/experiments/wyckoff_codiffusion"
        / "wq_schedule_correct_bridge_parity_v1.json",
    )
    parser.add_argument("--source-generation", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execution-patch-sha256", required=True)
    parser.add_argument("--device", type=torch.device, default=torch.device("cuda"))
    args = parser.parse_args()
    args.contract = args.contract.resolve()
    args.source_generation = args.source_generation.resolve()
    args.checkpoint = args.checkpoint.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    try:
        report = _run(args, output)
    except Exception as exc:
        terminal = output / "terminal_report.json"
        if not terminal.exists():
            write_json_exclusive(
                terminal,
                {
                    "schema": "wq_schedule_correct_bridge_parity_terminal_v1",
                    "ok": False,
                    "acceptance": "FAIL",
                    "reason": f"{type(exc).__name__}:{exc}",
                    "traceback": traceback.format_exc(),
                    "training_performed": False,
                    "new_generation_performed": False,
                    "retry_or_replacement_used": False,
                    "mlip_used": False,
                    "external_api_used": False,
                },
            )
        raise
    if not report["ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
