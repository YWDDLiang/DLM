"""Frozen topology-preserving common refiner used by every compared method."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .charts import PyXtalChartCatalog
from .contracts import (
    ArtifactLedger,
    AttemptLedger,
    AttemptRecord,
    AttemptStatus,
    SeedDeriver,
    write_json_exclusive,
)
from .model import WQCoDenoiser, WQModelConfig, WQVariant
from .protocol import load_protocol
from .runtime import compute_geometry_evidence, expand_state, tensorize_state
from .sampling import _autocast, _continuous_step
from .state import StratifiedState


COMMON_REFINER_CALLS = 16
COMMON_REFINER_START_TIME = 0.1


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def freeze_common_refiner(
    *,
    checkpoint: str | Path,
    output: str | Path,
    protocol_path: str | Path,
    frozen_day: int = 14,
) -> dict[str, Any]:
    """Create the immutable Day-14 lock; it cannot be selected on test MLIP results."""

    protocol = load_protocol(protocol_path)
    checkpoint_path = Path(checkpoint).resolve()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "wqcodiff_ema_model_v1":
        raise ValueError("common refiner requires a final EMA checkpoint")
    if payload.get("protocol_sha256") != protocol.sha256:
        raise ValueError("common-refiner checkpoint/protocol mismatch")
    if not bool(payload.get("paper_eligible")):
        raise ValueError("non-paper checkpoint cannot be frozen as the common refiner")
    training = payload.get("training_config", {})
    variant = WQVariant(str(training.get("variant")))
    lock = {
        "schema": "wqcodiff_common_refiner_lock_v1",
        "protocol_name": protocol.name,
        "protocol_sha256": protocol.sha256,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_variant": variant.value,
        "checkpoint_training_seed": int(training["training_seed"]),
        "inference_variant": WQVariant.STRAT_GEO.value,
        "topology_updates": False,
        "calls": COMMON_REFINER_CALLS,
        "start_time": COMMON_REFINER_START_TIME,
        "frozen_day": int(frozen_day),
        "selection_data": "validation_only_no_test_no_mlip",
    }
    if lock["frozen_day"] > 14:
        raise ValueError("the common refiner must be frozen no later than Day 14")
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(lock, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return {**lock, "lock_sha256": _sha256(destination)}


@dataclasses.dataclass(frozen=True, slots=True)
class CommonRefinerLock:
    path: Path
    sha256: str
    payload: Mapping[str, Any]

    @classmethod
    def load(cls, path: str | Path, *, protocol_path: str | Path) -> "CommonRefinerLock":
        location = Path(path).resolve()
        payload = json.loads(location.read_text(encoding="utf-8"))
        protocol = load_protocol(protocol_path)
        if payload.get("schema") != "wqcodiff_common_refiner_lock_v1":
            raise ValueError("invalid common-refiner lock schema")
        if payload.get("protocol_sha256") != protocol.sha256:
            raise ValueError("common-refiner lock/protocol mismatch")
        if payload.get("inference_variant") != WQVariant.STRAT_GEO.value:
            raise ValueError("common refiner must use the permutation-equivariant inference path")
        if payload.get("topology_updates") is not False:
            raise ValueError("common refiner is not topology preserving")
        if int(payload.get("calls", -1)) != COMMON_REFINER_CALLS:
            raise ValueError("common-refiner call count changed")
        if float(payload.get("start_time", -1.0)) != COMMON_REFINER_START_TIME:
            raise ValueError("common-refiner start time changed")
        checkpoint = Path(str(payload["checkpoint"]))
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        if _sha256(checkpoint) != payload.get("checkpoint_sha256"):
            raise ValueError("common-refiner checkpoint hash mismatch")
        return cls(location, _sha256(location), payload)


@dataclasses.dataclass(frozen=True, slots=True)
class RefineConfig:
    input_jsonl: str
    output_jsonl: str
    attempt_ledger: str
    experiment_id: str
    refiner_lock: str
    device: str = "cuda"


def _load_generation(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("schema") != "wqcodiff_generation_attempt_v1":
                raise ValueError(f"generation line {line_number} has the wrong schema")
            attempt_id = str(row.get("attempt_id") or "")
            if not attempt_id or attempt_id in seen:
                raise ValueError(f"missing/duplicate generation attempt at line {line_number}")
            seen.add(attempt_id)
            records.append(row)
    if not records:
        raise ValueError("common-refiner input is empty")
    return records


def _load_model(lock: CommonRefinerLock, device: torch.device) -> WQCoDenoiser:
    checkpoint = Path(str(lock.payload["checkpoint"]))
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("schema") != "wqcodiff_ema_model_v1":
        raise ValueError("common-refiner checkpoint schema changed")
    if _sha256(checkpoint) != lock.payload["checkpoint_sha256"]:
        raise ValueError("common-refiner checkpoint mutated after lock")
    training = payload.get("training_config", {})
    if str(training.get("variant")) != lock.payload["checkpoint_variant"]:
        raise ValueError("common-refiner checkpoint variant mismatch")
    if int(training.get("training_seed", -1)) != int(lock.payload["checkpoint_training_seed"]):
        raise ValueError("common-refiner checkpoint seed mismatch")
    model = WQCoDenoiser(WQModelConfig(**payload["model_config"]))
    model.load_state_dict(payload["model"], strict=True)
    return model.to(device).eval()


def refine(config: RefineConfig, *, protocol_path: str | Path) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    lock = CommonRefinerLock.load(config.refiner_lock, protocol_path=protocol_path)
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("registered common refinement requires CUDA inside Slurm")
    model = _load_model(lock, device)
    catalog = PyXtalChartCatalog()
    generation = _load_generation(config.input_jsonl)
    output = ArtifactLedger(config.output_jsonl)
    if output.records():
        raise ValueError("common-refiner output is immutable and nonempty")
    attempts = AttemptLedger(config.attempt_ledger)
    ledger_stage = "common_refiner"
    existing = {record.key for record in attempts.records()}
    deriver = SeedDeriver(protocol.name, config.experiment_id)
    for row in generation:
        key = (str(row["attempt_id"]), ledger_stage)
        if key in existing:
            raise ValueError(f"common refiner would retry immutable stage {key}")
        seed = deriver.derive(
            training_seed=int(row["training_seed"]),
            sampling_seed=int(row["sampling_seed"]),
            attempt_id=str(row["attempt_id"]),
            stage=ledger_stage,
        )
        attempts.append(
            AttemptRecord(
                attempt_id=str(row["attempt_id"]),
                method=str(row["method"]),
                training_seed=int(row["training_seed"]),
                sampling_seed=int(row["sampling_seed"]),
                stage=ledger_stage,
                status=AttemptStatus.SUBMITTED,
                seed=seed,
            )
        )

    succeeded = 0
    failed = 0
    started_all = time.monotonic()
    for upstream in generation:
        started = time.monotonic()
        attempt_id = str(upstream["attempt_id"])
        seed = deriver.derive(
            training_seed=int(upstream["training_seed"]),
            sampling_seed=int(upstream["sampling_seed"]),
            attempt_id=attempt_id,
            stage=ledger_stage,
        )
        calls = {"common_refiner": 0, "projection": 0}
        try:
            if upstream.get("status") != AttemptStatus.SUCCEEDED.value:
                raise RuntimeError(
                    f"upstream_generation:{upstream.get('reason', upstream.get('status'))}"
                )
            state = StratifiedState.from_dict(dict(upstream["state"]))
            topology_before = state.topology_hash()
            score_norms: tuple[float, ...] = (0.0,) * len(state.orbits)
            uncertainties: tuple[float, ...] = (0.0,) * len(state.orbits)
            times = np.linspace(
                float(lock.payload["start_time"]),
                0.0,
                int(lock.payload["calls"]) + 1,
            )
            for index in range(int(lock.payload["calls"])):
                expanded = expand_state(state, catalog)
                calls["projection"] += 1
                evidence = compute_geometry_evidence(
                    state,
                    expanded,
                    score_norms=score_norms,
                    basin_uncertainties=uncertainties,
                )
                batch = tensorize_state(
                    state,
                    expanded,
                    evidence,
                    time=float(times[index]),
                ).to(device)
                with torch.no_grad(), _autocast(device):
                    prediction = model(batch, variant=WQVariant.STRAT_GEO)
                calls["common_refiner"] += 1
                state, score_norms = _continuous_step(
                    state,
                    expanded,
                    prediction,
                    current_time=float(times[index]),
                    next_time=float(times[index + 1]),
                )
            final = expand_state(state, catalog)
            calls["projection"] += 1
            if state.topology_hash() != topology_before:
                raise RuntimeError("common refiner changed discrete topology")
            structure = final.pymatgen_structure()
            elapsed = time.monotonic() - started
            refiner_flops = float(
                2 * model.parameter_count() * calls["common_refiner"]
            )
            artifact = {
                **upstream,
                "stage": "common_refiner",
                "state": state.to_dict(),
                "structure": structure.as_dict(),
                "structure_cif_sha256": hashlib.sha256(
                    structure.to(fmt="cif").encode("utf-8")
                ).hexdigest(),
                "redetected_space_group": final.redetected_space_group,
                "common_refiner_lock_sha256": lock.sha256,
                "common_refiner_checkpoint_sha256": lock.payload["checkpoint_sha256"],
                "common_refiner_calls": calls,
                "parent_artifact_sha256": _canonical_hash(upstream),
                "walltime_s": elapsed,
                "common_refiner_flops_lower_bound": refiner_flops,
                "common_refiner_flops_estimator": (
                    "2x_parameter_count_per_refiner_call_lower_bound_not_actual_flops"
                ),
                "status": AttemptStatus.SUCCEEDED.value,
            }
            digest = output.append(artifact)
            status = AttemptStatus.SUCCEEDED
            reason = ""
            succeeded += 1
        except Exception as exc:
            elapsed = time.monotonic() - started
            status = AttemptStatus.FAILED
            reason = f"{type(exc).__name__}:{exc}"
            refiner_flops = float(
                2 * model.parameter_count() * calls["common_refiner"]
            )
            digest = output.append(
                {
                    "schema": "wqcodiff_generation_attempt_v1",
                    "attempt_id": attempt_id,
                    "method": upstream["method"],
                    "training_seed": upstream["training_seed"],
                    "sampling_seed": upstream["sampling_seed"],
                    "ordinal": upstream.get("ordinal"),
                    "pair_id": upstream.get("pair_id"),
                    "paired_seed": upstream.get("paired_seed"),
                    "stage": "common_refiner",
                    "status": status.value,
                    "reason": reason,
                    "common_refiner_lock_sha256": lock.sha256,
                    "common_refiner_calls": calls,
                    "parent_artifact_sha256": _canonical_hash(upstream),
                    "walltime_s": elapsed,
                    "common_refiner_flops_lower_bound": refiner_flops,
                    "common_refiner_flops_estimator": (
                        "2x_parameter_count_per_refiner_call_lower_bound_not_actual_flops"
                    ),
                }
            )
            failed += 1
        attempts.append(
            AttemptRecord(
                attempt_id=attempt_id,
                method=str(upstream["method"]),
                training_seed=int(upstream["training_seed"]),
                sampling_seed=int(upstream["sampling_seed"]),
                stage=ledger_stage,
                status=status,
                reason=reason,
                artifact_hash=digest,
                seed=seed,
                calls=calls,
                flops=refiner_flops,
                walltime_s=elapsed,
                metadata={
                    "pair_id": upstream.get("pair_id"),
                    "refiner_lock_sha256": lock.sha256,
                    "topology_preserving": True,
                },
            )
        )
    result = {
        "ok": succeeded + failed == len(generation),
        "schema": "wqcodiff_common_refiner_summary_v1",
        "protocol_name": protocol.name,
        "protocol_sha256": protocol.sha256,
        "attempts": len(generation),
        "succeeded": succeeded,
        "failed": failed,
        "all_attempts_terminal": succeeded + failed == len(generation),
        "all_attempts_succeeded": failed == 0,
        "denominator": "all_upstream_attempts",
        "refiner_lock_sha256": lock.sha256,
        "elapsed_s": time.monotonic() - started_all,
        "output_jsonl": str(Path(config.output_jsonl).resolve()),
    }
    summary_path = Path(config.output_jsonl).with_suffix(".summary.json")
    write_json_exclusive(summary_path, result)
    return result
