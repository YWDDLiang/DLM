"""Evaluator-specific MP20 reference single points and closure relaxations."""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import ArtifactLedger, write_json_exclusive
from .mlip import EvaluatorLock, MLIPCalculator


@dataclasses.dataclass(frozen=True, slots=True)
class ReferenceEvaluationConfig:
    csv_splits: tuple[str, ...]
    output_jsonl: str
    evaluator: str
    asset_lock: str
    model_root: str
    stage: str
    device: str = "cuda"
    queue_path: str | None = None
    limit: int | None = None

    def __post_init__(self) -> None:
        if not self.csv_splits:
            raise ValueError("at least one SPLIT=CSV reference source is required")
        if self.evaluator not in {"chgnet", "mattersim", "mace"}:
            raise ValueError("unknown evaluator")
        if self.stage not in {"raw", "relaxed"}:
            raise ValueError("reference stage must be raw or relaxed")
        if self.stage == "relaxed" and not self.queue_path:
            raise ValueError("relaxed reference evaluation requires a hull pending queue")
        if self.limit is not None and self.limit <= 0:
            raise ValueError("limit must be positive")


def _sources(specifications: Sequence[str]) -> list[tuple[str, Path]]:
    result: list[tuple[str, Path]] = []
    for specification in specifications:
        split, separator, raw_path = specification.partition("=")
        if not separator or not split or not raw_path:
            raise ValueError("reference CSVs must use SPLIT=PATH")
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        result.append((split, path))
    return result


def _queue(
    path: str | None,
    *,
    evaluator: str,
    contract_hash: str,
) -> set[str] | None:
    if path is None:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != "wqcodiff_evaluator_hull_v1":
        raise ValueError("relaxation queue must be a hull-closure artifact")
    if str(payload.get("evaluator")) != evaluator:
        raise ValueError(
            "relaxation queue evaluator mismatch: "
            f"expected {evaluator}, found {payload.get('evaluator')}"
        )
    if str(payload.get("contract_hash")) != contract_hash:
        raise ValueError("relaxation queue contract hash mismatch")
    if payload.get("closed"):
        return set()
    return {str(value) for value in payload["pending_relaxation_ids"]}


def _structure_hash(structure: Any) -> str:
    return hashlib.sha256(structure.to(fmt="cif").encode("utf-8")).hexdigest()


def evaluate_references(config: ReferenceEvaluationConfig) -> dict[str, Any]:
    if config.device.startswith("cuda"):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("reference MLIP evaluation requires CUDA inside Slurm")
    lock = EvaluatorLock.load(config.asset_lock)
    calculator = MLIPCalculator(
        evaluator=config.evaluator,
        asset_lock=lock,
        model_root=config.model_root,
        device=config.device,
    )
    pending = _queue(
        config.queue_path,
        evaluator=config.evaluator,
        contract_hash=calculator.contract_hash,
    )
    output = ArtifactLedger(
        config.output_jsonl,
        key_fields=("reference_id", "stage", "evaluator", "contract_hash"),
    )
    existing = {
        (
            str(record["reference_id"]),
            str(record["stage"]),
            str(record["evaluator"]),
            str(record["contract_hash"]),
        )
        for record in output.records()
    }
    selected: list[tuple[str, str, Mapping[str, str]]] = []
    ids: set[str] = set()
    for split, path in _sources(config.csv_splits):
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                material_id = str(row.get("material_id") or row.get("id") or "").strip()
                if not material_id:
                    raise ValueError(f"{path} contains a row without material_id")
                if material_id in ids:
                    raise ValueError(f"duplicate MP20 reference ID across splits: {material_id}")
                ids.add(material_id)
                if pending is not None and material_id not in pending:
                    continue
                selected.append((material_id, split, row))
    if pending is not None:
        missing = sorted(pending - {value[0] for value in selected})
        if missing:
            raise ValueError(f"hull queue contains {len(missing)} unknown IDs; first={missing[0]}")
    if config.limit is not None:
        selected = selected[: config.limit]
    if not selected and pending != set():
        raise ValueError("reference evaluation selected no structures")

    from pymatgen.core import Structure

    succeeded = 0
    failed = 0
    started_all = time.monotonic()
    for reference_id, split, row in selected:
        key = (reference_id, config.stage, config.evaluator, calculator.contract_hash)
        if key in existing:
            raise ValueError(f"reference evaluation would retry immutable record {key}")
        started = time.monotonic()
        cif = str(row.get("cif") or row.get("cif.conv") or "")
        try:
            if not cif:
                raise ValueError("missing CIF column (expected cif or cif.conv)")
            structure = Structure.from_str(cif, fmt="cif")
            source_hash = _structure_hash(structure)
            result = (
                calculator.relax(structure)
                if config.stage == "relaxed"
                else calculator.single_point(structure)
            )
            evaluated_structure = (
                Structure.from_dict(result["structure"])
                if "structure" in result
                else structure
            )
            output.append(
                {
                    "schema": "wqcodiff_reference_energy_v1",
                    "reference_id": reference_id,
                    "split": split,
                    "evaluator": config.evaluator,
                    "contract_hash": calculator.contract_hash,
                    "stage": config.stage,
                    "status": "succeeded",
                    "source_structure_hash": source_hash,
                    "structure_hash": _structure_hash(evaluated_structure),
                    "structure": evaluated_structure.as_dict(),
                    "composition": evaluated_structure.composition.as_dict(),
                    **result,
                    "walltime_s": time.monotonic() - started,
                }
            )
            succeeded += 1
        except Exception as exc:
            output.append(
                {
                    "schema": "wqcodiff_reference_energy_v1",
                    "reference_id": reference_id,
                    "split": split,
                    "evaluator": config.evaluator,
                    "contract_hash": calculator.contract_hash,
                    "stage": config.stage,
                    "status": "failed",
                    "reason": f"{type(exc).__name__}:{exc}",
                    "structure_hash": hashlib.sha256(cif.encode("utf-8")).hexdigest(),
                    "walltime_s": time.monotonic() - started,
                }
            )
            failed += 1
    summary = {
        "schema": "wqcodiff_reference_evaluation_summary_v1",
        "evaluator": config.evaluator,
        "contract_hash": calculator.contract_hash,
        "stage": config.stage,
        "selected": len(selected),
        "succeeded": succeeded,
        "failed": failed,
        "complete": failed == 0,
        "queue_path": config.queue_path,
        "output_jsonl": str(Path(config.output_jsonl).resolve()),
        "elapsed_s": time.monotonic() - started_all,
    }
    write_json_exclusive(
        Path(config.output_jsonl).with_suffix(f".{config.stage}.summary.json"),
        summary,
    )
    return summary
