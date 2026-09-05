#!/usr/bin/env python3
"""Offline paired EFSM evaluation for fixed before/after PMTR bodies.

The script performs no Direct evaluation, Materials Project query, filtering,
or outcome selection.  CHGNet is loaded lazily only when no evaluator is
injected by a caller or test.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from crystal_dlm.dynamic_crystal import parse_dynamic_answer  # noqa: E402
from crystal_dlm.manifold_corruption import CrystalGeometry  # noqa: E402
from crystal_dlm.offline_pmtr_certification import (  # noqa: E402
    EFSMBatchEvaluator,
    EFSMObservation,
)


METRICS = (
    "energy_eV_per_atom",
    "force_rms_eV_per_A",
    "force_max_eV_per_A",
    "stress_frobenius_GPa",
    "abs_hydrostatic_stress_GPa",
)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number} is not a JSON object")
            yield value


def _body(row: Mapping[str, Any], stage: str) -> str:
    for key in (f"{stage}_body", f"body_{stage}", f"{stage}_dynamic_body"):
        value = row.get(key)
        if value is not None:
            return str(value)
    nested = row.get(stage)
    if isinstance(nested, Mapping) and nested.get("body") is not None:
        return str(nested["body"])
    raise KeyError(f"row lacks {stage}_body")


def _geometry(body: str) -> CrystalGeometry:
    parsed = parse_dynamic_answer(str(body), strict=True)
    tokens = tuple(str(token) for token in parsed["tokens"])
    if "".join(tokens) != str(body):
        raise ValueError("body must be canonical separator-free exact 7+4N")
    return CrystalGeometry.from_mapping(parsed)


def _endpoint_unknown(*, body_valid: bool, error: str) -> dict[str, Any]:
    return {
        "body_valid": bool(body_valid),
        "efsm_known": False,
        **{metric: None for metric in METRICS},
        "error": str(error)[:300],
    }


def _endpoint_known(observation: EFSMObservation) -> dict[str, Any]:
    forces = np.asarray(observation.forces_eV_per_A, dtype=float)
    force_norms = np.linalg.norm(forces, axis=1)
    stress = np.asarray(observation.stress, dtype=float)
    return {
        "body_valid": True,
        "efsm_known": True,
        "energy_eV_per_atom": float(observation.energy_eV_per_atom),
        "force_rms_eV_per_A": float(
            np.sqrt(np.mean(force_norms * force_norms))
        ),
        "force_max_eV_per_A": float(np.max(force_norms)),
        "stress_frobenius_GPa": float(np.linalg.norm(stress)),
        "abs_hydrostatic_stress_GPa": float(abs(np.trace(stress) / 3.0)),
        "error": None,
    }


def _identity(row: Mapping[str, Any], input_row_index: int) -> dict[str, Any]:
    output: dict[str, Any] = {"input_row_index": int(input_row_index)}
    for key in ("pair_id", "source_row_idx", "sample_idx", "request_key"):
        if row.get(key) is not None:
            output[key] = row[key]
    if "pair_id" not in output:
        value = row.get("sample_idx", row.get("source_row_idx", input_row_index))
        output["pair_id"] = str(value)
    return output


def _pair_record(
    row: Mapping[str, Any],
    *,
    input_row_index: int,
) -> tuple[dict[str, Any], list[tuple[str, CrystalGeometry]]]:
    record: dict[str, Any] = {
        "schema": "pmtr_transfer_physics_pair_v1",
        **_identity(row, input_row_index),
        "before_body": None,
        "after_body": None,
        "before": None,
        "after": None,
        "delta_after_minus_before": {metric: None for metric in METRICS},
    }
    pending: list[tuple[str, CrystalGeometry]] = []
    for stage in ("before", "after"):
        try:
            body = _body(row, stage)
            record[f"{stage}_body"] = body
            geometry = _geometry(body)
            record[stage] = _endpoint_unknown(
                body_valid=True, error="efsm_not_evaluated"
            )
            pending.append((stage, geometry))
        except Exception as exc:  # noqa: BLE001 - invalid endpoints remain reported.
            record[stage] = _endpoint_unknown(
                body_valid=False,
                error=f"body:{type(exc).__name__}:{exc}",
            )
    return record, pending


def evaluate_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    input_indices: Sequence[int],
    evaluator: EFSMBatchEvaluator,
    batch_size: int = 16,
) -> list[dict[str, Any]]:
    """Evaluate all parseable endpoints while retaining every input pair."""

    if int(batch_size) != 16:
        raise ValueError("PMTR transfer EFSM batch_size is fixed at 16")
    if len(rows) != len(input_indices):
        raise ValueError("one input index is required per row")
    records: list[dict[str, Any]] = []
    geometries: list[CrystalGeometry] = []
    destinations: list[tuple[int, str]] = []
    for row, input_index in zip(rows, input_indices, strict=True):
        record, pending = _pair_record(row, input_row_index=int(input_index))
        record_index = len(records)
        records.append(record)
        for stage, geometry in pending:
            destinations.append((record_index, stage))
            geometries.append(geometry)

    predictions = list(
        evaluator.evaluate(geometries, batch_size=int(batch_size))
    )
    if len(predictions) != len(geometries):
        raise RuntimeError("EFSM evaluator changed batch cardinality")
    for geometry, raw, (record_index, stage) in zip(
        geometries, predictions, destinations, strict=True
    ):
        try:
            observation = EFSMObservation.from_mapping(
                raw, num_sites=len(geometry.species)
            )
            records[record_index][stage] = _endpoint_known(observation)
        except Exception as exc:  # noqa: BLE001 - unknown EFSM remains in denominator.
            records[record_index][stage] = _endpoint_unknown(
                body_valid=True,
                error=f"efsm:{type(exc).__name__}:{exc}",
            )

    for record in records:
        before = record["before"]
        after = record["after"]
        if before["efsm_known"] and after["efsm_known"]:
            record["delta_after_minus_before"] = {
                metric: float(after[metric] - before[metric]) for metric in METRICS
            }
    return records


def _bootstrap_mean_ci(
    values: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> list[float] | None:
    if int(samples) <= 0 or values.size == 0:
        return None
    rng = np.random.default_rng(int(seed))
    draws = rng.integers(0, values.size, size=(int(samples), values.size))
    means = values[draws].mean(axis=1)
    lower, upper = np.quantile(means, (0.025, 0.975))
    return [float(lower), float(upper)]


def summarize_pairs(
    records: Sequence[Mapping[str, Any]],
    *,
    bootstrap_samples: int = 0,
    bootstrap_seed: int = 20260905,
    tie_tolerance: float = 1.0e-12,
) -> dict[str, Any]:
    if int(bootstrap_samples) < 0:
        raise ValueError("bootstrap_samples must be non-negative")
    if float(tie_tolerance) < 0.0:
        raise ValueError("tie_tolerance must be non-negative")
    metrics: dict[str, Any] = {}
    for metric_index, metric in enumerate(METRICS):
        deltas = np.asarray(
            [
                float(row["delta_after_minus_before"][metric])
                for row in records
                if row["delta_after_minus_before"][metric] is not None
            ],
            dtype=float,
        )
        wins = int(np.sum(deltas < -float(tie_tolerance)))
        losses = int(np.sum(deltas > float(tie_tolerance)))
        ties = int(deltas.size - wins - losses)
        summary: dict[str, Any] = {
            "direction": "lower_is_better",
            "known_pairs": int(deltas.size),
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "mean_delta_after_minus_before": (
                None if deltas.size == 0 else float(np.mean(deltas))
            ),
            "median_delta_after_minus_before": (
                None if deltas.size == 0 else float(np.median(deltas))
            ),
        }
        bootstrap = _bootstrap_mean_ci(
            deltas,
            samples=int(bootstrap_samples),
            seed=int(bootstrap_seed) + metric_index,
        )
        if bootstrap is not None:
            summary["bootstrap_mean_delta_95_ci"] = bootstrap
        metrics[metric] = summary
    return {
        "pairs": len(records),
        "body_valid_pairs": sum(
            bool(row["before"]["body_valid"] and row["after"]["body_valid"])
            for row in records
        ),
        "efsm_known_pairs": sum(
            bool(row["before"]["efsm_known"] and row["after"]["efsm_known"])
            for row in records
        ),
        "before_invalid": sum(not bool(row["before"]["body_valid"]) for row in records),
        "after_invalid": sum(not bool(row["after"]["body_valid"]) for row in records),
        "before_efsm_unknown": sum(
            bool(row["before"]["body_valid"] and not row["before"]["efsm_known"])
            for row in records
        ),
        "after_efsm_unknown": sum(
            bool(row["after"]["body_valid"] and not row["after"]["efsm_known"])
            for row in records
        ),
        "bootstrap_samples": int(bootstrap_samples),
        "metrics": metrics,
    }


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _new_output_dir(path: Path) -> Path:
    output = path.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=False)
    return output


def evaluate_file(
    *,
    input_path: Path,
    output_dir: Path,
    evaluator: EFSMBatchEvaluator,
    shard_rank: int,
    shard_count: int,
    batch_size: int = 16,
    bootstrap_samples: int = 0,
    bootstrap_seed: int = 20260905,
) -> dict[str, Any]:
    if int(shard_count) <= 0 or not 0 <= int(shard_rank) < int(shard_count):
        raise ValueError("shard_rank must lie in 0..shard_count-1")
    all_rows = list(iter_jsonl(input_path))
    indices = [
        index
        for index in range(len(all_rows))
        if index % int(shard_count) == int(shard_rank)
    ]
    selected = [all_rows[index] for index in indices]
    records = evaluate_rows(
        selected,
        input_indices=indices,
        evaluator=evaluator,
        batch_size=int(batch_size),
    )
    output = _new_output_dir(output_dir)
    _write_jsonl(output / "pairs.jsonl", records)
    summary = {
        "schema": "pmtr_transfer_physics_summary_v1",
        "mode": "shard",
        "input": str(input_path),
        "shard": {"rank": int(shard_rank), "count": int(shard_count)},
        "batch_size": int(batch_size),
        "selection": "none",
        "direct": False,
        "materials_project_query": False,
        **summarize_pairs(
            records,
            bootstrap_samples=int(bootstrap_samples),
            bootstrap_seed=int(bootstrap_seed),
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def merge_shards(
    *,
    shard_dirs: Sequence[Path],
    output_dir: Path,
    bootstrap_samples: int = 0,
    bootstrap_seed: int = 20260905,
) -> dict[str, Any]:
    if not shard_dirs:
        raise ValueError("at least one shard directory is required")
    records: list[dict[str, Any]] = []
    counts: set[int] = set()
    ranks: set[int] = set()
    inputs: set[str] = set()
    for directory in shard_dirs:
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        if summary.get("mode") != "shard":
            raise ValueError(f"{directory} is not a shard output")
        counts.add(int(summary["shard"]["count"]))
        rank = int(summary["shard"]["rank"])
        if rank in ranks:
            raise ValueError(f"duplicate shard rank {rank}")
        ranks.add(rank)
        inputs.add(str(summary["input"]))
        records.extend(iter_jsonl(directory / "pairs.jsonl"))
    if len(counts) != 1 or len(inputs) != 1:
        raise ValueError("shards disagree on count or input")
    shard_count = counts.pop()
    if ranks != set(range(shard_count)):
        raise ValueError("merge requires every shard rank exactly once")
    indices = [int(row["input_row_index"]) for row in records]
    if len(indices) != len(set(indices)):
        raise ValueError("shards contain duplicate input rows")
    records.sort(key=lambda row: int(row["input_row_index"]))

    output = _new_output_dir(output_dir)
    _write_jsonl(output / "pairs.jsonl", records)
    summary = {
        "schema": "pmtr_transfer_physics_summary_v1",
        "mode": "merged",
        "input": inputs.pop(),
        "merged_shards": shard_count,
        "selection": "none",
        "direct": False,
        "materials_project_query": False,
        **summarize_pairs(
            records,
            bootstrap_samples=int(bootstrap_samples),
            bootstrap_seed=int(bootstrap_seed),
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def load_default_evaluator(device: str) -> EFSMBatchEvaluator:
    """Reuse the offline certifier's lazy CHGNet adapter."""

    from scripts.certify_pmtr_corruptions import load_chgnet_evaluator

    return load_chgnet_evaluator(str(device))


def main(
    argv: Sequence[str] | None = None,
    *,
    evaluator: EFSMBatchEvaluator | None = None,
) -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--input-jsonl", type=Path)
    mode.add_argument("--merge-shards", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--shard-rank", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--bootstrap-samples", type=int, default=0)
    parser.add_argument("--bootstrap-seed", type=int, default=20260905)
    args = parser.parse_args(argv)

    if args.merge_shards is not None:
        summary = merge_shards(
            shard_dirs=[path.resolve(strict=True) for path in args.merge_shards],
            output_dir=args.output_dir,
            bootstrap_samples=int(args.bootstrap_samples),
            bootstrap_seed=int(args.bootstrap_seed),
        )
    else:
        active_evaluator = (
            evaluator
            if evaluator is not None
            else load_default_evaluator(str(args.device))
        )
        summary = evaluate_file(
            input_path=args.input_jsonl.resolve(strict=True),
            output_dir=args.output_dir,
            evaluator=active_evaluator,
            shard_rank=int(args.shard_rank),
            shard_count=int(args.shard_count),
            batch_size=int(args.batch_size),
            bootstrap_samples=int(args.bootstrap_samples),
            bootstrap_seed=int(args.bootstrap_seed),
        )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
