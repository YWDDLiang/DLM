#!/usr/bin/env python3
"""Build train-only VPA_Q8 and CN_ENV8 sidecar labels for SI-LWA-v1."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
from typing import Any, Iterable, Mapping, Sequence
import warnings


SCHEMA = "h1a2_si_lwa_intent_row_v1"
MANIFEST_SCHEMA = "h1a2_si_lwa_intent_manifest_v1"
CN_EDGES = (0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, float("inf"))
INTENT_CLASSES = 8
FIT_SEED = 82000


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _entropy(counts: Sequence[int]) -> float:
    total = sum(int(value) for value in counts)
    if total <= 0:
        return 0.0
    result = 0.0
    for value in counts:
        if value:
            probability = float(value) / float(total)
            result -= probability * math.log(probability)
    return result


def cn_histogram(values: Sequence[float]) -> list[float]:
    if not values:
        raise ValueError("coordination list is empty")
    counts = [0] * INTENT_CLASSES
    for raw in values:
        value = float(raw)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("effective coordination must be finite and non-negative")
        for index in range(INTENT_CLASSES):
            if CN_EDGES[index] <= value < CN_EDGES[index + 1]:
                counts[index] += 1
                break
        else:  # pragma: no cover - +inf terminal edge is exhaustive.
            raise AssertionError(value)
    total = float(sum(counts))
    return [float(value) / total for value in counts]


def quantile_edges(values: Sequence[float], *, classes: int = INTENT_CLASSES) -> list[float]:
    import numpy as np

    array = np.asarray([float(value) for value in values], dtype=np.float64)
    if array.ndim != 1 or len(array) < int(classes):
        raise ValueError("too few VPA values for frozen quantiles")
    if not np.isfinite(array).all():
        raise ValueError("VPA values must be finite")
    edges = np.quantile(array, np.arange(classes + 1) / float(classes)).tolist()
    if any(float(left) > float(right) for left, right in zip(edges, edges[1:])):
        raise RuntimeError("VPA quantiles are not monotone")
    if len(set(float(value) for value in edges)) != len(edges):
        raise RuntimeError("VPA quantile edges are not distinct")
    return [float(value) for value in edges]


def assign_quantile(value: float, edges: Sequence[float]) -> int:
    import numpy as np

    if len(edges) != INTENT_CLASSES + 1:
        raise ValueError("VPA_Q8 requires nine frozen edges")
    result = int(np.searchsorted(np.asarray(edges[1:-1]), float(value), side="right"))
    if result not in range(INTENT_CLASSES):
        raise RuntimeError("VPA class escaped 0..7")
    return result


def fit_representative_medoids(
    rows: Sequence[Sequence[float]],
    *,
    classes: int = INTENT_CLASSES,
    seed: int = FIT_SEED,
    maximum_iterations: int = 100,
) -> tuple[list[int], list[list[float]]]:
    """Fit deterministic k-means, then select nearest observed representatives."""

    import numpy as np

    values = np.asarray(rows, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < int(classes):
        raise ValueError("too few CN rows for eight representatives")
    if values.shape[1] != INTENT_CLASSES or not np.isfinite(values).all():
        raise ValueError("CN histograms must be finite eight-vectors")
    rng = np.random.default_rng(int(seed))
    first = int(rng.integers(values.shape[0]))
    center_indices = [first]
    minimum_distance = ((values - values[first]) ** 2).sum(axis=1)
    while len(center_indices) < int(classes):
        total = float(minimum_distance.sum())
        if total <= 0.0:
            candidate = next(
                index for index in range(values.shape[0]) if index not in center_indices
            )
        else:
            candidate = int(rng.choice(values.shape[0], p=minimum_distance / total))
            if candidate in center_indices:
                candidate = next(
                    index
                    for index in np.argsort(-minimum_distance).tolist()
                    if int(index) not in center_indices
                )
        center_indices.append(candidate)
        distance = ((values - values[candidate]) ** 2).sum(axis=1)
        minimum_distance = np.minimum(minimum_distance, distance)
    centers = values[np.asarray(center_indices)].copy()
    assignments = np.full(values.shape[0], -1, dtype=np.int64)
    for _ in range(int(maximum_iterations)):
        distances = ((values[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        updated_assignments = distances.argmin(axis=1)
        if np.array_equal(updated_assignments, assignments):
            break
        assignments = updated_assignments
        for cluster in range(int(classes)):
            members = values[assignments == cluster]
            if len(members) == 0:
                farthest = int(distances.min(axis=1).argmax())
                centers[cluster] = values[farthest]
            else:
                centers[cluster] = members.mean(axis=0)
    representative_indices: list[int] = []
    representatives: list[list[float]] = []
    for cluster in range(int(classes)):
        distances = ((values - centers[cluster]) ** 2).sum(axis=1)
        index = next(
            int(candidate)
            for candidate in np.argsort(distances).tolist()
            if int(candidate) not in representative_indices
        )
        representative_indices.append(index)
        representatives.append([float(value) for value in values[index].tolist()])
    if len(set(representative_indices)) != int(classes):
        raise RuntimeError("CN representative medoids are not distinct")
    return representative_indices, representatives


def assign_medoid(row: Sequence[float], representatives: Sequence[Sequence[float]]) -> int:
    if len(representatives) != INTENT_CLASSES:
        raise ValueError("CN_ENV8 requires eight representatives")
    distances = [
        sum((float(left) - float(right)) ** 2 for left, right in zip(row, medoid))
        for medoid in representatives
    ]
    if any(not math.isfinite(value) for value in distances):
        raise ValueError("CN distance is not finite")
    return min(range(len(distances)), key=lambda index: (distances[index], index))


def _process_record(task: tuple[str, int, Mapping[str, Any]]) -> dict[str, Any]:
    split, source_row, raw = task
    material_id = str(raw.get("material_id") or "")
    formula = str(raw.get("pretty_formula") or "")
    cif = str(raw.get("cif") or "")
    base = {
        "schema": SCHEMA,
        "split": str(split),
        "source_row": int(source_row),
        "material_id": material_id,
        "pretty_formula": formula,
        "cif_sha256": hashlib.sha256(cif.encode("utf-8")).hexdigest(),
    }
    try:
        from pymatgen.analysis.local_env import CrystalNN
        from pymatgen.core import Structure

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            structure = Structure.from_str(cif, fmt="cif")
            number_of_sites = len(structure)
            if number_of_sites not in range(1, 21):
                raise ValueError("site count outside dynamic 1..20")
            volume_per_atom = float(structure.volume) / float(number_of_sites)
            if not math.isfinite(volume_per_atom) or volume_per_atom <= 0.0:
                raise ValueError("volume per atom must be finite and positive")
            crystal_nn = CrystalNN()
            effective_cn = [
                sum(float(item.get("weight", 0.0)) for item in crystal_nn.get_nn_info(structure, index))
                for index in range(number_of_sites)
            ]
        return {
            **base,
            "label_status": "valid",
            "N": int(number_of_sites),
            "log_volume_per_atom": math.log(volume_per_atom),
            "cn_histogram": cn_histogram(effective_cn),
        }
    except Exception as exc:  # noqa: BLE001 - every failed row remains audited.
        return {
            **base,
            "label_status": "missing",
            "failure_type": type(exc).__name__,
            "failure_message": str(exc)[:500],
        }


def _read_csv(path: Path, split: str) -> list[tuple[str, int, Mapping[str, Any]]]:
    import pandas as pd

    frame = pd.read_csv(path)
    required = {"material_id", "pretty_formula", "cif"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    return [
        (str(split), int(index), row)
        for index, row in enumerate(frame.to_dict(orient="records"))
    ]


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> str:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(dict(row)) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return sha256_file(path)


def _audit_split(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row["label_status"] == "valid"]
    failures = Counter(
        str(row.get("failure_type") or "unknown")
        for row in rows
        if row["label_status"] != "valid"
    )
    vpa_counts = Counter(int(row["vpa_class"]) for row in valid)
    cn_counts = Counter(int(row["cn_class"]) for row in valid)
    return {
        "rows": len(rows),
        "valid": len(valid),
        "missing": len(rows) - len(valid),
        "coverage": float(len(valid)) / float(max(1, len(rows))),
        "failure_types": dict(sorted(failures.items())),
        "vpa_class_counts": [int(vpa_counts[index]) for index in range(INTENT_CLASSES)],
        "vpa_entropy_nats": _entropy([vpa_counts[index] for index in range(INTENT_CLASSES)]),
        "cn_class_counts": [int(cn_counts[index]) for index in range(INTENT_CLASSES)],
        "cn_entropy_nats": _entropy([cn_counts[index] for index in range(INTENT_CLASSES)]),
    }


def build_dataset(
    train_csv: Path,
    validation_csv: Path,
    output_dir: Path,
    *,
    workers: int,
) -> dict[str, Any]:
    train_source = train_csv.resolve()
    validation_source = validation_csv.resolve()
    if train_source == validation_source:
        raise ValueError("train and validation sources must differ")
    if not 1 <= int(workers) <= 48:
        raise ValueError("workers must lie in 1..48")
    final = output_dir.resolve()
    if final.exists():
        raise FileExistsError(final)
    preparing = final.with_name(final.name + ".preparing")
    failed = final.with_name(final.name + ".FAILED")
    if preparing.exists() or failed.exists():
        raise FileExistsError(preparing if preparing.exists() else failed)
    preparing.mkdir(parents=True, exist_ok=False)
    try:
        tasks_by_split = {
            "train": _read_csv(train_source, "train"),
            "validation": _read_csv(validation_source, "validation"),
        }
        processed: dict[str, list[dict[str, Any]]] = {}
        for split, tasks in tasks_by_split.items():
            if int(workers) == 1:
                rows = [_process_record(task) for task in tasks]
            else:
                with ProcessPoolExecutor(max_workers=int(workers)) as executor:
                    rows = list(executor.map(_process_record, tasks, chunksize=16))
            processed[split] = rows
        valid_train = [row for row in processed["train"] if row["label_status"] == "valid"]
        edges = quantile_edges([float(row["log_volume_per_atom"]) for row in valid_train])
        representative_indices, representatives = fit_representative_medoids(
            [row["cn_histogram"] for row in valid_train], seed=FIT_SEED
        )
        representative_rows = [valid_train[index] for index in representative_indices]
        for rows in processed.values():
            for row in rows:
                if row["label_status"] != "valid":
                    row["vpa_class"] = None
                    row["cn_class"] = None
                    continue
                row["vpa_class"] = assign_quantile(
                    float(row["log_volume_per_atom"]), edges
                )
                row["cn_class"] = assign_medoid(row["cn_histogram"], representatives)
        output_hashes = {
            split: _write_jsonl(preparing / f"{split}.jsonl", rows)
            for split, rows in processed.items()
        }
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "source_files": [
                {
                    "split": "train",
                    "path": str(train_source),
                    "bytes": train_source.stat().st_size,
                    "sha256": sha256_file(train_source),
                },
                {
                    "split": "validation",
                    "path": str(validation_source),
                    "bytes": validation_source.stat().st_size,
                    "sha256": sha256_file(validation_source),
                },
            ],
            "test_source_supported": False,
            "generated_test_outcomes_read": False,
            "VPA_Q8": {
                "definition": "log(volume/number_of_sites)",
                "train_quantile_edges": edges,
            },
            "CN_ENV8": {
                "definition": "normalized CrystalNN effective-CN histogram",
                "histogram_edges": [None if math.isinf(value) else value for value in CN_EDGES],
                "fit_seed": FIT_SEED,
                "representatives": [
                    {
                        "class": index,
                        "histogram": representatives[index],
                        "material_id": representative_rows[index]["material_id"],
                        "source_row": representative_rows[index]["source_row"],
                    }
                    for index in range(INTENT_CLASSES)
                ],
            },
            "splits": {
                split: _audit_split(rows) for split, rows in processed.items()
            },
            "workers": int(workers),
            "code_sha256": sha256_file(Path(__file__).resolve()),
            "output_hashes": output_hashes,
            "gpu_jobs_used": 0,
        }
        manifest_path = preparing / "SI_LWA_INTENT_MANIFEST.json"
        with manifest_path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n")
        manifest_sha256 = sha256_file(manifest_path)
        (preparing / "_SUCCESS").write_text(manifest_sha256 + "\n", encoding="utf-8")
        preparing.rename(final)
        return {**manifest, "manifest_sha256": manifest_sha256}
    except Exception:
        if preparing.exists():
            shutil.move(str(preparing), str(failed))
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--validation-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = build_dataset(
        args.train_csv,
        args.validation_csv,
        args.output_dir,
        workers=args.workers,
    )
    print(json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
