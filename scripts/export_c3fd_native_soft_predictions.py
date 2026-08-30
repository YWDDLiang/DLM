#!/usr/bin/env python3
"""Export frozen per-row C3FD soft-head predictions on CPU.

The exporter deliberately writes a narrow, outcome-blind payload.  Every output
row contains only the semantic source-row identity (``N/elements/counts``) and
the predictions from *both* frozen C3FD checkpoints.  No source metadata or
structure/outcome fields are copied.

Torch and the C3FD model are imported lazily so the validation/assembly helpers
remain usable in lightweight unit tests.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import suppress
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


ROW_SCHEMA = "c3fd_native_soft_prediction_row_v1"
MANIFEST_SCHEMA = "c3fd_native_soft_prediction_manifest_v1"
CHECKPOINT_SCHEMA = "h1a2_c3fd_planner_checkpoint_v1"
SEMANTIC_ROW_SCHEMA = "h1a2_c3fd_semantic_row_v1"
SEMANTIC_MANIFEST_SCHEMA = "h1a2_c3fd_planner_data_manifest_v1"
SEMANTIC_VOCABULARY_SCHEMA = "h1a2_c3fd_semantic_vocabulary_v1"
REQUIRED_SPLITS = ("train", "val")
PREDICTION_FIELDS = (
    "lattice_system",
    "spacegroup_bucket",
    "volume_per_atom_bin",
)
EXPECTED_CHECKPOINT_SEEDS = frozenset((17, 18))
UNKNOWN_LABEL = "<UNKNOWN>"


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"non-object JSONL row in {path}:{line_number}")
            yield value


def parse_checkpoint_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("checkpoint must be NAME=PATH")
    name, raw_path = value.split("=", 1)
    name = name.strip()
    raw_path = raw_path.strip()
    if not name or not raw_path:
        raise argparse.ArgumentTypeError("checkpoint must be NAME=PATH")
    return name, Path(raw_path)


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a JSON object")
    return value


def _strict_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    return int(value)


def _validate_vocabulary(
    vocabulary: Mapping[str, Any],
) -> dict[str, tuple[str, ...]]:
    if vocabulary.get("schema") != SEMANTIC_VOCABULARY_SCHEMA:
        raise ValueError("semantic vocabulary schema mismatch")
    raw_soft = vocabulary.get("soft_vocabulary")
    if not isinstance(raw_soft, Mapping):
        raise ValueError("semantic vocabulary lacks soft_vocabulary")
    labels: dict[str, tuple[str, ...]] = {}
    for field, raw_values in raw_soft.items():
        if not isinstance(field, str):
            raise TypeError("soft vocabulary field must be a string")
        if not isinstance(raw_values, list) or not raw_values:
            raise ValueError(f"soft vocabulary {field!r} must be a non-empty list")
        values = tuple(str(item) for item in raw_values)
        if any(not isinstance(item, str) for item in raw_values):
            raise TypeError(f"soft vocabulary {field!r} contains non-string labels")
        if len(values) != len(set(values)):
            raise ValueError(f"soft vocabulary {field!r} contains duplicate labels")
        labels[field] = values
    missing = set(PREDICTION_FIELDS) - set(labels)
    if missing:
        raise ValueError(f"semantic vocabulary lacks prediction fields: {sorted(missing)}")
    for field in PREDICTION_FIELDS:
        if UNKNOWN_LABEL not in labels[field]:
            raise ValueError(f"semantic vocabulary {field!r} lacks {UNKNOWN_LABEL}")
    return labels


def _identity_from_row(row: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    plan = row.get("plan_state")
    if not isinstance(plan, Mapping):
        raise ValueError(f"{label} lacks plan_state")
    n_value = _strict_int(plan.get("N"), label=f"{label} N")
    raw_elements = plan.get("elements")
    raw_counts = plan.get("counts")
    if not isinstance(raw_elements, list) or not isinstance(raw_counts, list):
        raise TypeError(f"{label} elements/counts must be lists")
    if not raw_elements or len(raw_elements) != len(raw_counts):
        raise ValueError(f"{label} elements/counts are not aligned")
    elements: list[str] = []
    counts: list[int] = []
    for index, (element, count) in enumerate(zip(raw_elements, raw_counts)):
        if not isinstance(element, str) or not element:
            raise TypeError(f"{label} element {index} is invalid")
        count_value = _strict_int(count, label=f"{label} count {index}")
        if count_value <= 0:
            raise ValueError(f"{label} count {index} must be positive")
        elements.append(element)
        counts.append(count_value)
    if len(elements) != len(set(elements)):
        raise ValueError(f"{label} repeats an element identity")
    if n_value <= 0 or sum(counts) != n_value:
        raise ValueError(f"{label} violates exact N/count identity")
    proposal = row.get("proposal_targets")
    if not isinstance(proposal, Mapping):
        raise ValueError(f"{label} lacks proposal_targets")
    if _strict_int(proposal.get("N"), label=f"{label} proposal N") != n_value:
        raise ValueError(f"{label} proposal N mismatches exact identity")
    if _strict_int(proposal.get("arity"), label=f"{label} proposal arity") != len(
        elements
    ):
        raise ValueError(f"{label} proposal arity mismatches exact identity")
    return {"N": n_value, "elements": elements, "counts": counts}


def _validate_soft_labels(
    row: Mapping[str, Any],
    *,
    labels: Mapping[str, Sequence[str]],
    label: str,
) -> None:
    values = row.get("soft_labels")
    if not isinstance(values, Mapping):
        raise ValueError(f"{label} lacks soft_labels")
    if set(values) != set(labels):
        raise ValueError(f"{label} soft-label fields mismatch vocabulary")
    for field, vocabulary in labels.items():
        class_index = _strict_int(values[field], label=f"{label} {field} label")
        if class_index < 0 or class_index >= len(vocabulary):
            raise ValueError(f"{label} {field} label is outside vocabulary")


def _load_semantic_split(
    path: Path,
    *,
    split: str,
    labels: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    source_rows = list(iter_jsonl(path))
    if not source_rows:
        raise ValueError(f"semantic {split} split is empty")
    eligible: list[dict[str, Any]] = []
    previous_index: int | None = None
    for ordinal, row in enumerate(source_rows):
        row_label = f"{split} row {ordinal}"
        if row.get("schema") != SEMANTIC_ROW_SCHEMA:
            raise ValueError(f"{row_label} schema mismatch")
        source_row_idx = _strict_int(
            row.get("source_row_idx"), label=f"{row_label} source_row_idx"
        )
        if previous_index is not None and source_row_idx <= previous_index:
            raise ValueError(f"{split} source_row_idx order mismatch")
        previous_index = source_row_idx
        declared_split = row.get("split") or row.get("source_split")
        if declared_split not in (None, "") and str(declared_split) != split:
            raise ValueError(f"{row_label} split mismatch")
        proposal_supervision = row.get("proposal_supervision")
        if not isinstance(proposal_supervision, bool):
            raise TypeError(f"{row_label} proposal_supervision must be boolean")
        if not proposal_supervision:
            continue
        identity = _identity_from_row(row, label=row_label)
        _validate_soft_labels(row, labels=labels, label=row_label)
        eligible.append(
            {
                "source_row_idx": source_row_idx,
                "identity": identity,
            }
        )
    if not eligible:
        raise ValueError(f"semantic {split} split has no supervised rows")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "source_rows": len(source_rows),
        "eligible_rows": len(eligible),
        "source_coverage": len(eligible) / len(source_rows),
        "rows": eligible,
    }


def load_semantic_inputs(semantic_data_dir: Path) -> dict[str, Any]:
    data_dir = semantic_data_dir.resolve()
    vocabulary_path = data_dir / "vocabulary.json"
    manifest_path = data_dir / "manifest.json"
    vocabulary = _read_json_object(vocabulary_path, label="semantic vocabulary")
    labels = _validate_vocabulary(vocabulary)
    manifest = _read_json_object(manifest_path, label="semantic manifest")
    if manifest.get("schema") != SEMANTIC_MANIFEST_SCHEMA:
        raise ValueError("semantic data manifest schema mismatch")
    gate = manifest.get("gate")
    if not isinstance(gate, Mapping) or gate.get("planner_training_data_authorized") is not True:
        raise RuntimeError("semantic data gate did not authorize Planner inference")
    if manifest.get("vocabulary") != vocabulary:
        raise ValueError("semantic manifest/vocabulary payload mismatch")
    splits: dict[str, dict[str, Any]] = {}
    manifest_splits = manifest.get("splits")
    if not isinstance(manifest_splits, Mapping):
        raise ValueError("semantic manifest lacks split summaries")
    for split in REQUIRED_SPLITS:
        split_path = data_dir / f"{split}.jsonl"
        split_payload = _load_semantic_split(split_path, split=split, labels=labels)
        manifest_split = manifest_splits.get(split)
        if not isinstance(manifest_split, Mapping):
            raise ValueError(f"semantic manifest lacks {split} split")
        if _strict_int(manifest_split.get("rows"), label=f"manifest {split} rows") != split_payload[
            "source_rows"
        ]:
            raise ValueError(f"semantic manifest/{split} row-count mismatch")
        splits[split] = split_payload
    return {
        "data_dir": str(data_dir),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "vocabulary_path": str(vocabulary_path.resolve()),
        "vocabulary_sha256": sha256_file(vocabulary_path),
        "vocabulary": vocabulary,
        "labels": labels,
        "splits": splits,
    }


def load_training_module() -> Any:
    path = PROJECT_ROOT / "scripts" / "train_c3fd_planner.py"
    spec = importlib.util.spec_from_file_location(
        "train_c3fd_planner_for_soft_export", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _infer_split(
    *,
    model: Any,
    context: Any,
    dataset_path: Path,
    expected_rows: Sequence[Mapping[str, Any]],
    vocabulary: Mapping[str, Any],
    labels: Mapping[str, Sequence[str]],
    batch_size: int,
    torch: Any,
    data_loader_class: Any,
    training: Any,
    semantic_head_flags: Any,
) -> dict[str, Any]:
    soft_fields = tuple(sorted(labels))
    dataset = training.C3FDDataset(dataset_path)
    observed_indices = [
        _strict_int(row.get("source_row_idx"), label="dataset source_row_idx")
        for row in dataset.rows
    ]
    expected_indices = [int(row["source_row_idx"]) for row in expected_rows]
    if observed_indices != expected_indices:
        raise ValueError("runtime dataset source_row_idx order mismatch")
    eos_id = _strict_int(
        vocabulary.get("species_eos_id"), label="semantic species_eos_id"
    )
    loader = data_loader_class(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=lambda rows: training.collate(
            rows,
            eos_species_id=eos_id,
            soft_fields=soft_fields,
        ),
    )
    predictions: dict[str, list[dict[str, Any]]] = {
        field: [] for field in PREDICTION_FIELDS
    }
    with torch.inference_mode():
        for batch in loader:
            batch_rows = int(batch["n_targets"].shape[0])
            output = model(
                context.expand(batch_rows, -1),
                previous_species_indices=batch["previous_species_indices"],
                previous_count_values=batch["previous_count_values"],
                previous_n_values=batch["previous_n_values"],
                ledger_features=batch["ledger_features"],
                flags=semantic_head_flags(use_physics=True),
            )
            rich_logits = output.rich_logits
            if not isinstance(rich_logits, Mapping):
                raise TypeError("C3FD output rich_logits is not a mapping")
            for field in PREDICTION_FIELDS:
                target = batch[f"rich:{field}"]
                valid = target != -100
                per_row = valid.sum(dim=1)
                if not bool(torch.all(per_row == 1).item()):
                    raise ValueError(f"{field} target mask does not identify one row position")
                positions = valid.to(dtype=torch.long).argmax(dim=1)
                logits = rich_logits.get(field)
                if logits is None:
                    raise ValueError(f"checkpoint output lacks independent {field} head")
                if tuple(logits.shape[:2]) != tuple(target.shape):
                    raise ValueError(f"checkpoint {field} logit shape mismatch")
                if int(logits.shape[-1]) != len(labels[field]):
                    raise ValueError(f"checkpoint {field} vocabulary dimension mismatch")
                row_indices = torch.arange(batch_rows, dtype=torch.long)
                selected = logits[row_indices, positions].float()
                probabilities = torch.softmax(selected, dim=-1)
                confidence, class_index = probabilities.max(dim=-1)
                if not bool(torch.isfinite(confidence).all().item()):
                    raise ValueError(f"checkpoint {field} confidence is non-finite")
                for raw_class, raw_confidence in zip(
                    class_index.tolist(), confidence.tolist()
                ):
                    class_value = int(raw_class)
                    predictions[field].append(
                        {
                            "prediction": str(labels[field][class_value]),
                            "confidence": float(raw_confidence),
                        }
                    )
    for field in PREDICTION_FIELDS:
        if len(predictions[field]) != len(expected_rows):
            raise ValueError(f"checkpoint {field} prediction coverage mismatch")
    return {
        "source_row_indices": expected_indices,
        "predictions": predictions,
    }


def run_checkpoint(
    *,
    name: str,
    checkpoint_path: Path,
    semantic_inputs: Mapping[str, Any],
    batch_size: int,
) -> dict[str, Any]:
    # Runtime imports stay lazy so pure assembly tests require no torch install.
    import torch
    from torch.utils.data import DataLoader

    from crystal_dlm.c3fd_planner_model import C3FDPlannerConfig, C3FDPlannerModel
    from crystal_dlm.semantic_composition_head import SemanticHeadFlags

    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    path = checkpoint_path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, Mapping) or checkpoint.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError(f"unexpected checkpoint schema at {path}")
    vocabulary_sha = str(semantic_inputs["vocabulary_sha256"])
    if str(checkpoint.get("vocabulary_sha256")) != vocabulary_sha:
        raise ValueError(f"checkpoint/vocabulary SHA mismatch at {path}")
    config_payload = checkpoint.get("config")
    if not isinstance(config_payload, Mapping):
        raise ValueError(f"checkpoint config mismatch at {path}")
    config = C3FDPlannerConfig(**dict(config_payload))
    labels: Mapping[str, Sequence[str]] = semantic_inputs["labels"]
    expected_dims = {
        field: len(values) for field, values in labels.items()
    }
    if dict(config.rich_soft_head_dims) != expected_dims:
        raise ValueError(f"checkpoint soft-head/vocabulary mismatch at {path}")
    vocabulary = semantic_inputs["vocabulary"]
    physics_payload = vocabulary.get("physics")
    if not isinstance(physics_payload, Mapping) or not isinstance(
        physics_payload.get("matrix"), list
    ):
        raise ValueError("semantic vocabulary lacks physics matrix")
    physics = torch.tensor(physics_payload["matrix"], dtype=torch.float32)
    model = C3FDPlannerModel(config, physics_features=physics)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    context = torch.as_tensor(checkpoint.get("context"), dtype=torch.float32)
    if tuple(context.shape) != (1, int(config.context_size)):
        raise ValueError(f"checkpoint context mismatch at {path}")
    training = load_training_module()
    splits: dict[str, dict[str, Any]] = {}
    for split in REQUIRED_SPLITS:
        split_payload = semantic_inputs["splits"][split]
        splits[split] = _infer_split(
            model=model,
            context=context,
            dataset_path=Path(split_payload["path"]),
            expected_rows=split_payload["rows"],
            vocabulary=vocabulary,
            labels=labels,
            batch_size=int(batch_size),
            torch=torch,
            data_loader_class=DataLoader,
            training=training,
            semantic_head_flags=SemanticHeadFlags,
        )
    return {
        "name": str(name),
        "seed": _strict_int(checkpoint.get("seed"), label=f"checkpoint {name} seed"),
        "checkpoint_schema": str(checkpoint.get("schema")),
        "checkpoint_path": str(path),
        "checkpoint_sha256": sha256_file(path),
        "config_sha256": sha256_json(dict(config_payload)),
        "vocabulary_sha256": vocabulary_sha,
        "device": "cpu",
        "confidence_definition": "maximum_uncalibrated_softmax_probability",
        "splits": splits,
    }


def _validate_checkpoint_results(
    results: Sequence[Mapping[str, Any]],
    *,
    vocabulary_sha256: str,
) -> tuple[str, ...]:
    if len(results) != 2:
        raise ValueError("exactly two frozen seed17/seed18 checkpoints are required")
    names = tuple(str(result.get("name") or "") for result in results)
    if any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("checkpoint names must be non-empty and unique")
    seeds = {
        _strict_int(result.get("seed"), label=f"checkpoint {name} seed")
        for name, result in zip(names, results)
    }
    if seeds != EXPECTED_CHECKPOINT_SEEDS:
        raise ValueError("checkpoint seeds must be exactly 17 and 18")
    paths = tuple(str(result.get("checkpoint_path") or "") for result in results)
    if any(not path for path in paths) or len(set(paths)) != len(paths):
        raise ValueError("checkpoint paths must be non-empty and distinct")
    config_hashes = {str(result.get("config_sha256") or "") for result in results}
    if len(config_hashes) != 1 or "" in config_hashes:
        raise ValueError("checkpoint configs mismatch")
    for name, result in zip(names, results):
        if result.get("checkpoint_schema") != CHECKPOINT_SCHEMA:
            raise ValueError(f"checkpoint {name} schema mismatch")
        if str(result.get("vocabulary_sha256")) != vocabulary_sha256:
            raise ValueError(f"checkpoint {name} vocabulary mismatch")
        digest = str(result.get("checkpoint_sha256") or "")
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError(f"checkpoint {name} SHA256 is invalid")
        if result.get("device") != "cpu":
            raise ValueError(f"checkpoint {name} was not inferred on CPU")
    return names


def _normalise_prediction(
    value: Any,
    *,
    field: str,
    vocabulary: Sequence[str],
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} {field} prediction must be an object")
    if set(value) != {"prediction", "confidence"}:
        raise ValueError(f"{label} {field} prediction fields mismatch")
    prediction = value.get("prediction")
    confidence = value.get("confidence")
    if not isinstance(prediction, str) or prediction not in vocabulary:
        raise ValueError(f"{label} {field} prediction is outside vocabulary")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise TypeError(f"{label} {field} confidence must be numeric")
    confidence_value = float(confidence)
    if not math.isfinite(confidence_value) or not 0.0 <= confidence_value <= 1.0:
        raise ValueError(f"{label} {field} confidence is invalid")
    return {"prediction": prediction, "confidence": confidence_value}


def _assemble_split(
    *,
    split: str,
    semantic_rows: Sequence[Mapping[str, Any]],
    checkpoint_results: Sequence[Mapping[str, Any]],
    checkpoint_names: Sequence[str],
    labels: Mapping[str, Sequence[str]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    expected_indices = [int(row["source_row_idx"]) for row in semantic_rows]
    per_checkpoint: dict[str, Mapping[str, Any]] = {}
    for name, result in zip(checkpoint_names, checkpoint_results):
        splits = result.get("splits")
        if not isinstance(splits, Mapping) or set(splits) != set(REQUIRED_SPLITS):
            raise ValueError(f"checkpoint {name} split fields mismatch")
        split_result = splits.get(split)
        if not isinstance(split_result, Mapping):
            raise ValueError(f"checkpoint {name} lacks {split} predictions")
        observed_indices = split_result.get("source_row_indices")
        if not isinstance(observed_indices, list) or [
            _strict_int(value, label=f"checkpoint {name} source_row_idx")
            for value in observed_indices
        ] != expected_indices:
            raise ValueError(f"checkpoint {name} {split} row/order mismatch")
        predictions = split_result.get("predictions")
        if not isinstance(predictions, Mapping) or set(predictions) != set(
            PREDICTION_FIELDS
        ):
            raise ValueError(f"checkpoint {name} {split} prediction fields mismatch")
        for field in PREDICTION_FIELDS:
            values = predictions[field]
            if not isinstance(values, list) or len(values) != len(semantic_rows):
                raise ValueError(f"checkpoint {name} {split} {field} coverage mismatch")
        per_checkpoint[name] = predictions

    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    for name in checkpoint_names:
        field_summary: dict[str, Any] = {}
        for field in PREDICTION_FIELDS:
            counts: Counter[str] = Counter()
            unknown = 0
            for ordinal in range(len(semantic_rows)):
                normalised = _normalise_prediction(
                    per_checkpoint[name][field][ordinal],
                    field=field,
                    vocabulary=labels[field],
                    label=f"checkpoint {name} {split} row {ordinal}",
                )
                counts[normalised["prediction"]] += 1
                unknown += int(normalised["prediction"] == UNKNOWN_LABEL)
            field_summary[field] = {
                "rows": len(semantic_rows),
                "coverage": 1.0,
                "unknown_predictions": unknown,
                "class_distribution": {
                    value: int(counts.get(value, 0)) for value in labels[field]
                },
            }
        summary[name] = field_summary

    for ordinal, source in enumerate(semantic_rows):
        by_checkpoint: dict[str, Any] = {}
        for name in checkpoint_names:
            by_checkpoint[name] = {
                field: _normalise_prediction(
                    per_checkpoint[name][field][ordinal],
                    field=field,
                    vocabulary=labels[field],
                    label=f"checkpoint {name} {split} row {ordinal}",
                )
                for field in PREDICTION_FIELDS
            }
        identity = source.get("identity")
        if not isinstance(identity, Mapping) or set(identity) != {
            "N",
            "elements",
            "counts",
        }:
            raise ValueError(f"semantic {split} row {ordinal} identity fields mismatch")
        row = {
            "schema": ROW_SCHEMA,
            "split": split,
            "source_row_idx": int(source["source_row_idx"]),
            "N": int(identity["N"]),
            "elements": [str(value) for value in identity["elements"]],
            "counts": [int(value) for value in identity["counts"]],
            "predictions_by_checkpoint": by_checkpoint,
        }
        if set(row) != {
            "schema",
            "split",
            "source_row_idx",
            "N",
            "elements",
            "counts",
            "predictions_by_checkpoint",
        }:
            raise AssertionError("export row whitelist changed")
        rows.append(row)
    return rows, summary


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True))
        handle.write("\n")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row))
            handle.write("\n")


def write_export(
    *,
    semantic_inputs: Mapping[str, Any],
    checkpoint_results: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    output = output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    checkpoint_names = _validate_checkpoint_results(
        checkpoint_results,
        vocabulary_sha256=str(semantic_inputs["vocabulary_sha256"]),
    )
    labels: Mapping[str, Sequence[str]] = semantic_inputs["labels"]
    assembled: dict[str, list[dict[str, Any]]] = {}
    summaries: dict[str, Any] = {}
    for split in REQUIRED_SPLITS:
        split_payload = semantic_inputs["splits"][split]
        assembled[split], summaries[split] = _assemble_split(
            split=split,
            semantic_rows=split_payload["rows"],
            checkpoint_results=checkpoint_results,
            checkpoint_names=checkpoint_names,
            labels=labels,
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.preparing.{os.getpid()}"
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir(exist_ok=False)
    try:
        for split in REQUIRED_SPLITS:
            _write_jsonl(staging / f"{split}.jsonl", assembled[split])
        row_hashes = {
            f"{split}.jsonl": sha256_file(staging / f"{split}.jsonl")
            for split in REQUIRED_SPLITS
        }
        checkpoint_manifest = [
            {
                key: result[key]
                for key in (
                    "name",
                    "seed",
                    "checkpoint_schema",
                    "checkpoint_path",
                    "checkpoint_sha256",
                    "config_sha256",
                    "vocabulary_sha256",
                    "device",
                    "confidence_definition",
                )
            }
            for result in checkpoint_results
        ]
        split_manifest = {
            split: {
                "source_path": semantic_inputs["splits"][split]["path"],
                "source_sha256": semantic_inputs["splits"][split]["sha256"],
                "source_rows": semantic_inputs["splits"][split]["source_rows"],
                "eligible_rows": semantic_inputs["splits"][split]["eligible_rows"],
                "source_coverage": semantic_inputs["splits"][split][
                    "source_coverage"
                ],
                "emitted_rows": len(assembled[split]),
                "eligible_prediction_coverage": 1.0,
                "output_sha256": row_hashes[f"{split}.jsonl"],
                "predictions": summaries[split],
            }
            for split in REQUIRED_SPLITS
        }
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "output_dir": str(output),
            "device": "cpu",
            "gpu_jobs_used": 0,
            "outcomes_read": False,
            "selection": "none",
            "all_frozen_checkpoints_preserved": True,
            "expected_checkpoint_seeds": sorted(EXPECTED_CHECKPOINT_SEEDS),
            "checkpoint_order": list(checkpoint_names),
            "independent_spacegroup_head": True,
            "prediction_fields": list(PREDICTION_FIELDS),
            "row_source_fields_copied": [
                "source_row_idx",
                "N",
                "elements",
                "counts",
            ],
            "forbidden_source_payloads_copied": [],
            "semantic_data": {
                "data_dir": semantic_inputs["data_dir"],
                "manifest_path": semantic_inputs["manifest_path"],
                "manifest_sha256": semantic_inputs["manifest_sha256"],
                "vocabulary_path": semantic_inputs["vocabulary_path"],
                "vocabulary_sha256": semantic_inputs["vocabulary_sha256"],
                "prediction_vocabulary": {
                    field: list(labels[field]) for field in PREDICTION_FIELDS
                },
            },
            "checkpoints": checkpoint_manifest,
            "splits": split_manifest,
            "outputs": row_hashes,
        }
        manifest_path = staging / "manifest.json"
        _write_json(manifest_path, manifest)
        checksummed = [
            *(f"{split}.jsonl" for split in REQUIRED_SPLITS),
            "manifest.json",
        ]
        checksum_path = staging / "SHA256SUMS"
        with checksum_path.open("x", encoding="utf-8", newline="\n") as handle:
            for name in sorted(checksummed):
                handle.write(f"{sha256_file(staging / name)}  {name}\n")
        success = {
            "manifest_sha256": sha256_file(manifest_path),
            "sha256sums_sha256": sha256_file(checksum_path),
        }
        _write_json(staging / "_SUCCESS", success)
        if output.exists():
            raise FileExistsError(output)
        staging.rename(output)
        return manifest
    except Exception:
        with suppress(Exception):
            shutil.rmtree(staging)
        raise


def export_predictions(
    *,
    checkpoints: Sequence[tuple[str, Path]],
    semantic_data_dir: Path,
    output_dir: Path,
    batch_size: int,
) -> dict[str, Any]:
    if len(checkpoints) != 2:
        raise ValueError("exactly two --checkpoint arguments are required")
    names = [name for name, _path in checkpoints]
    if len(set(names)) != len(names):
        raise ValueError("checkpoint names must be unique")
    if output_dir.resolve().exists():
        raise FileExistsError(output_dir.resolve())
    semantic_inputs = load_semantic_inputs(semantic_data_dir)
    results = [
        run_checkpoint(
            name=name,
            checkpoint_path=path,
            semantic_inputs=semantic_inputs,
            batch_size=int(batch_size),
        )
        for name, path in checkpoints
    ]
    return write_export(
        semantic_inputs=semantic_inputs,
        checkpoint_results=results,
        output_dir=output_dir,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export frozen C3FD per-row native soft predictions on CPU."
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        type=parse_checkpoint_arg,
        required=True,
        help="repeat exactly twice as NAME=PATH for frozen seed17/seed18",
    )
    parser.add_argument(
        "--semantic-data-dir",
        "--data-dir",
        dest="semantic_data_dir",
        type=Path,
        required=True,
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = export_predictions(
        checkpoints=args.checkpoint,
        semantic_data_dir=args.semantic_data_dir,
        output_dir=args.output_dir,
        batch_size=int(args.batch_size),
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.resolve()),
                "checkpoints": manifest["checkpoint_order"],
                "splits": {
                    split: manifest["splits"][split]["emitted_rows"]
                    for split in REQUIRED_SPLITS
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
