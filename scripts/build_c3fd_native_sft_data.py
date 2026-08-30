#!/usr/bin/env python3
"""Build outcome-blind C3FD-native multi-view crystal-DLM SFT data.

Each aligned source row is expanded into equally weighted views that retain the
exact same dynamic crystal body:

* ``teacher-native`` uses train-only structural fields from the MP20 row;
* one ``predicted-native-<checkpoint>`` view preserves every frozen C3FD model;
* ``soft-masked`` masks only lattice/space-group/volume hints; and
* ``minimal-reference`` retains the established CTV minimal prompt.

The builder deliberately emits a curated payload instead of copying source rows,
so source-side outcome annotations cannot leak into prompts or training records.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import suppress
import hashlib
from itertools import zip_longest
import json
import math
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
SCRIPTS = PROJECT_ROOT / "scripts"
for import_path in (SRC, SCRIPTS):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from build_ctv_minimal_spec_data import (  # noqa: E402
    minimal_prompt,
    minimal_spec_from_plan,
)
from crystal_dlm.c3fd_native_plan import (  # noqa: E402
    build_native_body_prompt,
    mask_native_soft_fields,
    parse_native_plan_line,
    serialize_native_plan,
)
from crystal_dlm.composition_identity import (  # noqa: E402
    canonical_symbol_counts,
    formula_from_symbol_counts,
    identity_from_plan_state,
    identity_text,
)
from crystal_dlm.composition_pair_prior import ValenceNode  # noqa: E402
from crystal_dlm.fixed_slot import SYMBOL_TO_Z, Z_TO_SYMBOL  # noqa: E402


SCHEMA = "c3fd_native_sft_row_v2"
MANIFEST_SCHEMA = "c3fd_native_sft_manifest_v2"
FORMAL_PREDICTION_MANIFEST_SCHEMA = "c3fd_native_soft_prediction_manifest_v1"
FORMAL_PREDICTION_ROW_SCHEMA = "c3fd_native_soft_prediction_row_v1"
FORMAL_CHECKPOINT_ORDER = ("seed17", "seed18")
FORMAL_CHECKPOINT_SEEDS = (17, 18)
LEGACY_DEVELOPMENT_CHECKPOINT = "development-single"
STATIC_VIEWS = ("teacher-native", "soft-masked", "minimal-reference")
SOFT_FIELDS = (
    "lattice_system",
    "spacegroup_bucket",
    "volume_per_atom_bin",
)
SOFT_LINE_FIELDS = {"LS", "SG", "VP"}
PREDICTED_CONTAINER_KEYS = (
    "frozen_predicted_soft_fields",
    "predicted_soft_fields",
    "predicted_plan_state",
    "predicted_plan",
    "target_proposal",
    "plan_state",
)
INLINE_PREDICTED_CONTAINER_KEYS = PREDICTED_CONTAINER_KEYS[:4]
SAFE_TRAINING_KEYS = (
    "loss_profile",
    "mask_policy",
    "module",
    "module_id",
    "sft_mask_policy",
)
FORBIDDEN_KEY_FRAGMENTS = ("energy", "hull", "stability", "chgnet")
FORBIDDEN_PROMPT_FRAGMENTS = (
    "energy",
    "e_above_hull",
    "hull",
    "stability",
    "prototype_key",
    "oxidation_candidates",
)
_MISSING = object()


def predicted_view_name(checkpoint_name: str) -> str:
    name = str(checkpoint_name)
    safe_characters = (
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
    )
    if not name or any(character not in safe_characters for character in name):
        raise ValueError(f"unsafe prediction checkpoint name {name!r}")
    return f"predicted-native-{name}"


def expanded_view_names(checkpoint_order: Sequence[str]) -> tuple[str, ...]:
    return (
        STATIC_VIEWS[0],
        *(predicted_view_name(name) for name in checkpoint_order),
        *STATIC_VIEWS[1:],
    )


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _load_prediction_contract(
    predicted_soft_dir: Path | None,
    *,
    allow_legacy_single_prediction_development: bool,
) -> dict[str, Any]:
    if predicted_soft_dir is None:
        if not allow_legacy_single_prediction_development:
            raise ValueError(
                "formal predictions require --predicted-soft-dir; legacy inline "
                "single-prediction input is development-only"
            )
        return {
            "mode": "legacy-single-development",
            "checkpoint_order": (LEGACY_DEVELOPMENT_CHECKPOINT,),
            "manifest_sha256": None,
            "selection": "none",
        }

    manifest_path = predicted_soft_dir / "manifest.json"
    if not manifest_path.is_file():
        if not allow_legacy_single_prediction_development:
            raise ValueError(
                "formal predicted-soft directory lacks its frozen manifest; "
                "legacy single-prediction input is development-only"
            )
        return {
            "mode": "legacy-single-development",
            "checkpoint_order": (LEGACY_DEVELOPMENT_CHECKPOINT,),
            "manifest_sha256": None,
            "selection": "none",
        }

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise TypeError("predicted-soft manifest is not an object")
    if manifest.get("schema") != FORMAL_PREDICTION_MANIFEST_SCHEMA:
        raise ValueError("predicted-soft manifest schema is not formal/frozen")
    checksum_path = predicted_soft_dir / "SHA256SUMS"
    success_path = predicted_soft_dir / "_SUCCESS"
    if not checksum_path.is_file() or not success_path.is_file():
        raise ValueError("formal predicted-soft export lacks frozen completion files")
    checksums: dict[str, str] = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        parts = line.split("  ", 1)
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise ValueError("formal predicted-soft SHA256SUMS is malformed")
        digest, name = parts
        if name in checksums:
            raise ValueError(f"duplicate formal predicted-soft checksum {name}")
        checksums[name] = digest
    expected_frozen_files = {"manifest.json", "train.jsonl", "val.jsonl"}
    if set(checksums) != expected_frozen_files:
        raise ValueError("formal predicted-soft frozen file support changed")
    for name, expected_sha in checksums.items():
        path = predicted_soft_dir / name
        if not path.is_file() or sha256_file(path) != expected_sha:
            raise ValueError(f"formal predicted-soft frozen hash changed for {name}")
    success = json.loads(success_path.read_text(encoding="utf-8"))
    if not isinstance(success, Mapping) or set(success) != {
        "manifest_sha256",
        "sha256sums_sha256",
    }:
        raise ValueError("formal predicted-soft _SUCCESS contract changed")
    if str(success["manifest_sha256"]) != sha256_file(manifest_path):
        raise ValueError("formal predicted-soft manifest completion hash changed")
    if str(success["sha256sums_sha256"]) != sha256_file(checksum_path):
        raise ValueError("formal predicted-soft checksum completion hash changed")
    checkpoint_order = manifest.get("checkpoint_order")
    if not isinstance(checkpoint_order, list) or tuple(
        checkpoint_order
    ) != FORMAL_CHECKPOINT_ORDER:
        raise ValueError(
            "formal checkpoint support/order must be exactly seed17,seed18"
        )
    expected_seeds = manifest.get("expected_checkpoint_seeds")
    if not isinstance(expected_seeds, list) or tuple(
        expected_seeds
    ) != FORMAL_CHECKPOINT_SEEDS:
        raise ValueError("formal expected checkpoint seeds must be exactly 17,18")
    checkpoints = manifest.get("checkpoints")
    if not isinstance(checkpoints, list) or len(checkpoints) != len(
        FORMAL_CHECKPOINT_ORDER
    ):
        raise ValueError("formal frozen checkpoint manifest support changed")
    observed_names: list[str] = []
    observed_seeds: list[int] = []
    for checkpoint in checkpoints:
        if not isinstance(checkpoint, Mapping):
            raise TypeError("formal checkpoint manifest entry is not an object")
        observed_names.append(str(checkpoint.get("name") or ""))
        observed_seeds.append(int(checkpoint.get("seed") or -1))
    if tuple(observed_names) != FORMAL_CHECKPOINT_ORDER or tuple(
        observed_seeds
    ) != FORMAL_CHECKPOINT_SEEDS:
        raise ValueError("formal frozen checkpoint names/seeds/order disagree")
    if manifest.get("selection") != "none":
        raise ValueError("formal predicted-soft export performed checkpoint selection")
    if manifest.get("all_frozen_checkpoints_preserved") is not True:
        raise ValueError("formal predicted-soft export did not preserve all checkpoints")
    if manifest.get("outcomes_read") is not False:
        raise ValueError("formal predicted-soft export is not outcome-blind")
    if manifest.get("prediction_fields") != list(SOFT_FIELDS):
        raise ValueError("formal predicted-soft field support/order changed")
    split_manifest = manifest.get("splits")
    outputs = manifest.get("outputs")
    if not isinstance(split_manifest, Mapping) or set(split_manifest) != {
        "train",
        "val",
    }:
        raise ValueError("formal predicted-soft split support changed")
    if not isinstance(outputs, Mapping) or set(outputs) != {
        "train.jsonl",
        "val.jsonl",
    }:
        raise ValueError("formal predicted-soft output support changed")
    for split in ("train", "val"):
        path = predicted_soft_dir / f"{split}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_sha = sha256_file(path)
        split_payload = split_manifest[split]
        if not isinstance(split_payload, Mapping):
            raise TypeError(f"formal predicted-soft {split} manifest is malformed")
        if str(split_payload.get("output_sha256") or "") != actual_sha:
            raise ValueError(f"formal predicted-soft {split} hash changed")
        if str(outputs.get(f"{split}.jsonl") or "") != actual_sha:
            raise ValueError(f"formal predicted-soft {split} output hash changed")
        predictions = split_payload.get("predictions")
        if not isinstance(predictions, Mapping) or tuple(
            predictions
        ) != FORMAL_CHECKPOINT_ORDER:
            raise ValueError(
                f"formal predicted-soft {split} checkpoint support/order changed"
            )
    return {
        "mode": "formal-multi-checkpoint",
        "checkpoint_order": FORMAL_CHECKPOINT_ORDER,
        "manifest_sha256": sha256_file(manifest_path),
        "selection": "none",
    }


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"non-object JSONL row in {path}:{line_number}")
            yield value


def _plan_from_row(row: Mapping[str, Any], *, label: str) -> Mapping[str, Any]:
    value = row.get("plan_state") or row.get("r5_plan_state")
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} lacks plan_state")
    return value


def _check_split(row: Mapping[str, Any], split: str, *, label: str) -> None:
    declared = row.get("split") or row.get("source_split")
    if declared not in (None, "") and str(declared) != split:
        raise ValueError(f"{label} split changed: {declared!r} != {split!r}")


def _source_row_idx(
    row: Mapping[str, Any],
    expected: int | None,
    *,
    label: str,
    allow_certificate_alias: bool,
) -> int:
    raw_values: list[Any] = []
    if "source_row_idx" in row:
        raw_values.append(row["source_row_idx"])
    if allow_certificate_alias and "c3fd_certificate_source_row_idx" in row:
        raw_values.append(row["c3fd_certificate_source_row_idx"])
    if not raw_values:
        raise ValueError(f"{label} lacks source_row_idx")
    values = {int(value) for value in raw_values}
    if len(values) != 1:
        raise ValueError(f"{label} carries conflicting source_row_idx values")
    actual = next(iter(values))
    if expected is not None and actual != int(expected):
        raise ValueError(
            f"{label} source_row_idx changed: {actual} != {int(expected)}"
        )
    return actual


def _index_rows_by_source_idx(
    rows: Iterable[Mapping[str, Any]],
    *,
    label: str,
    allow_certificate_alias: bool,
) -> dict[int, Mapping[str, Any]]:
    indexed: dict[int, Mapping[str, Any]] = {}
    for row in rows:
        source_idx = _source_row_idx(
            row,
            None,
            label=label,
            allow_certificate_alias=allow_certificate_alias,
        )
        if source_idx in indexed:
            raise ValueError(f"{label} duplicates source_row_idx {source_idx}")
        indexed[source_idx] = row
    return indexed


def _composition(plan: Mapping[str, Any]) -> tuple[tuple[str, int], ...]:
    return canonical_symbol_counts(
        [str(value) for value in (plan.get("elements") or ())],
        [int(value) for value in (plan.get("counts") or ())],
    )


def _n_value(plan: Mapping[str, Any], *, label: str) -> int:
    value = int(plan.get("N") or 0)
    if value < 1 or sum(count for _symbol, count in _composition(plan)) != value:
        raise ValueError(f"{label} violates exact N/count conservation")
    return value


def _assert_plan_alignment(
    source_plan: Mapping[str, Any],
    aligned_plan: Mapping[str, Any],
    *,
    label: str,
) -> None:
    if _n_value(source_plan, label="DLM Plan") != _n_value(
        aligned_plan, label=label
    ):
        raise ValueError(f"DLM/{label} N alignment changed")
    if _composition(source_plan) != _composition(aligned_plan):
        raise ValueError(f"DLM/{label} composition alignment changed")
    # Composition/family must align exactly.  The semantic sidecar's frozen
    # valence certificate is authoritative for charge: older DLM rows can
    # legitimately carry the pre-certificate bucket and are audited below.
    for key in ("anion_framework",):
        left = source_plan.get(key)
        right = aligned_plan.get(key)
        if left not in (None, "") and right not in (None, "") and left != right:
            raise ValueError(f"DLM/{label} {key} alignment changed")


def _species_vocabulary(semantic_dir: Path) -> tuple[dict[int, tuple[str, int]], str | None]:
    path = semantic_dir / "vocabulary.json"
    if not path.is_file():
        return {}, None
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("species") if isinstance(payload, Mapping) else None
    if not isinstance(rows, list):
        raise ValueError("semantic vocabulary lacks species rows")
    result: dict[int, tuple[str, int]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("semantic vocabulary species entry is not an object")
        species_id = int(row["id"])
        atomic_number = int(row["atomic_number"])
        if atomic_number not in Z_TO_SYMBOL:
            raise ValueError(f"unsupported atomic number {atomic_number}")
        if species_id in result:
            raise ValueError(f"duplicate semantic species id {species_id}")
        result[species_id] = (
            Z_TO_SYMBOL[atomic_number],
            int(row["oxidation_state"]),
        )
    return result, sha256_file(path)


def _node_from_value(
    value: Any,
    vocabulary: Mapping[int, tuple[str, int]],
) -> tuple[str, int]:
    if isinstance(value, Mapping):
        atomic_number = int(value["atomic_number"])
        oxidation = int(value["oxidation_state"])
        if atomic_number not in Z_TO_SYMBOL:
            raise ValueError(f"unsupported atomic number {atomic_number}")
        return Z_TO_SYMBOL[atomic_number], oxidation
    if isinstance(value, int) or (isinstance(value, str) and value.isdigit()):
        species_id = int(value)
        if species_id not in vocabulary:
            raise ValueError(
                f"semantic species id {species_id} is unavailable in vocabulary"
            )
        return vocabulary[species_id]
    if isinstance(value, str):
        node = ValenceNode.from_token(value)
        if int(node.atomic_number) not in Z_TO_SYMBOL:
            raise ValueError(f"unsupported semantic species {value!r}")
        return Z_TO_SYMBOL[int(node.atomic_number)], int(node.oxidation_state)
    raise TypeError(f"unsupported semantic species label {value!r}")


def _normalise_valence_species(
    values: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    merged: Counter[tuple[str, int]] = Counter()
    for value in values:
        symbol = str(value.get("element") or value.get("symbol") or "")
        if symbol not in SYMBOL_TO_Z:
            raise ValueError(f"unsupported valence element {symbol!r}")
        count = int(value.get("count") or 0)
        oxidation = value.get("oxidation_state")
        if count <= 0 or oxidation in (None, "", "unknown"):
            raise ValueError(f"invalid frozen valence species {value!r}")
        merged[(symbol, int(oxidation))] += count
    return [
        {
            "element": symbol,
            "count": int(merged[(symbol, oxidation)]),
            "oxidation_state": int(oxidation),
        }
        for symbol, oxidation in sorted(
            merged,
            key=lambda item: (SYMBOL_TO_Z[item[0]], item[1]),
        )
    ]


def _valence_species(
    semantic_row: Mapping[str, Any],
    semantic_plan: Mapping[str, Any],
    vocabulary: Mapping[int, tuple[str, int]],
) -> list[dict[str, Any]]:
    if semantic_row.get("composition_supervision") is not True:
        raise ValueError("semantic row does not authorize composition supervision")
    plan_species = semantic_plan.get("valence_species")
    if isinstance(plan_species, Sequence) and not isinstance(
        plan_species, (str, bytes)
    ):
        values = [
            value
            for value in plan_species
            if isinstance(value, Mapping)
        ]
        if len(values) != len(plan_species):
            raise ValueError("semantic Plan valence_species is malformed")
        if all(
            value.get("oxidation_state") not in (None, "", "unknown")
            for value in values
        ):
            species = _normalise_valence_species(values)
        else:
            # Some historical semantic Plan rows retain unknown oxidation in
            # their convenience field even though the frozen sidecar carries
            # the exact species-label witness used by C3FD supervision.
            plan_species = None
    else:
        plan_species = None
    if plan_species is None:
        labels = semantic_row.get("nodes") or semantic_row.get("species_labels")
        counts = semantic_row.get("counts") or semantic_row.get("count_targets")
        if not isinstance(labels, list) or not isinstance(counts, list):
            raise ValueError("semantic row lacks frozen valence labels/counts")
        if len(labels) != len(counts) or not labels:
            raise ValueError("semantic valence labels/counts are not aligned")
        species = _normalise_valence_species(
            [
                {
                    "element": _node_from_value(label, vocabulary)[0],
                    "oxidation_state": _node_from_value(label, vocabulary)[1],
                    "count": int(count),
                }
                for label, count in zip(labels, counts)
            ]
        )
    species_composition = canonical_symbol_counts(
        [str(row["element"]) for row in species],
        [int(row["count"]) for row in species],
    )
    if species_composition != _composition(semantic_plan):
        raise ValueError("semantic valence species changed exact composition")
    charge_sum = sum(
        int(row["count"]) * int(row["oxidation_state"]) for row in species
    )
    if charge_sum != 0:
        raise ValueError("semantic valence certificate is not charge neutral")
    return species


def _expected_charge_bucket(species: Sequence[Mapping[str, Any]]) -> str:
    elements = {str(row["element"]) for row in species}
    oxidations = [int(row["oxidation_state"]) for row in species]
    if len(elements) == 1 and all(value == 0 for value in oxidations):
        return "single_element"
    if len(elements) > 1 and all(value == 0 for value in oxidations):
        return "all_metal"
    return "neutral_plausible"


def _soft_fields(value: Mapping[str, Any], *, label: str) -> dict[str, str]:
    output: dict[str, str] = {}
    for key in SOFT_FIELDS:
        item = value.get(key)
        if item in (None, ""):
            raise ValueError(f"{label} lacks frozen {key}")
        output[key] = str(item)
    return output


def _predicted_container(
    row: Mapping[str, Any],
    *,
    inline_only: bool,
) -> Mapping[str, Any] | None:
    if not inline_only and all(
        row.get(key) not in (None, "") for key in SOFT_FIELDS
    ):
        return row
    keys = INLINE_PREDICTED_CONTAINER_KEYS if inline_only else PREDICTED_CONTAINER_KEYS
    for key in keys:
        value = row.get(key)
        if isinstance(value, Mapping) and all(
            value.get(field) not in (None, "") for field in SOFT_FIELDS
        ):
            return value
    return None


def _predicted_hard_plan(
    row: Mapping[str, Any],
    soft_container: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    candidates: list[Mapping[str, Any]] = [soft_container]
    for key in ("plan_state", "predicted_plan_state", "predicted_plan", "target_proposal"):
        value = row.get(key)
        if isinstance(value, Mapping) and value not in candidates:
            candidates.append(value)
    if row not in candidates:
        candidates.append(row)
    hard_keys = {"N", "elements", "counts"}
    for candidate in candidates:
        present = hard_keys.intersection(candidate)
        if present and present != hard_keys:
            raise ValueError("predicted Plan carries only a partial hard composition")
        if present == hard_keys:
            return candidate
    return None


def _formal_predicted_soft_fields(
    predicted_row: Mapping[str, Any],
    *,
    checkpoint_order: Sequence[str],
) -> dict[str, dict[str, str]]:
    if predicted_row.get("schema") != FORMAL_PREDICTION_ROW_SCHEMA:
        raise ValueError("formal predicted row schema changed")
    expected_row_fields = {
        "schema",
        "split",
        "source_row_idx",
        "N",
        "elements",
        "counts",
        "predictions_by_checkpoint",
    }
    if set(predicted_row) != expected_row_fields:
        raise ValueError("formal predicted row field support changed")
    by_checkpoint = predicted_row.get("predictions_by_checkpoint")
    if not isinstance(by_checkpoint, Mapping):
        raise TypeError("formal predictions_by_checkpoint is not an object")
    if tuple(by_checkpoint) != tuple(checkpoint_order):
        missing = sorted(set(checkpoint_order) - set(by_checkpoint))
        extra = sorted(set(by_checkpoint) - set(checkpoint_order))
        raise ValueError(
            "formal row checkpoint support/order disagrees with manifest: "
            f"missing={missing}, extra={extra}, observed={list(by_checkpoint)}"
        )
    output: dict[str, dict[str, str]] = {}
    for checkpoint_name in checkpoint_order:
        checkpoint = by_checkpoint[checkpoint_name]
        if not isinstance(checkpoint, Mapping) or tuple(checkpoint) != SOFT_FIELDS:
            raise ValueError(
                f"formal checkpoint {checkpoint_name} soft-field support/order changed"
            )
        values: dict[str, str] = {}
        for field in SOFT_FIELDS:
            prediction = checkpoint[field]
            if not isinstance(prediction, Mapping) or set(prediction) != {
                "prediction",
                "confidence",
            }:
                raise ValueError(
                    f"formal checkpoint {checkpoint_name} {field} payload changed"
                )
            value = prediction.get("prediction")
            confidence = prediction.get("confidence")
            if not isinstance(value, str) or not value:
                raise ValueError(
                    f"formal checkpoint {checkpoint_name} {field} prediction is invalid"
                )
            if isinstance(confidence, bool) or not isinstance(
                confidence, (int, float)
            ):
                raise TypeError(
                    f"formal checkpoint {checkpoint_name} {field} confidence is invalid"
                )
            confidence_value = float(confidence)
            if not math.isfinite(confidence_value) or not 0.0 <= confidence_value <= 1.0:
                raise ValueError(
                    f"formal checkpoint {checkpoint_name} {field} confidence is invalid"
                )
            values[field] = value
        output[str(checkpoint_name)] = values
    return output


def _resolve_predicted_soft_fields_by_checkpoint(
    source_row: Mapping[str, Any],
    semantic_row: Mapping[str, Any],
    predicted_row: Mapping[str, Any] | None,
    *,
    split: str,
    row_index: int,
    source_plan: Mapping[str, Any],
    semantic_species: Sequence[Mapping[str, Any]],
    prediction_mode: str,
    checkpoint_order: Sequence[str],
) -> dict[str, dict[str, str]]:
    if prediction_mode == "formal-multi-checkpoint":
        if predicted_row is None:
            raise ValueError("formal predicted row is missing")
        _check_split(predicted_row, split, label="formal predicted row")
        _source_row_idx(
            predicted_row,
            row_index,
            label="formal predicted row",
            allow_certificate_alias=False,
        )
        _assert_plan_alignment(
            source_plan,
            predicted_row,
            label="formal predicted row",
        )
        return _formal_predicted_soft_fields(
            predicted_row,
            checkpoint_order=checkpoint_order,
        )
    if tuple(checkpoint_order) != (LEGACY_DEVELOPMENT_CHECKPOINT,):
        raise ValueError("legacy development checkpoint support changed")
    if predicted_row is not None:
        _check_split(predicted_row, split, label="predicted row")
        _source_row_idx(
            predicted_row,
            row_index,
            label="predicted row",
            allow_certificate_alias=False,
        )
        container = _predicted_container(predicted_row, inline_only=False)
        if container is None:
            raise ValueError("predicted row lacks all three frozen soft fields")
        predicted_plan = _predicted_hard_plan(predicted_row, container)
        if predicted_plan is not None:
            _assert_plan_alignment(
                source_plan,
                predicted_plan,
                label="predicted Plan",
            )
            raw_species = predicted_plan.get("valence_species")
            if isinstance(raw_species, Sequence) and not isinstance(
                raw_species, (str, bytes)
            ):
                if not all(isinstance(value, Mapping) for value in raw_species):
                    raise ValueError("predicted Plan valence_species is malformed")
                predicted_species = _normalise_valence_species(
                    list(raw_species)
                )
                if predicted_species != list(semantic_species):
                    raise ValueError("predicted Plan valence certificate changed")
        return {
            LEGACY_DEVELOPMENT_CHECKPOINT: _soft_fields(
                container, label="predicted row"
            )
        }

    candidates: list[dict[str, str]] = []
    for label, row in (("DLM row", source_row), ("semantic row", semantic_row)):
        container = _predicted_container(row, inline_only=True)
        if container is not None:
            candidates.append(_soft_fields(container, label=label))
    if not candidates:
        raise ValueError(
            "predicted soft fields are missing; provide --predicted-soft-dir or "
            "an aligned per-row predicted_soft_fields mapping"
        )
    if any(candidate != candidates[0] for candidate in candidates[1:]):
        raise ValueError("inline frozen predicted soft fields disagree")
    return {LEGACY_DEVELOPMENT_CHECKPOINT: candidates[0]}


def _native_plan(
    source_plan: Mapping[str, Any],
    semantic_plan: Mapping[str, Any],
    species: Sequence[Mapping[str, Any]],
    soft: Mapping[str, str],
) -> dict[str, Any]:
    composition = _composition(source_plan)
    elements = [symbol for symbol, _count in composition]
    counts = [int(count) for _symbol, count in composition]
    family = str(
        semantic_plan.get("anion_framework")
        or source_plan.get("anion_framework")
        or ""
    )
    expected_charge = _expected_charge_bucket(species)
    declared_charge = str(
        semantic_plan.get("charge_bucket")
        or source_plan.get("charge_bucket")
        or expected_charge
    )
    if declared_charge == "certified_neutral":
        declared_charge = "neutral_plausible"
    plan = {
        "N": _n_value(source_plan, label="DLM Plan"),
        "elements": elements,
        "counts": counts,
        "formula": formula_from_symbol_counts(composition),
        "anion_framework": family,
        "charge_bucket": expected_charge,
        "valence_species": [dict(row) for row in species],
        **{key: str(soft[key]) for key in SOFT_FIELDS},
    }
    line = serialize_native_plan(plan)
    parsed = parse_native_plan_line(line)
    if _composition(parsed) != composition or int(parsed["N"]) != int(plan["N"]):
        raise ValueError("native Plan roundtrip changed composition/N")
    if parsed.get("charge_bucket_match") is not True:
        raise ValueError("native Plan roundtrip changed charge certificate")
    for key in SOFT_FIELDS:
        if str(parsed.get(key)) != str(plan[key]):
            raise ValueError(f"native Plan roundtrip changed {key}")
    return plan


def _line_fields(line: str) -> dict[str, str]:
    output: dict[str, str] = {}
    for chunk in str(line).split(";"):
        if "=" not in chunk:
            continue
        key, value = chunk.split("=", 1)
        output[key.strip().upper()] = value.strip()
    return output


def _assert_only_soft_line_differences(teacher: str, predicted: str) -> None:
    left = _line_fields(teacher)
    right = _line_fields(predicted)
    if set(left) != set(right):
        raise ValueError("predicted native Plan changed serialized field support")
    changed = {key for key in left if left[key] != right[key]}
    if not changed <= SOFT_LINE_FIELDS:
        raise ValueError(
            "predicted native Plan changed hard fields: "
            + ",".join(sorted(changed - SOFT_LINE_FIELDS))
        )


def _mask_selected_soft_line_fields(
    line: str,
    fields: set[str],
) -> str:
    line_keys = {
        "lattice_system": "LS",
        "spacegroup_bucket": "SG",
        "volume_per_atom_bin": "VP",
    }
    selected = {line_keys[field] for field in fields}
    output: list[str] = []
    seen: set[str] = set()
    for chunk in str(line).split(";"):
        if "=" not in chunk:
            output.append(chunk)
            continue
        key, _value = chunk.split("=", 1)
        key = key.strip().upper()
        if key in selected:
            output.append(f"{key}=<SOFT_MASK>")
            seen.add(key)
        else:
            output.append(chunk)
    if seen != selected:
        raise ValueError("predicted unknown soft-field masking support changed")
    return ";".join(output)


def _materialise_predicted_plan(
    source_plan: Mapping[str, Any],
    semantic_plan: Mapping[str, Any],
    species: Sequence[Mapping[str, Any]],
    teacher_soft: Mapping[str, str],
    predicted_soft: Mapping[str, str],
) -> tuple[dict[str, Any], str, str]:
    unknown_fields = {
        field
        for field in SOFT_FIELDS
        if str(predicted_soft[field]) in {"<UNKNOWN>", "<SOFT_MASK>"}
    }
    serializable_soft = {
        field: (
            str(teacher_soft[field])
            if field in unknown_fields
            else str(predicted_soft[field])
        )
        for field in SOFT_FIELDS
    }
    plan = _native_plan(
        source_plan,
        semantic_plan,
        species,
        serializable_soft,
    )
    line = serialize_native_plan(plan)
    prompt = build_native_body_prompt(plan)
    if unknown_fields:
        masked_line = _mask_selected_soft_line_fields(line, unknown_fields)
        if line not in prompt:
            raise RuntimeError("native prompt does not contain its serialized Plan")
        prompt = prompt.replace(line, masked_line, 1)
        line = masked_line
        plan = {
            **plan,
            **{field: "<SOFT_MASK>" for field in unknown_fields},
        }
    return plan, line, prompt


def _validate_output_payload(row: Mapping[str, Any]) -> None:
    def visit(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for raw_key, child in value.items():
                key = str(raw_key).lower()
                if any(fragment in key for fragment in FORBIDDEN_KEY_FRAGMENTS):
                    raise ValueError(f"forbidden outcome field in output payload: {path}{raw_key}")
                visit(child, f"{path}{raw_key}.")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{path}{index}.")

    visit(row, "")
    prompt = str(row.get("prompt") or "").lower()
    leaked = [value for value in FORBIDDEN_PROMPT_FRAGMENTS if value in prompt]
    if leaked:
        raise ValueError(f"forbidden field leaked into prompt: {','.join(leaked)}")


def convert_aligned_row(
    source_row: Mapping[str, Any],
    semantic_row: Mapping[str, Any],
    predicted_row: Mapping[str, Any] | None,
    *,
    split: str,
    row_index: int,
    vocabulary: Mapping[int, tuple[str, int]],
    prediction_mode: str,
    checkpoint_order: Sequence[str],
) -> list[dict[str, Any]]:
    _check_split(source_row, split, label="DLM row")
    _check_split(semantic_row, split, label="semantic row")
    source_idx = _source_row_idx(
        source_row,
        row_index,
        label="DLM row",
        allow_certificate_alias=True,
    )
    _source_row_idx(
        semantic_row,
        row_index,
        label="semantic row",
        allow_certificate_alias=False,
    )
    source_plan = _plan_from_row(source_row, label="DLM row")
    semantic_plan = _plan_from_row(semantic_row, label="semantic row")
    _assert_plan_alignment(source_plan, semantic_plan, label="semantic Plan")
    species = _valence_species(semantic_row, semantic_plan, vocabulary)
    certificate_charge_bucket = _expected_charge_bucket(species)
    semantic_declared_charge_bucket = str(
        semantic_plan.get("charge_bucket") or ""
    )
    if semantic_declared_charge_bucket == "certified_neutral":
        semantic_declared_charge_bucket = "neutral_plausible"
    plan_species = semantic_plan.get("valence_species")
    semantic_plan_valence_used_label_fallback = bool(
        isinstance(plan_species, Sequence)
        and not isinstance(plan_species, (str, bytes))
        and any(
            isinstance(value, Mapping)
            and value.get("oxidation_state") in (None, "", "unknown")
            for value in plan_species
        )
    )
    predicted_soft_by_checkpoint = _resolve_predicted_soft_fields_by_checkpoint(
        source_row,
        semantic_row,
        predicted_row,
        split=split,
        row_index=row_index,
        source_plan=source_plan,
        semantic_species=species,
        prediction_mode=prediction_mode,
        checkpoint_order=checkpoint_order,
    )
    teacher_soft = _soft_fields(semantic_plan, label="semantic teacher Plan")
    teacher_plan = _native_plan(
        source_plan, semantic_plan, species, teacher_soft
    )
    teacher_line = serialize_native_plan(teacher_plan)
    predicted_materialized: dict[str, tuple[dict[str, Any], str, str]] = {}
    for checkpoint_name in checkpoint_order:
        if checkpoint_name not in predicted_soft_by_checkpoint:
            raise ValueError(f"missing checkpoint prediction {checkpoint_name}")
        materialized = _materialise_predicted_plan(
            source_plan,
            semantic_plan,
            species,
            teacher_soft,
            predicted_soft_by_checkpoint[checkpoint_name],
        )
        _assert_only_soft_line_differences(teacher_line, materialized[1])
        predicted_materialized[str(checkpoint_name)] = materialized

    answer = source_row.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("DLM row lacks a nonempty answer/body")
    answer_sha = sha256_text(answer)
    composition = _composition(source_plan)
    chemsys = "-".join(symbol for symbol, _count in composition)
    source_weight = float(source_row.get("sample_weight", 1.0) or 1.0)
    if source_weight <= 0:
        raise ValueError("DLM row sample_weight must be positive")
    view_names = expanded_view_names(checkpoint_order)
    common: dict[str, Any] = {
        "schema": SCHEMA,
        "source_split": split,
        "source_row_idx": source_idx,
        "view_group_id": f"{split}:{source_idx}",
        "answer": answer,
        "answer_sha256": answer_sha,
        "chemsys": chemsys,
        "num_atoms": int(teacher_plan["N"]),
        "reduced_composition_identity": identity_text(
            identity_from_plan_state(source_plan)
        ),
        "sample_weight": source_weight / float(len(view_names)),
        "source_charge_bucket": str(source_plan.get("charge_bucket") or ""),
        "semantic_charge_bucket": str(semantic_plan.get("charge_bucket") or ""),
        "source_charge_bucket_matches_semantic": str(
            source_plan.get("charge_bucket") or ""
        )
        == str(semantic_plan.get("charge_bucket") or ""),
        "semantic_plan_valence_used_label_fallback": (
            semantic_plan_valence_used_label_fallback
        ),
        "semantic_charge_bucket_matches_valence_certificate": (
            semantic_declared_charge_bucket == certificate_charge_bucket
        ),
        "certificate_charge_bucket": certificate_charge_bucket,
    }
    for key in SAFE_TRAINING_KEYS:
        if key in source_row:
            common[key] = source_row[key]

    teacher_row = {
        **common,
        "view": "teacher-native",
        "prompt_schema": "C3FD_NATIVE_PLAN_V1",
        "prompt": build_native_body_prompt(teacher_plan),
        "native_plan_line": teacher_line,
        "plan_state": teacher_plan,
    }
    predicted_views: list[dict[str, Any]] = []
    for checkpoint_name in checkpoint_order:
        predicted_plan, predicted_line, predicted_prompt = predicted_materialized[
            str(checkpoint_name)
        ]
        predicted_views.append(
            {
                **common,
                "view": predicted_view_name(str(checkpoint_name)),
                "prediction_checkpoint": str(checkpoint_name),
                "prompt_schema": "C3FD_NATIVE_PLAN_V1",
                "prompt": predicted_prompt,
                "native_plan_line": predicted_line,
                "plan_state": predicted_plan,
            }
        )
    masked_plan = {
        **teacher_plan,
        **{key: "<SOFT_MASK>" for key in SOFT_FIELDS},
    }
    masked_line = mask_native_soft_fields(teacher_line)
    masked_row = {
        **common,
        "view": "soft-masked",
        "prompt_schema": "C3FD_NATIVE_PLAN_V1_SOFT_MASKED",
        "prompt": build_native_body_prompt(teacher_plan, mask_soft_fields=True),
        "native_plan_line": masked_line,
        "plan_state": masked_plan,
    }
    minimal_spec = minimal_spec_from_plan(source_plan, semantic_row)
    minimal_row = {
        **common,
        "view": "minimal-reference",
        "prompt_schema": "h1a2_ctv_minimal_spec_v1",
        "prompt": minimal_prompt(minimal_spec),
        "minimal_spec": minimal_spec,
    }
    rows = [teacher_row, *predicted_views, masked_row, minimal_row]
    if {row["answer"] for row in rows} != {answer}:
        raise RuntimeError("multi-view conversion changed the answer/body")
    if not math.isclose(
        sum(float(row["sample_weight"]) for row in rows),
        source_weight,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError("multi-view conversion changed source sample weight")
    for row in rows:
        _validate_output_payload(row)
    return rows


def _discover_splits(
    input_dir: Path,
    semantic_dir: Path,
    predicted_soft_dir: Path | None,
) -> tuple[str, ...]:
    splits = ["train", "val"]
    for split in splits:
        for label, directory in (
            ("DLM", input_dir),
            ("semantic", semantic_dir),
        ):
            if not (directory / f"{split}.jsonl").is_file():
                raise FileNotFoundError(f"{label} {split} split is missing")
        if predicted_soft_dir is not None and not (
            predicted_soft_dir / f"{split}.jsonl"
        ).is_file():
            raise FileNotFoundError(f"predicted {split} split is missing")
    return tuple(splits)


def _write_sha256sums(directory: Path, names: Sequence[str]) -> None:
    lines = [f"{sha256_file(directory / name)}  {name}" for name in names]
    (directory / "SHA256SUMS").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def build_dataset(
    *,
    input_dir: Path,
    semantic_dir: Path,
    output_dir: Path,
    predicted_soft_dir: Path | None = None,
    allow_legacy_single_prediction_development: bool = False,
    split_policy: str = "chemsys-held-out",
    allow_source_ordinal_index: bool = False,
) -> dict[str, Any]:
    input_dir = Path(input_dir)
    semantic_dir = Path(semantic_dir)
    output_dir = Path(output_dir)
    predicted_soft_dir = (
        None if predicted_soft_dir is None else Path(predicted_soft_dir)
    )
    if split_policy not in {"chemsys-held-out", "mp20-standard"}:
        raise ValueError(f"unsupported split policy {split_policy!r}")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    prediction_contract = _load_prediction_contract(
        predicted_soft_dir,
        allow_legacy_single_prediction_development=(
            allow_legacy_single_prediction_development
        ),
    )
    checkpoint_order = tuple(prediction_contract["checkpoint_order"])
    view_names = expanded_view_names(checkpoint_order)
    predicted_view_names = tuple(
        predicted_view_name(name) for name in checkpoint_order
    )
    splits = _discover_splits(input_dir, semantic_dir, predicted_soft_dir)
    vocabulary, vocabulary_sha = _species_vocabulary(semantic_dir)
    staging = output_dir.with_name(f".{output_dir.name}.preparing.{os.getpid()}")
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir(parents=True)
    try:
        split_reports: dict[str, Any] = {}
        chemsys_by_split: dict[str, set[str]] = {}
        for split in splits:
            source_path = input_dir / f"{split}.jsonl"
            semantic_path = semantic_dir / f"{split}.jsonl"
            predicted_path = (
                None
                if predicted_soft_dir is None
                else predicted_soft_dir / f"{split}.jsonl"
            )
            output_path = staging / f"{split}.jsonl"
            source_rows = list(iter_jsonl(source_path))
            semantic_by_idx = _index_rows_by_source_idx(
                iter_jsonl(semantic_path),
                label=f"semantic {split} row",
                allow_certificate_alias=False,
            )
            predicted_by_idx = (
                {}
                if predicted_path is None
                else _index_rows_by_source_idx(
                    iter_jsonl(predicted_path),
                    label=f"predicted {split} row",
                    allow_certificate_alias=False,
                )
            )
            source_count = 0
            source_semantic_charge_bucket_mismatches = 0
            semantic_plan_valence_label_fallbacks = 0
            semantic_charge_bucket_certificate_mismatches = 0
            source_indices: set[int] = set()
            view_counts: Counter[str] = Counter()
            chemsystems: set[str] = set()
            source_rows_assigned_ordinal_index = 0
            with output_path.open("x", encoding="utf-8", newline="\n") as handle:
                for source_ordinal, source_row in enumerate(source_rows):
                    if (
                        allow_source_ordinal_index
                        and "source_row_idx" not in source_row
                        and "c3fd_certificate_source_row_idx" not in source_row
                    ):
                        source_row = {
                            **source_row,
                            "source_row_idx": source_ordinal,
                        }
                        source_rows_assigned_ordinal_index += 1
                    source_idx = _source_row_idx(
                        source_row,
                        None,
                        label=f"DLM {split} row",
                        allow_certificate_alias=True,
                    )
                    if source_idx in source_indices:
                        raise ValueError(
                            f"DLM {split} rows duplicate source_row_idx {source_idx}"
                        )
                    source_indices.add(source_idx)
                    semantic_row = semantic_by_idx.get(source_idx)
                    if semantic_row is None:
                        raise ValueError(
                            f"semantic {split} is missing source_row_idx {source_idx}"
                        )
                    predicted_row = (
                        None
                        if predicted_path is None
                        else predicted_by_idx.get(source_idx)
                    )
                    if predicted_path is not None and predicted_row is None:
                        raise ValueError(
                            f"predicted {split} is missing source_row_idx {source_idx}"
                        )
                    converted = convert_aligned_row(
                        source_row,
                        semantic_row,
                        predicted_row,
                        split=split,
                        row_index=source_idx,
                        vocabulary=vocabulary,
                        prediction_mode=str(prediction_contract["mode"]),
                        checkpoint_order=checkpoint_order,
                    )
                    source_count += 1
                    source_semantic_charge_bucket_mismatches += int(
                        converted[0]["source_charge_bucket_matches_semantic"] is False
                    )
                    semantic_plan_valence_label_fallbacks += int(
                        converted[0]["semantic_plan_valence_used_label_fallback"]
                        is True
                    )
                    semantic_charge_bucket_certificate_mismatches += int(
                        converted[0][
                            "semantic_charge_bucket_matches_valence_certificate"
                        ]
                        is False
                    )
                    for row in converted:
                        view_counts[str(row["view"])] += 1
                        chemsystems.add(str(row["chemsys"]))
                        handle.write(
                            json.dumps(
                                row,
                                ensure_ascii=False,
                                sort_keys=True,
                                separators=(",", ":"),
                            )
                            + "\n"
                        )
            if source_count == 0:
                raise ValueError(f"{split} split is empty")
            if view_counts != Counter({view: source_count for view in view_names}):
                raise RuntimeError(f"{split} view expansion is incomplete")
            chemsys_by_split[split] = chemsystems
            split_reports[split] = {
                "source_rows": source_count,
                "output_rows": source_count * len(view_names),
                "views": dict(sorted(view_counts.items())),
                "chemsys": len(chemsystems),
                "source_sha256": sha256_file(source_path),
                "semantic_sha256": sha256_file(semantic_path),
                "predicted_sha256": (
                    None if predicted_path is None else sha256_file(predicted_path)
                ),
                "output_sha256": sha256_file(output_path),
                "source_semantic_charge_bucket_mismatches": (
                    source_semantic_charge_bucket_mismatches
                ),
                "semantic_plan_valence_label_fallbacks": (
                    semantic_plan_valence_label_fallbacks
                ),
                "semantic_charge_bucket_certificate_mismatches": (
                    semantic_charge_bucket_certificate_mismatches
                ),
                "source_rows_assigned_ordinal_index": (
                    source_rows_assigned_ordinal_index
                ),
                "semantic_rows": len(semantic_by_idx),
                "predicted_rows": (
                    None if predicted_path is None else len(predicted_by_idx)
                ),
                "unused_semantic_rows": len(set(semantic_by_idx) - source_indices),
                "unused_predicted_rows": (
                    None
                    if predicted_path is None
                    else len(set(predicted_by_idx) - source_indices)
                ),
            }

        overlap: dict[str, int] = {}
        for left_index, left in enumerate(splits):
            for right in splits[left_index + 1 :]:
                key = f"{left}__{right}"
                overlap[key] = len(chemsys_by_split[left] & chemsys_by_split[right])
        if split_policy == "chemsys-held-out" and any(overlap.values()):
            raise ValueError(f"chemsys-held-out split leakage detected: {overlap}")

        static_assets: dict[str, Any] = {}
        vocab_tokens = input_dir / "vocab_tokens.txt"
        if vocab_tokens.is_file():
            target = staging / vocab_tokens.name
            shutil.copyfile(vocab_tokens, target)
            if sha256_file(target) != sha256_file(vocab_tokens):
                raise RuntimeError("vocab_tokens.txt changed while copying")
            static_assets[vocab_tokens.name] = {
                "source_sha256": sha256_file(vocab_tokens),
                "output_sha256": sha256_file(target),
            }

        gate = {
            "outcome_blind_curated_payload": True,
            "source_row_alignment_exact": True,
            "composition_N_valence_exact": True,
            "predicted_rows_complete_and_aligned": True,
            "predicted_native_changes_only_soft_fields": True,
            "checkpoint_support_and_order_identical_across_splits_rows": True,
            "all_frozen_checkpoints_preserved_without_selection": True,
            "all_views_share_answer_body": True,
            "source_sample_weight_preserved_across_views": True,
            "split_policy_honored": (
                split_policy == "mp20-standard" or not any(overlap.values())
            ),
            "chemsys_overlap_disclosed": True,
            "legacy_rich_fields_absent": True,
            "semantic_valence_certificate_authoritative_for_charge": True,
            "unknown_plan_valence_replaced_only_by_frozen_species_labels": True,
            "frozen_valence_witness_authoritative_over_semantic_charge_bucket": True,
        }
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "input_dir": str(input_dir.resolve()),
            "semantic_dir": str(semantic_dir.resolve()),
            "predicted_soft_dir": (
                None
                if predicted_soft_dir is None
                else str(predicted_soft_dir.resolve())
            ),
            "output_dir": str(output_dir.resolve()),
            "split_policy": split_policy,
            "chemsys_held_out": not any(overlap.values()),
            "source_index_policy": (
                "file_ordinal_if_absent"
                if allow_source_ordinal_index
                else "explicit_source_row_idx_required"
            ),
            "views": list(view_names),
            "predicted_view_names": list(predicted_view_names),
            "prediction_mode": str(prediction_contract["mode"]),
            "prediction_checkpoint_order": list(checkpoint_order),
            "prediction_selection": str(prediction_contract["selection"]),
            "formal_prediction_manifest_sha256": prediction_contract[
                "manifest_sha256"
            ],
            "legacy_single_prediction_development": (
                prediction_contract["mode"] == "legacy-single-development"
            ),
            "unknown_prediction_policy": "per-field-soft-mask",
            "splits": split_reports,
            "semantic_vocabulary_sha256": vocabulary_sha,
            "static_assets": static_assets,
            "chemsys_overlap": overlap,
            "gate": gate,
        }
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        hash_names = [f"{split}.jsonl" for split in splits]
        hash_names.extend(sorted(static_assets))
        hash_names.append("manifest.json")
        _write_sha256sums(staging, hash_names)
        (staging / "_SUCCESS").touch()
        if output_dir.exists():
            raise FileExistsError(output_dir)
        staging.rename(output_dir)
        return manifest
    except Exception:
        with suppress(FileNotFoundError):
            shutil.rmtree(staging)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--semantic-dir", type=Path, required=True)
    parser.add_argument("--predicted-soft-dir", type=Path)
    parser.add_argument(
        "--allow-legacy-single-prediction-development",
        action="store_true",
        help="allow one legacy prediction view for tests/development only",
    )
    parser.add_argument(
        "--split-policy",
        choices=("chemsys-held-out", "mp20-standard"),
        default="chemsys-held-out",
        help=(
            "validation split contract; mp20-standard preserves and discloses "
            "the official material-level split even when chemsys overlap"
        ),
    )
    parser.add_argument(
        "--allow-source-ordinal-index",
        action="store_true",
        help=(
            "assign source_row_idx from immutable file ordinal only when the "
            "original source row has no explicit index"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = build_dataset(
        input_dir=args.input_dir,
        semantic_dir=args.semantic_dir,
        predicted_soft_dir=args.predicted_soft_dir,
        output_dir=args.output_dir,
        allow_legacy_single_prediction_development=(
            args.allow_legacy_single_prediction_development
        ),
        split_policy=args.split_policy,
        allow_source_ordinal_index=args.allow_source_ordinal_index,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
