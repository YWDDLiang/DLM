#!/usr/bin/env python3
"""Build outcome-blind C3FD-native multi-view crystal-DLM SFT data.

Each aligned source row is expanded into four equally weighted views that retain
the exact same dynamic crystal body:

* ``teacher-native`` uses train-only structural fields from the MP20 row;
* ``predicted-native`` uses frozen C3FD soft-field predictions;
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


SCHEMA = "c3fd_native_sft_row_v1"
MANIFEST_SCHEMA = "c3fd_native_sft_manifest_v1"
VIEWS = (
    "teacher-native",
    "predicted-native",
    "soft-masked",
    "minimal-reference",
)
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


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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
    expected: int,
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
    if actual != int(expected):
        raise ValueError(
            f"{label} source_row_idx changed: {actual} != {int(expected)}"
        )
    return actual


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
    for key in ("anion_framework", "charge_bucket"):
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
        species = _normalise_valence_species(values)
    else:
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


def _resolve_predicted_soft_fields(
    source_row: Mapping[str, Any],
    semantic_row: Mapping[str, Any],
    predicted_row: Mapping[str, Any] | None,
    *,
    split: str,
    row_index: int,
    source_plan: Mapping[str, Any],
    semantic_species: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
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
        return _soft_fields(container, label="predicted row")

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
    return candidates[0]


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
    if declared_charge != expected_charge:
        raise ValueError(
            "semantic charge bucket disagrees with frozen valence certificate"
        )
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
    predicted_soft = _resolve_predicted_soft_fields(
        source_row,
        semantic_row,
        predicted_row,
        split=split,
        row_index=row_index,
        source_plan=source_plan,
        semantic_species=species,
    )
    teacher_soft = _soft_fields(semantic_plan, label="semantic teacher Plan")
    teacher_plan = _native_plan(
        source_plan, semantic_plan, species, teacher_soft
    )
    predicted_plan = _native_plan(
        source_plan, semantic_plan, species, predicted_soft
    )
    teacher_line = serialize_native_plan(teacher_plan)
    predicted_line = serialize_native_plan(predicted_plan)
    _assert_only_soft_line_differences(teacher_line, predicted_line)

    answer = source_row.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("DLM row lacks a nonempty answer/body")
    answer_sha = sha256_text(answer)
    composition = _composition(source_plan)
    chemsys = "-".join(symbol for symbol, _count in composition)
    source_weight = float(source_row.get("sample_weight", 1.0) or 1.0)
    if source_weight <= 0:
        raise ValueError("DLM row sample_weight must be positive")
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
        "sample_weight": source_weight / float(len(VIEWS)),
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
    predicted_view = {
        **common,
        "view": "predicted-native",
        "prompt_schema": "C3FD_NATIVE_PLAN_V1",
        "prompt": build_native_body_prompt(predicted_plan),
        "native_plan_line": predicted_line,
        "plan_state": predicted_plan,
    }
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
    rows = [teacher_row, predicted_view, masked_row, minimal_row]
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
    test_presence = [
        (input_dir / "test.jsonl").is_file(),
        (semantic_dir / "test.jsonl").is_file(),
    ]
    if predicted_soft_dir is not None:
        test_presence.append((predicted_soft_dir / "test.jsonl").is_file())
    if any(test_presence):
        if not all(test_presence):
            raise ValueError("test split is not present in every aligned input")
        splits.append("test")
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
) -> dict[str, Any]:
    input_dir = Path(input_dir)
    semantic_dir = Path(semantic_dir)
    output_dir = Path(output_dir)
    predicted_soft_dir = (
        None if predicted_soft_dir is None else Path(predicted_soft_dir)
    )
    if output_dir.exists():
        raise FileExistsError(output_dir)
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
            source_rows = iter_jsonl(source_path)
            semantic_rows = iter_jsonl(semantic_path)
            predicted_rows: Iterable[dict[str, Any] | object]
            if predicted_path is None:
                predicted_rows = ()
                iterator = (
                    (source, semantic, None)
                    for source, semantic in zip_longest(
                        source_rows,
                        semantic_rows,
                        fillvalue=_MISSING,
                    )
                )
            else:
                predicted_rows = iter_jsonl(predicted_path)
                iterator = zip_longest(
                    source_rows,
                    semantic_rows,
                    predicted_rows,
                    fillvalue=_MISSING,
                )
            source_count = 0
            view_counts: Counter[str] = Counter()
            chemsystems: set[str] = set()
            with output_path.open("x", encoding="utf-8", newline="\n") as handle:
                for row_index, values in enumerate(iterator):
                    source_row, semantic_row, predicted_row = values
                    if source_row is _MISSING or semantic_row is _MISSING:
                        raise ValueError(
                            f"DLM/semantic split length changed for {split}"
                        )
                    if predicted_path is not None and predicted_row is _MISSING:
                        raise ValueError(
                            f"predicted split length changed for {split}"
                        )
                    if not isinstance(source_row, Mapping) or not isinstance(
                        semantic_row, Mapping
                    ):
                        raise TypeError("aligned DLM/semantic row is not an object")
                    if predicted_row is not None and not isinstance(
                        predicted_row, Mapping
                    ):
                        raise TypeError("aligned predicted row is not an object")
                    converted = convert_aligned_row(
                        source_row,
                        semantic_row,
                        predicted_row,
                        split=split,
                        row_index=row_index,
                        vocabulary=vocabulary,
                    )
                    source_count += 1
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
            if view_counts != Counter({view: source_count for view in VIEWS}):
                raise RuntimeError(f"{split} view expansion is incomplete")
            chemsys_by_split[split] = chemsystems
            split_reports[split] = {
                "source_rows": source_count,
                "output_rows": source_count * len(VIEWS),
                "views": dict(sorted(view_counts.items())),
                "chemsys": len(chemsystems),
                "source_sha256": sha256_file(source_path),
                "semantic_sha256": sha256_file(semantic_path),
                "predicted_sha256": (
                    None if predicted_path is None else sha256_file(predicted_path)
                ),
                "output_sha256": sha256_file(output_path),
            }

        overlap: dict[str, int] = {}
        for left_index, left in enumerate(splits):
            for right in splits[left_index + 1 :]:
                key = f"{left}__{right}"
                overlap[key] = len(chemsys_by_split[left] & chemsys_by_split[right])
        if any(overlap.values()):
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
            "all_views_share_answer_body": True,
            "source_sample_weight_preserved_across_views": True,
            "chemsys_held_out_split_preserved": not any(overlap.values()),
            "legacy_rich_fields_absent": True,
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
            "views": list(VIEWS),
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
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = build_dataset(
        input_dir=args.input_dir,
        semantic_dir=args.semantic_dir,
        predicted_soft_dir=args.predicted_soft_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
