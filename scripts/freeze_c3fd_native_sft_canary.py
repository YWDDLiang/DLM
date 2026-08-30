#!/usr/bin/env python3
"""Freeze an outcome-blind MP20 train/validation canary for native DLM SFT."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crystal_dlm.c3fd_native_plan import (  # noqa: E402
    SOFT_FIELD_KEYS,
    build_native_inference_prompt,
    native_plan_from_parts,
)


SCHEMA = "c3fd_native_sft_canary_cohort_v1"
CHECKPOINTS = ("seed17", "seed18")
SPLITS = ("train", "val")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number} is not an object")
            rows.append(value)
    return rows


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True))
        handle.write("\n")


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(dict(row)))
            handle.write("\n")


def prediction_values(
    row: Mapping[str, Any], checkpoint: str
) -> dict[str, str]:
    by_checkpoint = row.get("predictions_by_checkpoint")
    if not isinstance(by_checkpoint, Mapping) or tuple(by_checkpoint) != CHECKPOINTS:
        raise ValueError("predicted checkpoint support/order changed")
    payload = by_checkpoint.get(checkpoint)
    if not isinstance(payload, Mapping) or tuple(payload) != SOFT_FIELD_KEYS:
        raise ValueError(f"{checkpoint} soft-field support/order changed")
    output: dict[str, str] = {}
    for field in SOFT_FIELD_KEYS:
        item = payload[field]
        if not isinstance(item, Mapping) or set(item) != {"prediction", "confidence"}:
            raise ValueError(f"{checkpoint} {field} payload changed")
        value = str(item.get("prediction") or "").strip()
        confidence = item.get("confidence")
        if not value or isinstance(confidence, bool) or not isinstance(
            confidence, (int, float)
        ):
            raise ValueError(f"{checkpoint} {field} prediction is invalid")
        if not 0.0 <= float(confidence) <= 1.0:
            raise ValueError(f"{checkpoint} {field} confidence is invalid")
        output[field] = value
    return output


def aligned_candidates(
    teacher_rows: Sequence[Mapping[str, Any]],
    predicted_rows: Sequence[Mapping[str, Any]],
    *,
    split: str,
    freeze_seed: int,
) -> list[dict[str, Any]]:
    predicted_by_index: dict[int, Mapping[str, Any]] = {}
    for row in predicted_rows:
        index = int(row["source_row_idx"])
        if index in predicted_by_index:
            raise ValueError(f"{split} predictions duplicate source_row_idx {index}")
        predicted_by_index[index] = row
    candidates: list[dict[str, Any]] = []
    for teacher in teacher_rows:
        index = int(teacher["source_row_idx"])
        predicted = predicted_by_index.get(index)
        if predicted is None:
            raise ValueError(f"{split} predictions lack source_row_idx {index}")
        if str(teacher.get("source_split")) != split:
            raise ValueError(f"teacher split changed for {split}:{index}")
        if str(predicted.get("split")) != split:
            raise ValueError(f"prediction split changed for {split}:{index}")
        plan = teacher.get("plan_state")
        if not isinstance(plan, Mapping):
            raise TypeError(f"teacher plan_state is invalid for {split}:{index}")
        expected_identity = str(teacher.get("reduced_composition_identity") or "")
        if not expected_identity:
            raise ValueError(f"teacher composition identity is empty for {split}:{index}")
        for field in ("N", "elements", "counts"):
            if predicted.get(field) != plan.get(field):
                raise ValueError(f"prediction {field} differs for {split}:{index}")
        soft = {
            checkpoint: prediction_values(predicted, checkpoint)
            for checkpoint in CHECKPOINTS
        }
        rank = hashlib.sha256(
            f"{freeze_seed}\t{split}\t{index}\t{expected_identity}".encode("utf-8")
        ).hexdigest()
        candidates.append(
            {
                "rank": rank,
                "source_split": split,
                "source_row_idx": index,
                "reduced_composition_identity": expected_identity,
                "teacher_plan": dict(plan),
                "predicted_soft": soft,
            }
        )
    return sorted(candidates, key=lambda row: (row["rank"], row["source_row_idx"]))


def select_rows(
    candidates_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    per_split: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    used_identities: set[str] = set()
    for split in SPLITS:
        split_rows: list[dict[str, Any]] = []
        for candidate in candidates_by_split[split]:
            identity = str(candidate["reduced_composition_identity"])
            if identity in used_identities:
                continue
            split_rows.append(dict(candidate))
            used_identities.add(identity)
            if len(split_rows) == per_split:
                break
        if len(split_rows) != per_split:
            raise ValueError(
                f"{split} has only {len(split_rows)} unique eligible compositions"
            )
        selected.extend(split_rows)
    return selected


def freeze(
    *,
    teacher_dir: Path,
    predicted_dir: Path,
    output_dir: Path,
    per_split: int,
    freeze_seed: int,
) -> dict[str, Any]:
    if per_split <= 0:
        raise ValueError("per_split must be positive")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    candidates: dict[str, list[dict[str, Any]]] = {}
    inputs: dict[str, str] = {}
    for split in SPLITS:
        teacher_path = teacher_dir / f"{split}.jsonl"
        predicted_path = predicted_dir / f"{split}.jsonl"
        inputs[f"teacher_{split}"] = sha256_file(teacher_path)
        inputs[f"predicted_{split}"] = sha256_file(predicted_path)
        candidates[split] = aligned_candidates(
            read_jsonl(teacher_path),
            read_jsonl(predicted_path),
            split=split,
            freeze_seed=freeze_seed,
        )
    selected = select_rows(candidates, per_split=per_split)
    output_dir.mkdir(parents=True, exist_ok=False)
    ledger: list[dict[str, Any]] = []
    plans: dict[str, list[dict[str, Any]]] = {name: [] for name in CHECKPOINTS}
    for sample_idx, selected_row in enumerate(selected):
        split = str(selected_row["source_split"])
        source_idx = int(selected_row["source_row_idx"])
        teacher_plan = selected_row["teacher_plan"]
        identity = str(selected_row["reduced_composition_identity"])
        ledger.append(
            {
                "sample_idx": sample_idx,
                "source_split": split,
                "source_row_idx": source_idx,
                "reduced_composition_identity": identity,
                "selection_rank_sha256": selected_row["rank"],
                "N": int(teacher_plan["N"]),
                "elements": [str(value) for value in teacher_plan["elements"]],
                "counts": [int(value) for value in teacher_plan["counts"]],
            }
        )
        for checkpoint in CHECKPOINTS:
            predicted_soft = selected_row["predicted_soft"][checkpoint]
            plan_state = native_plan_from_parts(teacher_plan, predicted_soft)
            plans[checkpoint].append(
                {
                    "schema": "c3fd_native_sft_canary_plan_row_v1",
                    "sample_idx": sample_idx,
                    "source_sample_idx": source_idx + (0 if split == "train" else 1_000_000),
                    "source_split": split,
                    "source_row_idx": source_idx,
                    "prediction_checkpoint": checkpoint,
                    "reduced_composition_identity": identity,
                    "plan_state": plan_state,
                    "prompt": build_native_inference_prompt(teacher_plan, predicted_soft),
                }
            )
    write_jsonl(output_dir / "ledger.jsonl", ledger)
    for checkpoint in CHECKPOINTS:
        write_jsonl(output_dir / f"planner_{checkpoint}.jsonl", plans[checkpoint])
    files = ["ledger.jsonl", *[f"planner_{name}.jsonl" for name in CHECKPOINTS]]
    hashes = {name: sha256_file(output_dir / name) for name in files}
    manifest = {
        "schema": SCHEMA,
        "freeze_seed": freeze_seed,
        "per_split": per_split,
        "requested": len(selected),
        "split_counts": {
            split: sum(row["source_split"] == split for row in ledger)
            for split in SPLITS
        },
        "unique_exact_compositions": len(
            {row["reduced_composition_identity"] for row in ledger}
        ),
        "planner_checkpoints": list(CHECKPOINTS),
        "checkpoint_selection": "none",
        "policy_or_test_outcomes_read": False,
        "teacher_answers_copied_to_cohort": False,
        "renderer": "build_native_inference_prompt",
        "only_checkpoint_dependent_fields": list(SOFT_FIELD_KEYS),
        "input_sha256": inputs,
        "output_sha256": hashes,
        "gates": {
            "requested_complete": len(selected) == 2 * per_split,
            "split_balance": all(
                sum(row["source_split"] == split for row in ledger) == per_split
                for split in SPLITS
            ),
            "exact_composition_unique": len(
                {row["reduced_composition_identity"] for row in ledger}
            ) == len(selected),
            "sample_idx_contiguous": [row["sample_idx"] for row in ledger]
            == list(range(len(selected))),
            "both_planner_checkpoints_preserved": all(
                len(plans[name]) == len(selected) for name in CHECKPOINTS
            ),
            "outcome_blind": True,
        },
    }
    if not all(manifest["gates"].values()):
        raise RuntimeError("canary cohort gates failed")
    write_json(output_dir / "manifest.json", manifest)
    hashes["manifest.json"] = sha256_file(output_dir / "manifest.json")
    with (output_dir / "SHA256SUMS").open("x", encoding="utf-8", newline="\n") as handle:
        for name in sorted(hashes):
            handle.write(f"{hashes[name]}  {name}\n")
    (output_dir / "_SUCCESS").touch()
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-dir", type=Path, required=True)
    parser.add_argument("--predicted-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-split", type=int, default=128)
    parser.add_argument("--freeze-seed", type=int, default=20260831)
    args = parser.parse_args()
    manifest = freeze(
        teacher_dir=args.teacher_dir.resolve(),
        predicted_dir=args.predicted_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        per_split=args.per_split,
        freeze_seed=args.freeze_seed,
    )
    print(json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
