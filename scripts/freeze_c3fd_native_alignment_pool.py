#!/usr/bin/env python3
"""Freeze a fresh MP20-train, same-prompt K-way native alignment pool."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.freeze_c3fd_native_sft_canary import (  # noqa: E402
    CHECKPOINTS,
    aligned_candidates,
    canonical_json,
    read_jsonl,
    sha256_file,
    write_json,
    write_jsonl,
)
from crystal_dlm.c3fd_native_plan import (  # noqa: E402
    build_native_inference_prompt,
    native_plan_from_parts,
)


SCHEMA = "c3fd_native_alignment_pool_cohort_v1"
IDENTITY_KEYS = (
    "reduced_composition_identity",
    "composition_identity",
    "reduced_identity",
)


def identities_from_jsonl(path: Path) -> set[str]:
    identities: set[str] = set()
    for row in read_jsonl(path):
        for key in IDENTITY_KEYS:
            value = str(row.get(key) or "").strip()
            if value:
                identities.add(value)
                break
    return identities


def exclusion_ledger(root: Path) -> tuple[set[str], list[dict[str, Any]]]:
    identities: set[str] = set()
    files: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*.jsonl")):
        found = identities_from_jsonl(path)
        if not found:
            continue
        identities.update(found)
        files.append(
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "exact_identities": len(found),
            }
        )
    if not files:
        raise ValueError("exclude cohort root contains no exact identities")
    return identities, files


def select_compositions(
    candidates: Sequence[Mapping[str, Any]],
    *,
    excluded: set[str],
    compositions: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    used: set[str] = set()
    for candidate in candidates:
        identity = str(candidate["reduced_composition_identity"])
        if identity in excluded or identity in used:
            continue
        selected.append(dict(candidate))
        used.add(identity)
        if len(selected) == compositions:
            break
    if len(selected) != compositions:
        raise ValueError(
            f"only {len(selected)} eligible train compositions after exclusions"
        )
    return selected


def freeze(
    *,
    teacher_dir: Path,
    predicted_dir: Path,
    exclude_cohort_root: Path,
    output_dir: Path,
    compositions: int,
    candidates_per_group: int,
    selection_seed: int,
) -> dict[str, Any]:
    if compositions <= 0 or candidates_per_group < 2:
        raise ValueError("pool requires positive compositions and K >= 2")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    excluded, exclusion_files = exclusion_ledger(exclude_cohort_root)
    teacher_path = teacher_dir / "train.jsonl"
    predicted_path = predicted_dir / "train.jsonl"
    candidates = aligned_candidates(
        read_jsonl(teacher_path),
        read_jsonl(predicted_path),
        split="train",
        freeze_seed=selection_seed,
    )
    selected = select_compositions(
        candidates,
        excluded=excluded,
        compositions=compositions,
    )
    rows: list[dict[str, Any]] = []
    groups: list[dict[str, Any]] = []
    for composition_idx, selected_row in enumerate(selected):
        teacher_plan = selected_row["teacher_plan"]
        identity = str(selected_row["reduced_composition_identity"])
        source_idx = int(selected_row["source_row_idx"])
        for checkpoint_idx, checkpoint in enumerate(CHECKPOINTS):
            group_ordinal = composition_idx * len(CHECKPOINTS) + checkpoint_idx
            group_id = f"train:{source_idx}:{checkpoint}"
            predicted_soft = selected_row["predicted_soft"][checkpoint]
            plan_state = native_plan_from_parts(teacher_plan, predicted_soft)
            prompt = build_native_inference_prompt(teacher_plan, predicted_soft)
            sample_indices: list[int] = []
            for candidate_idx in range(candidates_per_group):
                sample_idx = group_ordinal * candidates_per_group + candidate_idx
                sample_indices.append(sample_idx)
                rows.append(
                    {
                        "schema": "c3fd_native_alignment_pool_plan_row_v1",
                        "sample_idx": sample_idx,
                        "source_sample_idx": source_idx,
                        "source_split": "train",
                        "source_row_idx": source_idx,
                        "composition_ordinal": composition_idx,
                        "group_ordinal": group_ordinal,
                        "group_id": group_id,
                        "candidate_idx": candidate_idx,
                        "prediction_checkpoint": checkpoint,
                        "reduced_composition_identity": identity,
                        "plan_state": plan_state,
                        "prompt": prompt,
                    }
                )
            groups.append(
                {
                    "schema": "c3fd_native_alignment_pool_group_v1",
                    "group_ordinal": group_ordinal,
                    "group_id": group_id,
                    "source_row_idx": source_idx,
                    "composition_ordinal": composition_idx,
                    "prediction_checkpoint": checkpoint,
                    "reduced_composition_identity": identity,
                    "K": candidates_per_group,
                    "sample_indices": sample_indices,
                    "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                }
            )
    expected_rows = compositions * len(CHECKPOINTS) * candidates_per_group
    if len(rows) != expected_rows or [row["sample_idx"] for row in rows] != list(
        range(expected_rows)
    ):
        raise RuntimeError("pool row accounting changed")
    for group in groups:
        group_rows = [row for row in rows if row["group_id"] == group["group_id"]]
        if len(group_rows) != candidates_per_group:
            raise RuntimeError("pool group K changed")
        if len({row["prompt"] for row in group_rows}) != 1:
            raise RuntimeError("pool group prompt changed across candidates")
    output_dir.mkdir(parents=True, exist_ok=False)
    write_jsonl(output_dir / "pool_plans.jsonl", rows)
    write_jsonl(output_dir / "groups.jsonl", groups)
    write_json(output_dir / "exclusion_inputs.json", exclusion_files)
    output_hashes = {
        name: sha256_file(output_dir / name)
        for name in ("pool_plans.jsonl", "groups.jsonl", "exclusion_inputs.json")
    }
    manifest = {
        "schema": SCHEMA,
        "selection_seed": selection_seed,
        "source_split": "MP20_train_only",
        "compositions": compositions,
        "planner_checkpoints": list(CHECKPOINTS),
        "planner_checkpoint_selection": "none",
        "groups": len(groups),
        "candidates_per_group": candidates_per_group,
        "rows": len(rows),
        "excluded_exact_identities": len(excluded),
        "exclusion_files": len(exclusion_files),
        "policy_or_test_outcomes_read": False,
        "teacher_answers_copied": False,
        "same_prompt_within_group": True,
        "input_sha256": {
            "teacher_train": sha256_file(teacher_path),
            "predicted_train": sha256_file(predicted_path),
        },
        "output_sha256": output_hashes,
        "gates": {
            "train_only": all(row["source_split"] == "train" for row in rows),
            "exact_composition_exclusion": all(
                row["reduced_composition_identity"] not in excluded for row in rows
            ),
            "unique_compositions": len(
                {row["reduced_composition_identity"] for row in rows}
            )
            == compositions,
            "both_planners_preserved": {group["prediction_checkpoint"] for group in groups}
            == set(CHECKPOINTS),
            "fixed_k": all(group["K"] == candidates_per_group for group in groups),
            "same_prompt_within_group": True,
            "fixed256": len(rows) == 256,
            "outcome_blind": True,
        },
    }
    if not all(manifest["gates"].values()):
        raise RuntimeError("alignment pool freeze gates failed")
    write_json(output_dir / "manifest.json", manifest)
    output_hashes["manifest.json"] = sha256_file(output_dir / "manifest.json")
    with (output_dir / "SHA256SUMS").open("x", encoding="utf-8", newline="\n") as handle:
        for name in sorted(output_hashes):
            handle.write(f"{output_hashes[name]}  {name}\n")
    (output_dir / "_SUCCESS").touch()
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-dir", type=Path, required=True)
    parser.add_argument("--predicted-dir", type=Path, required=True)
    parser.add_argument("--exclude-cohort-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--compositions", type=int, default=32)
    parser.add_argument("--candidates-per-group", type=int, default=4)
    parser.add_argument("--selection-seed", type=int, default=20260901)
    args = parser.parse_args()
    manifest = freeze(
        teacher_dir=args.teacher_dir.resolve(),
        predicted_dir=args.predicted_dir.resolve(),
        exclude_cohort_root=args.exclude_cohort_root.resolve(),
        output_dir=args.output_dir.resolve(),
        compositions=args.compositions,
        candidates_per_group=args.candidates_per_group,
        selection_seed=args.selection_seed,
    )
    print(json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
