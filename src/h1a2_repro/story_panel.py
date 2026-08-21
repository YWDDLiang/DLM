"""Build preregistered no-training story-panel ledgers."""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

from crystal_dlm.h1_formula_only_body import build_formula_only_body_prompt
from crystal_dlm.r5_plan_state import build_body_prompt


SHUFFLED_FIELDS = ("lattice_system", "spacegroup_bucket", "volume_per_atom_bin")


def load_plan_rows(path: Path, source: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_plan_ids: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            plan = payload.get("plan_state") or payload.get("r5_plan_state") or payload.get("parsed_plan")
            if not isinstance(plan, Mapping):
                raise ValueError(f"{path}:{line_number}: no plan_state/r5_plan_state/parsed_plan")
            plan = dict(plan)
            validate_plan_for_panel(plan, path=path, line_number=line_number)
            plan_id = str(
                payload.get("plan_id")
                or payload.get("sample_idx")
                or payload.get("raw_ordinal")
                or f"line-{line_number:06d}"
            )
            scoped_plan_id = f"{source}:{plan_id}"
            if scoped_plan_id in seen_plan_ids:
                raise ValueError(f"{path}:{line_number}: duplicate Plan id {scoped_plan_id!r}")
            seen_plan_ids.add(scoped_plan_id)
            rows.append({"source": source, "plan_id": scoped_plan_id, "plan_state": plan, "source_row": dict(payload)})
    if not rows:
        raise ValueError(f"{path}: no Plan rows")
    return rows


def validate_plan_for_panel(plan: Mapping[str, Any], *, path: Path | None = None, line_number: int | None = None) -> None:
    prefix = "Plan" if path is None else f"{path}:{line_number}"
    required = ("N", "elements", "counts", "formula", "anion_framework") + SHUFFLED_FIELDS
    missing = [key for key in required if key not in plan]
    if missing:
        raise ValueError(f"{prefix}: missing rich Plan fields {missing}")
    num_atoms = int(plan["N"])
    counts = [int(value) for value in plan["counts"]]
    if sum(counts) != num_atoms:
        raise ValueError(f"{prefix}: counts sum {sum(counts)} != N {num_atoms}")
    if len(plan["elements"]) != len(counts):
        raise ValueError(f"{prefix}: elements/counts length mismatch")


def plan_stratum(plan: Mapping[str, Any]) -> tuple[int, int, str]:
    return int(plan["N"]), len(plan["elements"]), str(plan["anion_framework"])


def coarse_signature(plan: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(str(plan[field]) for field in SHUFFLED_FIELDS)


def stable_seed(root_seed: int, *parts: str) -> int:
    digest = hashlib.sha256((str(root_seed) + "|" + "|".join(parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def select_matched_pairs(
    learned: Sequence[dict[str, Any]],
    gold: Sequence[dict[str, Any]],
    *,
    num_pairs: int,
    seed: int,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    learned_groups: dict[tuple[int, int, str], list[dict[str, Any]]] = defaultdict(list)
    gold_groups: dict[tuple[int, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in learned:
        learned_groups[plan_stratum(row["plan_state"])].append(row)
    for row in gold:
        gold_groups[plan_stratum(row["plan_state"])].append(row)
    common = sorted(set(learned_groups) & set(gold_groups))
    rng = random.Random(int(seed))
    for key in common:
        learned_groups[key].sort(key=lambda row: row["plan_id"])
        gold_groups[key].sort(key=lambda row: row["plan_id"])
        rng.shuffle(learned_groups[key])
        rng.shuffle(gold_groups[key])

    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    offsets = {key: 0 for key in common}
    while len(pairs) < int(num_pairs):
        progressed = False
        for key in common:
            index = offsets[key]
            if index < min(len(learned_groups[key]), len(gold_groups[key])):
                pairs.append((learned_groups[key][index], gold_groups[key][index]))
                offsets[key] += 1
                progressed = True
                if len(pairs) == int(num_pairs):
                    break
        if not progressed:
            break
    if len(pairs) != int(num_pairs):
        raise ValueError(
            f"only {len(pairs)} exact N/arity/anion matched pairs are available; requested {num_pairs}"
        )
    return pairs


def shuffled_plans(selected_rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    groups: dict[tuple[int, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in selected_rows:
        groups[plan_stratum(row["plan_state"])].append(row)
    result: dict[str, dict[str, Any]] = {}
    for key, group in groups.items():
        ordered = sorted(group, key=lambda row: row["plan_id"])
        if len(ordered) < 2:
            raise ValueError(f"stratum {key} has no donor for coarse-field shuffling")
        for index, row in enumerate(ordered):
            candidates = ordered[index + 1 :] + ordered[:index]
            donor = next(
                (
                    candidate
                    for candidate in candidates
                    if coarse_signature(candidate["plan_state"]) != coarse_signature(row["plan_state"])
                ),
                None,
            )
            if donor is None:
                raise ValueError(
                    f"stratum {key} has no donor with a different lattice/SG/volume tuple for {row['plan_id']}"
                )
            shuffled = deepcopy(row["plan_state"])
            for field in SHUFFLED_FIELDS:
                shuffled[field] = donor["plan_state"][field]
            result[row["plan_id"]] = {
                "plan_state": shuffled,
                "donor_plan_id": donor["plan_id"],
            }
    return result


def build_task(
    row: Mapping[str, Any],
    *,
    pair_id: str,
    arm: str,
    replicate: int,
    sample_idx: int,
    root_seed: int,
    shuffled: Mapping[str, Any] | None = None,
    shuffle_donor_plan_id: str | None = None,
) -> dict[str, Any]:
    plan = dict(row["plan_state"] if shuffled is None else shuffled)
    if arm == "formula":
        prompt = build_formula_only_body_prompt(plan).rstrip() + "\n"
    else:
        prompt = build_body_prompt(plan).rstrip() + "\n"
    shared_seed_label = f"rep-{replicate}" if replicate < 4 else f"{arm}-extra-{replicate}"
    scientific_seed = stable_seed(int(root_seed), pair_id, str(row["source"]), shared_seed_label)
    task = {
        "sample_idx": int(sample_idx),
        "task_id": f"{pair_id}/{row['source']}/{arm}/r{replicate}",
        "pair_id": pair_id,
        "plan_id": row["plan_id"],
        "plan_source": row["source"],
        "arm": arm,
        "replicate": int(replicate),
        "scientific_seed": scientific_seed,
        "plan_state": plan,
        "prompt": prompt,
        "body_prompt_style": "formula_only" if arm == "formula" else "full_plan_state",
        "shuffle_fields": list(SHUFFLED_FIELDS) if arm == "shuffle" else [],
    }
    if shuffle_donor_plan_id is not None:
        task["shuffle_donor_plan_id"] = str(shuffle_donor_plan_id)
    return task


def build_panels(
    learned: Sequence[dict[str, Any]],
    gold: Sequence[dict[str, Any]],
    *,
    num_pairs: int = 24,
    e2_pairs: int = 8,
    seed: int = 20260822,
) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    if not 1 <= int(e2_pairs) <= int(num_pairs):
        raise ValueError("e2_pairs must be between 1 and num_pairs")
    pairs = select_matched_pairs(learned, gold, num_pairs=num_pairs, seed=seed)
    selected_rows = [row for pair in pairs for row in pair]
    shuffled = shuffled_plans(selected_rows)
    tasks: list[dict[str, Any]] = []
    e2_task_ids: list[str] = []
    sample_idx = 0
    for pair_index, pair in enumerate(pairs):
        pair_id = f"pair-{pair_index:03d}"
        for row in pair:
            for arm, replicates in (("full", 8), ("formula", 4), ("shuffle", 4)):
                for replicate in range(replicates):
                    shuffle_record = shuffled[row["plan_id"]] if arm == "shuffle" else None
                    task = build_task(
                        row,
                        pair_id=pair_id,
                        arm=arm,
                        replicate=replicate,
                        sample_idx=sample_idx,
                        root_seed=seed,
                        shuffled=None if shuffle_record is None else shuffle_record["plan_state"],
                        shuffle_donor_plan_id=(
                            None if shuffle_record is None else str(shuffle_record["donor_plan_id"])
                        ),
                    )
                    tasks.append(task)
                    if pair_index < int(e2_pairs) and replicate < 4:
                        e2_task_ids.append(task["task_id"])
                    sample_idx += 1
    report = {
        "schema": "h1a2_story_panel_selection_v1",
        "root_seed": int(seed),
        "num_pairs": int(num_pairs),
        "selected_plans": len(selected_rows),
        "e1_requested_tasks": len(tasks),
        "e2_requested_tasks": len(e2_task_ids),
        "matching_fields": ["N", "arity", "anion_framework"],
        "shuffle_fields": list(SHUFFLED_FIELDS),
        "shuffle_donors": {
            plan_id: record["donor_plan_id"] for plan_id, record in sorted(shuffled.items())
        },
        "pairs": [
            {
                "pair_id": f"pair-{index:03d}",
                "stratum": list(plan_stratum(pair[0]["plan_state"])),
                "learned_plan_id": pair[0]["plan_id"],
                "gold_plan_id": pair[1]["plan_id"],
            }
            for index, pair in enumerate(pairs)
        ],
    }
    return tasks, e2_task_ids, report


__all__ = [
    "SHUFFLED_FIELDS",
    "build_panels",
    "build_task",
    "coarse_signature",
    "load_plan_rows",
    "plan_stratum",
    "select_matched_pairs",
    "shuffled_plans",
    "stable_seed",
    "validate_plan_for_panel",
]
