#!/usr/bin/env python3
"""Freeze matched MP20-clean/on-policy cell and anchor closure states."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crystal_dlm.r5_dynamic_length import validate_answer_matches_plan  # noqa: E402
from crystal_dlm.spad_program import LATTICE_POSITIONS, coordinate_positions  # noqa: E402


SCHEMA = "potential_closure_state_v1"
STRATA = (
    "mp20_clean_cell",
    "mp20_clean_site",
    "on_policy_cell",
    "on_policy_site",
)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number} is not an object")
            yield value


def deterministic_key(row: Mapping[str, Any], seed: int) -> tuple[str, int]:
    prompt_record = row.get("prompt_record") or {}
    identity = {
        "sample_idx": int(row["sample_idx"]),
        "source_row_idx": int(prompt_record.get("source_row_idx", -1)),
        "plan_state": row["plan_state"],
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(f"{int(seed)}|{encoded}".encode()).hexdigest()
    return digest, int(row["sample_idx"])


def limited_anchor_slots_for_species(
    species: Sequence[str],
    species_program: Sequence[str],
    *,
    max_anchors: int = 2,
) -> tuple[int, ...]:
    values = [str(value) for value in species]
    program = [str(value) for value in species_program]
    if not values or not program:
        raise ValueError("species and program must be nonempty")
    if len(program) != len(set(program)):
        raise ValueError("species program contains duplicates")
    limit = int(max_anchors)
    if limit <= 0:
        raise ValueError("max_anchors must be positive")
    anchors: list[int] = []
    for symbol in program[:limit]:
        try:
            anchors.append(values.index(symbol))
        except ValueError as exc:
            raise ValueError(f"program species {symbol} is absent from body") from exc
    return tuple(reversed(anchors))


def _normalized_source(row: Mapping[str, Any]) -> dict[str, Any]:
    if row.get("parsed") is not True or not isinstance(row.get("text"), str):
        raise ValueError("on-policy predictor row is not parsed")
    plan = row.get("plan_state")
    prompt_record = row.get("prompt_record")
    if not isinstance(plan, Mapping) or not isinstance(prompt_record, Mapping):
        raise ValueError("predictor row lacks plan/pointer metadata")
    teacher_answer = prompt_record.get("teacher_answer")
    generated_answer = row.get("text")
    prompt = row.get("conditioning_prompt")
    program = prompt_record.get("species_program")
    if not isinstance(teacher_answer, str) or not isinstance(generated_answer, str):
        raise ValueError("predictor row lacks teacher/generated answers")
    if not isinstance(prompt, str) or not prompt.rstrip():
        raise ValueError("predictor row lacks conditioning prompt")
    if not isinstance(program, list) or not program:
        raise ValueError("predictor row lacks species program")
    teacher = validate_answer_matches_plan(plan, teacher_answer)
    generated = validate_answer_matches_plan(plan, generated_answer)
    if Counter(teacher["species"]) != Counter(generated["species"]):
        raise ValueError("teacher/generated composition mismatch")
    return {
        "source_sample_idx": int(row["sample_idx"]),
        "source_row_idx": int(prompt_record.get("source_row_idx", -1)),
        "plan_state": dict(plan),
        "prompt": prompt.rstrip() + "\n",
        "species_program": [str(value) for value in program],
        "species_program_source": str(
            prompt_record.get("species_program_source")
            or "frozen_planner_llama_pointer"
        ),
        "teacher_answer": teacher_answer,
        "teacher_species": [str(value) for value in teacher["species"]],
        "generated_answer": generated_answer,
        "generated_species": [str(value) for value in generated["species"]],
    }


def build_states(
    rows: Sequence[Mapping[str, Any]],
    *,
    requested_sources: int,
    selection_seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    rejected = Counter()
    seen_sample_indices: set[int] = set()
    for row in rows:
        sample_idx = int(row["sample_idx"])
        if sample_idx in seen_sample_indices:
            raise ValueError("predictor body contains duplicate sample_idx")
        seen_sample_indices.add(sample_idx)
        try:
            normalized.append(_normalized_source(row))
        except Exception as exc:
            rejected[f"{type(exc).__name__}:{str(exc)[:120]}"] += 1
    normalized.sort(
        key=lambda row: deterministic_key(
            {
                "sample_idx": row["source_sample_idx"],
                "plan_state": row["plan_state"],
                "prompt_record": {"source_row_idx": row["source_row_idx"]},
            },
            int(selection_seed),
        )
    )
    selected = normalized[: int(requested_sources)]
    if len(selected) != int(requested_sources):
        raise RuntimeError("insufficient eligible matched closure sources")

    by_stratum: dict[str, list[dict[str, Any]]] = {name: [] for name in STRATA}
    for source_rank, source in enumerate(selected):
        for domain, answer_key, species_key in (
            ("mp20_clean", "teacher_answer", "teacher_species"),
            ("on_policy", "generated_answer", "generated_species"),
        ):
            answer = str(source[answer_key])
            species = list(source[species_key])
            limited_slots = limited_anchor_slots_for_species(
                species,
                source["species_program"],
                max_anchors=2,
            )
            chosen_rank = int(source_rank % len(limited_slots))
            slot = int(limited_slots[chosen_rank])
            common = {
                "schema": SCHEMA,
                "source_rank": int(source_rank),
                "source_sample_idx": int(source["source_sample_idx"]),
                "source_row_idx": int(source["source_row_idx"]),
                "source_domain": domain,
                "prompt": str(source["prompt"]),
                "plan_state": dict(source["plan_state"]),
                "species_program": list(source["species_program"]),
                "species_program_source": str(source["species_program_source"]),
                "source_answer": answer,
                "clean_teacher_answer": str(source["teacher_answer"]),
                "limited_anchor_slots": list(limited_slots),
                "proposal_temperature": 0.7,
                "maximum_proposal_attempts": 8,
                "candidate_retention": "first_distinct_legal_in_request_order",
                "outcomes_read": False,
            }
            by_stratum[f"{domain}_cell"].append(
                {
                    **common,
                    "stratum": f"{domain}_cell",
                    "transaction_kind": "cell",
                    "active_positions": list(LATTICE_POSITIONS),
                    "backfill_slot": None,
                    "anchor_rank": None,
                    "anchor_symbol": None,
                }
            )
            by_stratum[f"{domain}_site"].append(
                {
                    **common,
                    "stratum": f"{domain}_site",
                    "transaction_kind": "site",
                    "active_positions": list(coordinate_positions(slot)),
                    "backfill_slot": slot,
                    "anchor_rank": chosen_rank,
                    "anchor_symbol": str(species[slot]),
                }
            )

    states: list[dict[str, Any]] = []
    for stratum in STRATA:
        for row in by_stratum[stratum]:
            row["group_idx"] = len(states)
            states.append(row)
    expected = int(requested_sources) * len(STRATA)
    if len(states) != expected:
        raise RuntimeError("closure state denominator changed")
    manifest = {
        "schema": "potential_closure_state_manifest_v1",
        "sources_available": len(normalized),
        "sources_selected": int(len(selected)),
        "groups": int(len(states)),
        "groups_per_stratum": {
            name: int(len(by_stratum[name])) for name in STRATA
        },
        "selection_seed": int(selection_seed),
        "selection": "outcome_blind_hash_order",
        "matched_domains": True,
        "maximum_proposal_attempts": 8,
        "proposal_temperature": 0.7,
        "outcomes_read": False,
        "rejected_sources": dict(rejected),
    }
    return states, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictor-body", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--requested-sources", type=int, default=512)
    parser.add_argument("--selection-seed", type=int, default=20260904)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if int(args.requested_sources) != 512:
        raise ValueError("first potential-closure study requires 512 matched sources")
    rows = list(iter_jsonl(args.predictor_body.resolve()))
    states, manifest = build_states(
        rows,
        requested_sources=int(args.requested_sources),
        selection_seed=int(args.selection_seed),
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "states.jsonl").open("x", encoding="utf-8") as handle:
        for row in states:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "_SUCCESS").touch()
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()

