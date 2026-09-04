#!/usr/bin/env python3
"""Freeze one deployment-stage transaction source for every MP20-train row.

This ledger is deliberately outcome blind.  It joins the full SPAD schedule
SFT train split to a full-row Planner file, validates their shared immutable
content, and assigns exactly one deployment stage by ``source_row_idx % 3``.
No row is dropped for a downstream generation, parsing, or energy failure.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crystal_dlm.dynamic_crystal import dynamic_answer_token_count  # noqa: E402
from crystal_dlm.fixed_slot import tokenize_answer_text  # noqa: E402
from crystal_dlm.r5_dynamic_length import validate_answer_matches_plan  # noqa: E402
from crystal_dlm.spad_program import LATTICE_POSITIONS  # noqa: E402


SCHEMA = "full_mp20_transaction_source_v1"
FORMAL_MP20_TRAIN_ROWS = 27_136
DEPLOYMENT_STAGES = ("cell", "anchor_second", "anchor_first")
DEFAULT_COMMON_RANDOM_SEED = 20_260_904
COMMON_RANDOM_SEED_RULE = (
    "base_seed + source_row_idx * 1000003 + deployment_stage_index * 1009"
)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number} is not a JSON object")
            yield value


def _indexed_rows(
    rows: Sequence[Mapping[str, Any]], *, label: str
) -> dict[int, Mapping[str, Any]]:
    indexed: dict[int, Mapping[str, Any]] = {}
    for ordinal, row in enumerate(rows):
        if "source_row_idx" not in row:
            raise ValueError(f"{label} row {ordinal} lacks source_row_idx")
        try:
            source_idx = int(row["source_row_idx"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{label} row {ordinal} has invalid source_row_idx"
            ) from exc
        if source_idx < 0:
            raise ValueError(f"{label} source_row_idx must be nonnegative")
        if source_idx in indexed:
            raise ValueError(f"{label} duplicates source_row_idx {source_idx}")
        indexed[source_idx] = row
    expected = set(range(len(rows)))
    actual = set(indexed)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"{label} source_row_idx is not contiguous from zero; "
            f"missing={missing[:8]} extra={extra[:8]}"
        )
    return indexed


def _require_mapping(row: Mapping[str, Any], key: str, *, label: str) -> Mapping[str, Any]:
    value = row.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} lacks mapping field {key}")
    return value


def _require_text(row: Mapping[str, Any], key: str, *, label: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} lacks nonempty text field {key}")
    return value


def _sft_teacher_answer(row: Mapping[str, Any], *, source_idx: int) -> str:
    available = {
        key: row[key]
        for key in ("teacher_answer", "answer", "source_answer")
        if key in row
    }
    if not available:
        raise ValueError(f"SFT source_row_idx {source_idx} lacks a teacher answer")
    if any(not isinstance(value, str) or not value for value in available.values()):
        raise ValueError(f"SFT source_row_idx {source_idx} has an invalid teacher answer")
    values = list(available.values())
    if any(value != values[0] for value in values[1:]):
        raise ValueError(
            f"SFT source_row_idx {source_idx} has inconsistent answer fields"
        )
    return str(values[0])


def _source_marker(row: Mapping[str, Any], *, source_idx: int) -> tuple[str, Any]:
    for key in ("source", "source_marker", "species_program_source"):
        if key in row and row[key] is not None and row[key] != "":
            value = row[key]
            if isinstance(value, (Mapping, list)):
                raise ValueError(
                    f"Plan source_row_idx {source_idx} has a non-scalar {key} marker"
                )
            return key, value
    raise ValueError(f"Plan source_row_idx {source_idx} lacks a source marker")


def anchor_slots_from_teacher_species(
    teacher_species: Sequence[str], species_program: Sequence[str]
) -> tuple[int, int, list[int]]:
    """Map the first two programmed species to actual teacher-body slots.

    The deployment revision order is second anchor followed by first anchor.
    Unary structures therefore use their sole actual slot for both named
    anchor stages, making the fallback deterministic without dropping a row.
    """

    species = [str(value) for value in teacher_species]
    program = [str(value) for value in species_program]
    if not species:
        raise ValueError("teacher species must be nonempty")
    if not program:
        raise ValueError("species_program must be nonempty")
    if len(program) != len(set(program)):
        raise ValueError("species_program contains duplicate species")
    unique_species = set(species)
    if set(program) != unique_species or len(program) != len(unique_species):
        raise ValueError(
            "species_program must permute every unique actual teacher species"
        )
    program_slots = [species.index(symbol) for symbol in program]
    first = int(program_slots[0])
    second = int(program_slots[1]) if len(program_slots) >= 2 else first
    return first, second, [int(value) for value in program_slots]


def common_random_seed(base_seed: int, source_idx: int, stage_index: int) -> int:
    return int(base_seed) + int(source_idx) * 1_000_003 + int(stage_index) * 1_009


def _stage_descriptor(
    stage: str,
    *,
    first_teacher_slot: int,
    second_teacher_slot: int,
    species_program: Sequence[str],
) -> dict[str, Any]:
    if stage == "cell":
        return {
            "stage": stage,
            "transaction_kind": "cell",
            "active_positions": list(LATTICE_POSITIONS),
            "anchor_rank": None,
            "anchor_slot": None,
            "teacher_anchor_slot": None,
            "anchor_symbol": None,
        }
    if stage == "anchor_second":
        rank = 1 if len(species_program) >= 2 else 0
        teacher_slot = int(second_teacher_slot)
    elif stage == "anchor_first":
        rank = 0
        teacher_slot = int(first_teacher_slot)
    else:  # pragma: no cover - guarded by the fixed stage tuple
        raise ValueError(f"unknown deployment stage {stage}")
    return {
        "stage": stage,
        "transaction_kind": "site",
        "active_positions": None,
        "active_positions_resolution": "resolve_from_generated_body_species",
        "anchor_rank": int(rank),
        "anchor_slot": None,
        "teacher_anchor_slot": teacher_slot,
        "anchor_symbol": str(species_program[rank]),
    }


def build_full_mp20_transaction_sources(
    sft_rows: Sequence[Mapping[str, Any]],
    plan_rows: Sequence[Mapping[str, Any]],
    *,
    common_seed: int = DEFAULT_COMMON_RANDOM_SEED,
    formal: bool = False,
) -> list[dict[str, Any]]:
    """Build one immutable, outcome-blind source row per MP20-train row."""

    if formal and len(sft_rows) != FORMAL_MP20_TRAIN_ROWS:
        raise ValueError(
            "formal mode requires exactly "
            f"{FORMAL_MP20_TRAIN_ROWS} SPAD schedule SFT train rows"
        )
    if formal and len(plan_rows) != FORMAL_MP20_TRAIN_ROWS:
        raise ValueError(
            "formal mode requires exactly "
            f"{FORMAL_MP20_TRAIN_ROWS} full-row Plan rows"
        )
    sft_by_idx = _indexed_rows(sft_rows, label="SFT")
    plan_by_idx = _indexed_rows(plan_rows, label="Plan")
    if set(sft_by_idx) != set(plan_by_idx):
        missing_plan = sorted(set(sft_by_idx) - set(plan_by_idx))
        missing_sft = sorted(set(plan_by_idx) - set(sft_by_idx))
        raise ValueError(
            "SFT/Plan source coverage mismatch; "
            f"missing Plan={missing_plan[:8]} missing SFT={missing_sft[:8]}"
        )

    output: list[dict[str, Any]] = []
    for source_idx in range(len(sft_rows)):
        sft = sft_by_idx[source_idx]
        plan_row = plan_by_idx[source_idx]
        sft_plan = _require_mapping(sft, "plan_state", label=f"SFT {source_idx}")
        plan = _require_mapping(
            plan_row, "plan_state", label=f"Plan {source_idx}"
        )
        if dict(sft_plan) != dict(plan):
            raise ValueError(f"source_row_idx {source_idx} Plan mismatch")

        sft_prompt = _require_text(sft, "prompt", label=f"SFT {source_idx}")
        plan_prompt = _require_text(
            plan_row, "prompt", label=f"Plan {source_idx}"
        )
        if sft_prompt != plan_prompt:
            raise ValueError(f"source_row_idx {source_idx} prompt mismatch")

        sft_answer = _sft_teacher_answer(sft, source_idx=source_idx)
        plan_answer = _require_text(
            plan_row, "teacher_answer", label=f"Plan {source_idx}"
        )
        if sft_answer != plan_answer:
            raise ValueError(f"source_row_idx {source_idx} teacher answer mismatch")

        arrays = validate_answer_matches_plan(plan, plan_answer)
        expected_tokens = dynamic_answer_token_count(int(plan["N"]))
        actual_tokens = len(tokenize_answer_text(plan_answer))
        if actual_tokens != expected_tokens:
            raise ValueError(
                f"source_row_idx {source_idx} is not exact 7+4N: "
                f"expected {expected_tokens}, found {actual_tokens}"
            )

        program_raw = plan_row.get("species_program")
        if not isinstance(program_raw, list):
            raise ValueError(f"Plan source_row_idx {source_idx} lacks species_program")
        program = [str(value) for value in program_raw]
        teacher_species = [str(value) for value in arrays["species"]]
        first_slot, second_slot, program_slots = anchor_slots_from_teacher_species(
            teacher_species, program
        )
        marker_field, marker_value = _source_marker(plan_row, source_idx=source_idx)

        stage_index = int(source_idx % len(DEPLOYMENT_STAGES))
        stage = DEPLOYMENT_STAGES[stage_index]
        descriptors = {
            name: _stage_descriptor(
                name,
                first_teacher_slot=first_slot,
                second_teacher_slot=second_slot,
                species_program=program,
            )
            for name in DEPLOYMENT_STAGES
        }
        remaining = []
        for reference_stage_index, name in enumerate(DEPLOYMENT_STAGES):
            if reference_stage_index <= stage_index:
                continue
            reference = dict(descriptors[name])
            reference["executor"] = "frozen_reference_dlm"
            reference["common_random_seed"] = common_random_seed(
                common_seed, source_idx, reference_stage_index
            )
            remaining.append(reference)
        active = descriptors[stage]
        output.append(
            {
                "schema": SCHEMA,
                "source_row_idx": int(source_idx),
                "sample_idx": int(source_idx),
                "source_marker_field": marker_field,
                "source_marker": marker_value,
                "source_split": str(sft.get("source_split") or "train"),
                "plan_state": dict(plan),
                "prompt": plan_prompt,
                "teacher_answer": plan_answer,
                "teacher_species": teacher_species,
                "species_program": program,
                "species_program_anchor_slots": program_slots,
                "deployment_stage_order": list(DEPLOYMENT_STAGES),
                "deployment_stage_index": stage_index,
                "deployment_stage": stage,
                "transaction_kind": active["transaction_kind"],
                "active_positions": (
                    None
                    if active["active_positions"] is None
                    else list(active["active_positions"])
                ),
                "active_positions_resolution": active.get(
                    "active_positions_resolution", "already_resolved"
                ),
                "anchor_rank": active["anchor_rank"],
                "anchor_slot": active["anchor_slot"],
                "teacher_anchor_slot": active["teacher_anchor_slot"],
                "anchor_symbol": active["anchor_symbol"],
                "anchor_first_teacher_slot": int(first_slot),
                "anchor_second_teacher_slot": int(second_slot),
                "generated_body_slot_resolution_required": stage != "cell",
                "unary_anchor_fallback": len(set(teacher_species)) < 2,
                "remaining_reference_stages": remaining,
                "remaining_reference_policy": (
                    "execute_in_declared_order_with_frozen_reference_dlm"
                ),
                "common_random_seed_base": int(common_seed),
                "common_random_seed_rule": COMMON_RANDOM_SEED_RULE,
                "common_random_seed": common_random_seed(
                    common_seed, source_idx, stage_index
                ),
                "source_weight": 1.0,
                "outcomes_read": False,
                "downstream_status": "pending",
                "downstream_failure": None,
                "retain_failure_placeholder": True,
            }
        )

    if len(output) != len(sft_rows):
        raise RuntimeError("source ledger denominator changed")
    if [row["source_row_idx"] for row in output] != list(range(len(output))):
        raise RuntimeError("output source_row_idx is not contiguous")
    if [row["sample_idx"] for row in output] != list(range(len(output))):
        raise RuntimeError("output sample_idx is not contiguous")
    return output


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-train", type=Path, required=True)
    parser.add_argument("--plans-for-dlm", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument(
        "--common-random-seed", type=int, default=DEFAULT_COMMON_RANDOM_SEED
    )
    parser.add_argument(
        "--formal",
        action="store_true",
        help=f"require the complete {FORMAL_MP20_TRAIN_ROWS}-row MP20 train split",
    )
    args = parser.parse_args()
    if args.output_jsonl.exists():
        raise FileExistsError(args.output_jsonl)
    sft_rows = list(iter_jsonl(args.sft_train.resolve()))
    plan_rows = list(iter_jsonl(args.plans_for_dlm.resolve()))
    ledger = build_full_mp20_transaction_sources(
        sft_rows,
        plan_rows,
        common_seed=int(args.common_random_seed),
        formal=bool(args.formal),
    )
    write_jsonl(args.output_jsonl.resolve(), ledger)
    summary = {
        "schema": "full_mp20_transaction_source_summary_v1",
        "rows": len(ledger),
        "formal": bool(args.formal),
        "stage_counts": {
            stage: sum(row["deployment_stage"] == stage for row in ledger)
            for stage in DEPLOYMENT_STAGES
        },
        "source_weight_one": all(row["source_weight"] == 1.0 for row in ledger),
        "outcomes_read": False,
        "failure_placeholders_retained": True,
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
