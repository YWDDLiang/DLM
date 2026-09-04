#!/usr/bin/env python3
"""Build deployment-matched terminal transaction actions for full MP20 train.

The output is one shared, outcome-blind terminal-action pool.  A future
single-point control and a future basin-value method must read this same file;
this program never reads or ranks candidate energies.

For every source, the frozen Planner-Llama ``species_program`` controls the
deployed transaction order::

    complete reference body -> cell -> anchor_second -> anchor_first

The source ledger assigns exactly one active stage.  Stages before it are
executed with the frozen reference DLM to construct the deployed state.  Up to
four actions are then proposed for that state (no-op, reference-DLM, and two
force/stress directions).  Every retained action is followed by the same
frozen reference continuation, with source/stage seeds shared across actions.
Failures and variable K remain in the fixed source denominator.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence

import numpy as np


SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from crystal_dlm.dynamic_crystal import (  # noqa: E402
    arrays_to_structure,
)
from crystal_dlm.fixed_slot import (  # noqa: E402
    MASK_TOKEN_ID,
    FixedSlotConfig,
    tokenize_answer_text,
)
from crystal_dlm.r5_dynamic_length import (  # noqa: E402
    exact_body_token_count,
    exact_dynamic_schema_constraints,
    validate_answer_matches_plan,
    validate_dynamic_tokenizer_contract,
)
from crystal_dlm.spad_program import (  # noqa: E402
    LATTICE_POSITIONS,
    coordinate_positions,
)
from crystal_dlm.transaction_physics import (  # noqa: E402
    TransactionProposal,
    propose_force_site_transactions,
    propose_stress_lattice_transactions,
)


SOURCE_SCHEMA = "full_mp20_transaction_source_v1"
OUTPUT_SCHEMA = "full_mp20_terminal_action_group_v1"
MANIFEST_SCHEMA = "full_mp20_terminal_action_manifest_v1"
FORMAL_MP20_TRAIN_ROWS = 27_136
DEPLOYMENT_STAGES = ("cell", "anchor_second", "anchor_first")
MAX_CANDIDATES = 4
TEMPERATURE = 0.7
STAGE_SEED_STRIDE = 1_009
SOURCE_SEED_STRIDE = 1_000_003
PHYSICS_TOKEN_CHANGE_THRESHOLD = 0.85


def _geometric_steps(start: float, growth: float, stop: float) -> tuple[float, ...]:
    """Fixed quantization scan; it observes tokens, never energies or outcomes."""

    values: list[float] = []
    current = float(start)
    while current < float(stop):
        values.append(current)
        current *= float(growth)
    values.append(float(stop))
    return tuple(values)


# These schedules bracket the smallest representable transaction change.  The
# proposal implementation deterministically keeps the first step whose actual
# special tokens differ from no-op.  The caps are fixed before any outcome is
# observed; there is no resampling, energy-based step choice, or support search.
FORCE_ONE_BIN_SCAN_STEPS_A = _geometric_steps(0.0025, 1.35, 0.40)
STRAIN_ONE_BIN_SCAN_STEPS = _geometric_steps(0.00025, 1.40, 0.05)


def actual_token_quantization_contract(
    config: FixedSlotConfig = FixedSlotConfig(),
) -> dict[str, Any]:
    """Describe the deployed vocabulary rather than an assumed '1000-bin' grid."""

    return {
        "coordinate_token_ids_per_axis": int(
            config.coord_max_bin - config.coord_min_bin + 1
        ),
        "coordinate_periodic_intervals": int(config.coord_max_bin),
        "coordinate_effective_canonical_values": int(config.coord_max_bin),
        "coordinate_fractional_step": float(1.0 / config.coord_max_bin),
        "length_token_bins_per_axis": int(
            config.length_max_bin - config.length_min_bin + 1
        ),
        "length_step_A": float(config.length_step),
        "angle_token_bins_per_axis": int(
            config.angle_max_bin - config.angle_min_bin + 1
        ),
        "angle_step_degree": 1.0,
        "force_step_scan_A": [float(value) for value in FORCE_ONE_BIN_SCAN_STEPS_A],
        "strain_step_scan": [float(value) for value in STRAIN_ONE_BIN_SCAN_STEPS],
        "selection_rule": "first_quantized_non_noop_in_fixed_ascending_scan",
        "energy_or_outcome_used_to_choose_step": False,
    }


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number} is not a JSON object")
            yield value


def json_line_bytes(value: Mapping[str, Any]) -> bytes:
    """Return the canonical bytes consumed unchanged by both future methods."""

    return (
        json.dumps(dict(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def write_jsonl_row(handle: Any, value: Mapping[str, Any]) -> None:
    handle.write(json_line_bytes(value).decode("utf-8"))


def contiguous_shard(total: int, world_size: int, rank: int) -> tuple[int, int]:
    if int(total) < 0 or int(world_size) <= 0 or not 0 <= int(rank) < int(world_size):
        raise ValueError("invalid contiguous shard parameters")
    start = int(total) * int(rank) // int(world_size)
    stop = int(total) * (int(rank) + 1) // int(world_size)
    return start, stop


def source_index_from_body(row: Mapping[str, Any]) -> int:
    """Resolve the source identity without silently accepting disagreement."""

    values: list[int] = []
    if row.get("source_row_idx") is not None:
        values.append(int(row["source_row_idx"]))
    prompt_record = row.get("prompt_record")
    if isinstance(prompt_record, Mapping) and prompt_record.get("source_row_idx") is not None:
        values.append(int(prompt_record["source_row_idx"]))
    if row.get("sample_idx") is not None:
        values.append(int(row["sample_idx"]))
    if not values:
        raise ValueError("reference body row has no source_row_idx or sample_idx")
    if len(set(values)) != 1:
        raise ValueError(f"reference body source identities disagree: {values}")
    return values[0]


def index_reference_bodies(
    rows: Iterable[Mapping[str, Any]], *, expected_sources: int
) -> dict[int, dict[str, Any]]:
    indexed: dict[int, dict[str, Any]] = {}
    for row in rows:
        source_idx = source_index_from_body(row)
        if not 0 <= source_idx < int(expected_sources):
            raise ValueError(f"reference body source {source_idx} is outside denominator")
        if source_idx in indexed:
            raise ValueError(f"reference body duplicates source_row_idx {source_idx}")
        indexed[source_idx] = dict(row)
    return indexed


def _answer_from_body(row: Mapping[str, Any]) -> str:
    for key in ("text", "answer", "source_answer"):
        value = row.get(key)
        if isinstance(value, str) and value:
            return value
    raise ValueError("reference body row has no generated answer text")


def resolve_species_program_slots(
    generated_species: Sequence[str], species_program: Sequence[str]
) -> dict[str, Any]:
    """Resolve Llama anchors against the *generated* body's species order."""

    species = [str(value) for value in generated_species]
    program = [str(value) for value in species_program]
    if not species:
        raise ValueError("generated body has no species")
    if not program or len(program) != len(set(program)):
        raise ValueError("species_program must contain distinct species")
    unique_species = set(species)
    if set(program) != unique_species or len(program) != len(unique_species):
        raise ValueError(
            "species_program must permute all unique generated-body species"
        )
    slots = [species.index(symbol) for symbol in program]
    first_slot = int(slots[0])
    second_slot = int(slots[1]) if len(slots) >= 2 else first_slot
    return {
        "program_slots": [int(value) for value in slots],
        "anchor_first_slot": first_slot,
        "anchor_second_slot": second_slot,
        "anchor_first_symbol": program[0],
        "anchor_second_symbol": program[1] if len(program) >= 2 else program[0],
        "unary_anchor_fallback": len(program) == 1,
    }


def stage_seed(source: Mapping[str, Any], stage_index: int) -> int:
    base = int(source["common_random_seed_base"])
    source_idx = int(source["source_row_idx"])
    return base + source_idx * SOURCE_SEED_STRIDE + int(stage_index) * STAGE_SEED_STRIDE


def resolved_deployment_stages(
    source: Mapping[str, Any], generated_species: Sequence[str]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Resolve all cell/site transactions and verify the ledger seed contract."""

    if source.get("schema") != SOURCE_SCHEMA:
        raise ValueError("full MP20 source ledger schema changed")
    program = source.get("species_program")
    if not isinstance(program, list):
        raise ValueError("source ledger lacks Llama species_program")
    anchors = resolve_species_program_slots(generated_species, program)
    stages: list[dict[str, Any]] = []
    for stage_index, stage_name in enumerate(DEPLOYMENT_STAGES):
        if stage_name == "cell":
            slot = None
            positions = tuple(int(value) for value in LATTICE_POSITIONS)
            symbol = None
            transaction_kind = "cell"
        elif stage_name == "anchor_second":
            slot = int(anchors["anchor_second_slot"])
            positions = tuple(int(value) for value in coordinate_positions(slot))
            symbol = str(anchors["anchor_second_symbol"])
            transaction_kind = "site"
        else:
            slot = int(anchors["anchor_first_slot"])
            positions = tuple(int(value) for value in coordinate_positions(slot))
            symbol = str(anchors["anchor_first_symbol"])
            transaction_kind = "site"
        stages.append(
            {
                "stage": stage_name,
                "stage_index": int(stage_index),
                "transaction_kind": transaction_kind,
                "anchor_slot": slot,
                "anchor_symbol": symbol,
                "active_positions": list(positions),
                "common_random_seed": stage_seed(source, stage_index),
                "suffix_visible": True,
            }
        )

    active_index = int(source.get("deployment_stage_index", -1))
    if not 0 <= active_index < len(stages):
        raise ValueError("deployment_stage_index lies outside the deployment chain")
    if str(source.get("deployment_stage")) != stages[active_index]["stage"]:
        raise ValueError("deployment stage name/index disagree")
    if int(source.get("common_random_seed", -1)) != stages[active_index][
        "common_random_seed"
    ]:
        raise ValueError("active-stage seed disagrees with ledger seed rule")
    declared_remaining = source.get("remaining_reference_stages")
    if not isinstance(declared_remaining, list):
        raise ValueError("source ledger lacks remaining_reference_stages")
    if [str(value.get("stage")) for value in declared_remaining] != [
        value["stage"] for value in stages[active_index + 1 :]
    ]:
        raise ValueError("remaining reference stage order changed")
    for declared, resolved in zip(
        declared_remaining, stages[active_index + 1 :], strict=True
    ):
        if int(declared.get("common_random_seed", -1)) != int(
            resolved["common_random_seed"]
        ):
            raise ValueError("remaining-stage seed disagrees with ledger")
    return stages, anchors


def continuation_stage_names(
    stages: Sequence[Mapping[str, Any]], active_stage_index: int
) -> list[str]:
    return [str(value["stage"]) for value in stages[int(active_stage_index) + 1 :]]


def validate_transaction_transition(
    before_answer: str,
    after_answer: str,
    *,
    plan_state: Mapping[str, Any],
    stage: Mapping[str, Any],
) -> tuple[list[str], list[str]]:
    """Require a complete transaction and forbid all out-of-block mutation."""

    before_arrays = validate_answer_matches_plan(plan_state, before_answer)
    after_arrays = validate_answer_matches_plan(plan_state, after_answer)
    if list(before_arrays["species"]) != list(after_arrays["species"]):
        raise ValueError("transaction changed generated-body species order")
    before = tokenize_answer_text(before_answer)
    after = tokenize_answer_text(after_answer)
    if len(before) != len(after):
        raise ValueError("transaction changed exact 7+4N length")
    active = tuple(int(value) for value in stage["active_positions"])
    changed = [
        index
        for index, pair in enumerate(zip(before, after, strict=True))
        if pair[0] != pair[1]
    ]
    escaped = [index for index in changed if index not in set(active)]
    if escaped:
        raise ValueError(f"transaction changed non-active positions {escaped[:8]}")
    return [before[index] for index in active], [after[index] for index in active]


def execute_stage_chain(
    answer: str,
    stages: Sequence[Mapping[str, Any]],
    executor: Callable[[str, Mapping[str, Any]], tuple[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    """Pure serial reference-chain helper used by tests and audits."""

    current = str(answer)
    logs: list[dict[str, Any]] = []
    for stage in stages:
        updated, runtime_log = executor(current, stage)
        logs.append(
            {
                "stage": str(stage["stage"]),
                "common_random_seed": int(stage["common_random_seed"]),
                "runtime_log": runtime_log,
            }
        )
        current = str(updated)
    return current, logs


def continuation_requests(
    candidates: Sequence[Mapping[str, Any]],
    stages: Sequence[Mapping[str, Any]],
    active_stage_index: int,
) -> list[dict[str, Any]]:
    """Expose candidate-shared future seeds without executing a model."""

    output: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_idx = int(candidate["candidate_idx"])
        for stage in stages[int(active_stage_index) + 1 :]:
            output.append(
                {
                    "candidate_idx": candidate_idx,
                    "stage": str(stage["stage"]),
                    "common_random_seed": int(stage["common_random_seed"]),
                }
            )
    return output


def _active_tokens(answer: str, stage: Mapping[str, Any]) -> list[str]:
    tokens = tokenize_answer_text(answer)
    return [tokens[int(position)] for position in stage["active_positions"]]


def candidate_attempt(
    *,
    source: str,
    state_answer: str,
    action_answer: str | None,
    stage: Mapping[str, Any],
    legal: bool,
    failure: str | None,
    proposal: TransactionProposal | None = None,
    runtime_log: Any = None,
) -> dict[str, Any]:
    active_tokens: list[str] = []
    if action_answer is not None:
        active_tokens = _active_tokens(action_answer, stage)
    elif proposal is not None:
        active_tokens = [str(value) for value in proposal.transaction_tokens]
    proposal_metadata = None
    if proposal is not None:
        proposal_metadata = {
            "kind": str(proposal.kind),
            "direction": str(proposal.direction),
            "status": str(proposal.status),
            "reason": str(proposal.reason),
            "step": None if proposal.step is None else float(proposal.step),
            "transaction_tokens": [
                str(value) for value in proposal.transaction_tokens
            ],
            "minimum_distance_A": (
                None
                if proposal.minimum_distance_A is None
                else float(proposal.minimum_distance_A)
            ),
        }
    return {
        "candidate_source": str(source),
        "state_answer": str(state_answer),
        "action_answer": action_answer,
        "active_positions": [int(value) for value in stage["active_positions"]],
        "active_action_tokens": active_tokens,
        "active_action_token_ids": [],
        "action_tokens": active_tokens,
        "action_token_ids": [],
        "active_legal": bool(legal),
        "active_failure": failure,
        "proposal": proposal_metadata,
        "runtime_log": runtime_log,
        "retention_status": None,
        "candidate_idx": None,
        "terminal_answer": None,
        "terminal_structure": None,
        "terminal_cif": None,
        "terminal_cif_audit": None,
        "terminal_legal": False,
        "valid_terminal": False,
        "legality": False,
        "terminal_failure": None,
        "future_execution": [],
    }


def retain_shared_candidates(
    attempts: Sequence[Mapping[str, Any]], *, maximum: int = MAX_CANDIDATES
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep first distinct legal actions; preserve every failed/duplicate slot."""

    if int(maximum) <= 0:
        raise ValueError("maximum candidate count must be positive")
    retained: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    for raw in attempts:
        item = dict(raw)
        signature = tuple(str(value) for value in item.get("active_action_tokens") or ())
        if item.get("active_legal") is not True:
            item["retention_status"] = "invalid"
        elif not signature:
            item["retention_status"] = "invalid"
            item["active_legal"] = False
            item["active_failure"] = item.get("active_failure") or "missing_action_tokens"
        elif signature in seen:
            item["retention_status"] = "duplicate"
        elif len(retained) >= int(maximum):
            item["retention_status"] = "over_limit"
        else:
            seen.add(signature)
            item["candidate_idx"] = len(retained)
            item["retention_status"] = "retained"
            retained.append(item)
        audit.append(item)
    if not retained or retained[0].get("candidate_source") != "noop":
        raise RuntimeError("the legal no-op must remain candidate zero")
    return retained, audit


def physics_quantization_audit(
    attempts: Sequence[Mapping[str, Any]],
    *,
    threshold: float = PHYSICS_TOKEN_CHANGE_THRESHOLD,
) -> dict[str, Any]:
    """Audit signed physics-token collisions without inspecting any value label.

    The scientific gate uses only direction-defined proposals.  A missing
    CHGNet state prediction or an exactly zero force/stress has no direction
    from which to define a candidate and is reported outside that denominator.
    Invalid geometry remains in the direction-defined denominator when its
    quantized transaction changed; validity and token resolution are separate.
    """

    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("physics token-change threshold must lie in [0,1]")
    noop = next(
        (
            tuple(str(value) for value in item.get("active_action_tokens") or ())
            for item in attempts
            if str(item.get("candidate_source")) == "noop"
        ),
        (),
    )
    physics = [
        item
        for item in attempts
        if str(item.get("candidate_source"))
        in {"physics_downhill", "physics_reverse"}
    ]
    direction_defined: list[Mapping[str, Any]] = []
    undefined = 0
    for item in physics:
        proposal = item.get("proposal")
        reason = "" if not isinstance(proposal, Mapping) else str(proposal.get("reason") or "")
        if not isinstance(proposal, Mapping) or reason.startswith("zero_"):
            undefined += 1
        else:
            direction_defined.append(item)
    changed = [
        item
        for item in direction_defined
        if tuple(str(value) for value in item.get("active_action_tokens") or ())
        and tuple(str(value) for value in item.get("active_action_tokens") or ())
        != noop
    ]
    noop_duplicates = len(direction_defined) - len(changed)
    signed_signatures = [
        tuple(str(value) for value in item.get("active_action_tokens") or ())
        for item in changed
    ]
    signed_pair_collision = int(
        len(signed_signatures) == 2 and signed_signatures[0] == signed_signatures[1]
    )
    denominator = len(direction_defined)
    rate = None if denominator == 0 else len(changed) / denominator
    return {
        "physics_proposal_slots": len(physics),
        "direction_defined_proposals": denominator,
        "undefined_direction_proposals": undefined,
        "quantized_changed_at_least_one_token": len(changed),
        "quantized_noop_duplicates": noop_duplicates,
        "quantized_noop_duplicate_rate": (
            None if denominator == 0 else noop_duplicates / denominator
        ),
        "quantized_token_change_rate": rate,
        "signed_pair_token_collision": signed_pair_collision,
        "accepted_legal_physics": sum(
            item.get("active_legal") is True for item in direction_defined
        ),
        "threshold": float(threshold),
        "threshold_passed": bool(rate is not None and rate >= float(threshold)),
        "outcomes_read": False,
        "energy_used": False,
        "retry_or_resample": False,
    }


def merge_physics_quantization_audits(
    audits: Sequence[Mapping[str, Any]],
    *,
    threshold: float = PHYSICS_TOKEN_CHANGE_THRESHOLD,
) -> dict[str, Any]:
    slots = sum(int(value.get("physics_proposal_slots", 0)) for value in audits)
    defined = sum(int(value.get("direction_defined_proposals", 0)) for value in audits)
    undefined = sum(int(value.get("undefined_direction_proposals", 0)) for value in audits)
    changed = sum(
        int(value.get("quantized_changed_at_least_one_token", 0))
        for value in audits
    )
    noop_duplicates = sum(
        int(value.get("quantized_noop_duplicates", 0)) for value in audits
    )
    signed_collisions = sum(
        int(
            value.get(
                "signed_pair_token_collision",
                value.get("signed_pair_token_collision_groups", 0),
            )
        )
        for value in audits
    )
    accepted = sum(int(value.get("accepted_legal_physics", 0)) for value in audits)
    rate = None if defined == 0 else changed / defined
    return {
        "physics_proposal_slots": slots,
        "direction_defined_proposals": defined,
        "undefined_direction_proposals": undefined,
        "quantized_changed_at_least_one_token": changed,
        "quantized_noop_duplicates": noop_duplicates,
        "quantized_noop_duplicate_rate": (
            None if defined == 0 else noop_duplicates / defined
        ),
        "quantized_token_change_rate": rate,
        "signed_pair_token_collision_groups": signed_collisions,
        "accepted_legal_physics": accepted,
        "threshold": float(threshold),
        "threshold_passed": bool(rate is not None and rate >= float(threshold)),
        "denominator_definition": (
            "finite_nonzero_CHGNet_force_or_stress_direction_proposal_slots"
        ),
        "undefined_directions_disclosed_outside_rate": True,
        "outcomes_read": False,
        "energy_used": False,
        "retry_or_resample": False,
    }


def attach_action_token_ids(
    attempts: Sequence[MutableMapping[str, Any]], tokenizer: Any
) -> None:
    """Attach deployed special-token ids after all proposals share one tokenizer."""

    unknown = getattr(tokenizer, "unk_token_id", None)
    for attempt in attempts:
        tokens = [str(value) for value in attempt.get("active_action_tokens") or ()]
        if not tokens:
            attempt["active_action_token_ids"] = []
            attempt["action_token_ids"] = []
            continue
        values = tokenizer.convert_tokens_to_ids(tokens)
        if isinstance(values, int):
            values = [values]
        ids = [int(value) for value in values]
        if len(ids) != len(tokens) or (
            unknown is not None and any(value == int(unknown) for value in ids)
        ):
            attempt["active_legal"] = False
            attempt["active_failure"] = "active_special_token_missing_from_tokenizer"
            attempt["active_action_token_ids"] = []
            attempt["action_token_ids"] = []
        else:
            attempt["active_action_token_ids"] = ids
            attempt["action_token_ids"] = ids


def source_failure_record(source: Mapping[str, Any], failure: str) -> dict[str, Any]:
    source_idx = int(source["source_row_idx"])
    stage = str(source.get("deployment_stage"))
    return {
        "schema": OUTPUT_SCHEMA,
        "group_idx": source_idx,
        "source_row_idx": source_idx,
        "sample_idx": int(source.get("sample_idx", source_idx)),
        "source_weight": float(source.get("source_weight", 1.0)),
        "source": {
            "source_row_idx": source_idx,
            "source_split": str(source.get("source_split") or "train"),
        },
        "stage": stage,
        "deployment_stage": stage,
        "status": "failed_source",
        "failure": str(failure),
        "state": {
            "source_idx": source_idx,
            "source_row_idx": source_idx,
            "source_weight": float(source.get("source_weight", 1.0)),
            "plan_state": dict(source.get("plan_state") or {}),
            "prompt": source.get("prompt"),
            "source_answer": None,
            "active_positions": source.get("active_positions"),
            "species_program": list(source.get("species_program") or ()),
            "deployment_stage": stage,
        },
        "state_answer": None,
        "candidate_attempts": [],
        "candidates": [],
        "candidate_count": 0,
        "outcomes_read": False,
        "energy_selection": False,
        "replacement": False,
    }


def _runtime_imports() -> dict[str, Any]:
    import torch

    from crystal_dlm.spad_generation import revise_spad_anchors, revise_spad_cell
    from scripts.sample_llada_dynamic_crystals import (
        build_dynamic_lightweight_constraints,
        graph_from_arrays,
        import_process_one,
        init_distributed,
        load_model_and_tokenizer,
        rank_path,
    )

    return {
        "torch": torch,
        "revise_spad_anchors": revise_spad_anchors,
        "revise_spad_cell": revise_spad_cell,
        "build_dynamic_lightweight_constraints": build_dynamic_lightweight_constraints,
        "graph_from_arrays": graph_from_arrays,
        "import_process_one": import_process_one,
        "init_distributed": init_distributed,
        "load_model_and_tokenizer": load_model_and_tokenizer,
        "rank_path": rank_path,
    }


class ReferenceDLMRuntime:
    """Batched frozen-reference executor for one complete SPAD transaction."""

    def __init__(self, model: Any, tokenizer: Any, device: Any, runtime: Mapping[str, Any]):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.runtime = runtime
        self._allowed: dict[int, list[list[int]]] = {}
        self.constraints = runtime["build_dynamic_lightweight_constraints"](
            tokenizer,
            duplicate_coordinate_mask=True,
            lattice_volume_mask=True,
            min_lattice_rad=1.0e-4,
            canonicalize_periodic_alias=True,
            pbc_min_distance_mask=True,
            pbc_min_distance_A=0.5,
            pbc_image_radius=2,
        )

    def _model_device(self) -> Any:
        return next(self.model.parameters()).device

    def execute_batch(
        self, requests: Sequence[Mapping[str, Any]]
    ) -> list[tuple[str, Any]]:
        if not requests:
            return []
        torch = self.runtime["torch"]
        num_atoms = int(requests[0]["plan_state"]["N"])
        kind = str(requests[0]["stage"]["transaction_kind"])
        if any(int(item["plan_state"]["N"]) != num_atoms for item in requests):
            raise ValueError("reference batch mixes N")
        if any(str(item["stage"]["transaction_kind"]) != kind for item in requests):
            raise ValueError("reference batch mixes transaction kinds")
        prompts = [str(item["prompt"]).rstrip() + "\n" for item in requests]
        encoded = self.tokenizer(
            prompts, add_special_tokens=False, padding=True, return_tensors="pt"
        )
        input_ids = encoded["input_ids"].to(self._model_device())
        attention = encoded["attention_mask"].to(self._model_device())
        gen_length = exact_body_token_count(num_atoms)
        body_ids: list[list[int]] = []
        for item in requests:
            values = [
                int(value)
                for value in self.tokenizer(
                    str(item["answer"]), add_special_tokens=False
                )["input_ids"]
            ]
            if len(values) != gen_length:
                raise ValueError("reference answer length differs from exact 7+4N")
            body_ids.append(values)
        complete = torch.cat(
            (
                input_ids,
                torch.tensor(body_ids, dtype=torch.long, device=input_ids.device),
            ),
            dim=1,
        )
        allowed = self._allowed.setdefault(
            num_atoms, exact_dynamic_schema_constraints(self.tokenizer, num_atoms)
        )
        seeds = [int(item["stage"]["common_random_seed"]) for item in requests]
        if kind == "cell":
            revised, logs = self.runtime["revise_spad_cell"](
                self.model,
                complete,
                prompt_length=int(input_ids.shape[1]),
                gen_length=gen_length,
                attention_mask=attention,
                temperature=TEMPERATURE,
                cfg_scale=0.0,
                remasking="low_confidence",
                mask_id=MASK_TOKEN_ID,
                allowed_token_ids_by_generation_pos=allowed,
                atom_count_grammar=None,
                lightweight_decoding_constraints=self.constraints,
                strict_geometry_fallback=True,
                sampling_seeds_by_batch=seeds,
            )
        else:
            revised, logs = self.runtime["revise_spad_anchors"](
                self.model,
                complete,
                prompt_length=int(input_ids.shape[1]),
                gen_length=gen_length,
                revision_slots_by_batch=[
                    [int(item["stage"]["anchor_slot"])] for item in requests
                ],
                attention_mask=attention,
                temperature=TEMPERATURE,
                cfg_scale=0.0,
                remasking="low_confidence",
                mask_id=MASK_TOKEN_ID,
                allowed_token_ids_by_generation_pos=allowed,
                atom_count_grammar=None,
                lightweight_decoding_constraints=self.constraints,
                suffix_visible=True,
                strict_pbc_no_legal_fallback=True,
                sampling_seeds_by_batch=seeds,
            )
        decoded = self.tokenizer.batch_decode(
            revised[:, int(input_ids.shape[1]) :],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        return [(str(answer), log) for answer, log in zip(decoded, logs, strict=True)]


def execute_reference_requests(
    requests: Sequence[Mapping[str, Any]],
    *,
    executor: ReferenceDLMRuntime,
    batch_size: int,
) -> dict[Any, dict[str, Any]]:
    """Batch by N/kind, while isolating per-row failures without replacement."""

    grouped: dict[tuple[int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for request in requests:
        grouped[
            (
                int(request["plan_state"]["N"]),
                str(request["stage"]["transaction_kind"]),
            )
        ].append(request)
    output: dict[Any, dict[str, Any]] = {}
    for key in sorted(grouped):
        group = grouped[key]
        for offset in range(0, len(group), int(batch_size)):
            chunk = group[offset : offset + int(batch_size)]
            try:
                values = executor.execute_batch(chunk)
                if len(values) != len(chunk):
                    raise RuntimeError("reference executor changed batch cardinality")
                for request, (answer, runtime_log) in zip(chunk, values, strict=True):
                    output[request["request_key"]] = {
                        "answer": answer,
                        "runtime_log": runtime_log,
                        "failure": None,
                    }
            except Exception as batch_error:  # noqa: BLE001
                for request in chunk:
                    try:
                        answer, runtime_log = executor.execute_batch([request])[0]
                        output[request["request_key"]] = {
                            "answer": answer,
                            "runtime_log": runtime_log,
                            "failure": None,
                            "batch_fallback": f"{type(batch_error).__name__}:{batch_error}",
                        }
                    except Exception as error:  # noqa: BLE001
                        output[request["request_key"]] = {
                            "answer": None,
                            "runtime_log": None,
                            "failure": f"{type(error).__name__}:{error}",
                        }
    return output


def _physics_predictions(
    chgnet: Any, arrays_rows: Sequence[Mapping[str, Any]], *, batch_size: int
) -> list[dict[str, Any] | None]:
    structures = [arrays_to_structure(dict(arrays)) for arrays in arrays_rows]
    raw: list[Any] = []
    for offset in range(0, len(structures), int(batch_size)):
        chunk = structures[offset : offset + int(batch_size)]
        try:
            values = chgnet.predict_structure(
                chunk, task="efsm", batch_size=int(batch_size)
            )
            if isinstance(values, Mapping):
                values = [values]
            if len(values) != len(chunk):
                raise RuntimeError("CHGNet changed batch cardinality")
            raw.extend(values)
        except Exception:  # noqa: BLE001
            for structure in chunk:
                try:
                    raw.append(chgnet.predict_structure(structure, task="efsm"))
                except Exception:  # noqa: BLE001
                    raw.append(None)
    output: list[dict[str, Any] | None] = []
    for arrays, value in zip(arrays_rows, raw, strict=True):
        try:
            if not isinstance(value, Mapping):
                raise ValueError("missing CHGNet state prediction")
            forces = np.asarray(value["f"], dtype=np.float64)
            stress = np.asarray(value["s"], dtype=np.float64)
            if forces.shape != (len(arrays["species"]), 3) or stress.shape != (3, 3):
                raise ValueError("CHGNet state force/stress shape changed")
            if not np.isfinite(forces).all() or not np.isfinite(stress).all():
                raise ValueError("CHGNet state force/stress is non-finite")
            output.append({"forces": forces, "stress": stress})
        except Exception:  # noqa: BLE001
            output.append(None)
    return output


def _physics_attempts(
    state_answer: str,
    *,
    plan_state: Mapping[str, Any],
    stage: Mapping[str, Any],
    prediction: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    if prediction is None:
        return [
            candidate_attempt(
                source=name,
                state_answer=state_answer,
                action_answer=None,
                stage=stage,
                legal=False,
                failure="state_CHGNet_force_or_stress_unavailable",
            )
            for name in ("physics_downhill", "physics_reverse")
        ]
    arrays = validate_answer_matches_plan(plan_state, state_answer)
    if stage["transaction_kind"] == "cell":
        proposals = propose_stress_lattice_transactions(
            arrays,
            prediction["stress"],
            strain_steps=STRAIN_ONE_BIN_SCAN_STEPS,
        )
    else:
        proposals = propose_force_site_transactions(
            arrays,
            int(stage["anchor_slot"]),
            prediction["forces"][int(stage["anchor_slot"])],
            step_sizes_A=FORCE_ONE_BIN_SCAN_STEPS_A,
        )
    attempts: list[dict[str, Any]] = []
    for name, proposal in zip(
        ("physics_downhill", "physics_reverse"), proposals, strict=True
    ):
        answer = "".join(str(value) for value in proposal.full_tokens)
        legal = proposal.status == "accepted"
        failure = None if legal else f"{proposal.status}:{proposal.reason}"
        if legal:
            try:
                validate_transaction_transition(
                    state_answer,
                    answer,
                    plan_state=plan_state,
                    stage=stage,
                )
            except Exception as error:  # noqa: BLE001
                legal = False
                failure = f"{type(error).__name__}:{error}"
        attempts.append(
            candidate_attempt(
                source=name,
                state_answer=state_answer,
                action_answer=answer,
                stage=stage,
                legal=legal,
                failure=failure,
                proposal=proposal,
            )
        )
    return attempts


def _prepare_context(
    source: Mapping[str, Any], body_row: Mapping[str, Any]
) -> dict[str, Any]:
    source_idx = int(source["source_row_idx"])
    if int(source.get("sample_idx", source_idx)) != source_idx:
        raise ValueError("source ledger sample_idx/source_row_idx changed")
    if body_row.get("parsed") is False:
        raise ValueError(
            f"reference body was not parsed: {body_row.get('reason')}:{body_row.get('message')}"
        )
    plan = source.get("plan_state")
    if not isinstance(plan, Mapping):
        raise ValueError("source ledger lacks plan_state")
    body_plan = body_row.get("plan_state")
    if isinstance(body_plan, Mapping) and dict(body_plan) != dict(plan):
        raise ValueError("reference body Plan differs from source ledger")
    answer = _answer_from_body(body_row)
    arrays = validate_answer_matches_plan(plan, answer)
    if len(tokenize_answer_text(answer)) != exact_body_token_count(int(plan["N"])):
        raise ValueError("reference body is not exact 7+4N")
    stages, anchors = resolved_deployment_stages(source, arrays["species"])
    return {
        "source": dict(source),
        "source_row_idx": source_idx,
        "sample_idx": source_idx,
        "plan_state": dict(plan),
        "prompt": str(source["prompt"]),
        "species_program": [str(value) for value in source["species_program"]],
        "generated_body_species": [str(value) for value in arrays["species"]],
        "anchors": anchors,
        "stages": stages,
        "active_stage_index": int(source["deployment_stage_index"]),
        "state_answer": answer,
        "prefix_execution": [],
        "failure": None,
    }


def _run_prefixes(
    contexts: Sequence[MutableMapping[str, Any]],
    *,
    executor: ReferenceDLMRuntime,
    batch_size: int,
) -> None:
    for stage_index in range(len(DEPLOYMENT_STAGES)):
        requests: list[dict[str, Any]] = []
        for context in contexts:
            if context.get("failure") or int(context["active_stage_index"]) <= stage_index:
                continue
            requests.append(
                {
                    "request_key": int(context["source_row_idx"]),
                    "prompt": context["prompt"],
                    "plan_state": context["plan_state"],
                    "answer": context["state_answer"],
                    "stage": context["stages"][stage_index],
                }
            )
        results = execute_reference_requests(
            requests, executor=executor, batch_size=int(batch_size)
        )
        for context in contexts:
            key = int(context["source_row_idx"])
            if key not in results:
                continue
            result = results[key]
            if result["failure"] is not None:
                context["failure"] = f"prefix_{DEPLOYMENT_STAGES[stage_index]}:{result['failure']}"
                continue
            try:
                _old, new = validate_transaction_transition(
                    str(context["state_answer"]),
                    str(result["answer"]),
                    plan_state=context["plan_state"],
                    stage=context["stages"][stage_index],
                )
            except Exception as error:  # noqa: BLE001
                context["failure"] = (
                    f"prefix_{DEPLOYMENT_STAGES[stage_index]}:"
                    f"{type(error).__name__}:{error}"
                )
                continue
            context["prefix_execution"].append(
                {
                    "stage": DEPLOYMENT_STAGES[stage_index],
                    "common_random_seed": int(
                        context["stages"][stage_index]["common_random_seed"]
                    ),
                    "active_action_tokens": new,
                    "runtime_log": result["runtime_log"],
                }
            )
            context["state_answer"] = str(result["answer"])


def _run_current_reference(
    contexts: Sequence[MutableMapping[str, Any]],
    *,
    executor: ReferenceDLMRuntime,
    batch_size: int,
) -> dict[int, dict[str, Any]]:
    requests = [
        {
            "request_key": int(context["source_row_idx"]),
            "prompt": context["prompt"],
            "plan_state": context["plan_state"],
            "answer": context["state_answer"],
            "stage": context["stages"][int(context["active_stage_index"])],
        }
        for context in contexts
        if not context.get("failure")
    ]
    return execute_reference_requests(
        requests, executor=executor, batch_size=int(batch_size)
    )


def _run_terminal_continuations(
    contexts: Sequence[MutableMapping[str, Any]],
    *,
    executor: ReferenceDLMRuntime,
    batch_size: int,
) -> None:
    max_active = len(DEPLOYMENT_STAGES) - 1
    for future_stage_index in range(1, len(DEPLOYMENT_STAGES)):
        requests: list[dict[str, Any]] = []
        for context in contexts:
            active = int(context["active_stage_index"])
            absolute_stage = active + future_stage_index
            if absolute_stage > max_active:
                continue
            for candidate in context.get("candidates", []):
                if candidate.get("terminal_failure") is not None:
                    continue
                requests.append(
                    {
                        "request_key": (
                            int(context["source_row_idx"]),
                            int(candidate["candidate_idx"]),
                        ),
                        "prompt": context["prompt"],
                        "plan_state": context["plan_state"],
                        "answer": candidate["terminal_answer"],
                        "stage": context["stages"][absolute_stage],
                    }
                )
        results = execute_reference_requests(
            requests, executor=executor, batch_size=int(batch_size)
        )
        for context in contexts:
            active = int(context["active_stage_index"])
            absolute_stage = active + future_stage_index
            if absolute_stage > max_active:
                continue
            stage = context["stages"][absolute_stage]
            for candidate in context.get("candidates", []):
                key = (int(context["source_row_idx"]), int(candidate["candidate_idx"]))
                if key not in results or candidate.get("terminal_failure") is not None:
                    continue
                result = results[key]
                if result["failure"] is not None:
                    candidate["terminal_failure"] = (
                        f"continuation_{stage['stage']}:{result['failure']}"
                    )
                    candidate["terminal_answer"] = None
                    continue
                try:
                    _old, new = validate_transaction_transition(
                        str(candidate["terminal_answer"]),
                        str(result["answer"]),
                        plan_state=context["plan_state"],
                        stage=stage,
                    )
                except Exception as error:  # noqa: BLE001
                    candidate["terminal_failure"] = (
                        f"continuation_{stage['stage']}:{type(error).__name__}:{error}"
                    )
                    candidate["terminal_answer"] = None
                    continue
                candidate["future_execution"].append(
                    {
                        "stage": str(stage["stage"]),
                        "common_random_seed": int(stage["common_random_seed"]),
                        "active_action_tokens": new,
                        "runtime_log": result["runtime_log"],
                    }
                )
                candidate["terminal_answer"] = str(result["answer"])


def _finalize_terminal_candidate(
    candidate: MutableMapping[str, Any],
    *,
    context: Mapping[str, Any],
    process_one: Any,
    graph_from_arrays: Callable[..., Any],
) -> None:
    if candidate.get("terminal_failure") is not None or not isinstance(
        candidate.get("terminal_answer"), str
    ):
        candidate["terminal_legal"] = False
        candidate["valid_terminal"] = False
        candidate["legality"] = False
        candidate["terminal_failure"] = (
            candidate.get("terminal_failure") or "missing_terminal_answer"
        )
        return
    try:
        arrays = validate_answer_matches_plan(
            context["plan_state"], str(candidate["terminal_answer"])
        )
        if list(arrays["species"]) != list(context["generated_body_species"]):
            raise ValueError("terminal continuation changed generated-body species order")
        _graph, cif = graph_from_arrays(dict(arrays), process_one)
        candidate["terminal_structure"] = {
            "num_atoms": int(arrays["num_atoms"]),
            "lengths": [float(value) for value in arrays["lengths"]],
            "angles": [float(value) for value in arrays["angles"]],
            "species": [str(value) for value in arrays["species"]],
            "frac_coords": [
                [float(value) for value in coordinate]
                for coordinate in arrays["frac_coords"]
            ],
        }
        # The shared value labeler accepts exactly one primary representation.
        # Keep the exact token answer primary and retain CIF as an audit copy.
        candidate["terminal_cif"] = None
        candidate["terminal_cif_audit"] = str(cif)
        candidate["terminal_legal"] = True
        candidate["valid_terminal"] = True
        candidate["legality"] = True
        candidate["terminal_failure"] = None
    except Exception as error:  # noqa: BLE001
        candidate["terminal_structure"] = None
        candidate["terminal_cif"] = None
        candidate["terminal_cif_audit"] = None
        candidate["terminal_legal"] = False
        candidate["valid_terminal"] = False
        candidate["legality"] = False
        candidate["terminal_failure"] = f"{type(error).__name__}:{error}"


def build_chunk(
    sources: Sequence[Mapping[str, Any]],
    body_by_source: Mapping[int, Mapping[str, Any]],
    *,
    executor: ReferenceDLMRuntime,
    chgnet: Any,
    process_one: Any,
    graph_from_arrays: Callable[..., Any],
    reference_batch_size: int,
    chgnet_batch_size: int,
) -> list[dict[str, Any]]:
    contexts: list[dict[str, Any]] = []
    failures: dict[int, dict[str, Any]] = {}
    for source in sources:
        source_idx = int(source["source_row_idx"])
        try:
            body = body_by_source[source_idx]
            contexts.append(_prepare_context(source, body))
        except Exception as error:  # noqa: BLE001
            failures[source_idx] = source_failure_record(
                source, f"prepare:{type(error).__name__}:{error}"
            )

    _run_prefixes(contexts, executor=executor, batch_size=int(reference_batch_size))
    for context in contexts:
        if context.get("failure"):
            failures[int(context["source_row_idx"])] = source_failure_record(
                context["source"], str(context["failure"])
            )

    live = [context for context in contexts if not context.get("failure")]
    current_results = _run_current_reference(
        live, executor=executor, batch_size=int(reference_batch_size)
    )
    state_arrays = [
        validate_answer_matches_plan(context["plan_state"], context["state_answer"])
        for context in live
    ]
    physics = _physics_predictions(
        chgnet, state_arrays, batch_size=int(chgnet_batch_size)
    )

    for context, prediction in zip(live, physics, strict=True):
        source_idx = int(context["source_row_idx"])
        stage = context["stages"][int(context["active_stage_index"])]
        state_answer = str(context["state_answer"])
        attempts = [
            candidate_attempt(
                source="noop",
                state_answer=state_answer,
                action_answer=state_answer,
                stage=stage,
                legal=True,
                failure=None,
            )
        ]
        reference = current_results.get(source_idx)
        if reference is None or reference.get("failure") is not None:
            attempts.append(
                candidate_attempt(
                    source="reference_dlm",
                    state_answer=state_answer,
                    action_answer=None,
                    stage=stage,
                    legal=False,
                    failure=(
                        "missing_reference_result"
                        if reference is None
                        else str(reference["failure"])
                    ),
                )
            )
        else:
            legal = True
            failure = None
            try:
                validate_transaction_transition(
                    state_answer,
                    str(reference["answer"]),
                    plan_state=context["plan_state"],
                    stage=stage,
                )
            except Exception as error:  # noqa: BLE001
                legal = False
                failure = f"{type(error).__name__}:{error}"
            attempts.append(
                candidate_attempt(
                    source="reference_dlm",
                    state_answer=state_answer,
                    action_answer=str(reference["answer"]),
                    stage=stage,
                    legal=legal,
                    failure=failure,
                    runtime_log=reference.get("runtime_log"),
                )
            )
        attempts.extend(
            _physics_attempts(
                state_answer,
                plan_state=context["plan_state"],
                stage=stage,
                prediction=prediction,
            )
        )
        attach_action_token_ids(attempts, executor.tokenizer)
        context["physics_quantization_audit"] = physics_quantization_audit(
            attempts
        )
        candidates, audit = retain_shared_candidates(attempts)
        for candidate in candidates:
            candidate["terminal_answer"] = str(candidate["action_answer"])
            candidate["terminal_failure"] = None
        context["candidate_attempts"] = audit
        context["candidates"] = candidates

    _run_terminal_continuations(
        live, executor=executor, batch_size=int(reference_batch_size)
    )

    output: dict[int, dict[str, Any]] = dict(failures)
    for context in live:
        for candidate in context["candidates"]:
            _finalize_terminal_candidate(
                candidate,
                context=context,
                process_one=process_one,
                graph_from_arrays=graph_from_arrays,
            )
        active_index = int(context["active_stage_index"])
        stage_name = DEPLOYMENT_STAGES[active_index]
        state = {
            "source_idx": int(context["source_row_idx"]),
            "source_row_idx": int(context["source_row_idx"]),
            "source_weight": float(
                context["source"].get("source_weight", 1.0)
            ),
            "plan_state": context["plan_state"],
            "prompt": context["prompt"],
            "source_answer": context["state_answer"],
            "active_positions": context["stages"][active_index][
                "active_positions"
            ],
            "species_program": context["species_program"],
            "deployment_stage": stage_name,
        }
        output[int(context["source_row_idx"])] = {
            "schema": OUTPUT_SCHEMA,
            "group_idx": int(context["source_row_idx"]),
            "source_row_idx": int(context["source_row_idx"]),
            "sample_idx": int(context["sample_idx"]),
            "source_weight": float(context["source"].get("source_weight", 1.0)),
            "status": "built",
            "failure": None,
            "source": {
                "source_row_idx": int(context["source_row_idx"]),
                "source_split": str(
                    context["source"].get("source_split") or "train"
                ),
                "species_program_source": str(
                    context["source"].get("source_marker") or "unspecified"
                ),
            },
            "stage": stage_name,
            "state": state,
            "plan_state": context["plan_state"],
            "prompt": context["prompt"],
            "species_program": context["species_program"],
            "generated_body_species": context["generated_body_species"],
            "generated_body_anchor_slots": context["anchors"],
            "deployment_stage_order": list(DEPLOYMENT_STAGES),
            "deployment_stage": stage_name,
            "deployment_stage_index": active_index,
            "active_stage": context["stages"][active_index],
            "prefix_execution": context["prefix_execution"],
            "state_answer": context["state_answer"],
            "future_stage_names": continuation_stage_names(
                context["stages"], active_index
            ),
            "candidate_attempts": context["candidate_attempts"],
            "candidates": context["candidates"],
            "candidate_count": len(context["candidates"]),
            "terminal_legal_count": sum(
                candidate["terminal_legal"] is True
                for candidate in context["candidates"]
            ),
            "shared_terminal_pool": True,
            "physics_quantization_audit": context[
                "physics_quantization_audit"
            ],
            "physics_support_rule": (
                "fixed_noop_reference_dlm_signed_physics_no_replacement"
            ),
            "future_seed_shared_across_candidates": True,
            "suffix_visible": True,
            "outcomes_read": False,
            "energy_selection": False,
            "replacement": False,
        }
    return [output[int(source["source_row_idx"])] for source in sources]


def _rank_path(output_dir: Path, filename: str, rank: int) -> Path:
    path = Path(filename)
    return output_dir / f"{path.stem}.rank{int(rank)}{path.suffix}"


def _rank_manifest(rows: Sequence[Mapping[str, Any]], elapsed: float, rank: int) -> dict[str, Any]:
    status = Counter(str(row["status"]) for row in rows)
    k_hist = Counter(int(row["candidate_count"]) for row in rows)
    stages = Counter(str(row["deployment_stage"]) for row in rows)
    physics = merge_physics_quantization_audits(
        [
            value
            for row in rows
            for value in [row.get("physics_quantization_audit")]
            if isinstance(value, Mapping)
        ]
    )
    return {
        "schema": MANIFEST_SCHEMA,
        "rank": int(rank),
        "sources": len(rows),
        "statuses": dict(sorted(status.items())),
        "stages": dict(sorted(stages.items())),
        "candidate_count_histogram": {
            str(key): int(value) for key, value in sorted(k_hist.items())
        },
        "retained_candidates": sum(int(row["candidate_count"]) for row in rows),
        "terminal_legal_candidates": sum(
            int(row.get("terminal_legal_count", 0)) for row in rows
        ),
        "physics_quantization": physics,
        "outcomes_read": False,
        "energy_selection": False,
        "replacement": False,
        "elapsed_sec": float(elapsed),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-ledger", type=Path, required=True)
    parser.add_argument("--reference-body", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--crysllmgen-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-sources", type=int, default=FORMAL_MP20_TRAIN_ROWS)
    parser.add_argument("--source-batch-size", type=int, default=128)
    parser.add_argument("--reference-batch-size", type=int, default=8)
    parser.add_argument("--chgnet-batch-size", type=int, default=16)
    parser.add_argument("--formal", action="store_true")
    args = parser.parse_args()
    if int(args.expected_sources) <= 0:
        raise ValueError("--expected-sources must be positive")
    if args.formal and int(args.expected_sources) != FORMAL_MP20_TRAIN_ROWS:
        raise ValueError("formal mode requires --expected-sources 27136")
    if min(
        int(args.source_batch_size),
        int(args.reference_batch_size),
        int(args.chgnet_batch_size),
    ) <= 0:
        raise ValueError("all batch sizes must be positive")

    runtime = _runtime_imports()
    dist_info = runtime["init_distributed"]()
    rank = int(dist_info["rank"])
    world_size = int(dist_info["world_size"])
    is_main = bool(dist_info["is_main"])
    torch = runtime["torch"]
    if world_size != 4:
        raise RuntimeError("full MP20 terminal action generation requires four GPUs")
    if not bool(dist_info["distributed"]):
        raise RuntimeError("formal terminal action generation requires torch.distributed")
    formal_gate_failed = False
    if is_main:
        if args.output_dir.exists():
            raise FileExistsError(args.output_dir)
        args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.distributed.barrier()

    sources = list(iter_jsonl(args.source_ledger.resolve()))
    if len(sources) != int(args.expected_sources):
        raise ValueError("source ledger denominator differs from --expected-sources")
    if [int(row.get("source_row_idx", -1)) for row in sources] != list(
        range(int(args.expected_sources))
    ):
        raise ValueError("source ledger source_row_idx is not contiguous")
    if any(float(row.get("source_weight", -1.0)) != 1.0 for row in sources):
        raise ValueError("source ledger source weights changed")
    if any(row.get("outcomes_read") is not False for row in sources):
        raise ValueError("source ledger is not outcome blind")
    bodies = index_reference_bodies(
        iter_jsonl(args.reference_body.resolve()),
        expected_sources=int(args.expected_sources),
    )

    model, tokenizer = runtime["load_model_and_tokenizer"](
        str(args.model_path), str(args.checkpoint_path), dist_info["device"]
    )
    tokenizer_report = validate_dynamic_tokenizer_contract(
        tokenizer, mask_token_id=MASK_TOKEN_ID
    )
    executor = ReferenceDLMRuntime(
        model, tokenizer, dist_info["device"], runtime
    )
    from chgnet.model.model import CHGNet

    chgnet = CHGNet.load(
        use_device=str(dist_info["device"]),
        check_cuda_mem=False,
        verbose=False,
    )
    process_one = runtime["import_process_one"](args.crysllmgen_dir)

    start, stop = contiguous_shard(len(sources), world_size, rank)
    assigned = sources[start:stop]
    rank_output = _rank_path(args.output_dir, "terminal_actions.jsonl", rank)
    local_summary_rows: list[dict[str, Any]] = []
    started = time.time()
    with rank_output.open("x", encoding="utf-8", newline="\n") as handle:
        for offset in range(0, len(assigned), int(args.source_batch_size)):
            chunk = assigned[offset : offset + int(args.source_batch_size)]
            rows = build_chunk(
                chunk,
                bodies,
                executor=executor,
                chgnet=chgnet,
                process_one=process_one,
                graph_from_arrays=runtime["graph_from_arrays"],
                reference_batch_size=int(args.reference_batch_size),
                chgnet_batch_size=int(args.chgnet_batch_size),
            )
            if [int(row["source_row_idx"]) for row in rows] != [
                int(row["source_row_idx"]) for row in chunk
            ]:
                raise RuntimeError("chunk output changed source order")
            for row in rows:
                write_jsonl_row(handle, row)
                local_summary_rows.append(
                    {
                        "status": row["status"],
                        "deployment_stage": row["deployment_stage"],
                        "candidate_count": int(row["candidate_count"]),
                        "terminal_legal_count": int(
                            row.get("terminal_legal_count", 0)
                        ),
                        "physics_quantization_audit": row.get(
                            "physics_quantization_audit"
                        ),
                    }
                )
    rank_report = _rank_manifest(
        local_summary_rows, time.time() - started, rank
    )
    rank_report["source_start"] = start
    rank_report["source_stop"] = stop
    _rank_path(args.output_dir, "manifest.json", rank).write_text(
        json.dumps(rank_report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    torch.distributed.barrier()

    if is_main:
        merged_path = args.output_dir / "terminal_actions.jsonl"
        total_rows = 0
        expected_idx = 0
        rank_reports: list[dict[str, Any]] = []
        with merged_path.open("x", encoding="utf-8", newline="\n") as merged:
            for source_rank in range(world_size):
                report = json.loads(
                    _rank_path(args.output_dir, "manifest.json", source_rank).read_text(
                        encoding="utf-8"
                    )
                )
                rank_reports.append(report)
                with _rank_path(
                    args.output_dir, "terminal_actions.jsonl", source_rank
                ).open("r", encoding="utf-8") as source_handle:
                    for line in source_handle:
                        if not line.strip():
                            continue
                        row = json.loads(line)
                        if int(row["source_row_idx"]) != expected_idx:
                            raise RuntimeError("distributed merge changed source coverage")
                        merged.write(line)
                        expected_idx += 1
                        total_rows += 1
        if total_rows != int(args.expected_sources):
            raise RuntimeError("distributed merge changed source denominator")
        status: Counter[str] = Counter()
        stages: Counter[str] = Counter()
        k_hist: Counter[int] = Counter()
        for report in rank_reports:
            status.update({str(k): int(v) for k, v in report["statuses"].items()})
            stages.update({str(k): int(v) for k, v in report["stages"].items()})
            k_hist.update(
                {int(k): int(v) for k, v in report["candidate_count_histogram"].items()}
            )
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "sources": total_rows,
            "formal": bool(args.formal),
            "world_size": world_size,
            "distributed_contiguous_shards": True,
            "statuses": dict(sorted(status.items())),
            "stages": dict(sorted(stages.items())),
            "candidate_count_histogram": {
                str(key): int(value) for key, value in sorted(k_hist.items())
            },
            "retained_candidates": sum(
                int(report["retained_candidates"]) for report in rank_reports
            ),
            "terminal_legal_candidates": sum(
                int(report["terminal_legal_candidates"]) for report in rank_reports
            ),
            "reference_body_rows_present": len(bodies),
            "failed_sources_retained": int(status.get("failed_source", 0)),
            "llama_species_program_controls_deployment_order": True,
            "deployment_chain": list(DEPLOYMENT_STAGES),
            "state_is_deployment_prefix_matched": True,
            "candidate_future_seeds_shared": True,
            "shared_terminal_pool_for_A_and_B": True,
            "maximum_candidates": MAX_CANDIDATES,
            "candidate_sources": [
                "noop",
                "reference_dlm",
                "physics_downhill",
                "physics_reverse",
            ],
            "temperature": TEMPERATURE,
            "actual_token_quantization": actual_token_quantization_contract(),
            "physics_quantization": merge_physics_quantization_audits(
                [report["physics_quantization"] for report in rank_reports]
            ),
            "physics_support_fixed_before_labels": True,
            "physics_retry_or_resample": False,
            "candidate_energy_read": False,
            "outcomes_read": False,
            "energy_selection": False,
            "replacement": False,
            "tokenizer": tokenizer_report,
            "elapsed_sec": max(float(report["elapsed_sec"]) for report in rank_reports),
        }
        (args.output_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        formal_gate_failed = bool(args.formal) and not manifest[
            "physics_quantization"
        ][
            "threshold_passed"
        ]
        if formal_gate_failed:
            (args.output_dir / "_PHYSICS_PREFLIGHT_FAILED").touch()
        else:
            (args.output_dir / "_SUCCESS").touch()
        print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    gate_tensor = torch.tensor(
        int(formal_gate_failed), dtype=torch.int64, device=dist_info["device"]
    )
    torch.distributed.broadcast(gate_tensor, src=0)
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()
    if int(gate_tensor.cpu().item()) != 0:
        raise RuntimeError(
            "formal physics token-change rate is below the frozen 85% threshold"
        )


if __name__ == "__main__":
    main()
