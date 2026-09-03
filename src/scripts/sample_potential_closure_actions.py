#!/usr/bin/env python3
"""Sample deterministic variable-K complete transactions for potential closure.

The sampler never reads an energy or outcome.  For every frozen closure state it
keeps the legal no-op first, optionally audits the fixed MP20 teacher, and then
keeps the first distinct legal DLM transactions in request order.  Row-local
sampling seeds make the result independent of distributed/batch packing.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
import json
from pathlib import Path
import time
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.distributed as dist
from tqdm import tqdm

from crystal_dlm.fixed_slot import MASK_TOKEN_ID
from crystal_dlm.r5_dynamic_length import (
    exact_body_token_count,
    exact_dynamic_schema_constraints,
    validate_answer_matches_plan,
    validate_dynamic_tokenizer_contract,
)
from crystal_dlm.spad_generation import revise_spad_anchors, revise_spad_cell
from crystal_dlm.spad_program import LATTICE_POSITIONS, coordinate_positions
from scripts.sample_llada_dynamic_crystals import (
    build_dynamic_lightweight_constraints,
    graph_from_arrays,
    import_process_one,
    init_distributed,
    load_model_and_tokenizer,
    rank_path,
)


STATE_SCHEMA = "potential_closure_state_v1"
GROUP_SCHEMA = "potential_closure_candidate_group_v1"
CANDIDATE_SCHEMA = "potential_closure_candidate_v1"
ATTEMPT_SCHEMA = "potential_closure_proposal_attempt_v1"
MANIFEST_SCHEMA = "potential_closure_action_manifest_v1"
EXPECTED_GROUPS = 2048
EXPECTED_STRATA = (
    "mp20_clean_cell",
    "mp20_clean_site",
    "on_policy_cell",
    "on_policy_site",
)
TEMPERATURE = 0.7
MAX_PROPOSAL_ATTEMPTS = 8
MAX_CANDIDATES = 4
MAX_DLM_DISTINCT = {"mp20_clean": 2, "on_policy": 3}


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number} is not an object")
            yield value


def request_seed(base_seed: int, group_idx: int, proposal_attempt: int) -> int:
    """Return a stable request-local seed independent of batch/rank ordering."""

    if int(group_idx) < 0:
        raise ValueError("group_idx must be nonnegative")
    if not 1 <= int(proposal_attempt) <= MAX_PROPOSAL_ATTEMPTS:
        raise ValueError("proposal attempt must be in 1..8")
    modulus = 2**63 - 1
    return int(
        (
            int(base_seed)
            + (int(group_idx) + 1) * 1_000_003
            + int(proposal_attempt) * 1_000_000_007
        )
        % modulus
    )


def differing_positions(
    source_token_ids: Sequence[int], candidate_token_ids: Sequence[int]
) -> tuple[int, ...]:
    if len(source_token_ids) != len(candidate_token_ids):
        raise ValueError("candidate changed exact body length")
    return tuple(
        index
        for index, (source, candidate) in enumerate(
            zip(source_token_ids, candidate_token_ids, strict=True)
        )
        if int(source) != int(candidate)
    )


def active_block_escape_positions(
    source_token_ids: Sequence[int],
    candidate_token_ids: Sequence[int],
    active_positions: Sequence[int],
) -> tuple[int, ...]:
    active = {int(value) for value in active_positions}
    return tuple(
        position
        for position in differing_positions(source_token_ids, candidate_token_ids)
        if position not in active
    )


def first_unique_legal_attempts(
    attempts: Sequence[Mapping[str, Any]],
    *,
    existing_signatures: Iterable[Sequence[int]] = (),
    limit: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep the first legal unseen actions and audit every examined attempt.

    This pure helper is also used by tests to pin first-in-request-order
    semantics.  Invalid and duplicate attempts remain in the returned audit.
    """

    if int(limit) < 0:
        raise ValueError("limit must be nonnegative")
    known = {tuple(int(value) for value in signature) for signature in existing_signatures}
    retained: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for attempt in list(attempts)[:MAX_PROPOSAL_ATTEMPTS]:
        item = dict(attempt)
        signature = tuple(int(value) for value in item.get("action_token_ids", ()))
        if item.get("valid_action") is not True:
            item.update({"retention_status": "invalid", "retained": False})
        elif signature in known:
            item.update({"retention_status": "duplicate", "retained": False})
        elif len(retained) >= int(limit):
            item.update({"retention_status": "not_needed", "retained": False})
        else:
            known.add(signature)
            item.update({"retention_status": "retained", "retained": True})
            retained.append(dict(item))
        audit.append(item)
    return retained, audit


def _model_device(model: Any) -> torch.device:
    return next(model.parameters()).device


def _tokenize_answer(tokenizer: Any, answer: str, expected_length: int) -> list[int]:
    token_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
    values = [int(value) for value in token_ids]
    if len(values) != int(expected_length):
        raise ValueError(
            f"answer token count {len(values)} does not equal exact {expected_length}"
        )
    return values


def _token_strings(tokenizer: Any, token_ids: Sequence[int]) -> list[str]:
    values = tokenizer.convert_ids_to_tokens([int(value) for value in token_ids])
    if isinstance(values, str):
        values = [values]
    return [str(value) for value in values]


def _validate_state(row: Mapping[str, Any]) -> dict[str, Any]:
    if row.get("schema") != STATE_SCHEMA:
        raise ValueError("state schema changed")
    group_idx = int(row["group_idx"])
    stratum = str(row["stratum"])
    if stratum not in EXPECTED_STRATA:
        raise ValueError(f"unknown state stratum {stratum}")
    source_domain = str(row["source_domain"])
    transaction_kind = str(row["transaction_kind"])
    if stratum != f"{source_domain}_{transaction_kind}":
        raise ValueError("stratum/domain/transaction mismatch")
    if source_domain not in MAX_DLM_DISTINCT:
        raise ValueError("unknown source domain")
    if transaction_kind not in {"cell", "site"}:
        raise ValueError("unknown transaction kind")
    plan = row.get("plan_state")
    if not isinstance(plan, Mapping):
        raise ValueError("state lacks Plan")
    num_atoms = int(plan["N"])
    source_answer = row.get("source_answer")
    prompt = row.get("prompt")
    if not isinstance(source_answer, str) or not isinstance(prompt, str):
        raise ValueError("state lacks prompt/source answer")
    source_arrays = validate_answer_matches_plan(plan, source_answer)
    active = tuple(int(value) for value in row.get("active_positions", ()))
    if transaction_kind == "cell":
        expected_active = tuple(int(value) for value in LATTICE_POSITIONS)
        if row.get("backfill_slot") is not None:
            raise ValueError("cell state unexpectedly has a site slot")
    else:
        slot = int(row["backfill_slot"])
        expected_active = tuple(int(value) for value in coordinate_positions(slot))
    if active != expected_active:
        raise ValueError("state active transaction changed")
    if int(row.get("maximum_proposal_attempts", -1)) != MAX_PROPOSAL_ATTEMPTS:
        raise ValueError("state proposal-attempt contract changed")
    if float(row.get("proposal_temperature", -1.0)) != TEMPERATURE:
        raise ValueError("state proposal-temperature contract changed")
    if row.get("outcomes_read") is not False:
        raise ValueError("closure state is not outcome blind")
    return {
        **dict(row),
        "group_idx": group_idx,
        "stratum": stratum,
        "source_domain": source_domain,
        "transaction_kind": transaction_kind,
        "plan_state": dict(plan),
        "num_atoms": num_atoms,
        "prompt": prompt.rstrip() + "\n",
        "source_answer": source_answer,
        "source_arrays": source_arrays,
        "active_positions": active,
    }


def _attempt_base(
    task: Mapping[str, Any],
    *,
    attempt_kind: str,
    proposal_attempt: int | None,
    answer: str | None,
) -> dict[str, Any]:
    return {
        "schema": ATTEMPT_SCHEMA,
        "group_idx": int(task["group_idx"]),
        "stratum": str(task["stratum"]),
        "source_domain": str(task["source_domain"]),
        "transaction_kind": str(task["transaction_kind"]),
        "attempt_kind": str(attempt_kind),
        "proposal_attempt": (
            None if proposal_attempt is None else int(proposal_attempt)
        ),
        "answer": answer,
        "active_positions": list(task["active_positions"]),
        "action_tokens": [],
        "action_token_ids": [],
        "differing_positions": [],
        "valid_action": False,
        "graphable": False,
        "failure": None,
        "retained": False,
        "retention_status": None,
        "outcomes_read": False,
        "energy_selection": False,
    }


def inspect_attempt(
    task: Mapping[str, Any],
    *,
    tokenizer: Any,
    process_one: Any,
    source_token_ids: Sequence[int],
    answer: str,
    attempt_kind: str,
    proposal_attempt: int | None,
    revision_log: Any,
    known_legal_actions: Mapping[tuple[int, ...], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Validate one complete action without applying selection or energy logic."""

    record = _attempt_base(
        task,
        attempt_kind=attempt_kind,
        proposal_attempt=proposal_attempt,
        answer=answer,
    )
    record["revision_log"] = revision_log
    try:
        token_ids = _tokenize_answer(
            tokenizer,
            answer,
            exact_body_token_count(int(task["num_atoms"])),
        )
        active = tuple(int(value) for value in task["active_positions"])
        action_ids = [token_ids[position] for position in active]
        changed = differing_positions(source_token_ids, token_ids)
        escaped = active_block_escape_positions(
            source_token_ids,
            token_ids,
            active,
        )
        record.update(
            {
                "action_token_ids": action_ids,
                "action_tokens": _token_strings(tokenizer, action_ids),
                "differing_positions": list(changed),
            }
        )
        if escaped:
            raise ValueError(
                "active_block_escape:" + ",".join(str(value) for value in escaped)
            )
        known = (known_legal_actions or {}).get(tuple(action_ids))
        if known is not None:
            record.update(
                {
                    "valid_action": True,
                    "graphable": True,
                    "cif": str(known["cif"]),
                    "failure": None,
                    "duplicate_prevalidated": True,
                }
            )
            return record
        arrays = validate_answer_matches_plan(task["plan_state"], answer)
        if list(arrays["species"]) != list(task["source_arrays"]["species"]):
            raise ValueError("transaction changed atom-type order")
        _graph, cif = graph_from_arrays(arrays, process_one)
        record.update(
            {
                "valid_action": True,
                "graphable": True,
                "cif": cif,
                "failure": None,
                "duplicate_prevalidated": False,
            }
        )
    except Exception as exc:
        record.update(
            {
                "valid_action": False,
                "graphable": False,
                "cif": None,
                "failure": f"{type(exc).__name__}:{exc}",
            }
        )
    return record


def _candidate_from_attempt(
    attempt: Mapping[str, Any], candidate_idx: int
) -> dict[str, Any]:
    if attempt.get("valid_action") is not True:
        raise ValueError("cannot retain an invalid action")
    return {
        "schema": CANDIDATE_SCHEMA,
        "candidate_idx": int(candidate_idx),
        "candidate_source": str(attempt["attempt_kind"]),
        "candidate_kind": str(attempt["attempt_kind"]),
        "proposal_attempt": attempt.get("proposal_attempt"),
        "answer": str(attempt["answer"]),
        "action_tokens": list(attempt["action_tokens"]),
        "action_token_ids": list(attempt["action_token_ids"]),
        "active_positions": list(attempt["active_positions"]),
        "differing_positions": list(attempt["differing_positions"]),
        "valid_action": True,
        "graphable": True,
        "cif": str(attempt["cif"]),
        "revision_log": attempt.get("revision_log"),
    }


@dataclass
class GroupAccumulator:
    task: dict[str, Any]
    source_token_ids: list[int]
    candidates: list[dict[str, Any]] = field(default_factory=list)
    attempts: list[dict[str, Any]] = field(default_factory=list)
    signatures: dict[tuple[int, ...], int] = field(default_factory=dict)
    retained_dlm_distinct: int = 0

    @property
    def target_dlm_distinct(self) -> int:
        return int(MAX_DLM_DISTINCT[str(self.task["source_domain"])])

    @property
    def needs_dlm_proposal(self) -> bool:
        return (
            self.retained_dlm_distinct < self.target_dlm_distinct
            and len(self.candidates) < MAX_CANDIDATES
        )

    @property
    def known_legal_actions(self) -> dict[tuple[int, ...], dict[str, Any]]:
        return {
            tuple(int(value) for value in candidate["action_token_ids"]): candidate
            for candidate in self.candidates
        }

    def add(self, attempt: dict[str, Any], *, counts_as_dlm: bool) -> bool:
        signature = tuple(int(value) for value in attempt.get("action_token_ids", ()))
        if attempt.get("valid_action") is not True:
            attempt.update({"retention_status": "invalid", "retained": False})
            self.attempts.append(attempt)
            return False
        if signature in self.signatures:
            attempt.update(
                {
                    "retention_status": "duplicate",
                    "retained": False,
                    "duplicate_of_candidate_idx": int(self.signatures[signature]),
                }
            )
            self.attempts.append(attempt)
            return False
        if len(self.candidates) >= MAX_CANDIDATES or (
            counts_as_dlm and self.retained_dlm_distinct >= self.target_dlm_distinct
        ):
            attempt.update({"retention_status": "not_needed", "retained": False})
            self.attempts.append(attempt)
            return False
        candidate_idx = len(self.candidates)
        candidate = _candidate_from_attempt(attempt, candidate_idx)
        self.signatures[signature] = candidate_idx
        self.candidates.append(candidate)
        if counts_as_dlm:
            self.retained_dlm_distinct += 1
        attempt.update(
            {
                "retention_status": "retained",
                "retained": True,
                "candidate_idx": candidate_idx,
            }
        )
        self.attempts.append(attempt)
        return True

    def finish(self, executed_proposal_attempts: int) -> dict[str, Any]:
        count = len(self.candidates)
        if count > MAX_CANDIDATES:
            raise RuntimeError("candidate K exceeds four")
        if not self.candidates or self.candidates[0]["candidate_kind"] != "noop":
            raise RuntimeError("legal no-op is not candidate zero")
        if self.candidates[0]["valid_action"] is not True:
            raise RuntimeError("candidate-zero no-op is not legal")
        return {
            "schema": GROUP_SCHEMA,
            "state_schema": STATE_SCHEMA,
            "group_idx": int(self.task["group_idx"]),
            "stratum": str(self.task["stratum"]),
            "source_domain": str(self.task["source_domain"]),
            "transaction_kind": str(self.task["transaction_kind"]),
            "source_sample_idx": int(self.task["source_sample_idx"]),
            "source_row_idx": int(self.task["source_row_idx"]),
            "prompt": str(self.task["prompt"]),
            "plan_state": dict(self.task["plan_state"]),
            "species_program": list(self.task["species_program"]),
            "source_answer": str(self.task["source_answer"]),
            "clean_teacher_answer": str(self.task["clean_teacher_answer"]),
            "active_positions": list(self.task["active_positions"]),
            "transaction_length": len(self.task["active_positions"]),
            "backfill_slot": self.task.get("backfill_slot"),
            "candidates": list(self.candidates),
            "candidate_count": count,
            "variable_K": count if 2 <= count <= MAX_CANDIDATES else None,
            "trainable_variable_K": bool(2 <= count <= MAX_CANDIDATES),
            "target_dlm_distinct": self.target_dlm_distinct,
            "retained_dlm_distinct": int(self.retained_dlm_distinct),
            "missing_dlm_slots": int(
                self.target_dlm_distinct - self.retained_dlm_distinct
            ),
            "proposal_attempts_executed": int(executed_proposal_attempts),
            "maximum_proposal_attempts": MAX_PROPOSAL_ATTEMPTS,
            "temperature": TEMPERATURE,
            "candidate_retention": "first_distinct_legal_in_request_order",
            "outcomes_read": False,
            "energy_selection": False,
        }


def _initialize_accumulator(
    task: dict[str, Any], tokenizer: Any, process_one: Any
) -> GroupAccumulator:
    source_ids = _tokenize_answer(
        tokenizer,
        str(task["source_answer"]),
        exact_body_token_count(int(task["num_atoms"])),
    )
    accumulator = GroupAccumulator(task=task, source_token_ids=source_ids)
    noop = inspect_attempt(
        task,
        tokenizer=tokenizer,
        process_one=process_one,
        source_token_ids=source_ids,
        answer=str(task["source_answer"]),
        attempt_kind="noop",
        proposal_attempt=None,
        revision_log=[],
    )
    if noop.get("valid_action") is not True:
        raise RuntimeError(
            f"group {task['group_idx']} no-op is not legal: {noop.get('failure')}"
        )
    if not accumulator.add(noop, counts_as_dlm=False):
        raise RuntimeError("legal no-op was not retained")

    if task["source_domain"] == "mp20_clean":
        teacher = inspect_attempt(
            task,
            tokenizer=tokenizer,
            process_one=process_one,
            source_token_ids=source_ids,
            answer=str(task["clean_teacher_answer"]),
            attempt_kind="clean_teacher",
            proposal_attempt=None,
            revision_log=[],
            known_legal_actions=accumulator.known_legal_actions,
        )
        accumulator.add(teacher, counts_as_dlm=False)
    return accumulator


def _sample_batch(
    batch: Sequence[dict[str, Any]],
    *,
    model: Any,
    tokenizer: Any,
    process_one: Any,
    base_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not batch:
        return [], []
    num_atoms = int(batch[0]["num_atoms"])
    transaction_kind = str(batch[0]["transaction_kind"])
    if any(int(task["num_atoms"]) != num_atoms for task in batch):
        raise ValueError("batch mixes atom counts")
    if any(str(task["transaction_kind"]) != transaction_kind for task in batch):
        raise ValueError("batch mixes transaction kinds")

    gen_length = exact_body_token_count(num_atoms)
    encoded = tokenizer(
        [str(task["prompt"]) for task in batch],
        add_special_tokens=False,
        padding=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(_model_device(model))
    attention_mask = encoded["attention_mask"].to(_model_device(model))
    source_ids = [
        _tokenize_answer(tokenizer, str(task["source_answer"]), gen_length)
        for task in batch
    ]
    complete = torch.cat(
        (
            input_ids,
            torch.tensor(source_ids, dtype=torch.long, device=input_ids.device),
        ),
        dim=1,
    )
    allowed = exact_dynamic_schema_constraints(tokenizer, num_atoms)
    constraints = build_dynamic_lightweight_constraints(
        tokenizer,
        duplicate_coordinate_mask=True,
        lattice_volume_mask=True,
        min_lattice_rad=1.0e-4,
        canonicalize_periodic_alias=True,
        pbc_min_distance_mask=True,
        pbc_min_distance_A=0.5,
        pbc_image_radius=2,
    )
    accumulators = [
        _initialize_accumulator(task, tokenizer, process_one) for task in batch
    ]
    executed = [0 for _ in batch]

    for proposal_attempt in range(1, MAX_PROPOSAL_ATTEMPTS + 1):
        pending = [
            index
            for index, accumulator in enumerate(accumulators)
            if accumulator.needs_dlm_proposal
        ]
        if not pending:
            break
        indices = torch.tensor(pending, dtype=torch.long, device=input_ids.device)
        sub_complete = complete.index_select(0, indices)
        sub_attention = attention_mask.index_select(0, indices)
        sampling_seeds = [
            request_seed(base_seed, int(batch[index]["group_idx"]), proposal_attempt)
            for index in pending
        ]
        if transaction_kind == "cell":
            revised, revision_logs = revise_spad_cell(
                model,
                sub_complete,
                prompt_length=int(input_ids.shape[1]),
                gen_length=gen_length,
                attention_mask=sub_attention,
                temperature=TEMPERATURE,
                cfg_scale=0.0,
                remasking="low_confidence",
                mask_id=MASK_TOKEN_ID,
                allowed_token_ids_by_generation_pos=allowed,
                atom_count_grammar=None,
                lightweight_decoding_constraints=constraints,
                strict_geometry_fallback=True,
                sampling_seeds_by_batch=sampling_seeds,
            )
        else:
            revised, revision_logs = revise_spad_anchors(
                model,
                sub_complete,
                prompt_length=int(input_ids.shape[1]),
                gen_length=gen_length,
                revision_slots_by_batch=[
                    [int(batch[index]["backfill_slot"])] for index in pending
                ],
                attention_mask=sub_attention,
                temperature=TEMPERATURE,
                cfg_scale=0.0,
                remasking="low_confidence",
                mask_id=MASK_TOKEN_ID,
                allowed_token_ids_by_generation_pos=allowed,
                atom_count_grammar=None,
                lightweight_decoding_constraints=constraints,
                suffix_visible=True,
                strict_pbc_no_legal_fallback=True,
                sampling_seeds_by_batch=sampling_seeds,
            )
        decoded = tokenizer.batch_decode(
            revised[:, int(input_ids.shape[1]) :],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        for local_index, answer, revision_log in zip(
            pending, decoded, revision_logs, strict=True
        ):
            executed[local_index] = proposal_attempt
            attempt = inspect_attempt(
                batch[local_index],
                tokenizer=tokenizer,
                process_one=process_one,
                source_token_ids=source_ids[local_index],
                answer=str(answer),
                attempt_kind="dlm_proposal",
                proposal_attempt=proposal_attempt,
                revision_log=revision_log,
                known_legal_actions=accumulators[
                    local_index
                ].known_legal_actions,
            )
            attempt["sampling_seed"] = int(sampling_seeds[pending.index(local_index)])
            accumulators[local_index].add(attempt, counts_as_dlm=True)

    groups = [
        accumulator.finish(executed[index])
        for index, accumulator in enumerate(accumulators)
    ]
    attempts = [
        attempt for accumulator in accumulators for attempt in accumulator.attempts
    ]
    return groups, attempts


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def _manifest(
    groups: Sequence[Mapping[str, Any]],
    attempts: Sequence[Mapping[str, Any]],
    *,
    world_size: int,
    elapsed_sec: float,
    seed: int,
) -> dict[str, Any]:
    candidate_histogram = Counter(int(group["candidate_count"]) for group in groups)
    status_histogram = Counter(str(item["retention_status"]) for item in attempts)
    attempt_kind_histogram = Counter(str(item["attempt_kind"]) for item in attempts)
    strata = Counter(str(group["stratum"]) for group in groups)
    return {
        "schema": MANIFEST_SCHEMA,
        "groups": len(groups),
        "groups_per_stratum": dict(sorted(strata.items())),
        "candidate_count_histogram": {
            str(key): int(value) for key, value in sorted(candidate_histogram.items())
        },
        "trainable_variable_K_groups": sum(
            group["trainable_variable_K"] is True for group in groups
        ),
        "groups_below_K2": sum(int(group["candidate_count"]) < 2 for group in groups),
        "retained_candidates": sum(int(group["candidate_count"]) for group in groups),
        "proposal_audit_rows": len(attempts),
        "retention_statuses": dict(sorted(status_histogram.items())),
        "attempt_kinds": dict(sorted(attempt_kind_histogram.items())),
        "invalid_attempts": int(status_histogram.get("invalid", 0)),
        "duplicate_attempts": int(status_histogram.get("duplicate", 0)),
        "temperature": TEMPERATURE,
        "maximum_proposal_attempts": MAX_PROPOSAL_ATTEMPTS,
        "candidate_retention": "first_distinct_legal_in_request_order",
        "request_keyed_sampling": True,
        "batch_size": 8,
        "distributed": True,
        "world_size": int(world_size),
        "seed": int(seed),
        "outcomes_read": False,
        "energy_selection": False,
        "dynamic_temperature": False,
        "elapsed_sec": float(elapsed_sec),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--crysllmgen-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260904)
    args = parser.parse_args()
    if int(args.batch_size) != 8:
        raise ValueError("potential-closure candidate sampling is frozen at batch 8")

    dist_info = init_distributed()
    rank = int(dist_info["rank"])
    world_size = int(dist_info["world_size"])
    distributed = bool(dist_info["distributed"])
    is_main = bool(dist_info["is_main"])
    if not distributed or world_size != 4:
        raise RuntimeError("formal action sampling requires four distributed GPUs")
    if is_main:
        if args.output_dir.exists():
            raise FileExistsError(args.output_dir)
        args.output_dir.mkdir(parents=True, exist_ok=False)
    dist.barrier()

    all_rows = list(iter_jsonl(args.states.resolve()))
    if len(all_rows) != EXPECTED_GROUPS:
        raise ValueError("potential-closure state denominator changed")
    group_indices = [int(row["group_idx"]) for row in all_rows]
    if sorted(group_indices) != list(range(EXPECTED_GROUPS)):
        raise ValueError("state group_idx coverage changed")
    if Counter(str(row.get("stratum")) for row in all_rows) != Counter(
        {name: EXPECTED_GROUPS // len(EXPECTED_STRATA) for name in EXPECTED_STRATA}
    ):
        raise ValueError("four-stratum balance changed")

    assigned = [_validate_state(row) for row in all_rows[rank::world_size]]
    assigned.sort(
        key=lambda row: (
            int(row["num_atoms"]),
            str(row["transaction_kind"]),
            int(row["group_idx"]),
        )
    )
    process_one = import_process_one(args.crysllmgen_dir)
    model, tokenizer = load_model_and_tokenizer(
        args.model_path, args.checkpoint_path, dist_info["device"]
    )
    tokenizer_report = validate_dynamic_tokenizer_contract(
        tokenizer, mask_token_id=MASK_TOKEN_ID
    )

    local_groups: list[dict[str, Any]] = []
    local_attempts: list[dict[str, Any]] = []
    started = time.time()
    offset = 0
    progress = tqdm(
        total=len(assigned),
        desc=f"potential closure actions rank{rank}",
        disable=not is_main,
    )
    while offset < len(assigned):
        first = assigned[offset]
        batch: list[dict[str, Any]] = []
        while (
            offset < len(assigned)
            and len(batch) < int(args.batch_size)
            and int(assigned[offset]["num_atoms"]) == int(first["num_atoms"])
            and str(assigned[offset]["transaction_kind"])
            == str(first["transaction_kind"])
        ):
            batch.append(assigned[offset])
            offset += 1
        groups, attempts = _sample_batch(
            batch,
            model=model,
            tokenizer=tokenizer,
            process_one=process_one,
            base_seed=int(args.seed),
        )
        local_groups.extend(groups)
        local_attempts.extend(attempts)
        progress.update(len(batch))
    progress.close()
    local_groups.sort(key=lambda row: int(row["group_idx"]))
    local_attempts.sort(
        key=lambda row: (
            int(row["group_idx"]),
            {"noop": 0, "clean_teacher": 1, "dlm_proposal": 2}[
                str(row["attempt_kind"])
            ],
            int(row.get("proposal_attempt") or 0),
        )
    )
    _write_jsonl(
        rank_path(args.output_dir, "candidate_groups.jsonl", rank, True),
        local_groups,
    )
    _write_jsonl(
        rank_path(args.output_dir, "proposal_attempts.jsonl", rank, True),
        local_attempts,
    )
    rank_manifest = _manifest(
        local_groups,
        local_attempts,
        world_size=world_size,
        elapsed_sec=time.time() - started,
        seed=int(args.seed),
    )
    rank_manifest["rank"] = rank
    rank_manifest["distributed"] = True
    rank_path(args.output_dir, "manifest.json", rank, True).write_text(
        json.dumps(rank_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    dist.barrier()

    if is_main:
        merged_groups: list[dict[str, Any]] = []
        merged_attempts: list[dict[str, Any]] = []
        rank_reports: list[dict[str, Any]] = []
        for source_rank in range(world_size):
            merged_groups.extend(
                iter_jsonl(
                    rank_path(
                        args.output_dir, "candidate_groups.jsonl", source_rank, True
                    )
                )
            )
            merged_attempts.extend(
                iter_jsonl(
                    rank_path(
                        args.output_dir, "proposal_attempts.jsonl", source_rank, True
                    )
                )
            )
            rank_reports.append(
                json.loads(
                    rank_path(
                        args.output_dir, "manifest.json", source_rank, True
                    ).read_text(encoding="utf-8")
                )
            )
        merged_groups.sort(key=lambda row: int(row["group_idx"]))
        merged_attempts.sort(
            key=lambda row: (
                int(row["group_idx"]),
                {"noop": 0, "clean_teacher": 1, "dlm_proposal": 2}[
                    str(row["attempt_kind"])
                ],
                int(row.get("proposal_attempt") or 0),
            )
        )
        if [int(row["group_idx"]) for row in merged_groups] != list(
            range(EXPECTED_GROUPS)
        ):
            raise RuntimeError("distributed candidate-group merge changed coverage")
        _write_jsonl(args.output_dir / "candidate_groups.jsonl", merged_groups)
        _write_jsonl(args.output_dir / "proposal_attempts.jsonl", merged_attempts)
        report = _manifest(
            merged_groups,
            merged_attempts,
            world_size=world_size,
            elapsed_sec=max(float(item["elapsed_sec"]) for item in rank_reports),
            seed=int(args.seed),
        )
        report["tokenizer"] = tokenizer_report
        (args.output_dir / "manifest.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (args.output_dir / "_SUCCESS").touch()
        print(json.dumps(report, sort_keys=True))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
