#!/usr/bin/env python3
"""Build outcome-blind K<=4 actions and exact SPAD closure terminals."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Iterable, Mapping, Sequence


GROUP_SCHEMA = "spad_basin_preflight_action_group_v1"
ATTEMPT_SOURCES = (
    "no_op",
    "reference_dlm",
    "physics_downhill",
    "physics_reverse",
)
MAX_CANDIDATES = 4


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"JSONL line {line_number} is not an object")
            yield value


def retain_fixed_order_actions(
    attempts: Sequence[Mapping[str, Any]], *, maximum: int = MAX_CANDIDATES
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Retain the first distinct legal supplied actions without replacement."""

    if int(maximum) <= 0:
        raise ValueError("maximum must be positive")
    retained: list[dict[str, Any]] = []
    audited: list[dict[str, Any]] = []
    seen: dict[tuple[int, ...], str] = {}
    for attempt_index, raw in enumerate(attempts):
        item = dict(raw)
        item["attempt_index"] = int(attempt_index)
        signature = tuple(int(value) for value in item.get("action_token_ids") or ())
        if item.get("legal_supplied_action") is not True or not signature:
            item["retention_status"] = "invalid"
            item["retention_reason"] = item.get("legality_reason") or "illegal_action"
        elif signature in seen:
            item["retention_status"] = "duplicate"
            item["retention_reason"] = f"same_action_as:{seen[signature]}"
        elif len(retained) >= int(maximum):
            item["retention_status"] = "over_limit"
            item["retention_reason"] = "fixed_capacity_reached"
        else:
            item["retention_status"] = "retained"
            item["retention_reason"] = None
            item["candidate_idx"] = len(retained)
            seen[signature] = str(item["source"])
            retained.append(dict(item))
        audited.append(item)
    return retained, audited


def summarize_groups(groups: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return mergeable accounting without reading terminal values."""

    k_hist: Counter[int] = Counter()
    attempt_sources: Counter[str] = Counter()
    proposal_statuses: Counter[str] = Counter()
    retention_statuses: Counter[str] = Counter()
    retained_sources: Counter[str] = Counter()
    terminal_valid = 0
    reference_retained = 0
    reference_matched = 0
    reference_mismatched = 0
    state_efsm_known = 0
    group_failures = 0
    for group in groups:
        candidates = list(group.get("candidates") or ())
        attempts = list(group.get("candidate_attempts") or ())
        k_hist[len(candidates)] += 1
        group_failures += int(group.get("group_failure") is not None)
        state_efsm_known += int(
            bool((group.get("state_diagnostics") or {}).get("efsm_known"))
        )
        for attempt in attempts:
            attempt_sources[str(attempt.get("source"))] += 1
            proposal_statuses[str(attempt.get("proposal_status"))] += 1
            retention_statuses[str(attempt.get("retention_status"))] += 1
        for candidate in candidates:
            source = str(candidate.get("source"))
            retained_sources[source] += 1
            terminal_valid += int(candidate.get("terminal_legal") is True)
            if source == "reference_dlm":
                reference_retained += 1
                if candidate.get("reference_replay_matches_final") is True:
                    reference_matched += 1
                else:
                    reference_mismatched += 1
    return {
        "groups": len(groups),
        "group_failures": group_failures,
        "candidate_count_histogram": {
            str(key): int(value) for key, value in sorted(k_hist.items())
        },
        "candidate_attempt_sources": dict(sorted(attempt_sources.items())),
        "proposal_statuses": dict(sorted(proposal_statuses.items())),
        "retention_statuses": dict(sorted(retention_statuses.items())),
        "retained_sources": dict(sorted(retained_sources.items())),
        "retained_candidates": sum(k * count for k, count in k_hist.items()),
        "terminal_valid_candidates": terminal_valid,
        "state_efsm_known": state_efsm_known,
        "reference_replay": {
            "retained": reference_retained,
            "matched": reference_matched,
            "mismatched": reference_mismatched,
        },
    }


def _runtime_imports() -> dict[str, Any]:
    import numpy as np
    import torch

    from crystal_dlm.dynamic_crystal import arrays_to_structure
    from crystal_dlm.feasible_force_teacher import minimum_image_vector
    from crystal_dlm.fixed_slot import MASK_TOKEN_ID
    from crystal_dlm.r5_dynamic_length import (
        exact_dynamic_schema_constraints,
        validate_answer_matches_plan,
        validate_dynamic_tokenizer_contract,
    )
    from crystal_dlm.spad_generation import (
        FixedBatchShapeModelView,
        continue_spad_species_blocks_from_cursor,
        revise_spad_species_blocks,
    )
    from crystal_dlm.spad_program import (
        program_from_element_order,
        reverse_species_block_revision_slots,
    )
    from crystal_dlm.transaction_physics import (
        lattice_matrix_from_dynamic_arrays,
        propose_force_site_transactions,
        propose_stress_lattice_transactions,
    )
    from scripts.sample_llada_dynamic_crystals import (
        build_dynamic_lightweight_constraints,
        graph_from_arrays,
        import_process_one,
        load_model_and_tokenizer,
    )

    return {
        "np": np,
        "torch": torch,
        "arrays_to_structure": arrays_to_structure,
        "minimum_image_vector": minimum_image_vector,
        "mask_token_id": MASK_TOKEN_ID,
        "exact_dynamic_schema_constraints": exact_dynamic_schema_constraints,
        "validate_answer_matches_plan": validate_answer_matches_plan,
        "validate_dynamic_tokenizer_contract": validate_dynamic_tokenizer_contract,
        "continue_from_cursor": continue_spad_species_blocks_from_cursor,
        "fixed_batch_model_view": FixedBatchShapeModelView,
        "revise_species_blocks": revise_spad_species_blocks,
        "program_from_element_order": program_from_element_order,
        "reverse_species_block_revision_slots": reverse_species_block_revision_slots,
        "lattice_matrix_from_dynamic_arrays": lattice_matrix_from_dynamic_arrays,
        "propose_force": propose_force_site_transactions,
        "propose_stress": propose_stress_lattice_transactions,
        "build_constraints": build_dynamic_lightweight_constraints,
        "graph_from_arrays": graph_from_arrays,
        "import_process_one": import_process_one,
        "load_model_and_tokenizer": load_model_and_tokenizer,
    }


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "detach"):
        return _jsonable(value.detach().cpu().tolist())
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if hasattr(value, "item"):
        return _jsonable(value.item())
    return str(value)


def _encode_ids(tokenizer: Any, text: str) -> list[int]:
    return [
        int(value)
        for value in tokenizer(str(text), add_special_tokens=False)["input_ids"]
    ]


def _tokens_for_ids(tokenizer: Any, token_ids: Sequence[int]) -> list[str]:
    values = tokenizer.convert_ids_to_tokens([int(value) for value in token_ids])
    if isinstance(values, str):
        values = [values]
    if len(values) != len(token_ids) or any(value is None for value in values):
        raise ValueError("action token IDs do not resolve through the tokenizer")
    return [str(value) for value in values]


def _action_is_legal(
    token_ids: Sequence[int],
    positions: Sequence[int],
    allowed: Sequence[Sequence[int]],
) -> tuple[bool, str | None]:
    if len(token_ids) != len(positions):
        return False, "action_arity_mismatch"
    if len(set(int(value) for value in positions)) != len(positions):
        return False, "duplicate_active_position"
    for position, token_id in zip(positions, token_ids, strict=True):
        index = int(position)
        if not 0 <= index < len(allowed):
            return False, "active_position_out_of_schema"
        if int(token_id) not in set(int(value) for value in allowed[index]):
            return False, f"token_outside_schema_at:{index}"
    return True, None


def _supplied_attempt(
    *,
    source: str,
    path: str,
    token_ids: Sequence[int],
    tokenizer: Any,
    positions: Sequence[int],
    allowed: Sequence[Sequence[int]],
) -> dict[str, Any]:
    legal, reason = _action_is_legal(token_ids, positions, allowed)
    try:
        tokens = _tokens_for_ids(tokenizer, token_ids)
    except Exception as error:  # noqa: BLE001
        tokens = []
        legal = False
        reason = f"token_resolution:{type(error).__name__}:{error}"
    return {
        "source": source,
        "path": path,
        "action_token_ids": [int(value) for value in token_ids],
        "action_tokens": tokens,
        "status": "accepted" if legal else "invalid",
        "reason": path if legal else reason,
        "step": None,
        "proposal_status": "supplied",
        "proposal_reason": path,
        "proposal_step": None,
        "proposal_direction": None,
        "proposal_minimum_distance_A": None,
        "legal_supplied_action": bool(legal),
        "legality_reason": reason,
    }


def _unavailable_physics_attempt(source: str, reason: str) -> dict[str, Any]:
    return {
        "source": source,
        "path": "state_CHGNet_0.3_EFSM",
        "action_token_ids": [],
        "action_tokens": [],
        "status": "invalid",
        "reason": str(reason),
        "step": None,
        "proposal_status": "invalid",
        "proposal_reason": str(reason),
        "proposal_step": None,
        "proposal_direction": None,
        "proposal_minimum_distance_A": None,
        "legal_supplied_action": False,
        "legality_reason": str(reason),
    }


def _physics_attempt(
    source: str,
    proposal: Any,
    *,
    tokenizer: Any,
    positions: Sequence[int],
    allowed: Sequence[Sequence[int]],
) -> dict[str, Any]:
    try:
        token_ids = [
            int(tokenizer.get_vocab()[str(token)])
            for token in proposal.transaction_tokens
        ]
        schema_legal, schema_reason = _action_is_legal(
            token_ids, positions, allowed
        )
    except Exception as error:  # noqa: BLE001
        token_ids = []
        schema_legal = False
        schema_reason = f"token_resolution:{type(error).__name__}:{error}"
    proposal_status = str(proposal.status)
    supplied_legal = proposal_status in {"accepted", "noop", "duplicate"}
    return {
        "source": source,
        "path": f"transaction_physics:{proposal.kind}:{proposal.direction}",
        "action_token_ids": token_ids,
        "action_tokens": [str(value) for value in proposal.transaction_tokens],
        "status": proposal_status,
        "reason": str(proposal.reason),
        "step": None if proposal.step is None else float(proposal.step),
        "proposal_status": proposal_status,
        "proposal_reason": str(proposal.reason),
        "proposal_step": None if proposal.step is None else float(proposal.step),
        "proposal_direction": str(proposal.direction),
        "proposal_minimum_distance_A": (
            None
            if proposal.minimum_distance_A is None
            else float(proposal.minimum_distance_A)
        ),
        "legal_supplied_action": bool(supplied_legal and schema_legal),
        "legality_reason": (
            schema_reason
            if not schema_legal
            else None if supplied_legal else f"proposal_{proposal_status}"
        ),
    }


def _normalize_efsm(
    prediction: Mapping[str, Any] | None,
    *,
    num_sites: int,
    np: Any,
    failure: str | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "source_body": "provisional_complete_body",
        "efsm_known": False,
        "energy_known": False,
        "force_known": False,
        "stress_known": False,
        "energy_eV_per_atom": None,
        "forces_eV_per_A": None,
        "force_rms_eV_per_A": None,
        "force_max_eV_per_A": None,
        "stress_GPa": None,
        "stress_frobenius_GPa": None,
        "failure": failure,
    }
    if prediction is None:
        return result
    try:
        energy = float(np.asarray(prediction["e"], dtype=np.float64).reshape(()))
        if math.isfinite(energy):
            result["energy_known"] = True
            result["energy_eV_per_atom"] = energy
    except Exception:  # noqa: BLE001
        pass
    try:
        forces = np.asarray(prediction["f"], dtype=np.float64)
        if forces.shape == (int(num_sites), 3) and np.isfinite(forces).all():
            norms = np.linalg.norm(forces, axis=1)
            result["force_known"] = True
            result["forces_eV_per_A"] = forces.tolist()
            result["force_rms_eV_per_A"] = float(
                np.sqrt(np.mean(norms * norms))
            )
            result["force_max_eV_per_A"] = float(np.max(norms))
    except Exception:  # noqa: BLE001
        pass
    try:
        stress = np.asarray(prediction["s"], dtype=np.float64)
        if stress.shape == (3, 3) and np.isfinite(stress).all():
            result["stress_known"] = True
            result["stress_GPa"] = stress.tolist()
            result["stress_frobenius_GPa"] = float(np.linalg.norm(stress))
    except Exception:  # noqa: BLE001
        pass
    result["efsm_known"] = bool(
        result["energy_known"] and result["force_known"] and result["stress_known"]
    )
    if result["efsm_known"]:
        result["failure"] = None
    elif result["failure"] is None:
        result["failure"] = "partial_or_nonfinite_EFSM"
    return result


def _predict_efsm_isolated(
    model: Any, structures: Sequence[Any], *, batch_size: int
) -> list[tuple[Mapping[str, Any] | None, str | None]]:
    output: list[tuple[Mapping[str, Any] | None, str | None]] = []
    for start in range(0, len(structures), int(batch_size)):
        chunk = list(structures[start : start + int(batch_size)])
        try:
            values = model.predict_structure(
                chunk, task="efsm", batch_size=int(batch_size)
            )
            if isinstance(values, Mapping):
                values = [values]
            values = list(values)
            if len(values) != len(chunk):
                raise RuntimeError("CHGNet changed EFSM batch cardinality")
            output.extend((value, None) for value in values)
        except Exception as batch_error:  # noqa: BLE001
            prefix = f"batch_fallback:{type(batch_error).__name__}:{batch_error}"
            for structure in chunk:
                try:
                    output.append(
                        (model.predict_structure(structure, task="efsm"), prefix)
                    )
                except Exception as error:  # noqa: BLE001
                    output.append(
                        (
                            None,
                            f"{prefix};single:{type(error).__name__}:{error}",
                        )
                    )
    if len(output) != len(structures):
        raise RuntimeError("CHGNet changed total EFSM cardinality")
    return output


def _prepare_state_diagnostics(
    states: Sequence[Mapping[str, Any]],
    *,
    chgnet: Any,
    batch_size: int,
    runtime: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Run all provisional-state EFSM calls before terminal continuation."""

    prepared: list[dict[str, Any]] = []
    structures: list[Any] = []
    destinations: list[int] = []
    for state in states:
        item: dict[str, Any] = {"arrays": None, "structure": None, "failure": None}
        try:
            arrays = runtime["validate_answer_matches_plan"](
                state["plan_state"], str(state["provisional_complete_body"])
            )
            structure = runtime["arrays_to_structure"](arrays)
            item.update({"arrays": arrays, "structure": structure})
            destinations.append(len(prepared))
            structures.append(structure)
        except Exception as error:  # noqa: BLE001
            item["failure"] = f"provisional_parse:{type(error).__name__}:{error}"
        prepared.append(item)

    predictions = _predict_efsm_isolated(
        chgnet, structures, batch_size=int(batch_size)
    )
    for destination, (prediction, failure) in zip(
        destinations, predictions, strict=True
    ):
        prepared[destination]["prediction"] = prediction
        prepared[destination]["prediction_failure"] = failure
    for item, state in zip(prepared, states, strict=True):
        failure = item.get("failure") or item.get("prediction_failure")
        item["diagnostics"] = _normalize_efsm(
            item.get("prediction"),
            num_sites=int(state["N"]),
            np=runtime["np"],
            failure=failure,
        )
    return prepared


def _candidate_attempts(
    state: Mapping[str, Any],
    prepared: Mapping[str, Any],
    *,
    tokenizer: Any,
    allowed: Sequence[Sequence[int]],
    runtime: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    positions = [int(value) for value in state["active_generation_positions"]]
    reference = state["reference_action"]
    attempts = [
        _supplied_attempt(
            source="no_op",
            path="reference_action.previous_token_ids",
            token_ids=reference["previous_token_ids"],
            tokenizer=tokenizer,
            positions=positions,
            allowed=allowed,
        ),
        _supplied_attempt(
            source="reference_dlm",
            path="reference_action.token_ids",
            token_ids=reference["token_ids"],
            tokenizer=tokenizer,
            positions=positions,
            allowed=allowed,
        ),
    ]
    diagnostics = prepared["diagnostics"]
    arrays = prepared.get("arrays")
    if arrays is None:
        physics = [
            _unavailable_physics_attempt(name, "provisional_state_unavailable")
            for name in ATTEMPT_SOURCES[2:]
        ]
    else:
        try:
            if state["state_type"] == "cell":
                if diagnostics.get("stress_known") is not True:
                    raise ValueError("state_stress_unavailable")
                proposals = runtime["propose_stress"](
                    arrays,
                    diagnostics["stress_GPa"],
                    minimum_distance_A=0.5,
                    image_radius=2,
                )
            elif state["state_type"] == "xyz":
                if diagnostics.get("force_known") is not True:
                    raise ValueError("state_force_unavailable")
                slot = int(state["cursor"]["slot_index"])
                proposals = runtime["propose_force"](
                    arrays,
                    slot,
                    diagnostics["forces_eV_per_A"][slot],
                    minimum_distance_A=0.5,
                    image_radius=2,
                )
            else:
                raise ValueError(f"unknown state type {state['state_type']!r}")
            physics = [
                _physics_attempt(
                    source,
                    proposal,
                    tokenizer=tokenizer,
                    positions=positions,
                    allowed=allowed,
                )
                for source, proposal in zip(
                    ATTEMPT_SOURCES[2:], proposals, strict=True
                )
            ]
        except Exception as error:  # noqa: BLE001
            reason = f"physics_proposal:{type(error).__name__}:{error}"
            physics = [
                _unavailable_physics_attempt(name, reason)
                for name in ATTEMPT_SOURCES[2:]
            ]
    attempts.extend(physics)
    if tuple(item["source"] for item in attempts) != ATTEMPT_SOURCES:
        raise RuntimeError("candidate attempt order changed")
    return retain_fixed_order_actions(attempts)


def _body_tensors(
    state: Mapping[str, Any],
    tokenizer: Any,
    device: Any,
    runtime: Mapping[str, Any],
    *,
    padded_prompt_length: int,
) -> dict[str, Any]:
    torch = runtime["torch"]

    def tensor(text: str) -> Any:
        ids = _encode_ids(tokenizer, text)
        return torch.tensor([ids], dtype=torch.long, device=device)

    unpadded_prompt = tensor(str(state["prompt"]).rstrip() + "\n")
    padding = int(padded_prompt_length) - int(unpadded_prompt.shape[1])
    if padding < 0:
        raise ValueError("recorded batch prompt length is shorter than this prompt")
    prompt = torch.cat(
        (
            torch.full(
                (1, padding),
                int(tokenizer.pad_token_id),
                dtype=torch.long,
                device=device,
            ),
            unpadded_prompt,
        ),
        dim=1,
    )
    attention = torch.cat(
        (
            torch.zeros((1, padding), dtype=torch.long, device=device),
            torch.ones_like(unpadded_prompt, dtype=torch.long),
        ),
        dim=1,
    )
    predictor = tensor(str(state["predictor_body"]))
    masked_state = tensor(str(state["state_body"]))
    final = tensor(str(state["final_body"]))
    if state["state_type"] == "cell":
        entry = predictor.clone()
    else:
        entry = tensor(str(state["block_entry_snapshot"]["body"]))
    gen_length = 7 + 4 * int(state["N"])
    for name, value in (
        ("predictor", predictor),
        ("state", masked_state),
        ("entry", entry),
        ("final", final),
    ):
        if int(value.shape[1]) != gen_length:
            raise ValueError(f"{name} body is not exact 7+4N")
    active = torch.tensor(
        [int(value) for value in state["active_generation_positions"]],
        dtype=torch.long,
        device=device,
    )
    if not bool((masked_state[0, active] == int(runtime["mask_token_id"])).all()):
        raise ValueError("active transaction is not masked in state_body")
    context = [
        int(value) for value in state["context_masked_generation_positions"]
    ]
    if context:
        context_tensor = torch.tensor(context, dtype=torch.long, device=device)
        if not bool(
            (masked_state[0, context_tensor] == int(runtime["mask_token_id"])).all()
        ):
            raise ValueError("future cursor context is not masked")
    previous = [int(value) for value in state["reference_action"]["previous_token_ids"]]
    provisional = tensor(str(state["provisional_complete_body"]))
    if [int(value) for value in provisional[0, active].tolist()] != previous:
        raise ValueError("no-op does not equal provisional active transaction")
    return {
        "prompt": prompt,
        "attention": attention,
        "predictor": predictor,
        "state": masked_state,
        "entry": entry,
        "final": final,
        "prompt_length": int(prompt.shape[1]),
        "gen_length": gen_length,
    }


def _minimum_distance_125(arrays: Mapping[str, Any], runtime: Mapping[str, Any]) -> float:
    np = runtime["np"]
    lattice = runtime["lattice_matrix_from_dynamic_arrays"](arrays)
    shifts = np.asarray(
        [
            (i, j, k)
            for i in range(-2, 3)
            for j in range(-2, 3)
            for k in range(-2, 3)
            if (i, j, k) != (0, 0, 0)
        ],
        dtype=np.float64,
    )
    minimum = float(np.min(np.linalg.norm(shifts @ lattice, axis=1)))
    coordinates = np.asarray(arrays["frac_coords"], dtype=np.float64)
    for left in range(len(coordinates)):
        for right in range(left + 1, len(coordinates)):
            _vector, distance = runtime["minimum_image_vector"](
                coordinates[left],
                coordinates[right],
                lattice,
                image_radius=2,
            )
            minimum = min(minimum, float(distance))
    return minimum


def _terminal_payload(
    body_ids: Sequence[int],
    *,
    tokenizer: Any,
    state: Mapping[str, Any],
    allowed: Sequence[Sequence[int]],
    process_one: Any,
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    if len(body_ids) != len(allowed):
        raise ValueError("terminal body length differs from exact schema")
    for position, token_id in enumerate(body_ids):
        if int(token_id) not in set(int(value) for value in allowed[position]):
            raise ValueError(f"terminal token outside schema at {position}")
    tokens = _tokens_for_ids(tokenizer, body_ids)
    body = "".join(tokens)
    arrays = runtime["validate_answer_matches_plan"](state["plan_state"], body)
    minimum_distance = _minimum_distance_125(arrays, runtime)
    if minimum_distance < 0.5 - 1.0e-10:
        raise ValueError(f"terminal PBC minimum distance {minimum_distance:.8f} < 0.5 A")
    graph, cif = runtime["graph_from_arrays"](dict(arrays), process_one)
    normalized_arrays = {
        "num_atoms": int(arrays["num_atoms"]),
        "lengths": [float(value) for value in arrays["lengths"]],
        "angles": [float(value) for value in arrays["angles"]],
        "species": [str(value) for value in arrays["species"]],
        "atom_types": [int(value) for value in arrays["atom_types"]],
        "frac_coords": [
            [float(value) for value in coordinate]
            for coordinate in arrays["frac_coords"]
        ],
    }
    return {
        "terminal_body": body,
        "terminal_body_token_ids": [int(value) for value in body_ids],
        "terminal_arrays": normalized_arrays,
        "terminal_structure": normalized_arrays,
        "terminal_graph": _jsonable(graph),
        "terminal_cif": str(cif),
        "terminal_minimum_distance_A_125_image": float(minimum_distance),
        "terminal_legal": True,
        "terminal_failure": None,
    }


def _execute_candidate(
    candidate: Mapping[str, Any],
    *,
    model: Any,
    tokenizer: Any,
    state: Mapping[str, Any],
    tensors: Mapping[str, Any],
    blocks: Sequence[Sequence[int]],
    allowed: Sequence[Sequence[int]],
    constraints: Mapping[str, Any],
    process_one: Any,
    device: Any,
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    torch = runtime["torch"]
    output = dict(candidate)
    output.update(
        {
            "action_id": f"{int(state['sample_idx']):03d}:{int(candidate['candidate_idx'])}:{candidate['source']}",
            "common_seed": int(state["continuation_seeds"]["species_blocks"]),
            "terminal_body": None,
            "terminal_body_token_ids": None,
            "terminal_arrays": None,
            "terminal_structure": None,
            "terminal_graph": None,
            "terminal_cif": None,
            "terminal_minimum_distance_A_125_image": None,
            "terminal_legal": False,
            "terminal_failure": None,
            "continuation_report": None,
            "reference_replay_matches_final": None,
            "reference_replay_mismatch_positions": [],
        }
    )
    prompt_length = int(tensors["prompt_length"])
    gen_length = int(tensors["gen_length"])
    attention = tensors["attention"]
    try:
        if state["state_type"] == "cell":
            complete = torch.cat(
                (tensors["prompt"], tensors["predictor"]), dim=1
            )
            for position, token_id in zip(
                state["active_generation_positions"],
                candidate["action_token_ids"],
                strict=True,
            ):
                complete[0, prompt_length + int(position)] = int(token_id)
            terminal, logs = runtime["revise_species_blocks"](
                model,
                complete,
                prompt_length=prompt_length,
                gen_length=gen_length,
                revision_blocks_by_batch=[[list(value) for value in blocks]],
                attention_mask=attention,
                temperature=0.7,
                cfg_scale=0.0,
                remasking="low_confidence",
                mask_id=int(runtime["mask_token_id"]),
                allowed_token_ids_by_generation_pos=list(allowed),
                atom_count_grammar=None,
                lightweight_decoding_constraints=dict(constraints),
                sampling_seeds_by_batch=[output["common_seed"]],
            )
            continuation_report = {
                "state_type": "cell",
                "species_block_revisions": logs[0],
            }
        else:
            complete = torch.cat((tensors["prompt"], tensors["state"]), dim=1)
            block_entry = torch.cat(
                (tensors["prompt"], tensors["entry"]), dim=1
            )
            terminal, continuation_report = runtime["continue_from_cursor"](
                model,
                complete,
                block_entry_tokens=block_entry,
                prompt_length=prompt_length,
                gen_length=gen_length,
                revision_blocks=[list(value) for value in blocks],
                block_index=int(state["cursor"]["block_index"]),
                site_order_index=int(state["cursor"]["site_order_index"]),
                action_token_ids=candidate["action_token_ids"],
                attention_mask=attention,
                temperature=0.7,
                cfg_scale=0.0,
                remasking="low_confidence",
                mask_id=int(runtime["mask_token_id"]),
                allowed_token_ids_by_generation_pos=list(allowed),
                atom_count_grammar=None,
                lightweight_decoding_constraints=dict(constraints),
                sampling_seed=output["common_seed"],
            )
        body_tensor = terminal[0, prompt_length : prompt_length + gen_length]
        body_ids = [int(value) for value in body_tensor.detach().cpu().tolist()]
        output["continuation_report"] = _jsonable(continuation_report)
        if output["source"] == "reference_dlm":
            final_ids = [int(value) for value in tensors["final"][0].detach().cpu().tolist()]
            mismatch = [
                index
                for index, (observed, expected) in enumerate(
                    zip(body_ids, final_ids, strict=True)
                )
                if observed != expected
            ]
            output["reference_replay_matches_final"] = not mismatch
            output["reference_replay_mismatch_positions"] = mismatch
        try:
            output.update(
                _terminal_payload(
                    body_ids,
                    tokenizer=tokenizer,
                    state=state,
                    allowed=allowed,
                    process_one=process_one,
                    runtime=runtime,
                )
            )
        except Exception as error:  # noqa: BLE001
            output["terminal_body_token_ids"] = body_ids
            try:
                output["terminal_body"] = "".join(
                    _tokens_for_ids(tokenizer, body_ids)
                )
            except Exception:  # noqa: BLE001
                pass
            output["terminal_failure"] = f"validation:{type(error).__name__}:{error}"
    except Exception as error:  # noqa: BLE001
        output["terminal_failure"] = f"continuation:{type(error).__name__}:{error}"
        if output["source"] == "reference_dlm":
            output["reference_replay_matches_final"] = False
    return output


def _build_group(
    state: Mapping[str, Any],
    prepared: Mapping[str, Any],
    *,
    model: Any,
    tokenizer: Any,
    process_one: Any,
    device: Any,
    allowed_cache: dict[int, list[list[int]]],
    constraints: Mapping[str, Any],
    padded_prompt_length: int,
    original_batch_size: int,
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    num_atoms = int(state["N"])
    if num_atoms not in allowed_cache:
        allowed_cache[num_atoms] = runtime["exact_dynamic_schema_constraints"](
            tokenizer, num_atoms
        )
    allowed = allowed_cache[num_atoms]
    retained, attempts = _candidate_attempts(
        state,
        prepared,
        tokenizer=tokenizer,
        allowed=allowed,
        runtime=runtime,
    )
    tensors = _body_tensors(
        state,
        tokenizer,
        device,
        runtime,
        padded_prompt_length=int(padded_prompt_length),
    )
    program = runtime["program_from_element_order"](
        state["plan_state"],
        state["species_program"],
        order_source="frozen_planner_llama_pointer",
    )
    blocks = runtime["reverse_species_block_revision_slots"](program)
    candidates = [
        _execute_candidate(
            candidate,
            model=runtime["fixed_batch_model_view"](
                model,
                batch_size=int(original_batch_size),
                row_index=int(state["sample_idx"]) % int(original_batch_size),
            ),
            tokenizer=tokenizer,
            state=state,
            tensors=tensors,
            blocks=blocks,
            allowed=allowed,
            constraints=constraints,
            process_one=process_one,
            device=device,
            runtime=runtime,
        )
        for candidate in retained
    ]
    reference = next(
        (candidate for candidate in candidates if candidate["source"] == "reference_dlm"),
        None,
    )
    return {
        "schema": GROUP_SCHEMA,
        "preflight_idx": int(state["preflight_idx"]),
        "sample_idx": int(state["sample_idx"]),
        "state_type": str(state["state_type"]),
        "cursor_bucket": state.get("cursor_bucket"),
        "state": dict(state),
        "state_diagnostics": dict(prepared["diagnostics"]),
        "candidate_attempts": attempts,
        "candidates": candidates,
        "candidate_count": len(candidates),
        "common_seed": int(state["continuation_seeds"]["species_blocks"]),
        "reference_replay": {
            "retained": reference is not None,
            "matches_final": (
                None
                if reference is None
                else reference["reference_replay_matches_final"]
            ),
            "mismatch_positions": (
                []
                if reference is None
                else reference["reference_replay_mismatch_positions"]
            ),
        },
        "group_failure": None,
        "outcomes_read": False,
        "selection": False,
        "replacement": False,
    }


def _failed_group(state: Mapping[str, Any], error: Exception) -> dict[str, Any]:
    return {
        "schema": GROUP_SCHEMA,
        "preflight_idx": int(state.get("preflight_idx", state["sample_idx"])),
        "sample_idx": int(state["sample_idx"]),
        "state_type": str(state.get("state_type") or "unknown"),
        "cursor_bucket": state.get("cursor_bucket"),
        "state": dict(state),
        "state_diagnostics": {
            "efsm_known": False,
            "failure": f"group_build:{type(error).__name__}:{error}",
        },
        "candidate_attempts": [],
        "candidates": [],
        "candidate_count": 0,
        "common_seed": (state.get("continuation_seeds") or {}).get("species_blocks"),
        "reference_replay": {
            "retained": False,
            "matches_final": None,
            "mismatch_positions": [],
        },
        "group_failure": f"{type(error).__name__}:{error}",
        "outcomes_read": False,
        "selection": False,
        "replacement": False,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not 8 <= int(args.chgnet_batch_size) <= 16:
        raise ValueError("--chgnet-batch-size must lie in 8..16")
    runtime = _runtime_imports()
    torch = runtime["torch"]
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError("invalid torchrun rank/world size")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    states = list(iter_jsonl(args.states_jsonl.resolve()))
    if len(states) != 128 or {int(row["sample_idx"]) for row in states} != set(
        range(128)
    ):
        raise ValueError("states JSONL must contain sample_idx 0..127 exactly once")
    if any(row.get("schema") != "spad_basin_preflight_state_v1" for row in states):
        raise ValueError("unexpected preflight state schema")
    if any(
        row.get("outcomes_read") is not False
        or row.get("selection") is not False
        or row.get("replacement") is not False
        for row in states
    ):
        raise ValueError("preflight states are not outcome blind")
    assigned = [row for row in states if int(row["sample_idx"]) % world_size == rank]
    if int(args.original_batch_size) <= 0:
        raise ValueError("--original-batch-size must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    group_path = args.output_dir / f"groups_rank{rank}.jsonl"
    report_path = args.output_dir / f"report_rank{rank}.json"
    if group_path.exists() or report_path.exists():
        raise FileExistsError(f"rank {rank} output already exists")

    model, tokenizer = runtime["load_model_and_tokenizer"](
        str(args.model_path), str(args.checkpoint_path), device
    )
    tokenizer_contract = runtime["validate_dynamic_tokenizer_contract"](tokenizer)
    prompt_lengths = {
        int(row["sample_idx"]): len(
            _encode_ids(tokenizer, str(row["prompt"]).rstrip() + "\n")
        )
        for row in states
    }
    batch_prompt_lengths: dict[int, int] = {}
    for start in range(0, len(states), int(args.original_batch_size)):
        batch = states[start : start + int(args.original_batch_size)]
        batch_prompt_lengths[start // int(args.original_batch_size)] = max(
            prompt_lengths[int(row["sample_idx"])] for row in batch
        )
    from chgnet.model.model import CHGNet

    chgnet = CHGNet.load(
        use_device=str(device), check_cuda_mem=False, verbose=False
    )
    process_one = runtime["import_process_one"](args.crysllmgen_dir)
    constraints = runtime["build_constraints"](
        tokenizer,
        duplicate_coordinate_mask=True,
        lattice_volume_mask=True,
        min_lattice_rad=1.0e-4,
        canonicalize_periodic_alias=True,
        pbc_min_distance_mask=True,
        pbc_min_distance_A=0.5,
        pbc_image_radius=2,
    )
    if constraints is None:
        raise RuntimeError("deployed geometry constraints were not built")

    started = time.time()
    prepared = _prepare_state_diagnostics(
        assigned,
        chgnet=chgnet,
        batch_size=int(args.chgnet_batch_size),
        runtime=runtime,
    )
    groups: list[dict[str, Any]] = []
    allowed_cache: dict[int, list[list[int]]] = {}
    for state, state_prepared in zip(assigned, prepared, strict=True):
        try:
            group = _build_group(
                state,
                state_prepared,
                model=model,
                tokenizer=tokenizer,
                process_one=process_one,
                device=device,
                allowed_cache=allowed_cache,
                constraints=constraints,
                padded_prompt_length=batch_prompt_lengths[
                    int(state["sample_idx"]) // int(args.original_batch_size)
                ],
                original_batch_size=int(args.original_batch_size),
                runtime=runtime,
            )
        except Exception as error:  # noqa: BLE001
            group = _failed_group(state, error)
        groups.append(group)

    with group_path.open("x", encoding="utf-8", newline="\n") as handle:
        for group in groups:
            handle.write(json.dumps(_jsonable(group), ensure_ascii=False, sort_keys=True) + "\n")
    report = {
        "schema": "spad_basin_preflight_action_rank_report_v1",
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "shard_rule": "sample_idx modulo world_size",
        "assigned_sample_indices": [int(row["sample_idx"]) for row in assigned],
        "temperature": 0.7,
        "exact_representation": "7+4N",
        "periodic_alias": "canonical",
        "pbc_image_radius": 2,
        "pbc_images": 125,
        "pbc_minimum_distance_A": 0.5,
        "chgnet_checkpoint": "0.3.0",
        "chgnet_batch_size": int(args.chgnet_batch_size),
        "model_loads_per_rank": {"closure_dlm": 1, "chgnet": 1},
        "tokenizer_contract": tokenizer_contract,
        **summarize_groups(groups),
        "outcomes_read": False,
        "selection": False,
        "replacement": False,
        "elapsed_seconds": time.time() - started,
        "original_batch_size": int(args.original_batch_size),
        "left_padding_replayed": True,
    }
    report_path.write_text(
        json.dumps(_jsonable(report), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states-jsonl", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--crysllmgen-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chgnet-batch-size", type=int, default=16)
    parser.add_argument("--original-batch-size", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), sort_keys=True))
