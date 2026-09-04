"""Deployment-matched probabilities for complete SPAD transactions.

The SPAD decoder samples a three-token ``XYZ`` or six-token lattice action by
masking the complete transaction and then resolving its components in native
order.  Each committed component is visible to the following model forward.
Because the deployed Gumbel rule is equivalent to categorical sampling from
``softmax(logits / temperature)``, this module scores a fixed action path with
the corresponding sequence of conditional log probabilities.

The implementation deliberately reuses the decoder's schema and lightweight
geometry masks.  It therefore measures legal *proposal-path* probability.  It
does not enumerate the probability of downstream whole-transaction fallback
outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import torch

from crystal_dlm.llada_generation import (
    _apply_lightweight_decoding_masks,
    _apply_schema_masks,
    _model_logits,
    _prepare_atom_count_grammar,
)
from crystal_dlm.spad_generation import (
    _complete_cell_is_supported,
    _full_attention_mask,
)
from crystal_dlm.spad_program import LATTICE_POSITIONS, coordinate_positions


DEPLOYED_TEMPERATURE = 0.7
VALID_TRANSACTION_LENGTHS = (3, 6)


@dataclass(frozen=True)
class TransactionActionAudit:
    """Detached audit information for one supplied complete action path."""

    action_index: int
    token_ids: tuple[int, ...]
    valid: bool
    finite: bool
    joint_logprob: float
    component_logprobs: tuple[float | None, ...]
    invalid_step: int | None
    invalid_generation_position: int | None
    invalid_reason: str | None
    duplicate_of: int | None


@dataclass(frozen=True)
class DeployedTransactionScores:
    """Differentiable scores plus detached path-level diagnostics.

    ``candidate_log_mass`` is the log probability of the union of distinct,
    valid supplied legal proposal paths.  It is conditional on any declared
    context masks and does not include transaction-fallback outcome mass.
    Duplicate input rows retain their own score and audit record, but are
    counted once in that union.
    """

    action_logprobs: torch.Tensor
    valid_mask: torch.Tensor
    candidate_log_mass: torch.Tensor
    action_audits: tuple[TransactionActionAudit, ...]
    generation_positions: tuple[int, ...]
    context_masked_generation_positions: tuple[int, ...]
    transaction_kind: str
    temperature: float
    cfg_scale: float
    suffix_visible: bool
    unique_valid_action_count: int


def _normalize_single_state(
    complete_tokens: torch.Tensor,
    *,
    prompt_length: int,
    gen_length: int,
) -> torch.Tensor:
    if complete_tokens.ndim == 1:
        state = complete_tokens.unsqueeze(0)
    elif complete_tokens.ndim == 2 and int(complete_tokens.shape[0]) == 1:
        state = complete_tokens
    else:
        raise ValueError("complete_tokens must describe exactly one state")
    if int(prompt_length) < 0 or int(gen_length) <= 0:
        raise ValueError("prompt_length and gen_length must be non-negative/positive")
    if int(state.shape[1]) != int(prompt_length) + int(gen_length):
        raise ValueError("complete token sequence does not match prompt+generation")
    if state.dtype != torch.long:
        raise TypeError("complete_tokens must use torch.long token IDs")
    return state


def _normalize_transaction_positions(
    generation_positions: Sequence[int],
    *,
    gen_length: int,
) -> tuple[tuple[int, ...], str]:
    positions = tuple(int(value) for value in generation_positions)
    if len(positions) not in VALID_TRANSACTION_LENGTHS:
        raise ValueError("a complete transaction must contain exactly 3 or 6 tokens")
    if len(set(positions)) != len(positions):
        raise ValueError("transaction positions must be unique")
    if any(value < 0 or value >= int(gen_length) for value in positions):
        raise ValueError("transaction position lies outside the generation canvas")
    if len(positions) == 6:
        expected = tuple(int(value) for value in LATTICE_POSITIONS)
        if positions != expected:
            raise ValueError("six-token transactions must use native lattice order")
        return positions, "cell"
    first = int(positions[0])
    if first < 8 or (first - 8) % 4:
        raise ValueError("three-token transactions must begin at an XYZ site")
    slot = (first - 8) // 4
    if positions != tuple(int(value) for value in coordinate_positions(slot)):
        raise ValueError("three-token transactions must use native X/Y/Z order")
    return positions, "xyz"


def _normalize_context_masked_positions(
    context_masked_generation_positions: Sequence[int],
    *,
    active_positions: Sequence[int],
    gen_length: int,
) -> tuple[int, ...]:
    positions = tuple(int(value) for value in context_masked_generation_positions)
    if len(set(positions)) != len(positions):
        raise ValueError("context-masked generation positions must be unique")
    if any(value < 0 or value >= int(gen_length) for value in positions):
        raise ValueError(
            "context-masked generation position lies outside the generation canvas"
        )
    if set(positions).intersection(int(value) for value in active_positions):
        raise ValueError("context-masked positions must not overlap active positions")
    return positions


def _normalize_actions(
    action_token_ids: torch.Tensor | Sequence[Sequence[int]],
    *,
    width: int,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(action_token_ids, torch.Tensor):
        actions = action_token_ids.to(device=device, dtype=torch.long)
    else:
        actions = torch.tensor(action_token_ids, dtype=torch.long, device=device)
    if actions.ndim != 2 or int(actions.shape[0]) <= 0:
        raise ValueError("action_token_ids must have shape [K, transaction_length]")
    if int(actions.shape[1]) != int(width):
        raise ValueError("action width does not match transaction positions")
    return actions


def _expand_attention(
    attention_mask: torch.Tensor | None,
    *,
    action_count: int,
    device: torch.device,
) -> torch.Tensor | None:
    if attention_mask is None:
        return None
    mask = attention_mask.to(device=device)
    if mask.ndim == 1:
        mask = mask.unsqueeze(0)
    if mask.ndim != 2 or int(mask.shape[0]) != 1:
        raise ValueError("attention_mask must describe exactly one source state")
    return mask.repeat(int(action_count), 1)


def _build_allowed_mask(
    allowed_token_ids_by_generation_pos: Sequence[Sequence[int]],
    *,
    gen_length: int,
    vocab_size: int,
    device: torch.device,
) -> torch.Tensor:
    if len(allowed_token_ids_by_generation_pos) != int(gen_length):
        raise ValueError("allowed token schema length changed")
    allowed = torch.zeros(
        (int(gen_length), int(vocab_size)),
        dtype=torch.bool,
        device=device,
    )
    for position, raw_ids in enumerate(allowed_token_ids_by_generation_pos):
        token_ids = sorted({int(value) for value in raw_ids})
        if not token_ids:
            raise ValueError(f"generation position {position} has no legal token")
        if token_ids[0] < 0 or token_ids[-1] >= int(vocab_size):
            raise ValueError(f"generation position {position} has out-of-vocab schema IDs")
        allowed[
            int(position),
            torch.tensor(token_ids, dtype=torch.long, device=device),
        ] = True
    return allowed


def _first_invalid_reason(
    *,
    target_in_vocab: bool,
    schema_target_legal: bool,
    dynamic_target_legal: bool,
    has_legal_token: bool,
    numerical_logits_valid: bool,
    pbc_no_legal_completion: bool,
) -> str | None:
    if not target_in_vocab:
        return "target_token_out_of_vocab"
    if not numerical_logits_valid:
        return "nonfinite_legal_logits"
    if not has_legal_token:
        return "no_legal_token_after_masks"
    if not schema_target_legal:
        return "schema_masked_target"
    if pbc_no_legal_completion:
        return "pbc_no_legal_completion"
    if not dynamic_target_legal:
        return "dynamic_geometry_masked_target"
    return None


def _deployment_model_logits(
    model: Any,
    x: torch.Tensor,
    attention_mask: torch.Tensor | None,
    prompt_index: torch.Tensor,
    mask_id: int,
) -> torch.Tensor:
    """Run one dropout-free deployment forward while preserving caller mode."""

    previous_training = getattr(model, "training", None)
    if hasattr(model, "eval"):
        model.eval()
    try:
        logits = _model_logits(
            model,
            x,
            attention_mask,
            prompt_index,
            0.0,
            int(mask_id),
        )
    finally:
        if previous_training is not None and hasattr(model, "train"):
            model.train(bool(previous_training))
    if not isinstance(logits, torch.Tensor):
        raise RuntimeError("CFG-zero deployment forward returned non-tensor logits")
    return logits


def score_deployed_transaction_actions(
    model: Any,
    complete_tokens: torch.Tensor,
    *,
    prompt_length: int,
    gen_length: int,
    generation_positions: Sequence[int],
    action_token_ids: torch.Tensor | Sequence[Sequence[int]],
    attention_mask: torch.Tensor | None,
    mask_id: int,
    allowed_token_ids_by_generation_pos: Sequence[Sequence[int]],
    atom_count_grammar: dict | None,
    lightweight_decoding_constraints: dict | None,
    temperature: float = DEPLOYED_TEMPERATURE,
    context_masked_generation_positions: Sequence[int] = (),
) -> DeployedTransactionScores:
    """Score fixed complete actions under the deployed SPAD proposal process.

    The input may contain old values or masks at the active positions; this
    function always starts by masking the *entire* active block.  Positions
    declared in ``context_masked_generation_positions`` must already contain
    ``mask_id`` and remain masked during every component score.  Every other
    generation token must already be committed.  The only supported contract
    is the deployed one: temperature 0.7, CFG zero, and a suffix-visible
    attention canvas.
    """

    if not math.isclose(
        float(temperature), DEPLOYED_TEMPERATURE, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise ValueError("deployment-matched scoring requires temperature=0.7")
    if not isinstance(lightweight_decoding_constraints, dict):
        raise ValueError("deployment-matched scoring requires lightweight constraints")
    required_geometry_masks = (
        "duplicate_coordinate_mask",
        "lattice_volume_mask",
        "pbc_min_distance_mask",
    )
    missing_geometry_masks = [
        name
        for name in required_geometry_masks
        if not bool(lightweight_decoding_constraints.get(name))
    ]
    if missing_geometry_masks:
        raise ValueError(
            "deployment-matched scoring requires active dynamic masks: "
            + ",".join(missing_geometry_masks)
        )
    state = _normalize_single_state(
        complete_tokens,
        prompt_length=int(prompt_length),
        gen_length=int(gen_length),
    )
    positions, transaction_kind = _normalize_transaction_positions(
        generation_positions,
        gen_length=int(gen_length),
    )
    context_positions = _normalize_context_masked_positions(
        context_masked_generation_positions,
        active_positions=positions,
        gen_length=int(gen_length),
    )
    actions = _normalize_actions(
        action_token_ids,
        width=len(positions),
        device=state.device,
    )
    action_count = int(actions.shape[0])
    absolute_positions = tuple(int(prompt_length) + value for value in positions)
    active_absolute_tensor = torch.tensor(
        absolute_positions, dtype=torch.long, device=state.device
    )
    outside_active_and_context = torch.ones(
        (int(gen_length),), dtype=torch.bool, device=state.device
    )
    outside_active_and_context[
        torch.tensor(positions, dtype=torch.long, device=state.device)
    ] = False
    if context_positions:
        outside_active_and_context[
            torch.tensor(context_positions, dtype=torch.long, device=state.device)
        ] = False
    generation = state[0, int(prompt_length) : int(prompt_length) + int(gen_length)]
    if context_positions:
        context_tensor = torch.tensor(
            context_positions, dtype=torch.long, device=state.device
        )
        if not bool(
            (generation[context_tensor] == int(mask_id)).all().detach().item()
        ):
            raise ValueError("declared context-masked positions must contain mask_id")
    if bool(
        (generation[outside_active_and_context] == int(mask_id)).any().detach().item()
    ):
        raise ValueError(
            "generation masks outside active and declared context positions are not allowed"
        )

    current = state.repeat(action_count, 1).clone()
    current[:, active_absolute_tensor] = int(mask_id)
    expanded_attention = _expand_attention(
        attention_mask,
        action_count=action_count,
        device=state.device,
    )
    full_attention = _full_attention_mask(
        current,
        expanded_attention,
        prompt_length=int(prompt_length),
        gen_length=int(gen_length),
    )
    if full_attention is not None and not bool(
        full_attention[
            :, int(prompt_length) : int(prompt_length) + int(gen_length)
        ]
        .bool()
        .all()
        .detach()
        .item()
    ):
        raise ValueError("deployment-matched scoring requires suffix-visible attention")
    prompt_index = torch.zeros_like(current, dtype=torch.bool)
    prompt_index[:, : int(prompt_length)] = True

    output_embeddings = model.get_output_embeddings()
    vocab_size = int(output_embeddings.weight.shape[0])
    if not 0 <= int(mask_id) < vocab_size:
        raise ValueError("mask_id lies outside the model vocabulary")
    allowed_mask = _build_allowed_mask(
        allowed_token_ids_by_generation_pos,
        gen_length=int(gen_length),
        vocab_size=vocab_size,
        device=state.device,
    )
    prepared_atom_count_grammar = _prepare_atom_count_grammar(
        atom_count_grammar,
        vocab_size,
        state.device,
    )

    valid = torch.ones((action_count,), dtype=torch.bool, device=state.device)
    joint_logprob = torch.zeros(
        (action_count,), dtype=torch.float32, device=state.device
    )
    component_logprobs: list[list[float | None]] = [
        [None] * len(positions) for _ in range(action_count)
    ]
    invalid_step: list[int | None] = [None] * action_count
    invalid_position: list[int | None] = [None] * action_count
    invalid_reason: list[str | None] = [None] * action_count
    rows = torch.arange(action_count, device=state.device)

    for step, (generation_position, absolute_position) in enumerate(
        zip(positions, absolute_positions, strict=True)
    ):
        logits = _deployment_model_logits(
            model,
            current,
            full_attention,
            prompt_index,
            int(mask_id),
        )
        if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
            raise RuntimeError("DLM did not return rank-three logits")
        if tuple(logits.shape[:2]) != tuple(current.shape):
            raise RuntimeError("DLM logits do not match the scoring canvas")
        if int(logits.shape[-1]) != vocab_size:
            raise RuntimeError("DLM output vocabulary changed during scoring")

        _apply_schema_masks(
            logits,
            current,
            int(prompt_length),
            int(gen_length),
            allowed_mask,
            prepared_atom_count_grammar,
        )
        sentinel = torch.finfo(logits.dtype).min
        target = actions[:, int(step)]
        target_in_vocab = (target >= 0) & (target < vocab_size)
        safe_target = target.clamp(min=0, max=vocab_size - 1)
        schema_vector = logits[rows, int(absolute_position)]
        schema_masked = schema_vector == sentinel
        schema_numerically_valid = (
            torch.isfinite(schema_vector) | schema_masked
        ).all(dim=1)
        schema_legal = ~schema_masked & torch.isfinite(schema_vector)
        schema_target_legal = target_in_vocab & schema_legal.gather(
            1, safe_target.unsqueeze(1)
        ).squeeze(1)

        active_generation_mask = torch.zeros(
            (action_count, int(gen_length)),
            dtype=torch.bool,
            device=state.device,
        )
        active_generation_mask[:, int(generation_position)] = True
        mask_report = _apply_lightweight_decoding_masks(
            logits,
            current,
            int(prompt_length),
            int(gen_length),
            lightweight_decoding_constraints,
            active_generation_mask,
            int(mask_id),
        )
        vector = logits[rows, int(absolute_position)]
        dynamically_masked = vector == sentinel
        finite_or_masked = torch.isfinite(vector) | dynamically_masked
        numerical_logits_valid = (
            schema_numerically_valid & finite_or_masked.all(dim=1)
        )
        # Retain the schema support explicitly.  In particular, a later
        # logaddexp-based alias canonicalization must never resurrect a token
        # that the exact position schema had already removed.
        legal = schema_legal & ~dynamically_masked & torch.isfinite(vector)
        has_legal_token = legal.any(dim=1)
        dynamic_target_legal = target_in_vocab & legal.gather(
            1, safe_target.unsqueeze(1)
        ).squeeze(1)
        pbc_no_legal = torch.tensor(
            [
                (row, int(generation_position))
                in mask_report.get("pbc_no_legal_completion", set())
                for row in range(action_count)
            ],
            dtype=torch.bool,
            device=state.device,
        )
        step_valid = (
            target_in_vocab
            & schema_target_legal
            & dynamic_target_legal
            & has_legal_token
            & numerical_logits_valid
            & ~pbc_no_legal
        )

        scaled = (vector.float() / float(temperature)).masked_fill(~legal, -torch.inf)
        raw_log_normalizer = torch.logsumexp(scaled, dim=1)
        safe_log_normalizer = torch.where(
            has_legal_token & numerical_logits_valid,
            raw_log_normalizer,
            torch.zeros_like(raw_log_normalizer),
        )
        target_scaled = scaled.gather(1, safe_target.unsqueeze(1)).squeeze(1)
        safe_target_scaled = torch.where(
            step_valid,
            target_scaled,
            torch.zeros_like(target_scaled),
        )
        finite_step_logprob = safe_target_scaled - safe_log_normalizer
        finite_step_logprob = torch.where(
            step_valid,
            finite_step_logprob,
            torch.zeros_like(finite_step_logprob),
        )
        alive_step = valid & step_valid
        joint_logprob = torch.where(
            alive_step,
            joint_logprob + finite_step_logprob,
            torch.full_like(joint_logprob, -torch.inf),
        )

        for row in range(action_count):
            if bool(alive_step[row].detach().item()):
                component_logprobs[row][step] = float(
                    finite_step_logprob[row].detach().cpu().item()
                )
            elif bool(valid[row].detach().item()):
                component_logprobs[row][step] = -math.inf
                invalid_step[row] = int(step)
                invalid_position[row] = int(generation_position)
                invalid_reason[row] = _first_invalid_reason(
                    target_in_vocab=bool(target_in_vocab[row].detach().item()),
                    schema_target_legal=bool(
                        schema_target_legal[row].detach().item()
                    ),
                    dynamic_target_legal=bool(
                        dynamic_target_legal[row].detach().item()
                    ),
                    has_legal_token=bool(has_legal_token[row].detach().item()),
                    numerical_logits_valid=bool(
                        numerical_logits_valid[row].detach().item()
                    ),
                    pbc_no_legal_completion=bool(
                        pbc_no_legal[row].detach().item()
                    ),
                )
        valid = alive_step
        commit_rows = torch.nonzero(valid, as_tuple=False).flatten()
        if int(commit_rows.numel()) > 0:
            current = current.clone()
            current[commit_rows, int(absolute_position)] = target[commit_rows]

    if transaction_kind == "cell":
        cell_supported_by_row: list[bool] = []
        for row in range(action_count):
            if not bool(valid[row].detach().item()):
                cell_supported_by_row.append(True)
                continue
            supported = _complete_cell_is_supported(
                current[row],
                prompt_length=int(prompt_length),
                constraints=lightweight_decoding_constraints,
            )
            if supported:
                cell_supported_by_row.append(True)
                continue
            cell_supported_by_row.append(False)
            invalid_step[row] = len(positions)
            invalid_position[row] = None
            invalid_reason[row] = "cell_geometry_unsupported"
        # Do not mutate ``valid`` in place: it participates in the autograd
        # graph through earlier ``torch.where`` operations.  An unsupported
        # completed cell is an out-of-place terminal support update.
        cell_supported = torch.tensor(
            cell_supported_by_row, dtype=torch.bool, device=state.device
        )
        newly_unsupported = valid & ~cell_supported
        valid = valid & cell_supported
        joint_logprob = torch.where(
            newly_unsupported,
            torch.full_like(joint_logprob, -torch.inf),
            joint_logprob,
        )

    duplicate_of: list[int | None] = [None] * action_count
    first_by_action: dict[tuple[int, ...], int] = {}
    unique_valid_indices: list[int] = []
    for row in range(action_count):
        action = tuple(int(value) for value in actions[row].detach().cpu().tolist())
        previous = first_by_action.get(action)
        if previous is None:
            first_by_action[action] = row
            if bool(valid[row].detach().item()):
                unique_valid_indices.append(row)
        else:
            duplicate_of[row] = int(previous)

    if unique_valid_indices:
        candidate_log_mass = torch.logsumexp(
            joint_logprob[
                torch.tensor(
                    unique_valid_indices, dtype=torch.long, device=state.device
                )
            ],
            dim=0,
        )
    else:
        candidate_log_mass = joint_logprob.new_tensor(-torch.inf)

    audits: list[TransactionActionAudit] = []
    for row in range(action_count):
        value = float(joint_logprob[row].detach().cpu().item())
        is_valid = bool(valid[row].detach().item())
        audits.append(
            TransactionActionAudit(
                action_index=int(row),
                token_ids=tuple(
                    int(token) for token in actions[row].detach().cpu().tolist()
                ),
                valid=is_valid,
                finite=bool(is_valid and math.isfinite(value)),
                joint_logprob=value,
                component_logprobs=tuple(component_logprobs[row]),
                invalid_step=invalid_step[row],
                invalid_generation_position=invalid_position[row],
                invalid_reason=invalid_reason[row],
                duplicate_of=duplicate_of[row],
            )
        )

    if bool(torch.isnan(joint_logprob).any().detach().item()) or bool(
        torch.isnan(candidate_log_mass).detach().item()
    ):
        raise FloatingPointError("transaction scoring produced NaN")
    return DeployedTransactionScores(
        action_logprobs=joint_logprob,
        valid_mask=valid,
        candidate_log_mass=candidate_log_mass,
        action_audits=tuple(audits),
        generation_positions=positions,
        context_masked_generation_positions=context_positions,
        transaction_kind=transaction_kind,
        temperature=float(temperature),
        cfg_scale=0.0,
        suffix_visible=True,
        unique_valid_action_count=len(unique_valid_indices),
    )


__all__ = [
    "DEPLOYED_TEMPERATURE",
    "DeployedTransactionScores",
    "TransactionActionAudit",
    "score_deployed_transaction_actions",
]
