#!/usr/bin/env python3
"""Single-trajectory sampler for the fused C3FD--Llama typed Planner.

The frozen C3FD-v2.5 model defines calibrated base logits and the constructive
Pauling-bitset support.  A trained Llama LoRA and typed residual module only
reweight that support through a unit-weight product of experts.  Every source
ordinal receives exactly one trajectory attempt; failures remain explicit.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Protocol, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
for candidate in (SRC_ROOT, SCRIPTS_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import torch
from torch import Tensor

from crystal_dlm.c3fd_calibration import StratumInteraction
from crystal_dlm.c3fd_native_plan import (
    C3FD_NATIVE_PLAN_VERSION,
    build_native_body_prompt,
    native_plan_from_parts,
    serialize_native_plan,
)
from crystal_dlm.c3fd_planner_model import C3FDPlannerConfig, C3FDPlannerModel
from crystal_dlm.c3fd_llama_typed_planner import (
    C3FDLlamaTypedPlannerConfig,
    C3FDLlamaTypedResidualPlanner,
    TypedResidualLogits,
    masked_log_softmax,
    unit_weight_poe_log_probs,
)
from crystal_dlm.ccfd import FormulaToken
from crystal_dlm.ccfd_v2 import CCFDv2State, SetAtomCount
from crystal_dlm.composition_pair_prior import ValenceNode
from crystal_dlm.family_reachability import PaulingBitsetReachability, state_symbols
from crystal_dlm.fixed_slot import SYMBOL_TO_Z, Z_TO_SYMBOL
from crystal_dlm.r5_plan_state import anion_framework_from_symbols
from crystal_dlm.semantic_composition_head import SemanticHeadFlags
from crystal_dlm.species_program_pointer import (
    PlanConditionedSpeciesPointer,
    SpeciesPointerConfig,
)
from sample_c3fd_plans import semantic_inputs


SCHEMA = "c3fd_llama_typed_single_trajectory_v1"
METRICS_SCHEMA = "c3fd_llama_typed_sampling_metrics_v1"
FINAL_CONFIG_SCHEMA = "c3fd_llama_typed_planner_final_config_v1"
FINAL_STATE_SCHEMA = "c3fd_llama_typed_planner_final_state_v1"
STABILITY_GOAL = "meta_or_better"
PROPOSAL_QUERY_STATE = 0
LEDGER_FEATURE_SIZE = 6


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256(path: Path, expected: str, *, label: str) -> str:
    value = str(expected).strip().lower()
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} expected SHA256 is malformed")
    observed = sha256_file(path)
    if observed != value:
        raise RuntimeError(f"{label} SHA256 mismatch")
    return observed


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number} is not an object")
            yield value


def load_requested_rows(path: Path, *, requested: int) -> list[dict[str, Any]]:
    rows = list(iter_jsonl(path))
    if len(rows) != int(requested):
        raise ValueError(f"source ledger must contain exactly {requested} rows")
    normalized: list[dict[str, Any]] = []
    seen: set[int] = set()
    for position, row in enumerate(rows):
        raw = row.get("sample_idx", row.get("ordinal"))
        if isinstance(raw, bool) or raw is None or int(raw) != raw:
            raise ValueError(f"source ledger row {position} lacks an integer ordinal")
        ordinal = int(raw)
        if ordinal < 0 or ordinal in seen:
            raise ValueError("source ledger ordinals must be unique nonnegative integers")
        seen.add(ordinal)
        normalized.append({"sample_idx": ordinal, "source_position": position})
    return normalized


def sample_log_probs(
    log_probs: Tensor,
    *,
    rng: torch.Generator,
    temperature: float,
    top_p: float,
    top_k: int,
) -> int:
    if int(top_k) != 0:
        raise ValueError("the fused typed contract fixes top_k=0")
    if not 0.0 < float(top_p) <= 1.0:
        raise ValueError("top_p must be in (0, 1]")
    if not math.isfinite(float(temperature)) or float(temperature) <= 0.0:
        raise ValueError("temperature must be finite and positive")
    values = log_probs.detach().float().cpu().clone() / float(temperature)
    finite = torch.isfinite(values)
    if not bool(finite.any().item()):
        raise ValueError("no finite sampling action")
    probabilities = torch.softmax(values, dim=-1)
    if float(top_p) < 1.0:
        sorted_probabilities, sorted_indices = torch.sort(probabilities, descending=True)
        cumulative = torch.cumsum(sorted_probabilities, dim=-1)
        remove = cumulative - sorted_probabilities > float(top_p)
        sorted_probabilities[remove] = 0.0
        probabilities.zero_().scatter_(0, sorted_indices, sorted_probabilities)
        probabilities /= probabilities.sum().clamp_min(1e-12)
    return int(torch.multinomial(probabilities, 1, generator=rng).item())


def distribution_audit(
    base_logits: Tensor,
    fused_log_probs: Tensor,
    legal_mask: Tensor,
    selected_index: int,
) -> dict[str, Any]:
    base_log_probs = masked_log_softmax(base_logits, legal_mask, name="audit base logits")
    legal_indices = torch.nonzero(legal_mask, as_tuple=False).flatten()
    if int(selected_index) not in legal_indices.tolist():
        raise ValueError("selected action is outside the legal support")
    base_probabilities = torch.exp(base_log_probs[legal_mask])
    fused_probabilities = torch.exp(fused_log_probs[legal_mask])
    kl = torch.sum(
        fused_probabilities * (fused_log_probs[legal_mask] - base_log_probs[legal_mask])
    )
    ordered = sorted(
        (int(index) for index in legal_indices.tolist()),
        key=lambda index: (-float(base_log_probs[index]), index),
    )
    return {
        "kl_fused_vs_c3fd": float(kl),
        "selected_action_base_c3fd_rank": ordered.index(int(selected_index)) + 1,
        "legal_support_size": len(ordered),
    }


def _branch_vector(branch: str | None) -> tuple[float, float, float]:
    values = {
        None: (1.0, 0.0, 0.0),
        "ionic": (0.0, 1.0, 0.0),
        "alloy": (0.0, 0.0, 1.0),
    }
    if branch not in values:
        raise ValueError(f"unsupported C3FD branch {branch!r}")
    return values[branch]


def _ledger_row(state: CCFDv2State, target_arity: int) -> list[float]:
    return [
        float(state.remaining_atoms or 0) / 20.0,
        float(state.net_charge) / 160.0,
        float(int(target_arity) - len(state.tokens)) / 7.0,
        *_branch_vector(state.branch),
    ]


@dataclass(frozen=True)
class TypedSequence:
    stability_goal_ids: Tensor
    proposal_state_ids: Tensor
    previous_species_indices: Tensor
    previous_count_values: Tensor
    ledger_features: Tensor

    @property
    def length(self) -> int:
        return int(self.proposal_state_ids.shape[1])


def build_typed_sequence(
    *,
    stability_goal_id: int,
    proposal_state_id: int = PROPOSAL_QUERY_STATE,
    target_arity: int | None = None,
    species_ids: Sequence[int] = (),
    counts: Sequence[int] = (),
    state_history: Sequence[CCFDv2State] = (),
) -> TypedSequence:
    if len(species_ids) != len(counts):
        raise ValueError("typed species/count history mismatch")
    if target_arity is None:
        if species_ids or state_history or int(proposal_state_id) != 0:
            raise ValueError("proposal query cannot contain composition history")
        return TypedSequence(
            stability_goal_ids=torch.tensor([int(stability_goal_id)], dtype=torch.long),
            proposal_state_ids=torch.tensor([[PROPOSAL_QUERY_STATE]], dtype=torch.long),
            previous_species_indices=torch.tensor([[-1]], dtype=torch.long),
            previous_count_values=torch.tensor([[0]], dtype=torch.long),
            ledger_features=torch.zeros((1, 1, LEDGER_FEATURE_SIZE), dtype=torch.float32),
        )
    if len(state_history) != len(species_ids) + 1:
        raise ValueError("state history must contain post-N and every post-action state")
    if int(proposal_state_id) <= 0:
        raise ValueError("composition queries require frozen stratum index plus one")
    proposal_states = [PROPOSAL_QUERY_STATE, int(proposal_state_id)]
    previous_species = [-1, -1]
    previous_counts = [0, 0]
    ledger = [[0.0] * LEDGER_FEATURE_SIZE, _ledger_row(state_history[0], target_arity)]
    for species_id, count, state in zip(species_ids, counts, state_history[1:]):
        proposal_states.append(int(proposal_state_id))
        previous_species.append(int(species_id))
        previous_counts.append(int(count))
        ledger.append(_ledger_row(state, target_arity))
    return TypedSequence(
        stability_goal_ids=torch.tensor([int(stability_goal_id)], dtype=torch.long),
        proposal_state_ids=torch.tensor([proposal_states], dtype=torch.long),
        previous_species_indices=torch.tensor([previous_species], dtype=torch.long),
        previous_count_values=torch.tensor([previous_counts], dtype=torch.long),
        ledger_features=torch.tensor([ledger], dtype=torch.float32),
    )


def recompute_llama_hidden(llama: Any, inputs_embeds: Tensor) -> Tensor:
    """Recompute the complete short sequence and explicitly disable KV cache."""

    attention_mask = torch.ones(
        inputs_embeds.shape[:2], dtype=torch.long, device=inputs_embeds.device
    )
    output = llama(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    hidden_states = getattr(output, "hidden_states", None)
    if not hidden_states:
        raise RuntimeError("Llama did not return hidden states")
    hidden = hidden_states[-1]
    if hidden.shape[:2] != inputs_embeds.shape[:2]:
        raise RuntimeError("Llama hidden-state sequence changed shape")
    return hidden


class RuntimeProtocol(Protocol):
    interaction: Any
    family_values: Sequence[str]
    nodes: Sequence[ValenceNode]
    node_to_id: Mapping[ValenceNode, int]
    max_count: int
    max_species: int
    eos_action_index: int
    stability_goal_id: int
    reachability: Any
    soft_values: Mapping[str, Sequence[str]]

    def proposal_logits(self) -> Tensor: ...
    def action_logits(
        self,
        state: CCFDv2State,
        *,
        target_n: int,
        target_arity: int,
        species_ids: Sequence[int],
        counts: Sequence[int],
        state_history: Sequence[CCFDv2State],
    ) -> tuple[Tensor, Any]: ...
    def residual_logits(self, sequence: TypedSequence) -> TypedResidualLogits: ...
    def soft_logits(self, c3fd_context: Any) -> Mapping[str, Tensor]: ...
    def terminal_certificate(self, state: CCFDv2State) -> Mapping[str, Any]: ...


def proposal_legal_mask(runtime: RuntimeProtocol, base_logits: Tensor) -> Tensor:
    if int(base_logits.numel()) != len(runtime.interaction.strata):
        raise ValueError("proposal logits/strata mismatch")
    mask = torch.zeros_like(base_logits, dtype=torch.bool)
    for index, (family_id, target_n, target_arity) in enumerate(runtime.interaction.strata):
        family = str(runtime.family_values[int(family_id)])
        if family == "<UNKNOWN>":
            continue
        state = CCFDv2State.start().apply(SetAtomCount(int(target_n)))
        mask[index] = bool(
            runtime.reachability.can_complete(
                state,
                family=family,
                target_arity=int(target_arity),
                max_species=int(runtime.max_species),
            )
        )
    if not bool(mask.any().item()):
        raise ValueError("proposal pre-mask contains no completable stratum")
    return mask


def action_legal_mask(
    runtime: RuntimeProtocol,
    state: CCFDv2State,
    *,
    family: str,
    target_arity: int,
) -> Tensor:
    size = int(runtime.eos_action_index) + 1
    mask = torch.zeros(size, dtype=torch.bool)
    exact_terminal = bool(
        state.eos_legal
        and len(state.tokens) == int(target_arity)
        and anion_framework_from_symbols(state_symbols(state)) == str(family)
    )
    if exact_terminal:
        certificate = runtime.terminal_certificate(state)
        if bool(certificate.get("benchmark_compatible")):
            mask[int(runtime.eos_action_index)] = True
        return mask
    legal_tokens = runtime.reachability.legal_species_counts(
        state,
        family=str(family),
        target_arity=int(target_arity),
        max_species=int(runtime.max_species),
    )
    for token in legal_tokens:
        node = ValenceNode(int(token.atomic_number), int(token.oxidation_state))
        species_id = runtime.node_to_id.get(node)
        if species_id is None:
            continue
        count = int(token.count)
        if 1 <= count <= int(runtime.max_count):
            mask[int(species_id) * int(runtime.max_count) + count - 1] = True
    return mask


def _soft_legal_mask(values: Sequence[str], logits: Tensor) -> Tensor:
    if len(values) != int(logits.numel()):
        raise ValueError("soft vocabulary/logits mismatch")
    mask = torch.tensor(
        [str(value) != "<UNKNOWN>" for value in values],
        dtype=torch.bool,
        device=logits.device,
    )
    if not bool(mask.any().item()):
        raise ValueError("soft field has no known value")
    return mask


def _benchmark_compatible(certificate: Mapping[str, Any]) -> bool:
    return bool(
        certificate.get("benchmark_compatible")
        or certificate.get("benchmark_valid") is True
        or certificate.get("certificate_class") == "benchmark_compatible"
    )


def sample_single_trajectory(
    runtime: RuntimeProtocol,
    *,
    sample_idx: int,
    seed: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> dict[str, Any]:
    rng = torch.Generator(device="cpu")
    rng.manual_seed(int(seed) * 1_000_003 + int(sample_idx))
    audit: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []

    proposal_base = runtime.proposal_logits().detach().float().cpu()
    proposal_mask = proposal_legal_mask(runtime, proposal_base)
    proposal_sequence = build_typed_sequence(
        stability_goal_id=int(runtime.stability_goal_id)
    )
    proposal_residuals = runtime.residual_logits(proposal_sequence)
    proposal_residual = proposal_residuals.proposal[0].detach().float().cpu()
    proposal_fused = unit_weight_poe_log_probs(
        proposal_base, proposal_residual, proposal_mask
    )
    proposal_index = sample_log_probs(
        proposal_fused,
        rng=rng,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )
    family_id, target_n, target_arity = runtime.interaction.strata[proposal_index]
    family = str(runtime.family_values[int(family_id)])
    proposal_event = {
        "step": "proposal",
        "selected_index": int(proposal_index),
        "family": family,
        "N": int(target_n),
        "arity": int(target_arity),
        "pre_masked_strata": int((~proposal_mask).sum().item()),
        **distribution_audit(
            proposal_base, proposal_fused, proposal_mask, proposal_index
        ),
    }
    audit.append(proposal_event)
    trace.append(
        {"action": "proposal", "family": family, "N": int(target_n), "arity": int(target_arity)}
    )

    state = CCFDv2State.start().apply(SetAtomCount(int(target_n)))
    state_history = [state]
    species_ids: list[int] = []
    counts: list[int] = []
    final_c3fd_context: Any = None
    eos_seen = False
    for action_step in range(int(runtime.max_species) + 1):
        sequence = build_typed_sequence(
            stability_goal_id=int(runtime.stability_goal_id),
            proposal_state_id=int(proposal_index) + 1,
            target_arity=int(target_arity),
            species_ids=species_ids,
            counts=counts,
            state_history=state_history,
        )
        base_action, c3fd_context = runtime.action_logits(
            state,
            target_n=int(target_n),
            target_arity=int(target_arity),
            species_ids=species_ids,
            counts=counts,
            state_history=state_history,
        )
        base_action = base_action.detach().float().cpu()
        legal = action_legal_mask(
            runtime, state, family=family, target_arity=int(target_arity)
        )
        if not bool(legal.any().item()):
            raise ValueError("semantic_dead_end")
        residual = runtime.residual_logits(sequence).actions[0, -1].detach().float().cpu()
        fused = unit_weight_poe_log_probs(base_action, residual, legal)
        selected = sample_log_probs(
            fused,
            rng=rng,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        event = {
            "step": "action",
            "action_step": int(action_step),
            "selected_index": int(selected),
            **distribution_audit(base_action, fused, legal, selected),
        }
        if selected == int(runtime.eos_action_index):
            if not bool(legal[int(runtime.eos_action_index)].item()):
                raise ValueError("illegal_eos")
            state = state.end()
            event["action"] = "EOS"
            trace.append({"action": "EOS"})
            audit.append(event)
            final_c3fd_context = c3fd_context
            eos_seen = True
            break
        species_id = int(selected) // int(runtime.max_count)
        count = int(selected) % int(runtime.max_count) + 1
        node = runtime.nodes[species_id]
        token = FormulaToken(int(node.atomic_number), int(node.oxidation_state), count)
        state = state.apply(token, max_species=int(runtime.max_species))
        state_history.append(state)
        species_ids.append(species_id)
        counts.append(count)
        event.update(
            {
                "action": "species_count",
                "atomic_number": int(node.atomic_number),
                "oxidation_state": int(node.oxidation_state),
                "count": count,
            }
        )
        trace.append(
            {
                "action": "species",
                "atomic_number": int(node.atomic_number),
                "oxidation_state": int(node.oxidation_state),
                "count": count,
            }
        )
        audit.append(event)
        final_c3fd_context = c3fd_context
    if not eos_seen or not state.ended:
        raise ValueError("trajectory_did_not_reach_exact_eos")
    certificate = runtime.terminal_certificate(state)
    if not _benchmark_compatible(certificate):
        raise ValueError("terminal_certificate_not_benchmark_compatible")
    if len(state.tokens) != int(target_arity) or int(state.target_atoms or 0) != int(target_n):
        raise ValueError("terminal_N_or_arity_mismatch")
    if anion_framework_from_symbols(state_symbols(state)) != family:
        raise ValueError("terminal_family_mismatch")

    terminal_sequence = build_typed_sequence(
        stability_goal_id=int(runtime.stability_goal_id),
        proposal_state_id=int(proposal_index) + 1,
        target_arity=int(target_arity),
        species_ids=species_ids,
        counts=counts,
        state_history=state_history,
    )
    terminal_residual = runtime.residual_logits(terminal_sequence).soft_fields
    c3fd_soft = runtime.soft_logits(final_c3fd_context)
    selected_soft: dict[str, str] = {}
    for field in ("lattice_system", "spacegroup_bucket", "volume_per_atom_bin"):
        base = c3fd_soft[field].detach().float().cpu()
        residual = terminal_residual[field][0].detach().float().cpu()
        legal = _soft_legal_mask(runtime.soft_values[field], base)
        fused = unit_weight_poe_log_probs(base, residual, legal)
        selected = sample_log_probs(
            fused,
            rng=rng,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        value = str(runtime.soft_values[field][selected])
        selected_soft[field] = value
        audit.append(
            {
                "step": "soft_field",
                "field": field,
                "selected_index": int(selected),
                "selected_value": value,
                **distribution_audit(base, fused, legal, selected),
            }
        )

    composition = {
        "N": int(target_n),
        "elements": [Z_TO_SYMBOL[int(token.atomic_number)] for token in state.tokens],
        "counts": [int(token.count) for token in state.tokens],
        "anion_framework": family,
    }
    native = native_plan_from_parts(composition, selected_soft)
    plan_text = serialize_native_plan(native)
    plan_state = json.loads(plan_text)
    prompt = build_native_body_prompt(native)
    program_method = getattr(runtime, "species_program", None)
    if callable(program_method):
        program = program_method(
            terminal_sequence,
            plan_state=plan_state,
            selected_soft=selected_soft,
        )
    else:
        program = {
            "elements": list(plan_state["elements"]),
            "indices": list(range(len(plan_state["elements"]))),
            "source": "canonical_compatibility",
        }
    if sorted(program["elements"], key=lambda value: SYMBOL_TO_Z[value]) != list(
        plan_state["elements"]
    ):
        raise ValueError("species program changed the certified Plan element set")
    return {
        "schema": SCHEMA,
        "sample_idx": int(sample_idx),
        "stability_goal": STABILITY_GOAL,
        "trajectory_attempts": 1,
        "parsed": True,
        "comp_valid": True,
        "plan_text": plan_text,
        "plan_state": plan_state,
        "prompt": prompt,
        "prompt_schema": C3FD_NATIVE_PLAN_VERSION,
        "species_program": list(program["elements"]),
        "species_program_indices": [int(value) for value in program["indices"]],
        "species_program_source": str(program["source"]),
        "semantic_trace": trace,
        "audit": audit,
        "certificate": dict(certificate),
        "failure": None,
    }


def sample_requests(
    runtime: RuntimeProtocol,
    source_rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    plans: list[dict[str, Any]] = []
    failures: Counter[str] = Counter()
    started = time.perf_counter()
    for source in source_rows:
        sample_idx = int(source["sample_idx"])
        try:
            record = sample_single_trajectory(
                runtime,
                sample_idx=sample_idx,
                seed=seed,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
            plans.append(
                {
                    key: record[key]
                    for key in (
                        "sample_idx",
                        "plan_text",
                        "plan_state",
                        "prompt",
                        "prompt_schema",
                        "species_program",
                        "species_program_indices",
                        "species_program_source",
                    )
                }
            )
        except Exception as exc:  # one requested ordinal becomes one final failure row
            reason = f"{type(exc).__name__}:{exc}"
            failures[reason] += 1
            record = {
                "schema": SCHEMA,
                "sample_idx": sample_idx,
                "stability_goal": STABILITY_GOAL,
                "trajectory_attempts": 1,
                "parsed": False,
                "comp_valid": False,
                "plan_text": None,
                "plan_state": None,
                "prompt": None,
                "prompt_schema": C3FD_NATIVE_PLAN_VERSION,
                "semantic_trace": [],
                "audit": [],
                "certificate": None,
                "failure": reason,
            }
        records.append(record)
    requested = len(source_rows)
    valid = sum(bool(row["comp_valid"]) for row in records)
    metrics = {
        "schema": METRICS_SCHEMA,
        "requested_samples": requested,
        "parsed_samples": len(plans),
        "all_request_benchmark_comp_valid": valid,
        "comp_valid_rate_requested_denominator": valid / requested if requested else 0.0,
        "stability_goal": STABILITY_GOAL,
        "seed": int(seed),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "top_k": int(top_k),
        "one_trajectory_per_ordinal": True,
        "retry": False,
        "filter": False,
        "replacement": False,
        "rerank": False,
        "best_of_n": False,
        "failures": dict(failures.most_common()),
        "species_program_sources": dict(
            Counter(
                str(row.get("species_program_source"))
                for row in records
                if row.get("parsed")
            )
        ),
        "elapsed_sec": time.perf_counter() - started,
    }
    return records, plans, metrics


class ProductionRuntime:
    def __init__(
        self,
        *,
        c3fd: C3FDPlannerModel,
        c3fd_context: Tensor,
        interaction: StratumInteraction,
        calibration: Mapping[str, Any],
        vocabulary: Mapping[str, Any],
        reachability: PaulingBitsetReachability,
        llama: Any,
        typed: C3FDLlamaTypedResidualPlanner,
        stability_goal_id: int,
        device: torch.device,
        max_species: int,
        species_pointer: PlanConditionedSpeciesPointer | None = None,
    ) -> None:
        self.c3fd = c3fd
        self.c3fd_context = c3fd_context
        self.interaction = interaction
        self.calibration = calibration
        self.family_values = vocabulary["soft_vocabulary"]["anion_framework"]
        self.soft_values = vocabulary["soft_vocabulary"]
        species_rows = sorted(vocabulary["species"], key=lambda row: int(row["id"]))
        self.nodes = tuple(
            ValenceNode(int(row["atomic_number"]), int(row["oxidation_state"]))
            for row in species_rows
        )
        self.node_to_id = {node: index for index, node in enumerate(self.nodes)}
        self.max_count = int(c3fd.head.max_count)
        self.max_species = int(max_species)
        self.species_pointer = species_pointer
        self.eos_action_index = int(c3fd.head.eos_action_index)
        self.reachability = reachability
        self.llama = llama
        self.typed = typed
        self.stability_goal_id = int(stability_goal_id)
        self.device = device

    def _c3fd_first(self) -> Any:
        sentinel_species = torch.tensor([[-1]], device=self.device)
        sentinel_count = torch.tensor([[0]], device=self.device)
        sentinel_n = torch.tensor([[0]], device=self.device)
        sentinel_ledger = torch.zeros((1, 1, 6), device=self.device)
        with torch.inference_mode():
            return self.c3fd(
                self.c3fd_context,
                previous_species_indices=sentinel_species,
                previous_count_values=sentinel_count,
                previous_n_values=sentinel_n,
                ledger_features=sentinel_ledger,
                flags=SemanticHeadFlags(use_physics=True),
            )

    def proposal_logits(self) -> Tensor:
        first = self._c3fd_first()
        return self.interaction.joint_scores(
            first.family_logits[0],
            first.n_logits[0],
            first.arity_logits[0],
            family_temperature=float(self.calibration["family"]["temperature"]),
            n_temperature=float(self.calibration["n"]["temperature"]),
            arity_temperature=float(self.calibration["arity"]["temperature"]),
        )

    def action_logits(
        self,
        state: CCFDv2State,
        *,
        target_n: int,
        target_arity: int,
        species_ids: Sequence[int],
        counts: Sequence[int],
        state_history: Sequence[CCFDv2State],
    ) -> tuple[Tensor, Any]:
        previous_species, previous_count, previous_n, ledger, position = semantic_inputs(
            target_n,
            species_ids,
            counts,
            state_history=state_history,
            target_arity=target_arity,
        )
        with torch.inference_mode():
            output = self.c3fd(
                self.c3fd_context,
                previous_species_indices=previous_species.to(self.device),
                previous_count_values=previous_count.to(self.device),
                previous_n_values=previous_n.to(self.device),
                ledger_features=ledger.to(self.device),
                flags=SemanticHeadFlags(use_physics=True),
            )
        species = output.species_logits[0, position] / float(
            self.calibration["species"]["temperature"]
        )
        count = output.count_logits[0, position] / float(
            self.calibration["count"]["temperature"]
        )
        unmasked = torch.ones(
            self.c3fd.head.num_joint_actions, dtype=torch.bool, device=self.device
        )
        joint = self.c3fd.head.joint_action_scores(
            species.unsqueeze(0),
            count.unsqueeze(0),
            legal_action_mask=unmasked,
            flags=SemanticHeadFlags(use_pair_prior=False, use_hard_mask=True),
        )[0]
        return joint, SimpleNamespace(output=output, position=position)

    def _sequence_hidden(self, sequence: TypedSequence) -> Tensor:
        embeds = self.typed.typed_inputs_embeds(
            stability_goal_ids=sequence.stability_goal_ids,
            proposal_state_ids=sequence.proposal_state_ids,
            previous_species_indices=sequence.previous_species_indices,
            previous_count_values=sequence.previous_count_values,
            ledger_features=sequence.ledger_features,
        )
        llama_dtype = next(self.llama.parameters()).dtype
        hidden = recompute_llama_hidden(
            self.llama, embeds.to(device=self.device, dtype=llama_dtype)
        )
        return hidden

    def residual_logits(self, sequence: TypedSequence) -> TypedResidualLogits:
        hidden = self._sequence_hidden(sequence)
        typed_dtype = next(self.typed.parameters()).dtype
        return self.typed(
            hidden.to(dtype=typed_dtype),
            soft_position_indices=torch.tensor(
                [sequence.length - 1], dtype=torch.long, device=self.device
            ),
        )

    def species_program(
        self,
        sequence: TypedSequence,
        *,
        plan_state: Mapping[str, Any],
        selected_soft: Mapping[str, str],
    ) -> dict[str, Any]:
        elements = [str(value) for value in plan_state["elements"]]
        counts = [int(value) for value in plan_state["counts"]]
        if self.species_pointer is None:
            return {
                "elements": elements,
                "indices": list(range(len(elements))),
                "source": "canonical_control",
            }
        hidden = self._sequence_hidden(sequence)
        terminal = hidden[:, sequence.length - 1, :].float()
        atomic = torch.tensor(
            [[int(SYMBOL_TO_Z[value]) for value in elements]],
            dtype=torch.long,
            device=self.device,
        )
        element_counts = torch.tensor(
            [counts], dtype=torch.long, device=self.device
        )
        valid = torch.ones_like(atomic, dtype=torch.bool)
        soft_ids = torch.tensor(
            [
                [
                    list(self.soft_values[field]).index(str(selected_soft[field]))
                    for field in (
                        "lattice_system",
                        "spacegroup_bucket",
                        "volume_per_atom_bin",
                    )
                ]
            ],
            dtype=torch.long,
            device=self.device,
        )
        with torch.inference_mode():
            order = self.species_pointer.decode(
                terminal,
                atomic,
                element_counts,
                valid,
                soft_ids,
            )[0, : len(elements)].tolist()
        indices = [int(value) for value in order]
        if sorted(indices) != list(range(len(elements))):
            raise RuntimeError("species pointer did not return an exact permutation")
        return {
            "elements": [elements[index] for index in indices],
            "indices": indices,
            "source": "planner_llama_pointer",
        }

    def soft_logits(self, c3fd_context: Any) -> Mapping[str, Tensor]:
        if c3fd_context is None:
            raise RuntimeError("missing terminal C3FD context")
        return {
            field: c3fd_context.output.rich_logits[field][0, c3fd_context.position]
            for field in ("lattice_system", "spacegroup_bucket", "volume_per_atom_bin")
        }

    def terminal_certificate(self, state: CCFDv2State) -> Mapping[str, Any]:
        terminal = state if state.ended else state.end()
        payload = terminal.certificate().to_dict()
        payload["benchmark_compatible"] = bool(
            payload.get("benchmark_valid") is True
            or payload.get("certificate_class") == "benchmark_compatible"
        )
        return payload


def load_production_runtime(args: argparse.Namespace) -> ProductionRuntime:
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("the fused typed Planner sampler requires its allocated GPU")
    vocabulary_path = args.data_dir / "vocabulary.json"
    vocabulary_bytes = vocabulary_path.read_bytes()
    vocabulary = json.loads(vocabulary_bytes)
    c3fd_payload = torch.load(args.c3fd_checkpoint, map_location="cpu")
    if c3fd_payload.get("vocabulary_sha256") != hashlib.sha256(vocabulary_bytes).hexdigest():
        raise RuntimeError("C3FD checkpoint/vocabulary mismatch")
    calibration = c3fd_payload.get("calibration") or {}
    if set(calibration) != {"family", "n", "arity", "species", "count"}:
        raise RuntimeError("frozen C3FD-v2.5 checkpoint lacks complete calibration")
    c3fd_config = C3FDPlannerConfig(**c3fd_payload["config"])
    physics = torch.tensor(vocabulary["physics"]["matrix"], dtype=torch.float32)
    c3fd = C3FDPlannerModel(c3fd_config, physics_features=physics)
    c3fd.load_state_dict(c3fd_payload["model_state"], strict=True)
    c3fd.to(device).eval()
    interaction = StratumInteraction.from_dict(c3fd_payload["stratum_interaction"])

    config_path = args.fused_planner_final / "typed_residual_config.json"
    state_path = args.fused_planner_final / "typed_residual_state.pt"
    adapter_dir = args.fused_planner_final / "llama_adapter"
    if not (args.fused_planner_final / "_SUCCESS").is_file():
        raise FileNotFoundError("fused Planner final lacks _SUCCESS")
    config_payload = json.loads(config_path.read_text(encoding="utf-8"))
    if config_payload.get("schema") != FINAL_CONFIG_SCHEMA:
        raise RuntimeError("typed residual final config schema changed")
    typed_config = C3FDLlamaTypedPlannerConfig(
        **config_payload["typed_planner_config"]
    )
    if config_payload.get("proposal_state_encoding") != (
        "zero_query_then_frozen_stratum_index_plus_one"
    ):
        raise RuntimeError("trained typed Planner uses different proposal-state encoding")
    goal_ids = config_payload.get("stability_goal_to_id") or {}
    if STABILITY_GOAL not in goal_ids:
        raise RuntimeError("trained typed Planner lacks meta_or_better goal")
    state_payload = torch.load(state_path, map_location="cpu")
    if state_payload.get("schema") != FINAL_STATE_SCHEMA:
        raise RuntimeError("typed residual final state schema changed")
    typed = C3FDLlamaTypedResidualPlanner(typed_config)
    typed.load_state_dict(state_payload["state_dict"], strict=True)
    typed.to(device).eval()

    species_pointer = None
    if args.species_pointer_state is not None:
        pointer_payload = torch.load(args.species_pointer_state, map_location="cpu")
        if pointer_payload.get("schema") != "spad_species_pointer_state_v1":
            raise RuntimeError("SPAD species-pointer state schema changed")
        pointer_config = SpeciesPointerConfig(**pointer_payload["config"])
        if int(pointer_config.llama_hidden_size) != int(typed_config.llama_hidden_size):
            raise RuntimeError("species pointer and Planner Llama hidden sizes differ")
        species_pointer = PlanConditionedSpeciesPointer(pointer_config)
        species_pointer.load_state_dict(pointer_payload["state_dict"], strict=True)
        species_pointer.to(device).eval()
        for parameter in species_pointer.parameters():
            parameter.requires_grad_(False)

    from transformers import AutoModelForCausalLM
    from peft import PeftModel

    llama = AutoModelForCausalLM.from_pretrained(
        args.llama_model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    llama = PeftModel.from_pretrained(llama, adapter_dir, is_trainable=False)
    llama.to(device).eval()
    if int(llama.config.hidden_size) != int(typed_config.llama_hidden_size):
        raise RuntimeError("Llama and typed residual hidden sizes differ")
    if int(typed_config.num_proposal_states) != len(interaction.strata) + 1:
        raise RuntimeError("typed residual proposal-state count changed")
    if int(typed_config.ledger_feature_size) != LEDGER_FEATURE_SIZE:
        raise RuntimeError("typed residual ledger contract changed")
    if int(typed_config.num_proposal_strata) != len(interaction.strata):
        raise RuntimeError("typed/C3FD proposal strata differ")
    species_rows = sorted(vocabulary["species"], key=lambda row: int(row["id"]))
    if int(typed_config.num_species) != len(species_rows):
        raise RuntimeError("typed/C3FD species vocabularies differ")
    if int(typed_config.max_count) != int(c3fd.head.max_count):
        raise RuntimeError("typed/C3FD count vocabularies differ")
    for field, expected in typed_config.soft_head_dims.items():
        if len(vocabulary["soft_vocabulary"][field]) != int(expected):
            raise RuntimeError(f"typed/C3FD {field} vocabularies differ")
    nodes = tuple(
        ValenceNode(int(row["atomic_number"]), int(row["oxidation_state"]))
        for row in species_rows
    )
    return ProductionRuntime(
        c3fd=c3fd,
        c3fd_context=torch.as_tensor(
            c3fd_payload["context"], dtype=torch.float32, device=device
        ),
        interaction=interaction,
        calibration=calibration,
        vocabulary=vocabulary,
        reachability=PaulingBitsetReachability(nodes),
        llama=llama,
        typed=typed,
        stability_goal_id=int(goal_ids[STABILITY_GOAL]),
        device=device,
        max_species=int(args.max_species),
        species_pointer=species_pointer,
    )


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--c3fd-checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--llama-model", type=Path, required=True)
    parser.add_argument("--fused-planner-final", type=Path, required=True)
    parser.add_argument("--species-pointer-state", type=Path)
    parser.add_argument("--source-ledger", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-c3fd-sha256", required=True)
    parser.add_argument("--expected-vocabulary-sha256", required=True)
    parser.add_argument("--expected-source-ledger-sha256", required=True)
    parser.add_argument("--expected-typed-config-sha256", required=True)
    parser.add_argument("--expected-typed-state-sha256", required=True)
    parser.add_argument("--expected-adapter-config-sha256", required=True)
    parser.add_argument("--expected-adapter-model-sha256", required=True)
    parser.add_argument("--requested", type=int, default=256)
    parser.add_argument(
        "--expected-requested",
        type=int,
        default=256,
        help="Fail closed unless --requested matches the frozen run contract.",
    )
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument(
        "--expected-seed",
        type=int,
        default=21,
        help="Fail closed unless --seed matches this preregistered contract seed.",
    )
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--max-species", type=int, default=7)
    parser.add_argument(
        "--minimum-comp-valid",
        type=float,
        default=0.95,
        help="Nonblocking requested-denominator composition-validity target.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if int(args.expected_requested) <= 0:
        raise ValueError("expected-requested must be positive")
    if int(args.requested) != int(args.expected_requested):
        raise ValueError("requested must match expected-requested")
    if int(args.seed) != int(args.expected_seed):
        raise ValueError("seed must match expected-seed")
    if int(args.top_k) != 0 or float(args.temperature) != 0.9 or float(args.top_p) != 0.95:
        raise ValueError("sampling contract changed")
    if not 0.0 <= float(args.minimum_comp_valid) <= 1.0:
        raise ValueError("minimum-comp-valid must be in [0, 1]")
    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    input_paths = {
        "c3fd_checkpoint": args.c3fd_checkpoint,
        "vocabulary": args.data_dir / "vocabulary.json",
        "source_ledger": args.source_ledger,
        "typed_config": args.fused_planner_final / "typed_residual_config.json",
        "typed_state": args.fused_planner_final / "typed_residual_state.pt",
        "adapter_config": args.fused_planner_final / "llama_adapter" / "adapter_config.json",
        "adapter_model": args.fused_planner_final / "llama_adapter" / "adapter_model.safetensors",
    }
    expected = {
        "c3fd_checkpoint": args.expected_c3fd_sha256,
        "vocabulary": args.expected_vocabulary_sha256,
        "source_ledger": args.expected_source_ledger_sha256,
        "typed_config": args.expected_typed_config_sha256,
        "typed_state": args.expected_typed_state_sha256,
        "adapter_config": args.expected_adapter_config_sha256,
        "adapter_model": args.expected_adapter_model_sha256,
    }
    observed = {
        label: verify_sha256(path, expected[label], label=label)
        for label, path in input_paths.items()
    }
    source_rows = load_requested_rows(args.source_ledger, requested=args.requested)
    runtime = load_production_runtime(args)
    records, plans, metrics = sample_requests(
        runtime,
        source_rows,
        seed=args.seed,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )
    args.output_dir.mkdir(parents=True)
    write_jsonl(args.output_dir / "raw_generations.jsonl", records)
    write_jsonl(args.output_dir / "plans_for_dlm.jsonl", plans)
    metrics["minimum_comp_valid"] = float(args.minimum_comp_valid)
    metrics["minimum_comp_valid_met"] = (
        float(metrics["comp_valid_rate_requested_denominator"])
        >= float(args.minimum_comp_valid)
    )
    metrics["input_sha256"] = observed
    (args.output_dir / "sample_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    run_config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if not key.startswith("expected_")
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    output_files = sorted(
        path for path in args.output_dir.iterdir() if path.is_file()
    )
    (args.output_dir / "OUTPUTS.sha256").write_text(
        "".join(f"{sha256_file(path)}  {path.name}\n" for path in output_files),
        encoding="utf-8",
    )
    (args.output_dir / "_SUCCESS").touch()
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
