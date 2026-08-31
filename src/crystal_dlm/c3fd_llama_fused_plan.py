"""Typed supervision contract for the stability-conditioned C3FD--Llama Planner.

The custom Planner trainer consumes typed residual-head targets.  A textual
transcript is retained only to make frozen datasets human-auditable; it is not
the training or inference interface.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from crystal_dlm.ccfd import FormulaToken
from crystal_dlm.ccfd_v2 import (
    CCFDv2State,
    END_COMPOSITION,
    SetAtomCount,
    replay_actions,
)
from crystal_dlm.composition_pair_prior import ValenceNode
from crystal_dlm.family_reachability import PaulingBitsetReachability


FUSED_TYPED_PLAN_SCHEMA = "c3fd_llama_fused_typed_targets_v1"
STABILITY_META_OR_BETTER = "meta_or_better"
STABILITY_HIGHER = "higher"
STABILITY_CONDITIONS = (STABILITY_META_OR_BETTER, STABILITY_HIGHER)
SOFT_FIELDS = (
    "lattice_system",
    "spacegroup_bucket",
    "volume_per_atom_bin",
)


@dataclass
class TypedTargetContext:
    species_vocabulary: dict[int, tuple[int, int]]
    max_count: int
    reachability: PaulingBitsetReachability


def build_typed_target_context(
    vocabulary: Mapping[str, Any],
) -> TypedTargetContext:
    species = _species_vocabulary(vocabulary)
    nodes = tuple(
        ValenceNode(int(value[0]), int(value[1]))
        for _species_id, value in sorted(species.items())
    )
    return TypedTargetContext(
        species_vocabulary=species,
        max_count=_max_count(vocabulary),
        reachability=PaulingBitsetReachability(nodes),
    )


def stability_condition_from_e_above_hull(value: Any) -> str:
    """Map one finite MP20 hull value to the frozen binary condition."""

    if value is None or value == "":
        raise ValueError("missing e_above_hull")
    if isinstance(value, bool):
        raise ValueError("malformed e_above_hull")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("malformed e_above_hull") from exc
    if not math.isfinite(numeric):
        raise ValueError("nonfinite e_above_hull")
    return STABILITY_META_OR_BETTER if numeric <= 0.1 else STABILITY_HIGHER


def _strict_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{label} must be an integer")
    return result


def _safe_text(value: Any, *, label: str) -> str:
    text = str(value or "").strip()
    if not text or any(character.isspace() for character in text):
        raise ValueError(f"{label} must be one non-empty token")
    if any(character in text for character in "\r\n<>"):
        raise ValueError(f"{label} contains transcript control characters")
    return text


def _species_vocabulary(vocabulary: Mapping[str, Any]) -> dict[int, tuple[int, int]]:
    result: dict[int, tuple[int, int]] = {}
    for ordinal, raw in enumerate(vocabulary.get("species") or ()):
        if not isinstance(raw, Mapping):
            raise ValueError(f"species vocabulary row {ordinal} is not an object")
        species_id = _strict_int(raw.get("id"), label="species id")
        if species_id in result:
            raise ValueError(f"duplicate species id {species_id}")
        result[species_id] = (
            _strict_int(raw.get("atomic_number"), label="atomic number"),
            _strict_int(raw.get("oxidation_state"), label="oxidation state"),
        )
    if not result:
        raise ValueError("species vocabulary is empty")
    if sorted(result) != list(range(len(result))):
        raise ValueError("species vocabulary ids must be contiguous from zero")
    return result


def _max_count(vocabulary: Mapping[str, Any]) -> int:
    values = vocabulary.get("count_values")
    if values is None:
        return 20
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError("count vocabulary is malformed")
    counts = [_strict_int(value, label="count vocabulary value") for value in values]
    if not counts or min(counts) != 1 or sorted(set(counts)) != list(
        range(1, max(counts) + 1)
    ):
        raise ValueError("count vocabulary must be contiguous from one")
    return max(counts)


def _legal_action_indices(
    *,
    tokens: Sequence[FormulaToken],
    species_vocabulary: Mapping[int, tuple[int, int]],
    target_n: int,
    arity: int,
    family: str,
    max_count: int,
    reachability: PaulingBitsetReachability,
) -> list[list[int]]:
    """Compile the exact inference-time legal support for every teacher step."""

    nodes_by_id = {
        int(species_id): ValenceNode(int(value[0]), int(value[1]))
        for species_id, value in species_vocabulary.items()
    }
    node_to_id = {node: species_id for species_id, node in nodes_by_id.items()}
    state = CCFDv2State.start().apply(SetAtomCount(int(target_n)))
    legal_steps: list[list[int]] = []
    for teacher in tokens:
        legal_tokens = reachability.legal_species_counts(
            state,
            family=str(family),
            target_arity=int(arity),
            max_species=7,
        )
        legal: list[int] = []
        for token in legal_tokens:
            node = ValenceNode(int(token.atomic_number), int(token.oxidation_state))
            if node not in node_to_id:
                raise ValueError("reachability returned a species outside vocabulary")
            count = int(token.count)
            if count < 1 or count > int(max_count):
                raise ValueError("reachability returned a count outside vocabulary")
            legal.append(int(node_to_id[node]) * int(max_count) + count - 1)
        legal = sorted(set(legal))
        teacher_node = ValenceNode(
            int(teacher.atomic_number), int(teacher.oxidation_state)
        )
        teacher_action = (
            int(node_to_id[teacher_node]) * int(max_count)
            + int(teacher.count)
            - 1
        )
        if teacher_action not in legal:
            raise ValueError("teacher action is illegal under Pauling-bitset mask")
        legal_steps.append(legal)
        state = state.apply(teacher, max_species=7)
    if not state.conservation_complete or len(state.tokens) != int(arity):
        raise ValueError("teacher terminal state is not conservation complete")
    eos_action = len(species_vocabulary) * int(max_count)
    legal_steps.append([eos_action])
    return legal_steps


def _soft_target(
    row: Mapping[str, Any],
    vocabulary: Mapping[str, Any],
    plan_state: Mapping[str, Any],
    field: str,
) -> dict[str, Any]:
    labels = row.get("soft_labels")
    if not isinstance(labels, Mapping) or field not in labels:
        raise ValueError(f"semantic row lacks soft label {field}")
    label = _strict_int(labels[field], label=f"{field} label")
    soft_vocabulary = vocabulary.get("soft_vocabulary")
    if not isinstance(soft_vocabulary, Mapping):
        raise ValueError("vocabulary lacks soft_vocabulary")
    values = soft_vocabulary.get(field)
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError(f"vocabulary lacks values for {field}")
    if label < 0 or label >= len(values):
        raise ValueError(f"{field} label is outside its vocabulary")
    value = _safe_text(plan_state.get(field), label=field)
    if str(values[label]) != value:
        raise ValueError(f"{field} label/value mismatch")
    return {"label": label, "value": value}


def _expected_ledger(
    tokens: Sequence[FormulaToken], *, target_n: int, arity: int
) -> list[dict[str, Any]]:
    remaining_atoms = int(target_n)
    remaining_species = int(arity)
    net_charge = 0
    branch = "unset"
    initial = {
        "remaining_atoms": remaining_atoms,
        "net_charge": net_charge,
        "remaining_species": remaining_species,
        "branch": branch,
    }
    # The frozen semantic format records state after family and after N/arity.
    steps = [dict(initial), dict(initial)]
    for token in tokens:
        token_branch = "alloy" if int(token.oxidation_state) == 0 else "ionic"
        if branch == "unset":
            branch = token_branch
        elif branch != token_branch:
            raise ValueError("teacher actions mix ionic and alloy branches")
        remaining_atoms -= int(token.count)
        remaining_species -= 1
        net_charge += int(token.oxidation_state) * int(token.count)
        steps.append(
            {
                "remaining_atoms": remaining_atoms,
                "net_charge": net_charge,
                "remaining_species": remaining_species,
                "branch": branch,
            }
        )
    return steps


def _normalize_ledger(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("semantic row lacks ledger_steps")
    normalized: list[dict[str, Any]] = []
    for ordinal, raw in enumerate(value):
        if not isinstance(raw, Mapping):
            raise ValueError(f"ledger step {ordinal} is not an object")
        normalized.append(
            {
                "remaining_atoms": _strict_int(
                    raw.get("remaining_atoms"), label="remaining_atoms"
                ),
                "net_charge": _strict_int(raw.get("net_charge"), label="net_charge"),
                "remaining_species": _strict_int(
                    raw.get("remaining_species"), label="remaining_species"
                ),
                "branch": str(raw.get("branch") or ""),
            }
        )
    return normalized


def audit_transcript_from_targets(targets: Mapping[str, Any]) -> str:
    """Render a deterministic audit-only transcript from validated targets."""

    proposal = targets["proposal_target"]
    lines = [
        "PROPOSAL "
        f"family={proposal['family_value']} family_id={proposal['family_id']} "
        f"N={proposal['N']} arity={proposal['arity']}"
    ]
    for species_id, count, species in zip(
        targets["species_ids"], targets["count_targets"], targets["species_actions"]
    ):
        lines.append(
            "SPECIES "
            f"id={species_id} Z={species['atomic_number']} "
            f"oxidation={species['oxidation_state']:+d} count={count}"
        )
    lines.append("EOS_COMPOSITION")
    for field in SOFT_FIELDS:
        soft = targets["soft_targets"][field]
        lines.append(f"SOFT field={field} label={soft['label']} value={soft['value']}")
    lines.append("END_TYPED_PLAN")
    return "\n".join(lines) + "\n"


def typed_targets_from_semantic_row(
    row: Mapping[str, Any],
    vocabulary: Mapping[str, Any],
    *,
    context: TypedTargetContext | None = None,
    compile_legal_masks: bool = False,
) -> dict[str, Any]:
    """Validate and extract one teacher-forced typed C3FD target sequence."""

    if row.get("composition_supervision") is not True:
        raise ValueError("composition supervision is not valid")
    if row.get("proposal_supervision") is not True:
        raise ValueError("proposal supervision is not valid")
    if str(row.get("certificate_class") or "") != "benchmark_compatible":
        raise ValueError("composition certificate is not benchmark compatible")
    if row.get("compile_error") not in (None, ""):
        raise ValueError("semantic row carries a compile error")

    plan_state = row.get("plan_state")
    proposal = row.get("proposal_targets")
    if not isinstance(plan_state, Mapping) or not isinstance(proposal, Mapping):
        raise ValueError("semantic row lacks Plan/proposal targets")
    family_value = _safe_text(
        plan_state.get("anion_framework"), label="proposal family"
    )
    family_id = _strict_int(proposal.get("family"), label="proposal family id")
    soft_labels = row.get("soft_labels")
    if not isinstance(soft_labels, Mapping):
        raise ValueError("semantic row lacks soft_labels")
    if _strict_int(
        soft_labels.get("anion_framework"), label="anion framework label"
    ) != family_id:
        raise ValueError("proposal family target disagrees with soft label")
    family_values = (vocabulary.get("soft_vocabulary") or {}).get(
        "anion_framework"
    )
    if not isinstance(family_values, Sequence) or isinstance(
        family_values, (str, bytes)
    ):
        raise ValueError("vocabulary lacks anion framework values")
    if family_id < 0 or family_id >= len(family_values):
        raise ValueError("proposal family id is outside its vocabulary")
    if str(family_values[family_id]) != family_value:
        raise ValueError("proposal family id/value mismatch")

    target_n = _strict_int(proposal.get("N"), label="proposal N")
    arity = _strict_int(proposal.get("arity"), label="proposal arity")
    if target_n < 1 or target_n > 20 or arity < 1 or arity > 7:
        raise ValueError("proposal N/arity is outside the C3FD contract")
    if _strict_int(row.get("N_target"), label="N_target") != target_n:
        raise ValueError("proposal N disagrees with N_target")

    raw_species = row.get("species_labels")
    raw_counts = row.get("count_targets")
    if not isinstance(raw_species, Sequence) or isinstance(raw_species, (str, bytes)):
        raise ValueError("semantic row lacks species_labels")
    if not isinstance(raw_counts, Sequence) or isinstance(raw_counts, (str, bytes)):
        raise ValueError("semantic row lacks count_targets")
    species_ids = [
        _strict_int(value, label="species id") for value in raw_species
    ]
    counts = [_strict_int(value, label="species count") for value in raw_counts]
    if len(species_ids) != arity or len(counts) != arity:
        raise ValueError("species/count sequence does not match proposal arity")

    species_vocabulary = _species_vocabulary(vocabulary)
    tokens: list[FormulaToken] = []
    species_actions: list[dict[str, int]] = []
    for species_id, count in zip(species_ids, counts):
        if species_id not in species_vocabulary:
            raise ValueError(f"unknown species id {species_id}")
        atomic_number, oxidation_state = species_vocabulary[species_id]
        token = FormulaToken(atomic_number, oxidation_state, count)
        tokens.append(token)
        species_actions.append(
            {
                "atomic_number": atomic_number,
                "oxidation_state": oxidation_state,
            }
        )
    if [token.species_key for token in tokens] != sorted(
        token.species_key for token in tokens
    ) or len({token.species_key for token in tokens}) != len(tokens):
        raise ValueError("species actions are not in strict canonical order")
    state = replay_actions((SetAtomCount(target_n), *tokens, END_COMPOSITION))
    if not state.ended or not state.conservation_complete:
        raise ValueError("teacher action sequence does not terminate exactly")

    ledger = _normalize_ledger(row.get("ledger_steps"))
    expected_ledger = _expected_ledger(tokens, target_n=target_n, arity=arity)
    if ledger != expected_ledger:
        raise ValueError("ledger_steps do not match teacher actions")
    soft_targets = {
        field: _soft_target(row, vocabulary, plan_state, field)
        for field in SOFT_FIELDS
    }
    targets: dict[str, Any] = {
        "schema": FUSED_TYPED_PLAN_SCHEMA,
        "proposal_target": {
            "family_id": family_id,
            "family_value": family_value,
            "N": target_n,
            "arity": arity,
        },
        "species_ids": species_ids,
        "count_targets": counts,
        "species_actions": species_actions,
        "ledger_steps": ledger,
        "soft_targets": soft_targets,
    }
    if compile_legal_masks:
        compiled = context or build_typed_target_context(vocabulary)
        targets["legal_action_indices"] = _legal_action_indices(
            tokens=tokens,
            species_vocabulary=compiled.species_vocabulary,
            target_n=target_n,
            arity=arity,
            family=family_value,
            max_count=int(compiled.max_count),
            reachability=compiled.reachability,
        )
        targets["max_count"] = int(compiled.max_count)
    targets["audit_transcript"] = audit_transcript_from_targets(targets)
    return targets


__all__ = [
    "FUSED_TYPED_PLAN_SCHEMA",
    "SOFT_FIELDS",
    "STABILITY_CONDITIONS",
    "STABILITY_HIGHER",
    "STABILITY_META_OR_BETTER",
    "TypedTargetContext",
    "audit_transcript_from_targets",
    "build_typed_target_context",
    "stability_condition_from_e_above_hull",
    "typed_targets_from_semantic_row",
]
