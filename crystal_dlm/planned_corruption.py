"""Pure position-group policies for dependency-aligned DLM corruption.

This module does not depend on PyTorch and does not modify the training loop.
It defines and validates answer-relative groups, then samples the exact masks
needed by the D1 current-order control, D2 PlanGraph policy, and legal
safe-axis PlanGraph policy:

* prerequisite groups remain visible;
* a stochastic subset of the active group is masked and supervised;
* all future groups are masked but receive no loss.

The separation between ``masked_input_positions`` and ``loss_positions`` is
the key invariant for planned denoising.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import random
from typing import Any, Dict, Mapping, Sequence

from crystal_dlm.dynamic_crystal import dynamic_answer_token_count
from crystal_dlm.plangraph_v1 import (
    ensure_valid_plangraph,
    plangraph_from_plan_state,
    plangraph_to_json,
)


class CorruptionScheduleError(ValueError):
    """Raised when position groups or corruption parameters are invalid."""


STATELESS_MODULUS = 2_147_483_647
STATELESS_SEED_MULTIPLIER = 104_729
STATELESS_STEP_MULTIPLIER = 1_000_003
STATELESS_STREAM_MULTIPLIER = 9_176
STATELESS_POSITION_MULTIPLIER = 611_953
STATELESS_MIX_MULTIPLIER = 1_103_515_245
STATELESS_MIX_INCREMENT = 12_345
STATELESS_FINAL_MULTIPLIER = 48_271


@dataclass(frozen=True)
class PositionGroup:
    name: str
    positions: tuple[int, ...]
    prerequisites: tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "positions": list(self.positions),
            "prerequisites": list(self.prerequisites),
        }


@dataclass(frozen=True)
class CorruptionMask:
    policy: str
    answer_length: int
    active_group_index: int | None
    active_group: str | None
    p_mask: float
    masked_input_positions: tuple[int, ...]
    loss_positions: tuple[int, ...]
    visible_positions: tuple[int, ...]
    future_positions: tuple[int, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy": self.policy,
            "answer_length": self.answer_length,
            "active_group_index": self.active_group_index,
            "active_group": self.active_group,
            "p_mask": self.p_mask,
            "masked_input_positions": list(self.masked_input_positions),
            "loss_positions": list(self.loss_positions),
            "visible_positions": list(self.visible_positions),
            "future_positions": list(self.future_positions),
        }


def _sequential_groups(
    named_positions: Sequence[tuple[str, Sequence[int]]],
) -> tuple[PositionGroup, ...]:
    groups: list[PositionGroup] = []
    prior_names: list[str] = []
    for name, positions in named_positions:
        groups.append(
            PositionGroup(
                name=str(name),
                positions=tuple(int(position) for position in positions),
                prerequisites=tuple(prior_names),
            )
        )
        prior_names.append(str(name))
    return tuple(groups)


def validate_position_groups(
    groups: Sequence[PositionGroup],
    *,
    answer_length: int,
) -> None:
    if int(answer_length) <= 0:
        raise CorruptionScheduleError("answer_length must be positive")
    if not groups:
        raise CorruptionScheduleError("at least one position group is required")
    names = [group.name for group in groups]
    if len(names) != len(set(names)):
        raise CorruptionScheduleError("position group names must be unique")
    all_positions: list[int] = []
    previous_names: list[str] = []
    for index, group in enumerate(groups):
        if not group.name:
            raise CorruptionScheduleError(f"group {index} has an empty name")
        if not group.positions:
            raise CorruptionScheduleError(f"group {group.name!r} is empty")
        if tuple(sorted(set(group.positions))) != group.positions:
            raise CorruptionScheduleError(
                f"group {group.name!r} positions must be sorted and unique"
            )
        if any(
            position < 0 or position >= int(answer_length)
            for position in group.positions
        ):
            raise CorruptionScheduleError(
                f"group {group.name!r} has a position outside 0..{answer_length - 1}"
            )
        if group.prerequisites != tuple(previous_names):
            raise CorruptionScheduleError(
                f"group {group.name!r} prerequisites must equal all earlier groups"
            )
        all_positions.extend(group.positions)
        previous_names.append(group.name)
    if sorted(all_positions) != list(range(int(answer_length))):
        raise CorruptionScheduleError(
            "position groups must cover each answer position exactly once"
        )


def position_group_ids(
    groups: Sequence[PositionGroup],
    *,
    answer_length: int,
) -> tuple[int, ...]:
    """Encode validated groups as one group index per answer position."""

    validate_position_groups(groups, answer_length=int(answer_length))
    encoded = [-1] * int(answer_length)
    for group_index, group in enumerate(groups):
        for position in group.positions:
            encoded[position] = group_index
    if any(group_index < 0 for group_index in encoded):
        raise CorruptionScheduleError(
            "position group encoding left an answer position unassigned"
        )
    return tuple(encoded)


def corruption_key_for_record(record: Mapping[str, Any]) -> int:
    """Return a stable signed-int64 key without relying on row order or IDs.

    PlanGraph records carry ``training_pair_sha256``.  Older records fall back
    to the prompt/answer pair only, deliberately excluding metadata, sample
    IDs, evaluation labels, and row position.
    """

    registered_sha = record.get("training_pair_sha256")
    if (
        isinstance(registered_sha, str)
        and len(registered_sha) == 64
        and all(character in "0123456789abcdefABCDEF" for character in registered_sha)
    ):
        digest = bytes.fromhex(registered_sha)
    else:
        identity = {
            "prompt": str(record.get("prompt", "")),
            "answer": str(record.get("answer", "")),
        }
        digest = hashlib.sha256(
            json.dumps(
                identity,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) & ((1 << 63) - 1)


def plan_condition_sha256(
    *,
    prompt: str,
    graph: Mapping[str, Any],
) -> str:
    """Hash only information available before body denoising starts.

    This identity deliberately excludes the target answer.  It is therefore
    safe to use for a deterministic training/inference schedule control.
    """

    ensure_valid_plangraph(graph)
    normalized_prompt = str(prompt).rstrip() + "\n"
    payload = normalized_prompt + plangraph_to_json(graph)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def stateless_uniform(
    key: int,
    *,
    step: int,
    seed: int,
    stream: int,
    position: int = 0,
) -> float:
    """Map a counter tuple to a deterministic uniform variate in ``(0, 1)``.

    The arithmetic stays below signed-int64 overflow so the same formula can
    be implemented with PyTorch integer tensors on CPU or CUDA.  It is a
    counter-based training RNG, not a cryptographic primitive.
    """

    if int(step) < 0:
        raise CorruptionScheduleError("stateless step must be non-negative")
    if int(stream) < 0:
        raise CorruptionScheduleError("stateless stream must be non-negative")
    if int(position) < 0:
        raise CorruptionScheduleError("stateless position must be non-negative")
    modulus = STATELESS_MODULUS
    value = (
        int(key) % modulus
        + (int(seed) % modulus) * STATELESS_SEED_MULTIPLIER
        + ((int(step) + 1) % modulus) * STATELESS_STEP_MULTIPLIER
        + ((int(stream) + 1) % modulus) * STATELESS_STREAM_MULTIPLIER
        + ((int(position) + 1) % modulus) * STATELESS_POSITION_MULTIPLIER
    ) % modulus
    value = (
        ((value ^ (value >> 16)) * STATELESS_MIX_MULTIPLIER) + STATELESS_MIX_INCREMENT
    ) % modulus
    value = ((value ^ (value >> 13)) * STATELESS_FINAL_MULTIPLIER + 1) % modulus
    return (float(value) + 0.5) / float(modulus)


def current_order_groups(num_atoms: int) -> tuple[PositionGroup, ...]:
    """D1 groups matching the current exact-dynamic inference order."""

    num_atoms = int(num_atoms)
    answer_length = dynamic_answer_token_count(num_atoms)
    element_positions = [7 + 4 * slot for slot in range(num_atoms)]
    x_positions = [8 + 4 * slot for slot in range(num_atoms)]
    y_positions = [9 + 4 * slot for slot in range(num_atoms)]
    z_positions = [10 + 4 * slot for slot in range(num_atoms)]
    groups = _sequential_groups(
        [
            ("atom_count", [0]),
            ("elements", element_positions),
            ("lattice", [1, 2, 3, 4, 5, 6]),
            ("x", x_positions),
            ("y", y_positions),
            ("z", z_positions),
        ]
    )
    validate_position_groups(groups, answer_length=answer_length)
    return groups


def plangraph_dependency_groups(
    graph: Mapping[str, Any],
) -> tuple[PositionGroup, ...]:
    """D2 groups derived from a valid PlanGraph v1.

    Count and element positions form the locked composition group.  Lattice
    positions form the second group.  Each registered element-multiplicity
    site group then owns the XYZ coordinates of its exact dynamic-v1 slots.
    """

    ensure_valid_plangraph(graph)
    num_atoms = int(graph["composition"]["N"])
    answer_length = dynamic_answer_token_count(num_atoms)
    element_positions = [7 + 4 * slot for slot in range(num_atoms)]
    named_positions: list[tuple[str, Sequence[int]]] = [
        ("composition", [0, *element_positions]),
        ("symmetry_lattice", [1, 2, 3, 4, 5, 6]),
    ]
    for site_group in graph["site_groups"]:
        coordinate_positions: list[int] = []
        for slot in site_group["slot_indices"]:
            coordinate_positions.extend(
                [8 + 4 * int(slot), 9 + 4 * int(slot), 10 + 4 * int(slot)]
            )
        named_positions.append(
            (
                str(site_group["group_id"]),
                sorted(coordinate_positions),
            )
        )
    groups = _sequential_groups(named_positions)
    validate_position_groups(groups, answer_length=answer_length)
    return groups


def safe_axis_dependency_groups(
    graph: Mapping[str, Any],
) -> tuple[PositionGroup, ...]:
    """PlanGraph groups matching the legal grouped-X, grouped-Y, then Z path.

    Composition and lattice are resolved first.  Each PlanGraph site group then
    owns one coordinate-axis group, with every X and Y group strictly preceding
    every Z group.  No coordinate group mixes axes.
    """

    ensure_valid_plangraph(graph)
    num_atoms = int(graph["composition"]["N"])
    answer_length = dynamic_answer_token_count(num_atoms)
    element_positions = [7 + 4 * slot for slot in range(num_atoms)]
    named_positions: list[tuple[str, Sequence[int]]] = [
        ("composition", [0, *element_positions]),
        ("symmetry_lattice", [1, 2, 3, 4, 5, 6]),
    ]
    for axis_name, axis_offset in (("x", 1), ("y", 2), ("z", 3)):
        for site_group in graph["site_groups"]:
            positions = sorted(
                7 + 4 * int(slot) + axis_offset
                for slot in site_group["slot_indices"]
            )
            named_positions.append(
                (f"{site_group['group_id']}_{axis_name}", positions)
            )
    groups = _sequential_groups(named_positions)
    validate_position_groups(groups, answer_length=answer_length)
    return groups


def h1a2_generation_schedule(
    plan_state: Mapping[str, Any],
    *,
    policy: str,
) -> list[list[int]]:
    """Build an H1-A2 body schedule from inference-available plan fields.

    ``d1`` is exactly the frozen R5-C ``exact-plan`` order. ``d2`` compiles
    the historical mixed-axis graph. ``d2_safe_axis`` compiles the legal
    grouped-X, grouped-Y, then grouped-Z graph used by B3.
    """

    normalized = str(policy).strip().lower().replace("-", "_")
    if normalized == "d1":
        groups = current_order_groups(int(plan_state["N"]))
    elif normalized == "d2":
        graph = plangraph_from_plan_state(plan_state)
        groups = plangraph_dependency_groups(graph)
    elif normalized == "d2_safe_axis":
        graph = plangraph_from_plan_state(plan_state)
        groups = safe_axis_dependency_groups(graph)
    else:
        raise CorruptionScheduleError(
            "H1-A2 generation policy must be d1, d2, or d2_safe_axis"
        )
    return [list(group.positions) for group in groups]


def _sample_mask_probability(
    rng: random.Random,
    *,
    eps: float,
    p_mask: float | None,
) -> float:
    if not 0.0 < float(eps) <= 1.0:
        raise CorruptionScheduleError("eps must be in (0, 1]")
    if p_mask is None:
        return float(eps) + (1.0 - float(eps)) * rng.random()
    probability = float(p_mask)
    if not 0.0 < probability <= 1.0:
        raise CorruptionScheduleError("p_mask must be in (0, 1]")
    return probability


def sample_iid_corruption(
    answer_length: int,
    *,
    rng: random.Random,
    eps: float = 1e-3,
    p_mask: float | None = None,
) -> CorruptionMask:
    """Sample a pure iid answer mask for simulation and parity tests."""

    answer_length = int(answer_length)
    if answer_length <= 0:
        raise CorruptionScheduleError("answer_length must be positive")
    probability = _sample_mask_probability(rng, eps=eps, p_mask=p_mask)
    masked = [
        position for position in range(answer_length) if rng.random() < probability
    ]
    if not masked:
        masked = [rng.randrange(answer_length)]
    masked_tuple = tuple(sorted(masked))
    masked_set = set(masked_tuple)
    visible = tuple(
        position for position in range(answer_length) if position not in masked_set
    )
    return CorruptionMask(
        policy="iid",
        answer_length=answer_length,
        active_group_index=None,
        active_group=None,
        p_mask=probability,
        masked_input_positions=masked_tuple,
        loss_positions=masked_tuple,
        visible_positions=visible,
    )


def sample_planned_corruption(
    groups: Sequence[PositionGroup],
    *,
    rng: random.Random,
    active_group_index: int | None = None,
    eps: float = 1e-3,
    p_mask: float | None = None,
    policy_name: str = "planned",
) -> CorruptionMask:
    """Sample one dependency-aligned corruption from validated groups."""

    answer_length = sum(len(group.positions) for group in groups)
    validate_position_groups(groups, answer_length=answer_length)
    if active_group_index is None:
        active_index = rng.randrange(len(groups))
    else:
        active_index = int(active_group_index)
        if not 0 <= active_index < len(groups):
            raise CorruptionScheduleError(
                f"active_group_index {active_index} outside 0..{len(groups) - 1}"
            )
    probability = _sample_mask_probability(rng, eps=eps, p_mask=p_mask)
    active_group = groups[active_index]
    active_masked = [
        position for position in active_group.positions if rng.random() < probability
    ]
    if not active_masked:
        active_masked = [rng.choice(active_group.positions)]

    prerequisite_positions = {
        position for group in groups[:active_index] for position in group.positions
    }
    future_positions = tuple(
        sorted(
            position
            for group in groups[active_index + 1 :]
            for position in group.positions
        )
    )
    masked_input = tuple(sorted({*active_masked, *future_positions}))
    loss_positions = tuple(sorted(active_masked))
    masked_set = set(masked_input)
    visible = tuple(
        position for position in range(answer_length) if position not in masked_set
    )
    if prerequisite_positions.intersection(masked_set):
        raise CorruptionScheduleError(
            "planned corruption masked a prerequisite position"
        )
    if not set(future_positions).issubset(masked_set):
        raise CorruptionScheduleError(
            "planned corruption failed to mask all future positions"
        )
    if not set(loss_positions).issubset(set(active_group.positions)):
        raise CorruptionScheduleError(
            "planned corruption loss escaped the active group"
        )
    return CorruptionMask(
        policy=str(policy_name),
        answer_length=answer_length,
        active_group_index=active_index,
        active_group=active_group.name,
        p_mask=probability,
        masked_input_positions=masked_input,
        loss_positions=loss_positions,
        visible_positions=visible,
        future_positions=future_positions,
    )


def sample_mixture_policy(
    *,
    rng: random.Random,
    iid_weight: float = 2.0,
    planned_weight: float = 1.0,
) -> str:
    """Return ``iid`` or ``planned`` according to explicit non-negative weights."""

    iid = float(iid_weight)
    planned = float(planned_weight)
    if iid < 0.0 or planned < 0.0 or iid + planned <= 0.0:
        raise CorruptionScheduleError(
            "iid_weight and planned_weight must be non-negative with positive sum"
        )
    return "planned" if rng.random() < planned / (iid + planned) else "iid"


def simulate_planned_policy(
    groups: Sequence[PositionGroup],
    *,
    trials: int,
    seed: int,
    policy_name: str,
) -> Dict[str, Any]:
    """Return a deterministic CPU-only mask-distribution summary."""

    trial_count = int(trials)
    if trial_count <= 0:
        raise CorruptionScheduleError("trials must be positive")
    answer_length = sum(len(group.positions) for group in groups)
    validate_position_groups(groups, answer_length=answer_length)
    rng = random.Random(int(seed))
    active_counts = {group.name: 0 for group in groups}
    input_mask_counts = [0] * answer_length
    loss_counts = [0] * answer_length
    p_mask_sum = 0.0
    for _ in range(trial_count):
        sample = sample_planned_corruption(
            groups,
            rng=rng,
            policy_name=policy_name,
        )
        active_counts[str(sample.active_group)] += 1
        p_mask_sum += sample.p_mask
        for position in sample.masked_input_positions:
            input_mask_counts[position] += 1
        for position in sample.loss_positions:
            loss_counts[position] += 1
    return {
        "policy": str(policy_name),
        "seed": int(seed),
        "trials": trial_count,
        "answer_length": answer_length,
        "groups": [group.to_dict() for group in groups],
        "active_group_counts": active_counts,
        "mean_p_mask": p_mask_sum / trial_count,
        "input_mask_frequency": [count / trial_count for count in input_mask_counts],
        "loss_frequency": [count / trial_count for count in loss_counts],
    }


__all__ = [
    "CorruptionMask",
    "CorruptionScheduleError",
    "PositionGroup",
    "STATELESS_FINAL_MULTIPLIER",
    "STATELESS_MIX_INCREMENT",
    "STATELESS_MIX_MULTIPLIER",
    "STATELESS_MODULUS",
    "STATELESS_POSITION_MULTIPLIER",
    "STATELESS_SEED_MULTIPLIER",
    "STATELESS_STEP_MULTIPLIER",
    "STATELESS_STREAM_MULTIPLIER",
    "corruption_key_for_record",
    "current_order_groups",
    "h1a2_generation_schedule",
    "plan_condition_sha256",
    "plangraph_dependency_groups",
    "position_group_ids",
    "sample_iid_corruption",
    "sample_mixture_policy",
    "sample_planned_corruption",
    "safe_axis_dependency_groups",
    "simulate_planned_policy",
    "stateless_uniform",
    "validate_position_groups",
]
