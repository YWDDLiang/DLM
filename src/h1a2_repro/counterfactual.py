"""Composition-matched counterfactual rich-Plan helpers."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from hashlib import sha256
import json
import math
from typing import Any, Mapping, Sequence

from h1a2_repro.difficulty import broad_family, n_bin


STRUCTURAL_FIELDS = ("lattice_system", "spacegroup_bucket", "volume_per_atom_bin")
ANCHORED_FIELDS = ("formula", "N", "elements", "counts", "anion_framework", "charge_bucket")


def structural_tuple(plan: Mapping[str, Any]) -> tuple[str, str, str]:
    return tuple(str(plan.get(field, "unknown")) for field in STRUCTURAL_FIELDS)


def plan_stratum(plan: Mapping[str, Any]) -> tuple[str, str, str]:
    elements = tuple(str(value) for value in (plan.get("elements") or ()))
    try:
        num_atoms = int(plan.get("N"))
    except (TypeError, ValueError):
        num_atoms = None
    charge = str(plan.get("charge_bucket", ""))
    family = broad_family(elements, all_metal=charge == "all_metal")
    return family, str(len(set(elements))) if elements else "unknown", n_bin(num_atoms)


def plan_key(plan: Mapping[str, Any]) -> str:
    payload = {field: plan.get(field) for field in (*ANCHORED_FIELDS, *STRUCTURAL_FIELDS)}
    return sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def choose_donors(plans: Sequence[Mapping[str, Any]], *, seed: int = 17) -> list[int | None]:
    groups: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for index, plan in enumerate(plans):
        groups[plan_stratum(plan)].append(index)
    result: list[int | None] = []
    for index, plan in enumerate(plans):
        candidates = [
            donor
            for donor in groups[plan_stratum(plan)]
            if donor != index and structural_tuple(plans[donor]) != structural_tuple(plan)
        ]
        if not candidates:
            result.append(None)
            continue
        digest = sha256(f"{seed}:{plan_key(plan)}".encode("utf-8")).hexdigest()
        result.append(candidates[int(digest[:16], 16) % len(candidates)])
    return result


def build_counterfactual_plan(plan: Mapping[str, Any], donor: Mapping[str, Any]) -> dict[str, Any]:
    if plan_stratum(plan) != plan_stratum(donor):
        raise ValueError("counterfactual donor is not matched on family/arity/N-bin")
    if structural_tuple(plan) == structural_tuple(donor):
        raise ValueError("counterfactual donor must change the structural tuple")
    counterfactual = deepcopy(dict(plan))
    for field in STRUCTURAL_FIELDS:
        counterfactual[field] = donor.get(field)
    for field in ANCHORED_FIELDS:
        if counterfactual.get(field) != plan.get(field):
            raise RuntimeError(f"counterfactual changed anchored field {field}")
    return counterfactual


def geometry_relative_positions(num_atoms: int) -> tuple[int, ...]:
    num_atoms = int(num_atoms)
    if num_atoms <= 0:
        raise ValueError("num_atoms must be positive")
    positions = list(range(1, 7))
    for index in range(num_atoms):
        start = 7 + 4 * index
        positions.extend((start + 1, start + 2, start + 3))
    expected = 6 + 3 * num_atoms
    if len(positions) != expected:
        raise RuntimeError("geometry position count changed")
    return tuple(positions)


def pairwise_logistic_from_margins(margins, *, temperature: float = 1.0):
    """Return softplus(-margin/T); imports torch only when training calls it."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    import torch.nn.functional as F

    return F.softplus(-margins / float(temperature)).mean()


def calibrated_grounding_weight(
    *,
    ce_gradient_norm: float,
    grounding_gradient_norm: float,
    target_ratio: float = 0.1,
) -> float:
    if ce_gradient_norm <= 0 or grounding_gradient_norm <= 0 or target_ratio <= 0:
        raise ValueError("gradient norms and target_ratio must be positive")
    value = target_ratio * ce_gradient_norm / grounding_gradient_norm
    if not math.isfinite(value):
        raise ValueError("calibrated grounding weight is not finite")
    return value


__all__ = [
    "ANCHORED_FIELDS",
    "STRUCTURAL_FIELDS",
    "build_counterfactual_plan",
    "calibrated_grounding_weight",
    "choose_donors",
    "geometry_relative_positions",
    "pairwise_logistic_from_margins",
    "plan_key",
    "plan_stratum",
    "structural_tuple",
]

