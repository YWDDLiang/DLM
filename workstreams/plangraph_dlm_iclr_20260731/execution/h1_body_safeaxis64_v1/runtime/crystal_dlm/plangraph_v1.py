"""Versioned, leakage-safe PlanGraph contract for dynamic-v1 crystals.

PlanGraph v1 is intentionally conservative: it is derived only from the
existing composition-first plan state and the dynamic-v1 site ordering.  Site
groups are element-multiplicity groups because the current representation does
not contain trustworthy Wyckoff-equivalence labels.  Energy, stability, S.U.N.,
CHGNet, and Materials Project metadata are never serialized into the graph.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Dict, Mapping, Sequence

from crystal_dlm.r5_dynamic_length import validate_answer_matches_plan
from crystal_dlm.r5_plan_state import (
    ALLOWED_LATTICE_SYSTEMS,
    ALLOWED_SPACEGROUP_BUCKETS,
    PLAN_STATE_VERSION,
    validate_plan_state,
)


PLANGRAPH_VERSION = "plangraph_v1"
PLANGRAPH_SITE_GROUP_STRATEGY = "element_multiplicity_v1"

PLANGRAPH_PLANNER_PROMPT = (
    "Generate one chemically structured PlanGraph v1 JSON object for an MP-20 "
    "bulk material. Use only schema_version, source_plan_state_version, "
    "site_group_strategy, composition, symmetry, lattice, site_groups, "
    "constraints, dependency_order. Hard rules: N is 1..20; counts sum to N; "
    "site_groups cover every slot exactly once; composition is locked; no "
    "additional fields are allowed. Return only JSON:"
)

PLANGRAPH_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "source_plan_state_version",
        "site_group_strategy",
        "composition",
        "symmetry",
        "lattice",
        "site_groups",
        "constraints",
        "dependency_order",
    }
)

FORBIDDEN_KEY_TOKENS = frozenset(
    {
        "eabovehull",
        "energyabovehull",
        "formationenergy",
        "formationenergyperatom",
        "energy",
        "stability",
        "stable",
        "metastable",
        "sun",
        "strictsun",
        "metasun",
        "chgnet",
        "mlip",
        "mpapi",
        "materialsproject",
    }
)

FORBIDDEN_KEY_FRAGMENTS = (
    "abovehull",
    "formationenergy",
    "chgnet",
    "materialsproject",
)


class PlanGraphError(ValueError):
    """Raised when a PlanGraph cannot be constructed or validated."""


@dataclass(frozen=True)
class PlanGraphValidation:
    errors: tuple[str, ...]
    forbidden_key_paths: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        return not self.errors and not self.forbidden_key_paths

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "errors": list(self.errors),
            "forbidden_key_paths": list(self.forbidden_key_paths),
        }


def _normalized_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def forbidden_key_paths(payload: Any, *, prefix: str = "$") -> list[str]:
    """Find forbidden metadata keys recursively without inspecting values."""

    matches: list[str] = []
    if isinstance(payload, Mapping):
        for raw_key, value in payload.items():
            key = str(raw_key)
            normalized = _normalized_key(key)
            path = f"{prefix}.{key}"
            if normalized in FORBIDDEN_KEY_TOKENS or any(
                fragment in normalized for fragment in FORBIDDEN_KEY_FRAGMENTS
            ):
                matches.append(path)
            matches.extend(forbidden_key_paths(value, prefix=path))
    elif isinstance(payload, Sequence) and not isinstance(
        payload, (str, bytes, bytearray)
    ):
        for index, value in enumerate(payload):
            matches.extend(forbidden_key_paths(value, prefix=f"{prefix}[{index}]"))
    return matches


def _normalize_oxidation_candidates(raw: Any, *, arity: int) -> list[list[int]]:
    if raw in (None, "", "unknown"):
        return []
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        return []
    values = list(raw)
    if values and all(isinstance(value, (int, float)) for value in values):
        candidate = [int(value) for value in values]
        return [candidate] if len(candidate) == arity else []
    candidates: list[list[int]] = []
    for item in values:
        if not isinstance(item, Sequence) or isinstance(item, (str, bytes, bytearray)):
            continue
        candidate = [int(value) for value in item if isinstance(value, (int, float))]
        if len(candidate) == arity:
            candidates.append(candidate)
    return candidates


def _canonical_site_species(
    elements: Sequence[str],
    counts: Sequence[int],
) -> list[str]:
    species: list[str] = []
    for element, count in zip(elements, counts):
        species.extend([str(element)] * int(count))
    return species


def _require_exact_keys(
    payload: Any,
    *,
    required: set[str],
    path: str,
    errors: list[str],
) -> Mapping[str, Any] | None:
    if not isinstance(payload, Mapping):
        errors.append(f"{path} must be an object")
        return None
    keys = {str(key) for key in payload}
    missing = sorted(required - keys)
    extra = sorted(keys - required)
    if missing:
        errors.append(f"{path} missing keys: {missing}")
    if extra:
        errors.append(f"{path} has unsupported keys: {extra}")
    return payload


def validate_plangraph(graph: Mapping[str, Any]) -> PlanGraphValidation:
    errors: list[str] = []
    forbidden = tuple(sorted(set(forbidden_key_paths(graph))))
    top = _require_exact_keys(
        graph,
        required=set(PLANGRAPH_TOP_LEVEL_FIELDS),
        path="$",
        errors=errors,
    )
    if top is None:
        return PlanGraphValidation(tuple(errors), forbidden)

    if graph.get("schema_version") != PLANGRAPH_VERSION:
        errors.append(f"$.schema_version must be {PLANGRAPH_VERSION!r}")
    if graph.get("site_group_strategy") != PLANGRAPH_SITE_GROUP_STRATEGY:
        errors.append(
            f"$.site_group_strategy must be {PLANGRAPH_SITE_GROUP_STRATEGY!r}"
        )
    if not isinstance(graph.get("source_plan_state_version"), str):
        errors.append("$.source_plan_state_version must be a string")

    composition = _require_exact_keys(
        graph.get("composition"),
        required={
            "N",
            "elements",
            "counts",
            "formula",
            "reduced_formula",
            "charge_bucket",
            "oxidation_candidates",
            "anion_framework",
        },
        path="$.composition",
        errors=errors,
    )
    num_atoms = -1
    elements: list[str] = []
    counts: list[int] = []
    if composition is not None:
        plan_validation = validate_plan_state(
            {
                "N": composition.get("N"),
                "elements": composition.get("elements"),
                "counts": composition.get("counts"),
                "formula": composition.get("formula"),
            }
        )
        if not plan_validation.valid:
            errors.append(
                "$.composition violates plan-state count/element/formula rules"
            )
        try:
            num_atoms = int(composition.get("N"))
        except Exception:
            errors.append("$.composition.N must be an integer")
        raw_elements = composition.get("elements")
        raw_counts = composition.get("counts")
        if isinstance(raw_elements, list):
            elements = [str(value) for value in raw_elements]
        if isinstance(raw_counts, list):
            try:
                counts = [int(value) for value in raw_counts]
            except Exception:
                errors.append("$.composition.counts must contain integers")
        if not isinstance(composition.get("reduced_formula"), str):
            errors.append("$.composition.reduced_formula must be a string")
        if not isinstance(composition.get("charge_bucket"), str):
            errors.append("$.composition.charge_bucket must be a string")
        if not isinstance(composition.get("anion_framework"), str):
            errors.append("$.composition.anion_framework must be a string")
        oxidation = composition.get("oxidation_candidates")
        if not isinstance(oxidation, list) or any(
            not isinstance(candidate, list)
            or len(candidate) != len(elements)
            or any(not isinstance(value, int) for value in candidate)
            for candidate in oxidation
            if isinstance(oxidation, list)
        ):
            errors.append(
                "$.composition.oxidation_candidates must be integer lists matching arity"
            )

    symmetry = _require_exact_keys(
        graph.get("symmetry"),
        required={"lattice_system", "spacegroup_bucket"},
        path="$.symmetry",
        errors=errors,
    )
    if symmetry is not None:
        allowed_lattice = set(ALLOWED_LATTICE_SYSTEMS) | {"unknown"}
        allowed_spacegroup = set(ALLOWED_SPACEGROUP_BUCKETS) | {"sg_unknown"}
        if symmetry.get("lattice_system") not in allowed_lattice:
            errors.append("$.symmetry.lattice_system is unsupported")
        if symmetry.get("spacegroup_bucket") not in allowed_spacegroup:
            errors.append("$.symmetry.spacegroup_bucket is unsupported")

    lattice = _require_exact_keys(
        graph.get("lattice"),
        required={"volume_per_atom_bin"},
        path="$.lattice",
        errors=errors,
    )
    if lattice is not None:
        volume_bin = lattice.get("volume_per_atom_bin")
        if (
            not isinstance(volume_bin, str)
            or re.fullmatch(
                r"volpa_(?:unknown|\d{3}_\d{3})",
                volume_bin,
            )
            is None
        ):
            errors.append("$.lattice.volume_per_atom_bin is malformed")

    constraints = _require_exact_keys(
        graph.get("constraints"),
        required={
            "atom_count",
            "element_counts",
            "charge_bucket",
            "composition_locked",
        },
        path="$.constraints",
        errors=errors,
    )
    if constraints is not None:
        if constraints.get("atom_count") != num_atoms:
            errors.append("$.constraints.atom_count must equal composition.N")
        expected_counts = dict(zip(elements, counts))
        if constraints.get("element_counts") != expected_counts:
            errors.append(
                "$.constraints.element_counts must equal composition elements/counts"
            )
        if composition is not None and constraints.get(
            "charge_bucket"
        ) != composition.get("charge_bucket"):
            errors.append(
                "$.constraints.charge_bucket must equal composition.charge_bucket"
            )
        if constraints.get("composition_locked") is not True:
            errors.append("$.constraints.composition_locked must be true")

    site_groups = graph.get("site_groups")
    group_ids: list[str] = []
    covered_slots: list[int] = []
    grouped_counts: Counter[str] = Counter()
    if not isinstance(site_groups, list) or not site_groups:
        errors.append("$.site_groups must be a non-empty list")
    else:
        for index, raw_group in enumerate(site_groups):
            path = f"$.site_groups[{index}]"
            group = _require_exact_keys(
                raw_group,
                required={
                    "group_id",
                    "element",
                    "multiplicity",
                    "slot_indices",
                    "depends_on",
                },
                path=path,
                errors=errors,
            )
            if group is None:
                continue
            group_id = group.get("group_id")
            element = group.get("element")
            multiplicity = group.get("multiplicity")
            slots = group.get("slot_indices")
            if not isinstance(group_id, str) or not re.fullmatch(
                r"site_group_\d{3}",
                group_id,
            ):
                errors.append(f"{path}.group_id is malformed")
            else:
                group_ids.append(group_id)
            if element not in elements:
                errors.append(f"{path}.element is absent from composition")
            if not isinstance(slots, list) or any(
                not isinstance(slot, int) for slot in slots
            ):
                errors.append(f"{path}.slot_indices must be an integer list")
                slots = []
            if list(slots) != sorted(set(slots)):
                errors.append(f"{path}.slot_indices must be sorted and unique")
            if not isinstance(multiplicity, int) or multiplicity != len(slots):
                errors.append(f"{path}.multiplicity must equal slot count")
            if group.get("depends_on") != ["composition", "symmetry_lattice"]:
                errors.append(
                    f"{path}.depends_on must be ['composition', 'symmetry_lattice']"
                )
            covered_slots.extend(int(slot) for slot in slots)
            if isinstance(element, str):
                grouped_counts[element] += len(slots)

    if len(group_ids) != len(set(group_ids)):
        errors.append("$.site_groups group_id values must be unique")
    if num_atoms >= 0 and sorted(covered_slots) != list(range(num_atoms)):
        errors.append("$.site_groups must cover each atom slot exactly once")
    if grouped_counts != Counter(dict(zip(elements, counts))):
        errors.append("$.site_groups multiplicities must match composition counts")

    expected_order = ["composition", "symmetry_lattice", *group_ids]
    if graph.get("dependency_order") != expected_order:
        errors.append("$.dependency_order must match the registered group order")

    return PlanGraphValidation(tuple(errors), forbidden)


def ensure_valid_plangraph(graph: Mapping[str, Any]) -> None:
    validation = validate_plangraph(graph)
    if not validation.valid:
        details = [*validation.errors]
        if validation.forbidden_key_paths:
            details.append(
                "forbidden keys: " + ", ".join(validation.forbidden_key_paths)
            )
        raise PlanGraphError("; ".join(details))


def plangraph_from_plan_state(
    plan_state: Mapping[str, Any],
    *,
    site_species: Sequence[str] | None = None,
) -> Dict[str, Any]:
    """Convert one valid plan state into a deterministic PlanGraph v1."""

    plan_validation = validate_plan_state(plan_state)
    if not plan_validation.valid:
        raise PlanGraphError(
            "plan_state violates count/element/formula rules: "
            + json.dumps(plan_validation.to_dict(), sort_keys=True)
        )
    num_atoms = int(plan_state["N"])
    elements = [str(value) for value in plan_state["elements"]]
    counts = [int(value) for value in plan_state["counts"]]
    if site_species is None:
        species = _canonical_site_species(elements, counts)
    else:
        species = [str(value) for value in site_species]
    if len(species) != num_atoms:
        raise PlanGraphError(
            f"site_species length {len(species)} does not match N={num_atoms}"
        )
    if Counter(species) != Counter(dict(zip(elements, counts))):
        raise PlanGraphError(
            f"site_species composition {dict(Counter(species))} does not match "
            f"{dict(zip(elements, counts))}"
        )

    site_groups: list[Dict[str, Any]] = []
    for group_index, element in enumerate(elements):
        slots = [
            slot_index
            for slot_index, site_element in enumerate(species)
            if site_element == element
        ]
        site_groups.append(
            {
                "group_id": f"site_group_{group_index:03d}",
                "element": element,
                "multiplicity": len(slots),
                "slot_indices": slots,
                "depends_on": ["composition", "symmetry_lattice"],
            }
        )

    charge_bucket = str(plan_state.get("charge_bucket") or "validator_unavailable")
    graph: Dict[str, Any] = {
        "schema_version": PLANGRAPH_VERSION,
        "source_plan_state_version": str(
            plan_state.get("plan_state_version") or PLAN_STATE_VERSION
        ),
        "site_group_strategy": PLANGRAPH_SITE_GROUP_STRATEGY,
        "composition": {
            "N": num_atoms,
            "elements": elements,
            "counts": counts,
            "formula": str(plan_state["formula"]),
            "reduced_formula": str(
                plan_state.get("reduced_formula") or plan_state["formula"]
            ),
            "charge_bucket": charge_bucket,
            "oxidation_candidates": _normalize_oxidation_candidates(
                plan_state.get("oxidation_candidates"),
                arity=len(elements),
            ),
            "anion_framework": str(plan_state.get("anion_framework") or "other"),
        },
        "symmetry": {
            "lattice_system": str(plan_state.get("lattice_system") or "unknown"),
            "spacegroup_bucket": str(
                plan_state.get("spacegroup_bucket") or "sg_unknown"
            ),
        },
        "lattice": {
            "volume_per_atom_bin": str(
                plan_state.get("volume_per_atom_bin") or "volpa_unknown"
            ),
        },
        "site_groups": site_groups,
        "constraints": {
            "atom_count": num_atoms,
            "element_counts": dict(zip(elements, counts)),
            "charge_bucket": charge_bucket,
            "composition_locked": True,
        },
        "dependency_order": [
            "composition",
            "symmetry_lattice",
            *(group["group_id"] for group in site_groups),
        ],
    }
    ensure_valid_plangraph(graph)
    return graph


def plangraph_from_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    """Convert a dynamic-v1 supervised row without reading its metadata."""

    representation = record.get("representation")
    if representation not in (None, "dynamic_v1"):
        raise PlanGraphError(
            f"record representation {representation!r} is not dynamic_v1"
        )
    plan_state = record.get("plan_state") or record.get("r5_plan_state")
    if not isinstance(plan_state, Mapping):
        raise PlanGraphError("record is missing plan_state/r5_plan_state")
    answer = record.get("answer")
    if not isinstance(answer, str):
        raise PlanGraphError("record is missing dynamic-v1 answer text")
    try:
        arrays = validate_answer_matches_plan(plan_state, answer)
    except Exception as exc:
        raise PlanGraphError(f"answer does not match plan_state: {exc}") from exc
    return plangraph_from_plan_state(
        plan_state,
        site_species=arrays["species"],
    )


def plangraph_to_json(graph: Mapping[str, Any]) -> str:
    ensure_valid_plangraph(graph)
    return json.dumps(
        graph,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def build_plangraph_planner_prompt() -> str:
    return PLANGRAPH_PLANNER_PROMPT


def build_plangraph_body_prompt(graph: Mapping[str, Any]) -> str:
    """Build a body prompt containing only the validated canonical graph."""

    encoded = plangraph_to_json(graph)
    return (
        "Generate only the exact-length dynamic crystal body for this fixed "
        "PlanGraph. The first token must match composition.N, element tokens "
        "must match the locked element counts and slot groups, and every site "
        "must have one XYZ triplet.\n"
        f"plangraph: {encoded}\n"
        "dynamic_crystal_body:"
    )


def build_plangraph_sft_records(
    source_record: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """Create leakage-safe Planner and body SFT records.

    Source metadata, prompts, weights, IDs, and diagnostics are deliberately
    not copied.  The returned identity hashes only the canonical graph and
    dynamic answer, providing reproducibility without carrying evaluation
    properties.
    """

    graph = plangraph_from_record(source_record)
    answer = str(source_record["answer"])
    graph_json = plangraph_to_json(graph)
    pair_sha256 = hashlib.sha256(
        (graph_json + "\n" + answer).encode("utf-8")
    ).hexdigest()
    planner_prompt = build_plangraph_planner_prompt().rstrip() + "\n"
    body_prompt = build_plangraph_body_prompt(graph).rstrip() + "\n"
    num_atoms = int(graph["composition"]["N"])
    return {
        "planner": {
            "task": "plangraph_v1_planner",
            "representation": PLANGRAPH_VERSION,
            "prompt": planner_prompt,
            "answer": graph_json,
            "text": planner_prompt + graph_json,
            "loss_profile": "text",
            "sample_weight": 1.0,
            "training_pair_sha256": pair_sha256,
        },
        "body": {
            "task": "plangraph_v1_dynamic_body",
            "representation": "dynamic_v1",
            "plangraph_version": PLANGRAPH_VERSION,
            "prompt": body_prompt,
            "answer": answer,
            "text": body_prompt + answer,
            "loss_profile": "fixed_slot",
            "sample_weight": 1.0,
            "num_atoms": num_atoms,
            "answer_semantic_length": 7 + 4 * num_atoms,
            "answer_token_count": 7 + 4 * num_atoms,
            "plangraph": graph,
            "training_pair_sha256": pair_sha256,
        },
    }


__all__ = [
    "FORBIDDEN_KEY_FRAGMENTS",
    "FORBIDDEN_KEY_TOKENS",
    "PLANGRAPH_SITE_GROUP_STRATEGY",
    "PLANGRAPH_PLANNER_PROMPT",
    "PLANGRAPH_TOP_LEVEL_FIELDS",
    "PLANGRAPH_VERSION",
    "PlanGraphError",
    "PlanGraphValidation",
    "build_plangraph_body_prompt",
    "build_plangraph_planner_prompt",
    "build_plangraph_sft_records",
    "ensure_valid_plangraph",
    "forbidden_key_paths",
    "plangraph_from_plan_state",
    "plangraph_from_record",
    "plangraph_to_json",
    "validate_plangraph",
]
