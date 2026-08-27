"""Deterministic element--valence--count assignments for Planner supervision.

The search mirrors the composition factor used by CrysVCD without copying its
electronic-configuration vectors or model code.  It first tries one oxidation
state per element, then permits adjacent, same-sign mixed valences within an
element.  All-metal compositions use explicit zero oxidation states.

Reference (MIT): https://github.com/vipandyc/CrysVCD/tree/main/formula_gen
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from functools import lru_cache
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from typing import Any, Mapping, Sequence

from crystal_dlm.fixed_slot import SYMBOL_TO_Z


# Oxidation-state support represented by CrysVCD's element/valence species.
# Zero is kept separately as an alloy/elemental state and is not considered in
# the ionic charge-balance search.
CRYSVCD_ELEMENT_STATES: dict[str, tuple[int, ...]] = {
    "Ag": (0, 1, 2, 3),
    "Al": (0, 3),
    "As": (-3, 0, 3, 5),
    "Au": (0, 1, 2, 3, 5),
    "B": (0, 3),
    "Ba": (0, 2),
    "Be": (0, 2),
    "Bi": (-3, 0, 3, 5),
    "Br": (-1, 1, 5),
    "C": (-4, -3, -2, -1, 0, 1, 2, 3, 4),
    "Ca": (0, 2),
    "Cd": (0, 2),
    "Cl": (-1, 1, 5, 7),
    "Co": (0, 2, 3, 4),
    "Cr": (0, 2, 3, 4, 6),
    "Cs": (0, 1),
    "Cu": (0, 1, 2, 3, 4),
    "F": (-1,),
    "Fe": (0, 2, 3, 4),
    "Ga": (0, 3),
    "Ge": (-4, 0, 2, 4),
    "H": (-1, 0, 1),
    "Hf": (0, 2, 3, 4),
    "Hg": (0, 1, 2),
    "I": (-1, 1, 5, 7),
    "In": (0, 1, 3),
    "Ir": (0, 3, 4),
    "K": (0, 1),
    "Li": (0, 1),
    "Mg": (0, 2),
    "Mn": (0, 2, 3, 4, 5, 6, 7),
    "Mo": (0, 2, 3, 4, 5, 6),
    "N": (-3, 3, 5),
    "Na": (0, 1),
    "Nb": (0, 3, 4, 5),
    "Ni": (0, 2, 3, 4),
    "O": (-2,),
    "Os": (0, 4, 5, 6, 8),
    "P": (-3, 0, 1, 3, 5),
    "Pb": (0, 2, 4),
    "Pd": (0, 2, 4),
    "Pt": (0, 2, 4),
    "Rb": (0, 1),
    "Re": (0, 4, 6, 7),
    "Rh": (0, 3, 4, 5),
    "Ru": (0, 2, 3, 4, 5),
    "S": (-2, 0, 2, 4, 6),
    "Sb": (-3, 0, 3, 5),
    "Sc": (0, 3),
    "Se": (-2, 0, 2, 4, 6),
    "Si": (-4, 0, 4),
    "Sn": (0, 2, 4),
    "Sr": (0, 2),
    "Ta": (0, 3, 4, 5),
    "Te": (-2, 0, 4, 6),
    "Ti": (0, 2, 3, 4),
    "Tl": (0, 1, 3),
    "V": (0, 2, 3, 4, 5),
    "W": (0, 3, 4, 5, 6),
    "Y": (0, 3),
    "Zn": (0, 2),
    "Zr": (0, 2, 3, 4),
}

CRYSVCD_ALLOY_ELEMENTS = frozenset(
    symbol for symbol, states in CRYSVCD_ELEMENT_STATES.items() if 0 in states
)
CRYSVCD_IONIC_STATES: dict[str, tuple[int, ...]] = {
    symbol: tuple(state for state in states if state != 0)
    for symbol, states in CRYSVCD_ELEMENT_STATES.items()
}


def _canonical_composition(
    symbols: Sequence[str], counts: Sequence[int]
) -> tuple[list[str], list[int]]:
    if len(symbols) != len(counts):
        raise ValueError("symbols and counts must have equal length")
    merged: Counter[str] = Counter()
    for symbol, count in zip(symbols, counts):
        symbol_value = str(symbol)
        count_value = int(count)
        if symbol_value not in SYMBOL_TO_Z:
            raise ValueError(f"unsupported element symbol {symbol_value!r}")
        if count_value < 0:
            raise ValueError(f"negative count {count_value} for {symbol_value}")
        if count_value == 0:
            continue
        merged[symbol_value] += count_value
    ordered = sorted(merged, key=lambda value: SYMBOL_TO_Z[value])
    return ordered, [int(merged[symbol]) for symbol in ordered]


def _species_option(symbol: str, pieces: Sequence[tuple[int, int]]) -> dict[str, Any]:
    species = [
        {"element": symbol, "oxidation_state": int(state), "count": int(count)}
        for state, count in pieces
        if int(count) > 0
    ]
    return {
        "species": species,
        "charge": int(sum(int(state) * int(count) for state, count in pieces)),
        "mixed": len(species) > 1,
    }


@lru_cache(maxsize=256)
def _pymatgen_states(symbol: str, *, common_only: bool) -> tuple[int, ...]:
    try:
        from pymatgen.core.periodic_table import Element

        element = Element(str(symbol))
        raw_states = (
            element.common_oxidation_states
            if common_only
            else element.oxidation_states
        )
    except Exception:  # noqa: BLE001 - optional local dependency.
        return ()
    states: set[int] = set()
    for value in raw_states:
        numeric = float(value)
        if numeric.is_integer() and int(numeric) != 0 and abs(int(numeric)) <= 9:
            states.add(int(numeric))
    return tuple(sorted(states))


@lru_cache(maxsize=128)
def _pymatgen_is_metal(symbol: str) -> bool:
    try:
        from pymatgen.core.periodic_table import Element

        return bool(Element(str(symbol)).is_metal)
    except Exception:  # noqa: BLE001 - optional local dependency.
        return False


def _states_for_tier(symbol: str, tier: str) -> tuple[int, ...]:
    states = set(CRYSVCD_IONIC_STATES.get(symbol, ()))
    if tier in {"common_extension", "full_extension"}:
        states.update(_pymatgen_states(symbol, common_only=True))
    if tier == "full_extension":
        states.update(_pymatgen_states(symbol, common_only=False))
    return tuple(sorted(state for state in states if state != 0))


def _element_options(
    symbol: str,
    count: int,
    *,
    states: Sequence[int],
    allow_mixed: bool,
) -> list[dict[str, Any]]:
    states = tuple(sorted(set(int(value) for value in states if int(value) != 0)))
    options = [_species_option(symbol, ((state, count),)) for state in states]
    if not allow_mixed or count <= 1:
        return options
    for left, right in zip(states, states[1:]):
        # CrysVCD forbids opposite signs and non-adjacent states for one element.
        if left * right < 0:
            continue
        for left_count in range(1, int(count)):
            right_count = int(count) - left_count
            options.append(
                _species_option(symbol, ((left, left_count), (right, right_count)))
            )
    return options


def _path_score(path: Sequence[dict[str, Any]]) -> tuple[int, int, tuple[tuple[str, int, int], ...]]:
    flattened = tuple(
        (
            str(species["element"]),
            int(species["oxidation_state"]),
            int(species["count"]),
        )
        for option in path
        for species in option["species"]
    )
    return (
        len(flattened),
        sum(int(bool(option["mixed"])) for option in path),
        flattened,
    )


def _solve_ionic(
    symbols: Sequence[str],
    counts: Sequence[int],
    *,
    states_by_element: Mapping[str, Sequence[int]],
    allow_mixed: bool,
    max_species: int,
) -> list[dict[str, Any]] | None:
    # Charge -> best deterministic partial path.  The charge range is naturally
    # bounded by at most 20 atoms and the supported oxidation table.
    frontier: dict[int, list[dict[str, Any]]] = {0: []}
    for symbol, count in zip(symbols, counts):
        options = _element_options(
            symbol,
            int(count),
            states=states_by_element.get(symbol, ()),
            allow_mixed=allow_mixed,
        )
        if not options:
            return None
        next_frontier: dict[int, list[dict[str, Any]]] = {}
        for partial_charge, path in frontier.items():
            for option in options:
                candidate = [*path, option]
                species_count = sum(len(item["species"]) for item in candidate)
                if species_count > int(max_species):
                    continue
                charge = int(partial_charge) + int(option["charge"])
                incumbent = next_frontier.get(charge)
                if incumbent is None or _path_score(candidate) < _path_score(incumbent):
                    next_frontier[charge] = candidate
        frontier = next_frontier
        if not frontier:
            return None
    return frontier.get(0)


def assign_crysvcd_valences(
    symbols: Sequence[str],
    counts: Sequence[int],
    *,
    max_species: int = 7,
) -> dict[str, Any]:
    """Find one exact charge-balanced assignment for a fixed composition."""

    canonical_symbols, canonical_counts = _canonical_composition(symbols, counts)
    if not canonical_symbols:
        return {
            "assigned": False,
            "reason": "empty_composition",
            "species": [],
            "charge_sum": None,
        }
    crysvcd_alloy = set(canonical_symbols) <= CRYSVCD_ALLOY_ELEMENTS
    extended_all_metal = all(_pymatgen_is_metal(symbol) for symbol in canonical_symbols)
    if crysvcd_alloy or extended_all_metal:
        species = [
            {"element": symbol, "oxidation_state": 0, "count": int(count)}
            for symbol, count in zip(canonical_symbols, canonical_counts)
        ]
        if len(species) > int(max_species):
            return {
                "assigned": False,
                "reason": "species_overflow",
                "required_species": len(species),
                "max_species": int(max_species),
                "species": species,
                "charge_sum": 0,
            }
        return {
            "assigned": True,
            "reason": "ok",
            "mode": "alloy_zero",
            "formula_type": "alloy",
            "state_catalog_tier": (
                "crysvcd" if crysvcd_alloy else "common_extension"
            ),
            "species": species,
            "charge_sum": 0,
            "species_count": len(species),
            "mixed_elements": [],
        }

    path: list[dict[str, Any]] | None = None
    mode = "ionic_uniform"
    selected_tier = "full_extension"
    selected_catalog: dict[str, tuple[int, ...]] = {}
    for tier in ("crysvcd", "common_extension", "full_extension"):
        states_by_element = {
            symbol: _states_for_tier(symbol, tier) for symbol in canonical_symbols
        }
        if any(not states for states in states_by_element.values()):
            continue
        path = _solve_ionic(
            canonical_symbols,
            canonical_counts,
            states_by_element=states_by_element,
            allow_mixed=False,
            max_species=max_species,
        )
        mode = "ionic_uniform"
        if path is None:
            path = _solve_ionic(
                canonical_symbols,
                canonical_counts,
                states_by_element=states_by_element,
                allow_mixed=True,
                max_species=max_species,
            )
            mode = "ionic_mixed"
        if path is not None:
            selected_tier = tier
            selected_catalog = states_by_element
            break
    if path is None:
        full_catalog = {
            symbol: _states_for_tier(symbol, "full_extension")
            for symbol in canonical_symbols
        }
        unsupported = sorted(
            symbol for symbol, states in full_catalog.items() if not states
        )
        return {
            "assigned": False,
            "reason": (
                "unsupported_elements"
                if unsupported
                else "no_charge_neutral_assignment"
            ),
            "unsupported_elements": unsupported,
            "state_catalog": {
                symbol: list(states) for symbol, states in full_catalog.items()
            },
            "species": [],
            "charge_sum": None,
        }

    species = [dict(value) for option in path for value in option["species"]]
    mixed_elements = [
        str(symbol)
        for symbol, option in zip(canonical_symbols, path)
        if bool(option["mixed"])
    ]
    charge_sum = sum(
        int(value["oxidation_state"]) * int(value["count"]) for value in species
    )
    return {
        "assigned": True,
        "reason": "ok",
        "mode": mode,
        "formula_type": "ionic",
        "state_catalog_tier": selected_tier,
        "selected_extension_elements": sorted(
            {
                str(value["element"])
                for value in species
                if int(value["oxidation_state"])
                not in CRYSVCD_IONIC_STATES.get(str(value["element"]), ())
            }
        ),
        "state_catalog": {
            symbol: list(states) for symbol, states in selected_catalog.items()
        },
        "species": species,
        "charge_sum": int(charge_sum),
        "species_count": len(species),
        "mixed_elements": mixed_elements,
    }


def annotate_plan_with_valence(
    plan: Mapping[str, Any],
    *,
    max_species: int = 7,
) -> dict[str, Any]:
    """Attach deterministic count-valence supervision without dropping a Plan."""

    annotated = deepcopy(dict(plan))
    symbols = [str(value) for value in (plan.get("elements") or ())]
    counts = [int(value) for value in (plan.get("counts") or ())]
    assignment = assign_crysvcd_valences(symbols, counts, max_species=max_species)
    annotated["valence_assignment"] = assignment
    annotated["source_charge_bucket"] = plan.get("charge_bucket")
    if not assignment["assigned"]:
        annotated["valence_species"] = [
            {"element": symbol, "oxidation_state": None, "count": int(count)}
            for symbol, count in zip(symbols, counts)
            if int(count) > 0
        ]
        annotated["oxidation_candidates"] = "unknown"
        return annotated

    species = [dict(value) for value in assignment["species"]]
    annotated["valence_species"] = species
    states_by_element: dict[str, set[int]] = {}
    for value in species:
        states_by_element.setdefault(str(value["element"]), set()).add(
            int(value["oxidation_state"])
        )
    annotated["oxidation_candidates"] = [
        (
            next(iter(states_by_element[symbol]))
            if len(states_by_element.get(symbol, ())) == 1
            else "mixed"
        )
        if int(count) > 0
        else "unknown"
        for symbol, count in zip(symbols, counts)
    ]
    if assignment["mode"] == "alloy_zero":
        annotated["charge_bucket"] = "single_element" if len(symbols) == 1 else "all_metal"
    else:
        annotated["charge_bucket"] = "neutral_plausible"
    return annotated


def valence_catalog_manifest() -> dict[str, Any]:
    canonical = json.dumps(
        {key: list(value) for key, value in sorted(CRYSVCD_ELEMENT_STATES.items())},
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    try:
        pymatgen_version = version("pymatgen")
    except PackageNotFoundError:
        pymatgen_version = None
    return {
        "schema": "h1a2_countvalence_catalog_v2",
        "crysvcd_reference": "https://github.com/vipandyc/CrysVCD/tree/main/formula_gen",
        "crysvcd_table_sha256": hashlib.sha256(canonical).hexdigest(),
        "pymatgen_version": pymatgen_version,
        "tiers": ["crysvcd", "common_extension", "full_extension"],
        "search": "uniform_then_adjacent_same_sign_mixed",
    }


__all__ = [
    "CRYSVCD_ALLOY_ELEMENTS",
    "CRYSVCD_ELEMENT_STATES",
    "CRYSVCD_IONIC_STATES",
    "annotate_plan_with_valence",
    "assign_crysvcd_valences",
    "valence_catalog_manifest",
]
