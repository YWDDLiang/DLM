"""Composition-validity helpers aligned with CrysLLMGen metrics."""

from __future__ import annotations

from collections import Counter
from functools import lru_cache, reduce
import itertools
import math
import random
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

from crystal_dlm.fixed_slot import CHEMICAL_SYMBOLS, SYMBOL_TO_Z, Z_TO_SYMBOL


def reduced_composition(atom_types: Sequence[int]) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Return sorted element atomic numbers and gcd-reduced counts."""

    counter = Counter(int(value) for value in atom_types)
    items = [(elem, counter[elem]) for elem in sorted(counter)]
    if not items:
        return (), ()
    elems, counts = zip(*items)
    gcd_value = reduce(math.gcd, (int(count) for count in counts))
    if gcd_value <= 0:
        gcd_value = 1
    return tuple(int(elem) for elem in elems), tuple(int(count // gcd_value) for count in counts)


def element_symbols(elems: Sequence[int]) -> Tuple[str, ...]:
    """Map atomic numbers to symbols with CrysLLMGen's invalid-value fallback."""

    symbols: List[str] = []
    for elem in elems:
        atomic_number = int(elem)
        if atomic_number <= 0 or atomic_number > 119:
            atomic_number = 1
        if atomic_number < len(CHEMICAL_SYMBOLS):
            symbols.append(CHEMICAL_SYMBOLS[atomic_number])
        else:
            symbols.append("H")
    return tuple(symbols)


def _neutral_ratios_compat(smact_module: Any, ox_states: Sequence[int], stoichs: Sequence[Tuple[int, ...]], threshold: int) -> Tuple[bool, Sequence[Any]]:
    """Handle SMACT 3.x and 4.x neutral_ratios return shapes."""

    result = smact_module.neutral_ratios(ox_states, stoichs=stoichs, threshold=threshold)
    if isinstance(result, tuple) and len(result) == 2:
        cn_e, cn_r = result
        return bool(cn_e), cn_r
    if result is None:
        return False, []
    if isinstance(result, list):
        return bool(result), result
    return False, []


def formula_from_composition(elems: Sequence[int], counts: Sequence[int]) -> str:
    parts: List[str] = []
    for elem, count in zip(elems, counts):
        symbol = Z_TO_SYMBOL.get(int(elem), f"Z{int(elem)}")
        count = int(count)
        parts.append(symbol if count == 1 else f"{symbol}{count}")
    return "".join(parts)


@lru_cache(maxsize=200_000)
def _smact_validity_cached(
    elems: Tuple[int, ...],
    counts: Tuple[int, ...],
    use_pauling_test: bool,
    include_alloys: bool,
) -> bool:
    return _smact_validity_uncached(
        elems,
        counts,
        use_pauling_test=use_pauling_test,
        include_alloys=include_alloys,
    )


def _smact_validity_uncached(
    elems: Sequence[int],
    counts: Sequence[int],
    *,
    use_pauling_test: bool = True,
    include_alloys: bool = True,
) -> bool:
    """Match reference/crysllmgen/eval_utils.py::smact_validity."""

    import numpy as np
    import smact
    from smact.screening import pauling_test

    elem_symbols = element_symbols(elems)
    space = smact.element_dictionary(elem_symbols)
    smact_elems = [element for _, element in space.items()]
    electronegs = [element.pauling_eneg for element in smact_elems]
    ox_combos = [element.oxidation_states or [] for element in smact_elems]
    if len(set(elem_symbols)) == 1:
        return True
    if include_alloys:
        is_metal_list = [elem_symbol in smact.metals for elem_symbol in elem_symbols]
        if all(is_metal_list):
            return True

    threshold = np.max(counts)
    compositions = []
    for ox_states in itertools.product(*ox_combos):
        stoichs = [(count,) for count in counts]
        cn_e, cn_r = _neutral_ratios_compat(smact, ox_states, stoichs, threshold)
        if cn_e:
            if use_pauling_test:
                try:
                    electroneg_ok = pauling_test(ox_states, electronegs)
                except TypeError:
                    electroneg_ok = True
            else:
                electroneg_ok = True
            if electroneg_ok:
                for ratio in cn_r:
                    compositions.append(tuple([elem_symbols, ox_states, ratio]))
    compositions = [(item[0], item[2]) for item in compositions]
    return len(set(compositions)) > 0


def smact_validity(
    elems: Sequence[int],
    counts: Sequence[int],
    *,
    use_pauling_test: bool = True,
    include_alloys: bool = True,
) -> bool:
    """Cached wrapper matching reference/crysllmgen/eval_utils.py::smact_validity."""

    return _smact_validity_cached(
        tuple(int(value) for value in elems),
        tuple(int(value) for value in counts),
        bool(use_pauling_test),
        bool(include_alloys),
    )


def smact_validity_from_atom_types(atom_types: Sequence[int]) -> bool:
    elems, counts = reduced_composition(atom_types)
    if not elems:
        return False
    return smact_validity(elems, counts)


def classify_smact_validity(
    elems: Sequence[int],
    counts: Sequence[int],
    *,
    use_pauling_test: bool = True,
    include_alloys: bool = True,
) -> Dict[str, Any]:
    """Approximate why a composition passes or fails the CrysLLMGen SMACT check."""

    import numpy as np
    import smact
    from smact.screening import pauling_test

    elem_symbols = element_symbols(elems)
    if not elem_symbols:
        return {"valid": False, "reason": "empty_composition"}

    space = smact.element_dictionary(elem_symbols)
    smact_elems = [element for _, element in space.items()]
    electronegs = [element.pauling_eneg for element in smact_elems]
    ox_combos = [element.oxidation_states or [] for element in smact_elems]
    if len(set(elem_symbols)) == 1:
        return {"valid": True, "reason": "single_element_shortcut"}
    if include_alloys:
        is_metal_list = [elem_symbol in smact.metals for elem_symbol in elem_symbols]
        if all(is_metal_list):
            return {"valid": True, "reason": "all_metal_shortcut"}
    if any(len(combo) == 0 for combo in ox_combos):
        return {"valid": False, "reason": "oxidation_state_missing"}

    threshold = np.max(counts)
    charge_neutral_seen = False
    pauling_error_seen = False
    for ox_states in itertools.product(*ox_combos):
        stoichs = [(count,) for count in counts]
        cn_e, cn_r = _neutral_ratios_compat(smact, ox_states, stoichs, threshold)
        if not cn_e:
            continue
        charge_neutral_seen = True
        if use_pauling_test:
            try:
                electroneg_ok = pauling_test(ox_states, electronegs)
            except TypeError:
                electroneg_ok = True
                pauling_error_seen = True
        else:
            electroneg_ok = True
        if electroneg_ok and cn_r:
            return {
                "valid": True,
                "reason": "charge_neutral_pauling_valid",
                "oxidation_states": tuple(int(value) for value in ox_states),
            }
    if not charge_neutral_seen:
        return {"valid": False, "reason": "charge_neutrality_fail"}
    return {
        "valid": False,
        "reason": "pauling_fail_or_ratio_rejected",
        "pauling_type_error_seen": pauling_error_seen,
    }


def composition_record(atom_types: Sequence[int]) -> Dict[str, Any]:
    elems, counts = reduced_composition(atom_types)
    symbols = element_symbols(elems)
    classification = classify_smact_validity(elems, counts) if elems else {"valid": False, "reason": "empty"}
    return {
        "elems": list(elems),
        "symbols": list(symbols),
        "counts": list(counts),
        "formula": formula_from_composition(elems, counts),
        "num_atoms": int(sum(Counter(int(value) for value in atom_types).values())),
        "num_elements": len(elems),
        "comp_valid": bool(classification["valid"]),
        "reason": classification["reason"],
    }


def pbc_duplicate_record(frac_coords: Sequence[Sequence[float]]) -> Dict[str, Any]:
    """Diagnose exact and periodic-boundary duplicate fractional coordinates."""

    exact_counts: Counter[Tuple[int, int, int]] = Counter()
    pbc_counts: Counter[Tuple[int, int, int]] = Counter()
    for coord in frac_coords:
        bins = tuple(int(round(float(value) * 100.0)) for value in coord)
        exact_counts[bins] += 1
        pbc_counts[tuple(value % 100 for value in bins)] += 1
    exact_duplicate_sites = sum(count - 1 for count in exact_counts.values() if count > 1)
    pbc_duplicate_sites = sum(count - 1 for count in pbc_counts.values() if count > 1)
    return {
        "exact_duplicate_site_count": exact_duplicate_sites,
        "pbc_equivalent_duplicate_site_count": pbc_duplicate_sites,
        "pbc_only_duplicate_site_count": max(0, pbc_duplicate_sites - exact_duplicate_sites),
        "has_exact_duplicate": exact_duplicate_sites > 0,
        "has_pbc_equivalent_duplicate": pbc_duplicate_sites > 0,
    }


def active_element_answer_positions(max_atoms: int = 20) -> List[int]:
    return [8 + 5 * slot_index for slot_index in range(max_atoms)]


def replace_active_elements(
    token_ids: Sequence[int],
    *,
    prompt_length: int,
    element_token_ids: Sequence[int],
    token_id_to_atomic_number: Mapping[int, int],
    max_atoms: int = 20,
    attempts: int = 20,
    rng: random.Random | None = None,
    validity_fn: Callable[[Sequence[int]], bool] = smact_validity_from_atom_types,
) -> Tuple[List[int] | None, Dict[str, Any]]:
    """Corrupt active element tokens until the composition is SMACT-invalid."""

    rng = rng or random.Random()
    ids = list(int(value) for value in token_ids)
    if prompt_length >= len(ids):
        return None, {"reason": "prompt_length_outside_sequence"}
    answer_start = int(prompt_length)
    active_positions = []
    atom_types = []
    for slot_index, rel_pos in enumerate(active_element_answer_positions(max_atoms=max_atoms)):
        abs_pos = answer_start + rel_pos
        if abs_pos >= len(ids):
            break
        token_id = int(ids[abs_pos])
        atomic_number = token_id_to_atomic_number.get(token_id)
        if atomic_number is None:
            continue
        active_positions.append(abs_pos)
        atom_types.append(atomic_number)
    if not active_positions:
        return None, {"reason": "no_active_element_positions"}
    if not validity_fn(atom_types):
        return None, {"reason": "positive_composition_not_valid"}

    element_token_ids = [int(value) for value in element_token_ids]
    atomic_number_to_token_id = {
        int(atomic_number): int(token_id)
        for token_id, atomic_number in token_id_to_atomic_number.items()
    }
    noble_gas_pattern = [
        atomic_number_to_token_id[z]
        for z in (2, 10, 18)
        if z in atomic_number_to_token_id
    ]
    if len(active_positions) >= 2 and len(noble_gas_pattern) >= 2:
        negative = list(ids)
        for position_idx, abs_pos in enumerate(active_positions):
            negative[abs_pos] = noble_gas_pattern[position_idx % len(noble_gas_pattern)]
        negative_atom_types = [
            int(token_id_to_atomic_number[int(negative[abs_pos])])
            for abs_pos in active_positions
        ]
        if not validity_fn(negative_atom_types):
            return negative, {
                "reason": "ok",
                "attempt": 0,
                "mode": "deterministic_noble_gas_mixture",
                "active_element_count": len(active_positions),
            }

    for attempt in range(int(attempts)):
        negative = list(ids)
        changed = False
        for abs_pos in active_positions:
            old_id = int(negative[abs_pos])
            new_id = old_id
            for _ in range(8):
                new_id = int(rng.choice(element_token_ids))
                if new_id != old_id:
                    break
            negative[abs_pos] = new_id
            changed = changed or new_id != old_id
        if not changed:
            continue
        negative_atom_types = [
            int(token_id_to_atomic_number[int(negative[abs_pos])])
            for abs_pos in active_positions
            if int(negative[abs_pos]) in token_id_to_atomic_number
        ]
        if len(negative_atom_types) != len(active_positions):
            continue
        if not validity_fn(negative_atom_types):
            return negative, {
                "reason": "ok",
                "attempt": attempt + 1,
                "active_element_count": len(active_positions),
            }
    return None, {"reason": "no_invalid_negative_found", "attempts": int(attempts)}
