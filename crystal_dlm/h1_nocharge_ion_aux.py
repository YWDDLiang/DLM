"""Matched no-charge/ion auxiliary helpers for the H1 Planner SFT route.

The module is deliberately independent of model code.  It defines a compact,
round-trippable per-atom oxidation witness, the exact C0/C1 task ledger, and a
SMACT-4 ICSD24 evaluator contract.  Importing it never requires SMACT; the
optional evaluator is loaded only by the data builder/audit.
"""

from __future__ import annotations

from collections import Counter
from functools import reduce
import hashlib
import itertools
import json
import math
import random
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from crystal_dlm.fixed_slot import SYMBOL_TO_Z
from crystal_dlm.r5_plan_body import formula_from_symbol_counts


H1_NOCHARGE_ION_AUX_SCHEMA = "h1_nocharge_ion_aux_v1"
H1_NOCHARGE_ION_AUX_SEED = 26080617
SMACT4_VERSION = "4.0.0"
SMACT4_WHEEL_SHA256 = "e3eb968da92d47a8ef9a4af42af5589a6de61cccbca9d329937e1f4e402f0551"
SMACT4_SOURCE_COMMIT = "c2b4d3ce6fa3a8c39fd21ab252934253dc66e131"
H1_NOCHARGE_ION_AUX_TASK_COUNTS = {
    "direct_nocharge_plan": 960,
    "sequence_to_formula": 160,
    "oxidation_infill": 160,
    "conditional_mp20_anchor": 1280,
    "p0_kl_anchor": 640,
}
H1_NOCHARGE_ION_AUX_VALIDATION_TASK_COUNTS = {
    "direct_nocharge_plan": 192,
    "sequence_to_formula": 32,
    "oxidation_infill": 32,
    "conditional_mp20_anchor": 256,
    "p0_kl_anchor": 128,
}
SMACT4_ICSD24_FILTER = {
    "include_zero": False,
    "consensus": 3,
    "commonality": "medium",
    "use_pauling_test": True,
    "include_alloys": True,
    "check_metallicity": False,
    "metallicity_threshold": 0.7,
    "mixed_valence": True,
}

# Frozen from the SMACT-4 mixed-valence implementation.  Other elements retain
# one oxidation state per element even when their count is greater than one.
SMACT4_MIXED_VALENCE_ELEMENTS = frozenset(
    {
        "Fe", "Mn", "Co", "Cu", "Ni", "V", "Ti", "Cr", "Nb", "Mo",
        "W", "Re", "Ru", "Os", "Pd", "Ag", "Au", "Sn", "Sb", "Bi",
        "Ce", "Eu", "Yb", "U",
    }
)
_ION_TOKEN = re.compile(r"^([A-Z][a-z]?):(Q[PMZXU]\d{2})$")
_ATOM_TOKEN = re.compile(r"^([A-Z][a-z]?):C(\d{3})$")


def canonical_json_sha256(payload: Any) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def oxidation_to_code(value: int | None, *, neutral_placeholder: bool = False) -> str:
    if neutral_placeholder:
        return "QX00"
    if value is None:
        return "QU00"
    oxidation = int(value)
    magnitude = abs(oxidation)
    if magnitude > 99:
        raise ValueError(f"oxidation state {oxidation} outside encodable range")
    if oxidation > 0:
        return f"QP{magnitude:02d}"
    if oxidation < 0:
        return f"QM{magnitude:02d}"
    return "QZ00"


def oxidation_from_code(value: str) -> int | None:
    text = str(value).strip().upper()
    match = re.fullmatch(r"Q([PMZXU])(\d{2})", text)
    if match is None:
        raise ValueError(f"invalid oxidation code {value!r}")
    sign, magnitude_text = match.groups()
    magnitude = int(magnitude_text)
    if sign in {"X", "U"}:
        if magnitude != 0:
            raise ValueError(f"placeholder oxidation code must end in 00, got {value!r}")
        return None
    if sign == "Z":
        if magnitude != 0:
            raise ValueError(f"zero oxidation code must be QZ00, got {value!r}")
        return 0
    return magnitude if sign == "P" else -magnitude


def expanded_symbols(symbols: Sequence[str], counts: Sequence[int]) -> list[str]:
    if len(symbols) != len(counts):
        raise ValueError("symbols and counts must be aligned")
    rows: list[tuple[str, int]] = []
    for symbol, count in zip(symbols, counts):
        symbol = str(symbol)
        if symbol not in SYMBOL_TO_Z:
            raise ValueError(f"unsupported element symbol {symbol!r}")
        count_value = int(count)
        if count_value <= 0:
            raise ValueError(f"element count for {symbol} must be positive")
        rows.append((symbol, count_value))
    rows.sort(key=lambda item: SYMBOL_TO_Z[item[0]])
    return [symbol for symbol, count in rows for _ in range(count)]


def canonicalize_ion_witness(witness: Sequence[tuple[str, int]]) -> list[tuple[str, int]]:
    rows = [(str(symbol), int(oxidation)) for symbol, oxidation in witness]
    for symbol, _oxidation in rows:
        if symbol not in SYMBOL_TO_Z:
            raise ValueError(f"unsupported ion element symbol {symbol!r}")
    # Preserve mixed-valence multiplicity while making the witness independent
    # of solver traversal order.
    return sorted(rows, key=lambda item: (SYMBOL_TO_Z[item[0]], item[1]))


def format_ion_sequence(
    symbols: Sequence[str],
    counts: Sequence[int],
    witness: Sequence[tuple[str, int]] | None = None,
    *,
    neutral_placeholder: bool = False,
) -> str:
    atoms = expanded_symbols(symbols, counts)
    if neutral_placeholder:
        rows = [(symbol, None) for symbol in atoms]
    else:
        if witness is None:
            raise ValueError("an oxidation witness is required for the C1 ion sequence")
        canonical = canonicalize_ion_witness(witness)
        if Counter(symbol for symbol, _ in canonical) != Counter(atoms):
            raise ValueError("ion witness element multiplicities do not match the formula")
        rows = [(symbol, oxidation) for symbol, oxidation in canonical]
    return "I=" + ",".join(
        f"{symbol}:{oxidation_to_code(oxidation, neutral_placeholder=neutral_placeholder)}"
        for symbol, oxidation in rows
    )


def format_atom_sequence(symbols: Sequence[str], counts: Sequence[int]) -> str:
    """Encode one C001 token per atom for the matched chemistry-free C0 arm."""

    return "A=" + ",".join(f"{symbol}:C001" for symbol in expanded_symbols(symbols, counts))


def parse_atom_sequence(text: str) -> list[str]:
    value = str(text).strip()
    if not value.startswith("A="):
        raise ValueError("atom sequence must start with A=")
    payload = value[2:]
    if not payload:
        raise ValueError("atom sequence is empty")
    atoms: list[str] = []
    for token in payload.split(","):
        match = _ATOM_TOKEN.fullmatch(token.strip())
        if match is None:
            raise ValueError(f"invalid atom token {token!r}")
        symbol, count_text = match.groups()
        if symbol not in SYMBOL_TO_Z:
            raise ValueError(f"unsupported atom element symbol {symbol!r}")
        if int(count_text) != 1:
            raise ValueError(f"repeated atom sequence requires C001, got C{count_text}")
        atoms.append(symbol)
    return atoms


def formula_from_atom_sequence(text: str) -> str:
    counts = Counter(parse_atom_sequence(text))
    symbols = sorted(counts, key=lambda symbol: SYMBOL_TO_Z[symbol])
    return formula_from_symbol_counts(symbols, [counts[symbol] for symbol in symbols])


def parse_ion_sequence(text: str) -> list[tuple[str, int | None]]:
    value = str(text).strip()
    if not value.startswith("I="):
        raise ValueError("ion sequence must start with I=")
    payload = value[2:]
    if not payload:
        raise ValueError("ion sequence is empty")
    rows: list[tuple[str, int | None]] = []
    for token in payload.split(","):
        match = _ION_TOKEN.fullmatch(token.strip())
        if match is None:
            raise ValueError(f"invalid ion token {token!r}")
        symbol, code = match.groups()
        if symbol not in SYMBOL_TO_Z:
            raise ValueError(f"unsupported ion element symbol {symbol!r}")
        rows.append((symbol, oxidation_from_code(code)))
    return rows


def formula_from_ion_sequence(text: str) -> str:
    rows = parse_ion_sequence(text)
    counts = Counter(symbol for symbol, _oxidation in rows)
    symbols = sorted(counts, key=lambda symbol: SYMBOL_TO_Z[symbol])
    return formula_from_symbol_counts(symbols, [counts[symbol] for symbol in symbols])


def ion_charge_sum(text: str) -> int | None:
    states = [oxidation for _symbol, oxidation in parse_ion_sequence(text)]
    if any(value is None for value in states):
        return None
    return sum(int(value) for value in states if value is not None)


def deterministic_task_schedule(
    task_counts: Mapping[str, int],
    *,
    seed: int = H1_NOCHARGE_ION_AUX_SEED,
) -> list[str]:
    tasks = [str(task) for task, count in sorted(task_counts.items()) for _ in range(int(count))]
    if any(int(count) < 0 for count in task_counts.values()):
        raise ValueError("task counts must be non-negative")
    random.Random(int(seed)).shuffle(tasks)
    return tasks


def deterministic_ranked_indices(
    population: Sequence[int],
    count: int,
    *,
    seed: int,
    role: str,
) -> list[int]:
    unique = sorted(set(int(value) for value in population))
    if int(count) > len(unique):
        raise ValueError(f"role {role!r} requests {count} rows from a population of {len(unique)}")
    ranked = sorted(
        unique,
        key=lambda value: hashlib.sha256(f"{int(seed)}|{role}|{value}".encode("utf-8")).digest(),
    )
    return ranked[: int(count)]


def _largest_remainder_targets(values: Sequence[str], total: int) -> dict[str, int]:
    counts = Counter(str(value) for value in values)
    denominator = sum(counts.values())
    if denominator <= 0:
        return {}
    quotas = {key: float(total) * value / denominator for key, value in counts.items()}
    targets = {key: int(quota) for key, quota in quotas.items()}
    remaining = int(total) - sum(targets.values())
    order = sorted(quotas, key=lambda key: (-(quotas[key] - targets[key]), key))
    for key in order[:remaining]:
        targets[key] += 1
    return targets


def raked_select_indices(
    rows: Sequence[Mapping[str, Any]],
    population: Sequence[int],
    target_rows: Sequence[Mapping[str, Any]],
    count: int,
    *,
    fields: Sequence[str],
    seed: int,
    role: str,
    exact_formula_cap: int = 4,
    reduced_formula_cap: int = 4,
) -> tuple[list[int], dict[str, Any]]:
    """Greedily rake a no-replacement subset toward fixed MP-20 marginals."""

    candidates = sorted(set(int(value) for value in population))
    if int(count) > len(candidates):
        raise ValueError(f"role {role!r} requests {count} rows from a population of {len(candidates)}")
    targets = {
        str(field): _largest_remainder_targets(
            [str((row.get("plan") or {}).get(field, "unknown")) for row in target_rows],
            int(count),
        )
        for field in fields
    }
    selected_hist = {str(field): Counter() for field in fields}
    exact_hist: Counter[str] = Counter()
    reduced_hist: Counter[str] = Counter()
    selected: list[int] = []
    available = set(candidates)

    for _slot in range(int(count)):
        best_idx: int | None = None
        best_key: tuple[float, bytes] | None = None
        for idx in available:
            plan = rows[idx].get("plan") or {}
            exact_formula = str(plan.get("formula", "unknown"))
            reduced_formula = str(plan.get("reduced_formula", exact_formula))
            if exact_hist[exact_formula] >= int(exact_formula_cap):
                continue
            if reduced_hist[reduced_formula] >= int(reduced_formula_cap):
                continue
            score = 0.0
            for field in fields:
                field = str(field)
                value = str(plan.get(field, "unknown"))
                target = int(targets[field].get(value, 0))
                deficit = max(0, target - int(selected_hist[field][value]))
                score += float(deficit) / max(1, target)
            tie = hashlib.sha256(f"{int(seed)}|{role}|{idx}".encode("utf-8")).digest()
            key = (score, bytes(255 - value for value in tie))
            if best_key is None or key > best_key:
                best_key = key
                best_idx = idx
        if best_idx is None:
            raise ValueError(
                f"raking role {role!r} exhausted support at {len(selected)}/{count} under "
                f"exact_formula_cap={exact_formula_cap}, reduced_formula_cap={reduced_formula_cap}"
            )
        selected.append(best_idx)
        available.remove(best_idx)
        plan = rows[best_idx].get("plan") or {}
        exact_formula = str(plan.get("formula", "unknown"))
        reduced_formula = str(plan.get("reduced_formula", exact_formula))
        exact_hist[exact_formula] += 1
        reduced_hist[reduced_formula] += 1
        for field in fields:
            field = str(field)
            selected_hist[field][str(plan.get(field, "unknown"))] += 1

    residual = {
        field: {
            value: int(target) - int(selected_hist[field][value])
            for value, target in sorted(targets[field].items())
            if int(target) != int(selected_hist[field][value])
        }
        for field in fields
    }
    report = {
        "role": str(role),
        "requested": int(count),
        "selected": len(selected),
        "population": len(candidates),
        "fields": [str(field) for field in fields],
        "targets": targets,
        "selected_histograms": {
            field: dict(sorted((key, int(value)) for key, value in histogram.items()))
            for field, histogram in selected_hist.items()
        },
        "residual": residual,
        "residual_l1": sum(abs(value) for field in residual.values() for value in field.values()),
        "exact_formula_cap": int(exact_formula_cap),
        "reduced_formula_cap": int(reduced_formula_cap),
        "max_exact_formula_exposure": max(exact_hist.values(), default=0),
        "max_reduced_formula_exposure": max(reduced_hist.values(), default=0),
    }
    return selected, report


def formula_weight_span(answer: str, formula: str, *, weight: float = 2.0) -> list[dict[str, Any]]:
    start = str(answer).find(str(formula))
    if start < 0:
        raise ValueError(f"formula {formula!r} is absent from the answer")
    return [{"start": start, "end": start + len(str(formula)), "weight": float(weight), "label": "formula"}]


def payload_weight_span(answer: str, payload: str, *, weight: float = 2.0, label: str = "chemistry_payload") -> list[dict[str, Any]]:
    start = str(answer).find(str(payload))
    if start < 0:
        raise ValueError(f"payload {payload!r} is absent from the answer")
    return [{"start": start, "end": start + len(str(payload)), "weight": float(weight), "label": str(label)}]


def _pauling_ok(states: Sequence[int], electronegs: Sequence[float | None]) -> bool:
    from smact.screening import pauling_test

    try:
        return bool(pauling_test(tuple(int(value) for value in states), list(electronegs)))
    except TypeError:
        # Match SMACT's compatibility behavior for missing electronegativity.
        return True


def load_smact4_icsd24_oxidation_map() -> tuple[dict[str, list[int]], dict[str, Any]]:
    import smact
    import smact.screening
    import smact.utils.oxidation
    from smact.utils.oxidation import ICSD24OxStatesFilter

    config = SMACT4_ICSD24_FILTER
    filtered = ICSD24OxStatesFilter().filter(
        consensus=int(config["consensus"]),
        include_zero=bool(config["include_zero"]),
        commonality=config["commonality"],
    )
    oxidation_map = {
        str(row["element"]): sorted({int(value) for value in str(row["oxidation_state"]).split()})
        for _, row in filtered.iterrows()
    }
    contract = {
        "schema": H1_NOCHARGE_ION_AUX_SCHEMA,
        "smact_version": str(getattr(smact, "__version__", "unknown")),
        "expected_smact_version": SMACT4_VERSION,
        "release_wheel_sha256": SMACT4_WHEEL_SHA256,
        "release_source_commit": SMACT4_SOURCE_COMMIT,
        "icsd24_filter": dict(config),
        "oxidation_map_sha256": canonical_json_sha256(oxidation_map),
        "element_count": len(oxidation_map),
        "state_count": sum(len(states) for states in oxidation_map.values()),
        "installed_screening_source_sha256": hashlib.sha256(
            Path(smact.screening.__file__).read_bytes()
        ).hexdigest(),
        "installed_oxidation_source_sha256": hashlib.sha256(
            Path(smact.utils.oxidation.__file__).read_bytes()
        ).hexdigest(),
    }
    contract["contract_sha256"] = canonical_json_sha256(contract)
    return oxidation_map, contract


def deterministic_smact4_witness(
    symbols: Sequence[str],
    counts: Sequence[int],
    oxidation_map: Mapping[str, Sequence[int]],
    *,
    allow_mixed_valence: bool = True,
    max_projected_combinations: int = 1_000_000,
) -> tuple[list[tuple[str, int]] | None, str]:
    """Return a deterministic fixed-stoichiometry SMACT-4 witness.

    Unary and all-metal fast paths are identified but intentionally return no
    witness because they are never positive de-novo targets in this route.
    """

    import smact

    ordered_raw = sorted(
        [(str(symbol), int(count)) for symbol, count in zip(symbols, counts)],
        key=lambda item: SYMBOL_TO_Z[item[0]],
    )
    if len(ordered_raw) == 1:
        return None, "single_element_shortcut"
    if all(symbol in smact.metals for symbol, _count in ordered_raw):
        return None, "all_metal_shortcut"
    if any(not oxidation_map.get(symbol) for symbol, _count in ordered_raw):
        return None, "oxidation_state_missing"

    # SMACT 4 reduces integer stoichiometries before both its uniform and
    # mixed-valence searches.  Search the same reduced problem, then expand a
    # successful witness back to the exact unreduced MP-20 atom sequence used
    # by the Planner auxiliary task.
    common_factor = reduce(math.gcd, (count for _symbol, count in ordered_raw))
    ordered = [(symbol, count // common_factor) for symbol, count in ordered_raw]

    def expand_to_raw_formula(witness: Sequence[tuple[str, int]]) -> list[tuple[str, int]]:
        return canonicalize_ion_witness(
            [item for item in witness for _ in range(common_factor)]
        )

    smact_elements = {symbol: smact.Element(symbol) for symbol, _count in ordered}
    uniform_options = [tuple(sorted(set(int(value) for value in oxidation_map[symbol]))) for symbol, _ in ordered]
    uniform_projected = 1
    for options in uniform_options:
        uniform_projected *= len(options)
    if uniform_projected <= int(max_projected_combinations):
        for states in itertools.product(*uniform_options):
            if sum(int(state) * count for state, (_symbol, count) in zip(states, ordered)) != 0:
                continue
            electronegs = [smact_elements[symbol].pauling_eneg for symbol, _count in ordered]
            if _pauling_ok(states, electronegs):
                witness = [
                    (symbol, int(state))
                    for state, (symbol, count) in zip(states, ordered)
                    for _ in range(count)
                ]
                return expand_to_raw_formula(witness), "uniform_primary"

    if not allow_mixed_valence:
        return None, "no_uniform_witness"

    options: list[tuple[int, ...]] = []
    site_rows: list[tuple[str, int]] = []
    electronegs: list[float | None] = []
    projected = 1
    for symbol, count in ordered:
        states = tuple(sorted(set(int(value) for value in oxidation_map[symbol])))
        if symbol in SMACT4_MIXED_VALENCE_ELEMENTS:
            for _ in range(count):
                options.append(states)
                site_rows.append((symbol, 1))
                electronegs.append(smact_elements[symbol].pauling_eneg)
                projected *= len(states)
        else:
            options.append(states)
            site_rows.append((symbol, count))
            electronegs.append(smact_elements[symbol].pauling_eneg)
            projected *= len(states)
        if projected > int(max_projected_combinations):
            return None, "mixed_valence_projection_overflow"

    for states in itertools.product(*options):
        if sum(int(state) * stoich for state, (_symbol, stoich) in zip(states, site_rows)) != 0:
            continue
        if not _pauling_ok(states, electronegs):
            continue
        witness = [
            (symbol, int(state))
            for state, (symbol, stoich) in zip(states, site_rows)
            for _ in range(stoich)
        ]
        return expand_to_raw_formula(witness), "mixed_valence_only"
    return None, "charge_or_pauling_fail"


def smact4_validity_with_witness(
    formula: str,
    symbols: Sequence[str],
    counts: Sequence[int],
    oxidation_map: Mapping[str, Sequence[int]],
) -> dict[str, Any]:
    from smact.screening import ICSD24FilterConfig, smact_validity

    config = SMACT4_ICSD24_FILTER
    official_valid = bool(
        smact_validity(
            str(formula),
            use_pauling_test=bool(config["use_pauling_test"]),
            include_alloys=bool(config["include_alloys"]),
            check_metallicity=bool(config["check_metallicity"]),
            metallicity_threshold=float(config["metallicity_threshold"]),
            oxidation_states_set=None,
            icsd_filter=ICSD24FilterConfig(
                include_zero=bool(config["include_zero"]),
                consensus=int(config["consensus"]),
                commonality=config["commonality"],
            ),
            mixed_valence=bool(config["mixed_valence"]),
        )
    )
    witness, stratum = deterministic_smact4_witness(
        symbols,
        counts,
        oxidation_map,
        allow_mixed_valence=bool(config["mixed_valence"]),
    )
    shortcut = stratum in {"single_element_shortcut", "all_metal_shortcut"}
    witness_valid = witness is not None or shortcut
    return {
        "valid": official_valid,
        "witness_valid": witness_valid,
        "official_witness_parity": official_valid == witness_valid,
        "stratum": stratum,
        "witness": witness,
        "charge_sum": None if witness is None else sum(oxidation for _symbol, oxidation in witness),
    }


def assert_task_contract(task_counts: Mapping[str, int], *, expected_total: int) -> None:
    unknown = set(task_counts) - set(H1_NOCHARGE_ION_AUX_TASK_COUNTS)
    if unknown:
        raise ValueError(f"unknown no-charge SFT tasks: {sorted(unknown)}")
    if sum(int(value) for value in task_counts.values()) != int(expected_total):
        raise ValueError(f"task ledger total is not {expected_total}: {dict(task_counts)}")


def validation_anchor_nll_gate(
    reference_nll: float,
    candidate_nll: float,
    *,
    maximum_relative_degradation: float = 0.01,
) -> dict[str, Any]:
    """Apply the frozen full-MP20 conditional-anchor NLL noninferiority gate."""

    reference = float(reference_nll)
    candidate = float(candidate_nll)
    margin = float(maximum_relative_degradation)
    if not math.isfinite(reference) or reference <= 0.0:
        raise ValueError(f"reference anchor NLL must be finite and positive, got {reference!r}")
    if not math.isfinite(candidate) or candidate < 0.0:
        raise ValueError(f"candidate anchor NLL must be finite and non-negative, got {candidate!r}")
    if not math.isfinite(margin) or margin < 0.0:
        raise ValueError(f"anchor NLL degradation margin must be finite and non-negative, got {margin!r}")
    relative = candidate / reference - 1.0
    return {
        "reference_nll": reference,
        "candidate_nll": candidate,
        "relative_degradation": relative,
        "maximum_relative_degradation": margin,
        "passed": relative <= margin + 1e-12,
    }


__all__ = [
    "H1_NOCHARGE_ION_AUX_SCHEMA",
    "H1_NOCHARGE_ION_AUX_SEED",
    "H1_NOCHARGE_ION_AUX_TASK_COUNTS",
    "H1_NOCHARGE_ION_AUX_VALIDATION_TASK_COUNTS",
    "SMACT4_VERSION",
    "SMACT4_WHEEL_SHA256",
    "SMACT4_SOURCE_COMMIT",
    "SMACT4_ICSD24_FILTER",
    "SMACT4_MIXED_VALENCE_ELEMENTS",
    "assert_task_contract",
    "canonical_json_sha256",
    "canonicalize_ion_witness",
    "deterministic_ranked_indices",
    "deterministic_smact4_witness",
    "deterministic_task_schedule",
    "expanded_symbols",
    "format_atom_sequence",
    "format_ion_sequence",
    "formula_from_atom_sequence",
    "formula_from_ion_sequence",
    "formula_weight_span",
    "ion_charge_sum",
    "load_smact4_icsd24_oxidation_map",
    "oxidation_from_code",
    "oxidation_to_code",
    "parse_ion_sequence",
    "parse_atom_sequence",
    "payload_weight_span",
    "raked_select_indices",
    "smact4_validity_with_witness",
    "validation_anchor_nll_gate",
]
