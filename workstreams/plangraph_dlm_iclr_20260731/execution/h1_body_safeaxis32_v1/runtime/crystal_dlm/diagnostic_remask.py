"""Helpers for fixed-slot diagnostic remasking and geometry diagnostics."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from crystal_dlm.fixed_slot import (
    ANSWER_TOKEN_COUNT,
    MASK_TOKEN_ID,
    FixedSlotConfig,
    arrays_to_tokens,
    answer_token_count,
)

HIGH_SYMMETRY_COORD_VALUES = {0, 25, 50, 75, 100}


def slot_base(slot_index: int) -> int:
    return 7 + int(slot_index) * 5


def lattice_positions() -> List[int]:
    return [1, 2, 3, 4, 5, 6]


def active_coord_positions(num_atoms: int, config: FixedSlotConfig = FixedSlotConfig()) -> List[int]:
    positions: List[int] = []
    for slot_index in range(min(int(num_atoms), int(config.max_atoms))):
        base = slot_base(slot_index)
        positions.extend([base + 2, base + 3, base + 4])
    return positions


def remask_positions_for_mode(
    arrays: Mapping[str, Any],
    mode: str,
    config: FixedSlotConfig = FixedSlotConfig(),
) -> List[int]:
    if mode == "lattice_only":
        return lattice_positions()
    if mode == "geometry":
        return lattice_positions() + active_coord_positions(int(arrays["num_atoms"]), config=config)
    raise ValueError(f"Unsupported remask mode {mode!r}")


def fixed_slot_tokens_from_arrays(
    arrays: Mapping[str, Any],
    config: FixedSlotConfig = FixedSlotConfig(),
) -> List[str]:
    tokens = arrays.get("tokens")
    expected = answer_token_count(config)
    if isinstance(tokens, list) and len(tokens) >= expected:
        return [str(token) for token in tokens[:expected]]
    tokens, _ = arrays_to_tokens(
        lengths=arrays["lengths"],
        angles=arrays["angles"],
        species=arrays["species"],
        frac_coords=arrays["frac_coords"],
        config=config,
    )
    return tokens


def build_prefill_token_ids_by_position(
    tokenizer: Any,
    arrays_batch: Sequence[Mapping[str, Any]],
    *,
    mode: str,
    config: FixedSlotConfig = FixedSlotConfig(),
    mask_token_id: int = MASK_TOKEN_ID,
) -> Dict[int, List[int]]:
    """Return per-generation-position prefill ids for a batch.

    Positions selected for remasking receive ``mask_token_id`` for that sample.
    This lets a single batch contain different atom counts while keeping
    inactive slot pads frozen.
    """

    vocab = tokenizer.get_vocab()
    batch_tokens = [fixed_slot_tokens_from_arrays(arrays, config=config) for arrays in arrays_batch]
    batch_remask = [
        set(remask_positions_for_mode(arrays, mode=mode, config=config)) for arrays in arrays_batch
    ]
    prefill: Dict[int, List[int]] = {}
    for position in range(answer_token_count(config)):
        values: List[int] = []
        has_frozen_value = False
        for tokens, remask in zip(batch_tokens, batch_remask):
            if position in remask:
                values.append(int(mask_token_id))
                continue
            token = tokens[position]
            if token not in vocab:
                raise RuntimeError(f"Tokenizer is missing fixed-slot token {token!r}")
            values.append(int(vocab[token]))
            has_frozen_value = True
        if has_frozen_value:
            prefill[position] = values
    return prefill


def pbc_coord_key(coord: Sequence[float]) -> Tuple[int, int, int]:
    return tuple(int(round(float(value) * 100.0)) % 100 for value in coord)  # type: ignore[return-value]


def composition_signature(arrays: Mapping[str, Any]) -> Tuple[Tuple[str, int], ...]:
    return tuple(sorted(Counter(str(symbol) for symbol in arrays["species"]).items()))


def composition_preserved(source: Mapping[str, Any], candidate: Mapping[str, Any]) -> bool:
    return (
        int(source["num_atoms"]) == int(candidate["num_atoms"])
        and list(source["species"]) == list(candidate["species"])
        and composition_signature(source) == composition_signature(candidate)
    )


def formula_key(arrays: Mapping[str, Any]) -> str:
    parts = []
    for symbol, count in composition_signature(arrays):
        parts.append(symbol if count == 1 else f"{symbol}{count}")
    return "".join(parts)


def pbc_duplicate_count(arrays: Mapping[str, Any]) -> int:
    counts = Counter(pbc_coord_key(coord) for coord in arrays["frac_coords"])
    return sum(count - 1 for count in counts.values() if count > 1)


def geometry_degeneracy_record(arrays: Mapping[str, Any]) -> Dict[str, Any]:
    lengths = [float(value) for value in arrays["lengths"]]
    angles = [float(value) for value in arrays["angles"]]
    coords = list(arrays["frac_coords"])
    high_sym_site_count = 0
    high_sym_component_count = 0
    component_count = 0
    for coord in coords:
        bins = [int(round(float(value) * 100.0)) for value in coord]
        component_count += len(bins)
        high_sym_component_count += sum(1 for value in bins if value in HIGH_SYMMETRY_COORD_VALUES)
        if all(value in HIGH_SYMMETRY_COORD_VALUES for value in bins):
            high_sym_site_count += 1
    all_lengths_equal = max(lengths) - min(lengths) < 1e-6 if lengths else False
    all_angles_90 = all(abs(value - 90.0) < 1e-6 for value in angles)
    return {
        "all_lengths_equal": bool(all_lengths_equal),
        "all_angles_90": bool(all_angles_90),
        "high_symmetry_coord_fraction": high_sym_site_count / max(1, len(coords)),
        "high_symmetry_coord_component_fraction": high_sym_component_count / max(1, component_count),
        "pbc_equivalent_duplicate_site_count": pbc_duplicate_count(arrays),
        "num_atoms": int(arrays["num_atoms"]),
        "formula": formula_key(arrays),
    }


def anti_high_symmetry_failures(
    arrays: Mapping[str, Any],
    *,
    max_high_symmetry_coord_fraction: float | None = None,
    reject_all_lengths_equal: bool = False,
    reject_all_angles_90: bool = False,
) -> List[str]:
    record = geometry_degeneracy_record(arrays)
    failures: List[str] = []
    if reject_all_lengths_equal and record["all_lengths_equal"]:
        failures.append("all_lengths_equal")
    if reject_all_angles_90 and record["all_angles_90"]:
        failures.append("all_angles_90")
    if (
        max_high_symmetry_coord_fraction is not None
        and float(record["high_symmetry_coord_fraction"]) > float(max_high_symmetry_coord_fraction)
    ):
        failures.append("high_symmetry_coord_fraction")
    return failures


def summarize_geometry(arrays_list: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    records = [geometry_degeneracy_record(arrays) for arrays in arrays_list]
    count = len(records)
    top_formula = Counter(str(record["formula"]) for record in records).most_common(20)
    return {
        "count": count,
        "all_lengths_equal_count": sum(1 for record in records if record["all_lengths_equal"]),
        "all_lengths_equal_rate": (
            sum(1 for record in records if record["all_lengths_equal"]) / max(1, count)
        ),
        "all_angles_90_count": sum(1 for record in records if record["all_angles_90"]),
        "all_angles_90_rate": sum(1 for record in records if record["all_angles_90"]) / max(1, count),
        "high_symmetry_coord_fraction_mean": (
            sum(float(record["high_symmetry_coord_fraction"]) for record in records) / max(1, count)
        ),
        "high_symmetry_coord_component_fraction_mean": (
            sum(float(record["high_symmetry_coord_component_fraction"]) for record in records)
            / max(1, count)
        ),
        "pbc_equivalent_duplicate_count": sum(
            int(record["pbc_equivalent_duplicate_site_count"]) for record in records
        ),
        "top_formula": top_formula,
    }


def write_geometry_markdown(summary: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# geometry diagnostics",
            "",
            f"- count: `{summary.get('count', 0)}`",
            f"- a=b=c count: `{summary.get('all_lengths_equal_count', 0)}`",
            f"- a=b=c rate: `{float(summary.get('all_lengths_equal_rate', 0.0)):.4f}`",
            f"- all-90 count: `{summary.get('all_angles_90_count', 0)}`",
            f"- all-90 rate: `{float(summary.get('all_angles_90_rate', 0.0)):.4f}`",
            f"- high_sym_coord_mean: `{float(summary.get('high_symmetry_coord_fraction_mean', 0.0)):.4f}`",
            f"- high_sym_component_mean: `{float(summary.get('high_symmetry_coord_component_fraction_mean', 0.0)):.4f}`",
            f"- PBC-equivalent duplicate sites: `{summary.get('pbc_equivalent_duplicate_count', 0)}`",
            "",
            "## top formulas",
            "",
            "```json",
            str(summary.get("top_formula", [])),
            "```",
            "",
        ]
    )

