"""Structure-aware doping utilities.

The compact doping task only emits a dopant combination.  This module adds two
structure-aware representations used by the validation experiment:

* ``DOPING_STRUCT20``: a 107-token compressed structural code.  It is not a
  physical 20-atom crystal.  The first 16 slots encode B-site species and
  fractional coordinates from the 80-atom supercell; the last 4 slots encode
  simple global centroids for Cs/I/Pb/Mn.  The code expands back to a full
  80-atom CIF by combining generated lattice and B-site slots with a framework
  template for Cs/I sites.
* ``DOPING_FULL80``: the normal fixed-slot schema with ``max_atoms=80``.
"""

from __future__ import annotations

from collections import Counter
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from crystal_dlm.doping import (
    DFE_TOKENS,
    FE_TOKENS,
    BG_TOKENS,
    DOPANT_SET,
    DOPANT_SYMBOLS,
    combo_name,
)
from crystal_dlm.fixed_slot import (
    FixedSlotConfig,
    SYMBOL_TO_Z,
    arrays_to_answer,
    parse_fixed_slot_answer,
)

STRUCT20_TASK_TOKEN = "<DOPING_STRUCT20>"
FULL80_TASK_TOKEN = "<DOPING_FULL80>"
MP20_TASK_TOKEN = "<MP20_FULL>"
STRUCT20_DIRECTED_PROMPT = f"{STRUCT20_TASK_TOKEN} <BG_TARGET> <FE_Q1> <DFE_Q4>"
FULL80_DIRECTED_PROMPT = f"{FULL80_TASK_TOKEN} <BG_TARGET> <FE_Q1> <DFE_Q4>"

STRUCT20_CONFIG = FixedSlotConfig(max_atoms=20)
FULL80_CONFIG = FixedSlotConfig(max_atoms=80)
EXPECTED_FULL80_COUNTS = {"Cs": 16, "I": 48, "Pb": 12, "Mn": 1}
B_SITE_COUNT = 16
STRUCT20_GLOBAL_SPECIES = ["Cs", "I", "Pb", "Mn"]


class DopingStructureError(ValueError):
    """Raised when a structure-aware doping artifact is invalid."""


def doping_structure_task_tokens() -> List[str]:
    return [STRUCT20_TASK_TOKEN, FULL80_TASK_TOKEN, MP20_TASK_TOKEN, *BG_TOKENS, *FE_TOKENS, *DFE_TOKENS]


def prompt_for_task(base_prompt: str, task_token: str) -> str:
    """Replace the compact task token in a property prompt with a structure task."""

    if "<DOPING_COMPACT>" in base_prompt:
        return base_prompt.replace("<DOPING_COMPACT>", task_token, 1)
    return f"{task_token} " + " ".join(base_prompt.split()[1:])


def bsite_structure_indices(arrays: Mapping[str, Any]) -> List[int]:
    indices = [idx for idx, symbol in enumerate(arrays["species"]) if symbol not in {"Cs", "I"}]
    if len(indices) != B_SITE_COUNT:
        raise DopingStructureError(f"Expected 16 B-sites, got {len(indices)}")
    return indices


def composition_counts(species: Iterable[str]) -> Dict[str, int]:
    return dict(Counter(str(symbol) for symbol in species))


def full80_composition_is_exact(arrays: Mapping[str, Any]) -> bool:
    counts = Counter(str(symbol) for symbol in arrays["species"])
    if counts.get("Cs", 0) != 16 or counts.get("I", 0) != 48:
        return False
    if counts.get("Pb", 0) != 12 or counts.get("Mn", 0) != 1:
        return False
    dopant_counts = {symbol: count for symbol, count in counts.items() if symbol in DOPANT_SET}
    return len(dopant_counts) == 3 and all(count == 1 for count in dopant_counts.values())


def dopants_from_arrays(arrays: Mapping[str, Any]) -> List[str]:
    dopants = sorted(symbol for symbol in arrays["species"] if symbol in DOPANT_SET)
    if len(dopants) != 3 or len(set(dopants)) != 3:
        raise DopingStructureError(f"Expected exactly 3 unique dopants, got {dopants}")
    return dopants


def bsite_dopants_from_arrays(arrays: Mapping[str, Any]) -> List[str]:
    indices = bsite_structure_indices(arrays)
    dopants = sorted(
        str(arrays["species"][idx])
        for idx in indices
        if str(arrays["species"][idx]) in DOPANT_SET
    )
    if len(dopants) != 3 or len(set(dopants)) != 3:
        raise DopingStructureError(f"Expected exactly 3 unique B-site dopants, got {dopants}")
    return dopants


def _mean_coord(coords: Sequence[Sequence[float]]) -> List[float]:
    if not coords:
        return [0.0, 0.0, 0.0]
    return [
        sum(float(coord[axis]) for coord in coords) / len(coords) % 1.0
        for axis in range(3)
    ]


def compress_full80_arrays(arrays: Mapping[str, Any]) -> Dict[str, Any]:
    """Convert a full80 fixed-slot array payload to the 20-slot code arrays."""

    if int(arrays["num_atoms"]) != 80:
        raise DopingStructureError(f"Expected 80 atoms, got {arrays['num_atoms']}")
    if not full80_composition_is_exact(arrays):
        raise DopingStructureError(f"Unexpected full80 composition: {composition_counts(arrays['species'])}")
    bsite_indices = bsite_structure_indices(arrays)
    species20 = [str(arrays["species"][idx]) for idx in bsite_indices]
    coords20 = [[float(value) for value in arrays["frac_coords"][idx]] for idx in bsite_indices]

    by_species: Dict[str, List[Sequence[float]]] = {symbol: [] for symbol in STRUCT20_GLOBAL_SPECIES}
    for symbol, coord in zip(arrays["species"], arrays["frac_coords"]):
        if symbol in by_species:
            by_species[str(symbol)].append(coord)
    for symbol in STRUCT20_GLOBAL_SPECIES:
        species20.append(symbol)
        coords20.append(_mean_coord(by_species[symbol]))

    answer, diagnostics = arrays_to_answer(
        lengths=arrays["lengths"],
        angles=arrays["angles"],
        species=species20,
        frac_coords=coords20,
        config=STRUCT20_CONFIG,
        separator=" ",
    )
    parsed = parse_fixed_slot_answer(answer, config=STRUCT20_CONFIG, strict=True)
    parsed["compress_diagnostics"] = diagnostics.to_dict()
    parsed["bsite_source_indices"] = bsite_indices
    return parsed


def parse_structure20_answer(text: str, strict: bool = False) -> Dict[str, Any]:
    arrays = parse_fixed_slot_answer(text, config=STRUCT20_CONFIG, strict=strict)
    if int(arrays["num_atoms"]) != 20:
        raise DopingStructureError(f"Expected <N_020>, got num_atoms={arrays['num_atoms']}")
    bsite_species = [str(symbol) for symbol in arrays["species"][:B_SITE_COUNT]]
    counts = Counter(bsite_species)
    dopant_counts = {symbol: count for symbol, count in counts.items() if symbol in DOPANT_SET}
    if counts.get("Pb", 0) != 12 or counts.get("Mn", 0) != 1:
        raise DopingStructureError(f"Invalid B-site Pb/Mn counts: {dict(counts)}")
    if len(dopant_counts) != 3 or any(count != 1 for count in dopant_counts.values()):
        raise DopingStructureError(f"Invalid B-site dopants: {dopant_counts}")
    return arrays


def expand_structure20_arrays(
    compressed: Mapping[str, Any],
    template_full80: Mapping[str, Any],
) -> Dict[str, Any]:
    """Expand a structure20 code into full80 arrays using a framework template."""

    compressed = parse_structure20_answer(compressed["answer"], strict=True)
    if int(template_full80["num_atoms"]) != 80:
        raise DopingStructureError("Template must contain 80 atoms")
    bsite_indices = bsite_structure_indices(template_full80)

    species = [str(symbol) for symbol in template_full80["species"]]
    coords = [[float(value) for value in coord] for coord in template_full80["frac_coords"]]
    for local_idx, structure_idx in enumerate(bsite_indices):
        species[structure_idx] = str(compressed["species"][local_idx])
        coords[structure_idx] = [float(value) % 1.0 for value in compressed["frac_coords"][local_idx]]

    atom_types = [SYMBOL_TO_Z[str(symbol)] for symbol in species]
    expanded = {
        "num_atoms": 80,
        "lengths": [float(value) for value in compressed["lengths"]],
        "angles": [float(value) for value in compressed["angles"]],
        "species": species,
        "atom_types": atom_types,
        "frac_coords": coords,
        "bsite_structure_indices": bsite_indices,
        "source": "structure20_expanded",
    }
    if not full80_composition_is_exact(expanded):
        raise DopingStructureError(f"Expanded composition mismatch: {composition_counts(species)}")
    return expanded


def parse_full80_answer(text: str, strict: bool = False) -> Dict[str, Any]:
    arrays = parse_fixed_slot_answer(text, config=FULL80_CONFIG, strict=strict)
    if int(arrays["num_atoms"]) != 80:
        raise DopingStructureError(f"Expected <N_080>, got num_atoms={arrays['num_atoms']}")
    if not full80_composition_is_exact(arrays):
        raise DopingStructureError(f"Full80 composition mismatch: {composition_counts(arrays['species'])}")
    return arrays


def coord_rmsd(coords_a: Sequence[Sequence[float]], coords_b: Sequence[Sequence[float]]) -> float:
    if len(coords_a) != len(coords_b):
        raise DopingStructureError("Coordinate lengths do not match")
    if not coords_a:
        return 0.0
    total = 0.0
    for a, b in zip(coords_a, coords_b):
        for axis in range(3):
            delta = abs(float(a[axis]) - float(b[axis]))
            delta = min(delta, 1.0 - delta)
            total += delta * delta
    return math.sqrt(total / (len(coords_a) * 3))


def bsite_rmsd(arrays_a: Mapping[str, Any], arrays_b: Mapping[str, Any]) -> float:
    idx_a = bsite_structure_indices(arrays_a)
    idx_b = bsite_structure_indices(arrays_b)
    coords_a = [arrays_a["frac_coords"][idx] for idx in idx_a]
    coords_b = [arrays_b["frac_coords"][idx] for idx in idx_b]
    return coord_rmsd(coords_a, coords_b)


def lattice_differences(arrays_a: Mapping[str, Any], arrays_b: Mapping[str, Any]) -> Dict[str, float]:
    length_terms = []
    for a, b in zip(arrays_a["lengths"], arrays_b["lengths"]):
        denom = max(abs(float(b)), 1e-8)
        length_terms.append(abs(float(a) - float(b)) / denom)
    angle_terms = [abs(float(a) - float(b)) for a, b in zip(arrays_a["angles"], arrays_b["angles"])]
    return {
        "length_rel_mean": sum(length_terms) / len(length_terms),
        "angle_abs_mean": sum(angle_terms) / len(angle_terms),
    }


def structure_matcher_pass(arrays_a: Mapping[str, Any], arrays_b: Mapping[str, Any]) -> bool | None:
    try:
        from pymatgen.analysis.structure_matcher import StructureMatcher
        from crystal_dlm.fixed_slot import arrays_to_structure
    except Exception:
        return None
    try:
        matcher = StructureMatcher(ltol=0.3, stol=0.3, angle_tol=5)
        return bool(matcher.fit(arrays_to_structure(arrays_a), arrays_to_structure(arrays_b)))
    except Exception:
        return False


def near_hit(
    generated: Mapping[str, Any],
    target: Mapping[str, Any],
    length_rel_threshold: float = 0.03,
    angle_abs_threshold: float = 3.0,
    bsite_rmsd_threshold: float = 0.08,
) -> Dict[str, Any]:
    same_elements = sorted(dopants_from_arrays(generated)) == sorted(dopants_from_arrays(target))
    diffs = lattice_differences(generated, target)
    rmsd = bsite_rmsd(generated, target)
    matcher = structure_matcher_pass(generated, target)
    threshold_pass = (
        same_elements
        and diffs["length_rel_mean"] <= length_rel_threshold
        and diffs["angle_abs_mean"] <= angle_abs_threshold
        and rmsd <= bsite_rmsd_threshold
    )
    return {
        "same_elements": same_elements,
        "length_rel_mean": diffs["length_rel_mean"],
        "angle_abs_mean": diffs["angle_abs_mean"],
        "bsite_rmsd": rmsd,
        "structure_matcher_pass": matcher,
        "near_hit": bool(same_elements and (matcher is True or threshold_pass)),
    }


def load_full80_template(full80_jsonl: Path, exclude_names: Iterable[str] = ()) -> Dict[str, Any]:
    exclude = set(exclude_names)
    with full80_jsonl.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = __import__("json").loads(line)
            name = row.get("metadata", {}).get("name")
            if name in exclude:
                continue
            return parse_full80_answer(row["answer"], strict=True)
    raise DopingStructureError(f"No usable full80 template found in {full80_jsonl}")
