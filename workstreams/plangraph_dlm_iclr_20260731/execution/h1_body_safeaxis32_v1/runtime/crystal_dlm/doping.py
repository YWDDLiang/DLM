"""Compact conditional-doping utilities.

The compact doping task is intentionally narrower than MP-20 fixed-slot
structure generation.  A model emits a dopant combination plus canonical
B-site indices, and deterministic post-processing reconstructs the full
80-atom template structure.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from crystal_dlm.fixed_slot import FixedSlotConfig, parse_fixed_slot_answer

DOPANT_SYMBOLS: List[str] = [
    "Al",
    "Ba",
    "Ca",
    "Cd",
    "Co",
    "Cu",
    "Fe",
    "In",
    "Mg",
    "Ni",
    "Sn",
    "Sr",
    "Zn",
]
DOPANT_SET = set(DOPANT_SYMBOLS)
CANONICAL_BSITE_INDICES: Tuple[int, int, int] = (0, 1, 14)
COMPACT_TASK_TOKEN = "<DOPING_COMPACT>"
BG_TOKENS = ("<BG_LOW>", "<BG_TARGET>", "<BG_HIGH>")
FE_TOKENS = ("<FE_Q1>", "<FE_Q2>", "<FE_Q3>", "<FE_Q4>")
DFE_TOKENS = ("<DFE_Q1>", "<DFE_Q2>", "<DFE_Q3>", "<DFE_Q4>")
DIRECTED_PROMPT = "<DOPING_COMPACT> <BG_TARGET> <FE_Q1> <DFE_Q4>"

PAIR_FEATURES: List[Tuple[str, str]] = list(itertools.combinations(DOPANT_SYMBOLS, 2))
COMPACT_TOKEN_RE = re.compile(r"<DOPANT_[A-Za-z]{1,2}>|<B\d{2}>")


class DopingFormatError(ValueError):
    """Raised when a compact doping answer is invalid."""


@dataclass(frozen=True)
class PropertyBins:
    """Quartile cut points for compact property prompts."""

    formation_energy: Tuple[float, float, float]
    defect_formation_energy: Tuple[float, float, float]

    def to_dict(self) -> Dict[str, List[float]]:
        return {
            "formation_energy": list(self.formation_energy),
            "defect_formation_energy": list(self.defect_formation_energy),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PropertyBins":
        return cls(
            formation_energy=tuple(float(x) for x in payload["formation_energy"]),
            defect_formation_energy=tuple(float(x) for x in payload["defect_formation_energy"]),
        )


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def normalize_combo(items: Iterable[str]) -> Tuple[str, str, str]:
    combo = tuple(sorted(str(item) for item in items))
    if len(combo) != 3:
        raise DopingFormatError(f"Expected 3 dopants, got {combo}")
    if len(set(combo)) != 3:
        raise DopingFormatError(f"Dopants must be unique, got {combo}")
    unknown = [item for item in combo if item not in DOPANT_SET]
    if unknown:
        raise DopingFormatError(f"Unknown dopant symbols: {unknown}")
    return combo  # type: ignore[return-value]


def combo_name(dopants: Sequence[str]) -> str:
    return "_".join(item.lower() for item in normalize_combo(dopants))


def combo_from_name(name: str) -> Tuple[str, str, str]:
    tokens = [item for item in str(name).replace("-", "_").split("_") if item]
    canonical = []
    by_lower = {item.lower(): item for item in DOPANT_SYMBOLS}
    for token in tokens:
        lowered = token.lower()
        if lowered in by_lower:
            canonical.append(by_lower[lowered])
    return normalize_combo(canonical)


def all_candidate_combos() -> List[Tuple[str, str, str]]:
    return [tuple(combo) for combo in itertools.combinations(DOPANT_SYMBOLS, 3)]


def compact_special_tokens() -> List[str]:
    tokens = [f"<DOPANT_{symbol}>" for symbol in DOPANT_SYMBOLS]
    tokens.extend(f"<B{idx:02d}>" for idx in range(16))
    tokens.extend([COMPACT_TASK_TOKEN, *BG_TOKENS, *FE_TOKENS, *DFE_TOKENS])
    return tokens


def compact_answer(dopants: Sequence[str], sites: Sequence[int] = CANONICAL_BSITE_INDICES) -> str:
    combo = normalize_combo(dopants)
    site_tuple = tuple(int(idx) for idx in sites)
    if site_tuple != CANONICAL_BSITE_INDICES:
        raise DopingFormatError(
            f"Only canonical B-sites {CANONICAL_BSITE_INDICES} are supported, got {site_tuple}"
        )
    return " ".join([f"<DOPANT_{symbol}>" for symbol in combo] + [f"<B{idx:02d}>" for idx in site_tuple])


def parse_compact_answer(text: str) -> Dict[str, Any]:
    tokens = COMPACT_TOKEN_RE.findall(text)
    if len(tokens) != 6:
        raise DopingFormatError(f"Expected 6 compact tokens, got {len(tokens)}: {tokens}")
    dopant_tokens = tokens[:3]
    site_tokens = tokens[3:]
    dopants: List[str] = []
    for token in dopant_tokens:
        if not token.startswith("<DOPANT_"):
            raise DopingFormatError(f"Expected dopant token, got {token}")
        symbol = token[len("<DOPANT_") : -1]
        dopants.append(symbol)
    combo = normalize_combo(dopants)
    if list(combo) != dopants:
        raise DopingFormatError(f"Dopants must be sorted, got {dopants}")
    sites: List[int] = []
    for token in site_tokens:
        if not re.fullmatch(r"<B\d{2}>", token):
            raise DopingFormatError(f"Expected B-site token, got {token}")
        sites.append(int(token[2:4]))
    if tuple(sites) != CANONICAL_BSITE_INDICES:
        raise DopingFormatError(f"Expected B-sites {CANONICAL_BSITE_INDICES}, got {sites}")
    return {
        "dopants": list(combo),
        "dopant_site_indices": list(sites),
        "answer": compact_answer(combo),
        "name": combo_name(combo),
    }


def _quantile(values: Sequence[float], fraction: float) -> float:
    sorted_values = sorted(float(value) for value in values)
    if not sorted_values:
        raise ValueError("Cannot compute quantile for empty values")
    idx = min(len(sorted_values) - 1, max(0, int(math.floor((len(sorted_values) - 1) * fraction))))
    return sorted_values[idx]


def property_bins_from_rows(rows: Sequence[Mapping[str, Any]]) -> PropertyBins:
    formation = [float(row["properties"]["formation_energy"]) for row in rows]
    defect = [float(row["properties"]["defect_formation_energy"]) for row in rows]
    return PropertyBins(
        formation_energy=(
            _quantile(formation, 0.25),
            _quantile(formation, 0.50),
            _quantile(formation, 0.75),
        ),
        defect_formation_energy=(
            _quantile(defect, 0.25),
            _quantile(defect, 0.50),
            _quantile(defect, 0.75),
        ),
    )


def band_gap_bin(value: float) -> str:
    if float(value) < 1.7:
        return "<BG_LOW>"
    if float(value) <= 2.7:
        return "<BG_TARGET>"
    return "<BG_HIGH>"


def quartile_bin(value: float, thresholds: Sequence[float], prefix: str) -> str:
    if len(thresholds) != 3:
        raise ValueError(f"Expected 3 thresholds, got {thresholds}")
    numeric = float(value)
    if numeric <= thresholds[0]:
        idx = 1
    elif numeric <= thresholds[1]:
        idx = 2
    elif numeric <= thresholds[2]:
        idx = 3
    else:
        idx = 4
    return f"<{prefix}_Q{idx}>"


def prompt_for_properties(properties: Mapping[str, Any], bins: PropertyBins) -> str:
    return " ".join(
        [
            COMPACT_TASK_TOKEN,
            band_gap_bin(float(properties["band_gap"])),
            quartile_bin(float(properties["formation_energy"]), bins.formation_energy, "FE"),
            quartile_bin(
                float(properties["defect_formation_energy"]),
                bins.defect_formation_energy,
                "DFE",
            ),
        ]
    )


def feature_vector(dopants: Sequence[str]) -> List[float]:
    combo = set(normalize_combo(dopants))
    features = [1.0 if symbol in combo else 0.0 for symbol in DOPANT_SYMBOLS]
    features.extend(1.0 if a in combo and b in combo else 0.0 for a, b in PAIR_FEATURES)
    return features


def ranking_tuple(candidate: Mapping[str, Any]) -> Tuple[float, float, float, float, float]:
    return (
        float(candidate.get("p_success", 0.0)),
        float(candidate.get("p_band_gap_target", 0.0)),
        -float(candidate.get("pred_formation_energy", 0.0)),
        float(candidate.get("pred_defect_formation_energy", 0.0)),
        -float(candidate.get("uncertainty", 0.0)),
    )


def objective_tuple_from_properties(properties: Mapping[str, Any]) -> Tuple[int, float, float]:
    band_gap = float(properties["band_gap"])
    target = int(1.7 <= band_gap <= 2.7)
    return (
        target,
        -float(properties["formation_energy"]),
        float(properties["defect_formation_energy"]),
    )


def load_first_full80_template(full80_jsonl: Path) -> Dict[str, Any]:
    rows = read_jsonl(full80_jsonl)
    if not rows:
        raise ValueError(f"No full80 rows found in {full80_jsonl}")
    config = FixedSlotConfig(max_atoms=80)
    arrays = parse_fixed_slot_answer(rows[0]["answer"], config=config, strict=True)
    bsite_structure_indices = [
        idx for idx, symbol in enumerate(arrays["species"]) if symbol not in {"Cs", "I"}
    ]
    if len(bsite_structure_indices) != 16:
        raise ValueError(f"Expected 16 B-sites in template, got {len(bsite_structure_indices)}")
    return {
        "arrays": arrays,
        "bsite_structure_indices": bsite_structure_indices,
        "source_name": rows[0].get("metadata", {}).get("name"),
    }


def reconstruct_arrays_from_template(
    template: Mapping[str, Any],
    dopants: Sequence[str],
    sites: Sequence[int] = CANONICAL_BSITE_INDICES,
) -> Dict[str, Any]:
    parsed = parse_compact_answer(compact_answer(dopants, sites))
    arrays = dict(template["arrays"])
    arrays["species"] = list(arrays["species"])
    bsite_structure_indices = list(template["bsite_structure_indices"])
    for dopant, site_idx in zip(parsed["dopants"], parsed["dopant_site_indices"]):
        arrays["species"][bsite_structure_indices[int(site_idx)]] = dopant
    arrays["answer"] = parsed["answer"]
    arrays["dopants"] = parsed["dopants"]
    arrays["dopant_site_indices"] = parsed["dopant_site_indices"]
    return arrays


def arrays_to_cif_text(arrays: Mapping[str, Any], data_name: str = "doped_candidate") -> str:
    lengths = [float(value) for value in arrays["lengths"]]
    angles = [float(value) for value in arrays["angles"]]
    species = list(arrays["species"])
    coords = list(arrays["frac_coords"])
    lines = [
        f"data_{data_name}",
        "_symmetry_space_group_name_H-M   'P 1'",
        "_symmetry_Int_Tables_number      1",
        "_cell_length_a   {:.6f}".format(lengths[0]),
        "_cell_length_b   {:.6f}".format(lengths[1]),
        "_cell_length_c   {:.6f}".format(lengths[2]),
        "_cell_angle_alpha   {:.6f}".format(angles[0]),
        "_cell_angle_beta    {:.6f}".format(angles[1]),
        "_cell_angle_gamma   {:.6f}".format(angles[2]),
        "loop_",
        "_atom_site_type_symbol",
        "_atom_site_label",
        "_atom_site_fract_x",
        "_atom_site_fract_y",
        "_atom_site_fract_z",
        "_atom_site_occupancy",
    ]
    counts: Dict[str, int] = {}
    for symbol, coord in zip(species, coords):
        counts[symbol] = counts.get(symbol, 0) + 1
        label = f"{symbol}{counts[symbol]}"
        lines.append(
            "{} {} {:.6f} {:.6f} {:.6f} 1.0".format(
                symbol,
                label,
                float(coord[0]) % 1.0,
                float(coord[1]) % 1.0,
                float(coord[2]) % 1.0,
            )
        )
    return "\n".join(lines) + "\n"


def status_priority(statuses: Iterable[str]) -> str:
    status_set = {str(status) for status in statuses if status is not None}
    if "SUCCESS" in status_set:
        return "SUCCESS"
    if "FAIL" in status_set:
        return "FAIL"
    if "NOT CAL" in status_set:
        return "NOT CAL"
    return "UNKNOWN"
