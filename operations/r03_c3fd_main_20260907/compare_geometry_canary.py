#!/usr/bin/env python3
"""Read-only comparison of exported geometry-canary I/G/P artifacts.

Expected filenames follow export_geometry_canary_records.py.  The input
directory must also include SOURCE_RECEIPTS.json containing the exported
source paths, byte counts and hashes.  No model, relaxation, chemistry
validator, network or sampling is called.  Optional output files are analysis
reports only; all source artifacts remain untouched.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import itertools
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

SCHEMA = "r03_matched_geometry_canary_comparison_v1"
DIRECT_FUNCTION_SOURCE_SHA256 = "68e6d0a9703f412cfd3215e6d0ae687e5b153e941d16f3fa4f2fffeedb505cb6"
ROLES = ("I_batch1_reference", "G", "P")
DISTANCE_CUTOFF = 0.5
DISTANCE_COMPARISON_TOLERANCE = 1e-8


def file_identity(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    return {"path": str(path.resolve()), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return result


def read_rows(path: Path, *, expected: int) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != expected or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"all-request denominator changed in {path}; expected {expected}")
    if [row.get("ordinal") for row in rows] != list(range(expected)):
        raise ValueError(f"request order/ordinal coverage changed in {path}")
    identifiers = [row.get("sample_idx") for row in rows]
    if any(type(value) is not int or value < 0 for value in identifiers) or len(set(identifiers)) != expected:
        raise ValueError(f"global sample IDs are absent/duplicated in {path}")
    return rows


def receipt_index(path: Path) -> dict[str, Mapping[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict) and isinstance(value.get("files"), dict):
        value = value["files"]
    if isinstance(value, dict):
        pairs = list(value.items())
    elif isinstance(value, list):
        pairs = []
        for row in value:
            raw_name = row.get("name", row.get("local", row.get("local_path", "")))
            name = str(raw_name).replace("\\", "/").split("/")[-1]
            pairs.append((name, row))
    else:
        raise ValueError("SOURCE_RECEIPTS must be the exported name map or local receipt list")
    result = {}
    for name, row in pairs:
        if not isinstance(name, str) or not name or Path(name).name != name or name in result or not isinstance(row, Mapping):
            raise ValueError("source receipts have invalid or duplicate local names")
        result[name] = row
    return result


def required_filenames() -> list[str]:
    names = ["GEOMETRY_CANARY_COMPLETE.json"]
    for role in ROLES:
        names.extend(f"{role}_{suffix}" for suffix in ("body.jsonl", "native_direct.json", "tau800_direct.json", "component.json"))
        if role != "I_batch1_reference":
            names.append(f"{role}_construction.jsonl")
    return names


def verified_assets(directory: Path) -> dict[str, Any]:
    receipts_path = directory / "SOURCE_RECEIPTS.json"
    receipts = receipt_index(receipts_path)
    identities = {}
    for name in required_filenames():
        if name not in receipts:
            raise ValueError(f"export receipt absent for {name}; incomplete downloads are not model failures")
        actual = file_identity(directory / name)
        expected = receipts[name]
        if actual["bytes"] != expected.get("bytes") or actual["sha256"] != expected.get("sha256"):
            raise ValueError(f"exported bytes/SHA disagree for {name}")
        identities[name] = {**actual, "remote_source": expected.get("source"), "verified": True}
    return {"receipt_file": file_identity(receipts_path), "files": identities}


def unavailable_geometry(record: Mapping[str, Any], reason: str) -> dict[str, Any]:
    return {
        "available": False, "reason": reason, "requested_N": (record.get("plan_state") or {}).get("N"),
        "generated_N": None, "construction_complete": record.get("body_generation_complete") is True,
        "attempt_status": record.get("attempt_status"), "pymatgen_direct_structure_valid": False,
        "recorded_construction_failure": (record.get("construction_geometry") or {}).get("failure"),
        "pymatgen_minimum_pair_distance_A": None, "shell125_minimum_pair_distance_A": None,
        "shell125_direct_style_valid": False, "shell125_intersite_supported": False,
        "shell125_repair_style_supported": False, "distance_conventions_disagree": None,
        "counted_in_full_denominator": True,
    }


def geometry_from_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Reconstruct the same native input as Direct and compare two PBC searches."""
    from scripts.export_r03_evaluation_inputs import native_structure, source_native_failure
    reason = source_native_failure(record)
    if reason is not None:
        return unavailable_geometry(record, reason)
    # Import failures must stop the analysis, never count as geometry-invalid.
    import numpy as np
    from pymatgen.core import Structure  # noqa: F401 - explicit environment requirement

    try:
        structure, _text, arrays = native_structure(record)
    except (ValueError, TypeError, KeyError, ArithmeticError) as error:
        return unavailable_geometry(record, f"native_reconstruction:{type(error).__name__}:{error}")
    n = len(structure)
    distances = np.asarray(structure.distance_matrix, dtype=float)
    volume = float(structure.volume)
    if n < 1 or distances.shape != (n, n) or not np.isfinite(distances).all() or not math.isfinite(volume):
        return unavailable_geometry(record, "nonfinite_or_empty_native_geometry")
    # This is the exact frozen Direct structure_validity predicate, including
    # its diagonal sentinel and minimum volume. Chemistry is not re-evaluated.
    direct_matrix = distances + np.diag(np.ones(n) * (DISTANCE_CUTOFF + 10.0))
    direct_valid = bool(direct_matrix.min() >= DISTANCE_CUTOFF and volume >= 0.1)
    off_diagonal = ~np.eye(n, dtype=bool)
    pmat_min = float(distances[off_diagonal].min()) if n > 1 else None
    shifts = np.asarray(list(itertools.product(range(-2, 3), repeat=3)), dtype=float)
    frac = np.asarray(structure.frac_coords, dtype=float) % 1.0
    lattice = np.asarray(structure.lattice.matrix, dtype=float)
    shell_distances = np.linalg.norm((frac[:, None, None, :] - frac[None, :, None, :] + shifts) @ lattice, axis=-1).min(axis=2)
    shell_min = float(shell_distances[off_diagonal].min()) if n > 1 else None
    self_min = float(np.linalg.norm(shifts[np.any(shifts != 0, axis=1)] @ lattice, axis=-1).min())
    angles = list(map(float, structure.lattice.angles))
    ca, cb, cg = (math.cos(math.radians(angle)) for angle in angles)
    radicand = 1 + 2 * ca * cb * cg - ca * ca - cb * cb - cg * cg
    intersite = shell_min is None or shell_min >= DISTANCE_CUTOFF
    shell_direct = bool(intersite and volume >= 0.1)
    shell_repair = bool(intersite and self_min >= DISTANCE_CUTOFF and radicand > 1e-4)
    pairs = []
    for left in range(n):
        for right in range(left + 1, n):
            if distances[left, right] < DISTANCE_CUTOFF or shell_distances[left, right] < DISTANCE_CUTOFF:
                pairs.append({"left": left, "right": right, "species_left": str(structure[left].specie),
                              "species_right": str(structure[right].specie),
                              "pymatgen_distance_A": float(distances[left, right]),
                              "shell125_distance_A": float(shell_distances[left, right])})
    maximum_delta = float(np.abs(distances[off_diagonal] - shell_distances[off_diagonal]).max()) if n > 1 else 0.0
    return {
        "available": True, "reason": None, "requested_N": record["plan_state"]["N"], "generated_N": n,
        "construction_complete": True, "attempt_status": record.get("attempt_status"),
        "lengths_A": list(map(float, structure.lattice.abc)), "angles_deg": angles,
        "volume_A3": volume, "volume_per_atom_A3": volume / n,
        "pymatgen_direct_structure_valid": direct_valid, "pymatgen_minimum_pair_distance_A": pmat_min,
        "pymatgen_direct_padded_matrix_minimum_A": float(direct_matrix.min()),
        "shell125_minimum_pair_distance_A": shell_min, "shell125_minimum_self_image_A": self_min,
        "shell125_intersite_supported": bool(intersite), "shell125_direct_style_valid": shell_direct,
        "shell125_repair_style_supported": shell_repair,
        "maximum_off_diagonal_distance_difference_A": maximum_delta,
        "distance_conventions_disagree": maximum_delta > DISTANCE_COMPARISON_TOLERANCE or direct_valid != shell_direct,
        "direct_vs_repair_rule_disagree": direct_valid != shell_repair,
        "pairs_below_cutoff_under_either_search": pairs,
        "counted_in_full_denominator": True,
    }


def logged_support_comparison(measured: Mapping[str, Any], logged: Any) -> dict[str, Any]:
    if not isinstance(logged, Mapping):
        return {"reported": False, "agrees": None}
    comparison = {"reported": True, "original_report": dict(logged), "agrees": None}
    if measured["available"]:
        flag_equal = logged.get("supported") == measured["shell125_repair_style_supported"]
        old = logged.get("minimum_distance_A")
        new = measured["shell125_minimum_pair_distance_A"]
        distance_equal = (old is None and new is None) or (old is not None and new is not None
            and abs(float(old) - float(new)) <= DISTANCE_COMPARISON_TOLERANCE)
        # Fixed-cell/self-image failures return before a minimum pair distance.
        if old is None and logged.get("reason") not in (None, "native_pair_below_0.5A"):
            distance_equal = True
        comparison.update(agrees=flag_equal and distance_equal, support_flag_agrees=flag_equal,
                          minimum_distance_agrees=distance_equal)
    return comparison


def validate_direct_report(report: Mapping[str, Any], expected: int) -> dict[str, Any]:
    if report.get("schema") != "crysllmgen_direct_validity_fast_v1" or report.get("attempts") != expected:
        raise ValueError("saved Direct report schema or all-request denominator differs")
    keys = ("generation_succeeded", "comp_valid_count", "struct_valid_count", "valid_count")
    if any(type(report.get(key)) is not int or not 0 <= report[key] <= expected for key in keys):
        raise ValueError("saved Direct report contains invalid counts")
    if report["valid_count"] > min(report["comp_valid_count"], report["struct_valid_count"]):
        raise ValueError("saved joint validity exceeds one of its components")
    return {"denominator": expected, "C": report["comp_valid_count"], "S": report["struct_valid_count"],
            "J": report["valid_count"], "generation_succeeded": report["generation_succeeded"],
            "source": "saved_frozen_Direct_report_not_recomputed_chemistry"}


def complete_tokens(record: Mapping[str, Any]) -> list[int] | None:
    tokens = record.get("raw_body_token_ids")
    return list(tokens) if record.get("body_generation_complete") is True and isinstance(tokens, list) and tokens else None


def partial_tokens(record: Mapping[str, Any]) -> Any:
    geometry = record.get("construction_geometry")
    failure = geometry.get("failure") if isinstance(geometry, Mapping) else None
    value = failure.get("partial_body_token_ids") if isinstance(failure, Mapping) else None
    return value if isinstance(value, list) and value else None


def explicit_equality(left: Any, right: Any, expected_type: type) -> bool | None:
    if left is None and right is None:
        return None
    if not isinstance(left, expected_type) or not isinstance(right, expected_type):
        return False
    if expected_type is int and (type(left) is not int or type(right) is not int):
        return False
    return left == right


def token_change_report(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    left, right = complete_tokens(before), complete_tokens(after)
    if left is None or right is None:
        return {"comparable_complete_bodies": False, "tokens_equal": None,
                "before_complete": left is not None, "after_complete": right is not None}
    if len(left) != len(right):
        return {"comparable_complete_bodies": False, "tokens_equal": False, "cardinality_changed": True}
    changed = [position for position, (x, y) in enumerate(zip(left, right)) if x != y]
    frozen_positions = [0, *range(1, 7), *range(7, len(left), 4)]
    return {"comparable_complete_bodies": True, "tokens_equal": not changed,
            "changed_positions": changed, "changed_token_count": len(changed),
            "composition_tokens_equal": all(left[p] == right[p] for p in [0, *range(7, len(left), 4)]),
            "lattice_tokens_equal": left[1:7] == right[1:7],
            "noncoordinate_tokens_equal": all(left[p] == right[p] for p in frozen_positions)}


def summary_for(metrics: Sequence[Mapping[str, Any]], records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "denominator": len(records), "construction_complete": sum(row.get("body_generation_complete") is True for row in records),
        "construction_constraint_failures": sum(row.get("attempt_status") == "construction_constraint_failure" for row in records),
        "all_failure_statuses": dict(Counter(str(row.get("attempt_status")) for row in records if row.get("parsed") is not True)),
        "geometry_available": sum(row["available"] for row in metrics),
        "pymatgen_direct_S_geometry": sum(row["pymatgen_direct_structure_valid"] for row in metrics),
        "shell125_direct_style_geometry": sum(row["shell125_direct_style_valid"] for row in metrics),
        "shell125_repair_support": sum(row["shell125_repair_style_supported"] for row in metrics),
        "pymatgen_vs_shell125_disagreement_ids": [record["sample_idx"] for record, row in zip(records, metrics)
                                                 if row["distance_conventions_disagree"] is True],
        "unavailable_rows_stay_in_denominator": True,
    }


def observed_token_table(records: Sequence[Mapping[str, Any]]) -> tuple[dict[int, str], dict[str, Any]]:
    """Recover the registered added-token offset from actual exported tokens."""
    from crystal_dlm.fixed_slot import build_special_tokens
    special = build_special_tokens()
    ordinal = {token: index for index, token in enumerate(special)}
    offsets = set()
    observed = {}
    occurrences = 0
    for record in records:
        ids = complete_tokens(record)
        text = record.get("raw_body_text", record.get("text"))
        if ids is None or not isinstance(text, str):
            continue
        tokens = re.findall(r"<[^>]+>", text)
        if len(tokens) != len(ids):
            raise ValueError("complete body token IDs cannot be joined to their saved symbolic representation")
        for token, token_id in zip(tokens, ids):
            if token not in ordinal or (token_id in observed and observed[token_id] != token):
                raise ValueError("saved body token-ID ABI is inconsistent")
            offsets.add(token_id - ordinal[token])
            observed[token_id] = token
            occurrences += 1
    if not offsets:
        return {}, {"available": False, "reason": "no_complete_token_table_observation"}
    if len(offsets) != 1:
        raise ValueError("saved bodies disagree with the fixed original added-token order")
    offset = next(iter(offsets))
    return {offset + index: token for index, token in enumerate(special)}, {
        "available": True, "added_token_offset": offset, "observed_unique_ids": len(observed),
        "observed_occurrences": occurrences, "fixed_token_count": len(special),
        "mapping": "declared_fixed_slot_order_with_offset_checked_against_actual_complete_bodies",
    }


def partial_prefix_diagnosis(record: Mapping[str, Any], token_table: Mapping[int, str]) -> dict[str, Any] | None:
    """Enumerate geometric support under the recorded prefix, never resample.

    Zero support means that this committed lattice/XY/Z prefix is a dead end.
    It does not establish that the composition, another cell, or another reveal
    trajectory is infeasible. No hypothetical structure files are generated.
    """
    failure = (record.get("construction_geometry") or {}).get("failure")
    partial = partial_tokens(record)
    if not isinstance(failure, Mapping) or partial is None:
        return None
    result = {"failure_reason": failure.get("reason"), "original_failure": dict(failure),
              "composition_infeasibility_inferred": False,
              "scope": "only_the_recorded_committed_prefix_and_fixed_XY_geometric_support",
              "new_model_or_energy_calls": 0, "new_samples": 0}
    if not token_table:
        result.update(decoded=False, reason="token_mapping_unavailable", partial_body_token_ids=partial)
        return result
    if len(partial) != 1 or not isinstance(partial[0], list):
        raise ValueError("construction geometry failure must contain exactly one partial body row")
    from crystal_dlm.fixed_slot import MASK_TOKEN_ID
    from pymatgen.core import Lattice
    import numpy as np
    ids = partial[0]
    tokens = ["[MASK]" if value == MASK_TOKEN_ID else token_table.get(value, "[UNKNOWN]") for value in ids]
    n = (record.get("plan_state") or {}).get("N")
    if type(n) is not int or len(ids) != 7 + 4 * n or "[UNKNOWN]" in tokens:
        raise ValueError("partial construction token ABI/cardinality cannot be decoded reliably")
    def number(position, prefix, scale=1.0):
        token = tokens[position]
        if token == "[MASK]":
            return None
        match = re.fullmatch(r"<" + prefix + r"_(\d{3})>", token)
        if not match:
            raise ValueError("partial coordinate/lattice field has the wrong token type")
        return int(match.group(1)) * scale
    lengths = [number(position, axis, .1) for position, axis in ((1, "LA"), (2, "LB"), (3, "LC"))]
    angles = [number(position, axis) for position, axis in ((4, "AA"), (5, "AB"), (6, "AG"))]
    atoms = []
    for slot in range(n):
        species = re.fullmatch(r"<E_([A-Z][a-z]?)>", tokens[7 + 4 * slot])
        if not species:
            raise ValueError("partial construction lost an element prefill")
        bins = [number(8 + 4 * slot + component, axis) for component, axis in enumerate("XYZ")]
        frac = [None if value is None else (value % 100) / 100.0 for value in bins]
        atoms.append({"slot": slot, "species": species.group(1), "XYZ_tokens": tokens[8 + 4 * slot:11 + 4 * slot],
                      "fractional_XYZ": frac, "complete_XYZ": all(value is not None for value in frac)})
    result.update(decoded=True, partial_body_tokens=tokens, lengths_A=lengths, angles_deg=angles,
                  atoms=atoms, committed_sites=[row["slot"] for row in atoms if row["complete_XYZ"]],
                  incomplete_sites=[row["slot"] for row in atoms if not row["complete_XYZ"]])
    if any(value is None for value in lengths + angles) or any(row["fractional_XYZ"][component] is None for row in atoms for component in (0, 1)):
        result["remaining_Z_support_available"] = False
        result["remaining_Z_support_reason"] = "lattice_or_XY_not_fully_committed"
        return result
    try:
        lattice = Lattice.from_parameters(*lengths, *angles)
    except (ValueError, ArithmeticError) as error:
        result["remaining_Z_support_available"] = False
        result["remaining_Z_support_reason"] = f"invalid_fixed_lattice:{type(error).__name__}"
        return result
    if not math.isfinite(float(lattice.volume)) or lattice.volume <= 0:
        result["remaining_Z_support_available"] = False
        result["remaining_Z_support_reason"] = "invalid_fixed_lattice"
        return result
    committed = [row for row in atoms if row["complete_XYZ"]]
    visible_frac = np.asarray([row["fractional_XYZ"] for row in committed], dtype=float).reshape((-1, 3))
    shifts = np.asarray(list(itertools.product(range(-2, 3), repeat=3)), dtype=float)
    supports = []
    active_positions = set(failure.get("active_positions") or ())
    for atom in atoms:
        if atom["complete_XYZ"]:
            continue
        candidates = np.asarray([[atom["fractional_XYZ"][0], atom["fractional_XYZ"][1], z / 100.0] for z in range(100)])
        if committed:
            exact = np.asarray(lattice.get_all_distances(candidates, visible_frac))
            shell = np.linalg.norm((candidates[:, None, None, :] - visible_frac[None, :, None, :] + shifts) @ lattice.matrix, axis=-1).min(2)
            exact_min, shell_min = exact.min(1), shell.min(1)
            exact_legal = np.flatnonzero(exact_min >= DISTANCE_CUTOFF).tolist()
            shell_legal = np.flatnonzero(shell_min >= DISTANCE_CUTOFF).tolist()
            blocking_atoms = [row["slot"] for index, row in enumerate(committed) if bool((exact[:, index] < DISTANCE_CUTOFF).any())]
            best_minimum = float(exact_min.max())
        else:
            exact_legal = shell_legal = list(range(100))
            blocking_atoms, best_minimum = [], None
        supports.append({"slot": atom["slot"], "species": atom["species"], "body_Z_position": 10 + 4 * atom["slot"],
                         "in_active_failure_group": 10 + 4 * atom["slot"] in active_positions,
                         "pymatgen_legal_Z_bins": exact_legal, "shell125_legal_Z_bins": shell_legal,
                         "pymatgen_legal_Z_count": len(exact_legal), "shell125_legal_Z_count": len(shell_legal),
                         "support_searches_agree": exact_legal == shell_legal,
                         "committed_atoms_blocking_some_Z": blocking_atoms,
                         "best_achievable_minimum_distance_to_committed_sites_A": best_minimum})
    result.update(remaining_Z_support_available=True, remaining_Z_support=supports,
                  zero_support_sites=[row["slot"] for row in supports if row["pymatgen_legal_Z_count"] == 0],
                  remaining_masked_Z_interactions_are_not_assumed_satisfied=True,
                  conclusion="No legal Z means this fixed committed prefix cannot continue without revising earlier choices; it is not a composition-level infeasibility proof.")
    return result


def compare_assets(assets_dir: Path, *, expected_requests: int = 16) -> dict[str, Any]:
    if expected_requests < 1:
        raise ValueError("expected requests must be positive")
    sources = verified_assets(assets_dir)
    marker = read_json(assets_dir / "GEOMETRY_CANARY_COMPLETE.json")
    if marker.get("requests_per_arm") != expected_requests or marker.get("roles") != ["G", "P"]:
        raise ValueError("completed geometry canary does not match the declared cohort")
    import pymatgen.core  # noqa: F401
    import numpy
    bodies = {role: read_rows(assets_dir / f"{role}_body.jsonl", expected=expected_requests) for role in ROLES}
    construction = {role: read_rows(assets_dir / f"{role}_construction.jsonl", expected=expected_requests) for role in ("G", "P")}
    components = {role: read_json(assets_dir / f"{role}_component.json") for role in ROLES}
    direct = {role: {endpoint: validate_direct_report(read_json(assets_dir / f"{role}_{endpoint}_direct.json"), expected_requests)
                     for endpoint in ("native", "tau800")} for role in ROLES}
    violations = []
    for role, component in components.items():
        if component.get("requests") != expected_requests or component.get("body_batch_size") != 1:
            violations.append(f"{role}:component request count or singleton batch differs")
        if component.get("construction_geometry") is not (role != "I_batch1_reference"):
            violations.append(f"{role}:construction geometry flag differs from intended treatment")
    reference = bodies["I_batch1_reference"]
    token_table, token_mapping = observed_token_table([row for records in (*bodies.values(), *construction.values()) for row in records])
    identifiers = [row["sample_idx"] for row in reference]
    for label, records in {**bodies, **{role + "_construction": rows for role, rows in construction.items()}}.items():
        if [row["sample_idx"] for row in records] != identifiers:
            raise ValueError(f"global request identities differ for {label}")
    measured = {"I_native": [geometry_from_record(row) for row in reference]}
    for role in ("G", "P"):
        measured[role + "_construction"] = [geometry_from_record(row) for row in construction[role]]
        measured[role + "_native"] = [geometry_from_record(row) for row in bodies[role]]
    rows = []
    for ordinal, sample_idx in enumerate(identifiers):
        g, p, original = construction["G"][ordinal], construction["P"][ordinal], reference[ordinal]
        g_tokens, p_tokens = complete_tokens(g), complete_tokens(p)
        gp = {
            "Plans_equal": explicit_equality(g.get("plan_state"), p.get("plan_state"), dict),
            "body_noise_seeds_equal": explicit_equality(g.get("body_noise_seed"), p.get("body_noise_seed"), int),
            "body_prompts_equal": explicit_equality(g.get("body_prompt"), p.get("body_prompt"), str),
            "schedules_equal": explicit_equality(g.get("generation_position_groups"), p.get("generation_position_groups"), list),
            "construction_complete_flags_equal": (g_tokens is not None) == (p_tokens is not None),
            "both_record_original_B0_construction": g.get("body_checkpoint_arm") == p.get("body_checkpoint_arm") == "B0",
            "complete_construction_tokens_equal": g_tokens == p_tokens if g_tokens is not None and p_tokens is not None else None,
            "partial_failed_tokens_equal": partial_tokens(g) == partial_tokens(p)
                if partial_tokens(g) is not None and partial_tokens(p) is not None else None,
        }
        if any(value is False for value in gp.values()):
            violations.append(f"sample {sample_idx}:G/P construction pairing differs")
        item = {"sample_idx": sample_idx, "ordinal": ordinal, "I_formula": (original.get("plan_state") or {}).get("formula"),
                "geometry": {key: values[ordinal] for key, values in measured.items()}, "G_P_initial_pairing": gp,
                "construction_treatment_comparison": {}, "repair_only_comparison": {},
                "partial_construction_diagnosis": {role: partial_prefix_diagnosis(construction[role][ordinal], token_table) for role in ("G", "P")}}
        for role in ("G", "P"):
            initial, final = construction[role][ordinal], bodies[role][ordinal]
            same_condition = {
                "Plans_equal": explicit_equality(original.get("plan_state"), initial.get("plan_state"), dict),
                "body_noise_seeds_equal": explicit_equality(original.get("body_noise_seed"), initial.get("body_noise_seed"), int),
                "body_prompts_equal": explicit_equality(original.get("body_prompt"), initial.get("body_prompt"), str),
                "schedules_equal": explicit_equality(original.get("generation_position_groups"), initial.get("generation_position_groups"), list),
            }
            if any(value is False for value in same_condition.values()):
                violations.append(f"sample {sample_idx}:{role} construction is not matched to I")
            item["construction_treatment_comparison"][role] = {**same_condition, **token_change_report(original, initial)}
            trace = final.get("repair_trace") or {}
            native_pre = final.get("construction_raw_body_token_ids")
            artifact_match = native_pre == complete_tokens(initial) if complete_tokens(initial) is not None else native_pre is None
            if not artifact_match:
                violations.append(f"sample {sample_idx}:{role} repair input differs from saved construction")
            if complete_tokens(initial) is None and (final.get("repair_used") is True or trace):
                violations.append(f"sample {sample_idx}:{role} incomplete construction was repaired")
            item["repair_only_comparison"][role] = {
                **token_change_report(initial, final), "input_matches_saved_construction": artifact_match,
                "repair_used": final.get("repair_used"), "repair_skip_reason": final.get("repair_skip_reason"),
                "attempted_anchors": trace.get("attempted_anchors", 0), "committed_anchors": trace.get("committed_anchors", 0),
                "rolled_back_anchors": trace.get("rolled_back_anchors", 0),
                "rollback_reasons": dict(Counter(str(t.get("rollback_reason")) for t in trace.get("transactions", []) if not t.get("committed"))),
                "recorded125_before_vs_recomputed": logged_support_comparison(measured[role + "_construction"][ordinal], trace.get("geometry_before")),
                "recorded125_after_vs_recomputed": logged_support_comparison(measured[role + "_native"][ordinal], trace.get("geometry_after")),
            }
        rows.append(item)
    summaries = {"I_native": summary_for(measured["I_native"], reference)}
    for role in ("G", "P"):
        summaries[role + "_construction"] = summary_for(measured[role + "_construction"], construction[role])
        summaries[role + "_native"] = summary_for(measured[role + "_native"], bodies[role])
    saved_direct_agreement = {role: {
        "reported_native_S": direct[role]["native"]["S"],
        "local_pymatgen_native_S": summaries[("I" if role == "I_batch1_reference" else role) + "_native"]["pymatgen_direct_S_geometry"],
        "agrees": direct[role]["native"]["S"] == summaries[("I" if role == "I_batch1_reference" else role) + "_native"]["pymatgen_direct_S_geometry"],
    } for role in ROLES}
    return {
        "schema": SCHEMA, "status": "matched" if not violations else "requires_attention", "expected_requests": expected_requests,
        "scope": {"new_model_calls": 0, "new_relaxation_or_energy_calls": 0, "new_sampling": False,
                  "chemistry_recomputed": False, "SUN_computed": False, "input_files_modified": False,
                  "construction_vs_I_is_separate_from_repair_vs_construction": True},
        "geometry_conventions": {
            "local_pymatgen_version": importlib.metadata.version("pymatgen"), "numpy_version": numpy.__version__,
            "Direct_function_source_sha256": DIRECT_FUNCTION_SOURCE_SHA256,
            "Direct_predicate": "min(structure.distance_matrix + diag(10.5)) >= 0.5 and volume >= 0.1",
            "pair_minimum_excludes_diagonal": True, "single_atom_pair_minimum_is_null": True,
            "shell125": "fractional coords mod1; shifts[-2,2]^3; minimum intersite distance",
            "repair125_additional_rules": "self image >=0.5A and angular radicand >1e-4",
            "comparison_tolerance_A": DISTANCE_COMPARISON_TOLERANCE,
            "validity_thresholds_are_not_relaxed_by_tolerance": True,
            "saved_Direct_report_does_not_include_pymatgen_version": True,
        },
        "verified_sources": sources, "completion_marker": marker, "components": components,
        "partial_body_token_mapping": token_mapping,
        "invariant_violations": violations, "summary": summaries,
        "saved_Direct_C_S_J": direct, "local_vs_saved_native_Direct_S": saved_direct_agreement,
        "Direct_C_note": "saved Direct counts upstream structure failures as invalid C; this is not a new test of formula-only C3FD validity",
        "tau800_note": "only exported aggregate Direct C/S/J is available; no tau800 per-request distance is inferred",
        "rows": rows,
    }


def markdown_report(report: Mapping[str, Any]) -> str:
    count = report["expected_requests"]
    text = ["# 匹配 batch1 的构造几何 canary 对照", "", f"每臂完整分母 **{count}**。本报告只比较已有几何与 Direct 文件，不计算 SUN，不重新验证化学组成。", "",
            "| 端点 | 完整构造 | pymatgen Direct 几何S | 125镜像Direct式S | 125镜像repair支持 | 构造约束失败 |",
            "|---|---:|---:|---:|---:|---:|"]
    for name, value in report["summary"].items():
        text.append(f"| {name} | {value['construction_complete']}/{count} | {value['pymatgen_direct_S_geometry']}/{count} | {value['shell125_direct_style_geometry']}/{count} | {value['shell125_repair_support']}/{count} | {value['construction_constraint_failures']} |")
    text.extend(["", "| 保存的Direct汇总 | native C/S/J | tau800 C/S/J |", "|---|---|---|"])
    for role, values in report["saved_Direct_C_S_J"].items():
        text.append(f"| {role} | {values['native']['C']}/{values['native']['S']}/{values['native']['J']} | {values['tau800']['C']}/{values['tau800']['S']}/{values['tau800']['J']} |")
    text.extend(["", "逐请求距离采用 pymatgen 完整周期距离；括号标记构造失败。G/P构造列用于判断新增构造约束，G/P终点列用于判断其后的修订。", "",
                 "| sample | N | I最短距Å | G构造/终点Å | P构造/终点Å | G/P构造token一致 |", "|---:|---:|---:|---|---|---|"])
    def distance(value):
        if not value["available"]:
            return "失败"
        number = value["pymatgen_minimum_pair_distance_A"]
        return "单原子" if number is None else f"{number:.6f}"
    for row in report["rows"]:
        m = row["geometry"]
        text.append(f"| {row['sample_idx']} | {m['I_native']['requested_N']} | {distance(m['I_native'])} | {distance(m['G_construction'])}/{distance(m['G_native'])} | {distance(m['P_construction'])}/{distance(m['P_native'])} | {row['G_P_initial_pairing']['complete_construction_tokens_equal']} |")
    differences = {name: value["pymatgen_vs_shell125_disagreement_ids"] for name, value in report["summary"].items()
                   if value["pymatgen_vs_shell125_disagreement_ids"]}
    text.extend(["", "pymatgen与125镜像距离/Direct式有效性差异：" + json.dumps(differences, ensure_ascii=False),
                 "本地pymatgen与保存Direct的S计数对照：" + json.dumps(report["local_vs_saved_native_Direct_S"], ensure_ascii=False),
                 "配对/来源异常：" + json.dumps(report["invariant_violations"], ensure_ascii=False), "",
                 "Direct C/S/J取自已有官方汇总；构造失败也计入分母。repair支持另含晶胞自身周期像与角度判据，与Direct规则不完全相同。每条原始125报告、重新计算的两种距离、修订token变化和来源SHA详见同名JSON。", ""])
    for row in report["rows"]:
        for role, diagnosis in row.get("partial_construction_diagnosis", {}).items():
            if not diagnosis:
                continue
            text.extend([f"sample {row['sample_idx']} / {role} 构造停止：{diagnosis['failure_reason']}。这是已提交前缀的失败，不能推断该组成不可生成。",
                         "已提交完整XYZ位点：" + str(diagnosis.get("committed_sites")),
                         "尚未完成位点：" + str(diagnosis.get("incomplete_sites")),
                         "固定XY且保持已提交位点时，精确Z支持为空的位点：" + str(diagnosis.get("zero_support_sites")),
                         "完整partial XYZ与逐位点合法Z bins见JSON；这是确定性约束枚举，没有新采样或能量计算。", ""])
    return "\n".join(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-dir", type=Path, required=True)
    parser.add_argument("--expected-requests", type=int, default=16)
    parser.add_argument("--output-prefix", type=Path, help="Optional analysis report prefix; writes .json and .md, never overwrites")
    args = parser.parse_args()
    outputs = []
    if args.output_prefix:
        outputs = [args.output_prefix.with_suffix(suffix) for suffix in (".json", ".md")]
        if any(path.exists() for path in outputs):
            raise FileExistsError("comparison reports already exist; source/report files are not overwritten")
    report = compare_assets(args.assets_dir, expected_requests=args.expected_requests)
    if outputs:
        args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
        outputs[0].write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        outputs[1].write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps({"status": report["status"], "outputs": [str(path.resolve()) for path in outputs],
                      "summary": report["summary"], "saved_Direct_C_S_J": report["saved_Direct_C_S_J"],
                      "invariant_violations": report["invariant_violations"]}, ensure_ascii=False, allow_nan=False))
    if report["invariant_violations"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
