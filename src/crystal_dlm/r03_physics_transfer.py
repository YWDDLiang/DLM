"""Offline K4/K8 native-body transfer into a repair-only copy of R5-C B0.

This is not the old full-path objective and never reassigns trajectory energy
to a new same-state action.  Construction stays on the immutable B0 adapter.
The public prompt, scalar view and support functions are shared with inference;
they do not import or require the old periodic-state conditioner.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

import torch

from crystal_dlm.dynamic_crystal import parse_dynamic_answer
from crystal_dlm.fixed_slot import FixedSlotConfig, MASK_TOKEN_ID, SYMBOL_TO_Z
from crystal_dlm.lattice_geometry import lattice_angle_rad
from crystal_dlm.llada_generation import (
    _apply_lightweight_decoding_masks, _apply_schema_masks,
    _lattice_matrix_from_token_ids,
)
from crystal_dlm.periodic_geometry_ops import minimum_image_distances
from crystal_dlm.programmed_path_data import trace_terminal_body
from crystal_dlm.terminal_energy_consistency import TERMINAL_VERIFICATION_PROTOCOL


REPAIR_VIEW_SCHEMA = "r03_composition_current_canvas_xyz_v1"
TRANSFER_SCHEMA = "r03_existing_native_physics_transfer_v1"
SOURCES_SCHEMA = "r03_physics_transfer_sources_v1"
B0_ADAPTER_SHA256 = "5c39976b6ab237cbab32cbfeb1c23a557571e1c7d2b60c1e60cbb450166ae76d"
REPAIR_SUPPORT_PROTOCOL = {
    "representation": "dynamic_v1", "max_atoms": 20, "coord_period": 100,
    "canonicalize_periodic_alias": True, "duplicate_coordinate_mask": True,
    "lattice_volume_mask": True, "min_lattice_rad": 1e-4,
    "pbc_min_distance_mask": True, "pbc_min_distance_A": 0.5,
    "pbc_image_radius": 2,
}
TERMINAL_PROTOCOL = {
    "model": "CHGNet-0.3.0", "optimizer": "FIRE", "relax_cell": True,
    "ase_filter": "FrechetCellFilter", "fmax": 0.1,
    "stress_tolerance_GPa": 0.5, "max_steps": 500,
    "scalar_pressure": 0.0, "constant_volume": False, "hydrostatic_strain": False,
}


class TransferContractError(ValueError):
    """A source or repair contract is not safe to reinterpret as this dataset."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False)


def content_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise TransferContractError(f"{name} must be an exact integer")
    if isinstance(value, str) and not value.isdecimal():
        raise TransferContractError(f"{name} must be an exact nonnegative integer")
    return int(value)


def composition_state(plan_state: Mapping[str, Any]) -> dict[str, Any]:
    """Retain exact composition only; never synthesize rich scientific fields."""
    n = _integer(plan_state.get("N"), "N")
    elements, counts = plan_state.get("elements"), plan_state.get("counts")
    if (not isinstance(elements, (list, tuple)) or not isinstance(counts, (list, tuple))
            or not elements or len(elements) != len(counts)):
        raise TransferContractError("aligned elements/counts are required")
    if len(set(elements)) != len(elements) or any(e not in SYMBOL_TO_Z for e in elements):
        raise TransferContractError("elements must be distinct known symbols")
    values = [_integer(c, "count") for c in counts]
    if not 1 <= n <= 20 or min(values) < 1 or sum(values) != n:
        raise TransferContractError("exact MP20 composition cardinality changed")
    ordered = sorted(zip(elements, values), key=lambda pair: SYMBOL_TO_Z[pair[0]])
    return {"N": n, "elements": [e for e, _ in ordered], "counts": [c for _, c in ordered]}


def build_repair_prompt(plan_state: Mapping[str, Any]) -> str:
    """Exact deployment prefix for both unadapted G and adapted P repairs."""
    return canonical_json(composition_state(plan_state)) + "\ndynamic_crystal_body:\n"


def build_repair_constraints(tokenizer: Any) -> dict[str, Any]:
    """Token-ID maps for the fixed legacy schema/alias/125-image support."""
    config, vocab = FixedSlotConfig(), tokenizer.get_vocab()

    def ids(prefix: str, low: int, high: int) -> dict[int, int]:
        result = {}
        for value in range(low, high + 1):
            token = f"<{prefix}_{value:03d}>"
            if token not in vocab:
                raise TransferContractError(f"B0 tokenizer lacks {token}; no vocabulary extension allowed")
            result[int(vocab[token])] = value
        if len(result) != high - low + 1:
            raise TransferContractError("crystal token IDs are not one-to-one")
        return result

    coords = {axis: ids(axis, 0, 100) for axis in "XYZ"}
    lengths = {axis: ids(axis, config.length_min_bin, config.length_max_bin)
               for axis in ("LA", "LB", "LC")}
    angles = {axis: ids(axis, config.angle_min_bin, config.angle_max_bin)
              for axis in ("AA", "AB", "AG")}
    return {
        **REPAIR_SUPPORT_PROTOCOL, "count_token_to_n": ids("N", 1, 20),
        "coord_token_to_bin": coords,
        "coord_bin_to_token_id": {a: {v: k for k, v in m.items()} for a, m in coords.items()},
        "coordinate_alias_token_ids": {
            a: (int(vocab[f"<{a}_000>"]), int(vocab[f"<{a}_100>"])) for a in "XYZ"},
        "length_token_to_bin": lengths, "length_step": config.length_step,
        "angle_token_to_bin": angles,
        "z_bin_to_token_id": {v: k for k, v in coords["Z"].items()},
        "gamma_bin_to_token_id": {v: k for k, v in angles["AG"].items()},
        "zero_length_token_ids_by_position": {
            i: int(vocab[f"<{a}_000>"]) for i, a in enumerate(("LA", "LB", "LC"), 1)},
    }


def _checked_constraints(tokenizer: Any, constraints: Mapping[str, Any] | None) -> dict:
    result = build_repair_constraints(tokenizer) if constraints is None else dict(constraints)
    for key, value in REPAIR_SUPPORT_PROTOCOL.items():
        if result.get(key) != value:
            raise TransferContractError(f"repair support changed at {key}")
    if result.get("body_offset", 0) != 0 or result.get("length_step") != 0.1:
        raise TransferContractError("repair requires the unchanged B0 body ABI")
    return result


def _fixed_cell(body: torch.Tensor, constraints: dict) -> tuple[torch.Tensor | None, str | None]:
    maps = constraints["angle_token_to_bin"]
    angles = [maps[a].get(int(body[p])) for a, p in (("AA", 4), ("AB", 5), ("AG", 6))]
    if any(v is None for v in angles) or lattice_angle_rad(*angles) <= 1e-4:
        return None, "invalid_fixed_lattice_angles"
    lattice = _lattice_matrix_from_token_ids(body, prompt_length=0, constraints=constraints)
    if lattice is None or not bool(torch.isfinite(lattice).all()):
        return None, "invalid_fixed_lattice"
    shell = torch.arange(-2, 3, dtype=lattice.dtype, device=lattice.device)
    shifts = torch.cartesian_prod(shell, shell, shell)
    shifts = shifts[(shifts != 0).any(-1)]
    if float(torch.linalg.vector_norm(shifts @ lattice, dim=-1).min()) < 0.5:
        return None, "periodic_self_image_below_0.5A"
    return lattice, None


def geometry_support_report(body_ids: Sequence[int] | torch.Tensor, *, tokenizer=None,
                            constraints=None) -> dict[str, Any]:
    """Check a complete native target without changing its physical structure."""
    constraints = _checked_constraints(tokenizer, constraints)
    body = torch.as_tensor(body_ids, dtype=torch.long)
    if body.ndim != 1 or len(body) < 11:
        return {"supported": False, "reason": "invalid_body_length"}
    n = constraints["count_token_to_n"].get(int(body[0]))
    if n is None or len(body) != 7 + 4 * n:
        return {"supported": False, "reason": "invalid_body_cardinality"}
    lattice, reason = _fixed_cell(body, constraints)
    if reason is not None:
        return {"supported": False, "reason": reason}
    coordinates = []
    for site in range(n):
        values = [constraints["coord_token_to_bin"][a].get(int(body[8 + 4 * site + k]))
                  for k, a in enumerate("XYZ")]
        if any(v is None for v in values):
            return {"supported": False, "reason": "incomplete_native_coordinates"}
        coordinates.append([float(v % 100) / 100 for v in values])
    minimum = None
    if n > 1:
        frac = torch.tensor(coordinates, dtype=lattice.dtype, device=lattice.device)
        distances = minimum_image_distances(frac[:, None] - frac[None, :], lattice, image_radius=2)
        distances.fill_diagonal_(torch.inf)
        minimum = float(distances.min())
        if minimum < 0.5:
            return {"supported": False, "reason": "native_pair_below_0.5A", "minimum_distance_A": minimum}
    return {"supported": True, "reason": None, "minimum_distance_A": minimum}


def canonicalize_body_aliases(body_ids: Sequence[int], *, tokenizer=None, constraints=None) -> list[int]:
    """Only 100→0 periodic aliases; token scale, atom order and cell are retained."""
    constraints = _checked_constraints(tokenizer, constraints)
    body = [int(v) for v in body_ids]
    n = constraints["count_token_to_n"].get(body[0]) if body else None
    if n is None or len(body) != 7 + 4 * n:
        raise TransferContractError("exact native body cardinality is required")
    for site in range(n):
        for axis_index, axis in enumerate("XYZ"):
            position = 8 + 4 * site + axis_index
            canonical, alias = constraints["coordinate_alias_token_ids"][axis]
            if body[position] == alias:
                body[position] = canonical
    return body


def repair_scalar_example(body_ids: Sequence[int], site_index: int, axis_index: int,
                          *, plan_state: Mapping[str, Any], tokenizer=None,
                          constraints=None, mask_id: int = MASK_TOKEN_ID) -> dict[str, Any]:
    """One XYZ-block conditional: XYZ masked, then X visible, then XY visible.

    Inference uses its already drawn X/Y prefix. Training uses the saved native
    X/Y prefix, as ordinary conditional likelihood, not as a new physics label.
    """
    constraints = _checked_constraints(tokenizer, constraints)
    body = canonicalize_body_aliases(body_ids, constraints=constraints)
    n = composition_state(plan_state)["N"]
    if len(body) != 7 + 4 * n or not 0 <= site_index < n or axis_index not in (0, 1, 2):
        raise TransferContractError("invalid fixed-composition repair position")
    start = 8 + 4 * site_index
    staged = body.copy()
    for offset in range(axis_index, 3):
        staged[start + offset] = int(mask_id)
    return {"repair_view_schema": REPAIR_VIEW_SCHEMA, "prompt": build_repair_prompt(plan_state),
            "input_body": staged, "position": start + axis_index,
            "target_token": body[start + axis_index], "site_index": site_index,
            "axis_index": axis_index, "num_atoms": n,
            "transaction_positions": [start, start + 1, start + 2]}


def supported_scalar_logits(raw_logits: torch.Tensor, input_body: Sequence[int] | torch.Tensor,
                            prompt_length: int, N: int, position: int, tokenizer=None,
                            constraints=None, mask_id: int = MASK_TOKEN_ID):
    """Return (active logits, report), preserving alias gradients and exact support.

    raw_logits may be V, L×V, or 1×L×V. position is body-relative. A false
    report['available'] means inference must roll back the whole staged XYZ;
    training must not restore an unsupported target to the action set.
    """
    constraints = _checked_constraints(tokenizer, constraints)
    n = _integer(N, "N")
    body = torch.as_tensor(input_body, dtype=torch.long, device=raw_logits.device)
    component = (position - 8) % 4
    if (body.ndim != 1 or len(body) != 7 + 4 * n or position < 8 or position >= len(body)
            or component > 2 or int(body[position]) != int(mask_id)
            or constraints["count_token_to_n"].get(int(body[0])) != n):
        raise TransferContractError("active scalar must be a masked XYZ position in exact 7+4N")
    if raw_logits.ndim == 1:
        vector = raw_logits
    elif raw_logits.ndim == 2:
        vector = raw_logits[prompt_length + position]
    elif raw_logits.ndim == 3 and raw_logits.shape[0] == 1:
        vector = raw_logits[0, prompt_length + position]
    else:
        raise TransferContractError("one logical scalar row is required")
    if not vector.is_floating_point():
        raise TransferContractError("LM logits must be floating point")
    minimum = torch.finfo(vector.dtype).min
    _, cell_reason = _fixed_cell(body, constraints)
    if cell_reason is not None:
        return vector.masked_fill(torch.ones_like(vector, dtype=torch.bool), minimum), {
            "available": False, "no_legal_completion": False, "reason": cell_reason, "legal_count": 0}
    axis = "XYZ"[component]
    allowed = torch.zeros((len(body), vector.numel()), dtype=torch.bool, device=vector.device)
    allowed[position, list(constraints["coord_token_to_bin"][axis])] = True
    with torch.no_grad():
        # Only a detached support calculation uses the temporary body canvas.
        # The retained active-vector graph below is the differentiable path.
        probe = vector.new_zeros((1, len(body), vector.numel()))
        probe[0, position] = vector.detach()
        _apply_schema_masks(probe, body[None], 0, len(body), allowed, None)
        active = torch.zeros((1, len(body)), dtype=torch.bool, device=vector.device)
        active[0, position] = True
        reports = _apply_lightweight_decoding_masks(probe, body[None], 0, len(body),
                                                    constraints, active, int(mask_id))
        no_completion = (0, position) in reports["pbc_no_legal_completion"]
        selected = probe[0, position]
        legal = torch.isfinite(selected) & (selected > minimum)
        nonfinite = bool(torch.isnan(selected).any() or torch.isposinf(selected).any())
    transformed = vector.masked_fill(~allowed[position], minimum)
    canonical, alias = constraints["coordinate_alias_token_ids"][axis]
    merged = torch.logaddexp(transformed[canonical], transformed[alias])
    transformed = transformed.scatter(0, torch.tensor([canonical], device=vector.device), merged.reshape(1))
    available = bool(legal.any()) and not no_completion and not nonfinite
    reason = ("no_legal_Z_completion" if no_completion else
              "nonfinite_active_logits" if nonfinite else "empty_scalar_support" if not available else None)
    if not available:
        legal = torch.zeros_like(legal)
    return transformed.masked_fill(~legal, minimum), {
        "available": available, "no_legal_completion": no_completion,
        "reason": reason, "legal_count": int(legal.sum()),
    }


def _tokens(tokenizer: Any, ids: Sequence[int], *, vocab=None, atomic_checked=None) -> list[str]:
    values = tokenizer.convert_ids_to_tokens([int(v) for v in ids])
    if not isinstance(values, (tuple, list)) or len(values) != len(ids):
        raise TransferContractError("source tokenizer cannot map every body ID")
    vocab = tokenizer.get_vocab() if vocab is None else vocab
    atomic_checked = set() if atomic_checked is None else atomic_checked
    for token, token_id in zip(values, ids):
        if not isinstance(token, str) or vocab.get(token) != int(token_id):
            raise TransferContractError("source token-ID/string mapping is inconsistent")
        if token not in atomic_checked:
            if list(tokenizer(token, add_special_tokens=False)["input_ids"]) != [int(token_id)]:
                raise TransferContractError("source crystal token is not atomic")
            atomic_checked.add(token)
    return list(values)


def _physical_signature(tokens: Sequence[str]) -> dict[str, Any]:
    parsed = parse_dynamic_answer("".join(tokens), strict=True)
    canonical = list(parsed["tokens"])
    for position in range(8, len(canonical)):
        if (position - 8) % 4 < 3 and canonical[position].endswith("_100>"):
            canonical[position] = canonical[position][:-4] + "000>"
    return {"canonical_periodic_tokens": canonical, "num_atoms": parsed["num_atoms"]}


def _check_structure_matches_body(structure: Mapping[str, Any], parsed: Mapping[str, Any]) -> None:
    """Check metric and ordered fractional sites, permitting a rotated cell frame."""
    matrix = structure.get("lattice", {}).get("matrix")
    if (not isinstance(matrix, (list, tuple)) or len(matrix) != 3
            or any(not isinstance(v, (list, tuple)) or len(v) != 3 for v in matrix)):
        raise TransferContractError("labelled native structure has no complete lattice")
    matrix = [[float(v) for v in row] for row in matrix]
    if not all(math.isfinite(v) for row in matrix for v in row):
        raise TransferContractError("nonfinite labelled native lattice")
    gram = [[sum(matrix[i][k] * matrix[j][k] for k in range(3)) for j in range(3)] for i in range(3)]
    lengths = list(map(float, parsed["lengths"]))
    angles = list(map(math.radians, parsed["angles"]))
    expected = [[lengths[i] ** 2 if i == j else 0.0 for j in range(3)] for i in range(3)]
    for i, j, angle in ((1, 2, angles[0]), (0, 2, angles[1]), (0, 1, angles[2])):
        expected[i][j] = expected[j][i] = lengths[i] * lengths[j] * math.cos(angle)
    if any(not math.isclose(gram[i][j], expected[i][j], rel_tol=1e-8, abs_tol=1e-7)
           for i in range(3) for j in range(3)):
        raise TransferContractError("body lattice differs from the physically labelled native structure")
    sites = structure.get("sites")
    if not isinstance(sites, list) or len(sites) != parsed["num_atoms"]:
        raise TransferContractError("labelled native structure changed site count")
    for site, symbol, expected_coord in zip(sites, parsed["species"], parsed["frac_coords"]):
        species = site.get("species")
        if (not isinstance(species, list) or len(species) != 1 or species[0].get("element") != symbol
                or float(species[0].get("occu", 1.0)) != 1.0):
            raise TransferContractError("labelled native species/order differs from body")
        coords = site.get("abc")
        if not isinstance(coords, list) or len(coords) != 3:
            raise TransferContractError("labelled native fractional coordinates are absent")
        for actual, expected_value in zip(coords, expected_coord):
            delta = float(actual) - float(expected_value)
            if not math.isfinite(delta) or abs(delta - round(delta)) > 1e-8:
                raise TransferContractError("native coordinates differ from the physically labelled object")


def map_native_body(path: Mapping[str, Any], label: Mapping[str, Any],
                    source_tokenizer: Any, b0_tokenizer: Any, *, constraints=None,
                    mapping_cache=None) -> dict[str, Any]:
    """Verify original label identity first, then map strings and periodic aliases."""
    if path.get("success") is not True or not isinstance(path.get("body"), str):
        raise TransferContractError("a successful saved native body is required")
    if path.get("endpoint") not in (None, "native") or label.get("endpoint") not in (None, "native"):
        raise TransferContractError("only original native training feedback is supported")
    source_ids = path.get("final_body_token_ids")
    if not isinstance(source_ids, list) or not source_ids:
        raise TransferContractError("saved final native token IDs are absent")
    if trace_terminal_body(path["trace"]) != source_ids:
        raise TransferContractError("saved body differs from the completed attempted trace")
    cache = mapping_cache if mapping_cache is not None else {
        "source_vocab": source_tokenizer.get_vocab(), "b0_vocab": b0_tokenizer.get_vocab(),
        "source_atomic": set(), "b0_atomic": set(),
    }
    source_tokens = _tokens(source_tokenizer, source_ids, vocab=cache["source_vocab"], atomic_checked=cache["source_atomic"])
    parsed = parse_dynamic_answer(path["body"], strict=True)
    if list(parsed["tokens"]) != source_tokens:
        raise TransferContractError("saved body text and original tokenizer IDs disagree")
    comp = composition_state(path["plan_state"])
    if parsed["num_atoms"] != comp["N"] or dict(Counter(parsed["species"])) != dict(zip(comp["elements"], comp["counts"])):
        raise TransferContractError("saved body does not preserve its recorded exact composition")
    labelled_text = json.dumps(path["structure"], sort_keys=True) if path.get("structure") is not None else path["body"]
    expected_cache_key = hashlib.sha256(labelled_text.encode()).hexdigest()
    if label.get("endpoint_cache_key") != expected_cache_key:
        raise TransferContractError("physical label cache key does not identify this native body")
    if path.get("structure") is not None:
        _check_structure_matches_body(path["structure"], parsed)
    vocab = cache["b0_vocab"]
    if any(token not in vocab for token in source_tokens):
        raise TransferContractError("saved geometry has a token absent from frozen B0")
    mapped = [int(vocab[token]) for token in source_tokens]
    if _tokens(b0_tokenizer, mapped, vocab=vocab, atomic_checked=cache["b0_atomic"]) != source_tokens:
        raise TransferContractError("source-to-B0 numeric token strings changed")
    constraints = _checked_constraints(b0_tokenizer, constraints)
    canonical = canonicalize_body_aliases(mapped, constraints=constraints)
    canonical_tokens = _tokens(b0_tokenizer, canonical, vocab=vocab, atomic_checked=cache["b0_atomic"])
    signature = _physical_signature(source_tokens)
    if signature != _physical_signature(canonical_tokens):
        raise TransferContractError("token transfer changed the periodic physical identity")
    geometry = geometry_support_report(canonical, constraints=constraints)
    if not geometry["supported"]:
        raise TransferContractError(str(geometry["reason"]))
    anchors = []
    seen = set()
    for site, symbol in enumerate(parsed["species"]):
        if symbol not in seen:
            anchors.append(site)
            seen.add(symbol)
    # Complete-cell support implies that this body's own XYZ completion is
    # admissible. The exact active support is additionally checked on every
    # sampled training scalar, without allocating a V×canvas tensor per source.
    return {"body_token_ids": canonical, "plan_state": comp, "prompt": build_repair_prompt(comp),
            "anchor_sites": anchors, "num_atoms": comp["N"],
            "source_body_token_ids": source_ids, "source_body_tokens": source_tokens,
            "source_body_sha256": hashlib.sha256(path["body"].encode()).hexdigest(),
            "label_endpoint_cache_key": expected_cache_key,
            "physical_signature_sha256": content_sha256(signature),
            "alias_changes": sum(a != b for a, b in zip(mapped, canonical)),
            "geometry_support": geometry}


def _condition_signature(path: Mapping[str, Any]) -> str:
    return content_sha256({key: path.get(key) for key in
                          ("group_id", "source_split", "source_row_idx", "plan_state", "prompt",
                           "species_program", "species_program_source")})


def adapt_round_rows(teacher: Mapping[str, Any], paths: Sequence[Mapping[str, Any]],
                     labels: Sequence[Mapping[str, Any]], *, role: str,
                     source_tokenizer: Any, b0_tokenizer: Any, constraints=None,
                     expected_conditions: int = 1024) -> tuple[list[dict], list[dict], dict]:
    """Check a complete old empirical pool without bypassing its reference checks.

    expected_conditions is explicit for tiny CPU fixtures. The production file
    adapter below fixes it to 1024 and accepts only the registered K4/K8 rounds.
    """
    if role not in ("k8", "k4"):
        raise TransferContractError("only accepted K8 and consistent-v2 K4 are supported")
    round_index, candidates_per_condition = (1, 8) if role == "k8" else (0, 4)
    summary, provenance = teacher["summary"], teacher["provenance"]
    if (summary.get("trainable_teacher") is not True or summary.get("diagnostic_only") is not False
            or summary.get("solver_status") != "optimal" or float(summary.get("primal_residual", math.inf)) > 1e-6):
        raise TransferContractError("teacher has not reached its accepted trainable terminal")
    if (provenance.get("collection_round") != round_index
            or provenance.get("candidates_per_condition") != candidates_per_condition
            or provenance.get("verification_protocol") != TERMINAL_VERIFICATION_PROTOCOL):
        raise TransferContractError("teacher round, budget or terminal verification protocol changed")
    if any(provenance.get("terminal_protocol", {}).get(key) != value for key, value in TERMINAL_PROTOCOL.items()):
        raise TransferContractError("teacher physical protocol differs from accepted native feedback")
    expected_count = expected_conditions * candidates_per_condition
    if len(paths) != expected_count or len(labels) != expected_count:
        raise TransferContractError("old path/label accounting is incomplete")
    path_by_id = {str(row["trajectory_id"]): row for row in paths}
    label_by_id = {str(row["trajectory_id"]): row for row in labels}
    if len(path_by_id) != len(paths) or len(label_by_id) != len(labels) or path_by_id.keys() != label_by_id.keys():
        raise TransferContractError("duplicate, missing or extra physical occurrences")
    teacher_rows = {}
    uniform_ids = {str(value) for value in summary.get("uniform_reference_group_ids", [])}
    uniform_records = {str(row["group_id"]): dict(row) for row in summary.get("fixed_reference_groups", [])}
    uniform_ids.update(uniform_records)
    for group in teacher["groups"]:
        if str(group["group_id"]) in uniform_ids:
            positive = [float(c["weight"]) for c in group["candidates"] if float(c["weight"]) > 0]
            if not positive or any(not math.isclose(w, 1.0 / len(positive), rel_tol=1e-9, abs_tol=1e-9)
                                   for w in positive):
                raise TransferContractError("an explicitly uniform-reference group lost its original uniform weights")
        for candidate in group["candidates"]:
            tid = str(candidate["trajectory_id"])
            if tid in teacher_rows or str(candidate.get("group_id")) != str(group["group_id"]):
                raise TransferContractError("teacher occurrence grouping is inconsistent")
            teacher_rows[tid] = candidate
    if teacher_rows.keys() != path_by_id.keys() or len(teacher["groups"]) != expected_conditions:
        raise TransferContractError("teacher omits original occurrences or conditions")
    constraints = _checked_constraints(b0_tokenizer, constraints)
    mapping_cache = {"source_vocab": source_tokenizer.get_vocab(), "b0_vocab": b0_tokenizer.get_vocab(),
                     "source_atomic": set(), "b0_atomic": set()}
    grouped = defaultdict(list)
    original_weight_sums = defaultdict(float)
    signatures = {}
    compatible, rejected = [], []
    for tid, path in path_by_id.items():
        label, candidate = label_by_id[tid], teacher_rows[tid]
        if path.get("source_split") != "train" or label.get("source_split") != "train":
            raise TransferContractError("evaluation or validation feedback cannot enter physics transfer")
        if path.get("diagnostic_only") or path.get("reference_closure") or path.get("legacy_reference_closure") is not None:
            raise TransferContractError("diagnostic or model494/reference-closure paths are not this native teacher")
        if (str(path.get("checkpoint")) != str(provenance.get("checkpoint"))
                or path.get("collection_round") != round_index):
            raise TransferContractError("old path reference policy differs from its own teacher")
        for key in ("group_id", "source_row_idx", "source_split"):
            if str(path.get(key)) != str(label.get(key)) or str(label.get(key)) != str(candidate.get(key)):
                raise TransferContractError(f"teacher/path/label provenance mismatch: {key}")
        for key in ("verified", "status", "raw_energy", "terminal_energy", "endpoint_cache_key", "terminal_consistency"):
            if candidate.get(key) != label.get(key):
                raise TransferContractError(f"teacher and saved physical label differ: {key}")
        group_id = str(path["group_id"])
        grouped[group_id].append(_integer(path["candidate_index"], "candidate_index"))
        signature = _condition_signature(path)
        if group_id in signatures and signatures[group_id] != signature:
            raise TransferContractError("one old condition group contains inconsistent scientific conditions")
        signatures[group_id] = signature
        weight = float(candidate["weight"])
        if not math.isfinite(weight) or weight < 0:
            raise TransferContractError("invalid original teacher weight")
        original_weight_sums[group_id] += weight
        base = {"source_role": role, "trajectory_id": tid, "source_group_id": group_id,
                "source_row_idx": path["source_row_idx"], "condition_signature": signature,
                "collection_round": round_index, "source_checkpoint": provenance["checkpoint"],
                "original_teacher_weight": weight, "label_status": label.get("status"),
                "uniform_reference_group": group_id in uniform_ids,
                "original_uniform_reference_record": uniform_records.get(group_id),
                "source_credibility_treatment": summary.get("credibility_treatment")}
        if weight == 0:
            rejected.append({**base, "reason": "original_teacher_zero_weight"})
            continue
        if (label.get("verified") is not True or label.get("status") != "verified"
                or label.get("optimizer_converged") is not True
                or label.get("terminal_consistency", {}).get("status") != "consistent"
                or not all(isinstance(label.get(k), (float, int)) and math.isfinite(float(label[k]))
                           for k in ("raw_energy", "terminal_energy"))):
            raise TransferContractError("positive teacher weight lacks unchanged verified finite native labels")
        try:
            mapped = map_native_body(path, label, source_tokenizer, b0_tokenizer, constraints=constraints,
                                     mapping_cache=mapping_cache)
        except (TransferContractError, ValueError, KeyError) as exc:
            rejected.append({**base, "reason": str(exc)})
            continue
        compatible.append({**base, **mapped, "schema": TRANSFER_SCHEMA, "source_split": "train",
                           "raw_energy": label["raw_energy"], "terminal_energy": label["terminal_energy"],
                           "label_payload_sha256": content_sha256(label),
                           "original_condition_payload_sha256": signature})
    if len(grouped) != expected_conditions or any(sorted(v) != list(range(candidates_per_condition)) for v in grouped.values()):
        raise TransferContractError("complete old candidate budget is not retained in source accounting")
    if any(total != 0 and not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-9)
           for total in original_weight_sums.values()):
        raise TransferContractError("original teacher condition mass was not normalized")
    return compatible, rejected, {"source_role": role, "requested_occurrences": expected_count,
                                   "original_conditions": expected_conditions, "compatible_occurrences": len(compatible),
                                   "original_positive_occurrences": sum(float(c["weight"]) > 0 for c in teacher_rows.values()),
                                   "positive_incompatible_occurrences": sum(row["original_teacher_weight"] > 0 for row in rejected),
                                   "compatible_conditions": len({r["condition_signature"] for r in compatible}),
                                   "uniform_reference_group_ids": sorted(uniform_ids),
                                   "uniform_reference_weights_preserved": True,
                                   "rejected_reasons": dict(Counter(row["reason"] for row in rejected)),
                                   "teacher_summary": dict(summary), "source_reference_verified": True}


def select_and_normalize_records(k8_rows: Sequence[Mapping[str, Any]],
                                 k4_rows: Sequence[Mapping[str, Any]] = ()) -> tuple[list[dict], list[dict], dict]:
    """K8 priority per exact old condition; equal total mass per exact composition."""
    supported = {row["condition_signature"] for row in k8_rows}
    omitted = [{"source_role": "k4", "trajectory_id": row["trajectory_id"],
                "reason": "compatible_K8_condition_takes_priority"}
               for row in k4_rows if row["condition_signature"] in supported]
    rows = [dict(row) for row in k8_rows] + [dict(row) for row in k4_rows if row["condition_signature"] not in supported]
    groups = defaultdict(list)
    composition_groups = defaultdict(set)
    for row in rows:
        key = (row["source_role"], row["condition_signature"])
        groups[key].append(row)
        comp = canonical_json(composition_state(row["plan_state"]))
        row["exact_composition_key"] = comp
        composition_groups[comp].add(key)
    if not rows:
        raise TransferContractError("no compatible existing native physics remains; do not collect replacement data")
    retained_masses = []
    for key, group in groups.items():
        total = sum(float(row["original_teacher_weight"]) for row in group)
        if not math.isfinite(total) or total <= 0:
            raise TransferContractError("retained condition has no positive original teacher mass")
        retained_masses.append(total)
        comp = group[0]["exact_composition_key"]
        if any(row["exact_composition_key"] != comp for row in group):
            raise TransferContractError("an original condition changed composition")
        for row in group:
            q = float(row["original_teacher_weight"]) / total
            row.update(within_condition_weight=q, compatible_original_mass=total,
                       dataset_weight=q / len(composition_groups[comp]) / len(composition_groups),
                       condition_occurrences=len(group),
                       source_condition_key=list(key))
    masses = {comp: sum(row["dataset_weight"] for row in rows if row["exact_composition_key"] == comp)
              for comp in composition_groups}
    if not math.isclose(sum(masses.values()), 1.0, rel_tol=1e-12, abs_tol=1e-12):
        raise TransferContractError("offline dataset normalization failed")
    return rows, omitted, {"compatible_occurrences": len(rows), "source_conditions": len(groups),
                           "exact_compositions": len(composition_groups),
                           "single_occurrence_conditions": sum(len(g) == 1 for g in groups.values()),
                           "conditions_with_dropped_positive_mass": sum(m < 1 - 1e-9 for m in retained_masses),
                           "retained_original_teacher_mass": {"min": min(retained_masses), "max": max(retained_masses),
                                                               "mean": sum(retained_masses) / len(retained_masses)},
                           "composition_mass_min": min(masses.values()), "composition_mass_max": max(masses.values()),
                           "normalization": "equal exact-composition mass; equal separate original-condition mass; renormalized original q",
                           "old_dual_objective_guarantee_transferred": False,
                           "new_same_state_counterfactual_claimed": False}


def sample_transfer_example(records: Sequence[Mapping[str, Any]], *, tokenizer, constraints,
                            rng: random.Random) -> dict[str, Any]:
    """Uniform record sampling with an explicit importance weight for offline CE."""
    row = records[rng.randrange(len(records))]
    site = row["anchor_sites"][rng.randrange(len(row["anchor_sites"]))]
    axis = rng.randrange(3)
    view = repair_scalar_example(row["body_token_ids"], site, axis, plan_state=row["plan_state"], constraints=constraints)
    return {**view, "weight": len(records) * float(row["dataset_weight"]),
            "trajectory_id": row["trajectory_id"], "source_role": row["source_role"],
            "source_condition_key": row["source_condition_key"]}


def weighted_scalar_ce(processed_logits: torch.Tensor, target: int, *, weight: float = 1.0,
                       temperature: float = 0.7) -> torch.Tensor:
    """Exact normalized CE on supported canonical actions; unsupported is an error."""
    if not math.isfinite(weight) or weight < 0 or temperature != 0.7:
        raise TransferContractError("fixed finite offline weight and deployment temperature 0.7 required")
    minimum = torch.finfo(processed_logits.dtype).min
    legal = torch.isfinite(processed_logits) & (processed_logits > minimum)
    if not 0 <= target < len(legal) or not bool(legal[target]):
        raise TransferContractError("unsupported target cannot be inserted back into legal actions")
    values = processed_logits[legal].float() / temperature
    target_logit = processed_logits[target].float() / temperature
    return float(weight) * (torch.logsumexp(values, dim=0) - target_logit)


def supported_reference_kl(student: torch.Tensor, reference: torch.Tensor, *, temperature=0.7) -> torch.Tensor:
    minimum = torch.finfo(student.dtype).min
    legal = torch.isfinite(student) & (student > minimum)
    reference_legal = torch.isfinite(reference) & (reference > torch.finfo(reference.dtype).min)
    if temperature != 0.7 or not bool(legal.any()) or not torch.equal(legal, reference_legal):
        raise TransferContractError("B0 retention must use exactly the student's fixed support")
    log_q = torch.log_softmax(student[legal].float() / temperature, dim=0)
    log_p = torch.log_softmax(reference[legal].detach().float() / temperature, dim=0)
    return (log_p.exp() * (log_p - log_q)).sum()


def _read_json(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as stream:
        result = json.load(stream)
    if not isinstance(result, dict):
        raise TransferContractError(f"expected a JSON object at {path}")
    return result


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for ordinal, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TransferContractError(f"non-object source row {path}:{ordinal}")
            rows.append(row)
    return rows


def _verify_file(spec: Mapping[str, Any]) -> Path:
    path = Path(spec["path"])
    expected = spec.get("sha256")
    if not isinstance(expected, str) or len(expected) != 64 or file_sha256(path) != expected:
        raise TransferContractError(f"source SHA256 mismatch or missing pin: {path}")
    return path


def _successful_parent(path: Path) -> None:
    if not (path.parent / "_SUCCESS").is_file() or (path.parent / "_INVALIDATED").exists():
        raise TransferContractError(f"source is incomplete or invalidated: {path}")


def load_round_manifest(spec: Mapping[str, Any], *, role: str, b0_tokenizer: Any,
                        source_tokenizer: Any, constraints=None):
    """Read pinned original artifacts. Paths may be relocated with recorded_path.

    Each round spec contains teacher:{path,sha256}, paths:[{path,sha256}],
    labels:[{path,sha256}], source_tokenizer_path and tokenizer_json_sha256.
    K4 additionally requires data_revision='consistent_v2'. A relocated file
    descriptor records its original teacher-provenance name in recorded_path.
    """
    if role == "k4" and spec.get("data_revision") != "consistent_v2":
        raise TransferContractError("K4 may supplement only as explicitly accepted consistent_v2")
    teacher_path = _verify_file(spec["teacher"])
    _successful_parent(teacher_path)
    teacher = _read_json(teacher_path)
    tokenizer_path = Path(spec["source_tokenizer_path"]) / "tokenizer.json"
    _verify_file({"path": str(tokenizer_path), "sha256": spec.get("tokenizer_json_sha256")})
    files = {kind: list(spec.get(kind, [])) for kind in ("paths", "labels")}
    if not files["paths"] or not files["labels"]:
        raise TransferContractError("pinned original path and native-label files are required")
    for kind, key in (("paths", "paths_jsonl"), ("labels", "labels_jsonl")):
        recorded = [str(item.get("recorded_path", item["path"])) for item in files[kind]]
        if len(set(recorded)) != len(recorded) or set(recorded) != set(teacher["provenance"][key]):
            raise TransferContractError(f"{kind} files differ from the teacher's actual provenance")
        for item, recorded_path in zip(files[kind], recorded):
            teacher_pin = teacher["provenance"].get("source_sha256", {}).get(recorded_path)
            if teacher_pin is not None and teacher_pin != item.get("sha256"):
                raise TransferContractError("source file pin disagrees with the accepted teacher's own SHA")
    paths, labels = [], []
    for item in files["paths"]:
        path = _verify_file(item)
        _successful_parent(path)
        paths.extend(read_jsonl(path))
    for item in files["labels"]:
        path = _verify_file(item)
        _successful_parent(path)
        report = _read_json(path.parent / "LABEL_FINAL.json")
        if report.get("purpose") != "train" or report.get("verification_protocol") != TERMINAL_VERIFICATION_PROTOCOL:
            raise TransferContractError("evaluation or unverified label protocol cannot enter transfer")
        if any(report.get("protocol", {}).get(k) != v for k, v in TERMINAL_PROTOCOL.items()):
            raise TransferContractError("native label settings differ from the accepted teacher")
        loaded = read_jsonl(path)
        if report.get("requested") != len(loaded) or report.get("completed") != len(loaded):
            raise TransferContractError("native label completion receipt differs from file rows")
        labels.extend(loaded)
    output = adapt_round_rows(teacher, paths, labels, role=role, source_tokenizer=source_tokenizer,
                              b0_tokenizer=b0_tokenizer, constraints=constraints)
    output[2].update(teacher_json=str(teacher_path), teacher_sha256=spec["teacher"]["sha256"],
                     source_file_pins=files, data_revision=spec.get("data_revision"))
    return output


def prepare_mp20_retention(spec: Mapping[str, Any] | None, *, b0_tokenizer: Any,
                           constraints=None) -> tuple[list[dict], dict]:
    """Read only a pinned original MP20 train file; no new scientific targets."""
    if spec is None:
        return [], {"available": False, "reason": "no_pinned_MP20_train_source", "compatible_sources": 0}
    if spec.get("source_split") != "train":
        raise TransferContractError("MP20 retention requires explicitly pinned train provenance")
    path = _verify_file(spec)
    vocab = b0_tokenizer.get_vocab()
    constraints = _checked_constraints(b0_tokenizer, constraints)
    retained, reasons = [], Counter()
    rows = read_jsonl(path)
    for ordinal, row in enumerate(rows):
        explicit_split = row.get("source_split", row.get("metadata", {}).get("split"))
        if explicit_split not in (None, "train"):
            raise TransferContractError("heldout source found in pinned MP20 train retention file")
        try:
            plan = composition_state(row.get("plan_state", row.get("r5_plan_state", {})))
            parsed = parse_dynamic_answer(row["answer"], strict=True)
            if dict(Counter(parsed["species"])) != dict(zip(plan["elements"], plan["counts"])):
                raise TransferContractError("MP20 body and composition disagree")
            if any(token not in vocab for token in parsed["tokens"]):
                raise TransferContractError("MP20 body token absent from frozen B0")
            body = canonicalize_body_aliases([int(vocab[t]) for t in parsed["tokens"]], constraints=constraints)
            support = geometry_support_report(body, constraints=constraints)
            if not support["supported"]:
                raise TransferContractError(str(support["reason"]))
            seen, anchors = set(), []
            for site, symbol in enumerate(parsed["species"]):
                if symbol not in seen:
                    anchors.append(site)
                    seen.add(symbol)
            retained.append({"source_row_idx": row.get("source_row_idx", ordinal), "source_split": "train",
                             "body_token_ids": body, "plan_state": plan, "anchor_sites": anchors,
                             "source_answer_sha256": hashlib.sha256(row["answer"].encode()).hexdigest()})
        except (TransferContractError, ValueError, KeyError) as exc:
            reasons[str(exc)] += 1
    return retained, {"available": bool(retained), "source_path": str(path), "source_sha256": spec["sha256"],
                       "sources": len(rows), "compatible_sources": len(retained), "rejected_reasons": dict(reasons),
                       "rich_scientific_fields_synthesized": False}


def prepare_transfer(sources: Mapping[str, Any], *, b0_tokenizer: Any,
                     source_tokenizers: Mapping[str, Any]) -> tuple[list[dict], list[dict], list[dict], dict]:
    """Materialize immutable compatibility views; no model or physics calls."""
    if sources.get("schema") != SOURCES_SCHEMA or "k8" not in sources:
        raise TransferContractError("a pinned accepted K8 source manifest is required")
    constraints = build_repair_constraints(b0_tokenizer)
    k8, rejected8, report8 = load_round_manifest(sources["k8"], role="k8", b0_tokenizer=b0_tokenizer,
                                                 source_tokenizer=source_tokenizers["k8"], constraints=constraints)
    k4, rejected4, report4 = [], [], None
    if sources.get("k4") is not None:
        k4, rejected4, report4 = load_round_manifest(sources["k4"], role="k4", b0_tokenizer=b0_tokenizer,
                                                     source_tokenizer=source_tokenizers["k4"], constraints=constraints)
    records, omitted, normalizer = select_and_normalize_records(k8, k4)
    retention, retention_report = prepare_mp20_retention(sources.get("mp20"), b0_tokenizer=b0_tokenizer,
                                                          constraints=constraints)
    report = {"schema": TRANSFER_SCHEMA, "status": "complete", "repair_view_schema": REPAIR_VIEW_SCHEMA,
              "support_protocol": dict(REPAIR_SUPPORT_PROTOCOL), "k8": report8, "k4": report4,
              "normalization": normalizer, "mp20_retention": retention_report,
              "new_model_calls": 0, "new_physics_calls": 0, "new_training_structures": 0,
              "physical_targets": "unchanged saved native geometry; original IDs mapped by token strings; periodic 100=0 aliases only",
              "energy_objective": "offline quality-weighted ordinary XYZ-block conditionals; not original full-path or same-state counterfactual",
              "source_manifest": dict(sources)}
    return records, rejected8 + rejected4 + omitted, retention, report
