"""One source-keyed, teacher-forced scalar state for MP20 revision warmup.

This is geometry-only data construction, not a trajectory/energy teacher.
``old_body`` is the complete corrupted pre-transaction snapshot; only the
already-committed transaction prefix in ``input_body`` sees clean targets.
Runtime support is deliberately left to the trainer: even an alias or a
collision-invalid native teacher must remain an identifiable example.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from crystal_dlm.dynamic_crystal import arrays_to_dynamic_tokens, parse_dynamic_answer
from crystal_dlm.fixed_slot import FixedSlotConfig
from crystal_dlm.llada_generation import _lattice_matrix_from_token_ids
from crystal_dlm.manifold_corruption import lattice_parameters_from_matrix
from crystal_dlm.programmed_path_runtime import cooperative_slots
from crystal_dlm.spad_program import (
    LATTICE_POSITIONS,
    coordinate_positions,
    element_position,
    program_from_element_order,
    reverse_species_block_revision_slots,
)


# Read only known provenance/conditioning fields, never bulk-copy a row that
# might also contain outcome labels. Old rollout masks are not reusable here.
_METADATA_KEYS = (
    "prompt", "answer", "plan_state", "r5_plan_state", "source_plan_state",
    "teacherPlan", "teacher_plan", "plan_state_version", "plan_source",
    "species_program", "species_program_source", "contact_tree_order_symbols",
    "source_row_idx", "source_split", "source_id", "material_id", "mp_id",
    "sample_weight", "loss_profile", "pointer_semantics_available",
)
_CELL_STRAIN_BOUND = 0.015
_SHARED_DISPLACEMENT_BOUND_A = 0.06
_LOCAL_DISPLACEMENT_BOUND_A = 0.03


def _lattice(body: Sequence[int], constraints: dict) -> np.ndarray:
    value = _lattice_matrix_from_token_ids(
        torch.tensor(body, dtype=torch.long), prompt_length=0, constraints=constraints,
    )
    if value is None or not bool(torch.isfinite(value).all()):
        raise ValueError("revision corruption requires a nondegenerate decoded cell")
    return value.numpy()


def _encode(arrays: dict, vocab: Mapping[str, int], config: FixedSlotConfig) -> list[int]:
    tokens, _ = arrays_to_dynamic_tokens(
        arrays["lengths"], arrays["angles"], arrays["species"], arrays["frac_coords"],
        config=config,
    )
    return [int(vocab[token]) for token in tokens]


def _cell_corruption(clean, arrays, rng, vocab, config, constraints):
    lattice = _lattice(clean, constraints)
    noise = rng.uniform(-_CELL_STRAIN_BOUND, _CELL_STRAIN_BOUND, size=(3, 3))
    strain = (noise + noise.T) / 2
    lengths, angles = lattice_parameters_from_matrix(lattice @ (np.eye(3) + strain))
    encoded = _encode({**arrays, "lengths": lengths, "angles": angles}, vocab, config)
    old = list(clean)
    old[1:7] = encoded[1:7]
    fallback = None
    try:
        _lattice(old, constraints)
    except ValueError:
        fallback = "strain_quantized_to_degenerate_cell"
    if old[1:7] == clean[1:7]:
        fallback = "strain_below_codec_resolution"
    if fallback is not None:
        # The smallest representable uniaxial strain. Keeping the native
        # angles avoids rejecting sources near a quantized cell boundary.
        lengths = list(arrays["lengths"])
        axis = int(np.argmax(lengths))
        direction = -1 if lengths[axis] >= config.length_max_bin * config.length_step else 1
        lengths[axis] += direction * config.length_step
        encoded = _encode({**arrays, "lengths": lengths}, vocab, config)
        old[1:7] = encoded[1:7]
        _lattice(old, constraints)
    return old, {"cell_strain_bound": _CELL_STRAIN_BOUND, "cell_fallback": fallback}


def _coordinate_corruption(base, arrays, slots, rng, vocab, config, constraints):
    lattice = _lattice(base, constraints)
    fractional = np.asarray(arrays["frac_coords"], dtype=float).copy()
    shared = rng.uniform(-_SHARED_DISPLACEMENT_BOUND_A, _SHARED_DISPLACEMENT_BOUND_A, 3)
    displacement = shared + rng.uniform(
        -_LOCAL_DISPLACEMENT_BOUND_A, _LOCAL_DISPLACEMENT_BOUND_A, (len(slots), 3),
    )
    fractional[list(slots)] += np.linalg.solve(lattice.T, displacement.T).T
    encoded = _encode({**arrays, "frac_coords": fractional}, vocab, config)
    positions = [p for site in slots for p in coordinate_positions(site)]
    maps = constraints["coord_token_to_bin"]
    period = config.coord_max_bin
    changed_geometry = any(
        maps[axis][encoded[p]] % period != maps[axis][base[p]] % period
        for site in slots for axis, p in zip("XYZ", coordinate_positions(site))
    )
    fallback = None
    if not changed_geometry:
        # Quantization can erase small Cartesian noise, especially in a large
        # cell. Use one native grid step, still through the existing codec.
        fallback = "displacement_below_codec_resolution"
        fractional = np.asarray(arrays["frac_coords"], dtype=float).copy()
        site, axis = int(slots[0]), int(np.argmin(np.linalg.norm(lattice, axis=1)))
        fractional[site, axis] += 1.0 / period
        encoded = _encode({**arrays, "frac_coords": fractional}, vocab, config)
    old = list(base)
    for p in positions:
        old[p] = encoded[p]
    return old, {
        "shared_displacement_bound_A": _SHARED_DISPLACEMENT_BOUND_A,
        "local_displacement_bound_A": _LOCAL_DISPLACEMENT_BOUND_A,
        "coordinate_fallback": fallback,
    }


def _mask_token_id(tokenizer, vocab) -> int:
    for attribute in ("mask_token_id", "mask_id"):
        value = getattr(tokenizer, attribute, None)
        if value is not None:
            return int(value)
    for token in ("<|mdm_mask|>", "<MASK>", "[MASK]"):
        if token in vocab:
            return int(vocab[token])
    raise ValueError("tokenizer must expose a mask token ID")


def make_state_revision_example(
    row: Mapping[str, Any], tokenizer: Any, constraints: dict, *, seed: int, epoch: int = 0,
) -> dict[str, Any]:
    """Sample one scalar, alternating phases by ``(source_row_idx + epoch) % 2``.

    Rows use the existing clean ``answer``, ``plan_state``, and recorded
    ``species_program`` / ``species_program_source``. Indices are body-relative.
    Closure first samples a reverse-species block uniformly, then a scalar
    uniformly within it; cooperative samples uniformly over its joint action.
    Only conditioning/provenance metadata is copied. Invalid row schemas raise
    an error; runtime-illegal teacher tokens are NOT filtered or canonicalized.
    The trainer must check and report runtime support before choosing its loss.
    """
    source_index, seed, epoch = int(row["source_row_idx"]), int(seed), int(epoch)
    if min(source_index, seed, epoch) < 0:
        raise ValueError("source_row_idx, seed and epoch must be nonnegative")
    if int(constraints.get("body_offset", 0)) != 0:
        raise ValueError("state revision requires body-relative constraints (body_offset=0)")
    config = FixedSlotConfig(
        length_step=float(constraints.get("length_step", 0.1)),
        coord_max_bin=int(constraints.get("coord_period", 100)),
    )
    arrays = parse_dynamic_answer(str(row["answer"]), strict=True, config=config)
    plan = row["plan_state"]
    order = row["species_program"]
    if not isinstance(plan, Mapping) or not isinstance(order, (list, tuple)) or not order:
        raise ValueError("row requires plan_state and a recorded species_program")
    program = program_from_element_order(
        plan, order, order_source=str(row.get("species_program_source") or "source_row_unspecified"),
    )
    if arrays["num_atoms"] != program.num_atoms:
        raise ValueError("teacher atom count disagrees with source Plan")
    for entry in program.entries:
        if any(arrays["species"][site] != entry.symbol for site in entry.slot_indices):
            raise ValueError("teacher element slots disagree with program; atom reordering is forbidden")
    vocab = tokenizer.get_vocab()
    clean = [int(vocab[token]) for token in arrays["tokens"]]
    mask_id = _mask_token_id(tokenizer, vocab)
    rng = np.random.default_rng(np.random.SeedSequence([seed, source_index, epoch]))
    phase = "cooperative" if (source_index + epoch) % 2 == 0 else "closure"
    info: dict[str, Any] = {
        "seed": seed, "epoch": epoch, "source_row_idx": source_index,
        "region_fallback": False, "cell_fallback": None,
        "target_source": "same_source_mp20_clean_native",
    }
    if phase == "cooperative":
        cell_only, cell_info = _cell_corruption(clean, arrays, rng, vocab, config, constraints)
        # No target-defined region: S0 is selected AFTER quantized cell strain,
        # with native fractional coordinates untouched (including 000/100).
        slots = cooperative_slots(torch.tensor(cell_only), program, constraints)
        proposed, coordinate_info = _coordinate_corruption(
            cell_only, arrays, slots, rng, vocab, config, constraints,
        )
        proposed_slots = cooperative_slots(torch.tensor(proposed), program, constraints)
        fallback = proposed_slots != slots  # order is part of the transaction
        old = cell_only if fallback else proposed
        info.update(cell_info)
        info.update(coordinate_info)
        info.update({
            "region_before_coordinate_corruption": list(slots),
            "region_after_coordinate_corruption": list(proposed_slots),
            "region_fallback": fallback,
            "region_fallback_reason": "coordinate_corruption_changed_region" if fallback else None,
            "coordinate_corruption_applied": not fallback,
        })
        transaction = [*LATTICE_POSITIONS, *(p for site in slots for p in coordinate_positions(site))]
    else:
        blocks = reverse_species_block_revision_slots(program)
        block_index = int(rng.integers(len(blocks)))
        slots = blocks[block_index]
        old, coordinate_info = _coordinate_corruption(
            clean, arrays, slots, rng, vocab, config, constraints,
        )
        info.update(coordinate_info)
        info.update({"block_index": block_index, "coordinate_corruption_applied": True})
        transaction = [p for site in slots for p in coordinate_positions(site)]

    step = int(rng.integers(len(transaction)))
    position = transaction[step]
    canvas = old.copy()
    for p in transaction:
        canvas[p] = mask_id
    for p in transaction[:step]:
        canvas[p] = clean[p]
    changed = [p for p, (a, b) in enumerate(zip(old, clean)) if a != b]
    protected = {0, *(element_position(i) for i in range(program.num_atoms))}
    if not changed or not set(changed) <= set(transaction) or protected.intersection(transaction):
        raise RuntimeError(f"source {source_index}: corruption violated transaction or exact N/E")
    info.update({"active_slots": list(slots), "changed_positions": changed})
    example = {key: deepcopy(row[key]) for key in _METADATA_KEYS if key in row}
    example.update({
        "schema": "state_revision_v1", "source_row_idx": source_index,
        "old_body": old.copy(), "input_body": canvas,
        "position": position, "target_token": clean[position],
        "transaction_positions": transaction, "transaction_step": step,
        "phase": phase, "num_atoms": program.num_atoms, "corruption_info": info,
        "outcomes_read": False, "runtime_support_checked": False,
        "requires_runtime_support_check": True,
    })
    return example
