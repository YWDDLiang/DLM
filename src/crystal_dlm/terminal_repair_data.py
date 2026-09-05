"""Geometry targets from native, train-generated paths and verified terminals."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import hashlib
import math

import numpy as np
import torch

from crystal_dlm.dynamic_crystal import arrays_to_dynamic_tokens, parse_dynamic_answer
from crystal_dlm.fixed_slot import FixedSlotConfig
from crystal_dlm.programmed_path_runtime import (
    complete_geometry_supported, full_cell_transaction_positions, replay_scalar_states,
)
from crystal_dlm.spad_program import program_from_element_order


CONDITION_KEYS = ("group_id", "source_row_idx", "source_split", "prompt", "plan_state",
                  "species_program", "species_program_source")
TARGET_ADMISSION = {
    "requires_common_quantized_relaxation_verified": True,
    "max_quantized_gap_eV_atom": .05,
    "max_quantized_force_eV_A": 1.0,
    "max_quantized_stress_GPa": 5.0,
    "max_target_minus_parent_energy_eV_atom": .001,
}
STRUCTURED_NOISE_LEVELS = ((.01, .05), (.03, .15), (.06, .30))  # strain operator norm, Cartesian A


def composition_key(record):
    program = program_from_element_order(record["plan_state"], record["species_program"],
                                        order_source=record["species_program_source"])
    counts = Counter()
    for entry in program.entries:
        counts[entry.symbol] += len(entry.slot_indices)
    divisor = math.gcd(*counts.values())
    return "|".join(f"{element}:{counts[element] // divisor}" for element in sorted(counts))


def repair_split(record):
    key = composition_key(record)
    digest = hashlib.sha256(("periodic-repair-holdout-v1:" + key).encode()).digest()
    return "validation" if int.from_bytes(digest[:8], "big") % 10 == 0 else "train"


def encode_terminal_pair(record, label, tokenizer, constraints):
    """No geometry is chosen using evaluation scores or alternative candidates."""
    if record.get("source_split") != "train" or label.get("source_split") != "train":
        raise ValueError("only originally train-generated structures may supply repair targets")
    if str(record["trajectory_id"]) != str(label["trajectory_id"]):
        raise ValueError("parent/label occurrence mismatch")
    if not record.get("success"):
        raise ValueError("parent_generation_failure")
    if label.get("verified") is not True or label.get("status") != "verified":
        raise ValueError("parent_terminal_unverified")
    if label.get("terminal_consistency", {}).get("status") != "consistent":
        raise ValueError("parent_terminal_consistency_missing")
    if label.get("endpoint_cache_key") != hashlib.sha256(record["body"].encode()).hexdigest():
        raise ValueError("parent_body_and_label_fingerprint_differ")
    original = parse_dynamic_answer(record["body"], strict=True)
    from pymatgen.core import Structure
    terminal = Structure.from_dict(label["final_structure"])
    species = [str(site.specie.symbol) for site in terminal]
    if species != list(original["species"]):
        raise ValueError("terminal_changed_native_atom_order_or_species")
    values = [*terminal.lattice.abc, *terminal.lattice.angles, *terminal.frac_coords.reshape(-1)]
    if not np.isfinite(values).all():
        raise ValueError("nonfinite_terminal_geometry")
    tokens, diagnostics = arrays_to_dynamic_tokens(
        terminal.lattice.abc, terminal.lattice.angles, species, terminal.frac_coords,
        config=FixedSlotConfig(),
    )
    if diagnostics.length_clips or diagnostics.angle_clips or diagnostics.coord_clips:
        raise ValueError("terminal_tokenization_clipped")
    tokens = [token.replace("_100>", "_000>") if token.startswith(("<X_", "<Y_", "<Z_"))
              else token for token in tokens]
    vocab = tokenizer.get_vocab()
    target = [int(vocab[token]) for token in tokens]
    parent = [int(vocab[token]) for token in original["tokens"]]
    if record.get("final_body_token_ids") != parent:
        raise ValueError("parent_body_token_identity_mismatch")
    if not complete_geometry_supported(torch.tensor(target), constraints):
        raise ValueError("quantized_terminal_outside_native_support")
    program = program_from_element_order(record["plan_state"], record["species_program"],
                                        order_source=record["species_program_source"])
    if len(target) != len(parent) or len(target) != 7 + 4 * program.num_atoms:
        raise ValueError("target_changed_native_length")
    for pos in (0, *(7 + 4 * i for i in range(program.num_atoms))):
        if target[pos] != parent[pos]:
            raise ValueError("target_changed_native_composition")
    return {
        **{key: record[key] for key in CONDITION_KEYS},
        "schema": "native_verified_terminal_repair_pair_v1",
        "trajectory_id": "quantized-terminal:" + str(record["trajectory_id"]),
        "parent_trajectory_id": str(record["trajectory_id"]),
        "repair_split": repair_split(record), "composition_key": composition_key(record),
        "parent_checkpoint": record["checkpoint"],
        "parent_body_token_ids": parent, "target_body_token_ids": target,
        "parent_trace": record["trace"], "num_atoms": program.num_atoms,
        "parent_raw_energy": float(label["raw_energy"]),
        "parent_terminal_energy": float(label["terminal_energy"]),
        "parent_gap": float(label["gap"]), "parent_verified": True,
        "body": " ".join(tokens), "success": True,
        "tokenization": asdict(diagnostics),
        "quantized_target_verified": False,
        "target_supervision_ready": False,
    }


def target_admission(pair, label):
    if label.get("trajectory_id") != pair["trajectory_id"]:
        raise ValueError("quantized target label identity differs")
    if label.get("endpoint_cache_key") != hashlib.sha256(pair["body"].encode()).hexdigest():
        raise ValueError("quantized target body and label fingerprint differ")
    reasons = []
    if label.get("verified") is not True or label.get("status") != "verified":
        reasons.append("quantized_relaxation_unverified")
    raw = label.get("raw") or {}
    checks = (
        (label.get("gap"), TARGET_ADMISSION["max_quantized_gap_eV_atom"], "quantized_gap"),
        (raw.get("force_max_eV_A"), TARGET_ADMISSION["max_quantized_force_eV_A"], "quantized_force"),
        (raw.get("stress_max_GPa"), TARGET_ADMISSION["max_quantized_stress_GPa"], "quantized_stress"),
    )
    for value, bound, reason in checks:
        if value is None or not math.isfinite(value) or value > bound:
            reasons.append(reason)
    energy = label.get("raw_energy")
    if (energy is None or not math.isfinite(energy)
            or energy - pair["parent_raw_energy"] > TARGET_ADMISSION["max_target_minus_parent_energy_eV_atom"]):
        reasons.append("target_does_not_lower_parent_energy")
    return reasons


def make_terminal_repair_example(pair, *, family, epoch, seed, mask_id,
                                 tokenizer=None, constraints=None):
    """One true conditional state; cycle equally over length, angle and XYZ.

The target prefix is teacher-forced. All remaining numeric positions are masked;
the complete erroneous source remains in old_body, exactly as at deployment.
    """
    if pair.get("target_supervision_ready") is not True:
        raise ValueError("quantized target has not passed its separate admission")
    program = program_from_element_order(pair["plan_state"], pair["species_program"],
                                        order_source=pair["species_program_source"])
    positions = full_cell_transaction_positions(program)
    if family not in (0, 1, 2):
        raise ValueError("family must be length, angle, or coordinate")
    candidates = ([p for p in positions if p in (1, 2, 3)] if family == 0
                  else [p for p in positions if p in (4, 5, 6)] if family == 1
                  else [p for p in positions if p >= 8])
    key = f"{seed}:{epoch}:{pair['parent_trajectory_id']}:{family}".encode()
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest()[:8], "big"))
    position = int(rng.choice(candidates))
    target = pair["target_body_token_ids"]
    old = list(pair["parent_body_token_ids"])
    phase, noise_level, fallback = "full_cell_repair", -1., False
    # Exactly one quarter of per-family examples over four epochs uses a
    # declared geometric corruption; the other three quarters uses real errors.
    if tokenizer is not None and (epoch + family) % 4 == 0:
        if constraints is None:
            raise ValueError("structured noise requires the deployment support")
        from crystal_dlm.manifold_corruption import (
            lattice_matrix_from_parameters, lattice_parameters_from_matrix, symmetric_matrix_exp,
        )
        clean = parse_dynamic_answer(pair["body"], strict=True)
        lattice = lattice_matrix_from_parameters(clean["lengths"], clean["angles"])
        index = int(rng.integers(len(STRUCTURED_NOISE_LEVELS)))
        strain, displacement = STRUCTURED_NOISE_LEVELS[index]
        symmetric = rng.normal(size=(3, 3))
        symmetric = (symmetric + symmetric.T) / 2
        symmetric *= strain / max(float(np.linalg.norm(symmetric, ord=2)), 1.)
        noisy_lattice = lattice @ symmetric_matrix_exp(symmetric)
        cartesian = rng.uniform(-displacement, displacement, size=(program.num_atoms, 3))
        fractional = np.mod(np.asarray(clean["frac_coords"]) + cartesian @ np.linalg.inv(noisy_lattice), 1.)
        lengths, angles = lattice_parameters_from_matrix(noisy_lattice)
        tokens, diagnostics = arrays_to_dynamic_tokens(lengths, angles, clean["species"], fractional)
        tokens = [token.replace("_100>", "_000>") if token.startswith(("<X_", "<Y_", "<Z_"))
                  else token for token in tokens]
        vocabulary = tokenizer.get_vocab()
        noisy = [int(vocabulary[token]) for token in tokens]
        if (not diagnostics.length_clips and not diagnostics.angle_clips
                and complete_geometry_supported(torch.tensor(noisy), constraints)):
            old, phase, noise_level = noisy, "structured_denoise", (index + 1) / 3
        else:
            fallback = True  # retain this occurrence as its actual parent error
    current = old.copy()
    for pos in positions:
        current[pos] = int(mask_id)
    for pos in positions:
        if pos == position:
            break
        current[pos] = target[pos]
    states = list(replay_scalar_states(pair["parent_trace"]))
    if not states:
        raise ValueError("self-generated source has no actual deployment states")
    reference = states[int(rng.integers(len(states)))]
    metadata = {key: pair[key] for key in CONDITION_KEYS}
    return {
        **metadata, "input_body": current, "old_body": old,
        "target_token": int(target[position]), "position": position,
        "transaction_positions": list(positions), "num_atoms": program.num_atoms,
        "phase": phase, "family": family, "numeric_noise_level": noise_level,
        "structured_corruption_fallback": fallback,
        "reference_state": {**metadata, **reference, "num_atoms": program.num_atoms},
        "sample_weight": float(pair.get("sample_weight", 1.)),
    }
