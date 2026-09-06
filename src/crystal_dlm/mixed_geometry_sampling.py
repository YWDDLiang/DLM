"""H-P33 inference bookkeeping with fixed logical batches and separate float/Q outputs.

No model loading, dataset selection, energy or distance filter occurs here.
Known row failures become unavailable outputs; internal padding never becomes a
replacement crystal. Unattributed model/integrator errors fail the remaining
logical batch without a neural retry. The frozen numerical sampler is reused.
"""
from __future__ import annotations

from collections import Counter
from contextlib import nullcontext
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from crystal_dlm.dynamic_crystal import arrays_to_dynamic_tokens, arrays_to_structure, parse_dynamic_answer
from crystal_dlm.fixed_slot import SYMBOL_TO_Z
from crystal_dlm.mixed_geometry_diffusion import (
    DEFAULT_CONFIG, GeometrySamplingError, GeometryState, sample_geometry, sample_prior,
)
from crystal_dlm.periodic_base_training_data import _canonical_tokens
from crystal_dlm.programmed_path_data import path_seed
from crystal_dlm.state_conditioned_model import context_from_programs


METHOD = "mixed_geometry_h_p33"


def validate_final_policy(checkpoint_path):
    """Read the new trainer's final endpoint, independently of model reconstruction."""
    root = Path(checkpoint_path).resolve()
    train = root.parent.parent
    checkpoint = json.loads((root / "CHECKPOINT_FINAL.json").read_text(encoding="utf-8"))
    final = json.loads((train / "TRAIN_FINAL.json").read_text(encoding="utf-8"))
    if (not (train / "_SUCCESS").is_file() or checkpoint.get("method") != METHOD or final.get("method") != METHOD
            or checkpoint.get("eligible_policy") is not True or final.get("eligible_policy") is not True
            or checkpoint.get("completed_epochs") != checkpoint.get("expected_epochs")
            or checkpoint.get("completed_epochs") != final.get("epochs")
            or checkpoint.get("completed_step") != final.get("updates")
            or int(final.get("updates", 0)) <= 0 or int(final.get("epochs", 0)) <= 0
            or Path(final["policy_path"]).resolve() != root):
        raise ValueError("mixed geometry sampling requires its completed, declared final policy")
    return {"checkpoint": checkpoint, "training": final}


def scaffold_for(compiled, *, mask_id, max_sites, device, max_length=382):
    shapes = {(c["program"].num_atoms, len(c["prompt_token_ids"])) for c in compiled}
    if not compiled or len(shapes) != 1:
        raise ValueError("one logical batch must have the frozen common N and prompt length")
    count, prompt_length = next(iter(shapes))
    if prompt_length+7+4*count > max_length or count > max_sites:
        raise ValueError("compiled geometry scaffold exceeds its declared dimensions")
    x = torch.tensor([c["prompt_token_ids"]+c["initial_body"] for c in compiled], dtype=torch.long, device=device)
    numeric = list(range(1, 7))+[8+4*i+axis for i in range(count) for axis in range(3)]
    if not bool((x[:, [prompt_length+p for p in numeric]] == mask_id).all()):
        raise ValueError("geometry sampling must begin with MASK-only numeric scaffold slots")
    programs = [c["program"] for c in compiled]
    context = context_from_programs(x, prompt_length=prompt_length, num_sites=count, programs=programs,
                                    active_positions={row: numeric for row in range(len(compiled))}, max_sites=max_sites)
    species = torch.zeros((len(compiled), max_sites), dtype=torch.long, device=device)
    symbols = []
    for row, program in enumerate(programs):
        entries = [None]*count
        for entry in program.entries:
            for slot in entry.slot_indices:
                species[row, slot] = SYMBOL_TO_Z[entry.symbol]
                entries[slot] = entry.symbol
        if any(symbol is None for symbol in entries):
            raise ValueError("program did not cover every canonical atom slot")
        symbols.append(entries)
    mask = torch.arange(max_sites, device=device)[None] < torch.full((len(compiled), 1), count, device=device)
    return x, torch.ones_like(x), context, species, mask, symbols


def keyed_cpu_prior(seeds, counts, *, max_sites, device):
    """Draw each source separately in CPU FP64, then pad and transfer without re-drawing."""
    z = torch.empty((len(seeds), 6), dtype=torch.float64)
    fractional = torch.zeros((len(seeds), max_sites, 3), dtype=torch.float64)
    mask = torch.zeros((len(seeds), max_sites), dtype=torch.bool)
    receipts = []
    for row, (seed, count) in enumerate(zip(seeds, counts, strict=True)):
        if not 1 <= int(count) <= max_sites:
            raise ValueError("prior atom count exceeds the supplied scaffold")
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        prior = sample_prior(torch.ones(int(count), dtype=torch.bool), generator=generator)
        z[row] = prior.z
        fractional[row, :count] = prior.fractional
        mask[row, :count] = True
        raw = prior.z.contiguous().numpy().tobytes()+prior.fractional.contiguous().numpy().tobytes()
        receipts.append({"seed": int(seed), "dtype": "float64", "device": "cpu", "num_atoms": int(count),
                         "sha256": hashlib.sha256(raw).hexdigest(),
                         "z": prior.z.tolist(), "fractional": prior.fractional.tolist()})
    state = GeometryState(z.to(device), fractional.to(device), torch.ones(len(seeds), dtype=torch.float64, device=device), mask.to(device))
    return state, receipts


class _AllRowsUnavailable(RuntimeError):
    pass


@torch.no_grad()
def run_joint_geometry(model, initial, input_ids, attention_mask, context, species):
    """Keep logical batch shape while quarantining attributable numerical failures."""
    batch = input_ids.shape[0]
    failures = [None]*batch
    live_calls, padding_calls = [0]*batch, [0]*batch
    callbacks = model_calls = 0
    last_time = 1.

    def fail(row, *, scope, stage, message):
        if failures[row] is None:
            failures[row] = {"scope": scope, "stage": stage, "message": str(message), "time": last_time,
                             "attempted_field_nfe": callbacks, "model_nfe_at_failure": live_calls[row]}

    def field(state):
        nonlocal callbacks, model_calls, last_time
        callbacks += 1
        last_time = float(state.t.reshape(-1)[0])
        network = state.as_dtype(torch.float32)  # Does not modify FP64 integration state.
        for row in range(batch):
            if failures[row] is not None:
                continue
            try:
                if not bool(torch.isfinite(network.z[row]).all() and torch.isfinite(network.fractional[row][state.atom_mask[row]]).all()):
                    raise FloatingPointError("state is nonfinite after FP32 network input conversion")
                lattice = model.normalizer.decode(network.z[row], state.num_atoms[row]).float()
                if not bool(torch.isfinite(lattice).all() and torch.isfinite(lattice @ lattice.T).all()):
                    raise FloatingPointError("decoded FP32 lattice or metric is nonfinite")
            except (ValueError, FloatingPointError, RuntimeError) as error:
                fail(row, scope="row", stage="geometry_input", message=f"{type(error).__name__}: {error}")
        if all(error is not None for error in failures):
            raise _AllRowsUnavailable("all logical batch rows have recorded numerical failures")
        dead = torch.tensor([error is not None for error in failures], dtype=torch.bool, device=state.z.device)
        # These values only occupy already-failed rows in a fixed-size neural
        # batch. They are never candidates and never returned as successful.
        network = GeometryState(torch.where(dead[:, None], 0., network.z),
                                torch.where(dead[:, None, None], 0., network.fractional), network.t, network.atom_mask)
        model_calls += 1
        for row in range(batch):
            if failures[row] is None:
                live_calls[row] += 1
            else:
                padding_calls[row] += 1
        try:
            autocast = torch.autocast("cuda", dtype=torch.bfloat16) if input_ids.device.type == "cuda" else nullcontext()
            with autocast:
                output = model(input_ids, attention_mask=attention_mask, mode="geometry", geometry_context=context,
                               geometry_state=network, species=species)
            v, u = output.v_prediction.detach(), output.u_prediction.detach()
            if v.shape != state.z.shape or u.shape != state.fractional.shape or v.device != state.z.device or u.device != state.z.device:
                raise ValueError("geometry field outputs differ from the fixed logical batch state")
            good = torch.isfinite(v).all(-1) & torch.where(state.atom_mask[..., None], torch.isfinite(u), True).all(dim=(-1, -2))
            for row in range(batch):
                if not bool(good[row]):
                    fail(row, scope="row", stage="model_output", message="nonfinite geometry field")
        except Exception as error:
            for row in range(batch):
                fail(row, scope="logical_batch", stage="model_forward", message=f"{type(error).__name__}: {error}")
            raise
        if all(error is not None for error in failures):
            raise _AllRowsUnavailable("all logical batch rows have recorded field failures")
        live = torch.tensor([error is None for error in failures], dtype=torch.bool, device=state.z.device)
        return torch.where(live[:, None], v, 0.), torch.where(live[:, None, None] & state.atom_mask[..., None], u, 0.)

    sample = None
    try:
        sample = sample_geometry(field, initial, config=model.diffusion_config)
    except GeometrySamplingError as error:
        for row in range(batch):
            fail(row, scope="logical_batch", stage="sampler_step", message=f"{type(error.__cause__).__name__}: {error}")
    except Exception as error:
        for row in range(batch):
            fail(row, scope="logical_batch", stage="sampler_initial_state", message=f"{type(error).__name__}: {error}")
    return sample, failures, live_calls, padding_calls, {"attempted_field_callbacks": callbacks,
            "model_forward_calls": model_calls, "model_row_evaluations": model_calls*batch,
            "live_row_model_evaluations": sum(live_calls), "failed_row_padding_evaluations": sum(padding_calls)}


def _checked_structure(lattice, fractional, symbols):
    from pymatgen.core import Structure
    lattice, fractional = np.asarray(lattice, dtype=np.float64), np.asarray(fractional, dtype=np.float64)
    if lattice.shape != (3, 3) or fractional.shape != (len(symbols), 3):
        raise ValueError("output crystal dimensions changed")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        volume = float(np.linalg.det(lattice))
    if not np.isfinite(lattice).all() or not np.isfinite(fractional).all() or not math.isfinite(volume) or volume <= 0:
        raise ValueError("output geometry is nonfinite or not a positive-volume cell")
    structure = Structure(lattice, symbols, fractional, coords_are_cartesian=False)
    cif = structure.to(fmt="cif", significant_figures=16)
    parsed = Structure.from_str(cif, fmt="cif")
    if parsed.num_sites != len(symbols) or Counter(parsed.atomic_numbers) != Counter(SYMBOL_TO_Z[s] for s in symbols):
        raise ValueError("CIF roundtrip changed exact composition")
    return structure, cif


def paired_representations(structure, symbols, tokenizer):
    """Create Q only from the exact same float output, retaining clip failures."""
    tokens, diagnostics = arrays_to_dynamic_tokens(structure.lattice.abc, structure.lattice.angles, symbols, structure.frac_coords)
    tokens, aliases = _canonical_tokens(tokens)
    body = "".join(tokens)
    diagnostic = {**asdict(diagnostics), "periodic_alias_tokens_canonicalized": aliases}
    preview = {"body": body}
    if diagnostics.length_clips or diagnostics.angle_clips or diagnostics.coord_clips:
        return {"success": False, "body": None, "structure": None, "cif": None, "final_body_token_ids": None,
                "failure": "Q codec clipped a lattice/coordinate field", "quantization": diagnostic,
                "diagnostic_candidate": preview}
    try:
        arrays = parse_dynamic_answer(body, strict=True)
        if list(arrays["species"]) != list(symbols):
            raise ValueError("Q changed canonical atom identities")
        quantized = arrays_to_structure(arrays)
        quantized, cif = _checked_structure(quantized.lattice.matrix, quantized.frac_coords, symbols)
        return {"success": True, "body": body, "structure": quantized.as_dict(), "cif": cif,
                "final_body_token_ids": [int(tokenizer.get_vocab()[token]) for token in tokens],
                "failure": None, "quantization": diagnostic}
    except Exception as error:
        return {"success": False, "body": None, "structure": None, "cif": None, "final_body_token_ids": None,
                "failure": f"{type(error).__name__}: {error}", "quantization": diagnostic, "diagnostic_candidate": preview}


@torch.no_grad()
def sample_compiled_batch(model, tokenizer, batch, *, device, checkpoint_path,
                          seed=20260905, collection_round=0, condition_start=0,
                          layout_world_size=2, max_length=382):
    """Return one float and one Q ledger row per original logical batch member."""
    compiled = [item[2] for item in batch]
    if any(candidate != 0 for _, candidate, _ in batch):
        raise ValueError("H-P33 has one fixed occurrence per condition")
    if asdict(model.diffusion_config) != asdict(DEFAULT_CONFIG):
        raise ValueError("formal H-P33 sampling requires the registered numerical configuration")
    x, attention, context, species, mask, symbols = scaffold_for(compiled, mask_id=model.mask_id,
            max_sites=model.state_config.max_sites, device=device, max_length=max_length)
    counts = [c["program"].num_atoms for c in compiled]
    seeds = [path_seed(seed, c["record"]["group_id"], collection_round, 0) for c in compiled]
    initial, priors = keyed_cpu_prior(seeds, counts, max_sites=model.state_config.max_sites, device=device)
    if not torch.equal(mask, initial.atom_mask):
        raise ValueError("prior and fixed scaffold atom masks differ")
    sample, failures, nfes, padding, statistics = run_joint_geometry(model, initial, x, attention, context, species)
    native, quantized = [], []
    time_grid = tuple(DEFAULT_CONFIG.epsilon+(1-DEFAULT_CONFIG.epsilon)*(i/DEFAULT_CONFIG.euler_steps)**2
                      for i in range(DEFAULT_CONFIG.euler_steps, -1, -1))
    for row, (ordinal, candidate, c) in enumerate(batch):
        common = {**c["record"], "prompt": c["prompt"], "prompt_token_ids": c["prompt_token_ids"],
                  "condition_ordinal": int(ordinal), "evaluation_ordinal": int(ordinal)-int(condition_start),
                  "candidate_index": candidate, "collection_round": int(collection_round),
                  "trajectory_id": f"{c['record']['group_id']}:{collection_round}:{candidate}",
                  "sampling_seed": int(seeds[row]), "sampling_batch_size": len(batch),
                  "sampling_layout_world_size": int(layout_world_size), "num_atoms": counts[row],
                  "checkpoint": str(checkpoint_path), "method": METHOD, "endpoint": "native",
                  "sampling_nfe": int(nfes[row]), "sampling_failure": failures[row],
                  "failed_row_padding_evaluations": int(padding[row]), "initial_geometry_prior": priors[row],
                  "sampler_time_grid": list(time_grid), "body": None, "structure": None, "cif": None,
                  "native_structure": None, "cif_path": None, "final_body_token_ids": None,
                  "success": False, "parseable": False, "inference_mlip": False}
        result = dict(common, native_source="structure", output_representation="continuous_float_primary")
        if failures[row] is None and sample is not None:
            try:
                lattice = model.normalizer.decode(sample.z[row], counts[row]).cpu().numpy()
                fractional = sample.fractional[row, :counts[row]].cpu().numpy()
                structure, cif = _checked_structure(lattice, fractional, symbols[row])
                result.update(success=True, parseable=True, structure=structure.as_dict(), native_structure=structure.as_dict(), cif=cif, failure=None)
            except Exception as error:
                result["failure"] = f"{type(error).__name__}: {error}"
                result["sampling_failure"] = {"scope": "row", "stage": "native_decode", "message": result["failure"],
                                               "attempted_field_nfe": statistics["attempted_field_callbacks"],
                                               "model_nfe_at_failure": nfes[row], "time": DEFAULT_CONFIG.epsilon}
                result["diagnostic_candidate"] = {"z": sample.z[row].cpu().tolist(),
                                                    "fractional": sample.fractional[row, :counts[row]].cpu().tolist()}
        else:
            result["failure"] = failures[row]["message"] if failures[row] is not None else "geometry sample unavailable"
        result["native_execution_success"] = result["success"]
        result["trace"] = {"schema": "mixed_geometry_sampling_v1", "success": result["success"], "failure": result["failure"],
                           "neural_nfe": nfes[row], "scope": "continuous_geometry_H_P33_no_token_log_probability"}
        native.append(result)
        secondary = dict(common, native_source="body", output_representation="Q_raw_secondary",
                         continuous_native_success=result["success"], additional_neural_nfe=0,
                         sampling_failure=result["sampling_failure"], quantization=None,
                         continuous_parent_trajectory_id=result["trajectory_id"])
        if result["success"]:
            try:
                q = paired_representations(structure, symbols[row], tokenizer)
            except Exception as error:
                q = {"success": False, "body": None, "structure": None, "cif": None,
                     "final_body_token_ids": None, "failure": f"{type(error).__name__}: {error}"}
            secondary.update(q)
        else:
            secondary["failure"] = "continuous parent failed: "+result["failure"]
        secondary.update(parseable=secondary["success"], native_execution_success=secondary["success"],
                         native_structure=secondary["structure"] if secondary["success"] else None)
        secondary["trace"] = {"schema": "mixed_geometry_sampling_v1", "success": secondary["success"],
                              "failure": secondary["failure"], "neural_nfe": nfes[row],
                              "scope": "same_continuous_sample_followed_by_fixed_Q"}
        quantized.append(secondary)
    statistics.update(logical_batch_size=len(batch), sample_indices=[int(c["record"]["sample_idx"]) for c in compiled],
                      native_successful=sum(record["success"] for record in native),
                      quantized_successful=sum(record["success"] for record in quantized),
                      failures_by_stage=dict(Counter(record["sampling_failure"]["stage"] for record in native if record["sampling_failure"])))
    return native, quantized, statistics


__all__ = ["METHOD", "validate_final_policy", "scaffold_for", "keyed_cpu_prior", "run_joint_geometry",
           "paired_representations", "sample_compiled_batch"]
