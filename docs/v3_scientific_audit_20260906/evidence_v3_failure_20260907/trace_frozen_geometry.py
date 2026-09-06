"""Two fixed logical batches, one original 33-field replay each; no training/MLIP.

The production model/sampler are imported from the original archived source.
A callable proxy records actual inputs/outputs without changing any method or
hook.  Recorded fields then replay the *same* numerical helpers, with zero extra
neural evaluations, to recover and verify the FP64 internal trajectory.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np


def file_hash(path):
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def select_batches(records, batch_iterator):
    """Use only original input/layout metadata, never physical outcomes."""
    items = []
    seen = set()
    for row in sorted(records, key=lambda value: value["condition_ordinal"]):
        if row["trajectory_id"] in seen:
            raise ValueError("duplicate original trajectory")
        seen.add(row["trajectory_id"])
        if row["candidate_index"] != 0 or row["sampling_layout_world_size"] != 2:
            raise ValueError("this diagnostic requires the original one-candidate layout2 contract")
        count = int(row["initial_geometry_prior"]["num_atoms"])
        placeholder = {"program": SimpleNamespace(num_atoms=count), "prompt_token_ids": row["prompt_token_ids"],
                       "record": row}
        items.append((row["condition_ordinal"], 0, placeholder))
    selected = {}
    for serial, batch in enumerate(batch_iterator(items, batch_size=4, rank=0, world_size=1, layout_world_size=2)):
        count = batch[0][2]["program"].num_atoms
        category = "first_N_le_6" if count <= 6 else "first_N_ge_16" if count >= 16 else None
        if category is None or category in selected:
            continue
        if any(item[2]["record"]["sampling_batch_size"] != len(batch) for item in batch):
            raise ValueError("reconstructed logical batch does not match recorded membership/size")
        selected[category] = {"serial": serial, "N": count, "rows": [item[2]["record"] for item in batch]}
        if len(selected) == 2:
            break
    if len(selected) != 2:
        raise ValueError("both prescribed atom-count categories must exist")
    return selected


def compact_structure_arrays(record):
    structure = record.get("structure") or record.get("native_structure")
    if not isinstance(structure, dict):
        raise ValueError("original selected output has no saved structure")
    lattice = structure["lattice"]
    if isinstance(lattice, dict):
        lattice = lattice["matrix"]
    return np.asarray(lattice, dtype=np.float64), np.asarray([site["abc"] for site in structure["sites"]], dtype=np.float64)


def decomposition(displacement):
    """Equal-site centroid translation, not mass-weighted physical COM."""
    mean = displacement.mean(axis=0, keepdims=True)
    internal = displacement - mean
    total = float(np.square(displacement).sum())
    return {"RMS_fractional": float(np.sqrt(np.mean(np.square(displacement)))),
            "internal_RMS_fractional": float(np.sqrt(np.mean(np.square(internal)))),
            "centroid_displacement": mean[0].tolist(),
            "translation_squared_fraction": float(len(displacement)*np.square(mean).sum()/total) if total else 0.0,
            "max_abs_component": float(np.abs(displacement).max())}, internal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--native-paths", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--expected-source-commit", default="0bf8e8870f96c54e86cbd080e64c527f55fb6943")
    args = parser.parse_args()
    source_root = args.source_root.resolve()
    commit_path = source_root.parent / "CODE_COMMIT"
    if not commit_path.is_file() or commit_path.read_text().strip() != args.expected_source_commit:
        raise ValueError("source-root must be the original run/code with its matching CODE_COMMIT")
    sys.path.insert(0, str(source_root / "src"))
    import torch
    from crystal_dlm.mixed_geometry_diffusion import GeometryState, probability_flow_step, terminal_readout
    from crystal_dlm.mixed_geometry_model import load_mixed_geometry_model
    from crystal_dlm.mixed_geometry_sampling import run_joint_geometry, scaffold_for, validate_final_policy
    from crystal_dlm.programmed_path_data import compile_condition
    from crystal_dlm.sampling_layout import sampling_batches

    torch.set_num_threads(2)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    with args.native_paths.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    selected = select_batches(records, sampling_batches)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selection = {name: {"logical_serial": entry["serial"], "N": entry["N"],
                        "trajectory_ids": [row["trajectory_id"] for row in entry["rows"]],
                        "sample_indices": [row["sample_idx"] for row in entry["rows"]],
                        "original_batch_size": len(entry["rows"])} for name, entry in selected.items()}
    write_json(args.output_dir / "SELECTION.json", {"rule": "first complete layout2/cap4 batch N<=6 and N>=16",
                                                    "outcomes_used_for_selection": False,
                                                    "native_paths_sha256": file_hash(args.native_paths), "batches": selection})
    validate_final_policy(args.checkpoint_path)
    model, tokenizer = load_mixed_geometry_model(args.model_path, args.checkpoint_path, device, trainable=False)
    model.requires_grad_(False).eval()
    max_sites = model.state_config.max_sites
    config = model.diffusion_config
    if config.euler_steps != 32 or config.epsilon != .002:
        raise ValueError("this probe must keep the original H-P33 numerical configuration")

    class Recorder:
        def __init__(self, underlying):
            self.underlying = underlying
            self.records = []

        def __getattr__(self, name):
            return getattr(self.underlying, name)

        def __call__(self, *positional, **keywords):
            state = keywords["geometry_state"]
            item = {"network_z": state.z.detach().cpu().clone(),
                    "network_f": state.fractional.detach().cpu().clone(),
                    "network_t": state.t.detach().cpu().clone()}
            output = self.underlying(*positional, **keywords)
            item.update(v=output.v_prediction.detach().cpu().clone(), u=output.u_prediction.detach().cpu().clone())
            self.records.append(item)
            return output

    summaries = {}
    with torch.no_grad():
        for category, entry in selected.items():
            original_rows = entry["rows"]
            if any(Path(row["checkpoint"]).resolve() != args.checkpoint_path.resolve() or row["sampling_nfe"] != 33
                   for row in original_rows):
                raise ValueError("selected records do not belong to the declared final checkpoint/P33 execution")
            compiled = [compile_condition(row, tokenizer, mask_id=model.mask_id, purpose="evaluation") for row in original_rows]
            if any(item["prompt_token_ids"] != row["prompt_token_ids"] or item["prompt"] != row["prompt"]
                   for item, row in zip(compiled, original_rows)):
                raise ValueError("recompiled prompt differs from original sampling")
            x, attention, context, species, mask, symbols = scaffold_for(
                compiled, mask_id=model.mask_id, max_sites=max_sites, device=device)
            batch = len(original_rows)
            z = torch.empty(batch, 6, dtype=torch.float64)
            fractional = torch.zeros(batch, max_sites, 3, dtype=torch.float64)
            for index, row in enumerate(original_rows):
                prior = row["initial_geometry_prior"]
                n = int(prior["num_atoms"])
                prior_z = torch.tensor(prior["z"], dtype=torch.float64)
                prior_f = torch.tensor(prior["fractional"], dtype=torch.float64)
                payload = prior_z.numpy().tobytes() + prior_f.numpy().tobytes()
                if sha256(payload).hexdigest() != prior["sha256"]:
                    raise ValueError("stored FP64 prior bytes do not reproduce their original hash")
                if prior["seed"] != row["sampling_seed"] or n != compiled[index]["program"].num_atoms:
                    raise ValueError("prior identity/count/seed differs from original request")
                z[index] = prior_z
                fractional[index, :n] = prior_f
            initial = GeometryState(z.to(device), fractional.to(device), torch.ones(batch, dtype=torch.float64, device=device), mask)
            recorder = Recorder(model)
            sampled, failures, live_nfe, padding_nfe, statistics = run_joint_geometry(
                recorder, initial, x, attention, context, species)
            if sampled is None or any(error is not None for error in failures):
                write_json(args.output_dir / f"{category}.FAILED.json", {"failures": failures, "statistics": statistics})
                raise RuntimeError("original selected logical batch did not replay numerically; no retry")
            if len(recorder.records) != 33 or sampled.nfe != 33 or any(value != 33 for value in live_nfe):
                raise RuntimeError("replay must have exactly its original 33 neural evaluations per live row")

            # Replay recorded fields on the same device with original numerical
            # helpers.  This adds no model forward and verifies the exact states.
            state = initial
            states_z, states_f, lattices, per_step = [], [], [], []
            cumulative_internal = np.zeros(batch)
            cumulative_total = np.zeros(batch)
            cumulative_internal_A = np.zeros(batch)
            integrated_translation_square = np.zeros(batch)
            integrated_total_square = np.zeros(batch)
            maximum_network_cast_delta = 0.0
            grid = sampled.time_grid
            for index, recorded in enumerate(recorder.records):
                network = state.as_dtype(torch.float32)
                for current, saved in ((network.z, recorded["network_z"]), (network.fractional, recorded["network_f"]),
                                       (network.t, recorded["network_t"])):
                    maximum_network_cast_delta = max(maximum_network_cast_delta, float((current.cpu()-saved).abs().max()))
                if maximum_network_cast_delta != 0:
                    raise RuntimeError("recorded model inputs differ from exact numerical reconstruction")
                states_z.append(state.z.cpu().numpy().copy())
                states_f.append(state.fractional.cpu().numpy().copy())
                lattice = model.normalizer.decode(state.z, state.num_atoms).cpu().numpy()
                lattices.append(lattice)
                v, u = recorded["v"].to(device), recorded["u"].to(device)
                rows = []
                factor = float(grid[index]-grid[index+1]) if index < 32 else config.epsilon
                next_state = probability_flow_step(state, v, u, grid[index+1]) if index < 32 else None
                for row_index, original in enumerate(original_rows):
                    n = int(state.num_atoms[row_index])
                    raw_u = recorded["u"][row_index, :n].double().numpy()
                    raw_v = recorded["v"][row_index].double().numpy()
                    displacement = factor * raw_u
                    movement, internal = decomposition(displacement)
                    internal_A = float(np.sqrt(np.mean(np.sum((internal @ lattice[row_index])**2, axis=-1))))
                    total_square = float(np.square(displacement).sum())
                    translation_square = n * float(np.square(displacement.mean(0)).sum())
                    if index < 32:
                        cumulative_internal[row_index] += movement["internal_RMS_fractional"]
                        cumulative_total[row_index] += movement["RMS_fractional"]
                        cumulative_internal_A[row_index] += internal_A
                        integrated_translation_square[row_index] += translation_square
                        integrated_total_square[row_index] += total_square
                    rows.append({"trajectory_id": original["trajectory_id"], "N": n,
                                 "v_RMS": float(np.sqrt(np.mean(raw_v**2))),
                                 "u_RMS": float(np.sqrt(np.mean(raw_u**2))),
                                 "u_internal_RMS": float(np.sqrt(np.mean((raw_u-raw_u.mean(0))**2))),
                                 "movement": movement, "internal_step_RMS_A_current_cell": internal_A})
                per_step.append({"field_index": index+1, "time_FP64": float(state.t[0]),
                                 "kind": "Euler" if index < 32 else "terminal_readout", "factor": factor, "rows": rows})
                if index < 32:
                    state = next_state
            replay_z, replay_f = terminal_readout(state, recorder.records[-1]["v"].to(device), recorder.records[-1]["u"].to(device))
            if (not torch.equal(state.z, sampled.pre_readout_state.z)
                    or not torch.equal(state.fractional, sampled.pre_readout_state.fractional)
                    or not torch.equal(replay_z, sampled.z) or not torch.equal(replay_f, sampled.fractional)):
                raise RuntimeError("recorded fields did not reproduce original FP64 integrator/readout exactly")
            final_lattices = model.normalizer.decode(sampled.z, sampled.atom_mask.sum(-1)).cpu().numpy()
            final_rows = []
            for index, original in enumerate(original_rows):
                n = int(initial.num_atoms[index])
                first_f = initial.fractional[index, :n].cpu().numpy()
                before_f = state.fractional[index, :n].cpu().numpy()
                final_f = sampled.fractional[index, :n].cpu().numpy()
                net = (before_f-first_f+.5) % 1 - .5
                net_summary, _ = decomposition(net)
                final_net, _ = decomposition((final_f-first_f+.5) % 1-.5)
                original_lattice, original_f = compact_structure_arrays(original)
                fractional_error = float(np.max(np.abs((final_f-original_f+.5) % 1-.5)))
                gram = final_lattices[index] @ final_lattices[index].T
                original_gram = original_lattice @ original_lattice.T
                gram_error = float(np.linalg.norm(gram-original_gram)/np.linalg.norm(original_gram))
                ratio = (float(cumulative_internal[index])/net_summary["internal_RMS_fractional"]
                         if net_summary["internal_RMS_fractional"] > 1e-15 else None)
                final_rows.append({"trajectory_id": original["trajectory_id"], "sample_idx": original["sample_idx"], "N": n,
                                   "Euler_cumulative_internal_RMS_sum_fractional": float(cumulative_internal[index]),
                                   "Euler_cumulative_total_RMS_sum_fractional": float(cumulative_total[index]),
                                   "Euler_cumulative_internal_RMS_sum_A_changing_cell": float(cumulative_internal_A[index]),
                                   "Euler_net_movement": net_summary, "final_net_movement": final_net,
                                   "Euler_cumulative_internal_to_net_ratio": ratio,
                                   "Euler_translation_fraction_of_sum_step_squares": float(integrated_translation_square[index]/integrated_total_square[index])
                                   if integrated_total_square[index] else 0.0,
                                   "readout_movement": per_step[-1]["rows"][index]["movement"],
                                   "readout_z_RMS": float((sampled.z[index]-state.z[index]).square().mean().sqrt()),
                                   "original_final_fractional_max_error": fractional_error,
                                   "original_final_Gram_relative_error": gram_error,
                                   "original_output_reproduced": fractional_error <= 1e-10 and gram_error <= 1e-10})
            array_path = args.output_dir / f"{category}.npz"
            np.savez_compressed(array_path, state_z=np.stack(states_z), state_fractional=np.stack(states_f),
                                state_lattice=np.stack(lattices), times=np.asarray(grid),
                                v=np.stack([item["v"].numpy() for item in recorder.records]),
                                u=np.stack([item["u"].numpy() for item in recorder.records]),
                                final_z=sampled.z.cpu().numpy(), final_fractional=sampled.fractional.cpu().numpy(),
                                final_lattice=final_lattices, atom_mask=mask.cpu().numpy())
            summary = {"selection": selection[category], "statistics": statistics, "per_row_neural_nfe": live_nfe,
                       "failed_row_padding_nfe": padding_nfe, "recorded_model_calls": len(recorder.records),
                       "extra_neural_calls_for_numeric_reconstruction": 0, "network_input_reconstruction_max_error": maximum_network_cast_delta,
                       "numerical_reconstruction_exact": True, "centroid_definition": "equal-site fractional translation, not mass-weighted COM",
                       "array_file": array_path.name, "array_sha256": file_hash(array_path), "steps": per_step, "rows": final_rows}
            write_json(args.output_dir / f"{category}.json", summary)
            summaries[category] = {key: value for key, value in summary.items() if key != "steps"}
    outputs_reproduced = all(row["original_output_reproduced"] for value in summaries.values() for row in value["rows"])
    result = {"status": "PASS" if outputs_reproduced else "REPLAY_DIFFERS", "training": False, "MLIP_calls": 0,
              "new_candidate_selection": False, "source_commit": args.expected_source_commit,
              "total_neural_forward_calls": sum(value["recorded_model_calls"] for value in summaries.values()),
              "source_hashes": {relative: file_hash(source_root/relative) for relative in
                                ("src/crystal_dlm/mixed_geometry_diffusion.py", "src/crystal_dlm/mixed_geometry_model.py",
                                 "src/crystal_dlm/mixed_geometry_sampling.py")},
              "checkpoint_path": str(args.checkpoint_path),
              "checkpoint_hashes": {name: file_hash(args.checkpoint_path/name) for name in
                                    ("mixed_geometry_config.json", "mixed_geometry_heads.pt", "mixed_geometry_normalizer.json", "CHECKPOINT_FINAL.json")},
              "torch": str(torch.__version__), "device": str(device), "native_paths_sha256": file_hash(args.native_paths),
              "all_original_outputs_reproduced": outputs_reproduced, "batches": summaries,
              "limits": ["Two outcome-blind original logical batches do not estimate cohort-wide SUN.",
                         "Cumulative RMS sums diagnose motion/cancellation, not an energy descent guarantee."]}
    write_json(args.output_dir / "SUMMARY.json", result)
    if outputs_reproduced:
        (args.output_dir/"_SUCCESS").touch()
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))
    if not outputs_reproduced:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
