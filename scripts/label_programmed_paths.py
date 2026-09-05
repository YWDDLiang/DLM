#!/usr/bin/env python3
"""Offline original/terminal energy labels for every requested native path."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import gzip
import hashlib
import importlib.metadata
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
from typing import Any

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np

EV_A3_TO_GPA = 160.21766208
_MODEL = None
_OPTIMIZER = None
_OPT_STATUS: dict[str, Any] = {}
_VERSIONS: dict[str, str] = {}


def array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=float)


def finite_scalar(value):
    values = array(value).reshape(-1)
    if len(values) != 1 or not np.isfinite(values).all():
        raise ValueError("energy must be one finite scalar")
    return float(values[0])


def force_and_stress(forces, stress, *, stress_unit):
    forces, stress = array(forces), array(stress)
    if forces.ndim != 2 or forces.shape[-1] != 3 or not len(forces):
        raise ValueError("forces must have shape [N,3]")
    if stress.shape not in ((3, 3), (6,)):
        raise ValueError("stress must be a matrix or Voigt six-vector")
    if not np.isfinite(forces).all() or not np.isfinite(stress).all():
        raise ValueError("nonfinite force/stress")
    if stress_unit == "eV/A3":
        stress = stress * EV_A3_TO_GPA
    elif stress_unit != "GPa":
        raise ValueError("unknown stress unit")
    norms = np.linalg.norm(forces, axis=-1)
    return {"force_max_eV_A": float(norms.max()),
            "force_rms_eV_A": float(np.sqrt(np.mean(norms ** 2))),
            "stress_max_GPa": float(np.abs(stress).max()), "stress_GPa": stress.tolist()}


def structure_from_record(record):
    if record.get("structure") is not None:
        from pymatgen.core import Structure
        return Structure.from_dict(record["structure"])
    from crystal_dlm.dynamic_crystal import parse_dynamic_answer, arrays_to_structure
    return arrays_to_structure(parse_dynamic_answer(record["body"], strict=True))


def validate_structure_geometry(structure):
    lattice, coords = array(structure.lattice.matrix), array(structure.frac_coords)
    if lattice.shape != (3, 3) or coords.shape != (int(structure.num_sites), 3):
        raise ValueError("invalid periodic geometry dimensions")
    if not np.isfinite(lattice).all() or not np.isfinite(coords).all() or abs(np.linalg.det(lattice)) <= 1e-10:
        raise ValueError("nonfinite or degenerate periodic structure")
    shifts = np.stack(np.meshgrid(*([np.arange(-2, 3)] * 3), indexing="ij"), axis=-1).reshape(-1, 3)
    self_images = shifts[np.any(shifts != 0, axis=1)] @ lattice
    minimum = float(np.linalg.norm(self_images, axis=-1).min())
    if len(coords) > 1:
        delta = coords[:, None, :] - coords[None, :, :]
        delta -= np.round(delta)
        distances = np.linalg.norm((delta[:, :, None, :] + shifts) @ lattice, axis=-1).min(axis=-1)
        np.fill_diagonal(distances, np.inf)
        minimum = min(minimum, float(distances.min()))
    if minimum < .5 - 1e-8:
        raise ValueError("periodic geometry violates the common 0.5 Angstrom support")
    return minimum


def label_record(record, *, model, optimizer, structure_factory=structure_from_record,
                 fmax=.1, stress_tolerance=.5, max_steps=500, optimizer_status=None):
    result = {key: record.get(key) for key in ("trajectory_id", "group_id", "source_row_idx", "source_split", "endpoint")}
    result.update(raw_energy=None, terminal_energy=None, gap=None, verified=False,
                  status="unknown", error=None, raw=None, terminal=None, actual_steps=None,
                  optimizer_converged=None, final_structure=None)
    if not record.get("success", False):
        result["status"] = "generation_failure"
        result["error"] = (record.get("trace") or {}).get("failure")
        return result
    try:
        structure = structure_factory(record)
        count = int(structure.num_sites)
        if count < 1:
            raise ValueError("empty structure")
        result["raw_min_distance_A"] = validate_structure_geometry(structure)
    except Exception as error:
        result.update(status="invalid_raw", error=f"{type(error).__name__}: {error}")
        return result
    try:
        raw = model.predict_structure(structure, task="efs")
        if isinstance(raw, list):
            raw = raw[0]
        result["raw_energy"] = finite_scalar(raw["e"])  # CHGNet predict returns eV/atom.
        result["raw"] = force_and_stress(raw["f"], raw["s"], stress_unit="GPa")
        if optimizer_status is not None:
            optimizer_status.clear()
        relaxed = optimizer.relax(structure, fmax=fmax, steps=max_steps, relax_cell=True,
                                  ase_filter="FrechetCellFilter", verbose=False)
        final = relaxed["final_structure"]
        trajectory = relaxed["trajectory"]
        if int(final.num_sites) != count or final.composition != structure.composition:
            raise ValueError("relaxation changed the fixed composition")
        geometry_valid = True
        try:
            result["terminal_min_distance_A"] = validate_structure_geometry(final)
        except ValueError as error:
            geometry_valid = False
            result["terminal_geometry_error"] = str(error)
        energies = list(trajectory.energies)
        if not energies:
            raise ValueError("missing relaxation energy trajectory")
        result["terminal_energy"] = finite_scalar(energies[-1]) / count
        result["terminal"] = force_and_stress(trajectory.forces[-1], trajectory.stresses[-1], stress_unit="eV/A3")
        result["gap"] = result["raw_energy"] - result["terminal_energy"]
        first_energy = finite_scalar(energies[0]) / count
        result["raw_vs_trajectory_first_delta"] = result["raw_energy"] - first_energy
        status = optimizer_status or relaxed.get("optimizer_status") or {}
        result["actual_steps"] = status.get("steps")
        result["optimizer_converged"] = status.get("converged")
        result["trajectory_frames"] = len(energies)
        result["relaxation_trajectory"] = {
            "energies_eV_atom": [finite_scalar(e) / count for e in energies],
            "forces_eV_A": [array(f).tolist() for f in trajectory.forces],
            "stresses_GPa": [(array(s) * EV_A3_TO_GPA).tolist() for s in trajectory.stresses],
            "cells_A": [array(v).tolist() for v in getattr(trajectory, "cells", [])],
            "positions_A": [array(v).tolist() for v in getattr(trajectory, "atom_positions", [])],
        }
        result["final_structure"] = final.as_dict()
        physical = (result["terminal"]["force_max_eV_A"] <= fmax + 1e-8
                    and result["terminal"]["stress_max_GPa"] <= stress_tolerance + 1e-8)
        same_energy = abs(result["raw_vs_trajectory_first_delta"]) <= .001
        monotone = result["gap"] >= -.001
        # Missing optimizer status is explicit, never synthesized as a success.
        stop_verified = result["optimizer_converged"] is True
        result["verified"] = bool(geometry_valid and physical and same_energy and monotone and stop_verified)
        if not same_energy:
            result["status"] = "energy_protocol_mismatch"
        elif not monotone:
            result["status"] = "relaxation_energy_increased"
        elif not geometry_valid:
            result["status"] = "invalid_terminal"
        elif not physical:
            result["status"] = "not_converged"
        elif not stop_verified:
            result["status"] = "optimizer_stop_unverified"
        else:
            result["status"] = "verified"
    except Exception as error:
        result.update(status="evaluation_error", error=f"{type(error).__name__}: {error}")
    return result


def worker_init(gpu_index):
    global _MODEL, _OPTIMIZER, _VERSIONS
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    import torch
    from ase.optimize import FIRE
    from ase.filters import FrechetCellFilter
    from chgnet.model.model import CHGNet
    from chgnet.model.dynamics import StructOptimizer
    torch.set_num_threads(1)
    torch.cuda.set_device(gpu_index)
    class RecordedFIRE(FIRE):
        def run(self, *args, **kwargs):
            outcome = super().run(*args, **kwargs)
            _OPT_STATUS.update(steps=int(self.nsteps), converged=None if outcome is None else bool(outcome))
            return outcome
    class PinnedOptimizer(StructOptimizer):
        def relax(self, structure, **kwargs):
            if kwargs.pop("ase_filter") != "FrechetCellFilter":
                raise ValueError("teacher protocol requires FrechetCellFilter")
            return super().relax(structure, ase_filter=partial(
                FrechetCellFilter, mask=np.ones((3, 3)), hydrostatic_strain=False,
                constant_volume=False, scalar_pressure=0.), loginterval=1,
                assign_magmoms=False, dt=.1, maxstep=.2, **kwargs)
    device = f"cuda:{gpu_index}"
    _MODEL = CHGNet.load(model_name="0.3.0", use_device=device)
    _MODEL.eval()
    _OPTIMIZER = PinnedOptimizer(model=_MODEL, optimizer_class=RecordedFIRE, use_device=device,
                                stress_weight=1 / EV_A3_TO_GPA)
    _VERSIONS = {"model": "CHGNet-0.3.0", "chgnet_package": importlib.metadata.version("chgnet"),
                 "ase_package": importlib.metadata.version("ase")}


def worker_label(record, fmax, stress_tolerance, max_steps):
    result = label_record(record, model=_MODEL, optimizer=_OPTIMIZER, fmax=fmax,
                          stress_tolerance=stress_tolerance, max_steps=max_steps,
                          optimizer_status=_OPT_STATUS)
    result["versions"] = _VERSIONS
    return result


def json_default(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(type(value).__name__)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-jsonl", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--gpu-count", type=int, default=2)
    p.add_argument("--workers-per-gpu", type=int, default=2)
    p.add_argument("--purpose", choices=("train", "evaluation"), default="train")
    p.add_argument("--fmax", type=float, default=.1)
    p.add_argument("--stress-tolerance", type=float, default=.5)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--shard-rank", type=int, default=0)
    p.add_argument("--shard-ranks", type=int, nargs="+")
    p.add_argument("--shard-count", type=int, default=1)
    args = p.parse_args()
    if args.shard_count < 1 or not 0 <= args.shard_rank < args.shard_count:
        raise ValueError("invalid label shard")
    shard_ranks = args.shard_ranks if args.shard_ranks is not None else [args.shard_rank]
    if not shard_ranks or len(set(shard_ranks)) != len(shard_ranks) or any(not 0 <= r < args.shard_count for r in shard_ranks):
        raise ValueError("invalid disjoint label shard ranks")
    if not 1 <= args.gpu_count <= 6 or not 1 <= args.workers_per_gpu <= 4:
        raise ValueError("respect six GPUs and four CPUs per GPU")
    if "SLURM_JOB_ID" not in os.environ or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("labeling requires its declared GPU allocation")
    if args.gpu_count > len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")):
        raise ValueError("requested workers exceed the GPUs actually allocated")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    records = []
    with args.input_jsonl.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip() or index % args.shard_count not in shard_ranks:
                continue
            record = json.loads(line)
            if args.purpose == "train" and record.get("source_split") != "train":
                raise ValueError("training labels cannot read heldout conditions")
            if args.purpose == "train" and record.get("endpoint") not in (None, "native"):
                raise ValueError("native path teachers cannot use model494 endpoints")
            # Do not ship the complete token trace to each physics worker.
            records.append({key: record.get(key) for key in
                            ("trajectory_id", "group_id", "source_row_idx", "source_split", "success", "body", "structure", "endpoint")})
    identities = [r["trajectory_id"] for r in records]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate trajectory occurrence in label input")
    # Identical endpoints share physics only. Every path occurrence is retained.
    by_endpoint = {}
    for record in records:
        geometry = json.dumps(record["structure"], sort_keys=True) if record.get("structure") is not None else str(record["body"])
        key = hashlib.sha256(geometry.encode()).hexdigest() if record["success"] else record["trajectory_id"]
        by_endpoint.setdefault(key, []).append(record)
    (args.output_dir / "trajectories").mkdir()
    pools = [ProcessPoolExecutor(max_workers=args.workers_per_gpu, mp_context=mp.get_context("spawn"),
                                 initializer=worker_init, initargs=(gpu,)) for gpu in range(args.gpu_count)]
    futures = {}
    started = time.monotonic()
    counts = {}
    completed = 0
    try:
        for index, (key, occurrences) in enumerate(by_endpoint.items()):
            future = pools[index % args.gpu_count].submit(worker_label, occurrences[0], args.fmax, args.stress_tolerance, args.max_steps)
            futures[future] = (key, occurrences)
        with (args.output_dir / "labels.jsonl").open("x", encoding="utf-8") as handle:
            for future in as_completed(futures):
                key, occurrences = futures[future]
                source = occurrences[0]
                try:
                    result = future.result()
                except Exception as error:
                    result = {key: source.get(key) for key in ("trajectory_id", "group_id", "source_row_idx", "source_split")}
                    result.update(raw_energy=None, terminal_energy=None, gap=None, verified=False,
                                  status="worker_error", error=f"{type(error).__name__}: {error}")
                trajectory = result.pop("relaxation_trajectory", None)
                if trajectory is not None:
                    destination = args.output_dir / "trajectories" / f"{key}.json.gz"
                    with gzip.open(destination, "wt", encoding="utf-8") as stream:
                        json.dump(trajectory, stream)
                    result["trajectory_file"] = str(destination)
                for occurrence in occurrences:
                    labelled = dict(result, **{name: occurrence.get(name) for name in
                                    ("trajectory_id", "group_id", "source_row_idx", "source_split", "endpoint")})
                    labelled["endpoint_cache_key"] = key
                    handle.write(json.dumps(labelled, default=json_default) + "\n")
                    counts[result["status"]] = counts.get(result["status"], 0) + 1
                    completed += 1
                handle.flush()
                if completed % 64 == 0:
                    print(json.dumps({"completed": completed, "requested": len(records), "statuses": counts,
                                      "seconds": time.monotonic() - started}), flush=True)
    finally:
        for pool in pools:
            pool.shutdown(wait=True)
    report = {"requested": len(records), "completed": completed, "statuses": counts,
              "distinct_endpoint_evaluations": len(by_endpoint),
              "protocol": {"model": "CHGNet-0.3.0", "optimizer": "FIRE", "relax_cell": True,
                           "ase_filter": "FrechetCellFilter", "fmax": args.fmax,
                           "scalar_pressure": 0., "constant_volume": False, "hydrostatic_strain": False,
                           "cell_mask": "all_six", "fire_dt": .1, "fire_maxstep": .2,
                           "stress_tolerance_GPa": args.stress_tolerance, "max_steps": args.max_steps},
              "gpu_count": args.gpu_count, "workers_per_gpu": args.workers_per_gpu,
              "shard_count": args.shard_count, "shard_ranks": shard_ranks,
              "elapsed_seconds": time.monotonic() - started, "purpose": args.purpose}
    (args.output_dir / "LABEL_FINAL.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "_SUCCESS").touch()  # Complete accounting, not universal convergence.
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
