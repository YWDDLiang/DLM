#!/usr/bin/env python3
"""Check every verified terminal against fresh and periodic-equivalent scores."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import importlib.metadata
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import time

MODEL = None


def compare_energies(stored, fresh, *, tolerance=.001):
    values = [float(stored), *map(float, fresh)]
    if len(fresh) != 3 or not all(math.isfinite(v) for v in values):
        raise ValueError("three finite terminal representation energies are required")
    delta = fresh[0] - stored
    spread = max(fresh) - min(fresh)
    return {"stored_energy_matches_fresh": abs(delta) <= tolerance,
            "periodic_representation_consistent": spread <= tolerance,
            "fresh_minus_stored_eV_atom": delta, "representation_spread_eV_atom": spread,
            "status": "consistent" if abs(delta) <= tolerance and spread <= tolerance else "inconsistent"}


def initialize(gpu):
    global MODEL
    import torch
    from chgnet.model.model import CHGNet
    torch.set_num_threads(1)
    torch.cuda.set_device(gpu)
    MODEL = CHGNet.load(model_name="0.3.0", use_device=f"cuda:{gpu}")
    MODEL.eval()


def audit(record):
    import numpy as np
    from pymatgen.core import Structure
    original = Structure.from_dict(record["final_structure"])
    wrapped = Structure(original.lattice, original.species, np.mod(original.frac_coords, 1.))
    shifted = Structure(original.lattice, original.species, np.mod(original.frac_coords + [.137, .271, .419], 1.))
    scores = []
    for name, structure in (("stored_geometry", original), ("wrapped_geometry", wrapped), ("shifted_geometry", shifted)):
        result = MODEL.predict_structure(structure, task="efs")
        if isinstance(result, list):
            result = result[0]
        energy = float(result["e"])
        def array(value):
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            return np.asarray(value, dtype=float)
        forces = array(result["f"])
        stress = array(result["s"])
        if not np.isfinite(forces).all() or not np.isfinite(stress).all():
            raise ValueError("nonfinite fresh force or stress")
        scores.append({"representation": name, "energy_eV_atom": energy,
                       "force_max_eV_A": float(np.linalg.norm(forces, axis=-1).max()),
                       "stress_max_GPa": float(np.abs(stress).max())})
    return {"trajectory_id": record["trajectory_id"], "group_id": record["group_id"],
            "stored_terminal_energy_eV_atom": record["terminal_energy"], "scores": scores,
            **compare_energies(record["terminal_energy"], [r["energy_eV_atom"] for r in scores])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-jsonl", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu-count", type=int, default=4)
    args = parser.parse_args()
    if "SLURM_JOB_ID" not in os.environ or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("terminal audit requires its GPU allocation")
    if not 1 <= args.gpu_count <= len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")):
        raise ValueError("audit workers exceed the GPU allocation")
    records = []
    for source in args.labels_jsonl:
        if not (source.parent / "_SUCCESS").is_file():
            raise ValueError("source label accounting is incomplete")
        records += [row for line in source.read_text().splitlines() if line.strip()
                    if (row := json.loads(line)).get("verified") is True]
    if len({r["trajectory_id"] for r in records}) != len(records):
        raise ValueError("duplicate audited occurrence")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    pools = [ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn"), initializer=initialize, initargs=(gpu,))
             for gpu in range(args.gpu_count)]
    futures, output = {}, []
    try:
        for i, row in enumerate(records):
            futures[pools[i % args.gpu_count].submit(audit, row)] = row
        with (args.output_dir / "audit.jsonl").open("x", encoding="utf-8") as handle:
            for future in as_completed(futures):
                source = futures[future]
                try:
                    result = future.result()
                except Exception as error:
                    result = {"trajectory_id": source["trajectory_id"], "group_id": source["group_id"],
                              "status": "audit_error", "error": f"{type(error).__name__}: {error}"}
                output.append(result)
                handle.write(json.dumps(result) + "\n")
                handle.flush()
                if len(output) % 64 == 0:
                    print(json.dumps({"completed": len(output), "requested": len(records),
                                      "statuses": dict(Counter(r["status"] for r in output))}), flush=True)
    finally:
        for pool in pools:
            pool.shutdown(wait=True)
    report = {"requested": len(records), "completed": len(output),
              "statuses": dict(Counter(r["status"] for r in output)),
              "all_consistent": all(r["status"] == "consistent" for r in output),
              "energy_tolerance_eV_atom": .001, "rigid_fractional_shift": [.137, .271, .419],
              "model": "CHGNet-0.3.0", "chgnet_package": importlib.metadata.version("chgnet"),
              "source_labels": [str(p) for p in args.labels_jsonl],
              "selection": "all verified terminal occurrences; no energy selection",
              "relaxation_rerun": False, "elapsed_seconds": time.monotonic() - started}
    (args.output_dir / "AUDIT_FINAL.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output_dir / "_SUCCESS").touch()
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
