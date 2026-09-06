#!/usr/bin/env python3
"""Audit saved label stopping and PBC support without running any model.

Uses only Python/NumPy. Inputs are the original labels.jsonl (one endpoint) and,
optionally, its exported paths.jsonl. Structures come exclusively from saved
final_structure/structure dictionaries. No CIF reconstruction, MLIP, relaxation,
sampling, selection or mutation of input files occurs.

The adaptive image box certifies only absence of a distance BELOW the declared
0.5 A boundary. It does not claim to compute the global nearest image at every
larger distance. Budget-exceeded geometries remain explicitly uncertified.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np


FMAX = .1
SMAX = .5
MAX_STEPS = 500
DISTANCE_CUTOFF = .5 - 1e-8


def read_rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def finite(value):
    try:
        return float(value) if value is not None and math.isfinite(float(value)) else None
    except (TypeError, ValueError):
        return None


def threshold_status(value, limit):
    value = finite(value)
    return "missing" if value is None else "pass" if value <= limit + 1e-8 else "above"


def step_bucket(value):
    if value is None:
        return "missing"
    value = int(value)
    return "zero" if value == 0 else "below_500" if value < MAX_STEPS else "at_or_above_500"


def geometry_from_saved(structure):
    if not isinstance(structure, dict):
        return None
    lattice = np.asarray(structure["lattice"]["matrix"], dtype=float)
    fractional = np.asarray([site["abc"] for site in structure["sites"]], dtype=float)
    volume = abs(float(np.linalg.det(lattice))) if lattice.shape == (3, 3) else math.nan
    if (lattice.shape != (3, 3) or fractional.ndim != 2 or fractional.shape[-1] != 3
            or not 1 <= len(fractional) <= 20 or not np.isfinite(lattice).all()
            or not np.isfinite(fractional).all() or not math.isfinite(volume) or volume <= 1e-10):
        raise ValueError("saved geometry is outside the finite MP20 cell contract")
    return lattice, fractional


def minimum_in_shifts(lattice, deltas, pairs, shifts):
    best, witness = math.inf, None
    for start in range(0, len(shifts), 256):
        block = shifts[start:start + 256]
        distances = np.linalg.norm((deltas[:, None] + block[None]) @ lattice, axis=-1)
        distances[0, np.all(block == 0, axis=-1)] = math.inf  # exclude zero self displacement
        flat = int(np.argmin(distances))
        pair, image = np.unravel_index(flat, distances.shape)
        if float(distances[pair, image]) < best:
            best = float(distances[pair, image])
            witness = {"atom_pair": pairs[pair], "image": block[image].tolist()}
    return best, witness


def audit_geometry(structure, max_images):
    decoded = geometry_from_saved(structure)
    if decoded is None:
        return {"status": "saved_geometry_missing", "certified_at_cutoff": False}
    lattice, fractional = decoded
    pairs = [[0, 0], *([i, j] for i in range(len(fractional)) for j in range(i + 1, len(fractional)))]
    deltas = np.asarray([np.zeros(3), *(fractional[i] - fractional[j] for i, j in pairs[1:])])
    deltas -= np.round(deltas)
    shifts = np.asarray(list(itertools.product(range(-2, 3), repeat=3)), dtype=int)
    minimum, witness = minimum_in_shifts(lattice, deltas, pairs, shifts)
    inverse_bounds = DISTANCE_CUTOFF * np.linalg.norm(np.linalg.inv(lattice), axis=0)
    if not np.isfinite(inverse_bounds).all():
        raise ValueError("nonfinite image-search bounds")
    # If r=(delta+n)L and ||r||<d, each |delta_k+n_k| is bounded by
    # d*||column_k(L^-1)||. The centered delta lies in [-.5,.5].
    bounds = [math.floor(.5 + float(value) + 1e-10 * max(1., float(value))) for value in inverse_bounds]
    result = {
        "status": "audited", "sites": len(fractional),
        "volume_per_atom_A3": abs(float(np.linalg.det(lattice))) / len(fractional),
        "lattice_condition_number": finite(np.linalg.cond(lattice)),
        "declared_radius2_min_A": minimum, "declared_witness": witness,
        "declared_radius2_pass": minimum >= DISTANCE_CUTOFF,
        "complete_image_box_halfwidths": bounds, "search_cutoff_A": DISTANCE_CUTOFF,
        "certified_at_cutoff": True, "missed_radius2_violation": False,
    }
    if max(bounds) <= 2:
        return result
    size = math.prod(2 * value + 1 for value in bounds)
    result["complete_image_box_size"] = size
    if size > max_images:
        result.update(status="image_budget_exceeded", certified_at_cutoff=False)
        return result
    extra = np.asarray([image for image in itertools.product(*(range(-value, value + 1) for value in bounds))
                        if max(map(abs, image)) > 2], dtype=int).reshape(-1, 3)
    extra_minimum, extra_witness = minimum_in_shifts(lattice, deltas, pairs, extra)
    result.update(extra_box_min_A=extra_minimum, extra_witness=extra_witness,
                  missed_radius2_violation=minimum >= DISTANCE_CUTOFF and extra_minimum < DISTANCE_CUTOFF)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--labels-jsonl", type=Path, nargs="+", required=True)
    p.add_argument("--paths-jsonl", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-images", type=int, default=200000)
    args = p.parse_args()
    if args.max_images < 125:
        raise ValueError("image budget must cover the declared shell")
    labels = [row for path in args.labels_jsonl for row in read_rows(path)]
    identifiers = [row["trajectory_id"] for row in labels]
    if not labels or len(set(identifiers)) != len(labels):
        raise ValueError("duplicate label identities; audit native and tau800 separately")
    path_rows = [] if args.paths_jsonl is None else read_rows(args.paths_jsonl)
    paths = {row["trajectory_id"]: row for row in path_rows}
    if len(paths) != len(path_rows):
        raise ValueError("duplicate path identities")
    if paths and set(paths) != set(identifiers):
        raise ValueError("path and label occurrence sets differ")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    counters, crossings, mic, fresh = Counter(), Counter(), Counter(), Counter()
    with (args.output_dir / "rows.jsonl").open("x", encoding="utf-8") as output:
        for label in labels:
            identity = label["trajectory_id"]
            optimizer = label.get("optimizer_converged")
            stop = "true" if optimizer is True else "false" if optimizer is False else "missing"
            terminal = label.get("terminal") or {}
            force = threshold_status(terminal.get("force_max_eV_A"), FMAX)
            stress = threshold_status(terminal.get("stress_max_GPa"), SMAX)
            steps = step_bucket(label.get("actual_steps"))
            crossings[(stop, steps, force, stress, str(label.get("status")))] += 1
            counters["labels"] += 1
            counters["verified"] += int(label.get("verified") is True)
            if stop == "true" and force == "pass" and stress == "above":
                counters["optimizer_true_force_pass_stress_above"] += 1
                counters["optimizer_true_force_pass_stress_above_" + steps] += 1
            row = {key: label.get(key) for key in ("trajectory_id", "group_id", "status", "verified", "optimizer_converged", "actual_steps", "raw_energy", "terminal_energy", "gap")}
            row.update(terminal_force_status=force, terminal_stress_status=stress)
            scores = (label.get("terminal_consistency") or {}).get("scores", [])
            for score in scores:
                key = str(score.get("representation"))
                f = threshold_status(score.get("force_max_eV_A"), FMAX)
                s = threshold_status(score.get("stress_max_GPa"), SMAX)
                fresh[(key, f, s, bool(label.get("verified")))] += 1
            for stage, structure in (("terminal", label.get("final_structure")),
                                     ("raw", paths.get(identity, {}).get("structure"))):
                try:
                    result = audit_geometry(structure, args.max_images)
                except (KeyError, TypeError, ValueError, np.linalg.LinAlgError) as error:
                    result = {"status": "invalid_saved_geometry", "error": str(error), "certified_at_cutoff": False}
                row[stage + "_geometry"] = result
                mic[(stage, result["status"], bool(result.get("missed_radius2_violation")))] += 1
            output.write(json.dumps(row, allow_nan=False) + "\n")
    report = {
        "schema": "saved_physics_lane_audit_v1", "model_calls": 0, "relaxations": 0, "new_samples": 0,
        "input_sha256": {str(path): digest(path) for path in [*args.labels_jsonl, *([args.paths_jsonl] if args.paths_jsonl else [])]},
        "thresholds": {"force_eV_A": FMAX, "entrywise_stress_GPa": SMAX, "max_steps": MAX_STEPS, "distance_A": DISTANCE_CUTOFF},
        "counters": dict(counters),
        "stop_crossings": [dict(optimizer_converged=k[0], step_bucket=k[1], force=k[2], stress=k[3], status=k[4], count=v) for k, v in sorted(crossings.items())],
        "fresh_representation_checks": [dict(representation=k[0], force=k[1], stress=k[2], labelled_verified=k[3], count=v) for k, v in sorted(fresh.items())],
        "image_audit": [dict(stage=k[0], status=k[1], missed_radius2_violation=k[2], count=v) for k, v in sorted(mic.items())],
        "max_images_per_structure": args.max_images,
        "claim_boundary": "Saved-artifact checks only; incomplete geometry/image searches are never labelled valid. No frozen evaluation result is changed.",
    }
    (args.output_dir / "SUMMARY.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "_AUDIT_SUCCESS").touch()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
