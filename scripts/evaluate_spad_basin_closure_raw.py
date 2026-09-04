#!/usr/bin/env python3
"""Fast paired native screen for Llama-programmed basin closure.

This screen intentionally stops before Direct, official hull queries and
model494.  It measures whether the trained native closure preserves the fixed
attempt denominator while moving complete structures toward lower CHGNet
energy, force and stress and toward the Plan volume-per-atom range.
"""

from __future__ import annotations

import argparse
from collections import Counter
from itertools import product
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from evaluate_potential_closure_native_raw import (
    cluster_bootstrap_mean_ci,
    describe,
    lower_fraction,
    predict_batches,
    prediction_fields,
    read_jsonl,
    validate_generation_rows,
)
from crystal_dlm.r5_plan_state import volume_per_atom_bin


ARMS = ("BS", "closure_ce")
DEFAULT_DENOMINATOR = 256


def volume_bin_bounds(value: Any) -> tuple[float, float] | None:
    parts = str(value or "").split("_")
    if len(parts) != 3 or parts[0] != "volpa":
        return None
    try:
        lower = int(parts[1])
        labelled_upper = int(parts[2])
        if labelled_upper != lower + 4:
            return None
        return float(lower), float(labelled_upper + 1)
    except ValueError:
        return None


def minimum_distance(structure: Any) -> float:
    """Return the 125-image PBC minimum, including nonzero self images."""

    lattice = np.asarray(structure.lattice.matrix, dtype=float)
    shifts = np.asarray(
        [shift for shift in product(range(-2, 3), repeat=3) if shift != (0, 0, 0)],
        dtype=float,
    )
    self_image = float(np.min(np.linalg.norm(shifts @ lattice, axis=1)))
    if len(structure) < 2:
        return self_image
    matrix = np.asarray(structure.distance_matrix, dtype=float)
    np.fill_diagonal(matrix, np.inf)
    return min(self_image, float(np.min(matrix)))


def composition_key(structure: Any) -> tuple[tuple[str, int], ...]:
    counts = Counter(str(site.specie.symbol) for site in structure.sites)
    return tuple(sorted((symbol, int(count)) for symbol, count in counts.items()))


def distance_tail(values: Sequence[float]) -> dict[str, Any]:
    finite = np.asarray(
        [float(value) for value in values if math.isfinite(float(value))],
        dtype=float,
    )
    if finite.size == 0:
        return {
            "known": 0,
            "q01_A": None,
            "q05_A": None,
            "q10_A": None,
            "below_0p75_A": 0,
            "below_1p00_A": 0,
            "below_1p50_A": 0,
        }
    return {
        "known": int(finite.size),
        "q01_A": float(np.quantile(finite, 0.01)),
        "q05_A": float(np.quantile(finite, 0.05)),
        "q10_A": float(np.quantile(finite, 0.10)),
        "below_0p75_A": int(np.sum(finite < 0.75)),
        "below_1p00_A": int(np.sum(finite < 1.00)),
        "below_1p50_A": int(np.sum(finite < 1.50)),
    }


def expected_composition(plan: Mapping[str, Any] | None) -> tuple[tuple[str, int], ...] | None:
    if not isinstance(plan, Mapping):
        return None
    elements = [str(value) for value in plan.get("elements") or ()]
    counts = [int(value) for value in plan.get("counts") or ()]
    if not elements or len(elements) != len(counts) or any(value <= 0 for value in counts):
        return None
    if sum(counts) != int(plan.get("N") or 0):
        return None
    return tuple(sorted(zip(elements, counts, strict=True)))


def prepare_arm(
    arm: str, generation_path: Path, *, denominator: int
) -> tuple[list[dict[str, Any]], list[Any], list[int]]:
    from pymatgen.core import Structure

    rows = read_jsonl(generation_path)
    validate_generation_rows(rows, denominator=denominator)
    records: list[dict[str, Any]] = []
    structures: list[Any] = []
    record_indices: list[int] = []
    for row in rows:
        plan = row.get("plan_state") if isinstance(row.get("plan_state"), Mapping) else None
        ordinal = int(row["ordinal"])
        sample_idx = int(row.get("sample_idx", -1))
        if sample_idx != ordinal:
            raise ValueError(f"sample_idx/ordinal mismatch in {arm} at {ordinal}")
        record: dict[str, Any] = {
            "schema": "spad_basin_closure_raw_attempt_v1",
            "arm": arm,
            "ordinal": ordinal,
            "sample_idx": sample_idx,
            "attempt_id": str(row["attempt_id"]),
            "pair_id": str(row.get("pair_id") or ""),
            "source_plan_state_sha256": row.get("source_plan_state_sha256"),
            "plan_signature": None if plan is None else json.dumps(plan, sort_keys=True),
            "generation_succeeded": row.get("status") == "succeeded",
            "structure_parsed": False,
            "composition_valid": False,
            "fast_structural_valid": False,
            "composition_cluster": None,
            "minimum_distance_A": None,
            "volume_per_atom_A3": None,
            "plan_volume_bin": None if plan is None else plan.get("volume_per_atom_bin"),
            "plan_volume_match": None,
            "unknown_reason": str(row.get("reason") or row.get("status") or ""),
            **prediction_fields(None, num_sites=0),
        }
        if row.get("status") == "succeeded" and isinstance(row.get("structure"), dict):
            try:
                structure = Structure.from_dict(dict(row["structure"]))
                lattice = np.asarray(structure.lattice.matrix, dtype=float)
                volume = float(structure.volume)
                distance = minimum_distance(structure)
                vpa = volume / max(1, len(structure))
                expected = expected_composition(plan)
                exact_composition = expected is not None and composition_key(structure) == expected
                bounds = volume_bin_bounds(record["plan_volume_bin"])
                computed_volume_bin = volume_per_atom_bin(
                    structure.lattice.abc, structure.lattice.angles, len(structure)
                )
                volume_match = (
                    None
                    if bounds is None
                    else computed_volume_bin == str(record["plan_volume_bin"])
                )
                structural_valid = bool(
                    structure.is_ordered
                    and np.isfinite(lattice).all()
                    and math.isfinite(volume)
                    and volume >= 0.1
                    and distance >= 0.5
                )
                record.update(
                    {
                        "structure_parsed": True,
                        "composition_valid": bool(exact_composition),
                        "fast_structural_valid": structural_valid,
                        "composition_cluster": "|".join(
                            f"{symbol}:{count}" for symbol, count in composition_key(structure)
                        ),
                        "minimum_distance_A": distance,
                        "volume_per_atom_A3": vpa,
                        "plan_volume_match": volume_match,
                        "unknown_reason": None,
                    }
                )
                structures.append(structure)
                record_indices.append(len(records))
            except Exception as exc:  # noqa: BLE001
                record["unknown_reason"] = f"{type(exc).__name__}:{exc}"
        records.append(record)
    return records, structures, record_indices


def summarize(records: Sequence[Mapping[str, Any]], *, denominator: int) -> dict[str, Any]:
    if len(records) != denominator:
        raise ValueError("raw screen denominator changed")
    distances = [
        float(row["minimum_distance_A"])
        for row in records
        if row["minimum_distance_A"] is not None
    ]
    return {
        "requested": denominator,
        "generation_succeeded": sum(bool(row["generation_succeeded"]) for row in records),
        "structure_parsed": sum(bool(row["structure_parsed"]) for row in records),
        "composition_valid": sum(bool(row["composition_valid"]) for row in records),
        "fast_structural_valid": sum(bool(row["fast_structural_valid"]) for row in records),
        "plan_volume_match": sum(row["plan_volume_match"] is True for row in records),
        "plan_volume_known": sum(row["plan_volume_match"] is not None for row in records),
        "energy_known": sum(bool(row["energy_known"]) for row in records),
        "force_known": sum(bool(row["force_known"]) for row in records),
        "stress_known": sum(bool(row["stress_known"]) for row in records),
        "minimum_distance_A": describe(
            distances
        ),
        "minimum_distance_tail": distance_tail(distances),
        "volume_per_atom_A3": describe(
            float(row["volume_per_atom_A3"])
            for row in records
            if row["volume_per_atom_A3"] is not None
        ),
        "energy_eV_per_atom": describe(
            float(row["energy_eV_per_atom"]) for row in records if row["energy_known"]
        ),
        "force_rms_eV_per_A": describe(
            float(row["force_rms_eV_per_A"]) for row in records if row["force_known"]
        ),
        "stress_frobenius_GPa": describe(
            float(row["stress_frobenius_GPa"]) for row in records if row["stress_known"]
        ),
    }


def paired_binary(
    closure: Mapping[int, Mapping[str, Any]],
    baseline: Mapping[int, Mapping[str, Any]],
    field: str,
) -> dict[str, int]:
    shared = sorted(set(closure) & set(baseline))
    return {
        "paired": len(shared),
        "wins": sum(bool(closure[i][field]) and not bool(baseline[i][field]) for i in shared),
        "losses": sum(not bool(closure[i][field]) and bool(baseline[i][field]) for i in shared),
        "both_true": sum(bool(closure[i][field]) and bool(baseline[i][field]) for i in shared),
        "both_false": sum(not bool(closure[i][field]) and not bool(baseline[i][field]) for i in shared),
    }


def paired_tristate(
    closure: Mapping[int, Mapping[str, Any]],
    baseline: Mapping[int, Mapping[str, Any]],
    field: str,
) -> dict[str, int]:
    shared = sorted(set(closure) & set(baseline))
    known = [
        index
        for index in shared
        if closure[index].get(field) is not None and baseline[index].get(field) is not None
    ]
    return {
        "paired": len(shared),
        "paired_known": len(known),
        "wins": sum(closure[i][field] is True and baseline[i][field] is False for i in known),
        "losses": sum(closure[i][field] is False and baseline[i][field] is True for i in known),
        "both_true": sum(closure[i][field] is True and baseline[i][field] is True for i in known),
        "both_false": sum(closure[i][field] is False and baseline[i][field] is False for i in known),
        "closure_unknown": sum(closure[i].get(field) is None for i in shared),
        "BS_unknown": sum(baseline[i].get(field) is None for i in shared),
        "both_unknown": sum(
            closure[i].get(field) is None and baseline[i].get(field) is None for i in shared
        ),
    }


def paired_delta(
    closure: Mapping[int, Mapping[str, Any]],
    baseline: Mapping[int, Mapping[str, Any]],
    *,
    field: str,
    known_field: str | None = None,
    bootstrap_seed: int | None = None,
) -> dict[str, Any]:
    values: list[float] = []
    clustered: list[tuple[str, float]] = []
    composition_mismatch_excluded = 0
    for ordinal in sorted(set(closure) & set(baseline)):
        candidate = closure[ordinal]
        control = baseline[ordinal]
        if known_field is not None and not (
            candidate.get(known_field) is True and control.get(known_field) is True
        ):
            continue
        left = candidate.get(field)
        right = control.get(field)
        if left is None or right is None:
            continue
        delta = float(left) - float(right)
        if not math.isfinite(delta):
            continue
        if candidate.get("composition_cluster") != control.get("composition_cluster"):
            composition_mismatch_excluded += 1
            continue
        values.append(delta)
        clustered.append((str(candidate["composition_cluster"]), delta))
    result = {
        "definition": f"closure_ce_minus_BS:{field}",
        "paired_known": len(values),
        "composition_mismatch_excluded": composition_mismatch_excluded,
        "delta": describe(values),
        **lower_fraction(values),
    }
    if bootstrap_seed is not None:
        result["composition_cluster_bootstrap_mean_95ci"] = cluster_bootstrap_mean_ci(
            clustered, seed=int(bootstrap_seed), replicates=10_000
        )
    return result


def run(
    *,
    bs_generation: Path,
    closure_generation: Path,
    output_dir: Path,
    denominator: int,
    device: str,
    batch_size: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if device != "cuda:0":
        raise ValueError("raw screen requires explicit job-local cuda:0")
    if batch_size != 16:
        raise ValueError("raw screen fixes CHGNet batch size to 16")
    paths = {"BS": bs_generation.resolve(), "closure_ce": closure_generation.resolve()}
    all_records: dict[str, list[dict[str, Any]]] = {}
    pending: list[Any] = []
    destinations: list[tuple[str, int, int]] = []
    for arm in ARMS:
        records, structures, indices = prepare_arm(arm, paths[arm], denominator=denominator)
        all_records[arm] = records
        pending.extend(structures)
        destinations.extend(
            (arm, record_index, len(structure))
            for record_index, structure in zip(indices, structures, strict=True)
        )

    from chgnet.model.model import CHGNet

    model = CHGNet.load(use_device=device, check_cuda_mem=False, verbose=False)
    predictions = predict_batches(model, pending, batch_size=batch_size)
    for (arm, record_index, num_sites), prediction in zip(destinations, predictions, strict=True):
        fields = prediction_fields(prediction, num_sites=num_sites)
        all_records[arm][record_index].update(fields)
        if not (fields["energy_known"] and fields["force_known"] and fields["stress_known"]):
            all_records[arm][record_index]["unknown_reason"] = "chgnet_partial_or_failed"

    indexed = {
        arm: {int(row["ordinal"]): row for row in all_records[arm]} for arm in ARMS
    }
    if any(set(rows) != set(range(denominator)) for rows in indexed.values()):
        raise ValueError("paired raw labels do not cover the fixed denominator")
    for ordinal in range(denominator):
        closure_row = indexed["closure_ce"][ordinal]
        baseline_row = indexed["BS"][ordinal]
        for field in (
            "sample_idx",
            "pair_id",
            "source_plan_state_sha256",
            "plan_signature",
        ):
            if closure_row.get(field) != baseline_row.get(field):
                raise ValueError(f"paired {field} mismatch at ordinal {ordinal}")
    paired = {
        "generation_succeeded": paired_binary(indexed["closure_ce"], indexed["BS"], "generation_succeeded"),
        "composition_valid": paired_binary(indexed["closure_ce"], indexed["BS"], "composition_valid"),
        "fast_structural_valid": paired_binary(indexed["closure_ce"], indexed["BS"], "fast_structural_valid"),
        "plan_volume_match": paired_tristate(
            indexed["closure_ce"], indexed["BS"], "plan_volume_match"
        ),
        "minimum_distance_A": paired_delta(indexed["closure_ce"], indexed["BS"], field="minimum_distance_A"),
        "volume_per_atom_A3": paired_delta(indexed["closure_ce"], indexed["BS"], field="volume_per_atom_A3"),
        "energy_eV_per_atom": paired_delta(
            indexed["closure_ce"], indexed["BS"], field="energy_eV_per_atom",
            known_field="energy_known", bootstrap_seed=bootstrap_seed,
        ),
        "force_rms_eV_per_A": paired_delta(
            indexed["closure_ce"], indexed["BS"], field="force_rms_eV_per_A", known_field="force_known"
        ),
        "stress_frobenius_GPa": paired_delta(
            indexed["closure_ce"], indexed["BS"], field="stress_frobenius_GPa", known_field="stress_known"
        ),
        "absolute_hydrostatic_stress_GPa": paired_delta(
            {
                i: {**row, "absolute_hydrostatic_stress_GPa": None if row.get("hydrostatic_stress_GPa") is None else abs(float(row["hydrostatic_stress_GPa"]))}
                for i, row in indexed["closure_ce"].items()
            },
            {
                i: {**row, "absolute_hydrostatic_stress_GPa": None if row.get("hydrostatic_stress_GPa") is None else abs(float(row["hydrostatic_stress_GPa"]))}
                for i, row in indexed["BS"].items()
            },
            field="absolute_hydrostatic_stress_GPa", known_field="stress_known",
        ),
    }
    report = {
        "schema": "spad_basin_closure_native_raw_screen_v1",
        "fixed_denominator": denominator,
        "device": device,
        "chgnet_batch_size": batch_size,
        "single_chgnet_model_load": True,
        "pbc_minimum_distance_image_radius": 2,
        "pbc_minimum_distance_images": 125,
        "arms": {arm: summarize(all_records[arm], denominator=denominator) for arm in ARMS},
        "paired_closure_ce_minus_BS": paired,
        "expensive_direct": False,
        "official_query": False,
        "model494": False,
        "selection_retry_replacement": False,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    for arm in ARMS:
        with (output_dir / f"{arm}.jsonl").open("x", encoding="utf-8", newline="\n") as handle:
            for row in all_records[arm]:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
    (output_dir / "RAW_SCREEN_FINAL.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "_SUCCESS").touch()
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bs-generation", type=Path, required=True)
    parser.add_argument("--closure-generation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--denominator", type=int, default=DEFAULT_DENOMINATOR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--bootstrap-seed", type=int, default=20260904)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run(
        bs_generation=args.bs_generation,
        closure_generation=args.closure_generation,
        output_dir=args.output_dir.resolve(),
        denominator=int(args.denominator),
        device=str(args.device),
        batch_size=int(args.batch_size),
        bootstrap_seed=int(args.bootstrap_seed),
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
