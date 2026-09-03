#!/usr/bin/env python3
"""Audit the deployed 7+4N representation before potential-closure training.

The audit compares untouched MP20 validation structures with their exact
deployed dynamic-token round trip.  CHGNet is used only as a paired diagnostic;
cached DFT hull values remain the stability reference and are shifted by the
within-structure CHGNet quantization delta to form an explicitly labelled
proxy-retention diagnostic.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import math
from pathlib import Path
import random
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from crystal_dlm.dynamic_crystal import (
    arrays_to_structure,
    dynamic_answer_token_count,
    parse_dynamic_answer,
    structure_to_dynamic_answer,
)


STRICT_HULL_EV_PER_ATOM = 0.0
META_HULL_EV_PER_ATOM = 0.1


def iter_csv(path: Path) -> Iterable[tuple[int, dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "cif" not in reader.fieldnames:
            raise ValueError(f"{path} must contain a cif column")
        for index, row in enumerate(reader):
            yield index, dict(row)


def minimum_distance(structure: Any) -> float:
    if len(structure) < 2:
        return math.inf
    matrix = np.asarray(structure.distance_matrix, dtype=float)
    np.fill_diagonal(matrix, np.inf)
    return float(np.min(matrix))


def fast_structure_valid(structure: Any) -> bool:
    try:
        volume = float(structure.volume)
        return (
            bool(structure.is_ordered)
            and math.isfinite(volume)
            and volume >= 0.1
            and minimum_distance(structure) >= 0.5
        )
    except Exception:
        return False


def _composition_key(structure: Any) -> tuple[tuple[str, int], ...]:
    counts = Counter(str(site.specie.symbol) for site in structure.sites)
    return tuple(sorted((symbol, int(count)) for symbol, count in counts.items()))


def _parse_quantize(payload: tuple[int, Mapping[str, str]]) -> dict[str, Any]:
    index, row = payload
    try:
        from pymatgen.core import Structure

        continuous = Structure.from_str(str(row["cif"]), fmt="cif")
        if not bool(continuous.is_ordered):
            raise ValueError("unordered structure")
        answer, diagnostics = structure_to_dynamic_answer(continuous)
        parsed = parse_dynamic_answer(answer, strict=True)
        quantized = arrays_to_structure(parsed)
        expected = dynamic_answer_token_count(len(continuous))
        exact_species_order = [
            str(site.specie.symbol) for site in continuous.sites
        ] == [str(site.specie.symbol) for site in quantized.sites]
        exact_composition = _composition_key(continuous) == _composition_key(quantized)
        finite_geometry = bool(
            np.isfinite(np.asarray(quantized.lattice.matrix, dtype=float)).all()
            and np.isfinite(np.asarray(quantized.frac_coords, dtype=float)).all()
        )
        original_metric = np.asarray(continuous.lattice.matrix, dtype=float)
        original_metric = original_metric @ original_metric.T
        quantized_metric = np.asarray(quantized.lattice.matrix, dtype=float)
        quantized_metric = quantized_metric @ quantized_metric.T
        metric_denominator = max(float(np.linalg.norm(original_metric)), 1.0e-12)
        e_hull = row.get("e_above_hull")
        return {
            "row_index": int(index),
            "material_id": str(row.get("material_id") or index),
            "status": "ok",
            "num_atoms": int(len(continuous)),
            "arity": int(len({site.specie.symbol for site in continuous.sites})),
            "answer_token_count": int(len(parsed["tokens"])),
            "expected_token_count": int(expected),
            "exact_species_order": bool(exact_species_order),
            "exact_composition": bool(exact_composition),
            "finite_geometry": bool(finite_geometry),
            "continuous_structure": continuous.as_dict(),
            "quantized_structure": quantized.as_dict(),
            "continuous_volume_A3": float(continuous.volume),
            "quantized_volume_A3": float(quantized.volume),
            "continuous_minimum_distance_A": minimum_distance(continuous),
            "quantized_minimum_distance_A": minimum_distance(quantized),
            "continuous_fast_valid": fast_structure_valid(continuous),
            "quantized_fast_valid": fast_structure_valid(quantized),
            "metric_relative_frobenius": float(
                np.linalg.norm(quantized_metric - original_metric) / metric_denominator
            ),
            "angle_max_abs_delta_degree": float(
                np.max(
                    np.abs(
                        np.asarray(quantized.lattice.angles, dtype=float)
                        - np.asarray(continuous.lattice.angles, dtype=float)
                    )
                )
            ),
            "length_max_abs_delta_A": float(
                np.max(
                    np.abs(
                        np.asarray(quantized.lattice.abc, dtype=float)
                        - np.asarray(continuous.lattice.abc, dtype=float)
                    )
                )
            ),
            "cached_dft_e_hull_eV_per_atom": (
                None if e_hull in (None, "") else float(e_hull)
            ),
            "encoding_diagnostics": diagnostics.to_dict(),
        }
    except Exception as exc:
        return {
            "row_index": int(index),
            "material_id": str(row.get("material_id") or index),
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc)[:240],
        }


def parse_quantize_rows(path: Path, workers: int) -> list[dict[str, Any]]:
    payloads = list(iter_csv(path))
    if int(workers) <= 1:
        return [_parse_quantize(payload) for payload in payloads]
    with ProcessPoolExecutor(max_workers=int(workers)) as executor:
        return list(executor.map(_parse_quantize, payloads, chunksize=16))


def _prediction_batches(
    model: Any,
    structures: Sequence[Any],
    *,
    batch_size: int,
) -> list[dict[str, Any] | None]:
    output: list[dict[str, Any] | None] = []
    for start in range(0, len(structures), int(batch_size)):
        chunk = list(structures[start : start + int(batch_size)])
        try:
            values = model.predict_structure(
                chunk,
                task="efsm",
                batch_size=int(batch_size),
            )
            if isinstance(values, dict):
                values = [values]
            output.extend(values)
        except Exception:
            for structure in chunk:
                try:
                    output.append(model.predict_structure(structure, task="efsm"))
                except Exception:
                    output.append(None)
    if len(output) != len(structures):
        raise RuntimeError("CHGNet prediction count changed")
    return output


def _finite_prediction(value: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    try:
        energy = float(np.asarray(value["e"], dtype=float).reshape(()))
        forces = np.asarray(value["f"], dtype=float)
        stress = np.asarray(value["s"], dtype=float)
        if forces.ndim != 2 or forces.shape[1] != 3:
            return None
        if stress.shape != (3, 3):
            return None
        if not (
            math.isfinite(energy)
            and np.isfinite(forces).all()
            and np.isfinite(stress).all()
        ):
            return None
        norms = np.linalg.norm(forces, axis=1)
        return {
            "energy_eV_per_atom": energy,
            "force_rms_eV_per_A": float(np.sqrt(np.mean(norms * norms))),
            "force_max_eV_per_A": float(np.max(norms)),
            "stress_frobenius_GPa": float(np.linalg.norm(stress)),
            "_forces": forces,
            "_stress": stress,
        }
    except Exception:
        return None


def attach_chgnet(
    rows: list[dict[str, Any]],
    *,
    selected_indices: Sequence[int],
    device: str,
    batch_size: int,
) -> None:
    from chgnet.model.model import CHGNet
    from pymatgen.core import Structure

    valid_rows = [rows[int(index)] for index in selected_indices]
    if any(row.get("status") != "ok" for row in valid_rows):
        raise ValueError("CHGNet subset contains failed round trips")
    continuous = [Structure.from_dict(row["continuous_structure"]) for row in valid_rows]
    quantized = [Structure.from_dict(row["quantized_structure"]) for row in valid_rows]
    model = CHGNet.load(use_device=device, check_cuda_mem=False, verbose=False)
    continuous_predictions = _prediction_batches(
        model,
        continuous,
        batch_size=int(batch_size),
    )
    quantized_predictions = _prediction_batches(
        model,
        quantized,
        batch_size=int(batch_size),
    )
    for row, continuous_value, quantized_value in zip(
        valid_rows,
        continuous_predictions,
        quantized_predictions,
        strict=True,
    ):
        left = _finite_prediction(continuous_value)
        right = _finite_prediction(quantized_value)
        row["continuous_chgnet"] = left
        row["quantized_chgnet"] = right
        if left is None or right is None:
            row["chgnet_pair_known"] = False
            continue
        row["chgnet_pair_known"] = True
        left_forces = np.asarray(left.pop("_forces"), dtype=float)
        right_forces = np.asarray(right.pop("_forces"), dtype=float)
        left_stress = np.asarray(left.pop("_stress"), dtype=float)
        right_stress = np.asarray(right.pop("_stress"), dtype=float)
        force_denominator = float(
            np.linalg.norm(left_forces) * np.linalg.norm(right_forces)
        )
        stress_denominator = float(
            np.linalg.norm(left_stress) * np.linalg.norm(right_stress)
        )
        row["force_direction_cosine"] = (
            None
            if force_denominator <= 1.0e-12
            else float(np.sum(left_forces * right_forces) / force_denominator)
        )
        row["stress_direction_cosine"] = (
            None
            if stress_denominator <= 1.0e-12
            else float(np.sum(left_stress * right_stress) / stress_denominator)
        )
        row["hydrostatic_stress_delta_GPa"] = float(
            np.trace(right_stress - left_stress) / 3.0
        )
        row["quantized_minus_continuous_energy_eV_per_atom"] = float(
            right["energy_eV_per_atom"] - left["energy_eV_per_atom"]
        )
        cached = row.get("cached_dft_e_hull_eV_per_atom")
        row["quantized_proxy_e_hull_eV_per_atom"] = (
            None
            if cached is None
            else float(cached)
            + float(row["quantized_minus_continuous_energy_eV_per_atom"])
        )


def attach_frozen_fast_validity(
    rows: list[dict[str, Any]],
    *,
    selected_indices: Sequence[int],
    snapshot_root: Path,
) -> None:
    from pymatgen.core import Structure

    sys.path.insert(0, str(snapshot_root.resolve()))
    try:
        from eval_utils import smact_validity, structure_validity
    finally:
        sys.path.pop(0)

    def evaluate(structure: Any) -> tuple[bool, bool]:
        counts = Counter(int(value) for value in structure.atomic_numbers)
        elements = tuple(sorted(counts))
        amounts = [counts[element] for element in elements]
        divisor = math.gcd(*amounts)
        reduced = tuple(int(value // divisor) for value in amounts)
        return bool(smact_validity(elements, reduced)), bool(
            structure_validity(structure)
        )

    for index in selected_indices:
        row = rows[int(index)]
        if row.get("status") != "ok":
            raise ValueError("fast-validity subset contains failed round trips")
        continuous = Structure.from_dict(row["continuous_structure"])
        quantized = Structure.from_dict(row["quantized_structure"])
        continuous_comp, continuous_struct = evaluate(continuous)
        quantized_comp, quantized_struct = evaluate(quantized)
        row.update(
            {
                "continuous_frozen_comp_valid": continuous_comp,
                "continuous_frozen_struct_valid": continuous_struct,
                "continuous_frozen_direct_joint": (
                    continuous_comp and continuous_struct
                ),
                "quantized_frozen_comp_valid": quantized_comp,
                "quantized_frozen_struct_valid": quantized_struct,
                "quantized_frozen_direct_joint": quantized_comp and quantized_struct,
            }
        )


def _n_bin(value: int) -> str:
    if value <= 5:
        return "N01_05"
    if value <= 10:
        return "N06_10"
    if value <= 15:
        return "N11_15"
    return "N16_20"


def _arity_bin(value: int) -> str:
    return str(value) if value <= 3 else "4plus"


def stratified_indices(
    rows: Sequence[Mapping[str, Any]],
    *,
    count: int,
    seed: int,
) -> list[int]:
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    rng = random.Random(int(seed))
    for position, row in enumerate(rows):
        if row.get("status") != "ok":
            continue
        groups[(_n_bin(int(row["num_atoms"])), _arity_bin(int(row["arity"])))].append(
            position
        )
    for key in sorted(groups):
        rng.shuffle(groups[key])
    selected: list[int] = []
    positions = Counter()
    keys = sorted(groups)
    while len(selected) < min(int(count), sum(len(value) for value in groups.values())):
        progressed = False
        for key in keys:
            offset = positions[key]
            if offset < len(groups[key]):
                selected.append(groups[key][offset])
                positions[key] += 1
                progressed = True
                if len(selected) >= int(count):
                    break
        if not progressed:
            break
    return sorted(selected)


def describe(values: Iterable[float]) -> dict[str, float | int | None]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "q05": None,
            "q95": None,
            "min": None,
            "max": None,
        }
    array = np.asarray(finite, dtype=float)
    return {
        "count": int(len(finite)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "q05": float(np.quantile(array, 0.05)),
        "q95": float(np.quantile(array, 0.95)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def stable_retention(
    rows: Sequence[Mapping[str, Any]],
    *,
    threshold: float,
) -> dict[str, float | int | None]:
    eligible = [
        row
        for row in rows
        if row.get("cached_dft_e_hull_eV_per_atom") is not None
        and row.get("quantized_proxy_e_hull_eV_per_atom") is not None
        and math.isfinite(float(row["cached_dft_e_hull_eV_per_atom"]))
        and math.isfinite(float(row["quantized_proxy_e_hull_eV_per_atom"]))
    ]
    continuous_stable = [
        row
        for row in eligible
        if float(row["cached_dft_e_hull_eV_per_atom"]) <= float(threshold)
    ]
    retained = sum(
        float(row["quantized_proxy_e_hull_eV_per_atom"]) <= float(threshold)
        for row in continuous_stable
    )
    return {
        "eligible": int(len(eligible)),
        "continuous_stable": int(len(continuous_stable)),
        "retained": int(retained),
        "retention": (
            None if not continuous_stable else float(retained / len(continuous_stable))
        ),
    }


def build_report(
    rows: list[dict[str, Any]],
    *,
    fast_indices: Sequence[int],
    selection_seed: int,
    expected_rows: int,
    median_energy_limit: float,
    validity_drop_limit: float,
    retention_limit: float,
) -> dict[str, Any]:
    status_counts = Counter(str(row.get("status")) for row in rows)
    ok = [row for row in rows if row.get("status") == "ok"]
    subset = [rows[int(index)] for index in fast_indices]
    pair_known = [row for row in subset if row.get("chgnet_pair_known") is True]
    continuous_comp_valid = sum(
        bool(row["continuous_frozen_comp_valid"]) for row in subset
    )
    quantized_comp_valid = sum(
        bool(row["quantized_frozen_comp_valid"]) for row in subset
    )
    continuous_struct_valid = sum(
        bool(row["continuous_frozen_struct_valid"]) for row in subset
    )
    quantized_struct_valid = sum(
        bool(row["quantized_frozen_struct_valid"]) for row in subset
    )
    continuous_valid = sum(
        bool(row["continuous_frozen_direct_joint"]) for row in subset
    )
    quantized_valid = sum(
        bool(row["quantized_frozen_direct_joint"]) for row in subset
    )
    denominator = len(subset)
    validity_delta = (
        None
        if not denominator
        else float((quantized_valid - continuous_valid) / denominator)
    )
    energy = describe(
        float(row["quantized_minus_continuous_energy_eV_per_atom"])
        for row in pair_known
    )
    strict = stable_retention(rows, threshold=STRICT_HULL_EV_PER_ATOM)
    meta = stable_retention(rows, threshold=META_HULL_EV_PER_ATOM)
    gates = {
        "expected_rows": int(len(rows)) == int(expected_rows),
        "all_rows_roundtrip": (
            int(len(ok)) == int(expected_rows)
            and all(
                bool(row["exact_composition"])
                and bool(row["exact_species_order"])
                and bool(row["finite_geometry"])
                and int(row["answer_token_count"]) == int(row["expected_token_count"])
                for row in ok
            )
        ),
        "chgnet_pair_coverage_ge_0p98": (
            bool(subset) and len(pair_known) / len(subset) >= 0.98
        ),
        "median_absolute_energy_shift_le_limit": (
            energy["median"] is not None
            and abs(float(energy["median"])) <= float(median_energy_limit)
        ),
        "fast_validity_drop_le_limit": (
            validity_delta is not None
            and float(validity_delta) >= -float(validity_drop_limit)
        ),
        "meta_proxy_retention_ge_limit": (
            meta["retention"] is not None
            and float(meta["retention"]) >= float(retention_limit)
        ),
    }
    return {
        "schema": "potential_closure_phase0_v1",
        "status_counts": dict(sorted(status_counts.items())),
        "rows": int(len(rows)),
        "roundtrip_rows": int(len(ok)),
        "chgnet_pair_known": int(len(pair_known)),
        "chgnet_pair_coverage": (
            None if not subset else float(len(pair_known) / len(subset))
        ),
        "representation": {
            "semantic": "exact deployed dynamic 7+4N",
            "lattice_length_step_A": 0.1,
            "angle_step_degree": 1.0,
            "fractional_coordinate_bins": 101,
        },
        "quantized_minus_continuous": {
            "energy_eV_per_atom": energy,
            "force_rms_eV_per_A": describe(
                float(row["quantized_chgnet"]["force_rms_eV_per_A"])
                - float(row["continuous_chgnet"]["force_rms_eV_per_A"])
                for row in pair_known
            ),
            "force_max_eV_per_A": describe(
                float(row["quantized_chgnet"]["force_max_eV_per_A"])
                - float(row["continuous_chgnet"]["force_max_eV_per_A"])
                for row in pair_known
            ),
            "force_direction_cosine": describe(
                float(row["force_direction_cosine"])
                for row in pair_known
                if row.get("force_direction_cosine") is not None
            ),
            "stress_frobenius_GPa": describe(
                float(row["quantized_chgnet"]["stress_frobenius_GPa"])
                - float(row["continuous_chgnet"]["stress_frobenius_GPa"])
                for row in pair_known
            ),
            "stress_direction_cosine": describe(
                float(row["stress_direction_cosine"])
                for row in pair_known
                if row.get("stress_direction_cosine") is not None
            ),
            "hydrostatic_stress_delta_GPa": describe(
                float(row["hydrostatic_stress_delta_GPa"]) for row in pair_known
            ),
            "volume_A3": describe(
                float(row["quantized_volume_A3"])
                - float(row["continuous_volume_A3"])
                for row in ok
            ),
            "minimum_distance_A": describe(
                float(row["quantized_minimum_distance_A"])
                - float(row["continuous_minimum_distance_A"])
                for row in ok
                if math.isfinite(float(row["continuous_minimum_distance_A"]))
                and math.isfinite(float(row["quantized_minimum_distance_A"]))
            ),
            "metric_relative_frobenius": describe(
                float(row["metric_relative_frobenius"]) for row in ok
            ),
            "angle_max_abs_delta_degree": describe(
                float(row["angle_max_abs_delta_degree"]) for row in ok
            ),
            "length_max_abs_delta_A": describe(
                float(row["length_max_abs_delta_A"]) for row in ok
            ),
        },
        "fast_structural_validity": {
            "subset_size": int(denominator),
            "selection_seed": int(selection_seed),
            "continuous_comp_valid": int(continuous_comp_valid),
            "quantized_comp_valid": int(quantized_comp_valid),
            "continuous_struct_valid": int(continuous_struct_valid),
            "quantized_struct_valid": int(quantized_struct_valid),
            "continuous_valid": int(continuous_valid),
            "quantized_valid": int(quantized_valid),
            "quantized_minus_continuous_rate": validity_delta,
            "definition": "frozen upstream smact_validity + structure_validity",
            "full_Direct_role": "cached-if-available diagnostic; not recomputed on critical path",
        },
        "cached_dft_plus_paired_chgnet_proxy": {
            "definition": "cached DFT e_hull + (CHGNet quantized - CHGNet continuous) at fixed composition",
            "official_recomputation": False,
            "strict_threshold_eV_per_atom": STRICT_HULL_EV_PER_ATOM,
            "strict": strict,
            "strict_role": "reported diagnostic; exact zero threshold is sensitive to proxy error",
            "meta_threshold_eV_per_atom": META_HULL_EV_PER_ATOM,
            "meta": meta,
            "formal_gate_uses": "meta retention",
        },
        "thresholds": {
            "median_energy_limit_eV_per_atom": float(median_energy_limit),
            "fast_validity_drop_limit": float(validity_drop_limit),
            "meta_proxy_retention_limit": float(retention_limit),
        },
        "gates": gates,
        "formal_training_authorized_by_phase0": bool(all(gates.values())),
    }


def _jsonable_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if key not in {"continuous_structure", "quantized_structure"}
    }


def write_outputs(output: Path, rows: list[dict[str, Any]], report: Mapping[str, Any]) -> None:
    output.mkdir(parents=True, exist_ok=False)
    with (output / "paired_rows.jsonl").open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_jsonable_row(row), sort_keys=True) + "\n")
    (output / "PHASE0_REPORT.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    energy = report["quantized_minus_continuous"]["energy_eV_per_atom"]
    validity = report["fast_structural_validity"]
    proxy = report["cached_dft_plus_paired_chgnet_proxy"]
    lines = [
        "# Potential-closure Phase 0",
        "",
        f"- rows / round trips: `{report['rows']} / {report['roundtrip_rows']}`",
        f"- CHGNet paired coverage: `{report['chgnet_pair_coverage']}`",
        f"- median quantized-minus-continuous energy: `{energy['median']}` eV/atom",
        f"- fast validity continuous/quantized: `{validity['continuous_valid']} / {validity['quantized_valid']}` over `{validity['subset_size']}`",
        f"- Strict proxy retention: `{proxy['strict']['retention']}`",
        f"- Meta proxy retention: `{proxy['meta']['retention']}`",
        f"- Phase 0 authorizes action-pool preflight: `{report['formal_training_authorized_by_phase0']}`",
        "",
        "## Gates",
        "",
    ]
    lines.extend(f"- {key}: `{value}`" for key, value in report["gates"].items())
    (output / "PHASE0_REPORT.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    marker = "_SUCCESS" if report["formal_training_authorized_by_phase0"] else "_BLOCKED"
    (output / marker).touch()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-csv", type=Path, required=True)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--expected-rows", type=int, default=9047)
    parser.add_argument("--fast-subset-size", type=int, default=512)
    parser.add_argument("--selection-seed", type=int, default=20260904)
    parser.add_argument("--median-energy-limit", type=float, default=0.015)
    parser.add_argument("--validity-drop-limit", type=float, default=0.01)
    parser.add_argument("--retention-limit", type=float, default=0.60)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if int(args.batch_size) <= 0 or int(args.workers) <= 0:
        raise ValueError("batch size and workers must be positive")
    rows = parse_quantize_rows(args.validation_csv.resolve(), int(args.workers))
    fast_indices = stratified_indices(
        rows,
        count=int(args.fast_subset_size),
        seed=int(args.selection_seed),
    )
    if len(fast_indices) != int(args.fast_subset_size):
        raise RuntimeError("insufficient successful rows for the fixed Phase 0 subset")
    attach_frozen_fast_validity(
        rows,
        selected_indices=fast_indices,
        snapshot_root=args.snapshot_root.resolve(),
    )
    attach_chgnet(
        rows,
        selected_indices=fast_indices,
        device=str(args.device),
        batch_size=int(args.batch_size),
    )
    report = build_report(
        rows,
        fast_indices=fast_indices,
        selection_seed=int(args.selection_seed),
        expected_rows=int(args.expected_rows),
        median_energy_limit=float(args.median_energy_limit),
        validity_drop_limit=float(args.validity_drop_limit),
        retention_limit=float(args.retention_limit),
    )
    write_outputs(args.output_dir.resolve(), rows, report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
