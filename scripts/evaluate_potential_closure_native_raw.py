#!/usr/bin/env python3
"""Evaluate fixed-256 native potential-closure raw structures.

The GPU phase loads CHGNet once and labels all three arms.  The finalize phase
combines those labels with the concurrently computed predicted-only Direct
results.  Generation failures stay in the registered denominator and are
reported as unknown; they are never dropped or replaced.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random
import statistics
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ARMS = ("BS", "closure_control", "potential_closed")
DEFAULT_DENOMINATOR = 256
DEFAULT_BOOTSTRAP_SEED = 20260904
DEFAULT_BOOTSTRAP_REPLICATES = 10_000


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} is not a JSON object")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def validate_generation_rows(
    rows: Sequence[Mapping[str, Any]], *, denominator: int
) -> None:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    if len(rows) != denominator:
        raise ValueError(f"generation must retain all {denominator} attempts")
    ordinals = [int(row.get("ordinal", -1)) for row in rows]
    if ordinals != list(range(denominator)):
        raise ValueError("generation ordinals must be contiguous and ordered")
    attempt_ids = [str(row.get("attempt_id") or "") for row in rows]
    if any(not value for value in attempt_ids) or len(set(attempt_ids)) != denominator:
        raise ValueError("generation attempt_id values must be nonempty and unique")


def describe(values: Iterable[float]) -> dict[str, float | int | None]:
    finite = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not finite:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "minimum": None,
            "maximum": None,
        }
    return {
        "count": len(finite),
        "mean": float(statistics.fmean(finite)),
        "median": float(statistics.median(finite)),
        "minimum": finite[0],
        "maximum": finite[-1],
    }


def vector_cosine(first: Sequence[float], second: Sequence[float]) -> float | None:
    left = np.asarray(first, dtype=float).reshape(-1)
    right = np.asarray(second, dtype=float).reshape(-1)
    if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
        return None
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1.0e-12:
        return None
    return float(np.dot(left, right) / denominator)


def composition_cluster(structure: Any) -> str:
    counts = Counter(str(site.specie.symbol) for site in structure.sites)
    return "|".join(f"{symbol}:{counts[symbol]}" for symbol in sorted(counts))


def predict_batches(
    model: Any, structures: Sequence[Any], *, batch_size: int
) -> list[Mapping[str, Any] | None]:
    output: list[Mapping[str, Any] | None] = []
    for start in range(0, len(structures), batch_size):
        chunk = list(structures[start : start + batch_size])
        try:
            values = model.predict_structure(chunk, task="efsm", batch_size=batch_size)
            if isinstance(values, Mapping):
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


def prediction_fields(
    prediction: Mapping[str, Any] | None, *, num_sites: int
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "energy_known": False,
        "force_known": False,
        "stress_known": False,
        "energy_eV_per_atom": None,
        "forces_eV_per_A": None,
        "force_rms_eV_per_A": None,
        "force_max_eV_per_A": None,
        "stress_GPa": None,
        "stress_frobenius_GPa": None,
        "hydrostatic_stress_GPa": None,
    }
    if prediction is None:
        return result

    try:
        energy = float(np.asarray(prediction["e"], dtype=float).reshape(()))
        if math.isfinite(energy):
            result["energy_known"] = True
            result["energy_eV_per_atom"] = energy
    except Exception:
        pass

    try:
        forces = np.asarray(prediction["f"], dtype=float)
        if forces.shape == (num_sites, 3) and np.isfinite(forces).all():
            norms = np.linalg.norm(forces, axis=1)
            result.update(
                {
                    "force_known": True,
                    "forces_eV_per_A": forces.tolist(),
                    "force_rms_eV_per_A": float(np.sqrt(np.mean(norms * norms))),
                    "force_max_eV_per_A": float(np.max(norms)),
                }
            )
    except Exception:
        pass

    try:
        stress = np.asarray(prediction["s"], dtype=float)
        if stress.shape == (3, 3) and np.isfinite(stress).all():
            result.update(
                {
                    "stress_known": True,
                    "stress_GPa": stress.tolist(),
                    "stress_frobenius_GPa": float(np.linalg.norm(stress)),
                    "hydrostatic_stress_GPa": float(np.trace(stress) / 3.0),
                }
            )
    except Exception:
        pass
    return result


def prepare_arm(
    arm: str,
    path: Path,
    *,
    denominator: int,
) -> tuple[list[dict[str, Any]], list[Any], list[int]]:
    from pymatgen.core import Structure

    rows = read_jsonl(path)
    validate_generation_rows(rows, denominator=denominator)
    records: list[dict[str, Any]] = []
    structures: list[Any] = []
    record_indices: list[int] = []
    for row in rows:
        ordinal = int(row["ordinal"])
        record: dict[str, Any] = {
            "schema": "potential_closure_native_raw_attempt_v1",
            "arm": arm,
            "ordinal": ordinal,
            "attempt_id": str(row["attempt_id"]),
            "generation_succeeded": row.get("status") == "succeeded",
            "structure_parsed": False,
            "composition_cluster": None,
            "species_order": None,
            "unknown_reason": str(row.get("reason") or row.get("status") or ""),
            **prediction_fields(None, num_sites=0),
        }
        if row.get("status") == "succeeded" and isinstance(row.get("structure"), dict):
            try:
                structure = Structure.from_dict(dict(row["structure"]))
                record.update(
                    {
                        "structure_parsed": True,
                        "composition_cluster": composition_cluster(structure),
                        "species_order": [str(site.specie.symbol) for site in structure.sites],
                        "unknown_reason": None,
                    }
                )
                record_indices.append(len(records))
                structures.append(structure)
            except Exception as exc:
                record["unknown_reason"] = f"{type(exc).__name__}:{exc}"
        records.append(record)
    return records, structures, record_indices


def summarize_arm(records: Sequence[Mapping[str, Any]], *, denominator: int) -> dict[str, Any]:
    if len(records) != denominator:
        raise ValueError("arm summary denominator changed")
    return {
        "requested": denominator,
        "generation_succeeded": sum(bool(row["generation_succeeded"]) for row in records),
        "structure_parsed": sum(bool(row["structure_parsed"]) for row in records),
        "energy_known": sum(bool(row["energy_known"]) for row in records),
        "force_known": sum(bool(row["force_known"]) for row in records),
        "stress_known": sum(bool(row["stress_known"]) for row in records),
        "energy_eV_per_atom": describe(
            float(row["energy_eV_per_atom"]) for row in records if row["energy_known"]
        ),
        "force_rms_eV_per_A": describe(
            float(row["force_rms_eV_per_A"]) for row in records if row["force_known"]
        ),
        "force_max_eV_per_A": describe(
            float(row["force_max_eV_per_A"]) for row in records if row["force_known"]
        ),
        "stress_frobenius_GPa": describe(
            float(row["stress_frobenius_GPa"]) for row in records if row["stress_known"]
        ),
        "hydrostatic_stress_GPa": describe(
            float(row["hydrostatic_stress_GPa"]) for row in records if row["stress_known"]
        ),
    }


def run_chgnet(
    generation_paths: Mapping[str, Path],
    *,
    output_dir: Path,
    denominator: int,
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if set(generation_paths) != set(ARMS):
        raise ValueError("exactly the three registered arms are required")
    if device != "cuda:0":
        raise ValueError("native raw screen requires explicit job-local cuda:0")
    if batch_size != 16:
        raise ValueError("native raw screen fixes CHGNet batch size to 16")

    all_records: dict[str, list[dict[str, Any]]] = {}
    pending: list[Any] = []
    destinations: list[tuple[str, int, int]] = []
    for arm in ARMS:
        records, structures, indices = prepare_arm(
            arm, generation_paths[arm].resolve(), denominator=denominator
        )
        all_records[arm] = records
        pending.extend(structures)
        destinations.extend(
            (arm, record_index, len(structure))
            for record_index, structure in zip(indices, structures)
        )

    from chgnet.model.model import CHGNet

    model = CHGNet.load(use_device=device, check_cuda_mem=False, verbose=False)
    predictions = predict_batches(model, pending, batch_size=batch_size)
    for (arm, record_index, num_sites), prediction in zip(destinations, predictions):
        fields = prediction_fields(prediction, num_sites=num_sites)
        all_records[arm][record_index].update(fields)
        if not (fields["energy_known"] and fields["force_known"] and fields["stress_known"]):
            all_records[arm][record_index]["unknown_reason"] = "chgnet_partial_or_failed"

    output_dir.mkdir(parents=True, exist_ok=False)
    summaries: dict[str, Any] = {}
    for arm in ARMS:
        with (output_dir / f"{arm}.jsonl").open("x", encoding="utf-8", newline="\n") as handle:
            for row in all_records[arm]:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        summaries[arm] = summarize_arm(all_records[arm], denominator=denominator)
    report = {
        "schema": "potential_closure_native_raw_chgnet_v1",
        "denominator_per_arm": denominator,
        "device": device,
        "batch_size": batch_size,
        "single_model_load": True,
        "arms": summaries,
        "selection": False,
    }
    (output_dir / "CHGNET_REPORT.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "_SUCCESS").touch()
    return report


def index_records(
    records: Sequence[Mapping[str, Any]], *, denominator: int
) -> dict[int, Mapping[str, Any]]:
    indexed = {int(row["ordinal"]): row for row in records}
    if len(records) != denominator or set(indexed) != set(range(denominator)):
        raise ValueError("attempt labels do not cover the fixed denominator")
    return indexed


def cluster_bootstrap_mean_ci(
    values: Sequence[tuple[str, float]],
    *,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    if replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")
    grouped: dict[str, list[float]] = defaultdict(list)
    for cluster, value in values:
        if math.isfinite(float(value)):
            grouped[str(cluster)].append(float(value))
    cluster_means = [statistics.fmean(grouped[key]) for key in sorted(grouped)]
    if not cluster_means:
        return {
            "seed": seed,
            "replicates": replicates,
            "clusters": 0,
            "point_mean": None,
            "ci95_lower": None,
            "ci95_upper": None,
        }
    rng = random.Random(seed)
    count = len(cluster_means)
    samples = sorted(
        statistics.fmean(cluster_means[rng.randrange(count)] for _ in range(count))
        for _ in range(replicates)
    )
    lower = samples[int(math.floor(0.025 * (replicates - 1)))]
    upper = samples[int(math.ceil(0.975 * (replicates - 1)))]
    return {
        "seed": seed,
        "replicates": replicates,
        "clusters": count,
        "point_mean": float(statistics.fmean(cluster_means)),
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
    }


def lower_fraction(values: Sequence[float]) -> dict[str, Any]:
    return {
        "lower_count": sum(value < 0.0 for value in values),
        "equal_count": sum(value == 0.0 for value in values),
        "higher_count": sum(value > 0.0 for value in values),
        "lower_fraction": None if not values else sum(value < 0.0 for value in values) / len(values),
    }


def paired_effect(
    potential_records: Sequence[Mapping[str, Any]],
    comparator_records: Sequence[Mapping[str, Any]],
    *,
    comparator: str,
    denominator: int,
    bootstrap_seed: int,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    potential = index_records(potential_records, denominator=denominator)
    baseline = index_records(comparator_records, denominator=denominator)
    energy_deltas: list[float] = []
    clustered_energy: list[tuple[str, float]] = []
    force_rms_deltas: list[float] = []
    force_cosines: list[float] = []
    stress_norm_deltas: list[float] = []
    stress_cosines: list[float] = []
    hydrostatic_abs_deltas: list[float] = []
    force_direction_incompatible = 0

    for ordinal in range(denominator):
        candidate = potential[ordinal]
        control = baseline[ordinal]
        if candidate.get("energy_known") is True and control.get("energy_known") is True:
            candidate_cluster = candidate.get("composition_cluster")
            control_cluster = control.get("composition_cluster")
            if not candidate_cluster or not control_cluster or candidate_cluster != control_cluster:
                raise ValueError(f"composition mismatch at ordinal {ordinal}")
            delta = float(candidate["energy_eV_per_atom"]) - float(control["energy_eV_per_atom"])
            energy_deltas.append(delta)
            clustered_energy.append((str(candidate_cluster), delta))

        if candidate.get("force_known") is True and control.get("force_known") is True:
            force_rms_deltas.append(
                float(candidate["force_rms_eV_per_A"])
                - float(control["force_rms_eV_per_A"])
            )
            if candidate.get("species_order") == control.get("species_order"):
                cosine = vector_cosine(candidate["forces_eV_per_A"], control["forces_eV_per_A"])
                if cosine is not None:
                    force_cosines.append(cosine)
            else:
                force_direction_incompatible += 1

        if candidate.get("stress_known") is True and control.get("stress_known") is True:
            stress_norm_deltas.append(
                float(candidate["stress_frobenius_GPa"])
                - float(control["stress_frobenius_GPa"])
            )
            cosine = vector_cosine(candidate["stress_GPa"], control["stress_GPa"])
            if cosine is not None:
                stress_cosines.append(cosine)
            hydrostatic_abs_deltas.append(
                abs(float(candidate["hydrostatic_stress_GPa"]))
                - abs(float(control["hydrostatic_stress_GPa"]))
            )

    return {
        "comparison": f"potential_closed-minus-{comparator}",
        "fixed_denominator": denominator,
        "energy": {
            "direction": "negative delta and lower fraction above 0.5 favor potential_closed",
            "paired_known": len(energy_deltas),
            "unknown_from_fixed_denominator": denominator - len(energy_deltas),
            "delta_eV_per_atom": describe(energy_deltas),
            **lower_fraction(energy_deltas),
            "composition_cluster_bootstrap_mean_95ci": cluster_bootstrap_mean_ci(
                clustered_energy,
                seed=bootstrap_seed,
                replicates=bootstrap_replicates,
            ),
        },
        "force": {
            "direction": "negative RMS delta is favorable; cosine reports field-direction agreement only",
            "paired_known": len(force_rms_deltas),
            "rms_delta_eV_per_A": describe(force_rms_deltas),
            **lower_fraction(force_rms_deltas),
            "direction_cosine": describe(force_cosines),
            "direction_incompatible_species_order": force_direction_incompatible,
        },
        "stress": {
            "direction": "negative norm deltas are favorable; cosine reports tensor-direction agreement only",
            "paired_known": len(stress_norm_deltas),
            "frobenius_delta_GPa": describe(stress_norm_deltas),
            "frobenius_lower": lower_fraction(stress_norm_deltas),
            "absolute_hydrostatic_delta_GPa": describe(hydrostatic_abs_deltas),
            "absolute_hydrostatic_lower": lower_fraction(hydrostatic_abs_deltas),
            "direction_cosine": describe(stress_cosines),
        },
    }


def load_direct(
    generation_path: Path,
    direct_dir: Path,
    *,
    denominator: int,
) -> tuple[dict[str, Any], dict[int, bool]]:
    generation = read_jsonl(generation_path)
    validate_generation_rows(generation, denominator=denominator)
    attempts = read_jsonl(direct_dir / "attempt_metrics.jsonl")
    report = read_json(direct_dir / "report.json")
    if len(attempts) != denominator or int(report.get("attempts", -1)) != denominator:
        raise ValueError("fast Direct output changed the denominator")
    indexed: dict[int, bool] = {}
    for row, metric in zip(generation, attempts):
        if str(row["attempt_id"]) != str(metric["attempt_id"]):
            raise ValueError("fast Direct attempt order changed")
        indexed[int(row["ordinal"])] = metric.get("valid") is True
    counts = {
        "requested": denominator,
        "generation_succeeded": int(report["generation_succeeded"]),
        "composition_valid": int(report["comp_valid_count"]),
        "structure_valid": int(report["struct_valid_count"]),
        "direct_joint": int(report["valid_count"]),
    }
    return counts, indexed


def paired_binary(
    potential: Mapping[int, bool], comparator: Mapping[int, bool]
) -> dict[str, int]:
    shared = sorted(set(potential) & set(comparator))
    return {
        "paired": len(shared),
        "wins": sum(potential[index] and not comparator[index] for index in shared),
        "losses": sum(not potential[index] and comparator[index] for index in shared),
        "both_valid": sum(potential[index] and comparator[index] for index in shared),
        "both_invalid": sum(not potential[index] and not comparator[index] for index in shared),
    }


def continuation_gate(
    *,
    potential_direct: Mapping[str, Any],
    control_direct: Mapping[str, Any],
    primary_energy: Mapping[str, Any],
) -> dict[str, Any]:
    denominator = int(potential_direct["requested"])
    energy = primary_energy["energy"]
    delta = energy["delta_eV_per_atom"]
    bootstrap = energy["composition_cluster_bootstrap_mean_95ci"]
    gates = {
        "composition_valid_ge_0p95": (
            int(potential_direct["composition_valid"]) / denominator >= 0.95
        ),
        "direct_drop_le_0p01": (
            (int(potential_direct["direct_joint"]) - int(control_direct["direct_joint"]))
            / denominator
            >= -0.01
        ),
        "paired_raw_energy_median_lt_zero": (
            delta["median"] is not None and float(delta["median"]) < 0.0
        ),
        "cluster_bootstrap_ci_upper_lt_zero": (
            bootstrap["ci95_upper"] is not None
            and float(bootstrap["ci95_upper"]) < 0.0
        ),
    }
    return {
        "definition": "native stream17 raw continuation only",
        "gates": gates,
        "continue_to_conditional_full_evaluator": all(gates.values()),
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Potential-closure native stream17 raw screen",
        "",
        "All three arms retain the registered fixed-256 denominator. Failed rows are unknown, not dropped.",
        "",
        "| Arm | Comp | Struct | Direct | E known | F known | Stress known |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        direct = report["direct"][arm]
        chgnet = report["chgnet"]["arms"][arm]
        lines.append(
            f"| {arm} | {direct['composition_valid']}/256 | {direct['structure_valid']}/256 | "
            f"{direct['direct_joint']}/256 | {chgnet['energy_known']}/256 | "
            f"{chgnet['force_known']}/256 | {chgnet['stress_known']}/256 |"
        )
    lines.extend(["", "## Paired raw effects", ""])
    for key in ("potential_closed_minus_closure_control", "potential_closed_minus_BS"):
        effect = report["paired"][key]["energy"]
        delta = effect["delta_eV_per_atom"]
        bootstrap = effect["composition_cluster_bootstrap_mean_95ci"]
        lines.append(
            f"- {key}: paired {effect['paired_known']}/256, mean {delta['mean']}, "
            f"median {delta['median']}, lower fraction {effect['lower_fraction']}, "
            f"cluster-bootstrap 95% CI [{bootstrap['ci95_lower']}, {bootstrap['ci95_upper']}]."
        )
    lines.extend(
        [
            "",
            f"Continuation gate: `{report['continuation']['continue_to_conditional_full_evaluator']}`.",
            "Strict/Meta S.U.N. are deferred until the conditional full evaluator; no values are inferred here.",
        ]
    )
    return "\n".join(lines) + "\n"


def finalize(
    generation_paths: Mapping[str, Path],
    direct_dirs: Mapping[str, Path],
    *,
    chgnet_dir: Path,
    output_dir: Path,
    denominator: int,
    bootstrap_seed: int,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if set(generation_paths) != set(ARMS) or set(direct_dirs) != set(ARMS):
        raise ValueError("finalizer requires all three registered arms")
    chgnet_report = read_json(chgnet_dir / "CHGNET_REPORT.json")
    if int(chgnet_report.get("denominator_per_arm", -1)) != denominator:
        raise ValueError("CHGNet report denominator changed")
    labels = {arm: read_jsonl(chgnet_dir / f"{arm}.jsonl") for arm in ARMS}
    for arm in ARMS:
        index_records(labels[arm], denominator=denominator)

    direct: dict[str, Any] = {}
    direct_valid: dict[str, dict[int, bool]] = {}
    for arm in ARMS:
        direct[arm], direct_valid[arm] = load_direct(
            generation_paths[arm], direct_dirs[arm], denominator=denominator
        )

    primary = paired_effect(
        labels["potential_closed"],
        labels["closure_control"],
        comparator="closure_control",
        denominator=denominator,
        bootstrap_seed=bootstrap_seed,
        bootstrap_replicates=bootstrap_replicates,
    )
    secondary = paired_effect(
        labels["potential_closed"],
        labels["BS"],
        comparator="BS",
        denominator=denominator,
        bootstrap_seed=bootstrap_seed,
        bootstrap_replicates=bootstrap_replicates,
    )
    report = {
        "schema": "potential_closure_native_stream17_raw_final_v1",
        "stream": 17,
        "fixed_denominator": denominator,
        "chgnet": chgnet_report,
        "direct": direct,
        "paired": {
            "potential_closed_minus_closure_control": primary,
            "potential_closed_minus_BS": secondary,
        },
        "direct_paired": {
            "potential_closed_vs_closure_control": paired_binary(
                direct_valid["potential_closed"], direct_valid["closure_control"]
            ),
            "potential_closed_vs_BS": paired_binary(
                direct_valid["potential_closed"], direct_valid["BS"]
            ),
        },
        "continuation": continuation_gate(
            potential_direct=direct["potential_closed"],
            control_direct=direct["closure_control"],
            primary_energy=primary,
        ),
        "strict_sun": {
            "value": None,
            "status": "deferred_until_conditional_full_evaluator",
        },
        "meta_sun": {
            "value": None,
            "status": "deferred_until_conditional_full_evaluator",
        },
        "selection": False,
        "replacement_or_denominator_drop": False,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "NATIVE_RAW_FINAL.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "NATIVE_RAW_FINAL.md").write_text(
        render_markdown(report), encoding="utf-8"
    )
    (output_dir / "_SUCCESS").touch()
    return report


def add_generation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--bs-generation", type=Path, required=True)
    parser.add_argument("--control-generation", type=Path, required=True)
    parser.add_argument("--potential-generation", type=Path, required=True)


def generation_paths(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "BS": args.bs_generation,
        "closure_control": args.control_generation,
        "potential_closed": args.potential_generation,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)
    evaluate = subparsers.add_parser("evaluate-chgnet")
    add_generation_arguments(evaluate)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--denominator", type=int, default=DEFAULT_DENOMINATOR)
    evaluate.add_argument("--device", default="cuda:0")
    evaluate.add_argument("--batch-size", type=int, default=16)

    final = subparsers.add_parser("finalize")
    add_generation_arguments(final)
    final.add_argument("--bs-direct-dir", type=Path, required=True)
    final.add_argument("--control-direct-dir", type=Path, required=True)
    final.add_argument("--potential-direct-dir", type=Path, required=True)
    final.add_argument("--chgnet-dir", type=Path, required=True)
    final.add_argument("--output-dir", type=Path, required=True)
    final.add_argument("--denominator", type=int, default=DEFAULT_DENOMINATOR)
    final.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    final.add_argument(
        "--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAP_REPLICATES
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = generation_paths(args)
    if args.mode == "evaluate-chgnet":
        report = run_chgnet(
            paths,
            output_dir=args.output_dir.resolve(),
            denominator=int(args.denominator),
            device=str(args.device),
            batch_size=int(args.batch_size),
        )
    else:
        report = finalize(
            paths,
            {
                "BS": args.bs_direct_dir,
                "closure_control": args.control_direct_dir,
                "potential_closed": args.potential_direct_dir,
            },
            chgnet_dir=args.chgnet_dir.resolve(),
            output_dir=args.output_dir.resolve(),
            denominator=int(args.denominator),
            bootstrap_seed=int(args.bootstrap_seed),
            bootstrap_replicates=int(args.bootstrap_replicates),
        )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
