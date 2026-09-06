#!/usr/bin/env python3
"""Descriptive CPU regression diagnostics from already completed evaluations.

No model, potential, new relaxation, training data selection or remote lookup is
used. Full-request binary counts remain the headline denominator; numerical
intersections, trimming and tail deletions are separately named diagnostics.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "scripts"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from compare_programmed_path_evaluations import (
    compare_results, index_requests, read_jsonl, validate_pairing,
)
from crystal_dlm.dynamic_crystal import parse_dynamic_answer
from crystal_dlm.manifold_corruption import lattice_matrix_from_parameters


SCHEMA = "programmed_path_regression_analysis_v1"
TRIM_FRACTION_EACH_TAIL = .1
ABSOLUTE_TAIL_COUNTS = (1, 3, 5)
IMAGE_RADIUS = 2
METRICS = {
    "A_gap_eV_atom": "gap_eV_atom",
    "eR_eV_atom_delta_equals_B_delta": "terminal_energy_eV_atom",
    "raw_energy_eV_atom": "raw_energy_eV_atom",
    "raw_force_max_eV_A": "raw.force_max_eV_A",
    "raw_stress_max_GPa": "raw.stress_max_GPa",
    "terminal_force_max_eV_A": "terminal.force_max_eV_A",
    "terminal_stress_max_GPa": "terminal.stress_max_GPa",
    "relaxation_steps": "actual_relaxation_steps",
}
BINARY_FIELDS = (
    "reconstructed", "native_execution_success", "endpoint_execution_success",
    "novel", "unique_representative", "novel_unique", "terminal_verified",
    "strict_stable", "meta_stable", "strict_sun", "meta_sun",
    "verified_strict_sun", "verified_meta_sun",
)
PROTOCOL_FIELDS = (
    "endpoint", "terminal_protocol", "verification_protocol",
    "frozen_nu_source_sha256", "official_cache", "cohort_role",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_value(row, field):
    for key in field.split("."):
        row = row.get(key) if isinstance(row, dict) else None
    return row


def finite_value(row, field):
    value = get_value(row, field)
    if value is None or isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def quantile(values, probability):
    if not values:
        return None
    ordered = sorted(values)
    location = (len(ordered) - 1) * probability
    low, high = math.floor(location), math.ceil(location)
    return ordered[low] + (ordered[high] - ordered[low]) * (location - low)


def describe(values):
    values = list(values)
    return {
        "n": len(values), "mean": statistics.fmean(values) if values else None,
        "median": statistics.median(values) if values else None,
        "p10": quantile(values, .1), "p90": quantile(values, .9),
        "minimum": min(values) if values else None, "maximum": max(values) if values else None,
    }


def exact_mcnemar(reference_only, method_only):
    discordant = reference_only + method_only
    if not discordant:
        return 1.
    numerator = 2 * sum(math.comb(discordant, k) for k in range(min(reference_only, method_only) + 1))
    return min(1., numerator / (1 << discordant))


def composition(path):
    plan = path.get("plan_state") or {}
    elements, counts = plan.get("elements") or [], plan.get("counts") or []
    if len(elements) != len(counts) or not elements:
        return {"formula": None, "elements": elements, "counts": counts, "N": plan.get("N")}
    formula = "".join(str(symbol) + (str(int(count)) if int(count) != 1 else "")
                      for symbol, count in zip(elements, counts))
    return {"formula": formula, "elements": list(elements), "counts": list(counts), "N": plan.get("N")}


def identity(index, reference_path, method_path):
    return {"sample_idx": index, "group_id": reference_path["group_id"],
            "composition": composition(reference_path),
            "reference_trajectory_id": reference_path["trajectory_id"],
            "method_trajectory_id": method_path["trajectory_id"]}


def validate_inputs(reference_rows, method_rows, reference_paths, method_paths, reference_manifest, method_manifest):
    for field in PROTOCOL_FIELDS:
        if field not in reference_manifest or field not in method_manifest or reference_manifest[field] != method_manifest[field]:
            raise ValueError(f"paired evaluation protocols differ: {field}")
    if reference_manifest["endpoint"] not in ("native", "tau800"):
        raise ValueError("unknown common evaluation endpoint")
    validate_pairing(reference_paths, method_paths)
    references, methods = map(index_requests, (reference_rows, method_rows))
    reference_paths, method_paths = map(index_requests, (reference_paths, method_paths))
    denominator = len(references)
    if not denominator or set(references) != set(methods) or set(references) != set(reference_paths) or set(references) != set(method_paths):
        raise ValueError("attempt results and complete path request denominators differ")
    for rows, paths, manifest in ((references, reference_paths, reference_manifest),
                                  (methods, method_paths, method_manifest)):
        if manifest.get("counts", {}).get("requests") != denominator:
            raise ValueError("evaluation manifest and full request denominator differ")
        for index, row in rows.items():
            path = paths[index]
            if row.get("trajectory_id") != path.get("trajectory_id") or row["group_id"] != path["group_id"]:
                raise ValueError(f"attempt/path identity mismatch at request {index}")
            if path.get("endpoint", manifest["endpoint"]) != manifest["endpoint"]:
                raise ValueError("supplied path endpoint differs from its evaluation")
            if any(not isinstance(row.get(field), bool) for field in BINARY_FIELDS):
                raise ValueError("all-request binary outcomes must be explicitly recorded booleans")
            if row["endpoint_execution_success"] != bool(path.get("success")):
                raise ValueError("endpoint execution result and supplied path differ")
            if row["native_execution_success"] != bool(path.get("native_execution_success", path.get("success"))):
                raise ValueError("native execution result and supplied path differ")
        for field in BINARY_FIELDS:
            if field in manifest["counts"] and manifest["counts"][field] != sum(row[field] for row in rows.values()):
                raise ValueError(f"evaluation manifest binary count differs: {field}")
    # This shared function also verifies per-request chemical systems and equal
    # paired hull energies whenever both hull references are available.
    comparison = compare_results(reference_rows, method_rows)
    return references, methods, reference_paths, method_paths, comparison


def paired_distribution(reference, method, identities, field, *, verified=False):
    pairs, missing = [], Counter()
    eligible = 0
    for index in sorted(reference):
        a, b = reference[index], method[index]
        if verified and not (a["terminal_verified"] and b["terminal_verified"]):
            continue
        eligible += 1
        av, bv = finite_value(a, field), finite_value(b, field)
        if av is None or bv is None:
            missing["both" if av is None and bv is None else "reference_only" if av is None else "method_only"] += 1
            continue
        pairs.append({**identities[index], "reference": av, "method": bv, "delta": bv - av})
    n = len(pairs)
    deltas = [row["delta"] for row in pairs]
    ranked = sorted(pairs, key=lambda row: (row["delta"], row["sample_idx"]))
    trimmed_per_tail = math.floor(TRIM_FRACTION_EACH_TAIL * n)
    trimmed = ranked[trimmed_per_tail:n - trimmed_per_tail] if n else []
    absolute_rank = sorted(pairs, key=lambda row: (-abs(row["delta"]), row["sample_idx"]))
    absolute_sum = math.fsum(abs(value) for value in deltas)
    tail_removals = {}
    for requested in ABSOLUTE_TAIL_COUNTS:
        removed = absolute_rank[:requested]
        remaining = absolute_rank[len(removed):]
        signed_sum = math.fsum(row["delta"] for row in removed)
        tail_removals[str(requested)] = {
            "requested_removed": requested, "actually_removed": len(removed), "retained_n": len(remaining),
            "mean_delta_without_tail": statistics.fmean(row["delta"] for row in remaining) if remaining else None,
            "removed_signed_delta_sum": signed_sum,
            "removed_contribution_to_original_mean": signed_sum / n if n else None,
            "removed_absolute_delta_share": math.fsum(abs(row["delta"]) for row in removed) / absolute_sum if absolute_sum else None,
            "removed_sample_idx": [row["sample_idx"] for row in removed],
        }
    return {
        "field": field, "n": n, "excluded_requests": len(reference) - n,
        "all_requests": len(reference), "eligible_requests": eligible,
        "excluded_by_verification": len(reference) - eligible,
        "nonfinite_or_missing_within_eligible": dict(missing),
        "reference": describe(row["reference"] for row in pairs),
        "method": describe(row["method"] for row in pairs), "delta": describe(deltas),
        "method_lower": sum(value < 0 for value in deltas),
        "method_higher": sum(value > 0 for value in deltas), "equal": sum(value == 0 for value in deltas),
        "positive_delta_sum": math.fsum(value for value in deltas if value > 0),
        "negative_delta_sum": math.fsum(value for value in deltas if value < 0),
        "net_delta_sum": math.fsum(deltas),
        "symmetric_trimmed_delta": {
            "fraction_each_tail": TRIM_FRACTION_EACH_TAIL, "removed_per_tail": trimmed_per_tail,
            "retained_n": len(trimmed),
            "mean": statistics.fmean(row["delta"] for row in trimmed) if trimmed else None,
            "selection": "rank by paired delta, remove floor(0.10*n) at each end",
        },
        "absolute_tail_removals": tail_removals,
        "largest_decreases": [row for row in ranked if row["delta"] < 0][:5],
        "largest_increases": [row for row in reversed(ranked) if row["delta"] > 0][:5],
        "diagnostic_only": "trimmed/deleted tails never replace full-request headline counts or the full finite-pair mean",
    }


def paired_energy_decomposition(reference, method, identities, *, verified):
    pairs = []
    for index in sorted(reference):
        a, b = reference[index], method[index]
        if verified and not (a["terminal_verified"] and b["terminal_verified"]):
            continue
        fields = ("gap_eV_atom", "raw_energy_eV_atom", "terminal_energy_eV_atom")
        av, bv = [finite_value(a, key) for key in fields], [finite_value(b, key) for key in fields]
        if any(value is None for value in av + bv):
            continue
        da, de0, der = [right - left for left, right in zip(av, bv)]
        pairs.append({**identities[index], "delta_A": da, "delta_raw_energy": de0, "delta_eR_equals_delta_B": der,
                      "reference_identity_residual": av[0] - (av[1] - av[2]),
                      "method_identity_residual": bv[0] - (bv[1] - bv[2]),
                      "delta_identity_residual": da - (de0 - der)})
    residuals = [abs(row[key]) for row in pairs for key in
                 ("reference_identity_residual", "method_identity_residual", "delta_identity_residual")]
    return {
        "n": len(pairs), "excluded_requests": len(reference) - len(pairs),
        "mean_delta_A": statistics.fmean(row["delta_A"] for row in pairs) if pairs else None,
        "mean_delta_raw_energy": statistics.fmean(row["delta_raw_energy"] for row in pairs) if pairs else None,
        "mean_delta_eR_equals_delta_B": statistics.fmean(row["delta_eR_equals_delta_B"] for row in pairs) if pairs else None,
        "identity": "delta A = delta raw energy - delta eR, on exactly these same pairs",
        "identity_tolerance_eV_atom": 1e-8, "maximum_absolute_identity_residual": max(residuals) if residuals else None,
        "identity_verified": bool(pairs) and max(residuals) <= 1e-8,
        "identity_violation_pairs": [row for row in pairs if max(abs(row[key]) for key in
            ("reference_identity_residual", "method_identity_residual", "delta_identity_residual")) > 1e-8],
        "A_higher_B_lower_pairs": sum(row["delta_A"] > 0 and row["delta_eR_equals_delta_B"] < 0 for row in pairs),
        "A_and_B_both_lower_pairs": sum(row["delta_A"] < 0 and row["delta_eR_equals_delta_B"] < 0 for row in pairs),
        "interpretation": "algebraic decomposition, not independent causal effects",
    }


def geometry_from_arrays(lattice, fractional, *, source):
    lattice, fractional = np.asarray(lattice, dtype=np.float64), np.asarray(fractional, dtype=np.float64)
    if lattice.shape != (3, 3) or fractional.ndim != 2 or fractional.shape[1] != 3 or not len(fractional):
        raise ValueError("periodic geometry requires a 3x3 lattice and nonempty Nx3 fractional coordinates")
    if not np.isfinite(lattice).all() or not np.isfinite(fractional).all():
        raise ValueError("nonfinite stored periodic geometry")
    volume = abs(float(np.linalg.det(lattice)))
    if volume <= 1e-12:
        raise ValueError("stored periodic cell is degenerate")
    shifts = np.asarray(list(itertools.product(range(-IMAGE_RADIUS, IMAGE_RADIUS + 1), repeat=3)), dtype=np.float64)
    self_shifts = shifts[np.any(shifts != 0, axis=1)]
    self_distances = np.linalg.norm(self_shifts @ lattice, axis=-1)
    self_index = int(np.argmin(self_distances))
    pair_distance, nearest_pair = None, None
    if len(fractional) > 1:
        i, j = np.triu_indices(len(fractional), 1)
        raw_delta = fractional[j] - fractional[i]
        centered = raw_delta - np.round(raw_delta)
        vectors = (centered[:, None, :] + shifts[None, :, :]) @ lattice
        distances = np.linalg.norm(vectors, axis=-1)
        pair_index, image_index = np.unravel_index(np.argmin(distances), distances.shape)
        pair_distance = float(distances[pair_index, image_index])
        nearest_pair = {"site_i": int(i[pair_index]), "site_j": int(j[pair_index]),
                        "image_shift_j": (shifts[image_index] - np.round(raw_delta[pair_index])).astype(int).tolist(),
                        "distance_A": pair_distance}
    self_distance = float(self_distances[self_index])
    lengths = np.linalg.norm(lattice, axis=-1)
    singular_values = np.linalg.svd(lattice, compute_uv=False)
    condition = float(singular_values[0] / singular_values[-1])
    if not math.isfinite(condition):
        raise ValueError("stored lattice condition number is nonfinite")
    return {
        "status": "available", "source": source, "num_atoms": len(fractional),
        "volume_A3": volume, "volume_per_atom_A3": volume / len(fractional),
        "lattice_matrix_A": lattice.tolist(), "lattice_lengths_A": lengths.tolist(),
        "lattice_condition_number_2": condition,
        "minimum_distinct_pair_distance_A": pair_distance,
        "minimum_self_image_distance_A": self_distance,
        "minimum_any_periodic_distance_A": self_distance if pair_distance is None else min(self_distance, pair_distance),
        "nearest_distinct_pair": nearest_pair,
        "nearest_self_image_shift": self_shifts[self_index].astype(int).tolist(),
        "periodic_image_radius": IMAGE_RADIUS,
        "periodic_search": "centered finite 125-image shell in the actual full row-vector lattice; not exact arbitrary-basis MIC",
    }


def _geometry_from_body(body):
    arrays = parse_dynamic_answer(body, strict=True)
    lattice = lattice_matrix_from_parameters(arrays["lengths"], arrays["angles"])
    return geometry_from_arrays(lattice, arrays["frac_coords"], source="stored_native_integer_tokens")


def stored_geometry(path, *, endpoint, native_body_only=False):
    """Read existing coordinates only, keeping missing tau800 geometry missing."""
    try:
        if native_body_only:
            return _geometry_from_body(path["body"])
        if path.get("parseable") is False:
            raise ValueError(path.get("artifact_error") or "recorded endpoint parser failure")
        structure = path.get("structure")
        if structure is not None:
            lattice = structure["lattice"]["matrix"]
            sites = structure["sites"]
            if all("abc" in site for site in sites):
                fractions = [site["abc"] for site in sites]
            elif all("xyz" in site for site in sites):
                fractions = np.asarray([site["xyz"] for site in sites]) @ np.linalg.inv(np.asarray(lattice))
            else:
                raise ValueError("stored structure lacks complete fractional or Cartesian site coordinates")
            return geometry_from_arrays(lattice, fractions, source="stored_endpoint_structure_dict")
        cif = path.get("cif")
        cif_path = path.get("cif_path")
        if isinstance(cif, str) or cif_path:
            from pymatgen.core import Structure
            if not isinstance(cif, str):
                cif = Path(cif_path).read_text(encoding="utf-8")
            structure = Structure.from_str(cif, fmt="cif")
            return geometry_from_arrays(structure.lattice.matrix, structure.frac_coords,
                                        source="stored_inline_cif" if isinstance(path.get("cif"), str) else "stored_cif_file")
        if endpoint == "tau800":
            raise ValueError("tau800 input geometry unavailable; native token body is not a refined substitute")
        return _geometry_from_body(path["body"])
    except (KeyError, TypeError, ValueError, RuntimeError, OSError, ImportError, np.linalg.LinAlgError) as error:
        return {"status": "unavailable", "reason": f"{type(error).__name__}: {error}"}


def recorded_failure_evidence(result, path, terminal_protocol):
    evidence = {"terminal_status": result.get("terminal_status"),
                "official_hull_status": result.get("official_hull_status"),
                "hull_energy_eV_atom": finite_value(result, "hull_energy_eV_atom")}
    for field in ("parser_error", "terminal_error", "label_error", "error", "terminal_geometry_error",
                  "optimizer_converged", "terminal_consistency", "raw_vs_trajectory_first_delta"):
        if field in result and result[field] is not None:
            evidence[field] = result[field]
    for field in ("artifact_error", "refiner_graph_error", "error", "failure"):
        if path.get(field) is not None:
            evidence["path_" + field] = path[field]
    if isinstance(path.get("trace"), dict) and path["trace"].get("failure") is not None:
        evidence["path_trace_failure"] = path["trace"]["failure"]
    force, stress = finite_value(result, "terminal.force_max_eV_A"), finite_value(result, "terminal.stress_max_GPa")
    fmax, smax = terminal_protocol.get("fmax"), terminal_protocol.get("stress_tolerance_GPa")
    evidence["recorded_terminal_force_eV_A"] = force
    evidence["recorded_terminal_stress_GPa"] = stress
    evidence["recorded_relaxation_steps"] = finite_value(result, "actual_relaxation_steps")
    evidence["force_exceeds_registered_tolerance"] = None if force is None or fmax is None else force > float(fmax) + 1e-8
    evidence["stress_exceeds_registered_tolerance"] = None if stress is None or smax is None else stress > float(smax) + 1e-8
    detailed = ("optimizer_converged", "terminal_consistency", "terminal_error", "label_error", "error", "terminal_geometry_error")
    evidence["unrecorded_label_detail_fields"] = [field for field in detailed if field not in result]
    evidence["reason_limit"] = "status and recorded thresholds only; absent optimizer/consistency errors cannot be reconstructed"
    return evidence


def transition_cases(reference, method, reference_paths, method_paths, identities, field, terminal_protocol):
    lost, gained = [], []
    for index in sorted(reference):
        before, after = reference[index][field], method[index][field]
        if before == after:
            continue
        entry = {**identities[index], "reference": before, "method": after,
                 "reference_evidence": recorded_failure_evidence(reference[index], reference_paths[index], terminal_protocol),
                 "method_evidence": recorded_failure_evidence(method[index], method_paths[index], terminal_protocol)}
        (lost if before else gained).append(entry)
    return {"lost_count": len(lost), "gained_count": len(gained),
            "lost": lost, "gained": gained,
            "lost_method_statuses": dict(Counter(row["method_evidence"]["terminal_status"] for row in lost)),
            "gained_reference_statuses": dict(Counter(row["reference_evidence"]["terminal_status"] for row in gained))}


def status_transitions(reference, method, field):
    pairs = Counter((str(row.get(field, "unrecorded")), str(method[index].get(field, "unrecorded")))
                    for index, row in reference.items())
    return [{"reference": left, "method": right, "requests": count} for (left, right), count in sorted(pairs.items())]


def analyze_regressions(reference_rows, method_rows, reference_paths, method_paths, reference_manifest, method_manifest):
    reference, method, reference_paths, method_paths, comparison = validate_inputs(
        reference_rows, method_rows, reference_paths, method_paths, reference_manifest, method_manifest)
    identities = {index: identity(index, reference_paths[index], method_paths[index]) for index in reference}
    binary = comparison["binary_all_requests"]
    for field, row in binary.items():
        row["requests"] = len(reference)
        row["exact_mcnemar_two_sided_descriptive"] = exact_mcnemar(row["reference_only"], row["method_only"])
    physical = {
        name: {metric: paired_distribution(reference, method, identities, field, verified=verified)
               for metric, field in METRICS.items()}
        for name, verified in (("finite_physical_intersections", False), ("common_verified_intersections", True))
    }
    endpoint = reference_manifest["endpoint"]
    geometry = {
        arm: {index: stored_geometry(path, endpoint=endpoint) for index, path in paths.items()}
        for arm, paths in (("reference", reference_paths), ("method", method_paths))
    }
    geometry_fields = ("volume_per_atom_A3", "lattice_condition_number_2", "minimum_distinct_pair_distance_A",
                       "minimum_self_image_distance_A", "minimum_any_periodic_distance_A")
    geometry_summary = {
        field: paired_distribution(geometry["reference"], geometry["method"], identities, field)
        for field in geometry_fields
    }
    tail_ids, tail_memberships = set(), {}
    for section, metrics in physical.items():
        for metric in ("A_gap_eV_atom", "eR_eV_atom_delta_equals_B_delta"):
            for direction in ("largest_increases", "largest_decreases"):
                for row in metrics[metric][direction]:
                    index = row["sample_idx"]
                    tail_ids.add(index)
                    tail_memberships.setdefault(index, []).append({"intersection": section, "metric": metric, "tail": direction})
    tails = []
    for index in sorted(tail_ids):
        raw_values = {arm: {name: finite_value(rows[index], field) for name, field in METRICS.items()}
                      for arm, rows in (("reference", reference), ("method", method))}
        entry = {
            **identities[index], "selected_as_tail": tail_memberships[index], "physical_values": raw_values,
            "physical_deltas": {name: None if raw_values["reference"][name] is None or raw_values["method"][name] is None
                                else raw_values["method"][name] - raw_values["reference"][name] for name in METRICS},
            "reference_input_geometry": geometry["reference"][index],
            "method_input_geometry": geometry["method"][index],
            "input_geometry_deltas": {field: None if finite_value(geometry["reference"][index], field) is None or finite_value(geometry["method"][index], field) is None
                                      else finite_value(geometry["method"][index], field) - finite_value(geometry["reference"][index], field)
                                      for field in geometry_fields},
            "reference_evidence": recorded_failure_evidence(reference[index], reference_paths[index], reference_manifest["terminal_protocol"]),
            "method_evidence": recorded_failure_evidence(method[index], method_paths[index], method_manifest["terminal_protocol"]),
        }
        if endpoint == "tau800":
            entry["upstream_native_token_geometry"] = {
                "reference": stored_geometry(reference_paths[index], endpoint="native", native_body_only=True),
                "method": stored_geometry(method_paths[index], endpoint="native", native_body_only=True),
                "interpretation": "upstream DLM input to refinement, separate from the evaluated tau800 geometry",
            }
        tails.append(entry)
    transitions = {field: transition_cases(reference, method, reference_paths, method_paths, identities, field,
                                           reference_manifest["terminal_protocol"])
                   for field in ("native_execution_success", "endpoint_execution_success", "reconstructed", "terminal_verified")}
    return {
        "schema": SCHEMA, "requests": len(reference), "endpoint": endpoint,
        "cohort_role": reference_manifest["cohort_role"],
        "reference_policy_stage": reference_manifest.get("policy_stage"),
        "method_policy_stage": method_manifest.get("policy_stage"),
        "binary_all_requests": binary, **physical,
        "energy_decomposition": {
            "finite_complete_energy_pairs": paired_energy_decomposition(reference, method, identities, verified=False),
            "common_verified_complete_energy_pairs": paired_energy_decomposition(reference, method, identities, verified=True),
        },
        "execution_and_verification_transitions": transitions,
        "terminal_status_transitions_all_requests": status_transitions(reference, method, "terminal_status"),
        "hull_status_transitions_all_requests": status_transitions(reference, method, "official_hull_status"),
        "hull_and_energy_missingness": {
            arm: {"requests": len(rows), "hull_nonfinite_or_missing": sum(finite_value(row, "hull_energy_eV_atom") is None for row in rows.values()),
                  "terminal_energy_nonfinite_or_missing": sum(finite_value(row, "terminal_energy_eV_atom") is None for row in rows.values()),
                  "statuses": dict(Counter(str(row.get("official_hull_status", "unrecorded")) for row in rows.values()))}
            for arm, rows in (("reference", reference), ("method", method))
        },
        "geometry_finite_intersections": geometry_summary,
        "geometry_coverage": {arm: {"available": sum(row["status"] == "available" for row in rows.values()),
                                    "unavailable": sum(row["status"] != "available" for row in rows.values()),
                                    "reasons": dict(Counter(row["reason"] for row in rows.values() if row["status"] != "available"))}
                              for arm, rows in geometry.items()},
        "A_B_tail_cases": tails,
        "protocol": {key: reference_manifest[key] for key in PROTOCOL_FIELDS},
        "diagnostic_rules": {
            "trim_each_tail_fraction": TRIM_FRACTION_EACH_TAIL,
            "absolute_tail_deletions": list(ABSOLUTE_TAIL_COUNTS), "top_cases_each_sign": 5,
            "periodic_image_radius": IMAGE_RADIUS,
            "delta_sign": "method minus reference; lower energy/force/stress/steps has negative sign",
            "B_semantics": "same-composition delta B equals delta eR; reported absolute eR means are not absolute B",
            "missingness": "no missing physical value or absent verification detail is imputed as zero",
            "geometry_semantics": "actual full cell with centered finite image shell; condition number depends on the stored lattice basis",
            "mcnemar": "exact two-sided discordant-pair calculation, exploratory and unadjusted for multiple comparisons",
            "causal_limit": "posthoc descriptive associations and algebraic decomposition; no module-level causal claim",
            "prohibited_actions_not_performed": ["MLIP evaluation", "new relaxation", "training data selection", "checkpoint selection"],
        },
    }


def render_markdown(report):
    def number(value):
        return "unavailable" if value is None else f"{value:.6g}"

    lines = [f"# Programmed-path regression analysis: {report['endpoint']}", "",
             f"{report['requests']} complete paired requests; method minus reference. "
             "This is a descriptive diagnostic, not a replacement evaluation or a causal attribution.", "",
             "| Outcome | Reference | Method | Lost | Gained | Neither | McNemar descriptive p |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for field, row in report["binary_all_requests"].items():
        lines.append(f"| {field} | {row['reference_count']} | {row['method_count']} | {row['reference_only']} | "
                     f"{row['method_only']} | {row['neither']} | {number(row['exact_mcnemar_two_sided_descriptive'])} |")
    for section in ("finite_physical_intersections", "common_verified_intersections"):
        lines += ["", f"## {section}", "", "| Metric | n / excluded | Mean delta | Median | p10 | p90 | 10% each-tail trimmed | Without abs top 1 / 3 / 5 |",
                  "|---|---:|---:|---:|---:|---:|---:|---|"]
        for field, row in report[section].items():
            trimmed = row["symmetric_trimmed_delta"]["mean"]
            removed = " / ".join(number(row["absolute_tail_removals"][str(k)]["mean_delta_without_tail"]) for k in ABSOLUTE_TAIL_COUNTS)
            lines.append(f"| {field} | {row['n']} / {row['excluded_requests']} | {number(row['delta']['mean'])} | "
                         f"{number(row['delta']['median'])} | {number(row['delta']['p10'])} | {number(row['delta']['p90'])} | {number(trimmed)} | {removed} |")
    lines += ["", "Trimming and absolute-tail deletion are sensitivity descriptions. Full-request counts and original finite-pair means remain unchanged.", "",
              "## Same-pair energy decomposition", ""]
    for name, row in report["energy_decomposition"].items():
        lines.append(f"- {name}: n={row['n']}; mean ΔA={number(row['mean_delta_A'])}, "
                     f"mean Δraw={number(row['mean_delta_raw_energy'])}, mean ΔeR=ΔB={number(row['mean_delta_eR_equals_delta_B'])} eV/atom; "
                     f"maximum identity residual={number(row['maximum_absolute_identity_residual'])}.")
    lines += ["", "Absolute terminal-energy means are eR; only the same-composition differences equal ΔB.", "",
              "## Execution, verification and missingness", ""]
    for field, value in report["execution_and_verification_transitions"].items():
        lines.append(f"- {field}: lost {value['lost_count']}, gained {value['gained_count']}; "
                     f"method statuses on lost requests: {json.dumps(value['lost_method_statuses'], ensure_ascii=False)}.")
        for case in value["lost"]:
            evidence = case["method_evidence"]
            detail = {key: item for key, item in evidence.items() if key.endswith("error") or key.endswith("failure")}
            lines.append(f"  - sample {case['sample_idx']} / group {case['group_id']} / {case['composition']['formula']}: "
                         f"{evidence['terminal_status']}; recorded details {json.dumps(detail, ensure_ascii=False)}.")
    lines += ["", "Only stored terminal statuses, force/stress values, parser/artifact errors and path failures are used. "
              "If optimizer-stop or representation-consistency details were omitted from attempt results, this script cannot reconstruct them.", "",
              "Hull status transitions: " + json.dumps(report["hull_status_transitions_all_requests"], ensure_ascii=False) + ".", "",
              "## A/B tail geometry", "",
              "The cases below are selected by paired A/eR tails in the two stated numerical intersections. They are not a new benchmark subset.", "",
              "| Sample / group / composition | ΔA | ΔeR=ΔB | VPA ref → method | Distinct-pair distance ref → method | Self-image ref → method | Lattice condition ref → method |",
              "|---|---:|---:|---|---|---|---|"]
    for case in report["A_B_tail_cases"]:
        a, b = case["reference_input_geometry"], case["method_input_geometry"]
        arrow = lambda field: f"{number(a.get(field))} → {number(b.get(field))}"
        lines.append(f"| {case['sample_idx']} / {case['group_id']} / {case['composition']['formula']} | "
                     f"{number(case['physical_deltas']['A_gap_eV_atom'])} | {number(case['physical_deltas']['eR_eV_atom_delta_equals_B_delta'])} | "
                     f"{arrow('volume_per_atom_A3')} | {arrow('minimum_distinct_pair_distance_A')} | "
                     f"{arrow('minimum_self_image_distance_A')} | {arrow('lattice_condition_number_2')} |")
    lines += ["", "Distances use the stored full row-vector lattice and a centered 125-image shell; this does not certify exact MIC for an arbitrary unreduced basis. "
              "For tau800, input geometry means the stored refined structure; upstream native token geometry is separate in the JSON.", "",
              "The JSON also contains top-five increases/decreases for every physical metric, signed tail contributions, coverage counts and per-case existing evidence.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference-eval-dir", "method-eval-dir", "reference-paths-jsonl", "method-paths-jsonl", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    input_paths = {"reference_paths": args.reference_paths_jsonl, "method_paths": args.method_paths_jsonl}
    manifests, rows = [], []
    for arm, directory in (("reference", args.reference_eval_dir), ("method", args.method_eval_dir)):
        if not (directory / "_SUCCESS").is_file():
            raise ValueError("evaluation accounting is incomplete")
        input_paths[arm + "_manifest"] = directory / "EVALUATION_FINAL.json"
        input_paths[arm + "_attempts"] = directory / "attempt_results.jsonl"
        manifests.append(json.loads(input_paths[arm + "_manifest"].read_text(encoding="utf-8")))
        rows.append(read_jsonl(input_paths[arm + "_attempts"]))
    before_hashes = {name: sha256_file(path) for name, path in input_paths.items()}
    report = analyze_regressions(rows[0], rows[1], read_jsonl(args.reference_paths_jsonl),
                                read_jsonl(args.method_paths_jsonl), manifests[0], manifests[1])
    after_hashes = {name: sha256_file(path) for name, path in input_paths.items()}
    if before_hashes != after_hashes:
        raise RuntimeError("an evaluation input changed during the read-only diagnostic")
    report["provenance"] = {"input_files": {name: str(path.resolve()) for name, path in input_paths.items()},
                            "input_sha256_before": before_hashes, "input_sha256_after": after_hashes,
                            "script_sha256": sha256_file(Path(__file__))}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "REGRESSION_ANALYSIS.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    (args.output_dir / "analysis.md").write_text(render_markdown(report), encoding="utf-8")
    (args.output_dir / "_SUCCESS").touch()
    print(json.dumps({"requests": report["requests"], "endpoint": report["endpoint"],
                      "binary_all_requests": report["binary_all_requests"],
                      "energy_decomposition": report["energy_decomposition"]}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
