"""CPU-only audit of whole-target integer translations and species assignments.

The objective is periodic fractional squared representation distance in integer
bin units. It is not Cartesian displacement, an atom trajectory, or an energy.
Original records and physics-label provenance are retained without modification.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import time

import numpy as np
import scipy
from scipy.optimize import linear_sum_assignment


PERIOD = 100
TOKENIZER_SHA256 = "3a21588abca8e56155cc7b6cabb81df51992ccd2e89704aec770912f24e75509"
SCHEMA = "gauge32_integer_translation_species_assignment_audit_v1"


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def canonical_hash(value):
    return sha256_bytes(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8"))


def parse_body(tokens):
    if not isinstance(tokens, list) or not tokens:
        raise ValueError("a token body must be a nonempty list")
    match = re.fullmatch(r"<N_(\d{3})>", tokens[0])
    if not match:
        raise ValueError("missing native count token")
    count = int(match[1])
    if not 1 <= count <= 20 or len(tokens) != 7 + 4 * count:
        raise ValueError("body violates the native 7+4N layout")
    for position, family in enumerate(("LA", "LB", "LC", "AA", "AB", "AG"), 1):
        if not re.fullmatch(rf"<{family}_(\d{{3}})>", tokens[position]):
            raise ValueError(f"wrong lattice token family at {position}")
    species, raw_bins = [], []
    for site in range(count):
        start = 7 + 4 * site
        element = re.fullmatch(r"<E_([A-Z][a-z]?)>", tokens[start])
        if not element:
            raise ValueError(f"wrong species token at site {site}")
        species.append(element[1])
        coordinates = []
        for axis, family in enumerate("XYZ"):
            coordinate = re.fullmatch(rf"<{family}_(\d{{3}})>", tokens[start + 1 + axis])
            if not coordinate or not 0 <= int(coordinate[1]) <= PERIOD:
                raise ValueError(f"wrong coordinate token at site {site}, axis {axis}")
            coordinates.append(int(coordinate[1]))
        raw_bins.append(coordinates)
    raw_bins = np.asarray(raw_bins, dtype=np.int64)
    return {
        "count": count, "species": species, "raw_bins": raw_bins,
        "bins": raw_bins % PERIOD, "alias_count": int((raw_bins == PERIOD).sum()),
    }


def render_coordinates(template, bins, *, alias_mask=None):
    result = list(template)
    for site, coordinates in enumerate(bins):
        for axis, value in enumerate(coordinates):
            value = int(value)
            if alias_mask is not None and bool(alias_mask[site, axis]):
                if value != 0:
                    raise AssertionError("inverse alias restoration did not recover bin zero")
                value = PERIOD
            result[8 + 4 * site + axis] = f"<{'XYZ'[axis]}_{value:03d}>"
    return result


def squared_distances(left, right):
    delta = np.abs(left[:, None, :] - right[None, :, :])
    delta = np.minimum(delta, PERIOD - delta)
    return np.sum(delta * delta, axis=-1, dtype=np.int64)


def paired_cost(left, right):
    delta = np.abs(left - right)
    delta = np.minimum(delta, PERIOD - delta)
    return int(np.sum(delta * delta, dtype=np.int64))


def minimum_assignment_cost(cost):
    if not len(cost):
        return 0
    rows, columns = linear_sum_assignment(cost)
    return int(cost[rows, columns].sum(dtype=np.int64))


def lexicographic_optimal_assignment(cost):
    """Smallest column-index tuple among assignments with minimum integer cost.

    Rows are in OLD slot order; callers sort columns by shifted XYZ and original
    target index. Repeated constrained Hungarian calls implement an exact tie
    rule without floating epsilon perturbations or solver-dependent ties.
    """
    count = len(cost)
    columns = list(range(count))
    remaining_cost = minimum_assignment_cost(cost)
    selected = []
    for row in range(count):
        for column in columns:
            rest = [value for value in columns if value != column]
            future = cost[np.ix_(list(range(row + 1, count)), rest)]
            future_cost = minimum_assignment_cost(future)
            if int(cost[row, column]) + future_cost == remaining_cost:
                selected.append(column)
                columns = rest
                remaining_cost = future_cost
                break
        else:
            raise AssertionError("no lexicographic optimal assignment continuation")
    if remaining_cost != 0:
        raise AssertionError("assignment cost accounting failed")
    return selected


def assignment_for_shift(old, target, species, shift, *, resolve_ties):
    shifted = (target + np.asarray(shift, dtype=np.int64)) % PERIOD
    mapping = [-1] * len(species)
    total = 0
    for element in sorted(set(species)):
        old_slots = [i for i, symbol in enumerate(species) if symbol == element]
        target_slots = sorted(
            old_slots, key=lambda j: (tuple(map(int, shifted[j])), j)
        )
        cost = squared_distances(old[old_slots], shifted[target_slots])
        total += minimum_assignment_cost(cost)
        if resolve_ties:
            chosen = lexicographic_optimal_assignment(cost)
            for row, column in enumerate(chosen):
                mapping[old_slots[row]] = target_slots[column]
    if not resolve_ties:
        return total
    aligned = shifted[mapping]
    if paired_cost(old, aligned) != total:
        raise AssertionError("assignment reconstruction does not attain its cost")
    return total, aligned, mapping


def align_target(old, target, species):
    shifts = {(0, 0, 0)}
    for i, element in enumerate(species):
        for j, other in enumerate(species):
            if element == other:
                shifts.add(tuple(map(int, (old[i] - target[j]) % PERIOD)))
    costs = [
        (assignment_for_shift(old, target, species, shift, resolve_ties=False), shift)
        for shift in sorted(shifts)
    ]
    best_cost = min(cost for cost, _ in costs)
    minimizers = [shift for cost, shift in costs if cost == best_cost]
    resolved = []
    for shift in minimizers:
        cost, aligned, mapping = assignment_for_shift(
            old, target, species, shift, resolve_ties=True
        )
        key = (cost, tuple(map(int, aligned.reshape(-1))), shift, tuple(mapping))
        resolved.append((key, aligned, mapping))
    key, aligned, mapping = min(resolved, key=lambda item: item[0])
    zero_cost, zero_aligned, zero_mapping = assignment_for_shift(
        old, target, species, (0, 0, 0), resolve_ties=True
    )
    return {
        "aligned": aligned, "permutation": mapping, "shift": list(key[2]),
        "cost": best_cost, "candidate_count": len(shifts),
        "min_cost_shift_count": len(minimizers),
        "zero_shift_cost": zero_cost, "zero_shift_aligned": zero_aligned,
        "zero_shift_permutation": zero_mapping,
    }


def element_xyz_multiset(species, bins):
    return Counter((symbol, *map(int, row)) for symbol, row in zip(species, bins))


def analyze_row(record):
    old, target = parse_body(record["old"]), parse_body(record["target"])
    if old["count"] != target["count"] or old["species"] != target["species"]:
        raise ValueError(f"{record['record_id']}: ordered N/species slots differ")
    result = align_target(old["bins"], target["bins"], old["species"])
    aligned = result["aligned"]
    shift = np.asarray(result["shift"], dtype=np.int64)
    permutation = result["permutation"]
    aligned_tokens = render_coordinates(record["target"], aligned)
    old_canonical = render_coordinates(record["old"], old["bins"])
    target_canonical = render_coordinates(record["target"], target["bins"])
    untranslated = (aligned - shift) % PERIOD
    recovered = np.empty_like(target["bins"])
    for old_slot, original_target_slot in enumerate(permutation):
        recovered[original_target_slot] = untranslated[old_slot]
    inverse_tokens = render_coordinates(
        record["target"], recovered, alias_mask=target["raw_bins"] == PERIOD
    )
    protected = [0, *[7 + 4 * site for site in range(old["count"])]]
    checks = {
        "permutation_is_bijective": sorted(permutation) == list(range(old["count"])),
        "permutation_is_species_preserving": all(
            old["species"][i] == target["species"][j] for i, j in enumerate(permutation)
        ),
        "inverse_recovers_original_integer_target": bool(np.array_equal(recovered, target["bins"])),
        "inverse_recovers_original_target_tokens_including_aliases": inverse_tokens == record["target"],
        "element_xyz_multiset_after_undoing_global_shift_equal": (
            element_xyz_multiset(old["species"], untranslated)
            == element_xyz_multiset(target["species"], target["bins"])
        ),
        "target_lattice_tokens_unchanged": aligned_tokens[1:7] == record["target"][1:7],
        "target_N_and_ordered_species_tokens_unchanged": all(
            aligned_tokens[p] == record["target"][p] for p in protected
        ),
        "OLD_N_and_ordered_species_contract_preserved": all(
            aligned_tokens[p] == record["old"][p] for p in protected
        ),
    }
    if not all(checks.values()):
        raise AssertionError(f"{record['record_id']}: equivalence check failed: {checks}")
    positions = set(record["action"]["positions"])
    original_changed = [i for i, (a, b) in enumerate(zip(old_canonical, target_canonical)) if a != b]
    aligned_changed = [i for i, (a, b) in enumerate(zip(old_canonical, aligned_tokens)) if a != b]
    action_checks = {
        "whole_coordinate_action": record["action"]["mode"] in ("full_cell", "all_xyz"),
        "original_target_changes_within_action": set(original_changed).issubset(positions),
        "aligned_target_changes_within_same_action": set(aligned_changed).issubset(positions),
        "aligned_changes_outside_action": sorted(set(aligned_changed) - positions),
    }
    before = paired_cost(old["bins"], target["bins"])
    if result["cost"] > result["zero_shift_cost"] or result["zero_shift_cost"] > before:
        raise AssertionError("zero-shift baseline was not retained in optimization")
    return {
        **record,  # Original OLD, target, action and source identifiers are untouched.
        "input_record_sha256": canonical_hash(record),
        "old_token_sequence_sha256": canonical_hash(record["old"]),
        "original_target_token_sequence_sha256": canonical_hash(record["target"]),
        "aligned_target": aligned_tokens,
        "aligned_target_token_sequence_sha256": canonical_hash(aligned_tokens),
        "num_atoms": old["count"],
        "coordinate_tokens": 3 * old["count"],
        "mapping": {
            "shift_bins_xyz": result["shift"],
            "old_slot_to_original_target_slot": permutation,
            "rule": "aligned[i] = (original_target[permutation[i]] + shift_bins) mod 100",
            "candidate_shifts": result["candidate_count"],
            "minimum_cost_candidate_shifts": result["min_cost_shift_count"],
        },
        "metrics": {
            "original_coordinate_token_changes": int((old["bins"] != target["bins"]).sum()),
            "permutation_only_coordinate_token_changes": int((old["bins"] != result["zero_shift_aligned"]).sum()),
            "aligned_coordinate_token_changes": int((old["bins"] != aligned).sum()),
            "original_integer_periodic_squared_distance": before,
            "permutation_only_integer_periodic_squared_distance": result["zero_shift_cost"],
            "aligned_integer_periodic_squared_distance": result["cost"],
            "original_fractional_squared_distance": before / PERIOD**2,
            "aligned_fractional_squared_distance": result["cost"] / PERIOD**2,
            "original_zero_action": old_canonical == target_canonical,
            "aligned_zero_action": old_canonical == aligned_tokens,
            "aligned_coordinate_identity": bool(np.array_equal(old["bins"], aligned)),
            "old_alias_100_tokens": old["alias_count"],
            "target_alias_100_tokens": target["alias_count"],
        },
        "equivalence_checks": checks,
        "action_contract_checks": action_checks,
        "physical_label_modified": False,
        "independently_physics_evaluated": False,
    }


def algorithm_self_checks():
    generator = np.random.default_rng(713)
    tested = 0
    for size in range(1, 6):
        for _ in range(6):
            cost = generator.integers(0, 4, size=(size, size), dtype=np.int64)
            expected = min(
                (sum(int(cost[i, j]) for i, j in enumerate(p)), p)
                for p in itertools.permutations(range(size))
            )[1]
            if tuple(lexicographic_optimal_assignment(cost)) != expected:
                raise AssertionError("lexicographic assignment differs from exhaustive optimum")
            tested += 1
    old = np.asarray([[0, 0, 0], [50, 50, 50], [20, 20, 20]], dtype=np.int64)
    teacher = old[[1, 0, 2]]
    first = align_target(old, teacher, ["Na", "Na", "Cl"])
    equivalent_teacher = (teacher[[1, 0, 2]] + np.asarray([23, 37, 11])) % PERIOD
    second = align_target(old, equivalent_teacher, ["Na", "Na", "Cl"])
    if first["cost"] != 0 or not np.array_equal(first["aligned"], second["aligned"]):
        raise AssertionError("equivalent teacher representatives changed the alignment result")
    return {
        "exhaustive_integer_assignment_cases": tested,
        "lexicographic_optimal_assignment_passed": True,
        "translated_permuted_teacher_same_representative_passed": True,
    }


def summarize(rows):
    metrics = [row["metrics"] for row in rows]
    total_tokens = sum(row["coordinate_tokens"] for row in rows)
    before_tokens = sum(row["original_coordinate_token_changes"] for row in metrics)
    after_tokens = sum(row["aligned_coordinate_token_changes"] for row in metrics)
    before_cost = sum(row["original_integer_periodic_squared_distance"] for row in metrics)
    after_cost = sum(row["aligned_integer_periodic_squared_distance"] for row in metrics)
    def distribution(key):
        values = [row[key] for row in metrics]
        return {"sum": int(sum(values)), "mean": float(np.mean(values)),
                "median": float(np.median(values)), "min": int(min(values)), "max": int(max(values))}
    return {
        "records": len(rows), "independent_ancestors": len({row["ancestor_id"] for row in rows}),
        "total_atoms": sum(row["num_atoms"] for row in rows), "coordinate_tokens": total_tokens,
        "atom_count_histogram": dict(sorted(Counter(row["num_atoms"] for row in rows).items())),
        "action_mode_counts": dict(Counter(row["action"]["mode"] for row in rows)),
        **{key: distribution(key) for key in (
            "original_coordinate_token_changes", "permutation_only_coordinate_token_changes",
            "aligned_coordinate_token_changes", "original_integer_periodic_squared_distance",
            "permutation_only_integer_periodic_squared_distance", "aligned_integer_periodic_squared_distance",
        )},
        "coordinate_token_change_reduction_fraction": (before_tokens - after_tokens) / before_tokens if before_tokens else 0.0,
        "integer_periodic_squared_distance_reduction_fraction": (before_cost - after_cost) / before_cost if before_cost else 0.0,
        "token_change_count_decreased_records": sum(row["aligned_coordinate_token_changes"] < row["original_coordinate_token_changes"] for row in metrics),
        "token_change_count_equal_records": sum(row["aligned_coordinate_token_changes"] == row["original_coordinate_token_changes"] for row in metrics),
        "token_change_count_increased_records": sum(row["aligned_coordinate_token_changes"] > row["original_coordinate_token_changes"] for row in metrics),
        "distance_strictly_decreased_records": sum(row["aligned_integer_periodic_squared_distance"] < row["original_integer_periodic_squared_distance"] for row in metrics),
        "original_zero_actions": sum(row["original_zero_action"] for row in metrics),
        "aligned_zero_actions": sum(row["aligned_zero_action"] for row in metrics),
        "aligned_coordinate_identity_records": sum(row["aligned_coordinate_identity"] for row in metrics),
        "nonzero_global_shift_records": sum(any(row["mapping"]["shift_bins_xyz"]) for row in rows),
        "nonidentity_permutation_records": sum(row["mapping"]["old_slot_to_original_target_slot"] != list(range(row["num_atoms"])) for row in rows),
        "all_strict_equivalence_checks_pass": all(all(row["equivalence_checks"].values()) for row in rows),
        "all_original_action_contracts_preserved": all(
            all(value for key, value in row["action_contract_checks"].items() if key != "aligned_changes_outside_action")
            for row in rows
        ),
        "labels_modified": 0, "new_physics_evaluations": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(__file__).with_name("GAUGE32_INPUT.json"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("GAUGE32_AUDIT.json"))
    args = parser.parse_args()
    started = time.monotonic()
    input_bytes = args.input.read_bytes()
    payload = json.loads(input_bytes)
    if not re.fullmatch(r"[0-9a-f]{64}", payload.get("source_sha256", "")):
        raise ValueError("missing original dataset hash")
    if payload.get("tokenizer_sha256") != TOKENIZER_SHA256:
        raise ValueError("input does not declare the preserved B0 tokenizer identity")
    records = payload["rows"]
    if (len(records) != 32 or len({row["ancestor_id"] for row in records}) != 32
            or len({row["record_id"] for row in records}) != 32
            or any(not row["record_id"].startswith("S:positive:") for row in records)):
        raise ValueError("audit requires the exact declared 32 independent S positive records")
    self_checks = algorithm_self_checks()
    rows = [analyze_row(row) for row in records]
    if args.input.read_bytes() != input_bytes:
        raise RuntimeError("input changed while the audit was running")
    output = {
        "schema": SCHEMA, "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_path": str(args.input.resolve()), "input_file_sha256": sha256_bytes(input_bytes),
        "source_sha256": payload["source_sha256"], "tokenizer_sha256": payload["tokenizer_sha256"],
        "source_hash_scope": "dataset/tokenizer hashes are retained input declarations; their original files were not re-read",
        "audit_script_sha256": sha256_bytes(Path(__file__).read_bytes()),
        "runtime": {"numpy": np.__version__, "scipy": scipy.__version__, "device": "CPU"},
        "method": {
            "coordinate_period_bins": PERIOD,
            "objective": "sum_i sum_axis min(abs(OLD_bin-aligned_bin), 100-abs(OLD_bin-aligned_bin))^2",
            "distance_interpretation": "periodic fractional representation distance; not Cartesian distance or physical motion",
            "candidate_shifts": "zero plus all (OLD_i - TARGET_j) mod 100 for equal-species pairs",
            "assignment": "one common global shift, then independent same-species Hungarian assignment of whole XYZ triples",
            "tie_rule": "minimum exact integer cost; lexicographically smallest aligned XYZ in OLD slot order; shift XYZ; original-target permutation",
            "assignment_tie_rule": "exact lexicographic optimum via constrained Hungarian calls; no floating epsilon",
            "lattice_and_atom_types": "unchanged; no basis conversion, rotation, clipping or re-quantization",
            "physical_equivalence_proof": "bijective same-species inverse reconstruction plus equality of complete element/XYZ multisets after undoing the global shift",
            "physics_labels": "not changed or re-evaluated; aligned_target is an audited representation only",
        },
        "algorithm_self_checks": self_checks, "summary": summarize(rows), "rows": rows,
        "elapsed_cpu_wall_seconds": time.monotonic() - started,
    }
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps({"output": str(args.output.resolve()), "summary": output["summary"],
                      "elapsed_cpu_wall_seconds": output["elapsed_cpu_wall_seconds"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
