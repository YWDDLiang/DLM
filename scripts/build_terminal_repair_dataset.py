#!/usr/bin/env python3
"""Prepare Q(R(x)) targets, then admit them only after separate physics labels."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from crystal_dlm.programmed_path_data import read_jsonl
from crystal_dlm.terminal_energy_consistency import TERMINAL_VERIFICATION_PROTOCOL
from crystal_dlm.terminal_repair_data import (
    TARGET_ADMISSION, composition_key, encode_terminal_pair, repair_split, target_admission,
)


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def write_jsonl(path, rows):
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def checked_labels(paths):
    result, protocol, versions = {}, None, set()
    for path in paths:
        if not (path.parent / "_SUCCESS").is_file():
            raise ValueError(f"incomplete label collection: {path}")
        report = json.loads((path.parent / "LABEL_FINAL.json").read_text())
        if report.get("purpose") != "train":
            raise ValueError("evaluation endpoints cannot supply training targets")
        if report.get("verification_protocol") != TERMINAL_VERIFICATION_PROTOCOL:
            raise ValueError("terminal representation verification differs")
        if protocol is not None and report["protocol"] != protocol:
            raise ValueError("relaxation protocols differ")
        protocol = report["protocol"]
        for row in read_jsonl(path):
            identity = row["trajectory_id"]
            if identity in result:
                raise ValueError("duplicate label identity")
            result[identity] = row
            if row.get("verified") is True:
                versions.add(json.dumps(row.get("versions"), sort_keys=True))
    expected = {"model": "CHGNet-0.3.0", "optimizer": "FIRE", "relax_cell": True,
                "ase_filter": "FrechetCellFilter", "fmax": .1, "stress_tolerance_GPa": .5,
                "max_steps": 500, "scalar_pressure": 0., "constant_volume": False,
                "hydrostatic_strain": False, "fire_dt": .1, "fire_maxstep": .2,
                "cell_mask": "all_six"}
    if any(protocol.get(key) != value for key, value in expected.items()):
        raise ValueError("the frozen common relaxation protocol changed")
    if len(versions) > 1 or "null" in versions:
        raise ValueError("verified labels require one recorded model/package version")
    return result, protocol


def prepare(args):
    from transformers import AutoTokenizer
    from scripts.sample_llada_dynamic_crystals import build_dynamic_lightweight_constraints
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), trust_remote_code=True)
    constraints = build_dynamic_lightweight_constraints(
        tokenizer, duplicate_coordinate_mask=True, lattice_volume_mask=True,
        min_lattice_rad=1e-4, canonicalize_periodic_alias=True,
        pbc_min_distance_mask=True, pbc_min_distance_A=.5, pbc_image_radius=2,
    )
    records = []
    for path in args.parent_paths:
        if not (path.parent / "_SUCCESS").is_file():
            raise ValueError(f"incomplete parent collection: {path}")
        records.extend(read_jsonl(path))
    labels, protocol = checked_labels(args.labels)
    identities = [row["trajectory_id"] for row in records]
    if len(records) != args.expected_requests or len(set(identities)) != len(records):
        raise ValueError("registered parent request count or occurrence uniqueness differs")
    if set(labels) != set(identities):
        raise ValueError("parent and label denominator mismatch")
    from crystal_dlm.programmed_path_data import training_candidates_per_condition
    groups, metadata = {}, {}
    rounds = {row["collection_round"] for row in records}
    if not rounds or not rounds <= {0, 1}:
        raise ValueError("repair preparation accepts only the recorded K4 and K8 collections")
    expected_keys = {(round_id, candidate) for round_id in rounds
                     for candidate in range(training_candidates_per_condition(round_id))}
    for row in records:
        group = str(row["group_id"])
        groups.setdefault(group, []).append((row["collection_round"], row["candidate_index"]))
        condition = json.dumps({key: row[key] for key in
                                ("plan_state", "species_program", "species_program_source", "prompt")}, sort_keys=True)
        if group in metadata and metadata[group] != condition:
            raise ValueError("a repeated condition changed its Planner/program metadata")
        metadata[group] = condition
    if len(groups) != 1024 or any(len(keys) != len(expected_keys) or set(keys) != expected_keys
                                  for keys in groups.values()):
        raise ValueError("each of the fixed 1024 conditions requires its complete K4/K8 occurrences")
    args.output.mkdir(parents=True, exist_ok=False)
    ledger, pairs, all_targets = [], [], []
    coverage = Counter()
    for row in records:
        split, chemistry = repair_split(row), composition_key(row)
        entry = {"parent_trajectory_id": row["trajectory_id"], "repair_split": split,
                 "group_id": row["group_id"], "composition_key": chemistry}
        coverage[f"{split}_requests"] += 1
        try:
            pair = encode_terminal_pair(row, labels[row["trajectory_id"]], tokenizer, constraints)
            pairs.append(pair)
            entry["status"] = "awaiting_quantized_physics"
            all_targets.append({key: pair[key] for key in
                                ("trajectory_id", "group_id", "source_row_idx", "source_split", "success", "body")})
        except (ValueError, KeyError, TypeError) as error:
            entry.update(status="unavailable_target", reason=f"{type(error).__name__}: {error}")
            all_targets.append({
                "trajectory_id": "quantized-terminal:" + row["trajectory_id"],
                "group_id": row["group_id"], "source_row_idx": row["source_row_idx"],
                "source_split": row["source_split"], "success": False, "body": None,
            })
        ledger.append(entry)
    write_jsonl(args.output / "pair_candidates.jsonl", pairs)
    write_jsonl(args.output / "target_paths.jsonl", all_targets)
    write_jsonl(args.output / "source_ledger.jsonl", ledger)
    write_jsonl(args.output / "validation_parent_paths.jsonl",
                [row for row in records if repair_split(row) == "validation"])
    report = {
        "requested_parent_paths": len(records), "encodable_verified_pairs": len(pairs),
        "source_compositions": len({composition_key(row) for row in records}),
        "split_before_outcomes": "sha256(reduced_composition), fixed 10% validation",
        "coverage": dict(coverage), "statuses": dict(Counter(row["status"] for row in ledger)),
        "unavailable_reasons": dict(Counter(row.get("reason") for row in ledger if row.get("reason"))),
        "target_admission": TARGET_ADMISSION, "terminal_protocol": protocol,
        "sources": {str(path): digest(path) for path in [*args.parent_paths, *args.labels]},
        "target_physics_complete": False, "new_database_geometry_targets": False,
        "collection_rounds": sorted(rounds), "condition_ids": sorted(groups),
        "repair_holdout_is_not_a_claim_of_unseen_base_model_compositions": True,
    }
    (args.output / "PREPARE_FINAL.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "_SUCCESS").touch()
    print(json.dumps(report), flush=True)


def finalize(args):
    prepared_paths = args.prepared
    if any(not (path / "_SUCCESS").is_file() for path in prepared_paths):
        raise ValueError("target preparation is incomplete")
    reports = [json.loads((path / "PREPARE_FINAL.json").read_text()) for path in prepared_paths]
    initial = reports[0]
    if any(report["target_admission"] != TARGET_ADMISSION for report in reports):
        raise ValueError("target criteria changed after preparation")
    if any(report["terminal_protocol"] != initial["terminal_protocol"]
           or report["condition_ids"] != initial["condition_ids"] for report in reports):
        raise ValueError("prepared collections differ in protocol or condition pool")
    labels, protocol = checked_labels(args.labels)
    expected = [row for path in prepared_paths for row in read_jsonl(path / "target_paths.jsonl")]
    if (args.expected_requests not in (4096, 12288) or len(expected) != args.expected_requests
            or len({row["trajectory_id"] for row in expected}) != len(expected)):
        raise ValueError("repair source must be complete unique K4 or complete K4+K8 requests")
    if set(labels) != {row["trajectory_id"] for row in expected}:
        raise ValueError("quantized target labels lost requests")
    if protocol != initial["terminal_protocol"]:
        raise ValueError("quantized targets used a different common relaxation")
    candidates = [row for path in prepared_paths for row in read_jsonl(path / "pair_candidates.jsonl")]
    admitted, ledger = [], []
    for pair in candidates:
        label = labels[pair["trajectory_id"]]
        reasons = target_admission(pair, label)
        reliable = label.get("verified") is True
        ledger.append({"trajectory_id": pair["trajectory_id"], "repair_split": pair["repair_split"],
                       "admitted": not reasons, "reasons": reasons,
                       "quantized_gap": label.get("gap"), "quantized_raw": label.get("raw"),
                       "delta_A": label["gap"] - pair["parent_gap"] if reliable else None,
                       "delta_B": label["terminal_energy"] - pair["parent_terminal_energy"] if reliable else None,
                       "delta_native_energy": label["raw_energy"] - pair["parent_raw_energy"]
                       if label.get("raw_energy") is not None else None})
        if not reasons:
            admitted.append(dict(
                pair, target_supervision_ready=True,
                quantized_relaxation_verified=True,
                quantized_target_verified=False,  # raw Q is not certified by terminal-R thresholds
                quantized_raw_energy=label["raw_energy"],
                quantized_relaxation_gap=label["gap"],
                quantized_terminal_energy=label["terminal_energy"],
            ))
    training = [pair for pair in admitted if pair["repair_split"] == "train"]
    validation = [pair for pair in admitted if pair["repair_split"] == "validation"]
    counts = Counter(pair["composition_key"] for pair in training)
    for pair in training:
        pair["sample_weight"] = len(training) / (len(counts) * counts[pair["composition_key"]])
    args.output.mkdir(parents=True, exist_ok=False)
    write_jsonl(args.output / "train.jsonl", training)
    write_jsonl(args.output / "validation_targets.jsonl", validation)
    write_jsonl(args.output / "quantized_admission_ledger.jsonl", ledger)
    validation_parents = [row for path in prepared_paths
                          for row in read_jsonl(path / "validation_parent_paths.jsonl")]
    write_jsonl(args.output / "validation_parent_paths.jsonl", validation_parents)
    quantized_changes = {}
    for scope, rows in (("all_reliably_compared", ledger), ("admitted", [row for row in ledger if row["admitted"]])):
        quantized_changes[scope] = {}
        for key in ("delta_A", "delta_B", "delta_native_energy"):
            values = [row[key] for row in rows if row[key] is not None]
            quantized_changes[scope][key] = {
                "count": len(values), "mean": sum(values) / len(values) if values else None,
                "worse_than_1meV_count": sum(value > .001 for value in values),
            }
    report = {
        "parent_requests": len(expected), "pair_candidates": len(candidates),
        "admitted_pairs": len(admitted), "train_pairs": len(training),
        "validation_pairs": len(validation), "train_compositions": len(counts),
        "validation_compositions": len({pair["composition_key"] for pair in validation}),
        "target_admission": TARGET_ADMISSION,
        "rejection_reasons": dict(Counter(reason for row in ledger for reason in row["reasons"])),
        "prepared_sources": {str(path): digest(path / "PREPARE_FINAL.json") for path in prepared_paths},
        "label_sources": {str(path): digest(path) for path in args.labels},
        "teacher_target_changes": quantized_changes,
        "unfiltered_repair_validation_requests": len(validation_parents),
        "raw_quantized_targets_are_not_certified_minima": True,
        "train_weighting": "equal reduced composition, uniform admitted occurrences within composition",
    }
    (args.output / "DATA_FINAL.json").write_text(json.dumps(report, indent=2) + "\n")
    if not training or not validation:
        raise ValueError("train or held-out target coverage is empty; artifacts retained, no success")
    (args.output / "_SUCCESS").touch()
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("prepare", "finalize"), required=True)
    parser.add_argument("--parent-paths", nargs="+", type=Path)
    parser.add_argument("--labels", nargs="+", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--prepared", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-requests", type=int, default=12288)
    args = parser.parse_args()
    if args.mode == "prepare":
        if not args.parent_paths or not args.tokenizer:
            parser.error("prepare requires parent paths and tokenizer")
        prepare(args)
    else:
        if args.prepared is None:
            parser.error("finalize requires prepared targets")
        finalize(args)


if __name__ == "__main__":
    main()
