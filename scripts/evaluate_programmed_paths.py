#!/usr/bin/env python3
"""Frozen input N/U, energy-threshold SUN, and its verified-terminal subset."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from crystal_dlm.dynamic_crystal import arrays_to_structure, parse_dynamic_answer
from crystal_dlm.programmed_path_data import read_jsonl


def describe(values):
    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return {"count": len(finite), "mean": statistics.fmean(finite) if finite else None,
            "median": statistics.median(finite) if finite else None,
            "minimum": min(finite) if finite else None, "maximum": max(finite) if finite else None}


def classify_stability(*, verified, energy, hull_energy, novel, unique):
    known = energy is not None and hull_energy is not None and math.isfinite(float(energy)) and math.isfinite(float(hull_energy))
    e_hull = float(energy) - float(hull_energy) if known else None
    strict = bool(known and e_hull <= 0.)
    meta = bool(known and e_hull <= .1)
    return {"e_above_hull_eV_atom": e_hull, "strict_stable": strict, "meta_stable": meta,
            "strict_sun": bool(strict and novel and unique), "meta_sun": bool(meta and novel and unique),
            "verified_strict_stable": bool(verified and strict), "verified_meta_stable": bool(verified and meta),
            "verified_strict_sun": bool(verified and strict and novel and unique),
            "verified_meta_sun": bool(verified and meta and novel and unique)}


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_input_structure(record):
    if record.get("parseable") is False:
        raise ValueError(record.get("artifact_error") or "recorded CIF parser failure")
    if record.get("structure") is not None:
        from pymatgen.core import Structure
        return Structure.from_dict(record["structure"])
    if record.get("endpoint") == "tau800":
        raise ValueError("refined endpoint is missing; native body is not a substitute")
    return arrays_to_structure(parse_dynamic_answer(record["body"], strict=True))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--paths-jsonl", type=Path, required=True)
    p.add_argument("--labels-jsonl", type=Path, nargs="+", required=True)
    p.add_argument("--frozen-config", type=Path, required=True)
    p.add_argument("--official-cache", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--expected-requests", type=int, choices=(256, 1000, 1200), required=True)
    p.add_argument("--selection-json", type=Path)
    p.add_argument("--endpoint", choices=("native", "tau800"), required=True)
    p.add_argument("--cohort-role", choices=("fixed_development", "independent_main"), required=True)
    args = p.parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("full N/U evaluation must use its allocated workflow CPUs")
    records = read_jsonl(args.paths_jsonl)
    indexed = {r["trajectory_id"]: r for r in records}
    if len(indexed) != len(records) or len(records) != args.expected_requests:
        raise ValueError("evaluation request denominator changed")
    if [int(r.get("evaluation_ordinal", r["sample_idx"])) for r in records] != list(range(len(records))):
        raise ValueError("evaluation source order changed")
    if any(r.get("source_split") != "evaluation" for r in records):
        raise ValueError("train paths cannot supply evaluation results")
    if any(r.get("endpoint", args.endpoint) != args.endpoint for r in records):
        raise ValueError("input evaluation endpoints were mixed")
    labels, protocols = {}, []
    for path in args.labels_jsonl:
        if not (path.parent / "_SUCCESS").is_file():
            raise ValueError("terminal label accounting has not completed")
        report = json.loads((path.parent / "LABEL_FINAL.json").read_text())
        if report["purpose"] != "evaluation":
            raise ValueError("wrong label purpose")
        protocols.append(report["protocol"])
        for row in read_jsonl(path):
            if row["trajectory_id"] in labels:
                raise ValueError("duplicate terminal label occurrence")
            labels[row["trajectory_id"]] = row
    if set(labels) != set(indexed) or any(p != protocols[0] for p in protocols):
        raise ValueError("label/request alignment or terminal protocol differs")
    expected = {"model": "CHGNet-0.3.0", "optimizer": "FIRE", "relax_cell": True,
                "ase_filter": "FrechetCellFilter", "fmax": .1, "stress_tolerance_GPa": .5,
                "max_steps": 500, "scalar_pressure": 0., "constant_volume": False, "hydrostatic_strain": False}
    if any(protocols[0].get(k) != v for k, v in expected.items()):
        raise ValueError("registered common terminal protocol changed")
    config = json.loads(args.frozen_config.read_text())
    evaluator_path = Path(config["assets"]["eval_sun_py"])
    evaluator_hash = sha256_file(evaluator_path)
    if evaluator_hash != config["frozen_code"]["eval_sun_sha256"]:
        raise ValueError("frozen novelty/uniqueness implementation changed")
    spec = importlib.util.spec_from_file_location("eval_sun", evaluator_path)
    evaluator = importlib.util.module_from_spec(spec)
    sys.modules["eval_sun"] = evaluator
    spec.loader.exec_module(evaluator)
    structures, reconstructed = [], {}
    parser_errors = {}
    for record in records:
        try:
            structure = read_input_structure(record)
            if not math.isfinite(float(structure.volume)) or structure.volume <= 0:
                raise ValueError("nonpositive or nonfinite cell")
            reconstructed[record["trajectory_id"]] = len(structures)
            structures.append(structure)
        except Exception as error:
            parser_errors[record["trajectory_id"]] = f"{type(error).__name__}: {error}"
    novel, unique_representatives = [], set()
    if structures:
        train_structures, train_index = evaluator.load_training_index(config["assets"]["train_csv"])
        matcher = evaluator.StructureMatcher(ltol=.2, stol=.3, angle_tol=5)
        novel = evaluator.compute_novelty(structures, train_structures, train_index, matcher)
        classes, count = evaluator.compute_uniqueness(structures, matcher)
        seen = set()
        for index, label in enumerate(classes):
            if int(label) not in seen:
                unique_representatives.add(index)
                seen.add(int(label))
        if len(unique_representatives) != count:
            raise ValueError("frozen unique representative accounting differs")
    if not (args.official_cache / "completion_SUCCESS").is_file():
        raise ValueError("official cache accounting is incomplete")
    unresolved = {r["chemsys"] for r in read_jsonl(args.official_cache / "unresolved_chemsys.jsonl")}
    systems = {"-".join(sorted(e.symbol for e in s.composition.elements)) for s in structures}
    from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
    from pymatgen.core import Composition
    diagrams = {}
    for row in read_jsonl(args.official_cache / "official_slim_cache.jsonl"):
        if row["chemsys"] in systems and row["chemsys"] not in unresolved:
            diagrams[row["chemsys"]] = PhaseDiagram([
                PDEntry(Composition(e["composition"]), float(e["energy"]), name=str(e.get("entry_id", "")))
                for e in row["entries"]])
    output = []
    for ordinal, record in enumerate(records):
        label = labels[record["trajectory_id"]]
        if str(label["group_id"]) != str(record["group_id"]):
            raise ValueError("label condition mismatch")
        if label.get("endpoint") not in (None, args.endpoint):
            raise ValueError("native and refined label endpoints were mixed")
        index = reconstructed.get(record["trajectory_id"])
        is_novel = bool(novel[index]) if index is not None else False
        is_unique = index in unique_representatives if index is not None else False
        system, hull = None, None
        hull_status = "input_not_reconstructed"
        if index is not None:
            structure = structures[index]
            system = "-".join(sorted(e.symbol for e in structure.composition.elements))
            if system in diagrams:
                hull = float(diagrams[system].get_hull_energy_per_atom(structure.composition))
                hull_status = "known"
            else:
                hull_status = "official_cache_unresolved" if system in unresolved else "official_cache_not_covered"
        output.append({"trajectory_id": record["trajectory_id"], "ordinal": ordinal,
                       "group_id": record["group_id"], "sample_idx": record["sample_idx"],
                       "reconstructed": index is not None, "parser_error": parser_errors.get(record["trajectory_id"]),
                       "endpoint_execution_success": record["success"],
                       "native_execution_success": record.get("native_execution_success", record["success"]), "novel": is_novel,
                       "unique_representative": is_unique, "novel_unique": is_novel and is_unique,
                       "terminal_verified": label["verified"], "terminal_status": label["status"],
                       "raw_energy_eV_atom": label["raw_energy"], "terminal_energy_eV_atom": label["terminal_energy"],
                       "gap_eV_atom": label.get("gap"), "raw": label.get("raw"), "terminal": label.get("terminal"),
                       "actual_relaxation_steps": label.get("actual_steps"), "chemsys": system,
                       "hull_energy_eV_atom": hull, "official_hull_status": hull_status,
                       **classify_stability(verified=label["verified"], energy=label["terminal_energy"],
                            hull_energy=hull, novel=is_novel, unique=is_unique)})
    counts = {key: sum(bool(r[key]) for r in output) for key in
              ("reconstructed", "native_execution_success", "endpoint_execution_success", "novel", "unique_representative", "novel_unique",
               "terminal_verified", "strict_stable", "meta_stable", "strict_sun", "meta_sun",
               "verified_strict_stable", "verified_meta_stable", "verified_strict_sun", "verified_meta_sun")}
    counts["requests"] = len(output)
    report = {"counts": counts, "endpoint": args.endpoint, "cohort_role": args.cohort_role,
              "terminal_protocol": protocols[0], "main_stability_criterion": "retained terminal-energy threshold",
              "verified_subset_adds": "optimizer stop, force/stress, geometry and consistent-energy checks",
              "novelty_uniqueness_endpoint": "input_structure_before_common_CHGNet_relaxation",
              "strict_sun_percent": 100 * counts["strict_sun"] / len(output),
              "meta_sun_percent": 100 * counts["meta_sun"] / len(output),
              "verified_strict_sun_percent": 100 * counts["verified_strict_sun"] / len(output),
              "verified_meta_sun_percent": 100 * counts["verified_meta_sun"] / len(output),
              "strictly_exceeds_10_and_50": counts["strict_sun"] * 100 > 10 * len(output) and counts["meta_sun"] * 100 > 50 * len(output),
              "verified_subset_strictly_exceeds_10_and_50": counts["verified_strict_sun"] * 100 > 10 * len(output) and counts["verified_meta_sun"] * 100 > 50 * len(output),
              "raw_energy_eV_atom": describe(r["raw_energy_eV_atom"] for r in output),
              "verified_gap_eV_atom": describe(r["gap_eV_atom"] for r in output if r["terminal_verified"]),
              "verified_terminal_eV_atom": describe(r["terminal_energy_eV_atom"] for r in output if r["terminal_verified"]),
              "raw_force_max_eV_A": describe((r["raw"] or {}).get("force_max_eV_A") for r in output),
              "raw_stress_max_GPa": describe((r["raw"] or {}).get("stress_max_GPa") for r in output),
              "label_statuses": dict(Counter(r["terminal_status"] for r in output)),
              "hull_statuses": dict(Counter(r["official_hull_status"] for r in output)),
              "frozen_nu_source_sha256": evaluator_hash, "official_cache": str(args.official_cache),
              "new_official_query": False, "historical_protocol_comparison": "new common relaxation protocol with verification reported separately; historical values are not relabeled"}
    selected_output = []
    if args.selection_json:
        selection = json.loads(args.selection_json.read_text())
        chosen = selection["selected_source_ordinals"]
        if len(chosen) != 1000 or chosen != sorted(set(chosen)) or max(chosen) >= len(output):
            raise ValueError("conditional main sample must have 1000 fixed source ordinals")
        if selection["selection_basis"] != "CIF_parser_only_in_source_order" or selection["energy_or_stability_selection"] is not False:
            raise ValueError("main sample selection was not parser-only")
        if args.endpoint == "native" and chosen != [i for i, r in enumerate(output) if r["reconstructed"]][:1000]:
            raise ValueError("frozen selection is not the first 1000 reconstructed inputs")
        selected_output = [dict(output[i], source_request_ordinal=i, ordinal=j) for j, i in enumerate(chosen)]
        selected_counts = {key: sum(bool(r[key]) for r in selected_output) for key in counts if key != "requests"}
        selected_counts["requests"] = len(selected_output)
        report["conditional_1000"] = {"counts": selected_counts,
            "strict_sun_percent": selected_counts["strict_sun"] / 10,
            "meta_sun_percent": selected_counts["meta_sun"] / 10,
            "verified_strict_sun_percent": selected_counts["verified_strict_sun"] / 10,
            "verified_meta_sun_percent": selected_counts["verified_meta_sun"] / 10,
            "selection": selection,
            "nu_semantics": "first parseable prefix: later rows cannot change earlier uniqueness representatives"}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "attempt_results.jsonl").open("x", encoding="utf-8") as handle:
        for row in output:
            handle.write(json.dumps(row) + "\n")
    (args.output_dir / "EVALUATION_FINAL.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if selected_output:
        with (args.output_dir / "conditional_1000_results.jsonl").open("x", encoding="utf-8") as handle:
            for row in selected_output:
                handle.write(json.dumps(row) + "\n")
    (args.output_dir / "_SUCCESS").touch()
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
