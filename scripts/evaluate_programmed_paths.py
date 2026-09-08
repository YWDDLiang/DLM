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
from crystal_dlm.terminal_energy_consistency import COMMON_RELAXATION_PROTOCOL, LABEL_GEOMETRY_PROTOCOL, TERMINAL_VERIFICATION_PROTOCOL
from crystal_dlm.exact_sun_nu import conjunction, evaluate_sun_predicates, fingerprint


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
            "strict_sun": conjunction(strict, novel, unique), "meta_sun": conjunction(meta, novel, unique),
            "verified_strict_stable": bool(verified and strict), "verified_meta_stable": bool(verified and meta),
            "verified_strict_sun": conjunction(bool(verified), strict, novel, unique),
            "verified_meta_sun": conjunction(bool(verified), meta, novel, unique)}


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


def positive_request_count(value):
    """Allow a declared cohort size without changing all-request accounting."""
    try:
        count = int(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("expected-requests must be a positive integer") from error
    if count <= 0:
        raise argparse.ArgumentTypeError("expected-requests must be a positive integer")
    return count


def endpoint_cache_key(record):
    if not record['success']:
        return record['trajectory_id']
    geometry = json.dumps(record['structure'], sort_keys=True) if record.get('structure') is not None else record.get('body')
    if not isinstance(geometry, str):
        raise ValueError('a successful endpoint has no actual geometry')
    return hashlib.sha256(geometry.encode()).hexdigest()


def load_bound_evaluation_labels(records, label_files, *, paths_file, endpoint, expected_model_sha256=None,
                                 purpose='evaluation', feedback_scope=None):
    if purpose not in ('evaluation', 'training_feedback') or (purpose == 'training_feedback' and feedback_scope is None):
        raise ValueError('explicit validated scope is required for training rewards')
    expected_split = 'evaluation' if purpose == 'evaluation' else 'train'
    inputs = {record['trajectory_id']: record for record in records}
    if len(inputs) != len(records):
        raise ValueError('duplicate endpoint request identity')
    expected_input_sha = sha256_file(paths_file)
    labels, reports, runtime = {}, [], None
    for path in map(Path, label_files):
        directory = path.parent
        if not (directory/'_SUCCESS').is_file() or (directory/'_ENGINEERING_FAILED').exists():
            raise ValueError('physics accounting is incomplete or contains engineering failures')
        report = json.loads((directory/'LABEL_FINAL.json').read_text())
        rows = read_jsonl(path)
        if (report.get('purpose') != purpose or report.get('protocol') != COMMON_RELAXATION_PROTOCOL
                or report.get('geometry_validation_protocol') != LABEL_GEOMETRY_PROTOCOL
                or report.get('verification_protocol') != TERMINAL_VERIFICATION_PROTOCOL
                or report.get('requested') != len(rows) or report.get('completed') != len(rows)
                or report.get('statuses') != dict(Counter(row['status'] for row in rows))
                or any(row['status'] in ('worker_error', 'worker_timeout') for row in rows)):
            raise ValueError('label purpose, full physical protocol or completion receipt differs')
        if purpose == 'training_feedback' and report.get('training_feedback_scope') != feedback_scope:
            raise ValueError('training labels and reward inputs have different source/exclusion contracts')
        identities = report.get('runtime_identities')
        if (not isinstance(identities, list) or len(identities) != 1 or
                not {'model','model_checkpoint_sha256','chgnet_package','ase_package','torch_package',
                     'pymatgen_package','labeler_sha256'}.issubset(identities[0])):
            raise ValueError('complete label model/runtime identity is required')
        if runtime is not None and runtime != identities[0]:
            raise ValueError('label shards use different physical/runtime implementations')
        runtime = identities[0]
        if (runtime['model'] != COMMON_RELAXATION_PROTOCOL['model']
                or any(not isinstance(runtime.get(key), str) or not runtime[key] for key in
                       ('model', 'model_checkpoint_sha256', 'chgnet_package', 'ase_package',
                        'torch_package', 'pymatgen_package', 'labeler_sha256'))
                or any(len(runtime[key]) != 64 or any(c not in '0123456789abcdef' for c in runtime[key])
                       for key in ('model_checkpoint_sha256','labeler_sha256'))
                or expected_model_sha256 is not None and runtime['model_checkpoint_sha256'] != expected_model_sha256):
            raise ValueError('label runtime does not identify the pinned physical model')
        if ('deterministic_algorithms_enabled' in runtime and type(runtime['deterministic_algorithms_enabled']) is not bool
                or runtime.get('cublas_workspace_config') is not None and not isinstance(runtime['cublas_workspace_config'], str)
                or runtime.get('deterministic_algorithms_enabled') is True and runtime.get('cublas_workspace_config') not in (':4096:8', ':16:8')):
            raise ValueError('invalid deterministic numerical runtime identity')
        if report.get('input_sha256') != expected_input_sha:
            source = Path(report.get('input_file') or '')
            if not source.is_file() or sha256_file(source) != report.get('input_sha256'):
                raise ValueError('label input bytes are not bound to this endpoint or a verified component')
            component_ids = {record['trajectory_id'] for record in read_jsonl(source)}
            if not {row['trajectory_id'] for row in rows}.issubset(component_ids):
                raise ValueError('label occurrences are absent from their pinned source component')
        for label in rows:
            key = label['trajectory_id']
            if key not in inputs or key in labels:
                raise ValueError('extra or duplicate terminal label occurrence')
            record = inputs[key]
            if (label.get('endpoint_cache_key') != endpoint_cache_key(record)
                    or label.get('versions') != runtime
                    or any(label.get(field) != record.get(field) for field in
                           ('group_id','source_row_idx','source_split','endpoint'))
                    or label.get('endpoint') != endpoint or label.get('source_split') != expected_split):
                raise ValueError('terminal label belongs to a different exact endpoint or occurrence')
            if not record['success'] and (label.get('status') != 'generation_failure' or label.get('verified') is not False):
                raise ValueError('a failed input acquired a physical label')
            if any(word in str(label.get('error') or '').lower() for word in
                   ('out of memory','cuda error','modulenotfounderror','brokenprocesspool')):
                raise ValueError('an engineering evaluation error cannot be scored as unstable')
            labels[key] = label
        reports.append({'labels_path': str(path.resolve()), 'labels_sha256': sha256_file(path),
                        'report_sha256': sha256_file(directory/'LABEL_FINAL.json'), 'runtime': runtime,
                        'input_sha256': report['input_sha256']})
    if set(labels) != set(inputs):
        raise ValueError('terminal labels do not cover every input request exactly')
    return labels, reports


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--paths-jsonl", type=Path, required=True)
    p.add_argument("--labels-jsonl", type=Path, nargs="+", required=True)
    p.add_argument("--frozen-config", type=Path, required=True)
    p.add_argument("--official-cache", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--expected-requests", type=positive_request_count, required=True)
    p.add_argument("--selection-json", type=Path)
    p.add_argument("--endpoint", choices=("native", "tau800"), required=True)
    p.add_argument("--cohort-role", choices=("fixed_development", "independent_main", "training_feedback"), required=True)
    p.add_argument('--feedback-manifest', type=Path)
    p.add_argument("--policy-stage", choices=("reference", "round0_diagnostic", "final", "unspecified"), default="unspecified")
    p.add_argument('--sun-only', action='store_true', help='Evaluate only the exact N/U predicates needed for SUN/MSUN')
    p.add_argument('--nu-workers', type=int, default=4)
    p.add_argument('--nu-pair-timeout', type=float, default=30.)
    p.add_argument('--nu-cache', type=Path)
    args = p.parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("full N/U evaluation must use its allocated workflow CPUs")
    if args.output_dir.exists():
        raise ValueError('evaluation output must be new; use the shared pair cache to resume work')
    if args.sun_only and not 1 <= args.nu_workers <= max(1, int(os.environ.get('SLURM_CPUS_PER_TASK', '1'))-1):
        raise ValueError('pair workers exceed the reserved CPU budget, including the coordinator')
    records = read_jsonl(args.paths_jsonl)
    indexed = {r["trajectory_id"]: r for r in records}
    if len(indexed) != len(records) or len(records) != args.expected_requests:
        raise ValueError("evaluation request denominator changed")
    if [int(r.get("evaluation_ordinal", r["sample_idx"])) for r in records] != list(range(len(records))):
        raise ValueError("evaluation source order changed")
    feedback_scope = None
    purpose = 'training_feedback' if args.cohort_role == 'training_feedback' else 'evaluation'
    if purpose == 'training_feedback':
        if args.feedback_manifest is None or args.selection_json is not None:
            raise ValueError('training feedback requires its bound manifest and all requests')
        from crystal_dlm.sun_feedback_contract import validate_training_feedback
        feedback_scope = validate_training_feedback(records, args.paths_jsonl, args.feedback_manifest, endpoint=args.endpoint)
    elif args.feedback_manifest is not None or any(r.get('source_split') != 'evaluation' for r in records):
        raise ValueError('train paths cannot supply evaluation results')
    if any(r.get("endpoint", args.endpoint) != args.endpoint for r in records):
        raise ValueError("input evaluation endpoints were mixed")
    config = json.loads(args.frozen_config.read_text())
    model_sha = sha256_file(config['assets']['chgnet_runtime_checkpoint'])
    labels, label_bindings = load_bound_evaluation_labels(records, args.labels_jsonl,
                             paths_file=args.paths_jsonl, endpoint=args.endpoint, expected_model_sha256=model_sha,
                             purpose=purpose, feedback_scope=feedback_scope)
    protocols = [COMMON_RELAXATION_PROTOCOL]
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
    if not (args.official_cache / "completion_SUCCESS").is_file():
        raise ValueError("official cache accounting is incomplete")
    unresolved = {r["chemsys"] for r in read_jsonl(args.official_cache / "unresolved_chemsys.jsonl")}
    systems = {"-".join(sorted(e.symbol for e in s.composition.elements)) for s in structures}
    from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
    from pymatgen.core import Composition
    diagrams = {}
    for row in read_jsonl(args.official_cache / "official_slim_cache.jsonl"):
        if row["chemsys"] in systems and row["chemsys"] not in unresolved:
            if row['chemsys'] in diagrams:
                raise ValueError('official cache contains duplicated chemistry records')
            diagrams[row["chemsys"]] = PhaseDiagram([
                PDEntry(Composition(e["composition"]), float(e["energy"]), name=str(e.get("entry_id", "")))
                for e in row["entries"]])
    missing_systems = systems - set(diagrams) - unresolved
    if missing_systems:
        raise ValueError('official cache coverage is missing; this is an engineering gap: '+', '.join(sorted(missing_systems)))
    hulls = {}
    for index, structure in enumerate(structures):
        system = '-'.join(sorted(e.symbol for e in structure.composition.elements))
        hull = float(diagrams[system].get_hull_energy_per_atom(structure.composition)) if system in diagrams else None
        if hull is not None and not math.isfinite(hull):
            raise ValueError('an official phase diagram produced a nonfinite hull energy')
        hulls[index] = (system, hull, 'known' if hull is not None else 'official_cache_unresolved')
    required = []
    for record in records:
        index = reconstructed.get(record['trajectory_id'])
        if index is None:
            continue
        energy, hull = labels[record['trajectory_id']]['terminal_energy'], hulls[index][1]
        if energy is not None and hull is not None and math.isfinite(float(energy)) and float(energy)-hull <= .1:
            required.append(index)
    novel, unique, nu_report = [None]*len(structures), [None]*len(structures), None
    if structures and (required or not args.sun_only):
        train_structures, train_index = evaluator.load_training_index(config['assets']['train_csv'])
        if args.sun_only:
            training_identity = {'csv_sha256': sha256_file(config['assets']['train_csv']),
                                 'index_order_sha256': fingerprint(sorted((key,list(value)) for key,value in train_index.items()))}
            cache_path = config['assets'].get('training_index_cache')
            if cache_path and Path(cache_path).is_file():
                training_identity['index_cache_sha256'] = sha256_file(cache_path)
            nu_report = evaluate_sun_predicates(structures, train_structures, train_index, required,
                         cache_dir=args.nu_cache or args.output_dir.parent/'directed_nu_cache',
                         frozen_nu_sha256=evaluator_hash, training_identity=training_identity,
                         workers=args.nu_workers, pair_timeout=args.nu_pair_timeout)
            novel, unique = nu_report['novel'], nu_report['unique']
        else:
            matcher = evaluator.StructureMatcher(ltol=.2, stol=.3, angle_tol=5)
            novel = list(map(bool, evaluator.compute_novelty(structures, train_structures, train_index, matcher)))
            classes, count = evaluator.compute_uniqueness(structures, matcher)
            seen, representatives = set(), set()
            for index, label in enumerate(classes):
                if int(label) not in seen:
                    representatives.add(index)
                    seen.add(int(label))
            if len(representatives) != count:
                raise ValueError('frozen unique representative accounting differs')
            unique = [index in representatives for index in range(len(structures))]
    nu_complete = nu_report is None or nu_report['complete_sun_predicates']
    output = []
    for ordinal, record in enumerate(records):
        label = labels[record["trajectory_id"]]
        if label.get("group_id") != record.get("group_id"):
            raise ValueError("label condition mismatch")
        if label.get("endpoint") not in (None, args.endpoint):
            raise ValueError("native and refined label endpoints were mixed")
        index = reconstructed.get(record["trajectory_id"])
        is_novel = novel[index] if index is not None else None if args.sun_only else False
        is_unique = unique[index] if index is not None else None if args.sun_only else False
        system, hull = None, None
        hull_status = "input_not_reconstructed"
        if index is not None:
            system, hull, hull_status = hulls[index]
        output.append({"trajectory_id": record["trajectory_id"], "ordinal": ordinal,
                       "group_id": record.get("group_id"), "sample_idx": record["sample_idx"],
                       "reconstructed": index is not None, "parser_error": parser_errors.get(record["trajectory_id"]),
                       "endpoint_execution_success": record["success"],
                       "native_execution_success": record.get("native_execution_success", record["success"]), "novel": is_novel,
                       "unique_representative": is_unique, "novel_unique": conjunction(is_novel, is_unique),
                       "nu_needed_for_sun": index in required,
                       "terminal_verified": label["verified"], "terminal_status": label["status"],
                       "raw_energy_eV_atom": label["raw_energy"], "terminal_energy_eV_atom": label["terminal_energy"],
                       "gap_eV_atom": label.get("gap"), "raw": label.get("raw"), "terminal": label.get("terminal"),
                       "actual_relaxation_steps": label.get("actual_steps"), "chemsys": system,
                       "hull_energy_eV_atom": hull, "official_hull_status": hull_status,
                       **classify_stability(verified=label["verified"], energy=label["terminal_energy"],
                            hull_energy=hull, novel=is_novel, unique=is_unique)})
    counts = {key: None if any(r[key] is None for r in output) else sum(r[key] is True for r in output) for key in
              ("reconstructed", "native_execution_success", "endpoint_execution_success", "novel", "unique_representative", "novel_unique",
               "terminal_verified", "strict_stable", "meta_stable", "strict_sun", "meta_sun",
               "verified_strict_stable", "verified_meta_stable", "verified_strict_sun", "verified_meta_sun")}
    counts["requests"] = len(output)
    def percent(value, denominator=len(output)):
        return None if value is None else 100*value/denominator
    report = {"counts": counts, "endpoint": args.endpoint, "cohort_role": args.cohort_role,
              'purpose': purpose, 'training_feedback_scope': feedback_scope,
              'independent_evaluation': args.cohort_role == 'independent_main',
              "status": 'complete' if nu_complete else 'incomplete_required_NU',
              "sun_only": args.sun_only, "standalone_NU_complete": all(row['novel'] is not None and row['unique_representative'] is not None for row in output),
              "nu_evaluation": nu_report, "nu_pair_timeout_seconds": args.nu_pair_timeout if args.sun_only else None,
              "input_sha256": sha256_file(args.paths_jsonl), "label_bindings": label_bindings,
              "physical_model_sha256": model_sha,
              "frozen_config_sha256": sha256_file(args.frozen_config),
              "official_cache_sha256": sha256_file(args.official_cache/'official_slim_cache.jsonl'),
              "official_unresolved_sha256": sha256_file(args.official_cache/'unresolved_chemsys.jsonl'),
              "policy_stage": args.policy_stage,
              "verification_protocol": TERMINAL_VERIFICATION_PROTOCOL,
              "terminal_protocol": protocols[0], "main_stability_criterion": "retained terminal-energy threshold",
              "geometry_validation_protocol": LABEL_GEOMETRY_PROTOCOL,
              "verified_subset_adds": "optimizer stop, force/stress, geometry and consistent-energy checks",
              "novelty_uniqueness_endpoint": "input_structure_before_common_CHGNet_relaxation",
              "strict_sun_percent": percent(counts['strict_sun']),
              "meta_sun_percent": percent(counts['meta_sun']),
              "verified_strict_sun_percent": percent(counts['verified_strict_sun']),
              "verified_meta_sun_percent": percent(counts['verified_meta_sun']),
              "strictly_exceeds_10_and_50": (counts['strict_sun']*100 > 10*len(output) and counts['meta_sun']*100 > 50*len(output))
                  if counts['strict_sun'] is not None and counts['meta_sun'] is not None else None,
              "verified_subset_strictly_exceeds_10_and_50": (counts['verified_strict_sun']*100 > 10*len(output) and counts['verified_meta_sun']*100 > 50*len(output))
                  if counts['verified_strict_sun'] is not None and counts['verified_meta_sun'] is not None else None,
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
        selected_counts = {key: None if any(r[key] is None for r in selected_output) else sum(r[key] is True for r in selected_output)
                           for key in counts if key != "requests"}
        selected_counts["requests"] = len(selected_output)
        report["conditional_1000"] = {"counts": selected_counts,
            "strict_sun_percent": percent(selected_counts['strict_sun'], 1000),
            "meta_sun_percent": percent(selected_counts['meta_sun'], 1000),
            "verified_strict_sun_percent": percent(selected_counts['verified_strict_sun'], 1000),
            "verified_meta_sun_percent": percent(selected_counts['verified_meta_sun'], 1000),
            "selection": selection,
            "nu_semantics": "first parseable prefix: later rows cannot change earlier uniqueness representatives"}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "attempt_results.jsonl").open("x", encoding="utf-8") as handle:
        for row in output:
            handle.write(json.dumps(row) + "\n")
    report_name = 'FEEDBACK_FINAL.json' if purpose == 'training_feedback' else 'EVALUATION_FINAL.json'
    (args.output_dir / report_name).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if selected_output:
        with (args.output_dir / "conditional_1000_results.jsonl").open("x", encoding="utf-8") as handle:
            for row in selected_output:
                handle.write(json.dumps(row) + "\n")
    if not nu_complete:
        (args.output_dir/'_INCOMPLETE_NU').touch()
        print(json.dumps(report), flush=True)
        raise SystemExit(2)
    (args.output_dir / "_SUCCESS").touch()
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
