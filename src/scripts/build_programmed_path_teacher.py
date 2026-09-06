#!/usr/bin/env python3
"""One empirical A/B teacher for a complete, fixed train-condition pool."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crystal_dlm.basin_path_objective import solve_basin_path_teacher
from crystal_dlm.programmed_path_data import read_jsonl, trace_summary, training_candidates_per_condition
from crystal_dlm.programmed_path_training import join_terminal_labels
from crystal_dlm.terminal_energy_consistency import TERMINAL_VERIFICATION_PROTOCOL


def uniform_reference_settings(group_ids, reason=None, manifest_path=None):
    """Read an explicit uncertainty decision; never infer groups from scores."""
    if not group_ids:
        if reason is not None or manifest_path is not None:
            raise ValueError("reference-constraint documentation requires --uniform-reference-group")
        return {}, None
    group_ids = [str(group_id) for group_id in group_ids]
    if any(not group_id for group_id in group_ids) or len(set(group_ids)) != len(group_ids):
        raise ValueError("uniform reference group IDs must be nonempty and unique")
    manifest = None
    provenance = {"schema": "uniform_reference_uncertainty_v1", "group_ids": group_ids}
    if manifest_path is not None:
        manifest_path = Path(manifest_path)
        payload = manifest_path.read_bytes()
        manifest = json.loads(payload.decode("utf-8-sig"))
        if not isinstance(manifest, dict):
            raise ValueError("uniform reference manifest must be a JSON object")
        recorded_ids = manifest.get("group_ids")
        if (not isinstance(recorded_ids, list)
                or len(recorded_ids) != len(group_ids)
                or set(map(str, recorded_ids)) != set(group_ids)):
            raise ValueError("manifest group_ids must exactly match the explicit CLI groups")
        manifest_reason = manifest.get("reason")
        if not isinstance(manifest_reason, str) or not manifest_reason.strip():
            raise ValueError("uniform reference manifest requires a nonempty reason")
        if reason is not None and reason.strip() != manifest_reason.strip():
            raise ValueError("CLI and manifest uncertainty reasons differ")
        reason = manifest_reason
        provenance.update(manifest_path=str(manifest_path.resolve()),
                          manifest_sha256=hashlib.sha256(payload).hexdigest(),
                          manifest=manifest)
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("--uniform-reference-group requires --uniform-reference-reason or a reason manifest")
    reason = reason.strip()
    provenance.update(
        reason=reason,
        constraint="q_c equals the original uniform reference over verified finite occurrences",
        interpretation="conservative training-data credibility variant; unresolved uncertainty",
        labels_and_verification_unchanged=True,
        physical_disproof_claimed=False,
        fixed_groups_retain_path_supervision=True,
    )
    return dict.fromkeys(group_ids, reason), provenance


def file_sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--paths-jsonl", type=Path, nargs="+", required=True)
    p.add_argument("--labels-jsonl", type=Path, nargs="+", required=True)
    p.add_argument("--expected-conditions", type=int, default=1024)
    p.add_argument("--candidates", type=int, choices=(4, 8), default=4)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--diagnostic-only", action="store_true")
    p.add_argument("--uniform-reference-group", action="append", default=[], metavar="GROUP_ID",
                   help="Explicitly constrain this group's original reference weights; repeat for more groups.")
    p.add_argument("--uniform-reference-reason",
                   help="Required uncertainty explanation unless supplied by the manifest.")
    p.add_argument("--uniform-reference-manifest", type=Path,
                   help="JSON with exactly the CLI group_ids and a nonempty reason; other evidence is preserved.")
    args = p.parse_args(argv)
    uniform_groups, constraint_provenance = uniform_reference_settings(
        args.uniform_reference_group, args.uniform_reference_reason, args.uniform_reference_manifest,
    )
    for path in [*args.paths_jsonl, *args.labels_jsonl]:
        if not (path.parent / "_SUCCESS").is_file():
            raise ValueError(f"input accounting has not completed: {path}")
    protocols = []
    for path in args.labels_jsonl:
        label_report = json.loads((path.parent / "LABEL_FINAL.json").read_text())
        if label_report["purpose"] != "train":
            raise ValueError("evaluation labels cannot become a train teacher")
        if label_report.get("verification_protocol") != TERMINAL_VERIFICATION_PROTOCOL:
            raise ValueError("formal labels require the uniform terminal-consistency verification")
        protocols.append(label_report["protocol"])
    if any(protocol != protocols[0] for protocol in protocols):
        raise ValueError("terminal protocols differ across teacher label shards")
    expected = {"model": "CHGNet-0.3.0", "optimizer": "FIRE", "relax_cell": True,
                "ase_filter": "FrechetCellFilter", "fmax": .1, "stress_tolerance_GPa": .5,
                "max_steps": 500, "scalar_pressure": 0., "constant_volume": False, "hydrostatic_strain": False}
    if any(protocols[0].get(k) != v for k, v in expected.items()):
        raise ValueError("registered train terminal protocol changed")
    if not args.diagnostic_only and args.expected_conditions != 1024:
        raise ValueError("formal teacher requires the preregistered 1024 train conditions")
    paths = [r for path in args.paths_jsonl for r in read_jsonl(path)]
    labels = [r for path in args.labels_jsonl for r in read_jsonl(path)]
    versions = {json.dumps(r.get("versions"), sort_keys=True) for r in labels if r.get("verified") is True}
    if len(versions) > 1 or "null" in versions:
        raise ValueError("verified teacher labels require one recorded model/package version")
    groups = join_terminal_labels(paths, labels, expected_conditions=args.expected_conditions, candidates=args.candidates)
    if args.candidates != training_candidates_per_condition(int(paths[0]["collection_round"])):
        raise ValueError("registered data budget is K4 initially and K8 for the one refresh")
    teacher = solve_basin_path_teacher(groups, uniform_reference_groups=uniform_groups)
    summary = teacher["summary"]
    summary["label_statuses"] = dict(Counter(r["status"] for r in labels))
    summary["verified_per_condition"] = dict(Counter(sum(c["verified"] is True for c in g["candidates"]) for g in groups))
    execution = [trace_summary(r["trace"]) for r in paths]
    summary["cooperative_attempted_paths"] = sum(bool(r["transactions_by_phase"].get("cooperative", 0)) for r in execution)
    summary["cooperative_accepted_paths"] = sum(bool(r["cooperative_accepted"]) for r in execution)
    summary["cooperative_changed_paths"] = sum(r["committed_changed_scalars_by_phase"].get("cooperative", 0) > 0 for r in execution)
    summary["successful_paths"] = sum(r["success"] for r in paths)
    summary["requested_candidates_per_condition"] = args.candidates
    summary["data_budget_amendment"] = "20260906_K4_then_K8"
    summary["diagnostic_only"] = args.diagnostic_only
    summary["trainable_teacher"] = bool(not args.diagnostic_only and summary["solver_status"] == "optimal"
        and summary["rho_max"] > 0 and summary["primal_residual"] <= 1e-6)
    spans = []
    for group in groups:
        values = [c["terminal_energy"] for c in group["candidates"] if c["verified"] is True]
        if len(values) >= 2:
            spans.append(max(values) - min(values))
    summary["conditions_with_multiple_verified_paths"] = len(spans)
    summary["mean_within_condition_terminal_span_eV_atom"] = sum(spans) / len(spans) if spans else None
    teacher["provenance"] = {"paths_jsonl": [str(p) for p in args.paths_jsonl],
                             "labels_jsonl": [str(p) for p in args.labels_jsonl],
                             "checkpoint": paths[0]["checkpoint"], "collection_round": paths[0]["collection_round"],
                             "candidates_per_condition": args.candidates,
                             "terminal_protocol": protocols[0], "verified_label_versions": [json.loads(v) for v in versions],
                             "verification_protocol": TERMINAL_VERIFICATION_PROTOCOL,
                             "objective": "separate mean improvements in e0-eR and centered eR; no cross-composition ranking"}
    if constraint_provenance is not None:
        teacher["provenance"]["uniform_reference_constraint"] = constraint_provenance
        teacher["provenance"]["source_sha256"] = {
            str(path): file_sha256(path) for path in [*args.paths_jsonl, *args.labels_jsonl]
        }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "teacher.json").write_text(json.dumps(teacher, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.output_dir / "TEACHER_FINAL.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "_SUCCESS").touch()  # Completed solver/accounting, not a positive-gain claim.
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
