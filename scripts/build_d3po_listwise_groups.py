#!/usr/bin/env python3
"""Build frozen same-composition groups for masked listwise alignment.

Only the historical noisy-eight-stream and retired-L7 training sources are
accepted.  Main and sealed D3PO holdout outcomes are deliberately absent from
the command-line interface and are never read by this module.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys
from types import ModuleType
from typing import Any, Mapping, Sequence


SCHEMA = "h1a2_shared_noise_listwise_group_v1"
MANIFEST_SCHEMA = "h1a2_shared_noise_listwise_group_manifest_v1"
ENERGY_LABEL = "post_model494_chgnet_energy_per_atom"


def _load_pair_builder() -> ModuleType:
    """Load the frozen source readers/deduplicator from the pair builder."""

    path = Path(__file__).resolve().with_name("build_d3po_pairs.py")
    spec = importlib.util.spec_from_file_location(
        "_h1a2_frozen_d3po_pair_builder", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load frozen D3PO pair builder: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


PAIR_BUILDER = _load_pair_builder()


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for one required source file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_record(role: str, path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"required listwise source is missing: {resolved}")
    return {
        "role": role,
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def source_file_records(
    noisy_plan_cohort: Path,
    noisy_body_runs: Sequence[Path],
    noisy_eval_runs: Sequence[Path],
    l7_generation: Path,
    l7_final: Path,
) -> list[dict[str, Any]]:
    """Enumerate and hash exactly the files read by the frozen source loaders."""

    if len(noisy_body_runs) != 2 or len(noisy_eval_runs) != 2:
        raise ValueError("listwise source requires exactly two noisy body/eval runs")
    records = [_source_record("noisy_plan_cohort", noisy_plan_cohort)]
    for stream in range(8):
        body_root = noisy_body_runs[0 if stream < 4 else 1]
        eval_root = noisy_eval_runs[0 if stream < 4 else 1]
        records.extend(
            [
                _source_record(
                    f"noisy_stream{stream}_body",
                    body_root / f"stream{stream}/body/raw_generations.jsonl",
                ),
                _source_record(
                    f"noisy_stream{stream}_preofficial_labels",
                    eval_root
                    / (
                        f"stream{stream}/evaluation/full_reconstructed/"
                        "attempt_labels_preofficial.jsonl"
                    ),
                ),
                _source_record(
                    f"noisy_stream{stream}_direct_metrics",
                    eval_root
                    / f"stream{stream}/evaluation/direct/attempt_metrics.jsonl",
                ),
            ]
        )
    for arm in ("base", "g0_all", "g1_strict"):
        records.extend(
            [
                _source_record(
                    f"retired_l7_{arm}_body",
                    l7_generation
                    / f"seed18/{arm}/body/raw_generations.jsonl",
                ),
                _source_record(
                    f"retired_l7_{arm}_official_results",
                    l7_final
                    / f"cells/seed18/{arm}/attempt_results_official.jsonl",
                ),
            ]
        )
    roles = [str(row["role"]) for row in records]
    if len(roles) != len(set(roles)):
        raise AssertionError("source roles must be unique")
    return records


def _candidate_row(row: Mapping[str, Any], *, rank: int) -> dict[str, Any]:
    answer = str(row.get("answer") or "")
    cif = str(row.get("cif") or "")
    energy = float(row["energy_per_atom"])
    if not answer or not cif:
        raise ValueError("every listwise candidate requires answer text and CIF")
    if not math.isfinite(energy):
        raise ValueError("candidate energy must be finite")
    replicate_count = int(row.get("replicate_count", 1))
    replicate_energy_std = float(row.get("replicate_energy_std", 0.0))
    if replicate_count < 1:
        raise ValueError("candidate replicate_count must be positive")
    if not math.isfinite(replicate_energy_std) or replicate_energy_std < 0.0:
        raise ValueError("candidate replicate_energy_std must be finite and non-negative")
    replicate_sources = sorted(
        str(value)
        for value in row.get("replicate_sources", [row.get("source")])
        if value is not None
    )
    source_ordinal = row.get("source_ordinal")
    return {
        "candidate_index": rank,
        "answer": answer,
        "answer_sha256": hashlib.sha256(answer.encode("utf-8")).hexdigest(),
        "cif_sha256": hashlib.sha256(cif.encode("utf-8")).hexdigest(),
        "post_model494_energy_per_atom": energy,
        "source": str(row["source"]),
        "source_ordinal": (
            None if source_ordinal is None else int(source_ordinal)
        ),
        "replicate_count": replicate_count,
        "replicate_energy_std": replicate_energy_std,
        "replicate_sources": replicate_sources,
    }


def _validate_split_rows(
    rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
) -> None:
    identities: set[str] = set()
    chemsys_by_split: dict[str, set[str]] = {}
    for split in ("train", "validation"):
        rows = rows_by_split.get(split)
        if rows is None:
            raise ValueError(f"missing {split} split")
        chemsys_by_split[split] = set()
        for row in rows:
            identity = str(row["composition_id"])
            if identity in identities:
                raise ValueError(f"composition appears more than once: {identity}")
            identities.add(identity)
            chemsys_by_split[split].add(str(row["chemsys"]))
            candidates = row.get("candidates") or ()
            if len(candidates) < 2:
                raise ValueError(f"listwise group {identity} has K < 2")
            if float(row.get("group_weight", 0.0)) != 1.0:
                raise ValueError(f"listwise group {identity} must have weight one")
    overlap = chemsys_by_split["train"] & chemsys_by_split["validation"]
    if overlap:
        raise ValueError(f"train/validation chemsys overlap: {sorted(overlap)}")


def build_listwise_rows(
    groups: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    physical_deduplication: bool,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Make one weight-one row for each eligible exact composition.

    Physical equivalence is delegated to ``build_d3po_pairs.py`` so pairwise
    and listwise assets use the identical strict StructureMatcher contract.
    """

    rows_by_split: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "validation": [],
    }
    dedup_totals: Counter[str] = Counter()
    candidate_sources: Counter[str] = Counter()
    skipped: Counter[str] = Counter()
    for identity, outcomes in sorted(groups.items()):
        if not outcomes:
            skipped["empty_source_group"] += 1
            continue
        plans = [row["plan"] for row in outcomes]
        observed_identities = {
            PAIR_BUILDER.composition_identity(plan) for plan in plans
        }
        if observed_identities != {identity}:
            raise ValueError(f"exact composition changed within group {identity}")
        prompts = {str(row["prompt"]) for row in outcomes}
        if len(prompts) != 1:
            raise ValueError(f"minimal prompt changed within {identity}")
        chemsystems = {PAIR_BUILDER.chemsys(plan) for plan in plans}
        if len(chemsystems) != 1:
            raise ValueError(f"chemsys changed within {identity}")

        unique, dedup_stats = PAIR_BUILDER.deduplicate_outcomes(
            outcomes,
            physical=physical_deduplication,
        )
        dedup_totals.update(dedup_stats)
        if len(unique) < 2:
            skipped["fewer_than_two_physical_candidates"] += 1
            continue
        ordered = sorted(
            unique,
            key=lambda row: (
                float(row["energy_per_atom"]),
                hashlib.sha256(str(row["answer"]).encode("utf-8")).hexdigest(),
            ),
        )
        candidates = [
            _candidate_row(row, rank=index) for index, row in enumerate(ordered)
        ]
        for candidate in candidates:
            for source in candidate["replicate_sources"]:
                candidate_sources[str(source)] += 1
        plan = plans[0]
        chemical_system = next(iter(chemsystems))
        split = PAIR_BUILDER.chemsys_split(chemical_system)
        candidate_fingerprint = [
            {
                "answer_sha256": candidate["answer_sha256"],
                "energy": candidate["post_model494_energy_per_atom"],
                "replicate_count": candidate["replicate_count"],
                "replicate_sources": candidate["replicate_sources"],
            }
            for candidate in candidates
        ]
        rows_by_split[split].append(
            {
                "schema": SCHEMA,
                "group_id": (
                    "d3po-listwise-"
                    + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
                ),
                "split": split,
                "composition_id": identity,
                "chemsys": chemical_system,
                "N": int(plan["N"]),
                "prompt": next(iter(prompts)),
                "energy_label": ENERGY_LABEL,
                "group_weight": 1.0,
                "candidate_count": len(candidates),
                "candidate_set_sha256": PAIR_BUILDER.canonical_sha256(
                    candidate_fingerprint
                ),
                "candidates": candidates,
            }
        )

    for rows in rows_by_split.values():
        rows.sort(key=lambda row: str(row["composition_id"]))
    _validate_split_rows(rows_by_split)
    candidate_counts = [
        int(row["candidate_count"])
        for rows in rows_by_split.values()
        for row in rows
    ]
    audit = {
        "input_exact_composition_groups": len(groups),
        "eligible_exact_composition_groups": len(candidate_counts),
        "skipped_groups": dict(sorted(skipped.items())),
        "deduplication": {
            "method": (
                "exact-answer mean then strict StructureMatcher; "
                "scale=false; no supercells"
            ),
            "physical_enabled": bool(physical_deduplication),
            **dict(sorted(dedup_totals.items())),
        },
        "candidate_source_counts": dict(sorted(candidate_sources.items())),
        "candidate_count_summary": (
            {
                "minimum": min(candidate_counts),
                "maximum": max(candidate_counts),
                "mean": statistics.fmean(candidate_counts),
                "total": sum(candidate_counts),
            }
            if candidate_counts
            else {"minimum": None, "maximum": None, "mean": None, "total": 0}
        ),
    }
    return rows_by_split, audit


def _source_bundle_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    portable = [
        {
            "role": row["role"],
            "bytes": row["bytes"],
            "sha256": row["sha256"],
        }
        for row in records
    ]
    return PAIR_BUILDER.canonical_sha256(portable)


def write_dataset(
    output_dir: Path,
    rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    source_records: Sequence[Mapping[str, Any]],
    audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Write immutable JSONL splits, provenance manifest, and success marker."""

    output = output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    output_hashes = {
        split: PAIR_BUILDER.write_jsonl(output / f"{split}.jsonl", rows)
        for split, rows in rows_by_split.items()
    }
    group_counts = {
        split: len(rows) for split, rows in rows_by_split.items()
    }
    chemsys_counts = {
        split: len({str(row["chemsys"]) for row in rows})
        for split, rows in rows_by_split.items()
    }
    code_hashes = {
        "build_d3po_listwise_groups.py": sha256_file(Path(__file__).resolve()),
        "build_d3po_pairs.py": sha256_file(
            Path(__file__).resolve().with_name("build_d3po_pairs.py")
        ),
    }
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "split_salt": PAIR_BUILDER.SPLIT_SALT,
        "energy_label": ENERGY_LABEL,
        "group_weight": 1.0,
        "minimum_candidates_per_group": 2,
        "group_counts": group_counts,
        "chemsys_counts": chemsys_counts,
        "chemsys_disjoint": True,
        "source_policy": "historical noisy-eight-stream plus retired L7 only",
        "main_holdout_outcomes_read": False,
        "sealed_holdout_outcomes_read": False,
        "source_files": list(source_records),
        "source_bundle_sha256": _source_bundle_sha256(source_records),
        "code_hashes": code_hashes,
        "output_hashes": output_hashes,
        "audit": dict(audit),
        "gpu_jobs_used": 0,
    }
    manifest_path = output / "D3PO_LISTWISE_GROUP_MANIFEST.json"
    with manifest_path.open("x", encoding="utf-8") as handle:
        handle.write(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n"
        )
    manifest_sha256 = sha256_file(manifest_path)
    with (output / "_SUCCESS").open("x", encoding="utf-8") as handle:
        handle.write(manifest_sha256 + "\n")
    return {**manifest, "manifest_sha256": manifest_sha256}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build train-only listwise groups; holdout outcome inputs are "
            "intentionally unsupported."
        )
    )
    parser.add_argument("--noisy-plan-cohort", type=Path, required=True)
    parser.add_argument("--noisy-body-run", type=Path, action="append", required=True)
    parser.add_argument("--noisy-eval-run", type=Path, action="append", required=True)
    parser.add_argument("--l7-generation", type=Path, required=True)
    parser.add_argument("--l7-final", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    noisy_body_runs = [path.resolve() for path in args.noisy_body_run]
    noisy_eval_runs = [path.resolve() for path in args.noisy_eval_run]
    noisy_plan_cohort = args.noisy_plan_cohort.resolve()
    l7_generation = args.l7_generation.resolve()
    l7_final = args.l7_final.resolve()

    source_records_before = source_file_records(
        noisy_plan_cohort,
        noisy_body_runs,
        noisy_eval_runs,
        l7_generation,
        l7_final,
    )
    noisy, exclusions = PAIR_BUILDER.noisy_outcomes(
        noisy_plan_cohort,
        noisy_body_runs,
        noisy_eval_runs,
    )
    retired_l7 = PAIR_BUILDER.l7_outcomes(l7_generation, l7_final)
    groups = PAIR_BUILDER.merge_groups(noisy, retired_l7)
    source_records_after = source_file_records(
        noisy_plan_cohort,
        noisy_body_runs,
        noisy_eval_runs,
        l7_generation,
        l7_final,
    )
    if source_records_before != source_records_after:
        raise RuntimeError("a frozen listwise source changed while it was being read")

    rows_by_split, audit = build_listwise_rows(
        groups,
        physical_deduplication=True,
    )
    for split, rows in rows_by_split.items():
        if not rows:
            raise RuntimeError(f"frozen listwise source produced an empty {split} split")
    audit["noisy_exclusions"] = dict(sorted(exclusions.items()))
    manifest = write_dataset(
        args.output_dir,
        rows_by_split,
        source_records=source_records_before,
        audit=audit,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
