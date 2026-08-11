#!/usr/bin/env python3
"""Recompute only E_hull and S labels, preserving explicit MP unknowns."""

from __future__ import annotations

import argparse
import copy
import math
import os
import shutil
import statistics
from pathlib import Path
from typing import Any

from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.core import Composition
from pymatgen.entries.computed_entries import ComputedEntry

from protocol import (
    ContractError,
    canonical_sha256,
    finite_float,
    identity,
    normalized_chemsys,
    read_json,
    read_jsonl,
    require_source_manifest,
    sha256_file,
    write_json_exclusive,
    write_jsonl_exclusive,
)


STRICT_THRESHOLD = 0.0
META_THRESHOLD = 0.1


def finite_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "median": None, "mean": None, "max": None}
    return {
        "count": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "max": max(values),
    }


def load_clean_cache(path: Path) -> dict[str, list[dict[str, Any]]]:
    cache: dict[str, list[dict[str, Any]]] = {}
    for row in read_jsonl(path):
        chemsys = str(row["chemsys"])
        entries = row.get("entries")
        if chemsys in cache or not isinstance(entries, list) or not entries:
            raise ContractError(f"invalid or duplicate clean cache row: {chemsys}")
        cache[chemsys] = entries
    return cache


def load_unresolved(path: Path) -> dict[str, dict[str, Any]]:
    unresolved: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        chemsys = str(row["chemsys"])
        if chemsys in unresolved or row.get("reason") != "official_gga_gga_u_missing_yb_unary_reference":
            raise ContractError(f"invalid or duplicate unresolved cache row: {chemsys}")
        unresolved[chemsys] = row
    return unresolved


def phase_diagram(entries: list[dict[str, Any]], chemsys: str) -> PhaseDiagram:
    decoded = [
        ComputedEntry(
            row["composition"],
            finite_float(row["energy"], "reference entry energy"),
            entry_id=row.get("entry_id"),
        )
        for row in entries
    ]
    diagram = PhaseDiagram(decoded)
    expected = set(chemsys.split("-"))
    observed = {element.symbol for element in diagram.elements}
    if observed != expected:
        raise ContractError(
            f"clean phase diagram elements {sorted(observed)} != {sorted(expected)}"
        )
    return diagram


def exact_hull(
    diagram: PhaseDiagram, composition: Composition, energy_per_atom: float
) -> float:
    _, raw = diagram.get_decomp_and_e_above_hull(
        PDEntry(composition, energy_per_atom * composition.num_atoms),
        allow_negative=True,
    )
    return max(finite_float(raw, "clean e_above_hull"), 0.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--cell-index", type=int, required=True)
    args = parser.parse_args()

    source = args.source_dir.resolve()
    require_source_manifest(source, args.source_manifest_sha256)
    run_root = args.run_root.resolve()
    input_manifest_path = run_root / "inputs/input_manifest.json"
    cache_manifest_path = run_root / "official_mp_cache/completion_manifest.json"
    cache_path = run_root / "official_mp_cache/official_slim_cache.jsonl"
    unresolved_path = run_root / "official_mp_cache/unresolved_chemsys.jsonl"
    if not (run_root / "official_mp_cache/completion_SUCCESS").is_file():
        raise ContractError("official MP cache is incomplete")
    input_manifest = read_json(input_manifest_path)
    cache_manifest = read_json(cache_manifest_path)
    if (
        cache_manifest.get("query_status") != "complete_with_explicit_hull_unknown"
        or cache_manifest.get("unresolved_policy")
        != "explicit_hull_unknown_excluded_from_skip_unknown_denominators"
        or int(cache_manifest.get("new_mp_queries", -1)) != 0
    ):
        raise ContractError("official MP skip-unknown cache contract changed")
    expected_cache_sha = cache_manifest["outputs"]["slim_evaluation_cache"]["sha256"]
    if sha256_file(cache_path) != expected_cache_sha:
        raise ContractError("official slim cache identity changed")
    expected_unresolved_sha = cache_manifest["outputs"]["unresolved_chemsys"]["sha256"]
    if sha256_file(unresolved_path) != expected_unresolved_sha:
        raise ContractError("official unresolved-chemsys identity changed")

    index = int(args.cell_index)
    cells = list(input_manifest["cells"])
    if index < 0 or index >= len(cells):
        raise ContractError(f"cell index out of range: {index}")
    cell = cells[index]
    if int(cell["cell_index"]) != index:
        raise ContractError("cell index mapping changed")
    output = run_root / f"cells/{index:03d}_{cell['cell_id']}"
    if output.exists():
        raise FileExistsError(output)
    preparing = run_root / f".cell_{index:03d}.preparing.{os.getpid()}"
    failed = run_root / f".cell_{index:03d}.FAILED.{os.getpid()}"
    preparing.mkdir(parents=True, exist_ok=False)

    try:
        old_attempt_path = Path(cell["source_files"]["attempt_results"]["path"])
        old_summary_path = Path(cell["source_files"]["attempt_summary"]["path"])
        alignment_path = Path(cell["alignment_file"]["path"])
        if sha256_file(old_attempt_path) != cell["source_files"]["attempt_results"]["sha256"]:
            raise ContractError("old attempt ledger identity changed")
        if sha256_file(old_summary_path) != cell["source_files"]["attempt_summary"]["sha256"]:
            raise ContractError("old attempt summary identity changed")
        if sha256_file(alignment_path) != cell["alignment_file"]["sha256"]:
            raise ContractError("alignment identity changed")

        old_attempts = read_jsonl(old_attempt_path)
        old_summary = read_json(old_summary_path)
        alignment = read_jsonl(alignment_path)
        alignment_by_id = {str(row["attempt_id"]): row for row in alignment}
        if len(alignment_by_id) != len(alignment):
            raise ContractError("duplicate alignment attempt IDs")
        cache = load_clean_cache(cache_path)
        unresolved = load_unresolved(unresolved_path)
        required_chemsys = {str(row["chemsys"]) for row in alignment}
        overlap = sorted(set(cache) & set(unresolved))
        if overlap:
            raise ContractError("resolved and unresolved cache sets overlap")
        missing = sorted(required_chemsys - set(cache) - set(unresolved))
        if missing:
            raise ContractError(f"cache coverage misses {len(missing)} required chemsys")
        diagrams = {
            chemsys: phase_diagram(cache[chemsys], chemsys)
            for chemsys in sorted(required_chemsys & set(cache))
        }

        new_attempts: list[dict[str, Any]] = []
        hull_deltas: list[float] = []
        strict_01 = strict_10 = meta_01 = meta_10 = 0
        old_unknown = new_relax_unknown = new_hull_unknown = 0
        newly_hull_resolved = 0
        evaluated = 0
        unknown_skipped_from_pairing = 0
        for old in old_attempts:
            attempt_id = str(old["attempt_id"])
            new = copy.deepcopy(old)
            new["schema"] = "crysllmgen_r5c_a100_sun_attempt_official_mp_skip_unknown_v2"
            metrics = new["metrics"]
            old_metrics = old["metrics"]
            for key in ("novel", "unique_representative", "novel_unique"):
                if metrics.get(key) != old_metrics.get(key):
                    raise AssertionError("non-stability metric changed during copy")
            repair: dict[str, Any] = {
                "old_e_above_hull": old_metrics.get("e_above_hull"),
                "old_strict_full_sun": bool(old_metrics.get("strict_full_sun")),
                "old_meta_full_sun": bool(old_metrics.get("meta_full_sun")),
                "old_evaluation_status": str(old["evaluation_status"]),
                "thermo_type": "GGA_GGA+U",
                "query_method": "MPRester.get_entries_in_chemsys",
                "unresolved_policy": "explicit_hull_unknown_excluded_from_skip_unknown_denominators",
            }
            paired_known = True
            if bool(metrics.get("novel_unique")):
                aligned = alignment_by_id.pop(attempt_id, None)
                if aligned is None:
                    raise ContractError(f"missing aligned relaxed energy: {attempt_id}")
                energy = aligned.get("energy_per_atom")
                old_hull = aligned.get("old_e_above_hull")
                if old_hull is None:
                    old_unknown += 1
                if energy is None:
                    new_hull = None
                    new["evaluation_status"] = "relaxation_unknown"
                    new_relax_unknown += 1
                else:
                    composition = Composition(aligned["composition"])
                    chemsys = normalized_chemsys(composition)
                    if chemsys != aligned["chemsys"]:
                        raise ContractError("aligned chemsys changed")
                    if chemsys in unresolved:
                        new_hull = None
                        new["evaluation_status"] = "hull_unknown"
                        repair["hull_unknown_reason"] = unresolved[chemsys]["reason"]
                        new_hull_unknown += 1
                        paired_known = False
                    else:
                        new_hull = exact_hull(
                            diagrams[chemsys], composition, float(energy)
                        )
                        new["evaluation_status"] = "evaluated"
                        evaluated += 1
                        if old_hull is None:
                            newly_hull_resolved += 1
                        else:
                            hull_deltas.append(new_hull - float(old_hull))
                metrics["energy_per_atom"] = energy
                metrics["e_above_hull"] = new_hull
                metrics["strict_full_sun"] = (
                    new_hull is not None and new_hull <= STRICT_THRESHOLD
                )
                metrics["meta_full_sun"] = (
                    new_hull is not None and new_hull <= META_THRESHOLD
                )
            elif attempt_id in alignment_by_id:
                raise ContractError(f"alignment includes non-novel-unique attempt: {attempt_id}")
            else:
                if (
                    metrics.get("e_above_hull") is not None
                    or bool(metrics.get("strict_full_sun"))
                    or bool(metrics.get("meta_full_sun"))
                ):
                    raise ContractError(
                        f"non-novel-unique attempt carries a stability label: {attempt_id}"
                    )
            repair.update(
                {
                    "clean_e_above_hull": metrics.get("e_above_hull"),
                    "clean_strict_full_sun": bool(metrics.get("strict_full_sun")),
                    "clean_meta_full_sun": bool(metrics.get("meta_full_sun")),
                    "clean_evaluation_status": str(new["evaluation_status"]),
                }
            )
            old_strict = bool(old_metrics.get("strict_full_sun"))
            new_strict = bool(metrics.get("strict_full_sun"))
            old_meta = bool(old_metrics.get("meta_full_sun"))
            new_meta = bool(metrics.get("meta_full_sun"))
            if paired_known:
                strict_01 += int(not old_strict and new_strict)
                strict_10 += int(old_strict and not new_strict)
                meta_01 += int(not old_meta and new_meta)
                meta_10 += int(old_meta and not new_meta)
            else:
                unknown_skipped_from_pairing += 1
            new["stability_repair"] = repair
            new_attempts.append(new)
        if alignment_by_id:
            raise ContractError("unused aligned attempts remain")

        attempts = int(cell["expected_attempts"])
        reconstructed = int(cell["reconstructed"])
        strict_count = sum(
            int(bool(row["metrics"].get("strict_full_sun"))) for row in new_attempts
        )
        meta_count = sum(
            int(bool(row["metrics"].get("meta_full_sun"))) for row in new_attempts
        )
        novel_unique = sum(
            int(bool(row["metrics"].get("novel_unique"))) for row in new_attempts
        )
        if novel_unique != int(cell["novel_unique"]):
            raise ContractError("novel-unique count changed")
        if evaluated + new_relax_unknown + new_hull_unknown != novel_unique:
            raise ContractError("clean stability accounting is incomplete")
        if old_unknown != int(cell["old_hull_unknown"]):
            raise ContractError("old hull-unknown accounting changed")
        if meta_count < strict_count:
            raise ContractError("clean meta-S.U.N. is smaller than strict S.U.N.")
        if sum(
            bool((row.get("metrics") or {}).get("strict_full_sun"))
            for row in old_attempts
        ) != int(cell["old_strict_full_sun"]):
            raise ContractError("old strict S.U.N. accounting changed")
        if sum(
            bool((row.get("metrics") or {}).get("meta_full_sun"))
            for row in old_attempts
        ) != int(cell["old_meta_full_sun"]):
            raise ContractError("old meta-S.U.N. accounting changed")
        skip_all_denominator = attempts - new_hull_unknown
        skip_reconstructed_denominator = reconstructed - new_hull_unknown
        if skip_all_denominator <= 0 or skip_reconstructed_denominator <= 0:
            raise ContractError("skip-unknown denominator is not positive")
        report = {
            "schema": "h1_sun_official_gga_u_skip_unknown_cell_report_v2",
            "ok": True,
            "cell": {key: cell[key] for key in (
                "cell_index", "cell_id", "panel", "arm", "repeat", "stage"
            )},
            "denominators": {
                "all_attempts": attempts,
                "reconstructed_exact_legacy": reconstructed,
                "novel_unique": novel_unique,
                "hull_evaluated": evaluated,
                "relaxation_unknown": new_relax_unknown,
                "hull_unknown": new_hull_unknown,
                "all_attempts_skip_mp_unknown": skip_all_denominator,
                "reconstructed_skip_mp_unknown": skip_reconstructed_denominator,
            },
            "frozen_non_stability_counts": {
                "reconstructed": reconstructed,
                "novel": int(cell["novel"]),
                "unique": int(cell["unique"]),
                "novel_unique": novel_unique,
            },
            "old": {
                "strict_full_sun": int(cell["old_strict_full_sun"]),
                "meta_full_sun": int(cell["old_meta_full_sun"]),
                "hull_unknown": old_unknown,
            },
            "clean": {
                "strict_full_sun": strict_count,
                "meta_full_sun": meta_count,
                "hull_evaluated": evaluated,
                "relaxation_unknown": new_relax_unknown,
                "hull_unknown": new_hull_unknown,
                "strict_rate_all_attempts": strict_count / attempts,
                "meta_rate_all_attempts": meta_count / attempts,
                "strict_rate_reconstructed": strict_count / reconstructed,
                "meta_rate_reconstructed": meta_count / reconstructed,
                "strict_rate_all_attempts_skip_mp_unknown": strict_count / skip_all_denominator,
                "meta_rate_all_attempts_skip_mp_unknown": meta_count / skip_all_denominator,
                "strict_rate_reconstructed_skip_mp_unknown": strict_count / skip_reconstructed_denominator,
                "meta_rate_reconstructed_skip_mp_unknown": meta_count / skip_reconstructed_denominator,
                "strict_rate_hull_evaluated_novel_unique": strict_count / evaluated if evaluated else None,
                "meta_rate_hull_evaluated_novel_unique": meta_count / evaluated if evaluated else None,
            },
            "paired_old_to_clean": {
                "strict_0_to_1": strict_01,
                "strict_1_to_0": strict_10,
                "meta_0_to_1": meta_01,
                "meta_1_to_0": meta_10,
                "old_unknown": old_unknown,
                "old_unknown_now_hull_resolved": newly_hull_resolved,
                "mp_unknown_skipped_from_pairing": unknown_skipped_from_pairing,
                "e_hull_clean_minus_old_ev_per_atom": finite_summary(hull_deltas),
            },
            "contract": {
                "thermo_type": "GGA_GGA+U",
                "official_get_entries_in_chemsys": True,
                "fresh_cache": True,
                "local_compatibility_reprocessing": False,
                "generation_or_relaxation_rerun": False,
                "novelty_or_uniqueness_recompute": False,
                "new_mp_queries": 0,
                "mp_unresolved_count": int(cache_manifest["unresolved_query_count"]),
                "mp_unresolved_policy": "explicit_hull_unknown_excluded_from_skip_unknown_denominators",
            },
            "inputs": {
                "old_attempt_results": identity(old_attempt_path),
                "old_attempt_summary": identity(old_summary_path),
                "alignment": identity(alignment_path),
                "clean_cache_manifest": identity(cache_manifest_path),
                "clean_slim_cache": identity(cache_path),
                "unresolved_chemsys": identity(unresolved_path),
            },
            "old_summary_schema": old_summary.get("schema"),
            "new_attempts_sha256": canonical_sha256(new_attempts),
        }
        write_jsonl_exclusive(preparing / "attempt_results_clean.jsonl", new_attempts)
        write_json_exclusive(preparing / "cell_report.json", report)
        (preparing / "_SUCCESS").touch(exist_ok=False)
        output.parent.mkdir(parents=True, exist_ok=True)
        preparing.rename(output)
    except Exception:
        if preparing.exists():
            shutil.move(str(preparing), str(failed))
        raise
    print(
        {
            "cell": cell["cell_id"],
            "old_strict": cell["old_strict_full_sun"],
            "clean_strict": strict_count,
            "old_meta": cell["old_meta_full_sun"],
            "clean_meta": meta_count,
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
