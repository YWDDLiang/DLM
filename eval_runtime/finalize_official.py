#!/usr/bin/env python3
"""Join fresh official hull references and emit the complete comparison report."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Any, Mapping

from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.core import Composition

import protocol


def _rate(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _phase_diagrams(cache_path: Path) -> dict[str, PhaseDiagram]:
    result: dict[str, PhaseDiagram] = {}
    for row in protocol.read_jsonl(cache_path):
        chemsys = str(row["chemsys"])
        entries = [
            PDEntry(Composition(entry["composition"]), float(entry["energy"]), name=str(entry.get("entry_id") or ""))
            for entry in row["entries"]
        ]
        result[chemsys] = PhaseDiagram(entries)
    return result


def _e_above_hull(pd: PhaseDiagram, composition: Mapping[str, Any], energy_per_atom: float) -> float:
    comp = Composition(composition)
    return float(energy_per_atom) - float(pd.get_hull_energy_per_atom(comp))


def _evaluate_cell(
    *,
    cell_id: str,
    labels_path: Path,
    generation_path: Path,
    direct_path: Path,
    phase_diagrams: Mapping[str, PhaseDiagram],
    unresolved: set[str],
    output_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    labels = sorted(protocol.read_jsonl(labels_path), key=lambda row: int(row["ordinal"]))
    generation = sorted(protocol.read_jsonl(generation_path), key=lambda row: int(row["ordinal"]))
    if len(labels) != len(generation) or [int(row["ordinal"]) for row in labels] != list(range(len(labels))):
        raise protocol.ContractError(f"cell alignment changed: {cell_id}")
    generation_by_id = {str(row["attempt_id"]): row for row in generation}
    if len(generation_by_id) != len(generation):
        raise protocol.ContractError(f"duplicate generation ID: {cell_id}")
    evaluated: list[dict[str, Any]] = []
    for row in labels:
        attempt_id = str(row["attempt_id"])
        if attempt_id not in generation_by_id:
            raise protocol.ContractError(f"missing generation row: {cell_id}:{attempt_id}")
        reconstructed = row.get("reconstructed") is True
        chemsys = None if row.get("chemsys") is None else str(row["chemsys"])
        energy = row.get("chgnet_energy_per_atom")
        e_hull: float | None = None
        hull_status = "not_reconstructed"
        if reconstructed and energy is None:
            hull_status = "chgnet_energy_unknown"
        elif reconstructed and chemsys in unresolved:
            hull_status = "official_hull_unknown"
        elif reconstructed:
            pd = phase_diagrams.get(str(chemsys))
            if pd is None:
                raise protocol.ContractError(f"official cache omitted {chemsys}")
            e_hull = _e_above_hull(pd, row["chgnet_composition"], float(energy))
            if not math.isfinite(e_hull):
                raise protocol.ContractError("nonfinite official e_above_hull")
            hull_status = "known"
        novel = bool(row.get("novel"))
        unique = bool(row.get("unique_representative"))
        novel_unique = novel and unique
        strict_stable = e_hull is not None and e_hull <= 0.0
        meta_stable = e_hull is not None and e_hull <= 0.1
        evaluated.append(
            {
                **row,
                "schema": "h1a2_epoch2_exactplan_official_attempt_v1",
                "official_hull_status": hull_status,
                "official_e_above_hull": e_hull,
                "strict_stable": strict_stable,
                "meta_stable": meta_stable,
                "strict_sun": strict_stable and novel_unique,
                "meta_sun": meta_stable and novel_unique,
            }
        )
    counts = {
        "raw_attempts": len(evaluated),
        "generation_succeeded": sum(row.get("status") == "succeeded" for row in generation),
        "reconstructed": sum(row["reconstructed"] for row in evaluated),
        "novel": sum(row["novel"] for row in evaluated),
        "unique_representatives": sum(row["unique_representative"] for row in evaluated),
        "novel_unique": sum(row["novel_unique"] for row in evaluated),
        "chgnet_energy_known": sum(row["chgnet_relaxation_known"] for row in evaluated),
        "hull_known_reconstructed": sum(row["official_hull_status"] == "known" for row in evaluated),
        "hull_unknown_reconstructed": sum(row["reconstructed"] and row["official_hull_status"] != "known" for row in evaluated),
        "hull_known_novel_unique": sum(row["official_hull_status"] == "known" and row["novel_unique"] for row in evaluated),
        "strict_stable_all_hull_known": sum(row["strict_stable"] for row in evaluated),
        "meta_stable_all_hull_known": sum(row["meta_stable"] for row in evaluated),
        "strict_sun": sum(row["strict_sun"] for row in evaluated),
        "meta_sun": sum(row["meta_sun"] for row in evaluated),
    }
    direct = protocol.read_json(direct_path)
    if int(direct["attempts"]) != len(evaluated):
        raise protocol.ContractError(f"Direct denominator changed: {cell_id}")
    report = {
        "schema": "h1a2_epoch2_exactplan_official_cell_report_v1",
        "cell_id": cell_id,
        "counts": counts,
        "direct": {
            "composition_valid": int(direct["comp_valid_count"]),
            "structure_valid": int(direct["struct_valid_count"]),
            "joint_valid": int(direct["valid_count"]),
            "metrics": direct["metrics_unchanged_upstream"],
        },
        "rates": {
            "strict_sun_all_attempts": _rate(counts["strict_sun"], counts["raw_attempts"]),
            "meta_sun_all_attempts": _rate(counts["meta_sun"], counts["raw_attempts"]),
            "strict_sun_hull_known_reconstructed": _rate(counts["strict_sun"], counts["hull_known_reconstructed"]),
            "meta_sun_hull_known_reconstructed": _rate(counts["meta_sun"], counts["hull_known_reconstructed"]),
            "strict_among_hull_known_novel_unique": _rate(counts["strict_sun"], counts["hull_known_novel_unique"]),
            "meta_among_hull_known_novel_unique": _rate(counts["meta_sun"], counts["hull_known_novel_unique"]),
        },
        "target": {
            "name": "hull_known_reconstructed",
            "value": 1000,
            "observed": counts["hull_known_reconstructed"],
            "met": counts["hull_known_reconstructed"] >= 1000,
            "target_is_nonblocking": True,
        },
        "unknown_policy": "excluded from hull-known denominators; never mapped to unstable",
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    protocol.write_jsonl_exclusive(output_dir / "attempt_results_official.jsonl", evaluated)
    protocol.write_json_exclusive(output_dir / "report.json", report)
    (output_dir / "_SUCCESS").touch(exist_ok=False)
    return evaluated, report


def _exact_mcnemar(control: list[bool], candidate: list[bool]) -> dict[str, Any]:
    if len(control) != len(candidate):
        raise protocol.ContractError("paired endpoint lengths differ")
    control_only = sum(left and not right for left, right in zip(control, candidate, strict=True))
    candidate_only = sum(right and not left for left, right in zip(control, candidate, strict=True))
    discordant = control_only + candidate_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, value) for value in range(min(control_only, candidate_only) + 1)) / (2 ** discordant)
        p_value = min(1.0, 2.0 * tail)
    return {
        "control_only": control_only,
        "candidate_only": candidate_only,
        "discordant": discordant,
        "two_sided_exact_p": p_value,
    }


def _survivor_prefix(
    attempts_path: Path,
    official_rows: list[dict[str, Any]],
    prefix: int = 1000,
) -> dict[str, Any]:
    attempts = sorted(protocol.read_jsonl(attempts_path), key=lambda row: int(row["ordinal"]))
    selected = [int(row["ordinal"]) for row in attempts if row.get("status") == "succeeded"][:prefix]
    if len(selected) != prefix:
        raise protocol.ContractError("fewer than 1000 body successes for survivor-prefix report")
    rows = [official_rows[index] for index in selected]
    return {
        "selection": "first_1000_body_status_succeeded_by_raw_ordinal_read_only",
        "denominator": prefix,
        "first_raw_ordinal": selected[0],
        "last_raw_ordinal": selected[-1],
        "selected_ordinals_sha256": protocol.canonical_sha256(selected),
        "reconstructed": sum(row["reconstructed"] for row in rows),
        "novel_unique": sum(row["novel_unique"] for row in rows),
        "hull_known": sum(row["official_hull_status"] == "known" for row in rows),
        "hull_unknown": sum(row["reconstructed"] and row["official_hull_status"] != "known" for row in rows),
        "strict_sun": sum(row["strict_sun"] for row in rows),
        "meta_sun": sum(row["meta_sun"] for row in rows),
    }


def _old_summary(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"Full S\.U\.N\. lower-bound:\s*([0-9.]+)%", text)
    if match is None:
        raise protocol.ContractError(f"cannot parse legacy S.U.N. summary: {path}")
    percent = float(match.group(1))

    def fraction(label: str) -> tuple[int, int] | None:
        found = re.search(rf"^{re.escape(label)}:\s*(\d+)\s*/\s*(\d+)", text, re.MULTILINE)
        return None if found is None else (int(found.group(1)), int(found.group(2)))

    return {
        "summary": protocol.identity(path),
        "full_sun_lower_bound_pct": percent,
        "full_sun_numerator_over_1000": int(round(percent * 10)),
        "novel_unique": fraction("Novel + Unique"),
        "e_hull_evaluated": fraction("E_hull evaluated"),
        "e_hull_unknown": fraction("E_hull unknown"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_root.resolve()
    config = protocol.read_json(args.config.resolve())
    protocol.validate_config(config)
    cache = run / "official_mp_cache"
    phase_diagrams = _phase_diagrams(cache / "official_slim_cache.jsonl")
    unresolved = {str(row["chemsys"]) for row in protocol.read_jsonl(cache / "unresolved_chemsys.jsonl")}
    output = run / "official_results"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    specs = {
        "historical": {
            "labels": run / "historical/evaluation/full_reconstructed/attempt_labels_preofficial.jsonl",
            "generation": run / "historical/generation/generation.jsonl",
            "direct": run / "historical/evaluation/direct/report.json",
        },
        "control": {
            "labels": run / "new/arms/control/evaluation/full_reconstructed/attempt_labels_preofficial.jsonl",
            "generation": run / "new/arms/control/generation/generation.jsonl",
            "direct": run / "new/arms/control/evaluation/direct/report.json",
        },
        "candidate": {
            "labels": run / "new/arms/candidate/evaluation/full_reconstructed/attempt_labels_preofficial.jsonl",
            "generation": run / "new/arms/candidate/generation/generation.jsonl",
            "direct": run / "new/arms/candidate/evaluation/direct/report.json",
        },
    }
    rows: dict[str, list[dict[str, Any]]] = {}
    reports: dict[str, dict[str, Any]] = {}
    for cell_id, spec in specs.items():
        rows[cell_id], reports[cell_id] = _evaluate_cell(
            cell_id=cell_id,
            labels_path=spec["labels"],
            generation_path=spec["generation"],
            direct_path=spec["direct"],
            phase_diagrams=phase_diagrams,
            unresolved=unresolved,
            output_dir=output / "cells" / cell_id,
        )
    prefix = {
        arm: _survivor_prefix(run / f"new/body/{arm}_body_attempts.jsonl", rows[arm])
        for arm in ("control", "candidate")
    }
    known_both = [
        index for index in range(protocol.PLANNER_ATTEMPTS)
        if rows["control"][index]["official_hull_status"] == "known"
        and rows["candidate"][index]["official_hull_status"] == "known"
    ]
    paired = {
        "pairs_total_raw": protocol.PLANNER_ATTEMPTS,
        "pairs_hull_known_both": len(known_both),
        "pairs_excluded": protocol.PLANNER_ATTEMPTS - len(known_both),
        "strict_sun": _exact_mcnemar(
            [bool(rows["control"][index]["strict_sun"]) for index in known_both],
            [bool(rows["candidate"][index]["strict_sun"]) for index in known_both],
        ),
        "meta_sun": _exact_mcnemar(
            [bool(rows["control"][index]["meta_sun"]) for index in known_both],
            [bool(rows["candidate"][index]["meta_sun"]) for index in known_both],
        ),
    }
    old_protocol = {
        "strict": _old_summary(run / "historical/old_protocol/strict_summary.md"),
        "meta": _old_summary(run / "historical/old_protocol/meta_summary.md"),
        "expected_frozen_record": {
            "strict_sun": int(config["historical_frozen"]["old_strict_sun"]),
            "meta_sun": int(config["historical_frozen"]["old_meta_sun"]),
        },
    }
    cohort = protocol.read_json(run / "new/cohort/cohort_report.json")
    plan_drift = protocol.read_json(run / "new/cohort/plan_distribution_and_drift.json")
    body = protocol.read_json(run / "new/body/generation_report.json")
    refiners = {
        arm: protocol.read_json(run / f"new/arms/{arm}/refinement/refinement_metrics.json")
        for arm in ("control", "candidate")
    }
    terminal = {
        "schema": "h1a2_epoch2_exactplan1200_h1a2_r03_fullsun_terminal_v1",
        "status": "complete",
        "ok": True,
        "scientific_contract": {
            "planner": "early epoch2 world2 batch4 seed17 legacy-rank exact archived sampler",
            "arms": {"control": "H1-A2 D1", "candidate": "R03 D2 safe-axis"},
            "body_selection": "all body successes",
            "refiner": "model494 exact800 batch1 with paired per-raw-ordinal seeds",
            "stability": "fresh official MP GGA_GGA+U over every reconstructed row",
            "retry_replacement_repair_filter_rerank_training_rl": False,
        },
        "historical_old_protocol_recomputation": old_protocol,
        "cells": reports,
        "historical_style_survivor_prefix1000": prefix,
        "paired_exact_mcnemar": paired,
        "planner_identity": cohort,
        "planner_distribution_and_drift": plan_drift,
        "body_generation": body,
        "refiners": refiners,
        "official_cache": protocol.read_json(cache / "completion_manifest.json"),
    }
    protocol.write_json_exclusive(output / "terminal_report.json", terminal)

    def pct(value: float | None) -> str:
        return "NA" if value is None else f"{100 * value:.3f}%"

    lines = [
        "# Early H1-A2 9.4% reproduction and paired H1-A2 vs R03 full S.U.N.",
        "",
        "## Historical frozen artifact — recomputed legacy protocol",
        "",
        f"- strict S.U.N.: {old_protocol['strict']['full_sun_numerator_over_1000']}/1000 ({old_protocol['strict']['full_sun_lower_bound_pct']:.2f}%)",
        f"- meta-S.U.N.: {old_protocol['meta']['full_sun_numerator_over_1000']}/1000 ({old_protocol['meta']['full_sun_lower_bound_pct']:.2f}%)",
        "",
        "## Full official results",
        "",
        "| Cell | Raw | Parser | Body | Refined | Reconstructed | Direct comp | Direct struct | Direct joint | N | U | N∩U | Hull known | Hull unknown | Strict S.U.N. / hull-known | Meta-S.U.N. / hull-known | Strict / all attempts | Meta / all attempts | Hull target |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    parser_count = int(cohort["parsed"])
    for cell_id in ("historical", "control", "candidate"):
        report = reports[cell_id]
        count = report["counts"]
        direct = report["direct"]
        if cell_id == "historical":
            raw, parsed, body_count, refined = 1200, int(config["historical_frozen"]["planner_parsed"]), int(config["historical_frozen"]["body_succeeded"]), 1000
        else:
            raw, parsed = 1200, parser_count
            body_count = int(body["arms"][cell_id]["succeeded"])
            refined = int(refiners[cell_id]["refiner_complete"])
        lines.append(
            f"| {cell_id} | {raw} | {parsed} | {body_count} | {refined} | {count['reconstructed']} | "
            f"{direct['composition_valid']} | {direct['structure_valid']} | {direct['joint_valid']} | "
            f"{count['novel']} | {count['unique_representatives']} | {count['novel_unique']} | "
            f"{count['hull_known_reconstructed']} | {count['hull_unknown_reconstructed']} | "
            f"{count['strict_sun']}/{count['hull_known_reconstructed']} ({pct(report['rates']['strict_sun_hull_known_reconstructed'])}) | "
            f"{count['meta_sun']}/{count['hull_known_reconstructed']} ({pct(report['rates']['meta_sun_hull_known_reconstructed'])}) | "
            f"{count['strict_sun']}/{count['raw_attempts']} ({pct(report['rates']['strict_sun_all_attempts'])}) | "
            f"{count['meta_sun']}/{count['raw_attempts']} ({pct(report['rates']['meta_sun_all_attempts'])}) | "
            f"{'met' if report['target']['met'] else 'not met'} |"
        )
    lines.extend([
        "",
        "## Historical-style survivor-prefix1000 (read-only view)",
        "",
        "| Arm | Reconstructed | N∩U | Hull known | Hull unknown | Strict S.U.N. | Meta-S.U.N. |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for arm in ("control", "candidate"):
        row = prefix[arm]
        lines.append(f"| {arm} | {row['reconstructed']} | {row['novel_unique']} | {row['hull_known']} | {row['hull_unknown']} | {row['strict_sun']}/1000 ({row['strict_sun']/10:.2f}%) | {row['meta_sun']}/1000 ({row['meta_sun']/10:.2f}%) |")
    lines.extend([
        "",
        "## Exact paired McNemar (same raw ordinal; official hull known in both arms)",
        "",
        "| Endpoint | H1-A2 only | R03 only | Discordant | Two-sided exact p |",
        "|---|---:|---:|---:|---:|",
        f"| strict S.U.N. | {paired['strict_sun']['control_only']} | {paired['strict_sun']['candidate_only']} | {paired['strict_sun']['discordant']} | {paired['strict_sun']['two_sided_exact_p']:.8g} |",
        f"| meta-S.U.N. | {paired['meta_sun']['control_only']} | {paired['meta_sun']['candidate_only']} | {paired['meta_sun']['discordant']} | {paired['meta_sun']['two_sided_exact_p']:.8g} |",
        "",
        "## Plan identity and drift",
        "",
        f"- raw planner bytes identical to the early run: `{cohort['raw_byte_identical']}`",
        f"- parsed-plan ledger bytes identical to the early run: `{cohort['plans_byte_identical']}`",
        f"- parsed plans: `{cohort['parsed']}/1200`",
        f"- maximum categorical total-variation drift: `{plan_drift['max_total_variation']:.8g}`",
        "",
        "Hull-unknown rows are excluded from hull-known denominators and are never silently classified as unstable.",
        "",
    ])
    (output / "RESULTS_COMPLETE.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    (output / "_SUCCESS").touch(exist_ok=False)
    (run / "terminal_report.json").hardlink_to(output / "terminal_report.json")
    (run / "RESULTS_COMPLETE.md").hardlink_to(output / "RESULTS_COMPLETE.md")
    (run / "_SUCCESS").touch(exist_ok=False)
    print(protocol.canonical_json({"status": "complete", "strict": {key: reports[key]["counts"]["strict_sun"] for key in reports}, "meta": {key: reports[key]["counts"]["meta_sun"] for key in reports}}), flush=True)


if __name__ == "__main__":
    main()
