#!/usr/bin/env python3
"""Diagnose R5-C vs current H1/free-geometry S.U.N. gap with e_hull labels."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path


ROOT = Path("/mnt/d/codex_work/ai4s/diffsion_language_model_meets_diffusion")
EHULL_ROOT = ROOT / "reports/20260604_ehull_label_export_remote/20260604_ehull_label_export/outputs/ehull_labels"
OUT_JSON = ROOT / "reports/20260604_r5c_vs_h1a3_freegeo_sun_gap_statistical_diagnosis.json"
OUT_MD = ROOT / "reports/20260604_r5c_vs_h1a3_freegeo_sun_gap_diagnosis.md"


BRANCHES = {
    "r5c_conditional": {
        "display": "R5-C conditional",
        "raw": ROOT / "runs/20260531_0040-r5c-full1000-sun/outputs/r5c_sample1000/raw_generations.jsonl",
        "sample_metrics": ROOT / "runs/20260531_0040-r5c-full1000-sun/outputs/r5c_sample1000/sample_metrics.json",
        "composition": ROOT / "reports/20260531_r5c_full1000_sun/notes/composition1000.json",
        "crys": ROOT / "runs/20260531_0040-r5c-full1000-sun/notes/crysllmgen_metrics1000.json",
        "a100_summary": ROOT / "runs/20260531_2200-a100-eval-sun-mpapi-cache-final2/notes/a100_eval_sun_comparison_summary.json",
        "sun_summary_keys": ("dlm_strict", "dlm_meta_like"),
    },
    "h1a3_epoch1": {
        "display": "H1-A3 epoch1",
        "planner": ROOT / "reports/20260604_h1a3_epoch1_2_artifacts/runs/20260603_h1a3_joint_planner/outputs/h1a3_epoch1_planner256/raw_generations.jsonl",
        "raw": ROOT / "reports/20260604_h1a3_epoch1_2_artifacts/runs/20260603_h1a3_joint_planner/outputs/h1a3_epoch1_hybrid_body256/raw_generations.jsonl",
        "sample_metrics": ROOT / "reports/20260604_h1a3_epoch1_2_artifacts/runs/20260603_h1a3_joint_planner/outputs/h1a3_epoch1_hybrid_body256/sample_metrics.json",
        "planner_gate": ROOT / "reports/20260604_h1a3_epoch1_2_artifacts/runs/20260603_h1a3_joint_planner/notes/h1a3_epoch1_planner_gate.json",
        "composition": ROOT / "reports/20260604_h1a3_epoch1_2_artifacts/runs/20260603_h1a3_joint_planner/notes/h1a3_epoch1_composition.json",
        "crys": ROOT / "reports/20260604_h1a3_epoch1_2_artifacts/runs/20260603_h1a3_joint_planner/notes/h1a3_epoch1_crysllmgen_metrics.json",
        "a100_summary": ROOT / "reports/20260604_h1a3_epoch1_2_artifacts/runs/20260603_h1a3_epoch1_refined256_a100_sun/notes/a100_eval_sun_dlm_only_summary.json",
        "sun_summary_keys": ("dlm_strict", "dlm_meta_like"),
    },
    "h1a3_epoch2": {
        "display": "H1-A3 epoch2",
        "planner": ROOT / "reports/20260604_h1a3_epoch1_2_artifacts/runs/20260603_h1a3_joint_planner/outputs/h1a3_epoch2_planner256/raw_generations.jsonl",
        "raw": ROOT / "reports/20260604_h1a3_epoch1_2_artifacts/runs/20260603_h1a3_joint_planner/outputs/h1a3_epoch2_hybrid_body256/raw_generations.jsonl",
        "sample_metrics": ROOT / "reports/20260604_h1a3_epoch1_2_artifacts/runs/20260603_h1a3_joint_planner/outputs/h1a3_epoch2_hybrid_body256/sample_metrics.json",
        "planner_gate": ROOT / "reports/20260604_h1a3_epoch1_2_artifacts/runs/20260603_h1a3_joint_planner/notes/h1a3_epoch2_planner_gate.json",
        "composition": ROOT / "reports/20260604_h1a3_epoch1_2_artifacts/runs/20260603_h1a3_joint_planner/notes/h1a3_epoch2_composition.json",
        "crys": ROOT / "reports/20260604_h1a3_epoch1_2_artifacts/runs/20260603_h1a3_joint_planner/notes/h1a3_epoch2_crysllmgen_metrics.json",
        "a100_summary": ROOT / "reports/20260604_h1a3_epoch1_2_artifacts/runs/20260604_h1a3_epoch2_refined256_a100_sun/notes/a100_eval_sun_dlm_only_summary.json",
        "sun_summary_keys": ("dlm_strict", "dlm_meta_like"),
    },
    "freegeo_ablation_default": {
        "display": "free-geometry default",
        "raw": ROOT / "reports/20260604_freegeo_default_full1000_artifacts/runs/20260603_h1_freegeo_remaining7_full1000_sun/outputs/ablation_default_sample1200/raw_generations.jsonl",
        "sample_metrics": ROOT / "reports/20260604_freegeo_default_full1000_artifacts/runs/20260603_h1_freegeo_remaining7_full1000_sun/outputs/ablation_default_sample1200/sample_metrics.json",
        "composition": ROOT / "reports/20260604_freegeo_default_full1000_artifacts/runs/20260603_h1_freegeo_remaining7_full1000_sun/notes/ablation_default_composition1000.json",
        "crys": ROOT / "reports/20260604_freegeo_default_full1000_artifacts/runs/20260603_h1_freegeo_remaining7_full1000_sun/notes/ablation_default_crysllmgen_metrics1000.json",
        "a100_summary": ROOT / "reports/20260604_freegeo_default_full1000_artifacts/runs/20260603_h1_freegeo_ablation_default_full1000_a100_retry/notes/a100_eval_sun_dlm_only_summary.json",
        "sun_summary_keys": ("dlm_strict", "dlm_meta_like"),
    },
}


FREEGEO_EXTRA = [
    "freegeo_ablation_default_schedule",
    "freegeo_ablation_no_lattice_volume_mask",
    "freegeo_ablation_no_duplicate_coordinate_mask",
    "freegeo_ablation_no_freeze_plan_composition",
    "freegeo_weighted_lattice_up",
    "freegeo_weighted_coord_up",
    "freegeo_weighted_balanced",
]


TOKEN_RE = re.compile(r"<([^>]+)>")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                out.append(json.loads(line))
    return out


def pct(n: float, d: float) -> float:
    return 100.0 * n / d if d else 0.0


def dist(counter: Counter) -> dict[str, float]:
    total = sum(counter.values())
    return {str(k): v / total for k, v in counter.items()} if total else {}


def tvd(a: dict, b: dict) -> float:
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys)


def js_div(a: dict, b: dict) -> float:
    keys = set(a) | set(b)
    m = {k: 0.5 * (a.get(k, 0.0) + b.get(k, 0.0)) for k in keys}

    def kl(p, q):
        total = 0.0
        for k in keys:
            if p.get(k, 0.0) > 0 and q.get(k, 0.0) > 0:
                total += p[k] * math.log(p[k] / q[k], 2)
        return total

    return 0.5 * kl(a, m) + 0.5 * kl(b, m)


def top_delta(a_counter: Counter, b_counter: Counter, limit: int = 12) -> list[dict]:
    a = dist(a_counter)
    b = dist(b_counter)
    rows = []
    for key in set(a) | set(b):
        rows.append(
            {
                "value": key,
                "current_pct": 100 * b.get(key, 0.0),
                "r5c_pct": 100 * a.get(key, 0.0),
                "delta_pp": 100 * (b.get(key, 0.0) - a.get(key, 0.0)),
            }
        )
    return sorted(rows, key=lambda row: abs(row["delta_pp"]), reverse=True)[:limit]


def as_metrics(payload: dict) -> dict:
    if "metrics" in payload:
        return payload["metrics"]
    return payload


def extract_sun_summary(branch_key: str, config: dict) -> dict:
    payload = load_json(config["a100_summary"])
    if branch_key == "r5c_conditional":
        strict = payload["dlm_strict"]
        meta = payload["dlm_meta_like"]
    else:
        strict = payload["dlm_strict"]
        meta = payload["dlm_meta_like"]
    return {
        "strict_adjusted_pct": strict.get("coverage-adjusted_sun_estimate_pct"),
        "strict_lower_bound_pct": strict.get("full_sun_lower-bound_pct"),
        "meta_adjusted_pct": meta.get("coverage-adjusted_sun_estimate_pct"),
        "meta_lower_bound_pct": meta.get("full_sun_lower-bound_pct"),
        "novel_unique_pct": strict.get("novel_+_unique_pct"),
        "novel_unique_num": strict.get("novel_+_unique_num"),
        "novel_unique_den": strict.get("novel_+_unique_den"),
        "e_hull_evaluated_num": strict.get("e_hull_evaluated_num"),
        "e_hull_evaluated_den": strict.get("e_hull_evaluated_den"),
        "unknown_num": strict.get("e_hull_unknown_num"),
        "unknown_den": strict.get("e_hull_unknown_den"),
    }


def parse_body_tokens(text: str) -> dict:
    tokens = TOKEN_RE.findall(text or "")
    data = {"tokens": tokens, "N": None, "lengths": [], "angles": [], "coords": []}
    for tok in tokens:
        if "_" not in tok:
            continue
        prefix, value = tok.split("_", 1)
        if prefix == "N":
            try:
                data["N"] = int(value)
            except ValueError:
                pass
        elif prefix in {"LA", "LB", "LC"}:
            data["lengths"].append(value)
        elif prefix in {"AA", "AB", "AG"}:
            data["angles"].append(value)
        elif prefix in {"X", "Y", "Z"}:
            data["coords"].append(value)
    return data


def raw_token_stats(path: Path) -> dict:
    rows = load_jsonl(path)
    length_counter = Counter()
    angle_counter = Counter()
    coord_counter = Counter()
    angle_triples = Counter()
    n_counter = Counter()
    all_angles_90 = 0
    all_angles_60 = 0
    two_lengths_equal = 0
    all_lengths_equal = 0
    quarter_coord = 0
    coord_total = 0
    examples = []
    for row in rows:
        text = row.get("text") or row.get("raw_model_text") or row.get("raw_plan_text") or ""
        parsed = parse_body_tokens(text)
        if parsed["N"] is not None:
            n_counter[str(parsed["N"])] += 1
        for val in parsed["lengths"]:
            length_counter[val] += 1
        for val in parsed["angles"]:
            angle_counter[val] += 1
        for val in parsed["coords"]:
            coord_counter[val] += 1
            coord_total += 1
            if val in {"000", "025", "050", "075"}:
                quarter_coord += 1
        if len(parsed["angles"]) >= 3:
            triple = tuple(parsed["angles"][:3])
            angle_triples["/".join(triple)] += 1
            if triple == ("090", "090", "090"):
                all_angles_90 += 1
            if triple == ("060", "060", "060"):
                all_angles_60 += 1
        if len(parsed["lengths"]) >= 3:
            a, b, c = parsed["lengths"][:3]
            if a == b == c:
                all_lengths_equal += 1
            if len({a, b, c}) <= 2:
                two_lengths_equal += 1
        if len(examples) < 3:
            plan = row.get("plan_state") or {}
            examples.append(
                {
                    "sample_idx": row.get("sample_idx"),
                    "text": text[:260],
                    "formula": plan.get("formula"),
                    "N": plan.get("N"),
                    "charge": plan.get("charge_bucket"),
                    "anion": plan.get("anion_framework"),
                    "lattice": plan.get("lattice_system"),
                    "spacegroup": plan.get("spacegroup_bucket"),
                    "volume": plan.get("volume_per_atom_bin"),
                }
            )
    n = len(rows)
    return {
        "count": n,
        "N_hist": dict(n_counter),
        "length_top20": length_counter.most_common(20),
        "angle_top20": angle_counter.most_common(20),
        "coord_top20": coord_counter.most_common(20),
        "angle_triple_top20": angle_triples.most_common(20),
        "all_angles_90_pct": pct(all_angles_90, n),
        "all_angles_60_pct": pct(all_angles_60, n),
        "two_lengths_equal_pct": pct(two_lengths_equal, n),
        "all_lengths_equal_pct": pct(all_lengths_equal, n),
        "quarter_coord_pct": pct(quarter_coord, coord_total),
        "coord_total": coord_total,
        "examples": examples,
    }


def plan_stats(path: Path) -> dict:
    rows = load_jsonl(path)
    counters = {k: Counter() for k in ["N", "arity", "anion", "charge", "lattice", "spacegroup", "volume"]}
    examples = []
    for row in rows:
        ps = row.get("parsed_plan") or row.get("plan_state") or {}
        if not ps:
            continue
        counters["N"][str(ps.get("N"))] += 1
        counters["arity"][str(len(ps.get("elements") or []))] += 1
        counters["anion"][ps.get("anion_framework")] += 1
        counters["charge"][ps.get("charge_bucket")] += 1
        counters["lattice"][ps.get("lattice_system")] += 1
        counters["spacegroup"][ps.get("spacegroup_bucket")] += 1
        counters["volume"][ps.get("volume_per_atom_bin")] += 1
        if len(examples) < 4:
            examples.append(row.get("raw_plan_text") or row.get("plan_text") or row.get("raw_model_text"))
    return {
        "count": len(rows),
        "hist": {k: dict(v) for k, v in counters.items()},
        "top": {k: v.most_common(20) for k, v in counters.items()},
        "examples": examples,
    }


def composition_stats(path: Path) -> dict:
    payload = load_json(path)
    raw = payload.get("raw_jsonl") or {}
    return {
        "count": raw.get("count"),
        "comp_valid_rate": raw.get("comp_valid_rate"),
        "pbc_equivalent_duplicate_fraction": raw.get("pbc_equivalent_duplicate_fraction"),
        "N_hist": raw.get("num_atoms_histogram", {}),
        "arity_hist": raw.get("num_elements_histogram", {}),
        "reason_counts": raw.get("reason_counts", {}),
        "top_formula": raw.get("formula_top30", {}),
    }


def load_ehull(branch: str) -> tuple[dict, list[dict]]:
    summary = load_json(EHULL_ROOT / branch / "ehull_summary.json")
    labels = load_jsonl(EHULL_ROOT / branch / "ehull_labels.jsonl")
    return summary, labels


def label_counters(labels: list[dict]) -> dict[str, Counter]:
    return {
        "N": Counter(str(r.get("N")) for r in labels),
        "arity": Counter(str(r.get("arity")) for r in labels),
        "elements": Counter(el for r in labels for el in r.get("elements", [])),
        "chemsys": Counter(r.get("chemsys") for r in labels),
        "flags": Counter(
            key
            for r in labels
            for key in [
                "oxygen" if r.get("contains_oxygen") else None,
                "chalcogen" if r.get("contains_chalcogen") else None,
                "halogen" if r.get("contains_halogen") else None,
                "common_anion" if r.get("contains_common_anion") else None,
                "all_metal_proxy" if r.get("all_metal_proxy") else None,
            ]
            if key
        ),
    }


def ehull_bins(labels: list[dict]) -> dict:
    bins = Counter()
    for row in labels:
        hull = row.get("e_hull")
        if hull is None:
            bins["unknown"] += 1
        elif hull <= 0:
            bins["<=0"] += 1
        elif hull <= 0.05:
            bins["0-0.05"] += 1
        elif hull <= 0.1:
            bins["0.05-0.1"] += 1
        elif hull <= 0.2:
            bins["0.1-0.2"] += 1
        else:
            bins[">0.2"] += 1
    return {k: {"count": v, "pct": pct(v, len(labels))} for k, v in bins.items()}


def group_rates(labels: list[dict], key_fn) -> list[dict]:
    groups = {}
    for row in labels:
        key = key_fn(row)
        groups.setdefault(key, []).append(row)
    out = []
    for key, rows in groups.items():
        evaluated = [r for r in rows if r.get("e_hull") is not None]
        out.append(
            {
                "group": str(key),
                "n": len(rows),
                "evaluated": len(evaluated),
                "strict_rate_eval_pct": pct(sum(bool(r.get("strict_stable")) for r in evaluated), len(evaluated)),
                "meta_rate_eval_pct": pct(sum(bool(r.get("meta_stable")) for r in evaluated), len(evaluated)),
                "mean_e_hull": sum(r["e_hull"] for r in evaluated) / len(evaluated) if evaluated else None,
            }
        )
    return sorted(out, key=lambda r: (-r["n"], r["group"]))


def build_analysis() -> dict:
    analysis = {
        "created": "2026-06-04",
        "ehull_export": str(EHULL_ROOT),
        "branches": {},
        "comparisons_vs_r5c": {},
        "freegeo_all": {},
    }
    r5c_labels = []
    r5c_counters = {}
    for key, cfg in BRANCHES.items():
        eh_summary, labels = load_ehull(key)
        if key == "r5c_conditional":
            r5c_labels = labels
            r5c_counters = label_counters(labels)
        branch = {
            "display": cfg["display"],
            "sun": extract_sun_summary(key, cfg),
            "ehull": {
                "summary": eh_summary,
                "bins": ehull_bins(labels),
                "rates_by_arity": group_rates(labels, lambda r: r.get("arity")),
                "rates_by_N_top": group_rates(labels, lambda r: r.get("N"))[:30],
                "rates_by_all_metal": group_rates(labels, lambda r: "all_metal_proxy" if r.get("all_metal_proxy") else "has_common_anion"),
                "rates_by_oxygen": group_rates(labels, lambda r: "contains_O" if r.get("contains_oxygen") else "no_O"),
            },
            "sample_metrics": load_json(cfg["sample_metrics"]),
            "composition": composition_stats(cfg["composition"]),
            "crysllmgen": as_metrics(load_json(cfg["crys"])),
            "raw_token": raw_token_stats(cfg["raw"]),
        }
        if cfg.get("planner"):
            branch["planner"] = plan_stats(cfg["planner"])
        analysis["branches"][key] = branch

    for key, branch in analysis["branches"].items():
        if key == "r5c_conditional":
            continue
        labels = load_jsonl(EHULL_ROOT / key / "ehull_labels.jsonl")
        counters = label_counters(labels)
        analysis["comparisons_vs_r5c"][key] = {
            "N_tvd": tvd(dist(r5c_counters["N"]), dist(counters["N"])),
            "N_js": js_div(dist(r5c_counters["N"]), dist(counters["N"])),
            "arity_tvd": tvd(dist(r5c_counters["arity"]), dist(counters["arity"])),
            "arity_js": js_div(dist(r5c_counters["arity"]), dist(counters["arity"])),
            "element_tvd": tvd(dist(r5c_counters["elements"]), dist(counters["elements"])),
            "flag_tvd": tvd(dist(r5c_counters["flags"]), dist(counters["flags"])),
            "top_N_delta": top_delta(r5c_counters["N"], counters["N"]),
            "top_arity_delta": top_delta(r5c_counters["arity"], counters["arity"]),
            "top_element_delta": top_delta(r5c_counters["elements"], counters["elements"]),
            "top_flag_delta": top_delta(r5c_counters["flags"], counters["flags"]),
        }

    for key in FREEGEO_EXTRA:
        summary, labels = load_ehull(key)
        analysis["freegeo_all"][key] = {
            "ehull": summary,
            "bins": ehull_bins(labels),
        }
    return analysis


def fmt(x, digits=2):
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def table(rows: list[list]) -> list[str]:
    if not rows:
        return []
    widths = [max(len(str(row[i])) for row in rows) for i in range(len(rows[0]))]
    out = []
    for idx, row in enumerate(rows):
        out.append("| " + " | ".join(str(row[i]).ljust(widths[i]) for i in range(len(row))) + " |")
        if idx == 0:
            out.append("| " + " | ".join("-" * widths[i] for i in range(len(row))) + " |")
    return out


def write_report(analysis: dict) -> None:
    b = analysis["branches"]
    lines = [
        "# R5-C vs H1-A3 / Free-Geometry S.U.N. Gap Diagnosis With e_hull Labels",
        "",
        "Date: 2026-06-04",
        "",
        "This report uses the A800 e_hull label export from existing A100 `relax_results.jsonl` files. It does not re-run CHGNet relaxation, does not re-run baseline S.U.N., and does not use the MP API. The per-sample labels are for the A100 novel+unique subset, so they explain the stability part of S.U.N.; raw proposal statistics are reported separately.",
        "",
        "## Executive Conclusion",
        "",
        "- The dominant gap is hull stability, especially meta-like stability. R5-C conditional has `722/927 = 77.89%` meta-stable among evaluated novel+unique samples; free-geometry default has `488/861 = 56.68%`; H1-A3 epoch1/2 have only `51.56% / 50.92%`.",
        "- Strict stability is closer but still lower: R5-C has `103/927 = 11.11%`, free-geometry default has `93/861 = 10.80%`, H1-A3 epoch1/2 have `7.56% / 9.17%`. This explains why free-geometry default can reach baseline-like strict while still being far below R5-C meta S.U.N.",
        "- Novel+Unique is not the main bottleneck for R5-C vs current H1. R5-C is `95.50%`, free-geometry default is `88.90%`, H1-A3 epoch1 is `92.71%`; the large meta gap remains after conditioning on evaluated novel+unique samples.",
        "- H1-A3 epoch2 shifts to larger N after extra planner training. Its A100 novel+unique subset has mean `N=12.04` vs R5-C `9.39`, and its refined density WDist worsens. Extra planner epochs are not monotonic improvement.",
        "- The DLM executor obeys hard constraints almost perfectly, but raw body tokens remain coarse-grid/lattice-template proposals. CrysLLMGen refinement repairs structural validity and coverage, yet cannot recover R5-C's hull-stable basin when the generated plan/body pair is off-distribution.",
        "",
        "## A100 e_hull Results",
    ]
    rows = [["Branch", "Novel+Unique", "Evaluated", "Strict stable/eval", "Meta stable/eval", "Strict adj", "Meta adj", "mean N"]]
    for key in ["r5c_conditional", "h1a3_epoch1", "h1a3_epoch2", "freegeo_ablation_default"]:
        branch = b[key]
        eh = branch["ehull"]["summary"]
        sun = branch["sun"]
        rows.append([
            branch["display"],
            f"{sun['novel_unique_num']}/{sun['novel_unique_den']} ({fmt(sun['novel_unique_pct'])}%)",
            f"{eh['evaluated']}/{eh['total_relax_records']}",
            f"{eh['strict_stable']}/{eh['evaluated']} ({fmt(eh['strict_stable_per_evaluated_pct'])}%)",
            f"{eh['meta_stable']}/{eh['evaluated']} ({fmt(eh['meta_stable_per_evaluated_pct'])}%)",
            f"{fmt(sun['strict_adjusted_pct'])}%",
            f"{fmt(sun['meta_adjusted_pct'])}%",
            fmt(eh["mean_N"]),
        ])
    lines.extend(table(rows))
    lines.extend(["", "The R5-C meta-like advantage is mostly the `77.89%` meta-stable/evaluated rate. The current branches do not lack e_hull cache coverage; unknown hull counts are similar (`28` for R5-C and freegeo default).", ""])

    lines.append("## e_hull Distribution")
    rows = [["Branch", "<=0", "0-0.05", "0.05-0.1", "0.1-0.2", ">0.2", "unknown"]]
    for key in ["r5c_conditional", "h1a3_epoch1", "h1a3_epoch2", "freegeo_ablation_default"]:
        bins = b[key]["ehull"]["bins"]
        rows.append([
            b[key]["display"],
            f"{bins.get('<=0', {}).get('count', 0)} ({fmt(bins.get('<=0', {}).get('pct', 0))}%)",
            f"{bins.get('0-0.05', {}).get('count', 0)} ({fmt(bins.get('0-0.05', {}).get('pct', 0))}%)",
            f"{bins.get('0.05-0.1', {}).get('count', 0)} ({fmt(bins.get('0.05-0.1', {}).get('pct', 0))}%)",
            f"{bins.get('0.1-0.2', {}).get('count', 0)} ({fmt(bins.get('0.1-0.2', {}).get('pct', 0))}%)",
            f"{bins.get('>0.2', {}).get('count', 0)} ({fmt(bins.get('>0.2', {}).get('pct', 0))}%)",
            f"{bins.get('unknown', {}).get('count', 0)} ({fmt(bins.get('unknown', {}).get('pct', 0))}%)",
        ])
    lines.extend(table(rows))
    lines.extend(["", "R5-C has a much larger low-hull shoulder: many more samples sit in `0-0.1 eV/atom`, which is exactly the meta-like S.U.N. threshold. The current branches produce many valid structures that land above `0.1 eV/atom` after relaxation.", ""])

    lines.append("## CrysLLMGen Metrics vs e_hull")
    rows = [["Branch", "comp_valid", "struct_valid", "cov_recall", "density WDist", "elem WDist", "meta stable/eval"]]
    for key in ["r5c_conditional", "h1a3_epoch1", "h1a3_epoch2", "freegeo_ablation_default"]:
        crys = b[key]["crysllmgen"]
        eh = b[key]["ehull"]["summary"]
        rows.append([
            b[key]["display"],
            fmt(crys.get("comp_valid")),
            fmt(crys.get("struct_valid")),
            fmt(crys.get("cov_recall")),
            fmt(crys.get("wdist_density"), 4),
            fmt(crys.get("wdist_num_elems"), 4),
            f"{fmt(eh['meta_stable_per_evaluated_pct'])}%",
        ])
    lines.extend(table(rows))
    lines.extend(["", "Crys metrics are necessary but not sufficient. Free-geometry default has `cov_recall=95.58%`, nearly R5-C level, but meta-stable/evaluated is only `56.68%`. Coverage says the structures occupy MP-like neighborhoods; e_hull says many are not in stable basins.", ""])

    lines.append("## Composition Distribution")
    rows = [["Branch", "raw comp_valid", "all-metal reason", "charge valid", "charge fail", "N TVD vs R5-C", "arity TVD", "element TVD", "flag TVD"]]
    for key in ["r5c_conditional", "h1a3_epoch1", "h1a3_epoch2", "freegeo_ablation_default"]:
        comp = b[key]["composition"]
        reasons = comp.get("reason_counts", {})
        if key == "r5c_conditional":
            compvals = ["-", "-", "-", "-"]
        else:
            c = analysis["comparisons_vs_r5c"][key]
            compvals = [fmt(c["N_tvd"], 3), fmt(c["arity_tvd"], 3), fmt(c["element_tvd"], 3), fmt(c["flag_tvd"], 3)]
        rows.append([
            b[key]["display"],
            fmt(100 * (comp.get("comp_valid_rate") or 0)),
            reasons.get("all_metal_shortcut", 0),
            reasons.get("charge_neutral_pauling_valid", 0),
            reasons.get("charge_neutrality_fail", 0),
            *compvals,
        ])
    lines.extend(table(rows))
    lines.extend(["", "Composition alone does not explain the whole gap. Free-geometry default is composition-close to R5-C at the novel+unique label level (`N TVD` and `arity TVD` are modest), yet meta stability is 21 pp lower. H1-A3 epoch2, however, clearly over-shifts N upward, which hurts both density and stability.", ""])

    lines.append("## Stable Composition Signals")
    for key in ["r5c_conditional", "freegeo_ablation_default", "h1a3_epoch1", "h1a3_epoch2"]:
        rates = b[key]["ehull"]["rates_by_all_metal"]
        lines.append(f"### {b[key]['display']}")
        rows = [["Group", "n", "evaluated", "strict/eval", "meta/eval", "mean e_hull"]]
        for row in rates:
            rows.append([row["group"], row["n"], row["evaluated"], f"{fmt(row['strict_rate_eval_pct'])}%", f"{fmt(row['meta_rate_eval_pct'])}%", fmt(row["mean_e_hull"], 4)])
        lines.extend(table(rows))
        lines.append("")
    lines.append("The gap is not simply all-metal vs anion. Both groups degrade in current branches. This points to higher-order plan/geometry compatibility: formula plus generated rich fields plus body geometry do not land in the same refinement basins as R5-C teacher-conditioned samples.")
    lines.append("")

    lines.append("## Raw DLM Token Behavior")
    rows = [["Branch", "raw count", "all 90 angles", "all 60 angles", "two lengths equal", "quarter-grid coords", "top angle triple"]]
    for key in ["r5c_conditional", "h1a3_epoch1", "h1a3_epoch2", "freegeo_ablation_default"]:
        raw = b[key]["raw_token"]
        top_angle = raw["angle_triple_top20"][0][0] if raw["angle_triple_top20"] else "-"
        rows.append([
            b[key]["display"],
            raw["count"],
            f"{fmt(raw['all_angles_90_pct'])}%",
            f"{fmt(raw['all_angles_60_pct'])}%",
            f"{fmt(raw['two_lengths_equal_pct'])}%",
            f"{fmt(raw['quarter_coord_pct'])}%",
            top_angle,
        ])
    lines.extend(table(rows))
    lines.extend(["", "The exact-DLM geometry language is still very coarse: coordinates are dominated by quarter-grid values and angles concentrate at common buckets. R5-C survives this better because the teacher plan is paired with real-structure metadata; generated H1 plans expose the same executor to less compatible condition tuples.", ""])

    lines.append("## H1-A3 Planner Output")
    for key in ["h1a3_epoch1", "h1a3_epoch2"]:
        planner = b[key].get("planner", {})
        gate = load_json(BRANCHES[key]["planner_gate"])
        gate_metrics = gate.get("metrics", {})
        lines.append(f"### {b[key]['display']}")
        lines.append(f"- Planner count: `{planner.get('count')}`")
        lines.append(
            "- Gate summary: "
            f"mean_N `{fmt(gate_metrics.get('mean_N'), 3)}`, "
            f"N TVD `{fmt(gate_metrics.get('n_tvd'), 3)}`, "
            f"arity TVD `{fmt(gate_metrics.get('arity_tvd'), 3)}`, "
            f"charge TVD `{fmt(gate_metrics.get('charge_bucket_tvd'), 3)}`, "
            f"volume TVD `{fmt(gate_metrics.get('volume_per_atom_bin_tvd'), 3)}`"
        )
        lines.append(f"- Top charge fields: `{planner.get('top', {}).get('charge', [])[:6]}`")
        lines.append(f"- Top volume fields: `{planner.get('top', {}).get('volume', [])[:6]}`")
        if planner.get("examples"):
            lines.append("Example raw plans:")
            for ex in planner["examples"][:3]:
                lines.append("```text")
                lines.append(str(ex))
                lines.append("```")
        lines.append("")

    lines.append("## Free-Geometry 8-Branch e_hull Summary")
    rows = [["Branch", "records", "eval", "strict/eval", "meta/eval", "mean N", "mean e_hull"]]
    for key in ["freegeo_ablation_default", *FREEGEO_EXTRA]:
        eh = (b[key]["ehull"]["summary"] if key in b else analysis["freegeo_all"][key]["ehull"])
        rows.append([
            key,
            eh["total_relax_records"],
            eh["evaluated"],
            f"{eh['strict_stable']}/{eh['evaluated']} ({fmt(eh['strict_stable_per_evaluated_pct'])}%)",
            f"{eh['meta_stable']}/{eh['evaluated']} ({fmt(eh['meta_stable_per_evaluated_pct'])}%)",
            fmt(eh["mean_N"]),
            fmt(eh["mean_e_hull"], 4),
        ])
    lines.extend(table(rows))
    lines.extend(["", "The no-duplicate-mask branch has high strict/meta rates only because the evaluated set is tiny and biased (`117` novel+unique records from a graph-valid survivor subset). It should not be promoted.", ""])

    lines.append("## Root Cause")
    lines.extend([
        "",
        "1. **R5-C conditional uses teacher plan tuples that are geometry-realizable.** Its plan is not merely a formula prior; it carries a coupled tuple `(formula, charge/anion, lattice, SG bucket, volume)` that came from a real MP-20 structure. The DLM/refiner pair sees a condition that is inside its learned basin.",
        "2. **H1-A3 generated rich plans are locally valid but not equally basin-compatible.** Epoch1 matches marginal distributions better, but strict remains low. Epoch2 increases N and large-cell rate, which worsens density WDist and meta stability. The missing object is joint compatibility, not string syntax.",
        "3. **The exact-DLM executor follows constraints but its geometry proposal is a coarse sketch.** Body parse and plan match are essentially perfect, but raw geometry tokens still overuse bucketed angles and quarter-grid coordinates. This sketch is adequate under teacher plans, but brittle under generated plans.",
        "4. **Diffusion refinement repairs validity, not composition-geometry thermodynamics.** High struct_valid and cov_recall do not imply low e_hull. The e_hull labels show the current branches mostly fail by landing above the meta threshold after relaxation.",
    ])
    lines.extend(["", "## Next Adjustments", ""])
    lines.extend([
        "- Stop treating CrysLLMGen metrics as promotion gates by themselves. A branch should be promoted only if e_hull stable/evaluated moves, especially meta-like stable/evaluated.",
        "- For planner training, optimize/model-select on joint tuple diagnostics, not marginal TVD only: `(N_bin, arity, anion, charge, lattice, SG, volume)` plus conditional tables such as `volume|N`, `SG|lattice`, `charge|anion+formula_family`.",
        "- Avoid simply adding epochs to H1-A3. Epoch2 proves more SFT can shift N upward and reduce meta stability.",
        "- Keep default exact-DLM constraints for now. Removing masks creates biased survivor subsets or violates plan execution. Weighted geometry CE did not improve e_hull enough.",
        "- If continuing special-token DLM, diagnose teacher-vs-generated raw token distributions within the same plan bins. If coarse grid/angle templates remain stable, move to H2 plain-text DLM proposal or a less quantized geometry proposal language.",
        "- Add e_hull label export as a standard post-S.U.N. artifact for every future branch. The useful diagnostic unit is not only S.U.N. scalar but which compositions/condition tuples are stable or meta-stable.",
    ])
    lines.extend(["", "## Artifacts", ""])
    lines.extend([
        f"- e_hull export: `{EHULL_ROOT}`",
        f"- diagnosis JSON: `{OUT_JSON}`",
        f"- cleanup manifest: `{ROOT / 'reports/20260604_h1a3_checkpoint_cleanup_manifest.md'}`",
        "- H1-A3 epoch1/2 checkpoints were removed remotely; logs, raw outputs, refined `.pt`, and metrics were preserved.",
    ])
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    analysis = build_analysis()
    OUT_JSON.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(analysis)
    print(OUT_JSON)
    print(OUT_MD)


if __name__ == "__main__":
    main()
