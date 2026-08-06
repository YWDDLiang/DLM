#!/usr/bin/env python3
"""Analyze next-round R5-C/H1/H2 S.U.N. gap from plan, DLM, and diffusion views."""

from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path("/mnt/d/codex_work/ai4s/diffsion_language_model_meets_diffusion")
OUT_JSON = ROOT / "reports/20260604_r5c_vs_next_round_plan_dlm_sun_comparison.json"
OUT_MD = ROOT / "reports/20260604_r5c_vs_next_round_plan_dlm_sun_comparison.md"

EHULL_CANDIDATES = [
    Path(os.environ["EHULL_ROOT"]) if os.environ.get("EHULL_ROOT") else None,
    ROOT / "reports/20260604_next_round_ehull_label_export_remote/20260604_next_round_ehull_label_export/outputs/ehull_labels",
    ROOT / "reports/20260604_next_round_remote_artifacts/runs/20260604_next_round_ehull_label_export/outputs/ehull_labels",
    ROOT / "reports/20260604_ehull_label_export_remote/20260604_ehull_label_export/outputs/ehull_labels",
]
EHULL = next((p for p in EHULL_CANDIDATES if p and p.exists()), EHULL_CANDIDATES[-1])

TOKEN_RE = re.compile(r"<([^>]+)>")


BRANCHES = {
    "r5c_conditional": {
        "display": "R5-C conditional",
        "kind": "teacher-plan upper reference",
        "raw": ROOT / "runs/20260531_0040-r5c-full1000-sun/outputs/r5c_sample1000/raw_generations.jsonl",
        "sample_metrics": ROOT / "runs/20260531_0040-r5c-full1000-sun/outputs/r5c_sample1000/sample_metrics.json",
        "crys": ROOT / "runs/20260531_0040-r5c-full1000-sun/notes/crysllmgen_metrics1000.json",
        "sun": {"strict_adjusted": 10.61, "meta_adjusted": 74.38},
    },
    "h1a3_epoch1": {
        "display": "H1-A3 epoch1",
        "kind": "old rich planner + exact-DLM",
        "raw": ROOT / "reports/20260604_h1a3_after_epoch3_cancel_artifacts/runs/20260603_h1a3_joint_planner/outputs/h1a3_epoch1_hybrid_body256/raw_generations.jsonl",
        "planner": ROOT / "reports/20260604_h1a3_after_epoch3_cancel_artifacts/runs/20260603_h1a3_joint_planner/outputs/h1a3_epoch1_planner256/raw_generations.jsonl",
        "sample_metrics": ROOT / "reports/20260604_h1a3_after_epoch3_cancel_artifacts/runs/20260603_h1a3_joint_planner/outputs/h1a3_epoch1_hybrid_body256/sample_metrics.json",
        "crys": ROOT / "reports/20260604_h1a3_after_epoch3_cancel_artifacts/runs/20260603_h1a3_joint_planner/notes/h1a3_epoch1_crysllmgen_metrics.json",
        "a100": ROOT / "reports/20260604_h1a3_after_epoch3_cancel_artifacts/runs/20260603_h1a3_epoch1_refined256_a100_sun/notes/a100_eval_sun_dlm_only_summary.json",
    },
    "h1a3_epoch2": {
        "display": "H1-A3 epoch2",
        "kind": "old rich planner + exact-DLM",
        "raw": ROOT / "reports/20260604_h1a3_after_epoch3_cancel_artifacts/runs/20260603_h1a3_joint_planner/outputs/h1a3_epoch2_hybrid_body256/raw_generations.jsonl",
        "planner": ROOT / "reports/20260604_h1a3_after_epoch3_cancel_artifacts/runs/20260603_h1a3_joint_planner/outputs/h1a3_epoch2_planner256/raw_generations.jsonl",
        "sample_metrics": ROOT / "reports/20260604_h1a3_after_epoch3_cancel_artifacts/runs/20260603_h1a3_joint_planner/outputs/h1a3_epoch2_hybrid_body256/sample_metrics.json",
        "crys": ROOT / "reports/20260604_h1a3_after_epoch3_cancel_artifacts/runs/20260603_h1a3_joint_planner/notes/h1a3_epoch2_crysllmgen_metrics.json",
        "a100": ROOT / "reports/20260604_h1a3_after_epoch3_cancel_artifacts/runs/20260604_h1a3_epoch2_refined256_a100_sun/notes/a100_eval_sun_dlm_only_summary.json",
    },
    "freegeo_ablation_default": {
        "display": "freegeo default",
        "kind": "constraint ablation reference",
        "raw": ROOT / "reports/20260604_freegeo_default_full1000_artifacts/runs/20260603_h1_freegeo_remaining7_full1000_sun/outputs/ablation_default_sample1200/raw_generations.jsonl",
        "sample_metrics": ROOT / "reports/20260604_freegeo_default_full1000_artifacts/runs/20260603_h1_freegeo_remaining7_full1000_sun/outputs/ablation_default_sample1200/sample_metrics.json",
        "crys": ROOT / "reports/20260604_freegeo_default_full1000_artifacts/runs/20260603_h1_freegeo_remaining7_full1000_sun/notes/ablation_default_crysllmgen_metrics1000.json",
        "a100": ROOT / "reports/20260604_freegeo_default_full1000_artifacts/runs/20260603_h1_freegeo_ablation_default_full1000_a100_retry/notes/a100_eval_sun_dlm_only_summary.json",
    },
    "h1g1_full_rich": {
        "display": "H1-G1 full-rich",
        "kind": "robust exact-DLM",
        "raw": ROOT / "reports/20260604_next_round_remote_artifacts/runs/20260604_h1g1_robust_exact_dlm_resume/outputs/h1g1_full_rich_sample1200/raw_generations.jsonl",
        "sample_metrics": ROOT / "reports/20260604_next_round_remote_artifacts/runs/20260604_h1g1_robust_exact_dlm_resume/outputs/h1g1_full_rich_sample1200/sample_metrics.json",
        "crys": ROOT / "reports/20260604_next_round_remote_artifacts/runs/20260604_h1g1_robust_exact_dlm_resume/notes/h1g1_full_rich_crysllmgen_metrics.json",
        "a100": ROOT / "reports/20260604_next_round_remote_artifacts/runs/20260604_h1g1_robust_exact_dlm_resume/notes/h1g1_full_rich_a100_sun_summary.json",
    },
    "h1g1_condition_dropout": {
        "display": "H1-G1 condition-dropout",
        "kind": "robust exact-DLM",
        "raw": ROOT / "reports/20260604_next_round_remote_artifacts/runs/20260604_h1g1_robust_exact_dlm_resume/outputs/h1g1_condition_dropout_sample1200/raw_generations.jsonl",
        "sample_metrics": ROOT / "reports/20260604_next_round_remote_artifacts/runs/20260604_h1g1_robust_exact_dlm_resume/outputs/h1g1_condition_dropout_sample1200/sample_metrics.json",
        "crys": ROOT / "reports/20260604_next_round_remote_artifacts/runs/20260604_h1g1_robust_exact_dlm_resume/notes/h1g1_condition_dropout_crysllmgen_metrics.json",
        "a100": ROOT / "reports/20260604_next_round_remote_artifacts/runs/20260604_h1g1_robust_exact_dlm_resume/notes/h1g1_condition_dropout_a100_sun_summary.json",
    },
    "h1g1_formula_volume_sg": {
        "display": "H1-G1 formula-volume-sg",
        "kind": "robust exact-DLM",
        "raw": ROOT / "reports/20260604_next_round_remote_artifacts/runs/20260604_h1g1_robust_exact_dlm_resume/outputs/h1g1_formula_volume_sg_sample1200/raw_generations.jsonl",
        "sample_metrics": ROOT / "reports/20260604_next_round_remote_artifacts/runs/20260604_h1g1_robust_exact_dlm_resume/outputs/h1g1_formula_volume_sg_sample1200/sample_metrics.json",
        "crys": ROOT / "reports/20260604_next_round_remote_artifacts/runs/20260604_h1g1_robust_exact_dlm_resume/notes/h1g1_formula_volume_sg_crysllmgen_metrics.json",
        "a100": ROOT / "reports/20260604_next_round_remote_artifacts/runs/20260604_h1g1_robust_exact_dlm_resume/notes/h1g1_formula_volume_sg_a100_sun_summary.json",
    },
    "h1g1_formula_volume_only": {
        "display": "H1-G1 formula-volume-only",
        "kind": "robust exact-DLM",
        "raw": ROOT / "reports/20260604_next_round_remote_artifacts/runs/20260604_h1g1_robust_exact_dlm_resume/outputs/h1g1_formula_volume_only_sample1200/raw_generations.jsonl",
        "sample_metrics": ROOT / "reports/20260604_next_round_remote_artifacts/runs/20260604_h1g1_robust_exact_dlm_resume/outputs/h1g1_formula_volume_only_sample1200/sample_metrics.json",
        "crys": ROOT / "reports/20260604_next_round_remote_artifacts/runs/20260604_h1g1_robust_exact_dlm_resume/notes/h1g1_formula_volume_only_crysllmgen_metrics.json",
        "a100": ROOT / "reports/20260604_next_round_remote_artifacts/runs/20260604_h1g1_robust_exact_dlm_resume/notes/h1g1_formula_volume_only_a100_sun_summary.json",
    },
    "h2p1_plaintext": {
        "display": "H2-P1 plain-text",
        "kind": "plain-text DLM proposal",
        "raw": ROOT / "reports/20260604_next_round_remote_artifacts/runs/20260604_h2p1_plaintext_dlm_proposal_clean/outputs/h2p1_plaintext_dlm_sample1200/raw_generations.jsonl",
        "planner": ROOT / "reports/20260604_next_round_remote_artifacts/runs/20260604_h2p1_plaintext_dlm_proposal_clean/outputs/h2p1_planner1200/raw_generations.jsonl",
        "sample_metrics": ROOT / "reports/20260604_next_round_remote_artifacts/runs/20260604_h2p1_plaintext_dlm_proposal_clean/outputs/h2p1_plaintext_dlm_sample1200/sample_metrics.json",
        "proposal_gate": ROOT / "reports/20260604_next_round_remote_artifacts/runs/20260604_h2p1_plaintext_dlm_proposal_clean/notes/h2p1_plaintext_gate.json",
        "crys": ROOT / "reports/20260604_next_round_remote_artifacts/runs/20260604_h2p1_plaintext_dlm_proposal_clean/notes/h2p1_crysllmgen_metrics.json",
        "a100": ROOT / "reports/20260604_next_round_remote_artifacts/runs/20260604_h2p1_plaintext_dlm_proposal_clean/notes/h2p1_a100_sun_summary.json",
    },
    "h1a4_epoch1": {
        "display": "H1-A4 epoch1",
        "kind": "joint-basin planner + old exact-DLM",
        "raw": ROOT / "reports/20260604_next_round_remote_artifacts/runs/20260604_h1a4_joint_basin_planner_clean/outputs/h1a4_epoch1_hybrid_body1200/raw_generations.jsonl",
        "planner": ROOT / "reports/20260604_next_round_remote_artifacts/runs/20260604_h1a4_joint_basin_planner_clean/outputs/h1a4_epoch1_planner1200/raw_generations.jsonl",
        "sample_metrics": ROOT / "reports/20260604_next_round_remote_artifacts/runs/20260604_h1a4_joint_basin_planner_clean/outputs/h1a4_epoch1_hybrid_body1200/sample_metrics.json",
        "planner_gate": ROOT / "reports/20260604_next_round_remote_artifacts/runs/20260604_h1a4_joint_basin_planner_clean/notes/h1a4_epoch1_planner_gate.json",
        "hybrid_gate": ROOT / "reports/20260604_next_round_remote_artifacts/runs/20260604_h1a4_joint_basin_planner_clean/notes/h1a4_epoch1_hybrid_gate.json",
        "crys": ROOT / "reports/20260604_next_round_remote_artifacts/runs/20260604_h1a4_joint_basin_planner_clean/notes/h1a4_epoch1_crysllmgen_metrics.json",
        "a100": ROOT / "reports/20260604_next_round_remote_artifacts/runs/20260604_h1a4_joint_basin_planner_clean/notes/h1a4_epoch1_a100_sun_summary.json",
    },
    "h1a4_epoch2": {
        "display": "H1-A4 epoch2",
        "kind": "joint-basin planner + old exact-DLM",
        "raw": ROOT / "reports/20260604_next_round_remote_artifacts/runs/20260604_h1a4_joint_basin_planner_clean/outputs/h1a4_epoch2_hybrid_body1200/raw_generations.jsonl",
        "planner": ROOT / "reports/20260604_next_round_remote_artifacts/runs/20260604_h1a4_joint_basin_planner_clean/outputs/h1a4_epoch2_planner1200/raw_generations.jsonl",
        "sample_metrics": ROOT / "reports/20260604_next_round_remote_artifacts/runs/20260604_h1a4_joint_basin_planner_clean/outputs/h1a4_epoch2_hybrid_body1200/sample_metrics.json",
        "planner_gate": ROOT / "reports/20260604_next_round_remote_artifacts/runs/20260604_h1a4_joint_basin_planner_clean/notes/h1a4_epoch2_planner_gate.json",
        "hybrid_gate": ROOT / "reports/20260604_next_round_remote_artifacts/runs/20260604_h1a4_joint_basin_planner_clean/notes/h1a4_epoch2_hybrid_gate.json",
        "crys": ROOT / "reports/20260604_next_round_remote_artifacts/runs/20260604_h1a4_joint_basin_planner_clean/notes/h1a4_epoch2_crysllmgen_metrics.json",
        "a100": ROOT / "reports/20260604_next_round_remote_artifacts/runs/20260604_h1a4_joint_basin_planner_clean/notes/h1a4_epoch2_a100_sun_summary.json",
    },
}


def load_json(path: Path | None) -> dict:
    if not path or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path | None) -> list[dict]:
    if not path or not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def pct(n: float, d: float) -> float:
    return 100.0 * n / d if d else 0.0


def fmt(value, digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def table(rows: list[list]) -> list[str]:
    widths = [max(len(str(row[i])) for row in rows) for i in range(len(rows[0]))]
    out = []
    for idx, row in enumerate(rows):
        out.append("| " + " | ".join(str(row[i]).ljust(widths[i]) for i in range(len(row))) + " |")
        if idx == 0:
            out.append("| " + " | ".join("-" * widths[i] for i in range(len(row))) + " |")
    return out


def as_metrics(payload: dict) -> dict:
    return payload.get("metrics", payload)


def sun_from_config(cfg: dict) -> dict:
    if "sun" in cfg:
        return cfg["sun"]
    payload = load_json(cfg.get("a100"))
    strict = payload.get("dlm_strict", {})
    meta = payload.get("dlm_meta_like", {})
    return {
        "strict_adjusted": strict.get("coverage-adjusted_sun_estimate_pct"),
        "strict_lower_bound": strict.get("full_sun_lower-bound_pct"),
        "meta_adjusted": meta.get("coverage-adjusted_sun_estimate_pct"),
        "meta_lower_bound": meta.get("full_sun_lower-bound_pct"),
        "novel_unique": meta.get("novel_+_unique_pct") or strict.get("novel_+_unique_pct"),
        "eval_num": meta.get("e_hull_evaluated_num") or strict.get("e_hull_evaluated_num"),
        "eval_den": meta.get("e_hull_evaluated_den") or strict.get("e_hull_evaluated_den"),
    }


def composition_key_from_plan(plan: dict) -> tuple[tuple[str, int], ...]:
    return tuple(
        sorted(
            (str(el), int(round(float(count))))
            for el, count in zip(plan.get("elements") or [], plan.get("counts") or [])
        )
    )


def composition_key_from_dict(comp: dict) -> tuple[tuple[str, int], ...]:
    return tuple(sorted((str(el), int(round(float(count)))) for el, count in comp.items()))


def parse_tokens(text: str) -> dict:
    out = {"lengths": [], "angles": [], "coords": []}
    for token in TOKEN_RE.findall(text or ""):
        if "_" not in token:
            continue
        prefix, value = token.split("_", 1)
        if prefix in {"LA", "LB", "LC"}:
            out["lengths"].append(value)
        elif prefix in {"AA", "AB", "AG"}:
            out["angles"].append(value)
        elif prefix in {"X", "Y", "Z"}:
            out["coords"].append(value)
    return out


def raw_token_stats(path: Path | None) -> dict:
    rows = load_jsonl(path)
    angle_templates = {("090", "090", "090"), ("060", "060", "060"), ("090", "090", "120"), ("090", "120", "120"), ("060", "060", "120")}
    angle_triples = Counter()
    n = 0
    all90 = all60 = template = two_lengths = all_lengths = 0
    coord_total = coord_quarter = coord_zero = 0
    examples = []
    for row in rows:
        text = row.get("text") or row.get("raw_model_text") or row.get("completion") or ""
        parsed = parse_tokens(text)
        if len(parsed["angles"]) >= 3:
            triple = tuple(parsed["angles"][:3])
            angle_triples["/".join(triple)] += 1
            template += int(triple in angle_templates)
            all90 += int(triple == ("090", "090", "090"))
            all60 += int(triple == ("060", "060", "060"))
        if len(parsed["lengths"]) >= 3:
            vals = parsed["lengths"][:3]
            all_lengths += int(len(set(vals)) == 1)
            two_lengths += int(len(set(vals)) <= 2)
        for coord in parsed["coords"]:
            coord_total += 1
            coord_quarter += int(coord in {"000", "025", "050", "075"})
            coord_zero += int(coord == "000")
        if len(examples) < 3:
            plan = row.get("plan_state") or row.get("parsed_plan") or {}
            examples.append(
                {
                    "sample_idx": row.get("sample_idx"),
                    "formula": plan.get("formula"),
                    "N": plan.get("N"),
                    "anion": plan.get("anion_framework"),
                    "charge": plan.get("charge_bucket"),
                    "spacegroup": plan.get("spacegroup_bucket"),
                    "volume": plan.get("volume_per_atom_bin"),
                    "text_head": text[:220],
                }
            )
        n += 1
    return {
        "count": n,
        "angle_template_pct": pct(template, n),
        "all90_pct": pct(all90, n),
        "all60_pct": pct(all60, n),
        "two_lengths_equal_pct": pct(two_lengths, n),
        "all_lengths_equal_pct": pct(all_lengths, n),
        "quarter_coord_pct": pct(coord_quarter, coord_total),
        "zero_coord_pct": pct(coord_zero, coord_total),
        "top_angle_triples": angle_triples.most_common(8),
        "examples": examples,
    }


def family(row: dict) -> str:
    elements = set(row.get("elements") or [])
    halides = {"F", "Cl", "Br", "I"}
    pnictides = {"N", "P", "As", "Sb", "Bi"}
    nonoxide_chalc = {"S", "Se", "Te"}
    if not row.get("contains_common_anion") and not (elements & pnictides) and "H" not in elements:
        return "intermetallic/all-metal"
    parts = []
    if "O" in elements:
        parts.append("oxide")
    if elements & nonoxide_chalc:
        parts.append("non-oxide chalcogenide")
    if elements & halides:
        parts.append("halide")
    if elements & pnictides:
        parts.append("pnictide")
    if "H" in elements:
        parts.append("hydride")
    if not parts:
        return "other-anion"
    if len(parts) == 1:
        return parts[0]
    return "mixed-anion/" + "+".join(parts[:3])


def summarize_group(rows: list[dict]) -> dict:
    evaluated = [row for row in rows if row.get("e_hull") is not None]
    strict = [row for row in evaluated if row.get("strict_stable")]
    meta = [row for row in evaluated if row.get("meta_stable")]
    return {
        "n": len(rows),
        "evaluated": len(evaluated),
        "strict": len(strict),
        "meta": len(meta),
        "strict_eval_pct": pct(len(strict), len(evaluated)),
        "meta_eval_pct": pct(len(meta), len(evaluated)),
        "mean_e_hull": sum(row["e_hull"] for row in evaluated) / len(evaluated) if evaluated else None,
    }


def group_by(rows: list[dict], key_fn) -> dict[str, dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(key_fn(row))].append(row)
    return {key: summarize_group(value) for key, value in groups.items()}


def raw_index(path: Path | None) -> dict:
    by_key: dict[tuple, list[dict]] = defaultdict(list)
    for row in load_jsonl(path):
        plan = row.get("plan_state") or row.get("parsed_plan") or {}
        key = composition_key_from_plan(plan)
        if not key:
            continue
        tokens = parse_tokens(row.get("text") or "")
        by_key[key].append(
            {
                "plan": plan,
                "sample_idx": row.get("sample_idx"),
                "text": row.get("text") or "",
                "angle_triple": "/".join(tokens["angles"][:3]) if len(tokens["angles"]) >= 3 else "",
                "lengths": tokens["lengths"][:3],
            }
        )
    return by_key


def labels_for(branch: str) -> list[dict]:
    rows = load_jsonl(EHULL / branch / "ehull_labels.jsonl")
    for row in rows:
        row["family"] = family(row)
    return rows


def plan_field_stats(branch: str, cfg: dict, labels: list[dict]) -> tuple[dict, list[dict]]:
    index = raw_index(cfg.get("raw"))
    matched = []
    for row in labels:
        candidates = index.get(composition_key_from_dict(row.get("composition") or {}), [])
        if not candidates:
            continue
        best = candidates[0]
        plan = best["plan"]
        matched.append(
            {
                **row,
                "plan_anion": plan.get("anion_framework"),
                "plan_charge": plan.get("charge_bucket"),
                "plan_lattice": plan.get("lattice_system"),
                "plan_spacegroup": plan.get("spacegroup_bucket"),
                "plan_volume": plan.get("volume_per_atom_bin"),
                "angle_triple": best["angle_triple"],
                "lengths": best["lengths"],
                "raw_text_head": best["text"][:180],
                "raw_sample_idx": best["sample_idx"],
                "ambiguous_composition_matches": len(candidates),
            }
        )
    return (
        {
            "matched": len(matched),
            "total_labels": len(labels),
            "by_anion": group_by(matched, lambda r: r.get("plan_anion")),
            "by_charge": group_by(matched, lambda r: r.get("plan_charge")),
            "by_spacegroup": group_by(matched, lambda r: r.get("plan_spacegroup")),
            "by_volume": group_by(matched, lambda r: r.get("plan_volume")),
            "by_angle_triple": group_by(matched, lambda r: r.get("angle_triple")),
        },
        matched,
    )


def contribution_vs_r5c(r5_groups: dict[str, dict], cur_groups: dict[str, dict]) -> list[dict]:
    total_eval = sum(item["evaluated"] for item in cur_groups.values())
    rows = []
    for key, cur in cur_groups.items():
        ref = r5_groups.get(key)
        if not ref or not cur["evaluated"] or not ref["evaluated"]:
            continue
        ref_rate = ref["meta"] / ref["evaluated"]
        cur_rate = cur["meta"] / cur["evaluated"]
        rows.append(
            {
                "group": key,
                "current_eval": cur["evaluated"],
                "current_meta_pct": pct(cur["meta"], cur["evaluated"]),
                "r5c_meta_pct": pct(ref["meta"], ref["evaluated"]),
                "current_mean_e_hull": cur["mean_e_hull"],
                "r5c_mean_e_hull": ref["mean_e_hull"],
                "gap_contribution_pp": 100.0 * (ref_rate - cur_rate) * cur["evaluated"] / total_eval if total_eval else 0.0,
            }
        )
    return sorted(rows, key=lambda row: row["gap_contribution_pp"], reverse=True)


def top_bad_examples(matched: list[dict], limit: int = 8) -> list[dict]:
    rows = sorted([row for row in matched if row.get("e_hull") is not None], key=lambda row: row["e_hull"], reverse=True)
    return [
        {
            "formula": row.get("formula"),
            "e_hull": row.get("e_hull"),
            "family": row.get("family"),
            "anion": row.get("plan_anion"),
            "charge": row.get("plan_charge"),
            "sg": row.get("plan_spacegroup"),
            "volume": row.get("plan_volume"),
            "angles": row.get("angle_triple"),
            "lengths": row.get("lengths"),
            "raw_text_head": row.get("raw_text_head"),
        }
        for row in rows[:limit]
    ]


def planner_gate_metrics(cfg: dict) -> dict:
    for key in ("planner_gate", "proposal_gate", "hybrid_gate"):
        payload = load_json(cfg.get(key))
        if payload:
            return payload.get("metrics", payload)
    return {}


def build() -> dict:
    branches = {}
    family_groups = {}
    plan_groups = {}
    matched_rows = {}
    for key, cfg in BRANCHES.items():
        labels = labels_for(key) if (EHULL / key / "ehull_labels.jsonl").exists() else []
        if not labels and key not in {"r5c_conditional"}:
            # Keep branch if ordinary metrics exist, but mark e_hull missing.
            pass
        fam = group_by(labels, lambda r: r["family"]) if labels else {}
        plan_stats, matched = plan_field_stats(key, cfg, labels) if labels and cfg.get("raw") and cfg.get("raw").exists() else ({}, [])
        family_groups[key] = fam
        plan_groups[key] = plan_stats
        matched_rows[key] = matched
        branches[key] = {
            "display": cfg["display"],
            "kind": cfg["kind"],
            "exists": bool(labels or load_json(cfg.get("sample_metrics")) or load_json(cfg.get("crys")) or load_json(cfg.get("a100"))),
            "sun": sun_from_config(cfg),
            "e_hull": load_json(EHULL / key / "ehull_summary.json"),
            "crysllmgen": as_metrics(load_json(cfg.get("crys"))),
            "sample_metrics": load_json(cfg.get("sample_metrics")),
            "planner_gate": planner_gate_metrics(cfg),
            "raw_token": raw_token_stats(cfg.get("raw")) if cfg.get("raw") else {},
            "family_stats": fam,
            "plan_field_stats": plan_stats,
            "top_bad_examples": top_bad_examples(matched),
        }
    r5 = family_groups.get("r5c_conditional", {})
    comparisons = {
        key: contribution_vs_r5c(r5, fam)
        for key, fam in family_groups.items()
        if key != "r5c_conditional" and fam
    }
    plan_comparisons = {}
    for key, stats in plan_groups.items():
        if key == "r5c_conditional" or not stats:
            continue
        plan_comparisons[key] = {}
        for field in ["by_anion", "by_charge", "by_spacegroup", "by_volume"]:
            if field in plan_groups.get("r5c_conditional", {}) and field in stats:
                plan_comparisons[key][field] = contribution_vs_r5c(plan_groups["r5c_conditional"][field], stats[field])
    return {"ehull_root": str(EHULL), "branches": branches, "family_gap_vs_r5c": comparisons, "plan_field_gap_vs_r5c": plan_comparisons}


def write_report(data: dict) -> None:
    b = data["branches"]
    lines = [
        "# R5-C vs Next-Round H1/H2 Plan-DLM-Diffusion S.U.N. Diagnosis",
        "",
        "Date: 2026-06-04",
        "",
        f"e_hull label root: `{data['ehull_root']}`",
        "",
        "## Executive Answer",
        "",
        "- The gap is primarily a **low-e_hull basin** gap, not a parse/graph/CrysLLMGen validity gap.",
        "- H1-G1 proves that exact-DLM can keep `struct_valid≈100` and `cov_recall≈95`, but meta-like S.U.N. stays around `48-50%`. That makes diffusion/refine insufficient by itself: it repairs structures but does not reliably move them onto the hull.",
        "- H2-P1 proves that dropping dense special tokens naively is not yet viable: the planner is acceptable, but plain-text DLM proposal parse and composition match collapse before diffusion has enough graphs.",
        "- H1-A4 is the key planner test. If its improved joint planner still remains near `meta≈50`, the dominant bottleneck is the DLM/refiner basin. If it moves materially upward, planner joint tuple mismatch is a major causal factor.",
        "",
        "## S.U.N. And CrysLLMGen Summary",
    ]
    order = [
        "r5c_conditional",
        "h1a3_epoch1",
        "h1a3_epoch2",
        "freegeo_ablation_default",
        "h1g1_full_rich",
        "h1g1_condition_dropout",
        "h1g1_formula_volume_sg",
        "h1g1_formula_volume_only",
        "h2p1_plaintext",
        "h1a4_epoch1",
        "h1a4_epoch2",
    ]
    rows = [["Branch", "kind", "strict adj", "meta adj", "Novel+Unique", "e_hull eval", "comp_valid", "struct_valid", "cov_recall", "graph"]]
    for key in order:
        item = b.get(key, {})
        if not item or not item.get("exists"):
            continue
        sun = item.get("sun", {})
        crys = item.get("crysllmgen", {})
        sm = item.get("sample_metrics", {})
        rows.append(
            [
                item["display"],
                item["kind"],
                fmt(sun.get("strict_adjusted")),
                fmt(sun.get("meta_adjusted")),
                fmt(sun.get("novel_unique")),
                f"{sun.get('eval_num')}/{sun.get('eval_den')}" if sun.get("eval_den") else "-",
                fmt(crys.get("comp_valid")),
                fmt(crys.get("struct_valid")),
                fmt(crys.get("cov_recall")),
                fmt(sm.get("graph_acceptance_rate") or item.get("planner_gate", {}).get("graph_acceptance")),
            ]
        )
    lines.extend(table(rows))
    lines.append("")

    lines.append("## Material Family e_hull Splits")
    important = [
        "oxide",
        "non-oxide chalcogenide",
        "halide",
        "intermetallic/all-metal",
        "mixed-anion/oxide+halide",
        "mixed-anion/oxide+pnictide",
    ]
    for key in order:
        item = b.get(key, {})
        stats = item.get("family_stats") or {}
        if not stats:
            continue
        lines.append(f"### {item['display']}")
        rows = [["Family", "eval", "strict/eval", "meta/eval", "mean e_hull"]]
        for family_name in important:
            row = stats.get(family_name)
            if row:
                rows.append(
                    [
                        family_name,
                        row["evaluated"],
                        f"{fmt(row['strict_eval_pct'])}%",
                        f"{fmt(row['meta_eval_pct'])}%",
                        fmt(row["mean_e_hull"], 3),
                    ]
                )
        lines.extend(table(rows))
        lines.append("")

    lines.append("## Largest Meta Gap Contributions vs R5-C")
    for key in ["h1a3_epoch1", "h1a3_epoch2", "freegeo_ablation_default", "h1g1_full_rich", "h1g1_condition_dropout", "h1g1_formula_volume_sg", "h1g1_formula_volume_only", "h2p1_plaintext", "h1a4_epoch1", "h1a4_epoch2"]:
        rows_data = data["family_gap_vs_r5c"].get(key)
        if not rows_data:
            continue
        lines.append(f"### {b[key]['display']}")
        rows = [["Family", "current eval", "current meta", "R5-C meta", "current e_hull", "R5-C e_hull", "gap pp"]]
        for row in rows_data[:8]:
            rows.append(
                [
                    row["group"],
                    row["current_eval"],
                    f"{fmt(row['current_meta_pct'])}%",
                    f"{fmt(row['r5c_meta_pct'])}%",
                    fmt(row["current_mean_e_hull"], 3),
                    fmt(row["r5c_mean_e_hull"], 3),
                    fmt(row["gap_contribution_pp"]),
                ]
            )
        lines.extend(table(rows))
        lines.append("")

    lines.append("## Plan Field e_hull Splits")
    for key in ["h1g1_full_rich", "h1g1_condition_dropout", "h1g1_formula_volume_sg", "h1g1_formula_volume_only", "h1a4_epoch1", "h1a4_epoch2"]:
        stats = b.get(key, {}).get("plan_field_stats") or {}
        if not stats:
            continue
        lines.append(f"### {b[key]['display']}")
        for title, field in [("Anion", "by_anion"), ("Charge", "by_charge"), ("Spacegroup", "by_spacegroup"), ("Volume", "by_volume")]:
            group = stats.get(field) or {}
            rows = [["Group", "eval", "strict/eval", "meta/eval", "mean e_hull"]]
            for name, row in sorted(group.items(), key=lambda kv: -kv[1]["evaluated"])[:8]:
                rows.append([name, row["evaluated"], f"{fmt(row['strict_eval_pct'])}%", f"{fmt(row['meta_eval_pct'])}%", fmt(row["mean_e_hull"], 3)])
            lines.append(f"**{title}**")
            lines.extend(table(rows))
            lines.append("")

    lines.append("## Raw DLM Output")
    rows = [["Branch", "raw n", "angle template", "all-90", "all-60", "two lengths eq", "quarter coords", "top angle"]]
    for key in order:
        raw = b.get(key, {}).get("raw_token") or {}
        if not raw.get("count"):
            continue
        top = raw.get("top_angle_triples") or []
        rows.append(
            [
                b[key]["display"],
                raw["count"],
                f"{fmt(raw['angle_template_pct'])}%",
                f"{fmt(raw['all90_pct'])}%",
                f"{fmt(raw['all60_pct'])}%",
                f"{fmt(raw['two_lengths_equal_pct'])}%",
                f"{fmt(raw['quarter_coord_pct'])}%",
                top[0][0] if top else "-",
            ]
        )
    lines.extend(table(rows))
    lines.append("")

    lines.append("## Bad Examples")
    for key in ["h1g1_full_rich", "h1g1_condition_dropout", "h2p1_plaintext", "h1a4_epoch1", "h1a4_epoch2"]:
        examples = b.get(key, {}).get("top_bad_examples") or []
        if not examples:
            continue
        lines.append(f"### {b[key]['display']}")
        rows = [["Formula", "e_hull", "family", "anion", "charge", "SG", "vol", "angles", "lengths"]]
        for row in examples[:6]:
            rows.append(
                [
                    row.get("formula"),
                    fmt(row.get("e_hull"), 3),
                    row.get("family"),
                    row.get("anion"),
                    row.get("charge"),
                    row.get("sg"),
                    row.get("volume"),
                    row.get("angles"),
                    "/".join(row.get("lengths") or []),
                ]
            )
        lines.extend(table(rows))
        lines.append("")

    lines.extend(
        [
            "## Attribution",
            "",
            "**Plan:** Planner syntax and marginal distributions can be made acceptable. H2-P1 and H1-A4 planner gates show parse/TVD are not the only problem. The remaining plan-side issue is joint tuple realizability, especially oxide/chalcogenide plus charge/volume/SG compatibility.",
            "",
            "**DLM:** Exact-DLM is the strongest current suspect after H1-G1: it produces valid and cover-relevant structures, but the generated-plan proposals do not land in R5-C's low-e_hull basin. H2-P1 shows that replacing special tokens with plain text is not solved yet; the proposal parser/graph acceptance collapses.",
            "",
            "**Diffusion/refine:** Refine is necessary for structural validity, but not sufficient for S.U.N. It can turn accepted proposals into valid structures, yet H1-G1's high `cov_recall` and `struct_valid` remain far below R5-C meta S.U.N. Therefore the refiner cannot be treated as a universal low-e_hull corrector.",
            "",
            "## Recommended Next Training Target",
            "",
            "Train and select on **stable basin entry**, not only token CE or Crys metrics. The target should be: generated plan tuples whose exact-DLM proposal, after fixed CrysLLMGen refinement, yields low e_hull in oxide/chalcogenide and charge_fail bins. Practically, the next useful loop is to log e_hull-conditioned plan/geometry failures and train a proposal model or planner reranker only as an analysis-time model, while keeping final generation de novo and unscreened.",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    data = build()
    OUT_JSON.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(data)
    print(OUT_JSON)
    print(OUT_MD)


if __name__ == "__main__":
    main()
