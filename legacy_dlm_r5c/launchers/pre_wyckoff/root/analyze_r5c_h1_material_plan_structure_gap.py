#!/usr/bin/env python3
"""Material/plan/structure attribution for R5-C vs H1 S.U.N. gap."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path("/mnt/d/codex_work/ai4s/diffsion_language_model_meets_diffusion")
EHULL = ROOT / "reports/20260604_ehull_label_export_remote/20260604_ehull_label_export/outputs/ehull_labels"
OUT_JSON = ROOT / "reports/20260604_r5c_vs_h1a3_freegeo_material_plan_structure_ehull_diagnosis.json"
OUT_MD = ROOT / "reports/20260604_r5c_vs_h1a3_freegeo_material_plan_structure_ehull_diagnosis.md"

BRANCH_RAW = {
    "r5c_conditional": ROOT / "runs/20260531_0040-r5c-full1000-sun/outputs/r5c_sample1000/raw_generations.jsonl",
    "h1a3_epoch1": ROOT / "reports/20260604_h1a3_epoch1_2_artifacts/runs/20260603_h1a3_joint_planner/outputs/h1a3_epoch1_hybrid_body256/raw_generations.jsonl",
    "h1a3_epoch2": ROOT / "reports/20260604_h1a3_epoch1_2_artifacts/runs/20260603_h1a3_joint_planner/outputs/h1a3_epoch2_hybrid_body256/raw_generations.jsonl",
    "freegeo_ablation_default": ROOT / "reports/20260604_freegeo_default_full1000_artifacts/runs/20260603_h1_freegeo_remaining7_full1000_sun/outputs/ablation_default_sample1200/raw_generations.jsonl",
}

DISPLAY = {
    "r5c_conditional": "R5-C conditional",
    "h1a3_epoch1": "H1-A3 epoch1",
    "h1a3_epoch2": "H1-A3 epoch2",
    "freegeo_ablation_default": "free-geometry default",
}

TOKEN_RE = re.compile(r"<([^>]+)>")


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def pct(n: float, d: float) -> float:
    return 100.0 * n / d if d else 0.0


def fmt(value, nd=2) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{nd}f}"
    return str(value)


def table(rows: list[list]) -> list[str]:
    widths = [max(len(str(row[i])) for row in rows) for i in range(len(rows[0]))]
    out = []
    for idx, row in enumerate(rows):
        out.append("| " + " | ".join(str(row[i]).ljust(widths[i]) for i in range(len(row))) + " |")
        if idx == 0:
            out.append("| " + " | ".join("-" * widths[i] for i in range(len(row))) + " |")
    return out


def composition_key_from_dict(comp: dict) -> tuple[tuple[str, int], ...]:
    return tuple(sorted((str(el), int(round(float(count)))) for el, count in comp.items()))


def composition_key_from_plan(plan: dict) -> tuple[tuple[str, int], ...]:
    return tuple(
        sorted(
            (str(el), int(round(float(count))))
            for el, count in zip(plan.get("elements") or [], plan.get("counts") or [])
        )
    )


def family(row: dict) -> str:
    els = set(row["elements"])
    halides = {"F", "Cl", "Br", "I"}
    pnictides = {"N", "P", "As", "Sb", "Bi"}
    nonoxide_chalc = {"S", "Se", "Te"}
    if not row.get("contains_common_anion") and not (els & pnictides) and "H" not in els:
        return "intermetallic/all-metal"
    parts = []
    if "O" in els:
        parts.append("oxide")
    if els & nonoxide_chalc:
        parts.append("non-oxide chalcogenide")
    if els & halides:
        parts.append("halide")
    if els & pnictides:
        parts.append("pnictide")
    if "H" in els:
        parts.append("hydride")
    if not parts:
        return "other-anion"
    if len(parts) == 1:
        return parts[0]
    return "mixed-anion/" + "+".join(parts[:3])


def parse_tokens(text: str) -> dict:
    tokens = TOKEN_RE.findall(text or "")
    out = {"lengths": [], "angles": [], "coords": []}
    for tok in tokens:
        if "_" not in tok:
            continue
        prefix, value = tok.split("_", 1)
        if prefix in {"LA", "LB", "LC"}:
            out["lengths"].append(value)
        elif prefix in {"AA", "AB", "AG"}:
            out["angles"].append(value)
        elif prefix in {"X", "Y", "Z"}:
            out["coords"].append(value)
    return out


def raw_index(path: Path) -> dict:
    by_key: dict[tuple, list[dict]] = defaultdict(list)
    for item in load_jsonl(path):
        plan = item.get("plan_state") or {}
        key = composition_key_from_plan(plan)
        if not key:
            continue
        text = item.get("text") or ""
        tokens = parse_tokens(text)
        angle_triple = "/".join(tokens["angles"][:3]) if len(tokens["angles"]) >= 3 else ""
        rec = {
            "sample_idx": item.get("sample_idx"),
            "plan": plan,
            "text": text,
            "angle_triple": angle_triple,
            "lengths": tokens["lengths"][:3],
            "coord_head": tokens["coords"][:9],
        }
        by_key[key].append(rec)
    return by_key


def branch_labels(branch: str) -> list[dict]:
    rows = load_jsonl(EHULL / branch / "ehull_labels.jsonl")
    for row in rows:
        row["family"] = family(row)
    return rows


def summarize_group(rows: list[dict]) -> dict:
    evaluated = [r for r in rows if r.get("e_hull") is not None]
    strict = [r for r in evaluated if r.get("strict_stable")]
    meta = [r for r in evaluated if r.get("meta_stable")]
    return {
        "n": len(rows),
        "evaluated": len(evaluated),
        "strict": len(strict),
        "meta": len(meta),
        "strict_eval_pct": pct(len(strict), len(evaluated)),
        "meta_eval_pct": pct(len(meta), len(evaluated)),
        "mean_e_hull": sum(r["e_hull"] for r in evaluated) / len(evaluated) if evaluated else None,
    }


def group_by(rows: list[dict], key_fn) -> dict[str, dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[str(key_fn(row))].append(row)
    return {key: summarize_group(value) for key, value in groups.items()}


def sorted_group_rows(groups: dict[str, dict], min_eval: int = 1) -> list[dict]:
    out = []
    for key, value in groups.items():
        if value["evaluated"] >= min_eval:
            out.append({"group": key, **value})
    return sorted(out, key=lambda row: (-row["n"], row["group"]))


def plan_field_stats(branch: str, rows: list[dict]) -> tuple[dict, list[dict]]:
    raw = raw_index(BRANCH_RAW[branch])
    matched = []
    for row in rows:
        candidates = raw.get(composition_key_from_dict(row["composition"]), [])
        if not candidates:
            continue
        best = candidates[0]
        plan = best["plan"]
        enriched = {
            **row,
            "plan_anion": plan.get("anion_framework"),
            "plan_charge": plan.get("charge_bucket"),
            "plan_lattice": plan.get("lattice_system"),
            "plan_spacegroup": plan.get("spacegroup_bucket"),
            "plan_volume": plan.get("volume_per_atom_bin"),
            "raw_sample_idx": best.get("sample_idx"),
            "raw_text": best.get("text"),
            "angle_triple": best.get("angle_triple"),
            "lengths": best.get("lengths"),
            "coord_head": best.get("coord_head"),
            "ambiguous_composition_matches": len(candidates),
        }
        matched.append(enriched)
    stats = {
        "matched": len(matched),
        "total_labels": len(rows),
        "ambiguous_matches": sum(1 for row in matched if row["ambiguous_composition_matches"] > 1),
        "by_anion": group_by(matched, lambda r: r.get("plan_anion")),
        "by_charge": group_by(matched, lambda r: r.get("plan_charge")),
        "by_lattice": group_by(matched, lambda r: r.get("plan_lattice")),
        "by_spacegroup": group_by(matched, lambda r: r.get("plan_spacegroup")),
        "by_volume": group_by(matched, lambda r: r.get("plan_volume")),
        "by_angle_triple": group_by(matched, lambda r: r.get("angle_triple")),
    }
    return stats, matched


def contribution_vs_r5c(r5_groups: dict[str, dict], cur_groups: dict[str, dict]) -> list[dict]:
    total_eval = sum(v["evaluated"] for v in cur_groups.values())
    out = []
    for key, cur in cur_groups.items():
        ref = r5_groups.get(key)
        if not ref or not cur["evaluated"] or not ref["evaluated"]:
            continue
        ref_rate = ref["meta"] / ref["evaluated"]
        cur_rate = cur["meta"] / cur["evaluated"]
        out.append(
            {
                "group": key,
                "current_eval": cur["evaluated"],
                "current_meta_pct": pct(cur["meta"], cur["evaluated"]),
                "r5c_meta_pct": pct(ref["meta"], ref["evaluated"]),
                "current_mean_e_hull": cur["mean_e_hull"],
                "r5c_mean_e_hull": ref["mean_e_hull"],
                "gap_contribution_pp": 100.0 * (ref_rate - cur_rate) * cur["evaluated"] / total_eval,
            }
        )
    return sorted(out, key=lambda row: row["gap_contribution_pp"], reverse=True)


def top_bad_examples(matched: list[dict], limit: int = 12) -> list[dict]:
    bad = sorted(
        [row for row in matched if row.get("e_hull") is not None],
        key=lambda row: row["e_hull"],
        reverse=True,
    )[:limit]
    out = []
    for row in bad:
        out.append(
            {
                "formula": row["formula"],
                "e_hull": row["e_hull"],
                "family": row["family"],
                "plan_anion": row.get("plan_anion"),
                "plan_charge": row.get("plan_charge"),
                "plan_lattice": row.get("plan_lattice"),
                "plan_spacegroup": row.get("plan_spacegroup"),
                "plan_volume": row.get("plan_volume"),
                "angle_triple": row.get("angle_triple"),
                "lengths": row.get("lengths"),
                "raw_sample_idx": row.get("raw_sample_idx"),
                "raw_text_head": (row.get("raw_text") or "")[:180],
            }
        )
    return out


def build() -> dict:
    labels = {branch: branch_labels(branch) for branch in BRANCH_RAW}
    family_groups = {branch: group_by(rows, lambda r: r["family"]) for branch, rows in labels.items()}
    plan_stats = {}
    matched = {}
    for branch, rows in labels.items():
        plan_stats[branch], matched[branch] = plan_field_stats(branch, rows)
    payload = {
        "families": family_groups,
        "family_gap_contribution_vs_r5c": {
            branch: contribution_vs_r5c(family_groups["r5c_conditional"], family_groups[branch])
            for branch in ["h1a3_epoch1", "h1a3_epoch2", "freegeo_ablation_default"]
        },
        "plan_field_stats": plan_stats,
        "plan_field_gap_contribution_vs_r5c": {
            branch: {
                field: contribution_vs_r5c(plan_stats["r5c_conditional"][field], plan_stats[branch][field])
                for field in ["by_anion", "by_charge", "by_lattice", "by_spacegroup", "by_volume"]
            }
            for branch in ["h1a3_epoch1", "h1a3_epoch2", "freegeo_ablation_default"]
        },
        "top_bad_examples": {
            branch: top_bad_examples(matched[branch])
            for branch in ["r5c_conditional", "h1a3_epoch1", "h1a3_epoch2", "freegeo_ablation_default"]
        },
    }
    return payload


def selected_groups_table(data: dict, branch: str, groups: list[str]) -> list[list]:
    rows = [["Type", "eval", "strict/eval", "meta/eval", "mean e_hull"]]
    fams = data["families"][branch]
    for group in groups:
        item = fams.get(group)
        if item:
            rows.append(
                [
                    group,
                    item["evaluated"],
                    f"{fmt(item['strict_eval_pct'])}%",
                    f"{fmt(item['meta_eval_pct'])}%",
                    fmt(item["mean_e_hull"], 3),
                ]
            )
    return rows


def write_md(data: dict) -> None:
    lines = [
        "# Material/Plan/Structure e_hull Gap Diagnosis",
        "",
        "Date: 2026-06-04",
        "",
        "This is a deeper follow-up to the S.U.N. gap report. It asks which material classes and which generated plan/structure patterns create the stability gap, rather than only comparing aggregate S.U.N. metrics.",
        "",
        "## Short Answer",
        "",
        "- The largest chemistry-side gap is in **oxygen-containing systems**, especially plain oxides and mixed oxide-anion compositions. R5-C oxide meta-stable/evaluated is `58.41%`; H1-A3 epoch1 is `41.67%`, H1-A3 epoch2 is `25.49%`, and free-geometry default is `32.76%`.",
        "- **Non-oxide chalcogenides** also degrade sharply: R5-C `70.09%` meta vs H1-A3 epoch1 `38.10%` and freegeo default `53.85%`.",
        "- **Halides and all-metal/intermetallics are comparatively healthier**, but still lower than R5-C. Halides remain one of the best current families; oxides are the clearest failure family.",
        "- The gap is both **plan-side** and **structure/executor-side**. Plan-side: H1-A3 epoch2 shifts to larger N and many `charge_fail` / oxide plans. Structure-side: even under the same generated plan fields (`anion=oxide`, `charge=charge_fail`, `volpa_010_014`, `sg_003_015/sg_143_167`), H1/freegeo e_hull is much worse than R5-C, so the DLM/refiner basin is not matching teacher-conditioned geometry.",
        "",
        "## Material Type Stability",
    ]
    important = [
        "intermetallic/all-metal",
        "oxide",
        "non-oxide chalcogenide",
        "pnictide",
        "halide",
        "mixed-anion/oxide+halide",
        "mixed-anion/oxide+pnictide",
        "mixed-anion/oxide+non-oxide chalcogenide",
    ]
    for branch in ["r5c_conditional", "h1a3_epoch1", "h1a3_epoch2", "freegeo_ablation_default"]:
        lines.append(f"### {DISPLAY[branch]}")
        lines.extend(table(selected_groups_table(data, branch, important)))
        lines.append("")

    lines.append("## Which Types Contribute Most To The Meta Gap")
    for branch in ["h1a3_epoch1", "h1a3_epoch2", "freegeo_ablation_default"]:
        lines.append(f"### {DISPLAY[branch]} vs R5-C")
        rows = [["Type", "current eval", "current meta", "R5-C meta", "current mean e_hull", "R5-C mean e_hull", "gap contribution"]]
        for row in data["family_gap_contribution_vs_r5c"][branch][:8]:
            rows.append(
                [
                    row["group"],
                    row["current_eval"],
                    f"{fmt(row['current_meta_pct'])}%",
                    f"{fmt(row['r5c_meta_pct'])}%",
                    fmt(row["current_mean_e_hull"], 3),
                    fmt(row["r5c_mean_e_hull"], 3),
                    f"{fmt(row['gap_contribution_pp'])} pp",
                ]
            )
        lines.extend(table(rows))
        lines.append("")

    lines.append("Interpretation: freegeo default's biggest losses are oxides and all-metal/intermetallics by contribution, but for different reasons. Oxides have a very large rate gap; all-metal has a smaller rate gap but high count. H1-A3 epoch2 is worse because oxides collapse to only `25.49%` meta-stable and N shifts larger.")
    lines.append("")

    lines.append("## Plan Field Stability")
    for branch in ["r5c_conditional", "h1a3_epoch1", "h1a3_epoch2", "freegeo_ablation_default"]:
        lines.append(f"### {DISPLAY[branch]}")
        for title, key, min_eval in [
            ("Anion framework", "by_anion", 5),
            ("Charge bucket", "by_charge", 5),
            ("Volume bin", "by_volume", 5),
            ("Spacegroup bucket", "by_spacegroup", 10),
        ]:
            lines.append(f"**{title}**")
            rows = [["Group", "eval", "strict/eval", "meta/eval", "mean e_hull"]]
            for item in sorted_group_rows(data["plan_field_stats"][branch][key], min_eval=min_eval)[:10]:
                rows.append([item["group"], item["evaluated"], f"{fmt(item['strict_eval_pct'])}%", f"{fmt(item['meta_eval_pct'])}%", fmt(item["mean_e_hull"], 3)])
            lines.extend(table(rows))
            lines.append("")

    lines.append("Plan-field diagnosis: `anion=oxide`, `charge=charge_fail`, `volpa_010_014/005_009`, and low/mid symmetry buckets such as `sg_003_015` and `sg_143_167` are consistently weak under generated H1/freegeo plans. R5-C has many of these same fields, but its e_hull is substantially lower, which means the teacher plan is coupled to a better geometry basin.")
    lines.append("")

    lines.append("## Concrete Bad Generated Examples")
    for branch in ["h1a3_epoch1", "h1a3_epoch2", "freegeo_ablation_default"]:
        lines.append(f"### {DISPLAY[branch]}")
        rows = [["Formula", "e_hull", "family", "anion", "charge", "lattice", "SG", "vol", "angles", "lengths"]]
        for ex in data["top_bad_examples"][branch][:10]:
            rows.append(
                [
                    ex["formula"],
                    fmt(ex["e_hull"], 3),
                    ex["family"],
                    ex["plan_anion"],
                    ex["plan_charge"],
                    ex["plan_lattice"],
                    ex["plan_spacegroup"],
                    ex["plan_volume"],
                    ex["angle_triple"],
                    "/".join(ex["lengths"] or []),
                ]
            )
        lines.extend(table(rows))
        lines.append("")

    lines.extend(
        [
            "Representative failure patterns:",
            "",
            "- H1-A3 epoch1: `MoSe2` (`e_hull≈1.047`) is a chalcogenide plan with `hexagonal / sg_143_167 / volpa_025_029`, but the raw body is a coarse special-token sketch with `090/090/120` angles and a very elongated `LC_153` body token. This is not a formula-only failure; it is plan-geometry incompatibility.",
            "- H1-A3 epoch1/2: several oxides such as `CeCuO2`, `MnCoO4`, `SrS2O`, `PrEu3(ClO4)2`, `EuClO2` have `charge_fail` or mixed oxide-halide plans and e_hull far above `0.1`.",
            "- Freegeo default: `Te3Mo3Se` (`e_hull≈2.325`) and `GaNO` (`e_hull≈1.451`) show extreme failures. `Te3Mo3Se` has a huge `LC_371` raw body token; `GaNO` is a mixed oxide/pnictide under `charge_fail / triclinic / sg_003_015 / volpa_010_014`.",
            "- These are exactly the classes where R5-C conditional still has moderate to high meta-stability, so the difference is not that MP-20 cannot contain such chemistry. The difference is that generated plans and special-token geometry are not coupled like teacher plans.",
        ]
    )
    lines.append("")

    lines.append("## Plan Or Structure?")
    lines.extend(
        [
            "",
            "**Plan-side evidence:** H1-A3 epoch2 over-shifts `N` and large-cell plans; generated plans contain many low-success buckets such as `charge_fail`, oxides in small volume bins, and `sg_003_015/sg_143_167`. This changes what the DLM is asked to execute.",
            "",
            "**Structure/executor evidence:** Under matched plan fields the current branches still have much worse e_hull than R5-C. For example, `anion=oxide` is `57.69%` meta in R5-C but only `30.86% / 29.55% / 33.46%` in H1-A3 epoch1 / epoch2 / freegeo default. `charge_fail` is `63.79%` in R5-C but `44.09% / 36.75% / 44.10%`. This cannot be fixed by marginal plan distribution alone.",
            "",
            "**Bottom line:** the gap is a coupled plan-structure problem. Planner quality matters, but the exact-DLM special-token body is only reliable when the plan is teacher-like and geometry-realizable. Generated rich plans expose brittleness in lattice/coord proposals, especially for oxides and chalcogenides.",
        ]
    )
    lines.append("")

    lines.append("## Next Changes Suggested By This Diagnosis")
    lines.extend(
        [
            "",
            "1. Prioritize oxide/chalcogenide-specific diagnostics before another broad H1 run. Track `anion=oxide`, `mixed oxide`, `chalcogenide`, and `charge_fail` e_hull separately.",
            "2. For planner training, do not only match global N/arity. Add joint calibration reports for `formula_family + charge + volume + SG`, especially oxide small-volume and oxide low-symmetry bins.",
            "3. For DLM/executor, compare teacher-conditioned vs generated-plan raw geometry inside the same oxide/chalcogenide bins. The current evidence says lattice/coord special-token sketches are too brittle for generated plans.",
            "4. Consider H2/plain-text proposal or less quantized geometry for oxide/chalcogenide branches first, rather than globally reweighting coordinate CE.",
            "5. Keep e_hull label export as mandatory; aggregate S.U.N. hides that oxides are failing much harder than halides/all-metals.",
        ]
    )
    lines.append("")
    lines.append("## Artifacts")
    lines.extend(
        [
            f"- JSON: `{OUT_JSON}`",
            f"- e_hull labels: `{EHULL}`",
            "- Previous aggregate report: `reports/20260604_r5c_vs_h1a3_freegeo_sun_gap_diagnosis.md`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    data = build()
    OUT_JSON.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_md(data)
    print(OUT_JSON)
    print(OUT_MD)


if __name__ == "__main__":
    main()
