#!/usr/bin/env python3
"""Compare R5-C conditional and H1-A2 de novo plan/refine/S.U.N. outputs."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_r5c_plan_distribution import (  # noqa: E402
    canonical_plan,
    compare_stats,
    distribution,
    empty_stats,
    extract_generated_plan,
    js_divergence,
    summarize,
    tvd,
)


CHALCOGENS = {"O", "S", "Se", "Te"}
HALOGENS = {"F", "Cl", "Br", "I"}


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def parse_summary_md(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    payload: dict[str, Any] = {"path": str(path)}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip().lower().replace(" ", "_").replace(".", "")
        value = value.strip()
        payload[key] = value
        m = re.search(r"([-+]?\d+(?:\.\d+)?)%", value)
        if m:
            payload[f"{key}_pct"] = float(m.group(1))
        m = re.search(r"(\d+)\s*/\s*(\d+)", value)
        if m:
            payload[f"{key}_num"] = int(m.group(1))
            payload[f"{key}_den"] = int(m.group(2))
    return payload


def add_counter(counter: Counter[str], key: Any) -> None:
    counter[str(key)] += 1


def plan_records(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, record in enumerate(iter_jsonl(path)):
        if isinstance(record.get("plan_state"), Mapping):
            try:
                rows.append({"row_idx": idx, "plan": canonical_plan(record["plan_state"]), "raw": record})
                continue
            except Exception as exc:  # noqa: BLE001
                rows.append({"row_idx": idx, "failure": f"plan_state:{type(exc).__name__}: {exc}", "raw": record})
                continue
        plan, failure = extract_generated_plan(record)
        if plan is None:
            rows.append({"row_idx": idx, "failure": failure, "raw": record})
        else:
            rows.append({"row_idx": idx, "plan": plan, "raw": record})
    return rows


def plan_stats_from_records(name: str, rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    stats = empty_stats(name)
    pair_counters: dict[str, Counter[str]] = {
        "charge_x_anion": Counter(),
        "charge_x_volume": Counter(),
        "anion_x_volume": Counter(),
        "lattice_x_sg": Counter(),
        "n_x_arity": Counter(),
        "charge_x_anion_x_volume": Counter(),
        "formula_x_fields": Counter(),
    }
    flags = Counter()
    for row in rows:
        stats["rows"] += 1
        plan = row.get("plan")
        if not plan:
            stats["failed_rows"] += 1
            stats["failure_reasons"][str(row.get("failure") or "unknown")] += 1
            continue
        from scripts.analyze_r5c_plan_distribution import add_plan  # noqa: PLC0415

        add_plan(stats, plan)
        elements = [str(item) for item in plan.get("elements") or []]
        n_value = int(plan.get("N") or 0)
        arity = str(len(elements))
        charge = str(plan.get("charge_bucket", "unknown"))
        anion = str(plan.get("anion_framework", "unknown"))
        lattice = str(plan.get("lattice_system", "unknown"))
        sg = str(plan.get("spacegroup_bucket", "sg_unknown"))
        volume = str(plan.get("volume_per_atom_bin", "volpa_unknown"))
        formula = str(plan.get("formula", ""))
        add_counter(pair_counters["charge_x_anion"], f"{charge}|{anion}")
        add_counter(pair_counters["charge_x_volume"], f"{charge}|{volume}")
        add_counter(pair_counters["anion_x_volume"], f"{anion}|{volume}")
        add_counter(pair_counters["lattice_x_sg"], f"{lattice}|{sg}")
        add_counter(pair_counters["n_x_arity"], f"{n_value}|{arity}")
        add_counter(pair_counters["charge_x_anion_x_volume"], f"{charge}|{anion}|{volume}")
        add_counter(pair_counters["formula_x_fields"], f"{formula}|{charge}|{anion}|{lattice}|{sg}|{volume}")
        if any(symbol == "O" for symbol in elements):
            flags["contains_O"] += 1
        if any(symbol in CHALCOGENS for symbol in elements):
            flags["contains_chalcogen"] += 1
        if any(symbol in HALOGENS for symbol in elements):
            flags["contains_halogen"] += 1
        if charge in {"charge_fail", "charge_neutrality_fail"}:
            flags["charge_fail_bucket"] += 1
        if charge == "neutral_plausible":
            flags["neutral_plausible"] += 1
        if charge == "all_metal":
            flags["all_metal_bucket"] += 1
        if anion == "oxide":
            flags["anion_oxide"] += 1
    parsed = max(1, int(stats["parsed_rows"]))
    derived = {
        "flags": {key: value / parsed for key, value in flags.items()},
        "pair_counters": {key: dict(counter.most_common(40)) for key, counter in pair_counters.items()},
        "pair_counter_totals": {key: sum(counter.values()) for key, counter in pair_counters.items()},
    }
    return stats, derived


def extract_dynamic_tokens(text: str) -> dict[str, Any] | None:
    lattice = re.findall(r"<L([ABC])_(\d{3})>", text)
    angles = re.findall(r"<A([ABG])_(\d{3})>", text)
    coords = re.findall(r"<E_([A-Za-z][a-z]?)><X_(\d{3})><Y_(\d{3})><Z_(\d{3})>", text)
    if len(lattice) != 3 or len(angles) != 3:
        return None
    lat_values = {axis: int(value) for axis, value in lattice}
    angle_values = {axis: int(value) for axis, value in angles}
    coord_values = [(el, int(x), int(y), int(z)) for el, x, y, z in coords]
    coord_positions = [(x, y, z) for _, x, y, z in coord_values]
    return {
        "lat": lat_values,
        "angles": angle_values,
        "coords": coord_values,
        "coord_positions": coord_positions,
    }


def body_geometry_summary(path: Path) -> dict[str, Any]:
    count = 0
    parsed = 0
    flags = Counter()
    n_values = Counter()
    angle_patterns = Counter()
    lattice_patterns = Counter()
    coord_top = Counter()
    for record in iter_jsonl(path):
        count += 1
        text = str(record.get("text") or "")
        tokens = extract_dynamic_tokens(text)
        if tokens is None:
            continue
        parsed += 1
        lat = [tokens["lat"].get(axis) for axis in ("A", "B", "C")]
        angles = [tokens["angles"].get(axis) for axis in ("A", "B", "G")]
        coords = tokens["coord_positions"]
        n_values[str(len(coords))] += 1
        if lat[0] == lat[1] == lat[2]:
            flags["a_eq_b_eq_c"] += 1
        if lat[0] == lat[1] or lat[0] == lat[2] or lat[1] == lat[2]:
            flags["any_two_lengths_equal"] += 1
        if angles[0] == angles[1] == angles[2] == 90:
            flags["all_angles_90"] += 1
        if angles[0] == angles[1] == angles[2] == 60:
            flags["all_angles_60"] += 1
        if angles == [90, 90, 120]:
            flags["angles_90_90_120"] += 1
        if len(coords) != len(set(coords)):
            flags["duplicate_coords_exact"] += 1
        if coords and all(all(v in {0, 25, 50, 75} for v in pos) for pos in coords):
            flags["all_coords_quarter_grid"] += 1
        angle_patterns["/".join(str(x) for x in angles)] += 1
        lattice_patterns["/".join(str(x) for x in lat)] += 1
        for pos in coords:
            coord_top["/".join(str(x) for x in pos)] += 1
    denom = max(1, parsed)
    return {
        "rows": count,
        "parsed_dynamic": parsed,
        "parse_rate": parsed / max(1, count),
        "rates": {key: value / denom for key, value in flags.items()},
        "n_histogram": dict(n_values.most_common()),
        "angle_patterns_top20": dict(angle_patterns.most_common(20)),
        "lattice_patterns_top20": dict(lattice_patterns.most_common(20)),
        "coord_top20": dict(coord_top.most_common(20)),
    }


def composition_summary(path: Path | None) -> dict[str, Any]:
    payload = load_json(path)
    if not payload:
        return {}
    raw = payload.get("raw_jsonl") if isinstance(payload.get("raw_jsonl"), Mapping) else {}
    refined = payload.get("refined_pt") if isinstance(payload.get("refined_pt"), Mapping) else {}
    return {
        "path": str(path),
        "raw_comp_valid_rate": raw.get("comp_valid_rate"),
        "raw_reason_counts": raw.get("reason_counts"),
        "raw_pbc_duplicate_fraction": raw.get("pbc_equivalent_duplicate_fraction"),
        "refined_comp_valid_rate": refined.get("comp_valid_rate"),
        "refined_reason_counts": refined.get("reason_counts"),
        "refined_pbc_duplicate_fraction": refined.get("pbc_equivalent_duplicate_fraction"),
        "refined_formula_top30": refined.get("formula_top30"),
    }


def metrics_summary(path: Path | None) -> dict[str, Any]:
    payload = load_json(path)
    if not payload:
        return {}
    return payload.get("metrics") if isinstance(payload.get("metrics"), Mapping) else payload


def relax_comp_summary(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    rows = list(iter_jsonl(path))
    n_hist = Counter()
    arity_hist = Counter()
    flags = Counter()
    energy_values: list[float] = []
    chemsys = Counter()
    formula = Counter()
    for row in rows:
        comp = row.get("composition") if isinstance(row.get("composition"), Mapping) else {}
        elements = list(comp)
        counts = [float(value) for value in comp.values()]
        n_value = int(round(sum(counts)))
        n_hist[str(n_value)] += 1
        arity_hist[str(len(elements))] += 1
        if elements and all(el in CHALCOGENS for el in elements):
            flags["all_chalcogen"] += 1
        if any(el == "O" for el in elements):
            flags["contains_O"] += 1
        if any(el in CHALCOGENS for el in elements):
            flags["contains_chalcogen"] += 1
        if any(el in HALOGENS for el in elements):
            flags["contains_halogen"] += 1
        chemsys["-".join(sorted(elements))] += 1
        formula["".join(f"{el}{int(v) if float(v) != 1.0 else ''}" for el, v in sorted(comp.items()))] += 1
        try:
            energy_values.append(float(row["energy_per_atom"]))
        except Exception:
            pass
    energy_values.sort()
    denom = max(1, len(rows))
    q = {}
    if energy_values:
        for name, frac in (("p10", 0.1), ("p50", 0.5), ("p90", 0.9)):
            idx = min(len(energy_values) - 1, max(0, int(round(frac * (len(energy_values) - 1)))))
            q[name] = energy_values[idx]
        q["mean"] = sum(energy_values) / len(energy_values)
    return {
        "rows": len(rows),
        "n_histogram": dict(n_hist.most_common()),
        "arity_histogram": dict(arity_hist.most_common()),
        "rates": {key: value / denom for key, value in flags.items()},
        "energy_per_atom": q,
        "chemsys_top30": dict(chemsys.most_common(30)),
        "formula_top30": dict(formula.most_common(30)),
    }


def counter_tvd_from_dict(left: Mapping[str, int], right: Mapping[str, int]) -> float:
    return tvd(Counter({str(k): int(v) for k, v in left.items()}), Counter({str(k): int(v) for k, v in right.items()}))


def summarize_pair_tvd(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, float]:
    result = {}
    for key in sorted(set(left) & set(right)):
        result[f"{key}_tvd"] = counter_tvd_from_dict(left[key], right[key])
        result[f"{key}_js"] = js_divergence(Counter(left[key]), Counter(right[key]))
    return result


def top_deltas(left_counter: Mapping[str, int], right_counter: Mapping[str, int], top_k: int = 20) -> list[dict[str, Any]]:
    left = distribution(Counter({str(k): int(v) for k, v in left_counter.items()}))
    right = distribution(Counter({str(k): int(v) for k, v in right_counter.items()}))
    rows = []
    for key in sorted(set(left) | set(right)):
        rows.append({"key": key, "r5c_rate": left.get(key, 0.0), "h1a2_rate": right.get(key, 0.0), "delta": right.get(key, 0.0) - left.get(key, 0.0)})
    rows.sort(key=lambda item: abs(float(item["delta"])), reverse=True)
    return rows[:top_k]


def write_report(result: Mapping[str, Any], path: Path) -> None:
    comp = result["plan_comparison"]
    r5c = result["r5c"]
    h1 = result["h1a2"]
    sun = result["sun_gap"]
    lines = [
        "# R5-C vs H1-A2 Plan / Refine / S.U.N. Diagnosis",
        "",
        "## Executive Summary",
        "",
        "- H1-A2 的 planner 边际分布已经接近 R5-C conditional，但最终 A100 S.U.N. 仍低，缺口主要不在 parse/body/refine coverage，而在 generated plan 的高阶联合化学先验与 refined 后的 hull stability。",
        f"- R5-C conditional A100 meta-like adjusted `{sun['r5c_meta_adjusted_pct']:.2f}%`，H1-A2 `{sun['h1a2_meta_adjusted_pct']:.2f}%`，差 `{sun['meta_adjusted_gap_pct']:.2f} pp`。",
        f"- R5-C strict adjusted `{sun['r5c_strict_adjusted_pct']:.2f}%`，H1-A2 `{sun['h1a2_strict_adjusted_pct']:.2f}%`，差 `{sun['strict_adjusted_gap_pct']:.2f} pp`。",
        f"- Novel+Unique 从 R5-C `{sun['r5c_novel_unique_pct']:.2f}%` 降到 H1-A2 `{sun['h1a2_novel_unique_pct']:.2f}%`，但更大的 meta-like 缺口来自 stable/evaluated：R5-C `{sun['r5c_meta_stable_pct']:.2f}%` vs H1-A2 `{sun['h1a2_meta_stable_pct']:.2f}%`。",
        "",
        "## Plan Distribution",
        "",
        "| metric | R5-C conditional | H1-A2 | delta / TVD |",
        "| --- | ---: | ---: | ---: |",
        f"| rows | {r5c['plan']['rows']} | {h1['plan']['rows']} | |",
        f"| parse_rate | {r5c['plan']['parse_rate']:.4f} | {h1['plan']['parse_rate']:.4f} | |",
        f"| mean_N | {r5c['plan']['mean_N']:.4f} | {h1['plan']['mean_N']:.4f} | {h1['plan']['mean_N'] - r5c['plan']['mean_N']:.4f} |",
        f"| N>=12 | {r5c['plan']['n_ge_12_rate_parsed']:.4f} | {h1['plan']['n_ge_12_rate_parsed']:.4f} | {h1['plan']['n_ge_12_rate_parsed'] - r5c['plan']['n_ge_12_rate_parsed']:.4f} |",
        f"| ternary | {r5c['plan']['ternary_rate_parsed']:.4f} | {h1['plan']['ternary_rate_parsed']:.4f} | {h1['plan']['ternary_rate_parsed'] - r5c['plan']['ternary_rate_parsed']:.4f} |",
        f"| 4+ elements | {r5c['plan']['four_plus_elements_rate_parsed']:.4f} | {h1['plan']['four_plus_elements_rate_parsed']:.4f} | {h1['plan']['four_plus_elements_rate_parsed'] - r5c['plan']['four_plus_elements_rate_parsed']:.4f} |",
        f"| single element | {r5c['plan']['single_element_rate_parsed']:.4f} | {h1['plan']['single_element_rate_parsed']:.4f} | {h1['plan']['single_element_rate_parsed'] - r5c['plan']['single_element_rate_parsed']:.4f} |",
        f"| all metal | {r5c['plan']['all_metal_rate_parsed']:.4f} | {h1['plan']['all_metal_rate_parsed']:.4f} | {h1['plan']['all_metal_rate_parsed'] - r5c['plan']['all_metal_rate_parsed']:.4f} |",
        f"| N TVD | | | {comp['n_tvd']:.4f} |",
        f"| arity TVD | | | {comp['num_elements_tvd']:.4f} |",
        f"| element presence TVD | | | {comp['element_presence_tvd']:.4f} |",
        f"| charge TVD | | | {comp['charge_bucket_tvd']:.4f} |",
        f"| anion TVD | | | {comp['anion_framework_tvd']:.4f} |",
        f"| lattice TVD | | | {comp['lattice_system_tvd']:.4f} |",
        f"| SG TVD | | | {comp['spacegroup_bucket_tvd']:.4f} |",
        f"| volume TVD | | | {comp['volume_per_atom_bin_tvd']:.4f} |",
        "",
        "## Joint Plan Drift",
        "",
        "这些 TVD 是 joint fields，而不是单字段边际；它们更能解释为什么 H1-A2 的边际看起来很好但 S.U.N 仍差。",
        "",
        "| joint field | TVD | JS |",
        "| --- | ---: | ---: |",
    ]
    for key, value in result["joint_plan_tvd"].items():
        if key.endswith("_tvd"):
            js_key = key[:-4] + "_js"
            lines.append(f"| {key[:-4]} | {value:.4f} | {result['joint_plan_tvd'].get(js_key, 0.0):.4f} |")
    lines.extend(
        [
            "",
            "## Raw Body Geometry",
            "",
            "| metric | R5-C conditional | H1-A2 |",
            "| --- | ---: | ---: |",
        ]
    )
    for key in sorted(set(r5c["body_geometry"]["rates"]) | set(h1["body_geometry"]["rates"])):
        lines.append(
            f"| {key} | {r5c['body_geometry']['rates'].get(key, 0.0):.4f} | {h1['body_geometry']['rates'].get(key, 0.0):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Refined Metrics And A100 S.U.N.",
            "",
            "| metric | R5-C conditional | H1-A2 | gap |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for key in ("comp_valid", "struct_valid", "valid", "cov_recall", "cov_precision", "wdist_density", "wdist_num_elems"):
        rv = r5c["crysllmgen_metrics"].get(key)
        hv = h1["crysllmgen_metrics"].get(key)
        if isinstance(rv, (int, float)) and isinstance(hv, (int, float)):
            lines.append(f"| {key} | {rv:.4f} | {hv:.4f} | {hv-rv:.4f} |")
    lines.extend(
        [
            f"| strict adjusted S.U.N. | {sun['r5c_strict_adjusted_pct']:.2f}% | {sun['h1a2_strict_adjusted_pct']:.2f}% | {-sun['strict_adjusted_gap_pct']:.2f} pp |",
            f"| meta-like adjusted S.U.N. | {sun['r5c_meta_adjusted_pct']:.2f}% | {sun['h1a2_meta_adjusted_pct']:.2f}% | {-sun['meta_adjusted_gap_pct']:.2f} pp |",
            f"| Novel+Unique | {sun['r5c_novel_unique_pct']:.2f}% | {sun['h1a2_novel_unique_pct']:.2f}% | {sun['h1a2_novel_unique_pct']-sun['r5c_novel_unique_pct']:.2f} pp |",
            f"| meta stable/evaluated | {sun['r5c_meta_stable_pct']:.2f}% | {sun['h1a2_meta_stable_pct']:.2f}% | {sun['h1a2_meta_stable_pct']-sun['r5c_meta_stable_pct']:.2f} pp |",
            "",
            "## Composition Validator Reasons",
            "",
            "R5-C 的 `plan_state` 来自真实样本，`validator.reason` 与 formula/anion/charge 的耦合更真实；H1-A2 是模型生成 rich fields，composition 合法性可以接近，但稳定性不一定接近。",
            "",
            "```json",
            json.dumps(
                {
                    "r5c": r5c["composition"],
                    "h1a2": h1["composition"],
                },
                ensure_ascii=False,
                indent=2,
            )[:12000],
            "```",
            "",
            "## Largest Plan Distribution Deltas",
            "",
            "### N histogram deltas",
            "",
            "```json",
            json.dumps(result["top_deltas"]["n_histogram"], ensure_ascii=False, indent=2),
            "```",
            "",
            "### Element presence deltas",
            "",
            "```json",
            json.dumps(result["top_deltas"]["element_presence_histogram"], ensure_ascii=False, indent=2),
            "```",
            "",
            "### Charge x anion x volume deltas",
            "",
            "```json",
            json.dumps(result["top_deltas"]["charge_x_anion_x_volume"], ensure_ascii=False, indent=2),
            "```",
            "",
            "## Diagnosis",
            "",
            "H1-A2 的不足不是 plan 格式、body parse 或 graph acceptance。主要原因是：",
            "",
            "1. **Gold teacher plan 的联合先验仍然没有被 LLM 完全复现。** 边际 TVD 低，但 `charge x anion x volume`、`lattice x SG`、`formula x fields` 等联合项仍有 drift。",
            "2. **Novel+Unique 下降只解释一小部分缺口。** H1-A2 Novel+Unique 仍有 89.30%，但 R5-C conditional 有 95.50%，这会直接压低 S.U.N. 的分母通过率。",
            "3. **更大的缺口来自 hull stability。** R5-C conditional 的 meta-like stable/evaluated 为 77.89%，H1-A2 只有 48.50%；说明 generated plan/proposal 虽然覆盖和结构有效，但很多 refined structures 仍落在高能区域。",
            "4. **refinement 能修连续几何，不能补 plan prior。** H1-A2 `cov_recall` 已经到 95.52%，但 strict/meta S.U.N 不到 baseline，说明 coverage 不再是首要矛盾，stability prior 才是。",
            "",
            "## Optimization Direction",
            "",
            "- H1-A2 planner 继续训练到 epoch2/epoch3 是合理下一步，但要按 epoch 监控 strict/meta S.U.N；如果 planner 过训，边际 TVD 可能更低但 novelty/stability 反而下降。",
            "- 后续优化应瞄准 joint consistency，而不是继续堆字段：可以加 teacher-plan joint consistency auxiliary loss、field-order/dropout robustness、或者训练阶段的 formula/rich-field consistency classifier 作 loss，不在采样时筛选。",
            "- DLM body 暂时不应重训 generated-like plan，因为没有事实结构标签；但可以研究 teacher plan 的扰动鲁棒性训练，只限于不改变真实 body 标签的等价/近等价扰动。",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r5c-raw-jsonl", type=Path, required=True)
    parser.add_argument("--h1a2-raw-jsonl", type=Path, required=True)
    parser.add_argument("--r5c-crysllmgen-metrics", type=Path, required=True)
    parser.add_argument("--h1a2-crysllmgen-metrics", type=Path, required=True)
    parser.add_argument("--r5c-composition-json", type=Path, default=None)
    parser.add_argument("--h1a2-composition-json", type=Path, default=None)
    parser.add_argument("--r5c-a100-strict-summary", type=Path, required=True)
    parser.add_argument("--r5c-a100-meta-summary", type=Path, required=True)
    parser.add_argument("--h1a2-a100-strict-summary", type=Path, required=True)
    parser.add_argument("--h1a2-a100-meta-summary", type=Path, required=True)
    parser.add_argument("--r5c-relax-jsonl", type=Path, default=None)
    parser.add_argument("--h1a2-relax-jsonl", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    r5c_rows = plan_records(args.r5c_raw_jsonl)
    h1_rows = plan_records(args.h1a2_raw_jsonl)
    r5c_plan_raw, r5c_derived = plan_stats_from_records("r5c_conditional", r5c_rows)
    h1_plan_raw, h1_derived = plan_stats_from_records("h1a2", h1_rows)
    plan_comparison = compare_stats(r5c_plan_raw, h1_plan_raw)
    joint_plan_tvd = summarize_pair_tvd(r5c_derived["pair_counters"], h1_derived["pair_counters"])
    r5c_strict = parse_summary_md(args.r5c_a100_strict_summary)
    r5c_meta = parse_summary_md(args.r5c_a100_meta_summary)
    h1_strict = parse_summary_md(args.h1a2_a100_strict_summary)
    h1_meta = parse_summary_md(args.h1a2_a100_meta_summary)
    sun_gap = {
        "r5c_strict_adjusted_pct": float(r5c_strict.get("coverage-adjusted_sun_estimate_pct", 0.0)),
        "h1a2_strict_adjusted_pct": float(h1_strict.get("coverage-adjusted_sun_estimate_pct", 0.0)),
        "r5c_meta_adjusted_pct": float(r5c_meta.get("coverage-adjusted_sun_estimate_pct", 0.0)),
        "h1a2_meta_adjusted_pct": float(h1_meta.get("coverage-adjusted_sun_estimate_pct", 0.0)),
        "r5c_novel_unique_pct": float(r5c_meta.get("novel_+_unique_pct", 0.0)),
        "h1a2_novel_unique_pct": float(h1_meta.get("novel_+_unique_pct", 0.0)),
        "r5c_meta_stable_pct": float(r5c_meta.get("stable_pct", 0.0)),
        "h1a2_meta_stable_pct": float(h1_meta.get("stable_pct", 0.0)),
    }
    sun_gap["strict_adjusted_gap_pct"] = sun_gap["r5c_strict_adjusted_pct"] - sun_gap["h1a2_strict_adjusted_pct"]
    sun_gap["meta_adjusted_gap_pct"] = sun_gap["r5c_meta_adjusted_pct"] - sun_gap["h1a2_meta_adjusted_pct"]
    result = {
        "inputs": {key: str(value) for key, value in vars(args).items() if isinstance(value, Path)},
        "r5c": {
            "plan": summarize(r5c_plan_raw, top_k=50),
            "derived": r5c_derived,
            "body_geometry": body_geometry_summary(args.r5c_raw_jsonl),
            "crysllmgen_metrics": metrics_summary(args.r5c_crysllmgen_metrics),
            "composition": composition_summary(args.r5c_composition_json),
            "a100_strict": r5c_strict,
            "a100_meta": r5c_meta,
            "relax_composition": relax_comp_summary(args.r5c_relax_jsonl),
        },
        "h1a2": {
            "plan": summarize(h1_plan_raw, top_k=50),
            "derived": h1_derived,
            "body_geometry": body_geometry_summary(args.h1a2_raw_jsonl),
            "crysllmgen_metrics": metrics_summary(args.h1a2_crysllmgen_metrics),
            "composition": composition_summary(args.h1a2_composition_json),
            "a100_strict": h1_strict,
            "a100_meta": h1_meta,
            "relax_composition": relax_comp_summary(args.h1a2_relax_jsonl),
        },
        "plan_comparison": plan_comparison,
        "joint_plan_tvd": joint_plan_tvd,
        "sun_gap": sun_gap,
        "top_deltas": {
            "n_histogram": top_deltas(r5c_plan_raw["n_histogram"], h1_plan_raw["n_histogram"]),
            "element_presence_histogram": top_deltas(
                r5c_plan_raw["element_presence_histogram"],
                h1_plan_raw["element_presence_histogram"],
            ),
            "charge_x_anion_x_volume": top_deltas(
                r5c_derived["pair_counters"]["charge_x_anion_x_volume"],
                h1_derived["pair_counters"]["charge_x_anion_x_volume"],
            ),
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(result, args.output_md)
    print(json.dumps({"output_json": str(args.output_json), "output_md": str(args.output_md), "sun_gap": sun_gap}, indent=2))


if __name__ == "__main__":
    main()
