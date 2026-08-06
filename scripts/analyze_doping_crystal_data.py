#!/usr/bin/env python3
"""Analyze CsPbI3 doped-crystal data and build small feasibility artifacts."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from crystal_dlm.fixed_slot import FixedSlotConfig, arrays_to_answer, build_special_tokens, write_json

DOPANT_SYMBOLS = ["Al", "Ba", "Ca", "Cd", "Co", "Cu", "Fe", "In", "Mg", "Ni", "Sn", "Sr", "Zn"]
DOPANT_LOWER = {item.lower(): item for item in DOPANT_SYMBOLS}
PROPERTY_KEYS = ["formation_energy", "defect_formation_energy", "band_gap", "VBM", "CBM", "fermi"]


def parse_float(text: str) -> float:
    return float(str(text).strip().strip('"'))


def normalize_combo(items: Iterable[str]) -> str:
    return "_".join(sorted(item.lower() for item in items))


def combo_from_cif_name(path: Path) -> str:
    tokens = re.findall(r"([A-Za-z]{1,2})0\.05", path.stem)
    dopants = [DOPANT_LOWER[token.lower()] for token in tokens if token.lower() in DOPANT_LOWER]
    return normalize_combo(dopants)


def read_cal_records(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        record["line_no"] = line_no
        record["combo_key"] = normalize_combo(record.get("name", "").split("_"))
        for key in PROPERTY_KEYS:
            if key in record:
                record[key] = float(record[key])
        records.append(record)
    return records


def parse_cif(path: Path) -> Dict[str, Any]:
    lengths: Dict[str, float] = {}
    angles: Dict[str, float] = {}
    formula_sum = ""
    sites: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if stripped.startswith("_chemical_formula_sum"):
            formula_sum = stripped.split(None, 1)[1].strip().strip('"')
        elif stripped.startswith("_cell_length_"):
            key, value = stripped.split(None, 1)
            lengths[key.rsplit("_", 1)[-1]] = parse_float(value)
        elif stripped.startswith("_cell_angle_"):
            key, value = stripped.split(None, 1)
            angles[key.rsplit("_", 1)[-1]] = parse_float(value)
        else:
            parts = stripped.split()
            if len(parts) >= 7 and re.fullmatch(r"[A-Z][a-z]?", parts[0]) and re.match(r"^[A-Za-z]+\d+", parts[1]):
                sites.append(
                    {
                        "species": parts[0],
                        "label": parts[1],
                        "frac_coords": [float(parts[3]), float(parts[4]), float(parts[5])],
                        "occupancy": float(parts[6]),
                    }
                )
    return {
        "path": str(path),
        "file": path.name,
        "combo_key": combo_from_cif_name(path),
        "formula_sum": formula_sum,
        "lengths": [lengths["a"], lengths["b"], lengths["c"]],
        "angles": [angles["alpha"], angles["beta"], angles["gamma"]],
        "sites": sites,
    }


def periodic_distance(a: Sequence[float], b: Sequence[float]) -> float:
    sq = 0.0
    for x, y in zip(a, b):
        delta = abs(float(x) - float(y))
        delta = min(delta, 1.0 - delta)
        sq += delta * delta
    return math.sqrt(sq)


def b_sites(cif: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [site for site in cif["sites"] if site["species"] not in {"Cs", "I"}]


def assign_b_sites(reference_sites: Sequence[Dict[str, Any]], candidate_sites: Sequence[Dict[str, Any]]) -> Tuple[List[int], float]:
    unused = set(range(len(reference_sites)))
    assignment: List[int] = []
    max_distance = 0.0
    for site in candidate_sites:
        best_idx = min(
            unused,
            key=lambda idx: periodic_distance(site["frac_coords"], reference_sites[idx]["frac_coords"]),
        )
        distance = periodic_distance(site["frac_coords"], reference_sites[best_idx]["frac_coords"])
        max_distance = max(max_distance, distance)
        assignment.append(best_idx)
        unused.remove(best_idx)
    return assignment, max_distance


def summarize(values: Sequence[float]) -> Dict[str, float]:
    sorted_values = sorted(float(v) for v in values)
    if not sorted_values:
        return {}
    return {
        "min": sorted_values[0],
        "p25": sorted_values[len(sorted_values) // 4],
        "median": statistics.median(sorted_values),
        "mean": statistics.mean(sorted_values),
        "p75": sorted_values[(3 * len(sorted_values)) // 4],
        "max": sorted_values[-1],
    }


def nearest_neighbor_predict(train_rows: Sequence[Dict[str, Any]], combo: Sequence[str], key: str, k: int = 5) -> float:
    combo_set = set(combo)
    scored = []
    for row in train_rows:
        train_set = set(row["dopants"])
        overlap = len(combo_set & train_set)
        scored.append((-overlap, len(combo_set ^ train_set), row[key]))
    scored.sort()
    return statistics.mean(item[2] for item in scored[:k])


def run_surrogate(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    evaluated = []
    for idx, row in enumerate(rows):
        train = [item for j, item in enumerate(rows) if j != idx]
        pred = {
            "band_gap": nearest_neighbor_predict(train, row["dopants"], "band_gap"),
            "formation_energy": nearest_neighbor_predict(train, row["dopants"], "formation_energy"),
            "defect_formation_energy": nearest_neighbor_predict(train, row["dopants"], "defect_formation_energy"),
        }
        actual_target = 1.7 <= row["band_gap"] <= 2.7
        pred_target = 1.7 <= pred["band_gap"] <= 2.7
        evaluated.append(
            {
                "name": row["name"],
                "dopants": row["dopants"],
                "actual": {key: row[key] for key in ("band_gap", "formation_energy", "defect_formation_energy")},
                "pred": pred,
                "actual_target_band_gap": actual_target,
                "pred_target_band_gap": pred_target,
                "pred_objective": (
                    int(pred_target),
                    -pred["formation_energy"],
                    pred["defect_formation_energy"],
                ),
            }
        )
    tp = sum(1 for row in evaluated if row["actual_target_band_gap"] and row["pred_target_band_gap"])
    fp = sum(1 for row in evaluated if (not row["actual_target_band_gap"]) and row["pred_target_band_gap"])
    tn = sum(1 for row in evaluated if (not row["actual_target_band_gap"]) and (not row["pred_target_band_gap"]))
    fn = sum(1 for row in evaluated if row["actual_target_band_gap"] and (not row["pred_target_band_gap"]))
    top20 = sorted(evaluated, key=lambda row: row["pred_objective"], reverse=True)[:20]
    top20_actual = [row["actual"] for row in top20]
    return {
        "method": "leave-one-out 5-nearest-combo baseline",
        "target_rule": "band_gap in [1.7, 2.7], low formation_energy, high defect_formation_energy",
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "accuracy": (tp + tn) / max(1, len(evaluated)),
        "precision": tp / max(1, tp + fp),
        "recall": tp / max(1, tp + fn),
        "base_target_rate": sum(row["actual_target_band_gap"] for row in evaluated) / max(1, len(evaluated)),
        "top20_target_rate": sum(row["actual_target_band_gap"] for row in top20) / max(1, len(top20)),
        "top20_actual_summary": {
            "band_gap": summarize([row["band_gap"] for row in top20_actual]),
            "formation_energy": summarize([row["formation_energy"] for row in top20_actual]),
            "defect_formation_energy": summarize([row["defect_formation_energy"] for row in top20_actual]),
        },
        "top20": top20,
    }


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def make_report(stats: Dict[str, Any], surrogate: Dict[str, Any]) -> str:
    top_targets = stats["top_target_low_formation_high_defect"][:12]
    lines = [
        "# CsPbI3 掺杂数据可行性分析",
        "",
        "## 数据概况",
        "",
        f"- 数据根目录：`{stats['data_root']}`",
        f"- CIF 文件数：{stats['cif_count']}",
        f"- `cal_data` 记录数：{stats['cal_record_count']}",
        f"- `SUCCESS` 记录数：{stats['state_counts'].get('SUCCESS', 0)}",
        f"- 可匹配 `SUCCESS + CIF`：{stats['matched_success_count']}",
        f"- 所有 matched CIF atom count：{stats['atom_count_histogram']}",
        f"- 目标 band gap `[1.7, 2.7] eV` 命中数：{stats['target_band_gap_count']}",
        "",
        "## 与 MP-20 fixed-slot 的差异",
        "",
        "- 当前 MP-20 schema 默认 `max_atoms=20`，answer 长度为 107。",
        "- 掺杂数据全部为 80 atoms，需要独立 `FixedSlotConfig(max_atoms=80)`，answer 长度为 407。",
        "- 掺杂数据不是一般晶体分布，而是同一 Cs-Pb-I 钙钛矿超胞上的组合掺杂；直接用无条件 MP-20 生成目标不合适。",
        "- 更合理的主路线是模板化掺杂编辑：模型先学 dopant identities / B-site indices / property condition，再由模板重构完整结构。",
        "",
        "## 结构族与 B-site 验证",
        "",
        f"- 期望结构族：`Cs16 I48 Pb12 Mn1 D1 D2 D3`",
        f"- B-site 验证成功：{stats['bsite_valid_count']} / {stats['matched_success_count']}",
        f"- compact reconstruction rate：{stats['compact_reconstruction_rate']:.4f}",
        f"- B-site alignment max distance 最大值：{stats['bsite_alignment_max_distance_summary'].get('max')}",
        "",
        "## 性质分布",
        "",
    ]
    for key, summary in stats["property_summary_all"].items():
        lines.append(f"- `{key}` all: {summary}")
    for key, summary in stats["property_summary_target_band_gap"].items():
        lines.append(f"- `{key}` target bandgap subset: {summary}")
    lines.extend(
        [
            "",
            "## 小规模 surrogate/ranker 结果",
            "",
            f"- 方法：{surrogate['method']}",
            f"- base target rate：{surrogate['base_target_rate']:.4f}",
            f"- LOO accuracy：{surrogate['accuracy']:.4f}",
            f"- precision：{surrogate['precision']:.4f}",
            f"- recall：{surrogate['recall']:.4f}",
            f"- top20 target rate：{surrogate['top20_target_rate']:.4f}",
            f"- top20 actual summary：{surrogate['top20_actual_summary']}",
            "",
            "## 推荐候选组合",
            "",
        ]
    )
    for row in top_targets:
        lines.append(
            f"- `{row['name']}`: band_gap={row['band_gap']:.4f}, "
            f"formation_energy={row['formation_energy']:.4f}, "
            f"defect_formation_energy={row['defect_formation_energy']:.4f}"
        )
    lines.extend(
        [
            "",
            "## 结论",
            "",
            "- Task 2 主线应先做 compact doping edit，而不是直接让模型生成完整 80 原子结构。",
            "- 完整 80-slot 适合作为 feasibility 对照：验证 tokenizer、round-trip 和 tiny decode 是否可行。",
            "- 当前小数据只有 225 条成功标签，足够做组合 ranker 和小规模 SFT smoke，不足以支撑直接大规模 inverse design。",
            "- 目标排序应固定为：先满足 `band_gap ∈ [1.7, 2.7] eV`，再降低 `formation_energy`，最后提高 `defect_formation_energy`。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("/zhdd/home/hengzhang/data/crystal"))
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data/doping_crystal")
    parser.add_argument("--report-path", type=Path, default=PROJECT_ROOT / "reports/20260519_doping_data_feasibility_report.md")
    parser.add_argument("--answer-separator", default="")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report_path.parent.mkdir(parents=True, exist_ok=True)

    cal_records = read_cal_records(args.data_root / "cal_data (1).txt")
    cif_by_combo = {combo_from_cif_name(path): parse_cif(path) for path in sorted((args.data_root / "cif_data").glob("*.cif"))}
    success_records = [record for record in cal_records if record.get("state") == "SUCCESS"]
    matched_success = [record for record in success_records if record["combo_key"] in cif_by_combo]
    target_records = [record for record in matched_success if 1.7 <= record["band_gap"] <= 2.7]

    reference_cif = cif_by_combo[matched_success[0]["combo_key"]]
    reference_b_sites = b_sites(reference_cif)
    compact_rows: List[Dict[str, Any]] = []
    full80_rows: List[Dict[str, Any]] = []
    bsite_valid_count = 0
    alignment_distances: List[float] = []
    atom_count_hist = Counter()
    element_hist = Counter()

    config80 = FixedSlotConfig(max_atoms=80)
    for record in matched_success:
        cif = cif_by_combo[record["combo_key"]]
        sites = cif["sites"]
        atom_count_hist[str(len(sites))] += 1
        element_hist.update(site["species"] for site in sites)
        current_b_sites = b_sites(cif)
        bsite_counts = Counter(site["species"] for site in current_b_sites)
        bsite_valid = (
            len(sites) == 80
            and len(current_b_sites) == 16
            and bsite_counts["Pb"] == 12
            and bsite_counts["Mn"] == 1
            and sum(bsite_counts[element] for element in DOPANT_SYMBOLS) == 3
        )
        assignment: List[int] = []
        max_distance = None
        dopant_site_indices: List[int] = []
        if bsite_valid:
            assignment, max_distance = assign_b_sites(reference_b_sites, current_b_sites)
            alignment_distances.append(max_distance)
            bsite_valid_count += 1
            for site, ref_idx in zip(current_b_sites, assignment):
                if site["species"] in DOPANT_SYMBOLS:
                    dopant_site_indices.append(ref_idx)
        dopants = [DOPANT_LOWER[item] for item in record["combo_key"].split("_") if item in DOPANT_LOWER]
        properties = {key: record[key] for key in ("band_gap", "formation_energy", "defect_formation_energy", "VBM", "CBM", "fermi")}
        compact_rows.append(
            {
                "task": "doping_compact",
                "name": record["name"],
                "dopants": dopants,
                "dopant_site_indices": sorted(dopant_site_indices),
                "target": {
                    "band_gap_range_eV": [1.7, 2.7],
                    "formation_energy": "lower_is_better",
                    "defect_formation_energy": "higher_is_better",
                },
                "answer": " ".join(
                    [f"<DOPANT_{item}>" for item in dopants]
                    + [f"<B{idx:02d}>" for idx in sorted(dopant_site_indices)]
                ),
                "properties": properties,
                "target_band_gap": 1.7 <= record["band_gap"] <= 2.7,
                "bsite_valid": bsite_valid,
                "bsite_alignment_max_distance": max_distance,
                "source_cif": cif["file"],
            }
        )
        species = [site["species"] for site in sites]
        frac_coords = [site["frac_coords"] for site in sites]
        answer, diagnostics = arrays_to_answer(
            lengths=cif["lengths"],
            angles=cif["angles"],
            species=species,
            frac_coords=frac_coords,
            config=config80,
            separator=args.answer_separator,
        )
        full80_rows.append(
            {
                "task": "doping_full80",
                "prompt": (
                    "Generate the 80-slot doped CsPbI3 structure with band_gap in [1.7, 2.7] eV, "
                    "low formation_energy, and high defect_formation_energy:"
                ),
                "answer": answer,
                "answer_semantic_tokens": 407,
                "metadata": {
                    "name": record["name"],
                    "dopants": dopants,
                    "properties": properties,
                    "source_cif": cif["file"],
                    "diagnostics": diagnostics.to_dict(),
                },
            }
        )

    surrogate_rows = []
    compact_by_name = {row["name"]: row for row in compact_rows}
    for record in matched_success:
        row = dict(record)
        row["dopants"] = compact_by_name[record["name"]]["dopants"]
        surrogate_rows.append(row)
    surrogate = run_surrogate(surrogate_rows)

    stats = {
        "data_root": str(args.data_root),
        "cal_record_count": len(cal_records),
        "state_counts": dict(Counter(record.get("state") for record in cal_records)),
        "cif_count": len(cif_by_combo),
        "matched_success_count": len(matched_success),
        "target_band_gap_count": len(target_records),
        "atom_count_histogram": dict(atom_count_hist),
        "element_histogram": dict(element_hist),
        "bsite_valid_count": bsite_valid_count,
        "compact_reconstruction_rate": bsite_valid_count / max(1, len(matched_success)),
        "bsite_alignment_max_distance_summary": summarize(alignment_distances),
        "property_summary_all": {
            key: summarize([record[key] for record in matched_success])
            for key in ("band_gap", "formation_energy", "defect_formation_energy")
        },
        "property_summary_target_band_gap": {
            key: summarize([record[key] for record in target_records])
            for key in ("band_gap", "formation_energy", "defect_formation_energy")
        },
        "dopant_frequency_all": dict(Counter(item for row in compact_rows for item in row["dopants"])),
        "dopant_frequency_target_band_gap": dict(Counter(item for row in compact_rows if row["target_band_gap"] for item in row["dopants"])),
        "top_target_low_formation_high_defect": sorted(
            [
                {
                    "name": record["name"],
                    "band_gap": record["band_gap"],
                    "formation_energy": record["formation_energy"],
                    "defect_formation_energy": record["defect_formation_energy"],
                }
                for record in target_records
            ],
            key=lambda row: (row["formation_energy"], -row["defect_formation_energy"]),
        ),
        "full80": {
            "max_atoms": 80,
            "answer_token_count": 407,
            "special_token_count": len(build_special_tokens(config80)),
        },
    }

    write_json(str(args.output_dir / "analysis_stats.json"), stats)
    write_json(str(args.output_dir / "surrogate_report.json"), surrogate)
    write_jsonl(args.output_dir / "compact_records.jsonl", compact_rows)
    write_jsonl(args.output_dir / "full80_success.jsonl", full80_rows)
    with (args.output_dir / "vocab_tokens_80.txt").open("w", encoding="utf-8") as handle:
        for token in build_special_tokens(config80):
            handle.write(token + "\n")
    args.report_path.write_text(make_report(stats, surrogate), encoding="utf-8")
    print(json.dumps({"stats": stats, "surrogate": {k: v for k, v in surrogate.items() if k != "top20"}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
