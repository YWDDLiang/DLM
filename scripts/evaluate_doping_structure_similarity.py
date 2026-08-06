#!/usr/bin/env python3
"""Evaluate generated full80 structures against hidden verified-good structures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from crystal_dlm.doping import (
    combo_from_name,
    combo_name,
    load_first_full80_template,
    read_jsonl,
    reconstruct_arrays_from_template,
    write_json,
)
from crystal_dlm.doping_structure import (
    bsite_dopants_from_arrays,
    dopants_from_arrays,
    full80_composition_is_exact,
    near_hit,
    parse_full80_answer,
)


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_generated_arrays(output_dir: Path) -> List[Dict[str, Any]]:
    rows = []
    for row in read_jsonl(output_dir / "expanded_arrays.jsonl"):
        arrays = row.get("expanded_arrays")
        if arrays:
            rows.append({**row, "expanded_arrays": arrays})
    return rows


def load_hidden_targets(data_dir: Path) -> Dict[str, Dict[str, Any]]:
    path = data_dir / "hidden_good_holdout.jsonl"
    if not path.exists():
        raise RuntimeError(f"Missing hidden-good file: {path}")
    hidden = {}
    for row in read_jsonl(path):
        name = row["name"]
        answer = row.get("full80_answer") or row.get("answer")
        if not answer:
            raise RuntimeError(f"Hidden row {name} does not contain full80 answer")
        hidden[name] = {
            "name": name,
            "dopants": row["dopants"],
            "properties": row["properties"],
            "arrays": parse_full80_answer(answer, strict=True),
        }
    return hidden


def best_hidden_matches(
    generated_rows: Iterable[Mapping[str, Any]],
    hidden: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    best: Dict[str, Dict[str, Any]] = {}
    for hidden_name, target in hidden.items():
        best[hidden_name] = {
            "hidden_name": hidden_name,
            "hidden_dopants": target["dopants"],
            "same_element_count": 0,
            "near_hit_count": 0,
            "best_bsite_rmsd": None,
            "best_length_rel_mean": None,
            "best_angle_abs_mean": None,
            "best_sample_idx": None,
            "best_generated_name": None,
            "structure_matcher_pass": None,
        }
    for row in generated_rows:
        arrays = row["expanded_arrays"]
        try:
            generated_combo = sorted(dopants_from_arrays(arrays))
        except Exception:
            continue
        for hidden_name, target in hidden.items():
            if generated_combo != sorted(target["dopants"]):
                continue
            match = near_hit(arrays, target["arrays"])
            entry = best[hidden_name]
            entry["same_element_count"] += 1
            if match["near_hit"]:
                entry["near_hit_count"] += 1
            current_best = entry["best_bsite_rmsd"]
            if current_best is None or match["bsite_rmsd"] < current_best:
                entry.update(
                    {
                        "best_bsite_rmsd": match["bsite_rmsd"],
                        "best_length_rel_mean": match["length_rel_mean"],
                        "best_angle_abs_mean": match["angle_abs_mean"],
                        "best_sample_idx": row.get("sample_idx"),
                        "best_generated_name": row.get("name"),
                        "structure_matcher_pass": match["structure_matcher_pass"],
                    }
                )
    return best


def load_compact_baseline_arrays(compact_output_dir: Path | None, full80_jsonl: Path) -> List[Dict[str, Any]]:
    if compact_output_dir is None or not compact_output_dir.exists():
        return []
    raw_path = compact_output_dir / "raw_generations.jsonl"
    if not raw_path.exists():
        return []
    template = load_first_full80_template(full80_jsonl)
    rows = []
    seen = set()
    for raw in read_jsonl(raw_path):
        name = raw.get("name")
        if not raw.get("parsed") or not name or name in seen:
            continue
        seen.add(name)
        try:
            arrays = reconstruct_arrays_from_template(template, combo_from_name(name))
            rows.append({"sample_idx": raw.get("sample_idx"), "name": name, "expanded_arrays": arrays})
        except Exception:
            continue
    return rows


def summarize_matches(matches: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    hidden_count = len(matches)
    same_element_names = [name for name, row in matches.items() if int(row["same_element_count"]) > 0]
    near_hit_names = [name for name, row in matches.items() if int(row["near_hit_count"]) > 0]
    return {
        "hidden_count": hidden_count,
        "same_element_hidden_count": len(same_element_names),
        "same_element_hidden_names": same_element_names,
        "structural_near_hit_count": len(near_hit_names),
        "structural_near_hit_names": near_hit_names,
        "best_matches": list(matches.values()),
    }


def make_markdown(metrics: Mapping[str, Any]) -> str:
    lines = [
        "# Structure-Aware Doping Similarity Evaluation",
        "",
        "## 结论",
        "",
        f"- offline_preliminary_pass：{metrics['offline_preliminary_pass']}",
        f"- claim_status：{metrics['claim_status']}",
        f"- generated structural near-hit：{metrics['generated_summary']['structural_near_hit_count']} / {metrics['generated_summary']['hidden_count']}",
        f"- compact-template baseline near-hit：{metrics['compact_baseline_summary']['structural_near_hit_count']} / {metrics['compact_baseline_summary']['hidden_count']}",
        "",
        "说明：本评估没有 DFT relaxation，因此不能证明性质更好，只能证明结构相似或结构生成能力。",
        "",
        "## Hidden-Good Best Matches",
        "",
    ]
    for row in metrics["generated_summary"]["best_matches"]:
        lines.append(
            "- `{}` same_element_count={} near_hit_count={} best_rmsd={} best_length_rel={} best_angle_abs={} sample={}".format(
                row["hidden_name"],
                row["same_element_count"],
                row["near_hit_count"],
                row["best_bsite_rmsd"],
                row["best_length_rel_mean"],
                row["best_angle_abs_mean"],
                row["best_sample_idx"],
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-output-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--full80-jsonl", type=Path, default=PROJECT_ROOT / "data/doping_crystal/full80_success.jsonl")
    parser.add_argument(
        "--compact-baseline-output-dir",
        type=Path,
        default=PROJECT_ROOT / "runs/20260519_210000-doping-good-holdout/outputs/compact_sample_goodholdout_2048",
    )
    args = parser.parse_args()

    generated = load_generated_arrays(args.sample_output_dir)
    hidden = load_hidden_targets(args.data_dir)
    generated_matches = best_hidden_matches(generated, hidden)
    baseline_rows = load_compact_baseline_arrays(args.compact_baseline_output_dir, args.full80_jsonl)
    baseline_matches = best_hidden_matches(baseline_rows, hidden)

    composition_exact_count = 0
    for row in generated:
        try:
            composition_exact_count += int(full80_composition_is_exact(row["expanded_arrays"]))
        except Exception:
            pass
    generated_summary = summarize_matches(generated_matches)
    baseline_summary = summarize_matches(baseline_matches)
    offline_pass = (
        generated_summary["structural_near_hit_count"] >= 1
        and generated_summary["structural_near_hit_count"] > baseline_summary["structural_near_hit_count"]
    )
    metrics = {
        "sample_output_dir": str(args.sample_output_dir),
        "data_dir": str(args.data_dir),
        "generated_count": len(generated),
        "generated_composition_exact_count": composition_exact_count,
        "generated_composition_exact_rate": composition_exact_count / max(1, len(generated)),
        "generated_unique_combo_count": len({combo_name(bsite_dopants_from_arrays(row["expanded_arrays"])) for row in generated}),
        "generated_summary": generated_summary,
        "compact_baseline_count": len(baseline_rows),
        "compact_baseline_summary": baseline_summary,
        "offline_preliminary_pass": offline_pass,
        "claim_status": (
            "可写：模型生成了 verified-good-like 的完整 80 原子结构初步证据。"
            if offline_pass
            else "不可写：尚未超过 compact-template baseline，不能声称结构生成更好。"
        ),
        "final_property_claim_allowed": False,
        "final_property_claim_reason": "未做统一 DFT relaxation，不能声称生成结构性质更好。",
    }
    write_json(args.sample_output_dir / "structure_similarity_eval.json", metrics)
    (args.sample_output_dir / "structure_similarity_eval.md").write_text(make_markdown(metrics), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
