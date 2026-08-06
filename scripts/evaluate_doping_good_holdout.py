#!/usr/bin/env python3
"""Evaluate whether compact sampling recovers hidden high-quality doping rows."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from crystal_dlm.doping import combo_name, read_jsonl, write_json


def load_json(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def is_target(properties: Mapping[str, Any]) -> bool:
    return 1.7 <= float(properties["band_gap"]) <= 2.7


def is_balanced_good(properties: Mapping[str, Any], definition: Mapping[str, Any]) -> bool:
    return (
        is_target(properties)
        and float(properties["formation_energy"]) <= float(definition["formation_energy_max"])
        and float(properties["defect_formation_energy"]) >= float(definition["defect_formation_energy_min"])
    )


def mean(values: Iterable[float]) -> float | None:
    values = list(values)
    if not values:
        return None
    return sum(values) / len(values)


def summarize_true(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    true_rows = [row for row in rows if row.get("true_properties")]
    if not true_rows:
        return {
            "count": 0,
            "bandgap_hit_rate": None,
            "mean_band_gap": None,
            "mean_formation_energy": None,
            "mean_defect_formation_energy": None,
        }
    return {
        "count": len(true_rows),
        "bandgap_hit_rate": mean(1.0 if is_target(row["true_properties"]) else 0.0 for row in true_rows),
        "mean_band_gap": mean(float(row["true_properties"]["band_gap"]) for row in true_rows),
        "mean_formation_energy": mean(float(row["true_properties"]["formation_energy"]) for row in true_rows),
        "mean_defect_formation_energy": mean(
            float(row["true_properties"]["defect_formation_energy"]) for row in true_rows
        ),
    }


def make_markdown(metrics: Mapping[str, Any]) -> str:
    lines = [
        "# Good Holdout Doping Evaluation",
        "",
        "## 结论",
        "",
        f"- hidden good holdout 命中：{metrics['hidden_good_generated_count']} / {metrics['hidden_good_count']}",
        f"- generated balanced-good unique combos：{metrics['generated_balanced_good_count']}",
        f"- top20 bandgap hit rate：{metrics['top20_true_summary']['bandgap_hit_rate']}",
        f"- top20 mean formation_energy：{metrics['top20_true_summary']['mean_formation_energy']}",
        f"- top20 mean defect_formation_energy：{metrics['top20_true_summary']['mean_defect_formation_energy']}",
        "",
        "## Hidden Good Holdout",
        "",
    ]
    for row in metrics["hidden_good_holdout"]:
        marker = "HIT" if row["generated_count"] else "MISS"
        lines.append(
            "- `{}` {} count={} bg={} fe={} dfe={}".format(
                row["name"],
                marker,
                row["generated_count"],
                row["band_gap"],
                row["formation_energy"],
                row["defect_formation_energy"],
            )
        )
    lines.extend(["", "## Top Generated Candidates", ""])
    for row in metrics["top20_candidates"]:
        lines.append(
            "- `{}` bg={} fe={} dfe={} balanced_good={}".format(
                row["name"],
                row.get("band_gap"),
                row.get("formation_energy"),
                row.get("defect_formation_energy"),
                row.get("balanced_good"),
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-output-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data/doping_compact_good_holdout")
    parser.add_argument("--compact-jsonl", type=Path, default=PROJECT_ROOT / "data/doping_crystal/compact_records.jsonl")
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    stats = load_json(args.data_dir / "stats.json")
    good_definition = stats["good_definition"]
    known_map = {combo_name(row["dopants"]): row for row in read_jsonl(args.compact_jsonl)}
    raw_rows = read_jsonl(args.sample_output_dir / "raw_generations.jsonl")
    sampled_counts = Counter(row["name"] for row in raw_rows if row.get("parsed") and row.get("name"))
    candidates_payload = load_json(args.sample_output_dir / "doping_design" / "candidates.json")
    candidates = candidates_payload["candidates"]

    hidden_rows = []
    for row in stats["hidden_good_holdout"]:
        name = row["name"]
        hidden_rows.append({**row, "generated_count": int(sampled_counts.get(name, 0))})

    generated_unique_names = set(sampled_counts)
    generated_known = [known_map[name] for name in generated_unique_names if name in known_map]
    generated_balanced_good = [
        row for row in generated_known if is_balanced_good(row["properties"], good_definition)
    ]
    top_candidates = []
    for row in candidates[: args.top_k]:
        true = row.get("true_properties") or {}
        top_candidates.append(
            {
                "name": row["name"],
                "band_gap": true.get("band_gap"),
                "formation_energy": true.get("formation_energy"),
                "defect_formation_energy": true.get("defect_formation_energy"),
                "balanced_good": bool(true) and is_balanced_good(true, good_definition),
                "generated_count": int(sampled_counts.get(row["name"], 0)),
            }
        )

    metrics = {
        "sample_output_dir": str(args.sample_output_dir),
        "data_dir": str(args.data_dir),
        "hidden_good_count": len(hidden_rows),
        "hidden_good_generated_count": sum(1 for row in hidden_rows if row["generated_count"] > 0),
        "hidden_good_generated_names": [row["name"] for row in hidden_rows if row["generated_count"] > 0],
        "hidden_good_holdout": hidden_rows,
        "generated_unique_count": len(generated_unique_names),
        "generated_known_count": len(generated_known),
        "generated_balanced_good_count": len(generated_balanced_good),
        "generated_balanced_good_names": sorted(row["name"] for row in generated_balanced_good),
        "top20_true_summary": summarize_true(candidates[: args.top_k]),
        "top20_balanced_good_count": sum(1 for row in top_candidates if row["balanced_good"]),
        "top20_candidates": top_candidates,
        "preliminary_validation_pass": (
            sum(1 for row in hidden_rows if row["generated_count"] > 0) >= 1
            and len(generated_balanced_good) >= len(hidden_rows)
            and (summarize_true(candidates[: args.top_k])["bandgap_hit_rate"] or 0.0) >= 0.8
        ),
    }

    write_json(args.sample_output_dir / "good_holdout_eval.json", metrics)
    (args.sample_output_dir / "good_holdout_eval.md").write_text(make_markdown(metrics), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
