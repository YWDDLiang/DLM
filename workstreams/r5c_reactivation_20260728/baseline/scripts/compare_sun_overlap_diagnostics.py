#!/usr/bin/env python3
"""Compare SUN overlap, high-symmetry, and concentration diagnostics across runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from crystal_dlm.fixed_slot import write_json  # noqa: E402


def read_json(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def parse_run_spec(spec: str) -> Tuple[str, Path]:
    if "=" in spec:
        name, path = spec.split("=", 1)
        return name.strip(), Path(path.strip())
    path = Path(spec)
    return path.name, path


def notes_dir(run_dir: Path) -> Path:
    if (run_dir / "notes").exists():
        return run_dir / "notes"
    return run_dir


def top_concentration(counter: Dict[str, Any] | None, denominator: int, top_k: int) -> float | None:
    if not counter or denominator <= 0:
        return None
    values = sorted((int(value) for value in counter.values()), reverse=True)
    return sum(values[:top_k]) / float(denominator)


def extract_crysllmgen_metrics(payload: Dict[str, Any] | None) -> Dict[str, Any]:
    if not payload:
        return {}
    metrics = payload.get("metrics", payload)

    def pick(*keys):
        for key in keys:
            if key in metrics:
                return metrics[key]
        return None

    return {
        "comp_valid": pick("comp_valid", "avg_comp_validity"),
        "struct_valid": pick("struct_valid", "avg_structure_validity"),
        "valid": pick("valid"),
        "wdist_density": pick("wdist_density"),
        "wdist_num_elems": pick("wdist_num_elems"),
        "cov_recall": pick("cov_recall", "coverage_recall"),
        "cov_precision": pick("cov_precision", "coverage_precision"),
    }


def extract_composition_metrics(payload: Dict[str, Any] | None) -> Dict[str, Any]:
    if not payload:
        return {}
    raw = payload.get("raw_jsonl") or payload.get("refined_pt") or payload
    reason_counts = raw.get("reason_counts", {})
    count = int(raw.get("count") or raw.get("total") or 0)
    strict = int(reason_counts.get("charge_neutral_pauling_valid", 0))
    single = int(reason_counts.get("single_element_shortcut", 0))
    all_metal = int(reason_counts.get("all_metal_shortcut", 0))
    return {
        "raw_comp_valid": raw.get("comp_valid_rate"),
        "strict_valid": strict / count if count else None,
        "single_element": single / count if count else None,
        "all_metal": all_metal / count if count else None,
        "pbc_duplicate": raw.get("pbc_equivalent_duplicate_fraction"),
    }


def extract_distribution_metrics(payload: Dict[str, Any] | None) -> Dict[str, Any]:
    if not payload:
        return {}
    total = int(payload.get("total") or payload.get("raw_record_count") or 0)
    all_lengths_equal = int(payload.get("records_all_lengths_equal") or 0)
    all_angles_90 = int(payload.get("records_all_angles_90") or 0)
    return {
        "distribution_total": total,
        "a_eq_b_eq_c": all_lengths_equal / total if total else None,
        "all_90": all_angles_90 / total if total else None,
        "high_sym_coord_mean": payload.get("high_symmetry_coord_fraction_mean"),
        "duplicate_pbc_site_failures": payload.get("duplicate_pbc_site_failures"),
    }


def extract_sun_metrics(payload: Dict[str, Any] | None) -> Dict[str, Any]:
    if not payload:
        return {}
    counts = payload.get("counts") or payload.get("sun_thresholds", {}).get("counts", {})
    rates = payload.get("rates_submitted") or payload.get("sun_thresholds", {}).get("rates", {})
    if not rates and "metrics" in payload:
        metrics = payload.get("metrics") or {}
        rates = {
            "meta_stable": metrics.get("frac_stable_structures"),
            "novel": metrics.get("frac_novel_structures"),
            "unique": metrics.get("frac_unique_structures"),
            "novel_unique": metrics.get("frac_novel_unique_structures"),
            "meta_sun": metrics.get("frac_novel_unique_stable_structures"),
        }
    submitted = int(counts.get("submitted") or payload.get("num_structures") or payload.get("sun_thresholds", {}).get("total_submitted") or 0)
    successful = int(counts.get("successful") or payload.get("sun_thresholds", {}).get("successful_rows") or 0)
    novel_unique = int(counts.get("novel_unique") or round(float(rates.get("novel_unique") or 0.0) * submitted))
    meta_sun = int(counts.get("meta_sun") or round(float(rates.get("meta_sun") or 0.0) * submitted))
    strict_sun = int(counts.get("strict_sun") or round(float(rates.get("strict_sun") or 0.0) * submitted))
    quantiles = payload.get("ehull_quantiles") or payload.get("sun_thresholds", {}).get("ehull_quantiles", {})
    top_formula_counts = payload.get("top_formula_counts") or {}
    top_chemsys_counts = payload.get("top_chemsys_counts") or {}
    return {
        "submitted": submitted,
        "successful": successful,
        "strict_stable": rates.get("strict_stable"),
        "meta_stable": rates.get("meta_stable"),
        "novel": rates.get("novel"),
        "unique": rates.get("unique"),
        "novel_unique": rates.get("novel_unique"),
        "strict_sun": rates.get("strict_sun"),
        "meta_sun": rates.get("meta_sun"),
        "p_meta_stable_given_novel_unique": meta_sun / novel_unique if novel_unique else None,
        "p_strict_stable_given_novel_unique": strict_sun / novel_unique if novel_unique else None,
        "novel_unique_ehull_q50": (quantiles.get("novel_unique") or {}).get("q50"),
        "novel_unique_ehull_mean": (quantiles.get("novel_unique") or {}).get("mean"),
        "top1_formula_concentration": top_concentration(top_formula_counts, submitted, 1),
        "top5_formula_concentration": top_concentration(top_formula_counts, submitted, 5),
        "top1_chemsys_concentration": top_concentration(top_chemsys_counts, submitted, 1),
        "top5_chemsys_concentration": top_concentration(top_chemsys_counts, submitted, 5),
    }


def summarize_run(name: str, run_dir: Path) -> Dict[str, Any]:
    explicit_summary_file = run_dir if run_dir.is_file() else None
    note_dir = run_dir.parent if explicit_summary_file is not None else notes_dir(run_dir)
    if explicit_summary_file is not None:
        threshold_path = explicit_summary_file if "threshold" in explicit_summary_file.name else None
        summary_path = explicit_summary_file
    else:
        threshold_path = first_existing(
            [
                note_dir / "mattergen_sun1000_threshold_analysis.json",
                *sorted(note_dir.glob("*threshold_analysis.json")),
            ]
        )
        summary_path = first_existing(
            [
                note_dir / "mattergen_sun1000_summary.json",
                *sorted(note_dir.glob("*sun1000_summary.json")),
                *sorted(note_dir.glob("*summary.json")),
            ]
        )
    distribution_path = first_existing(
        [
            note_dir / "sample1000_distribution.json",
            note_dir / "sample256_distribution.json",
            *sorted(note_dir.glob("*distribution*.json")),
        ]
    )
    composition_path = first_existing(
        [
            note_dir / "composition1000.json",
            *sorted(note_dir.glob("*composition*.json")),
        ]
    )
    crysllmgen_path = first_existing(
        [
            note_dir / "crysllmgen_metrics1000.json",
            *sorted(note_dir.glob("*crysllmgen*metrics*.json")),
        ]
    )
    sun_payload = read_json(threshold_path) if threshold_path else None
    if sun_payload is None and summary_path is not None:
        sun_payload = read_json(summary_path)
    row: Dict[str, Any] = {
        "name": name,
        "run_dir": str(run_dir),
        "threshold_path": None if threshold_path is None else str(threshold_path),
        "distribution_path": None if distribution_path is None else str(distribution_path),
        "composition_path": None if composition_path is None else str(composition_path),
        "crysllmgen_path": None if crysllmgen_path is None else str(crysllmgen_path),
    }
    row.update(extract_sun_metrics(sun_payload))
    row.update(extract_distribution_metrics(read_json(distribution_path) if distribution_path else None))
    row.update(extract_composition_metrics(read_json(composition_path) if composition_path else None))
    row.update(extract_crysllmgen_metrics(read_json(crysllmgen_path) if crysllmgen_path else None))
    return row


def fmt_percent(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value) * 100:.2f}%"
    except Exception:
        return str(value)


def fmt_float(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


def write_markdown(rows: List[Dict[str, Any]], path: Path) -> None:
    lines = [
        "# R3 E0 SUN Overlap Diagnostics",
        "",
        "| run | meta stable | strict stable | novel unique | meta SUN | strict SUN | P(meta|NU) | P(strict|NU) | high-sym coord | a=b=c | all-90 | top5 formula | top5 chemsys |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {name} | {meta_stable} | {strict_stable} | {novel_unique} | {meta_sun} | {strict_sun} | "
            "{p_meta} | {p_strict} | {hs} | {abc} | {all90} | {top_formula} | {top_chemsys} |".format(
                name=row["name"],
                meta_stable=fmt_percent(row.get("meta_stable")),
                strict_stable=fmt_percent(row.get("strict_stable")),
                novel_unique=fmt_percent(row.get("novel_unique")),
                meta_sun=fmt_percent(row.get("meta_sun")),
                strict_sun=fmt_percent(row.get("strict_sun")),
                p_meta=fmt_percent(row.get("p_meta_stable_given_novel_unique")),
                p_strict=fmt_percent(row.get("p_strict_stable_given_novel_unique")),
                hs=fmt_float(row.get("high_sym_coord_mean")),
                abc=fmt_percent(row.get("a_eq_b_eq_c")),
                all90=fmt_percent(row.get("all_90")),
                top_formula=fmt_percent(row.get("top5_formula_concentration")),
                top_chemsys=fmt_percent(row.get("top5_chemsys_concentration")),
            )
        )
    lines.extend(
        [
            "",
            "## Paths",
            "",
        ]
    )
    for row in rows:
        lines.append(f"- `{row['name']}`: `{row['run_dir']}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", default=[], help="Run spec: name=path or path. May be repeated.")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    rows = [summarize_run(name, path) for name, path in (parse_run_spec(spec) for spec in args.run)]
    payload = {"runs": rows}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    write_json(str(args.output_json), payload)
    write_markdown(rows, args.output_md)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
