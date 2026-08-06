#!/usr/bin/env python3
"""R5-0 data, baseline, and failure-pool audit."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from crystal_dlm.fixed_slot import write_json  # noqa: E402
from crystal_dlm.r5_repair import normalize_violation_labels  # noqa: E402


DEFAULT_RUNS = [
    "runs/20260527_semalign_selfimprove_r2",
    "runs/20260529_r3_e3_physical_header_r2abs_fix2",
    "runs/20260529_r4_e4a_geometry_inpaint_r2abs",
]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def summarize_data_dir(data_dir: Path) -> Dict[str, Any]:
    stats_path = data_dir / "stats.json"
    success_path = data_dir / "_SUCCESS"
    payload: Dict[str, Any] = {
        "path": str(data_dir),
        "exists": data_dir.exists(),
        "stats_exists": stats_path.exists(),
        "success_exists": success_path.exists(),
        "splits": {},
    }
    if stats_path.exists():
        stats = read_json(stats_path)
        payload["representation"] = stats.get("representation")
        payload["prompt_version"] = stats.get("prompt_version")
        payload["r5_representation"] = stats.get("r5_representation")
        payload["max_answer_model_length"] = stats.get("max_answer_model_length")
        payload["max_length_recommended"] = stats.get("max_length_recommended")
        for split in ("train", "val", "test"):
            split_stats = stats.get("splits", {}).get(split, {})
            payload["splits"][split] = {
                "rows_seen": split_stats.get("rows_seen"),
                "rows_written": split_stats.get("rows_written"),
                "failures": split_stats.get("failures"),
                "module_counts": split_stats.get("module_counts"),
                "atom_count_histogram": split_stats.get("atom_count_histogram"),
            }
    return payload


def find_run_files(run_dir: Path) -> Dict[str, list[str]]:
    if not run_dir.exists():
        return {}
    patterns = {
        "sample_metrics": "outputs/**/sample_metrics.json",
        "composition": "notes/*composition*.json",
        "distribution": "notes/*distribution*.json",
        "crysllmgen_metrics": "notes/*crysllmgen*metrics*.json",
        "mattergen": "notes/*mattergen*.json",
        "raw_generations": "outputs/**/raw_generations.jsonl",
        "failure_cases": "outputs/**/failure_cases.jsonl",
    }
    return {key: [str(path) for path in sorted(run_dir.glob(pattern))] for key, pattern in patterns.items()}


def summarize_run(run_dir: Path) -> Dict[str, Any]:
    files = find_run_files(run_dir)
    summary: Dict[str, Any] = {"run_dir": str(run_dir), "exists": run_dir.exists(), "files": files}
    sample_metrics = []
    for path_text in files.get("sample_metrics", [])[:20]:
        path = Path(path_text)
        try:
            metric = read_json(path)
            sample_metrics.append(
                {
                    "path": str(path),
                    "decoded_samples": metric.get("decoded_samples"),
                    "parse_rate": metric.get("parse_rate"),
                    "graph_acceptance_rate": metric.get("graph_acceptance_rate", metric.get("graph_rate")),
                    "valid_array_count": metric.get("valid_array_count"),
                }
            )
        except Exception:
            pass
    summary["sample_metrics"] = sample_metrics
    composition_summaries = []
    for path_text in files.get("composition", [])[:20]:
        path = Path(path_text)
        try:
            payload = read_json(path)
            composition_summaries.append(
                {
                    "path": str(path),
                    "raw_jsonl_comp_valid": payload.get("raw_jsonl", {}).get("comp_valid_rate"),
                    "raw_pt_comp_valid": payload.get("raw_pt", {}).get("comp_valid_rate"),
                    "reason_counts": payload.get("raw_jsonl", {}).get("reason_counts")
                    or payload.get("raw_pt", {}).get("reason_counts"),
                }
            )
        except Exception:
            pass
    summary["composition_summaries"] = composition_summaries
    return summary


def labels_from_failure(row: Mapping[str, Any]) -> list[str]:
    text = " ".join(str(row.get(key, "")) for key in ("reason", "stage", "message")).lower()
    labels = []
    if "high_sym" in text or "symmetry" in text:
        labels.append("high_sym")
    if "duplicate" in text or "pbc" in text:
        labels.append("pbc_duplicate")
    if "single" in text:
        labels.append("single_element")
    if "all_metal" in text or "all-metal" in text:
        labels.append("all_metal")
    if "charge" in text:
        labels.append("charge_fail")
    if "pauling" in text:
        labels.append("pauling_fail")
    if "refine" in text or "relax" in text:
        labels.append("refinement_fail")
    if "parse" in text:
        labels.append("parse_fail")
    if "graph" in text:
        labels.append("graph_fail")
    return normalize_violation_labels(labels or [row.get("reason", "unknown")])


def build_failure_pool(run_dirs: Iterable[Path], output_jsonl: Path, max_rows: int) -> Dict[str, Any]:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    label_counter: Counter[str] = Counter()
    source_counter: Counter[str] = Counter()
    written = 0
    with output_jsonl.open("w", encoding="utf-8") as out:
        for run_dir in run_dirs:
            if not run_dir.exists():
                continue
            paths = list(sorted(run_dir.glob("outputs/**/failure_cases.jsonl")))
            paths.extend(sorted(run_dir.glob("**/failure_cases.jsonl")))
            seen = set()
            for path in paths:
                if path in seen or not path.exists():
                    continue
                seen.add(path)
                for row in read_jsonl(path):
                    if written >= max_rows:
                        break
                    labels = labels_from_failure(row)
                    record = {
                        "source_run": str(run_dir),
                        "source_file": str(path),
                        "source_row": row,
                        "violation_labels": labels,
                        "reason": row.get("reason"),
                        "message": row.get("message"),
                        "visible_proposal": row.get("text") or row.get("visible_proposal"),
                    }
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    written += 1
                    source_counter[str(run_dir.name)] += 1
                    for label in labels:
                        label_counter[label] += 1
                if written >= max_rows:
                    break
    return {
        "rows_written": written,
        "label_histogram": dict(label_counter.most_common()),
        "source_histogram": dict(source_counter.most_common()),
    }


def write_markdown(payload: Mapping[str, Any], path: Path) -> None:
    lines = ["# R5-0 Data Audit", ""]
    lines.extend(["## Data builders", ""])
    lines.extend(["| name | exists | representation | train rows | val rows | test rows |", "| --- | ---: | --- | ---: | ---: | ---: |"])
    for name, item in payload["data_dirs"].items():
        splits = item.get("splits", {})
        lines.append(
            f"| {name} | {item.get('exists')} | {item.get('representation')} | "
            f"{splits.get('train', {}).get('rows_written')} | {splits.get('val', {}).get('rows_written')} | "
            f"{splits.get('test', {}).get('rows_written')} |"
        )
    lines.extend(["", "## Baseline runs", ""])
    lines.extend(["| run | exists | sample metric files | composition files | failure files |", "| --- | ---: | ---: | ---: | ---: |"])
    for item in payload["runs"]:
        files = item.get("files", {})
        lines.append(
            f"| `{Path(item['run_dir']).name}` | {item.get('exists')} | "
            f"{len(files.get('sample_metrics', []))} | {len(files.get('composition', []))} | "
            f"{len(files.get('failure_cases', []))} |"
        )
    failure = payload.get("failure_pool", {})
    lines.extend(
        [
            "",
            "## Failure Pool",
            "",
            f"- rows_written: `{failure.get('rows_written', 0)}`",
            f"- label_histogram: `{json.dumps(failure.get('label_histogram', {}), sort_keys=True)}`",
            f"- source_histogram: `{json.dumps(failure.get('source_histogram', {}), sort_keys=True)}`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--crysllmgen-text-dir", type=Path, default=PROJECT_ROOT / "data/dlm_sft/mp_20_crysllmgen_text")
    parser.add_argument("--crysllmgen-modular-dir", type=Path, default=PROJECT_ROOT / "data/dlm_sft/mp_20_crysllmgen_modular_v2")
    parser.add_argument("--run", action="append", default=[])
    parser.add_argument("--max-failure-pool-rows", type=int, default=20000)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_dirs = [PROJECT_ROOT / item for item in (args.run or DEFAULT_RUNS)]
    data_dirs = {
        "crysllmgen_text": summarize_data_dir(args.crysllmgen_text_dir),
        "crysllmgen_modular": summarize_data_dir(args.crysllmgen_modular_dir),
    }
    runs = [summarize_run(path) for path in run_dirs]
    failure_pool_path = args.output_dir / "r5_0_failure_pool.jsonl"
    failure_summary = build_failure_pool(run_dirs, failure_pool_path, args.max_failure_pool_rows)
    payload = {
        "data_dirs": data_dirs,
        "runs": runs,
        "failure_pool": failure_summary,
        "failure_pool_path": str(failure_pool_path),
    }
    write_json(str(args.output_dir / "r5_0_baseline_distribution.json"), payload)
    write_json(str(args.output_dir / "r5_0_data_audit.json"), payload)
    write_markdown(payload, args.output_dir / "r5_0_data_audit.md")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
