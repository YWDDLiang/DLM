#!/usr/bin/env python3
"""R5 conditional bucket diagnostics for existing DLM/S.U.N. runs."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from crystal_dlm.fixed_slot import parse_fixed_slot_answer, write_json  # noqa: E402
from crystal_dlm.r5_conditioning import build_z_payload_from_arrays  # noqa: E402


BUCKET_KEYS = [
    "full_formula",
    "chemsys",
    "anion_framework",
    "lattice_system",
    "high_symmetry_token",
    "prototype_key",
]


def parse_run_spec(spec: str) -> tuple[str, Path]:
    if "=" in spec:
        name, path = spec.split("=", 1)
        return name.strip(), Path(path.strip())
    path = Path(spec)
    return path.name, path


def first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def find_raw_jsonl(run_dir: Path) -> Path | None:
    if run_dir.is_file():
        return run_dir if run_dir.suffix == ".jsonl" else None
    candidates = [
        run_dir / "outputs/sample1000/raw_generations.jsonl",
        run_dir / "outputs/sample256/raw_generations.jsonl",
        run_dir / "outputs/r2_baseline256/raw_generations.jsonl",
        run_dir / "outputs/e4a_geometry256/raw_generations.jsonl",
        run_dir / "raw_generations.jsonl",
    ]
    candidates.extend(sorted(run_dir.glob("outputs/**/raw_generations.jsonl")))
    return first_existing(candidates)


def find_detailed_metrics(run_dir: Path) -> Path | None:
    if run_dir.is_file():
        return None
    candidates = [
        run_dir / "notes/mattergen_sun1000_detailed_metrics.json",
    ]
    if (run_dir / "notes").exists():
        candidates.extend(sorted((run_dir / "notes").glob("*detailed_metrics.json")))
    return first_existing(candidates)


def read_json(path: Path | None) -> Any:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def detailed_row(metrics: Mapping[str, Any] | None, idx: int) -> dict[str, Any]:
    if not metrics:
        return {}
    out: dict[str, Any] = {}
    for key, values in metrics.items():
        if isinstance(values, list) and idx < len(values):
            out[key] = values[idx]
    return out


def bool_value(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def record_from_raw(row: Mapping[str, Any], idx: int, metrics: Mapping[str, Any] | None) -> dict[str, Any] | None:
    text = str(row.get("text") or "")
    try:
        arrays = parse_fixed_slot_answer(text)
        z = build_z_payload_from_arrays(arrays, metadata={})
    except Exception:
        return None
    metric = detailed_row(metrics, idx)
    ehull = metric.get("energy_above_hull_per_atom")
    try:
        ehull_value = None if ehull is None else float(ehull)
    except Exception:
        ehull_value = None
    novel_unique = bool_value(metric.get("novel_unique"))
    return {
        "z": z,
        "ehull": ehull_value,
        "meta_stable": None if ehull_value is None else ehull_value < 0.1,
        "strict_stable": None if ehull_value is None else ehull_value < 0.0,
        "novel_unique": novel_unique,
        "meta_sun": None if ehull_value is None or novel_unique is None else (ehull_value < 0.1 and novel_unique),
        "strict_sun": None if ehull_value is None or novel_unique is None else (ehull_value < 0.0 and novel_unique),
    }


def summarize_bool(values: list[bool | None]) -> float | None:
    known = [value for value in values if value is not None]
    if not known:
        return None
    return sum(bool(value) for value in known) / len(known)


def summarize_bucket(records: list[dict[str, Any]], key: str, min_count: int) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record["z"].get(key, "unknown"))].append(record)
    rows: list[dict[str, Any]] = []
    for value, items in groups.items():
        if len(items) < min_count:
            continue
        ehulls = [item["ehull"] for item in items if item["ehull"] is not None]
        novel_unique_ehulls = [
            item["ehull"]
            for item in items
            if item["ehull"] is not None and item.get("novel_unique") is True
        ]
        rows.append(
            {
                "bucket_key": key,
                "bucket_value": value,
                "count": len(items),
                "meta_stable_rate": summarize_bool([item.get("meta_stable") for item in items]),
                "strict_stable_rate": summarize_bool([item.get("strict_stable") for item in items]),
                "novel_unique_rate": summarize_bool([item.get("novel_unique") for item in items]),
                "meta_sun_rate": summarize_bool([item.get("meta_sun") for item in items]),
                "strict_sun_rate": summarize_bool([item.get("strict_sun") for item in items]),
                "ehull_q50": statistics.median(ehulls) if ehulls else None,
                "novel_unique_ehull_q50": statistics.median(novel_unique_ehulls) if novel_unique_ehulls else None,
            }
        )
    rows.sort(key=lambda row: (-int(row["count"]), str(row["bucket_value"])))
    return rows


def summarize_run(name: str, run_dir: Path, min_count: int) -> dict[str, Any]:
    raw_path = find_raw_jsonl(run_dir)
    metrics_path = find_detailed_metrics(run_dir)
    metrics = read_json(metrics_path)
    records: list[dict[str, Any]] = []
    parse_fail = 0
    if raw_path is not None:
        with raw_path.open(encoding="utf-8") as handle:
            for idx, line in enumerate(handle):
                if not line.strip():
                    continue
                row = json.loads(line)
                record = record_from_raw(row, idx, metrics)
                if record is None:
                    parse_fail += 1
                else:
                    records.append(record)
    buckets: dict[str, list[dict[str, Any]]] = {}
    for key in BUCKET_KEYS:
        buckets[key] = summarize_bucket(records, key, min_count)
    z_counter = Counter(record["z"].get("prototype_key", "unknown") for record in records)
    return {
        "name": name,
        "run_dir": str(run_dir),
        "raw_path": None if raw_path is None else str(raw_path),
        "detailed_metrics_path": None if metrics_path is None else str(metrics_path),
        "record_count": len(records),
        "parse_fail": parse_fail,
        "has_sun_metrics": metrics is not None,
        "overall": {
            "meta_stable_rate": summarize_bool([record.get("meta_stable") for record in records]),
            "strict_stable_rate": summarize_bool([record.get("strict_stable") for record in records]),
            "novel_unique_rate": summarize_bool([record.get("novel_unique") for record in records]),
            "meta_sun_rate": summarize_bool([record.get("meta_sun") for record in records]),
            "strict_sun_rate": summarize_bool([record.get("strict_sun") for record in records]),
        },
        "top_prototype_keys": z_counter.most_common(20),
        "buckets": buckets,
    }


def fmt_percent(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value) * 100:.2f}%"


def write_markdown(payload: Mapping[str, Any], path: Path) -> None:
    lines = ["# R5 Conditional Bucket Diagnostics", ""]
    lines.extend(
        [
            "| run | records | meta stable | strict stable | novel unique | meta SUN | strict SUN | raw | metrics |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for row in payload["runs"]:
        overall = row["overall"]
        lines.append(
            f"| {row['name']} | {row['record_count']} | {fmt_percent(overall.get('meta_stable_rate'))} | "
            f"{fmt_percent(overall.get('strict_stable_rate'))} | {fmt_percent(overall.get('novel_unique_rate'))} | "
            f"{fmt_percent(overall.get('meta_sun_rate'))} | {fmt_percent(overall.get('strict_sun_rate'))} | "
            f"`{row.get('raw_path')}` | `{row.get('detailed_metrics_path')}` |"
        )
    for row in payload["runs"]:
        lines.extend(["", f"## {row['name']}", ""])
        for key in BUCKET_KEYS:
            bucket_rows = row["buckets"].get(key, [])[:10]
            if not bucket_rows:
                continue
            lines.extend(
                [
                    f"### {key}",
                    "",
                    "| value | count | meta | strict | novel_unique | meta SUN | strict SUN | NU Ehull q50 |",
                    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
                ]
            )
            for bucket in bucket_rows:
                nu_q50 = bucket.get("novel_unique_ehull_q50")
                nu_q50_text = "-" if nu_q50 is None else f"{float(nu_q50):.4f}"
                lines.append(
                    f"| `{bucket['bucket_value']}` | {bucket['count']} | "
                    f"{fmt_percent(bucket.get('meta_stable_rate'))} | {fmt_percent(bucket.get('strict_stable_rate'))} | "
                    f"{fmt_percent(bucket.get('novel_unique_rate'))} | {fmt_percent(bucket.get('meta_sun_rate'))} | "
                    f"{fmt_percent(bucket.get('strict_sun_rate'))} | "
                    f"{nu_q50_text} |"
                )
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, help="name=run_dir")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--min-count", type=int, default=5)
    args = parser.parse_args()
    runs = [
        summarize_run(name, path, int(args.min_count))
        for name, path in (parse_run_spec(spec) for spec in args.run)
    ]
    payload = {"runs": runs, "min_count": int(args.min_count)}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    write_json(str(args.output_json), payload)
    write_markdown(payload, args.output_md)


if __name__ == "__main__":
    main()
