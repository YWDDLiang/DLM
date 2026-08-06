#!/usr/bin/env python3
"""Analyze fixed-slot DLM sample outputs and failure modes."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from crystal_dlm.fixed_slot import (  # noqa: E402
    FixedSlotError,
    parse_fixed_slot_answer,
    tokenize_answer_text,
    write_json,
)
from crystal_dlm.fixed_slot_compressed import (  # noqa: E402
    CompressedFixedSlotConfig,
    parse_compressed_fixed_slot_answer,
    tokenize_compressed_answer_text,
)
from crystal_dlm.physical_header import (  # noqa: E402
    parse_physical_header_answer,
    tokenize_physical_header_text,
)


HIGH_SYMMETRY_VALUES = {0.0, 0.25, 0.5, 0.75, 1.0}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def iter_text_records(path: Path, text_key: str) -> Iterable[Tuple[int, str, Mapping[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        for row_idx, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            text = row.get(text_key)
            if text is None:
                text = row.get("answer") or row.get("text")
            if text is None:
                continue
            sample_idx = int(row.get("sample_idx", row_idx))
            yield sample_idx, str(text), row


def coord_key(coord: Sequence[float]) -> Tuple[int, int, int]:
    return tuple(int(round(float(value) * 100)) for value in coord)  # type: ignore[return-value]


def pbc_coord_key(coord: Sequence[float]) -> Tuple[int, int, int]:
    return tuple(int(round(float(value) * 100)) % 100 for value in coord)  # type: ignore[return-value]


def min_fractional_distance(coords: Sequence[Sequence[float]]) -> float | None:
    if len(coords) < 2:
        return None
    best = math.inf
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            delta = [
                abs(float(coords[i][axis]) - float(coords[j][axis])) for axis in range(3)
            ]
            delta = [min(value, 1.0 - value) for value in delta]
            dist = math.sqrt(sum(value * value for value in delta))
            best = min(best, dist)
    return best if best < math.inf else None


def duplicate_summary(arrays: Mapping[str, Any]) -> Dict[str, Any]:
    species = list(arrays["species"])
    coords = list(arrays["frac_coords"])
    coord_counts: Counter[Tuple[int, int, int]] = Counter(coord_key(coord) for coord in coords)
    pbc_coord_counts: Counter[Tuple[int, int, int]] = Counter(pbc_coord_key(coord) for coord in coords)
    species_coord_counts: Counter[Tuple[str, Tuple[int, int, int]]] = Counter(
        (symbol, coord_key(coord)) for symbol, coord in zip(species, coords)
    )
    exact_duplicate_groups = {
        ",".join(f"{axis:03d}" for axis in key): count
        for key, count in coord_counts.items()
        if count > 1
    }
    pbc_duplicate_groups = {
        ",".join(f"{axis:03d}" for axis in key): count
        for key, count in pbc_coord_counts.items()
        if count > 1
    }
    same_species_duplicate_groups = {
        f"{symbol}@{','.join(f'{axis:03d}' for axis in key)}": count
        for (symbol, key), count in species_coord_counts.items()
        if count > 1
    }
    high_symmetry_coord_count = 0
    for coord in coords:
        if all(float(value) in HIGH_SYMMETRY_VALUES for value in coord):
            high_symmetry_coord_count += 1
    lengths = [float(value) for value in arrays["lengths"]]
    angles = [float(value) for value in arrays["angles"]]
    return {
        "exact_duplicate_site_count": sum(count - 1 for count in coord_counts.values() if count > 1),
        "pbc_equivalent_duplicate_site_count": sum(count - 1 for count in pbc_coord_counts.values() if count > 1),
        "pbc_only_duplicate_site_count": max(
            0,
            sum(count - 1 for count in pbc_coord_counts.values() if count > 1)
            - sum(count - 1 for count in coord_counts.values() if count > 1),
        ),
        "same_species_duplicate_site_count": sum(
            count - 1 for count in species_coord_counts.values() if count > 1
        ),
        "exact_duplicate_groups": exact_duplicate_groups,
        "pbc_duplicate_groups": pbc_duplicate_groups,
        "same_species_duplicate_groups": same_species_duplicate_groups,
        "min_fractional_distance": min_fractional_distance(coords),
        "all_angles_90": all(abs(value - 90.0) < 1e-6 for value in angles),
        "all_lengths_equal": max(lengths) - min(lengths) < 1e-6,
        "high_symmetry_coord_fraction": high_symmetry_coord_count / max(1, len(coords)),
    }


def summarize_records(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {}
    texts = [str(row.get("text", "")) for row in rows]
    return {
        "raw_record_count": len(rows),
        "unique_text_count": len(set(texts)),
        "raw_parsed_true_count": sum(1 for row in rows if row.get("parsed") is True),
        "raw_reason_counts": dict(Counter(str(row.get("reason")) for row in rows if row.get("reason"))),
    }


def parse_answer_for_representation(
    text: str,
    representation: str,
    compressed_config: CompressedFixedSlotConfig | None,
) -> Dict[str, Any]:
    if representation == "fixed_slot_compressed_v1":
        return parse_compressed_fixed_slot_answer(
            text,
            config=compressed_config or CompressedFixedSlotConfig(),
        )
    if representation == "fixed_slot_physical_header":
        return parse_physical_header_answer(text)
    return parse_fixed_slot_answer(text)


def tokenize_for_representation(
    text: str,
    representation: str,
) -> List[str]:
    if representation == "fixed_slot_compressed_v1":
        return tokenize_compressed_answer_text(text)
    if representation == "fixed_slot_physical_header":
        return tokenize_physical_header_text(text) + tokenize_answer_text(text)
    return tokenize_answer_text(text)


def analyze_file(
    path: Path,
    text_key: str,
    failure_rows: List[Dict[str, Any]],
    representation: str,
    compressed_config: CompressedFixedSlotConfig | None,
) -> Dict[str, Any]:
    failure_by_idx: Dict[int, Dict[str, Any]] = {}
    for row in failure_rows:
        if "sample_idx" in row:
            failure_by_idx[int(row["sample_idx"])] = row

    metrics: Dict[str, Any] = {
        "input_path": str(path),
        "total": 0,
        "unique_outputs": 0,
        "schema_parse_success": 0,
        "schema_parse_failures": 0,
        "parse_error_counts": {},
        "token_count_histogram": {},
        "num_atoms_histogram": {},
        "element_histogram": {},
        "graph_success_from_raw": 0,
        "graph_failure_from_raw": 0,
        "graph_failure_message_counts": {},
        "duplicate_exact_site_failures": 0,
        "duplicate_pbc_site_failures": 0,
        "same_species_duplicate_failures": 0,
        "records_with_exact_duplicate_sites": 0,
        "records_with_pbc_equivalent_duplicate_sites": 0,
        "records_with_same_species_duplicate_sites": 0,
        "records_all_angles_90": 0,
        "records_all_lengths_equal": 0,
        "high_symmetry_coord_fraction_mean": None,
        "min_fractional_distance_min": None,
        "examples": {
            "parse_failures": [],
            "graph_failures_with_duplicates": [],
            "graph_success": [],
        },
    }
    parse_error_counts: Counter[str] = Counter()
    token_count_histogram: Counter[str] = Counter()
    num_atoms_histogram: Counter[str] = Counter()
    element_histogram: Counter[str] = Counter()
    graph_failure_messages: Counter[str] = Counter()
    high_symmetry_fractions: List[float] = []
    min_distances: List[float] = []
    seen_texts = set()

    for sample_idx, text, row in iter_text_records(path, text_key):
        metrics["total"] += 1
        seen_texts.add(text)
        tokens = tokenize_for_representation(text, representation)
        token_count_histogram[str(len(tokens))] += 1
        raw_graph_success = row.get("parsed") is True
        if raw_graph_success:
            metrics["graph_success_from_raw"] += 1
        elif "parsed" in row:
            metrics["graph_failure_from_raw"] += 1

        try:
            arrays = parse_answer_for_representation(text, representation, compressed_config)
        except FixedSlotError as exc:
            metrics["schema_parse_failures"] += 1
            message = str(exc)
            parse_error_counts[message] += 1
            if len(metrics["examples"]["parse_failures"]) < 5:
                metrics["examples"]["parse_failures"].append(
                    {
                        "sample_idx": sample_idx,
                        "message": message,
                        "token_count": len(tokens),
                        "text_prefix": text[:300],
                    }
                )
            continue

        metrics["schema_parse_success"] += 1
        num_atoms_histogram[str(arrays["num_atoms"])] += 1
        element_histogram.update(arrays["species"])
        dup = duplicate_summary(arrays)
        high_symmetry_fractions.append(float(dup["high_symmetry_coord_fraction"]))
        if dup["min_fractional_distance"] is not None:
            min_distances.append(float(dup["min_fractional_distance"]))
        if dup["all_angles_90"]:
            metrics["records_all_angles_90"] += 1
        if dup["all_lengths_equal"]:
            metrics["records_all_lengths_equal"] += 1
        if dup["exact_duplicate_site_count"] > 0:
            metrics["records_with_exact_duplicate_sites"] += 1
        if dup["pbc_equivalent_duplicate_site_count"] > 0:
            metrics["records_with_pbc_equivalent_duplicate_sites"] += 1
        if dup["same_species_duplicate_site_count"] > 0:
            metrics["records_with_same_species_duplicate_sites"] += 1

        failure = failure_by_idx.get(sample_idx)
        failure_message = None
        if failure is not None:
            failure_message = str(failure.get("message", ""))
        elif row.get("message"):
            failure_message = str(row.get("message"))
        if failure_message:
            graph_failure_messages[failure_message] += 1
            if dup["exact_duplicate_site_count"] > 0:
                metrics["duplicate_exact_site_failures"] += 1
            if dup["pbc_equivalent_duplicate_site_count"] > 0:
                metrics["duplicate_pbc_site_failures"] += 1
            if dup["same_species_duplicate_site_count"] > 0:
                metrics["same_species_duplicate_failures"] += 1
            if (
                (dup["exact_duplicate_site_count"] > 0 or dup["pbc_equivalent_duplicate_site_count"] > 0)
                and len(metrics["examples"]["graph_failures_with_duplicates"]) < 8
            ):
                metrics["examples"]["graph_failures_with_duplicates"].append(
                    {
                        "sample_idx": sample_idx,
                        "message": failure_message,
                        "num_atoms": arrays["num_atoms"],
                        "species": arrays["species"],
                        "frac_coords": arrays["frac_coords"],
                        "duplicates": dup,
                        "text_prefix": text[:300],
                    }
                )
        elif raw_graph_success and len(metrics["examples"]["graph_success"]) < 5:
            metrics["examples"]["graph_success"].append(
                {
                    "sample_idx": sample_idx,
                    "num_atoms": arrays["num_atoms"],
                    "species": arrays["species"],
                    "frac_coords": arrays["frac_coords"],
                    "duplicates": dup,
                    "text_prefix": text[:300],
                }
            )

    metrics["unique_outputs"] = len(seen_texts)
    metrics["parse_error_counts"] = dict(parse_error_counts.most_common())
    metrics["token_count_histogram"] = dict(token_count_histogram.most_common())
    metrics["num_atoms_histogram"] = dict(num_atoms_histogram.most_common())
    metrics["element_histogram"] = dict(element_histogram.most_common(30))
    metrics["graph_failure_message_counts"] = dict(graph_failure_messages.most_common(20))
    if high_symmetry_fractions:
        metrics["high_symmetry_coord_fraction_mean"] = sum(high_symmetry_fractions) / len(
            high_symmetry_fractions
        )
    if min_distances:
        metrics["min_fractional_distance_min"] = min(min_distances)
    metrics["schema_parse_rate"] = metrics["schema_parse_success"] / max(1, metrics["total"])
    metrics["raw_graph_rate"] = metrics["graph_success_from_raw"] / max(1, metrics["total"])
    return metrics


def write_markdown(summary: Mapping[str, Any], path: Path) -> None:
    lines = [
        "# sample failure analysis",
        "",
        "## 诊断结论",
        "",
        f"- 总样本数：{summary.get('total', 0)}",
        f"- 唯一输出数：{summary.get('unique_outputs', 0)}",
        f"- fixed-slot 语法 parse rate：{summary.get('schema_parse_rate', 0):.4f}",
        f"- raw graph rate：{summary.get('raw_graph_rate', 0):.4f}",
        f"- 有精确重复坐标的样本数：{summary.get('records_with_exact_duplicate_sites', 0)}",
        f"- 有同元素同坐标重复的样本数：{summary.get('records_with_same_species_duplicate_sites', 0)}",
        f"- graph 失败中带精确重复坐标的数量：{summary.get('duplicate_exact_site_failures', 0)}",
        f"- 全 90 度晶格样本数：{summary.get('records_all_angles_90', 0)}",
        f"- a=b=c 晶格样本数：{summary.get('records_all_lengths_equal', 0)}",
        f"- 高对称坐标比例均值：{summary.get('high_symmetry_coord_fraction_mean')}",
        "",
        "## parse 错误",
        "",
        "```json",
        json.dumps(summary.get("parse_error_counts", {}), ensure_ascii=False, indent=2),
        "```",
        "",
        "## graph 失败消息",
        "",
        "```json",
        json.dumps(summary.get("graph_failure_message_counts", {}), ensure_ascii=False, indent=2),
        "```",
        "",
        "## 重复坐标失败样例",
        "",
        "```json",
        json.dumps(
            summary.get("examples", {}).get("graph_failures_with_duplicates", [])[:5],
            ensure_ascii=False,
            indent=2,
        ),
        "```",
        "",
        "## graph 成功样例",
        "",
        "```json",
        json.dumps(
            summary.get("examples", {}).get("graph_success", [])[:3],
            ensure_ascii=False,
            indent=2,
        ),
        "```",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--failure-jsonl", type=Path, default=None)
    parser.add_argument("--text-key", default="text")
    parser.add_argument(
        "--representation",
        choices=["fixed_slot", "fixed_slot_compressed_v1", "fixed_slot_physical_header"],
        default="fixed_slot",
    )
    parser.add_argument("--compressed-token-config", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    compressed_config = (
        CompressedFixedSlotConfig.from_path(args.compressed_token_config)
        if args.compressed_token_config is not None
        else None
    )
    failure_rows = read_jsonl(args.failure_jsonl) if args.failure_jsonl else []
    summary = analyze_file(
        args.input_jsonl,
        args.text_key,
        failure_rows,
        representation=args.representation,
        compressed_config=compressed_config,
    )
    summary["raw_record_summary"] = summarize_records(read_jsonl(args.input_jsonl))
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    write_json(str(args.output_json), summary)
    write_markdown(summary, args.output_md)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
