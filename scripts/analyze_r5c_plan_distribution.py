#!/usr/bin/env python3
"""Compare R5-C generated composition plans against teacher-plan distribution."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from crystal_dlm.r5_plan_body import parse_composition_plan  # noqa: E402


METAL_SYMBOLS = {
    "Li",
    "Be",
    "Na",
    "Mg",
    "Al",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
}


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def canonical_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    elements = [str(item) for item in (plan.get("elements") or [])]
    counts = [int(item) for item in (plan.get("counts") or [])]
    formula = str(plan.get("formula") or "")
    n_value = int(plan.get("N"))
    result = {
        "formula": formula,
        "elements": elements,
        "counts": counts,
        "N": n_value,
    }
    for key in (
        "family",
        "arity",
        "size",
        "anion_framework",
        "charge_bucket",
        "lattice_system",
        "spacegroup_bucket",
        "volume_per_atom_bin",
        "generated_rich_fields",
        "rich_field_valid",
        "expected_anion_framework",
        "anion_match_formula",
        "generated_family",
        "generated_arity",
        "generated_size",
        "family_match_formula",
        "arity_match_formula",
        "size_match_formula",
        "semantic_consistency",
    ):
        if key in plan:
            result[key] = plan[key]
    return result


def extract_generated_plan(record: Mapping[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    parsed = record.get("parsed_plan")
    if isinstance(parsed, Mapping):
        try:
            return canonical_plan(parsed), None
        except Exception as exc:  # noqa: BLE001 - keep bad records auditable.
            return None, f"{type(exc).__name__}: {exc}"
    text = str(record.get("plan_text") or record.get("raw_plan_text") or "")
    if not text:
        return None, "missing plan text"
    try:
        return canonical_plan(parse_composition_plan(text)), None
    except Exception as exc:  # noqa: BLE001 - generation failures are diagnostics.
        return None, f"{type(exc).__name__}: {exc}"


def extract_teacher_plan(record: Mapping[str, Any]) -> tuple[str, dict[str, Any] | None, str | None]:
    metadata = record.get("metadata") if isinstance(record.get("metadata"), Mapping) else {}
    key = str(
        metadata.get("material_id")
        or record.get("material_id")
        or metadata.get("id")
        or record.get("idx")
        or ""
    )
    parsed = record.get("plan_state")
    if isinstance(parsed, Mapping):
        try:
            return key, canonical_plan(parsed), None
        except Exception as exc:  # noqa: BLE001
            return key, None, f"{type(exc).__name__}: {exc}"
    answer = record.get("answer")
    if isinstance(answer, str) and answer.strip():
        try:
            parsed_answer = json.loads(answer)
            if isinstance(parsed_answer, Mapping):
                return key, canonical_plan(parsed_answer), None
        except Exception as exc:  # noqa: BLE001
            return key, None, f"answer_json:{type(exc).__name__}: {exc}"
    text = str(record.get("plan_text") or "")
    if not text:
        return key, None, "missing plan text"
    try:
        return key, canonical_plan(parse_composition_plan(text)), None
    except Exception as exc:  # noqa: BLE001
        return key, None, f"{type(exc).__name__}: {exc}"


def add_plan(stats: dict[str, Any], plan: Mapping[str, Any]) -> None:
    elements = [str(item) for item in plan["elements"]]
    counts = [int(item) for item in plan["counts"]]
    n_value = int(plan["N"])
    formula = str(plan["formula"])
    stats["parsed_rows"] += 1
    stats["n_sum"] += n_value
    if n_value >= 12:
        stats["n_ge_12_count"] += 1
    if len(elements) == 3:
        stats["ternary_count"] += 1
    if len(elements) >= 4:
        stats["four_plus_elements_count"] += 1
    stats["n_histogram"][str(n_value)] += 1
    stats["num_elements_histogram"][str(len(elements))] += 1
    stats["formula_histogram"][formula] += 1
    stats["family_histogram"][str(plan.get("family", "unknown"))] += 1
    stats["arity_histogram"][str(plan.get("arity", "unknown"))] += 1
    stats["size_histogram"][str(plan.get("size", "unknown"))] += 1
    stats["anion_framework_histogram"][str(plan.get("anion_framework", "unknown"))] += 1
    stats["charge_bucket_histogram"][str(plan.get("charge_bucket", "unknown"))] += 1
    stats["lattice_system_histogram"][str(plan.get("lattice_system", "unknown"))] += 1
    stats["spacegroup_bucket_histogram"][str(plan.get("spacegroup_bucket", "sg_unknown"))] += 1
    stats["volume_per_atom_bin_histogram"][str(plan.get("volume_per_atom_bin", "volpa_unknown"))] += 1
    consistency = plan.get("semantic_consistency") if isinstance(plan.get("semantic_consistency"), Mapping) else {}
    if consistency.get("family_match_formula") is True or plan.get("family_match_formula") is True:
        stats["family_match_formula_count"] += 1
    if consistency.get("arity_match_formula") is True or plan.get("arity_match_formula") is True:
        stats["arity_match_formula_count"] += 1
    if consistency.get("size_match_formula") is True or plan.get("size_match_formula") is True:
        stats["size_match_formula_count"] += 1
    if len(elements) == 1:
        stats["single_element_count"] += 1
    if elements and all(symbol in METAL_SYMBOLS for symbol in elements):
        stats["all_metal_count"] += 1
    for symbol, count in zip(elements, counts):
        stats["element_histogram"][symbol] += int(count)
        stats["element_presence_histogram"][symbol] += 1


def empty_stats(name: str) -> dict[str, Any]:
    return {
        "name": name,
        "rows": 0,
        "parsed_rows": 0,
        "failed_rows": 0,
        "n_sum": 0,
        "n_ge_12_count": 0,
        "ternary_count": 0,
        "four_plus_elements_count": 0,
        "single_element_count": 0,
        "all_metal_count": 0,
        "family_match_formula_count": 0,
        "arity_match_formula_count": 0,
        "size_match_formula_count": 0,
        "n_histogram": Counter(),
        "num_elements_histogram": Counter(),
        "family_histogram": Counter(),
        "arity_histogram": Counter(),
        "size_histogram": Counter(),
        "anion_framework_histogram": Counter(),
        "charge_bucket_histogram": Counter(),
        "lattice_system_histogram": Counter(),
        "spacegroup_bucket_histogram": Counter(),
        "volume_per_atom_bin_histogram": Counter(),
        "formula_histogram": Counter(),
        "element_histogram": Counter(),
        "element_presence_histogram": Counter(),
        "failure_reasons": Counter(),
        "failure_examples": [],
    }


def summarize(stats: dict[str, Any], *, top_k: int) -> dict[str, Any]:
    rows = int(stats["rows"])
    parsed = int(stats["parsed_rows"])
    result = {
        "name": stats["name"],
        "rows": rows,
        "parsed_rows": parsed,
        "failed_rows": int(stats["failed_rows"]),
        "parse_rate": parsed / max(1, rows),
        "mean_N": float(stats["n_sum"]) / max(1, parsed),
        "n_ge_12_rate_all": int(stats["n_ge_12_count"]) / max(1, rows),
        "n_ge_12_rate_parsed": int(stats["n_ge_12_count"]) / max(1, parsed),
        "ternary_rate_all": int(stats["ternary_count"]) / max(1, rows),
        "ternary_rate_parsed": int(stats["ternary_count"]) / max(1, parsed),
        "four_plus_elements_rate_all": int(stats["four_plus_elements_count"]) / max(1, rows),
        "four_plus_elements_rate_parsed": int(stats["four_plus_elements_count"]) / max(1, parsed),
        "single_element_rate_all": int(stats["single_element_count"]) / max(1, rows),
        "single_element_rate_parsed": int(stats["single_element_count"]) / max(1, parsed),
        "all_metal_rate_all": int(stats["all_metal_count"]) / max(1, rows),
        "all_metal_rate_parsed": int(stats["all_metal_count"]) / max(1, parsed),
        "family_match_formula_rate_all": int(stats["family_match_formula_count"]) / max(1, rows),
        "arity_match_formula_rate_all": int(stats["arity_match_formula_count"]) / max(1, rows),
        "size_match_formula_rate_all": int(stats["size_match_formula_count"]) / max(1, rows),
        "n_histogram": dict(stats["n_histogram"].most_common()),
        "num_elements_histogram": dict(stats["num_elements_histogram"].most_common()),
        "family_histogram": dict(stats["family_histogram"].most_common()),
        "arity_histogram": dict(stats["arity_histogram"].most_common()),
        "size_histogram": dict(stats["size_histogram"].most_common()),
        "anion_framework_histogram": dict(stats["anion_framework_histogram"].most_common()),
        "charge_bucket_histogram": dict(stats["charge_bucket_histogram"].most_common()),
        "lattice_system_histogram": dict(stats["lattice_system_histogram"].most_common()),
        "spacegroup_bucket_histogram": dict(stats["spacegroup_bucket_histogram"].most_common()),
        "volume_per_atom_bin_histogram": dict(stats["volume_per_atom_bin_histogram"].most_common()),
        "formula_top": dict(stats["formula_histogram"].most_common(top_k)),
        "element_top": dict(stats["element_histogram"].most_common(top_k)),
        "element_presence_top": dict(stats["element_presence_histogram"].most_common(top_k)),
        "failure_reasons": dict(stats["failure_reasons"].most_common(top_k)),
        "failure_examples": stats["failure_examples"][:top_k],
    }
    return result


def distribution(counter: Mapping[str, int]) -> dict[str, float]:
    total = float(sum(int(value) for value in counter.values()))
    if total <= 0:
        return {}
    return {str(key): float(value) / total for key, value in counter.items()}


def tvd(left: Mapping[str, int], right: Mapping[str, int]) -> float:
    p = distribution(left)
    q = distribution(right)
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(key, 0.0) - q.get(key, 0.0)) for key in keys)


def js_divergence(left: Mapping[str, int], right: Mapping[str, int]) -> float:
    p = distribution(left)
    q = distribution(right)
    keys = set(p) | set(q)
    if not keys:
        return 0.0
    m = {key: 0.5 * (p.get(key, 0.0) + q.get(key, 0.0)) for key in keys}

    def kl(source: Mapping[str, float], target: Mapping[str, float]) -> float:
        total = 0.0
        for key, value in source.items():
            if value > 0 and target.get(key, 0.0) > 0:
                total += value * math.log(value / target[key], 2)
        return total

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def compare_stats(teacher: dict[str, Any], generated: dict[str, Any]) -> dict[str, Any]:
    return {
        "n_tvd": tvd(teacher["n_histogram"], generated["n_histogram"]),
        "n_js": js_divergence(teacher["n_histogram"], generated["n_histogram"]),
        "num_elements_tvd": tvd(teacher["num_elements_histogram"], generated["num_elements_histogram"]),
        "num_elements_js": js_divergence(
            teacher["num_elements_histogram"],
            generated["num_elements_histogram"],
        ),
        "element_presence_tvd": tvd(
            teacher["element_presence_histogram"],
            generated["element_presence_histogram"],
        ),
        "element_presence_js": js_divergence(
            teacher["element_presence_histogram"],
            generated["element_presence_histogram"],
        ),
        "family_tvd": tvd(teacher["family_histogram"], generated["family_histogram"]),
        "arity_tvd": tvd(teacher["arity_histogram"], generated["arity_histogram"]),
        "size_tvd": tvd(teacher["size_histogram"], generated["size_histogram"]),
        "anion_framework_tvd": tvd(teacher["anion_framework_histogram"], generated["anion_framework_histogram"]),
        "charge_bucket_tvd": tvd(teacher["charge_bucket_histogram"], generated["charge_bucket_histogram"]),
        "lattice_system_tvd": tvd(teacher["lattice_system_histogram"], generated["lattice_system_histogram"]),
        "spacegroup_bucket_tvd": tvd(
            teacher["spacegroup_bucket_histogram"],
            generated["spacegroup_bucket_histogram"],
        ),
        "volume_per_atom_bin_tvd": tvd(
            teacher["volume_per_atom_bin_histogram"],
            generated["volume_per_atom_bin_histogram"],
        ),
        "all_metal_rate_delta_all": (
            generated["all_metal_count"] / max(1, generated["rows"])
            - teacher["all_metal_count"] / max(1, teacher["rows"])
        ),
        "single_element_rate_delta_all": (
            generated["single_element_count"] / max(1, generated["rows"])
            - teacher["single_element_count"] / max(1, teacher["rows"])
        ),
    }


def load_teacher(path: Path, *, dedupe: bool) -> dict[str, Any]:
    stats = empty_stats("teacher")
    seen: set[str] = set()
    for idx, record in enumerate(iter_jsonl(path)):
        key, plan, failure = extract_teacher_plan(record)
        if dedupe:
            dedupe_key = key or str(record.get("plan_text") or record.get("answer") or idx)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
        stats["rows"] += 1
        if plan is None:
            stats["failed_rows"] += 1
            stats["failure_reasons"][failure or "unknown"] += 1
            if len(stats["failure_examples"]) < 20:
                stats["failure_examples"].append({"row": idx, "reason": failure})
            continue
        add_plan(stats, plan)
    return stats


def load_generated(path: Path) -> dict[str, Any]:
    stats = empty_stats("generated")
    for idx, record in enumerate(iter_jsonl(path)):
        stats["rows"] += 1
        plan, failure = extract_generated_plan(record)
        if plan is None:
            stats["failed_rows"] += 1
            reason = failure or str(record.get("message") or record.get("reason") or "unknown")
            stats["failure_reasons"][reason] += 1
            if len(stats["failure_examples"]) < 20:
                stats["failure_examples"].append(
                    {
                        "row": idx,
                        "sample_idx": record.get("sample_idx"),
                        "reason": reason,
                        "raw_plan_text": str(record.get("raw_plan_text") or record.get("plan_text") or "")[:320],
                    }
                )
            continue
        add_plan(stats, plan)
    return stats


def write_markdown(result: Mapping[str, Any], path: Path) -> None:
    teacher = result["teacher"]
    generated = result["generated"]
    comparison = result["comparison"]
    lines = [
        "# R5-C Plan Distribution Audit",
        "",
        "## Summary",
        "",
        f"- teacher rows: `{teacher['rows']}`",
        f"- generated rows: `{generated['rows']}`",
        f"- generated parse_rate: `{generated['parse_rate']:.6f}`",
        f"- N TVD: `{comparison['n_tvd']:.6f}`",
        f"- num-elements TVD: `{comparison['num_elements_tvd']:.6f}`",
        f"- element-presence TVD: `{comparison['element_presence_tvd']:.6f}`",
        f"- family TVD: `{comparison['family_tvd']:.6f}`",
        f"- arity TVD: `{comparison['arity_tvd']:.6f}`",
        f"- size TVD: `{comparison['size_tvd']:.6f}`",
        f"- generated mean_N: `{generated['mean_N']:.6f}`",
        f"- generated N>=12: `{generated['n_ge_12_rate_all']:.6f}`",
        f"- generated ternary: `{generated['ternary_rate_all']:.6f}`",
        f"- generated 4+ elements: `{generated['four_plus_elements_rate_all']:.6f}`",
        f"- generated all-metal rate: `{generated['all_metal_rate_all']:.6f}`",
        f"- generated single-element rate: `{generated['single_element_rate_all']:.6f}`",
        "",
        "## Teacher",
        "",
        "```json",
        json.dumps(teacher, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Generated",
        "",
        "```json",
        json.dumps(generated, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Comparison",
        "",
        "```json",
        json.dumps(comparison, ensure_ascii=False, indent=2),
        "```",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-jsonl", type=Path, required=True)
    parser.add_argument("--generated-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument(
        "--no-dedupe-teacher",
        action="store_true",
        help="Count every teacher row. Default dedupes mixture rows by metadata/material id.",
    )
    args = parser.parse_args()

    teacher_raw = load_teacher(args.teacher_jsonl, dedupe=not args.no_dedupe_teacher)
    generated_raw = load_generated(args.generated_jsonl)
    result = {
        "teacher_jsonl": str(args.teacher_jsonl),
        "generated_jsonl": str(args.generated_jsonl),
        "teacher": summarize(teacher_raw, top_k=args.top_k),
        "generated": summarize(generated_raw, top_k=args.top_k),
        "comparison": compare_stats(teacher_raw, generated_raw),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(result, args.output_md)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
