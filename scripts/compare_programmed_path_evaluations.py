#!/usr/bin/env python3
"""Compare the one fixed development reference and final policy by request."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def index_requests(rows):
    indexed = {int(row["sample_idx"]): row for row in rows}
    if len(indexed) != len(rows) or set(indexed) != set(range(len(rows))):
        raise ValueError("paired development requests must be unique and complete")
    return indexed


def validate_pairing(reference_paths, method_paths):
    reference, method = map(index_requests, (reference_paths, method_paths))
    if set(reference) != set(method):
        raise ValueError("reference and method request denominators differ")
    fields = ("group_id", "prompt", "plan_state", "species_program", "species_program_source",
              "num_atoms", "sampling_seed", "sampling_batch_size", "candidate_index")
    for index in reference:
        for row in (reference[index], method[index]):
            if row.get("source_split") != "evaluation":
                raise ValueError("training outcomes cannot enter the paired evaluation")
        for field in fields:
            if field not in reference[index] or field not in method[index] or reference[index][field] != method[index][field]:
                raise ValueError(f"paired condition differs at request {index}: {field}")


def field_value(row, field):
    value = row
    for key in field.split("."):
        value = value.get(key) if isinstance(value, dict) else None
    return float(value) if value is not None and math.isfinite(float(value)) else None


def paired_metric(pairs, field, *, verified=False):
    values = []
    for reference, method in pairs:
        if verified and not (reference["terminal_verified"] and method["terminal_verified"]):
            continue
        a, b = field_value(reference, field), field_value(method, field)
        if a is not None and b is not None:
            values.append((a, b))
    deltas = [b - a for a, b in values]
    return {"pairs": len(values), "excluded_requests": len(pairs) - len(values),
            "reference_mean": statistics.fmean(a for a, _ in values) if values else None,
            "method_mean": statistics.fmean(b for _, b in values) if values else None,
            "mean_method_minus_reference": statistics.fmean(deltas) if values else None,
            "median_method_minus_reference": statistics.median(deltas) if values else None,
            "method_lower": sum(d < 0 for d in deltas), "method_higher": sum(d > 0 for d in deltas),
            "equal": sum(d == 0 for d in deltas)}


def compare_results(reference_rows, method_rows):
    reference, method = map(index_requests, (reference_rows, method_rows))
    if set(reference) != set(method):
        raise ValueError("evaluated request denominators differ")
    pairs = [(reference[i], method[i]) for i in sorted(reference)]
    for a, b in pairs:
        if a["group_id"] != b["group_id"]:
            raise ValueError("evaluated conditions differ")
        if a.get("chemsys") and b.get("chemsys") and a["chemsys"] != b["chemsys"]:
            raise ValueError("evaluated chemical systems differ")
        ha, hb = a.get("hull_energy_eV_atom"), b.get("hull_energy_eV_atom")
        if ha is not None and hb is not None and not math.isclose(ha, hb, rel_tol=0., abs_tol=1e-10):
            raise ValueError("paired hull references differ")
    binary = {}
    for field in ("reconstructed", "native_execution_success", "endpoint_execution_success", "novel", "unique_representative",
                  "novel_unique", "terminal_verified", "strict_stable", "meta_stable", "strict_sun", "meta_sun",
                  "verified_strict_sun", "verified_meta_sun"):
        cells = Counter((bool(a[field]), bool(b[field])) for a, b in pairs)
        ref_count, method_count = sum(bool(a[field]) for a, _ in pairs), sum(bool(b[field]) for _, b in pairs)
        binary[field] = {"reference_count": ref_count, "method_count": method_count,
                         "reference_percent": 100 * ref_count / len(pairs), "method_percent": 100 * method_count / len(pairs),
                         "change_percentage_points": 100 * (method_count - ref_count) / len(pairs),
                         "both": cells[True, True], "reference_only": cells[True, False],
                         "method_only": cells[False, True], "neither": cells[False, False]}
    raw = {name: paired_metric(pairs, field) for name, field in
           (("energy_eV_atom", "raw_energy_eV_atom"), ("force_max_eV_A", "raw.force_max_eV_A"),
            ("stress_max_GPa", "raw.stress_max_GPa"))}
    terminal_fields = (("A_gap_eV_atom", "gap_eV_atom"), ("B_terminal_energy_eV_atom", "terminal_energy_eV_atom"),
                       ("relaxation_steps", "actual_relaxation_steps"), ("force_max_eV_A", "terminal.force_max_eV_A"),
                       ("stress_max_GPa", "terminal.stress_max_GPa"))
    verified = {name: paired_metric(pairs, field, verified=True) for name, field in terminal_fields}
    available = {name: paired_metric(pairs, field) for name, field in terminal_fields}
    return {"requests": len(pairs), "binary_all_requests": binary, "raw_finite_pairs": raw,
            "verified_terminal_pairs": verified, "available_terminal_pairs": available,
            "label_statuses": {"reference": dict(Counter(a["terminal_status"] for a, _ in pairs)),
                               "method": dict(Counter(b["terminal_status"] for _, b in pairs))},
            "hull_statuses": {"reference": dict(Counter(a["official_hull_status"] for a, _ in pairs)),
                              "method": dict(Counter(b["official_hull_status"] for _, b in pairs))},
            "delta_sign": "method minus reference; negative energy, force, stress and steps is lower",
            "B_difference": "same-composition hull cancels: delta B = method eR - reference eR; absolute means shown are eR",
            "coverage": "physical paired means exclude missing values; verified means additionally require both terminal verifications",
            "interpretation": "descriptive fixed-development comparison; teacher feasibility does not imply these outcomes"}


def render_markdown(report):
    text = [f"# Fixed development comparison: {report['endpoint']}", "",
            f"{report['requests']} paired requests. Changes are method minus reference.", "",
            "| Metric | Reference | Method | Change (percentage points) |", "|---|---:|---:|---:|"]
    for key in ("reconstructed", "terminal_verified", "strict_sun", "meta_sun", "verified_strict_sun", "verified_meta_sun"):
        row = report["binary_all_requests"][key]
        text.append(f"| {key} | {row['reference_count']} ({row['reference_percent']:.3f}%) | {row['method_count']} ({row['method_percent']:.3f}%) | {row['change_percentage_points']:+.3f} |")
    text += ["", "| Physical metric | Pairs | Reference mean | Method mean | Paired mean change |", "|---|---:|---:|---:|---:|"]
    def number(value):
        return "unavailable" if value is None else f"{value:.6g}"
    for section in ("raw_finite_pairs", "verified_terminal_pairs"):
        for key, row in report[section].items():
            text.append(f"| {section}: {key} | {row['pairs']} | {number(row['reference_mean'])} | {number(row['method_mean'])} | {number(row['mean_method_minus_reference'])} |")
    text += ["", report["B_difference"] + ".", "", report["coverage"] + ".", "",
             "SUN uses input N/U and common relaxed-energy thresholds; the verified subset adds optimizer, force, stress and geometry checks.", ""]
    return "\n".join(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-eval-dir", type=Path, required=True)
    parser.add_argument("--method-eval-dir", type=Path, required=True)
    parser.add_argument("--reference-paths-jsonl", type=Path, required=True)
    parser.add_argument("--method-paths-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifests = []
    for directory in (args.reference_eval_dir, args.method_eval_dir):
        if not (directory / "_SUCCESS").is_file():
            raise ValueError("evaluation accounting is incomplete")
        manifests.append(json.loads((directory / "EVALUATION_FINAL.json").read_text()))
    for key in ("endpoint", "terminal_protocol", "frozen_nu_source_sha256", "official_cache", "cohort_role"):
        if manifests[0][key] != manifests[1][key]:
            raise ValueError(f"paired evaluation protocols differ: {key}")
    if any(m["cohort_role"] != "fixed_development" or m["counts"]["requests"] != 256 for m in manifests):
        raise ValueError("this comparison is restricted to the frozen development256")
    validate_pairing(read_jsonl(args.reference_paths_jsonl), read_jsonl(args.method_paths_jsonl))
    report = compare_results(read_jsonl(args.reference_eval_dir / "attempt_results.jsonl"),
                             read_jsonl(args.method_eval_dir / "attempt_results.jsonl"))
    if report["requests"] != 256:
        raise ValueError("fixed development denominator changed")
    report.update(endpoint=manifests[0]["endpoint"], provenance={k: str(v) for k, v in vars(args).items()},
                  terminal_protocol=manifests[0]["terminal_protocol"], frozen_nu_source_sha256=manifests[0]["frozen_nu_source_sha256"])
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "COMPARISON_FINAL.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "comparison.md").write_text(render_markdown(report), encoding="utf-8")
    (args.output_dir / "_SUCCESS").touch()
    print(json.dumps({k: report[k] for k in ("requests", "endpoint", "binary_all_requests", "verified_terminal_pairs")}), flush=True)


if __name__ == "__main__":
    main()
