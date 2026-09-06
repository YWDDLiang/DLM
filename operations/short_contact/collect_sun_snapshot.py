#!/usr/bin/env python3
"""Collect completed, comparable SUN endpoints without changing any evaluation."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def measure(directory, expected, endpoint, *, conditional=False):
    path = directory/"EVALUATION_FINAL.json"
    if not (directory/"_SUCCESS").is_file():
        return None
    report = json.loads(path.read_text())
    if report["endpoint"] != endpoint:
        raise ValueError(f"endpoint identity changed: {path}")
    value = report["conditional_1000"] if conditional else report
    counts = value["counts"]
    if counts["requests"] != expected:
        raise ValueError(f"request denominator changed: {path}")
    for key in ("strict_sun", "meta_sun", "verified_strict_sun", "verified_meta_sun"):
        if type(counts[key]) is not int or not 0 <= counts[key] <= expected:
            raise ValueError(f"invalid {key}: {path}")
    if counts["strict_sun"] > counts["meta_sun"]:
        raise ValueError(f"Strict is not nested in Meta: {path}")
    protocol = {key: report[key] for key in ("terminal_protocol", "verification_protocol",
                 "novelty_uniqueness_endpoint", "frozen_nu_source_sha256")}
    return {"counts": counts, "strict_percent": 100*counts["strict_sun"]/expected,
            "meta_percent": 100*counts["meta_sun"]/expected,
            "joint_target_met": counts["strict_sun"]*10 >= expected and counts["meta_sun"]*2 >= expected,
            "source": str(path), "source_sha256": digest(path), "protocol": protocol,
            "conditional_parser_prefix": conditional}


def pareto(rows, endpoint):
    complete = [row for row in rows if row["complete"]]
    result = []
    for row in complete:
        a = row[endpoint]["counts"]
        if not any((b["strict_sun"] >= a["strict_sun"] and b["meta_sun"] >= a["meta_sun"])
                   and (b["strict_sun"] > a["strict_sun"] or b["meta_sun"] > a["meta_sun"])
                   for other in complete if other is not row for b in [other[endpoint]["counts"]]):
            result.append(row["method"])
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate-root", type=Path, required=True)
    p.add_argument("--contact-run", type=Path, action="append", default=[])
    p.add_argument("--mixture-run", type=Path, action="append", default=[])
    p.add_argument("--mixture-supplement", type=Path, action="append", default=[])
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    root = args.candidate_root
    runs = root/"runs"
    mixed = root/"experiments/mixed_geometry_h_p33_20260906/runs"
    v2 = root/"experiments/periodic_self_repair_20260906/v2_from_original_llada/runs/eval_39998"
    entries = [
        ("Reference", runs/"spad_state_reference_verification_39893/native-evaluation", runs/"spad_state_reference_verification_39893/tau800-evaluation", False),
        ("K4", runs/"spad_state_eval_method_39910/evaluation", runs/"spad_state_tau800_39910/evaluation", True),
        ("K8", runs/"spad_state_eval_method_39945/evaluation", runs/"spad_state_tau800_39948/evaluation", True),
        ("V2 repair", v2/"native-evaluation", v2/"tau800-evaluation", True),
        ("V3 G", mixed/"eval_40066/native-evaluation", mixed/"eval_40066/tau800-evaluation", True),
        ("V2 construction", v2/"construction-evaluation", mixed/"token_eval_40071/v2-construction-tau800-evaluation", False),
        ("V3 T construction", mixed/"token_eval_40071/native-evaluation", mixed/"token_eval_40071/tau800-evaluation", True),
    ]
    for run in args.contact_run:
        for method in ("k4", "k8"):
            entries.append((method.upper()+" short-contact v1 ("+run.name+")",
                            run/method/"native-evaluation", run/method/"tau800-evaluation", True))
    for run in args.mixture_run:
        entries.append(("K4/K8 equal probability mixture ("+run.name+")",
                        run/"native-evaluation", run/"tau800-evaluation", True))
    rows, common = [], None
    for method, native, refined, candidate in entries:
        row = {"method": method, "candidate": candidate,
               "raw": measure(native, 256, "native"), "tau800": measure(refined, 256, "tau800")}
        row["complete"] = row["raw"] is not None and row["tau800"] is not None
        for endpoint in (row["raw"], row["tau800"]):
            if endpoint is not None:
                if common is None:
                    common = endpoint["protocol"]
                if endpoint["protocol"] != common:
                    raise ValueError(f"development protocols differ: {method}")
        rows.append(row)
    independent = []
    for n in (1000, 1200):
        run = runs/"spad_state_main_39951"
        independent.append({"method": "K8 original", "requests": n,
            "raw": measure(run/"native-evaluation-hull-complete", n, "native", conditional=n == 1000),
            "tau800": measure(run/"tau800-evaluation-hull-complete", n, "tau800", conditional=n == 1000)})
    mixture_supplements = []
    for run in args.mixture_supplement:
        for n in (1000, 1200):
            mixture_supplements.append({"method": "K4/K8 equal probability mixture ("+run.name+")", "requests": n,
                "raw": measure(run/"native-evaluation", n, "native", conditional=n == 1000),
                "tau800": measure(run/"tau800-evaluation", n, "tau800", conditional=n == 1000)})
    completed = [row for row in rows if row["complete"] and row["candidate"]]
    report = {"created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "development_requests_per_method": 256, "development": rows,
              "latest_complete_registered_candidate": completed[-1]["method"] if completed else None,
              "development_pareto_by_endpoint": {name: pareto(rows, name) for name in ("raw", "tau800")},
              "qualified_complete_endpoints": [{"method": row["method"], "endpoint": name}
                  for row in rows if row["complete"] for name in ("raw", "tau800") if row[name]["joint_target_met"]],
              "independent_original_k8": independent, "combined_best_metrics_across_methods": False,
              "independent_mixture": mixture_supplements,
              "score_or_selection_mutations": 0}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir/"SUN_SNAPSHOT.json").write_text(json.dumps(report, indent=2)+"\n")
    def cell(value):
        if value is None:
            return "pending"
        c = value["counts"]
        return f"{c['strict_sun']}/{c['meta_sun']} ({value['strict_percent']:.2f}%/{value['meta_percent']:.2f}%)"
    lines = ["# SUN snapshot", "", "Strict / Meta SUN; development rows each retain all 256 requests.", "",
             "| Method | Raw | Tau800 | Both endpoints complete |", "|---|---:|---:|---|"]
    lines.extend(f"| {row['method']} | {cell(row['raw'])} | {cell(row['tau800'])} | {row['complete']} |" for row in rows)
    lines += ["", "Original K8 independent cohorts are separate from development:", "",
              "| Requests | Raw | Tau800 |", "|---:|---:|---:|"]
    lines.extend(f"| {row['requests']} | {cell(row['raw'])} | {cell(row['tau800'])} |" for row in independent)
    if mixture_supplements:
        lines += ["", "Frozen mixture on the same independent cohort:", "",
                  "| Method | Requests | Raw | Tau800 |", "|---|---:|---:|---:|"]
        lines.extend(f"| {row['method']} | {row['requests']} | {cell(row['raw'])} | {cell(row['tau800'])} |" for row in mixture_supplements)
    lines += ["", "Source paths, original report hashes, verified counts, incomplete endpoints and target checks are in SUN_SNAPSHOT.json.",
              "No metric from one method is combined with another method. No evaluation file or selection is changed.", ""]
    (args.output_dir/"SUN_SNAPSHOT.md").write_text("\n".join(lines))
    print(json.dumps({key: report[key] for key in ("latest_complete_registered_candidate", "development_pareto_by_endpoint", "qualified_complete_endpoints")}), flush=True)


if __name__ == "__main__":
    main()
