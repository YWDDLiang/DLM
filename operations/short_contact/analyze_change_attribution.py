#!/usr/bin/env python3
"""Attribute paired contact outcome changes to changed native bodies, without scoring again."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def read_rows(path, receipts):
    raw = path.read_bytes()
    receipts[str(path)] = hashlib.sha256(raw).hexdigest()
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    result = {row["trajectory_id"]: row for row in rows}
    if len(rows) != 256 or len(result) != 256:
        raise ValueError(f"expected the complete unique development ledger: {path}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--contact-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not (args.contact_run/"_SUCCESS").is_file():
        raise ValueError("both contact methods must have completed before this attribution")
    receipts, summary, cases = {}, {}, []
    fields = ("strict_stable", "meta_stable", "strict_sun", "meta_sun", "novel", "unique_representative",
              "terminal_verified", "verified_strict_sun", "verified_meta_sun")
    for method, native_id, tau_id in (("k4",39910,39910),("k8",39945,39948)):
        old_native = args.candidate_root/f"runs/spad_state_eval_method_{native_id}"
        old_tau = args.candidate_root/f"runs/spad_state_tau800_{tau_id}"
        new = args.contact_run/method
        baseline = read_rows(old_native/"native/paths.jsonl", receipts)
        changed = read_rows(new/"native/paths.jsonl", receipts)
        for stage, original in (("native",old_native),("tau800",old_tau)):
            a = read_rows(original/stage/"paths.jsonl", receipts)
            b = read_rows(new/stage/"paths.jsonl", receipts)
            ae = read_rows(original/"evaluation/attempt_results.jsonl", receipts)
            be = read_rows(new/f"{stage}-evaluation/attempt_results.jsonl", receipts)
            if not all(set(x) == set(a) for x in (b,ae,be,baseline,changed)):
                raise ValueError("paired request identity differs")
            groups = {key:Counter() for key in ("native_body_same", "native_body_changed")}
            for key in a:
                for name in ("group_id", "sample_idx", "species_program", "sampling_seed"):
                    if baseline[key].get(name) != changed[key].get(name):
                        raise ValueError(f"paired conditioning changed: {key}/{name}")
                same_body = baseline[key].get("body") == changed[key].get("body")
                same_structure = a[key].get("structure") == b[key].get("structure")
                group = "native_body_same" if same_body else "native_body_changed"
                count = groups[group]
                count["requests"] += 1
                count["exact_endpoint_structure_same"] += int(same_structure)
                flips = {}
                for field in fields:
                    left, right = bool(ae[key][field]), bool(be[key][field])
                    count[field+"_old"] += int(left)
                    count[field+"_new"] += int(right)
                    count[field+"_gain"] += int(not left and right)
                    count[field+"_loss"] += int(left and not right)
                    if left != right:
                        flips[field] = [left,right]
                cases.append({"method":method,"endpoint":stage,"trajectory_id":key,"sample_idx":a[key]["sample_idx"],
                    "native_body_same":same_body,"exact_endpoint_structure_same":same_structure,"flips":flips,
                    "old_terminal_status":ae[key]["terminal_status"],"new_terminal_status":be[key]["terminal_status"]})
            summary[method+"_"+stage] = {key:dict(value) for key,value in groups.items()}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {"status":"PASS","requests_per_method":256,"summary":summary,"source_sha256":receipts,
              "interpretation":"Equal native bodies isolate changes not caused by sampled token geometry. A repeated evaluator/refiner may differ numerically. This is descriptive attribution, not a new evaluation or quality gate.",
              "energy_evaluations":0,"model_forwards":0,"metric_or_selection_changes":0}
    (args.output_dir/"ATTRIBUTION.json").write_text(json.dumps(report,indent=2)+"\n")
    (args.output_dir/"CASES.json").write_text(json.dumps(cases,indent=2)+"\n")
    (args.output_dir/"_SUCCESS").touch()
    print(json.dumps(report),flush=True)


if __name__ == "__main__":
    main()
