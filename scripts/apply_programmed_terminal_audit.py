#!/usr/bin/env python3
"""Add uniform verification evidence while preserving every original energy."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from crystal_dlm.programmed_path_data import read_jsonl
from crystal_dlm.terminal_energy_consistency import TERMINAL_VERIFICATION_PROTOCOL


def apply_audit(labels, audits):
    ids = {r["trajectory_id"] for r in labels}
    by_id = {r["trajectory_id"]: r for r in audits}
    if len(ids) != len(labels) or len(by_id) != len(audits):
        raise ValueError("duplicate label or audit occurrence")
    if set(by_id) != {r["trajectory_id"] for r in labels if r["verified"] is True}:
        raise ValueError("all and only the originally verified occurrences must be audited")
    output = []
    for row in labels:
        result = dict(row)
        if row["verified"] is True:
            check = by_id[row["trajectory_id"]]
            if str(check["group_id"]) != str(row["group_id"]):
                raise ValueError("audit condition mismatch")
            if check["status"] not in ("consistent", "inconsistent", "audit_error"):
                raise ValueError("unknown audit status")
            if check["status"] != "audit_error" and check["stored_terminal_energy_eV_atom"] != row["terminal_energy"]:
                raise ValueError("audit was not performed on this terminal energy")
            result["before_terminal_consistency"] = {"verified": row["verified"], "status": row["status"]}
            result["terminal_consistency"] = check
            if check["status"] != "consistent":
                result["verified"] = False
                result["status"] = "terminal_consistency_unverified" if check["status"] == "inconsistent" else "terminal_consistency_unknown"
        output.append(result)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-jsonl", type=Path, nargs="+", required=True)
    parser.add_argument("--audit-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not (args.audit_jsonl.parent / "_SUCCESS").is_file():
        raise ValueError("audit accounting is incomplete")
    audit_report = json.loads((args.audit_jsonl.parent / "AUDIT_FINAL.json").read_text())
    if audit_report["energy_tolerance_eV_atom"] != .001 or audit_report["rigid_fractional_shift"] != [.137, .271, .419]:
        raise ValueError("uniform audit settings changed")
    if {Path(p).resolve() for p in audit_report["source_labels"]} != {p.resolve() for p in args.labels_jsonl}:
        raise ValueError("audit and original label sources differ")
    labels, manifests = [], []
    for source in args.labels_jsonl:
        if not (source.parent / "_SUCCESS").is_file():
            raise ValueError("original label accounting is incomplete")
        manifests.append(json.loads((source.parent / "LABEL_FINAL.json").read_text()))
        labels += read_jsonl(source)
    if any(m["protocol"] != manifests[0]["protocol"] or m["purpose"] != manifests[0]["purpose"] for m in manifests):
        raise ValueError("original label protocols or purposes differ")
    result = apply_audit(labels, read_jsonl(args.audit_jsonl))
    report = {"requested": len(result), "completed": len(result),
              "statuses": dict(Counter(r["status"] for r in result)),
              "protocol": manifests[0]["protocol"], "purpose": manifests[0]["purpose"],
              "verification_protocol": TERMINAL_VERIFICATION_PROTOCOL,
              "original_labels": [str(p) for p in args.labels_jsonl], "audit": str(args.audit_jsonl),
              "all_original_energies_and_occurrences_preserved": True,
              "verification_withheld": sum(a["verified"] and not b["verified"] for a, b in zip(labels, result)),
              "physical_correctness_claim": "passing reproducibility does not prove the surrogate energy is physically correct"}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "labels.jsonl").open("x", encoding="utf-8") as handle:
        for row in result:
            handle.write(json.dumps(row) + "\n")
    (args.output_dir / "LABEL_FINAL.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output_dir / "_SUCCESS").touch()
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
