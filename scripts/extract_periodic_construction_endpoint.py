#!/usr/bin/env python3
"""Measure the construction endpoint already present in each V2 attempt trace.

This performs no extra DLM sampling and selects no outcomes. Failed construction
requests remain in the same ledger. The diagnostic is never a training teacher.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from crystal_dlm.programmed_path_data import read_jsonl, trace_terminal_body, trace_summary


def construction_endpoint(record, decode):
    if record.get("source_split") != "evaluation":
        raise ValueError("construction diagnostics are evaluation-only")
    trace = deepcopy(record["trace"])
    start = next((i for i, event in enumerate(trace["events"])
                  if event["op"] == "begin" and event["phase"] == "full_cell_repair"), None)
    if start is not None:
        trace["events"] = trace["events"][:start]
        if any(event.get("phase") != "construct" for event in trace["events"]):
            raise ValueError("unexpected phase before V2 full-cell repair")
        trace["success"] = True
        trace["failure"] = None
    body = trace_terminal_body(trace)
    output = dict(record, success=bool(trace["success"]), body=decode(body),
                  final_body_token_ids=body, trace=trace, trace_summary=trace_summary(trace),
                  trace_scope="construction_prefix_of_same_attempt", diagnostic_only=True,
                  diagnostic="before_full_cell_repair", repair_transaction_entered=start is not None)
    # The input is the sampler output, not a subsequently exported/relaxed artifact.
    for field in ("structure", "cif_path", "parseable", "endpoint", "native_execution_success"):
        output.pop(field, None)
    return output


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--paths-jsonl", type=Path, required=True)
    p.add_argument("--checkpoint-path", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    rows = read_jsonl(args.paths_jsonl)
    if len(rows) != 256 or sorted(int(row["sample_idx"]) for row in rows) != list(range(256)):
        raise ValueError("the registered diagnostic retains all fixed 256 requests")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_path, trust_remote_code=True, local_files_only=True)
    output = [construction_endpoint(row, lambda ids: tokenizer.decode(
        ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)) for row in rows]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "paths.jsonl").open("x", encoding="utf-8") as handle:
        for row in output:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.output_dir / "CONSTRUCTION_DIAGNOSTIC.json").write_text(json.dumps({
        "requests": len(rows), "extra_DLM_samples": 0, "source": str(args.paths_jsonl),
        "repair_transactions_entered": sum(row["repair_transaction_entered"] for row in output),
        "outcome_selection": False, "training_data": False}, indent=2) + "\n")
    (args.output_dir / "_SUCCESS").touch()


if __name__ == "__main__":
    main()
