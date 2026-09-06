#!/usr/bin/env python3
"""Uniform recorded cooperative-END ablation of original K4/K8; zero NN."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"));sys.path.insert(0,str(ROOT/"scripts"))
from extract_legacy_construction_endpoint import extract_one as construct,sha256
from crystal_dlm.programmed_path_data import compile_condition,read_jsonl,trace_terminal_body,trace_summary,validate_completed_body
from crystal_dlm.programmed_path_runtime import complete_geometry_supported


def extract_one(record,tokenizer,constraints,method):
    output=construct(record,tokenizer,constraints,method)
    output.pop("construction_only",None)
    output.update(method=f"{method}_cooperative_endpoint_v1",path_mode="cooperative_endpoint",cooperative_only=True,
                  cooperative_executed=False,trace_scope="construct_failed_before_cooperative")
    events=record["trace"]["events"]
    begins=[i for i,e in enumerate(events) if e.get("op")=="begin" and e.get("phase")=="cooperative"]
    ends=[i for i,e in enumerate(events) if e.get("op")=="end" and e.get("phase")=="cooperative"]
    if not output["success"]:
        if begins or ends:raise ValueError("failed constructor has cooperative metadata")
        return output
    if len(begins)!=1 or len(ends)!=1 or begins[0]!=len(output["trace"]["events"]) or ends[0]<=begins[0]:
        raise ValueError("successful constructor requires one complete following cooperative transaction")
    begin,end=begins[0],ends[0]
    positions=list(events[begin]["positions"])
    if len(positions)!=len(set(positions)):raise ValueError("duplicate cooperative transaction position")
    for event in events[begin+1:end]:
        if event["op"] in {"draw","no_support"}:
            if event.get("phase")!="cooperative" or event["position"] not in positions:
                raise ValueError("action is outside the recorded cooperative transaction")
        elif event["op"]=="rollback":
            if not set(event["positions"])<=set(positions):raise ValueError("cooperative rollback exceeds its snapshot")
        else:raise ValueError("foreign event inside cooperative transaction")
    trace={**deepcopy(record["trace"]),"events":deepcopy(events[:end+1]),"success":True,"failure":None}
    ids=trace_terminal_body(trace)
    body=tokenizer.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
    compiled=compile_condition(record,tokenizer,mask_id=trace["mask_id"],purpose="evaluation")
    validate_completed_body(body,compiled)
    import torch
    if not complete_geometry_supported(torch.tensor(ids,dtype=torch.long),constraints):
        raise RuntimeError("cooperative END support replay violates the original rollback guarantee")
    output.update(success=True,failure=None,sampling_failure=None,native_execution_success=True,parseable=True,
        body=body,final_body_token_ids=ids,trace=trace,trace_summary=trace_summary(trace),cooperative_executed=True,
        cooperative_end_event_index=end,omitted_later_events=len(events)-end-1,
        trace_scope="cooperative_prefix_of_same_original_attempt")
    return output


def extract_rows(rows,tokenizer,constraints,method,expected_requests):
    expected=list(range(expected_requests))
    if (len(rows)!=expected_requests or sorted(int(r["sample_idx"]) for r in rows)!=expected
            or sorted(int(r["condition_ordinal"]) for r in rows)!=expected
            or len({r["trajectory_id"] for r in rows})!=expected_requests):
        raise ValueError("cooperative extraction must retain the entire frozen cohort")
    return [extract_one(r,tokenizer,constraints,method) for r in sorted(rows,key=lambda r:r["condition_ordinal"])]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--paths-jsonl",type=Path,required=True);p.add_argument("--checkpoint-path",type=Path,required=True)
    p.add_argument("--method",choices=("k4","k8"),required=True);p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--expected-requests",type=int,default=256);p.add_argument("--source-sha256")
    p.add_argument("--input-manifest",type=Path,default=ROOT/"docs/v3_scientific_audit_20260906/evidence_v3_failure_20260907/k4k8_prefix_inputs.json")
    args=p.parse_args()
    if args.expected_requests<1:raise ValueError("positive request count is required")
    if args.output_dir.resolve().is_relative_to(args.paths_jsonl.parent.resolve()):raise ValueError("write outside the original sample directory")
    expected_hash=args.source_sha256
    if expected_hash is None:
        manifest=json.loads(args.input_manifest.read_text());binding=manifest["methods"][args.method]
        if Path(binding["source_path"]).resolve()!=args.paths_jsonl.resolve() or Path(binding["checkpoint"]).resolve()!=args.checkpoint_path.resolve():
            raise ValueError("new original source requires explicit --source-sha256")
        expected_hash=manifest["source_receipts"][binding["source_path"]]
    if sha256(args.paths_jsonl)!=expected_hash or not (args.paths_jsonl.parent/"_SUCCESS").is_file():raise ValueError("original source differs or is incomplete")
    records=read_jsonl(args.paths_jsonl)
    if any(Path(r["checkpoint"]).resolve()!=args.checkpoint_path.resolve() for r in records):raise ValueError("source checkpoint differs")
    from crystal_dlm.short_contact_sampling import validate_legacy_contact_policy
    if validate_legacy_contact_policy(args.checkpoint_path)["collection_round"]!={"k4":0,"k8":1}[args.method]:raise ValueError("method/policy mismatch")
    from transformers import AutoTokenizer
    from scripts.sample_llada_dynamic_crystals import build_dynamic_lightweight_constraints
    import torch
    torch.set_num_threads(2)
    tokenizer=AutoTokenizer.from_pretrained(args.checkpoint_path,trust_remote_code=True,local_files_only=True)
    constraints=build_dynamic_lightweight_constraints(tokenizer,duplicate_coordinate_mask=True,lattice_volume_mask=True,
        min_lattice_rad=1e-4,canonicalize_periodic_alias=True,pbc_min_distance_mask=True,pbc_min_distance_A=.5,pbc_image_radius=2)
    rows=extract_rows(records,tokenizer,constraints,args.method,args.expected_requests)
    if sha256(args.paths_jsonl)!=expected_hash:raise ValueError("source changed during extraction")
    args.output_dir.mkdir(parents=True,exist_ok=False)
    (args.output_dir/"paths.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False,allow_nan=False)+"\n" for r in rows),encoding="utf-8")
    report={"method":f"{args.method}_cooperative_endpoint_v1","requests":len(rows),"successful":sum(r["success"] for r in rows),
            "failed":sum(not r["success"] for r in rows),"cooperative_executed":sum(r["cooperative_executed"] for r in rows),
            "source":str(args.paths_jsonl),"source_sha256":expected_hash,"checkpoint":str(args.checkpoint_path),
            "failure_reasons":dict(Counter(r["trace"]["failure"] for r in rows if not r["success"])),
            "finite_named_stage_ablation":True,"per_request_stage_search":False,"model_forwards":0,"energy_evaluations":0,"optimizer_steps":0}
    (args.output_dir/"COOPERATIVE_EXTRACTION.json").write_text(json.dumps(report,indent=2)+"\n")
    (args.output_dir/"_SUCCESS").touch();print(json.dumps(report),flush=True)


if __name__=="__main__":main()
