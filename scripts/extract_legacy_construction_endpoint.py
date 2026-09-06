#!/usr/bin/env python3
"""Deterministically extract every legacy K4/K8 construction endpoint; no NN."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from crystal_dlm.programmed_path_data import compile_condition,read_jsonl,trace_terminal_body,trace_summary,validate_completed_body
from crystal_dlm.programmed_path_runtime import complete_geometry_supported
from crystal_dlm.spad_program import spad_predictor_position_groups


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):h.update(chunk)
    return h.hexdigest()


def extract_one(record,tokenizer,constraints,method):
    if method not in {"k4","k8"} or record.get("source_split")!="evaluation":
        raise ValueError("construction extraction requires an explicitly named evaluation policy")
    full=record["trace"]
    if full.get("temperature")!=.7:
        raise ValueError("construction-only v1 retains original temperature .7")
    if full.get("short_contact_policy") or full.get("probability_mixture_policy"):
        raise ValueError("this ablation uses original K4/K8, without later inference transforms")
    events=full["events"]
    if not events or events[0].get("op")!="begin" or events[0].get("phase")!="construct":
        raise ValueError("original full path does not begin with construction")
    end=next((i for i,e in enumerate(events) if e.get("op")=="end" and e.get("phase")=="construct"),None)
    if end is None:raise ValueError("original construction transaction has no recorded end")
    kept=deepcopy(events[:end+1])
    if (any(e.get("phase")!="construct" or e["op"] not in {"begin","draw","no_support","end"} for e in kept)
            or sum(e["op"]=="begin" for e in kept)!=1 or sum(e["op"]=="end" for e in kept)!=1):
        raise ValueError("construction prefix has a foreign or malformed transaction")
    mask=int(full["mask_id"])
    compiled=compile_condition(record,tokenizer,mask_id=mask,purpose="evaluation")
    if compiled["initial_body"]!=full["initial_body"] or compiled["prompt_token_ids"]!=record["prompt_token_ids"]:
        raise ValueError("tokenizer, Plan or prompt differs from the original construction canvas")
    if list(compiled["program"].element_order)!=full["element_order"]:
        raise ValueError("original constructor program differs from the declared P")
    schedule=[g[0] for g in spad_predictor_position_groups(compiled["program"])[1:]]
    actions=[e for e in kept if e["op"] in {"draw","no_support"}]
    if ([e["position"] for e in actions]!=schedule[:len(actions)] or len(actions)>len(schedule)
            or any(e["op"]=="no_support" for e in actions[:-1])):
        raise ValueError("constructor action order is not the original fixed P prefix")
    trace={**deepcopy(full),"events":kept,"success":False,"failure":None}
    ids=trace_terminal_body(trace)
    body=tokenizer.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
    schema_error=None
    try:
        validate_completed_body(body,compiled)
        schema_pass=True
    except Exception as error:
        schema_pass=False;schema_error=f"{type(error).__name__}: {error}"
    supported=False
    support_error=None
    if schema_pass:
        import torch
        supported=bool(complete_geometry_supported(torch.tensor(ids,dtype=torch.long,device="cpu"),constraints))
    entered_later=any(e.get("op")=="begin" and e.get("phase") in {"cooperative","closure"} for e in events[end+1:])
    if entered_later and not supported:
        raise RuntimeError("construction support replay disagrees with the original entry into a later phase")
    no_support=any(e["op"]=="no_support" for e in kept)
    success=not no_support and schema_pass and supported
    failure=("construct_empty_support" if no_support else
             "construction_schema_failure" if not schema_pass else "unsupported_predictor" if not supported else None)
    trace.update(success=success,failure=failure)
    output=deepcopy(record)
    for field in ("structure","native_structure","cif","cif_path","artifact_error","refiner_graph_error",
                  "diagnostic_candidate","legacy_reference_closure","probability_mixture","short_contact_policy"):
        output.pop(field,None)
    output.update(method=f"{method}_construction_only_v1",base_policy=method,path_mode="construction_only",
        success=success,failure=failure,sampling_failure=None if success else {"stage":"construction","message":failure},
        native_execution_success=success,parseable=success,body=body,final_body_token_ids=ids,
        structure=None,native_structure=None,cif=None,cif_path=None,endpoint="native",trace=trace,
        trace_summary=trace_summary(trace),trace_scope="construction_prefix_of_same_original_attempt",
        construction_only=True,full_cycle_success=bool(record.get("success")),
        full_cycle_failure=full.get("failure"),omitted_later_events=len(events)-len(kept),
        construction_check={"no_support_event":no_support,"schema_pass":schema_pass,"complete_geometry_supported":supported,
                            "schema_error":schema_error,"support_error":support_error},
        inference_mlip=False,extra_DLM_samples=0,model_forwards=0,optimizer_steps=0,outcome_selection=False)
    output.setdefault("evaluation_ordinal",int(record["sample_idx"]))
    if not success:
        # The exact attempted mask-containing body remains evidence; parseable=False
        # prevents the legacy exporter from re-admitting a failed complete preview.
        output["diagnostic_candidate"]={"body":body,"final_body_token_ids":ids}
    return output


def extract_rows(rows,tokenizer,constraints,method,expected_requests):
    expected=list(range(expected_requests))
    if (len(rows)!=expected_requests or sorted(int(r["sample_idx"]) for r in rows)!=expected
            or sorted(int(r["condition_ordinal"]) for r in rows)!=expected
            or len({r["trajectory_id"] for r in rows})!=expected_requests):
        raise ValueError("extraction must retain the entire frozen original request population")
    return [extract_one(row,tokenizer,constraints,method) for row in sorted(rows,key=lambda r:r["condition_ordinal"])]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--paths-jsonl",type=Path,required=True)
    p.add_argument("--checkpoint-path",type=Path,required=True)
    p.add_argument("--method",choices=("k4","k8"),required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--expected-requests",type=int,default=256)
    p.add_argument("--source-sha256")
    p.add_argument("--input-manifest",type=Path,default=ROOT/"docs/v3_scientific_audit_20260906/evidence_v3_failure_20260907/k4k8_prefix_inputs.json")
    args=p.parse_args()
    if args.expected_requests<1:raise ValueError("expected request count must be positive")
    if args.output_dir.resolve().is_relative_to(args.paths_jsonl.parent.resolve()):
        raise ValueError("write the extracted endpoint outside the immutable source sample directory")
    expected_hash=args.source_sha256
    if expected_hash is None:
        manifest=json.loads(args.input_manifest.read_text())
        binding=manifest["methods"][args.method]
        if (Path(binding["source_path"]).resolve()!=args.paths_jsonl.resolve()
                or Path(binding["checkpoint"]).resolve()!=args.checkpoint_path.resolve()):
            raise ValueError("new source needs an explicit --source-sha256 from its frozen manifest")
        expected_hash=manifest["source_receipts"][binding["source_path"]]
    actual_hash=sha256(args.paths_jsonl)
    if actual_hash!=expected_hash:raise ValueError("original full-path source hash differs")
    if not (args.paths_jsonl.parent/"_SUCCESS").is_file():raise ValueError("original full-path ledger is incomplete")
    rows=read_jsonl(args.paths_jsonl)
    if any(Path(r["checkpoint"]).resolve()!=args.checkpoint_path.resolve() for r in rows):
        raise ValueError("input rows do not belong to the requested original checkpoint")
    from crystal_dlm.short_contact_sampling import validate_legacy_contact_policy
    policy=validate_legacy_contact_policy(args.checkpoint_path)
    if policy.get("collection_round")!={"k4":0,"k8":1}[args.method]:
        raise ValueError("requested method name differs from the original policy round")
    from transformers import AutoTokenizer
    from scripts.sample_llada_dynamic_crystals import build_dynamic_lightweight_constraints
    import torch
    torch.set_num_threads(2)
    tokenizer=AutoTokenizer.from_pretrained(args.checkpoint_path,trust_remote_code=True,local_files_only=True)
    constraints=build_dynamic_lightweight_constraints(tokenizer,duplicate_coordinate_mask=True,lattice_volume_mask=True,
        min_lattice_rad=1e-4,canonicalize_periodic_alias=True,pbc_min_distance_mask=True,pbc_min_distance_A=.5,pbc_image_radius=2)
    result=extract_rows(rows,tokenizer,constraints,args.method,args.expected_requests)
    if sha256(args.paths_jsonl)!=actual_hash:raise ValueError("source file changed during extraction")
    args.output_dir.mkdir(parents=True,exist_ok=False)
    with (args.output_dir/"paths.jsonl").open("x",encoding="utf-8") as stream:
        for row in result:stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+"\n")
    report={"method":f"{args.method}_construction_only_v1","requests":len(result),"successful":sum(r["success"] for r in result),
        "failed":sum(not r["success"] for r in result),"source":str(args.paths_jsonl),"source_sha256":actual_hash,
        "checkpoint":str(args.checkpoint_path),"omitted_later_events":sum(r["omitted_later_events"] for r in result),
        "failure_reasons":dict(Counter(r["trace"]["failure"] for r in result if not r["success"])),
        "uniform_phase_ablation":True,"outcome_selection":False,"all_failures_retained":True,
        "model_forwards":0,"extra_DLM_samples":0,"optimizer_steps":0,"energy_evaluations":0}
    (args.output_dir/"CONSTRUCTION_EXTRACTION.json").write_text(json.dumps(report,indent=2)+"\n")
    (args.output_dir/"_SUCCESS").touch();print(json.dumps(report),flush=True)


if __name__=="__main__":main()
