#!/usr/bin/env python3
"""Read-only geometry of three recorded legacy phases; no stage selection or NN."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"));sys.path.insert(0,str(ROOT/"scripts"))
from extract_legacy_construction_endpoint import extract_one,sha256
from crystal_dlm.dynamic_crystal import arrays_to_structure
from crystal_dlm.programmed_path_data import compile_condition,read_jsonl,validate_completed_body
from crystal_dlm.programmed_path_runtime import complete_geometry_supported
from crystal_dlm.spad_program import reverse_species_block_revision_slots,coordinate_positions

PHASES=("construct","cooperative","closure")


def periodic_minimum(structure):
    """Full pair MIC plus shortest nonzero lattice translation, not only 125 images."""
    reduced=structure.lattice.get_lll_reduced_lattice()
    bound=min(reduced.abc)
    _points,distances,_ids,images=reduced.get_points_in_sphere(
        [[0.,0.,0.]],[0.,0.,0.],bound*(1+1e-12)+1e-10,zip_results=False)
    nonzero=np.any(np.rint(images)!=0,axis=1)
    if not np.any(nonzero):raise RuntimeError("periodic self-image search did not include a basis translation")
    self_min=float(np.asarray(distances)[nonzero].min())
    pair_min=None
    if len(structure)>1:
        distances=np.asarray(structure.distance_matrix).copy()
        np.fill_diagonal(distances,np.inf)
        pair_min=float(distances.min())
    return {"minimum_A":min(self_min,pair_min) if pair_min is not None else self_min,
            "distinct_atom_minimum_A":pair_min,"periodic_self_minimum_A":self_min,
            "self_search_upper_bound_A":bound,"self_images_in_bound":int(nonzero.sum())}


def stage_geometry(ids,tokenizer,compiled,constraints):
    body=tokenizer.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
    output={"token_ids":list(ids),"body":body,"schema_valid":False,"support125_valid":False,
            "geometry":None,"geometry_error":None}
    try:arrays=validate_completed_body(body,compiled)
    except ValueError as error:
        output["geometry_error"]=f"schema: {error}";return output
    output["schema_valid"]=True
    output["support125_valid"]=bool(complete_geometry_supported(torch.tensor(ids,dtype=torch.long),constraints))
    structure=arrays_to_structure(arrays)
    matrix=np.asarray(structure.lattice.matrix)
    if not np.isfinite(matrix).all() or not np.isfinite(structure.frac_coords).all() or structure.volume<=0:
        output["geometry_error"]="nonfinite or degenerate decoded geometry";return output
    distance=periodic_minimum(structure)
    output["geometry"]={"structure":structure.as_dict(),"periodic_distance":distance,
        "volume_per_atom_A3":structure.volume/len(structure),"volume_A3":structure.volume,
        "lattice_condition_number":float(np.linalg.cond(matrix)),
        "below_common_0p5_A":distance["minimum_A"]<.5-1e-8}
    return output


def analyze_one(record,tokenizer,constraints,method):
    constructor=extract_one(record,tokenizer,constraints,method)
    compiled=compile_condition(record,tokenizer,mask_id=record["trace"]["mask_id"],purpose="evaluation")
    events=record["trace"]["events"];body=list(record["trace"]["initial_body"])
    snapshot=None;phase=None;active_positions=[];ends={p:[] for p in PHASES};counts={p:Counter() for p in PHASES}
    rollback_reasons={p:Counter() for p in PHASES};transaction_rolled_back=False;last_phase=-1;closure_index=0
    closure_blocks=reverse_species_block_revision_slots(compiled["program"])
    for index,event in enumerate(events):
        op=event["op"]
        if op=="begin":
            if snapshot is not None:raise ValueError("nested phase transaction")
            phase=event.get("phase")
            if phase not in PHASES or PHASES.index(phase)<last_phase:raise ValueError("unexpected or reversed phase")
            if phase!="closure" and counts[phase]["transactions"]:raise ValueError("duplicated single phase")
            last_phase=PHASES.index(phase);active_positions=list(event["positions"])
            if len(set(active_positions))!=len(active_positions):raise ValueError("duplicate transaction positions")
            if phase=="closure":
                if closure_index>=len(closure_blocks):raise ValueError("extra closure block")
                expected=[p for site in closure_blocks[closure_index] for p in coordinate_positions(site)]
                if active_positions!=expected:raise ValueError("closure block differs from original reverse P")
                closure_index+=1
            snapshot=body.copy();transaction_rolled_back=False
            for pos in active_positions:
                if pos<0 or pos>=len(body):raise ValueError("transaction position out of canvas")
                body[pos]=int(record["trace"]["mask_id"])
            counts[phase]["transactions"]+=1
        elif op in {"draw","no_support"}:
            if snapshot is None or event.get("phase")!=phase:raise ValueError("action lacks matching phase begin")
            pos=int(event["position"])
            if pos<0 or pos>=len(body) or (phase!="construct" and pos not in active_positions):raise ValueError("action outside its transaction")
            counts[phase][op]+=1
            if op=="draw":body[pos]=int(event["token"])
        elif op=="rollback":
            if snapshot is None:raise ValueError("rollback without an active snapshot")
            positions=list(event["positions"])
            if not set(positions)<=set(active_positions):raise ValueError("rollback exceeds its recorded transaction")
            for pos in positions:body[pos]=snapshot[pos]
            counts[phase]["rollback_events"]+=1;counts[phase]["restored_scalar_writes"]+=len(positions)
            rollback_reasons[phase][event["reason"]]+=1;transaction_rolled_back=True
        elif op=="end":
            if snapshot is None or event.get("phase")!=phase:raise ValueError("phase end lacks matching begin")
            counts[phase]["transactions_with_rollback"]+=int(transaction_rolled_back)
            ends[phase].append((index,body.copy()));snapshot=None;phase=None
        else:raise ValueError(f"unknown trace operation: {op}")
    if snapshot is not None:raise ValueError("unfinished phase transaction")
    if len(ends["construct"])!=1 or ends["construct"][0][1]!=constructor["final_body_token_ids"]:
        raise ValueError("construction boundary replay differs")
    if body!=record["final_body_token_ids"]:raise ValueError("full event replay differs from original final body")
    if constructor["success"]:
        if len(ends["cooperative"])!=1 or len(ends["closure"])!=len(closure_blocks):
            raise ValueError("successful full path is missing cooperative or closure metadata")
    elif ends["cooperative"] or ends["closure"]:
        raise ValueError("failed constructor unexpectedly entered later phases")
    stages={}
    for name in PHASES:
        if not ends[name]:
            stages[name]={"executed":False,"status":"not_executed_after_construct_failure","valid":False,
                          "token_ids":None,"body":None,"geometry":None,"counts":dict(counts[name])}
            continue
        end_index,ids=ends[name][-1]
        stage=stage_geometry(ids,tokenizer,compiled,constraints)
        valid=stage["schema_valid"] and stage["support125_valid"]
        if name=="construct":valid=valid and constructor["success"]
        if name!="construct" and not valid:raise RuntimeError(f"{name} end violates the original rollback support guarantee")
        stages[name]={**stage,"executed":True,"status":"valid" if valid else "failed_constructor",
                      "valid":valid,"end_event_index":end_index,"recorded_end_count":len(ends[name]),
                      "counts":dict(counts[name]),"rollback_reasons":dict(rollback_reasons[name])}
    transitions={}
    for left,right in zip(PHASES,PHASES[1:]):
        a,b=stages[left],stages[right]
        pair={"available":a["executed"] and b["executed"]}
        if pair["available"]:
            pair.update(exact_tokens_unchanged=a["token_ids"]==b["token_ids"],
                        changed_scalar_tokens=sum(x!=y for x,y in zip(a["token_ids"],b["token_ids"],strict=True)))
            if a["geometry"] is not None and b["geometry"] is not None:
                first,last=a["geometry"],b["geometry"]
                pair.update(minimum_distance_delta_A=last["periodic_distance"]["minimum_A"]-first["periodic_distance"]["minimum_A"],
                            volume_per_atom_ratio=last["volume_per_atom_A3"]/first["volume_per_atom_A3"],
                            introduced_below_0p5=not first["below_common_0p5_A"] and last["below_common_0p5_A"],
                            removed_below_0p5=first["below_common_0p5_A"] and not last["below_common_0p5_A"])
        transitions[left+"_to_"+right]=pair
    return {"sample_idx":record["sample_idx"],"condition_ordinal":record["condition_ordinal"],
            "trajectory_id":record["trajectory_id"],"group_id":record["group_id"],"sampling_seed":record["sampling_seed"],
            "plan_state":record["plan_state"],"species_program":record["species_program"],"method":method,
            "metadata_complete":True,"recorded_full_cycle_success":record["success"],"stages":stages,"transitions":transitions}


def describe(values):
    a=np.asarray(list(values),dtype=np.float64)
    return {"n":int(len(a)),**({"mean":float(a.mean()),"median":float(np.median(a)),"p10":float(np.quantile(a,.1)),
                               "p90":float(np.quantile(a,.9)),"min":float(a.min()),"max":float(a.max())} if len(a) else {})}


def analyze_rows(records,tokenizer,constraints,method):
    result=[]
    for row in records:
        try:result.append(analyze_one(row,tokenizer,constraints,method))
        except Exception as error:
            result.append({"sample_idx":row.get("sample_idx"),"trajectory_id":row.get("trajectory_id"),
                           "metadata_complete":False,"error":f"{type(error).__name__}: {error}","stages":None,"transitions":None})
    valid=[r for r in result if r["metadata_complete"]]
    summary={"status":"PASS" if len(valid)==len(result) else "FAIL","requests":len(result),"metadata_errors":len(result)-len(valid),
             "stages":{},"transitions":{},"model_forwards":0,"energy_evaluations":0,"optimizer_steps":0,
             "SUN_claim":False,"per_case_stage_selection":False,"distance_comparison_tolerance_A":1e-8}
    for name in PHASES:
        stages=[r["stages"][name] for r in valid];geometry=[s["geometry"] for s in stages if s.get("geometry") is not None]
        summary["stages"][name]={"known_requests":len(stages),"executed":sum(s["executed"] for s in stages),
            "valid":sum(s["valid"] for s in stages),"failed":sum(s["executed"] and not s["valid"] for s in stages),
            "not_executed":sum(not s["executed"] for s in stages),"unknown_due_metadata_error":len(result)-len(valid),
            "rollback_events":sum(s["counts"].get("rollback_events",0) for s in stages),
            "requests_with_rollback":sum(s["counts"].get("rollback_events",0)>0 for s in stages),
            "transactions_with_rollback":sum(s["counts"].get("transactions_with_rollback",0) for s in stages),
            "minimum_distance_A":describe(g["periodic_distance"]["minimum_A"] for g in geometry),
            "volume_per_atom_A3":describe(g["volume_per_atom_A3"] for g in geometry),
            "below_0p5_A":sum(g["below_common_0p5_A"] for g in geometry)}
    for name in ("construct_to_cooperative","cooperative_to_closure"):
        pairs=[r["transitions"][name] for r in valid if r["transitions"][name]["available"]]
        changes=[r["minimum_distance_delta_A"] for r in pairs if "minimum_distance_delta_A" in r]
        summary["transitions"][name]={"pairs":len(pairs),"exact_tokens_unchanged":sum(p["exact_tokens_unchanged"] for p in pairs),
            "minimum_distance_delta_A":describe(changes),"distance_decreased":sum(v < -1e-8 for v in changes),
            "distance_increased":sum(v > 1e-8 for v in changes),
            "volume_per_atom_ratio":describe(p["volume_per_atom_ratio"] for p in pairs if "volume_per_atom_ratio" in p),
            "introduced_below_0p5":sum(p.get("introduced_below_0p5",False) for p in pairs),
            "removed_below_0p5":sum(p.get("removed_below_0p5",False) for p in pairs)}
    return result,summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--paths-jsonl",type=Path,required=True);p.add_argument("--checkpoint-path",type=Path,required=True)
    p.add_argument("--method",choices=("k4","k8"),required=True);p.add_argument("--source-sha256",required=True)
    p.add_argument("--expected-requests",type=int,default=256);p.add_argument("--output-dir",type=Path,required=True)
    args=p.parse_args()
    if sha256(args.paths_jsonl)!=args.source_sha256:raise ValueError("original full source hash differs")
    if not (args.paths_jsonl.parent/"_SUCCESS").is_file():raise ValueError("original full sample is incomplete")
    if args.output_dir.resolve().is_relative_to(args.paths_jsonl.parent.resolve()):raise ValueError("write outside the original sample directory")
    records=read_jsonl(args.paths_jsonl)
    if len(records)!=args.expected_requests or sorted(int(r["sample_idx"]) for r in records)!=list(range(args.expected_requests)):
        raise ValueError("audit must retain the whole original cohort")
    if any(Path(r["checkpoint"]).resolve()!=args.checkpoint_path.resolve() for r in records):raise ValueError("source checkpoint differs")
    from crystal_dlm.short_contact_sampling import validate_legacy_contact_policy
    policy=validate_legacy_contact_policy(args.checkpoint_path)
    if policy["collection_round"]!={"k4":0,"k8":1}[args.method]:raise ValueError("method differs from original policy")
    from transformers import AutoTokenizer
    from scripts.sample_llada_dynamic_crystals import build_dynamic_lightweight_constraints
    torch.set_num_threads(2)
    tokenizer=AutoTokenizer.from_pretrained(args.checkpoint_path,trust_remote_code=True,local_files_only=True)
    constraints=build_dynamic_lightweight_constraints(tokenizer,duplicate_coordinate_mask=True,lattice_volume_mask=True,
        min_lattice_rad=1e-4,canonicalize_periodic_alias=True,pbc_min_distance_mask=True,pbc_min_distance_A=.5,pbc_image_radius=2)
    rows,summary=analyze_rows(sorted(records,key=lambda r:r["condition_ordinal"]),tokenizer,constraints,args.method)
    summary.update(source=str(args.paths_jsonl),source_sha256=args.source_sha256,checkpoint=str(args.checkpoint_path),method=args.method,
                   distance_protocol="pymatgen full pair MIC plus nonzero translation sphere bounded by shortest LLL basis vector")
    if sha256(args.paths_jsonl)!=args.source_sha256:raise ValueError("original source changed during audit")
    args.output_dir.mkdir(parents=True,exist_ok=False)
    (args.output_dir/"per_request.jsonl").write_text("".join(json.dumps(r,allow_nan=False)+"\n" for r in rows))
    (args.output_dir/"SUMMARY.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    passed=summary["status"]=="PASS";(args.output_dir/("_SUCCESS" if passed else "_FAILED")).touch()
    print(json.dumps(summary,allow_nan=False),flush=True)
    if not passed:raise SystemExit(1)


if __name__=="__main__":main()
