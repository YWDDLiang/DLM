#!/usr/bin/env python3
"""Real K4/K8 prefix counterfactuals while replaying unchanged complete batches."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from crystal_dlm.programmed_path_data import compile_condition, load_path_model, read_jsonl
from crystal_dlm.sampling_layout import sampling_batches
from crystal_dlm.short_contact_sampling import ContactSpec, validate_legacy_contact_policy
from scripts.sample_state_programmed_paths import make_sampler


def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):h.update(chunk)
    return h.hexdigest()


def bind_trace(actual, expected, tolerance):
    if actual["success"]!=expected["success"] or actual["failure"]!=expected["failure"]:
        raise RuntimeError("baseline replay changed construction/revision failure outcome")
    if actual["initial_body"]!=expected["initial_body"] or len(actual["events"])!=len(expected["events"]):
        raise RuntimeError("baseline replay changed the attempt ledger")
    maximum=0.
    semantic=("op","phase","kind","position","token","positions","reason","salt")
    for a,b in zip(actual["events"],expected["events"],strict=True):
        if any(a.get(key)!=b.get(key) for key in semantic):
            raise RuntimeError("baseline replay changed an original seeded scalar action or rollback")
        if a["op"]=="draw":
            error=abs(a["log_probability"]-b["log_probability"])
            if error>tolerance:
                raise RuntimeError(f"baseline prefix log probability changed: {error}")
            maximum=max(maximum,error)
    return maximum


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method",choices=("k4","k8"),required=True)
    p.add_argument("--model-path",required=True)
    p.add_argument("--input-manifest",type=Path,required=True)
    p.add_argument("--short-contact-spec",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--reference-world-size",type=int,default=2)
    p.add_argument("--batch-size",type=int,default=4)
    p.add_argument("--first-conditions",type=int,default=4)
    p.add_argument("--replay-tolerance",type=float,default=1e-6)
    args=p.parse_args()
    if args.first_conditions!=4 or args.reference_world_size<1 or args.batch_size<1:
        raise ValueError("this prefix probe uses the fixed first four original conditions")
    manifest=json.loads(args.input_manifest.read_text())
    data=manifest["methods"][args.method]
    source=Path(data["source_path"])
    if digest(source)!=manifest["source_receipts"][str(source)]:
        raise ValueError("actual complete sample source changed from the frozen prefix receipt")
    if not (source.parent/"_SUCCESS").is_file():
        raise ValueError("original sample ledger is incomplete")
    validate_legacy_contact_policy(data["checkpoint"])
    spec=ContactSpec.load(args.short_contact_spec)
    records=read_jsonl(source)
    lookup={int(row["condition_ordinal"]):row for row in records}
    if len(lookup)!=len(records):raise ValueError("duplicate original request identity")
    wanted={int(row["condition_ordinal"]) for row in data["first_four"]}
    if wanted!={int(row["condition_ordinal"]) for row in sorted(records,key=lambda r:r["condition_ordinal"])[:4]}:
        raise ValueError("probe subset changed from the original first four")
    for row in data["first_four"]:
        if row!=lookup[int(row["condition_ordinal"])]:
            raise ValueError("saved prefix package differs from the complete original row")
    import torch
    from scripts.sample_llada_dynamic_crystals import build_dynamic_lightweight_constraints
    if not torch.cuda.is_available():raise RuntimeError("real prefix probing requires its allocated CUDA device")
    sample_meta=next(value for key,value in data["metadata"].items() if key.endswith("/SAMPLE_FINAL.json"))
    local=int(os.environ.get("LOCAL_RANK",0));torch.cuda.set_device(local);torch.set_num_threads(2)
    torch.manual_seed(int(sample_meta["seed"]))
    device=torch.device("cuda",local)
    model,tokenizer=load_path_model(args.model_path,data["checkpoint"],device,trainable=False)
    if hasattr(model,"raw_initialization"):raise ValueError("probe expected legacy K4/K8 state-conditioned model")
    constraints=build_dynamic_lightweight_constraints(tokenizer,duplicate_coordinate_mask=True,lattice_volume_mask=True,
        min_lattice_rad=1e-4,canonicalize_periodic_alias=True,pbc_min_distance_mask=True,pbc_min_distance_A=.5,pbc_image_radius=2)
    items=[]
    for row in sorted(records,key=lambda r:r["condition_ordinal"]):
        c=compile_condition(row,tokenizer,mask_id=126336,purpose="evaluation")
        if c["prompt_token_ids"]!=row["prompt_token_ids"] or c["initial_body"]!=row["trace"]["initial_body"]:
            raise ValueError("tokenizer or fixed composition changed the recorded prefix")
        items.append((int(row["condition_ordinal"]),int(row["candidate_index"]),c))
    selected=[]
    for batch in sampling_batches(items,batch_size=args.batch_size,rank=0,world_size=1,
                                  layout_world_size=args.reference_world_size):
        if any(item[2]["record"]["sampling_batch_size"]!=len(batch) for item in batch):
            raise ValueError("legacy modulo layout does not reconstruct recorded full batch sizes")
        if any(item[0] in wanted for item in batch):selected.append(batch)
    args.output_dir.mkdir(parents=True,exist_ok=False)
    bindings=[];vectors=[];started=time.monotonic();forwards=0
    for batch in selected:
        compiled=[item[2] for item in batch];seeds=[c["record"]["sampling_seed"] for c in compiled]
        sampler=make_sampler(model,tokenizer,constraints,compiled,seeds,.7,spec,probe_only=True,collect_vectors=True)
        x=torch.tensor([c["prompt_token_ids"]+c["initial_body"] for c in compiled],device=device)
        calls=[0]
        def count(_m,_a):calls[0]+=1
        handle=model.register_forward_pre_hook(count)
        try:
            with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16):
                final,traces=sampler.run(x,torch.ones_like(x),construct=True,cooperative=True,closure=True,full_cell_repair=False)
        finally:handle.remove()
        forwards+=calls[0]
        for row,(_,_,c) in enumerate(batch):
            original=c["record"];maximum=bind_trace(traces[row],original["trace"],args.replay_tolerance)
            if final[row,len(c["prompt_token_ids"]):].tolist()!=original["final_body_token_ids"]:
                raise RuntimeError("unchanged-policy probe changed the original attempted terminal body")
            bindings.append({"sample_idx":original["sample_idx"],"trajectory_id":original["trajectory_id"],
                "requested_probe":int(original["condition_ordinal"]) in wanted,"batch_ordinals":[item[0] for item in batch],
                "success":original["success"],"failure":original["trace"]["failure"],"maximum_logp_error":maximum,
                "model_forward_calls_for_batch":calls[0]})
        for record in sampler.contact_diagnostics:
            original=compiled[record["row"]]["record"]
            if int(original["condition_ordinal"]) in wanted:
                vectors.append({**record,"sample_idx":original["sample_idx"],"trajectory_id":original["trajectory_id"],
                                "baseline_sampling_seed":original["sampling_seed"]})
        print(json.dumps({"bound_requests_including_companions":len(bindings),"Z_prefixes":len(vectors),
                          "elapsed_seconds":time.monotonic()-started}),flush=True)
    with (args.output_dir/"prefix_vectors.jsonl").open("x") as stream:
        for row in vectors:stream.write(json.dumps(row,allow_nan=False)+"\n")
    if not vectors or not any(r["positive_candidates"]>0 for r in vectors):
        raise RuntimeError("fixed real prefix probe did not exercise a nonzero soft penalty")
    if any(not r["hard_support_unchanged"] or not r["penalty_in_unit_interval"]
           or abs(r["baseline_probability_sum"]-1)>1e-12 or abs(r["tilted_probability_sum"]-1)>1e-12 for r in vectors):
        raise RuntimeError("real prefix support, bound or normalization check failed")
    report={"status":"PASS","method":args.method,"checkpoint":data["checkpoint"],"probe_only":True,
            "source":str(source),"source_sha256":digest(source),"input_manifest_sha256":digest(args.input_manifest),
            "short_contact_policy":spec.metadata(),"requested_first_conditions":sorted(wanted),"bindings":bindings,
            "Z_prefixes":len(vectors),"positive_penalty_prefixes":sum(r["positive_candidates"]>0 for r in vectors),
            "Z_prefixes_by_phase":dict(Counter(r["phase"] for r in vectors)),
            "changed_same_seed_tokens":sum(r["baseline_selected_token"]!=r["tilted_selected_token"] for r in vectors),
            "zero_penalty_winners":sum(r["baseline_winner_penalty"]==0 for r in vectors),
            "zero_penalty_winner_violations":sum(r["zero_penalty_winner_preserved"] is False for r in vectors),
            "maximum_logp_error":max(r["maximum_logp_error"] for r in bindings),"model_forward_calls":forwards,
            "extra_counterfactual_model_forwards":0,"new_candidates":0,"energy_evaluations":0,"optimizer_steps":0,
            "sampling_layout_world_size_in_original":None,"effective_legacy_layout":args.reference_world_size,
            "elapsed_seconds":time.monotonic()-started}
    (args.output_dir/"PROBE.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    (args.output_dir/"_SUCCESS").touch();print(json.dumps(report,allow_nan=False),flush=True)


if __name__=="__main__":main()
