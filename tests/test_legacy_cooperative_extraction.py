from contextlib import nullcontext
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from crystal_dlm.programmed_path_data import compile_condition,trace_terminal_body
from scripts.sample_state_programmed_paths import make_sampler,parse_args,main as live_main,replay
from test_legacy_construction_extraction import environment,recorded
from test_legacy_phase_geometry import full_trace
from test_state_programmed_runtime import body,PreferredModel

SPEC=importlib.util.spec_from_file_location("legacy_coop_extract_test",ROOT/"scripts/extract_legacy_cooperative_endpoint.py")
COOP=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(COOP)


def test_exact_cooperative_prefix_ignores_later_closure_and_failure(environment):
    tok,support=environment;row=full_trace(tok)
    row["trace"]["events"]=[e for e in row["trace"]["events"] if e["op"]!="rollback"]
    row["final_body_token_ids"]=trace_terminal_body(row["trace"])
    row["success"]=False;row["trace"].update(success=False,failure="later failure")
    saved=deepcopy(row);out=COOP.extract_one(row,tok,support,"k4")
    end=next(i for i,e in enumerate(row["trace"]["events"]) if e["op"]=="end" and e["phase"]=="cooperative")
    assert out["trace"]["events"]==row["trace"]["events"][:end+1]
    assert out["success"] and out["method"]=="k4_cooperative_endpoint_v1" and not out["full_cycle_success"]
    assert out["final_body_token_ids"][14]==tok.vocab["<Z_025>"] and row["final_body_token_ids"][14]==tok.vocab["<Z_050>"]
    assert row==saved and out["sampling_seed"]==row["sampling_seed"]


def test_cooperative_no_support_rollback_is_successful_supported_endpoint(environment):
    tok,support=environment;row=full_trace(tok);events=row["trace"]["events"]
    begin=next(i for i,e in enumerate(events) if e["op"]=="begin" and e["phase"]=="cooperative")
    end=next(i for i,e in enumerate(events) if e["op"]=="end" and e["phase"]=="cooperative")
    positions=events[begin]["positions"]
    row["trace"]["events"]=events[:begin+1]+[{"op":"no_support","phase":"cooperative","position":positions[0]},
        {"op":"rollback","positions":positions,"reason":"cooperative_empty_support"}]+events[end:]
    row["final_body_token_ids"]=trace_terminal_body(row["trace"])
    out=COOP.extract_one(row,tok,support,"k8")
    assert out["success"] and out["parseable"] and out["cooperative_executed"]
    assert out["final_body_token_ids"]==body(tok,count=2,length=40,x=50)
    assert out["trace_summary"]["rollback_reasons"]["cooperative_empty_support"]==1


def test_construct_failure_retained_and_missing_or_duplicate_coop_boundaries_rejected(environment):
    tok,support=environment
    failed=COOP.extract_one(recorded(tok,case="construct_failure"),tok,support,"k8")
    assert not failed["success"] and not failed["cooperative_executed"] and failed["parseable"] is False
    assert failed["method"]=="k8_cooperative_endpoint_v1" and tok.mask_id in failed["final_body_token_ids"]
    with pytest.raises(ValueError,match="one complete"):
        COOP.extract_one(recorded(tok),tok,support,"k4")
    row=full_trace(tok);row["trace"]["events"] += [{"op":"begin","phase":"cooperative","kind":"cell_sites","positions":[14]},
                                                  {"op":"end","phase":"cooperative"}]
    with pytest.raises(ValueError,match="one complete"):
        COOP.extract_one(row,tok,support,"k4")


def test_live_flag_defaults_off_and_replays_construct_plus_coop(environment,tmp_path,monkeypatch):
    tok,support=environment
    argv=["sample","--model-path","raw","--checkpoint-path","fixture","--conditions","conditions","--output-dir","out","--purpose","evaluation"]
    monkeypatch.setattr(sys,"argv",argv);assert not parse_args().cooperative_only
    for extra in (["--cooperative-only","--construction-only"],["--cooperative-only","--temperature",".8"]):
        monkeypatch.setattr(sys,"argv",argv+extra)
        with pytest.raises(ValueError,match="cooperative-only"):
            live_main()
    row=recorded(tok);c=compile_condition(row,tok,mask_id=tok.mask_id,purpose="evaluation")
    model=PreferredModel(tok,body(tok,count=2,length=40,x=50)).eval()
    model.geometry_inputs=lambda context:{}
    model.state_conditioner=lambda **kwargs:{"cell_embedding":torch.ones(1),"site_embeddings":torch.ones(1)}
    sampler=make_sampler(model,tok,support,[c],[row["sampling_seed"]],.7)
    x=torch.tensor([[0]+c["initial_body"]]);result,traces=sampler.run(x,torch.ones_like(x),construct=True,cooperative=True,closure=False)
    row.update(trace=traces[0],final_body_token_ids=result[0,1:].tolist(),method="k4_cooperative_endpoint_v1")
    path=tmp_path/"paths.jsonl";path.write_text(json.dumps(row)+"\n")
    args=SimpleNamespace(replay_jsonl=path,purpose="evaluation",checkpoint_path="fixture",output_dir=tmp_path/"replay",
        replay_tolerance=1e-6,cooperative_only=True,_cooperative_method="k4_cooperative_endpoint_v1")
    monkeypatch.setattr(torch,"autocast",lambda *a,**kw:nullcontext())
    replay(args,model,tok,support,"cpu")
    report=json.loads((args.output_dir/"REPLAY.json").read_text())
    assert report["all_recorded_decisions_checked"]==24 and report["trained_state_residual_l1"]>0
    assert {e["phase"] for e in row["trace"]["events"] if e["op"]!="rollback"}=={"construct","cooperative"}
