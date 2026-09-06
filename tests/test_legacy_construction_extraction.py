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
from crystal_dlm import fixed_slot
from crystal_dlm.programmed_path_data import compile_condition,path_seed,trace_terminal_body
from crystal_dlm.spad_program import spad_predictor_position_groups
from scripts.sample_state_programmed_paths import make_sampler,replay,parse_args,main as live_main
from test_periodic_v2_training_data import codec
from test_state_programmed_runtime import body,PreferredModel

SPEC=importlib.util.spec_from_file_location("legacy_construction_extract_test",ROOT/"scripts/extract_legacy_construction_endpoint.py")
EXTRACT=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(EXTRACT)


@pytest.fixture
def environment(monkeypatch):
    tok,support=codec()
    monkeypatch.setattr(type(tok),"__call__",lambda self,text,**kw:{"input_ids":[0]},raising=False)
    monkeypatch.setattr(type(tok),"decode",lambda self,ids,**kw:"".join({v:k for k,v in self.vocab.items()}[int(i)] for i in ids),raising=False)
    monkeypatch.setattr(fixed_slot,"MASK_TOKEN_ID",tok.mask_id)
    return tok,support


def recorded(tok,index=0,case="success"):
    row={"source_split":"evaluation","source_row_idx":index,"sample_idx":index,"condition_ordinal":index,
         "group_id":f"eval:{index}","trajectory_id":f"eval:{index}:0:0","prompt":"fixed condition\n",
         "prompt_token_ids":[0],"plan_state":{"N":2,"elements":["H"],"counts":[2]},
         "species_program":["H"],"species_program_source":"frozen_pointer","sampling_seed":path_seed(20260905,f"eval:{index}",0,0),
         "sampling_batch_size":1,"sampling_layout_world_size":None,"collection_round":0,"candidate_index":0,"checkpoint":"fixture"}
    c=compile_condition(row,tok,mask_id=tok.mask_id,purpose="evaluation")
    target=body(tok,count=2,length=40,x=50 if case!="unsupported" else 10)
    schedule=[g[0] for g in spad_predictor_position_groups(c["program"])[1:]]
    events=[{"op":"begin","phase":"construct","kind":"construct","positions":[]}]
    for step,pos in enumerate(schedule):
        if case=="construct_failure" and step==4:
            events.append({"op":"no_support","position":pos,"phase":"construct"});break
        events.append({"op":"draw","phase":"construct","position":pos,"token":target[pos],
                       "salt":100000000+10007*step,"log_probability":-.123-step*.001})
    events.append({"op":"end","phase":"construct"})
    if case=="later_failure":
        events += [{"op":"begin","phase":"cooperative","kind":"cell_sites","positions":[14]},
                   {"op":"draw","phase":"cooperative","position":14,"token":tok.vocab["<Z_025>"],"salt":200000000,"log_probability":-.4},
                   {"op":"rollback","positions":[14],"reason":"cooperative_empty_support"},
                   {"op":"end","phase":"cooperative"}]
    trace={"schema":"programmed_crystal_attempt_path_v1","initial_body":c["initial_body"],"mask_id":tok.mask_id,
           "temperature":.7,"element_order":["H"],"events":events,"success":case=="success",
           "failure":None if case=="success" else "later_or_original_failure"}
    row.update(trace=trace,success=trace["success"],final_body_token_ids=trace_terminal_body(trace),
               body="old full-cycle preview",structure={"later":"geometry"},parseable=False)
    return row


def test_constructor_success_ignores_later_failure_and_preserves_exact_prefix(environment,monkeypatch):
    tok,support=environment;source=recorded(tok,case="later_failure");saved=deepcopy(source)
    out=EXTRACT.extract_one(source,tok,support,"k4")
    assert out["method"]=="k4_construction_only_v1" and out["success"] and out["parseable"]
    assert not out["full_cycle_success"] and out["trace"]["failure"] is None and out["failure"] is None
    assert out["trace"]["events"]==source["trace"]["events"][:14]
    assert out["final_body_token_ids"]==body(tok,count=2,length=40,x=50)
    assert out["structure"] is None and source==saved
    for key in ("prompt","prompt_token_ids","plan_state","species_program","species_program_source","sampling_seed","condition_ordinal"):
        assert out[key]==source[key]
    monkeypatch.setattr(EXTRACT,"complete_geometry_supported",lambda *a:(_ for _ in ()).throw(RuntimeError("fixture support interface error")))
    with pytest.raises(RuntimeError,match="fixture support interface"):
        EXTRACT.extract_one(source,tok,support,"k4")


def test_constructor_failure_retains_masks_without_rescue(environment):
    tok,support=environment;source=recorded(tok,case="construct_failure")
    out=EXTRACT.extract_one(source,tok,support,"k8")
    assert not out["success"] and out["parseable"] is False and out["trace"]["failure"]=="construct_empty_support"
    assert tok.mask_id in out["final_body_token_ids"] and "<MASK>" in out["body"]
    assert out["diagnostic_candidate"]["final_body_token_ids"]==out["final_body_token_ids"]
    assert out["trace"]["events"]==source["trace"]["events"]
    source["trace"]["temperature"]=.8
    with pytest.raises(ValueError,match="temperature"):
        EXTRACT.extract_one(source,tok,support,"k8")


def test_complete_but_unsupported_constructor_is_not_reaccepted(environment):
    tok,support=environment;out=EXTRACT.extract_one(recorded(tok,case="unsupported"),tok,support,"k4")
    assert out["construction_check"]["schema_pass"] and not out["construction_check"]["complete_geometry_supported"]
    assert not out["success"] and not out["parseable"] and out["trace"]["failure"]=="unsupported_predictor"
    impossible=recorded(tok,case="unsupported")
    impossible["trace"]["events"] += [{"op":"begin","phase":"closure","kind":"species_block","positions":[14]},
                                         {"op":"end","phase":"closure"}]
    with pytest.raises(RuntimeError,match="support replay disagrees"):
        EXTRACT.extract_one(impossible,tok,support,"k4")


def test_complete256_ledger_and_exact_64bit_seed_with_no_source_mutation(environment):
    tok,support=environment
    rows=[recorded(tok,i,"construct_failure" if i%4==0 else "later_failure" if i%4==1 else "success") for i in range(256)]
    before=json.dumps(rows,sort_keys=True)
    output=EXTRACT.extract_rows(rows,tok,support,"k8",256)
    assert len(output)==256 and sum(r["success"] for r in output)==192
    assert [r["sample_idx"] for r in output]==list(range(256))
    assert output[0]["sampling_seed"]>2**53 and json.loads(json.dumps(output))[0]["sampling_seed"]==rows[0]["sampling_seed"]
    assert json.dumps(rows,sort_keys=True)==before
    with pytest.raises(ValueError,match="entire"):
        EXTRACT.extract_rows(rows[:-1],tok,support,"k8",256)


def test_live_construction_flag_defaults_off_and_fresh_replay_needs_no_cooperative_residual(environment,tmp_path,monkeypatch):
    tok,support=environment
    argv=["sample","--model-path","raw","--checkpoint-path","fixture","--conditions","conditions","--output-dir","out"]
    monkeypatch.setattr(sys,"argv",argv);assert parse_args().construction_only is False
    monkeypatch.setattr(sys,"argv",argv+["--construction-only"]);assert parse_args().construction_only is True
    monkeypatch.setattr(sys,"argv",argv+["--construction-only","--purpose","evaluation","--temperature",".8"])
    with pytest.raises(ValueError,match="construction-only stage ablation"):
        live_main()
    row=recorded(tok);c=compile_condition(row,tok,mask_id=tok.mask_id,purpose="evaluation")
    model=PreferredModel(tok,body(tok,count=2,length=40,x=50)).eval()
    sampler=make_sampler(model,tok,support,[c],[row["sampling_seed"]],.7)
    x=torch.tensor([[0]+c["initial_body"]]);result,traces=sampler.run(x,torch.ones_like(x),construct=True,cooperative=False,closure=False)
    row.update(trace=traces[0],final_body_token_ids=result[0,1:].tolist(),method="k4_construction_only_v1")
    path=tmp_path/"paths.jsonl";path.write_text(json.dumps(row)+"\n")
    args=SimpleNamespace(replay_jsonl=path,purpose="evaluation",checkpoint_path="fixture",output_dir=tmp_path/"replay",
                         replay_tolerance=1e-6,construction_only=True,_construction_method="k4_construction_only_v1")
    monkeypatch.setattr(torch,"autocast",lambda *a,**kw:nullcontext())
    replay(args,model,tok,support,"cpu")
    report=json.loads((args.output_dir/"REPLAY.json").read_text())
    assert report["all_recorded_decisions_checked"]==12 and report["trained_state_residual_l1"]==0
    assert (args.output_dir/"_SUCCESS").is_file()
