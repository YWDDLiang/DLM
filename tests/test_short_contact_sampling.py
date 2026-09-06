from copy import deepcopy
import importlib.util
from pathlib import Path
import sys

import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from crystal_dlm.programmed_path_runtime import ProgrammedPathSampler
from crystal_dlm.r5_dynamic_length import exact_dynamic_schema_constraints
from crystal_dlm.short_contact_sampling import ContactSpec, ShortContactProgrammedPathSampler
from crystal_dlm.spad_program import program_from_element_order
from test_periodic_v2_training_data import codec
from test_state_programmed_runtime import PreferredModel

SPEC=ROOT/"docs/v3_scientific_audit_20260906/evidence_v3_failure_20260907/k4k8_physics/candidate_spec.json"


def fixture(dtype=torch.float32,probe_only=False):
    tok,support=codec();v=tok.vocab
    body=[v["<N_002>"]]+[v[f"<{a}_040>"] for a in ("LA","LB","LC")]+[v[f"<{a}_090>"] for a in ("AA","AB","AG")]
    for z in (0,50):body += [v["<E_Zn>"],v["<X_000>"],v["<Y_000>"],v[f"<Z_{z:03d}>"]]
    program=program_from_element_order({"N":2,"elements":["Zn"],"counts":[2]},["Zn"],order_source="fixture")
    class Model(PreferredModel):
        def forward(self,*args,**kwargs):
            result=super().forward(*args,**kwargs);result.logits=result.logits.to(dtype);return result
    model=Model(tok,body).eval()
    kwargs=dict(prompt_length=1,gen_length=15,mask_id=tok.mask_id,programs=[program],
                allowed_token_ids=exact_dynamic_schema_constraints(tok,2),atom_count_grammar=None,
                constraints=support,temperature=.7,sampling_seeds=[8732798746829386280])
    old=ProgrammedPathSampler(model,**kwargs)
    new=ShortContactProgrammedPathSampler(model,contact_spec=ContactSpec.load(SPEC),probe_only=probe_only,collect_vectors=True,**kwargs)
    x=torch.tensor([[0]+body]);x[0,15]=tok.mask_id
    return tok,old,new,x,body


@pytest.mark.parametrize("dtype",[torch.bfloat16,torch.float32,torch.float64])
def test_tilt_formula_preserves_original_support_alias_and_precision(dtype):
    tok,old,new,x,body=fixture(dtype)
    old._prepare(x);new._prepare(x)
    before,bad=old.processed_logits(x,x,{0:14},{0:[14]},torch.ones_like(x),phase="closure")
    after,after_bad=new.processed_logits(x,x,{0:14},{0:[14]},torch.ones_like(x),phase="closure")
    assert bad==after_bad==set()
    vector=before[0,15];legal=torch.isfinite(vector)&(vector>torch.finfo(dtype).min)
    ids=legal.nonzero(as_tuple=True)[0];b,reason,n=new.penalty(x,0,14,ids)
    assert reason=="evaluated" and n==1 and float(b.min())==0 and float(b.max())>0
    expected=vector[ids].double()-.7*b
    torch.testing.assert_close(after[0,15,ids].double(),expected,atol=2e-6,rtol=0)
    assert torch.equal(after[0,15,~legal],vector[~legal].to(after.dtype))
    assert after[0,15,tok.vocab["<Z_100>"]]==before[0,15,tok.vocab["<Z_100>"]]
    assert after.dtype==(torch.float64 if dtype==torch.float64 else torch.float32)
    assert new.contact_diagnostics[-1]["expected_penalty_after"]<=new.contact_diagnostics[-1]["expected_penalty_before"]+1e-10
    other=torch.ones_like(before,dtype=torch.bool);other[0,15,ids]=False
    assert torch.equal(after[other],before.to(after.dtype)[other])


def test_known_geometry_only_and_max_pair_barrier():
    tok,old,new,x,body=fixture()
    ids=torch.tensor([tok.vocab["<Z_025>"],tok.vocab["<Z_050>"]])
    b,_,_=new.penalty(x,0,14,ids)
    torch.testing.assert_close(b,torch.tensor([torch.log(torch.tensor(1.22)).item(),0.],dtype=torch.float64),atol=1e-7,rtol=0)
    missing=x.clone();missing[0,13]=tok.mask_id
    assert not bool(new.penalty(missing,0,14,ids)[0].any())
    missing=x.clone();missing[0,11]=tok.mask_id
    assert new.penalty(missing,0,14,ids)[1]=="no_complete_current_neighbours"
    assert not bool(new.penalty(x,0,13,ids)[0].any())
    missing=x.clone();missing[0,2]=tok.mask_id
    assert new.penalty(missing,0,14,ids)[1]=="incomplete_current_cell"


def test_all_zero_returns_exact_original_tensor(monkeypatch):
    tok,old,new,x,body=fixture(torch.bfloat16)
    x[0,11]=tok.mask_id  # No complete neighbour, even though old could contain it.
    hard=torch.full((1,16,len(tok.vocab)),torch.finfo(torch.bfloat16).min,dtype=torch.bfloat16)
    hard[0,15,tok.vocab["<Z_050>"]]=3.
    monkeypatch.setattr(ProgrammedPathSampler,"processed_logits",lambda *a,**k:(hard,set()))
    result,bad=new.processed_logits(x,torch.tensor([[0]+body]),{0:14},{0:[14]},torch.ones_like(x),phase="closure")
    assert result is hard and bad==set()


def test_exactly_constant_positive_barrier_preserves_original_policy(monkeypatch):
    tok,old,new,x,body=fixture(torch.bfloat16)
    hard=torch.full((1,16,len(tok.vocab)),torch.finfo(torch.bfloat16).min,dtype=torch.bfloat16)
    hard[0,15,tok.vocab["<Z_025>"]]=3.
    hard[0,15,tok.vocab["<Z_050>"]]=2.
    monkeypatch.setattr(ProgrammedPathSampler,"processed_logits",lambda *a,**k:(hard,set()))
    monkeypatch.setattr(new,"penalty",lambda x,row,pos,ids:(torch.ones(len(ids),dtype=torch.float64),"evaluated",1))
    result,_=new.processed_logits(x,x,{0:14},{0:[14]},torch.ones_like(x),phase="closure")
    assert result is hard and new.contact_diagnostics[-1]["constant_penalty_removed"]==1.
    assert new.contact_diagnostics[-1]["tv"]==0


def test_probe_keeps_original_path_and_applied_tilt_keeps_zero_penalty_winners():
    tok,old,probe,x,body=fixture(probe_only=True)
    initial=torch.tensor([[0]+body]);initial[:,2:8]=tok.mask_id
    for site in range(2):initial[:,9+4*site:12+4*site]=tok.mask_id
    result,trace=old.run(initial,torch.ones_like(initial),construct=True,cooperative=True,closure=True)
    probed,probe_trace=probe.run(initial,torch.ones_like(initial),construct=True,cooperative=True,closure=True)
    assert torch.equal(result,probed) and len(probe.contact_diagnostics)>0
    spec=importlib.util.spec_from_file_location("probe_contact_test",ROOT/"src/scripts/probe_short_contact_prefixes.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert module.bind_trace(probe_trace[0],trace[0],0)==0
    _,_,applied,_,_=fixture()
    biased,biased_trace=applied.run(initial,torch.ones_like(initial),construct=True,cooperative=True,closure=True)
    assert torch.equal(biased,result)
    z_events=[e for e in biased_trace[0]["events"] if e.get("short_contact")]
    assert z_events and all(e["short_contact"]["chosen_penalty"]==0 for e in z_events)
    assert {e["phase"] for e in biased_trace[0]["events"]}=={"construct","cooperative","closure"}


def test_missing_element_radius_fails_before_model_call():
    tok,old,new,x,body=fixture()
    invalid=ContactSpec({},"fixture",new.contact_spec.radii_sha256)
    with pytest.raises(ValueError,match="does not cover Zn"):
        ShortContactProgrammedPathSampler(new.model,prompt_length=1,gen_length=15,mask_id=tok.mask_id,
            programs=new.programs,allowed_token_ids=new.allowed_ids,atom_count_grammar=None,
            constraints=new.constraints,temperature=.7,sampling_seeds=[1],contact_spec=invalid)
