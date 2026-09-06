import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from crystal_dlm.probability_mixture_sampling import mix_post_hard_vectors, ProbabilityMixtureProgrammedPathSampler
from crystal_dlm.programmed_path_runtime import replay_scalar_states
from crystal_dlm.spad_generation import _transaction_candidate_tokens
from test_short_contact_sampling import fixture
from test_state_programmed_runtime import PreferredModel


def test_probability_average_is_normalized_before_mixing_not_logit_average():
    a=torch.tensor([.9,.1],dtype=torch.float64).log()*.7+70
    b=torch.tensor([.2,.8],dtype=torch.float64).log()*.7-19
    out,diag=mix_post_hard_vectors(a,b)
    torch.testing.assert_close(torch.softmax(out/.7,-1),torch.tensor([.55,.45],dtype=torch.float64),atol=1e-14,rtol=0)
    assert abs(diag["probability_sum"]-1)<1e-14
    assert diag["kl_k4_to_mixture"]<=math.log(2) and diag["kl_k8_to_mixture"]<=math.log(2)
    assert not torch.allclose(torch.softmax((a+b)/1.4,-1),torch.softmax(out/.7,-1))


@pytest.mark.parametrize("dtype",[torch.bfloat16,torch.float32,torch.float64])
def test_identical_policies_keep_support_and_same_seed_winner(dtype):
    v=torch.tensor([2.,-.3,4.,torch.finfo(dtype).min],dtype=dtype)
    out,diag=mix_post_hard_vectors(v,v)
    assert out.dtype==torch.float64 and out[-1]==torch.finfo(torch.float64).min
    torch.testing.assert_close(torch.softmax(v.double()/.7,-1),torch.softmax(out/.7,-1),atol=1e-14,rtol=0)
    for seed in range(10):
        kwargs=dict(active_absolute_positions={0:0},temperature=.7,remasking="low_confidence",sampling_seeds_by_batch=[seed],salt=10007)
        assert int(_transaction_candidate_tokens(v[None,None],**kwargs)[0,0])==int(_transaction_candidate_tokens(out[None,None],**kwargs)[0,0])


def test_mismatched_or_invalid_component_is_not_replaced_by_the_other():
    a=torch.tensor([1.,2.,torch.finfo(torch.float32).min])
    for b in (torch.tensor([1.,torch.finfo(torch.float32).min,0.]),torch.tensor([1.,torch.nan,-2.]),
              torch.tensor([1.,torch.inf,-2.]),torch.full_like(a,torch.finfo(torch.float32).min)):
        with pytest.raises(ValueError):mix_post_hard_vectors(a,b)


def models_and_sampler():
    tok,old,_,x,body=fixture()
    changed=list(body);changed[14]=tok.vocab["<Z_030>"]
    peer=PreferredModel(tok,changed).eval()
    sampler=ProbabilityMixtureProgrammedPathSampler(old.model,peer_model=peer,prompt_length=old.prompt_length,
        gen_length=old.gen_length,mask_id=old.mask_id,programs=old.programs,allowed_token_ids=old.allowed_ids,
        atom_count_grammar=old.grammar,constraints=old.constraints,temperature=.7,sampling_seeds=old.seeds)
    return tok,sampler,x,body


def test_both_models_read_identical_current_old_phase_and_active_context():
    tok,sampler,x,body=models_and_sampler();sampler._prepare(x)
    captured=[]
    def hook(_m,args,kwargs):
        c=kwargs["geometry_context"]
        captured.append((args[0].clone(),c.old_token_ids.clone(),c.program_rank.clone(),c.task_ids.clone(),c.active_token_mask.clone()))
    hooks=[m.register_forward_pre_hook(hook,with_kwargs=True) for m in (sampler.model,sampler.peer.model)]
    try:out,bad=sampler.processed_logits(x,torch.tensor([[0]+body]),{0:14},{0:[14]},torch.ones_like(x),phase="closure")
    finally:
        for h in hooks:h.remove()
    assert bad==set() and sampler.component_forward_calls==[1,1]
    assert len(captured)==2 and all(torch.equal(a,b) for a,b in zip(*captured))
    assert out.shape==(1,16,len(tok.vocab)) and out.untyped_storage().nbytes()==len(tok.vocab)*8


def test_mixture_full_path_and_fresh_scalar_replay_preserve_original_phases():
    tok,sampler,x,body=models_and_sampler()
    initial=torch.tensor([[0]+body]);initial[:,2:8]=tok.mask_id
    for site in range(2):initial[:,9+4*site:12+4*site]=tok.mask_id
    result,traces=sampler.run(initial,torch.ones_like(initial),construct=True,cooperative=True,closure=True)
    assert traces[0]["success"] and {e["phase"] for e in traces[0]["events"]}=={"construct","cooperative","closure"}
    assert sampler.component_forward_calls[0]==sampler.component_forward_calls[1]>0
    _,fresh,_,_=models_and_sampler()
    checked=0
    for state in replay_scalar_states(traces[0]):
        current=torch.tensor([[0]+state["input_body"]]);old=torch.tensor([[0]+state["old_body"]]);fresh._prepare(current)
        output,bad=fresh.processed_logits(current,old,{0:state["position"]},{0:state["transaction_positions"]},torch.ones_like(current),phase=state["phase"])
        logp=float(torch.log_softmax(output[0,1+state["position"]]/.7,-1)[state["target_token"]])
        assert not bad and abs(logp-state["recorded_log_probability"])<1e-12
        checked+=1
    assert checked==fresh.component_forward_calls[0]==fresh.component_forward_calls[1]


def test_one_nonfinite_peer_marks_action_unavailable_without_single_model_fallback():
    tok,sampler,x,body=models_and_sampler();sampler._prepare(x)
    original=sampler.peer.model.forward
    def invalid(*args,**kwargs):
        output=original(*args,**kwargs);output.logits.fill_(torch.nan);return output
    sampler.peer.model.forward=invalid
    output,bad=sampler.processed_logits(x,x,{0:14},{0:[14]},torch.ones_like(x),phase="closure")
    assert bad=={0} and sampler.component_forward_calls==[1,1]
    assert bool((output==torch.finfo(torch.float64).min).all())
    assert sampler._mixture_last[0]["component_unavailable"]==[False,True]
    assert sampler._mixture_last[0]["single_component_fallback"] is False
