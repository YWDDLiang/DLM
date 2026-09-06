"""Behavioral CPU tests of T deployment through the original scalar runtime."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from crystal_dlm import fixed_slot
from crystal_dlm import mixed_token_sampling as sampling
from crystal_dlm.programmed_path_data import compile_condition, path_seed, trace_terminal_body
from crystal_dlm.sampling_layout import sampling_batches
from crystal_dlm.spad_program import spad_predictor_position_groups
from scripts.sample_state_programmed_paths import make_sampler
from test_periodic_v2_training_data import codec

SPEC = importlib.util.spec_from_file_location("mixed_token_cli_test", ROOT/"src/scripts/sample_mixed_token_paths.py")
CLI = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CLI)


def condition(index, *, multiple_species=False):
    plan = {"N": 4, "elements": ["H", "O"], "counts": [2, 2]} if multiple_species else {"N": 2, "elements": ["H"], "counts": [2]}
    return {"sample_idx": index, "source_row_idx": index, "group_id": f"eval:{index}", "source_split": "evaluation",
            "prompt": "fixed source condition", "plan_state": plan,
            "species_program": ["O", "H"] if multiple_species else ["H"],
            "species_program_source": "fixed_pointer", "upstream_fallback": False}


@pytest.fixture
def tokenizer_and_support(monkeypatch):
    tokenizer, support = codec()
    tokenizer_type = type(tokenizer)
    monkeypatch.setattr(tokenizer_type, "__call__", lambda self, text, **kw: {"input_ids": [0]*len(text.split())}, raising=False)
    monkeypatch.setattr(tokenizer_type, "decode", lambda self, ids, **kw: "".join(
        {value:key for key,value in self.vocab.items()}[int(index)] for index in ids), raising=False)
    monkeypatch.setattr(fixed_slot, "MASK_TOKEN_ID", tokenizer.mask_id)
    return tokenizer, support


class PreferredTokenModel(nn.Module):
    def __init__(self, tokenizer, *, bad_row=None, bad_call=4, raise_call=None):
        super().__init__()
        self.table = nn.Embedding(len(tokenizer.vocab), 1)
        self.mask_id = tokenizer.mask_id
        self.raw_initialization = {"test": True}
        self.tokenizer = tokenizer
        self.bad_row, self.bad_call, self.raise_call = bad_row, bad_call, raise_call
        self.calls = []

    @property
    def device(self):
        return self.table.weight.device

    def get_output_embeddings(self):
        return self.table

    def forward(self, input_ids, attention_mask=None, geometry_context=None):
        assert geometry_context is not None
        assert torch.equal(input_ids, geometry_context.old_token_ids)
        assert bool((geometry_context.numeric_noise_level == -1).all())
        self.calls.append({"shape": tuple(input_ids.shape), "input": input_ids.clone(),
                           "active": geometry_context.active_token_mask.clone(),
                           "ranks": geometry_context.program_rank.clone()})
        if self.raise_call == len(self.calls):
            raise RuntimeError("fixture unassigned model failure")
        logits = torch.full((*input_ids.shape, self.table.weight.shape[0]), -10.)
        coordinates = [(0,0,0),(50,50,0),(0,50,50),(50,0,50)]
        vocabulary = self.tokenizer.vocab
        for row in range(input_ids.shape[0]):
            prompt = int(geometry_context.prompt_lengths[row])
            count = int(geometry_context.num_sites[row])
            for pos, family in enumerate(("LA","LB","LC","AA","AB","AG"), 1):
                token = f"<{family}_{40 if pos<=3 else 90:03d}>"
                logits[row,prompt+pos,vocabulary[token]] = 8.
            for site in range(count):
                for axis, family in enumerate("XYZ"):
                    value = coordinates[site][axis]
                    # The preferred periodic 100 alias must become canonical 0.
                    token = f"<{family}_{100 if site==0 and axis==0 else value:03d}>"
                    logits[row,prompt+8+4*site+axis,vocabulary[token]] = 8.
        if self.bad_row is not None and len(self.calls) >= self.bad_call:
            logits[self.bad_row] = torch.nan
        return SimpleNamespace(logits=logits)


def compiled(tokenizer, indices=(0,2), *, multiple_species=False):
    return [(index, 0, compile_condition(condition(index,multiple_species=multiple_species), tokenizer,
                                         mask_id=tokenizer.mask_id,purpose="evaluation")) for index in indices]


def final_policy(tmp_path):
    checkpoint = tmp_path/"train/checkpoints/step-4524"
    checkpoint.mkdir(parents=True)
    (checkpoint/"CHECKPOINT_FINAL.json").write_text(json.dumps({"method":sampling.TRAINING_METHOD,
        "eligible_policy":True,"completed_step":4524,"completed_epochs":2,"expected_epochs":2}))
    (tmp_path/"train/TRAIN_FINAL.json").write_text(json.dumps({"method":sampling.TRAINING_METHOD,
        "eligible_policy":True,"updates":4524,"epochs":2,"policy_path":str(checkpoint)}))
    (tmp_path/"train/_SUCCESS").touch()
    return checkpoint


def test_final_mixed_gate_precedes_loading_and_extracts_only_t(tokenizer_and_support,tmp_path,monkeypatch):
    tokenizer,_ = tokenizer_and_support
    checkpoint = final_policy(tmp_path)
    token = PreferredTokenModel(tokenizer)
    calls=[]
    def load(model_path, checkpoint_path, device, **kwargs):
        calls.append((model_path,checkpoint_path,device,kwargs))
        return SimpleNamespace(token_model=token,geometry_heads=lambda:pytest.fail("G called")),tokenizer
    monkeypatch.setitem(sys.modules,"crystal_dlm.mixed_geometry_model",SimpleNamespace(load_mixed_geometry_model=load))
    actual, restored, policy = sampling.load_final_mixed_token_policy("raw",checkpoint,"cpu")
    assert actual is token and restored is tokenizer and not token.training
    assert all(not p.requires_grad for p in token.parameters())
    assert calls[0][3]["trainable"] is False and policy["training"]["method"] == sampling.TRAINING_METHOD
    (tmp_path/"train/_SUCCESS").unlink()
    with pytest.raises(ValueError,match="final policy"):
        sampling.load_final_mixed_token_policy("raw",checkpoint,"cpu")
    assert len(calls)==1


def test_constructor_matches_original_runtime_and_noncanonical_p(tokenizer_and_support):
    tokenizer,support=tokenizer_and_support
    batch=compiled(tokenizer,multiple_species=True)
    model=PreferredTokenModel(tokenizer).eval()
    rows,statistics=sampling.sample_token_batch(model,tokenizer,support,batch,device="cpu",checkpoint_path="fixture")
    assert all(row["success"] for row in rows) and statistics["model_forward_calls"]==18
    schedule=[group[0] for group in spad_predictor_position_groups(batch[0][2]["program"])[1:]]
    assert schedule[6]==16  # O anchor precedes the canonical H storage slots.
    for row in rows:
        assert row["method"]=="mixed_token_construction" and row["training_method"]==sampling.TRAINING_METHOD
        assert [event["position"] for event in row["trace"]["events"] if event["op"]=="draw"]==schedule
        assert {event["phase"] for event in row["trace"]["events"]}=={"construct"}
        assert "<X_100>" not in row["body"] and "<X_000>" in row["body"]
        assert row["sampling_nfe"]==18 and row["geometry_forward_calls"]==0
        assert row["sampling_seed"]==path_seed(20260905,row["group_id"],0,0)
    assert int(model.calls[0]["active"][0].sum())==18 and int(model.calls[-1]["active"][0].sum())==1
    fresh=PreferredTokenModel(tokenizer).eval()
    cs=[item[2] for item in batch]
    sampler=make_sampler(fresh,tokenizer,support,cs,[row["sampling_seed"] for row in rows],.7)
    x=torch.tensor([c["prompt_token_ids"]+c["initial_body"] for c in cs])
    result,traces=sampler.run(x,torch.ones_like(x),construct=True,cooperative=False,closure=False,full_cell_repair=False)
    assert traces==[row["trace"] for row in rows]
    assert result[:,len(cs[0]["prompt_token_ids"]):].tolist()==[row["final_body_token_ids"] for row in rows]


def test_fresh_full_batch_scalar_replay_checks_logp_seed_and_final_state(tokenizer_and_support):
    tokenizer,support=tokenizer_and_support
    batch=compiled(tokenizer,multiple_species=True)
    rows,_=sampling.sample_token_batch(PreferredTokenModel(tokenizer).eval(),tokenizer,support,batch,
                                      device="cpu",checkpoint_path="fixture")
    fresh=PreferredTokenModel(tokenizer).eval()
    result=sampling.replay_token_batch(fresh,tokenizer,support,batch,rows,device="cpu",checkpoint_path="fixture")
    assert result["checked_draws"]==result["seeded_token_matches"]==36
    assert result["model_forward_calls"]==18 and result["maximum_logp_error"]==0
    assert all(call["shape"][0]==2 for call in fresh.calls)
    changed=deepcopy(rows);next(e for e in changed[0]["trace"]["events"] if e["op"]=="draw")["log_probability"]+=.01
    with pytest.raises(RuntimeError,match="logp replay"):
        sampling.replay_token_batch(PreferredTokenModel(tokenizer).eval(),tokenizer,support,batch,changed,
                                    device="cpu",checkpoint_path="fixture")
    changed=deepcopy(rows);changed[0]["sampling_seed"]+=1
    with pytest.raises(ValueError,match="identity"):
        sampling.replay_token_batch(PreferredTokenModel(tokenizer).eval(),tokenizer,support,batch,changed,
                                    device="cpu",checkpoint_path="fixture")


def test_empty_support_row_is_preserved_and_replayed_in_original_batch(tokenizer_and_support):
    tokenizer,support=tokenizer_and_support;batch=compiled(tokenizer)
    model=PreferredTokenModel(tokenizer,bad_row=0,bad_call=4).eval()
    rows,stats=sampling.sample_token_batch(model,tokenizer,support,batch,device="cpu",checkpoint_path="fixture")
    assert not rows[0]["success"] and rows[1]["success"] and stats["model_forward_calls"]==12
    assert rows[0]["sampling_nfe"]==4 and rows[0]["failed_row_padding_evaluations"]==8
    assert rows[0]["sampling_failure"]["scope"]=="row" and rows[0]["parseable"] is False
    for name in ("body","structure","native_structure","cif","cif_path","final_body_token_ids"):
        assert rows[0][name] is None
    assert trace_terminal_body(rows[0]["trace"])==rows[0]["diagnostic_candidate"]["final_body_token_ids"]
    replay=sampling.replay_token_batch(PreferredTokenModel(tokenizer,bad_row=0,bad_call=4).eval(),tokenizer,support,
                    batch,rows,device="cpu",checkpoint_path="fixture")
    assert replay["checked_no_support"]==1 and replay["checked_draws"]==15 and replay["model_forward_calls"]==12
    json.dumps([rows,stats,replay],allow_nan=False)


def test_unattributed_error_retains_entire_attempt_batch_without_retry(tokenizer_and_support):
    tokenizer,support=tokenizer_and_support;batch=compiled(tokenizer)
    model=PreferredTokenModel(tokenizer,raise_call=3).eval()
    rows,stats=sampling.sample_token_batch(model,tokenizer,support,batch,device="cpu",checkpoint_path="fixture")
    assert len(model.calls)==stats["model_forward_calls"]==3 and len(rows)==2
    assert all(not row["success"] and row["body"] is None and row["parseable"] is False for row in rows)
    assert all(row["sampling_failure"]["scope"]=="logical_batch" for row in rows)
    fresh=PreferredTokenModel(tokenizer).eval()
    with pytest.raises(ValueError,match="aborted"):
        sampling.replay_token_batch(fresh,tokenizer,support,batch,rows,device="cpu",checkpoint_path="fixture")
    assert not fresh.calls
    json.dumps([rows,stats],allow_nan=False)


def test_parser_failure_has_no_active_preview_and_cannot_pass_acceptance(tokenizer_and_support,monkeypatch):
    tokenizer,support=tokenizer_and_support;batch=compiled(tokenizer)
    monkeypatch.setattr(sampling,"validate_completed_body",lambda *_:(_ for _ in ()).throw(ValueError("fixture parse failure")))
    rows,_=sampling.sample_token_batch(PreferredTokenModel(tokenizer).eval(),tokenizer,support,batch,
                                      device="cpu",checkpoint_path="fixture")
    assert all(not row["success"] and row["runtime_success"] and row["body"] is None for row in rows)
    assert all(row["diagnostic_candidate"]["body"] and row["parseable"] is False for row in rows)
    with pytest.raises(ValueError,match="native-parser"):
        sampling.replay_token_batch(PreferredTokenModel(tokenizer).eval(),tokenizer,support,batch,rows,
                                    device="cpu",checkpoint_path="fixture")


@pytest.mark.parametrize("key,value",[("seed",8),("collection_round",1),("batch_cap",2),("layout_world_size",6),("temperature",1.)])
def test_frozen_protocol_rejects_unregistered_changes(key,value):
    args=dict(seed=20260905,collection_round=0,batch_cap=4,layout_world_size=2,temperature=.7);args[key]=value
    with pytest.raises(ValueError,match="fixed"):
        sampling.validate_protocol(**args)


def test_merge_and_cpu_replay_entry_preserve_full_failures_and_64bit_seeds(tokenizer_and_support,tmp_path):
    tokenizer,support=tokenizer_and_support
    checkpoint=final_policy(tmp_path)
    source=tmp_path/"conditions.jsonl";source.write_text("".join(json.dumps(condition(i))+"\n" for i in range(4)))
    output=tmp_path/"sample";output.mkdir()
    args=SimpleNamespace(conditions=source,checkpoint_path=checkpoint,condition_start=0,condition_stop=4,
        seed=20260905,collection_round=0,temperature=.7,batch_size=4,sampling_layout_world_size=2,
        world_size=2,rank=0,max_length=382,output_dir=output,replay_tolerance=1e-6)
    selected=CLI.expected_conditions(args)
    items=[(i,0,compile_condition(row,tokenizer,mask_id=tokenizer.mask_id,purpose="evaluation")) for i,row in selected]
    for rank in range(2):
        records=[];batch_stats=[]
        for batch in sampling_batches(items,batch_size=4,rank=rank,world_size=2,layout_world_size=2):
            rows,stat=sampling.sample_token_batch(PreferredTokenModel(tokenizer).eval(),tokenizer,support,batch,
                device="cpu",checkpoint_path=checkpoint);records+=rows;batch_stats.append(stat)
        (output/f"records.rank{rank}.jsonl").write_text("".join(json.dumps(row)+"\n" for row in records))
        totals={key:sum(s[key] for s in batch_stats) for key in ("requested","successful","model_forward_calls","model_row_evaluations",
                "live_row_model_evaluations","failed_row_padding_evaluations","geometry_forward_calls")}
        (output/f"SAMPLING.rank{rank}.json").write_text(json.dumps({**CLI.protocol(args),"rank":rank,**totals}))
        (output/f"_SUCCESS.rank{rank}").touch()
    final=CLI.merge_outputs(args)
    assert final["method"]=="mixed_token_construction" and final["requested"]==4 and final["successful"]==4
    merged=CLI.read_jsonl(output/"native/paths.jsonl")
    assert isinstance(merged[0]["sampling_seed"],int) and merged[0]["sampling_seed"]>2**53
    assert [row["sample_idx"] for row in merged]==list(range(4))
    args.replay_jsonl=output/"native/paths.jsonl";args.output_dir=tmp_path/"replay";args.output_dir.mkdir();args.world_size=1
    replay=CLI.replay_records(args,PreferredTokenModel(tokenizer).eval(),tokenizer,support,items)
    assert replay["status"]=="PASS" and replay["requests"]==4 and replay["checked_draws"]==48
    assert (args.output_dir/"REPLAY.json").is_file() and (args.output_dir/"_SUCCESS").is_file()


def test_logical_batch_membership_is_unchanged_across_execution_workers(tokenizer_and_support):
    tokenizer,_=tokenizer_and_support;items=compiled(tokenizer,tuple(range(19)))
    layouts=[]
    for world in (2,4,6):
        membership={}
        for rank in range(world):
            for batch in sampling_batches(items,batch_size=4,rank=rank,world_size=world,layout_world_size=2):
                ids=[item[0] for item in batch]
                for index in ids:
                    assert index not in membership
                    membership[index]=ids
        layouts.append(membership)
    assert layouts[0]==layouts[1]==layouts[2]
