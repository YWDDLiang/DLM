from types import SimpleNamespace

import math
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from crystal_dlm.fixed_slot import build_special_tokens
from crystal_dlm.periodic_state_conditioning import PeriodicStateConfig
from crystal_dlm.programmed_path_runtime import (
    ProgrammedPathSampler, complete_geometry_supported, cooperative_slots, replay_scalar_states,
)
from crystal_dlm.r5_dynamic_length import exact_dynamic_schema_constraints
from crystal_dlm.spad_program import program_from_element_order
from crystal_dlm.state_conditioned_model import (
    CrystalStateContext, StateConditionedDLM, context_from_programs, set_state_lora_trainable,
)


class TinyTokenizer:
    def __init__(self):
        self.vocab = {token: i + 1 for i, token in enumerate(build_special_tokens())}
        self.vocab["<PAD>"] = 0
        self.vocab["<MASK>"] = len(self.vocab)
        self.mask_id = self.vocab["<MASK>"]

    def get_vocab(self):
        return self.vocab

    def convert_tokens_to_ids(self, token):
        return self.vocab[token]


class TinyBase(nn.Module):
    def __init__(self, vocab_size, hidden=16, recompute=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden)
        self.output = nn.Linear(hidden, vocab_size, bias=False)
        self.lora_A = nn.ModuleDict({"default": nn.Linear(hidden, 2, bias=False)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(2, hidden, bias=False)})
        nn.init.zeros_(self.lora_B["default"].weight)
        self.config = SimpleNamespace(hidden_size=hidden)
        self.recompute = recompute

    def get_input_embeddings(self):
        return self.embedding

    def get_output_embeddings(self):
        return self.output

    def forward(self, input_ids=None, *, inputs_embeds=None, attention_mask=None):
        embeddings = self.embedding(input_ids) if inputs_embeds is None else inputs_embeds
        self.last_embeddings = embeddings
        def transform(x):
            x = x + self.lora_B["default"](self.lora_A["default"](x))
            return self.output(x + x.mean(1, keepdim=True))
        logits = checkpoint(transform, embeddings, use_reentrant=False) if self.recompute else transform(embeddings)
        return SimpleNamespace(logits=logits)


def body(tokenizer, *, length=20, x=30, count=2):
    v = tokenizer.vocab
    values = [v[f"<N_{count:03d}>"]]
    values += [v[f"<{axis}_{length:03d}>"] for axis in ("LA", "LB", "LC")]
    values += [v[f"<{axis}_090>"] for axis in ("AA", "AB", "AG")]
    for i in range(count):
        values += [v["<E_H>"], v[f"<X_{(x*i)%100:03d}>"], v["<Y_000>"], v["<Z_000>"]]
    return values


def constraints(tokenizer):
    v = tokenizer.vocab
    coords = {a: {v[f"<{a}_{i:03d}>"]: i for i in range(101)} for a in "XYZ"}
    return {
        "representation": "dynamic_v1", "max_atoms": 20, "coord_period": 100,
        "count_token_to_n": {v[f"<N_{i:03d}>"]: i for i in range(1, 21)},
        "length_token_to_bin": {a: {v[f"<{a}_{i:03d}>"]: i for i in range(501)} for a in ("LA", "LB", "LC")},
        "angle_token_to_bin": {a: {v[f"<{a}_{i:03d}>"]: i for i in range(1, 180)} for a in ("AA", "AB", "AG")},
        "length_step": .1, "coord_token_to_bin": coords,
        "coord_bin_to_token_id": {a: {b: t for t, b in table.items()} for a, table in coords.items()},
        "coordinate_alias_token_ids": {a: (v[f"<{a}_000>"], v[f"<{a}_100>"]) for a in "XYZ"},
        "pbc_min_distance_mask": True, "pbc_min_distance_A": .5, "pbc_image_radius": 2,
        "canonicalize_periodic_alias": True,
    }


def program(count=2):
    return program_from_element_order({"N": count, "elements": ["H"], "counts": [count]}, ["H"], order_source="test")


def test_explicit_embedding_injection_and_checkpoint_gradients():
    torch.manual_seed(14)
    tok = TinyTokenizer()
    base = TinyBase(len(tok.vocab), recompute=True)
    model = StateConditionedDLM(base, tok, PeriodicStateConfig(16, width=12, radial_basis_count=4))
    counts = set_state_lora_trainable(model)
    assert counts["lora"] > 0 and counts["conditioner"] > 0
    assert not base.embedding.weight.requires_grad and not base.output.weight.requires_grad
    x = torch.tensor([[0] + body(tok)])
    active = {0: list(range(1, 7))}
    context = context_from_programs(x.clone(), prompt_length=1, num_sites=2, programs=[program()], active_positions=active)
    x[:, 2:8] = tok.mask_id
    baseline = base(x).logits.detach()
    assert torch.equal(baseline, model(x, geometry_context=context).logits.detach())
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
    for _ in range(2):
        optimizer.zero_grad()
        logits = model(x, geometry_context=context).logits
        loss = torch.nn.functional.cross_entropy(logits[:, 2], torch.tensor([tok.vocab["<LA_020>"]]))
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()
    assert model.state_conditioner.cell_encoder[0].weight.grad.abs().sum() > 0


def test_mixed_length_context_does_not_inject_prompt_or_padding():
    tok = TinyTokenizer()
    base = TinyBase(len(tok.vocab))
    model = StateConditionedDLM(base, tok, PeriodicStateConfig(16, width=12, radial_basis_count=4))
    with torch.no_grad():
        model.state_conditioner.cell_projection.weight.fill_(.1)
        model.state_conditioner.site_projection.weight.fill_(.1)
    values = [[0] + body(tok) + [0] * 5, [0] * 3 + body(tok, count=1) + [0] * 7]
    x = torch.tensor(values)
    context = CrystalStateContext(x.clone(), torch.tensor([1, 3]), torch.tensor([2, 1]),
                                  torch.zeros(2, 20, dtype=torch.long), torch.zeros_like(x, dtype=torch.bool))
    before = base.embedding(x).detach()
    model(x, geometry_context=context)
    difference = base.last_embeddings - before
    assert torch.equal(difference[0, 0], torch.zeros(16))
    assert torch.equal(difference[1, :3], torch.zeros(3, 16))
    assert torch.equal(difference[0, 16:], torch.zeros_like(difference[0, 16:]))
    assert torch.equal(difference[1, 14:], torch.zeros_like(difference[1, 14:]))


class PreferredModel(TinyBase):
    def __init__(self, tokenizer, preferred):
        super().__init__(len(tokenizer.vocab))
        self.preferred = preferred

    def forward(self, input_ids=None, attention_mask=None, geometry_context=None):
        batch, length = input_ids.shape
        logits = torch.full((batch, length, self.output.weight.shape[0]), -30.)
        for pos, token in enumerate(self.preferred):
            logits[:, 1 + pos, token] = 30.
        return SimpleNamespace(logits=logits)


def sample_case(tok, source, preferred, *, closure=False):
    model = PreferredModel(tok, preferred)
    count = (len(source) - 7) // 4
    sampler = ProgrammedPathSampler(
        model, prompt_length=1, gen_length=len(source), mask_id=tok.mask_id,
        programs=[program(count)], allowed_token_ids=exact_dynamic_schema_constraints(tok, count),
        atom_count_grammar=None, constraints=constraints(tok), temperature=0., sampling_seeds=[71],
    )
    x = torch.tensor([[0] + source])
    return sampler, sampler.run(x, torch.ones_like(x), construct=False, closure=closure)


def test_cooperative_cell_sites_pass_where_cell_alone_collides():
    tok = TinyTokenizer()
    source, target = body(tok, length=20, x=30), body(tok, length=10, x=50)
    intermediate = source.copy()
    intermediate[1:7] = target[1:7]
    assert complete_geometry_supported(torch.tensor(source), constraints(tok))
    assert not complete_geometry_supported(torch.tensor(intermediate), constraints(tok))
    assert complete_geometry_supported(torch.tensor(target), constraints(tok))
    _, (output, traces) = sample_case(tok, source, target, closure=True)
    assert output[0, 1:].tolist() == target
    assert traces[0]["success"]
    assert not any(e["op"] == "rollback" for e in traces[0]["events"])


def test_joint_failure_rolls_back_all_fields_and_keeps_attempts():
    tok = TinyTokenizer()
    source, target = body(tok), body(tok, length=4, x=50)
    _, (output, traces) = sample_case(tok, source, target)
    assert output[0, 1:].tolist() == source
    events = traces[0]["events"]
    rollback = [e for e in events if e["op"] == "rollback"]
    assert len(rollback) == 1 and len(rollback[0]["positions"]) == 12
    states = list(replay_scalar_states(traces[0]))
    assert len(states) == sum(e["op"] == "draw" for e in events)
    assert states[0]["old_body"] == source
    assert all(states[0]["input_body"][p] == tok.mask_id for p in rollback[0]["positions"])
    assert any(s["target_token"] != source[s["position"]] for s in states)


def test_single_atom_self_images_are_checked():
    tok = TinyTokenizer()
    assert not complete_geometry_supported(torch.tensor(body(tok, count=1, length=4)), constraints(tok))
    assert complete_geometry_supported(torch.tensor(body(tok, count=1, length=10)), constraints(tok))
    assert cooperative_slots(torch.tensor(body(tok, count=1)), program(1), constraints(tok)) == (0,)


def test_positive_temperature_draw_probabilities_replay():
    tok = TinyTokenizer()
    source = body(tok)
    sampler, _ = sample_case(tok, source, source)
    sampler.temperature = .7
    x = torch.tensor([[0] + source])
    _, traces = sampler.run(x, torch.ones_like(x), construct=False, closure=False)
    for state in replay_scalar_states(traces[0]):
        current = torch.tensor([[0] + state["input_body"]])
        old = torch.tensor([[0] + state["old_body"]])
        logits, bad = sampler.processed_logits(current, old, {0: state["position"]},
                                                {0: state["transaction_positions"]}, torch.ones_like(current))
        assert not bad
        actual = torch.log_softmax(logits[0, state["position"] + 1].double() / .7, -1)[state["target_token"]]
        assert math.isclose(float(actual), state["recorded_log_probability"], rel_tol=0, abs_tol=1e-10)
