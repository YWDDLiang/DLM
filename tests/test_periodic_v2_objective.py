from copy import deepcopy
import itertools
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from crystal_dlm import periodic_v2_training_data as data
from crystal_dlm.periodic_v2_objective import PeriodicV2Objective, dense_field_coefficients
from crystal_dlm.programmed_path_runtime import process_path_logits
from crystal_dlm.r5_dynamic_length import exact_dynamic_schema_constraints
from test_periodic_base_training_data import source
from test_periodic_v2_training_data import codec, force_state


def batch(examples):
    width = max(len(example["input_body"]) for example in examples)
    tokens = torch.zeros(len(examples), width, dtype=torch.long)
    for row, example in enumerate(examples):
        tokens[row, :len(example["input_body"])] = torch.tensor(example["input_body"])
    return {"examples": examples, "input_ids": tokens,
            "geometry_context": SimpleNamespace(prompt_lengths=torch.zeros(len(examples), dtype=torch.long))}


@pytest.mark.parametrize("probability", [.1, .65])
def test_dense_inclusion_estimator_matches_forced_target_risk_including_empty_masks(probability):
    positions = [1, 2, 3, 4, 5, 6, 8, 9, 10]
    base_loss = np.arange(1., 10.)
    expectation = 0.
    empty_mass = 0.
    for mask in itertools.product((False, True), repeat=9):
        count = sum(mask)
        mass = probability ** count * (1 - probability) ** (9 - count)
        selected = [position for position, take in zip(positions, mask) if take]
        losses = [base_loss[index] + .2 * count for index, take in enumerate(mask) if take]
        coefficients = dense_field_coefficients(selected, 1, probability)
        value = sum(weight * loss for weight, loss in zip(coefficients, losses))
        expectation += mass * value
        if not selected:
            assert value == 0.
            empty_mass += mass
    # Conditional on M_j=1, the other eight masks remain Bernoulli(p).
    expected_losses = base_loss + .2 * (1 + 8 * probability)
    expected = .5 * expected_losses[:6].mean() + .5 * expected_losses[6:].mean()
    assert abs(expectation - expected) < 2e-13
    assert empty_mass == (1 - probability) ** 9 > 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_prefix_exact_loss_matches_actual_alias_policy_temperature_and_gradients(monkeypatch, dtype):
    tokenizer, support = codec()
    force_state(monkeypatch, branch="prefix", position=8, object_index=1)
    example = data.make_periodic_v2_training_example(source(count=1), tokenizer, support, view=0, epoch=0, seed=8)
    materialized = batch([example])
    vocabulary_size = max(tokenizer.vocab.values()) + 1
    logits = torch.zeros(1, len(example["input_body"]), vocabulary_size, dtype=dtype)
    canonical, alias = support["coordinate_alias_token_ids"]["X"]
    logits[0, 8, canonical], logits[0, 8, alias] = .2, .6
    logits.requires_grad_(True)
    objective = PeriodicV2Objective(tokenizer, support, "cpu")
    loss, metrics, conflicts = objective(logits, materialized)
    allowed = torch.zeros(len(example["input_body"]), vocabulary_size, dtype=torch.bool)
    for position, ids in enumerate(exact_dynamic_schema_constraints(tokenizer, 1)):
        allowed[position, ids] = True
    reference_logits = logits.detach().clone().requires_grad_(True)
    actual, unavailable = process_path_logits(
        reference_logits, materialized["input_ids"], prompt_length=0, gen_length=len(example["input_body"]),
        allowed=allowed, grammar=None, constraints=support, positions={0: 8}, mask_id=tokenizer.mask_id,
    )
    expected = -torch.log_softmax(actual[0, 8].float() / .7, -1)[canonical]
    torch.testing.assert_close(loss, expected, atol=2e-6, rtol=1e-6)
    loss.backward()
    expected.backward()
    torch.testing.assert_close(logits.grad, reference_logits.grad, atol=2e-6, rtol=1e-5)
    assert not conflicts and not unavailable and metrics["prefix_eligible_states"] == 1
    assert metrics["loss_sum"] == float(loss.detach())
    assert logits.grad[0, 8, canonical] != 0 and logits.grad[0, 8, alias] != 0
    # Tempering physical classes is intentionally different from tempering
    # original alias tokens and then adding their probabilities.
    token_vector = logits.detach()[0, 8, objective.tables[("coord", "X")]["ids"]].float()
    token_probability = torch.softmax(token_vector / .7, -1)
    original_ids = objective.tables[("coord", "X")]["ids"]
    wrong_mass = token_probability[(original_ids == canonical) | (original_ids == alias)].sum()
    assert abs(float(expected.detach().exp().reciprocal()) - float(wrong_mass)) > .003


def test_ineligible_prefix_and_padding_keep_original_state_denominator(monkeypatch):
    tokenizer, support = codec()
    force_state(monkeypatch, branch="prefix", position=2)
    valid = data.make_periodic_v2_training_example(source(count=1), tokenizer, support, view=0, epoch=0, seed=6)
    invalid_row = source(index=1, count=1)
    invalid_row["answer"] = invalid_row["answer"].replace("<LA_040>", "<LA_000>")
    invalid = data.make_periodic_v2_training_example(invalid_row, tokenizer, support, view=0, epoch=0, seed=6)
    padding = dict(deepcopy(valid), is_padding=True, sample_weight=0.)
    materialized = batch([valid, invalid, padding])
    logits = torch.zeros(3, materialized["input_ids"].shape[1], max(tokenizer.vocab.values()) + 1, requires_grad=True)
    loss, metrics, conflicts = PeriodicV2Objective(tokenizer, support, "cpu")(logits, materialized)
    # Legal positive lengths have 500 tokens. Only one of three states contributes.
    torch.testing.assert_close(loss, torch.tensor(np.log(500) / 3, dtype=torch.float32))
    loss.backward()
    assert not logits.grad[1:].any()
    assert metrics["state_count"] == 3 and metrics["real_states"] == 2 and metrics["padding_states"] == 1
    assert metrics["prefix_eligible_states"] == 1 and metrics["prefix_ineligible_states"] == 1
    assert len(conflicts) == 1 and "teacher_predecessor_unreachable" in conflicts[0]["reasons"]
    assert metrics["construction_prefix_length_B_requested_tokens"] == 2
    assert metrics["construction_prefix_length_B_eligible_tokens"] == 1
    json.dumps({"metrics": metrics, "conflicts": conflicts}, allow_nan=False)


def test_dense_exact_ce_has_field_weight_and_no_extra_legal_term(monkeypatch):
    tokenizer, support = codec()
    force_state(monkeypatch, branch="dense", masks=[True, False, False, False, False, False, True, False, False])
    example = data.make_periodic_v2_training_example(source(count=1), tokenizer, support, view=0, epoch=0, seed=3)
    materialized = batch([example])
    logits = torch.zeros(1, len(example["input_body"]), max(tokenizer.vocab.values()) + 1, requires_grad=True)
    loss, metrics, conflicts = PeriodicV2Objective(tokenizer, support, "cpu")(logits, materialized)
    # Length typed support retains 000; coord 000 contains the two raw aliases.
    alias_odds = torch.exp(torch.logaddexp(torch.tensor(0.), torch.tensor(0.)) / .7)
    coord_ce = -torch.log(alias_odds / (99 + alias_odds))
    expected = torch.tensor(np.log(501) / (12 * .4), dtype=torch.float32) + coord_ce / (6 * .4)
    torch.testing.assert_close(loss, expected, atol=2e-6, rtol=1e-6)
    assert not conflicts and "prefix_states" not in metrics
    assert metrics["construction_dense_eligible_tokens"] == 2
    force_state(monkeypatch, branch="dense")
    empty = data.make_periodic_v2_training_example(source(count=1), tokenizer, support, view=0, epoch=0, seed=3)
    empty_logits = torch.ones_like(logits, requires_grad=True)
    empty_loss, empty_metrics, _ = PeriodicV2Objective(tokenizer, support, "cpu")(empty_logits, batch([empty]))
    assert float(empty_loss.detach()) == 0 and empty_metrics["state_count"] == 1 and empty_metrics["empty_states"] == 1
    empty_loss.backward()
    assert not empty_logits.grad.any()
