from copy import deepcopy

import numpy as np
import pytest

from crystal_dlm import periodic_base_training_data as data
from crystal_dlm.dynamic_crystal import arrays_to_dynamic_tokens, parse_dynamic_answer
from crystal_dlm.fixed_slot import EncodeDiagnostics
from test_state_programmed_runtime import TinyTokenizer, constraints, program


@pytest.fixture(scope="module")
def codec():
    tokenizer = TinyTokenizer()
    support = constraints(tokenizer)
    support.update(lattice_volume_mask=True, min_lattice_rad=1e-4)
    return tokenizer, support


def source(index=0, *, count=2, split="train"):
    tokens, _ = arrays_to_dynamic_tokens(
        [4., 5., 6.], [85., 95., 105.], ["H"] * count,
        [[i * .3, i * .2, i * .1] for i in range(count)],
    )
    return {
        "source_row_idx": index, "source_split": split, "prompt": "MP20 Plan prompt",
        "answer": " ".join(tokens), "plan_state": {"N": count, "elements": ["H"], "counts": [count]},
        "species_program": ["H"], "species_program_source": "contact_tree_teacher",
        "sample_weight": 1., "forced_mask_positions": [1], "loss_positions": [1],
    }


def clean_ids(row, tokenizer):
    tokens = parse_dynamic_answer(row["answer"], strict=True)["tokens"]
    return [tokenizer.vocab[token.replace("_100>", "_000>")
                            if token.startswith(("<X_", "<Y_", "<Z_")) else token]
            for token in tokens]


def test_construction_cannot_access_masked_clean_geometry(codec):
    tokenizer, support = codec
    row = source()
    example = data.make_periodic_base_training_example(row, tokenizer, support, view=0, epoch=0, seed=47)
    clean = clean_ids(row, tokenizer)
    assert example["phase"] == "construct"
    assert example["old_body"] == example["input_body"]
    assert example["old_body"] is not example["input_body"]
    assert .1 <= example["mask_probability"] <= 1.
    assert example["transaction_positions"] == example["positions"]
    assert example["targets"] == [clean[p] for p in example["positions"]]
    for position, token in enumerate(example["input_body"]):
        assert token == (tokenizer.mask_id if position in example["positions"] else clean[position])
    assert not {0, 7, 11}.intersection(example["positions"])
    assert "forced_mask_positions" not in example and "loss_positions" not in example


def test_empty_construction_mask_is_retained_with_no_fake_loss(codec, monkeypatch):
    tokenizer, support = codec

    class NoMasks:
        def uniform(self, low, high):
            return .1

        def random(self, size=None):
            return .9 if size is None else np.full(size, .9)

    monkeypatch.setattr(data, "_rng", lambda *args, **kwargs: NoMasks())
    row = source(count=1)
    example = data.make_periodic_base_training_example(row, tokenizer, support, view=0, epoch=0, seed=1)
    assert example["positions"] == example["targets"] == []
    assert example["empty_supervision"] and example["sample_weight"] == 0.
    assert example["source_sample_weight"] == 1.
    assert example["old_body"] == example["input_body"] == clean_ids(row, tokenizer)
    assert example["position"] == 1 and example["target_token"] == clean_ids(row, tokenizer)[1]


def test_repair_is_one_actual_full_cell_conditional_state(codec):
    tokenizer, support = codec
    row = source(3)
    example = data.make_periodic_base_training_example(row, tokenizer, support, view=1, epoch=1, seed=37)
    clean = clean_ids(row, tokenizer)
    transaction = list(data.full_cell_transaction_positions(program()))
    step = transaction.index(example["position"])
    assert example["phase"] == "full_cell_repair"
    assert example["transaction_positions"] == transaction
    assert example["positions"] == [example["position"]]
    assert example["targets"] == [clean[example["position"]]]
    assert example["target_families"] == [example["family"]]
    assert example["corruption_info"]["applied"]
    assert not example["structured_corruption_fallback"]
    for position in transaction[:step]:
        assert example["input_body"][position] == clean[position]
    for position in transaction[step:]:
        assert example["input_body"][position] == tokenizer.mask_id
    for position in (0, 7, 11):
        assert example["old_body"][position] == example["input_body"][position] == clean[position]
    assert example["old_body"] != clean


def test_corruption_clipping_falls_back_without_losing_source(codec, monkeypatch):
    tokenizer, support = codec
    row = source(8)
    clean_tokens = parse_dynamic_answer(row["answer"], strict=True)["tokens"]
    monkeypatch.setattr(data, "arrays_to_dynamic_tokens", lambda *args, **kwargs:
                        (clean_tokens, EncodeDiagnostics(length_clips=1)))
    example = data.make_periodic_base_training_example(row, tokenizer, support, view=1, epoch=0, seed=5)
    assert example["old_body"] == clean_ids(row, tokenizer)
    assert example["structured_corruption_fallback"]
    assert example["corruption_info"]["fallback_reason"] == "perturbation_tokenization_clipped"
    assert example["corruption_info"]["encoding"]["length_clips"] == 1
    assert example["declared_numeric_noise_level"] == 0.
    assert example["sample_weight"] == 1. and len(example["targets"]) == 1


def test_unsupported_clean_geometry_still_has_both_views(codec):
    tokenizer, support = codec
    row = source(count=1)
    row["answer"] = row["answer"].replace("<LA_040>", "<LA_000>")
    dataset = data.PeriodicBaseTrainingDataset([row], tokenizer, support, seed=2, expected_rows=1, effective_batch=2)
    construction, repair = dataset[0], dataset[1]
    assert dataset.real_length == 2
    assert construction["source_row_idx"] == repair["source_row_idx"] == 0
    assert repair["structured_corruption_fallback"]
    assert repair["old_body"] == clean_ids(row, tokenizer)
    assert repair["sample_weight"] == 1.
    assert repair["requires_runtime_support_check"] and not repair["runtime_support_checked"]


def test_alias_canonicalization_and_metadata_whitelist(codec):
    tokenizer, support = codec

    class Forbidden:
        def __deepcopy__(self, memo):
            raise AssertionError("read forbidden outcome data")

    row = source()
    row["answer"] = row["answer"].replace("<X_000>", "<X_100>")
    for field in ("raw_energy", "terminal_energy", "final_structure", "trace", "parent_trajectory_id"):
        row[field] = Forbidden()
    example = data.make_periodic_base_training_example(row, tokenizer, support, view=1, epoch=0, seed=6)
    assert example["alias_tokens_canonicalized"] == 1
    assert "<X_100>" not in example["answer"]
    assert "<X_100>" in row["answer"]
    assert example["source_answer_sha256"] != example["canonical_answer_sha256"]
    assert all(field not in example for field in ("raw_energy", "terminal_energy", "final_structure", "trace", "parent_trajectory_id"))
    assert example["outcomes_read"] is False


def test_family_and_noise_metadata_are_source_keyed_not_source_modulo(codec):
    tokenizer, support = codec
    prepared = data.prepare_periodic_base_source(source(0), tokenizer, support)
    families, hidden = set(), {0: set(), 1: set()}
    for epoch in range(16):
        for view in (0, 1):
            example = data.make_periodic_base_training_example(prepared, tokenizer, support, view=view, epoch=epoch, seed=27)
            assert example == data.make_periodic_base_training_example(prepared, tokenizer, support, view=view, epoch=epoch, seed=27)
            hidden[view].add(example["noise_metadata_dropped"])
            expected_noise = -1. if example["noise_metadata_dropped"] else example["declared_numeric_noise_level"]
            assert example["numeric_noise_level"] == expected_noise
            if view:
                families.add(example["family"])
    assert families == {0, 1, 2}
    assert hidden == {0: {False, True}, 1: {False, True}}


def test_dataset_keeps_each_source_twice_and_padding_weight_zero(codec):
    tokenizer, support = codec
    rows = [source(1), source(7)]
    rows[1]["sample_weight"] = 0.
    original = deepcopy(rows)
    dataset = data.PeriodicBaseTrainingDataset(rows, tokenizer, support, seed=7, expected_rows=2, effective_batch=8)
    assert rows == original
    assert dataset.real_length == 4 and len(dataset) == 8
    assert [(dataset[i]["source_row_idx"], dataset[i]["view"]) for i in range(4)] == [(1, 0), (1, 1), (7, 0), (7, 1)]
    assert dataset[2]["sample_weight"] == dataset[3]["sample_weight"] == 0.
    for index in range(4, 8):
        assert dataset[index]["is_padding"] and dataset[index]["sample_weight"] == 0.
        assert dataset[index]["input_body"] == dataset[index - 4]["input_body"]
    dataset.epoch = 1
    assert dataset[0]["epoch"] == 1
    with pytest.raises(IndexError):
        dataset[8]
    with pytest.raises(ValueError, match="duplicate"):
        data.PeriodicBaseTrainingDataset([rows[0], rows[0]], tokenizer, support, seed=7, expected_rows=2)
    with pytest.raises(ValueError, match="mixes"):
        data.PeriodicBaseTrainingDataset([source(split="val")], tokenizer, support, seed=7, expected_rows=1)
