from copy import deepcopy
import json

import numpy as np
import pytest
import torch

from crystal_dlm import periodic_v2_training_data as data
from crystal_dlm.dynamic_crystal import arrays_to_dynamic_tokens
from crystal_dlm.periodic_base_training_data import prepare_periodic_base_source
from crystal_dlm.programmed_path_runtime import process_path_logits
from crystal_dlm.r5_dynamic_length import exact_dynamic_schema_constraints
from test_periodic_base_training_data import source
from test_state_programmed_runtime import TinyTokenizer, constraints


def codec(*, sparse_ids=False):
    tokenizer = TinyTokenizer()
    if sparse_ids:
        tokenizer.vocab = {name: 2 + 3 * index for index, name in enumerate(tokenizer.vocab)}
        tokenizer.mask_id = tokenizer.vocab["<MASK>"]
    tokenizer.mask_token_id = tokenizer.mask_id
    tokenizer.pad_token_id = tokenizer.vocab["<PAD>"]
    support = constraints(tokenizer)
    support.update(
        duplicate_coordinate_mask=True, lattice_volume_mask=True, min_lattice_rad=1e-4,
        gamma_bin_to_token_id={i: tokenizer.vocab[f"<AG_{i:03d}>"] for i in range(1, 180)},
        z_bin_to_token_id={i: tokenizer.vocab[f"<Z_{i:03d}>"] for i in range(101)},
        zero_length_token_ids_by_position={i + 1: tokenizer.vocab[f"<{axis}_000>"]
                                          for i, axis in enumerate(("LA", "LB", "LC"))},
    )
    return tokenizer, support


def force_state(monkeypatch, *, branch, position=1, object_index=0, masks=None, dropped=False):
    original = data._rng

    class Choices:
        def random(self):
            return 0. if branch == "prefix" else .99

        def integers(self, high):
            return object_index

        def choice(self, candidates):
            assert position in candidates
            return position

    class Masks:
        def uniform(self, low, high):
            return .4

        def random(self, size):
            if masks is None:
                return np.ones(size)
            assert len(masks) == size
            return np.where(masks, 0., 1.)

    class Metadata:
        def random(self):
            return 0. if dropped else .99

    def selected_rng(*args, stream, **kwargs):
        return {"branch_and_position": Choices(), "mask": Masks(),
                "noise_metadata": Metadata()}.get(stream) or original(*args, stream=stream, **kwargs)

    monkeypatch.setattr(data, "_rng", selected_rng)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_compact_support_alias_logits_and_gradients_match_original_integer_ids(dtype):
    tokenizer, support = codec(sparse_ids=True)
    prepared = prepare_periodic_base_source(source(), tokenizer, support)
    auditor = data.PrefixSupportAuditor(tokenizer, support)
    vocabulary_size = max(tokenizer.vocab.values()) + 1
    allowed = torch.zeros(len(prepared.clean_tokens), vocabulary_size, dtype=torch.bool)
    for position, ids in enumerate(exact_dynamic_schema_constraints(tokenizer, 2)):
        allowed[position, ids] = True
    native_ids = torch.tensor(auditor.original_ids)
    for position in (1, 6, 8, 10, 14):
        current = list(prepared.clean_tokens)
        current[position] = tokenizer.mask_id
        x = torch.tensor([current])
        compact_x = auditor.remap_body(current)
        generator = torch.Generator().manual_seed(119 + position)
        raw = torch.randn(1, len(current), vocabulary_size, generator=generator, dtype=dtype, requires_grad=True)
        compact_raw = raw.detach()[..., native_ids].clone().requires_grad_(True)
        full, full_bad = process_path_logits(
            raw, x, prompt_length=0, gen_length=len(current), allowed=allowed,
            grammar=None, constraints=support, positions={0: position}, mask_id=tokenizer.mask_id,
        )
        compact, compact_bad = auditor.process_compact_logits(compact_raw, compact_x, position=position, count=2)
        assert full_bad == compact_bad
        assert torch.equal(full[0, position, native_ids], compact[0, position])
        target = prepared.clean_tokens[position]
        full_loss = -torch.log_softmax(full[0, position].float() / .7, -1)[target]
        compact_loss = -torch.log_softmax(compact[0, position].float() / .7, -1)[auditor.to_compact[target]]
        torch.testing.assert_close(full_loss, compact_loss, atol=2e-6, rtol=1e-6)
        full_loss.backward()
        compact_loss.backward()
        torch.testing.assert_close(raw.grad[..., native_ids], compact_raw.grad, atol=2e-6, rtol=1e-5)
        if position == 8:
            canonical, alias = support["coordinate_alias_token_ids"]["X"]
            assert raw.grad[0, position, canonical] != 0
            assert raw.grad[0, position, alias] != 0


def test_illegal_predecessor_keeps_later_legal_target_ineligible(monkeypatch):
    tokenizer, support = codec()
    row = source(count=1)
    row["answer"] = row["answer"].replace("<LA_040>", "<LA_000>")
    force_state(monkeypatch, branch="prefix", position=2)
    example = data.make_periodic_v2_training_example(row, tokenizer, support, view=0, epoch=0, seed=4)
    assert example["teacher_target_legal"]
    assert not example["prefix_predecessors_reachable"]
    assert not example["prefix_reachable"] and not example["legal_eligible"]
    assert example["first_illegal_teacher_position"] == 1
    assert example["sample_weight"] == example["source_sample_weight"] == 1.


def test_prefix_uses_species_program_order_and_runtime_active_scope(monkeypatch):
    tokenizer, support = codec()
    tokens, _ = arrays_to_dynamic_tokens(
        [4., 5., 6.], [85., 95., 105.], ["H", "H", "He", "He"],
        [[0., 0., 0.], [.3, .2, .1], [.6, .4, .2], [.9, .6, .3]],
    )
    row = dict(source(count=4), answer=" ".join(tokens),
               plan_state={"N": 4, "elements": ["H", "He"], "counts": [2, 2]},
               species_program=["He", "H"])
    force_state(monkeypatch, branch="prefix", position=8, object_index=1)
    prepared = prepare_periodic_base_source(row, tokenizer, support)
    transaction = list(prepared.transaction_positions)
    step = transaction.index(8)
    assert transaction[6:9] == [16, 17, 18]
    construction = data.make_periodic_v2_training_example(prepared, tokenizer, support, view=0, epoch=0, seed=9)
    repair = data.make_periodic_v2_training_example(prepared, tokenizer, support, view=1, epoch=0, seed=9)
    for example in (construction, repair):
        assert example["input_body"][16:19] == list(prepared.clean_tokens[16:19])
        assert example["input_body"][12:15] == [tokenizer.mask_id] * 3
        assert example["positions"] == [8] and example["transaction_step"] == step
    assert construction["transaction_positions"] == transaction[step:]
    assert construction["old_body"] == construction["input_body"]
    assert repair["transaction_positions"] == transaction
    assert all(token != tokenizer.mask_id for token in repair["old_body"])
    assert construction["phase"] == "construct" and repair["phase"] == "full_cell_repair"


def test_dense_masks_do_not_leak_clean_values_and_empty_mask_is_kept(monkeypatch):
    tokenizer, support = codec()
    force_state(monkeypatch, branch="dense", masks=[True, False, False, False, False, False, True, False, False])
    row = source(count=1)
    original = data.make_periodic_v2_training_example(row, tokenizer, support, view=0, epoch=0, seed=3)
    changed = dict(row, answer=row["answer"].replace("<LA_040>", "<LA_080>").replace("<X_000>", "<X_030>"))
    other = data.make_periodic_v2_training_example(changed, tokenizer, support, view=0, epoch=0, seed=3)
    assert original["input_body"] == other["input_body"]
    assert original["old_body"] == other["old_body"] == original["input_body"]
    assert original["targets"] != other["targets"]
    assert original["phase"] == "structured_denoise"
    assert original["transaction_positions"] == original["positions"] == [1, 8]
    force_state(monkeypatch, branch="dense")
    empty = data.make_periodic_v2_training_example(row, tokenizer, support, view=0, epoch=0, seed=3)
    assert empty["positions"] == empty["targets"] == [] and empty["empty_supervision"]
    assert empty["sample_weight"] == 1. and empty["transaction_positions"] == []


def test_repair_dense_visible_values_are_noisy_and_noise_dropout_is_joint(monkeypatch):
    tokenizer, support = codec()
    prepared = prepare_periodic_base_source(source(count=1), tokenizer, support)
    noisy = list(prepared.clean_tokens)
    noisy[1], noisy[2], noisy[8] = [tokenizer.vocab[t] for t in ("<LA_050>", "<LB_060>", "<X_020>")]
    info = {"applied": True, "fallback_to_clean": False, "old_state_admitted": True,
            "attempt_count": 1, "rejected_attempts": 0, "quantized_unchanged": False}
    monkeypatch.setattr(data, "corrupt_periodic_v2_source", lambda *args, **kwargs: (noisy.copy(), (.1, .03, .15), info.copy()))
    masks = [True, False, False, False, False, False, False, False, False]
    force_state(monkeypatch, branch="dense", masks=masks, dropped=False)
    known = data.make_periodic_v2_training_example(prepared, tokenizer, support, view=1, epoch=0, seed=2)
    assert known["old_body"] == noisy
    assert known["input_body"][1] == tokenizer.mask_id
    assert known["input_body"][2] == noisy[2] != prepared.clean_tokens[2]
    assert known["input_body"][8] == noisy[8] != prepared.clean_tokens[8]
    assert known["targets"] == [prepared.clean_tokens[1]]
    assert known["numeric_noise_components"] == [.1, .03, .15]
    force_state(monkeypatch, branch="dense", masks=masks, dropped=True)
    unknown = data.make_periodic_v2_training_example(prepared, tokenizer, support, view=1, epoch=0, seed=2)
    assert unknown["numeric_noise_components"] == [-1., -1., -1.]
    assert unknown["declared_numeric_noise_components"] == [.1, .03, .15]
    assert known["numeric_noise_level"] == unknown["numeric_noise_level"] == -1.


def test_dataset_keeps_two_views_and_global24_padding_without_mutating_sources(monkeypatch):
    tokenizer, support = codec()
    force_state(monkeypatch, branch="dense")
    rows = [source(index=i, count=1) for i in range(13)]
    original = deepcopy(rows)
    dataset = data.PeriodicV2TrainingDataset(rows, tokenizer, support, seed=9, expected_rows=13)
    assert rows == original
    assert dataset.real_length == 26 and dataset.padded_length == 48
    assert [(dataset[i]["source_row_idx"], dataset[i]["view"]) for i in range(26)] == [
        (source_index, view) for source_index in range(13) for view in (0, 1)]
    for i in range(26, 48):
        assert dataset[i]["is_padding"] and dataset[i]["sample_weight"] == 0.
    assert 2 * ((2 * 27136 + 23) // 24) == 4524
    assert ((2 * 27136 + 23) // 24) * 24 - 2 * 27136 == 16


def test_complete_examples_and_state_audit_are_strict_json_serializable(monkeypatch):
    tokenizer, support = codec()
    prepared = prepare_periodic_base_source(source(), tokenizer, support)
    auditor = data.PrefixSupportAuditor(tokenizer, support)
    fields = (
        "input_body", "old_body", "positions", "targets", "corruption_info",
        "numeric_noise_components", "declared_numeric_noise_components",
        "legal_eligible", "prefix_reachable", "prefix_predecessors_reachable",
        "teacher_target_legal", "old_state_admitted",
    )
    for epoch in (0, 1):
        for view in (0, 1):
            for branch in ("prefix", "dense"):
                force_state(monkeypatch, branch=branch, position=8, object_index=1,
                            masks=[True] * 12, dropped=bool(epoch))
                example = data.make_periodic_v2_training_example(
                    prepared, tokenizer, support, view=view, epoch=epoch, seed=91,
                    prefix_auditor=auditor,
                )
                # No default converter: NumPy arrays/scalars and NaN must fail.
                assert json.loads(json.dumps(example, allow_nan=False)) == example
                audit = {field: example[field] for field in fields}
                assert json.loads(json.dumps(audit, allow_nan=False)) == audit
