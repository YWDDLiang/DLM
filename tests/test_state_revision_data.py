from collections.abc import Mapping
from copy import deepcopy

import numpy as np
import pytest
import torch

from crystal_dlm import state_revision_data as data
from crystal_dlm.dynamic_crystal import arrays_to_dynamic_tokens, parse_dynamic_answer
from crystal_dlm.fixed_slot import build_special_tokens
from crystal_dlm.programmed_path_runtime import complete_geometry_supported, cooperative_slots
from crystal_dlm.spad_program import (
    coordinate_positions, prefilled_positions, program_from_element_order,
    reverse_species_block_revision_slots,
)


class Tokenizer:
    def __init__(self):
        self.vocab = {token: i for i, token in enumerate(build_special_tokens())}
        self.mask_token_id = len(self.vocab)
        self.vocab["<MASK>"] = self.mask_token_id

    def get_vocab(self):
        return self.vocab


@pytest.fixture(scope="module")
def codec():
    tok = Tokenizer()
    v = tok.vocab
    constraints = {
        "representation": "dynamic_v1", "coord_period": 100, "length_step": .1,
        "count_token_to_n": {v[f"<N_{n:03d}>"]: n for n in range(1, 21)},
        "length_token_to_bin": {
            a: {v[f"<{a}_{n:03d}>"]: n for n in range(501)} for a in ("LA", "LB", "LC")
        },
        "angle_token_to_bin": {
            a: {v[f"<{a}_{n:03d}>"]: n for n in range(1, 180)} for a in ("AA", "AB", "AG")
        },
        "coord_token_to_bin": {
            a: {v[f"<{a}_{n:03d}>"]: n for n in range(101)} for a in "XYZ"
        },
        "pbc_min_distance_A": .5, "pbc_image_radius": 2,
        "pbc_min_distance_mask": True, "canonicalize_periodic_alias": True,
    }
    return tok, constraints


def source(index=0, *, n=None):
    species = ["O", "O", "Na", "Cl"] if n is None else ["H"] * n
    elements, counts = (["O", "Na", "Cl"], [2, 1, 1]) if n is None else (["H"], [n])
    fractional = [[(i * .21) % 1, (i * .37) % 1, (i * .13) % 1] for i in range(len(species))]
    tokens, _ = arrays_to_dynamic_tokens([4., 4.8, 5.4], [81., 93., 107.], species, fractional)
    return {
        "source_row_idx": index, "source_split": "train", "prompt": "teacher-rich prompt",
        "answer": "".join(tokens), "plan_state": {"N": len(species), "elements": elements, "counts": counts},
        "species_program": ["Cl", "O", "Na"] if n is None else ["H"],
        "species_program_source": "contact_tree_teacher", "teacherPlan": {"kind": "rich"},
        "sample_weight": 0.,
        # Legacy closure masks must not be mistaken for this joint transaction.
        "forced_mask_positions": [1, 2, 3], "loss_positions": [1],
    }


def native(row, tok):
    return [tok.vocab[t] for t in parse_dynamic_answer(row["answer"], strict=True)["tokens"]]


def program(row):
    return program_from_element_order(
        row["plan_state"], row["species_program"], order_source=row["species_program_source"],
    )


def assert_layout(row, example, tok):
    clean = native(row, tok)
    old, canvas = example["old_body"], example["input_body"]
    transaction, step = example["transaction_positions"], example["transaction_step"]
    assert len(old) == len(canvas) == 7 + 4 * example["num_atoms"]
    assert all(type(t) is int for t in old + canvas)
    assert old != clean
    assert old is not canvas
    assert example["position"] == transaction[step]
    assert example["target_token"] == clean[example["position"]]
    assert canvas[example["position"]] == tok.mask_token_id
    assert [canvas[p] for p in transaction[:step]] == [clean[p] for p in transaction[:step]]
    assert all(canvas[p] == tok.mask_token_id for p in transaction[step:])
    assert {p for p, t in enumerate(canvas) if t == tok.mask_token_id} == set(transaction[step:])
    assert all(canvas[p] == old[p] == clean[p] for p in range(len(clean)) if p not in transaction)
    assert all(old[p] == canvas[p] == clean[p] for p in prefilled_positions(example["num_atoms"]))
    assert example["corruption_info"]["changed_positions"] == [p for p in range(len(clean)) if old[p] != clean[p]]
    assert not set(prefilled_positions(example["num_atoms"])) & set(transaction)
    completed = canvas.copy()
    for p in transaction[step:]:
        completed[p] = clean[p]
    assert completed == clean  # Never a chimera of different structures.


@pytest.mark.parametrize("index", [0, 1, 24, 25])
@pytest.mark.parametrize("epoch", [0, 1, 2])
def test_reproducible_per_source_epoch_without_global_rng_or_mutation(codec, index, epoch):
    tok, constraints = codec
    row = source(index)
    before = deepcopy(row)
    first = data.make_state_revision_example(row, tok, constraints, seed=731, epoch=epoch)
    np.random.seed(99)
    np.random.random(100)
    torch.manual_seed(77)
    second = data.make_state_revision_example(row, tok, constraints, seed=731, epoch=epoch)
    assert first == second and row == before
    assert first["phase"] == ("cooperative" if (index + epoch) % 2 == 0 else "closure")
    assert first["sample_weight"] == 0.
    for key in ("prompt", "plan_state", "teacherPlan", "species_program", "species_program_source"):
        assert first[key] == row[key]
    first["plan_state"]["counts"][0] = -1
    assert row == before
    assert "forced_mask_positions" not in first and "loss_positions" not in first
    assert_layout(row, first, tok)


@pytest.mark.parametrize("index", [0, 1])
def test_actual_prefix_suffix_and_runtime_transaction_order(codec, index):
    tok, constraints = codec
    row = source(index)
    clean = native(row, tok)
    positions_seen, blocks_seen = set(), set()
    endpoints = set()
    changed_prefix_seen = False
    for seed in range(64):
        example = data.make_state_revision_example(row, tok, constraints, seed=seed)
        assert_layout(row, example, tok)
        slots = example["corruption_info"]["active_slots"]
        if index == 0:
            assert tuple(slots) == cooperative_slots(torch.tensor(example["old_body"]), program(row), constraints)
            expected = [*range(1, 7), *(p for s in slots for p in coordinate_positions(s))]
        else:
            block = example["corruption_info"]["block_index"]
            blocks_seen.add(block)
            assert tuple(slots) == reverse_species_block_revision_slots(program(row))[block]
            expected = [p for s in slots for p in coordinate_positions(s)]
            assert example["old_body"][1:7] == clean[1:7]
        assert example["transaction_positions"] == expected
        step = example["transaction_step"]
        if step == 0:
            endpoints.add("first")
        if step == len(expected) - 1:
            endpoints.add("last")
        changed_prefix_seen |= any(example["old_body"][p] != clean[p] for p in expected[:step])
        positions_seen.add(example["position"])
    assert endpoints == {"first", "last"}
    assert changed_prefix_seen
    assert len(positions_seen) > 6
    if index == 1:
        assert blocks_seen == {0, 1, 2}


def test_region_change_reverts_only_coordinates_and_keeps_source(codec, monkeypatch):
    tok, constraints = codec
    row = source(n=4)
    tokens, _ = arrays_to_dynamic_tokens(
        [4., 4., 4.], [90., 90., 90.], ["H"] * 4,
        [[0., 0., 0.], [.20, 0., 0.], [.21, 0., 0.], [.45, .4, .4]],
    )
    row["answer"] = "".join(tokens)
    snapshots = {}

    def switch_region(base, arrays, slots, rng, vocab, config, constraints):
        assert slots == (0, 1)
        snapshots["cell_only"] = base.copy()
        candidate = base.copy()
        candidate[coordinate_positions(1)[0]] = vocab["<X_040>"]
        return candidate, {"coordinate_fallback": None}

    monkeypatch.setattr(data, "_coordinate_corruption", switch_region)
    example = data.make_state_revision_example(row, tok, constraints, seed=17)
    info = example["corruption_info"]
    assert info["region_fallback"] and not info["coordinate_corruption_applied"]
    assert info["region_before_coordinate_corruption"] == [0, 1]
    assert info["region_after_coordinate_corruption"] == [0, 2]
    assert example["old_body"] == snapshots["cell_only"]
    assert example["old_body"][1:7] != native(row, tok)[1:7]
    assert tuple(info["active_slots"]) == cooperative_slots(torch.tensor(example["old_body"]), program(row), constraints)
    assert_layout(row, example, tok)


@pytest.mark.parametrize("n", [1, 2, 3, 20])
@pytest.mark.parametrize("index", [0, 1])
def test_small_and_full_mp20_sizes(codec, n, index):
    tok, constraints = codec
    row = source(index, n=n)
    example = data.make_state_revision_example(row, tok, constraints, seed=43)
    assert_layout(row, example, tok)
    if index == 0:
        assert len(example["corruption_info"]["active_slots"]) == min(n, max(2, (n + 1) // 2))


@pytest.mark.parametrize("index", [0, 1])
def test_quantization_resolution_fallback_is_not_a_clean_identity(codec, monkeypatch, index):
    tok, constraints = codec
    monkeypatch.setattr(data, "_CELL_STRAIN_BOUND", 0.)
    monkeypatch.setattr(data, "_SHARED_DISPLACEMENT_BOUND_A", 0.)
    monkeypatch.setattr(data, "_LOCAL_DISPLACEMENT_BOUND_A", 0.)
    row = source(index, n=1)
    example = data.make_state_revision_example(row, tok, constraints, seed=0)
    assert_layout(row, example, tok)
    assert example["corruption_info"]["coordinate_fallback"] == "displacement_below_codec_resolution"
    if index == 0:
        assert example["corruption_info"]["cell_fallback"] == "strain_below_codec_resolution"


class NoOutcomeReads(Mapping):
    def __init__(self, row):
        self.row = row

    def __getitem__(self, key):
        if "energy" in key.lower():
            raise AssertionError("outcome was read")
        return self.row[key]

    def __iter__(self):
        raise AssertionError("bulk copying a row can expose outcomes")

    def __len__(self):
        return len(self.row)


@pytest.mark.parametrize("index", [0, 1])
def test_runtime_illegal_teacher_is_retained_without_reading_energy(codec, index):
    tok, constraints = codec
    row = source(index, n=2)
    tokens, _ = arrays_to_dynamic_tokens([4.] * 3, [90.] * 3, ["H"] * 2, [[0.] * 3] * 2)
    row["answer"] = "".join(tokens)
    row["energy"] = object()
    assert not complete_geometry_supported(torch.tensor(native(row, tok)), constraints)
    example = data.make_state_revision_example(NoOutcomeReads(row), tok, constraints, seed=9)
    assert_layout(row, example, tok)
    assert example["source_row_idx"] == index
    assert example["requires_runtime_support_check"] and not example["runtime_support_checked"]
    assert not example["outcomes_read"] and "energy" not in example


def test_native_periodic_alias_teacher_not_silently_rewritten(codec):
    tok, constraints = codec
    row = source(1, n=1)
    row["answer"] = row["answer"].replace("<X_000>", "<X_100>")
    for seed in range(32):
        example = data.make_state_revision_example(row, tok, constraints, seed=seed)
        assert_layout(row, example, tok)
        if example["position"] == 8:
            assert example["target_token"] == tok.vocab["<X_100>"]
            assert example["requires_runtime_support_check"]
            break
    else:
        pytest.fail("did not sample the native alias target")


def test_plan_mismatch_is_explicit_and_never_reorders_atoms(codec):
    tok, constraints = codec
    row = source()
    row["answer"] = row["answer"].replace("<E_Na>", "<E_Cl>", 1)
    with pytest.raises(ValueError, match="reordering is forbidden"):
        data.make_state_revision_example(row, tok, constraints, seed=1)
