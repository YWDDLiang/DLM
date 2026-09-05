from copy import deepcopy
from dataclasses import replace
import json
import hashlib

import pytest
import torch
from pymatgen.core import Lattice, Structure

from crystal_dlm.programmed_path_runtime import ProgrammedPathSampler, full_cell_transaction_positions
from crystal_dlm.programmed_path_data import trace_terminal_body
from crystal_dlm.terminal_repair_data import (
    encode_terminal_pair, make_terminal_repair_example, composition_key, repair_split, target_admission,
)
from test_state_programmed_runtime import TinyTokenizer, TinyBase, body, constraints, program
from crystal_dlm.r5_dynamic_length import exact_dynamic_schema_constraints


def pair_inputs():
    tok = TinyTokenizer()
    native = body(tok, count=2, length=40, x=35)
    reverse = {value: key for key, value in tok.vocab.items()}
    record = {
        "trajectory_id": "0:1:0", "group_id": "0", "source_row_idx": 0, "source_split": "train",
        "prompt": "crystal", "plan_state": {"N": 2, "elements": ["H"], "counts": [2]},
        "species_program": ["H"], "species_program_source": "test",
        "success": True, "body": " ".join(reverse[value] for value in native),
        "final_body_token_ids": native, "checkpoint": "own-policy",
        "trace": {"initial_body": native, "mask_id": tok.mask_id,
                  "events": [{"op": "begin", "phase": "closure", "kind": "species_block", "positions": [8, 9, 10]},
                             {"op": "draw", "phase": "closure", "position": 8, "token": native[8], "log_probability": -1.},
                             {"op": "rollback", "positions": [8, 9, 10], "reason": "test"},
                             {"op": "end", "phase": "closure"}],
                  "temperature": .7},
    }
    final = Structure(Lattice.cubic(5), ["H", "H"], [[.9999, 0, 0], [.5, 0, 0]])
    label = {
        "trajectory_id": record["trajectory_id"], "source_split": "train", "status": "verified",
        "verified": True, "terminal_consistency": {"status": "consistent"},
        "raw_energy": 2., "terminal_energy": -1., "gap": 3., "final_structure": final.as_dict(),
        "endpoint_cache_key": hashlib.sha256(record["body"].encode()).hexdigest(),
    }
    return tok, record, label


def test_terminal_quantization_preserves_atoms_and_does_not_inherit_verification():
    tok, record, label = pair_inputs()
    pair = encode_terminal_pair(record, label, tok, constraints(tok))
    assert pair["target_body_token_ids"][8] == tok.vocab["<X_000>"]
    assert not pair["quantized_target_verified"] and not pair["target_supervision_ready"]
    assert [pair["target_body_token_ids"][p] for p in (0, 7, 11)] == [record["final_body_token_ids"][p] for p in (0, 7, 11)]
    assert pair["parent_body_token_ids"] == record["final_body_token_ids"]


def test_targets_reject_atom_identity_change_and_encoding_clipping():
    tok, record, label = pair_inputs()
    wrong = deepcopy(label)
    wrong["final_structure"] = Structure(Lattice.cubic(5), ["He", "H"], [[0, 0, 0], [.5, 0, 0]]).as_dict()
    with pytest.raises(ValueError, match="atom_order_or_species"):
        encode_terminal_pair(record, wrong, tok, constraints(tok))
    clipped = deepcopy(label)
    clipped["final_structure"] = Structure(Lattice.cubic(80), ["H", "H"], [[0, 0, 0], [.5, 0, 0]]).as_dict()
    with pytest.raises(ValueError, match="clipped"):
        encode_terminal_pair(record, clipped, tok, constraints(tok))


def test_holdout_is_by_reduced_composition_before_outcomes():
    _, record, _ = pair_inputs()
    doubled = dict(record, group_id="other", success=False, plan_state={"N": 4, "elements": ["H"], "counts": [4]})
    assert composition_key(record) == composition_key(doubled)
    assert repair_split(record) == repair_split(doubled)


def test_quantized_admission_checks_physics_and_target_energy():
    tok, record, label = pair_inputs()
    pair = encode_terminal_pair(record, label, tok, constraints(tok))
    q = dict(trajectory_id=pair["trajectory_id"], verified=True, status="verified",
             endpoint_cache_key=hashlib.sha256(pair["body"].encode()).hexdigest(),
             gap=.01, raw_energy=-.99,
             raw={"force_max_eV_A": .3, "stress_max_GPa": 1.})
    assert target_admission(pair, q) == []
    assert "quantized_gap" in target_admission(pair, dict(q, gap=.4))
    assert "target_does_not_lower_parent_energy" in target_admission(pair, dict(q, raw_energy=3.))
    assert "quantized_relaxation_unverified" in target_admission(pair, dict(q, verified=False))


def test_teacher_state_has_full_old_error_and_only_a_reachable_target_prefix():
    tok, record, label = pair_inputs()
    pair = encode_terminal_pair(record, label, tok, constraints(tok))
    with pytest.raises(ValueError, match="separate admission"):
        make_terminal_repair_example(pair, family=0, epoch=0, seed=8, mask_id=tok.mask_id)
    pair["target_supervision_ready"] = True
    positions = full_cell_transaction_positions(program())
    for family in range(3):
        example = make_terminal_repair_example(pair, family=family, epoch=0, seed=8, mask_id=tok.mask_id)
        assert example["old_body"] == pair["parent_body_token_ids"]
        before = True
        for pos in positions:
            if pos == example["position"]:
                before = False
            assert example["input_body"][pos] == (pair["target_body_token_ids"][pos] if before else tok.mask_id)
        assert example["target_token"] == pair["target_body_token_ids"][example["position"]]
        assert set(example["transaction_positions"]) == set(range(1, 7)) | {8, 9, 10, 12, 13, 14}


def test_structured_corruption_supplies_noise_without_changing_composition():
    tok, record, label = pair_inputs()
    pair = encode_terminal_pair(record, label, tok, constraints(tok))
    pair["target_supervision_ready"] = True
    example = make_terminal_repair_example(pair, family=0, epoch=0, seed=9, mask_id=tok.mask_id,
                                           tokenizer=tok, constraints=constraints(tok))
    assert example["phase"] == "structured_denoise"
    assert 0 < example["numeric_noise_level"] <= 1
    assert example["old_body"] != pair["parent_body_token_ids"]
    assert [example["old_body"][i] for i in (0, 7, 11)] == [pair["target_body_token_ids"][i] for i in (0, 7, 11)]


def test_full_cell_failure_rolls_back_all_lattice_and_coordinates():
    tok, record, label = pair_inputs()
    target = encode_terminal_pair(record, label, tok, constraints(tok))["target_body_token_ids"]
    class FailLast(ProgrammedPathSampler):
        def _draw(self, x, old, positions, transactions, attention_mask, *, phase, salt):
            assert phase == "full_cell_repair"
            if list(positions.values()) == [14]:
                return set(positions)
            for row, pos in positions.items():
                x[row, self.prompt_length + pos] = target[pos]
                self.traces[row].events.append({"op": "draw", "phase": phase, "position": pos,
                                               "token": target[pos], "log_probability": 0.})
            return set()
    sampler = FailLast(TinyBase(len(tok.vocab)), prompt_length=1, gen_length=15, mask_id=tok.mask_id,
                       programs=[program()], allowed_token_ids=exact_dynamic_schema_constraints(tok, 2),
                       atom_count_grammar=None, constraints=constraints(tok), sampling_seeds=[8])
    x = torch.tensor([[0] + record["final_body_token_ids"]])
    result, traces = sampler.run(x, torch.ones_like(x), construct=False, cooperative=False,
                                closure=False, full_cell_repair=True)
    assert torch.equal(x, result)
    assert trace_terminal_body(traces[0]) == record["final_body_token_ids"]
    rollback = next(event for event in traces[0]["events"] if event["op"] == "rollback")
    assert set(rollback["positions"]) == set(full_cell_transaction_positions(program()))


def test_target_support_includes_the_actual_lattice_rad_threshold():
    from crystal_dlm.programmed_path_runtime import complete_geometry_supported
    tok = TinyTokenizer()
    narrow = body(tok, count=1, length=100)
    for pos, axis in ((4, "AA"), (5, "AB"), (6, "AG")):
        narrow[pos] = tok.vocab[f"<{axis}_005>"]
    limits = dict(constraints(tok), lattice_volume_mask=True, min_lattice_rad=1e-4)
    assert not complete_geometry_supported(torch.tensor(narrow), limits)
