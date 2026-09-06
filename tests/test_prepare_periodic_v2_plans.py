from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from crystal_dlm.c3fd_llama_typed_planner import SOFT_FIELDS, unit_weight_poe_log_probs
from crystal_dlm.c3fd_native_plan import build_native_body_prompt
from crystal_dlm.dynamic_crystal import arrays_to_dynamic_tokens
from crystal_dlm.periodic_v2_plan_data import (
    DIRECT, REUSED, FALLBACK, SAMPLING, build_source_requests,
    collate_prediction_requests, finish_source, sample_soft_fields,
    source_field_seed, summarize_outputs,
)
from scripts.prepare_periodic_v2_plans import assert_frozen_modules, merge_split, predict_batch


GOALS = {"meta_or_better": 0, "higher": 1}


def vocabulary():
    return {
        "species": [{"id": 0, "atomic_number": 11, "oxidation_state": 1},
                    {"id": 1, "atomic_number": 17, "oxidation_state": -1}],
        "soft_vocabulary": {
            "anion_framework": ["halide"],
            "lattice_system": ["<UNKNOWN>", "cubic", "tetragonal"],
            "spacegroup_bucket": ["<UNKNOWN>", "sg_195_230", "sg_075_142"],
            "volume_per_atom_bin": ["<UNKNOWN>", "vpa_010_020", "vpa_020_030"],
        },
    }


def sft(index, *, split="train", count=1, goal=None):
    plan = {"N": 2 * count, "elements": ["Na", "Cl"], "counts": [count, count],
            "anion_framework": "halide", "lattice_system": "cubic",
            "spacegroup_bucket": "sg_195_230", "volume_per_atom_bin": "vpa_010_020"}
    tokens, _ = arrays_to_dynamic_tokens([4., 4., 4.], [90., 90., 90.],
                                         ["Na"] * count + ["Cl"] * count,
                                         [[i / (2 * count), .1, .2] for i in range(2 * count)])
    answer = " ".join(tokens)
    result = {"source_row_idx": index, "source_split": split, "plan_state": plan,
              "prompt": build_native_body_prompt(plan), "answer": answer, "source_answer": answer,
              "species_program": ["Cl", "Na"], "species_program_source": "contact_tree_teacher",
              "sample_weight": 1., "outcomes_read": False,
              "forced_mask_positions": [12, 13, 14], "loss_positions": [12]}
    if goal is not None:
        result["stability_condition"] = goal
    return result


def metadata(index, *, split="train", count=1, goal="meta_or_better"):
    return {
        "schema": "spad_species_pointer_row_v1", "source_row_idx": index, "source_split": split,
        "stability_condition": goal, "proposal_target": {"family_id": 0, "N": 2 * count, "arity": 2},
        "species_ids": [0, 1], "count_targets": [count, count],
        "ledger_steps": [
            {"remaining_atoms": 2 * count, "net_charge": 0, "remaining_species": 2, "branch": "unset"},
            {"remaining_atoms": 2 * count, "net_charge": 0, "remaining_species": 2, "branch": "unset"},
            {"remaining_atoms": count, "net_charge": count, "remaining_species": 1, "branch": "ionic"},
            {"remaining_atoms": 0, "net_charge": 0, "remaining_species": 0, "branch": "ionic"},
        ],
        "canonical_atomic_numbers": [11, 17], "canonical_element_counts": [count, count],
        "contact_tree_order_indices": [1, 0], "contact_tree_order_symbols": ["Cl", "Na"],
        "soft_targets": {field: {"label": 1} for field in SOFT_FIELDS},
        "sample_weight": 1., "outcomes_read": False,
    }


class FakeC3FD(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.))
        self.config = SimpleNamespace(num_species=2, max_count=20, max_sequence_length=10)
        self.inputs = []

    def forward(self, context, **kwargs):
        self.inputs.append(kwargs)
        size, width = kwargs["previous_species_indices"].shape
        values = torch.tensor([1000., 0., 0.]).expand(size, width, 3)
        return SimpleNamespace(rich_logits={field: values for field in SOFT_FIELDS})


class FakeLlama(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.))

    def forward(self, *, inputs_embeds, **kwargs):
        assert kwargs["use_cache"] is False and kwargs["output_hidden_states"] is True
        return SimpleNamespace(hidden_states=(inputs_embeds,))


class FakeTyped(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.))
        self.inputs = []

    def typed_inputs_embeds(self, **kwargs):
        self.inputs.append(kwargs)
        size, width = kwargs["proposal_state_ids"].shape
        return torch.ones(size, width, 4)

    def forward(self, hidden, *, soft_position_indices):
        values = torch.zeros(hidden.shape[0], 3)
        return SimpleNamespace(soft_fields={field: values for field in SOFT_FIELDS})


class SpyPointer(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.))
        self.received_soft = []

    def decode(self, terminal, atomic, counts, valid, soft_ids):
        self.received_soft.append(soft_ids.clone())
        assert soft_ids.shape[1] == 3 and bool((soft_ids > 0).all())
        assert torch.equal(atomic, torch.tensor([[11, 17]]).expand_as(atomic))
        return torch.stack([torch.tensor([0, 1]) if row[0] == 1 else torch.tensor([1, 0]) for row in soft_ids])


def setup():
    c3fd, llama, typed, pointer = FakeC3FD(), FakeLlama(), FakeTyped(), SpyPointer()
    for module in (c3fd, llama, typed, pointer):
        module.eval().requires_grad_(False)
    bundle = SimpleNamespace(model=c3fd, vocabulary=vocabulary(), context=torch.zeros(1, 4),
                             stratum_to_index={(0, 2, 2): 0, (0, 4, 2): 1},
                             proposal_legal_mask=torch.tensor([True, True]))
    return bundle, llama, typed, pointer


def infer(requests, objects, seed=109):
    bundle, llama, typed, pointer = objects
    return predict_batch(requests, bundle=bundle, llama=llama, typed=typed, pointer=pointer,
                         goal_to_id=GOALS, seed=seed, device=torch.device("cpu"))


def test_full_source_denominator_and_unknown_goal_fallback_are_preserved():
    bundle, *_ = setup()
    rows = [sft(0), sft(1), sft(2)]  # Repeated compositions are independent source occurrences.
    requests = build_source_requests(rows, [metadata(0)], bundle, split="train", goal_to_id=GOALS, expected_rows=3)
    assert [request.source_row_idx for request in requests] == [0, 1, 2]
    assert [request.status for request in requests] == [DIRECT, FALLBACK, FALLBACK]
    fallback = finish_source(requests[1])
    assert fallback["species_program"] == ["Na", "Cl"]
    assert fallback["plan_state"] == rows[1]["plan_state"]
    assert fallback["condition_prediction"]["stability_condition"] is None
    assert not fallback["condition_prediction"]["goal_was_defaulted"]
    assert fallback["soft_plan_source"] == FALLBACK
    assert fallback["condition_prediction"]["typed_metadata_source_row_idx"] is None


def test_metadata_reuse_requires_same_split_exact_composition_and_explicit_goal():
    bundle, *_ = setup()
    rows = [sft(5), sft(9), sft(10, goal="meta_or_better"),
            sft(11, goal="higher"), sft(12, count=2, goal="meta_or_better")]
    requests = build_source_requests(rows, [metadata(9), metadata(5)], bundle, split="train", goal_to_id=GOALS)
    by_id = {request.source_row_idx: request for request in requests}
    assert by_id[10].status == REUSED and by_id[10].typed_source_row_idx == 5
    assert by_id[11].status == by_id[12].status == FALLBACK
    val_requests = build_source_requests([sft(10, split="val", goal="meta_or_better")], [], bundle,
                                        split="val", goal_to_id=GOALS)
    assert val_requests[0].status == FALLBACK
    with pytest.raises(ValueError, match="split/source"):
        build_source_requests([sft(5, split="val")], [metadata(5)], bundle, split="val", goal_to_id=GOALS)


def test_unsupported_stratum_and_missing_ledger_still_retain_sources():
    bundle, *_ = setup()
    bundle.stratum_to_index = {(0, 2, 2): 0}
    bundle.proposal_legal_mask = torch.tensor([True])
    partial = metadata(2)
    partial.pop("ledger_steps")
    requests = build_source_requests([sft(0), sft(1, count=2), sft(2)],
                                    [metadata(0), metadata(1, count=2), partial], bundle,
                                    split="train", goal_to_id=GOALS)
    assert requests[1].status == FALLBACK and requests[1].original_unavailable_reason == "unsupported_frozen_stratum"
    assert requests[2].status == REUSED and requests[2].typed_source_row_idx == 0


def test_teacher_structural_metadata_is_never_read_or_forwarded():
    class Forbidden:
        def __deepcopy__(self, memo):
            raise AssertionError("read teacher structural target")

    objects = setup()
    bundle, _, typed, pointer = objects
    raw = metadata(0)
    for field in ("soft_targets", "contact_tree_order_indices", "contact_tree_order_symbols", "audit_transcript"):
        raw[field] = Forbidden()
    requests = build_source_requests([sft(0)], [raw], bundle, split="train", goal_to_id=GOALS)
    batch = collate_prediction_requests(requests, bundle, goal_to_id=GOALS)
    assert not any("target" in name or name == "pointer_soft_field_ids" for name in batch)
    prediction = infer(requests, objects)[0]
    assert pointer.received_soft[-1].tolist() == [[prediction["soft_ids"][field] for field in SOFT_FIELDS]]
    assert set(typed.inputs[-1]) == {"stability_goal_ids", "proposal_state_ids", "previous_species_indices",
                                   "previous_count_values", "ledger_features"}


def test_inference_collator_matches_original_chemical_tensor_sequence():
    from scripts.train_c3fd_llama_typed_planner import collate_typed_rows

    bundle, *_ = setup()
    original = metadata(0)
    requests = build_source_requests([sft(0)], [original], bundle, split="train", goal_to_id=GOALS)
    actual = collate_prediction_requests(requests, bundle, goal_to_id=GOALS)
    legacy = collate_typed_rows([dict(original, schema="c3fd_llama_fused_typed_dataset_v1")], bundle=bundle)
    for name in ("stability_goal_ids", "proposal_state_ids", "previous_species_indices",
                 "previous_count_values", "previous_n_values", "ledger_features", "attention_mask", "soft_position_indices"):
        assert torch.equal(actual[name], legacy[name]), name


def test_soft_sampling_exactly_matches_the_production_transform_order():
    from scripts.sample_c3fd_llama_typed_planner import sample_log_probs

    values = vocabulary()["soft_vocabulary"]
    base = {field: torch.tensor([200., 1., .6]) for field in SOFT_FIELDS}
    residual = {field: torch.tensor([400., -.2, .3]) for field in SOFT_FIELDS}
    selected, audit = sample_soft_fields(base, residual, values, seed=88, split="train", source_row_idx=17)
    for field in SOFT_FIELDS:
        fused = unit_weight_poe_log_probs(base[field], residual[field], torch.tensor([False, True, True]))
        generator = torch.Generator().manual_seed(source_field_seed(88, "train", 17, field))
        expected = sample_log_probs(fused, rng=generator, **SAMPLING)
        assert selected[field] == expected and selected[field] != 0
        assert audit[field]["legacy_map_id"] == int(fused.argmax())


def test_source_rng_and_pointer_prediction_do_not_depend_on_batch_or_world_order():
    objects = setup()
    bundle, *_ = objects
    requests = build_source_requests([sft(i) for i in range(8)], [metadata(i) for i in range(8)],
                                    bundle, split="train", goal_to_id=GOALS)
    all_at_once = {row["source_row_idx"]: row for row in infer(requests, objects)}
    shards = [requests[rank::3] for rank in range(3)]
    shuffled = {row["source_row_idx"]: row for shard in reversed(shards) for row in infer(list(reversed(shard)), objects)}
    singles = {row["source_row_idx"]: row for request in requests for row in infer([request], objects)}
    assert all_at_once == shuffled == singles
    assert source_field_seed(3, "train", 0, SOFT_FIELDS[0]) != source_field_seed(3, "val", 0, SOFT_FIELDS[0])


def test_reused_metadata_uses_recipient_seed_and_preserves_exact_answer():
    objects = setup()
    bundle, *_ = objects
    rows = [sft(4), sft(8, goal="meta_or_better")]
    original = deepcopy(rows)
    requests = build_source_requests(rows, [metadata(4)], bundle, split="train", goal_to_id=GOALS)
    predictions = infer(requests, objects)
    outputs = [finish_source(request, prediction) for request, prediction in zip(requests, predictions)]
    assert rows == original
    assert outputs[1]["condition_prediction"]["status"] == REUSED
    assert outputs[1]["condition_prediction"]["typed_metadata_source_row_idx"] == 4
    for row, source in zip(outputs, rows):
        assert row["answer"] == source["answer"] and row["source_answer"] == source["source_answer"]
        assert row["condition_prediction"]["clean_answer_sha256_before"] == row["condition_prediction"]["clean_answer_sha256_after"]
        assert sorted(row["species_program"]) == sorted(source["plan_state"]["elements"])
    for field in SOFT_FIELDS:
        assert predictions[1]["soft_sampling_audit"][field]["seed"] == source_field_seed(109, "train", 8, field)
        assert predictions[0]["soft_sampling_audit"][field]["seed"] != predictions[1]["soft_sampling_audit"][field]["seed"]


def test_merge_and_report_keep_all_sources_and_expose_teacher_fallback():
    objects = setup()
    bundle, *_ = objects
    sources = [sft(2), sft(0), sft(1)]
    requests = build_source_requests(sources, [metadata(0), metadata(2)], bundle, split="train", goal_to_id=GOALS, expected_rows=3)
    predictions = {row["source_row_idx"]: row for row in infer([request for request in requests if request.status != FALLBACK], objects)}
    output = [finish_source(request, predictions.get(request.source_row_idx)) for request in requests]
    merged = merge_split([[output[1]], [output[2], output[0]]], sources, split="train")
    assert [row["source_row_idx"] for row in merged] == [2, 0, 1]
    report = summarize_outputs(merged)
    assert report["source_rows"] == 3 and report["unique_exact_compositions"] == 1
    assert report["teacher_soft_fallback_rows"] == 1 and report["teacher_soft_fallback_fraction"] == pytest.approx(1 / 3)
    assert not report["all_sources_predicted"] and report["all_clean_answers_unchanged"]
    with pytest.raises(ValueError, match="all original"):
        merge_split([[output[0]]], sources, split="train")
    mutated = deepcopy(output)
    mutated[0]["answer"] = mutated[0]["answer"].replace("<LA_040>", "<LA_050>")
    mutated[0]["source_answer"] = mutated[0]["answer"]
    with pytest.raises(ValueError, match="exact source clean target"):
        merge_split([mutated], sources, split="train")


def test_invalid_input_identity_or_pointer_output_fails_without_silent_filtering():
    bundle, *_ = setup()
    with pytest.raises(ValueError, match="duplicate"):
        build_source_requests([sft(0), sft(0)], [], bundle, split="train", goal_to_id=GOALS)
    wrong = metadata(0)
    wrong["canonical_element_counts"] = [2, 1]
    with pytest.raises(ValueError, match="composition differs"):
        build_source_requests([sft(0)], [wrong], bundle, split="train", goal_to_id=GOALS)
    objects = setup()
    requests = build_source_requests([sft(0)], [metadata(0)], objects[0], split="train", goal_to_id=GOALS)
    prediction = infer(requests, objects)[0]
    prediction["species_program_indices"] = [0, 0]
    with pytest.raises(ValueError, match="permutation"):
        finish_source(requests[0], prediction)
    objects[1].train()
    with pytest.raises(RuntimeError, match="frozen"):
        assert_frozen_modules(*[objects[0].model, *objects[1:]])
