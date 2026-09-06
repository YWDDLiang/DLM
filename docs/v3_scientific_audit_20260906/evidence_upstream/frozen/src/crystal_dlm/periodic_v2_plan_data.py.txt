"""Source-complete, frozen Planner conditioning for original MP20 targets.

Only the original chemical typed history may enter the Planner. Contact-tree
orders, teacher structural soft labels, crystal coordinates and energies are
not inputs to prediction. Missing metadata has an explicit, auditable fallback.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any

import torch

from crystal_dlm.c3fd_llama_typed_planner import SOFT_FIELDS, unit_weight_poe_log_probs
from crystal_dlm.c3fd_native_plan import build_native_body_prompt
from crystal_dlm.composition_identity import canonical_symbol_counts
from crystal_dlm.dynamic_crystal import parse_dynamic_answer
from crystal_dlm.fixed_slot import SYMBOL_TO_Z, Z_TO_SYMBOL
from crystal_dlm.spad_program import program_from_element_order


SCHEMA = "periodic_v2_frozen_plan_conditioning_v1"
SAMPLING = {"temperature": .9, "top_p": .95, "top_k": 0}
DIRECT = "predicted_direct"
REUSED = "predicted_metadata_reuse"
FALLBACK = "teacher_soft_fallback"
TYPED_INPUT_KEYS = (
    "schema", "source_row_idx", "source_split", "stability_condition",
    "proposal_target", "species_ids", "count_targets", "ledger_steps",
    "canonical_atomic_numbers", "canonical_element_counts",
)


def json_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def answer_sha256(answer: str) -> str:
    return hashlib.sha256(answer.encode("utf-8")).hexdigest()


def _source_index(row: Mapping[str, Any]) -> int:
    value = row["source_row_idx"]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("source_row_idx must be its original nonnegative integer")
    return value


def exact_composition(plan: Mapping[str, Any]) -> tuple[tuple[str, int], ...]:
    values = canonical_symbol_counts(plan["elements"], plan["counts"])
    if len(values) != len(plan["elements"]) or sum(count for _, count in values) != int(plan["N"]):
        raise ValueError("original Plan is not an exact unique-element composition")
    return values


def validate_clean_source(row: Mapping[str, Any], *, split: str) -> None:
    _source_index(row)
    if row.get("source_split") != split or split not in ("train", "val"):
        raise ValueError("original MP20 source split differs")
    if row.get("outcomes_read") is True:
        raise ValueError("original MP20 preparation cannot consume outcome-trained data")
    if not isinstance(row.get("answer"), str) or not isinstance(row.get("prompt"), str):
        raise ValueError("original SFT source requires its clean answer and prompt")
    if "source_answer" in row and row["source_answer"] != row["answer"]:
        raise ValueError("source_answer differs from the original clean MP20 target")
    plan = row["plan_state"]
    composition = exact_composition(plan)
    parsed = parse_dynamic_answer(row["answer"], strict=True)
    expected_species = [symbol for symbol, count in composition for _ in range(count)]
    if parsed["num_atoms"] != int(plan["N"]) or parsed["species"] != expected_species:
        raise ValueError("clean target must already use the canonical Plan site order")
    # Validate the prompt schema without replacing or re-encoding the answer.
    build_native_body_prompt(plan)


def _explicit_goal(source: Mapping[str, Any], metadata: Mapping[str, Any] | None) -> str | None:
    source_goal = source.get("stability_condition")
    metadata_goal = None if metadata is None else metadata.get("stability_condition")
    if source_goal is not None and metadata_goal is not None and source_goal != metadata_goal:
        raise ValueError("SFT and original typed metadata have different stability conditions")
    value = metadata_goal if metadata_goal is not None else source_goal
    return str(value) if isinstance(value, str) and value else None


def _reuse_key(source: Mapping[str, Any], goal: str | None):
    if goal is None:
        return None  # A missing goal must never be silently treated as meta_or_better.
    plan = source["plan_state"]
    return (source["source_split"], exact_composition(plan), str(plan["anion_framework"]), goal)


def normalized_ledger(raw: Mapping[str, Any]) -> tuple[float, ...]:
    branch = str(raw.get("branch") or "unset")
    branch_vector = {"unset": (1., 0., 0.), "ionic": (0., 1., 0.), "alloy": (0., 0., 1.)}.get(branch)
    if branch_vector is None:
        raise ValueError("unknown original chemical ledger branch")
    values = (float(raw["remaining_atoms"]) / 20., float(raw["net_charge"]) / 160.,
              float(raw["remaining_species"]) / 7., *branch_vector)
    if not all(math.isfinite(value) and abs(value) <= 1. + 1e-6 for value in values):
        raise ValueError("original chemical ledger exceeds the six-feature contract")
    return values


def validate_typed_metadata(metadata, source, bundle, *, goal_to_id,
                            max_elements=7, max_count=20) -> tuple[dict[str, Any] | None, str | None]:
    """Return only actual chemical metadata, or a disclosed eligibility reason.

    Split, source identity and composition mismatches are input corruption and
    fail the run. Missing fields or unsupported frozen-model strata are retained
    as fallback sources. Structural soft targets and contact orders are unread.
    """
    if metadata is None:
        return None, "missing_typed_metadata"
    if metadata.get("source_split") != source["source_split"] or _source_index(metadata) != _source_index(source):
        raise ValueError("typed metadata is not the same original split/source")
    if metadata.get("outcomes_read") is True:
        raise ValueError("typed metadata cannot be derived from generated outcomes")
    typed = {key: deepcopy(metadata[key]) for key in TYPED_INPUT_KEYS if key in metadata}
    if typed.get("schema") != "spad_species_pointer_row_v1":
        return None, "unsupported_typed_metadata_schema"
    missing = [key for key in TYPED_INPUT_KEYS if key not in typed]
    if missing:
        return None, "missing_typed_fields:" + ",".join(missing)
    expected = exact_composition(source["plan_state"])
    try:
        atomic = [int(value) for value in typed["canonical_atomic_numbers"]]
        counts = [int(value) for value in typed["canonical_element_counts"]]
        observed = tuple((Z_TO_SYMBOL[z], count) for z, count in zip(atomic, counts, strict=True))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("typed canonical composition is malformed") from error
    if observed != expected:
        raise ValueError("typed canonical composition differs from the clean MP20 source")
    proposal = typed["proposal_target"]
    try:
        stratum = (int(proposal["family_id"]), int(proposal["N"]), int(proposal["arity"]))
        species = [int(value) for value in typed["species_ids"]]
        actions = [int(value) for value in typed["count_targets"]]
        vocab_species = {int(row["id"]): int(row["atomic_number"]) for row in bundle.vocabulary["species"]}
        merged = Counter()
        for species_id, count in zip(species, actions, strict=True):
            merged[Z_TO_SYMBOL[vocab_species[species_id]]] += count
    except (KeyError, TypeError, ValueError) as error:
        return None, "malformed_typed_actions:" + type(error).__name__
    if canonical_symbol_counts(list(merged), list(merged.values())) != expected or stratum[1] != int(source["plan_state"]["N"]):
        raise ValueError("original typed actions change the fixed MP20 composition")
    families = bundle.vocabulary["soft_vocabulary"]["anion_framework"]
    if not 0 <= stratum[0] < len(families) or str(families[stratum[0]]) != str(source["plan_state"]["anion_framework"]):
        raise ValueError("original typed family differs from the source chemical condition")
    if stratum not in bundle.stratum_to_index:
        return None, "unsupported_frozen_stratum"
    if not bool(bundle.proposal_legal_mask[int(bundle.stratum_to_index[stratum])]):
        return None, "stratum_outside_frozen_legal_support"
    if typed["stability_condition"] not in goal_to_id:
        return None, "unsupported_original_stability_condition"
    if len(species) != stratum[2] or len(species) + 2 > int(bundle.model.config.max_sequence_length):
        return None, "typed_sequence_outside_frozen_length_support"
    if len(atomic) > max_elements or any(count < 1 or count > max_count for count in counts):
        return None, "composition_outside_pointer_support"
    if any(value < 0 or value >= int(bundle.model.config.num_species) for value in species):
        return None, "typed_species_outside_frozen_support"
    if any(value < 1 or value > int(bundle.model.config.max_count) for value in actions):
        return None, "typed_count_outside_frozen_support"
    try:
        if len(typed["ledger_steps"]) != len(species) + 2:
            return None, "missing_or_misaligned_original_ledger"
        for ledger_row in typed["ledger_steps"]:
            normalized_ledger(ledger_row)
    except (KeyError, TypeError, ValueError) as error:
        return None, "malformed_original_ledger:" + type(error).__name__
    return typed, None


@dataclass(frozen=True)
class PlanRequest:
    ordinal: int
    source: Mapping[str, Any]
    status: str
    typed_metadata: Mapping[str, Any] | None
    typed_source_row_idx: int | None
    stability_condition: str | None
    original_unavailable_reason: str | None

    @property
    def source_row_idx(self) -> int:
        return int(self.source["source_row_idx"])

    @property
    def source_split(self) -> str:
        return str(self.source["source_split"])


def build_source_requests(sft_rows, pointer_rows, bundle, *, split, goal_to_id,
                          expected_rows=None, max_elements=7, max_count=20):
    """Retain each source once; donor choice uses metadata and never outcomes."""
    if expected_rows is not None and (len(sft_rows) != expected_rows
            or {_source_index(row) for row in sft_rows} != set(range(expected_rows))):
        raise ValueError("the complete registered original MP20 source denominator changed")
    sources = {}
    for row in sft_rows:
        validate_clean_source(row, split=split)
        index = _source_index(row)
        if index in sources:
            raise ValueError("duplicate original SFT source identity")
        sources[index] = row
    metadata_by_source = {}
    for row in pointer_rows:
        index = _source_index(row)
        if index in metadata_by_source:
            raise ValueError("duplicate original typed source identity")
        if index not in sources or row.get("source_split") != split:
            raise ValueError("typed metadata is outside this original split/source denominator")
        metadata_by_source[index] = row
    eligible, unavailable, goals, donors = {}, {}, {}, {}
    for index, source in sources.items():
        metadata = metadata_by_source.get(index)
        goals[index] = _explicit_goal(source, metadata)
        typed, reason = validate_typed_metadata(metadata, source, bundle, goal_to_id=goal_to_id,
                                              max_elements=max_elements, max_count=max_count)
        if typed is None:
            unavailable[index] = reason
        else:
            eligible[index] = typed
            key = _reuse_key(source, goals[index])
            donors[key] = min(index, donors.get(key, index))
    requests = []
    for ordinal, source in enumerate(sft_rows):
        index = _source_index(source)
        donor = index if index in eligible else donors.get(_reuse_key(source, goals[index]))
        status = DIRECT if index in eligible else REUSED if donor is not None else FALLBACK
        requests.append(PlanRequest(
            ordinal=ordinal, source=source, status=status,
            typed_metadata=None if donor is None else eligible[donor],
            typed_source_row_idx=donor, stability_condition=goals[index],
            original_unavailable_reason=unavailable.get(index),
        ))
    return requests


def collate_prediction_requests(requests, bundle, *, goal_to_id, device="cpu"):
    """Inference-only collator: no soft target IDs or contact-tree labels."""
    if not requests or any(request.typed_metadata is None for request in requests):
        raise ValueError("prediction requires actual original chemical metadata")
    lengths = [len(request.typed_metadata["species_ids"]) + 2 for request in requests]
    width, size = max(lengths), len(requests)
    pointer_width = max(len(request.source["plan_state"]["elements"]) for request in requests)
    batch = {
        "stability_goal_ids": torch.empty(size, dtype=torch.long),
        "proposal_state_ids": torch.zeros(size, width, dtype=torch.long),
        "previous_species_indices": torch.full((size, width), -1, dtype=torch.long),
        "previous_count_values": torch.zeros(size, width, dtype=torch.long),
        "previous_n_values": torch.zeros(size, width, dtype=torch.long),
        "ledger_features": torch.zeros(size, width, 6, dtype=torch.float32),
        "attention_mask": torch.zeros(size, width, dtype=torch.long),
        "soft_position_indices": torch.empty(size, dtype=torch.long),
        "pointer_atomic_numbers": torch.zeros(size, pointer_width, dtype=torch.long),
        "pointer_counts": torch.zeros(size, pointer_width, dtype=torch.long),
        "pointer_valid_mask": torch.zeros(size, pointer_width, dtype=torch.bool),
    }
    for row, (request, length) in enumerate(zip(requests, lengths)):
        typed = request.typed_metadata
        proposal = typed["proposal_target"]
        stratum = (int(proposal["family_id"]), int(proposal["N"]), int(proposal["arity"]))
        batch["stability_goal_ids"][row] = int(goal_to_id[typed["stability_condition"]])
        batch["proposal_state_ids"][row, 1:length] = int(bundle.stratum_to_index[stratum]) + 1
        batch["previous_n_values"][row, 1] = int(proposal["N"])
        batch["previous_species_indices"][row, 2:length] = torch.tensor(typed["species_ids"])
        batch["previous_count_values"][row, 2:length] = torch.tensor(typed["count_targets"])
        for position in range(1, length):
            batch["ledger_features"][row, position] = torch.tensor(normalized_ledger(typed["ledger_steps"][position]))
        batch["attention_mask"][row, :length] = 1
        batch["soft_position_indices"][row] = length - 1
        plan = request.source["plan_state"]
        count = len(plan["elements"])
        batch["pointer_atomic_numbers"][row, :count] = torch.tensor([SYMBOL_TO_Z[value] for value in plan["elements"]])
        batch["pointer_counts"][row, :count] = torch.tensor(plan["counts"])
        batch["pointer_valid_mask"][row, :count] = True
    return {key: value.to(device) for key, value in batch.items()}


def source_field_seed(seed: int, split: str, source_row_idx: int, field: str) -> int:
    if int(seed) < 0 or split not in ("train", "val") or int(source_row_idx) < 0 or field not in SOFT_FIELDS:
        raise ValueError("invalid independent soft-field sampling key")
    payload = ["periodic-v2-soft-draw-v1", int(seed), split, int(source_row_idx), field]
    return int(json_sha256(payload)[:16], 16) % (2**63 - 1)


def sample_soft_fields(base_logits, residual_logits, soft_vocabulary, *, seed, split, source_row_idx):
    """Production order: legal PoE -> temperature -> nucleus -> CPU draw.

    Independent per-source/field generators remove dependence on sharding,
    batch order and the presence of other source records. Model floating-point
    kernels are a separate source of possible numerical variation.
    """
    selected, audits = {}, {}
    for field in SOFT_FIELDS:
        values = list(soft_vocabulary[field])
        base = torch.as_tensor(base_logits[field]).detach().float().cpu()
        residual = torch.as_tensor(residual_logits[field]).detach().float().cpu()
        if base.shape != (len(values),) or residual.shape != base.shape:
            raise ValueError("frozen soft logits and vocabulary differ")
        legal = torch.tensor([str(value) != "<UNKNOWN>" for value in values], dtype=torch.bool)
        fused = unit_weight_poe_log_probs(base, residual, legal)
        probabilities = torch.softmax(fused.clone() / SAMPLING["temperature"], dim=-1)
        sorted_probabilities, sorted_indices = torch.sort(probabilities, descending=True)
        cumulative = torch.cumsum(sorted_probabilities, dim=-1)
        remove = cumulative - sorted_probabilities > SAMPLING["top_p"]
        sorted_probabilities[remove] = 0.
        probabilities.zero_().scatter_(0, sorted_indices, sorted_probabilities)
        probabilities /= probabilities.sum().clamp_min(1e-12)
        draw_seed = source_field_seed(seed, split, source_row_idx, field)
        generator = torch.Generator(device="cpu").manual_seed(draw_seed)
        choice = int(torch.multinomial(probabilities, 1, generator=generator).item())
        map_id = int(fused.argmax().item())
        if not bool(legal[choice]) or not bool(torch.isfinite(probabilities).all()):
            raise RuntimeError("sampled soft value is outside finite frozen legal support")
        selected[field] = choice
        audits[field] = {
            "seed": draw_seed, "selected_id": choice, "selected_value": str(values[choice]),
            "legacy_map_id": map_id, "legacy_map_value": str(values[map_id]),
            "differs_from_legacy_map": choice != map_id,
            "selected_sampling_probability": float(probabilities[choice]),
            "selected_legal_poe_probability": float(fused[choice].exp()),
            "legal_support_size": int(legal.sum()),
        }
    return selected, audits


def finish_source(request: PlanRequest, prediction: Mapping[str, Any] | None = None):
    """Replace conditioning only; exact clean answer bytes remain unchanged."""
    source = request.source
    output = deepcopy(dict(source))
    plan = deepcopy(source["plan_state"])
    if request.status == FALLBACK:
        if prediction is not None:
            raise ValueError("teacher-soft fallback cannot pretend to have a prediction")
        indices = list(range(len(plan["elements"])))
        program_source = "canonical_teacher_soft_fallback"
        soft_source = FALLBACK
        soft_audit, soft_ids = None, None
    else:
        if prediction is None:
            raise ValueError("a predicted source is missing its frozen-model output")
        if (prediction["source_row_idx"] != request.source_row_idx
                or prediction["source_split"] != request.source_split):
            raise ValueError("frozen prediction belongs to a different source")
        for field in SOFT_FIELDS:
            plan[field] = str(prediction["soft_values"][field])
        indices = [int(value) for value in prediction["species_program_indices"]]
        program_source = ("frozen_llama_pointer_sampled_soft" if request.status == DIRECT
                          else "frozen_llama_pointer_sampled_soft_reused_typed_metadata")
        soft_source = "frozen_C3FD_Llama_legal_PoE_sampled_given_original_chemical_metadata"
        soft_audit, soft_ids = prediction["soft_sampling_audit"], prediction["soft_ids"]
    if sorted(indices) != list(range(len(plan["elements"]))):
        raise ValueError("frozen species program is not an exact Plan-element permutation")
    if exact_composition(plan) != exact_composition(source["plan_state"]):
        raise RuntimeError("frozen preprocessing changed the source composition")
    order = [plan["elements"][index] for index in indices]
    program_from_element_order(plan, order, order_source=program_source)
    before = answer_sha256(source["answer"])
    evidence = {
        "schema": SCHEMA, "status": request.status,
        "typed_metadata_source_split": request.source_split if request.typed_metadata is not None else None,
        "typed_metadata_source_row_idx": request.typed_source_row_idx,
        "typed_input_sha256": None if request.typed_metadata is None else json_sha256(request.typed_metadata),
        "original_metadata_unavailable_reason": request.original_unavailable_reason,
        "stability_condition": request.stability_condition,
        "goal_was_defaulted": False,
        "reuse_condition": None if request.status != REUSED else
            "same_split_exact_composition_anion_framework_and_explicit_stability_condition",
        "soft_ids": soft_ids, "soft_sampling_audit": soft_audit,
        "legacy_train_soft_selection": "legal_PoE_MAP",
        "current_soft_selection": "legal_PoE_temperature_nucleus_sample" if prediction is not None else FALLBACK,
        "clean_answer_sha256_before": before,
        "clean_answer_sha256_after": answer_sha256(output["answer"]),
        "clean_answer_unchanged": output["answer"] == source["answer"],
        "teacher_structure_in_planner_forward": False,
        "teacher_soft_ids_in_pointer_forward": False,
        "chemical_transcript_resampled": False,
        # Post-prediction diagnostics only. They never enter either model's input.
        "soft_matches_original_annotation": {
            field: plan[field] == source["plan_state"][field] for field in SOFT_FIELDS
        },
        "program_matches_original_order": order == source["species_program"],
    }
    output.update(
        plan_state=plan, prompt=build_native_body_prompt(plan).rstrip() + "\n",
        species_program=order, species_program_indices=indices,
        species_program_source=program_source, soft_plan_source=soft_source,
        pointer_semantics_available=prediction is not None,
        condition_prediction=evidence, composition_resampled=False,
        outcomes_read=False, planner_training=False,
    )
    if evidence["clean_answer_sha256_after"] != before:
        raise RuntimeError("frozen preprocessing altered the original clean answer")
    return output


def summarize_outputs(rows):
    statuses = Counter(row["condition_prediction"]["status"] for row in rows)
    reasons = Counter(row["condition_prediction"]["original_metadata_unavailable_reason"]
                      for row in rows if row["condition_prediction"]["original_metadata_unavailable_reason"])
    predicted = [row for row in rows if row["condition_prediction"]["status"] != FALLBACK]
    return {
        "source_rows": len(rows), "prediction_statuses": dict(statuses),
        "predicted_direct_rows": statuses[DIRECT], "predicted_with_reused_metadata_rows": statuses[REUSED],
        "teacher_soft_fallback_rows": statuses[FALLBACK],
        "teacher_soft_fallback_fraction": statuses[FALLBACK] / len(rows) if rows else None,
        "original_metadata_unavailable_reasons": dict(reasons),
        "stability_conditions": dict(Counter(str(row["condition_prediction"]["stability_condition"]) for row in rows)),
        "sampled_soft_differs_from_legacy_map_rows": sum(any(
            item["differs_from_legacy_map"] for item in row["condition_prediction"]["soft_sampling_audit"].values()
        ) for row in predicted),
        "sampled_soft_differs_from_legacy_map_by_field": {
            field: sum(row["condition_prediction"]["soft_sampling_audit"][field]["differs_from_legacy_map"] for row in predicted)
            for field in SOFT_FIELDS
        },
        "unique_exact_compositions": len({exact_composition(row["plan_state"]) for row in rows}),
        "all_clean_answers_unchanged": all(row["condition_prediction"]["clean_answer_unchanged"] for row in rows),
        "all_sources_predicted": statuses[FALLBACK] == 0,
        "predicted_soft_matches_original_annotation_rows_by_field": {
            field: sum(row["condition_prediction"]["soft_matches_original_annotation"][field]
                       for row in predicted) for field in SOFT_FIELDS
        },
        "predicted_program_matches_original_order_rows": sum(
            row["condition_prediction"]["program_matches_original_order"] for row in predicted),
        "source_annotation_agreement_denominator": len(predicted),
    }
