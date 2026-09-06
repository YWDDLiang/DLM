"""Pure canonical exact CE for registered periodic v2 state distributions.

Prefix states fit actual legal actions at T=.7. Dense states fit canonical typed
actions at the same temperature with inverse inclusion weights. The caller's
task/branch/position sampling already carries its mixture weights; this module
never adds them a second time. Ineligible and padding states retain denominator.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch

from crystal_dlm.periodic_base_objective import numeric_family_axis
from crystal_dlm.periodic_base_training_data import _mask_id
from crystal_dlm.periodic_geometry_objective import build_geometry_token_support
from crystal_dlm.programmed_path_runtime import process_scalar_path_logits
from crystal_dlm.r5_dynamic_length import exact_dynamic_schema_constraints


V2_POLICY_TEMPERATURE = .7
V2_OBJECTIVE_PROTOCOL = {
    "schema": "periodic_v2_exact_conditional_field_risk_v1",
    "temperature": V2_POLICY_TEMPERATURE,
    "ordinal_smoothing_weight": 0.,
    "alias": "raw_dtype_logaddexp_before_physical_class_temperature",
    "prefix": "actual_legal_exact_CE_with_explicit_old_and_cumulative_prefix_admission",
    "dense": "canonical_typed_exact_CE_with_independent_mask_inclusion_correction",
    "objects": {"lattice": .5, "coord": .5},
    "state_denominator": "original_batch_size_including_zero_weight_padding_and_ineligible_states",
    "extra_legal_auxiliary": False,
}


def fold_typed_logits(
    logits: torch.Tensor, ids: torch.Tensor, values: torch.Tensor, family: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Restrict an axis and fold 100 into 000 before temperature, in raw dtype."""
    vector = logits[..., ids]
    if family != "coord":
        return vector, ids
    canonical = torch.nonzero(values == 0, as_tuple=False).flatten()
    alias = torch.nonzero(values == 1, as_tuple=False).flatten()
    if canonical.numel() != 1 or alias.numel() != 1:
        raise ValueError("v2 requires the native coordinate 000/100 alias pair")
    canonical_index, alias_index = int(canonical[0]), int(alias[0])
    merged = torch.logaddexp(vector[..., canonical_index], vector[..., alias_index])
    vector = vector.clone()
    vector[..., canonical_index] = merged
    keep = torch.arange(ids.numel(), device=ids.device) != alias_index
    return vector[..., keep], ids[keep]


def exact_ce_on_support(vector: torch.Tensor, ids: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Exact canonical-target CE, normalized only on this vector's support."""
    matches = ids[None] == targets[:, None]
    if not bool(matches.any(-1).all()):
        raise ValueError("target is outside its canonical declared support")
    target_indices = matches.long().argmax(-1)
    logp = torch.log_softmax(vector.float() / V2_POLICY_TEMPERATURE, -1)
    return -logp.gather(-1, target_indices[:, None]).squeeze(-1)


def dense_field_coefficients(positions: list[int], num_atoms: int, probability: float) -> list[float]:
    """Independent-mask coefficients for equal lattice/coordinate object risk."""
    if not 0 < float(probability) <= 1 or not 1 <= int(num_atoms) <= 20:
        raise ValueError("dense field risk requires N in 1..20 and mask probability in (0,1]")
    coefficients = []
    for position in positions:
        _, family, _, _ = numeric_family_axis(int(position))
        size = 3 * int(num_atoms) if family == "coord" else 6
        coefficients.append(.5 / (size * float(probability)))
    return coefficients


class PeriodicV2Objective:
    """Return (state batch mean, additive numeric metrics, conflict records).

    Metrics ending in *_sum/*_states/*_tokens are additive across microbatches
    and ranks. loss_sum/state_count reconstructs the batch mean. Per-axis
    requested_tokens includes ineligible targets; exact_ce_sum/eligible_tokens
    gives the conditional CE without disguising support loss as improvement.
    """
    def __init__(self, tokenizer: Any, constraints: dict, device):
        if not constraints.get("canonicalize_periodic_alias"):
            raise ValueError("v2 requires the registered canonical alias policy")
        self.tokenizer, self.constraints, self.device = tokenizer, constraints, torch.device(device)
        self.mask_id = _mask_id(tokenizer, tokenizer.get_vocab())
        self.allowed_cache: dict[tuple[int, int], torch.Tensor] = {}
        self.tables = {
            (family, axis): {
                "ids": torch.tensor(table["ids"], dtype=torch.long, device=self.device),
                "values": torch.tensor(table["values"], dtype=torch.float32, device=self.device),
            }
            for family, axes in build_geometry_token_support(tokenizer).items()
            for axis, table in axes.items()
        }

    def _legal_vector(self, logits, batch, row, position):
        count = int(batch["examples"][row]["num_atoms"])
        key = (count, logits.shape[-1])
        if key not in self.allowed_cache:
            allowed = torch.zeros(7 + 4 * count, logits.shape[-1], dtype=torch.bool, device=self.device)
            for body_position, ids in enumerate(exact_dynamic_schema_constraints(self.tokenizer, count)):
                allowed[body_position, ids] = True
            self.allowed_cache[key] = allowed
        prompt = int(batch["geometry_context"].prompt_lengths[row])
        length = prompt + 7 + 4 * count
        return process_scalar_path_logits(
            logits[row:row + 1, :length], batch["input_ids"][row:row + 1, :length],
            prompt_length=prompt, gen_length=7 + 4 * count, allowed=self.allowed_cache[key],
            constraints=self.constraints, position=position, mask_id=self.mask_id,
        )

    @staticmethod
    def _conflict(example, reasons):
        return {
            "source_row_idx": int(example["source_row_idx"]),
            "source_split": example.get("source_split"), "epoch": int(example["epoch"]),
            "view": int(example["view"]), "branch": example["branch_name"],
            "position": int(example["position"]), "reasons": reasons,
            "first_illegal_teacher_position": example.get("first_illegal_teacher_position"),
            "first_illegal_teacher_reason": example.get("first_illegal_teacher_reason"),
        }

    def __call__(self, logits, batch):
        if logits.shape[0] != len(batch["examples"]):
            raise ValueError("v2 logits and example rows differ")
        metrics = defaultdict(float)
        conflicts, state_losses = [], []
        metrics["state_count"] = len(batch["examples"])
        for row, example in enumerate(batch["examples"]):
            zero = logits[row, 0, 0].float() * 0.
            branch = example["branch"]
            name = example["branch_name"]
            if branch not in ("prefix", "dense"):
                raise ValueError("v2 objective only supports declared prefix/dense branches")
            positions, targets = example["positions"], example["targets"]
            if len(positions) != len(targets):
                raise ValueError("numerical targets and positions differ")
            padding = bool(example["is_padding"])
            weight = float(example["sample_weight"])
            if padding:
                if weight != 0.:
                    raise ValueError("padding state must have zero weight")
                metrics["padding_states"] += 1
                state_losses.append(zero)
                continue
            metrics["real_states"] += 1
            metrics["source_weight_sum"] += float(example["source_sample_weight"])
            metrics[f"{name}_states"] += 1
            metrics[f"{name}_source_weight_sum"] += float(example["source_sample_weight"])
            metrics[f"{name}_requested_tokens"] += len(positions)
            metrics["requested_numerical_tokens"] += len(positions)
            metrics["empty_states"] += int(not positions)
            metrics["noise_metadata_unknown_states"] += int(example["noise_metadata_dropped"])
            corruption = example.get("corruption_info", {})
            if int(example["view"]) == 1:
                metrics["repair_old_states"] += 1
                metrics["repair_old_admitted_states"] += int(bool(example["old_state_admitted"]))
                metrics["corruption_proposals"] += int(corruption.get("attempt_count", 0))
                metrics["corruption_rejections"] += int(corruption.get("rejected_attempts", 0))
                metrics["corruption_fallback_states"] += int(corruption.get("fallback_to_clean", False))
                metrics["corruption_quantized_unchanged_states"] += int(corruption.get("quantized_unchanged", False))
            for position in positions:
                _, family, axis, _ = numeric_family_axis(position)
                metrics[f"{name}_{family}_{axis}_requested_tokens"] += 1

            def record_axis(family, axis, ces, coefficients):
                values = ces.detach().float().cpu().tolist()
                prefix = f"{name}_{family}_{axis}"
                metrics[f"{prefix}_eligible_tokens"] += len(values)
                metrics[f"{prefix}_exact_ce_sum"] += sum(values)
                metrics[f"{prefix}_risk_sum"] += weight * sum(v * c for v, c in zip(values, coefficients, strict=True))
                metrics[f"{name}_eligible_tokens"] += len(values)
                metrics["eligible_numerical_tokens"] += len(values)

            field_loss = zero
            if branch == "prefix":
                if len(positions) != 1:
                    raise ValueError("a prefix state supervises exactly one sampled scalar")
                position, target = int(positions[0]), int(targets[0])
                reasons = []
                if example["old_admission_required"] and not example["old_state_admitted"]:
                    reasons.append("repair_old_outside_complete_support")
                    metrics["prefix_old_admission_conflicts"] += 1
                if not example["prefix_predecessors_reachable"]:
                    reasons.append("teacher_predecessor_unreachable")
                    metrics["prefix_predecessor_conflicts"] += 1
                if not example["teacher_target_legal"]:
                    reasons.append("teacher_target_outside_actual_support")
                    metrics["prefix_target_conflicts"] += 1
                eligible = not reasons and bool(example["prefix_reachable"])
                if bool(example["legal_eligible"]) != eligible:
                    raise ValueError("cached prefix admission differs from its explicit factors")
                if eligible:
                    vector, bad = self._legal_vector(logits, batch, row, position)
                    if bool(torch.isnan(vector).any()) or bool(torch.isposinf(vector).any()):
                        raise FloatingPointError("nonfinite v2 actual-policy logits")
                    if bad or not bool(vector[target] > torch.finfo(vector.dtype).min):
                        eligible = False
                        reasons.append("actual_state_support_disagrees_with_teacher_admission")
                    else:
                        support = vector > torch.finfo(vector.dtype).min
                        ids = torch.nonzero(support, as_tuple=False).flatten()
                        ces = exact_ce_on_support(vector[ids][None], ids, torch.tensor([target], device=logits.device))
                        field_loss = ces[0]
                        _, family, axis, _ = numeric_family_axis(position)
                        record_axis(family, axis, ces, [1.])
                metrics["prefix_states"] += 1
                metrics["prefix_eligible_states"] += int(eligible)
                metrics[f"{name}_eligible_states"] += int(eligible)
                if not eligible:
                    metrics["prefix_ineligible_states"] += 1
                    conflicts.append(self._conflict(example, reasons))
            else:
                probability = float(example["mask_probability"])
                coefficients = dense_field_coefficients(positions, int(example["num_atoms"]), probability)
                inclusion = example.get("inclusion_probabilities", [probability] * len(positions))
                if len(inclusion) != len(positions) or any(float(value) != probability for value in inclusion):
                    raise ValueError("v2 dense masks must use the declared independent inclusion probability")
                prompt = int(batch["geometry_context"].prompt_lengths[row])
                by_axis = defaultdict(list)
                for position, target, coefficient in zip(positions, targets, coefficients, strict=True):
                    _, family, axis, _ = numeric_family_axis(position)
                    by_axis[(family, axis)].append((position, target, coefficient))
                for (family, axis), items in by_axis.items():
                    table = self.tables[(family, axis)]
                    absolute = torch.tensor([prompt + p for p, _, _ in items], device=logits.device)
                    vector, ids = fold_typed_logits(logits[row, absolute], table["ids"], table["values"], family)
                    target_ids = torch.tensor([target for _, target, _ in items], device=logits.device)
                    ces = exact_ce_on_support(vector, ids, target_ids)
                    axis_coefficients = [coefficient for _, _, coefficient in items]
                    field_loss = field_loss + (ces * ces.new_tensor(axis_coefficients)).sum()
                    record_axis(family, axis, ces, axis_coefficients)
                metrics["dense_states"] += 1
                metrics[f"{name}_mask_probability_sum"] += probability
            weighted = weight * field_loss
            state_losses.append(weighted)
            value = float(weighted.detach())
            metrics["loss_sum"] += value
            metrics[f"{branch}_loss_sum"] += value
            metrics[f"{name}_loss_sum"] += value
        return torch.stack(state_losses).mean(), dict(metrics), conflicts


__all__ = [
    "V2_POLICY_TEMPERATURE", "V2_OBJECTIVE_PROTOCOL", "fold_typed_logits",
    "exact_ce_on_support", "dense_field_coefficients", "PeriodicV2Objective",
]
