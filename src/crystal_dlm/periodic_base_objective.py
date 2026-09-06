"""MP20-only numerical denoising risk and explicitly gated legal-policy CE."""
from __future__ import annotations

from collections import defaultdict
import torch

from crystal_dlm.periodic_geometry_objective import build_geometry_token_support
from crystal_dlm.terminal_repair_objective import RepairObjective


def numeric_family_axis(position):
    if 1 <= position <= 3:
        return 0, "length", "ABC"[position - 1], .1
    if 4 <= position <= 6:
        return 1, "angle", "ABG"[position - 4], 1.
    if position >= 8 and (position - 8) % 4 < 3:
        return 2, "coord", "XYZ"[(position - 8) % 4], .01
    raise ValueError("only native numerical positions are supervised")


class PeriodicBaseObjective:
    def __init__(self, tokenizer, constraints, device, *, legal_weight=.25, smooth_weight=.1):
        self.device, self.legal_weight, self.smooth_weight = device, legal_weight, smooth_weight
        self.legal = RepairObjective(tokenizer, constraints, device, temperature=.7)
        self.tables = {}
        for family, axes in build_geometry_token_support(tokenizer).items():
            for axis, table in axes.items():
                self.tables[(family, axis)] = {
                    "ids": torch.tensor(table["ids"], dtype=torch.long, device=device),
                    "values": torch.tensor(table["values"], dtype=torch.float32, device=device),
                }

    def _numerical_loss(self, vector, ids, values, targets, family, step,
                        *, sigma_bins, temperature):
        """Alias folding occurs in the original model-logits dtype."""
        matches = ids[None] == targets[:, None]
        if not bool(matches.any(-1).all()):
            raise ValueError("source target is outside its declared typed vocabulary")
        logp = torch.log_softmax(vector.float() / temperature, -1)
        target_values = (matches.float() * values[None]).sum(-1)
        exact = -(logp * matches).sum(-1)
        distances = (values[None] - target_values[:, None]).abs()
        if family == "coord":
            distances = distances.remainder(1)
            distances = torch.minimum(distances, 1 - distances)
        q = torch.softmax(-.5 * (distances / (step * sigma_bins)).square(), -1)
        smooth = -(q * logp).sum(-1)
        return (1 - self.smooth_weight) * exact + self.smooth_weight * smooth, exact

    def __call__(self, logits, batch, *, sigma_bins=.5):
        source_losses, metrics = [], defaultdict(float)
        conflicts = []
        for row, example in enumerate(batch["examples"]):
            zero = logits[row, 0, 0].float() * 0.
            by_axis, family_losses, family_exacts = defaultdict(list), defaultdict(list), defaultdict(list)
            prompt = int(batch["geometry_context"].prompt_lengths[row])
            for position, target in zip(example["positions"], example["targets"], strict=True):
                family_index, family, axis, step = numeric_family_axis(position)
                by_axis[(family_index, family, axis, step)].append((position, target))
            for (family_index, family, axis, step), items in by_axis.items():
                table = self.tables[(family, axis)]
                ids, values = table["ids"], table["values"]
                positions = torch.tensor([prompt + position for position, _ in items], device=logits.device)
                targets = torch.tensor([target for _, target in items], device=logits.device)
                vector = logits[row, positions][:, ids]
                if family == "coord":
                    canonical = torch.nonzero(values == 0, as_tuple=False).flatten()
                    alias = torch.nonzero(values == 1, as_tuple=False).flatten()
                    if canonical.numel() != 1 or alias.numel() != 1:
                        raise ValueError("the retained periodic alias pair changed")
                    canonical_index, alias_index = int(canonical[0]), int(alias[0])
                    merged = torch.logaddexp(vector[:, canonical_index], vector[:, alias_index])
                    vector = vector.scatter(1, canonical.expand(vector.shape[0], 1), merged[:, None])
                    keep = torch.arange(ids.numel(), device=ids.device) != alias_index
                    ids, values, vector = ids[keep], values[keep], vector[:, keep]
                losses, exacts = self._numerical_loss(
                    vector, ids, values, targets, family, step, sigma_bins=sigma_bins, temperature=1.,
                )
                family_losses[family_index].append(losses)
                family_exacts[family_index].append(exacts)
            means = [torch.cat(family_losses[index]).mean() if family_losses[index] else zero
                     for index in range(3)]
            exact_means = [torch.cat(family_exacts[index]).mean() if family_exacts[index] else zero
                           for index in range(3)]
            # Dense masks use the fixed three-family denominator; an empty
            # family contributes zero. Scalar repair has already sampled its
            # family uniformly, so its one CE is the Monte Carlo estimator.
            schema = sum(means) / 3 if example["view"] == 0 else sum(means)
            schema_exact = sum(exact_means) / 3 if example["view"] == 0 else sum(exact_means)
            legal_loss, eligible = zero, False
            probe = example.get("legal_probe_position")
            if probe is not None and not example["is_padding"]:
                target = int(example["legal_probe_target"])
                modified = list(batch["examples"])
                modified[row] = dict(example, position=int(probe), target_token=target)
                try:
                    vector = self.legal.processed_vector(logits, dict(batch, examples=modified), row)
                    if vector[target] <= torch.finfo(vector.dtype).min:
                        raise ValueError("target_outside_actual_support")
                    family_index, family, axis, step = numeric_family_axis(probe)
                    table = self.tables[(family, axis)]
                    ids, values = table["ids"], table["values"]
                    support = vector[ids] > torch.finfo(vector.dtype).min
                    ids, values = ids[support], values[support]
                    legal_losses, _ = self._numerical_loss(
                        vector[ids][None], ids, values, torch.tensor([target], device=logits.device),
                        family, step, sigma_bins=sigma_bins, temperature=.7,
                    )
                    legal_loss, eligible = legal_losses[0], True
                except ValueError as error:
                    conflicts.append({"source_row_idx": example["source_row_idx"], "view": example["view"],
                                      "epoch": example["epoch"], "position": probe, "reason": str(error)})
            weight = float(example["sample_weight"])
            source_losses.append(weight * (schema + self.legal_weight * legal_loss))
            metrics["schema_loss_sum"] += float(schema.detach()) * weight
            metrics["schema_exact_ce_sum"] += float(schema_exact.detach()) * weight
            metrics["legal_loss_sum"] += float(legal_loss.detach()) * weight
            metrics["legal_eligible_states"] += int(eligible)
            metrics["legal_probed_states"] += int(probe is not None and not example["is_padding"])
            metrics["supervised_numerical_tokens"] += len(example["positions"]) if not example["is_padding"] else 0
            metrics["empty_states"] += int(example["empty_supervision"] and not example["is_padding"])
        return torch.stack(source_losses).mean(), dict(metrics), conflicts
