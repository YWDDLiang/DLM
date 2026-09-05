"""Typed numerical CE on actual legal support, plus actual-policy KL."""
from __future__ import annotations

import torch
from crystal_dlm.periodic_geometry_objective import build_geometry_token_support
from crystal_dlm.programmed_path_runtime import process_scalar_path_logits
from crystal_dlm.r5_dynamic_length import exact_dynamic_schema_constraints


class RepairObjective:
    def __init__(self, tokenizer, constraints, device, temperature=.7):
        self.tokenizer, self.constraints = tokenizer, constraints
        self.device, self.temperature = device, temperature
        self.supports = build_geometry_token_support(tokenizer)
        self.allowed_cache = {}

    def processed_vector(self, logits, batch, row):
        example = batch["examples"][row]
        count = example["num_atoms"]
        if count not in self.allowed_cache:
            support = exact_dynamic_schema_constraints(self.tokenizer, count)
            allowed = torch.zeros(7 + 4 * count, logits.shape[-1], dtype=torch.bool, device=self.device)
            for position, values in enumerate(support):
                allowed[position, values] = True
            self.allowed_cache[count] = allowed
        prompt = int(batch["geometry_context"].prompt_lengths[row])
        length = prompt + 7 + 4 * count
        from crystal_dlm.fixed_slot import MASK_TOKEN_ID
        vector, bad = process_scalar_path_logits(
            logits[row:row + 1, :length], batch["input_ids"][row:row + 1, :length],
            prompt_length=prompt, gen_length=7 + 4 * count,
            allowed=self.allowed_cache[count], constraints=self.constraints,
            position=example["position"], mask_id=MASK_TOKEN_ID,
        )
        if bad:
            raise ValueError("training state has no actual legal deployment support")
        return vector

    def terminal_ce(self, logits, batch, *, sigma_bins=.5, smooth_weight=.1):
        losses, exacts, smooths = [], [], []
        for row, example in enumerate(batch["examples"]):
            vector = self.processed_vector(logits, batch, row)
            target = int(example["target_token"])
            position = int(example["position"])
            family, axis, step = (("length", "ABC"[position - 1], .1) if position <= 3
                                 else ("angle", "ABG"[position - 4], 1.) if position <= 6
                                 else ("coord", "XYZ"[(position - 8) % 4], .01))
            table = self.supports[family][axis]
            ids = torch.tensor(table["ids"], dtype=torch.long, device=vector.device)
            values = torch.tensor(table["values"], dtype=torch.float32, device=vector.device)
            legal = vector[ids] > torch.finfo(vector.dtype).min
            ids, values = ids[legal], values[legal]
            matches = ids == target
            if not bool(matches.any()):
                raise ValueError("terminal target is unreachable in its teacher-forced legal state")
            logp = torch.log_softmax(vector[ids].float() / self.temperature, -1)
            exact = -logp[matches][0]
            distance = (values - values[matches][0]).abs()
            if family == "coord":
                distance = distance.remainder(1)
                distance = torch.minimum(distance, 1 - distance)
            q = torch.softmax(-.5 * (distance / (step * sigma_bins)).square(), -1)
            smooth = -(q * logp).sum()
            losses.append(((1 - smooth_weight) * exact + smooth_weight * smooth)
                          * float(example["sample_weight"]))
            exacts.append(exact.detach())
            smooths.append(smooth.detach())
        return torch.stack(losses).mean(), {
            "exact_ce": float(torch.stack(exacts).mean()),
            "numeric_ce": float(torch.stack(smooths).mean()),
        }

    def policy_kl(self, student_logits, reference_logits, batch):
        values = []
        for row, example in enumerate(batch["examples"]):
            student = self.processed_vector(student_logits, batch, row)
            reference = self.processed_vector(reference_logits, batch, row).detach()
            support = reference > torch.finfo(reference.dtype).min
            if not torch.equal(support, student > torch.finfo(student.dtype).min):
                raise ValueError("student/reference legal supports differ at the same actual state")
            log_reference = torch.log_softmax(reference[support].float() / self.temperature, -1)
            log_student = torch.log_softmax(student[support].float() / self.temperature, -1)
            kl = (log_reference.exp() * (log_reference - log_student)).sum()
            values.append(kl * float(example.get("sample_weight", 1.)))
        return torch.stack(values).mean()
