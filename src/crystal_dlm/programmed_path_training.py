"""Occurrence-level terminal supervision and unbiased within-path subsampling."""
from __future__ import annotations

from collections import defaultdict
import math

import numpy as np
import torch

from crystal_dlm.fixed_slot import MASK_TOKEN_ID
from crystal_dlm.programmed_path_data import path_seed, trace_terminal_body
from crystal_dlm.programmed_path_runtime import process_scalar_path_logits, replay_scalar_states
from crystal_dlm.r5_dynamic_length import exact_dynamic_schema_constraints


def join_terminal_labels(paths, labels, *, expected_conditions, candidates=4):
    by_id = {}
    for label in labels:
        key = label["trajectory_id"]
        if key in by_id:
            raise ValueError("duplicate terminal label occurrence")
        by_id[key] = label
    if len(paths) != expected_conditions * candidates:
        raise ValueError("incomplete requested path pool")
    if len({p["trajectory_id"] for p in paths}) != len(paths):
        raise ValueError("duplicate generated path occurrence")
    if set(by_id) != {p["trajectory_id"] for p in paths}:
        raise ValueError("unknown/missing label rows must be explicitly accounted for")
    groups = {}
    references = {(p["checkpoint"], p["collection_round"]) for p in paths}
    if len(references) != 1:
        raise ValueError("one empirical teacher must have one collection policy and round")
    for path in paths:
        label = by_id[path["trajectory_id"]]
        if path.get("source_split") != "train" or label.get("source_split") != "train":
            raise ValueError("terminal teachers cannot use heldout outcomes")
        for key in ("group_id", "source_row_idx"):
            if str(path[key]) != str(label[key]):
                raise ValueError("label-to-condition provenance mismatch")
        if label.get("verified") is True and not path["success"]:
            raise ValueError("generation failure cannot have a verified teacher")
        if trace_terminal_body(path["trace"]) != path["final_body_token_ids"]:
            raise ValueError("terminal path does not match its attempted trace")
        group_id = str(path["group_id"])
        group = groups.setdefault(group_id, {"group_id": group_id, "candidates": []})
        group["candidates"].append({**label, "candidate_index": path["candidate_index"]})
    if len(groups) != expected_conditions:
        raise ValueError("condition coverage changed")
    for group in groups.values():
        if sorted(c["candidate_index"] for c in group["candidates"]) != list(range(candidates)):
            raise ValueError("a condition is missing a requested candidate occurrence")
        group["candidates"].sort(key=lambda c: c["candidate_index"])
    return list(groups.values())


def sample_path_decisions(path, *, seed, pass_index, budget=6):
    """Fixed-size stratified sample with m_h/T_h inclusion probabilities.

    Every nonempty phase gets positive allocation. Short phases are exhausted;
    unused slots move deterministically to phases with remaining decisions. If
    T<budget all decisions are returned once with pi=1 (no false multiplicity).
    """
    if budget < 3:
        raise ValueError("budget must cover all three possible phases")
    states = list(replay_scalar_states(path["trace"]))
    strata = defaultdict(list)
    for state in states:
        if state["phase"] not in ("construct", "cooperative", "closure"):
            raise ValueError("unknown deployed phase")
        strata[state["phase"]].append(state)
    phases = [p for p in ("construct", "cooperative", "closure") if strata[p]]
    allocation = {p: 0 for p in phases}
    for _ in range(min(budget, len(states))):
        available = [p for p in phases if allocation[p] < len(strata[p])]
        chosen = min(available, key=lambda p: (allocation[p], phases.index(p)))
        allocation[chosen] += 1
    rng = np.random.default_rng(path_seed(seed, path["trajectory_id"], pass_index, 0))
    selected = []
    for phase in phases:
        count = allocation[phase]
        pi = count / len(strata[phase])
        indices = rng.choice(len(strata[phase]), count, replace=False)
        for index in indices:
            selected.append(dict(strata[phase][int(index)], inclusion_probability=pi))
    return selected


def training_decision_budget(collection_round, positive_paths):
    """Reuse the registered 98,304 refresh-state budget when labels are sparse.

    Depend only on usable path count, never energies or evaluation outcomes.
    More within-path observations reduce HT sampling noise; they are not new
    trajectories. Round0 remains unchanged and the single refresh has two passes.
    """
    if collection_round not in (0, 1) or positive_paths < 1:
        raise ValueError("a registered collection round and positive path count are required")
    return 6 if collection_round == 0 else min(24, max(6, 98304 // (2 * positive_paths)))


def sampled_training_examples(paths, teacher, *, seed, pass_index, decision_budget=6):
    weights = {c["trajectory_id"]: c["weight"] for g in teacher["groups"] for c in g["candidates"]}
    examples = []
    for path in paths:
        weight = float(weights[path["trajectory_id"]])
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("invalid fixed teacher weight")
        if weight == 0:
            continue
        states = sample_path_decisions(path, seed=seed, pass_index=pass_index, budget=decision_budget)
        if not states:
            raise ValueError("positive teacher weight requires actual path decisions")
        for state in states:
            examples.append({**state, "weight": weight, "trajectory_id": path["trajectory_id"],
                             "group_id": path["group_id"], "num_atoms": path["num_atoms"],
                             "prompt": path["prompt"], "prompt_token_ids": path["prompt_token_ids"],
                             "plan_state": path["plan_state"], "species_program": path["species_program"],
                             "species_program_source": path["species_program_source"]})
            examples[-1]["sampling_batch_size"] = int(path.get("sampling_batch_size", 4))
    return examples


class PathLogProbability:
    """Score mixed-length rows through exactly the sampler's support transform."""
    def __init__(self, tokenizer, constraints):
        self.tokenizer, self.constraints, self.allowed = tokenizer, constraints, {}

    def __call__(self, raw, batch):
        values = []
        for row, example in enumerate(batch["examples"]):
            count = int(example["num_atoms"])
            body_length = 7 + 4 * count
            prompt = int(batch["geometry_context"].prompt_lengths[row])
            width = prompt + body_length
            key = (count, raw.device)
            if key not in self.allowed:
                mask = torch.zeros(body_length, raw.shape[-1], device=raw.device, dtype=torch.bool)
                for pos, ids in enumerate(exact_dynamic_schema_constraints(self.tokenizer, count)):
                    mask[pos, ids] = True
                self.allowed[key] = mask
            vector, bad = process_scalar_path_logits(
                raw[row:row+1, :width], batch["input_ids"][row:row+1, :width],
                prompt_length=prompt, gen_length=body_length, allowed=self.allowed[key],
                constraints=self.constraints, position=example["position"], mask_id=MASK_TOKEN_ID,
            )
            target = int(example["target_token"])
            if bad or vector[target] <= torch.finfo(vector.dtype).min:
                raise ValueError("recorded teacher decision is outside deployed support")
            temperature = float(example["temperature"])
            if not temperature > 0:
                raise ValueError("path fitting requires stochastic deployed probabilities")
            values.append(torch.log_softmax(vector.float() / temperature, dim=-1)[target])
        return torch.stack(values)


def minibatch_path_loss(log_probs, examples, *, dataset_size, validated_groups):
    """Uniform scalar minibatches estimate the CONDITION mean, not scalar mean.

    D = number of selected positive-weight states in this pass. Multiplying the
    minibatch mean by D/C gives E[-sum_cj w_cj sum_t log p_t / C] after HT. There
    is no extra division by the number of decisions in a trajectory.
    """
    from crystal_dlm.basin_path_objective import weighted_sampled_path_nll
    return weighted_sampled_path_nll(
        log_probs, [e["weight"] for e in examples],
        [e["inclusion_probability"] for e in examples],
        len(examples) * validated_groups / dataset_size,
    )


def shape_matched_batches(examples, *, global_batch, seed, pass_index):
    """One exact pass, grouping by unpadded sequence width for deployed numerics.

    Each real state occurs once. Short buckets receive zero-mass copies only;
    the loss uses this padded population size, preserving the condition mean.
    Batch order is shuffled without consulting energy, weights or outcomes.
    """
    rng = np.random.default_rng(np.random.SeedSequence([seed, pass_index, 71]))
    buckets = defaultdict(list)
    for example in examples:
        width = len(example["prompt_token_ids"]) + len(example["input_body"])
        buckets[width].append(example)
    batches = []
    for width in sorted(buckets):
        rows = buckets[width]
        order = rng.permutation(len(rows))
        shuffled = [rows[int(i)] for i in order]
        missing = (-len(shuffled)) % global_batch
        shuffled += [dict(shuffled[0], weight=0., inclusion_probability=1.) for _ in range(missing)]
        batches.extend(shuffled[i:i + global_batch] for i in range(0, len(shuffled), global_batch))
    order = rng.permutation(len(batches))
    return [batches[int(i)] for i in order]
