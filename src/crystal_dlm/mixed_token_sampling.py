"""Construction-only deployment of a completed mixed policy's token module.

The original integer sampler, supports, alias transform and scalar RNG remain
unchanged. This module supplies endpoint bookkeeping and teacher-forced replay.
"""
from __future__ import annotations

from collections import Counter
from contextlib import nullcontext
from copy import deepcopy
import math
from pathlib import Path

import torch

from crystal_dlm.mixed_geometry_sampling import METHOD as TRAINING_METHOD, validate_final_policy
from crystal_dlm.programmed_path_data import path_seed, trace_summary, trace_terminal_body, validate_completed_body
from crystal_dlm.programmed_path_runtime import ProgrammedPathTrace, complete_geometry_supported, replay_scalar_states
from crystal_dlm.spad_generation import _transaction_candidate_tokens
from crystal_dlm.spad_program import spad_predictor_position_groups
from scripts.sample_state_programmed_paths import make_sampler


METHOD = "mixed_token_construction"
DEPLOYMENT = "token_construction_only"
BASE_SEED = 20260905
TEMPERATURE = .7
LAYOUT_WORLD_SIZE = 2
BATCH_CAP = 4


def load_final_mixed_token_policy(model_path, checkpoint_path, device, *, torch_dtype=None):
    """Validate the mixed final endpoint before loading and extracting its T module."""
    policy = validate_final_policy(checkpoint_path)
    from crystal_dlm.mixed_geometry_model import load_mixed_geometry_model
    mixed, tokenizer = load_mixed_geometry_model(model_path, checkpoint_path, device,
                                                trainable=False, torch_dtype=torch_dtype)
    token_model = mixed.token_model.requires_grad_(False).eval()
    if not hasattr(token_model, "raw_initialization"):
        raise ValueError("mixed token deployment lost the V2 construction context contract")
    return token_model, tokenizer, policy


def inference_context(device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if torch.device(device).type == "cuda" else nullcontext()


def validate_protocol(*, seed, collection_round, batch_cap, layout_world_size, temperature):
    if (seed != BASE_SEED or collection_round != 0 or batch_cap != BATCH_CAP
            or layout_world_size != LAYOUT_WORLD_SIZE or temperature != TEMPERATURE):
        raise ValueError("mixed T construction uses the fixed seed/round/cap/layout/temperature protocol")


class ForwardCounter:
    """Count attempted model calls, including a call that raises, without changing it."""
    def __init__(self, model):
        self.model = model
        self.calls = 0

    def __enter__(self):
        self.handle = self.model.register_forward_pre_hook(self._count)
        return self

    def _count(self, _module, _args):
        self.calls += 1

    def __exit__(self, *_error):
        self.handle.remove()


def _canvas(model, batch, device, max_length):
    compiled = [item[2] for item in batch]
    if not compiled or len(batch) > BATCH_CAP or any(candidate != 0 for _, candidate, _ in batch):
        raise ValueError("one fixed occurrence and at most four members are required")
    shapes = {(c["program"].num_atoms, len(c["prompt_token_ids"])) for c in compiled}
    if len(shapes) != 1 or model.training or not hasattr(model, "raw_initialization"):
        raise ValueError("T construction requires equal N/prompt lengths and an eval V2 token module")
    from crystal_dlm.fixed_slot import MASK_TOKEN_ID
    if model.mask_id != MASK_TOKEN_ID:
        raise ValueError("token model and original sampler MASK ABI differ")
    x = torch.tensor([c["prompt_token_ids"]+c["initial_body"] for c in compiled], dtype=torch.long, device=device)
    if x.shape[1] > max_length:
        raise ValueError("frozen construction canvas exceeds maximum length")
    return compiled, x


def _abort_traces(sampler, compiled, error):
    """Close an explicitly aborted construction ledger, without inventing a draw."""
    traces = []
    for row, c in enumerate(compiled):
        trace = (deepcopy(sampler.traces[row].to_dict()) if row < len(sampler.traces)
                 else ProgrammedPathTrace(c["initial_body"], sampler.mask_id, sampler.temperature,
                                          list(c["program"].element_order),
                                          construct_active_scope="masked_numeric").to_dict())
        if trace["success"]:
            trace.update(success=False, failure=f"construction_runtime_error: {type(error).__name__}: {error}")
        opened = sum(event["op"] == "begin" for event in trace["events"])-sum(event["op"] == "end" for event in trace["events"])
        if opened == 1:
            trace["events"].append({"op": "end", "phase": "construct", "aborted": True})
        elif opened != 0:
            raise RuntimeError("unrecoverable construction transaction ledger") from error
        traces.append(trace)
    return traces


@torch.no_grad()
def sample_token_batch(model, tokenizer, constraints, batch, *, device, checkpoint_path,
                       condition_start=0, seed=BASE_SEED, collection_round=0,
                       layout_world_size=LAYOUT_WORLD_SIZE, temperature=TEMPERATURE, max_length=382):
    validate_protocol(seed=seed, collection_round=collection_round, batch_cap=BATCH_CAP,
                      layout_world_size=layout_world_size, temperature=temperature)
    compiled, x = _canvas(model, batch, device, max_length)
    seeds = [path_seed(seed, c["record"]["group_id"], collection_round, 0) for c in compiled]
    sampler = make_sampler(model, tokenizer, constraints, compiled, seeds, temperature)
    runtime_error = None
    with ForwardCounter(model) as count, inference_context(device):
        try:
            result, traces = sampler.run(x, torch.ones_like(x), construct=True,
                                         cooperative=False, closure=False, full_cell_repair=False)
        except Exception as error:
            runtime_error = error
            traces = _abort_traces(sampler, compiled, error)
            result = None
    members = [int(item[0]) for item in batch]
    records, live_calls = [], []
    for row, (ordinal, candidate, c) in enumerate(batch):
        trace = traces[row]
        attempted = trace_terminal_body(trace)
        if result is not None and attempted != result[row, sampler.prompt_length:].tolist():
            raise RuntimeError("construction trace does not reconstruct its actual terminal state")
        attempted_body = tokenizer.decode(attempted, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        no_support = [e for e in trace["events"] if e["op"] == "no_support"]
        row_calls = min(count.calls, sum(e["op"] == "draw" for e in trace["events"])+1) if no_support else count.calls
        live_calls.append(row_calls)
        failure = trace["failure"]
        sampling_failure = None
        if not trace["success"]:
            scope = "logical_batch" if runtime_error is not None and not no_support else "row"
            sampling_failure = {"scope": scope, "stage": "model_or_runtime" if scope == "logical_batch" else "construct_support",
                                "message": failure, "model_nfe_at_failure": row_calls}
        record = {**c["record"], "prompt": c["prompt"], "prompt_token_ids": c["prompt_token_ids"],
                  "condition_ordinal": int(ordinal), "evaluation_ordinal": int(ordinal)-int(condition_start),
                  "candidate_index": candidate, "collection_round": int(collection_round),
                  "trajectory_id": f"{c['record']['group_id']}:{collection_round}:{candidate}",
                  "sampling_seed": int(seeds[row]), "sampling_batch_size": len(batch),
                  "sampling_layout_world_size": int(layout_world_size), "sampling_batch_ordinals": members,
                  "sampling_row_index": row, "num_atoms": c["program"].num_atoms,
                  "checkpoint": str(checkpoint_path), "method": METHOD, "training_method": TRAINING_METHOD,
                  "deployment_mode": "token",
                  "path_mode": "construction_only", "deployment": DEPLOYMENT, "endpoint": "native",
                  "native_source": "body", "output_representation": "discrete_token_construction",
                  "sampling_nfe": row_calls, "failed_row_padding_evaluations": count.calls-row_calls,
                  "expected_healthy_nfe": 6+3*c["program"].num_atoms,
                  "sampling_failure": sampling_failure, "success": bool(trace["success"]),
                  "runtime_success": bool(trace["success"]), "failure": failure,
                  "body": None, "final_body_token_ids": None, "structure": None,
                  "native_structure": None, "cif": None, "cif_path": None, "parseable": False,
                  "inference_mlip": False, "geometry_forward_calls": 0,
                  "trace": trace, "trace_summary": trace_summary(trace), "trace_scope": "construction_only_attempt"}
        if record["success"]:
            try:
                validate_completed_body(attempted_body, c)
                record.update(body=attempted_body, final_body_token_ids=attempted, parseable=True)
            except Exception as error:
                record.update(success=False, failure=f"{type(error).__name__}: {error}",
                              sampling_failure={"scope": "row", "stage": "native_parser",
                                                "message": str(error), "model_nfe_at_failure": row_calls})
        if not record["success"]:
            record["diagnostic_candidate"] = {"body": attempted_body, "final_body_token_ids": attempted}
        record["native_execution_success"] = record["success"]
        records.append(record)
    statistics = {"logical_batch_size": len(batch), "condition_ordinals": members,
                  "sample_indices": [int(c["record"]["sample_idx"]) for c in compiled],
                  "requested": len(batch), "successful": sum(r["success"] for r in records),
                  "model_forward_calls": count.calls, "model_row_evaluations": count.calls*len(batch),
                  "live_row_model_evaluations": sum(live_calls),
                  "failed_row_padding_evaluations": count.calls*len(batch)-sum(live_calls),
                  "geometry_forward_calls": 0,
                  "failures_by_stage": dict(Counter(r["sampling_failure"]["stage"] for r in records if r["sampling_failure"]))}
    return records, statistics


def validate_record_identity(record, source, ordinal, *, checkpoint_path, condition_start=0):
    """Check source keys and protocol independently of the generated body."""
    expected = {"sample_idx": source["sample_idx"], "group_id": source["group_id"],
                "plan_state": source["plan_state"], "species_program": source["species_program"],
                "species_program_source": source["species_program_source"],
                "prompt": source["prompt"].rstrip()+"\n", "condition_ordinal": ordinal,
                "evaluation_ordinal": ordinal-condition_start, "candidate_index": 0, "collection_round": 0,
                "sampling_seed": path_seed(BASE_SEED, source["group_id"], 0, 0),
                "sampling_layout_world_size": LAYOUT_WORLD_SIZE, "method": METHOD, "training_method": TRAINING_METHOD,
                "deployment": DEPLOYMENT, "deployment_mode": "token", "path_mode": "construction_only",
                "trajectory_id": f"{source['group_id']}:0:0", "source_split": "evaluation",
                "source_row_idx": int(source["sample_idx"]), "num_atoms": int(source["plan_state"]["N"])}
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError("T record changed frozen condition, occurrence or deployment identity")
    if Path(record["checkpoint"]).resolve() != Path(checkpoint_path).resolve():
        raise ValueError("T record belongs to another checkpoint")
    if not record["success"] and (record.get("parseable") is not False or any(record.get(field) is not None
            for field in ("body", "structure", "native_structure", "cif", "cif_path", "final_body_token_ids"))):
        raise ValueError("a failed T request retained an active geometry preview")


def _trace_actions(record, compiled, mask_id):
    trace = record["trace"]
    if (trace["initial_body"] != compiled["initial_body"] or trace["mask_id"] != mask_id
            or trace["temperature"] != TEMPERATURE or trace["element_order"] != list(compiled["program"].element_order)
            or trace.get("construct_active_scope") != "masked_numeric"):
        raise ValueError("recorded T trace changed its initial state or scalar policy")
    if any(e.get("phase") != "construct" or e["op"] not in {"begin", "end", "draw", "no_support"}
           for e in trace["events"]):
        raise ValueError("T replay accepts construction events only")
    if ((record.get("sampling_failure") or {}).get("scope") == "logical_batch"
            or any(e.get("aborted") for e in trace["events"])):
        raise ValueError("an aborted runtime batch cannot pass the replay acceptance")
    if (record.get("sampling_failure") or {}).get("stage") == "native_parser":
        raise ValueError("a native-parser failure cannot pass the replay acceptance")
    if (sum(e["op"] == "begin" for e in trace["events"]) != 1
            or sum(e["op"] == "end" for e in trace["events"]) != 1):
        raise ValueError("T trace requires exactly one completed construction transaction")
    schedule = [group[0] for group in spad_predictor_position_groups(compiled["program"])[1:]]
    actions = [e for e in trace["events"] if e["op"] in {"draw", "no_support"}]
    if ([e["position"] for e in actions] != schedule[:len(actions)] or len(actions) > len(schedule)
            or any(e["op"] == "no_support" for e in actions[:-1])
            or (trace["success"] and (len(actions) != len(schedule) or any(e["op"] != "draw" for e in actions)))):
        raise ValueError("T trace changed the fixed P scalar schedule")
    for step, event in enumerate(actions):
        if event["op"] == "draw" and (event["salt"] != 100_000_000+10_007*step
                                      or not math.isfinite(event["log_probability"])):
            raise ValueError("T trace changed a scalar salt or recorded a nonfinite likelihood")
    terminal = trace_terminal_body(trace)
    if record["success"] and terminal != record["final_body_token_ids"]:
        raise ValueError("T trace terminal tokens differ from the active endpoint")
    if not record["success"] and terminal != record["diagnostic_candidate"]["final_body_token_ids"]:
        raise ValueError("T trace terminal tokens differ from its failed-attempt diagnostic")
    states = {state["position"]: state for state in replay_scalar_states(trace)}
    return actions, states, terminal


@torch.no_grad()
def replay_token_batch(model, tokenizer, constraints, batch, records, *, device, checkpoint_path,
                       condition_start=0, tolerance=1e-6, max_length=382):
    """Replay recorded scalar states with original batch shape and no G calls.

    Targets are teacher-forced after checking likelihoods and the existing seeded
    scalar draw. Failed-support rows remain at their recorded state in the batch.
    """
    if not math.isfinite(tolerance) or tolerance < 0 or len(batch) != len(records):
        raise ValueError("invalid replay tolerance or logical batch coverage")
    compiled, x = _canvas(model, batch, device, max_length)
    members = [int(item[0]) for item in batch]
    actions, states, terminal = [], [], []
    for row, ((ordinal, _, c), record) in enumerate(zip(batch, records, strict=True)):
        validate_record_identity(record, c["record"], ordinal, checkpoint_path=checkpoint_path, condition_start=condition_start)
        if (record["prompt_token_ids"] != c["prompt_token_ids"] or record["sampling_batch_ordinals"] != members
                or record["sampling_row_index"] != row or record["sampling_batch_size"] != len(batch)):
            raise ValueError("replay changed the original tokenizer or logical batch packing")
        a, s, t = _trace_actions(record, c, model.mask_id)
        actions.append(a)
        states.append(s)
        terminal.append(t)
    seeds = [record["sampling_seed"] for record in records]
    sampler = make_sampler(model, tokenizer, constraints, compiled, seeds, TEMPERATURE)
    sampler._prepare(x)
    attention = torch.ones_like(x)
    checked = empty = 0
    maximum = 0.
    with ForwardCounter(model) as count, inference_context(device):
        for step in range(max(map(len, actions), default=0)):
            active = {row: row_actions[step]["position"] for row, row_actions in enumerate(actions) if step < len(row_actions)}
            transaction = {}
            for row, position in active.items():
                schedule = [g[0] for g in spad_predictor_position_groups(compiled[row]["program"])[1:]]
                transaction[row] = [p for p in schedule if int(x[row, sampler.prompt_length+p]) == model.mask_id]
                if actions[row][step]["op"] == "draw":
                    recorded = states[row][position]
                    if (x[row, sampler.prompt_length:].tolist() != recorded["input_body"]
                            or recorded["old_body"] != recorded["input_body"]
                            or sorted(transaction[row]) != sorted(recorded["transaction_positions"])):
                        raise ValueError("scalar replay input/context differs from its recorded state")
            logits, unavailable = sampler.processed_logits(x, x.clone(), active, transaction, attention, phase="construct")
            sampled = _transaction_candidate_tokens(logits,
                active_absolute_positions={row: sampler.prompt_length+pos for row, pos in active.items() if row not in unavailable},
                temperature=TEMPERATURE, remasking="low_confidence", sampling_seeds_by_batch=seeds,
                salt=100_000_000+10_007*step)
            for row, pos in active.items():
                event = actions[row][step]
                if event["op"] == "no_support":
                    if row not in unavailable:
                        raise RuntimeError("replay found support for a recorded empty-support decision")
                    empty += 1
                    continue
                if row in unavailable:
                    raise RuntimeError("recorded construction target has no replay support")
                token = event["token"]
                actual = float(torch.log_softmax(logits[row, sampler.prompt_length+pos].double()/TEMPERATURE, -1)[token])
                difference = abs(actual-event["log_probability"])
                if not math.isfinite(actual) or difference > tolerance:
                    raise RuntimeError(f"scalar logp replay differs at {records[row]['trajectory_id']}:{step}: {difference}")
                if int(sampled[row, sampler.prompt_length+pos]) != token:
                    raise RuntimeError("recorded scalar token differs from the original seeded Gumbel draw")
                maximum = max(maximum, difference)
                checked += 1
                x[row, sampler.prompt_length+pos] = token
    for row, record in enumerate(records):
        if x[row, sampler.prompt_length:].tolist() != terminal[row]:
            raise RuntimeError("scalar replay does not reconstruct the attempted endpoint")
        supported = complete_geometry_supported(x[row, sampler.prompt_length:], constraints)
        if supported != bool(record["trace"]["success"]):
            raise RuntimeError("scalar replay and recorded construction support status differ")
        if record["success"]:
            body = tokenizer.decode(terminal[row], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            if body != record["body"]:
                raise ValueError("saved T body differs from final tokenizer decoding")
            validate_completed_body(body, compiled[row])
    if checked+empty == 0:
        raise ValueError("replay exercised no model decisions")
    return {"requests": len(batch), "checked_draws": checked, "checked_no_support": empty,
            "maximum_logp_error": maximum, "seeded_token_matches": checked,
            "model_forward_calls": count.calls, "model_row_evaluations": count.calls*len(batch),
            "logical_batch_ordinals": members, "terminal_reconstruction_pass": True,
            "geometry_forward_calls": 0}
