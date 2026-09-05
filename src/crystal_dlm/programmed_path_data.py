"""Strict condition compilation and occurrence accounting for native paths."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

from crystal_dlm.dynamic_crystal import parse_dynamic_answer
from crystal_dlm.spad_program import program_from_element_order


def training_candidates_per_condition(collection_round: int) -> int:
    """2026-09-06 data-budget amendment: K4 first round, K8 single refresh."""
    if collection_round not in (0, 1):
        raise ValueError("only the initial collection and one refresh are registered")
    return 4 if collection_round == 0 else 8


def path_seed(seed: int, group_id: str, collection_round: int, occurrence: int) -> int:
    payload = json.dumps([seed, str(group_id), collection_round, occurrence]).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def compile_condition(record, tokenizer, *, mask_id: int, purpose: str = "train"):
    if purpose == "train" and record.get("source_split") != "train":
        raise ValueError("path teachers accept only explicitly recorded train conditions")
    if not record.get("group_id") and record.get("group_id") != 0:
        raise ValueError("every condition requires a stable group_id")
    program = program_from_element_order(
        record["plan_state"], record["species_program"],
        order_source=record["species_program_source"],
    )
    vocabulary = tokenizer.get_vocab()
    initial = [int(mask_id)] * (7 + 4 * program.num_atoms)
    initial[0] = int(vocabulary[f"<N_{program.num_atoms:03d}>"])
    # Slots come from the canonical compiler, not the Plan's display order.
    for entry in program.entries:
        for slot in entry.slot_indices:
            initial[7 + 4 * slot] = int(vocabulary[f"<E_{entry.symbol}>"])
    prompt = record["prompt"].rstrip() + "\n"
    prefix = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    if not prefix:
        raise ValueError("empty native prompt")
    return {"record": record, "program": program, "initial_body": initial,
            "prompt": prompt, "prompt_token_ids": prefix}


def validate_completed_body(body: str, condition) -> dict:
    parsed = parse_dynamic_answer(body, strict=True)
    program = condition["program"]
    expected = [None] * program.num_atoms
    for entry in program.entries:
        for slot in entry.slot_indices:
            expected[slot] = entry.symbol
    if list(parsed["species"]) != expected:
        raise ValueError("completed body differs from the canonical fixed composition")
    if len(parsed["tokens"]) != 7 + 4 * program.num_atoms:
        raise ValueError("completed body changed exact native length")
    return parsed


def trace_terminal_body(trace):
    """Replay deterministic transitions too, including a final rollback."""
    body = list(trace["initial_body"])
    old = None
    for event in trace["events"]:
        if event["op"] == "begin":
            if old is not None:
                raise ValueError("nested trace transaction")
            old = body.copy()
            for pos in event["positions"]:
                body[pos] = int(trace["mask_id"])
        elif event["op"] == "draw":
            if old is None:
                raise ValueError("draw outside trace transaction")
            body[event["position"]] = int(event["token"])
        elif event["op"] == "rollback":
            if old is None:
                raise ValueError("rollback without transaction")
            for pos in event["positions"]:
                body[pos] = old[pos]
        elif event["op"] == "end":
            old = None
        elif event["op"] != "no_support":
            raise ValueError("unknown trace event")
    if old is not None:
        raise ValueError("unfinished trace transaction")
    return body


def trace_summary(trace):
    body = list(trace["initial_body"])
    snapshot, phase, positions = None, "", []
    changes = Counter()
    transactions = Counter()
    for event in trace["events"]:
        if event["op"] == "begin":
            snapshot, phase, positions = body.copy(), event["phase"], event["positions"]
            transactions[phase] += 1
            for pos in positions:
                body[pos] = trace["mask_id"]
        elif event["op"] == "draw":
            body[event["position"]] = event["token"]
        elif event["op"] == "rollback":
            for pos in event["positions"]:
                body[pos] = snapshot[pos]
        elif event["op"] == "end":
            changes[phase] += sum(body[pos] != snapshot[pos] for pos in positions)
            snapshot = None
    return {
        "sampled_decisions": sum(e["op"] == "draw" for e in trace["events"]),
        "decisions_by_phase": dict(Counter(e["phase"] for e in trace["events"] if e["op"] == "draw")),
        "rollback_reasons": dict(Counter(e["reason"] for e in trace["events"] if e["op"] == "rollback")),
        "cooperative_accepted": any(e["op"] == "begin" and e["phase"] == "cooperative" for e in trace["events"])
            and not any(e["op"] == "rollback" and e["reason"].startswith("cooperative_") for e in trace["events"]),
        "transactions_by_phase": dict(transactions), "committed_changed_scalars_by_phase": dict(changes),
    }


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_path_model(model_path, checkpoint_path, device, *, trainable=False):
    """Load both retained LoRA/tables and the mandatory trained state input."""
    from crystal_dlm.periodic_state_conditioning import PeriodicStateConfig
    from crystal_dlm.state_conditioned_model import StateConditionedDLM, set_state_lora_trainable
    root = Path(checkpoint_path)
    for marker in (root.parent.parent / "TRAIN_FINAL.json", root.parent / "PREFLIGHT.json"):
        if marker.is_file() and json.loads(marker.read_text()).get("eligible_policy") is False:
            raise ValueError(f"engineering checkpoint is not an eligible collection policy: {root}")
    config = PeriodicStateConfig(**json.loads((root / "periodic_state_config.json").read_text()))
    if not (root / "periodic_state.pt").is_file():
        raise FileNotFoundError("checkpoint is missing the trained periodic state conditioner")
    from scripts.sample_llada_dynamic_crystals import load_model_and_tokenizer
    base, tokenizer = load_model_and_tokenizer(str(model_path), str(root), device)
    model = StateConditionedDLM(base, tokenizer, config).to(device)
    model.load_state_conditioner(root)
    if trainable:
        set_state_lora_trainable(model)
        model.train()
    else:
        model.requires_grad_(False)
        model.eval()
    return model, tokenizer
