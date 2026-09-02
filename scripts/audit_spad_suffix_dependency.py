#!/usr/bin/env python3
"""Prove that a real retained DLM uses visible future sites for anchor logits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch

from crystal_dlm.fixed_slot import MASK_TOKEN_ID  # noqa: E402
from crystal_dlm.spad_program import (  # noqa: E402
    coordinate_positions,
    program_from_element_order,
)
from scripts.sample_llada_dynamic_crystals import load_model_and_tokenizer  # noqa: E402


def rows_by_index(path: Path) -> dict[int, dict[str, Any]]:
    return {
        int(row["source_row_idx"]): row
        for row in (
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
        )
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--pointer-data", type=Path, required=True)
    parser.add_argument("--teacher-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError("suffix dependency audit requires its allocated GPU")

    pointer = rows_by_index(args.pointer_data)
    teacher = rows_by_index(args.teacher_data)
    selected = None
    for source_idx in sorted(set(pointer) & set(teacher)):
        plan = teacher[source_idx]["plan_state"]
        if int(plan["N"]) < 4 or len(plan["elements"]) < 2:
            continue
        program = program_from_element_order(
            plan,
            pointer[source_idx]["contact_tree_order_symbols"],
            order_source="mp20_contact_teacher_mechanism_audit",
        )
        candidates = [slot for slot in program.anchor_slots if slot < int(plan["N"]) - 1]
        if candidates:
            selected = (source_idx, program, candidates[0])
            break
    if selected is None:
        raise RuntimeError("no teacher row supports an earlier-anchor suffix audit")
    source_idx, program, anchor_slot = selected
    row = teacher[source_idx]
    device = torch.device("cuda")
    model, tokenizer = load_model_and_tokenizer(
        args.model_path, args.checkpoint_path, device
    )
    prompt_text = str(row["prompt"]).rstrip() + "\n"
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt")[
        "input_ids"
    ].to(device)
    answer_ids = tokenizer(
        str(row["answer"]), add_special_tokens=False, return_tensors="pt"
    )["input_ids"].to(device)
    expected = 7 + 4 * int(row["plan_state"]["N"])
    if answer_ids.shape[1] != expected:
        raise RuntimeError("teacher answer no longer tokenizes to exact 7+4N")
    base = torch.cat((prompt_ids, answer_ids), dim=1)
    anchor_positions = coordinate_positions(anchor_slot)
    absolute_anchor = [prompt_ids.shape[1] + value for value in anchor_positions]
    base[0, absolute_anchor] = int(MASK_TOKEN_ID)

    suffix_slot = int(row["plan_state"]["N"]) - 1
    suffix_z = prompt_ids.shape[1] + coordinate_positions(suffix_slot)[-1]
    old_suffix_token = int(base[0, suffix_z].item())
    old_suffix_text = str(tokenizer.convert_ids_to_tokens(old_suffix_token))
    if not old_suffix_text.startswith("<Z_"):
        raise RuntimeError("selected future token is not a Z coordinate")
    old_bin = int(old_suffix_text[3:6]) % 100
    new_bin = (old_bin + 1) % 100
    new_suffix_token = int(tokenizer.get_vocab()[f"<Z_{new_bin:03d}>"])
    changed = base.clone()
    changed[0, suffix_z] = new_suffix_token
    attention = torch.ones_like(base)
    with torch.inference_mode():
        base_logits = model(base, attention_mask=attention).logits
        changed_logits = model(changed, attention_mask=attention).logits

    per_axis = {}
    for absolute, axis in zip(absolute_anchor, ("X", "Y", "Z"), strict=True):
        ids = torch.tensor(
            [tokenizer.get_vocab()[f"<{axis}_{value:03d}>"] for value in range(101)],
            dtype=torch.long,
            device=device,
        )
        left = base_logits[0, absolute, ids].float()
        right = changed_logits[0, absolute, ids].float()
        p = torch.softmax(left, dim=0)
        q = torch.softmax(right, dim=0)
        midpoint = (p + q) / 2.0
        js = 0.5 * (
            torch.sum(p * (torch.log(p + 1e-12) - torch.log(midpoint + 1e-12)))
            + torch.sum(q * (torch.log(q + 1e-12) - torch.log(midpoint + 1e-12)))
        )
        per_axis[axis] = {
            "max_abs_logit_change": float((left - right).abs().max().item()),
            "jensen_shannon": float(js.item()),
        }
    report = {
        "schema": "spad_real_suffix_dependency_v1",
        "source_row_idx": int(source_idx),
        "num_atoms": int(row["plan_state"]["N"]),
        "species_program": list(program.element_order),
        "anchor_slot": int(anchor_slot),
        "future_slot": int(suffix_slot),
        "future_z_token_before": old_suffix_text,
        "future_z_token_after": f"<Z_{new_bin:03d}>",
        "per_axis": per_axis,
        "any_context_effect": any(
            value["max_abs_logit_change"] > 0.0 for value in per_axis.values()
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not report["any_context_effect"]:
        raise RuntimeError("real DLM anchor logits were insensitive to future context")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
