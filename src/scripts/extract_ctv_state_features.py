#!/usr/bin/env python3
"""Extract frozen state/action features for CTV value-head training."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from crystal_dlm.ctv_features import (  # noqa: E402
    exact_prompt_length,
    geometry_token_family,
    selected_probability_error,
)
from crystal_dlm.ctv_rollout import _allowed_mask, _constrained_logits  # noqa: E402
from crystal_dlm.fixed_slot import MASK_TOKEN_ID  # noqa: E402
from crystal_dlm.r5_dynamic_length import (  # noqa: E402
    exact_body_token_count,
    exact_dynamic_schema_constraints,
)
from scripts.sample_llada_dynamic_crystals import (  # noqa: E402
    build_dynamic_lightweight_constraints,
    load_model_and_tokenizer,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_state_records(split: str, rollout_run: Path) -> list[dict[str, Any]]:
    states = read_jsonl(rollout_run / "branch/states.jsonl")
    branches = read_jsonl(rollout_run / "branch/branches.jsonl")
    atoms_by_state: dict[str, set[int]] = {}
    for branch in branches:
        atoms_by_state.setdefault(str(branch["state_id"]), set()).add(
            int(branch["num_atoms"])
        )
    if len(atoms_by_state) != len(states):
        raise ValueError(f"CTV {split} state/branch identities changed")
    records = []
    for state in states:
        state_id = str(state["state_id"])
        atom_values = atoms_by_state.get(state_id, set())
        if len(atom_values) != 1:
            raise ValueError(f"CTV {split} state has ambiguous atom count")
        records.append({"split": str(split), "num_atoms": atom_values.pop(), **state})
    records.sort(key=lambda row: (int(row["canary_plan_idx"]), float(row["milestone"])))
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--train-rollout-run", type=Path, required=True)
    parser.add_argument("--validation-rollout-run", type=Path, required=True)
    parser.add_argument("--dataset-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--projection-seed", type=int, default=73017)
    parser.add_argument("--probability-atol", type=float, default=1e-5)
    args = parser.parse_args()

    audit = json.loads(args.dataset_audit.read_text(encoding="utf-8"))
    if audit.get("dataset_authorized") is not True:
        raise ValueError("CTV Branch dataset audit is not authorized")
    if int(args.projection_dim) != 256 or int(args.projection_seed) != 73017:
        raise ValueError("CTV frozen projection contract changed")
    if not torch.cuda.is_available():
        raise RuntimeError("CTV state feature extraction requires one CUDA device")
    device = torch.device("cuda", 0)
    model, tokenizer = load_model_and_tokenizer(
        args.model_path, args.checkpoint_path, device
    )
    output_head = model.get_output_embeddings()
    captures: list[torch.Tensor] = []

    def capture_hidden(_module: Any, inputs: tuple[Any, ...]) -> None:
        if not inputs or not isinstance(inputs[0], torch.Tensor):
            raise RuntimeError("CTV output-head hook did not receive hidden states")
        captures.append(inputs[0].detach())

    hook = output_head.register_forward_pre_hook(capture_hidden)
    records = [
        *load_state_records("train", args.train_rollout_run),
        *load_state_records("validation", args.validation_rollout_run),
    ]
    if len(records) != 320:
        raise ValueError("CTV feature extraction requires exactly 320 states")

    projection: torch.Tensor | None = None
    hidden_dim: int | None = None
    projected_states: list[torch.Tensor] = []
    output_records: list[dict[str, Any]] = []
    legal_union: set[int] = set()
    maximum_probability_error = 0.0
    lightweight = build_dynamic_lightweight_constraints(
        tokenizer,
        duplicate_coordinate_mask=True,
        lattice_volume_mask=True,
        min_lattice_rad=1e-4,
    )
    try:
        with torch.no_grad():
            for record in records:
                tokens = [int(value) for value in record["state_token_ids"]]
                num_atoms = int(record["num_atoms"])
                gen_length = exact_body_token_count(num_atoms)
                prompt_length = exact_prompt_length(len(tokens), num_atoms)
                if gen_length != len(tokens) - prompt_length:
                    raise ValueError("CTV exact state length changed")
                position = int(record["intervention_position"])
                absolute_position = prompt_length + position
                x = torch.tensor([tokens], dtype=torch.long, device=device)
                attention = torch.ones_like(x)
                allowed = exact_dynamic_schema_constraints(tokenizer, num_atoms)
                allowed_mask = _allowed_mask(
                    model=model,
                    gen_length=gen_length,
                    allowed_token_ids_by_generation_pos=allowed,
                )
                captures.clear()
                logits = _constrained_logits(
                    model=model,
                    x=x,
                    attention_mask=attention,
                    prompt_index=x != int(MASK_TOKEN_ID),
                    prompt_length=prompt_length,
                    gen_length=gen_length,
                    mask_id=int(MASK_TOKEN_ID),
                    allowed_mask=allowed_mask,
                    lightweight_decoding_constraints=lightweight,
                )
                if len(captures) != 1:
                    raise RuntimeError("CTV output-head hook count changed")
                hidden = captures[0]
                if hidden.ndim != 3 or hidden.shape[:2] != x.shape:
                    raise RuntimeError("CTV captured hidden-state shape changed")
                vector = hidden[0, absolute_position].to(torch.float32)
                if hidden_dim is None:
                    hidden_dim = int(vector.numel())
                    generator = torch.Generator(device="cpu")
                    generator.manual_seed(int(args.projection_seed))
                    projection = torch.randn(
                        hidden_dim,
                        int(args.projection_dim),
                        generator=generator,
                        dtype=torch.float32,
                    ).div_(math.sqrt(float(args.projection_dim))).to(device)
                if int(vector.numel()) != hidden_dim or projection is None:
                    raise RuntimeError("CTV hidden dimension changed across states")
                projected_states.append((vector @ projection).cpu())

                position_logits = logits[0, absolute_position]
                minimum = torch.finfo(position_logits.dtype).min
                legal_mask = position_logits > (minimum / 2)
                legal_ids = torch.nonzero(legal_mask, as_tuple=False).reshape(-1)
                if legal_ids.numel() < 8:
                    raise ValueError("CTV reproduced state has fewer than eight legal tokens")
                legal_logits = position_logits.index_select(0, legal_ids).to(torch.float64)
                legal_probabilities = torch.softmax(legal_logits, dim=0)
                legal_id_list = [int(value) for value in legal_ids.cpu().tolist()]
                legal_probability_list = [
                    float(value) for value in legal_probabilities.cpu().tolist()
                ]
                probability_error = selected_probability_error(
                    selected_token_ids=record["action_token_ids"],
                    selected_probabilities=record["action_probabilities"],
                    legal_token_ids=legal_id_list,
                    legal_probabilities=legal_probability_list,
                )
                maximum_probability_error = max(
                    maximum_probability_error, probability_error
                )
                if probability_error > float(args.probability_atol):
                    raise ValueError(
                        f"CTV base probability reproduction changed by {probability_error}"
                    )
                legal_union.update(legal_id_list)
                output_records.append(
                    {
                        "split": str(record["split"]),
                        "state_id": str(record["state_id"]),
                        "composition_id": str(record["composition_id"]),
                        "sample_idx": int(record["sample_idx"]),
                        "plan_ordinal": int(record["canary_plan_idx"]),
                        "milestone": float(record["milestone"]),
                        "intervention_position": position,
                        "absolute_position": absolute_position,
                        "num_atoms": num_atoms,
                        "action_token_ids": [
                            int(value) for value in record["action_token_ids"]
                        ],
                        "action_probabilities": [
                            float(value) for value in record["action_probabilities"]
                        ],
                        "legal_token_ids": legal_id_list,
                        "legal_probabilities": legal_probability_list,
                    }
                )
    finally:
        hook.remove()

    if projection is None or hidden_dim is None:
        raise RuntimeError("CTV extracted no hidden states")
    token_ids = sorted(legal_union)
    token_strings = tokenizer.convert_ids_to_tokens(token_ids)
    token_families = [geometry_token_family(token) for token in token_strings]
    weight = output_head.weight.detach().to(device=device, dtype=torch.float32)
    token_tensor = torch.tensor(token_ids, dtype=torch.long, device=device)
    projected_actions = weight.index_select(0, token_tensor) @ projection

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    artifact_path = output / "CTV_STATE_ACTION_FEATURES.pt"
    torch.save(
        {
            "schema": "h1a2_ctv_state_action_features_v1",
            "projection_dim": int(args.projection_dim),
            "projection_seed": int(args.projection_seed),
            "hidden_dim": hidden_dim,
            "state_features": torch.stack(projected_states),
            "state_records": output_records,
            "action_token_ids": torch.tensor(token_ids, dtype=torch.long),
            "action_token_families": torch.tensor(token_families, dtype=torch.long),
            "action_features": projected_actions.cpu(),
        },
        artifact_path,
    )
    manifest = {
        "schema": "h1a2_ctv_state_action_features_manifest_v1",
        "states": len(output_records),
        "train_states": sum(row["split"] == "train" for row in output_records),
        "validation_states": sum(
            row["split"] == "validation" for row in output_records
        ),
        "hidden_dim": hidden_dim,
        "projection_dim": int(args.projection_dim),
        "projection_seed": int(args.projection_seed),
        "geometry_tokens": len(token_ids),
        "maximum_selected_probability_error": maximum_probability_error,
        "probability_atol": float(args.probability_atol),
        "generator_frozen": True,
        "output_head_hook": "forward_pre_hook_input",
        "artifact_sha256": sha256(artifact_path),
        "train_states_sha256": sha256(
            args.train_rollout_run / "branch/states.jsonl"
        ),
        "validation_states_sha256": sha256(
            args.validation_rollout_run / "branch/states.jsonl"
        ),
    }
    (output / "CTV_STATE_ACTION_FEATURES_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "_SUCCESS").touch()
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
