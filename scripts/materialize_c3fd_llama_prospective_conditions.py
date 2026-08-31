#!/usr/bin/env python3
"""Replay frozen C3FD traces through both checkpoints and build F/M conditions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crystal_dlm.c3fd_rich_expander import (  # noqa: E402
    CHECKPOINT_ORDER,
    C3FD_SOFT_PREFIX_FEATURE_VERSION,
    pack_soft_prefix_features,
)
from crystal_dlm.ccfd import FormulaToken  # noqa: E402
from crystal_dlm.ccfd_v2 import CCFDv2State, SetAtomCount  # noqa: E402
FIELDS = ("lattice_system", "spacegroup_bucket", "volume_per_atom_bin")
SCHEMA = "c3fd_llama_prospective_conditions_v1"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    count = 0
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")
            count += 1
    return count


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def state_from_record(
    record: Mapping[str, Any], vocabulary: Mapping[str, Any]
) -> tuple[dict[str, Any], list[CCFDv2State], list[int], list[int]]:
    proposal = record.get("target_proposal") or {}
    target_n = int(proposal["N"])
    target_arity = int(proposal["arity"])
    family = str(proposal["family"])
    node_to_id = {
        (int(row["atomic_number"]), int(row["oxidation_state"])): int(row["id"])
        for row in vocabulary["species"]
    }
    state = CCFDv2State.start().apply(SetAtomCount(target_n))
    history = [state]
    species_ids: list[int] = []
    counts: list[int] = []
    for action in record.get("semantic_trace") or ():
        if action.get("action") != "species":
            continue
        key = (int(action["atomic_number"]), int(action["oxidation_state"]))
        if key not in node_to_id:
            raise ValueError(f"sampled C3FD node is outside vocabulary: {key}")
        count = int(action["count"])
        state = state.apply(FormulaToken(key[0], key[1], count), max_species=7)
        history.append(state)
        species_ids.append(node_to_id[key])
        counts.append(count)
    if len(species_ids) != target_arity or not state.eos_legal:
        raise ValueError("sampled C3FD semantic trace is incomplete")
    family_values = [
        str(value)
        for value in vocabulary["soft_vocabulary"]["anion_framework"]
    ]
    if family not in family_values:
        raise ValueError("sampled C3FD family is outside vocabulary")
    semantic = {
        "certificate_class": str(
            (record.get("certificate") or {}).get("certificate_class") or ""
        ),
        "composition_supervision": True,
        "proposal_supervision": True,
        "proposal_targets": {
            "N": target_n,
            "arity": target_arity,
            "family": family_values.index(family),
        },
        "species_labels": species_ids,
        "count_targets": counts,
        "ledger_steps": [
            {
                "remaining_atoms": int(item.remaining_atoms or 0),
                "net_charge": int(item.net_charge),
                "remaining_species": target_arity - len(item.tokens),
                "branch": "unset" if item.branch is None else str(item.branch),
            }
            for item in history
        ],
    }
    return semantic, history, species_ids, counts


def semantic_tensors(
    *,
    target_n: int,
    target_arity: int,
    history: Sequence[CCFDv2State],
    species_ids: Sequence[int],
    counts: Sequence[int],
):
    import torch

    target_position = len(species_ids) + 1
    width = target_position + 1
    previous_species = torch.full((1, width), -1, dtype=torch.long)
    previous_count = torch.zeros((1, width), dtype=torch.long)
    previous_n = torch.zeros((1, width), dtype=torch.long)
    ledger = torch.zeros((1, width, 6), dtype=torch.float32)
    previous_n[0, 1] = int(target_n)
    for index, (species_id, count) in enumerate(
        zip(species_ids, counts), start=2
    ):
        previous_species[0, index] = int(species_id)
        previous_count[0, index] = int(count)
    for index, state in enumerate(history, start=1):
        branch = {
            None: (1.0, 0.0, 0.0),
            "ionic": (0.0, 1.0, 0.0),
            "alloy": (0.0, 0.0, 1.0),
        }[state.branch]
        ledger[0, index] = torch.tensor(
            [
                float(state.remaining_atoms or 0) / 20.0,
                float(state.net_charge) / 160.0,
                float(target_arity - len(state.tokens)) / 7.0,
                *branch,
            ]
        )
    return previous_species, previous_count, previous_n, ledger, target_position


def load_checkpoint(path: Path, vocabulary: Mapping[str, Any]):
    import torch
    from crystal_dlm.c3fd_planner_model import C3FDPlannerConfig, C3FDPlannerModel

    payload = torch.load(path, map_location="cpu")
    config = C3FDPlannerConfig(**payload["config"])
    physics = torch.tensor(
        vocabulary["physics"]["matrix"], dtype=torch.float32
    )
    model = C3FDPlannerModel(config, physics_features=physics)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    context = torch.as_tensor(payload["context"], dtype=torch.float32)
    return model, context


def checkpoint_predictions(
    model,
    context,
    tensors,
    vocabulary: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    import torch
    from crystal_dlm.semantic_composition_head import SemanticHeadFlags

    previous_species, previous_count, previous_n, ledger, target_position = tensors
    with torch.inference_mode():
        output = model(
            context,
            previous_species_indices=previous_species,
            previous_count_values=previous_count,
            previous_n_values=previous_n,
            ledger_features=ledger,
            flags=SemanticHeadFlags(use_physics=True),
        )
    result: dict[str, dict[str, Any]] = {}
    for field in FIELDS:
        values = [
            str(value) for value in vocabulary["soft_vocabulary"][field]
        ]
        logits = output.rich_logits[field][0, target_position].float().clone()
        for index, value in enumerate(values):
            if value == "<UNKNOWN>":
                logits[index] = float("-inf")
        probabilities = torch.softmax(logits, dim=-1)
        index = int(torch.argmax(probabilities).item())
        result[field] = {
            "prediction": values[index],
            "confidence": float(probabilities[index].item()),
        }
    return result


def materialize(
    *,
    source_rows: Sequence[Mapping[str, Any]],
    ledger_rows: Sequence[Mapping[str, Any]],
    vocabulary: Mapping[str, Any],
    models: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_by_idx = {int(row["sample_idx"]): row for row in source_rows}
    f_rows: list[dict[str, Any]] = []
    m_rows: list[dict[str, Any]] = []
    for target_idx, ledger in enumerate(ledger_rows):
        source_idx = int(ledger["source_sample_idx"])
        record = source_by_idx[source_idx]
        plan = record.get("plan_state")
        if not isinstance(plan, Mapping):
            raise ValueError("selected C3FD source lacks plan_state")
        semantic, history, species_ids, counts = state_from_record(
            record, vocabulary
        )
        proposal = semantic["proposal_targets"]
        tensors = semantic_tensors(
            target_n=int(proposal["N"]),
            target_arity=int(proposal["arity"]),
            history=history,
            species_ids=species_ids,
            counts=counts,
        )
        by_checkpoint = {
            name: checkpoint_predictions(model, context, tensors, vocabulary)
            for name, (model, context) in models.items()
        }
        predicted = {"predictions_by_checkpoint": by_checkpoint}
        minimal = {
            "N": int(plan["N"]),
            "elements": [str(value) for value in plan["elements"]],
            "counts": [int(value) for value in plan["counts"]],
        }
        common = {
            "schema": SCHEMA,
            "sample_idx": target_idx,
            "source_sample_idx": source_idx,
            "expander_plan_state": minimal,
            "outcomes_read": False,
            "exact_composition_identity": str(ledger["exact_composition_identity"]),
            "reduced_composition_identity": str(ledger["reduced_composition_identity"]),
            "chemsys": str(ledger["chemsys"]),
        }
        f_rows.append({**common, "route": "F"})
        m_rows.append(
            {
                **common,
                "route": "M",
                "soft_prefix_feature_version": C3FD_SOFT_PREFIX_FEATURE_VERSION,
                "soft_prefix_features": pack_soft_prefix_features(
                    semantic, vocabulary, predicted
                ),
                "C3FD_predictions_by_checkpoint": by_checkpoint,
            }
        )
    return f_rows, m_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-raw", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint", action="append", required=True, help="name=path"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    vocabulary = json.loads(
        (args.data_dir / "vocabulary.json").read_text(encoding="utf-8")
    )
    models = {}
    checkpoint_inputs = {}
    for item in args.checkpoint:
        name, separator, raw_path = item.partition("=")
        if not separator or name in models:
            raise ValueError("--checkpoint must be unique name=path")
        path = Path(raw_path).resolve()
        models[name] = load_checkpoint(path, vocabulary)
        checkpoint_inputs[name] = {
            "path": str(path),
            "sha256": sha256_file(path),
        }
    if tuple(models) != CHECKPOINT_ORDER:
        raise ValueError(f"checkpoint order must be {CHECKPOINT_ORDER}")
    source_rows = read_jsonl(args.source_raw)
    ledger_rows = read_jsonl(args.ledger)
    f_rows, m_rows = materialize(
        source_rows=source_rows,
        ledger_rows=ledger_rows,
        vocabulary=vocabulary,
        models=models,
    )
    args.output_dir.mkdir(parents=True)
    write_jsonl(args.output_dir / "F.jsonl", f_rows)
    write_jsonl(args.output_dir / "M.jsonl", m_rows)
    manifest = {
        "schema": SCHEMA,
        "selected": len(f_rows),
        "routes": ["F", "M"],
        "checkpoint_order": list(models),
        "checkpoints": checkpoint_inputs,
        "source_raw_sha256": sha256_file(args.source_raw),
        "ledger_sha256": sha256_file(args.ledger),
        "outcomes_read": False,
        "output_sha256": {
            "F.jsonl": sha256_file(args.output_dir / "F.jsonl"),
            "M.jsonl": sha256_file(args.output_dir / "M.jsonl"),
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "_SUCCESS").touch()
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
