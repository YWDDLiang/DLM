#!/usr/bin/env python3
"""Train the full typed C³FD-v2 semantic Planner head."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch
from torch.utils.data import DataLoader, Dataset

from crystal_dlm.c3fd_planner_model import (  # noqa: E402
    C3FDPlannerConfig,
    C3FDPlannerModel,
)
from crystal_dlm.semantic_composition_head import SemanticHeadFlags  # noqa: E402


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


class C3FDDataset(Dataset):
    def __init__(self, path: Path) -> None:
        self.rows = [row for row in iter_jsonl(path) if row.get("composition_supervision") is True]
        if not self.rows:
            raise ValueError(f"no supervised C3FD rows in {path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


def collate(rows: list[dict[str, Any]], *, eos_species_id: int, soft_fields: tuple[str, ...]) -> dict[str, torch.Tensor]:
    lengths = [len(row["species_labels"]) + 2 for row in rows]
    width = max(lengths)
    batch = len(rows)
    previous_species = torch.full((batch, width), -1, dtype=torch.long)
    previous_count = torch.zeros((batch, width), dtype=torch.long)
    previous_n = torch.zeros((batch, width), dtype=torch.long)
    species_target = torch.full((batch, width), -100, dtype=torch.long)
    count_target = torch.full((batch, width), -100, dtype=torch.long)
    n_target = torch.empty((batch,), dtype=torch.long)
    rich_targets = {
        field: torch.full((batch, width), -100, dtype=torch.long)
        for field in soft_fields
    }
    for row_idx, (row, length) in enumerate(zip(rows, lengths)):
        n_value = int(row["N_target"])
        species = [int(value) for value in row["species_labels"]]
        counts = [int(value) for value in row["count_targets"]]
        if len(species) != len(counts) or not species:
            raise ValueError("invalid semantic training sequence")
        n_target[row_idx] = n_value
        previous_n[row_idx, 1] = n_value
        for action_idx, (species_id, count) in enumerate(zip(species, counts)):
            target_pos = action_idx + 1
            species_target[row_idx, target_pos] = species_id
            count_target[row_idx, target_pos] = count
            previous_pos = action_idx + 2
            if previous_pos < length:
                previous_species[row_idx, previous_pos] = species_id
                previous_count[row_idx, previous_pos] = count
        eos_pos = len(species) + 1
        species_target[row_idx, eos_pos] = int(eos_species_id)
        count_target[row_idx, eos_pos] = 0
        for field in soft_fields:
            rich_targets[field][row_idx, eos_pos] = int(row["soft_labels"][field])
    return {
        "previous_species_indices": previous_species,
        "previous_count_values": previous_count,
        "previous_n_values": previous_n,
        "n_targets": n_target,
        "species_targets": species_target,
        "count_targets": count_target,
        **{f"rich:{field}": value for field, value in rich_targets.items()},
    }


@torch.no_grad()
def evaluate(
    model: C3FDPlannerModel,
    loader: DataLoader,
    context: torch.Tensor,
    device: torch.device,
    soft_fields: tuple[str, ...],
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    batches = 0
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        rich = {field: batch[f"rich:{field}"] for field in soft_fields}
        output = model(
            context.expand(batch["n_targets"].shape[0], -1),
            previous_species_indices=batch["previous_species_indices"],
            previous_count_values=batch["previous_count_values"],
            previous_n_values=batch["previous_n_values"],
            n_targets=batch["n_targets"],
            species_targets=batch["species_targets"],
            count_targets=batch["count_targets"],
            rich_targets=rich,
            flags=SemanticHeadFlags(use_physics=True),
            loss_weights={f"rich:{field}": 0.2 for field in soft_fields},
        )
        for name, loss in output.losses.items():
            totals[name] = totals.get(name, 0.0) + float(loss.item())
        batches += 1
    model.train()
    return {name: value / max(1, batches) for name, value in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--context-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--semantic-size", type=int, default=256)
    parser.add_argument("--decoder-layers", type=int, default=2)
    parser.add_argument("--decoder-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--logging-steps", type=int, default=25)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocabulary_path = args.data_dir / "vocabulary.json"
    vocabulary_bytes = vocabulary_path.read_bytes()
    vocabulary = json.loads(vocabulary_bytes)
    data_manifest = json.loads((args.data_dir / "manifest.json").read_text(encoding="utf-8"))
    if data_manifest.get("gate", {}).get("planner_training_data_authorized") is not True:
        raise RuntimeError("C3FD data gate did not authorize training")
    context_payload = torch.load(args.context_path, map_location="cpu")
    context = torch.as_tensor(context_payload["context"], dtype=torch.float32)
    if context.shape[0] != 1:
        raise ValueError("frozen Planner context must contain one prompt row")
    soft_fields = tuple(sorted(vocabulary["soft_vocabulary"]))
    rich_dims = {
        field: len(vocabulary["soft_vocabulary"][field]) for field in soft_fields
    }
    physics = torch.tensor(vocabulary["physics"]["matrix"], dtype=torch.float32)
    config = C3FDPlannerConfig(
        context_size=int(context.shape[-1]),
        semantic_size=int(args.semantic_size),
        num_species=len(vocabulary["species"]),
        physics_feature_size=int(physics.shape[-1]),
        rich_soft_head_dims=rich_dims,
        decoder_layers=int(args.decoder_layers),
        decoder_heads=int(args.decoder_heads),
        decoder_dropout=float(args.dropout),
    )
    model = C3FDPlannerModel(config, physics_features=physics).to(device)
    context = context.to(device)
    train_ds = C3FDDataset(args.data_dir / "train.jsonl")
    val_ds = C3FDDataset(args.data_dir / "val.jsonl")
    eos_id = int(vocabulary["species_eos_id"])
    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=4,
        collate_fn=lambda rows: collate(rows, eos_species_id=eos_id, soft_fields=soft_fields),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=4,
        collate_fn=lambda rows: collate(rows, eos_species_id=eos_id, soft_fields=soft_fields),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay)
    )
    total_steps = int(args.epochs) * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_steps))
    args.output_dir.mkdir(parents=True)
    config_payload = {
        "schema": "h1a2_c3fd_planner_train_v1",
        "config": config.to_dict(),
        "data_dir": str(args.data_dir.resolve()),
        "context_path": str(args.context_path.resolve()),
        "vocabulary_sha256": hashlib.sha256(vocabulary_bytes).hexdigest(),
        "seed": int(args.seed),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "train_rows": len(train_ds),
        "val_rows": len(val_ds),
        "formula_bpe": False,
        "stability_condition": False,
        "rl": False,
        "rerank": False,
    }
    (args.output_dir / "train_config.json").write_text(
        json.dumps(config_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    start = time.time()
    step = 0
    history = []
    for epoch in range(int(args.epochs)):
        model.train()
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            rich = {field: batch[f"rich:{field}"] for field in soft_fields}
            output = model(
                context.expand(batch["n_targets"].shape[0], -1),
                previous_species_indices=batch["previous_species_indices"],
                previous_count_values=batch["previous_count_values"],
                previous_n_values=batch["previous_n_values"],
                n_targets=batch["n_targets"],
                species_targets=batch["species_targets"],
                count_targets=batch["count_targets"],
                rich_targets=rich,
                flags=SemanticHeadFlags(use_physics=True),
                loss_weights={f"rich:{field}": 0.2 for field in soft_fields},
            )
            if output.loss is None or not torch.isfinite(output.loss):
                raise RuntimeError("non-finite C3FD loss")
            optimizer.zero_grad(set_to_none=True)
            output.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1
            if step == 1 or step % int(args.logging_steps) == 0:
                event = {
                    "epoch": epoch,
                    "step": step,
                    "loss": float(output.loss.item()),
                    "lr": float(scheduler.get_last_lr()[0]),
                    "elapsed_sec": time.time() - start,
                }
                print(json.dumps(event), flush=True)
                history.append(event)
        validation = evaluate(model, val_loader, context, device, soft_fields)
        event = {"epoch": epoch + 1, "step": step, "validation": validation}
        print(json.dumps(event), flush=True)
        history.append(event)
    checkpoint = {
        "schema": "h1a2_c3fd_planner_checkpoint_v1",
        "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "config": config.to_dict(),
        "context": context_payload["context"],
        "context_manifest": {key: value for key, value in context_payload.items() if key != "context"},
        "vocabulary_sha256": config_payload["vocabulary_sha256"],
        "seed": int(args.seed),
    }
    torch.save(checkpoint, args.output_dir / "checkpoint.pt")
    metrics = {
        "steps": step,
        "elapsed_sec": time.time() - start,
        "final_validation": evaluate(model, val_loader, context, device, soft_fields),
        "history": history,
    }
    (args.output_dir / "train_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "_SUCCESS").touch()
    print(json.dumps(metrics["final_validation"], indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
