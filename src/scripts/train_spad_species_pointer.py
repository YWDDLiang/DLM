#!/usr/bin/env python3
"""Train the geometry-supervised species-program pointer on a frozen Planner Llama."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from crystal_dlm.c3fd_calibration import StratumInteraction
from crystal_dlm.c3fd_llama_typed_planner import (
    C3FDLlamaTypedPlannerConfig,
    C3FDLlamaTypedResidualPlanner,
)
from crystal_dlm.c3fd_planner_model import C3FDPlannerConfig
from crystal_dlm.species_program_pointer import (
    PlanConditionedSpeciesPointer,
    SpeciesPointerConfig,
    species_pointer_loss,
)
from scripts.train_c3fd_llama_typed_planner import (
    DATASET_SCHEMA as FUSED_SCHEMA,
    collate_typed_rows,
    compile_proposal_legal_mask,
)


DATASET_SCHEMA = "spad_species_pointer_row_v1"
TRAIN_SCHEMA = "spad_species_pointer_train_v1"
POINTER_STATE_SCHEMA = "spad_species_pointer_state_v1"
SEED = 86017
SOFT_FIELDS = (
    "lattice_system",
    "spacegroup_bucket",
    "volume_per_atom_bin",
)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number} is not an object")
            yield value


class PointerDataset(Dataset):
    def __init__(self, path: Path) -> None:
        self.rows = list(iter_jsonl(path))
        if not self.rows:
            raise ValueError(f"empty pointer dataset {path}")
        if any(row.get("schema") != DATASET_SCHEMA for row in self.rows):
            raise ValueError("pointer dataset schema changed")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


def metadata_bundle(checkpoint_path: Path, vocabulary_path: Path) -> Any:
    vocabulary = json.loads(vocabulary_path.read_text(encoding="utf-8"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    interaction = StratumInteraction.from_dict(checkpoint["stratum_interaction"])
    config = C3FDPlannerConfig(**checkpoint["config"])
    return SimpleNamespace(
        model=SimpleNamespace(config=config),
        interaction=interaction,
        vocabulary=vocabulary,
        proposal_legal_mask=compile_proposal_legal_mask(interaction, vocabulary),
        stratum_to_index={
            tuple(value): index for index, value in enumerate(interaction.strata)
        },
    )


def collate_pointer_rows(rows: list[dict[str, Any]], *, bundle: Any) -> dict[str, Tensor]:
    typed_rows = [{**row, "schema": FUSED_SCHEMA} for row in rows]
    batch = collate_typed_rows(typed_rows, bundle=bundle)
    width = max(len(row["canonical_atomic_numbers"]) for row in rows)
    batch_size = len(rows)
    atomic_numbers = torch.zeros(batch_size, width, dtype=torch.long)
    counts = torch.zeros(batch_size, width, dtype=torch.long)
    valid = torch.zeros(batch_size, width, dtype=torch.bool)
    teacher = torch.zeros(batch_size, width, dtype=torch.long)
    soft = torch.zeros(batch_size, 3, dtype=torch.long)
    for row_index, row in enumerate(rows):
        atomic = [int(value) for value in row["canonical_atomic_numbers"]]
        element_counts = [int(value) for value in row["canonical_element_counts"]]
        target = [int(value) for value in row["contact_tree_order_indices"]]
        if len(atomic) != len(element_counts) or sorted(target) != list(range(len(atomic))):
            raise ValueError("pointer row is not an exact Plan-element permutation")
        size = len(atomic)
        atomic_numbers[row_index, :size] = torch.tensor(atomic)
        counts[row_index, :size] = torch.tensor(element_counts)
        valid[row_index, :size] = True
        teacher[row_index, :size] = torch.tensor(target)
        soft[row_index] = torch.tensor(
            [int(row["soft_targets"][field]["label"]) for field in SOFT_FIELDS]
        )
    batch.update(
        {
            "pointer_atomic_numbers": atomic_numbers,
            "pointer_counts": counts,
            "pointer_valid_mask": valid,
            "pointer_teacher_order": teacher,
            "pointer_soft_field_ids": soft,
        }
    )
    return batch


def load_frozen_planner(
    *,
    llama_model: Path,
    planner_final: Path,
    device: torch.device,
) -> tuple[Any, C3FDLlamaTypedResidualPlanner, C3FDLlamaTypedPlannerConfig]:
    config_payload = json.loads(
        (planner_final / "typed_residual_config.json").read_text(encoding="utf-8")
    )
    typed_config = C3FDLlamaTypedPlannerConfig(
        **config_payload["typed_planner_config"]
    )
    typed_state = torch.load(
        planner_final / "typed_residual_state.pt", map_location="cpu"
    )
    typed = C3FDLlamaTypedResidualPlanner(typed_config)
    typed.load_state_dict(typed_state["state_dict"], strict=True)

    from transformers import AutoModelForCausalLM
    from peft import PeftModel

    llama = AutoModelForCausalLM.from_pretrained(
        llama_model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    llama = PeftModel.from_pretrained(
        llama,
        planner_final / "llama_adapter",
        is_trainable=False,
    )
    llama.to(device).eval()
    typed.to(device).eval()
    for module in (llama, typed):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in llama.parameters()) or any(
        parameter.requires_grad for parameter in typed.parameters()
    ):
        raise RuntimeError("Planner Llama or typed adapter was not frozen")
    return llama, typed, typed_config


@torch.no_grad()
def terminal_hidden(
    llama: Any,
    typed: C3FDLlamaTypedResidualPlanner,
    batch: Mapping[str, Tensor],
) -> Tensor:
    inputs = typed.typed_inputs_embeds(
        stability_goal_ids=batch["stability_goal_ids"],
        proposal_state_ids=batch["proposal_state_ids"],
        previous_species_indices=batch["previous_species_indices"],
        previous_count_values=batch["previous_count_values"],
        ledger_features=batch["ledger_features"],
    ).to(dtype=next(llama.parameters()).dtype)
    output = llama(
        inputs_embeds=inputs,
        attention_mask=batch["attention_mask"],
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    hidden = output.hidden_states[-1]
    rows = torch.arange(hidden.shape[0], device=hidden.device)
    return hidden[rows, batch["soft_position_indices"]].float()


def move_batch(batch: Mapping[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def order_metrics(predicted: Tensor, teacher: Tensor, valid: Tensor) -> dict[str, float]:
    exact = 0
    root = 0
    pair_correct = 0
    pair_total = 0
    for row in range(predicted.shape[0]):
        size = int(valid[row].sum().item())
        pred = [int(value) for value in predicted[row, :size].tolist()]
        target = [int(value) for value in teacher[row, :size].tolist()]
        if sorted(pred) != list(range(size)):
            raise RuntimeError("pointer decode changed the certified candidate set")
        exact += int(pred == target)
        root += int(pred[0] == target[0])
        pred_rank = {value: index for index, value in enumerate(pred)}
        target_rank = {value: index for index, value in enumerate(target)}
        for left in range(size):
            for right in range(left + 1, size):
                pair_correct += int(
                    (pred_rank[left] < pred_rank[right])
                    == (target_rank[left] < target_rank[right])
                )
                pair_total += 1
    rows = max(1, predicted.shape[0])
    return {
        "exact_permutation_accuracy": exact / rows,
        "root_accuracy": root / rows,
        "pairwise_order_accuracy": pair_correct / max(1, pair_total),
    }


@torch.inference_mode()
def evaluate(
    *,
    llama: Any,
    typed: C3FDLlamaTypedResidualPlanner,
    pointer: PlanConditionedSpeciesPointer,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    pointer.eval()
    losses: list[float] = []
    totals = CounterFloat()
    rows = 0
    for raw in loader:
        batch = move_batch(raw, device)
        hidden = terminal_hidden(llama, typed, batch)
        logits = pointer.permutation_logits(
            hidden,
            batch["pointer_atomic_numbers"],
            batch["pointer_counts"],
            batch["pointer_valid_mask"],
            batch["pointer_soft_field_ids"],
            teacher_order=batch["pointer_teacher_order"],
        )
        loss = species_pointer_loss(
            logits,
            batch["pointer_teacher_order"],
            batch["pointer_valid_mask"],
        )
        losses.append(float(loss.item()) * hidden.shape[0])
        decoded = pointer.decode(
            hidden,
            batch["pointer_atomic_numbers"],
            batch["pointer_counts"],
            batch["pointer_valid_mask"],
            batch["pointer_soft_field_ids"],
        )
        metrics = order_metrics(
            decoded,
            batch["pointer_teacher_order"],
            batch["pointer_valid_mask"],
        )
        for key, value in metrics.items():
            totals[key] += value * hidden.shape[0]
        rows += hidden.shape[0]
    return {
        "loss": sum(losses) / max(1, rows),
        **{key: value / max(1, rows) for key, value in totals.items()},
    }


class CounterFloat(dict[str, float]):
    def __missing__(self, key: str) -> float:
        return 0.0


def cosine_multiplier(step: int, *, total_steps: int, warmup_steps: int) -> float:
    if step < warmup_steps:
        return float(step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama-model", type=Path, required=True)
    parser.add_argument("--planner-final", type=Path, required=True)
    parser.add_argument("--pointer-data-dir", type=Path, required=True)
    parser.add_argument("--c3fd-checkpoint", type=Path, required=True)
    parser.add_argument("--vocabulary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    if int(args.seed) != SEED:
        raise ValueError(f"pointer seed is frozen at {SEED}")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if not torch.cuda.is_available():
        raise RuntimeError("SPAD pointer training requires its allocated GPU")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda")

    bundle = metadata_bundle(args.c3fd_checkpoint, args.vocabulary)
    llama, typed, typed_config = load_frozen_planner(
        llama_model=args.llama_model,
        planner_final=args.planner_final,
        device=device,
    )
    pointer_config = SpeciesPointerConfig(
        llama_hidden_size=int(typed_config.llama_hidden_size),
        pointer_size=256,
        max_elements=7,
        max_count=20,
        num_lattice_systems=int(typed_config.num_lattice_systems),
        num_spacegroup_buckets=int(typed_config.num_spacegroup_buckets),
        num_volume_per_atom_bins=int(typed_config.num_volume_per_atom_bins),
    )
    pointer = PlanConditionedSpeciesPointer(pointer_config).to(device)
    train_data = PointerDataset(args.pointer_data_dir / "train.jsonl")
    val_data = PointerDataset(args.pointer_data_dir / "val.jsonl")
    collate = lambda rows: collate_pointer_rows(rows, bundle=bundle)
    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        train_data,
        batch_size=int(args.batch_size),
        shuffle=True,
        generator=generator,
        num_workers=int(args.num_workers),
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        collate_fn=collate,
    )
    total_steps = len(train_loader)
    if total_steps <= int(args.warmup_steps):
        raise ValueError("pointer training is shorter than warmup")
    optimizer = torch.optim.AdamW(pointer.parameters(), lr=float(args.lr), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_multiplier(
            step,
            total_steps=total_steps,
            warmup_steps=int(args.warmup_steps),
        ),
    )
    config = {
        "schema": TRAIN_SCHEMA,
        "seed": SEED,
        "epochs": 1,
        "rows": len(train_data),
        "batch_size": int(args.batch_size),
        "total_steps": total_steps,
        "lr": float(args.lr),
        "teacher": "MP20 periodic maximum-contact-tree",
        "planner_llama_frozen": True,
        "c3fd_composition_model_frozen": True,
        "composition_candidate_set_mutable": False,
        "checkpoint_selection": "none",
        "pointer_config": asdict(pointer_config),
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    started = time.time()
    pointer.train()
    optimizer.zero_grad(set_to_none=True)
    with (args.output_dir / "training_log.jsonl").open(
        "x", encoding="utf-8", newline="\n"
    ) as log:
        for step, raw in enumerate(train_loader, start=1):
            batch = move_batch(raw, device)
            hidden = terminal_hidden(llama, typed, batch)
            logits = pointer.permutation_logits(
                hidden,
                batch["pointer_atomic_numbers"],
                batch["pointer_counts"],
                batch["pointer_valid_mask"],
                batch["pointer_soft_field_ids"],
                teacher_order=batch["pointer_teacher_order"],
            )
            loss = species_pointer_loss(
                logits,
                batch["pointer_teacher_order"],
                batch["pointer_valid_mask"],
            )
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError("non-finite pointer loss")
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(pointer.parameters(), 1.0))
            if not math.isfinite(grad_norm):
                raise RuntimeError("non-finite pointer gradient")
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            if any(parameter.grad is not None for parameter in llama.parameters()) or any(
                parameter.grad is not None for parameter in typed.parameters()
            ):
                raise RuntimeError("frozen Planner accumulated gradients")
            event = {
                "step": step,
                "loss": float(loss.item()),
                "grad_norm": grad_norm,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "elapsed_sec": time.time() - started,
            }
            log.write(json.dumps(event, sort_keys=True) + "\n")
            if step == 1 or step % 20 == 0 or step == total_steps:
                print(json.dumps(event, sort_keys=True), flush=True)
    if step != total_steps:
        raise RuntimeError("pointer update count changed")

    validation = evaluate(
        llama=llama,
        typed=typed,
        pointer=pointer,
        loader=val_loader,
        device=device,
    )
    if not all(math.isfinite(value) for value in validation.values()):
        raise RuntimeError("non-finite pointer validation metric")
    torch.save(
        {
            "schema": POINTER_STATE_SCHEMA,
            "config": asdict(pointer_config),
            "state_dict": pointer.state_dict(),
        },
        args.output_dir / "pointer_state.pt",
    )
    metrics = {
        "schema": TRAIN_SCHEMA,
        "status": "complete",
        "global_step": total_steps,
        "epochs_completed": 1,
        "elapsed_sec": time.time() - started,
        "validation": validation,
        "composition_candidate_set_preserved": True,
        "planner_llama_frozen_verified": True,
    }
    (args.output_dir / "train_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "_SUCCESS").touch()
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
