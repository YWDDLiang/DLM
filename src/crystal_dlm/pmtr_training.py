"""Reusable head-only training primitives for PMTR.

The retained SPAD DLM (including its LoRA adapter) is a frozen feature model.
Each single-component materialized row is reconstructed at the beginning of
its enclosing cell or XYZ transaction: the complete transaction is supervised
from one frozen-model forward and one cached manifold proposal.  Clean identity
and corrupted repair batches are separate optimizer updates.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, NamedTuple, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from crystal_dlm.llada_generation import _model_logits
from crystal_dlm.manifold_repair_head import ManifoldRepairOutput
from crystal_dlm.manifold_repair_objective import (
    ManifoldRepairLossConfig,
    manifold_repair_losses,
)
from crystal_dlm.pmtr_runtime import PMTRLogitTransform
from crystal_dlm.transaction_logits import (
    TransactionContext,
    TransactionModelStep,
    apply_transaction_logit_transform,
)
from scripts import llada_sft as SFT


TRAINING_MODES = ("clean_identity", "corrupt_repair")


@dataclass(frozen=True)
class PMTRTransactionSpec:
    kind: str
    active_positions: tuple[int, ...]
    component_indices: tuple[int, ...]
    site_index: int | None
    block_index: int | None
    site_order_index: int | None


@dataclass(frozen=True)
class MaterializedPMTRBatch:
    mode: str
    complete_tokens: Tensor
    noisy_tokens: Tensor
    model_mask: Tensor
    transaction_loss_mask: Tensor
    specs: tuple[PMTRTransactionSpec, ...]


@dataclass(frozen=True)
class PMTRTrainingExample:
    context: TransactionContext
    model_step: TransactionModelStep
    target_token_ids: Tensor
    repair_target: Mapping[str, Any] | None


class PMTRStepLosses(NamedTuple):
    total: Tensor
    token: Tensor
    spd: Tensor
    torus: Tensor
    step: Tensor


class PMTRJsonlDataset(Dataset):
    """Augment the established JSONL SFT dataset with PMTR row metadata."""

    def __init__(
        self,
        path: Path,
        tokenizer: Any,
        max_length: int,
        *,
        limit: int | None = None,
    ) -> None:
        self.sft = SFT.JsonlSftDataset(
            Path(path),
            tokenizer,
            int(max_length),
            fail_on_truncation=True,
        )
        upper = len(self.sft) if limit is None else min(int(limit), len(self.sft))
        if upper <= 0:
            raise ValueError("PMTR dataset is empty")
        self.indices = tuple(
            index
            for index in range(upper)
            if self.sft.rows[index].get("repair_target") is not None
        )
        if not self.indices:
            raise ValueError("PMTR dataset contains no corrupted repair targets")

    @property
    def source_row_count(self) -> int:
        return len(self.sft)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        source_index = self.indices[int(index)]
        item = dict(self.sft[source_index])
        row = self.sft.rows[source_index]
        program = row.get("species_program")
        if not isinstance(program, list) or not program:
            raise ValueError("PMTR row lacks a nonempty species_program")
        target = row.get("repair_target")
        if not isinstance(target, Mapping):
            raise ValueError("PMTR row lacks repair_target")
        item["pmtr_repair_target"] = dict(target)
        item["pmtr_plan_metadata"] = dict(row.get("plan_state") or {})
        item["pmtr_program_metadata"] = {
            "species_order": [str(value) for value in program],
            "source": row.get("species_program_source"),
        }
        item["pmtr_closure"] = dict(row.get("closure") or {})
        item["pmtr_source_row_idx"] = int(row.get("source_row_idx", source_index))
        return item


class PMTRDataCollator:
    """Preserve PMTR metadata while reusing the established SFT collator."""

    def __init__(self, tokenizer: Any) -> None:
        self.base = SFT.DataCollator(tokenizer)

    def __call__(self, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = self.base(list(rows))
        result["pmtr_repair_targets"] = [row["pmtr_repair_target"] for row in rows]
        result["pmtr_plan_metadata"] = [row["pmtr_plan_metadata"] for row in rows]
        result["pmtr_program_metadata"] = [
            row["pmtr_program_metadata"] for row in rows
        ]
        result["pmtr_closure"] = [row["pmtr_closure"] for row in rows]
        result["pmtr_source_row_indices"] = [
            int(row["pmtr_source_row_idx"]) for row in rows
        ]
        return result


def freeze_spad_model(model: nn.Module) -> int:
    """Freeze the complete retained DLM/LoRA and return its parameter count."""

    model.eval()
    count = 0
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        count += int(parameter.numel())
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("retained SPAD model was not fully frozen")
    return count


def _transaction_spec(
    repair_target: Mapping[str, Any],
    closure: Mapping[str, Any],
    *,
    prompt_length: int,
    sequence_length: int,
) -> PMTRTransactionSpec:
    raw_kind = str(repair_target.get("kind") or "")
    if raw_kind == "cell":
        relative = tuple(range(1, 7))
        kind = "cell"
        site_index = None
        components = tuple(range(6))
    elif raw_kind in ("site", "site_xyz"):
        site_index = int(repair_target.get("site_slot_index"))
        relative = tuple(8 + 4 * site_index + component for component in range(3))
        kind = "site_xyz"
        components = (0, 1, 2)
    else:
        raise ValueError(f"unsupported PMTR repair target kind {raw_kind!r}")
    absolute = tuple(int(prompt_length) + position for position in relative)
    if any(not 0 <= position < int(sequence_length) for position in absolute):
        raise ValueError("PMTR transaction lies outside the token sequence")
    block_index = closure.get("reverse_block_index")
    site_order = closure.get("site_index_within_block")
    return PMTRTransactionSpec(
        kind=kind,
        active_positions=absolute,
        component_indices=components,
        site_index=site_index,
        block_index=None if block_index is None else int(block_index),
        site_order_index=None if site_index is None else int(site_order or 0),
    )


def materialize_transaction_start(
    batch: Mapping[str, Any],
    *,
    mode: str,
    mask_id: int,
) -> MaterializedPMTRBatch:
    """Expand a component row into its inference-time transaction-start state.

    Existing forced masks retain the unrepaired block suffix.  The active cell
    or XYZ positions are added back to the mask, recreating component zero/X;
    only the complete active transaction is supervised.
    """

    if mode not in TRAINING_MODES:
        raise ValueError(f"unknown PMTR training mode {mode!r}")
    target_ids = batch["input_ids"]
    source_ids = batch.get("source_input_ids")
    forced = batch.get("forced_mask_indices")
    original_loss = batch.get("forced_loss_indices")
    if source_ids is None or forced is None or original_loss is None:
        raise ValueError("PMTR requires paired source tokens and forced masks")
    if not (
        target_ids.shape
        == source_ids.shape
        == forced.shape
        == original_loss.shape
        == batch["attention_mask"].shape
    ):
        raise ValueError("PMTR batch tensor shapes differ")
    repair_targets = batch["pmtr_repair_targets"]
    closures = batch["pmtr_closure"]
    if len(repair_targets) != int(target_ids.shape[0]) or len(closures) != int(
        target_ids.shape[0]
    ):
        raise ValueError("PMTR metadata batch size differs from token batch")

    complete = target_ids if mode == "clean_identity" else source_ids
    model_mask = forced.clone()
    transaction_loss = torch.zeros_like(forced, dtype=torch.bool)
    specs: list[PMTRTransactionSpec] = []
    for row_index in range(int(target_ids.shape[0])):
        length = int(batch["attention_mask"][row_index].sum().detach().item())
        prompt = int(batch["prompt_lengths"][row_index].detach().item())
        spec = _transaction_spec(
            repair_targets[row_index],
            closures[row_index],
            prompt_length=prompt,
            sequence_length=length,
        )
        specs.append(spec)
        active = torch.tensor(
            spec.active_positions,
            dtype=torch.long,
            device=target_ids.device,
        )
        transaction_loss[row_index, active] = True
        model_mask[row_index, active] = True
        row_loss = set(
            int(value)
            for value in torch.nonzero(original_loss[row_index], as_tuple=False)
            .flatten()
            .tolist()
        )
        if len(row_loss) != 1 or not row_loss <= set(spec.active_positions):
            raise ValueError("component row does not belong to reconstructed transaction")
        if bool((forced[row_index] & ~batch["attention_mask"][row_index].bool()).any()):
            raise ValueError("forced mask extends into sequence padding")

    processed = SFT.forced_rollout_process(
        complete,
        batch["attention_mask"],
        batch["prompt_lengths"],
        model_mask,
        transaction_loss,
        mask_id=int(mask_id),
    )
    return MaterializedPMTRBatch(
        mode=mode,
        complete_tokens=complete,
        noisy_tokens=processed["noisy"],
        model_mask=model_mask,
        transaction_loss_mask=transaction_loss,
        specs=tuple(specs),
    )


@torch.no_grad()
def frozen_spad_forward(
    model: nn.Module,
    batch: Mapping[str, Any],
    materialized: MaterializedPMTRBatch,
    *,
    mask_id: int,
) -> TransactionModelStep:
    """Run the frozen DLM once and expose detached logits plus final hidden state."""

    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("base SPAD forward requires a fully frozen model")
    context_model = model.module if hasattr(model, "module") else model
    if hasattr(context_model, "set_geometry_context"):
        context_model.set_geometry_context(batch["prompt_lengths"], batch["num_atoms"])
    positions = torch.arange(
        materialized.noisy_tokens.shape[1], device=materialized.noisy_tokens.device
    ).unsqueeze(0)
    prompt_index = positions < batch["prompt_lengths"].unsqueeze(1)
    step = _model_logits(
        model,
        materialized.noisy_tokens,
        batch["attention_mask"],
        prompt_index,
        0.0,
        int(mask_id),
        return_model_step=True,
    )
    if not isinstance(step, TransactionModelStep) or step.hidden_states is None:
        raise RuntimeError("retained DLM did not expose final hidden states")
    return TransactionModelStep(
        token_ids=step.token_ids.detach(),
        logits=step.logits.detach(),
        hidden_states=step.hidden_states.detach(),
    )


def build_training_examples(
    batch: Mapping[str, Any],
    materialized: MaterializedPMTRBatch,
    model_step: TransactionModelStep,
) -> tuple[PMTRTrainingExample, ...]:
    examples: list[PMTRTrainingExample] = []
    for row_index, spec in enumerate(materialized.specs):
        length = int(batch["attention_mask"][row_index].sum().detach().item())
        prompt = int(batch["prompt_lengths"][row_index].detach().item())
        row_step = TransactionModelStep(
            token_ids=model_step.token_ids[row_index, :length].detach().clone(),
            logits=model_step.logits[row_index, :length].detach(),
            hidden_states=model_step.hidden_states[row_index, :length].detach(),
        )
        complete = materialized.complete_tokens[row_index, :length].detach().clone()
        context = TransactionContext(
            kind=spec.kind,
            active_positions=spec.active_positions,
            complete_pre_remask_tokens=complete,
            previous_active_token_ids=tuple(
                int(complete[position].detach().item())
                for position in spec.active_positions
            ),
            prompt_length=prompt,
            gen_length=length - prompt,
            plan_metadata=batch["pmtr_plan_metadata"][row_index],
            program_metadata=batch["pmtr_program_metadata"][row_index],
            batch_index=row_index,
            block_index=spec.block_index,
            site_index=spec.site_index,
            site_order_index=spec.site_order_index,
            component_indices=spec.component_indices,
            lattice_version=0,
        )
        repair_target = (
            None
            if materialized.mode == "clean_identity"
            else batch["pmtr_repair_targets"][row_index]
        )
        examples.append(
            PMTRTrainingExample(
                context=context,
                model_step=row_step,
                target_token_ids=batch["input_ids"][
                    row_index,
                    torch.tensor(
                        spec.active_positions,
                        dtype=torch.long,
                        device=batch["input_ids"].device,
                    ),
                ],
                repair_target=repair_target,
            )
        )
    return tuple(examples)


def _mean_or_graph_zero(values: list[Tensor], reference: Tensor) -> Tensor:
    if values:
        return torch.stack(values).mean()
    return reference.sum() * 0.0


class PMTRHeadOnlyModule(nn.Module):
    """Train only a manifold head through the exact token-transport renderer."""

    def __init__(
        self,
        repair_head: nn.Module,
        tokenizer: Any,
        *,
        loss_config: ManifoldRepairLossConfig,
        runtime_config: Any,
    ) -> None:
        super().__init__()
        self.transform = PMTRLogitTransform(repair_head, tokenizer, runtime_config)
        self.loss_config = loss_config

    @property
    def repair_head(self) -> nn.Module:
        return self.transform.repair_head

    def forward(
        self,
        examples: Sequence[PMTRTrainingExample],
    ) -> PMTRStepLosses:
        if not examples:
            raise ValueError("PMTR head update requires at least one example")
        token_losses: list[Tensor] = []
        spd_losses: list[Tensor] = []
        torus_losses: list[Tensor] = []
        step_losses: list[Tensor] = []
        reference = None
        for example in examples:
            proposal = self.transform.prepare(example.context, example.model_step)
            reference = proposal.head_output.lattice_tangent
            component_logits = []
            for component, position in enumerate(example.context.active_positions):
                component_logits.append(
                    apply_transaction_logit_transform(
                        self.transform,
                        component=component,
                        active_logits=example.model_step.logits[position],
                        proposal=proposal,
                        context=example.context,
                    )
                )
            token_losses.append(
                F.cross_entropy(
                    torch.stack(component_logits),
                    example.target_token_ids,
                    reduction="mean",
                )
            )

            sites = int(proposal.head_output.cartesian_site_delta.shape[1])
            target_lattice = proposal.head_output.lattice_tangent.new_zeros((1, 3, 3))
            target_sites = proposal.head_output.cartesian_site_delta.new_zeros(
                (1, sites, 3)
            )
            site_mask = torch.ones(
                (1, sites), dtype=torch.bool, device=target_sites.device
            )
            lattice_active = torch.tensor(
                [example.context.kind == "cell"],
                dtype=torch.bool,
                device=target_sites.device,
            )
            active_site = torch.zeros_like(site_mask)
            if example.context.kind == "site_xyz":
                active_site[0, int(example.context.site_index)] = True
            if example.repair_target is not None:
                target_kind = str(example.repair_target.get("kind") or "")
                if example.context.kind == "cell":
                    if target_kind != "cell":
                        raise ValueError("cell context received a non-cell repair target")
                    target_lattice[0] = torch.as_tensor(
                        example.repair_target["lattice_tangent"],
                        dtype=target_lattice.dtype,
                        device=target_lattice.device,
                    )
                else:
                    if target_kind not in ("site", "site_xyz") or int(
                        example.repair_target["site_slot_index"]
                    ) != int(example.context.site_index):
                        raise ValueError("site context received a mismatched repair target")
                    target_sites[0, int(example.context.site_index)] = torch.as_tensor(
                        example.repair_target["cartesian_site_delta_A"],
                        dtype=target_sites.dtype,
                        device=target_sites.device,
                    )
            losses = manifold_repair_losses(
                proposal.head_output,
                target_lattice_tangent=target_lattice,
                target_cartesian_site_delta=target_sites,
                site_mask=site_mask,
                lattice_active=lattice_active,
                active_site_mask=active_site,
                config=self.loss_config,
            )
            if example.context.kind == "cell":
                spd_losses.append(losses.lattice)
            else:
                torus_losses.append(losses.coordinate)
            step_losses.append(losses.step)

        assert reference is not None
        token = torch.stack(token_losses).mean()
        spd = _mean_or_graph_zero(spd_losses, reference)
        torus = _mean_or_graph_zero(torus_losses, reference)
        step = torch.stack(step_losses).mean()
        total = token + spd + torus + step
        return PMTRStepLosses(
            total=total,
            token=token,
            spd=spd,
            torus=torus,
            step=step,
        )


def gradient_l2(parameters: Sequence[nn.Parameter]) -> float:
    squared = torch.zeros((), dtype=torch.float64)
    found = False
    for parameter in parameters:
        if parameter.grad is None:
            continue
        found = True
        squared += parameter.grad.detach().double().square().sum().cpu()
    if not found:
        return 0.0
    value = float(torch.sqrt(squared).item())
    if not torch.isfinite(torch.tensor(value)):
        raise FloatingPointError("PMTR gradient norm is not finite")
    return value


def probe_component_gradient_scales(
    module: nn.Module,
    examples: Sequence[PMTRTrainingExample],
) -> dict[str, float]:
    """Measure each loss gradient with independent forwards; never tune weights."""

    result: dict[str, float] = {}
    parameters = tuple(parameter for parameter in module.parameters() if parameter.requires_grad)
    for name in ("token", "spd", "torus", "step"):
        module.zero_grad(set_to_none=True)
        losses = module(examples)
        getattr(losses, name).backward()
        result[name] = gradient_l2(parameters)
    module.zero_grad(set_to_none=True)
    return result


def move_tensor_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, Tensor) else value
        for key, value in batch.items()
    }


__all__ = [
    "MaterializedPMTRBatch",
    "PMTRDataCollator",
    "PMTRHeadOnlyModule",
    "PMTRJsonlDataset",
    "PMTRStepLosses",
    "PMTRTrainingExample",
    "PMTRTransactionSpec",
    "TRAINING_MODES",
    "build_training_examples",
    "freeze_spad_model",
    "frozen_spad_forward",
    "gradient_l2",
    "materialize_transaction_start",
    "move_tensor_batch",
    "probe_component_gradient_scales",
]
