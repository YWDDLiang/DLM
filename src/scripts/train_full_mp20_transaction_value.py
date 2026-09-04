#!/usr/bin/env python3
"""Train full-MP20 transaction-value DLMs with two comparable teachers.

The two routes consume the *same* 27,136-row Llama-programmed deployment
ledger and differ only in the terminal value field used to construct the
finite-candidate posterior:

``single_point_full``
    uses the terminal single-point CHGNet energy;

``basin_consistent_full``
    uses the terminal energy after the frozen continuation and relaxation.

Each route is launched independently with ``torchrun --nproc_per_node=2``.
Every rank processes eight sources per optimizer update.  Gradients are
manually all-reduced and averaged, giving a global batch of sixteen without
wrapping the adapter-switching policy/reference model in DDP.  Clean SPAD CE
and transaction posterior updates alternate, and each objective sees every
MP20-train source exactly once.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.distributed as dist

from crystal_dlm.fixed_slot import MASK_TOKEN_ID
from crystal_dlm.potential_closure import (
    MAX_KL_BUDGET_NATS,
    build_potential_closure_posterior,
    potential_closure_loss,
)
from crystal_dlm.r5_dynamic_length import exact_body_token_count


EXPECTED_ROWS = 27_136
WORLD_SIZE = 2
LOCAL_BATCH_SIZE = 8
GLOBAL_BATCH_SIZE = WORLD_SIZE * LOCAL_BATCH_SIZE
POSTERIOR_UPDATES = EXPECTED_ROWS // GLOBAL_BATCH_SIZE
CLEAN_CE_UPDATES = POSTERIOR_UPDATES
TOTAL_UPDATES = POSTERIOR_UPDATES + CLEAN_CE_UPDATES
LEARNING_RATE = 5.0e-6
WARMUP_UPDATES = 100
MAX_LENGTH = 382
LOGGING_UPDATES = 10
ROUTES = ("single_point_full", "basin_consistent_full")
VALUE_FIELDS = {
    "single_point_full": "terminal_single_point_energy_eV_per_atom",
    "basin_consistent_full": "terminal_basin_energy_eV_per_atom",
}
DEPLOYMENT_STAGES = ("cell", "anchor_second", "anchor_first")
RUN_SCHEMA = "full_mp20_transaction_value_train_v1"
DATASET_SCHEMA = "full_mp20_transaction_value_dataset_v1"


def _runtime_modules() -> tuple[Any, Any, Any]:
    """Import heavyweight training modules only for an actual run."""

    from scripts import llada_d3po as d3po
    from scripts import llada_sft as sft
    from scripts import train_potential_closure as legacy

    return d3po, sft, legacy


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number} is not a JSON object")
            yield value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    Path(path).write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(value), ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def optimizer_objective(update: int) -> str:
    """Odd updates are clean CE and even updates are posterior updates."""

    if not 1 <= int(update) <= TOTAL_UPDATES:
        raise ValueError(f"update must lie in [1, {TOTAL_UPDATES}]")
    return "clean_ce" if int(update) % 2 else "transaction_posterior"


def learning_rate_for_update(update: int) -> float:
    if not 1 <= int(update) <= TOTAL_UPDATES:
        raise ValueError(f"update must lie in [1, {TOTAL_UPDATES}]")
    return LEARNING_RATE * min(1.0, float(update) / float(WARMUP_UPDATES))


def checkpoint_steps() -> tuple[int, ...]:
    """The formal contract exposes only the endpoint checkpoint."""

    return (TOTAL_UPDATES,)


def frozen_source_permutation(size: int, *, seed: int) -> tuple[int, ...]:
    if int(size) <= 0:
        raise ValueError("source count must be positive")
    values = list(range(int(size)))
    random.Random(int(seed)).shuffle(values)
    return tuple(values)


def rank_batch_indices(
    permutation: Sequence[int],
    batch_index: int,
    rank: int,
    *,
    world_size: int = WORLD_SIZE,
    local_batch_size: int = LOCAL_BATCH_SIZE,
) -> tuple[int, ...]:
    """Return one rank's disjoint contiguous shard of a global batch."""

    if not 0 <= int(rank) < int(world_size):
        raise ValueError("rank lies outside world_size")
    global_batch = int(world_size) * int(local_batch_size)
    start = int(batch_index) * global_batch
    end = start + global_batch
    if start < 0 or end > len(permutation):
        raise IndexError("batch_index lies outside the frozen epoch")
    local_start = start + int(rank) * int(local_batch_size)
    local_end = local_start + int(local_batch_size)
    result = tuple(int(value) for value in permutation[local_start:local_end])
    if len(result) != int(local_batch_size):
        raise RuntimeError("rank batch size changed")
    return result


def average_gradients_(
    parameters: Sequence[torch.nn.Parameter],
    *,
    world_size: int,
    all_reduce_fn: Callable[[torch.Tensor], Any] | None = None,
) -> None:
    """Sum then average every policy gradient across ranks.

    Missing local gradients are materialized as zeros so all ranks execute the
    same collective sequence.  This matters when a local batch contains only
    retained failure/uninformative rows.
    """

    if int(world_size) <= 0:
        raise ValueError("world_size must be positive")
    reducer = all_reduce_fn
    if reducer is None:
        if not dist.is_initialized():
            raise RuntimeError("distributed process group is not initialized")
        reducer = dist.all_reduce
    for parameter in parameters:
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        if parameter.grad.is_sparse:
            raise TypeError("sparse gradients are unsupported")
        reducer(parameter.grad)
        parameter.grad.div_(float(world_size))


def _path_value(mapping: Mapping[str, Any], path: str) -> Any:
    value: Any = mapping
    for component in str(path).split("."):
        if not isinstance(value, Mapping) or component not in value:
            raise KeyError(path)
        value = value[component]
    return value


def _first_value(
    mappings: Sequence[Mapping[str, Any]],
    primary: str,
    aliases: Sequence[str] = (),
) -> Any:
    for field in (primary, *aliases):
        for mapping in mappings:
            try:
                return _path_value(mapping, field)
            except KeyError:
                continue
    raise KeyError(primary)


def _optional_value(
    mappings: Sequence[Mapping[str, Any]],
    primary: str,
    aliases: Sequence[str] = (),
    *,
    default: Any = None,
) -> Any:
    try:
        return _first_value(mappings, primary, aliases)
    except KeyError:
        return default


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    return [
        int(value)
        for value in tokenizer(text, add_special_tokens=False)["input_ids"]
    ]


@dataclass(frozen=True)
class FieldConfig:
    state: str = "state"
    source_index: str = "source_idx"
    source_weight: str = "source_weight"
    prompt: str = "prompt"
    source_answer: str = "source_answer"
    active_positions: str = "active_positions"
    candidates: str = "candidates"
    candidate_action: str = "action_token_ids"
    candidate_legality: str = "legality"
    single_point_energy: str = VALUE_FIELDS["single_point_full"]
    basin_energy: str = VALUE_FIELDS["basin_consistent_full"]
    species_program: str = "species_program"
    deployment_stage: str = "deployment_stage"


def route_value_field(route: str, fields: FieldConfig) -> str:
    if route == "single_point_full":
        return fields.single_point_energy
    if route == "basin_consistent_full":
        return fields.basin_energy
    raise ValueError(f"unknown route {route}")


def _state_mapping(row: Mapping[str, Any], fields: FieldConfig) -> Mapping[str, Any]:
    state = _optional_value([row], fields.state, default=row)
    if not isinstance(state, Mapping):
        raise ValueError("state must be a mapping when present")
    return state


def _candidate_action(candidate: Mapping[str, Any], fields: FieldConfig) -> tuple[int, ...]:
    raw = _first_value(
        [candidate],
        fields.candidate_action,
        ("action_tokens", "tokens"),
    )
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise ValueError("candidate action token ids must be a sequence")
    action = tuple(int(value) for value in raw)
    if any(value < 0 for value in action):
        raise ValueError("candidate action token ids must be nonnegative")
    return action


def _candidate_legal(candidate: Mapping[str, Any], fields: FieldConfig) -> bool:
    raw = _first_value(
        [candidate],
        fields.candidate_legality,
        ("legal", "valid_action", "is_legal"),
    )
    if not isinstance(raw, bool):
        raise ValueError("candidate legality must be boolean")
    return raw


def _energy_value(
    row: Mapping[str, Any],
    candidate: Mapping[str, Any],
    candidate_index: int,
    field: str,
) -> float:
    """Read a candidate-nested scalar or a row-level list/mapping field."""

    value: Any
    try:
        value = _path_value(candidate, field)
    except KeyError:
        terminal_values = candidate.get("terminal_values")
        if isinstance(terminal_values, Mapping) and field in terminal_values:
            value = terminal_values[field]
        else:
            try:
                container = _path_value(row, field)
            except KeyError as exc:
                raise ValueError(f"energy field {field!r} is absent") from exc
            if isinstance(container, Sequence) and not isinstance(container, (str, bytes)):
                if int(candidate_index) >= len(container):
                    raise ValueError(f"energy field {field!r} is shorter than K")
                value = container[int(candidate_index)]
            elif isinstance(container, Mapping):
                key = str(int(candidate_index))
                if key in container:
                    value = container[key]
                elif int(candidate_index) in container:
                    value = container[int(candidate_index)]
                else:
                    raise ValueError(f"energy field {field!r} lacks candidate index")
            else:
                raise ValueError(
                    f"row-level energy field {field!r} must be a list or mapping"
                )
    if value is None:
        return math.nan
    result = float(value)
    return result if math.isfinite(result) else math.nan


def _has_energy_container(
    row: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]], field: str
) -> bool:
    try:
        _path_value(row, field)
        return True
    except KeyError:
        pass
    for candidate in candidates:
        try:
            _path_value(candidate, field)
            continue
        except KeyError:
            terminal_values = candidate.get("terminal_values")
            if isinstance(terminal_values, Mapping) and field in terminal_values:
                continue
            return False
    return True


def _explicit_zero_posterior(row: Mapping[str, Any]) -> bool:
    if row.get("informative") is False or row.get("failed") is True:
        return True
    if row.get("failure") not in (None, "", False):
        return True
    if row.get("downstream_failure") not in (None, "", False):
        return True
    return str(row.get("status", "")).lower() in {
        "failed",
        "error",
        "unavailable",
        "no_information",
    }


class FullMP20TransactionValueDataset:
    """Strict-denominator, flexible-schema full-MP20 transaction groups."""

    def __init__(
        self,
        path: Path,
        tokenizer: Any,
        *,
        fields: FieldConfig,
        max_length: int = MAX_LENGTH,
        expected_rows: int = EXPECTED_ROWS,
        active_positions_absolute: bool = False,
        require_llama_program: bool = True,
    ) -> None:
        raw_rows = list(iter_jsonl(path))
        if len(raw_rows) != int(expected_rows):
            raise ValueError(
                f"labelled_groups must contain exactly {expected_rows} rows"
            )
        self.path = Path(path).resolve()
        self.tokenizer = tokenizer
        self.fields = fields
        self.max_length = int(max_length)
        self.active_positions_absolute = bool(active_positions_absolute)
        self.require_llama_program = bool(require_llama_program)
        indexed: dict[int, dict[str, Any]] = {}
        self.k_histogram: Counter[int] = Counter()
        self.stage_counts: Counter[str] = Counter()
        self.program_source_counts: Counter[str] = Counter()
        self.explicit_zero_rows = 0
        for ordinal, row in enumerate(raw_rows):
            state = _state_mapping(row, fields)
            source_idx = int(
                _first_value(
                    [row, state],
                    fields.source_index,
                    ("source_row_idx", "group_idx", "sample_idx"),
                )
            )
            if source_idx in indexed:
                raise ValueError(f"duplicate source_idx {source_idx}")
            weight = float(
                _first_value([row, state], fields.source_weight, ("weight",))
            )
            if weight != 1.0:
                raise ValueError(f"source {source_idx} does not have source_weight=1")
            prompt = _first_value([state, row], fields.prompt)
            answer = _first_value(
                [state, row], fields.source_answer, ("teacher_answer", "answer")
            )
            if not isinstance(prompt, str) or not isinstance(answer, str) or not answer:
                raise ValueError(f"source {source_idx} lacks prompt/source_answer text")
            active = _first_value([state, row], fields.active_positions)
            if isinstance(active, (str, bytes)) or not isinstance(active, Sequence):
                raise ValueError(f"source {source_idx} active_positions is not a list")
            active_int = tuple(int(value) for value in active)
            if len(active_int) not in (3, 6) or len(set(active_int)) != len(active_int):
                raise ValueError(f"source {source_idx} is not a complete 3/6-token transaction")
            candidates = _first_value([row, state], fields.candidates)
            if (
                isinstance(candidates, (str, bytes))
                or not isinstance(candidates, Sequence)
                or not 2 <= len(candidates) <= 4
                or not all(isinstance(value, Mapping) for value in candidates)
            ):
                raise ValueError(f"source {source_idx} candidate K must lie in [2,4]")
            actions = [_candidate_action(value, fields) for value in candidates]
            if any(len(action) != len(active_int) for action in actions):
                raise ValueError(f"source {source_idx} candidate transaction width changed")
            if len(set(actions)) != len(actions):
                raise ValueError(f"source {source_idx} candidate actions are not unique")
            for candidate in candidates:
                _candidate_legal(candidate, fields)
            for value_field in (fields.single_point_energy, fields.basin_energy):
                if not _has_energy_container(row, candidates, value_field):
                    raise ValueError(
                        f"source {source_idx} lacks declared value field {value_field!r}"
                    )
            stage = str(
                _optional_value(
                    [state, row], fields.deployment_stage, ("stage",), default="unknown"
                )
            )
            program = _optional_value(
                [state, row], fields.species_program, ("program",), default=None
            )
            if self.require_llama_program:
                if stage not in DEPLOYMENT_STAGES:
                    raise ValueError(
                        f"source {source_idx} lacks a recognized Llama-programmed stage"
                    )
                if (
                    isinstance(program, (str, bytes))
                    or not isinstance(program, Sequence)
                    or not program
                ):
                    raise ValueError(
                        f"source {source_idx} lacks its Llama species program"
                    )
            self.k_histogram[len(candidates)] += 1
            self.stage_counts[stage] += 1
            self.program_source_counts[
                str(row.get("species_program_source", state.get("species_program_source", "unspecified")))
            ] += 1
            self.explicit_zero_rows += int(_explicit_zero_posterior(row))
            indexed[source_idx] = row
        expected_indices = set(range(int(expected_rows)))
        if set(indexed) != expected_indices:
            missing = sorted(expected_indices - set(indexed))
            extra = sorted(set(indexed) - expected_indices)
            raise ValueError(
                "source_idx must cover the complete contiguous denominator; "
                f"missing={missing[:8]} extra={extra[:8]}"
            )
        self.rows = tuple(indexed[index] for index in range(int(expected_rows)))

    def __len__(self) -> int:
        return len(self.rows)

    def _candidate_rows(self, row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
        state = _state_mapping(row, self.fields)
        return list(_first_value([row, state], self.fields.candidates))

    def materialize(self, index: int, *, route: str) -> dict[str, Any]:
        row = self.rows[int(index)]
        state = _state_mapping(row, self.fields)
        source_idx = int(
            _first_value(
                [row, state],
                self.fields.source_index,
                ("source_row_idx", "group_idx", "sample_idx"),
            )
        )
        prompt = str(_first_value([state, row], self.fields.prompt)).rstrip() + "\n"
        answer = str(
            _first_value(
                [state, row],
                self.fields.source_answer,
                ("teacher_answer", "answer"),
            )
        )
        prompt_ids = _token_ids(self.tokenizer, prompt)
        source_ids = _token_ids(self.tokenizer, answer)
        plan = _optional_value([state, row], "plan_state", default={})
        if isinstance(plan, Mapping) and plan.get("N") is not None:
            if len(source_ids) != exact_body_token_count(int(plan["N"])):
                raise ValueError(f"source {source_idx} is not exact 7+4N")
        if len(prompt_ids) + len(source_ids) > self.max_length:
            raise ValueError(f"source {source_idx} exceeds max_length")
        active_raw = tuple(
            int(value)
            for value in _first_value(
                [state, row], self.fields.active_positions
            )
        )
        if self.active_positions_absolute:
            active_relative = tuple(value - len(prompt_ids) for value in active_raw)
        else:
            active_relative = active_raw
        if any(value < 0 or value >= len(source_ids) for value in active_relative):
            raise ValueError(f"source {source_idx} active position is outside the answer")
        active_absolute = torch.tensor(
            [len(prompt_ids) + value for value in active_relative], dtype=torch.long
        )
        full = torch.tensor(prompt_ids + source_ids, dtype=torch.long)
        full[active_absolute] = int(MASK_TOKEN_ID)

        candidates = self._candidate_rows(row)
        actions = [_candidate_action(candidate, self.fields) for candidate in candidates]
        legal = [_candidate_legal(candidate, self.fields) for candidate in candidates]
        no_op = tuple(source_ids[position] for position in active_relative)
        try:
            no_op_index = actions.index(no_op)
        except ValueError as exc:
            raise ValueError(f"source {source_idx} has no no-op candidate") from exc
        order = [no_op_index] + [value for value in range(len(actions)) if value != no_op_index]
        actions = [actions[value] for value in order]
        legal = [legal[value] for value in order]
        if not legal[0]:
            raise ValueError(f"source {source_idx} marks its no-op illegal")
        value_field = route_value_field(route, self.fields)
        energies = [
            _energy_value(row, candidates[original], original, value_field)
            for original in order
        ]
        action_tensor = torch.tensor(actions, dtype=torch.long)
        legal_tensor = torch.tensor(legal, dtype=torch.bool)
        stage = str(
            _optional_value(
                [state, row], self.fields.deployment_stage, ("stage",), default="unknown"
            )
        )
        metadata = {
            "source_idx": source_idx,
            "deployment_stage": stage,
            "species_program": _optional_value(
                [state, row], self.fields.species_program, ("program",), default=[]
            ),
            "plan_state": dict(plan) if isinstance(plan, Mapping) else {},
        }
        explicit_zero = _explicit_zero_posterior(row)
        legal_known = sum(
            bool(is_legal and math.isfinite(energy))
            for is_legal, energy in zip(legal, energies, strict=True)
        )
        return {
            "source_idx": source_idx,
            "source_weight": 1.0,
            "input_ids": full,
            "attention_mask": torch.ones_like(full),
            "active_absolute": active_absolute,
            "active_relative": active_relative,
            "action_tokens": action_tensor,
            "no_op_tokens": no_op,
            "legal_mask": legal_tensor,
            "energies": torch.tensor(energies, dtype=torch.float64),
            "differing_positions": tuple(
                tuple(
                    position
                    for position, candidate_token, source_token in zip(
                        active_relative, action, no_op, strict=True
                    )
                    if candidate_token != source_token
                )
                for action in actions
            ),
            "state_metadata_by_action": tuple(metadata for _ in actions),
            "transaction_length": len(active_relative),
            "deployment_stage": stage,
            "value_field": value_field,
            "force_zero_posterior": bool(explicit_zero or legal_known < 2),
            "legal_known_value_count": int(legal_known),
        }

    def summary(self) -> dict[str, Any]:
        return {
            "schema": DATASET_SCHEMA,
            "rows": len(self),
            "source_weight_one": True,
            "failure_or_declared_uninformative_rows_retained": self.explicit_zero_rows,
            "candidate_k_histogram": dict(sorted(self.k_histogram.items())),
            "deployment_stage_counts": dict(sorted(self.stage_counts.items())),
            "species_program_source_counts": dict(
                sorted(self.program_source_counts.items())
            ),
            "llama_program_required": self.require_llama_program,
        }


def move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    return value


def transaction_value_loss(runtime: Any, batch: Mapping[str, Any]) -> dict[str, Any]:
    """Distill one finite-candidate transaction value into the policy."""

    _, _, legacy = _runtime_modules()
    if bool(batch["force_zero_posterior"]):
        policy_scores = legacy.sequential_action_scores(
            runtime, batch, reference=False
        )
        return {
            "loss": policy_scores.sum() * 0.0,
            "informative": False,
            "teacher_kl_nats": 0.0,
            "action_count": int(policy_scores.numel()),
        }
    reference_scores = legacy.sequential_action_scores(runtime, batch, reference=True)
    policy_scores = legacy.sequential_action_scores(runtime, batch, reference=False)
    posterior = build_potential_closure_posterior(
        reference_scores,
        batch["energies"],
        batch["legal_mask"],
        action_tokens=batch["action_tokens"].detach().cpu().tolist(),
        no_op_tokens=batch["no_op_tokens"],
        state_metadata_by_action=batch["state_metadata_by_action"],
        active_positions=batch["active_relative"],
        differing_positions_by_action=batch["differing_positions"],
        kl_budget_nats=MAX_KL_BUDGET_NATS,
    )
    if not posterior.informative:
        return {
            "loss": policy_scores.sum() * 0.0,
            "informative": False,
            "teacher_kl_nats": 0.0,
            "action_count": int(policy_scores.numel()),
        }
    output = potential_closure_loss(policy_scores, posterior)
    return {
        "loss": output.loss,
        "informative": True,
        "teacher_kl_nats": float(posterior.kl_nats),
        "action_count": int(posterior.action_count),
    }


def init_distributed() -> dict[str, Any]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != WORLD_SIZE:
        raise RuntimeError(f"formal training requires WORLD_SIZE={WORLD_SIZE}")
    if not torch.cuda.is_available():
        raise RuntimeError("formal training requires CUDA")
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    torch.cuda.set_device(local_rank)
    return {
        "rank": rank,
        "local_rank": local_rank,
        "world_size": dist.get_world_size(),
        "is_main": rank == 0,
        "device": torch.device("cuda", local_rank),
    }


def load_runtime(args: argparse.Namespace, *, is_main: bool) -> tuple[Any, Any, dict[str, Any]]:
    d3po, _, _ = _runtime_modules()
    loader_args = SimpleNamespace(
        model_path=Path(args.model_path),
        checkpoint_path=Path(args.checkpoint_path),
        data_dir=Path(args.sft_data_dir),
    )
    tokenizer, runtime, report = d3po.load_policy_and_reference_adapters(loader_args)
    if not is_main:
        report = {"loaded": True, "rank0_reported": True}
    return tokenizer, runtime, report


def load_clean_dataset(data_dir: Path, tokenizer: Any) -> tuple[Any, Any, dict[str, Any]]:
    _, _, legacy = _runtime_modules()
    dataset, collator, loss_config = legacy.load_clean_dataset(data_dir, tokenizer)
    if len(dataset) != EXPECTED_ROWS:
        raise ValueError("clean SPAD SFT must contain the full MP20 train split")
    for ordinal, row in enumerate(dataset.rows):
        source_idx = row.get("source_row_idx")
        if source_idx is not None and int(source_idx) != ordinal:
            raise ValueError("clean SFT source_row_idx order differs from full MP20")
    return dataset, collator, loss_config


def clean_batch_loss(
    runtime: Any,
    clean_dataset: Any,
    collator: Any,
    indices: Sequence[int],
    device: torch.device,
    loss_config: Mapping[str, Any],
) -> torch.Tensor:
    _, sft, _ = _runtime_modules()
    if len(indices) != LOCAL_BATCH_SIZE:
        raise ValueError("clean local batch must contain eight sources")
    runtime.activate_policy(trainable=True)
    batch = move_to_device(
        collator([clean_dataset[int(index)] for index in indices]), device
    )
    loss = sft.compute_loss(runtime.model, batch, dict(loss_config))
    if not bool(torch.isfinite(loss).item()):
        raise FloatingPointError("clean SPAD CE is nonfinite")
    return loss


def _gradient_norm(parameters: Sequence[torch.nn.Parameter]) -> float:
    total = torch.zeros((), dtype=torch.float64, device=parameters[0].device)
    for parameter in parameters:
        if parameter.grad is not None:
            total = total + torch.sum(parameter.grad.detach().double().square())
    return math.sqrt(float(total.detach().cpu()))


def _reduce_scalar(value: float, device: torch.device, *, op: Any = None) -> float:
    tensor = torch.tensor(float(value), dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM if op is None else op)
    return float(tensor.detach().cpu())


def step0_policy_reference_equality(
    runtime: Any,
    dataset: FullMP20TransactionValueDataset,
    device: torch.device,
    *,
    route: str,
) -> dict[str, Any]:
    _, _, legacy = _runtime_modules()
    batch = move_to_device(dataset.materialize(0, route=route), device)
    reference = legacy.sequential_action_scores(runtime, batch, reference=True)
    policy = legacy.sequential_action_scores(runtime, batch, reference=False)
    local_delta = float(torch.max(torch.abs(reference.detach() - policy.detach())).cpu())
    delta_tensor = torch.tensor(local_delta, dtype=torch.float64, device=device)
    dist.all_reduce(delta_tensor, op=dist.ReduceOp.MAX)
    global_delta = float(delta_tensor.detach().cpu())
    if global_delta > 1.0e-6:
        raise RuntimeError("step0 policy/reference transaction scores differ")
    return {"passed": True, "max_abs_score_delta": global_delta}


def save_endpoint(
    runtime: Any,
    tokenizer: Any,
    output_dir: Path,
    data_dir: Path,
) -> dict[str, Any]:
    d3po, sft, _ = _runtime_modules()
    runtime.activate_policy(trainable=False)
    if d3po.REFERENCE_ADAPTER not in runtime.model.peft_config:
        raise RuntimeError("reference adapter disappeared before endpoint save")
    runtime.model.delete_adapter(d3po.REFERENCE_ADAPTER)
    runtime.model.set_adapter(d3po.POLICY_ADAPTER)
    sft.save_checkpoint(
        runtime.model,
        tokenizer,
        output_dir,
        TOTAL_UPDATES,
        save_embedding_layers="auto",
        data_dir=data_dir,
        is_main=True,
    )
    checkpoints = output_dir / "checkpoints"
    directories = sorted(path.name for path in checkpoints.iterdir() if path.is_dir())
    if directories != [f"step-{TOTAL_UPDATES}"]:
        raise RuntimeError(f"unexpected checkpoint directories: {directories}")
    if (output_dir / "final").exists():
        raise RuntimeError("unexpected final checkpoint alias")
    checkpoint = checkpoints / f"step-{TOTAL_UPDATES}"
    adapter_configs = list(checkpoint.rglob("adapter_config.json"))
    adapter_models = list(checkpoint.rglob("adapter_model.safetensors"))
    if len(adapter_configs) != 1 or len(adapter_models) != 1:
        raise RuntimeError("endpoint must contain exactly one policy adapter")
    adapter_dir = adapter_configs[0].parent
    if adapter_models[0].parent != adapter_dir:
        raise RuntimeError("adapter config/model directories differ")
    tokenizer.save_pretrained(adapter_dir)
    return {
        "checkpoint_root": str(checkpoint.resolve()),
        "policy_adapter_path": str(adapter_dir.resolve()),
        "only_step3392": True,
        "reference_adapter_removed": True,
    }


def train(
    args: argparse.Namespace,
    dist_info: Mapping[str, Any],
    tokenizer: Any,
    runtime: Any,
    dataset: FullMP20TransactionValueDataset,
    clean_dataset: Any,
    clean_collator: Any,
    loss_config: Mapping[str, Any],
) -> dict[str, Any]:
    rank = int(dist_info["rank"])
    device = dist_info["device"]
    is_main = bool(dist_info["is_main"])
    permutation = frozen_source_permutation(EXPECTED_ROWS, seed=int(args.seed))
    parameters = tuple(
        parameter for parameter in runtime.policy_parameters if parameter.requires_grad
    )
    if not parameters:
        raise RuntimeError("policy adapter has no trainable parameters")
    optimizer = torch.optim.AdamW(parameters, lr=LEARNING_RATE, weight_decay=0.0)
    clean_seen = torch.zeros(EXPECTED_ROWS, dtype=torch.int16, device=device)
    posterior_seen = torch.zeros(EXPECTED_ROWS, dtype=torch.int16, device=device)
    objective_counts = Counter()
    informative_local = 0
    zero_local = 0
    finite_gradients = True
    started = time.time()
    log_path = Path(args.output_dir) / "training_log.jsonl"

    for batch_index in range(POSTERIOR_UPDATES):
        indices = rank_batch_indices(permutation, batch_index, rank)
        index_tensor = torch.tensor(indices, dtype=torch.long, device=device)

        clean_update = 2 * batch_index + 1
        if optimizer_objective(clean_update) != "clean_ce":
            raise RuntimeError("clean/posterior alternation changed")
        lr = learning_rate_for_update(clean_update)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        clean_loss = clean_batch_loss(
            runtime,
            clean_dataset,
            clean_collator,
            indices,
            device,
            loss_config,
        )
        clean_loss.backward()
        average_gradients_(parameters, world_size=WORLD_SIZE)
        clean_grad = _gradient_norm(parameters)
        if not math.isfinite(clean_grad) or clean_grad <= 0.0:
            finite_gradients = False
            raise FloatingPointError("clean gradient is nonfinite or zero")
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        clean_seen[index_tensor] += 1
        objective_counts["clean_ce"] += 1

        posterior_update = clean_update + 1
        if optimizer_objective(posterior_update) != "transaction_posterior":
            raise RuntimeError("clean/posterior alternation changed")
        lr = learning_rate_for_update(posterior_update)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        posterior_loss_sum = 0.0
        teacher_kl_max = 0.0
        local_informative_batch = 0
        local_zero_batch = 0
        for source_idx in indices:
            batch = move_to_device(
                dataset.materialize(source_idx, route=args.route), device
            )
            output = transaction_value_loss(runtime, batch)
            loss = output["loss"]
            if not isinstance(loss, torch.Tensor) or not bool(torch.isfinite(loss).item()):
                raise FloatingPointError("transaction posterior loss is nonfinite")
            (loss / float(LOCAL_BATCH_SIZE)).backward()
            posterior_loss_sum += float(loss.detach().cpu())
            teacher_kl_max = max(teacher_kl_max, float(output["teacher_kl_nats"]))
            local_informative_batch += int(output["informative"])
            local_zero_batch += int(not output["informative"])
        average_gradients_(parameters, world_size=WORLD_SIZE)
        posterior_grad = _gradient_norm(parameters)
        if not math.isfinite(posterior_grad):
            finite_gradients = False
            raise FloatingPointError("posterior gradient is nonfinite")
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        posterior_seen[index_tensor] += 1
        objective_counts["transaction_posterior"] += 1
        informative_local += local_informative_batch
        zero_local += local_zero_batch

        if is_main and (
            posterior_update == 2
            or batch_index % LOGGING_UPDATES == LOGGING_UPDATES - 1
            or posterior_update == TOTAL_UPDATES
        ):
            global_clean_loss = _reduce_scalar(
                float(clean_loss.detach().cpu()), device
            ) / WORLD_SIZE
            global_posterior_loss = _reduce_scalar(
                posterior_loss_sum, device
            ) / GLOBAL_BATCH_SIZE
            global_informative = int(_reduce_scalar(local_informative_batch, device))
            global_zero = int(_reduce_scalar(local_zero_batch, device))
            global_kl_max_tensor = torch.tensor(
                teacher_kl_max, dtype=torch.float64, device=device
            )
            dist.all_reduce(global_kl_max_tensor, op=dist.ReduceOp.MAX)
            append_jsonl(
                log_path,
                {
                    "event": "train",
                    "route": args.route,
                    "update": posterior_update,
                    "source_batch": batch_index + 1,
                    "clean_loss": global_clean_loss,
                    "posterior_loss": global_posterior_loss,
                    "clean_gradient_norm": clean_grad,
                    "posterior_gradient_norm": posterior_grad,
                    "informative_sources": global_informative,
                    "zero_posterior_sources": global_zero,
                    "max_teacher_kl_nats": float(global_kl_max_tensor.cpu()),
                    "learning_rate": lr,
                    "elapsed_sec": time.time() - started,
                },
            )
        elif not is_main and (
            posterior_update == 2
            or batch_index % LOGGING_UPDATES == LOGGING_UPDATES - 1
            or posterior_update == TOTAL_UPDATES
        ):
            # Match rank0's collective calls exactly.
            _reduce_scalar(float(clean_loss.detach().cpu()), device)
            _reduce_scalar(posterior_loss_sum, device)
            _reduce_scalar(local_informative_batch, device)
            _reduce_scalar(local_zero_batch, device)
            tensor = torch.tensor(teacher_kl_max, dtype=torch.float64, device=device)
            dist.all_reduce(tensor, op=dist.ReduceOp.MAX)

    dist.all_reduce(clean_seen, op=dist.ReduceOp.SUM)
    dist.all_reduce(posterior_seen, op=dist.ReduceOp.SUM)
    if not bool(torch.all(clean_seen == 1).item()):
        raise RuntimeError("clean CE did not cover every source exactly once")
    if not bool(torch.all(posterior_seen == 1).item()):
        raise RuntimeError("posterior did not cover every source exactly once")
    expected_objectives = Counter(
        {"clean_ce": CLEAN_CE_UPDATES, "transaction_posterior": POSTERIOR_UPDATES}
    )
    if objective_counts != expected_objectives:
        raise RuntimeError("objective update accounting changed")

    informative_global = int(_reduce_scalar(informative_local, device))
    zero_global = int(_reduce_scalar(zero_local, device))
    if informative_global + zero_global != EXPECTED_ROWS:
        raise RuntimeError("posterior source denominator changed")
    dist.barrier()
    checkpoint: dict[str, Any] | None = None
    if is_main:
        checkpoint = save_endpoint(
            runtime, tokenizer, Path(args.output_dir), Path(args.sft_data_dir)
        )
    dist.barrier()
    return {
        "schema": RUN_SCHEMA,
        "status": "success",
        "route": args.route,
        "value_field": route_value_field(args.route, args.fields),
        "seed": int(args.seed),
        "world_size": WORLD_SIZE,
        "local_microexamples": LOCAL_BATCH_SIZE,
        "global_batch_size": GLOBAL_BATCH_SIZE,
        "optimizer_updates": TOTAL_UPDATES,
        "objective_counts": dict(expected_objectives),
        "clean_source_coverage": EXPECTED_ROWS,
        "posterior_source_coverage": EXPECTED_ROWS,
        "informative_posterior_sources": informative_global,
        "zero_posterior_sources_retained": zero_global,
        "source_weight": 1.0,
        "finite_gradients": finite_gradients,
        "manual_gradient_allreduce_mean": True,
        "checkpoint": checkpoint,
        "elapsed_sec": time.time() - started,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--labelled-groups", type=Path, required=True)
    parser.add_argument("--sft-data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--route", choices=ROUTES, required=True)
    parser.add_argument("--seed", type=int, default=99_017)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--active-positions-absolute", action="store_true")
    parser.add_argument("--require-llama-program", action="store_true")
    parser.add_argument("--state-field", default=FieldConfig.state)
    parser.add_argument("--source-index-field", default=FieldConfig.source_index)
    parser.add_argument("--source-weight-field", default=FieldConfig.source_weight)
    parser.add_argument("--prompt-field", default=FieldConfig.prompt)
    parser.add_argument("--source-answer-field", default=FieldConfig.source_answer)
    parser.add_argument("--active-positions-field", default=FieldConfig.active_positions)
    parser.add_argument("--candidates-field", default=FieldConfig.candidates)
    parser.add_argument("--candidate-action-field", default=FieldConfig.candidate_action)
    parser.add_argument(
        "--candidate-legality-field", default=FieldConfig.candidate_legality
    )
    parser.add_argument(
        "--single-point-energy-field", default=FieldConfig.single_point_energy
    )
    parser.add_argument("--basin-energy-field", default=FieldConfig.basin_energy)
    parser.add_argument("--species-program-field", default=FieldConfig.species_program)
    parser.add_argument(
        "--deployment-stage-field", default=FieldConfig.deployment_stage
    )
    return parser


def _fields_from_args(args: argparse.Namespace) -> FieldConfig:
    return FieldConfig(
        state=str(args.state_field),
        source_index=str(args.source_index_field),
        source_weight=str(args.source_weight_field),
        prompt=str(args.prompt_field),
        source_answer=str(args.source_answer_field),
        active_positions=str(args.active_positions_field),
        candidates=str(args.candidates_field),
        candidate_action=str(args.candidate_action_field),
        candidate_legality=str(args.candidate_legality_field),
        single_point_energy=str(args.single_point_energy_field),
        basin_energy=str(args.basin_energy_field),
        species_program=str(args.species_program_field),
        deployment_stage=str(args.deployment_stage_field),
    )


def main() -> None:
    args = build_parser().parse_args()
    args.fields = _fields_from_args(args)
    dist_info = init_distributed()
    rank = int(dist_info["rank"])
    is_main = bool(dist_info["is_main"])
    device = dist_info["device"]
    output_dir = Path(args.output_dir).resolve()
    args.output_dir = output_dir
    try:
        if is_main:
            if output_dir.exists():
                raise FileExistsError(output_dir)
            output_dir.mkdir(parents=True, exist_ok=False)
        dist.barrier()
        random.seed(int(args.seed) + rank)
        torch.manual_seed(int(args.seed) + rank)
        torch.cuda.manual_seed_all(int(args.seed) + rank)
        tokenizer, runtime, adapter_report = load_runtime(args, is_main=is_main)
        runtime.model.to(device)
        dataset = FullMP20TransactionValueDataset(
            Path(args.labelled_groups),
            tokenizer,
            fields=args.fields,
            max_length=int(args.max_length),
            expected_rows=EXPECTED_ROWS,
            active_positions_absolute=bool(args.active_positions_absolute),
            require_llama_program=bool(args.require_llama_program),
        )
        clean_dataset, clean_collator, loss_config = load_clean_dataset(
            Path(args.sft_data_dir), tokenizer
        )
        dataset_identity = {
            "labelled_groups_sha256": sha256_file(Path(args.labelled_groups)),
            "clean_train_sha256": sha256_file(Path(args.sft_data_dir) / "train.jsonl"),
            "rows": EXPECTED_ROWS,
            "source_shuffle_seed": int(args.seed),
            "a_b_shared_dataset_identity": True,
        }
        step0 = step0_policy_reference_equality(
            runtime, dataset, device, route=args.route
        )
        config = {
            "schema": RUN_SCHEMA,
            "route": args.route,
            "value_field": route_value_field(args.route, args.fields),
            "other_route_value_field_not_used": route_value_field(
                ROUTES[1 - ROUTES.index(args.route)], args.fields
            ),
            "seed": int(args.seed),
            "model_path": str(Path(args.model_path).resolve()),
            "checkpoint_path": str(Path(args.checkpoint_path).resolve()),
            "sft_data_dir": str(Path(args.sft_data_dir).resolve()),
            "labelled_groups": str(Path(args.labelled_groups).resolve()),
            "field_config": asdict(args.fields),
            "dataset_identity": dataset_identity,
            "dataset_summary": dataset.summary(),
            "llama_controls_dlm_transaction_order": bool(
                args.require_llama_program
            ),
            "deployment_stage_order": list(DEPLOYMENT_STAGES),
            "step0_policy_reference_equality": step0,
            "adapter_report": adapter_report,
            "world_size": WORLD_SIZE,
            "local_microexamples": LOCAL_BATCH_SIZE,
            "global_batch_size": GLOBAL_BATCH_SIZE,
            "updates": TOTAL_UPDATES,
            "objective_updates": {
                "clean_ce": CLEAN_CE_UPDATES,
                "transaction_posterior": POSTERIOR_UPDATES,
            },
            "clean_ce_full_mp20_epochs": 1,
            "posterior_full_mp20_epochs": 1,
            "interleaving": ["clean_ce", "transaction_posterior"],
            "learning_rate": LEARNING_RATE,
            "warmup_updates": WARMUP_UPDATES,
            "kl_budget_nats": MAX_KL_BUDGET_NATS,
            "checkpoint_steps": list(checkpoint_steps()),
            "on_policy_or_failure_rows_filtered": False,
            "automatic_tuning": False,
        }
        if is_main:
            write_json(output_dir / "RUN_CONFIG.json", config)
            append_jsonl(output_dir / "training_log.jsonl", {"event": "start", **config})
        report = train(
            args,
            dist_info,
            tokenizer,
            runtime,
            dataset,
            clean_dataset,
            clean_collator,
            loss_config,
        )
        if is_main:
            report["dataset_identity"] = dataset_identity
            report["step0_policy_reference_equality"] = step0
            write_json(output_dir / "TRAIN_FINAL.json", report)
            append_jsonl(output_dir / "training_log.jsonl", {"event": "success", **report})
            (output_dir / "_SUCCESS").touch()
            print(json.dumps(report, ensure_ascii=False, sort_keys=True))
        dist.barrier()
    except Exception as exc:
        if is_main and output_dir.exists():
            write_json(
                output_dir / "_FAILED.json",
                {
                    "schema": RUN_SCHEMA,
                    "status": "failed",
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
