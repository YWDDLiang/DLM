#!/usr/bin/env python3
"""Low-resource shared-noise masked-D3PO trainer for dynamic crystal bodies.

This is intentionally a dedicated trainer.  Historical ``llada_sft.py`` is
used only for model/tokenizer loading, the LR scheduler, and checkpoint I/O;
its SFT loss and corruption path are never called here.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass
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
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from crystal_dlm.d3po import (
    D3POLossOutput,
    SharedGeometryCorruption,
    d3po_pair_loss,
    legal_target_log_probs,
    masked_sequence_log_ratio,
    shared_geometry_corruption,
    winner_denoising_anchor,
)
from crystal_dlm.dynamic_crystal import parse_dynamic_answer
from crystal_dlm.fixed_slot import MASK_TOKEN_ID
from crystal_dlm.r5_dynamic_length import exact_dynamic_schema_constraints
from scripts.llada_sft import (
    build_lr_scheduler,
    load_tokenizer_and_model,
    save_checkpoint,
)


PAIR_SCHEMA = "h1a2_shared_noise_soft_d3po_pair_v1"
PAIR_MANIFEST_SCHEMA = "h1a2_shared_noise_soft_d3po_pair_manifest_v1"
TRAIN_MANIFEST_SCHEMA = "h1a2_shared_noise_soft_d3po_train_manifest_v1"
POLICY_ADAPTER = "policy"
REFERENCE_ADAPTER = "reference"
ALLOWED_TRAINING_SEEDS = (81017, 81018)
BASE_ADAPTER_CONFIG_SHA256 = (
    "8101ee2a917dd1b08d5ef5d90472207a01161a6bcd2b03c78f9e037e756e6300"
)
BASE_ADAPTER_MODEL_SHA256 = (
    "6ea3c2a633706968e4b3e3cf77e98e46399c23e1568333522283472634553ecb"
)
PAIR_MANIFEST_SHA256 = (
    "90028189c1f3631e6d86a875713f037aa556df583824ff127ffd630050e751b1"
)
PAIR_DATA_SHA256 = {
    "train": "103f672bac29d913141f3e927efa3abb70767be79fbace87e0a31c14dcce4320",
    "validation": "a175d68907f4d9450478ae1fcce6ce0b822017f71806855a9615a36613b6d05e",
}

# Frozen D3PO-256-Min optimization contract.
TOTAL_UPDATES = 348
MICROBATCH_PAIRS = 1
GRADIENT_ACCUMULATION = 16
BETA = 0.1
LEARNING_RATE = 5e-6
LORA_RANK = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.0
WINNER_ANCHOR_WEIGHT = 0.2
ENERGY_TEMPERATURE = 0.03
MAX_SEQUENCE_LENGTH = 382
LOGGING_STEPS = 10

MINIMAL_PROMPT_SUFFIX = "\ndynamic_crystal_body:"
MINIMAL_PROMPT_KEYS = frozenset(
    {"N", "charge", "counts", "elements", "family", "formula"}
)
ALLOWED_CHARGE_CERTIFICATES = frozenset({"all_metal", "certified_neutral"})
ALLOWED_FAMILIES = frozenset(
    {
        "oxide",
        "halide",
        "sulfide",
        "chalcogenide",
        "nitride",
        "phosphide_or_phosphate",
        "other",
    }
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(canonical_json(dict(payload)) + "\n")
        handle.flush()


def _formula(elements: Sequence[str], counts: Sequence[int]) -> str:
    return "".join(
        element if count == 1 else f"{element}{count}"
        for element, count in zip(elements, counts)
    )


def _composition_identity(elements: Sequence[str], counts: Sequence[int]) -> str:
    return "|".join(
        f"{element}:{count}" for element, count in zip(elements, counts)
    )


def parse_canonical_minimal_prompt(prompt: str) -> dict[str, Any]:
    """Parse and fail closed on the frozen C3FD minimal typed condition."""

    if not isinstance(prompt, str) or not prompt.endswith(MINIMAL_PROMPT_SUFFIX):
        raise ValueError("prompt must end with the canonical dynamic body suffix")
    payload_text = prompt[: -len(MINIMAL_PROMPT_SUFFIX)]
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError as exc:
        raise ValueError("prompt does not contain canonical minimal JSON") from exc
    if not isinstance(payload, dict) or set(payload) != MINIMAL_PROMPT_KEYS:
        raise ValueError("prompt keys differ from the frozen minimal C3FD schema")
    if canonical_json(payload) != payload_text:
        raise ValueError("prompt JSON is not canonical")

    elements = payload.get("elements")
    counts = payload.get("counts")
    if not isinstance(elements, list) or not isinstance(counts, list):
        raise ValueError("prompt elements/counts must be lists")
    if not elements or len(elements) != len(counts):
        raise ValueError("prompt elements/counts are malformed")
    normalized_elements = [str(value) for value in elements]
    if normalized_elements != sorted(normalized_elements):
        raise ValueError("prompt elements must use canonical sorted order")
    if len(set(normalized_elements)) != len(normalized_elements):
        raise ValueError("prompt elements must be unique")
    try:
        normalized_counts = [int(value) for value in counts]
        num_atoms = int(payload["N"])
    except (TypeError, ValueError) as exc:
        raise ValueError("prompt N/counts must be integers") from exc
    if any(count <= 0 for count in normalized_counts):
        raise ValueError("prompt counts must be positive")
    if not 1 <= num_atoms <= 20 or sum(normalized_counts) != num_atoms:
        raise ValueError("prompt N/count conservation failed")
    if str(payload["formula"]) != _formula(normalized_elements, normalized_counts):
        raise ValueError("prompt formula disagrees with elements/counts")
    if str(payload["charge"]) not in ALLOWED_CHARGE_CERTIFICATES:
        raise ValueError("prompt charge certificate is not authorized")
    if str(payload["family"]) not in ALLOWED_FAMILIES:
        raise ValueError("prompt family is not authorized")

    return {
        **payload,
        "N": num_atoms,
        "elements": normalized_elements,
        "counts": normalized_counts,
    }


def validate_canonical_dynamic_answer(
    answer: str,
    condition: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate compact canonical dynamic ``7 + 4N`` and exact composition."""

    if not isinstance(answer, str) or not answer:
        raise ValueError("dynamic answer must be a non-empty string")
    if answer != answer.strip():
        raise ValueError("dynamic answer may not contain surrounding whitespace")
    parsed = parse_dynamic_answer(answer, strict=True)
    if str(parsed["answer"]) != answer:
        raise ValueError("dynamic answer is not the canonical compact token form")
    num_atoms = int(condition["N"])
    expected_length = 7 + 4 * num_atoms
    if int(parsed["num_atoms"]) != num_atoms:
        raise ValueError("dynamic answer N differs from the prompt")
    if len(parsed["tokens"]) != expected_length:
        raise ValueError("dynamic answer does not have exact 7+4N length")
    expected_counts = dict(zip(condition["elements"], condition["counts"]))
    observed_counts = Counter(str(symbol) for symbol in parsed["species"])
    if dict(observed_counts) != expected_counts:
        raise ValueError("dynamic answer composition differs from the prompt")
    return parsed


def validate_pair_row(
    raw_row: Mapping[str, Any],
    *,
    expected_split: str,
) -> dict[str, Any]:
    """Validate a frozen pair row without deriving training targets anew."""

    row = dict(raw_row)
    if row.get("schema") != PAIR_SCHEMA:
        raise ValueError("pair row schema changed")
    if str(row.get("split")) != expected_split:
        raise ValueError("pair row split disagrees with its file")
    pair_id = str(row.get("pair_id") or "")
    composition_id = str(row.get("composition_id") or "")
    if not pair_id or not composition_id:
        raise ValueError("pair_id and composition_id are required")

    condition = parse_canonical_minimal_prompt(str(row.get("prompt") or ""))
    expected_identity = _composition_identity(
        condition["elements"], condition["counts"]
    )
    if composition_id != expected_identity:
        raise ValueError("composition_id disagrees with the canonical prompt")
    if int(row.get("N") or 0) != int(condition["N"]):
        raise ValueError("pair N disagrees with the canonical prompt")
    expected_chemsys = "-".join(condition["elements"])
    if str(row.get("chemsys") or "") != expected_chemsys:
        raise ValueError("pair chemsys disagrees with the canonical prompt")

    winner_answer = str(row.get("winner_answer") or "")
    loser_answer = str(row.get("loser_answer") or "")
    validate_canonical_dynamic_answer(winner_answer, condition)
    validate_canonical_dynamic_answer(loser_answer, condition)
    if winner_answer == loser_answer:
        raise ValueError("winner and loser bodies must differ")

    numeric_fields = (
        "winner_energy_per_atom",
        "loser_energy_per_atom",
        "energy_gap_eV_per_atom",
        "soft_target",
        "pair_weight",
    )
    values: dict[str, float] = {}
    for name in numeric_fields:
        try:
            values[name] = float(row[name])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"pair field {name} must be numeric") from exc
        if not math.isfinite(values[name]):
            raise ValueError(f"pair field {name} must be finite")
    winner_energy = values["winner_energy_per_atom"]
    loser_energy = values["loser_energy_per_atom"]
    gap = values["energy_gap_eV_per_atom"]
    if not winner_energy < loser_energy:
        raise ValueError("pair winner must have lower post-refiner energy")
    if gap <= 0.0 or not math.isclose(
        gap,
        loser_energy - winner_energy,
        rel_tol=1e-8,
        abs_tol=1e-10,
    ):
        raise ValueError("pair energy gap is inconsistent")
    expected_soft_target = 1.0 / (1.0 + math.exp(-gap / ENERGY_TEMPERATURE))
    if not math.isclose(
        values["soft_target"],
        expected_soft_target,
        rel_tol=1e-8,
        abs_tol=1e-10,
    ):
        raise ValueError("soft_target differs from the frozen pair data")
    if not 0.5 < values["soft_target"] <= 1.0:
        raise ValueError("soft_target must express a strict winner preference")
    if values["pair_weight"] <= 0.0:
        raise ValueError("pair_weight must be positive")

    return {
        **row,
        **values,
        "pair_id": pair_id,
        "composition_id": composition_id,
        "chemsys": expected_chemsys,
        "N": int(condition["N"]),
        "prompt": str(row["prompt"]),
        "winner_answer": winner_answer,
        "loser_answer": loser_answer,
        "condition": condition,
    }


class D3POPairDataset(Dataset):
    """Eagerly validated/tokenized JSONL preference pairs."""

    def __init__(
        self,
        path: Path,
        tokenizer: Any,
        *,
        expected_split: str,
        max_length: int = MAX_SEQUENCE_LENGTH,
    ) -> None:
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.expected_split = str(expected_split)
        self.max_length = int(max_length)
        self._legal_supports_by_n: dict[int, tuple[frozenset[int], ...]] = {}
        if self.expected_split not in {"train", "validation"}:
            raise ValueError("expected_split must be train or validation")
        if not self.path.is_file():
            raise FileNotFoundError(self.path)

        rows: list[dict[str, Any]] = []
        seen_pair_ids: set[str] = set()
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                    row = validate_pair_row(raw, expected_split=self.expected_split)
                except Exception as exc:
                    raise ValueError(
                        f"invalid D3PO row {self.path}:{line_number}: {exc}"
                    ) from exc
                if row["pair_id"] in seen_pair_ids:
                    raise ValueError(f"duplicate pair_id {row['pair_id']}")
                seen_pair_ids.add(row["pair_id"])
                rows.append(row)
        if not rows:
            raise ValueError(f"D3PO split is empty: {self.path}")

        weight_totals: dict[str, float] = {}
        for row in rows:
            identity = str(row["composition_id"])
            weight_totals[identity] = weight_totals.get(identity, 0.0) + float(
                row["pair_weight"]
            )
        bad_totals = {
            identity: total
            for identity, total in weight_totals.items()
            if not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-6)
        }
        if bad_totals:
            first = next(iter(sorted(bad_totals.items())))
            raise ValueError(
                f"composition pair weights must sum to one; first mismatch={first}"
            )

        self.rows = rows
        self.items = [self._tokenize_row(row) for row in rows]
        self.pair_weights = [float(row["pair_weight"]) for row in rows]
        self.composition_ids = {str(row["composition_id"]) for row in rows}
        self.chemsys = {str(row["chemsys"]) for row in rows}
        self.composition_count = len(self.composition_ids)

    def _tokenize_row(self, row: Mapping[str, Any]) -> dict[str, Any]:
        prompt_text = str(row["prompt"]).rstrip() + "\n"
        prompt_ids = list(
            self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        )
        if not prompt_ids:
            raise ValueError(f"pair {row['pair_id']} has an empty tokenized prompt")
        answer_ids: dict[str, list[int]] = {}
        full_ids: dict[str, list[int]] = {}
        num_atoms = int(row["N"])
        if num_atoms not in self._legal_supports_by_n:
            self._legal_supports_by_n[num_atoms] = tuple(
                frozenset(int(value) for value in support)
                for support in exact_dynamic_schema_constraints(
                    self.tokenizer, num_atoms
                )
            )
        legal_supports = self._legal_supports_by_n[num_atoms]
        expected_body_length = 7 + 4 * num_atoms
        for side in ("winner", "loser"):
            answer = str(row[f"{side}_answer"])
            encoded_answer = list(
                self.tokenizer(answer, add_special_tokens=False)["input_ids"]
            )
            encoded_full = list(
                self.tokenizer(
                    prompt_text + answer,
                    add_special_tokens=False,
                )["input_ids"]
            )
            if len(encoded_answer) != expected_body_length:
                raise ValueError(
                    f"pair {row['pair_id']} {side} is not one tokenizer token per 7+4N field"
                )
            if encoded_full[-expected_body_length:] != encoded_answer:
                raise ValueError(
                    f"pair {row['pair_id']} {side} answer is not an exact token suffix"
                )
            if len(encoded_full) != len(prompt_ids) + expected_body_length:
                raise ValueError(
                    f"pair {row['pair_id']} {side} prompt/body token boundary changed"
                )
            if len(encoded_full) > self.max_length:
                raise ValueError(
                    f"pair {row['pair_id']} exceeds max_length={self.max_length}"
                )
            for position, (token_id, support) in enumerate(
                zip(encoded_answer, legal_supports)
            ):
                if int(token_id) not in support:
                    raise ValueError(
                        f"pair {row['pair_id']} {side} token at body position {position} "
                        "is outside its position-specific legal support"
                    )
            answer_ids[side] = encoded_answer
            full_ids[side] = encoded_full
        if len(full_ids["winner"]) != len(full_ids["loser"]):
            raise ValueError("winner/loser tokenized lengths differ")

        return {
            "winner_input_ids": torch.tensor(full_ids["winner"], dtype=torch.long),
            "loser_input_ids": torch.tensor(full_ids["loser"], dtype=torch.long),
            "prompt_length": len(prompt_ids),
            "num_atoms": int(row["N"]),
            "soft_target": float(row["soft_target"]),
            "pair_weight": float(row["pair_weight"]),
            "pair_id": str(row["pair_id"]),
            "composition_id": str(row["composition_id"]),
            "chemsys": str(row["chemsys"]),
        }

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.items[index]


class D3POPairCollator:
    """Pad winner/loser identically while preserving pair metadata."""

    def __init__(self, tokenizer: Any) -> None:
        if tokenizer.pad_token_id is None:
            raise ValueError("tokenizer must define pad_token_id")
        self.pad_token_id = int(tokenizer.pad_token_id)

    def __call__(self, batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if not batch:
            raise ValueError("D3PO collator received an empty batch")
        max_length = max(int(item["winner_input_ids"].shape[0]) for item in batch)
        winner = torch.full(
            (len(batch), max_length), self.pad_token_id, dtype=torch.long
        )
        loser = torch.full_like(winner, self.pad_token_id)
        attention = torch.zeros((len(batch), max_length), dtype=torch.long)
        prompt_lengths = torch.zeros((len(batch),), dtype=torch.long)
        num_atoms = torch.zeros((len(batch),), dtype=torch.long)
        soft_targets = torch.zeros((len(batch),), dtype=torch.float32)
        pair_weights = torch.zeros((len(batch),), dtype=torch.float32)
        pair_ids: list[str] = []
        composition_ids: list[str] = []
        chemsys: list[str] = []
        for index, item in enumerate(batch):
            winner_ids = item["winner_input_ids"]
            loser_ids = item["loser_input_ids"]
            if winner_ids.shape != loser_ids.shape:
                raise ValueError("winner/loser shapes differ inside a pair")
            length = int(winner_ids.shape[0])
            winner[index, :length] = winner_ids
            loser[index, :length] = loser_ids
            attention[index, :length] = 1
            prompt_lengths[index] = int(item["prompt_length"])
            num_atoms[index] = int(item["num_atoms"])
            soft_targets[index] = float(item["soft_target"])
            pair_weights[index] = float(item["pair_weight"])
            pair_ids.append(str(item["pair_id"]))
            composition_ids.append(str(item["composition_id"]))
            chemsys.append(str(item["chemsys"]))
        return {
            "winner_input_ids": winner,
            "loser_input_ids": loser,
            "attention_mask": attention,
            "prompt_lengths": prompt_lengths,
            "num_atoms": num_atoms,
            "soft_target": soft_targets,
            "pair_weight": pair_weights,
            "pair_ids": pair_ids,
            "composition_ids": composition_ids,
            "chemsys": chemsys,
        }


class DynamicLegalSupportCache:
    """Cache position-specific dynamic schema supports by atom count."""

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer
        self._by_n: dict[int, tuple[tuple[int, ...], ...]] = {}

    def schema(self, num_atoms: int) -> tuple[tuple[int, ...], ...]:
        atoms = int(num_atoms)
        if atoms not in self._by_n:
            supports = exact_dynamic_schema_constraints(self.tokenizer, atoms)
            self._by_n[atoms] = tuple(
                tuple(int(token_id) for token_id in support)
                for support in supports
            )
        return self._by_n[atoms]

    def selected(
        self,
        *,
        num_atoms: torch.Tensor,
        prompt_lengths: torch.Tensor,
        masked_positions: torch.Tensor,
    ) -> list[tuple[int, ...]]:
        selected: list[tuple[int, ...]] = []
        atoms_cpu = num_atoms.detach().cpu().tolist()
        prompts_cpu = prompt_lengths.detach().cpu().tolist()
        positions = torch.nonzero(masked_positions, as_tuple=False).detach().cpu()
        for row_value, absolute_value in positions.tolist():
            row = int(row_value)
            relative = int(absolute_value) - int(prompts_cpu[row])
            schema = self.schema(int(atoms_cpu[row]))
            if not 0 <= relative < len(schema):
                raise ValueError("masked position falls outside the dynamic body")
            selected.append(schema[relative])
        return selected


def build_llada_loader_args(args: argparse.Namespace) -> SimpleNamespace:
    """Create only the fields consumed by the frozen SFT loading helper."""

    return SimpleNamespace(
        model_path=str(args.model_path),
        checkpoint_path=Path(args.checkpoint_path),
        data_dir=Path(args.data_dir),
        representation="dynamic_v1",
        skip_data_vocab_resize=True,
        semantic_init_element_tokens=False,
        use_lora=False,
        lora_rank=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        lora_target_modules="q_proj,k_proj,v_proj,ff_proj,up_proj",
        modules_to_save="model.transformer.wte,model.transformer.ff_out",
    )


def _adapter_config_value_is(config_value: Any, expected: int) -> bool:
    if isinstance(config_value, Mapping):
        return bool(config_value) and all(
            int(value) == int(expected) for value in config_value.values()
        )
    return int(config_value) == int(expected)


def _parameter_belongs_to_adapter(parameter_name: str, adapter_name: str) -> bool:
    pieces = parameter_name.replace("[", ".").replace("]", ".").split(".")
    return adapter_name in pieces


def _is_policy_lora_parameter(parameter_name: str) -> bool:
    if not _parameter_belongs_to_adapter(parameter_name, POLICY_ADAPTER):
        return False
    pieces = set(parameter_name.replace("[", ".").replace("]", ".").split("."))
    return bool(
        pieces
        & {
            "lora_A",
            "lora_B",
            "lora_embedding_A",
            "lora_embedding_B",
        }
    )


@dataclass
class AdapterRuntime:
    """One backbone with sequentially activated policy/reference adapters."""

    model: Any
    policy_parameters: tuple[torch.nn.Parameter, ...]
    reference_parameters: tuple[torch.nn.Parameter, ...]
    frozen_parameters: tuple[torch.nn.Parameter, ...]

    def activate_reference(self) -> None:
        self.model.set_adapter(REFERENCE_ADAPTER)
        for parameter in self.frozen_parameters:
            parameter.requires_grad_(False)
        for parameter in self.policy_parameters:
            parameter.requires_grad_(False)

    def activate_policy(self, *, trainable: bool) -> None:
        self.model.set_adapter(POLICY_ADAPTER)
        for parameter in self.frozen_parameters:
            parameter.requires_grad_(False)
        for parameter in self.policy_parameters:
            parameter.requires_grad_(bool(trainable))


def _force_zero_dropout(model: Any) -> int:
    changed = 0
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout) and float(module.p) != 0.0:
            module.p = 0.0
            changed += 1
    for adapter_name in (POLICY_ADAPTER, REFERENCE_ADAPTER):
        config = model.peft_config[adapter_name]
        if hasattr(config, "lora_dropout"):
            config.lora_dropout = 0.0
    return changed


def load_policy_and_reference_adapters(
    args: argparse.Namespace,
) -> tuple[Any, AdapterRuntime, dict[str, Any]]:
    """Load one backbone and two named copies of the same step696 adapter."""

    helper_args = build_llada_loader_args(args)
    (
        tokenizer,
        model,
        num_new_tokens,
        tokenizer_source,
        model_source,
        semantic_report,
    ) = load_tokenizer_and_model(helper_args, is_main=True)
    if num_new_tokens != 0:
        raise RuntimeError("D3PO may not resize the frozen crystal vocabulary")
    for method in ("load_adapter", "set_adapter", "delete_adapter"):
        if not hasattr(model, method):
            raise RuntimeError(f"loaded model lacks PEFT method {method}")
    if "default" not in getattr(model, "peft_config", {}):
        raise RuntimeError("SFT helper did not load the starting adapter")

    checkpoint = str(Path(args.checkpoint_path))
    model.load_adapter(checkpoint, adapter_name=POLICY_ADAPTER, is_trainable=True)
    model.load_adapter(
        checkpoint,
        adapter_name=REFERENCE_ADAPTER,
        is_trainable=False,
    )
    model.set_adapter(POLICY_ADAPTER)
    model.delete_adapter("default")
    if set(model.peft_config) != {POLICY_ADAPTER, REFERENCE_ADAPTER}:
        raise RuntimeError("policy/reference adapter set is not exact")

    for adapter_name in (POLICY_ADAPTER, REFERENCE_ADAPTER):
        config = model.peft_config[adapter_name]
        if not _adapter_config_value_is(getattr(config, "r"), LORA_RANK):
            raise RuntimeError(f"{adapter_name} adapter is not rank {LORA_RANK}")
        if not _adapter_config_value_is(
            getattr(config, "lora_alpha"), LORA_ALPHA
        ):
            raise RuntimeError(
                f"{adapter_name} adapter does not use alpha {LORA_ALPHA}"
            )
        if hasattr(config, "inference_mode"):
            config.inference_mode = adapter_name == REFERENCE_ADAPTER

    dropout_modules_changed = _force_zero_dropout(model)
    named_parameters = list(model.named_parameters())
    all_policy_adapter_parameters = tuple(
        parameter
        for name, parameter in named_parameters
        if _parameter_belongs_to_adapter(name, POLICY_ADAPTER)
    )
    policy_parameters = tuple(
        parameter
        for name, parameter in named_parameters
        if _is_policy_lora_parameter(name)
    )
    reference_parameters = tuple(
        parameter
        for name, parameter in named_parameters
        if _parameter_belongs_to_adapter(name, REFERENCE_ADAPTER)
    )
    if (
        not all_policy_adapter_parameters
        or not policy_parameters
        or not reference_parameters
    ):
        raise RuntimeError("named policy/reference adapter parameters were not found")
    policy_parameter_ids = {id(parameter) for parameter in policy_parameters}
    frozen_parameters = tuple(
        parameter
        for _name, parameter in named_parameters
        if id(parameter) not in policy_parameter_ids
    )

    runtime = AdapterRuntime(
        model=model,
        policy_parameters=policy_parameters,
        reference_parameters=reference_parameters,
        frozen_parameters=frozen_parameters,
    )
    runtime.activate_policy(trainable=True)
    unexpected_trainable = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and not _parameter_belongs_to_adapter(name, POLICY_ADAPTER)
    ]
    if unexpected_trainable:
        raise RuntimeError(
            "non-policy parameters are trainable: " + ",".join(unexpected_trainable[:5])
        )
    if not any(parameter.requires_grad for parameter in policy_parameters):
        raise RuntimeError("policy adapter has no trainable parameters")

    model.config.use_cache = False
    gradient_checkpointing = False
    if bool(getattr(model, "supports_gradient_checkpointing", False)):
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        gradient_checkpointing = True
    model.train()

    report = {
        "single_backbone_object_id": id(model),
        "adapters": [POLICY_ADAPTER, REFERENCE_ADAPTER],
        "starting_adapter": str(Path(args.checkpoint_path).resolve()),
        "tokenizer_source": tokenizer_source,
        "model_source": model_source,
        "num_new_tokens": num_new_tokens,
        "semantic_report": semantic_report,
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "dropout_modules_zeroed": dropout_modules_changed,
        "policy_adapter_parameter_count": sum(
            int(parameter.numel()) for parameter in all_policy_adapter_parameters
        ),
        "policy_trainable_lora_parameter_count": sum(
            int(parameter.numel()) for parameter in policy_parameters
        ),
        "policy_frozen_modules_to_save_parameter_count": sum(
            int(parameter.numel())
            for parameter in all_policy_adapter_parameters
            if id(parameter) not in policy_parameter_ids
        ),
        "reference_parameter_count": sum(
            int(parameter.numel()) for parameter in reference_parameters
        ),
        "gradient_checkpointing": gradient_checkpointing,
    }
    return tokenizer, runtime, report


def _model_logits(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = getattr(outputs, "logits", None)
    if logits is None or logits.ndim != 3:
        raise RuntimeError("model forward did not return rank-3 logits")
    return logits


def _autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


@dataclass
class PairLossComputation:
    output: D3POLossOutput
    corruption: SharedGeometryCorruption
    winner_score: torch.Tensor
    loser_score: torch.Tensor
    max_policy_reference_log_prob_delta: float


def compute_pair_loss(
    runtime: AdapterRuntime,
    batch: Mapping[str, Any],
    legal_support_cache: DynamicLegalSupportCache,
    *,
    generator: torch.Generator | None,
    require_grad: bool,
    corruption: SharedGeometryCorruption | None = None,
) -> PairLossComputation:
    """Reference forward first, then policy forward on one shared corruption."""

    winner_ids = batch["winner_input_ids"]
    loser_ids = batch["loser_input_ids"]
    attention_mask = batch["attention_mask"]
    if corruption is None:
        corruption = shared_geometry_corruption(
            winner_ids,
            loser_ids,
            batch["prompt_lengths"],
            batch["num_atoms"],
            attention_mask=attention_mask,
            generator=generator,
        )
    selected_supports = legal_support_cache.selected(
        num_atoms=batch["num_atoms"],
        prompt_lengths=batch["prompt_lengths"],
        masked_positions=corruption.masked_positions,
    )
    pair_input_ids = torch.cat(
        [corruption.winner_noisy_ids, corruption.loser_noisy_ids], dim=0
    )
    pair_attention = torch.cat([attention_mask, attention_mask], dim=0)
    pair_count = winner_ids.shape[0]

    runtime.activate_reference()
    with torch.no_grad(), _autocast_context(winner_ids.device):
        reference_logits = _model_logits(
            runtime.model, pair_input_ids, pair_attention
        )
    reference_winner = legal_target_log_probs(
        reference_logits[:pair_count][corruption.masked_positions],
        winner_ids[corruption.masked_positions],
        selected_supports,
    )
    reference_loser = legal_target_log_probs(
        reference_logits[pair_count:][corruption.masked_positions],
        loser_ids[corruption.masked_positions],
        selected_supports,
    )
    del reference_logits

    runtime.activate_policy(trainable=require_grad)
    policy_context = nullcontext() if require_grad else torch.no_grad()
    with policy_context, _autocast_context(winner_ids.device):
        policy_logits = _model_logits(runtime.model, pair_input_ids, pair_attention)
    policy_zero_gradient_anchor = policy_logits.reshape(-1)[0].to(torch.float32) * 0.0
    policy_winner = legal_target_log_probs(
        policy_logits[:pair_count][corruption.masked_positions],
        winner_ids[corruption.masked_positions],
        selected_supports,
    )
    policy_loser = legal_target_log_probs(
        policy_logits[pair_count:][corruption.masked_positions],
        loser_ids[corruption.masked_positions],
        selected_supports,
    )
    del policy_logits

    for name, values in (
        ("reference_winner", reference_winner),
        ("reference_loser", reference_loser),
        ("policy_winner", policy_winner),
        ("policy_loser", policy_loser),
    ):
        if not bool(torch.isfinite(values).all().item()):
            raise FloatingPointError(f"{name} contains NaN/Inf")

    winner_score = masked_sequence_log_ratio(
        policy_winner,
        reference_winner,
        corruption.masked_positions,
        corruption.p_mask,
        geometry_mask=corruption.geometry_mask,
    )
    loser_score = masked_sequence_log_ratio(
        policy_loser,
        reference_loser,
        corruption.masked_positions,
        corruption.p_mask,
        geometry_mask=corruption.geometry_mask,
    )
    anchor = winner_denoising_anchor(
        policy_winner,
        corruption.masked_positions,
        corruption.p_mask,
        corruption.geometry_mask,
    )
    output = d3po_pair_loss(
        winner_score,
        loser_score,
        target_probabilities=batch["soft_target"],
        winner_denoising_losses=anchor,
        pair_weights=batch["pair_weight"],
        beta=BETA,
        energy_temperature=ENERGY_TEMPERATURE,
        winner_anchor_weight=WINNER_ANCHOR_WEIGHT,
    )
    if require_grad:
        # The pure masking-state core intentionally returns a disconnected zero
        # for an empty sampled mask.  Keep that valid event differentiable with
        # an exactly-zero policy path so backward produces zero gradients.
        output = D3POLossOutput(
            loss=output.loss + policy_zero_gradient_anchor,
            preference_loss=output.preference_loss,
            winner_anchor_loss=output.winner_anchor_loss,
            margin=output.margin,
            target_probability=output.target_probability,
            per_pair_preference_loss=output.per_pair_preference_loss,
        )
    if not bool(torch.isfinite(output.loss).item()):
        raise FloatingPointError("D3PO objective is NaN/Inf")
    if policy_winner.numel() == 0:
        max_delta = 0.0
    else:
        max_delta = max(
            float((policy_winner - reference_winner).abs().max().detach().cpu()),
            float((policy_loser - reference_loser).abs().max().detach().cpu()),
        )
    return PairLossComputation(
        output=output,
        corruption=corruption,
        winner_score=winner_score,
        loser_score=loser_score,
        max_policy_reference_log_prob_delta=max_delta,
    )


def move_batch_to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def run_step0_canary(
    runtime: AdapterRuntime,
    batch: Mapping[str, Any],
    legal_support_cache: DynamicLegalSupportCache,
) -> dict[str, Any]:
    """Fail closed on adapter equality and the frozen shared-mask invariants."""

    p_mask = torch.ones(
        (batch["winner_input_ids"].shape[0],),
        dtype=torch.float32,
        device=batch["winner_input_ids"].device,
    )
    corruption = shared_geometry_corruption(
        batch["winner_input_ids"],
        batch["loser_input_ids"],
        batch["prompt_lengths"],
        batch["num_atoms"],
        attention_mask=batch["attention_mask"],
        p_mask=p_mask,
    )
    if bool((corruption.masked_positions & ~corruption.geometry_mask).any().item()):
        raise RuntimeError("step0 canary masked N, element, prompt, or padding tokens")
    if not bool(
        (
            corruption.winner_noisy_ids[corruption.masked_positions]
            == int(MASK_TOKEN_ID)
        ).all().item()
    ):
        raise RuntimeError("winner shared mask was not applied exactly")
    if not bool(
        (
            corruption.loser_noisy_ids[corruption.masked_positions]
            == int(MASK_TOKEN_ID)
        ).all().item()
    ):
        raise RuntimeError("loser shared mask was not applied exactly")
    unmasked = ~corruption.masked_positions
    if not torch.equal(
        corruption.winner_noisy_ids[unmasked], batch["winner_input_ids"][unmasked]
    ) or not torch.equal(
        corruption.loser_noisy_ids[unmasked], batch["loser_input_ids"][unmasked]
    ):
        raise RuntimeError("shared corruption changed an unmasked token")

    computation = compute_pair_loss(
        runtime,
        batch,
        legal_support_cache,
        generator=None,
        require_grad=False,
        corruption=corruption,
    )
    margin = computation.winner_score - computation.loser_score
    tolerance = 1e-6
    if computation.max_policy_reference_log_prob_delta > tolerance:
        raise RuntimeError("policy/reference target log-probabilities differ at step0")
    if not bool((computation.winner_score.abs() <= tolerance).all().item()):
        raise RuntimeError("step0 winner reference-corrected score is nonzero")
    if not bool((computation.loser_score.abs() <= tolerance).all().item()):
        raise RuntimeError("step0 loser reference-corrected score is nonzero")
    if not torch.allclose(-margin, computation.loser_score - computation.winner_score):
        raise RuntimeError("winner/loser swap does not reverse the D3PO margin")
    hard = d3po_pair_loss(
        computation.winner_score,
        computation.loser_score,
        beta=BETA,
        winner_anchor_weight=0.0,
    )
    expected_log2 = math.log(2.0)
    if not math.isclose(
        float(hard.loss.detach().cpu()), expected_log2, rel_tol=1e-6, abs_tol=1e-6
    ):
        raise RuntimeError("step0 hard-label loss is not log(2)")
    runtime.activate_policy(trainable=True)
    return {
        "passed": True,
        "masked_geometry_tokens": int(corruption.masked_positions.sum().item()),
        "non_geometry_tokens_masked": 0,
        "max_policy_reference_log_prob_delta": computation.max_policy_reference_log_prob_delta,
        "max_abs_reference_corrected_margin": float(margin.abs().max().cpu()),
        "hard_label_loss": float(hard.loss.cpu()),
        "expected_log2": expected_log2,
        "swap_reverses_margin": True,
    }


def evaluate_pair_preferences(
    runtime: AdapterRuntime,
    loader: DataLoader,
    device: torch.device,
    legal_support_cache: DynamicLegalSupportCache,
    *,
    seed: int,
) -> dict[str, Any]:
    """Evaluate the complete frozen validation split without selecting anything."""

    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    weighted_preference = 0.0
    weighted_anchor = 0.0
    weighted_margin = 0.0
    weighted_accuracy = 0.0
    weight_sum = 0.0
    pair_count = 0
    for raw_batch in loader:
        batch = move_batch_to_device(raw_batch, device)
        computation = compute_pair_loss(
            runtime,
            batch,
            legal_support_cache,
            generator=generator,
            require_grad=False,
        )
        weights = batch["pair_weight"].to(torch.float32)
        margin = computation.output.margin.detach()
        accuracy = (margin > 0).to(torch.float32) + 0.5 * (margin == 0).to(
            torch.float32
        )
        weighted_preference += float(
            (computation.output.per_pair_preference_loss.detach() * weights).sum().cpu()
        )
        # The core returns a weighted batch mean.  The frozen validation
        # microbatch is one pair, so multiplying by that row's pair_weight
        # recovers the composition-normalized aggregate.
        weighted_anchor += float(
            computation.output.winner_anchor_loss.detach().cpu()
            * weights.sum().cpu()
        )
        weighted_margin += float((margin * weights).sum().cpu())
        weighted_accuracy += float((accuracy * weights).sum().cpu())
        weight_sum += float(weights.sum().cpu())
        pair_count += int(weights.numel())
    if pair_count == 0 or weight_sum <= 0.0:
        raise RuntimeError("validation loader produced no weighted pairs")
    runtime.activate_policy(trainable=True)
    return {
        "pair_count": pair_count,
        "composition_normalized_weight_sum": weight_sum,
        "preference_loss": weighted_preference / weight_sum,
        "winner_anchor_loss": weighted_anchor / weight_sum,
        "mean_reference_corrected_margin": weighted_margin / weight_sum,
        "preference_accuracy_with_half_credit_for_ties": weighted_accuracy
        / weight_sum,
    }


def verify_pair_manifest(data_dir: Path) -> dict[str, Any]:
    success = data_dir / "_SUCCESS"
    manifest_path = data_dir / "D3PO_PAIR_MANIFEST.json"
    train_path = data_dir / "train.jsonl"
    validation_path = data_dir / "validation.jsonl"
    for path in (success, manifest_path, train_path, validation_path):
        if not path.exists():
            raise FileNotFoundError(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != PAIR_MANIFEST_SCHEMA:
        raise ValueError("D3PO pair manifest schema changed")
    if sha256_file(manifest_path) != PAIR_MANIFEST_SHA256:
        raise ValueError("D3PO pair manifest differs from the frozen v3 build")
    if manifest.get("l7_retired_as_test") is not True:
        raise ValueError("pair manifest did not retire SGTC L7 as a test")
    observed_hashes = {
        "train": sha256_file(train_path),
        "validation": sha256_file(validation_path),
    }
    if manifest.get("hashes") != observed_hashes:
        raise ValueError("D3PO pair JSONL hashes differ from the frozen manifest")
    if observed_hashes != PAIR_DATA_SHA256:
        raise ValueError("D3PO pair JSONL files differ from the frozen v3 build")
    return {
        "manifest": manifest,
        "manifest_sha256": sha256_file(manifest_path),
        "data_hashes": observed_hashes,
    }


def validate_runtime() -> torch.device:
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("D3PO-256-Min must use one process and one backbone")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("D3PO-256-Min requires exactly one visible CUDA GPU")
    gpu_name = torch.cuda.get_device_name(0)
    if "A800" not in gpu_name.upper():
        raise RuntimeError(f"D3PO-256-Min requires one A800, got {gpu_name!r}")
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus is not None and int(slurm_cpus) != 8:
        raise RuntimeError("D3PO-256-Min requires exactly 8 Slurm CPUs")
    return torch.device("cuda", 0)


def validate_arguments(args: argparse.Namespace) -> None:
    if int(args.seed) not in ALLOWED_TRAINING_SEEDS:
        raise ValueError(f"seed must be one of {ALLOWED_TRAINING_SEEDS}")
    if Path(args.output_dir).exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")
    for path in (Path(args.model_path), Path(args.checkpoint_path), Path(args.data_dir)):
        if not path.exists():
            raise FileNotFoundError(path)
    checkpoint = Path(args.checkpoint_path)
    if checkpoint.name != "step-696":
        raise ValueError("D3PO reference must be ctv_minimal_base step-696")
    for filename in ("adapter_config.json", "adapter_model.safetensors"):
        if not (checkpoint / filename).is_file():
            raise FileNotFoundError(checkpoint / filename)
    if sha256_file(checkpoint / "adapter_config.json") != BASE_ADAPTER_CONFIG_SHA256:
        raise ValueError("D3PO base adapter config differs from frozen step696")
    if sha256_file(checkpoint / "adapter_model.safetensors") != BASE_ADAPTER_MODEL_SHA256:
        raise ValueError("D3PO base adapter weights differ from frozen step696")


def build_train_loader(
    dataset: D3POPairDataset,
    collator: D3POPairCollator,
    *,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    # Per-composition pair weights sum to one, so replacement sampling gives
    # every composition equal expected mass even with the frozen microbatch=1.
    sampler = WeightedRandomSampler(
        weights=torch.tensor(dataset.pair_weights, dtype=torch.double),
        num_samples=len(dataset),
        replacement=True,
        generator=generator,
    )
    return DataLoader(
        dataset,
        batch_size=MICROBATCH_PAIRS,
        sampler=sampler,
        collate_fn=collator,
        num_workers=0,
        pin_memory=True,
    )


def _next_batch(loader: DataLoader, iterator: Iterable[Any]) -> tuple[Any, Iterable[Any]]:
    try:
        return next(iterator), iterator  # type: ignore[arg-type]
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def save_policy_step348(
    runtime: AdapterRuntime,
    tokenizer: Any,
    output_dir: Path,
) -> dict[str, Any]:
    """Delete the redundant reference and call the historical checkpoint helper once."""

    runtime.activate_policy(trainable=False)
    if REFERENCE_ADAPTER not in runtime.model.peft_config:
        raise RuntimeError("reference adapter disappeared before checkpoint save")
    runtime.model.delete_adapter(REFERENCE_ADAPTER)
    runtime.model.set_adapter(POLICY_ADAPTER)
    save_checkpoint(
        runtime.model,
        tokenizer,
        output_dir,
        TOTAL_UPDATES,
        save_embedding_layers="auto",
        data_dir=None,
        is_main=True,
    )
    checkpoints_root = output_dir / "checkpoints"
    step_dirs = sorted(path.name for path in checkpoints_root.iterdir() if path.is_dir())
    if step_dirs != [f"step-{TOTAL_UPDATES}"]:
        raise RuntimeError(f"unexpected checkpoint directories: {step_dirs}")
    checkpoint_root = checkpoints_root / f"step-{TOTAL_UPDATES}"
    adapter_configs = list(checkpoint_root.rglob("adapter_config.json"))
    adapter_models = list(checkpoint_root.rglob("adapter_model.safetensors"))
    if len(adapter_configs) != 1 or len(adapter_models) != 1:
        raise RuntimeError("step348 must contain exactly one policy adapter")
    adapter_dir = adapter_configs[0].parent
    if adapter_models[0].parent != adapter_dir:
        raise RuntimeError("policy adapter config/model paths disagree")
    # Non-default named PEFT adapters save in a subdirectory.  Place tokenizer
    # files there as well so existing SFT/sampling loaders can consume the path.
    tokenizer.save_pretrained(adapter_dir)
    saved_config = json.loads(adapter_configs[0].read_text(encoding="utf-8"))
    if int(saved_config.get("r", -1)) != LORA_RANK:
        raise RuntimeError("saved policy adapter rank changed")
    if int(saved_config.get("lora_alpha", -1)) != LORA_ALPHA:
        raise RuntimeError("saved policy adapter alpha changed")
    if float(saved_config.get("lora_dropout", -1.0)) != LORA_DROPOUT:
        raise RuntimeError("saved policy adapter dropout changed")
    return {
        "checkpoint_root": str(checkpoint_root.resolve()),
        "policy_adapter_path": str(adapter_dir.resolve()),
        "adapter_model_sha256": sha256_file(adapter_models[0]),
        "adapter_config_sha256": sha256_file(adapter_configs[0]),
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    validate_arguments(args)
    pair_manifest = verify_pair_manifest(Path(args.data_dir))
    device = validate_runtime()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    log_path = output_dir / "training_log.jsonl"
    failure_path = output_dir / "_FAILED.json"
    started_at = time.time()
    run_config = {
        "schema": "h1a2_shared_noise_soft_d3po_run_config_v1",
        "model_path": str(Path(args.model_path).resolve()),
        "checkpoint_path": str(Path(args.checkpoint_path).resolve()),
        "data_dir": str(Path(args.data_dir).resolve()),
        "output_dir": str(output_dir),
        "seed": int(args.seed),
        "optimization": {
            "updates": TOTAL_UPDATES,
            "microbatch_pairs": MICROBATCH_PAIRS,
            "sequences_per_pair": 2,
            "gradient_accumulation": GRADIENT_ACCUMULATION,
            "beta": BETA,
            "learning_rate": LEARNING_RATE,
            "lora_rank": LORA_RANK,
            "lora_alpha": LORA_ALPHA,
            "lora_dropout": LORA_DROPOUT,
            "winner_anchor_weight": WINNER_ANCHOR_WEIGHT,
            "energy_temperature": ENERGY_TEMPERATURE,
            "lr_scheduler": "constant",
            "gradient_checkpointing": "if_supported_non_scientific_memory_optimization",
        },
        "adapters": {
            "single_backbone": True,
            "policy": POLICY_ADAPTER,
            "reference": REFERENCE_ADAPTER,
            "reference_forward_first": True,
        },
        "pair_manifest_sha256": pair_manifest["manifest_sha256"],
        "data_hashes": pair_manifest["data_hashes"],
    }
    write_json_atomic(output_dir / "RUN_CONFIG.json", run_config)
    append_jsonl(
        log_path,
        {"event": "start", "time": started_at, **run_config},
    )

    try:
        random.seed(int(args.seed))
        torch.manual_seed(int(args.seed))
        torch.cuda.manual_seed_all(int(args.seed))
        tokenizer, runtime, adapter_report = load_policy_and_reference_adapters(args)
        runtime.model.to(device)
        collator = D3POPairCollator(tokenizer)
        train_dataset = D3POPairDataset(
            Path(args.data_dir) / "train.jsonl",
            tokenizer,
            expected_split="train",
            max_length=MAX_SEQUENCE_LENGTH,
        )
        validation_dataset = D3POPairDataset(
            Path(args.data_dir) / "validation.jsonl",
            tokenizer,
            expected_split="validation",
            max_length=MAX_SEQUENCE_LENGTH,
        )
        if train_dataset.composition_ids & validation_dataset.composition_ids:
            raise RuntimeError("train/validation composition leakage detected")
        if train_dataset.chemsys & validation_dataset.chemsys:
            raise RuntimeError("train/validation chemsys leakage detected")
        expected_counts = pair_manifest["manifest"].get("pair_counts") or {}
        if expected_counts != {
            "train": len(train_dataset),
            "validation": len(validation_dataset),
        }:
            raise RuntimeError("dataset row counts differ from the pair manifest")

        train_loader = build_train_loader(
            train_dataset, collator, seed=int(args.seed) + 101
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=MICROBATCH_PAIRS,
            shuffle=False,
            collate_fn=collator,
            num_workers=0,
            pin_memory=True,
        )
        legal_support_cache = DynamicLegalSupportCache(tokenizer)
        canary_batch = move_batch_to_device(
            collator([train_dataset[0]]), device
        )
        step0_canary = run_step0_canary(
            runtime, canary_batch, legal_support_cache
        )
        append_jsonl(log_path, {"event": "step0_canary", **step0_canary})

        runtime.activate_policy(trainable=True)
        trainable_parameters = [
            parameter
            for parameter in runtime.policy_parameters
            if parameter.requires_grad
        ]
        if not trainable_parameters:
            raise RuntimeError("optimizer would receive no policy parameters")
        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=LEARNING_RATE,
            weight_decay=0.0,
        )
        scheduler_args = SimpleNamespace(
            warmup_steps=0,
            lr_scheduler="constant",
            min_lr_ratio=1.0,
        )
        scheduler = build_lr_scheduler(optimizer, scheduler_args, TOTAL_UPDATES)
        optimizer.zero_grad(set_to_none=True)
        train_iterator = iter(train_loader)
        corruption_generator = torch.Generator(device=device)
        corruption_generator.manual_seed(int(args.seed) + 202)

        for global_step in range(1, TOTAL_UPDATES + 1):
            component_sums = {
                "loss": 0.0,
                "preference_loss": 0.0,
                "winner_anchor_loss": 0.0,
                "margin": 0.0,
                "masked_tokens": 0.0,
            }
            for _ in range(GRADIENT_ACCUMULATION):
                raw_batch, train_iterator = _next_batch(
                    train_loader, train_iterator
                )
                batch = move_batch_to_device(raw_batch, device)
                computation = compute_pair_loss(
                    runtime,
                    batch,
                    legal_support_cache,
                    generator=corruption_generator,
                    require_grad=True,
                )
                (computation.output.loss / GRADIENT_ACCUMULATION).backward()
                component_sums["loss"] += float(
                    computation.output.loss.detach().cpu()
                )
                component_sums["preference_loss"] += float(
                    computation.output.preference_loss.detach().cpu()
                )
                component_sums["winner_anchor_loss"] += float(
                    computation.output.winner_anchor_loss.detach().cpu()
                )
                component_sums["margin"] += float(
                    computation.output.margin.detach().mean().cpu()
                )
                component_sums["masked_tokens"] += float(
                    computation.corruption.masked_positions.sum().item()
                )
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                trainable_parameters, max_norm=1.0
            )
            if not bool(torch.isfinite(torch.as_tensor(gradient_norm)).item()):
                raise FloatingPointError("policy gradient norm is NaN/Inf")
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            if global_step % LOGGING_STEPS == 0 or global_step == TOTAL_UPDATES:
                append_jsonl(
                    log_path,
                    {
                        "event": "train",
                        "step": global_step,
                        **{
                            key: value / GRADIENT_ACCUMULATION
                            for key, value in component_sums.items()
                        },
                        "gradient_norm": float(
                            torch.as_tensor(gradient_norm).detach().cpu()
                        ),
                        "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    },
                )

        if global_step != TOTAL_UPDATES:
            raise RuntimeError("training did not complete exactly 348 updates")
        validation = evaluate_pair_preferences(
            runtime,
            validation_loader,
            device,
            legal_support_cache,
            seed=int(args.seed) + 303,
        )
        append_jsonl(
            log_path,
            {"event": "validation", "step": TOTAL_UPDATES, **validation},
        )
        checkpoint = save_policy_step348(runtime, tokenizer, output_dir)
        elapsed = time.time() - started_at
        manifest = {
            "schema": TRAIN_MANIFEST_SCHEMA,
            "status": "success",
            "seed": int(args.seed),
            "optimizer_updates": TOTAL_UPDATES,
            "adapter_report": adapter_report,
            "step0_canary": step0_canary,
            "data": {
                "pair_manifest_sha256": pair_manifest["manifest_sha256"],
                "hashes": pair_manifest["data_hashes"],
                "train_pairs": len(train_dataset),
                "validation_pairs": len(validation_dataset),
                "train_compositions": train_dataset.composition_count,
                "validation_compositions": validation_dataset.composition_count,
                "chemsys_disjoint": True,
                "pair_weight_sampling": "composition-normalized WeightedRandomSampler",
            },
            "validation": validation,
            "checkpoint": checkpoint,
            "elapsed_seconds": elapsed,
            "search_or_selection": False,
            "intermediate_checkpoints": 0,
        }
        write_json_atomic(output_dir / "D3PO_TRAIN_MANIFEST.json", manifest)
        append_jsonl(log_path, {"event": "success", **manifest})
        (output_dir / "_SUCCESS").touch(exist_ok=False)
        return manifest
    except Exception as exc:
        failure = {
            "schema": "h1a2_shared_noise_soft_d3po_failure_v1",
            "status": "failed",
            "seed": int(args.seed),
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
            "success_marker_written": False,
        }
        write_json_atomic(failure_path, failure)
        append_jsonl(log_path, {"event": "failure", **failure})
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Frozen one-GPU shared-noise masked-D3PO trainer"
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--seed",
        type=int,
        choices=ALLOWED_TRAINING_SEEDS,
        required=True,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = train(args)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
