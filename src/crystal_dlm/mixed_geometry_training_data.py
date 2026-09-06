"""Matched token/continuous views with source-keyed noise and exact coverage.

The continuous target is the verified original CIF, already aligned to native
element slots. Discrete geometry is used only by the inherited token task.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from crystal_dlm.fixed_slot import SYMBOL_TO_Z
from crystal_dlm.mixed_geometry_diffusion import (
    DEFAULT_CONFIG, GeometryState, LatticeNormalizer, MixedGeometryConfig,
    corrupt_geometry, sample_log_uniform_times,
)
from crystal_dlm.periodic_base_training_data import _mask_id, prepare_periodic_base_source
from crystal_dlm.periodic_v2_training_data import PrefixSupportAuditor, make_periodic_v2_training_example
from crystal_dlm.state_training import materialize_state_batch


VALIDATION_TIME_BANDS = ((.002, .01), (.01, .05), (.05, .2), (.2, .5), (.5, .9), (.9, 1.))


def read_rows(path_or_rows):
    if isinstance(path_or_rows, (str, Path)):
        with Path(path_or_rows).open(encoding="utf-8") as stream:
            return [json.loads(line) for line in stream if line.strip()]
    return list(path_or_rows)


def source_key(row: Mapping[str, Any]) -> tuple[str, int]:
    return str(row["source_split"]), int(row["source_row_idx"])


def source_seed(seed: int, key: tuple[str, int], epoch: int, stream: str) -> int:
    payload = json.dumps(["mixed_geometry_v1", int(seed), *key, int(epoch), stream]).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


@dataclass(frozen=True)
class MixedBatchSchedule:
    """Every rank sees one mode per microstep, with matched T/G source windows."""

    source_count: int
    world_size: int
    microbatch: int = 1
    effective_batch: int = 24

    def __post_init__(self):
        if min(self.source_count, self.world_size, self.microbatch, self.effective_batch) < 1:
            raise ValueError("source and batch sizes must be positive")
        if self.effective_batch % (2 * self.world_size * self.microbatch):
            raise ValueError("effective batch must contain equal complete T/G rank groups")

    @property
    def sources_per_window(self):
        return self.effective_batch // 2

    @property
    def accumulation(self):
        return self.effective_batch // (self.world_size * self.microbatch)

    @property
    def updates_per_epoch(self):
        return math.ceil(self.source_count / self.sources_per_window)

    @property
    def padded_sources(self):
        return self.updates_per_epoch * self.sources_per_window

    @property
    def epoch_normalization(self):
        return self.padded_sources / self.source_count

    def manifest(self, epochs: int):
        return {"source_count": self.source_count, "world_size": self.world_size,
                "microbatch": self.microbatch, "effective_batch": self.effective_batch,
                "gradient_accumulation": self.accumulation, "epochs": epochs,
                "epoch_updates": self.updates_per_epoch, "updates": epochs * self.updates_per_epoch,
                "real_states_per_epoch": 2 * self.source_count,
                "padded_states_per_epoch": 2 * self.padded_sources,
                "padding_per_epoch": 2 * (self.padded_sources - self.source_count),
                "effective_states": epochs * 2 * self.source_count,
                "padding_normalization": self.epoch_normalization}

    def rank_batches(self, *, rank: int, seed: int, epoch: int):
        if not 0 <= rank < self.world_size:
            raise ValueError("rank outside world")
        rng = np.random.default_rng(source_seed(seed, ("train_order", 0), epoch, "permutation"))
        permutation = rng.permutation(self.source_count).tolist()
        for window in range(self.updates_per_epoch):
            for microstep in range(self.accumulation):
                mode = "token" if microstep % 2 == 0 else "geometry"
                first = (window * self.sources_per_window
                         + (microstep // 2 * self.world_size + rank) * self.microbatch)
                yield window, microstep, mode, [
                    (permutation[i % self.source_count], i >= self.source_count)
                    for i in range(first, first + self.microbatch)]


class MixedGeometrySources:
    """Verified clean sources; views are generated without outcome fields."""

    def __init__(self, path_or_rows, identity_path_or_rows, tokenizer, constraints,
                 normalizer: LatticeNormalizer, *, split: str, seed: int,
                 max_sites: int = 20, diffusion_config: MixedGeometryConfig = DEFAULT_CONFIG):
        rows, identities = read_rows(path_or_rows), read_rows(identity_path_or_rows)
        if split not in ("train", "val") or not rows or max_sites < 1:
            raise ValueError("nonempty original train/val sources are required")
        keys = [source_key(row) for row in rows]
        indexed = {source_key(row): row for row in identities}
        if (len(set(keys)) != len(keys) or len(indexed) != len(identities)
                or set(keys) != set(indexed) or any(key[0] != split for key in keys)):
            raise ValueError("continuous identity must be one-to-one with the complete source split")
        self.tokenizer, self.constraints = tokenizer, constraints
        self.normalizer, self.max_sites = normalizer, int(max_sites)
        self.seed, self.config = int(seed), diffusion_config
        self.vocabulary = tokenizer.get_vocab()
        self.mask_id = _mask_id(tokenizer, self.vocabulary)
        self.prefix_auditor = PrefixSupportAuditor(tokenizer, constraints)
        self.sources, self.scaffolds, self.prefixes, lattices = [], [], [], []
        self.fractional = torch.zeros(len(rows), max_sites, 3, dtype=torch.float64)
        self.atom_mask = torch.zeros(len(rows), max_sites, dtype=torch.bool)
        self.species = torch.zeros(len(rows), max_sites, dtype=torch.long)
        for index, (row, key) in enumerate(zip(rows, keys)):
            identity = indexed[key]
            answer = row.get("source_answer", row.get("answer"))
            if row.get("answer") != answer:
                raise ValueError("the token target differs from the verified clean source target")
            if (identity.get("identity_verified") is not True or not isinstance(answer, str)
                    or hashlib.sha256(answer.encode()).hexdigest() != identity.get("target_answer_sha256_bytes")):
                raise ValueError(f"unverified or changed original target: {key}")
            source = prepare_periodic_base_source(row, tokenizer, constraints, vocabulary=self.vocabulary)
            geometry = identity["continuous_aligned"]
            count = int(source.arrays["num_atoms"])
            if not 1 <= count <= max_sites or list(geometry["species"]) != list(source.arrays["species"]):
                raise ValueError("continuous species must use the exact native element slot order")
            lattice = torch.as_tensor(geometry["lattice_matrix_A"], dtype=torch.float64)
            fractional = torch.as_tensor(geometry["fractional"], dtype=torch.float64)
            if (lattice.shape != (3, 3) or fractional.shape != (count, 3)
                    or not bool(torch.isfinite(lattice).all() and torch.isfinite(fractional).all())):
                raise ValueError("continuous source geometry must have finite original dimensions")
            self.fractional[index, :count] = fractional.remainder(1)
            self.atom_mask[index, :count] = True
            self.species[index, :count] = torch.tensor([SYMBOL_TO_Z[s] for s in geometry["species"]])
            scaffold = list(source.clean_tokens)
            for position in source.transaction_positions:
                scaffold[position] = self.mask_id
            self.sources.append(source)
            self.scaffolds.append(scaffold)
            self.prefixes.append(tokenizer(source.metadata["prompt"].rstrip() + "\n",
                                           add_special_tokens=False)["input_ids"])
            lattices.append(lattice)
        self.keys = keys
        self.lattices = torch.stack(lattices)
        self.clean_z = normalizer.encode(self.lattices, self.atom_mask.sum(-1))

    def __len__(self):
        return len(self.sources)

    def validation_indices(self, count: int, seed: int):
        if not 1 <= count <= len(self):
            raise ValueError("validation count exceeds source split")
        return sorted(range(len(self)), key=lambda i: source_seed(seed, self.keys[i], 0, "validation_selection"))[:count]

    def example(self, index: int, *, mode: str, epoch: int, is_padding: bool = False,
                fixed_time: float | None = None, noise_stream: str = "training"):
        source = self.sources[index]
        if mode == "token":
            example = make_periodic_v2_training_example(
                source, self.tokenizer, self.constraints, view=0, epoch=epoch, seed=self.seed,
                is_padding=is_padding, vocabulary=self.vocabulary, prefix_auditor=self.prefix_auditor)
        elif mode == "geometry":
            generator = torch.Generator().manual_seed(source_seed(self.seed, self.keys[index], epoch,
                                                                  "geometry:" + noise_stream))
            time = (sample_log_uniform_times(generator=generator, config=self.config)
                    if fixed_time is None else torch.tensor(fixed_time, dtype=torch.float64))
            if not self.config.epsilon <= float(time) <= 1:
                raise ValueError("geometry training time must lie in the registered interval")
            noisy = corrupt_geometry(self.clean_z[index], self.fractional[index], time,
                                     atom_mask=self.atom_mask[index], generator=generator, config=self.config)
            scaffold = self.scaffolds[index]
            example = {**source.metadata, "view": 1, "epoch": epoch,
                       "input_body": scaffold, "old_body": scaffold, "phase": "structured_denoise",
                       "num_atoms": int(source.arrays["num_atoms"]),
                       "transaction_positions": list(source.transaction_positions),
                       # The shared materializer's scalar indices are unused in G.
                       "position": source.transaction_positions[0], "target_token": self.mask_id,
                       "is_padding": bool(is_padding), "source_sample_weight": source.metadata["sample_weight"],
                       "sample_weight": 0. if is_padding else source.metadata["sample_weight"],
                       "numeric_noise_level": -1., "numeric_noise_components": [-1., -1., -1.],
                       "noisy_geometry": noisy, "atomic_numbers": self.species[index],
                       "outcomes_read": False}
        else:
            raise ValueError("mixed mode must be token or geometry")
        return {**example, "mode": mode, "prompt_token_ids": self.prefixes[index], "source_index": index}


def materialize_mixed_batch(examples: Sequence[dict], tokenizer, *, device,
                            max_length: int = 382, max_sites: int = 20):
    modes = {example["mode"] for example in examples}
    if len(modes) != 1:
        raise ValueError("a mixed microbatch must contain exactly one mode")
    batch = materialize_state_batch(examples, tokenizer, device=device, max_length=max_length, max_sites=max_sites)
    batch["mode"] = next(iter(modes))
    batch["source_weights"] = torch.tensor([e["sample_weight"] for e in examples], dtype=torch.float32, device=device)
    if batch["mode"] == "geometry":
        noisy = [e["noisy_geometry"] for e in examples]
        batch["geometry_state"] = GeometryState(*(
            torch.stack([getattr(n.state, name) for n in noisy]).to(device)
            for name in ("z", "fractional", "t", "atom_mask")))
        batch["v_target"] = torch.stack([n.v_target for n in noisy]).to(device)
        batch["u_target"] = torch.stack([n.u_target for n in noisy]).to(device)
        batch["species"] = torch.stack([e["atomic_numbers"] for e in examples]).to(device)
    return batch


def forward_mixed(model, batch, *, output_hidden_states=False):
    kwargs = {"mode": batch["mode"], "geometry_context": batch["geometry_context"]}
    if batch["mode"] == "geometry":
        kwargs.update(geometry_state=batch["geometry_state"], species=batch["species"],
                      output_hidden_states=output_hidden_states)
    return model(batch["input_ids"], attention_mask=batch["attention_mask"], **kwargs)


__all__ = ["MixedBatchSchedule", "MixedGeometrySources", "materialize_mixed_batch", "forward_mixed",
           "source_key", "source_seed", "VALIDATION_TIME_BANDS"]
