#!/usr/bin/env python3
"""Response-only LLaDA SFT for fixed-slot and dynamic crystal generation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Sampler, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer

from crystal_dlm.fixed_slot import (
    ANSWER_TOKEN_COUNT,
    CHEMICAL_SYMBOLS,
    MASK_TOKEN_ID,
    PROMPT_POOL,
    build_special_tokens,
    structure_to_answer,
    write_json,
)
from crystal_dlm.dynamic_crystal import (
    DYNAMIC_MAX_ANSWER_TOKEN_COUNT,
    DYNAMIC_PROMPT_POOL,
    structure_to_dynamic_answer,
)
from crystal_dlm.cif_lite import CIF_LITE_PROMPT_POOL, MODULE_TO_ID
from crystal_dlm.crysllmgen_text import (
    CRYSLLMGEN_COMPOSITION_PROMPT_TEMPLATE,
    CRYSLLMGEN_COORDS_PROMPT_TEMPLATE,
    CRYSLLMGEN_LATTICE_PROMPT,
    CRYSLLMGEN_MODULE_TO_ID,
    CRYSLLMGEN_SITE_COORD_PROMPT_TEMPLATE,
    CRYSLLMGEN_SPECIES_PROMPT_TEMPLATE,
    CRYSLLMGEN_TEXT_PROMPT,
)
from crystal_dlm.fixed_plain import (
    FIXED_PLAIN_COORDS_PROMPT_TEMPLATE,
    FIXED_PLAIN_COUNT_PROMPT,
    FIXED_PLAIN_ELEMENTS_PROMPT_TEMPLATE,
    FIXED_PLAIN_LATTICE_PROMPT_TEMPLATE,
    FIXED_PLAIN_MODULE_TO_ID,
)
from crystal_dlm.physical_header import (
    PHYSICAL_HEADER_ANSWER_TOKEN_COUNT,
    PHYSICAL_HEADER_BODY_OFFSET,
    PHYSICAL_HEADER_PROMPT_POOL,
)
from crystal_dlm.llada_resize import ensure_llada_vocab_size
from crystal_dlm.transformers_compat import (
    ensure_create_bidirectional_mask,
    ensure_llada2_rope_parameters,
)


MASK_POLICY_TO_ID = {
    "normal": 0,
    "active_element": 1,
    "n_active_element": 2,
    "active_element_empty": 3,
}
ID_TO_MASK_POLICY = {value: key for key, value in MASK_POLICY_TO_ID.items()}
LOSS_PROFILE_TO_ID = {
    "fixed_slot": 0,
    "text": 1,
}
ID_TO_LOSS_PROFILE = {value: key for key, value in LOSS_PROFILE_TO_ID.items()}
TEXT_ONLY_REPRESENTATIONS = {
    "cif_lite_modular",
    "crysllmgen_text",
    "fixed_plain",
    "r5_plan_state",
    "r5_repair_text",
}

ELEMENT_NAMES = {
    "H": "hydrogen",
    "He": "helium",
    "Li": "lithium",
    "Be": "beryllium",
    "B": "boron",
    "C": "carbon",
    "N": "nitrogen",
    "O": "oxygen",
    "F": "fluorine",
    "Ne": "neon",
    "Na": "sodium",
    "Mg": "magnesium",
    "Al": "aluminum",
    "Si": "silicon",
    "P": "phosphorus",
    "S": "sulfur",
    "Cl": "chlorine",
    "Ar": "argon",
    "K": "potassium",
    "Ca": "calcium",
    "Sc": "scandium",
    "Ti": "titanium",
    "V": "vanadium",
    "Cr": "chromium",
    "Mn": "manganese",
    "Fe": "iron",
    "Co": "cobalt",
    "Ni": "nickel",
    "Cu": "copper",
    "Zn": "zinc",
    "Ga": "gallium",
    "Ge": "germanium",
    "As": "arsenic",
    "Se": "selenium",
    "Br": "bromine",
    "Kr": "krypton",
    "Rb": "rubidium",
    "Sr": "strontium",
    "Y": "yttrium",
    "Zr": "zirconium",
    "Nb": "niobium",
    "Mo": "molybdenum",
    "Tc": "technetium",
    "Ru": "ruthenium",
    "Rh": "rhodium",
    "Pd": "palladium",
    "Ag": "silver",
    "Cd": "cadmium",
    "In": "indium",
    "Sn": "tin",
    "Sb": "antimony",
    "Te": "tellurium",
    "I": "iodine",
    "Xe": "xenon",
    "Cs": "cesium",
    "Ba": "barium",
    "La": "lanthanum",
    "Ce": "cerium",
    "Pr": "praseodymium",
    "Nd": "neodymium",
    "Pm": "promethium",
    "Sm": "samarium",
    "Eu": "europium",
    "Gd": "gadolinium",
    "Tb": "terbium",
    "Dy": "dysprosium",
    "Ho": "holmium",
    "Er": "erbium",
    "Tm": "thulium",
    "Yb": "ytterbium",
    "Lu": "lutetium",
    "Hf": "hafnium",
    "Ta": "tantalum",
    "W": "tungsten",
    "Re": "rhenium",
    "Os": "osmium",
    "Ir": "iridium",
    "Pt": "platinum",
    "Au": "gold",
    "Hg": "mercury",
    "Tl": "thallium",
    "Pb": "lead",
    "Bi": "bismuth",
    "Po": "polonium",
    "At": "astatine",
    "Rn": "radon",
    "Fr": "francium",
    "Ra": "radium",
    "Ac": "actinium",
    "Th": "thorium",
    "Pa": "protactinium",
    "U": "uranium",
    "Np": "neptunium",
    "Pu": "plutonium",
}

COMMON_OXIDATION_STATES = {
    "H": [1, -1],
    "Li": [1],
    "Na": [1],
    "K": [1],
    "Rb": [1],
    "Cs": [1],
    "Be": [2],
    "Mg": [2],
    "Ca": [2],
    "Sr": [2],
    "Ba": [2],
    "B": [3],
    "Al": [3],
    "Ga": [3],
    "In": [3, 1],
    "C": [4, -4],
    "Si": [4],
    "Ge": [4, 2],
    "Sn": [4, 2],
    "Pb": [4, 2],
    "N": [-3, 3, 5],
    "P": [5, 3, -3],
    "As": [5, 3, -3],
    "Sb": [5, 3],
    "Bi": [3, 5],
    "O": [-2],
    "S": [-2, 4, 6],
    "Se": [-2, 4, 6],
    "Te": [-2, 4, 6],
    "F": [-1],
    "Cl": [-1, 1, 3, 5, 7],
    "Br": [-1, 1, 3, 5],
    "I": [-1, 1, 5, 7],
    "Sc": [3],
    "Y": [3],
    "La": [3],
    "Ce": [3, 4],
    "Pr": [3, 4],
    "Nd": [3],
    "Sm": [3, 2],
    "Eu": [3, 2],
    "Gd": [3],
    "Tb": [3, 4],
    "Dy": [3],
    "Ho": [3],
    "Er": [3],
    "Tm": [3],
    "Yb": [3, 2],
    "Lu": [3],
    "Ti": [4, 3],
    "Zr": [4],
    "Hf": [4],
    "V": [5, 4, 3],
    "Nb": [5],
    "Ta": [5],
    "Cr": [3, 6],
    "Mo": [6, 4],
    "W": [6, 4],
    "Mn": [2, 3, 4, 7],
    "Fe": [2, 3],
    "Co": [2, 3],
    "Ni": [2, 3],
    "Cu": [1, 2],
    "Zn": [2],
    "Ru": [3, 4],
    "Rh": [3],
    "Pd": [2, 4],
    "Ag": [1],
    "Cd": [2],
    "Re": [4, 6, 7],
    "Os": [4],
    "Ir": [3, 4],
    "Pt": [2, 4],
    "Au": [1, 3],
    "Hg": [1, 2],
    "Tl": [1, 3],
    "Ac": [3],
    "Th": [4],
    "Pa": [5],
    "U": [4, 5, 6],
    "Np": [4, 5, 6],
    "Pu": [3, 4, 5, 6],
}

ALKALI_ELEMENTS = {"Li", "Na", "K", "Rb", "Cs"}
ALKALINE_EARTH_ELEMENTS = {"Be", "Mg", "Ca", "Sr", "Ba"}
LANTHANIDE_ELEMENTS = {"La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"}
ACTINIDE_ELEMENTS = {"Ac", "Th", "Pa", "U", "Np", "Pu"}
HALOGEN_ELEMENTS = {"F", "Cl", "Br", "I"}
CHALCOGEN_ELEMENTS = {"O", "S", "Se", "Te"}
PNICTOGEN_ELEMENTS = {"N", "P", "As", "Sb", "Bi"}
TRANSITION_METALS = {
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
}


def oxidation_state_aliases(symbol: str, name: str) -> list[str]:
    aliases: list[str] = []
    for state in COMMON_OXIDATION_STATES.get(symbol, [])[:4]:
        sign = "+" if state > 0 else "-"
        magnitude = abs(int(state))
        aliases.extend(
            [
                f"{symbol} {sign}{magnitude} oxidation state",
                f"{name} {sign}{magnitude} oxidation state",
                f"{symbol}{magnitude}{sign} ion",
                f"{name} {magnitude}{sign} ion",
            ]
        )
        if state > 0:
            aliases.extend(
                [
                    f"{symbol} cation charge {magnitude}",
                    f"{name} oxide {symbol}{magnitude}{sign}",
                ]
            )
        else:
            aliases.extend(
                [
                    f"{symbol} anion charge {magnitude}",
                    f"{name} anion {magnitude}{sign}",
                ]
            )
    return aliases


def mask_policy_id(name: str | None) -> int:
    if not name:
        return MASK_POLICY_TO_ID["normal"]
    normalized = str(name).strip()
    if normalized not in MASK_POLICY_TO_ID:
        raise ValueError(f"Unknown SFT mask_policy={normalized!r}")
    return MASK_POLICY_TO_ID[normalized]


def loss_profile_id(name: str | None) -> int:
    if not name:
        return LOSS_PROFILE_TO_ID["fixed_slot"]
    normalized = str(name).strip()
    if normalized not in LOSS_PROFILE_TO_ID:
        raise ValueError(f"Unknown SFT loss_profile={normalized!r}")
    return LOSS_PROFILE_TO_ID[normalized]


class JsonlSftDataset(Dataset):
    def __init__(self, path: Path, tokenizer, max_length: int):
        self.rows: List[Dict[str, Any]] = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    self.rows.append(json.loads(line))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.rows[index]
        prompt_text = row["prompt"].rstrip() + "\n"
        full_text = prompt_text + row["answer"]
        input_ids = self.tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )["input_ids"]
        prompt_length = len(
            self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        )
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "prompt_length": min(prompt_length, len(input_ids)),
            "mask_policy_id": mask_policy_id(row.get("mask_policy") or row.get("sft_mask_policy")),
            "loss_profile_id": loss_profile_id(row.get("loss_profile")),
            "module_id": int(row.get("module_id") or MODULE_TO_ID.get(str(row.get("module", "")), 0)),
            "sample_weight": float(row.get("sample_weight", 1.0) or 1.0),
        }

    def sample_weights(
        self,
        multipliers: Optional[Dict[str, float]] = None,
        *,
        use_jsonl_sample_weight: bool = True,
    ) -> List[float]:
        weights: List[float] = []
        for row in self.rows:
            if use_jsonl_sample_weight:
                try:
                    weight = float(row.get("sample_weight", 1.0) or 1.0)
                except (TypeError, ValueError):
                    weight = 1.0
            else:
                weight = 1.0
            if multipliers:
                weight *= sample_weight_multiplier_for_row(row, multipliers)
            weights.append(max(weight, 0.0))
        return weights


class CsvCrystalSftDataset(Dataset):
    def __init__(
        self,
        path: Path,
        tokenizer,
        max_length: int,
        split: str,
        seed: int,
        answer_separator: str,
        augment_origin_shift: bool,
        niggli: bool,
        primitive: bool,
        answer_representation: str,
    ):
        self.rows: List[Dict[str, str]] = []
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            self.rows.extend(reader)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        self.seed = seed
        self.epoch = 0
        self.answer_separator = answer_separator
        self.augment_origin_shift = augment_origin_shift
        self.niggli = niggli
        self.primitive = primitive
        self.answer_representation = answer_representation

    def __len__(self) -> int:
        return len(self.rows)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _structure_from_cif(self, cif: str, rng: random.Random):
        from pymatgen.core import Lattice, Structure

        structure = Structure.from_str(cif, fmt="cif")
        if self.primitive:
            structure = structure.get_primitive_structure()
        if self.niggli:
            structure = structure.get_reduced_structure()
        structure = Structure(
            lattice=Lattice.from_parameters(*structure.lattice.parameters),
            species=structure.species,
            coords=structure.frac_coords,
            coords_are_cartesian=False,
        )
        if self.split == "train" and self.augment_origin_shift:
            shift = [rng.random(), rng.random(), rng.random()]
            structure.translate_sites(
                indices=range(len(structure.sites)),
                vector=shift,
                frac_coords=True,
                to_unit_cell=True,
            )
        return structure

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.rows[index]
        rng = random.Random(self.seed + self.epoch * 1_000_003 + index)
        prompt_pool = DYNAMIC_PROMPT_POOL if self.answer_representation == "dynamic_v1" else PROMPT_POOL
        prompt = rng.choice(prompt_pool) if self.split == "train" else prompt_pool[0]
        structure = self._structure_from_cif(row["cif"], rng)
        if self.answer_representation == "dynamic_v1":
            answer, _ = structure_to_dynamic_answer(structure, separator=self.answer_separator)
        else:
            answer, _ = structure_to_answer(structure, separator=self.answer_separator)
        prompt_text = prompt.rstrip() + "\n"
        full_text = prompt_text + answer
        input_ids = self.tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )["input_ids"]
        prompt_length = len(
            self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        )
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "prompt_length": min(prompt_length, len(input_ids)),
            "mask_policy_id": MASK_POLICY_TO_ID["normal"],
            "loss_profile_id": LOSS_PROFILE_TO_ID["fixed_slot"],
            "module_id": 0,
            "sample_weight": 1.0,
        }


class DataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_len = max(item["input_ids"].shape[0] for item in batch)
        input_ids = torch.full(
            (len(batch), max_len),
            self.tokenizer.pad_token_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        prompt_lengths = torch.zeros((len(batch),), dtype=torch.long)
        mask_policy_ids = torch.zeros((len(batch),), dtype=torch.long)
        loss_profile_ids = torch.zeros((len(batch),), dtype=torch.long)
        module_ids = torch.zeros((len(batch),), dtype=torch.long)
        sample_weights = torch.ones((len(batch),), dtype=torch.float32)
        for i, item in enumerate(batch):
            ids = item["input_ids"]
            input_ids[i, : ids.shape[0]] = ids
            attention_mask[i, : ids.shape[0]] = 1
            prompt_lengths[i] = item["prompt_length"]
            mask_policy_ids[i] = int(item.get("mask_policy_id", MASK_POLICY_TO_ID["normal"]))
            loss_profile_ids[i] = int(item.get("loss_profile_id", LOSS_PROFILE_TO_ID["fixed_slot"]))
            module_ids[i] = int(item.get("module_id", 0))
            sample_weights[i] = float(item.get("sample_weight", 1.0) or 1.0)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "prompt_lengths": prompt_lengths,
            "mask_policy_ids": mask_policy_ids,
            "loss_profile_ids": loss_profile_ids,
            "module_ids": module_ids,
            "sample_weights": sample_weights,
        }


class DistributedWeightedSampler(Sampler[int]):
    """Distributed replacement sampler backed by per-row JSONL weights."""

    def __init__(
        self,
        weights: Iterable[float],
        *,
        num_replicas: int,
        rank: int,
        seed: int = 0,
    ) -> None:
        self.weights = torch.as_tensor(list(weights), dtype=torch.double)
        if self.weights.numel() == 0:
            raise ValueError("DistributedWeightedSampler requires at least one weight")
        if torch.any(self.weights < 0):
            raise ValueError("sample weights must be non-negative")
        if float(self.weights.sum().item()) <= 0.0:
            raise ValueError("at least one sample weight must be positive")
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.weights) / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(
            self.weights,
            self.total_size,
            replacement=True,
            generator=generator,
        ).tolist()
        return iter(indices[self.rank : self.total_size : self.num_replicas])

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)


def summarize_sample_weights(weights: Iterable[float]) -> Dict[str, Any]:
    values = [float(weight) for weight in weights]
    positive = [weight for weight in values if weight > 0.0]
    unique = sorted(set(round(weight, 8) for weight in values))
    summary: Dict[str, Any] = {
        "count": len(values),
        "positive_count": len(positive),
        "zero_count": len(values) - len(positive),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "mean": (sum(values) / len(values)) if values else None,
        "unique_count": len(unique),
        "unique_values_head": unique[:20],
        "has_nonuniform_weights": len(unique) > 1,
    }
    if positive:
        total = sum(positive)
        summary["positive_mass"] = total
        summary["top_decile_mass_fraction"] = None
        if len(positive) >= 10 and total > 0:
            sorted_pos = sorted(positive, reverse=True)
            top_count = max(1, math.ceil(len(sorted_pos) * 0.1))
            summary["top_decile_mass_fraction"] = sum(sorted_pos[:top_count]) / total
    return summary


def parse_sample_weight_multipliers(text: str | None) -> Dict[str, float]:
    if not text:
        return {}
    multipliers: Dict[str, float] = {}
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        if "=" in item:
            key, value = item.split("=", 1)
        elif ":" in item:
            key, value = item.split(":", 1)
        else:
            raise ValueError(
                f"Invalid --sample-weight-multipliers item {item!r}; expected key=value"
            )
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid empty multiplier key in {item!r}")
        try:
            parsed = float(value)
        except ValueError as exc:
            raise ValueError(f"Invalid multiplier value in {item!r}") from exc
        if parsed < 0:
            raise ValueError(f"Multiplier for {key!r} must be non-negative")
        multipliers[key] = parsed
    return multipliers


def sample_weight_keys_for_row(row: Dict[str, Any]) -> List[str]:
    keys: List[str] = []
    for field in (
        "composition_bucket",
        "composition_reason",
        "sample_weight_tier",
        "selection_role",
        "source_kind",
    ):
        value = row.get(field)
        if value is None:
            continue
        value_str = str(value).strip()
        if value_str:
            keys.append(value_str)
            keys.append(f"{field}:{value_str}")
    reason = str(row.get("composition_reason") or "").strip()
    if reason == "charge_neutral_pauling_valid":
        keys.append("strict")
    elif reason == "all_metal_shortcut":
        keys.append("all_metal")
    elif reason == "single_element_shortcut":
        keys.append("single_element")
    elif reason:
        keys.append("invalid")
    if bool(row.get("strict_valid")):
        keys.append("strict_valid")
    if bool(row.get("comp_valid")):
        keys.append("comp_valid")
    return keys


def sample_weight_multiplier_for_row(row: Dict[str, Any], multipliers: Dict[str, float]) -> float:
    if not multipliers:
        return 1.0
    for key in sample_weight_keys_for_row(row):
        if key in multipliers:
            return float(multipliers[key])
    return 1.0


def summarize_sample_weight_multipliers(
    rows: Iterable[Dict[str, Any]],
    multipliers: Dict[str, float],
) -> Dict[str, Any]:
    bucket_counts: Counter[str] = Counter()
    matched_counts: Counter[str] = Counter()
    matched_weight_mass: Counter[str] = Counter()
    for row in rows:
        bucket = str(row.get("composition_bucket") or "missing")
        bucket_counts[bucket] += 1
        matched_key = None
        for key in sample_weight_keys_for_row(row):
            if key in multipliers:
                matched_key = key
                break
        if matched_key is not None:
            matched_counts[matched_key] += 1
            try:
                base_weight = float(row.get("sample_weight", 1.0) or 1.0)
            except (TypeError, ValueError):
                base_weight = 1.0
            matched_weight_mass[matched_key] += max(base_weight, 0.0) * float(multipliers[matched_key])
    return {
        "configured": dict(multipliers),
        "composition_bucket_counts": dict(bucket_counts.most_common()),
        "matched_counts": dict(matched_counts.most_common()),
        "matched_weight_mass": dict(matched_weight_mass.most_common()),
    }


def resize_tokenizer_and_model(tokenizer, model, vocab_file: Path) -> int:
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    with vocab_file.open(encoding="utf-8") as handle:
        special_tokens = [line.strip() for line in handle if line.strip()]
    num_new = tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    if num_new:
        model.resize_token_embeddings(len(tokenizer))
        ensure_llada_vocab_size(model, len(tokenizer))
        embeddings = model.get_input_embeddings().weight.data
        embeddings[-num_new:] = embeddings[:-num_new].mean(dim=0, keepdim=True)
        output_embeddings = model.get_output_embeddings()
        if output_embeddings is not None:
            output_embeddings.weight.data[-num_new:] = output_embeddings.weight.data[
                :-num_new
            ].mean(dim=0, keepdim=True)
    else:
        ensure_llada_vocab_size(model, len(tokenizer))
    return num_new


def data_vocab_missing_tokens(tokenizer, vocab_file: Path) -> list[str]:
    if not vocab_file.exists():
        return []
    vocab = tokenizer.get_vocab()
    with vocab_file.open(encoding="utf-8") as handle:
        tokens = [line.strip() for line in handle if line.strip()]
    return [token for token in tokens if token not in vocab]


def element_semantic_aliases(symbol: str) -> list[str]:
    name = ELEMENT_NAMES.get(symbol, symbol)
    aliases = [
        symbol,
        f" {symbol}",
        name,
        f" {name}",
        f"{symbol} element",
        f"{name} element",
        f"{symbol} ion",
        f"{name} ion",
        f"{symbol} oxide",
        f"{name} oxide",
        f"{symbol} crystal",
        f"{name} crystal",
    ]
    aliases.extend(oxidation_state_aliases(symbol, name))
    if symbol in CHALCOGEN_ELEMENTS:
        aliases.extend(
            [
                f"{name} anion",
                f"{symbol} anion",
                f"metal {name}",
                f"metal {name} compound",
                f"{name} chalcogenide",
                f"{symbol} chalcogenide",
            ]
        )
        if symbol == "O":
            aliases.extend(["oxide anion", "metal oxide", "perovskite oxide", "oxide crystal"])
        elif symbol == "S":
            aliases.extend(["sulfide anion", "metal sulfide", "sulfide crystal"])
        elif symbol == "Se":
            aliases.extend(["selenide anion", "metal selenide"])
        elif symbol == "Te":
            aliases.extend(["telluride anion", "metal telluride"])
    elif symbol in HALOGEN_ELEMENTS:
        aliases.extend(
            [
                f"{name} anion",
                f"{symbol} anion",
                f"{name} halide",
                f"{symbol} halide",
                f"alkali {name}",
                f"metal {name}",
                "halide crystal",
            ]
        )
    elif symbol in PNICTOGEN_ELEMENTS:
        aliases.extend(
            [
                f"{name} anion",
                f"{symbol} anion",
                f"{name} pnictide",
                f"{symbol} pnictide",
                f"metal {name}",
                f"{name} oxide",
            ]
        )
        if symbol == "N":
            aliases.extend(["nitride anion", "metal nitride", "nitride crystal"])
        elif symbol == "P":
            aliases.extend(["phosphate", "phosphide", "lithium iron phosphate"])
    else:
        aliases.extend([f"{name} cation", f"{symbol} cation", f"{symbol} oxygen compound"])
        if symbol in ALKALI_ELEMENTS:
            aliases.extend([f"{name} alkali metal", f"{symbol} alkali cation", f"{symbol} oxide", f"{symbol} halide"])
        if symbol in ALKALINE_EARTH_ELEMENTS:
            aliases.extend([f"{name} alkaline earth metal", f"{symbol} divalent cation", f"{symbol} oxide"])
        if symbol in TRANSITION_METALS:
            aliases.extend([f"{name} transition metal", f"{symbol} transition metal oxide", f"{symbol} redox ion"])
        if symbol in LANTHANIDE_ELEMENTS:
            aliases.extend([f"{name} rare earth", f"{symbol} rare earth oxide", f"{symbol} trivalent cation"])
        if symbol in ACTINIDE_ELEMENTS:
            aliases.extend([f"{name} actinide", f"{symbol} actinide oxide", f"{symbol} radioactive metal"])
    # A few common MP-20 motifs help the new special tokens inherit the base
    # model's latent chemical neighborhoods without changing the generation
    # format or adding property conditioning.
    if symbol in {"Li", "Co", "Mn", "Fe", "Ni", "V", "Ti", "O"}:
        aliases.extend(["lithium transition metal oxide", "battery cathode oxide"])
    if symbol in {"Ba", "Sr", "Ca", "Ti", "Zr", "W", "O"}:
        aliases.extend(["perovskite oxide", "alkaline earth oxide"])
    return aliases


@torch.no_grad()
def semantic_init_element_tokens(tokenizer, model, *, enabled: bool, num_new_tokens: int) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "enabled": bool(enabled),
        "attempted": False,
        "skipped_reason": None,
        "num_new_tokens": int(num_new_tokens),
        "initialized_count": 0,
        "element_reports": {},
    }
    if not enabled:
        report["skipped_reason"] = "disabled"
        return report
    if int(num_new_tokens) <= 0:
        report["skipped_reason"] = "no_new_tokens_added"
        return report

    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()
    if input_embeddings is None or not hasattr(input_embeddings, "weight"):
        report["skipped_reason"] = "missing_input_embeddings"
        return report
    input_weight = input_embeddings.weight.data
    output_weight = output_embeddings.weight.data if output_embeddings is not None and hasattr(output_embeddings, "weight") else None
    vocab_size = input_weight.shape[0]
    report["attempted"] = True

    def alias_vector(alias: str) -> torch.Tensor | None:
        token_ids = tokenizer(alias, add_special_tokens=False).get("input_ids", [])
        token_ids = [int(token_id) for token_id in token_ids if 0 <= int(token_id) < vocab_size]
        if not token_ids:
            return None
        return input_weight[torch.tensor(token_ids, dtype=torch.long, device=input_weight.device)].mean(dim=0)

    for symbol in CHEMICAL_SYMBOLS[1:]:
        special = f"<E_{symbol}>"
        token_id = tokenizer.convert_tokens_to_ids(special)
        if token_id is None or int(token_id) < 0 or int(token_id) >= vocab_size:
            continue
        vectors: list[torch.Tensor] = []
        used_aliases: list[dict[str, Any]] = []
        for alias in element_semantic_aliases(symbol):
            vec = alias_vector(alias)
            tokenized = tokenizer(alias, add_special_tokens=False).get("input_ids", [])
            if vec is None:
                continue
            vectors.append(vec)
            used_aliases.append({"alias": alias, "token_count": len(tokenized)})
        if not vectors:
            report["element_reports"][symbol] = {
                "token": special,
                "token_id": int(token_id),
                "initialized": False,
                "reason": "no_alias_vectors",
            }
            continue
        vector = torch.stack(vectors, dim=0).mean(dim=0).to(dtype=input_weight.dtype)
        input_weight[int(token_id)] = vector
        if output_weight is not None and int(token_id) < output_weight.shape[0]:
            output_weight[int(token_id)] = vector.to(dtype=output_weight.dtype, device=output_weight.device)
        report["initialized_count"] += 1
        report["element_reports"][symbol] = {
            "token": special,
            "token_id": int(token_id),
            "initialized": True,
            "alias_count": len(used_aliases),
            "aliases": used_aliases[:8],
        }
    return report


def write_semantic_init_markdown(path: Path, report: Dict[str, Any]) -> None:
    initialized = [
        (symbol, payload)
        for symbol, payload in sorted(report.get("element_reports", {}).items())
        if payload.get("initialized")
    ]
    lines = [
        "# Element Special-Token Semantic Initialization",
        "",
        f"- enabled: `{bool(report.get('enabled'))}`",
        f"- attempted: `{bool(report.get('attempted'))}`",
        f"- skipped_reason: `{report.get('skipped_reason')}`",
        f"- num_new_tokens: `{report.get('num_new_tokens')}`",
        f"- initialized_count: `{report.get('initialized_count')}`",
        "",
        "| element | token_id | alias_count | first_aliases |",
        "| --- | ---: | ---: | --- |",
    ]
    for symbol, payload in initialized:
        aliases = ", ".join(str(item.get("alias")) for item in payload.get("aliases", [])[:4])
        lines.append(
            f"| {symbol} | {payload.get('token_id')} | {payload.get('alias_count')} | `{aliases}` |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@torch.no_grad()
def build_element_semantic_alignment_targets(
    tokenizer,
    model,
    *,
    device: torch.device,
    symbols: Optional[List[str]] = None,
) -> Dict[str, Any]:
    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()
    if input_embeddings is None or not hasattr(input_embeddings, "weight"):
        return {
            "enabled": False,
            "skipped_reason": "missing_input_embeddings",
            "token_ids": torch.empty(0, dtype=torch.long, device=device),
            "input_targets": torch.empty(0, device=device),
            "output_targets": torch.empty(0, device=device),
            "report": {"enabled": False, "skipped_reason": "missing_input_embeddings"},
        }
    input_weight = input_embeddings.weight.detach()
    output_weight = (
        output_embeddings.weight.detach()
        if output_embeddings is not None and hasattr(output_embeddings, "weight")
        else None
    )
    vocab_size = input_weight.shape[0]
    selected_symbols = symbols or list(CHEMICAL_SYMBOLS[1:])
    token_ids: List[int] = []
    input_targets: List[torch.Tensor] = []
    output_targets: List[torch.Tensor] = []
    element_reports: Dict[str, Any] = {}

    for symbol in selected_symbols:
        special = f"<E_{symbol}>"
        token_id = tokenizer.convert_tokens_to_ids(special)
        if token_id is None or int(token_id) < 0 or int(token_id) >= vocab_size:
            element_reports[symbol] = {"token": special, "matched": False, "reason": "missing_special_token"}
            continue
        alias_input_vectors: List[torch.Tensor] = []
        alias_output_vectors: List[torch.Tensor] = []
        used_aliases: List[Dict[str, Any]] = []
        for alias in element_semantic_aliases(symbol):
            alias_ids = tokenizer(alias, add_special_tokens=False).get("input_ids", [])
            alias_ids = [int(item) for item in alias_ids if 0 <= int(item) < vocab_size]
            if not alias_ids:
                continue
            ids_tensor = torch.tensor(alias_ids, dtype=torch.long, device=input_weight.device)
            alias_input_vectors.append(input_weight[ids_tensor].float().mean(dim=0))
            if output_weight is not None and max(alias_ids) < output_weight.shape[0]:
                alias_output_vectors.append(output_weight[ids_tensor].float().mean(dim=0))
            used_aliases.append({"alias": alias, "token_count": len(alias_ids)})
        if not alias_input_vectors:
            element_reports[symbol] = {"token": special, "token_id": int(token_id), "matched": False, "reason": "no_alias_vectors"}
            continue
        input_target = torch.stack(alias_input_vectors, dim=0).mean(dim=0)
        output_target = (
            torch.stack(alias_output_vectors, dim=0).mean(dim=0)
            if alias_output_vectors
            else input_target
        )
        current_input = input_weight[int(token_id)].float()
        current_output = (
            output_weight[int(token_id)].float()
            if output_weight is not None and int(token_id) < output_weight.shape[0]
            else current_input
        )
        token_ids.append(int(token_id))
        input_targets.append(input_target.to(device=device))
        output_targets.append(output_target.to(device=device))
        element_reports[symbol] = {
            "token": special,
            "token_id": int(token_id),
            "matched": True,
            "alias_count": len(used_aliases),
            "aliases": used_aliases[:8],
            "input_cosine_to_alias_target": float(F.cosine_similarity(current_input, input_target, dim=0).detach().cpu()),
            "output_cosine_to_alias_target": float(F.cosine_similarity(current_output, output_target, dim=0).detach().cpu()),
        }

    if input_targets:
        input_target_tensor = torch.stack(input_targets, dim=0).detach()
        output_target_tensor = torch.stack(output_targets, dim=0).detach()
        token_id_tensor = torch.tensor(token_ids, dtype=torch.long, device=device)
    else:
        hidden = int(input_weight.shape[-1])
        input_target_tensor = torch.empty((0, hidden), dtype=torch.float32, device=device)
        output_target_tensor = torch.empty((0, hidden), dtype=torch.float32, device=device)
        token_id_tensor = torch.empty(0, dtype=torch.long, device=device)
    cosines = [
        payload["input_cosine_to_alias_target"]
        for payload in element_reports.values()
        if payload.get("matched")
    ]
    report = {
        "enabled": True,
        "matched_count": len(token_ids),
        "symbol_count": len(selected_symbols),
        "mean_input_cosine_to_alias_target": (sum(cosines) / len(cosines)) if cosines else None,
        "min_input_cosine_to_alias_target": min(cosines) if cosines else None,
        "element_reports": element_reports,
    }
    return {
        "enabled": True,
        "token_ids": token_id_tensor,
        "input_targets": input_target_tensor,
        "output_targets": output_target_tensor,
        "report": report,
    }


def element_semantic_alignment_loss(
    model,
    targets: Optional[Dict[str, Any]],
    *,
    output_weight: float,
) -> torch.Tensor | None:
    if not targets or int(targets.get("token_ids", torch.empty(0)).numel()) == 0:
        return None
    target_model = model.module if hasattr(model, "module") else model
    input_embeddings = target_model.get_input_embeddings()
    if input_embeddings is None or not hasattr(input_embeddings, "weight"):
        return None
    token_ids = targets["token_ids"].to(input_embeddings.weight.device)
    input_targets = targets["input_targets"].to(input_embeddings.weight.device, dtype=torch.float32)
    current_input = input_embeddings.weight[token_ids].float()
    input_loss = 1.0 - F.cosine_similarity(current_input, input_targets, dim=-1).mean()
    output_loss = input_loss * 0.0
    output_embeddings = target_model.get_output_embeddings()
    if output_weight > 0 and output_embeddings is not None and hasattr(output_embeddings, "weight"):
        output_targets = targets["output_targets"].to(output_embeddings.weight.device, dtype=torch.float32)
        current_output = output_embeddings.weight[token_ids].float()
        output_loss = 1.0 - F.cosine_similarity(current_output, output_targets, dim=-1).mean()
    return input_loss + float(output_weight) * output_loss


def write_element_alignment_markdown(path: Path, report: Dict[str, Any]) -> None:
    rows = []
    for symbol, payload in sorted(report.get("element_reports", {}).items()):
        if payload.get("matched"):
            rows.append((symbol, payload))
    rows.sort(key=lambda item: item[1].get("input_cosine_to_alias_target", 1.0))
    lines = [
        "# Element Token Semantic Alignment",
        "",
        f"- matched_count: `{report.get('matched_count')}`",
        f"- symbol_count: `{report.get('symbol_count')}`",
        f"- mean_input_cosine_to_alias_target: `{report.get('mean_input_cosine_to_alias_target')}`",
        f"- min_input_cosine_to_alias_target: `{report.get('min_input_cosine_to_alias_target')}`",
        "",
        "| element | token_id | input_cos | output_cos | aliases |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for symbol, payload in rows[:40]:
        aliases = ", ".join(str(item.get("alias")) for item in payload.get("aliases", [])[:4])
        lines.append(
            f"| {symbol} | {payload.get('token_id')} | {payload.get('input_cosine_to_alias_target')} | {payload.get('output_cosine_to_alias_target')} | `{aliases}` |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def maybe_apply_lora(model, args, is_main: bool = True):
    if not args.use_lora:
        return model
    from peft import LoraConfig, get_peft_model

    named_modules = list(model.named_modules())
    module_names = [name for name, _ in named_modules]
    requested_targets = [item.strip() for item in args.lora_target_modules.split(",") if item.strip()]
    target_modules = [
        item for item in requested_targets if any(name.endswith(item) for name in module_names)
    ]
    if not target_modules:
        raise RuntimeError(
            "None of the requested LoRA target modules were found. "
            f"Requested={requested_targets[:20]}"
        )
    requested_modules_to_save = [
        item.strip() for item in args.modules_to_save.split(",") if item.strip()
    ]
    modules_to_save = [
        item
        for item in requested_modules_to_save
        if any(name.endswith(item) for name in module_names)
    ]

    def module_name_for(target_module) -> str | None:
        if target_module is None:
            return None
        target_id = id(target_module)
        for name, module in named_modules:
            if name and id(module) == target_id:
                return name
        return None

    # New crystal tokens are added before LoRA wrapping for fixed-slot/dynamic
    # runs. Plain-text CIF-lite/CrysLLMGen/fixed-plain variants deliberately have no new
    # tokens, so avoid silently saving multi-GB embedding/head modules unless
    # the caller explicitly asks for them.
    if getattr(args, "representation", "fixed_slot") not in TEXT_ONLY_REPRESENTATIONS:
        for module_name in (
            module_name_for(model.get_input_embeddings()),
            module_name_for(model.get_output_embeddings()),
        ):
            if module_name and module_name not in modules_to_save:
                modules_to_save.append(module_name)

    missing_modules_to_save = sorted(set(requested_modules_to_save) - set(modules_to_save))
    if missing_modules_to_save:
        print(
            "Warning: modules_to_save not found and will be ignored:",
            ",".join(missing_modules_to_save),
        )
    if is_main and modules_to_save:
        print("Using modules_to_save:", ",".join(modules_to_save))
    config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=target_modules,
        modules_to_save=modules_to_save or None,
    )
    model = get_peft_model(model, config)
    if is_main:
        model.print_trainable_parameters()
    return model


def load_tokenizer_and_model(args, is_main: bool = True):
    checkpoint_path = Path(args.checkpoint_path) if args.checkpoint_path else None
    if checkpoint_path is not None and not checkpoint_path.exists():
        raise FileNotFoundError(f"--checkpoint-path does not exist: {checkpoint_path}")
    has_adapter_checkpoint = bool(
        checkpoint_path is not None and (checkpoint_path / "adapter_config.json").exists()
    )
    tokenizer_source = (
        checkpoint_path
        if checkpoint_path is not None and checkpoint_path.exists()
        else args.model_path
    )
    model_source = args.model_path if has_adapter_checkpoint else tokenizer_source

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    ensure_create_bidirectional_mask()
    model_config = AutoConfig.from_pretrained(model_source, trust_remote_code=True)
    ensure_llada2_rope_parameters(model_config)
    model_class = (
        AutoModelForCausalLM
        if getattr(model_config, "model_type", None) == "llada2_moe"
        else AutoModel
    )
    model = model_class.from_pretrained(
        model_source,
        config=model_config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    semantic_report: Dict[str, Any] = {
        "enabled": bool(getattr(args, "semantic_init_element_tokens", False)),
        "attempted": False,
        "skipped_reason": None,
    }
    if has_adapter_checkpoint:
        from peft import PeftModel

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.resize_token_embeddings(len(tokenizer))
        ensure_llada_vocab_size(model, len(tokenizer))
        skip_resize = bool(args.skip_data_vocab_resize or args.representation in TEXT_ONLY_REPRESENTATIONS)
        missing_data_tokens = []
        if not skip_resize:
            missing_data_tokens = data_vocab_missing_tokens(tokenizer, args.data_dir / "vocab_tokens.txt")
        if missing_data_tokens:
            # PEFT wraps saved embedding/head modules in ModulesToSaveWrapper,
            # and Transformers cannot resize those wrappers. Merge the warm-start
            # adapter into a plain model first, then extend the vocab and attach a
            # fresh trainable LoRA adapter for the new representation.
            model = PeftModel.from_pretrained(
                model,
                str(checkpoint_path),
                is_trainable=False,
            )
            model = model.merge_and_unload()
            if hasattr(model, "peft_config"):
                try:
                    delattr(model, "peft_config")
                except AttributeError:
                    pass
            num_new_tokens = resize_tokenizer_and_model(
                tokenizer,
                model,
                args.data_dir / "vocab_tokens.txt",
            )
            if args.use_lora:
                model = maybe_apply_lora(model, args, is_main=is_main)
            semantic_report["skipped_reason"] = "adapter_checkpoint_resume_merged_for_vocab_resize"
            semantic_report["missing_data_tokens_before_resize"] = len(missing_data_tokens)
            if is_main and hasattr(model, "print_trainable_parameters"):
                model.print_trainable_parameters()
            return tokenizer, model, num_new_tokens, str(tokenizer_source), str(model_source), semantic_report
        model = PeftModel.from_pretrained(
            model,
            str(checkpoint_path),
            is_trainable=args.use_lora,
        )
        if skip_resize:
            ensure_llada_vocab_size(model, len(tokenizer))
            num_new_tokens = 0
        else:
            num_new_tokens = resize_tokenizer_and_model(tokenizer, model, args.data_dir / "vocab_tokens.txt")
        semantic_report["skipped_reason"] = "adapter_checkpoint_resume"
        if is_main and hasattr(model, "print_trainable_parameters"):
            model.print_trainable_parameters()
    else:
        skip_resize = bool(args.skip_data_vocab_resize or args.representation in TEXT_ONLY_REPRESENTATIONS)
        if skip_resize:
            ensure_llada_vocab_size(model, len(tokenizer))
            num_new_tokens = 0
        else:
            num_new_tokens = resize_tokenizer_and_model(tokenizer, model, args.data_dir / "vocab_tokens.txt")
        semantic_report = semantic_init_element_tokens(
            tokenizer,
            model,
            enabled=bool(getattr(args, "semantic_init_element_tokens", False)),
            num_new_tokens=num_new_tokens,
        )
        if args.use_lora:
            model = maybe_apply_lora(model, args, is_main=is_main)
    return tokenizer, model, num_new_tokens, str(tokenizer_source), str(model_source), semantic_report


def init_distributed() -> Dict[str, Any]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed SFT requires CUDA.")
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        rank = dist.get_rank()
    else:
        local_rank = 0
        rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return {
        "distributed": distributed,
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "device": device,
        "is_main": rank == 0,
    }


def forward_process(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_lengths: torch.Tensor,
    mask_policy_ids: torch.Tensor | None = None,
    empty_token_id: int | None = None,
    prefill_slot_tokens: bool = False,
    fixed_slot_body_offset: int = 0,
    mask_id: int = MASK_TOKEN_ID,
    eps: float = 1e-3,
) -> Dict[str, torch.Tensor]:
    bsz, seq_len = input_ids.shape
    device = input_ids.device
    positions = torch.arange(seq_len, device=device).expand(bsz, seq_len)
    answer_mask = (positions >= prompt_lengths.unsqueeze(1)) & attention_mask.bool()

    t = torch.rand(bsz, device=device)
    p_mask = ((1 - eps) * t + eps).unsqueeze(1).expand(bsz, seq_len)
    random_mask = torch.rand((bsz, seq_len), device=device) < p_mask
    candidate_mask = answer_mask.clone()
    rel_positions = positions - prompt_lengths.unsqueeze(1)
    body_offset = int(fixed_slot_body_offset)
    body_rel_positions = rel_positions - body_offset
    in_slot = answer_mask & (body_rel_positions >= 7)
    slot_offsets = body_rel_positions - 7
    field_offsets = slot_offsets.remainder(5)
    slot_marker = in_slot & (field_offsets == 0)
    if prefill_slot_tokens:
        candidate_mask = candidate_mask & ~slot_marker
    if mask_policy_ids is not None and empty_token_id is not None and int(empty_token_id) >= 0:
        occupancy = in_slot & (field_offsets == 1)
        empty_boundary = occupancy & (input_ids == int(empty_token_id))
        n_token = answer_mask & (body_rel_positions == 0)
        for i in range(bsz):
            policy = ID_TO_MASK_POLICY.get(int(mask_policy_ids[i].detach().item()), "normal")
            sample_occupancy = occupancy[i]
            sample_empty = empty_boundary[i]
            if sample_empty.any():
                first_empty_abs = int(torch.nonzero(sample_empty, as_tuple=False).flatten()[0].detach().item())
                active_prefix = sample_occupancy & (positions[i] < first_empty_abs)
                first_empty_boundary = sample_empty & (positions[i] == first_empty_abs)
            else:
                active_prefix = sample_occupancy
                first_empty_boundary = torch.zeros_like(sample_empty)
            active_element = active_prefix & (input_ids[i] != int(empty_token_id))
            if policy == "active_element":
                selected = active_element
            elif policy == "n_active_element":
                selected = n_token[i] | active_element
            elif policy == "active_element_empty":
                selected = active_element | first_empty_boundary
            else:
                selected = answer_mask[i]
            if prefill_slot_tokens:
                selected = selected & ~slot_marker[i]
            if selected.any():
                candidate_mask[i] = selected
    masked_indices = random_mask & candidate_mask

    # Ensure every non-empty answer contributes at least one supervised token.
    for i in range(bsz):
        if candidate_mask[i].any() and not masked_indices[i].any():
            candidate_positions = torch.nonzero(candidate_mask[i], as_tuple=False).flatten()
            masked_indices[i, candidate_positions[torch.randint(len(candidate_positions), (1,), device=device)]] = True

    noisy = torch.where(masked_indices, torch.full_like(input_ids, mask_id), input_ids)
    return {
        "noisy": noisy,
        "masked_indices": masked_indices,
        "p_mask": p_mask,
        "answer_mask": answer_mask,
        "candidate_mask": candidate_mask,
    }


def build_loss_config(tokenizer, args) -> Dict[str, Any]:
    vocab = tokenizer.get_vocab()

    def optional_token_id(token: str) -> int:
        if token not in vocab:
            return -1
        token_id = tokenizer.convert_tokens_to_ids(token)
        return -1 if token_id is None else int(token_id)

    if getattr(args, "representation", "fixed_slot") == "fixed_slot_compressed_v1" and "<C_PAD>" in vocab:
        pad_coord_token_ids = [optional_token_id("<C_PAD>")]
    else:
        pad_coord_token_ids = [
            optional_token_id(token)
            for token in ("<X_PAD>", "<Y_PAD>", "<Z_PAD>")
        ]
    return {
        "representation": getattr(args, "representation", "fixed_slot"),
        "answer_token_count": int(getattr(args, "answer_token_count", ANSWER_TOKEN_COUNT)),
        "fixed_slot_body_offset": (
            PHYSICAL_HEADER_BODY_OFFSET
            if getattr(args, "representation", "fixed_slot") == "fixed_slot_physical_header"
            else 0
        ),
        "physical_header_loss_weight": float(getattr(args, "physical_header_loss_weight", 2.0)),
        "atom_count_loss_weight": args.atom_count_loss_weight,
        "slot_marker_loss_weight": args.slot_marker_loss_weight,
        "empty_slot_loss_weight": args.empty_slot_loss_weight,
        "nonempty_slot_loss_weight": args.nonempty_slot_loss_weight,
        "late_slot_start": args.late_slot_start,
        "late_nonempty_slot_loss_weight": (
            args.nonempty_slot_loss_weight
            if args.late_nonempty_slot_loss_weight is None
            else args.late_nonempty_slot_loss_weight
        ),
        "coordinate_loss_weight": args.coordinate_loss_weight,
        "pad_coordinate_loss_weight": args.pad_coordinate_loss_weight,
        "train_prefill_slot_tokens": bool(getattr(args, "train_prefill_slot_tokens", False)),
        "empty_token_id": optional_token_id("<EMPTY>"),
        "pad_coord_token_ids": pad_coord_token_ids,
        "composition_module_loss_weight": float(getattr(args, "composition_module_loss_weight", 2.0)),
        "lattice_module_loss_weight": float(getattr(args, "lattice_module_loss_weight", 1.0)),
        "sites_module_loss_weight": float(getattr(args, "sites_module_loss_weight", 1.25)),
        "crysllmgen_lattice_loss_weight": float(getattr(args, "crysllmgen_lattice_loss_weight", 1.0)),
        "crysllmgen_composition_loss_weight": float(getattr(args, "crysllmgen_composition_loss_weight", 2.5)),
        "crysllmgen_species_loss_weight": float(getattr(args, "crysllmgen_species_loss_weight", 2.0)),
        "crysllmgen_coords_loss_weight": float(getattr(args, "crysllmgen_coords_loss_weight", 1.1)),
        "crysllmgen_site_coord_loss_weight": float(getattr(args, "crysllmgen_site_coord_loss_weight", 1.0)),
        "fixed_plain_count_loss_weight": float(getattr(args, "fixed_plain_count_loss_weight", 3.0)),
        "fixed_plain_lattice_loss_weight": float(getattr(args, "fixed_plain_lattice_loss_weight", 1.0)),
        "fixed_plain_elements_loss_weight": float(getattr(args, "fixed_plain_elements_loss_weight", 2.0)),
        "fixed_plain_coords_loss_weight": float(getattr(args, "fixed_plain_coords_loss_weight", 1.1)),
        "dynamic_lattice_length_loss_weight": float(getattr(args, "dynamic_lattice_length_loss_weight", 1.0)),
        "dynamic_lattice_angle_loss_weight": float(getattr(args, "dynamic_lattice_angle_loss_weight", 1.0)),
        "dynamic_coord_loss_weight": float(
            getattr(args, "dynamic_coord_loss_weight", getattr(args, "coordinate_loss_weight", 1.0))
        ),
    }


def answer_position_ids(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_lengths: torch.Tensor,
    answer_token_count: int = ANSWER_TOKEN_COUNT,
) -> tuple[torch.Tensor, torch.Tensor]:
    bsz, seq_len = input_ids.shape
    positions = torch.arange(seq_len, device=input_ids.device).expand(bsz, seq_len)
    rel_positions = positions - prompt_lengths.unsqueeze(1)
    answer_mask = (
        (rel_positions >= 0)
        & (rel_positions < int(answer_token_count))
        & attention_mask.bool()
    )
    return rel_positions, answer_mask


def build_answer_position_weights(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_lengths: torch.Tensor,
    loss_config: Dict[str, Any],
    module_ids: torch.Tensor | None = None,
    loss_profile_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    rel_positions, answer_mask = answer_position_ids(
        input_ids,
        attention_mask,
        prompt_lengths,
        answer_token_count=int(loss_config.get("answer_token_count", ANSWER_TOKEN_COUNT)),
    )
    weights = torch.zeros(input_ids.shape, dtype=torch.float32, device=input_ids.device)
    weights = torch.where(answer_mask, torch.ones_like(weights), weights)
    text_profile_mask = None
    if loss_profile_ids is not None:
        text_profile_mask = (
            loss_profile_ids.to(input_ids.device).unsqueeze(1)
            == int(LOSS_PROFILE_TO_ID["text"])
        )

    if loss_config.get("representation") == "cif_lite_modular":
        if module_ids is not None:
            sample_weights = torch.ones((input_ids.shape[0],), dtype=torch.float32, device=input_ids.device)
            sample_weights = torch.where(
                module_ids.to(input_ids.device) == int(MODULE_TO_ID["composition"]),
                torch.full_like(sample_weights, float(loss_config["composition_module_loss_weight"])),
                sample_weights,
            )
            sample_weights = torch.where(
                module_ids.to(input_ids.device) == int(MODULE_TO_ID["lattice"]),
                torch.full_like(sample_weights, float(loss_config["lattice_module_loss_weight"])),
                sample_weights,
            )
            sample_weights = torch.where(
                module_ids.to(input_ids.device) == int(MODULE_TO_ID["sites"]),
                torch.full_like(sample_weights, float(loss_config["sites_module_loss_weight"])),
                sample_weights,
            )
            weights = weights * sample_weights.unsqueeze(1)
        if text_profile_mask is not None:
            weights = torch.where(text_profile_mask & answer_mask, torch.ones_like(weights), weights)
        return weights

    if loss_config.get("representation") == "crysllmgen_text":
        if module_ids is not None:
            module_ids = module_ids.to(input_ids.device)
            sample_weights = torch.ones((input_ids.shape[0],), dtype=torch.float32, device=input_ids.device)
            sample_weights = torch.where(
                module_ids == int(CRYSLLMGEN_MODULE_TO_ID["lattice"]),
                torch.full_like(sample_weights, float(loss_config["crysllmgen_lattice_loss_weight"])),
                sample_weights,
            )
            sample_weights = torch.where(
                module_ids == int(CRYSLLMGEN_MODULE_TO_ID["composition"]),
                torch.full_like(sample_weights, float(loss_config["crysllmgen_composition_loss_weight"])),
                sample_weights,
            )
            sample_weights = torch.where(
                module_ids == int(CRYSLLMGEN_MODULE_TO_ID["species"]),
                torch.full_like(sample_weights, float(loss_config["crysllmgen_species_loss_weight"])),
                sample_weights,
            )
            sample_weights = torch.where(
                module_ids == int(CRYSLLMGEN_MODULE_TO_ID["coords"]),
                torch.full_like(sample_weights, float(loss_config["crysllmgen_coords_loss_weight"])),
                sample_weights,
            )
            sample_weights = torch.where(
                module_ids == int(CRYSLLMGEN_MODULE_TO_ID["site_coord"]),
                torch.full_like(sample_weights, float(loss_config["crysllmgen_site_coord_loss_weight"])),
                sample_weights,
            )
            weights = weights * sample_weights.unsqueeze(1)
        if text_profile_mask is not None:
            weights = torch.where(text_profile_mask & answer_mask, torch.ones_like(weights), weights)
        return weights

    if loss_config.get("representation") == "fixed_plain":
        if module_ids is not None:
            module_ids = module_ids.to(input_ids.device)
            sample_weights = torch.ones((input_ids.shape[0],), dtype=torch.float32, device=input_ids.device)
            sample_weights = torch.where(
                module_ids == int(FIXED_PLAIN_MODULE_TO_ID["count"]),
                torch.full_like(sample_weights, float(loss_config["fixed_plain_count_loss_weight"])),
                sample_weights,
            )
            sample_weights = torch.where(
                module_ids == int(FIXED_PLAIN_MODULE_TO_ID["lattice"]),
                torch.full_like(sample_weights, float(loss_config["fixed_plain_lattice_loss_weight"])),
                sample_weights,
            )
            sample_weights = torch.where(
                module_ids == int(FIXED_PLAIN_MODULE_TO_ID["elements"]),
                torch.full_like(sample_weights, float(loss_config["fixed_plain_elements_loss_weight"])),
                sample_weights,
            )
            sample_weights = torch.where(
                module_ids == int(FIXED_PLAIN_MODULE_TO_ID["coords"]),
                torch.full_like(sample_weights, float(loss_config["fixed_plain_coords_loss_weight"])),
                sample_weights,
            )
            weights = weights * sample_weights.unsqueeze(1)
        if text_profile_mask is not None:
            weights = torch.where(text_profile_mask & answer_mask, torch.ones_like(weights), weights)
        return weights

    if loss_config.get("representation") in {"r5_plan_state", "r5_repair_text"}:
        return torch.where(answer_mask, torch.ones_like(weights), weights)

    body_offset = int(loss_config.get("fixed_slot_body_offset", 0))
    body_rel_positions = rel_positions - body_offset
    if body_offset > 0:
        header_mask = answer_mask & (rel_positions >= 0) & (rel_positions < body_offset)
        weights = torch.where(
            header_mask,
            torch.full_like(weights, float(loss_config.get("physical_header_loss_weight", 2.0))),
            weights,
        )

    weights = torch.where(
        answer_mask & (body_rel_positions == 0),
        torch.full_like(weights, float(loss_config["atom_count_loss_weight"])),
        weights,
    )
    if loss_config.get("representation") == "dynamic_v1":
        weights = torch.where(
            answer_mask & (body_rel_positions >= 1) & (body_rel_positions <= 3),
            torch.full_like(weights, float(loss_config["dynamic_lattice_length_loss_weight"])),
            weights,
        )
        weights = torch.where(
            answer_mask & (body_rel_positions >= 4) & (body_rel_positions <= 6),
            torch.full_like(weights, float(loss_config["dynamic_lattice_angle_loss_weight"])),
            weights,
        )
        in_site = answer_mask & (body_rel_positions >= 7)
        site_offsets = body_rel_positions - 7
        field_offsets = site_offsets.remainder(4)
        weights = torch.where(
            in_site & (field_offsets == 0),
            torch.full_like(weights, float(loss_config["nonempty_slot_loss_weight"])),
            weights,
        )
        weights = torch.where(
            in_site & (field_offsets >= 1),
            torch.full_like(weights, float(loss_config["dynamic_coord_loss_weight"])),
            weights,
        )
        if text_profile_mask is not None:
            weights = torch.where(text_profile_mask & answer_mask, torch.ones_like(weights), weights)
        return weights

    in_slot = answer_mask & (body_rel_positions >= 7)
    slot_offsets = body_rel_positions - 7
    slot_indices = torch.div(slot_offsets, 5, rounding_mode="floor")
    field_offsets = slot_offsets.remainder(5)

    weights = torch.where(
        in_slot & (field_offsets == 0),
        torch.full_like(weights, float(loss_config["slot_marker_loss_weight"])),
        weights,
    )

    is_occupancy = in_slot & (field_offsets == 1)
    is_empty = input_ids == int(loss_config["empty_token_id"])
    is_nonempty = is_occupancy & ~is_empty
    is_late_nonempty = is_nonempty & (slot_indices >= int(loss_config["late_slot_start"]))
    weights = torch.where(
        is_occupancy & is_empty,
        torch.full_like(weights, float(loss_config["empty_slot_loss_weight"])),
        weights,
    )
    weights = torch.where(
        is_nonempty,
        torch.full_like(weights, float(loss_config["nonempty_slot_loss_weight"])),
        weights,
    )
    weights = torch.where(
        is_late_nonempty,
        torch.full_like(weights, float(loss_config["late_nonempty_slot_loss_weight"])),
        weights,
    )

    is_coordinate = in_slot & (field_offsets >= 2)
    pad_coord_token_ids = torch.tensor(
        loss_config["pad_coord_token_ids"],
        dtype=input_ids.dtype,
        device=input_ids.device,
    )
    is_pad_coord = is_coordinate & torch.isin(input_ids, pad_coord_token_ids)
    weights = torch.where(
        is_coordinate & ~is_pad_coord,
        torch.full_like(weights, float(loss_config["coordinate_loss_weight"])),
        weights,
    )
    weights = torch.where(
        is_pad_coord,
        torch.full_like(weights, float(loss_config["pad_coordinate_loss_weight"])),
        weights,
    )
    if text_profile_mask is not None:
        weights = torch.where(text_profile_mask & answer_mask, torch.ones_like(weights), weights)
    return weights


def compute_loss(model, batch: Dict[str, torch.Tensor], loss_config: Dict[str, Any]) -> torch.Tensor:
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    prompt_lengths = batch["prompt_lengths"]
    processed = forward_process(
        input_ids,
        attention_mask,
        prompt_lengths,
        mask_policy_ids=batch.get("mask_policy_ids"),
        empty_token_id=int(loss_config.get("empty_token_id", -1)),
        prefill_slot_tokens=bool(loss_config.get("train_prefill_slot_tokens", False)),
        fixed_slot_body_offset=int(loss_config.get("fixed_slot_body_offset", 0)),
    )
    outputs = model(input_ids=processed["noisy"], attention_mask=attention_mask)
    masked = processed["masked_indices"]
    if not masked.any():
        return outputs.logits.sum() * 0.0
    max_target = int(input_ids[masked].max().detach().cpu())
    if max_target >= outputs.logits.shape[-1]:
        raise RuntimeError(
            f"Target token id {max_target} exceeds logits vocab size {outputs.logits.shape[-1]}. "
            "Run ensure_llada_vocab_size after adding crystal special tokens."
        )
    token_loss = F.cross_entropy(outputs.logits[masked], input_ids[masked], reduction="none")
    token_loss = token_loss / processed["p_mask"][masked]
    loss_weights = build_answer_position_weights(
        input_ids,
        attention_mask,
        prompt_lengths,
        loss_config,
        module_ids=batch.get("module_ids"),
        loss_profile_ids=batch.get("loss_profile_ids"),
    )
    if loss_config.get("representation") in {"cif_lite_modular", "crysllmgen_text", "fixed_plain"}:
        # CIF-lite module weights are intentionally sample-level weights.  Do
        # not include them in the denominator, otherwise composition/lattice/site
        # module scaling cancels out for each example.
        sample_weight_sums = processed["candidate_mask"].sum(dim=1).to(torch.float32).clamp_min(1.0)
    else:
        sample_weight_sums = (loss_weights * processed["candidate_mask"]).sum(dim=1).clamp_min(1.0)
    per_token_norm = sample_weight_sums.unsqueeze(1).expand_as(input_ids)[masked]
    weighted_token_loss = token_loss * loss_weights[masked] / per_token_norm
    sample_indices = torch.arange(input_ids.shape[0], device=input_ids.device).unsqueeze(1).expand_as(input_ids)[masked]
    sample_loss = torch.zeros((input_ids.shape[0],), dtype=weighted_token_loss.dtype, device=input_ids.device)
    sample_loss.scatter_add_(0, sample_indices, weighted_token_loss)
    sample_weights = batch.get("sample_weights")
    if sample_weights is None:
        return sample_loss.sum() / input_ids.shape[0]
    sample_weights = sample_weights.to(device=input_ids.device, dtype=sample_loss.dtype).clamp_min(0.0)
    return (sample_loss * sample_weights).sum() / sample_weights.sum().clamp_min(1.0)


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    max_batches: int,
    loss_config: Dict[str, Any],
    distributed: bool = False,
) -> float:
    model.eval()
    losses = []
    for idx, batch in enumerate(loader):
        if idx >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        losses.append(float(compute_loss(model, batch, loss_config).detach().cpu()))
    if distributed:
        payload = torch.tensor([sum(losses), len(losses)], dtype=torch.float64, device=device)
        dist.all_reduce(payload, op=dist.ReduceOp.SUM)
        result = float((payload[0] / payload[1].clamp_min(1.0)).detach().cpu())
    else:
        result = sum(losses) / max(1, len(losses))
    model.train()
    return result


@torch.no_grad()
def evaluate_position_diagnostics(
    model,
    loader,
    device,
    max_batches: int,
    loss_config: Dict[str, Any],
    distributed: bool = False,
) -> Dict[str, Any]:
    model.eval()
    answer_token_count = int(loss_config["answer_token_count"])
    position_totals = torch.zeros(answer_token_count, dtype=torch.float64, device=device)
    position_counts = torch.zeros(answer_token_count, dtype=torch.float64, device=device)
    if loss_config.get("representation") == "dynamic_v1":
        group_names = [
            "forced_N",
            "free_lattice_length",
            "free_lattice_angle",
            "forced_element",
            "free_coord_x",
            "free_coord_y",
            "free_coord_z",
        ]
    elif loss_config.get("representation") == "cif_lite_modular":
        group_names = ["composition_module", "lattice_module", "sites_module"]
    elif loss_config.get("representation") == "crysllmgen_text":
        group_names = [
            "full_module",
            "lattice_module",
            "composition_module",
            "species_module",
            "coords_module",
            "site_coord_module",
        ]
    elif loss_config.get("representation") == "fixed_plain":
        group_names = ["count_module", "lattice_module", "elements_module", "coords_module"]
    elif loss_config.get("representation") in {"r5_plan_state", "r5_repair_text"}:
        group_names = ["text_answer"]
    elif loss_config.get("representation") == "fixed_slot_physical_header":
        group_names = [
            "physical_header",
            "atom_count",
            "lattice",
            "slot_marker",
            "occupancy_empty",
            "occupancy_nonempty",
            "occupancy_nonempty_late",
            "coordinate_real",
            "coordinate_pad",
        ]
    else:
        group_names = [
            "atom_count",
            "lattice",
            "slot_marker",
            "occupancy_empty",
            "occupancy_nonempty",
            "occupancy_nonempty_late",
            "coordinate_real",
            "coordinate_pad",
        ]
    group_totals = torch.zeros(len(group_names), dtype=torch.float64, device=device)
    group_counts = torch.zeros(len(group_names), dtype=torch.float64, device=device)
    for idx, batch in enumerate(loader):
        if idx >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        prompt_lengths = batch["prompt_lengths"]
        rel_positions, answer_mask = answer_position_ids(
            input_ids,
            attention_mask,
            prompt_lengths,
            answer_token_count=answer_token_count,
        )
        body_offset = int(loss_config.get("fixed_slot_body_offset", 0))
        body_rel_positions = rel_positions - body_offset
        if bool(loss_config.get("train_prefill_slot_tokens", False)):
            in_slot_for_prefill = answer_mask & (body_rel_positions >= 7)
            slot_marker_for_prefill = in_slot_for_prefill & (((body_rel_positions - 7).remainder(5)) == 0)
            answer_mask = answer_mask & ~slot_marker_for_prefill
        noisy = torch.where(answer_mask, torch.full_like(input_ids, MASK_TOKEN_ID), input_ids)
        outputs = model(input_ids=noisy, attention_mask=attention_mask)
        losses = F.cross_entropy(
            outputs.logits[answer_mask],
            input_ids[answer_mask],
            reduction="none",
        ).to(torch.float64)
        rel_answer = rel_positions[answer_mask].to(torch.long)
        position_totals.scatter_add_(0, rel_answer, losses)
        position_counts.scatter_add_(0, rel_answer, torch.ones_like(losses))

        if loss_config.get("representation") == "cif_lite_modular":
            flat_answer_losses = torch.zeros(input_ids.shape, dtype=torch.float64, device=device)
            flat_answer_losses[answer_mask] = losses
            module_ids = batch.get("module_ids")
            if module_ids is None:
                module_ids = torch.zeros((input_ids.shape[0],), dtype=torch.long, device=device)
            module_ids = module_ids.to(device)
            group_masks = [
                answer_mask & (module_ids.unsqueeze(1) == MODULE_TO_ID["composition"]),
                answer_mask & (module_ids.unsqueeze(1) == MODULE_TO_ID["lattice"]),
                answer_mask & (module_ids.unsqueeze(1) == MODULE_TO_ID["sites"]),
            ]
            for group_idx, group_mask in enumerate(group_masks):
                group_totals[group_idx] += flat_answer_losses[group_mask].sum()
                group_counts[group_idx] += group_mask.sum()
            continue

        if loss_config.get("representation") == "crysllmgen_text":
            flat_answer_losses = torch.zeros(input_ids.shape, dtype=torch.float64, device=device)
            flat_answer_losses[answer_mask] = losses
            module_ids = batch.get("module_ids")
            if module_ids is None:
                module_ids = torch.zeros((input_ids.shape[0],), dtype=torch.long, device=device)
            module_ids = module_ids.to(device)
            group_masks = [
                answer_mask & (module_ids.unsqueeze(1) == CRYSLLMGEN_MODULE_TO_ID["full"]),
                answer_mask & (module_ids.unsqueeze(1) == CRYSLLMGEN_MODULE_TO_ID["lattice"]),
                answer_mask & (module_ids.unsqueeze(1) == CRYSLLMGEN_MODULE_TO_ID["composition"]),
                answer_mask & (module_ids.unsqueeze(1) == CRYSLLMGEN_MODULE_TO_ID["species"]),
                answer_mask & (module_ids.unsqueeze(1) == CRYSLLMGEN_MODULE_TO_ID["coords"]),
                answer_mask & (module_ids.unsqueeze(1) == CRYSLLMGEN_MODULE_TO_ID["site_coord"]),
            ]
            for group_idx, group_mask in enumerate(group_masks):
                group_totals[group_idx] += flat_answer_losses[group_mask].sum()
                group_counts[group_idx] += group_mask.sum()
            continue

        if loss_config.get("representation") == "fixed_plain":
            flat_answer_losses = torch.zeros(input_ids.shape, dtype=torch.float64, device=device)
            flat_answer_losses[answer_mask] = losses
            module_ids = batch.get("module_ids")
            if module_ids is None:
                module_ids = torch.zeros((input_ids.shape[0],), dtype=torch.long, device=device)
            module_ids = module_ids.to(device)
            group_masks = [
                answer_mask & (module_ids.unsqueeze(1) == FIXED_PLAIN_MODULE_TO_ID["count"]),
                answer_mask & (module_ids.unsqueeze(1) == FIXED_PLAIN_MODULE_TO_ID["lattice"]),
                answer_mask & (module_ids.unsqueeze(1) == FIXED_PLAIN_MODULE_TO_ID["elements"]),
                answer_mask & (module_ids.unsqueeze(1) == FIXED_PLAIN_MODULE_TO_ID["coords"]),
            ]
            for group_idx, group_mask in enumerate(group_masks):
                group_totals[group_idx] += flat_answer_losses[group_mask].sum()
                group_counts[group_idx] += group_mask.sum()
            continue

        if loss_config.get("representation") in {"r5_plan_state", "r5_repair_text"}:
            group_totals[0] += losses.sum()
            group_counts[0] += answer_mask.sum()
            continue

        if loss_config.get("representation") == "dynamic_v1":
            in_site = answer_mask & (rel_positions >= 7)
            site_offsets = rel_positions - 7
            field_offsets = site_offsets.remainder(4)
            group_masks = [
                answer_mask & (rel_positions == 0),
                answer_mask & (rel_positions >= 1) & (rel_positions <= 3),
                answer_mask & (rel_positions >= 4) & (rel_positions <= 6),
                in_site & (field_offsets == 0),
                in_site & (field_offsets == 1),
                in_site & (field_offsets == 2),
                in_site & (field_offsets == 3),
            ]
            flat_answer_losses = torch.zeros(input_ids.shape, dtype=torch.float64, device=device)
            flat_answer_losses[answer_mask] = losses
            for group_idx, group_mask in enumerate(group_masks):
                group_totals[group_idx] += flat_answer_losses[group_mask].sum()
                group_counts[group_idx] += group_mask.sum()
            continue

        in_slot = answer_mask & (body_rel_positions >= 7)
        slot_offsets = body_rel_positions - 7
        slot_indices = torch.div(slot_offsets, 5, rounding_mode="floor")
        field_offsets = slot_offsets.remainder(5)
        empty_token_id = int(loss_config["empty_token_id"])
        pad_coord_token_ids = torch.tensor(
            loss_config["pad_coord_token_ids"],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )

        group_masks = []
        if body_offset > 0:
            group_masks.append(answer_mask & (rel_positions >= 0) & (rel_positions < body_offset))
        group_masks.extend(
            [
                answer_mask & (body_rel_positions == 0),
                answer_mask & (body_rel_positions >= 1) & (body_rel_positions <= 6),
                in_slot & (field_offsets == 0),
                in_slot & (field_offsets == 1) & (input_ids == empty_token_id),
                in_slot & (field_offsets == 1) & (input_ids != empty_token_id),
                in_slot
                & (field_offsets == 1)
                & (input_ids != empty_token_id)
                & (slot_indices >= int(loss_config["late_slot_start"])),
                in_slot
                & (field_offsets >= 2)
                & ~torch.isin(input_ids, pad_coord_token_ids),
                in_slot
                & (field_offsets >= 2)
                & torch.isin(input_ids, pad_coord_token_ids),
            ]
        )
        flat_answer_losses = torch.zeros(input_ids.shape, dtype=torch.float64, device=device)
        flat_answer_losses[answer_mask] = losses
        for group_idx, group_mask in enumerate(group_masks):
            group_totals[group_idx] += flat_answer_losses[group_mask].sum()
            group_counts[group_idx] += group_mask.sum()

    if distributed:
        dist.all_reduce(position_totals, op=dist.ReduceOp.SUM)
        dist.all_reduce(position_counts, op=dist.ReduceOp.SUM)
        dist.all_reduce(group_totals, op=dist.ReduceOp.SUM)
        dist.all_reduce(group_counts, op=dist.ReduceOp.SUM)

    position_ce = {}
    for idx in range(answer_token_count):
        count = float(position_counts[idx].detach().cpu())
        if count:
            position_ce[str(idx)] = float((position_totals[idx] / position_counts[idx]).detach().cpu())
    group_ce = {}
    for idx, name in enumerate(group_names):
        count = float(group_counts[idx].detach().cpu())
        group_ce[name] = {
            "count": count,
            "ce": None
            if count == 0
            else float((group_totals[idx] / group_counts[idx]).detach().cpu()),
        }
    group_summary: Dict[str, Any] = {}
    if "slot_marker" in group_names:
        slot_idx = group_names.index("slot_marker")
        hard_mask = torch.ones(len(group_names), dtype=torch.bool, device=device)
        hard_mask[slot_idx] = False
        hard_total = group_totals[hard_mask].sum()
        hard_count = group_counts[hard_mask].sum()
        all_total = group_totals.sum()
        all_count = group_counts.sum()
        group_summary = {
            "all_token_ce": None
            if float(all_count.detach().cpu()) == 0.0
            else float((all_total / all_count.clamp_min(1.0)).detach().cpu()),
            "slot_marker_free_ce": None
            if float(hard_count.detach().cpu()) == 0.0
            else float((hard_total / hard_count.clamp_min(1.0)).detach().cpu()),
            "slot_marker_count": float(group_counts[slot_idx].detach().cpu()),
            "slot_marker_fraction": None
            if float(all_count.detach().cpu()) == 0.0
            else float((group_counts[slot_idx] / all_count.clamp_min(1.0)).detach().cpu()),
        }
    model.train()
    return {"position_ce": position_ce, "group_ce": group_ce, "group_ce_summary": group_summary}


def save_model_pretrained(model, output_dir: Path, save_embedding_layers: str = "auto") -> None:
    kwargs = {}
    if save_embedding_layers == "true":
        kwargs["save_embedding_layers"] = True
    elif save_embedding_layers == "false":
        kwargs["save_embedding_layers"] = False
    try:
        model.save_pretrained(output_dir, **kwargs)
    except TypeError:
        model.save_pretrained(output_dir)


def save_checkpoint(
    model,
    tokenizer,
    output_dir: Path,
    step: int,
    save_embedding_layers: str = "auto",
    data_dir: Path | None = None,
    is_main: bool = True,
) -> None:
    if not is_main:
        return
    checkpoint_dir = output_dir / "checkpoints" / f"step-{step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    target_model = model.module if hasattr(model, "module") else model
    save_model_pretrained(target_model, checkpoint_dir, save_embedding_layers=save_embedding_layers)
    tokenizer.save_pretrained(checkpoint_dir)
    if data_dir is not None:
        for name in ("compressed_token_config.json", "token_map.json", "physical_header_config.json"):
            source = data_dir / name
            if source.exists():
                (checkpoint_dir / name).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


def build_lr_scheduler(optimizer, args, total_steps: int):
    warmup_steps = max(0, int(args.warmup_steps))

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(1e-8, float(step + 1) / float(warmup_steps))
        if args.lr_scheduler == "constant":
            return 1.0
        if args.lr_scheduler == "cosine":
            decay_steps = max(1, total_steps - warmup_steps)
            progress = min(1.0, max(0.0, float(step - warmup_steps) / float(decay_steps)))
            return max(float(args.min_lr_ratio), 0.5 * (1.0 + math.cos(math.pi * progress)))
        raise ValueError(f"Unsupported lr_scheduler={args.lr_scheduler!r}")

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def infer_answer_token_count(data_dir: Path) -> int:
    """Infer fixed answer length from dataset metadata when available."""

    for filename in ("stats.json", "feasibility_metrics.json"):
        path = data_dir / filename
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if payload.get("representation") == "dynamic_v1":
                    return int(payload.get("max_answer_token_count") or DYNAMIC_MAX_ANSWER_TOKEN_COUNT)
                if payload.get("representation") == "cif_lite_modular":
                    return int(payload.get("answer_token_count") or (payload.get("max_answer_model_length", 512) + 8))
                if payload.get("representation") == "crysllmgen_text":
                    return int(payload.get("answer_token_count") or (payload.get("max_answer_model_length", 512) + 8))
                if payload.get("representation") == "fixed_plain":
                    return int(payload.get("answer_token_count") or (payload.get("max_answer_model_length", 512) + 8))
                if payload.get("representation") in {"r5_plan_state", "r5_repair_text"}:
                    return int(payload.get("answer_token_count") or (payload.get("max_answer_model_length", 512) + 8))
                value = payload.get("answer_token_count")
                if value:
                    return int(value)
            except Exception:
                pass
    return ANSWER_TOKEN_COUNT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Optional existing full checkpoint or PEFT adapter to continue SFT from.",
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--representation",
        choices=[
            "fixed_slot",
            "fixed_slot_compressed_v1",
            "fixed_slot_physical_header",
            "dynamic_v1",
            "cif_lite_modular",
            "crysllmgen_text",
            "fixed_plain",
            "r5_plan_state",
            "r5_repair_text",
        ],
        default="fixed_slot",
        help="Answer representation used by the SFT data.",
    )
    parser.add_argument(
        "--train-csv-dir",
        type=Path,
        default=None,
        help="Optional MP-20 CSV directory for online CrysLLMGen-style train augmentation.",
    )
    parser.add_argument(
        "--answer-separator",
        default="",
        help="Separator used when dynamically encoding CSV structures. Empty keeps compact answer tokens.",
    )
    parser.add_argument(
        "--augment-origin-shift",
        action="store_true",
        help="Apply CrysLLMGen-style global fractional origin shift to train CSV rows each epoch.",
    )
    parser.add_argument(
        "--niggli",
        action="store_true",
        help="Use pymatgen reduced structure before dynamic fixed-slot encoding.",
    )
    parser.add_argument(
        "--primitive",
        action="store_true",
        help="Use primitive structure before dynamic fixed-slot encoding.",
    )
    parser.add_argument("--data-seed", type=int, default=20260515)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument(
        "--answer-token-count",
        type=int,
        default=None,
        help="Semantic answer length. Defaults to stats.json/feasibility_metrics.json or 107.",
    )
    parser.add_argument(
        "--skip-data-vocab-resize",
        action="store_true",
        help="Do not add data_dir/vocab_tokens.txt. Use only when the checkpoint tokenizer already contains all answer tokens.",
    )
    parser.add_argument(
        "--semantic-init-element-tokens",
        action="store_true",
        help=(
            "When adding fixed-slot crystal tokens from a raw base model, initialize <E_*> "
            "rows from existing element-name/symbol text embeddings. Skipped for adapter resume."
        ),
    )
    parser.add_argument(
        "--element-token-alignment-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Add a cosine regularizer that keeps <E_*> embeddings close to existing "
            "element symbol/name/oxide/ion text embeddings. Works for raw starts and adapter resumes."
        ),
    )
    parser.add_argument(
        "--element-token-alignment-output-weight",
        type=float,
        default=1.0,
        help="Relative weight for aligning output/head rows in the element-token semantic regularizer.",
    )
    parser.add_argument(
        "--save-embedding-layers",
        choices=["auto", "true", "false"],
        default="auto",
        help="Forward PEFT save_embedding_layers to save_pretrained. Use false only when the base/checkpoint tokenizer and embeddings are supplied elsewhere.",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--max-train-steps",
        type=int,
        default=0,
        help="Optional optimizer-step cap. 0 means run all requested epochs.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-scheduler", choices=["constant", "cosine"], default="constant")
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--eval-max-batches", type=int, default=50)
    parser.add_argument("--position-diagnostics-steps", type=int, default=0)
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    parser.add_argument(
        "--weighted-sampling",
        action="store_true",
        help="Use per-row sample_weight from JSONL as a replacement sampling distribution for train rows.",
    )
    parser.add_argument(
        "--weighted-sampling-power",
        type=float,
        default=1.0,
        help="Raise JSONL sample_weight to this power before weighted sampling.",
    )
    parser.add_argument(
        "--sample-weight-multipliers",
        default="",
        help=(
            "Optional comma-separated multipliers applied before weighted sampling, "
            "for example strict=1.0,all_metal=0.7,single_element=0.05,invalid=0.6. "
            "Keys may also target composition_reason, sample_weight_tier, or field:value."
        ),
    )
    parser.add_argument(
        "--ignore-jsonl-sample-weight",
        action="store_true",
        help="Use uniform base train row weights before applying --sample-weight-multipliers.",
    )
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--atom-count-loss-weight", type=float, default=1.0)
    parser.add_argument("--slot-marker-loss-weight", type=float, default=1.0)
    parser.add_argument("--empty-slot-loss-weight", type=float, default=1.0)
    parser.add_argument("--nonempty-slot-loss-weight", type=float, default=1.0)
    parser.add_argument("--late-slot-start", type=int, default=4)
    parser.add_argument("--late-nonempty-slot-loss-weight", type=float, default=None)
    parser.add_argument("--coordinate-loss-weight", type=float, default=1.0)
    parser.add_argument("--pad-coordinate-loss-weight", type=float, default=1.0)
    parser.add_argument("--physical-header-loss-weight", type=float, default=2.0)
    parser.add_argument("--composition-module-loss-weight", type=float, default=2.0)
    parser.add_argument("--lattice-module-loss-weight", type=float, default=1.0)
    parser.add_argument("--sites-module-loss-weight", type=float, default=1.25)
    parser.add_argument("--crysllmgen-lattice-loss-weight", type=float, default=1.0)
    parser.add_argument("--crysllmgen-composition-loss-weight", type=float, default=2.5)
    parser.add_argument("--crysllmgen-species-loss-weight", type=float, default=2.0)
    parser.add_argument("--crysllmgen-coords-loss-weight", type=float, default=1.1)
    parser.add_argument("--crysllmgen-site-coord-loss-weight", type=float, default=1.0)
    parser.add_argument("--fixed-plain-count-loss-weight", type=float, default=3.0)
    parser.add_argument("--fixed-plain-lattice-loss-weight", type=float, default=1.0)
    parser.add_argument("--fixed-plain-elements-loss-weight", type=float, default=2.0)
    parser.add_argument("--fixed-plain-coords-loss-weight", type=float, default=1.1)
    parser.add_argument("--dynamic-lattice-length-loss-weight", type=float, default=1.0)
    parser.add_argument("--dynamic-lattice-angle-loss-weight", type=float, default=1.0)
    parser.add_argument("--dynamic-coord-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--train-prefill-slot-tokens",
        action="store_true",
        help=(
            "Keep fixed slot marker tokens visible during SFT denoising instead of "
            "masking/predicting them. This matches sampling with --prefill-slot-tokens."
        ),
    )
    parser.add_argument("--use-lora", action="store_true", default=True)
    parser.add_argument("--no-lora", dest="use_lora", action="store_false")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,ff_proj,up_proj",
    )
    parser.add_argument("--modules-to-save", default="model.transformer.wte,model.transformer.ff_out")
    args = parser.parse_args()
    if args.answer_token_count is None:
        args.answer_token_count = infer_answer_token_count(args.data_dir)
    if args.representation == "dynamic_v1" and args.answer_token_count is None:
        args.answer_token_count = DYNAMIC_MAX_ANSWER_TOKEN_COUNT
    if args.representation == "fixed_slot_physical_header" and args.answer_token_count == ANSWER_TOKEN_COUNT:
        args.answer_token_count = PHYSICAL_HEADER_ANSWER_TOKEN_COUNT

    dist_info = init_distributed()
    is_main = dist_info["is_main"]
    device = dist_info["device"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "training_log.jsonl"
    run_config = vars(args).copy()
    run_config["checkpoint_path"] = None if args.checkpoint_path is None else str(args.checkpoint_path)
    run_config["data_dir"] = str(args.data_dir)
    run_config["train_csv_dir"] = None if args.train_csv_dir is None else str(args.train_csv_dir)
    run_config["output_dir"] = str(args.output_dir)
    run_config["distributed"] = dist_info["distributed"]
    run_config["world_size"] = dist_info["world_size"]
    if is_main:
        write_json(str(args.output_dir / "run_config.json"), run_config)

    tokenizer, model, num_new_tokens, tokenizer_source, model_source, semantic_report = load_tokenizer_and_model(
        args,
        is_main=is_main,
    )
    loss_config = build_loss_config(tokenizer, args)
    if is_main:
        if args.representation == "cif_lite_modular":
            prompt_pool_payload = CIF_LITE_PROMPT_POOL
            canonical_prompt = CIF_LITE_PROMPT_POOL["composition"][0]
        elif args.representation == "crysllmgen_text":
            prompt_pool_payload = {
                "full": [CRYSLLMGEN_TEXT_PROMPT],
                "lattice": [CRYSLLMGEN_LATTICE_PROMPT],
                "composition": [CRYSLLMGEN_COMPOSITION_PROMPT_TEMPLATE],
                "species": [CRYSLLMGEN_SPECIES_PROMPT_TEMPLATE],
                "coords": [CRYSLLMGEN_COORDS_PROMPT_TEMPLATE],
                "site_coord": [CRYSLLMGEN_SITE_COORD_PROMPT_TEMPLATE],
            }
            canonical_prompt = CRYSLLMGEN_LATTICE_PROMPT
        elif args.representation == "fixed_plain":
            prompt_pool_payload = {
                "count": [FIXED_PLAIN_COUNT_PROMPT],
                "lattice": [FIXED_PLAIN_LATTICE_PROMPT_TEMPLATE],
                "elements": [FIXED_PLAIN_ELEMENTS_PROMPT_TEMPLATE],
                "coords": [FIXED_PLAIN_COORDS_PROMPT_TEMPLATE],
            }
            canonical_prompt = FIXED_PLAIN_COUNT_PROMPT
        elif args.representation == "r5_plan_state":
            prompt_pool_payload = ["R5 plan_state JSON generation"]
            canonical_prompt = "R5 plan_state JSON generation"
        elif args.representation == "r5_repair_text":
            prompt_pool_payload = ["R5 corrective repair"]
            canonical_prompt = "R5 corrective repair"
        elif args.representation == "dynamic_v1":
            data_stats = {}
            try:
                data_stats = json.loads((args.data_dir / "stats.json").read_text(encoding="utf-8"))
            except Exception:
                data_stats = {}
            if data_stats.get("r5_representation") in {
                "r5c_plan_body_v1",
                "r5c_composition_text_plan_body_v1",
            } and data_stats.get("prompt"):
                canonical_prompt = str(data_stats["prompt"]).rstrip()
                prompt_pool_payload = [canonical_prompt]
            else:
                prompt_pool_payload = DYNAMIC_PROMPT_POOL
                canonical_prompt = DYNAMIC_PROMPT_POOL[0]
        elif args.representation == "fixed_slot_physical_header":
            prompt_pool_payload = PHYSICAL_HEADER_PROMPT_POOL
            canonical_prompt = PHYSICAL_HEADER_PROMPT_POOL[0]
        else:
            prompt_pool_payload = PROMPT_POOL
            canonical_prompt = PROMPT_POOL[0]
        write_json(
            str(args.output_dir / "prompt_pool.json"),
            {
                "prompt_pool": prompt_pool_payload,
                "canonical_prompt": canonical_prompt,
            },
        )
        write_json(
            str(args.output_dir / "tokenizer_report.json"),
            {
                "model_path": args.model_path,
                "checkpoint_path": None if args.checkpoint_path is None else str(args.checkpoint_path),
                "tokenizer_source": tokenizer_source,
                "model_source": model_source,
                "vocab_size": len(tokenizer),
                "num_new_tokens": num_new_tokens,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "mask_token_id": MASK_TOKEN_ID,
                "pad_token_id_ne_mask_token_id": tokenizer.pad_token_id != MASK_TOKEN_ID,
                "semantic_init_element_tokens": semantic_report,
            },
        )
        write_json(
            str(args.output_dir / "element_special_token_alignment_report.json"),
            semantic_report,
        )
        write_semantic_init_markdown(
            args.output_dir / "element_special_token_alignment_report.md",
            semantic_report,
        )
    model.to(device)
    element_alignment_targets = None
    if args.element_token_alignment_loss_weight > 0:
        element_alignment_targets = build_element_semantic_alignment_targets(
            tokenizer,
            model,
            device=device,
        )
        if is_main:
            alignment_report = element_alignment_targets.get("report", {})
            write_json(
                str(args.output_dir / "element_token_semantic_alignment_report.json"),
                alignment_report,
            )
            write_element_alignment_markdown(
                args.output_dir / "element_token_semantic_alignment_report.md",
                alignment_report,
            )
    if dist_info["distributed"]:
        model = DDP(
            model,
            device_ids=[dist_info["local_rank"]],
            output_device=dist_info["local_rank"],
            find_unused_parameters=False,
        )
    model.train()

    if args.train_csv_dir is None:
        train_ds = JsonlSftDataset(args.data_dir / "train.jsonl", tokenizer, args.max_length)
        train_data_source = str(args.data_dir / "train.jsonl")
    else:
        train_ds = CsvCrystalSftDataset(
            args.train_csv_dir / "train.csv",
            tokenizer=tokenizer,
            max_length=args.max_length,
            split="train",
            seed=args.data_seed,
            answer_separator=args.answer_separator,
            augment_origin_shift=args.augment_origin_shift,
            niggli=args.niggli,
            primitive=args.primitive,
            answer_representation=args.representation,
        )
        train_data_source = str(args.train_csv_dir / "train.csv")
    val_ds = JsonlSftDataset(args.data_dir / "val.jsonl", tokenizer, args.max_length)
    if args.limit_train:
        train_ds.rows = train_ds.rows[: args.limit_train]
    if args.limit_val:
        val_ds.rows = val_ds.rows[: args.limit_val]
    collator = DataCollator(tokenizer)
    sample_weight_multipliers = parse_sample_weight_multipliers(args.sample_weight_multipliers)
    raw_train_weights = (
        train_ds.sample_weights(
            sample_weight_multipliers,
            use_jsonl_sample_weight=not args.ignore_jsonl_sample_weight,
        )
        if args.weighted_sampling and hasattr(train_ds, "sample_weights")
        else [1.0 for _ in range(len(train_ds))]
    )
    if args.weighted_sampling_power != 1.0:
        raw_train_weights = [math.pow(max(weight, 0.0), args.weighted_sampling_power) for weight in raw_train_weights]
    train_weight_summary = summarize_sample_weights(raw_train_weights)
    multiplier_summary = summarize_sample_weight_multipliers(train_ds.rows, sample_weight_multipliers)
    weighted_sampling_active = bool(args.weighted_sampling and train_weight_summary["positive_count"] > 0)
    if weighted_sampling_active and dist_info["distributed"]:
        train_sampler = DistributedWeightedSampler(
            raw_train_weights,
            num_replicas=dist_info["world_size"],
            rank=dist_info["rank"],
            seed=args.data_seed,
        )
    elif weighted_sampling_active:
        generator = torch.Generator()
        generator.manual_seed(args.data_seed)
        train_sampler = WeightedRandomSampler(
            raw_train_weights,
            num_samples=len(train_ds),
            replacement=True,
            generator=generator,
        )
    else:
        train_sampler = (
            DistributedSampler(train_ds, num_replicas=dist_info["world_size"], rank=dist_info["rank"], shuffle=True)
            if dist_info["distributed"]
            else None
        )
    val_sampler = (
        DistributedSampler(val_ds, num_replicas=dist_info["world_size"], rank=dist_info["rank"], shuffle=False)
        if dist_info["distributed"]
        else None
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        collate_fn=collator,
        num_workers=args.dataloader_num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        collate_fn=collator,
        num_workers=args.dataloader_num_workers,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    total_steps = math.ceil(len(train_loader) * args.epochs / args.grad_accum)
    if int(args.max_train_steps) > 0:
        total_steps = min(total_steps, int(args.max_train_steps))
    scheduler = build_lr_scheduler(optimizer, args, total_steps)
    progress = tqdm(total=total_steps, desc="LLaDA SFT", disable=not is_main)
    log_context = log_path.open("a", encoding="utf-8") if is_main else nullcontext()
    with log_context as log_handle:
        if is_main:
            log_handle.write(
                json.dumps(
                    {
                        "event": "start",
                        "num_new_tokens": num_new_tokens,
                        "distributed": dist_info["distributed"],
                        "world_size": dist_info["world_size"],
                        "train_data_source": train_data_source,
                        "weighted_sampling": {
                            "enabled": bool(args.weighted_sampling),
                            "active": weighted_sampling_active,
                            "power": args.weighted_sampling_power,
                            "use_jsonl_sample_weight": not args.ignore_jsonl_sample_weight,
                            "sampler": None if train_sampler is None else type(train_sampler).__name__,
                            "multipliers": multiplier_summary,
                            **train_weight_summary,
                        },
                        "loss_config": {
                            key: value
                            for key, value in loss_config.items()
                            if key not in {"empty_token_id", "pad_coord_token_ids"}
                        },
                        "element_token_alignment": {
                            "loss_weight": args.element_token_alignment_loss_weight,
                            "output_weight": args.element_token_alignment_output_weight,
                            "matched_count": (
                                None
                                if element_alignment_targets is None
                                else element_alignment_targets.get("report", {}).get("matched_count")
                            ),
                            "mean_input_cosine_to_alias_target": (
                                None
                                if element_alignment_targets is None
                                else element_alignment_targets.get("report", {}).get("mean_input_cosine_to_alias_target")
                            ),
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        stop_training = False
        for epoch in range(args.epochs):
            if hasattr(train_ds, "set_epoch"):
                train_ds.set_epoch(epoch)
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            for micro_step, batch in enumerate(train_loader):
                if int(args.max_train_steps) > 0 and global_step >= int(args.max_train_steps):
                    stop_training = True
                    break
                batch = {k: v.to(device) for k, v in batch.items()}
                sync_now = (micro_step + 1) % args.grad_accum == 0
                sync_context = (
                    model.no_sync()
                    if dist_info["distributed"] and not sync_now
                    else nullcontext()
                )
                with sync_context:
                    task_loss = compute_loss(model, batch, loss_config)
                    alignment_component = element_semantic_alignment_loss(
                        model,
                        element_alignment_targets,
                        output_weight=args.element_token_alignment_output_weight,
                    )
                    if alignment_component is not None:
                        loss = (
                            task_loss
                            + float(args.element_token_alignment_loss_weight) * alignment_component
                        ) / args.grad_accum
                    else:
                        loss = task_loss / args.grad_accum
                    loss.backward()
                if (micro_step + 1) % args.grad_accum != 0:
                    continue
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                progress.update(1)

                if is_main and global_step % args.logging_steps == 0:
                    alignment_value = (
                        None
                        if alignment_component is None
                        else float(alignment_component.detach().cpu())
                    )
                    log_handle.write(
                        json.dumps(
                            {
                                "event": "train",
                                "step": global_step,
                                "epoch": epoch,
                                "loss": float(loss.detach().cpu()) * args.grad_accum,
                                "task_loss": float(task_loss.detach().cpu()),
                                "element_alignment_loss": alignment_value,
                                "lr": optimizer.param_groups[0]["lr"],
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    log_handle.flush()

                if global_step % args.eval_steps == 0:
                    val_loss = evaluate(
                        model,
                        val_loader,
                        device,
                        args.eval_max_batches,
                        loss_config,
                        distributed=dist_info["distributed"],
                    )
                    if is_main:
                        log_handle.write(
                            json.dumps(
                                {"event": "eval", "step": global_step, "val_loss": val_loss},
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        log_handle.flush()
                    if args.position_diagnostics_steps > 0 and global_step % args.position_diagnostics_steps == 0:
                        diagnostics = evaluate_position_diagnostics(
                            model,
                            val_loader,
                            device,
                            args.eval_max_batches,
                            loss_config,
                            distributed=dist_info["distributed"],
                        )
                        if is_main:
                            log_handle.write(
                                json.dumps(
                                    {
                                        "event": "position_diagnostics",
                                        "step": global_step,
                                        **diagnostics,
                                    },
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
                            log_handle.flush()

                if global_step % args.save_steps == 0:
                    save_checkpoint(
                        model,
                        tokenizer,
                        args.output_dir,
                        global_step,
                        save_embedding_layers=args.save_embedding_layers,
                        data_dir=args.data_dir,
                        is_main=is_main,
                    )
                    if dist_info["distributed"]:
                        dist.barrier()
                if int(args.max_train_steps) > 0 and global_step >= int(args.max_train_steps):
                    stop_training = True
                    break
            if stop_training:
                break

    save_checkpoint(
        model,
        tokenizer,
        args.output_dir,
        global_step,
        save_embedding_layers=args.save_embedding_layers,
        data_dir=args.data_dir,
        is_main=is_main,
    )
    if is_main:
        tokenizer.save_pretrained(args.output_dir / "final")
        target_model = model.module if hasattr(model, "module") else model
        save_model_pretrained(
            target_model,
            args.output_dir / "final",
            save_embedding_layers=args.save_embedding_layers,
        )
        for name in ("compressed_token_config.json", "token_map.json", "physical_header_config.json"):
            source = args.data_dir / name
            if source.exists():
                (args.output_dir / "final" / name).write_text(
                    source.read_text(encoding="utf-8"),
                    encoding="utf-8",
                )
    if dist_info["distributed"]:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
