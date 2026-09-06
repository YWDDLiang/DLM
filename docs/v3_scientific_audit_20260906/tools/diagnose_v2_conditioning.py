#!/usr/bin/env python3
"""Frozen V2 validation sensitivity: no training, generation, or physical model."""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import torch
import torch.distributed as dist

from crystal_dlm.c3fd_native_plan import SOFT_FIELD_KEYS, build_native_body_prompt
from crystal_dlm.periodic_base_objective import numeric_family_axis
from crystal_dlm.periodic_base_training_data import _mask_id, prepare_periodic_base_source
from crystal_dlm.periodic_geometry_objective import build_geometry_token_support
from crystal_dlm.periodic_repair_model import task_features
from crystal_dlm.periodic_v2_objective import (
    PeriodicV2Objective, V2_POLICY_TEMPERATURE, dense_field_coefficients, fold_typed_logits,
)
from crystal_dlm.periodic_v2_training_data import (
    PrefixSupportAuditor, V2_TRAINING_DATA_PROTOCOL, make_periodic_v2_training_example,
)
from crystal_dlm.state_training import materialize_state_batch

VARIANTS = ("baseline", "noise_true", "noise_unknown", "target_soft_same_program", "old_geometry_masked")
PAIRS = {
    "true_vs_unknown_noise": ("noise_unknown", "noise_true"),
    "target_vs_predicted_soft_same_program": ("baseline", "target_soft_same_program"),
    "masked_vs_available_old_geometry": ("baseline", "old_geometry_masked"),
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def file_sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def select_sources(prepared, original, *, count, seed):
    """Selection only reads split and stable source identity, never outcomes."""
    def index(rows):
        result = {}
        for row in rows:
            key = int(row["source_row_idx"])
            if row.get("source_split") != "val" or key in result:
                raise ValueError("inputs must contain unique, explicitly original-validation source identities")
            result[key] = row
        return result
    prepared_by_id, original_by_id = index(prepared), index(original)
    if not 1 <= count <= len(prepared_by_id):
        raise ValueError("requested source count exceeds the validation input")
    chosen = sorted(prepared_by_id, key=lambda key: (digest(["v2-validation-diagnostic", seed, "val", key]), key))[:count]
    result = []
    for key in chosen:
        row, target = prepared_by_id[key], original_by_id[key]
        if row["answer"] != target["answer"]:
            raise ValueError(f"prepared and original clean answers differ for validation source {key}")
        hard = lambda plan: (int(plan["N"]), sorted(zip(plan["elements"], plan["counts"])), plan["anion_framework"])
        if hard(row["plan_state"]) != hard(target["plan_state"]):
            raise ValueError(f"Plan intervention would change hard composition/family for source {key}")
        result.append((row, target))
    return result


def constraints_for(tokenizer):
    from scripts.sample_llada_dynamic_crystals import build_dynamic_lightweight_constraints
    return build_dynamic_lightweight_constraints(
        tokenizer, duplicate_coordinate_mask=True, lattice_volume_mask=True, min_lattice_rad=1e-4,
        canonicalize_periodic_alias=True, pbc_min_distance_mask=True,
        pbc_min_distance_A=.5, pbc_image_radius=2,
    )


def make_variants(example, original_plan, tokenizer):
    """Body, target, active positions, and compiled species order stay fixed."""
    variants = {name: deepcopy(example) for name in VARIANTS}
    variants["noise_true"].update(numeric_noise_components=list(example["declared_numeric_noise_components"]),
                                  noise_metadata_dropped=False)
    variants["noise_unknown"].update(numeric_noise_components=[-1., -1., -1.], noise_metadata_dropped=True)
    target = variants["target_soft_same_program"]
    for field in SOFT_FIELD_KEYS:
        target["plan_state"][field] = original_plan[field]
    target["prompt"] = build_native_body_prompt(target["plan_state"]).rstrip() + "\n"
    masked = variants["old_geometry_masked"]
    mask = _mask_id(tokenizer, tokenizer.get_vocab())
    for position in example["full_transaction_positions"]:
        masked["old_body"][position] = mask
    for row in variants.values():
        row["prompt_token_ids"] = tokenizer(row["prompt"].rstrip() + "\n", add_special_tokens=False)["input_ids"]
    return variants


def input_signature(example):
    fields = ("prompt_token_ids", "input_body", "old_body", "num_atoms", "species_program",
              "transaction_positions", "phase", "numeric_noise_level", "numeric_noise_components")
    return digest({name: example[name] for name in fields})


def support_tables(tokenizer, device):
    result = {}
    for family, axes in build_geometry_token_support(tokenizer).items():
        for axis, table in axes.items():
            ids = torch.tensor(table["ids"], device=device, dtype=torch.long)
            values = torch.tensor(table["values"], device=device, dtype=torch.float32)
            _, canonical_ids = fold_typed_logits(torch.zeros(1, len(tokenizer.get_vocab()), device=device), ids, values, family)
            result[(family, axis)] = {"ids": ids, "values": values, "canonical_ids": canonical_ids.cpu().tolist()}
    return result


def common_targets(example, auditor, tables):
    """Freeze the baseline's original risk support, independently of any intervention."""
    coefficients = ([1.] if example["branch"] == "prefix" else
                    dense_field_coefficients(example["positions"], example["num_atoms"], example["mask_probability"]))
    descriptors = []
    for position, target, coefficient in zip(example["positions"], example["targets"], coefficients, strict=True):
        _, family, axis, codec_step = numeric_family_axis(position)
        schema_ids = tables[(family, axis)]["canonical_ids"]
        common_ids, bad = schema_ids, False
        if example["branch"] == "prefix":
            current = auditor.remap_body(example["input_body"])
            raw = torch.zeros(1, current.shape[1], auditor.vocabulary_size)
            projected, rejected = auditor.process_compact_logits(raw, current, position=position, count=example["num_atoms"])
            selected = projected[0, position] > torch.finfo(projected.dtype).min
            common_ids = [auditor.original_ids[i] for i in torch.nonzero(selected).flatten().tolist()]
            bad = bool(rejected)
        eligible = (not bad and target in common_ids and
                    (example["branch"] != "prefix" or bool(example["legal_eligible"])))
        schema_index = {token: index for index, token in enumerate(schema_ids)}
        descriptors.append({
            "position": int(position), "target_token_id": int(target), "family": family, "axis": axis,
            "site_index": (position-8)//4 if family == "coord" else None, "codec_step": float(codec_step),
            "coefficient": float(coefficient), "baseline_eligible": eligible,
            "common_support_kind": "actual_prefix_policy" if example["branch"] == "prefix" else "canonical_typed",
            "common_support_ids": common_ids, "schema_support_ids": schema_ids,
            "common_indices_in_schema": [schema_index[token] for token in common_ids],
        })
    return descriptors


def finite_stats(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = values[np.isfinite(values)]
    return {"count": int(values.size), "finite_count": int(finite.size),
            "mean": float(finite.mean()) if finite.size else None,
            "median": float(np.median(finite)) if finite.size else None,
            "q10": float(np.quantile(finite, .1)) if finite.size else None,
            "q90": float(np.quantile(finite, .9)) if finite.size else None}


def tensor_stats(value):
    array = np.asarray(value, dtype=np.float64)
    finite = bool(np.isfinite(array).all())
    return {"shape": list(array.shape), "elements": int(array.size), "finite": finite,
            "rms": float(np.sqrt(np.mean(array**2))) if array.size and finite else None,
            "max_abs": float(np.max(np.abs(array))) if array.size and finite else None}


def sampler_exp_contract():
    """Read the explicit cast at the actual row-seeded sampler's exp call."""
    from crystal_dlm.spad_generation import _transaction_candidate_tokens
    source = inspect.getsource(_transaction_candidate_tokens)
    contract = {"function": "spad_generation._transaction_candidate_tokens", "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "branch": "nonzero temperature with sampling_seeds_by_batch", "exp_dtype": None}
    for node in ast.walk(ast.parse(source)):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "exp"):
            continue
        cast = node.func.value
        if isinstance(cast, ast.Call) and isinstance(cast.func, ast.Attribute) and cast.func.attr == "to":
            for keyword in cast.keywords:
                value = keyword.value
                if keyword.arg == "dtype" and isinstance(value, ast.Attribute) and isinstance(value.value, ast.Name) and value.value.id == "torch":
                    dtype = getattr(torch, value.attr)
                    info = torch.finfo(dtype)
                    contract.update(exp_dtype=str(dtype), log_max_finite=math.log(info.max), log_min_normal=math.log(info.tiny),
                                    measurement="actual device exp on genuine legal logits; no finfo-min mask sentinels")
                    return dtype, contract
    contract["measurement"] = "no recognized explicitly cast exp call; no dtype or overflow inference made"
    return None, contract


def policy_exp_stats(vector, *, unavailable, exp_dtype):
    """The sampler exponentiates before temperature scaling; count its real dtype outcomes."""
    legal = torch.isfinite(vector) & (vector > torch.finfo(vector.dtype).min)
    values = vector[legal]
    result = {"raw_policy_dtype": str(vector.dtype), "exp_input_dtype": str(exp_dtype),
              "policy_unavailable": bool(unavailable), "legal_values": int(values.numel()),
              "logit_min": float(values.min()) if values.numel() else None,
              "logit_max": float(values.max()) if values.numel() else None,
              "masked_sentinel_values_excluded": int((vector == torch.finfo(vector.dtype).min).sum()),
              "nonfinite_values": int((~torch.isfinite(vector)).sum())}
    if exp_dtype is None:
        result["measured"] = False
        return result
    exponentials = values.to(dtype=exp_dtype).exp()
    zeros = int((exponentials == 0).sum())
    result.update(measured=True, overflow_values=int(torch.isposinf(exponentials).sum()),
                  zero_exp_values=zeros, all_legal_exp_zero=bool(values.numel() and zeros == values.numel()),
                  subnormal_exp_values=int(((exponentials > 0) & (exponentials < torch.finfo(exp_dtype).tiny)).sum()))
    return result


class ActivationProbe:
    """Read-only output hooks: no replacement, parameter mutation, or extra model forward."""
    def __init__(self, model, tables):
        self.tables = tables
        self.handles = [model.state_conditioner.register_forward_hook(self.conditioner),
                        model.repair_task_projection.register_forward_hook(self.task),
                        model.geometry_attention.register_forward_hook(self.geometry),
                        model.numeric_adapter.register_forward_hook(self.numeric)]

    def reset(self, batch):
        self.batch = batch
        self.rows = [dict() for _ in batch["examples"]]
        self.post_numeric = [[] for _ in self.rows]

    @staticmethod
    def copy(value):
        return value.detach().float().cpu().numpy().copy()

    def conditioner(self, _module, _args, output):
        for row, example in enumerate(self.batch["examples"]):
            self.rows[row]["conditioner_cell"] = self.copy(output["cell_embedding"][row])
            self.rows[row]["conditioner_sites"] = self.copy(output["site_embeddings"][row, :example["num_atoms"]])

    def task(self, _module, _args, output):
        for row in range(len(self.rows)):
            self.rows[row]["task_projection"] = self.copy(output[row])

    def geometry(self, _module, _args, output):
        for row, example in enumerate(self.batch["examples"]):
            prompt = int(self.batch["geometry_context"].prompt_lengths[row])
            body = output[row, :, prompt:prompt+len(example["input_body"]), prompt:prompt+len(example["input_body"])]
            self.rows[row]["geometry_bias_body"] = self.copy(body)
            self.rows[row]["geometry_bias_target_queries"] = self.copy(body[:, example["positions"], :])

    def numeric(self, _module, args, output):
        before, hidden, context = args
        for row, example in enumerate(self.batch["examples"]):
            prompt = int(context.prompt_lengths[row])
            self.rows[row]["final_target_hidden"] = self.copy(hidden[row, [prompt+p for p in example["positions"]]])
            increments = []
            for position in example["positions"]:
                _, family, axis, _ = numeric_family_axis(position)
                ids = self.tables[(family, axis)]["ids"]
                post = self.copy(output[row, prompt+position, ids])
                self.post_numeric[row].append(post)
                increments.append(post-self.copy(before[row, prompt+position, ids]))
            self.rows[row]["numeric_adapter_increment"] = np.concatenate(increments) if increments else np.empty(0)

    def finish(self, logits):
        for row, example in enumerate(self.batch["examples"]):
            prompt = int(self.batch["geometry_context"].prompt_lengths[row])
            increments = []
            for index, position in enumerate(example["positions"]):
                _, family, axis, _ = numeric_family_axis(position)
                ids = self.tables[(family, axis)]["ids"]
                increments.append(self.copy(logits[row, prompt+position, ids])-self.post_numeric[row][index])
            self.rows[row]["new_token_output_increment"] = np.concatenate(increments) if increments else np.empty(0)
        return self.rows

    def close(self):
        for handle in self.handles:
            handle.remove()


def log_distribution(vector, indices):
    chosen = np.asarray(vector, dtype=np.float64)[indices]
    if not len(chosen) or not np.isfinite(chosen).all():
        return None
    scores = torch.from_numpy(chosen) / V2_POLICY_TEMPERATURE
    return torch.log_softmax(scores, -1).numpy()


def position_result(vector, target):
    schema_ids = target["schema_support_ids"]
    common_indices = target["common_indices_in_schema"]
    result = {"finite_typed_logits": bool(np.isfinite(vector).all()), "supports": {}}
    for name, indices, ids in (("schema", list(range(len(schema_ids))), schema_ids),
                               ("common", common_indices, target["common_support_ids"])):
        logp = log_distribution(vector, indices)
        supported = target["target_token_id"] in ids
        result["supports"][name] = {
            "support_size": len(ids), "finite": logp is not None, "target_supported": supported,
            "ce": float(-logp[ids.index(target["target_token_id"])]) if logp is not None and supported else None,
            "top_token_id": int(ids[int(logp.argmax())]) if logp is not None else None,
        }
    return result


def compare_vectors(reference, method, target):
    result = {}
    schema_ids = target["schema_support_ids"]
    for kind, ids in (("schema", schema_ids), ("common", target["common_support_ids"])):
        indices = list(range(len(schema_ids))) if kind == "schema" else target["common_indices_in_schema"]
        a, b = log_distribution(reference, indices), log_distribution(method, indices)
        if a is None or b is None:
            result[kind] = {"finite": False}
            continue
        p, q = np.exp(a), np.exp(b)
        mixture = np.logaddexp(a, b)-math.log(2)
        delta = np.asarray(method)[indices].astype(np.float64)-np.asarray(reference)[indices].astype(np.float64)
        target_index = ids.index(target["target_token_id"]) if target["target_token_id"] in ids else None
        result[kind] = {"finite": True, "tv": float(.5*np.abs(p-q).sum()),
                        "kl_reference_to_method": float(np.sum(p*(a-b))),
                        "js": float(.5*np.sum(p*(a-mixture)+q*(b-mixture))),
                        "centered_logit_delta_rms": float(np.sqrt(np.mean((delta-delta.mean())**2))),
                        "max_abs_logit_delta": float(np.max(np.abs(delta))),
                        "target_ce_delta": float(a[target_index]-b[target_index]) if target_index is not None else None,
                        "top1_changed": int(a.argmax()) != int(b.argmax())}
    return result


def state_risk(position_results, targets, weight, kind):
    terms = []
    for value, target in zip(position_results, targets, strict=True):
        eligible = target["baseline_eligible"] if kind == "common" else True
        if not eligible:
            continue
        ce = value["supports"][kind]["ce"]
        if ce is None:
            return None
        terms.append(target["coefficient"]*ce)
    return float(weight*sum(terms))


def paired_record(example, variants, results, targets, original_plan, prediction_status):
    source = int(example["source_row_idx"])
    record = {"source_row_idx": source, "source_split": "val", "view": int(example["view"]),
              "epoch": int(example["epoch"]), "branch": example["branch"], "branch_name": example["branch_name"],
              "phase": example["phase"], "num_atoms": int(example["num_atoms"]),
              "source_weight": float(example["sample_weight"]), "mask_probability": example["mask_probability"],
              "declared_noise_components": example["declared_numeric_noise_components"],
              "corruption": example["corruption_info"], "prediction_status": prediction_status,
              "predicted_soft": {key: example["plan_state"][key] for key in SOFT_FIELD_KEYS},
              "target_annotation_soft": {key: original_plan[key] for key in SOFT_FIELD_KEYS},
              "species_program": example["species_program"], "species_program_source": example["species_program_source"],
              "source_answer_sha256": example["source_answer_sha256"],
              "canonical_answer_sha256": example["canonical_answer_sha256"],
              "input_body": example["input_body"], "baseline_old_body": example["old_body"],
              "active_positions": example["transaction_positions"], "targets": targets,
              "baseline_old_admitted": example["old_state_admitted"], "baseline_legal_eligible": example["legal_eligible"],
              "empty_supervision": not targets, "geometry_dimensions": 6+3*int(example["num_atoms"]),
              "requested_target_scalars": len(targets), "variants": {}, "comparisons": {}}
    for name in VARIANTS:
        row, value = variants[name], results[name]
        positions = [position_result(vector, target) for vector, target in zip(value["vectors"], targets, strict=True)]
        record["variants"][name] = {
            "input_signature": input_signature(row), "computed_as": value["computed_as"],
            "prompt": row["prompt"], "prompt_length": len(row["prompt_token_ids"]),
            "noise_components": row["numeric_noise_components"], "geometry": value["geometry"],
            "raw_logits_dtype": value["raw_logits_dtype"], "sampler_exp": value["sampler_exp"],
            "common_risk": state_risk(positions, targets, example["sample_weight"], "common"),
            "schema_risk": state_risk(positions, targets, example["sample_weight"], "schema"),
            "positions": positions, "modules": {key: tensor_stats(array) for key, array in value["modules"].items()},
        }
    for name, (reference, method) in PAIRS.items():
        a, b = record["variants"][reference], record["variants"][method]
        comparison = {"reference": reference, "method": method,
                      "identical_model_input": a["input_signature"] == b["input_signature"],
                      "prompt_length_changed": a["prompt_length"] != b["prompt_length"],
                      "positions": [compare_vectors(x, y, target) for x, y, target in
                                    zip(results[reference]["vectors"], results[method]["vectors"], targets, strict=True)],
                      "module_deltas": {key: tensor_stats(results[method]["modules"][key]-array)
                                        for key, array in results[reference]["modules"].items()}}
        for kind in ("common", "schema"):
            x, y = a[f"{kind}_risk"], b[f"{kind}_risk"]
            comparison[f"{kind}_risk_delta"] = None if x is None or y is None else y-x
        record["comparisons"][name] = comparison
    return record


def summarize(records, *, expected_sources):
    by_source = defaultdict(list)
    for row in records:
        by_source[row["source_row_idx"]].append(row)
    if len(by_source) != expected_sources or any(sorted(x["view"] for x in rows) != [0, 1] for rows in by_source.values()):
        raise ValueError("diagnostic output did not retain both views for every selected source")
    summary = {"sources": len(by_source), "states": len(records), "branches": dict(Counter(r["branch_name"] for r in records)),
               "empty_states": sum(r["empty_supervision"] for r in records),
               "requested_target_scalars": sum(r["requested_target_scalars"] for r in records),
               "dense_available_geometry_scalars": sum(r["geometry_dimensions"] for r in records),
               "source_target_coverage": [], "variants": {}, "comparisons": {}}
    for source, rows in sorted(by_source.items()):
        union = sorted({t["position"] for row in rows for t in row["targets"]})
        summary["source_target_coverage"].append({"source_row_idx": source, "geometry_dimensions": rows[0]["geometry_dimensions"],
                                                  "unique_supervised_positions": union,
                                                  "repair_target_scalars": next(r["requested_target_scalars"] for r in rows if r["view"] == 1)})
    for name in VARIANTS:
        summary["variants"][name] = {
            f"{kind}_risk": finite_stats([r["variants"][name][f"{kind}_risk"] if r["variants"][name][f"{kind}_risk"] is not None else np.nan for r in records])
            for kind in ("common", "schema")}
        activations = defaultdict(list)
        for row in records:
            for module, value in row["variants"][name]["modules"].items():
                activations[f"{row['branch_name']}:{module}:rms"].append(value["rms"] if value["rms"] is not None else np.nan)
        summary["variants"][name]["module_activations"] = {key: finite_stats(values) for key, values in sorted(activations.items())}
        exp_checks = [value for row in records for value in row["variants"][name]["sampler_exp"]]
        measured = [value for value in exp_checks if value["measured"]]
        nonempty = [value for value in exp_checks if value["legal_values"]]
        summary["variants"][name]["sampler_exp"] = {
            "scope": "teacher-prefix states, not generated-trajectory logits; baseline and unknown-noise variants only",
            "vectors_checked": len(exp_checks), "vectors_measured": len(measured),
            "raw_dtype_counts": dict(Counter(value["raw_policy_dtype"] for value in exp_checks)),
            "pre_exp_dtype_counts": dict(Counter(value["exp_input_dtype"] for value in exp_checks)),
            "policy_unavailable_vectors": sum(value["policy_unavailable"] for value in exp_checks),
            "legal_logit_min": min((value["logit_min"] for value in nonempty), default=None),
            "legal_logit_max": max((value["logit_max"] for value in nonempty), default=None),
            "legal_values": sum(value["legal_values"] for value in exp_checks),
            "overflow_values": sum(value["overflow_values"] for value in measured),
            "any_overflow_vectors": sum(value["overflow_values"] > 0 for value in measured),
            "zero_exp_values": sum(value["zero_exp_values"] for value in measured),
            "all_legal_exp_zero_vectors": sum(value["all_legal_exp_zero"] for value in measured),
            "subnormal_exp_values": sum(value["subnormal_exp_values"] for value in measured)}
    for name in PAIRS:
        group_rows = defaultdict(list)
        for row in records:
            group_rows["all"].append(row)
            group_rows[row["branch_name"]].append(row)
            group_rows["prompt_length_changed" if row["comparisons"][name]["prompt_length_changed"] else "prompt_length_equal"].append(row)
            group_rows["identical_input" if row["comparisons"][name]["identical_model_input"] else "changed_input"].append(row)
        groups = {}
        for group, rows in group_rows.items():
            groups[group] = {"states": len(rows), "sources": len({r["source_row_idx"] for r in rows})}
            for kind in ("common", "schema"):
                groups[group][f"{kind}_risk_delta"] = finite_stats([
                    r["comparisons"][name][f"{kind}_risk_delta"] if r["comparisons"][name][f"{kind}_risk_delta"] is not None else np.nan for r in rows])
        macro = []
        fields, modules = defaultdict(list), defaultdict(list)
        for rows in by_source.values():
            values = [r["comparisons"][name]["common_risk_delta"] for r in rows]
            macro.append(sum(values)/2 if all(v is not None for v in values) else np.nan)
        for row in records:
            for module, response in row["comparisons"][name]["module_deltas"].items():
                modules[f"{row['branch_name']}:{module}:rms_delta"].append(response["rms"] if response["rms"] is not None else np.nan)
            for target, response in zip(row["targets"], row["comparisons"][name]["positions"], strict=True):
                item = response["common"]
                fields[f"{row['branch_name']}:{target['family']}:{target['axis']}:tv"].append(item["tv"] if item["finite"] else np.nan)
                ce_delta = item.get("target_ce_delta")
                fields[f"{row['branch_name']}:{target['family']}:{target['axis']}:ce_delta"].append(ce_delta if ce_delta is not None else np.nan)
        summary["comparisons"][name] = {"groups": groups, "source_macro_common_risk_delta": finite_stats(macro),
                                          "field_sensitivity": {key: finite_stats(values) for key, values in sorted(fields.items())},
                                          "module_sensitivity": {key: finite_stats(values) for key, values in sorted(modules.items())}}
    summary["interpretation"] = "Frozen validation sensitivity and target-risk diagnostics, not SUN, free-generation performance, or evidence of module necessity."
    return summary


def model_versions(model):
    return {name: parameter._version for name, parameter in model.named_parameters()}


def run(args):
    from crystal_dlm.periodic_v2_initialization import load_periodic_v2_model
    world, rank, local = (int(os.environ.get(name, default)) for name, default in (("WORLD_SIZE", 1), ("RANK", 0), ("LOCAL_RANK", 0)))
    if not torch.cuda.is_available():
        raise RuntimeError("real checkpoint diagnostics require CUDA; --self-check is the CPU-only entry")
    torch.cuda.set_device(local)
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.state_seed)
    if world > 1:
        dist.init_process_group("nccl")
    device = torch.device("cuda", local)
    chosen = select_sources(read_rows(args.prepared_val), read_rows(args.original_val), count=args.source_count, seed=args.selection_seed)
    exp_dtype, exp_contract = sampler_exp_contract()
    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        config = {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  "world_size": world, "source_ids_in_selection_order": [int(row["source_row_idx"]) for row, _ in chosen],
                  "input_sha256": {"prepared_val": file_sha256(args.prepared_val), "original_val": file_sha256(args.original_val)},
                  "script_sha256": file_sha256(__file__), "temperature": V2_POLICY_TEMPERATURE,
                  "variants": list(VARIANTS), "pairs": PAIRS, "state_protocol": V2_TRAINING_DATA_PROTOCOL,
                  "common_support": "baseline actual prefix policy or canonical typed dense support; admission fixed to baseline",
                  "score_precision": "alias fold in model raw dtype, then float64 log_softmax",
                  "sampler_exp_contract": exp_contract,
                  "target_soft_semantics": "original source annotation, not a newly verified post-codec symmetry label",
                  "masked_old_semantics": "only explicit old numeric values are hidden; current tokens, composition, program, and task remain",
                  "no_training": True, "no_generation": True, "no_MLIP": True,
                  "core_source_sha256": {str(p.relative_to(PROJECT_ROOT)): file_sha256(p) for p in (
                      PROJECT_ROOT / "src/crystal_dlm/periodic_v2_initialization.py",
                      PROJECT_ROOT / "src/crystal_dlm/periodic_repair_initialization.py",
                      PROJECT_ROOT / "src/crystal_dlm/periodic_repair_model.py",
                      PROJECT_ROOT / "src/crystal_dlm/periodic_v2_model.py",
                      PROJECT_ROOT / "src/crystal_dlm/periodic_v2_training_data.py",
                      PROJECT_ROOT / "src/crystal_dlm/periodic_v2_objective.py",
                      PROJECT_ROOT / "src/crystal_dlm/periodic_v2_corruption.py",
                      PROJECT_ROOT / "src/crystal_dlm/state_conditioned_model.py",
                      PROJECT_ROOT / "src/crystal_dlm/state_training.py",
                      PROJECT_ROOT / "src/crystal_dlm/programmed_path_runtime.py",
                      PROJECT_ROOT / "src/crystal_dlm/spad_generation.py")},
                  "checkpoint_files_sha256": {str(p.relative_to(args.checkpoint_path)): file_sha256(p) for p in sorted(args.checkpoint_path.rglob("*"))
                                               if p.is_file() and p.suffix in (".json", ".pt", ".bin", ".safetensors")}}
        write_json(args.output_dir / "CONFIG.json", config)
    if world > 1:
        dist.barrier()
    model, tokenizer = load_periodic_v2_model(args.model_path, args.checkpoint_path, device, trainable=False)
    model.requires_grad_(False).eval()
    versions = model_versions(model)
    constraints = constraints_for(tokenizer)
    tables = support_tables(tokenizer, device)
    auditor = PrefixSupportAuditor(tokenizer, constraints)
    objective = PeriodicV2Objective(tokenizer, constraints, device)
    items = []
    for source, original in chosen[rank::world]:
        prepared = prepare_periodic_base_source(source, tokenizer, constraints)
        for view in (0, 1):
            example = make_periodic_v2_training_example(prepared, tokenizer, constraints, view=view,
                                                        epoch=args.state_epoch, seed=args.state_seed, prefix_auditor=auditor)
            variants = make_variants(example, original["plan_state"], tokenizer)
            items.append((example, variants, common_targets(example, auditor, tables), original["plan_state"],
                          source.get("condition_prediction", {}).get("status", source.get("soft_plan_source", "unrecorded"))))
    if rank == 0:
        write_json(args.output_dir / "TOKEN_SUPPORTS.json", {
            f"{family}:{axis}": {"canonical_ids": table["canonical_ids"],
                                  "tokens": [tokenizer.convert_ids_to_tokens(i) for i in table["canonical_ids"]]}
            for (family, axis), table in tables.items()})
    probe, arrays, records = ActivationProbe(model, tables), {}, []
    parity, forward_batches, forward_rows = [], 0, 0
    started = time.monotonic()
    try:
        with torch.inference_mode():
            for start in range(0, len(items), args.batch_size):
                chunk = items[start:start+args.batch_size]
                output_by_variant, cache = {}, {}
                for variant in VARIANTS:
                    examples = [item[1][variant] for item in chunk]
                    key = tuple(input_signature(example) for example in examples)
                    if key in cache:
                        previous = cache[key]
                        output_by_variant[variant] = [dict(result, computed_as=previous,
                                                          sampler_exp=result["sampler_exp"] if variant in ("baseline", "noise_unknown") else [])
                                                       for result in output_by_variant[previous]]
                        continue
                    batch = materialize_state_batch(examples, tokenizer, device=device, max_length=args.max_length,
                                                    max_sites=model.state_config.max_sites)
                    geometry = model.geometry_inputs(batch["geometry_context"])
                    tasks = task_features(batch["geometry_context"], batch["input_ids"], geometry, model.mask_id)
                    probe.reset(batch)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        logits = model(batch["input_ids"], attention_mask=batch["attention_mask"], geometry_context=batch["geometry_context"]).logits
                    modules = probe.finish(logits)
                    values = []
                    for row, example in enumerate(examples):
                        prompt = int(batch["geometry_context"].prompt_lengths[row])
                        vectors = []
                        for position in example["positions"]:
                            _, family, axis, _ = numeric_family_axis(position)
                            table = tables[(family, axis)]
                            vector, _ = fold_typed_logits(logits[row, prompt+position], table["ids"], table["values"], family)
                            vectors.append(vector.detach().float().cpu().numpy().copy())
                        exp_checks = []
                        if example["branch"] == "prefix" and variant in ("baseline", "noise_unknown"):
                            projected, bad = objective._legal_vector(logits, batch, row, example["position"])
                            exp_checks.append(policy_exp_stats(projected, unavailable=bad, exp_dtype=exp_dtype))
                        values.append({"vectors": vectors, "modules": modules[row], "computed_as": variant,
                                       "raw_logits_dtype": str(logits.dtype), "sampler_exp": exp_checks,
                                       "geometry": {"lattice_known": bool(geometry["lattice_known"][row]),
                                                    "site_known_count": int(geometry["site_known"][row].sum()),
                                                    "task_features": tasks[row].cpu().tolist()}})
                    if variant == "baseline":
                        calculated = [state_risk([position_result(vector, target) for vector, target in zip(value["vectors"], item[2], strict=True)],
                                                 item[2], item[0]["sample_weight"], "common") for value, item in zip(values, chunk, strict=True)]
                        official, conflicts = None, []
                        if all(v is not None for v in calculated):
                            official_tensor, _metrics, conflicts = objective(logits, batch)
                            official = float(official_tensor) if bool(torch.isfinite(official_tensor)) else None
                        parity.append({"official_batch_loss": official,
                                       "diagnostic_batch_loss": sum(calculated)/len(calculated) if all(v is not None for v in calculated) else None,
                                       "absolute_difference": abs(official-sum(calculated)/len(calculated)) if official is not None and all(v is not None for v in calculated) else None,
                                       "states": len(chunk), "conflicts": conflicts})
                    output_by_variant[variant], cache[key] = values, variant
                    forward_batches += 1
                    forward_rows += len(examples)
                    del logits, batch
                for row, (example, variants, targets, original_plan, status) in enumerate(chunk):
                    results = {variant: output_by_variant[variant][row] for variant in VARIANTS}
                    records.append(paired_record(example, variants, results, targets, original_plan, status))
                    for variant in VARIANTS:
                        for target, vector in zip(targets, results[variant]["vectors"], strict=True):
                            arrays[f"source{example['source_row_idx']}_view{example['view']}_{variant}_position{target['position']}"] = vector
                print(json.dumps({"rank": rank, "completed_states": len(records), "total_states": len(items)}, ensure_ascii=False), flush=True)
    finally:
        probe.close()
    unchanged = versions == model_versions(model)
    if not unchanged or any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("frozen model parameter contract changed during diagnosis")
    shard = f"rank{rank:03d}"
    with (args.output_dir / f"paired.{shard}.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    np.savez_compressed(args.output_dir / f"typed_logits.{shard}.npz", **arrays)
    write_json(args.output_dir / f"execution.{shard}.json", {
        "rank": rank, "states": len(records), "forward_batches": forward_batches, "forward_rows": forward_rows,
        "elapsed_seconds": time.monotonic()-started, "parameters_unchanged": unchanged,
        "trainable_parameters": 0, "model_training": model.training,
        "device_name": torch.cuda.get_device_name(device), "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
        "dropout_modules_training": sum(isinstance(m, torch.nn.Dropout) and m.training for m in model.modules()),
        "baseline_objective_parity": parity})
    if world > 1:
        dist.barrier()
    if rank == 0:
        all_records = [record for path in sorted(args.output_dir.glob("paired.rank*.jsonl")) for record in read_rows(path)]
        if {record["source_row_idx"] for record in all_records} != {int(row["source_row_idx"]) for row, _ in chosen}:
            raise ValueError("written diagnostic sources differ from the predeclared identity-hash selection")
        summary = summarize(all_records, expected_sources=args.source_count)
        summary["execution"] = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(args.output_dir.glob("execution.rank*.json"))]
        write_json(args.output_dir / "SUMMARY.json", summary)
        write_json(args.output_dir / "_SUCCESS", {"sources": args.source_count, "states": 2*args.source_count})
        print(json.dumps({"summary": str(args.output_dir / "SUMMARY.json"), "sources": args.source_count}), flush=True)
    if world > 1:
        dist.destroy_process_group()


def self_check():
    """Only parser, intervention, support, batch-shape and identity arithmetic checks."""
    from crystal_dlm.dynamic_crystal import arrays_to_dynamic_tokens
    # The repository's existing CPU codec fixture avoids importing Transformers
    # just to build tiny shape-check constraints. The GPU path uses the real builder.
    sys.path.insert(0, str(PROJECT_ROOT / "tests"))
    from test_periodic_v2_training_data import codec
    fixture, constraints = codec()
    class Tokenizer:
        def __getattr__(self, name): return getattr(fixture, name)
        def __call__(self, text, **_kwargs): return {"input_ids": [1]*len(text.split())}
    tokenizer = Tokenizer()
    tokens, _ = arrays_to_dynamic_tokens([4., 4., 4.], [90., 90., 90.], ["H", "H"], [[0., 0., 0.], [.5, .5, .5]])
    plan = {"N": 2, "elements": ["H"], "counts": [2], "anion_framework": "other",
            "lattice_system": "cubic", "spacegroup_bucket": "low", "volume_per_atom_bin": "medium"}
    original = {"source_split": "val", "source_row_idx": 7, "answer": " ".join(tokens), "plan_state": plan,
                "prompt": build_native_body_prompt(plan), "species_program": ["H"], "species_program_source": "self_check"}
    prepared = deepcopy(original)
    prepared["plan_state"]["volume_per_atom_bin"] = "high"
    prepared["prompt"] = build_native_body_prompt(prepared["plan_state"])
    assert len(select_sources([prepared], [original], count=1, seed=11)) == 1
    tables, auditor = support_tables(tokenizer, torch.device("cpu")), PrefixSupportAuditor(tokenizer, constraints)
    example = make_periodic_v2_training_example(prepared, tokenizer, constraints, view=1, epoch=1, seed=11, prefix_auditor=auditor)
    variants = make_variants(example, plan, tokenizer)
    for row in variants.values():
        assert row["input_body"] == example["input_body"] and row["targets"] == example["targets"]
        assert row["species_program"] == example["species_program"]
    assert variants["noise_true"]["numeric_noise_components"] == example["declared_numeric_noise_components"]
    assert variants["noise_unknown"]["numeric_noise_components"] == [-1., -1., -1.]
    for position in example["full_transaction_positions"]:
        assert variants["old_geometry_masked"]["old_body"][position] == tokenizer.mask_token_id
    batch = materialize_state_batch(list(variants.values()), tokenizer, device=torch.device("cpu"))
    assert batch["input_ids"].shape == batch["geometry_context"].old_token_ids.shape
    assert batch["geometry_context"].numeric_noise_components.shape == (len(VARIANTS), 3)
    targets = common_targets(example, auditor, tables)
    for target in targets:
        raw = np.linspace(-1, 1, len(target["schema_support_ids"]), dtype=np.float32)
        response = compare_vectors(raw, raw, target)
        assert response["schema"]["tv"] == 0 and response["schema"]["target_ce_delta"] == 0
    # Exercise read-only hook indexing with shaped tensors, without constructing
    # a model or executing any real neural forward.
    batch_size, length = batch["input_ids"].shape
    probe = ActivationProbe.__new__(ActivationProbe)
    probe.tables = tables
    probe.reset(batch)
    hidden = torch.zeros(batch_size, length, 8)
    logits = torch.zeros(batch_size, length, len(tokenizer.get_vocab()))
    probe.conditioner(None, (), {"cell_embedding": torch.zeros(batch_size, 8), "site_embeddings": torch.zeros(batch_size, 20, 8)})
    probe.task(None, (), torch.zeros(batch_size, 8))
    probe.geometry(None, (), torch.zeros(batch_size, 2, length, length))
    probe.numeric(None, (logits, hidden, batch["geometry_context"]), logits+1)
    observed = probe.finish(logits+3)
    assert all(np.all(value["new_token_output_increment"] == 2) for value in observed)
    assert observed[0]["geometry_bias_target_queries"].shape == (2, len(targets), len(example["input_body"]))
    # Reuse the real objective on synthetic logits to check support and coefficient
    # agreement, independently of neural weights.
    one = materialize_state_batch([variants["baseline"]], tokenizer, device=torch.device("cpu"))
    zero_logits = torch.zeros(1, one["input_ids"].shape[1], len(tokenizer.get_vocab()))
    values = []
    for target in targets:
        table = tables[(target["family"], target["axis"])]
        vector, _ = fold_typed_logits(zero_logits[0, 0], table["ids"], table["values"], target["family"])
        values.append(position_result(vector.numpy(), target))
    official, _, _ = PeriodicV2Objective(tokenizer, constraints, "cpu")(zero_logits, one)
    diagnostic = state_risk(values, targets, example["sample_weight"], "common")
    assert abs(float(official)-diagnostic) < 1e-5
    # Check the other source view too, including its real prefix policy support
    # when selected by the unchanged maker (the main run does not force branches).
    other = make_periodic_v2_training_example(prepared, tokenizer, constraints, view=0, epoch=1, seed=0, prefix_auditor=auditor)
    assert other["branch"] == "prefix"
    other_targets = common_targets(other, auditor, tables)
    other_variant = make_variants(other, plan, tokenizer)["baseline"]
    other_batch = materialize_state_batch([other_variant], tokenizer, device=torch.device("cpu"))
    other_logits = torch.zeros(1, other_batch["input_ids"].shape[1], len(tokenizer.get_vocab()))
    other_values = []
    for target in other_targets:
        table = tables[(target["family"], target["axis"])]
        vector, _ = fold_typed_logits(other_logits[0, 0], table["ids"], table["values"], target["family"])
        other_values.append(position_result(vector.numpy(), target))
    other_official, _, _ = PeriodicV2Objective(tokenizer, constraints, "cpu")(other_logits, other_batch)
    assert abs(float(other_official)-state_risk(other_values, other_targets, other["sample_weight"], "common")) < 1e-5
    exp_dtype, exp_contract = sampler_exp_contract()
    assert exp_dtype is torch.float64 and exp_contract["exp_dtype"] == "torch.float64"
    sentinel = torch.finfo(torch.float32).min
    measured = policy_exp_stats(torch.tensor([sentinel, -1000., 1000.]), unavailable=False, exp_dtype=exp_dtype)
    assert measured["legal_values"] == 2 and measured["overflow_values"] == 1 and measured["zero_exp_values"] == 1
    assert not measured["all_legal_exp_zero"]
    print(json.dumps({"self_check": "pass", "variants": list(VARIANTS), "targets": len(targets),
                      "branches_checked": [example["branch"], other["branch"]],
                      "batch_shape": list(batch["input_ids"].shape), "models_loaded": 0, "GPU_calls": 0}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--prepared-val", type=Path)
    parser.add_argument("--original-val", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--source-count", type=int, default=100)
    parser.add_argument("--selection-seed", type=int, default=202609061)
    parser.add_argument("--state-seed", type=int, default=20260906)
    parser.add_argument("--state-epoch", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=382)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    missing = [name for name in ("model_path", "checkpoint_path", "prepared_val", "original_val", "output_dir") if getattr(args, name) is None]
    if missing:
        parser.error("required for the GPU diagnosis: " + ", ".join("--"+name.replace("_", "-") for name in missing))
    if min(args.source_count, args.batch_size, args.cpu_threads, args.max_length) < 1 or min(args.selection_seed, args.state_seed, args.state_epoch) < 0:
        parser.error("counts must be positive and seeds/epoch nonnegative")
    run(args)


if __name__ == "__main__":
    main()
