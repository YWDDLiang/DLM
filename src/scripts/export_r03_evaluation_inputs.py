#!/usr/bin/env python3
"""Export every R03 request to the existing evaluation-only endpoint schema.

No neural trace is synthesized. The common label/N-U/hull entrypoints consume
real structures and request identities, not historical sampling log likelihoods.
Refiner rows are joined by global sample_indices, never by their tensor offsets.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from crystal_dlm.dynamic_crystal import arrays_to_structure, parse_dynamic_answer
from crystal_dlm.fixed_slot import SYMBOL_TO_Z

SCHEMA = "r03_common_evaluation_input_v1"
SOURCE_SCHEMA = "r03_integrated_body_v1"
EDITOR_SOURCE_SCHEMA = "r03_autonomous_editor_body_v1"
COMPONENTS_SCHEMA = "r03_evaluation_components_v1"
LEGACY_PANEL_SCHEMA = "h1a2_frozen_legacy_evaluation_v1"
LEGACY_SOURCE_SCHEMA = "h1_body_safeaxis256_attempt_v1"
P0_ADAPTER_SHA256 = "65766c7485bd5ad8e180f3f5d99b83bef0488c251acd9278cb8bc2ad2518aa3a"
MODEL494_SHA256 = "573e9b10af64b266b7c6cde4d0f8bdd8a7388fa98d36e2e82db341af3e511e7e"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("every saved request must be a JSON object")
    return rows


def valid_sha(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value) is not None


def pinned_json(descriptor: Mapping[str, Any], name: str):
    if not isinstance(descriptor, Mapping) or not descriptor.get("path") or not valid_sha(descriptor.get("sha256")):
        raise ValueError(f"multi-component {name} requires explicit path and SHA256")
    path = Path(descriptor["path"])
    if sha256_file(path) != descriptor["sha256"].lower():
        raise ValueError(f"pinned {name} file changed")
    return read_json(path), {"path": str(path.resolve()), "sha256": descriptor["sha256"].lower()}


def planner_component_identity(component, planners, *, planner_seed, begin, count, c3fd_enabled):
    config, pin = pinned_json(component.get("planner_run_config"), "planner_run_config")
    required = {"planner_identity", "native_prompt_file", "native_prompt_sha256", "native_prompt_input_ids_sha256",
                "c3fd_domain", "sampling", "seed", "num_samples", "sample_index_offset", "batch_size", "seed_mode",
                "prompt_style", "include_sample_id", "stop_after_plan_marker", "truncate_after_plan_marker",
                "eos_token_id", "pad_token_id", "planner_weights_changed", "tokenizer_resized",
                "formula_prefill_added", "external_formula_composer", "formula_bridge_source_sha256", "sampler_source_sha256"}
    if not required.issubset(config) or any(config[key] is None for key in required - {"c3fd_domain"}):
        raise ValueError("multi-component Planner identity/interface fields are missing; unknown is not equality")
    if config["seed"] != planner_seed or config["num_samples"] != count or config["sample_index_offset"] != begin:
        raise ValueError("pinned Planner config does not describe this seed and global component range")
    identity = config["planner_identity"]
    identity_keys = {"adapter_sha256", "adapter_path", "base_model_path", "base_config_sha256",
                     "llama_hidden_size", "tokenizer_source", "tokenizer_size", "tokenizer_files", "p0_trainable_parameters"}
    if (not isinstance(identity, Mapping) or not identity_keys.issubset(identity)
            or any(identity[key] is None for key in identity_keys)
            or identity["adapter_sha256"] != P0_ADAPTER_SHA256 or identity["p0_trainable_parameters"] != 0
            or not valid_sha(identity["base_config_sha256"]) or not identity["tokenizer_files"]
            or not all(valid_sha(value) for value in identity["tokenizer_files"].values())):
        raise ValueError("pinned Planner config lacks the observed frozen P0/tokenizer identity")
    for name in ("native_prompt_sha256", "native_prompt_input_ids_sha256", "formula_bridge_source_sha256", "sampler_source_sha256"):
        if not valid_sha(config[name]):
            raise ValueError(f"missing Planner scientific/source SHA: {name}")
    if sha256_file(Path(config["native_prompt_file"])) != config["native_prompt_sha256"]:
        raise ValueError("the actual native Planner prompt differs from its recorded SHA")
    for planner in planners:
        ids = planner.get("prompt_input_ids")
        if not isinstance(ids, list) or not ids:
            raise ValueError("saved Planner row lacks its actual native prompt token IDs")
        digest = hashlib.sha256(json.dumps(ids, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if digest != config["native_prompt_input_ids_sha256"]:
            raise ValueError("saved request prompt IDs differ from the pinned Planner interface")
    domain = config["c3fd_domain"]
    if not c3fd_enabled:
        if domain is not None:
            raise ValueError("saved disabled-C3FD rows conflict with the Planner domain config")
        domain_identity = {"enabled": False}
    else:
        keys = {"path", "sha256", "schema", "max_atoms", "max_species", "symbols", "declared_strata"}
        if not isinstance(domain, Mapping) or not keys.issubset(domain) or any(domain[key] is None for key in keys):
            raise ValueError("enabled C3FD requires complete domain identity, not missing defaults")
        if not valid_sha(domain["sha256"]) or sha256_file(Path(domain["path"])) != domain["sha256"]:
            raise ValueError("actual C3FD domain differs from its recorded SHA")
        actual = read_json(Path(domain["path"]))
        if (actual.get("schema") != domain["schema"] or actual.get("max_atoms", actual.get("max")) != domain["max_atoms"]
                or actual.get("max_species") != domain["max_species"]
                or len(actual.get("nodes", {})) != domain["symbols"] or not actual.get("allowed_strata")):
            raise ValueError("C3FD domain parameters disagree with the pinned artifact")
        domain_identity = {key: domain[key] for key in keys - {"path"}}
        domain_identity["enabled"] = True
    sampling = config["sampling"]
    if not isinstance(sampling, Mapping) or not {"max_new_tokens", "temperature", "top_p", "top_k", "do_sample"}.issubset(sampling):
        raise ValueError("explicit original Planner sampling parameters are required")
    contract = {key: config[key] for key in required - {"native_prompt_file", "seed", "num_samples", "sample_index_offset", "c3fd_domain"}}
    contract["c3fd_domain"] = domain_identity
    return contract, pin


def exact_index(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer; it cannot be inferred or renumbered")
    return value


def validate_ledger(rows: Sequence[Mapping[str, Any]], expected_requests: int, *, source_schema=SOURCE_SCHEMA) -> None:
    if expected_requests < 1 or len(rows) != expected_requests:
        raise ValueError("all-request source denominator changed")
    if [exact_index(row.get("ordinal"), "ordinal") for row in rows] != list(range(expected_requests)):
        raise ValueError("saved body source order is not the complete original request ledger")
    sample_ids = [exact_index(row.get("sample_idx"), "sample_idx") for row in rows]
    attempts = [row.get("attempt_id") for row in rows]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("duplicate global request sample_idx")
    if any(not isinstance(value, str) or not value for value in attempts) or len(set(attempts)) != len(attempts):
        raise ValueError("original attempt IDs must be present, nonempty and unique")
    if source_schema not in (SOURCE_SCHEMA, LEGACY_SOURCE_SCHEMA, EDITOR_SOURCE_SCHEMA) or any(row.get("schema") != source_schema for row in rows):
        raise ValueError("this adapter accepts only explicit R03 integrated-body artifacts")
    if any(row.get("purpose") == "train" or row.get("artifact_role") == "training" for row in rows):
        raise ValueError("a training artifact cannot be relabelled as a generated evaluation run")


def validate_composition(structure, plan: Mapping[str, Any]) -> None:
    if not isinstance(plan, Mapping):
        raise ValueError("the original exact Plan composition is unavailable")
    n = exact_index(plan.get("N"), "Plan N")
    elements, counts = plan.get("elements"), plan.get("counts")
    if (not isinstance(elements, (list, tuple)) or not isinstance(counts, (list, tuple))
            or not elements or len(elements) != len(counts) or len(set(elements)) != len(elements)):
        raise ValueError("original Plan element/count arrays are invalid")
    expected = Counter()
    for symbol, count in zip(elements, counts):
        if symbol not in SYMBOL_TO_Z or exact_index(count, "Plan count") < 1:
            raise ValueError("invalid original Plan species/count")
        expected[SYMBOL_TO_Z[symbol]] += count
    if sum(expected.values()) != n or structure.num_sites != n or Counter(structure.atomic_numbers) != expected:
        raise ValueError("endpoint changed the original exact N or species multiset")


def source_native_failure(row: Mapping[str, Any]) -> str | None:
    """Distinguish a genuine native endpoint from a stale failure preview."""
    if row.get("body_eligible") is not True or row.get("attempt_status") == "planner_failure":
        return str(row.get("reason") or "planner_failure")
    if row.get("attempt_status") == "controller_failure" or row.get("reason") == "controller_failure":
        return str(row.get("message") or "controller_failure")
    if row.get("body_generation_complete") is not True or row.get("body_plan_match") is not True:
        return str(row.get("message") or row.get("reason") or "native_body_incomplete_or_plan_mismatch")
    if row.get("parsed") is not True and row.get("earliest_failure_stage") != "body_graph":
        return str(row.get("message") or row.get("reason") or "recorded_native_endpoint_failure")
    return None


def native_structure(row: Mapping[str, Any]):
    reason = source_native_failure(row)
    if reason is not None:
        raise ValueError(reason)
    text = row.get("raw_body_text", row.get("text"))
    if not isinstance(text, str) or not text:
        raise ValueError("actual final R03 body text is absent")
    if row.get("raw_body_text") is not None and row.get("text") is not None and row["raw_body_text"] != row["text"]:
        raise ValueError("two active native body fields disagree")
    arrays = parse_dynamic_answer(text, strict=True)
    if row.get("raw_body_token_ids") is not None and len(row["raw_body_token_ids"]) != 7 + 4 * arrays["num_atoms"]:
        raise ValueError("saved final body token count changed")
    recorded = row.get("arrays")
    if recorded is not None:
        for key in ("num_atoms", "species", "lengths", "angles", "frac_coords"):
            if key in recorded and recorded[key] != arrays[key]:
                raise ValueError(f"saved native arrays and token body disagree at {key}")
    structure = arrays_to_structure(arrays)
    if not math.isfinite(float(structure.volume)) or structure.volume <= 0:
        raise ValueError("native cell is nonpositive or nonfinite")
    validate_composition(structure, row.get("plan_state"))
    return structure, text, arrays


def validate_refined_shapes(payload: Mapping[str, Any]) -> list[int]:
    import torch
    required = ("frac_coords", "num_atoms", "atom_types", "lengths", "angles", "sample_indices")
    if any(key not in payload for key in required):
        raise ValueError("refined tensor must include real global sample_indices and complete geometry")
    indices = torch.as_tensor(payload["sample_indices"])
    counts = torch.as_tensor(payload["num_atoms"])
    atom_types = torch.as_tensor(payload["atom_types"])
    coords = torch.as_tensor(payload["frac_coords"])
    lengths, angles = (torch.as_tensor(payload[k]) for k in ("lengths", "angles"))
    integer_types = (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)
    if indices.ndim != 1 or indices.dtype not in integer_types:
        raise ValueError("sample_indices must be a one-dimensional integer ledger")
    sample_ids = indices.tolist()
    if any(v < 0 for v in sample_ids) or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("refined sample_indices are negative or duplicated")
    m = len(sample_ids)
    if counts.shape != (1, m) or counts.dtype not in integer_types or bool((counts < 0).any()):
        raise ValueError("one refinement evaluation and exact nonnegative atom counts are required")
    total_atoms = int(counts.sum())
    if (atom_types.shape != (1, total_atoms) or atom_types.dtype not in integer_types
            or coords.shape != (1, total_atoms, 3) or lengths.shape != (1, m, 3) or angles.shape != (1, m, 3)):
        raise ValueError("refined atom/geometry tensor boundaries do not match their declared counts")
    return sample_ids


def refined_structures(payload: Mapping[str, Any]):
    """Use the existing common refined parser, adding strict ledger/shape checks."""
    sample_ids = validate_refined_shapes(payload)
    helper_path = ROOT / "scripts" / "assemble_grounding_repeat.py"
    spec = importlib.util.spec_from_file_location("_r03_common_refined_parser", helper_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    structures, failures = module._refined_structures(payload, invalid_as_failure=True)
    if set(structures).union(failures) != set(sample_ids) or set(structures).intersection(failures):
        raise ValueError("the common parser did not account for every refined proposal")
    return structures, failures


def export_records(rows: Sequence[Mapping[str, Any]], *, endpoint: str,
                   expected_requests: int, method_id: str, refined_payload=None, source_schema=SOURCE_SCHEMA):
    if endpoint not in ("native", "tau800") or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", method_id):
        raise ValueError("explicit native/tau800 endpoint and a stable method ID are required")
    if (endpoint == "tau800") != (refined_payload is not None):
        raise ValueError("tau800 needs its real refined tensor; native must not read a refined tensor")
    validate_ledger(rows, expected_requests, source_schema=source_schema)
    refined, malformed = ({}, {}) if refined_payload is None else refined_structures(refined_payload)
    known_ids = {row["sample_idx"] for row in rows}
    if not set(refined).union(malformed) <= known_ids:
        raise ValueError("refiner returned a global sample_idx outside the original request ledger")
    if endpoint == "tau800":
        expected_refined = {row["sample_idx"] for row in rows if row.get("body_graph_complete") is True}
        if set(refined).union(malformed) != expected_refined:
            raise ValueError("refiner artifact does not exactly cover the source graph ledger; missing/extra IDs are engineering errors")
    results, errors = [], Counter()
    from pymatgen.core import Structure
    for ordinal, row in enumerate(rows):
        sample_idx, attempt_id = row["sample_idx"], row["attempt_id"]
        planner = row.get("planner_record") or {}
        group_id = row.get("group_id", planner.get("group_id", f"eval:{sample_idx}"))
        result = {
            "schema": SCHEMA, "method_id": method_id, "purpose": "evaluation", "artifact_role": "evaluation",
            "source_artifact_schema": source_schema,
            "source_split": "evaluation", "trainable_teacher": False,
            "trajectory_id": f"r03-eval:{method_id}:{endpoint}:{sample_idx}",
            "attempt_id": attempt_id, "source_attempt_id": attempt_id,
            "original_attempt_id": attempt_id, "planner_seed": planner.get("seed"),
            "sample_idx": sample_idx, "evaluation_ordinal": ordinal, "source_request_ordinal": row["ordinal"],
            "group_id": str(group_id), "source_row_idx": row.get("source_row_idx", sample_idx),
            "endpoint": endpoint, "plan_state": row.get("plan_state"), "planner_record": planner,
            "prompt": row.get("body_prompt"), "source_body_status": row.get("status"),
            "source_attempt_status": row.get("attempt_status"), "source_failure_stage": row.get("earliest_failure_stage"),
            "source_reason": row.get("reason"), "source_message": row.get("message"),
            "native_execution_success": row.get("status") == "succeeded",
            "native_graph_complete": row.get("body_graph_complete") is True,
            "success": False, "parseable": False, "body": None, "structure": None, "native_structure": None,
            "artifact_error": None, "neural_trace_available": False,
            "trace_scope": "endpoint_artifact_only_no_sampling_likelihood_trace",
            "native_source": "saved_final_body", "body_noise_seed": row.get("body_noise_seed"),
            "repair_trace": row.get("repair_trace"), "expert_trace": row.get("expert_trace"),
            "site_order_mapping_verified": endpoint == "native",
        }
        try:
            native, text, arrays = native_structure(row)
            result.update(native_structure=native.as_dict(), num_atoms=native.num_sites,
                          source_native_body_sha256=hashlib.sha256(text.encode()).hexdigest(),
                          native_species_order=list(arrays["species"]))
            if endpoint == "tau800":
                if row.get("body_graph_complete") is not True:
                    raise ValueError("native_graph_unavailable_for_refinement")
                if sample_idx not in refined:
                    raise ValueError(malformed.get(sample_idx, "refined_endpoint_missing_for_global_request"))
                structure = Structure.from_dict(refined[sample_idx])
                validate_composition(structure, row["plan_state"])
                # Niggli/preprocessing can change the site order. No slotwise
                # identity or displacement is inferred without a verified map.
                result.update(diffusion_refinement_steps=800, refined_sample_idx=sample_idx,
                              site_order_mapping_verified=False,
                              site_order_contract="original refined order retained; composition checked, no slot map assumed")
            else:
                structure = native
                result["body"] = text
            result.update(structure=structure.as_dict(), success=True, parseable=True,
                          endpoint_species_order=[site.specie.symbol for site in structure],
                          exact_composition_preserved=True)
            result["endpoint_structure_sha256"] = hashlib.sha256(json.dumps(result["structure"], sort_keys=True).encode()).hexdigest()
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            result.update(success=False, parseable=False, body=None, structure=None,
                          artifact_error=f"{type(exc).__name__}: {exc}")
            errors[str(exc)] += 1
        results.append(result)
    return results, {"schema": SCHEMA, "method_id": method_id, "endpoint": endpoint,
                     "purpose": "evaluation", "source_split": "evaluation", "requests": len(rows),
                     "parseable": sum(r["parseable"] for r in results), "successful": sum(r["success"] for r in results),
                     "native_execution_success": sum(r["native_execution_success"] for r in results),
                     "errors": dict(errors), "global_sample_indices": [r["sample_idx"] for r in results],
                     "original_attempt_ids_preserved": True, "failed_requests_retained": True,
                     "all_request_denominator": expected_requests, "tensor_join_key": "global_sample_idx",
                     "neural_trace_synthesized": False, "site_order_mapping_assumed_for_refinement": False,
                     "new_model_calls": 0, "new_physics_calls": 0, "new_labels": 0,
                     "training_use_allowed": False, "continuous_refined_geometry_preserved": endpoint == "tau800"}


def export_legacy_panel(manifest, *, endpoint, expected_requests, method_id):
    """Explicitly adapt the frozen H1A2/D1/D2 ledger without rewriting its schema."""
    if (manifest.get('schema') != LEGACY_PANEL_SCHEMA or manifest.get('method_id') != method_id
            or manifest.get('endpoint') != endpoint or manifest.get('expected_requests') != expected_requests
            or manifest.get('arm') not in ('control', 'candidate')):
        raise ValueError('legacy panel manifest identity differs from requested evaluation')
    pins = {}
    def pinned(name, jsonl=False, binary=False):
        item = manifest['files'][name]
        path = Path(item['path'])
        if not valid_sha(item.get('sha256')) or sha256_file(path) != item['sha256']:
            raise ValueError(f'frozen legacy {name} hash differs')
        pins[str(path.resolve())] = item['sha256']
        return path if binary else read_rows(path) if jsonl else read_json(path)
    cohort = pinned('cohort', jsonl=True)
    rows = pinned('body', jsonl=True)
    seeds = pinned('seed_ledger', jsonl=True)
    validate_ledger(rows, expected_requests, source_schema=LEGACY_SOURCE_SCHEMA)
    if (len(cohort) != expected_requests or len(seeds) != expected_requests
            or [row.get('cohort_ordinal') for row in cohort] != list(range(expected_requests))
            or [row.get('ordinal') for row in seeds] != list(range(expected_requests))):
        raise ValueError('frozen cohort or random-seed ledger is incomplete/reordered')
    adapted = []
    policy = 'd1' if manifest['arm'] == 'control' else 'd2_safe_axis'
    for index, (row, plan, ledger) in enumerate(zip(rows, cohort, seeds)):
        ordinal_fields = [row.get('ordinal'), row.get('sample_idx'), plan.get('cohort_ordinal'),
                          plan.get('global_raw_ordinal'), ledger.get('ordinal'), ledger.get('sample_idx'),
                          ledger.get('raw_ordinal'), ledger.get('seed_derivation_ordinal')]
        if any(type(value) is not int or value != index for value in ordinal_fields):
            raise ValueError('legacy complete-panel request ordinals or seed derivation were shifted')
        if (plan.get('schema') != 'h1a2_epoch2_exactplan_paired_cohort_v1'
                or ledger.get('schema') != 'h1a2_epoch2_exactplan1200_paired_seed_ledger_v1'
                or row['attempt_id'] != plan.get('planner_attempt_id')
                or row['sample_idx'] != ledger.get('sample_idx')
                or ledger.get('raw_ordinal') != plan.get('global_raw_ordinal')
                or ledger.get('seed_derivation_ordinal') != plan.get('global_raw_ordinal')
                or ledger.get('paired_across_arms') is not True
                or ledger.get('sampling_seed_root') != 17029
                or any(value.get('repeat') != 0 for value in (plan,ledger))
                or row.get('body_noise_seed') != ledger.get('body_noise_seed')
                or row.get('body_prompt_sha256') != plan.get('body_prompt_sha256')
                or row.get('plan_state_sha256') != plan.get('plan_state_sha256')
                or row.get('generation_policy') != policy):
            raise ValueError('legacy body, Plan, arm or noise identity differs')
        if (row.get('body_eligible') != plan.get('body_eligible')
                or any(row.get(name) is not False for name in ('filter_used','repair_used','replacement_used','rerank_used','retry_used'))
                or plan.get('retry_or_replacement_used') is not False):
            raise ValueError('legacy panel contains selection/replacement or changed eligibility')
        if plan.get('body_eligible'):
            if hashlib.sha256(plan['body_prompt'].encode()).hexdigest() != plan['body_prompt_sha256']:
                raise ValueError('legacy rich prompt text differs from its recorded identity')
            actual_plan = hashlib.sha256(json.dumps(plan['plan_state'], sort_keys=True,
                                                   separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()
            if actual_plan != plan['plan_state_sha256']:
                raise ValueError('legacy Plan contents differ from their recorded identity')
        if (row.get('body_graph_complete') and (not row.get('body_generation_complete') or not row.get('body_plan_match'))
                or row.get('body_generation_complete') and not row.get('body_eligible')
                or row.get('status') not in ('succeeded', 'failed')
                or (row.get('status') == 'succeeded') != bool(row.get('body_graph_complete'))
                or row.get('body_graph_complete') and row.get('earliest_failure_stage') is not None):
            raise ValueError('legacy failure/completion flags contradict the endpoint provenance')
        if row.get('body_generation_complete'):
            if hashlib.sha256(row['text'].encode()).hexdigest() != row['raw_body_text_sha256']:
                raise ValueError('legacy native endpoint text changed')
        # These are explicit adapter fields; the original source schema and all
        # original row fields are retained and the cohort itself is pinned.
        adapted.append(dict(row, plan_state=plan.get('plan_state'), body_prompt=plan.get('body_prompt'),
                            planner_record=dict(plan, seed=plan.get('planner_sampling_seed')),
                            parsed=row.get('body_plan_match') is True,
                            attempt_status='planner_failure' if not row['body_eligible'] else
                                'complete' if row.get('body_graph_complete') else 'body_failure'))
    payload, refinement = None, {}
    if endpoint == 'tau800':
        import torch
        config, metrics = pinned('refiner_config'), pinned('refiner_metrics')
        attempts = pinned('refinement_attempts', jsonl=True)
        graphs, tensor_path = pinned('proposal_graphs', binary=True), pinned('refined_pt', binary=True)
        if (config.get('schema') != 'h1_r03e_refiner_run_v1' or metrics.get('schema') != 'h1_r03e_refiner_metrics_v1'
                or config.get('arm') != manifest['arm'] or metrics.get('arm') != manifest['arm']
                or config.get('num_samples') != expected_requests or metrics.get('all_attempt_denominator') != expected_requests
                or metrics.get('status') != 'complete' or config.get('timesteps') != 1000
                or config.get('seed_mode') != 'frozen_h1_ordinal_refiner_noise_seed'
                or config.get('checkpoint_sha256_recorded') != MODEL494_SHA256
                or Path(config['proposal_graphs']).resolve() != graphs.resolve()
                or Path(config['body_attempts']).resolve() != Path(manifest['files']['body']['path']).resolve()
                or Path(config['attempt_ledger']).resolve() != Path(manifest['files']['seed_ledger']['path']).resolve()
                or Path(metrics['output_file']).resolve() != tensor_path.resolve()):
            raise ValueError('frozen legacy refinement input/model/protocol differs')
        for value in (config, metrics):
            if (value.get('diff_steps') != 800 or value.get('num_evals') != 1 or value.get('effective_batch_size') != 1
                    or value.get('repeat') != 0
                    or any(value.get(name) is not False for name in ('filter','repair','replacement','rerank','retry',
                                                                     'new_scientific_seed_per_repeat'))):
                raise ValueError('legacy refinement is not the exact frozen single-draw protocol')
        if len(attempts) != expected_requests:
            raise ValueError('legacy refiner lost failed requests')
        for row, recorded, ledger in zip(rows, attempts, seeds):
            if (recorded.get('schema') != 'h1_r03e_refinement_attempt_v1'
                    or recorded.get('ordinal') != row['ordinal'] or recorded.get('attempt_id') != row['attempt_id']
                    or recorded.get('sample_idx') != row['sample_idx'] or recorded.get('body_noise_seed') != row['body_noise_seed']
                    or recorded.get('body_graph_complete') != row['body_graph_complete']
                    or recorded.get('refiner_sampling_seed') != ledger['refiner_noise_seed']
                    or recorded.get('repeat') != ledger['repeat']
                    or recorded.get('refiner_complete') != row['body_graph_complete']):
                raise ValueError('legacy refinement occurrences/noise differ from the frozen ledger')
        payload = torch.load(tensor_path, map_location='cpu', weights_only=False)
        if 'sample_idx' not in payload or 'sample_indices' in payload:
            raise ValueError('frozen h1_r03e tensor must carry its original unambiguous sample_idx ledger')
        # The historical producer saves sample_idx; the current shared parser
        # calls the same global-index tensor sample_indices. No values move.
        payload = {**payload,'sample_indices':payload['sample_idx']}
        indices = validate_refined_shapes(payload)
        if metrics.get('refiner_complete') != len(indices):
            raise ValueError('legacy refiner tensor count differs from its completion receipt')
        refinement = {'refined_run_config': config, 'refined_metrics': metrics,
                      'legacy_refiner_index_field':'sample_idx','parser_refiner_index_field':'sample_indices',
                      'refined_tensor_values_changed':False}
    result, report = export_records(adapted, endpoint=endpoint, expected_requests=expected_requests,
                                    method_id=method_id, refined_payload=payload, source_schema=LEGACY_SOURCE_SCHEMA)
    for output, ledger in zip(result, seeds):
        output['refiner_noise_seed'] = ledger['refiner_noise_seed']
        output['legacy_arm'] = manifest['arm']
    report.update(source_files_sha256=pins, legacy_source_schema=LEGACY_SOURCE_SCHEMA,
                   source_schema_rewritten=False, retrospective_frozen_panel=True, **refinement)
    return result, report


def load_body_directory(body_dir: Path, expected_requests: int):
    if not ((body_dir / "_SUCCESS").is_file() or (body_dir / "_FAILED").is_file()):
        raise ValueError("body run has no terminal marker; do not export a live partial ledger")
    primary = body_dir / "body_attempts.jsonl"
    raw = body_dir / "raw_generations.jsonl"
    if not primary.is_file():
        primary = raw
    rows = read_rows(primary)
    if raw.is_file() and primary != raw and sha256_file(raw) != sha256_file(primary):
        raise ValueError("body_attempts and raw_generations disagree")
    source_schema = rows[0].get('schema') if rows else None
    if source_schema not in (SOURCE_SCHEMA, EDITOR_SOURCE_SCHEMA):
        raise ValueError('body directory requires an explicit integrated or autonomous editor schema')
    validate_ledger(rows, expected_requests, source_schema=source_schema)
    metrics = read_json(body_dir / "sample_metrics.json")
    if metrics.get("schema") != source_schema or metrics.get("requested_samples") != expected_requests or metrics.get("denominator") != expected_requests:
        raise ValueError("source metrics and request ledger have different denominators")
    expected_counts = {"decoded_samples": sum(row.get("body_generation_complete") is True for row in rows),
                       "parse_success": sum(row.get("body_plan_match") is True for row in rows),
                       "graph_success": sum(row.get("body_graph_complete") is True for row in rows)}
    if any(key in metrics and metrics[key] != count for key, count in expected_counts.items()):
        raise ValueError("source success counts disagree with the actual all-request ledger")
    if (body_dir / "run_config.json").is_file():
        config = read_json(body_dir / "run_config.json")
        if config.get("schema") != source_schema or config.get("expected_requests") != expected_requests:
            raise ValueError("source run configuration differs from the exported request denominator")
    elif source_schema == EDITOR_SOURCE_SCHEMA:
        raise ValueError('autonomous editor body lacks its formal run configuration')
    pins = {str(primary): sha256_file(primary), str(body_dir / "sample_metrics.json"): sha256_file(body_dir / "sample_metrics.json")}
    for name in ("run_config.json", "attempt_ledger.jsonl", "proposal_graphs.pt"):
        if (body_dir / name).is_file():
            pins[str(body_dir / name)] = sha256_file(body_dir / name)
    return rows, {"source_files_sha256": pins, "source_metrics": metrics,
                   "source_run_success_marker": (body_dir / "_SUCCESS").is_file(),
                   "source_run_failed_marker": (body_dir / "_FAILED").is_file()}


def load_tau800(path: Path, *, body_dir: Path):
    import torch
    config = read_json(path.parent / "run_config.json")
    metrics = read_json(path.parent / "refinement_metrics.json")
    if any(record.get("diff_steps") != 800 or record.get("num_evals") != 1 for record in (config, metrics)):
        raise ValueError("tau800 export requires recorded 800-step, one-evaluation refinement")
    if Path(config.get("proposal_graphs", "")).resolve() != (body_dir / "proposal_graphs.pt").resolve():
        raise ValueError("refiner input graphs do not belong to this R03 body directory")
    if Path(metrics.get("output_file", "")).resolve() != path.resolve():
        raise ValueError("refined tensor is not the output named by its terminal metrics")
    body_config = read_json(body_dir/'run_config.json') if (body_dir/'run_config.json').is_file() else {}
    if body_config.get('schema') == EDITOR_SOURCE_SCHEMA:
        required = {'frozen_single_request_seeding':True,'seed_mode':'frozen_h1_ordinal_refiner_noise_seed',
            'seed_before_dataloader':True,'seed_from_graph_field':'refiner_noise_seed','seed_by_sample_index':False,
            'batch_size':1,'effective_batch_size':1,'timesteps':1000,'checkpoint_sha256':MODEL494_SHA256}
        if any(config.get(key) != value for key,value in required.items()):
            raise ValueError('editor tau800 does not use the exact frozen request-seeding/model494 protocol')
        for target,key in ((body_dir/'proposal_graphs.pt','proposal_graphs_sha256'),
                           (body_dir/'attempt_ledger.jsonl','frozen_seed_ledger_sha256'),
                           (Path(config['checkpoint']),'checkpoint_sha256')):
            if not valid_sha(config.get(key)) or sha256_file(target) != config[key]:
                raise ValueError('editor tau800 input/seed/model identity changed')
        seed_rows = read_rows(body_dir/'attempt_ledger.jsonl')
        if [row.get('sample_idx') for row in seed_rows] != list(range(len(seed_rows))):
            raise ValueError('editor tau800 original seed ledger was reordered')
        graphs = torch.load(body_dir/'proposal_graphs.pt',map_location='cpu',weights_only=False)
        graph_ids = [int(graph['sample_idx']) for graph in graphs]
        if (len(set(graph_ids)) != len(graph_ids) or any(not 0 <= index < len(seed_rows) for index in graph_ids)
                or any(graph.get('refiner_noise_seed') != seed_rows[int(graph['sample_idx'])]['refiner_noise_seed'] for graph in graphs)):
            raise ValueError('editor tau800 graph seeds differ from the frozen original request seeds')
    payload = torch.load(path, map_location="cpu", weights_only=False)
    sample_ids = validate_refined_shapes(payload)
    if metrics.get("assigned_proposals") != len(sample_ids):
        raise ValueError("refined completion receipt and actual tensor proposal count differ")
    return payload, {"refined_payload_sha256": sha256_file(path), "refined_run_config": config,
                     "refined_metrics": metrics, "refined_run_config_sha256": sha256_file(path.parent / "run_config.json"),
                     "refined_metrics_sha256": sha256_file(path.parent / "refinement_metrics.json")}


def export_components(manifest: Mapping[str, Any], *, endpoint: str, expected_requests: int, method_id: str):
    """Join same-arm Planner seed components once before common N/U evaluation."""
    if (manifest.get("schema") != COMPONENTS_SCHEMA or manifest.get("endpoint") != endpoint
            or manifest.get("method_id") != method_id or manifest.get("expected_requests") != expected_requests):
        raise ValueError("component manifest and requested arm/endpoint/denominator differ")
    components = manifest.get("components")
    if not isinstance(components, list) or not components:
        raise ValueError("explicit evaluation components are required")
    output, reports, source_dirs, component_ids, identities, pointer_ids = [], [], set(), set(), [], set()
    for component in components:
        if component.get("method_id") != method_id:
            raise ValueError("cross-arm components cannot be pooled")
        component_id = component.get("component_id")
        if not isinstance(component_id, str) or not component_id or component_id in component_ids:
            raise ValueError("component IDs must be explicit and unique")
        component_ids.add(component_id)
        count = exact_index(component.get("expected_requests"), "component expected_requests")
        begin = exact_index(component.get("sample_idx_start"), "component sample_idx_start")
        planner_seed = exact_index(component.get("planner_seed"), "component planner_seed")
        body_dir = Path(component["body_dir"])
        if body_dir.resolve() in source_dirs:
            raise ValueError("one body source cannot be counted twice")
        source_dirs.add(body_dir.resolve())
        rows, source = load_body_directory(body_dir, count)
        if {row["sample_idx"] for row in rows} != set(range(begin, begin + count)):
            raise ValueError("component global sample_idx range differs from its explicit manifest")
        planners = [row.get("planner_record") or {} for row in rows]
        if any(planner.get("seed") != planner_seed for planner in planners):
            raise ValueError("component Planner seed differs from the saved original per-request seed")
        enabled = {planner.get("c3fd_enabled") for planner in planners}
        if len(enabled) != 1 or next(iter(enabled)) not in (True, False):
            raise ValueError("component lacks one unambiguous original C3FD mode")
        config = read_json(body_dir / "run_config.json")
        body_keys = {"frozen_runtime", "b0", "base_model", "temperature", "cfg_scale", "remasking",
                     "geometry_support_scope", "post_construction_repair", "repair_checkpoint"}
        if not body_keys.issubset(config) or any(config[key] is None for key in body_keys - {"repair_checkpoint"}):
            raise ValueError("multi-component body policy identity is incomplete")
        body_identity = {key: config[key] for key in body_keys}
        body_identity["c3fd_enabled"] = next(iter(enabled))
        planner_identity, planner_pin = planner_component_identity(
            component, planners, planner_seed=planner_seed, begin=begin, count=count,
            c3fd_enabled=body_identity["c3fd_enabled"])
        body_identity["planner_contract"] = planner_identity
        identities.append(body_identity)
        for planner in planners:
            pointer = (planner.get("r03_control") or {}).get("pointer_sha256")
            if pointer is not None:
                pointer_ids.add(pointer)
        refined_path = component.get("refined_pt")
        if (endpoint == "tau800") != (refined_path is not None):
            raise ValueError("each tau800 component needs its own corresponding refined tensor")
        payload, refinement = (None, {}) if refined_path is None else load_tau800(Path(refined_path), body_dir=body_dir)
        if refined_path is not None:
            refiner, refiner_pin = pinned_json(component.get("refiner_identity"), "refiner_identity")
            if (refiner.get("checkpoint_sha256") != MODEL494_SHA256 or not refiner.get("checkpoint")
                    or Path(refiner["checkpoint"]).resolve() != Path(refinement["refined_run_config"].get("checkpoint", "")).resolve()
                    or ("tau" in refiner and refiner["tau"] != 800)):
                raise ValueError("actual refiner identity differs from the frozen model494 endpoint")
            refinement.update(refiner_identity=refiner, refiner_identity_file=refiner_pin)
            body_identity["refiner_checkpoint_sha256"] = refiner["checkpoint_sha256"]
        body_identity["refiner_identity"] = (None if refined_path is None else
            {key: refinement["refined_run_config"].get(key) for key in
             ("checkpoint", "diff_steps", "num_evals", "timesteps", "run_type",
              "seed_by_sample_index", "seed_from_graph_field")})
        selected, report = export_records(rows, endpoint=endpoint, expected_requests=count,
                                          method_id=method_id, refined_payload=payload)
        for row in selected:
            row.update(component_id=component_id, component_planner_seed=planner_seed,
                       component_body_seed=config.get("seed"), source_component_ordinal=row["evaluation_ordinal"],
                       source_body_directory=str(body_dir.resolve()))
        output.extend(selected)
        reports.append({**report, **source, **refinement, "component_id": component_id,
                        "planner_seed": planner_seed, "body_directory": str(body_dir.resolve()),
                        "planner_run_config_file": planner_pin, "component_manifest": dict(component)})
    if any(identity != identities[0] for identity in identities[1:]) or len(pointer_ids) > 1:
        raise ValueError("component source identities conflict; arms or checkpoints cannot be mixed")
    ids = [row["sample_idx"] for row in output]
    attempts = [row["original_attempt_id"] for row in output]
    if (len(output) != expected_requests or len(set(ids)) != len(ids)
            or set(ids) != set(range(expected_requests)) or len(set(attempts)) != len(attempts)):
        raise ValueError("global pooled request ledger has duplicates, gaps, or conflicting original attempt IDs")
    output.sort(key=lambda row: row["sample_idx"])
    for ordinal, row in enumerate(output):
        row["evaluation_ordinal"] = ordinal
    report = {"schema": SCHEMA, "method_id": method_id, "endpoint": endpoint,
              "purpose": "evaluation", "source_split": "evaluation", "requests": expected_requests,
              "all_request_denominator": expected_requests, "parseable": sum(r["parseable"] for r in output),
              "successful": sum(r["success"] for r in output),
              "native_execution_success": sum(r["native_execution_success"] for r in output),
              "errors": dict(Counter(r["artifact_error"] for r in output if r["artifact_error"])),
              "components": reports, "component_count": len(reports), "shared_method_identity": identities[0],
              "observed_pointer_sha256": sorted(pointer_ids), "global_sample_indices": ids,
              "identity_binding_scope": "pinned observed P0 load identity, actual native prompt/domain files, sampler code hashes, body policy, and recorded actual model494 weight hash; base-model weight shards are not rehashed here",
              "pooling": "one ordered all-request ledger; run common N/U once over the pooled structures",
              "per_component_NU_sums_used": False, "tensor_join_key": "global_sample_idx",
              "original_attempt_ids_preserved": True, "failed_requests_retained": True,
              "neural_trace_synthesized": False, "site_order_mapping_assumed_for_refinement": False,
              "new_model_calls": 0, "new_physics_calls": 0, "new_labels": 0, "training_use_allowed": False}
    report["global_sample_indices"] = [row["sample_idx"] for row in output]
    return output, report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--body-dir", type=Path)
    source.add_argument("--input-manifest", type=Path, help="Explicit same-arm seed components, pooled once before N/U")
    parser.add_argument("--endpoint", choices=("native", "tau800"), required=True)
    parser.add_argument("--refined-pt", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-requests", type=int, required=True)
    parser.add_argument("--method-id", required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error("output-dir must be new; existing endpoint identities are not overwritten")
    if args.input_manifest is not None and args.refined_pt is not None:
        parser.error("component manifests contain their own refined_pt paths")
    if args.body_dir is not None and (args.endpoint == "tau800") != (args.refined_pt is not None):
        parser.error("--refined-pt is required exactly for --endpoint tau800")
    if args.input_manifest is not None:
        manifest = read_json(args.input_manifest)
        exporter = export_legacy_panel if manifest.get('schema') == LEGACY_PANEL_SCHEMA else export_components
        output, report = exporter(manifest, endpoint=args.endpoint,
                                 expected_requests=args.expected_requests, method_id=args.method_id)
        report.update(input_manifest=str(args.input_manifest.resolve()), input_manifest_sha256=sha256_file(args.input_manifest))
    else:
        rows, source_report = load_body_directory(args.body_dir, args.expected_requests)
        payload, refinement = (None, {}) if args.refined_pt is None else load_tau800(args.refined_pt, body_dir=args.body_dir)
        output, report = export_records(rows, endpoint=args.endpoint, expected_requests=args.expected_requests,
                                        method_id=args.method_id, refined_payload=payload, source_schema=rows[0]['schema'])
        report.update(**source_report, **refinement, body_directory=str(args.body_dir.resolve()))
    report.update(
                  authoritative_parser_sha256=sha256_file(ROOT / "scripts" / "assemble_grounding_repeat.py"),
                  evaluation_entrypoints={name: {"path": str(ROOT / "scripts" / name),
                                                "sha256": sha256_file(ROOT / "scripts" / name)}
                                         for name in ("label_programmed_paths.py", "evaluate_programmed_paths.py")},
                  evaluation_contract="call label_programmed_paths with purpose=evaluation, then unchanged evaluate_programmed_paths with frozen_config/official_cache; no scoring in this adapter")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    paths = args.output_dir / "paths.jsonl"
    with paths.open("x", encoding="utf-8") as stream:
        for row in output:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    report["paths_sha256"] = sha256_file(paths)
    for name in ("ARTIFACT_FINAL.json", "EVALUATION_INPUTS_FINAL.json"):
        (args.output_dir / name).write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False,
                                                     allow_nan=False) + "\n", encoding="utf-8")
    (args.output_dir / "_SUCCESS").touch()
    print(json.dumps({k: report[k] for k in ("requests", "parseable", "endpoint", "method_id", "paths_sha256")}), flush=True)


if __name__ == "__main__":
    main()
