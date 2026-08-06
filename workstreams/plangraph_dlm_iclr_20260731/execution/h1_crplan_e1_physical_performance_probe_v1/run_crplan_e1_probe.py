#!/usr/bin/env python3
"""Run the bounded H1 CR-Plan E1 same-node physical-performance probe."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import time
from typing import Any, Mapping, Sequence

from crystal_dlm.h1_crplan import (
    CRPLAN_MODES,
    CRPLAN_SCHEMA,
    CRPlanDeadEndError,
    CRPlanIdentityError,
    CRPlanLogitsProcessor,
    CRPlanTokenVocabulary,
    PlanFormulaCursor,
    TokenSupportBundle,
    certificate_for_symbol_counts,
    load_frozen_smact_table,
    validate_crplan_parsed_identity,
)
from crystal_dlm.h1_llm_planner import (
    canonical_plan_record_for_style,
    clean_generated_plan_text,
    disable_peft_bnb_autodetect,
    ensure_peft_cache_compat,
    format_planner_prompt,
    load_llama3_compatible_config,
)
from crystal_dlm.ordinal_rng import derive_ordinal_seed
from crystal_dlm.r5_plan_body import (
    has_plan_end_marker,
    has_plan_tail_after_end_marker,
)


ATTEMPTS = 18
BASE_SEED = 26080517
REFERENCE_ORDINALS = (2, 11)
MODES = ("off", "terminal_only", "full_prefix")
ORDER_CYCLE = (
    ("off", "terminal_only", "full_prefix"),
    ("terminal_only", "full_prefix", "off"),
    ("full_prefix", "off", "terminal_only"),
)
PROMPT_STYLE = "h1_rich_plan_v1"
MAX_NEW_TOKENS = 96
TEMPERATURE = 0.9
TOP_P = 0.95
TOP_K = 50
MAX_ATOMS = 20
MISSING_STATE_POLICY = "fail_closed"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(value), ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def build_schedule(attempts: int = ATTEMPTS) -> list[dict[str, Any]]:
    if int(attempts) <= 0 or int(attempts) % len(ORDER_CYCLE) != 0:
        raise ValueError("E1 attempts must be a positive multiple of three")
    schedule: list[dict[str, Any]] = []
    sequence = 0
    for ordinal in range(int(attempts)):
        for position, mode in enumerate(ORDER_CYCLE[ordinal % 3]):
            schedule.append(
                {
                    "sequence": sequence,
                    "ordinal": ordinal,
                    "position_within_ordinal": position,
                    "mode": mode,
                }
            )
            sequence += 1
    return schedule


class GeneratedPlanEndStoppingCriteria:
    """Transformers stopping criterion without importing Transformers at load."""

    marker = re.compile(r"(?i)\bend\s*:\s*plan\b")

    def __init__(self, tokenizer: Any, start_length: int) -> None:
        self.tokenizer = tokenizer
        self.start_length = int(start_length)

    def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
        del scores, kwargs
        generated_ids = input_ids[:, self.start_length :]
        if int(generated_ids.numel()) == 0:
            return False
        decoded = self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return all(self.marker.search(text) is not None for text in decoded)


class ScalarBundleVocabulary:
    """Scalar support oracle exposed through the combined-bundle interface."""

    def __init__(self, base: CRPlanTokenVocabulary) -> None:
        self.base = base
        self.fragments = base.fragments
        self.eos_token_id = base.eos_token_id
        self.vocab_size = base.vocab_size
        self.fragment_sha256 = base.fragment_sha256

    def support_bundle(self, cursor: PlanFormulaCursor) -> TokenSupportBundle:
        supports = {
            mode: self.base.support_scalar_reference(
                dataclasses.replace(cursor, mode=mode)
            )
            for mode in CRPLAN_MODES
        }
        return TokenSupportBundle(
            grammar_only=supports["grammar_only"],
            terminal_only=supports["terminal_only"],
            full_prefix=supports["full_prefix"],
        )


def load_planner(
    model_path: str,
    checkpoint_path: str,
    device: Any,
) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer_source = (
        checkpoint_path
        if (Path(checkpoint_path) / "tokenizer_config.json").exists()
        else model_path
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        trust_remote_code=True,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    config = load_llama3_compatible_config(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
    )
    ensure_peft_cache_compat()
    disable_peft_bnb_autodetect()
    from peft import PeftModel

    model = PeftModel.from_pretrained(
        model,
        checkpoint_path,
    )
    model.to(device).eval()
    return model, tokenizer


def cuda_sync(torch_module: Any, device: Any) -> None:
    torch_module.cuda.synchronize(device)


def common_model_warmup(
    model: Any,
    tokenizer: Any,
    device: Any,
) -> float:
    import torch

    prompt = format_planner_prompt(
        tokenizer,
        sample_idx=None,
        prompt_style=PROMPT_STYLE,
    )
    encoded = tokenizer(
        [prompt],
        padding=True,
        add_special_tokens=False,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    cuda_sync(torch, device)
    started = time.perf_counter()
    with torch.inference_mode():
        model(**encoded, use_cache=False)
    cuda_sync(torch, device)
    return time.perf_counter() - started


def mode_context(tokenizer: Any) -> dict[str, Any]:
    started = time.perf_counter()
    reachability = load_frozen_smact_table(
        max_atoms=MAX_ATOMS,
        missing_state_policy=MISSING_STATE_POLICY,
    )
    vocabulary = CRPlanTokenVocabulary.from_tokenizer(tokenizer)
    return {
        "reachability": reachability,
        "vocabulary": vocabulary,
        "setup_seconds": time.perf_counter() - started,
    }


def _diagnostic_step_sha256(diagnostics: Mapping[str, Any] | None) -> str | None:
    if not isinstance(diagnostics, Mapping):
        return None
    return sha256_json(diagnostics.get("steps") or [])


def sample_one(
    *,
    model: Any,
    tokenizer: Any,
    device: Any,
    mode: str,
    ordinal: int,
    sequence: int | None,
    context: Mapping[str, Any] | None,
    scalar_reference: bool,
) -> dict[str, Any]:
    import torch
    from transformers import LogitsProcessorList, StoppingCriteriaList

    if mode not in MODES:
        raise ValueError(f"unknown E1 mode {mode!r}")
    if mode == "off" and context is not None:
        raise ValueError("off mode must not carry a constraint context")
    if mode != "off" and context is None:
        raise ValueError("constrained mode requires a context")

    prompt = format_planner_prompt(
        tokenizer,
        sample_idx=None,
        prompt_style=PROMPT_STYLE,
    )
    encoded = tokenizer(
        [prompt],
        padding=True,
        add_special_tokens=False,
        return_tensors="pt",
    )
    input_ids_cpu = encoded["input_ids"]
    input_ids = input_ids_cpu.to(device)
    attention_mask = encoded["attention_mask"].to(device)
    sampling_seed = derive_ordinal_seed(
        BASE_SEED,
        sample_idx=int(ordinal),
        stage="planner_sampling",
        role="shared",
    )
    torch.manual_seed(sampling_seed)
    torch.cuda.manual_seed_all(sampling_seed)

    processor = None
    if context is not None:
        vocabulary = context["vocabulary"]
        if scalar_reference:
            vocabulary = ScalarBundleVocabulary(vocabulary)
        processor = CRPlanLogitsProcessor(
            tokenizer,
            vocabulary,
            context["reachability"],
            prompt_width=int(input_ids.shape[1]),
            mode=mode,
            attempt_ordinal=int(ordinal),
        )

    generation_error: CRPlanDeadEndError | None = None
    generated_ids = None
    torch.cuda.reset_peak_memory_stats(device)
    cuda_sync(torch, device)
    started = time.perf_counter()
    try:
        generate_kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": MAX_NEW_TOKENS,
            "do_sample": True,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "top_k": TOP_K,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "stopping_criteria": StoppingCriteriaList(
                [
                    GeneratedPlanEndStoppingCriteria(
                        tokenizer,
                        int(input_ids.shape[1]),
                    )
                ]
            ),
        }
        if processor is not None:
            generate_kwargs["logits_processor"] = LogitsProcessorList(
                [processor]
            )
            generate_kwargs["renormalize_logits"] = True
        with torch.inference_mode():
            outputs = model.generate(**generate_kwargs)
        generated_ids = outputs[:, input_ids.shape[1] :]
    except CRPlanDeadEndError as exc:
        generation_error = exc
    cuda_sync(torch, device)
    latency_seconds = time.perf_counter() - started

    generated_token_ids: list[int] | None
    if generated_ids is None:
        generated_token_ids = None
        raw_decoded = ""
    else:
        generated_token_ids = [
            int(value) for value in generated_ids[0].tolist()
        ]
        raw_decoded = tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    diagnostics = (
        None
        if processor is None
        else processor.diagnostics(generated_token_ids)
    )
    raw_model_text = clean_generated_plan_text(
        raw_decoded,
        prompt_style=PROMPT_STYLE,
        truncate_after_marker=False,
    )
    raw_plan_text = clean_generated_plan_text(
        raw_model_text,
        prompt_style=PROMPT_STYLE,
        truncate_after_marker=True,
    )
    record: dict[str, Any] = {
        "schema": "h1_crplan_e1_attempt_v1",
        "sample_idx": int(ordinal),
        "sequence": sequence,
        "mode": mode,
        "execution_variant": (
            "scalar_reference" if scalar_reference else "optimized_primary"
        ),
        "planner_sampling_seed": int(sampling_seed),
        "seed_mode": "stateless_ordinal_v1",
        "planner_input_prompt_sha256": sha256_bytes(
            prompt.encode("utf-8")
        ),
        "planner_input_ids_sha256": sha256_json(
            [int(value) for value in input_ids_cpu[0].tolist()]
        ),
        "generated_token_ids": generated_token_ids,
        "generated_token_count": (
            None
            if generated_token_ids is None
            else len(generated_token_ids)
        ),
        "generated_token_ids_sha256": (
            None
            if generated_token_ids is None
            else sha256_json(generated_token_ids)
        ),
        "raw_model_text": raw_model_text,
        "raw_plan_text": raw_plan_text,
        "raw_model_text_sha256": sha256_bytes(
            raw_model_text.encode("utf-8")
        ),
        "plan_end_marker_present": has_plan_end_marker(raw_plan_text),
        "plan_tail_after_end_marker": has_plan_tail_after_end_marker(
            raw_model_text
        ),
        "planner_generation_latency_sec": latency_seconds,
        "cuda_peak_allocated_bytes": int(
            torch.cuda.max_memory_allocated(device)
        ),
        "cuda_peak_reserved_bytes": int(
            torch.cuda.max_memory_reserved(device)
        ),
        "process_max_rss_kib": int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        ),
        "crplan_schema": None if mode == "off" else CRPLAN_SCHEMA,
        "crplan_diagnostics": diagnostics,
        "diagnostic_steps_sha256": _diagnostic_step_sha256(diagnostics),
        "parsed": False,
        "formula_parse": False,
        "generation_error": generation_error is not None,
        "reason": (
            None if generation_error is None else type(generation_error).__name__
        ),
        "message": (
            None if generation_error is None else str(generation_error)
        ),
        "fail_closed": generation_error is not None,
        "retry": false_value(),
        "replacement": false_value(),
        "repair": false_value(),
        "filter": false_value(),
        "rerank": false_value(),
        "fallback": false_value(),
        "body_rerun": false_value(),
        "refiner_rerun": false_value(),
        "direct_rerun": false_value(),
        "sun_rerun": false_value(),
        "network": false_value(),
    }
    if generation_error is not None:
        return record

    try:
        plan_record = canonical_plan_record_for_style(
            raw_plan_text,
            sample_idx=int(ordinal),
            max_atoms=MAX_ATOMS,
            prompt_style=PROMPT_STYLE,
        )
        plan = plan_record["plan_state"]
        record.update(
            {
                "parsed": True,
                "formula_parse": True,
                "plan_state": plan,
                "formula": plan.get("formula"),
            }
        )
        if context is not None:
            identity = validate_crplan_parsed_identity(
                raw_model_text=raw_model_text,
                prompt_style=PROMPT_STYLE,
                parsed_symbols=plan.get("elements") or (),
                parsed_counts=plan.get("counts") or (),
                diagnostics=diagnostics,
                mode=mode,
            )
            certificate = certificate_for_symbol_counts(
                context["reachability"],
                plan.get("elements") or (),
                plan.get("counts") or (),
            )
            if (
                not isinstance(diagnostics, Mapping)
                or diagnostics.get("terminal_certificate") != certificate
            ):
                raise CRPlanIdentityError(
                    "decoder terminal certificate does not equal the "
                    "independently parsed certificate"
                )
            record["crplan_identity"] = identity
            record["crplan_terminal_certificate"] = certificate
    except CRPlanIdentityError as exc:
        record.update(
            {
                "parsed": False,
                "formula_parse": False,
                "reason": type(exc).__name__,
                "message": str(exc),
                "fail_closed": True,
            }
        )
    except Exception as exc:  # parser/model-output failure, not retried
        record.update(
            {
                "parsed": False,
                "formula_parse": False,
                "reason": type(exc).__name__,
                "message": str(exc),
            }
        )
    return record


def false_value() -> bool:
    """Make forbidden-operation fields visibly generated, not copied."""

    return False


def trace_support_audit(
    *,
    tokenizer: Any,
    rows_by_mode: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    started = time.perf_counter()
    reachability = load_frozen_smact_table(
        max_atoms=MAX_ATOMS,
        missing_state_policy=MISSING_STATE_POLICY,
    )
    vocabulary = CRPlanTokenVocabulary.from_tokenizer(tokenizer)
    seen: set[tuple[Any, ...]] = set()
    errors: list[str] = []
    checked_formula_cursors = 0
    sampled_legal_checks = 0
    diagnostic_step_checks = 0

    for mode in ("terminal_only", "full_prefix"):
        for row in rows_by_mode[mode]:
            token_ids = row.get("generated_token_ids")
            if not isinstance(token_ids, list):
                errors.append(
                    f"{mode}:{row.get('sample_idx')}:missing token trace"
                )
                continue
            cursor = PlanFormulaCursor(
                mode=mode,
                reachability=reachability,
            )
            masked_token_count = 0
            for token_id_value in token_ids:
                if cursor.phase == "after_formula":
                    break
                token_id = int(token_id_value)
                if not 0 <= token_id < vocabulary.vocab_size:
                    errors.append(
                        f"{mode}:{row.get('sample_idx')}:token outside vocab"
                    )
                    break
                if cursor.phase == "formula_value":
                    signature = vocabulary._bundle_signature(cursor)
                    bundle = vocabulary.support_bundle(cursor)
                    if signature not in seen:
                        seen.add(signature)
                        checked_formula_cursors += 1
                        for check_mode in CRPLAN_MODES:
                            optimized = bundle.for_mode(check_mode)
                            scalar = vocabulary.support_scalar_reference(
                                dataclasses.replace(cursor, mode=check_mode)
                            )
                            if optimized != scalar:
                                errors.append(
                                    f"{mode}:{row.get('sample_idx')}:"
                                    f"{check_mode}:support mismatch:"
                                    f"{repr(cursor.signature())}"
                                )
                    selected = bundle.for_mode(mode)
                    sampled_legal_checks += 1
                    if token_id not in selected.token_ids:
                        errors.append(
                            f"{mode}:{row.get('sample_idx')}:"
                            f"sampled illegal token {token_id}"
                        )
                masked_token_count += 1
                try:
                    cursor = cursor.feed(vocabulary.fragments[token_id])
                except Exception as exc:  # noqa: BLE001
                    errors.append(
                        f"{mode}:{row.get('sample_idx')}:trace feed:"
                        f"{type(exc).__name__}:{exc}"
                    )
                    break
            diagnostics = row.get("crplan_diagnostics")
            steps = (
                diagnostics.get("steps")
                if isinstance(diagnostics, Mapping)
                else None
            )
            if not isinstance(steps, list) or len(steps) != masked_token_count:
                errors.append(
                    f"{mode}:{row.get('sample_idx')}:"
                    f"diagnostic step count mismatch"
                )
            else:
                diagnostic_step_checks += 1

    return {
        "schema": "h1_crplan_e1_actual_trace_support_audit_v1",
        "all_equal": not errors,
        "errors": errors[:32],
        "error_count": len(errors),
        "checked_formula_cursor_count": checked_formula_cursors,
        "unique_bundle_signature_count": len(seen),
        "sampled_legal_check_count": sampled_legal_checks,
        "diagnostic_step_trace_count": diagnostic_step_checks,
        "optimized_scalar_modes_checked": list(CRPLAN_MODES),
        "elapsed_seconds": time.perf_counter() - started,
        "logical_state_gate_reused_from_v4": False,
        "logical_state_definition_changed": False,
        "model_rerun": False,
        "network": False,
    }


def reference_parity(
    optimized_rows: Sequence[Mapping[str, Any]],
    scalar_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    optimized = {
        int(row["sample_idx"]): row for row in optimized_rows
    }
    scalar = {int(row["sample_idx"]): row for row in scalar_rows}
    fields = (
        "planner_sampling_seed",
        "planner_input_prompt_sha256",
        "planner_input_ids_sha256",
        "generated_token_ids",
        "generated_token_ids_sha256",
        "raw_model_text",
        "raw_model_text_sha256",
        "parsed",
        "formula_parse",
        "reason",
        "crplan_terminal_certificate",
        "diagnostic_steps_sha256",
    )
    comparisons: list[dict[str, Any]] = []
    for ordinal in REFERENCE_ORDINALS:
        left = optimized.get(ordinal)
        right = scalar.get(ordinal)
        equal_fields = {
            field: (
                left is not None
                and right is not None
                and left.get(field) == right.get(field)
            )
            for field in fields
        }
        comparisons.append(
            {
                "ordinal": ordinal,
                "equal_fields": equal_fields,
                "all_equal": all(equal_fields.values()),
                "optimized_latency_sec": (
                    None
                    if left is None
                    else left.get("planner_generation_latency_sec")
                ),
                "scalar_latency_sec": (
                    None
                    if right is None
                    else right.get("planner_generation_latency_sec")
                ),
            }
        )
    return {
        "schema": "h1_crplan_e1_scalar_reference_parity_v1",
        "reference_ordinals": list(REFERENCE_ORDINALS),
        "comparisons": comparisons,
        "all_equal": (
            len(comparisons) == len(REFERENCE_ORDINALS)
            and all(value["all_equal"] for value in comparisons)
        ),
        "scalar_reruns_excluded_from_primary_latency": True,
        "scalar_reruns_included_in_total_job_wall": True,
    }


def run(args: argparse.Namespace) -> None:
    import torch

    run_root = args.run_root.resolve()
    probe_root = run_root / "probe"
    if probe_root.exists():
        raise FileExistsError(f"refusing to overwrite {probe_root}")
    probe_root.mkdir(parents=True)
    job_started = time.perf_counter()
    schedule = build_schedule()
    atomic_write_json(
        probe_root / "schedule.json",
        {
            "schema": "h1_crplan_e1_balanced_schedule_v1",
            "attempts": ATTEMPTS,
            "base_seed": BASE_SEED,
            "reference_ordinals": list(REFERENCE_ORDINALS),
            "order_cycle": [list(value) for value in ORDER_CYCLE],
            "schedule": schedule,
        },
    )

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"expected one CUDA device, observed {torch.cuda.device_count()}"
        )
    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(device)
    if "A800" not in gpu_name:
        raise RuntimeError(f"expected A800, observed {gpu_name!r}")

    load_started = time.perf_counter()
    model, tokenizer = load_planner(
        args.model_path,
        args.checkpoint_path,
        device,
    )
    cuda_sync(torch, device)
    model_load_seconds = time.perf_counter() - load_started
    warmup_seconds = common_model_warmup(model, tokenizer, device)
    contexts = {
        mode: mode_context(tokenizer)
        for mode in ("terminal_only", "full_prefix")
    }
    fragment_shas = {
        mode: contexts[mode]["vocabulary"].fragment_sha256
        for mode in contexts
    }
    contract_shas = {
        mode: contexts[mode][
            "reachability"
        ].constraint_contract_sha256
        for mode in contexts
    }
    if len(set(fragment_shas.values())) != 1:
        raise RuntimeError("mode token fragments differ")
    if len(set(contract_shas.values())) != 1:
        raise RuntimeError("mode constraint contracts differ")

    setup = {
        "schema": "h1_crplan_e1_setup_v1",
        "gpu_name": gpu_name,
        "torch_version": torch.__version__,
        "model_load_seconds": model_load_seconds,
        "common_model_forward_warmup_seconds": warmup_seconds,
        "mode_support_setup_seconds": {
            mode: contexts[mode]["setup_seconds"] for mode in contexts
        },
        "token_fragment_sha256": next(iter(fragment_shas.values())),
        "constraint_contract_sha256": next(iter(contract_shas.values())),
        "vocab_size": len(tokenizer),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "mode_specific_support_warmup": False,
        "network": False,
    }
    atomic_write_json(probe_root / "setup.json", setup)

    rows_by_mode: dict[str, list[dict[str, Any]]] = {
        mode: [] for mode in MODES
    }
    for item in schedule:
        mode = str(item["mode"])
        row = sample_one(
            model=model,
            tokenizer=tokenizer,
            device=device,
            mode=mode,
            ordinal=int(item["ordinal"]),
            sequence=int(item["sequence"]),
            context=contexts.get(mode),
            scalar_reference=False,
        )
        rows_by_mode[mode].append(row)
        append_jsonl(
            probe_root / "primary" / mode / "raw_generations.jsonl",
            row,
        )
        print(
            json.dumps(
                {
                    "event": "primary_attempt_complete",
                    "sequence": item["sequence"],
                    "ordinal": item["ordinal"],
                    "mode": mode,
                    "latency_sec": row["planner_generation_latency_sec"],
                    "parsed": row["parsed"],
                    "reason": row["reason"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    trace_audit = trace_support_audit(
        tokenizer=tokenizer,
        rows_by_mode=rows_by_mode,
    )
    atomic_write_json(
        probe_root / "trace_support_audit.json",
        trace_audit,
    )
    print(
        json.dumps(
            {
                "event": "trace_support_audit_complete",
                "all_equal": trace_audit["all_equal"],
                "cursor_count": trace_audit["checked_formula_cursor_count"],
                "elapsed_sec": trace_audit["elapsed_seconds"],
            },
            sort_keys=True,
        ),
        flush=True,
    )

    scalar_context = mode_context(tokenizer)
    scalar_rows: list[dict[str, Any]] = []
    for ordinal in REFERENCE_ORDINALS:
        row = sample_one(
            model=model,
            tokenizer=tokenizer,
            device=device,
            mode="full_prefix",
            ordinal=int(ordinal),
            sequence=None,
            context=scalar_context,
            scalar_reference=True,
        )
        scalar_rows.append(row)
        append_jsonl(
            probe_root
            / "reference"
            / "full_prefix_scalar"
            / "raw_generations.jsonl",
            row,
        )
        print(
            json.dumps(
                {
                    "event": "scalar_reference_attempt_complete",
                    "ordinal": ordinal,
                    "latency_sec": row["planner_generation_latency_sec"],
                    "parsed": row["parsed"],
                    "reason": row["reason"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    parity = reference_parity(
        rows_by_mode["full_prefix"],
        scalar_rows,
    )
    atomic_write_json(probe_root / "reference_parity.json", parity)

    runner_report = {
        "schema": "h1_crplan_e1_runner_report_v1",
        "status": "runner_complete",
        "attempts_per_mode": ATTEMPTS,
        "primary_attempt_count": sum(
            len(value) for value in rows_by_mode.values()
        ),
        "scalar_reference_attempt_count": len(scalar_rows),
        "trace_support_parity": trace_audit["all_equal"],
        "scalar_reference_parity": parity["all_equal"],
        "setup": setup,
        "job_wall_seconds_through_runner": time.perf_counter() - job_started,
        "logical_state_gate_reused_from_v4": False,
        "v4_terminal_modified": False,
        "body_rerun": False,
        "refiner_rerun": False,
        "direct_rerun": False,
        "sun_rerun": False,
        "network": False,
        "training": False,
        "downstream": False,
    }
    atomic_write_json(probe_root / "runner_report.json", runner_report)
    print(json.dumps(runner_report, sort_keys=True), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
