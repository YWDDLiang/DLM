#!/usr/bin/env python3
"""Run the byte-frozen R03 constructor on an explicit all-request Plan ledger.

The original safe-axis sampler, PlanGraph grouping, schema/prefill helpers,
trained B0 embedding/head loader, and stateless noise are imported from the
registered historical runtime.  Current crystal_dlm modules cannot replace
them through PYTHONPATH precedence.  No input row is repeated to fill a quota.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
from types import ModuleType
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

SCHEMA = "r03_integrated_body_v1"
EDITOR_PANEL_SCHEMA = "r03_autonomous_editor_body_v1"
FROZEN_RUNNER_SHA256 = "9da1379fbe33dc9c0b76fdcfb2497f24fc973847c366adefb3753e9b647038ca"
SAFE_AXIS_SHA256 = "754487d39ababb95cfb4e2cecc20cad5fac0a90b0fc150306b8316e789f44df9"
RUNTIME_MANIFEST_SHA256 = "ed2223e2a931fbc13b16113bbc7a1b28bcb2deceea310652a33db215623d9675"
B0_ADAPTER_SHA256 = "5c39976b6ab237cbab32cbfeb1c23a557571e1c7d2b60c1e60cbb450166ae76d"
B0_BYTES = 6391016776
B0_TOKENIZER_JSON_SHA256 = "3a21588abca8e56155cc7b6cabb81df51992ccd2e89704aec770912f24e75509"
B0_TOKENIZER_CONFIG_SHA256 = "8e89acaa54a8fb8fc7d228165ac483f61b7fef7c4c9761214092511190f75de2"
B0_VOCAB_SHA256 = "3acc073da85047265769f2dccd93543fa9d7cbfa95021aef54ef282b13ce2f37"
_FROZEN_PREFIXES = ("crystal_dlm", "scripts")
_FROZEN_NAMES = {"paired_llada", "paired_noise", "safe_axis_schedule"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=True, sort_keys=True,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()


def require_file_hash(path: Path, expected: str) -> None:
    observed = file_sha256(path)
    if observed != expected:
        raise ValueError(f"frozen file identity changed: {path}: {observed}")


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            rows.append(row)
    if not rows:
        raise ValueError("input request ledger is empty")
    return rows


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _frozen_name(name: str) -> bool:
    return name in _FROZEN_NAMES or any(name == prefix or name.startswith(prefix + ".") for prefix in _FROZEN_PREFIXES)


@dataclass
class FrozenRuntime:
    root: Path
    module: Any = None
    modules: dict[str, ModuleType] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)


@contextmanager
def frozen_imports(runtime: FrozenRuntime):
    """Temporarily install historical names, restoring current modules after use.

    This CLI runs each GPU in a separate process and never calls this context
    concurrently.  Generation functions retain their frozen module globals;
    the context also protects any delayed imports in old loader/graph helpers.
    """
    prior = {name: module for name, module in list(sys.modules.items()) if _frozen_name(name)}
    paths = list(sys.path)
    for name in prior:
        del sys.modules[name]
    sys.modules.update(runtime.modules)
    sys.path[:0] = [str(runtime.root), str(runtime.root / "runtime")]
    try:
        yield
    finally:
        runtime.modules.update({name: module for name, module in list(sys.modules.items()) if _frozen_name(name)})
        for name in list(sys.modules):
            if _frozen_name(name):
                del sys.modules[name]
        sys.modules.update(prior)
        sys.path[:] = paths


def load_frozen_runtime(root: Path) -> FrozenRuntime:
    root = root.resolve()
    require_file_hash(root / "run_schedule256.py", FROZEN_RUNNER_SHA256)
    require_file_hash(root / "safe_axis_schedule.py", SAFE_AXIS_SHA256)
    manifest_path = root / "H1_BODY_RUNTIME_SHA256.json"
    require_file_hash(manifest_path, RUNTIME_MANIFEST_SHA256)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pins = dict(manifest["files"])
    for relative, digest in pins.items():
        path = (root / relative).resolve()
        if root not in path.parents:
            raise ValueError("frozen runtime manifest escaped its root")
        require_file_hash(path, digest)
    runtime = FrozenRuntime(root=root)
    with frozen_imports(runtime):
        spec = importlib.util.spec_from_file_location("_r03_native_safeaxis_binding", root / "run_schedule256.py")
        if spec is None or spec.loader is None:
            raise ImportError("cannot import frozen safe-axis runner")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        runtime.module = module
    # Pin every actually loaded historical helper as additional provenance,
    # including PlanGraph files not listed in the smaller archived manifest.
    loaded = {}
    for name, module in runtime.modules.items():
        location = getattr(module, "__file__", None)
        if location and Path(location).is_file():
            path = Path(location).resolve()
            if root not in path.parents:
                raise ValueError(f"historical import escaped its runtime: {name}: {path}")
            loaded[str(path.relative_to(root))] = file_sha256(path)
    runtime.provenance = {
        "root": str(root), "runner_sha256": FROZEN_RUNNER_SHA256,
        "safe_axis_sha256": SAFE_AXIS_SHA256, "registered_runtime_manifest_sha256": RUNTIME_MANIFEST_SHA256,
        "registered_file_pins": pins, "loaded_historical_files": loaded,
        "constructor": "original_generate_paired_exact_plan",
    }
    return runtime


def request_sample_idx(row: Mapping[str, Any], ordinal: int) -> int:
    value = row.get("sample_idx", row.get("ordinal", row.get("cohort_ordinal", ordinal)))
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("every request needs a nonnegative integer sample_idx/ordinal")
    return value


def upstream_failure(row: Mapping[str, Any]) -> str | None:
    if row.get("body_eligible") is False or row.get("parsed") is False:
        return str(row.get("ineligible_reason", row.get("reason", "planner_failure")))
    if "attempt_status" in row and row["attempt_status"] != "complete":
        return str(row.get("ineligible_reason", row.get("reason", row["attempt_status"])))
    if not isinstance(row.get("plan_state"), Mapping):
        return "planner_plan_state_absent"
    return None


def prepare_tasks(rows: Sequence[Mapping[str, Any]], runtime: Any, *, seed: int) -> list[dict[str, Any]]:
    api = runtime.module
    derive = runtime.modules["paired_noise"].derive_subseed
    tasks = []
    identities = set()
    for ordinal, row in enumerate(rows):
        sample_idx = request_sample_idx(row, ordinal)
        if sample_idx in identities:
            raise ValueError(f"duplicate request sample_idx {sample_idx}")
        identities.add(sample_idx)
        noise = row.get("body_noise_seed", derive(seed, "r03_body", sample_idx))
        if isinstance(noise, bool) or not isinstance(noise, int) or not 0 <= noise < 2**63:
            raise ValueError("body_noise_seed must be a nonnegative signed-63-bit integer")
        reason = upstream_failure(row)
        task = {"ordinal": ordinal, "sample_idx": sample_idx, "source_row": dict(row),
                "attempt_id": str(row.get("attempt_id", row.get("planner_attempt_id", f"r03:{seed}:{sample_idx}"))),
                "body_noise_seed": noise, "eligible": reason is None, "reason": reason}
        if reason is not None:
            tasks.append(task)
            continue
        try:
            plan = dict(row["plan_state"])
            # These are the native rich ABI fields.  Missing science is never
            # replaced by defaults just to make a failed Planner row executable.
            required = {"N", "elements", "counts", "anion_framework", "charge_bucket",
                        "lattice_system", "spacegroup_bucket", "volume_per_atom_bin"}
            if not required.issubset(plan):
                raise ValueError(f"native rich Plan is missing {sorted(required - set(plan))}")
            prompt = api.build_body_prompt(plan).rstrip() + "\n"
            for field_name in ("body_prompt", "prompt"):
                if field_name in row and row[field_name] != prompt:
                    raise ValueError(f"original {field_name} disagrees with native rich body prompt")
            schedule = api.h1a2_safe_axis_generation_schedule(plan)
            invariant = api.require_safe_axis_schedule(schedule, num_atoms=int(plan["N"]))
            task.update(plan_state=plan, body_prompt=prompt, schedule=schedule,
                        schedule_sha256=canonical_sha256(schedule), schedule_invariant=invariant,
                        plan_state_sha256=canonical_sha256(plan),
                        body_prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest())
        except (ValueError, TypeError, KeyError) as exc:
            task.update(eligible=False, reason=f"planner_contract:{type(exc).__name__}:{exc}")
        tasks.append(task)
    return tasks


def make_batches(tasks: Sequence[Mapping[str, Any]], *, batch_size: int) -> list[list[Mapping[str, Any]]]:
    if not 1 <= batch_size <= 8:
        raise ValueError("native R03 batch size must be in 1..8")
    buckets = defaultdict(list)
    for task in tasks:
        if task["eligible"]:
            buckets[(int(task["plan_state"]["N"]), task["schedule_sha256"])].append(task)
    batches = []
    for key in sorted(buckets):
        bucket = sorted(buckets[key], key=lambda task: int(task["ordinal"]))
        batches.extend(bucket[offset:offset + batch_size] for offset in range(0, len(bucket), batch_size))
    covered = sorted(int(task["ordinal"]) for batch in batches for task in batch)
    if covered != [int(task["ordinal"]) for task in tasks if task["eligible"]]:
        raise RuntimeError("native batch partition lost or duplicated a request")
    return batches


def construct_batch(
    model: Any, tokenizer: Any, batch: Sequence[Mapping[str, Any]], runtime: Any,
    *, constraints: Any, geometry_api: Any = None,
) -> Any:
    """Call the frozen constructor, optionally adding the registered geometry hook."""
    api = runtime.module
    schedule = batch[0]["schedule"]
    if any(task["schedule"] != schedule for task in batch):
        raise ValueError("R03 batch has nonhomogeneous PlanGraph schedules")
    n = int(batch[0]["plan_state"]["N"])
    encoded = tokenizer([task["body_prompt"] for task in batch], add_special_tokens=False,
                        padding=True, return_tensors="pt")
    device = next(model.parameters()).device
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    prefill = api.merge_prefill_maps(
        api.count_prefill_for_batch(tokenizer, n, len(batch)),
        api.element_prefill_for_batch(tokenizer, [task["plan_state"] for task in batch]),
    )
    if geometry_api is not None and len(batch) != 1:
        raise ValueError("construction geometry requires singleton requests for exact failure accounting")
    bridge = (geometry_api.construction_geometry_bridge(
        runtime.modules["paired_llada"], tokenizer=tokenizer,
        generation_position_groups=schedule, native_constraints=constraints,
        enabled=True, mask_id=api.MASK_TOKEN_ID,
    ) if geometry_api is not None else nullcontext(None))
    with bridge as monitor:
        generated = api.generate_paired_exact_plan(
            model, input_ids, base_seeds=[task["body_noise_seed"] for task in batch],
            attention_mask=attention_mask, gen_length=api.exact_body_token_count(n),
            temperature=0.7, cfg_scale=0.0, remasking="low_confidence", mask_id=api.MASK_TOKEN_ID,
            allowed_token_ids_by_generation_pos=api.exact_dynamic_schema_constraints(tokenizer, n),
            prefill_token_ids_by_generation_pos=prefill, generation_position_groups=schedule,
            lightweight_decoding_constraints=constraints,
        )
        geometry_report = monitor.report() if monitor is not None else None
    suffix = generated[:, input_ids.shape[1]:]
    if suffix.shape[1] != 7 + 4 * n:
        raise RuntimeError("native R03 constructor changed its exact-length answer ABI")
    for position, values in prefill.items():
        if suffix[:, position].detach().cpu().tolist() != list(values):
            raise RuntimeError("native R03 constructor changed a prefilled count/element")
    metadata = {"prompt_token_lengths": attention_mask.sum(dim=1).cpu().tolist(),
                "prefill": {str(key): values for key, values in prefill.items()}}
    if geometry_report is not None:
        metadata["construction_geometry"] = geometry_report
    return suffix.detach().cpu(), metadata


def base_record(task: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": SCHEMA, "sample_idx": task["sample_idx"], "ordinal": task["ordinal"],
        "attempt_id": task["attempt_id"], "body_noise_seed": task["body_noise_seed"],
        "body_eligible": task["eligible"], "status": "failed", "parsed": False,
        "attempt_status": "body_pending" if task["eligible"] else "planner_failure",
        "reason": task.get("reason"), "planner_record": task["source_row"],
        "plan_state": task.get("plan_state", task["source_row"].get("plan_state")),
        "r5_plan_state": task.get("plan_state", task["source_row"].get("plan_state")),
        "generation_policy": "d2_safe_axis", "body_checkpoint_arm": "B0",
        "body_generation_complete": False, "body_plan_match": False, "body_graph_complete": False,
        "generation_position_groups": task.get("schedule"), "schedule_sha256": task.get("schedule_sha256"),
        "schedule_invariant": task.get("schedule_invariant"),
        "body_prompt": task.get("body_prompt"), "body_prompt_sha256": task.get("body_prompt_sha256"),
        "retry_used": False, "replacement_used": False, "filter_used": False, "rerank_used": False,
        "repair_used": False,
    }


def construction_failure_record(task: Mapping[str, Any], failure: Mapping[str, Any]) -> dict[str, Any]:
    """Keep one infeasible construction attempt without retry or partial repair."""
    record = base_record(task)
    record.update(
        attempt_status="construction_constraint_failure", reason="construction_geometry_no_legal_support",
        message=str(failure.get("reason", "no legal coordinate support")),
        earliest_failure_stage="body_construction_geometry",
        construction_geometry={"enabled": True, "status": "no_legal_support", "failure": dict(failure)},
        repair_skip_reason="construction_incomplete",
    )
    return record


def materialize_record(task: Mapping[str, Any], body_ids: Sequence[int], *, runtime: Any, tokenizer: Any, process_one: Any) -> tuple[dict[str, Any], dict[str, Any] | None]:
    api = runtime.module
    record = base_record(task)
    text = tokenizer.decode(list(body_ids), skip_special_tokens=False, clean_up_tokenization_spaces=False)
    record.update(text=text, raw_body_text=text, raw_body_token_ids=list(body_ids),
                  body_generation_complete=True, expected_body_token_count=7 + 4 * int(task["plan_state"]["N"]),
                  actual_body_token_count=len(body_ids), exact_length_match=len(body_ids) == 7 + 4 * int(task["plan_state"]["N"]))
    stage = "body_parse"
    try:
        arrays = api.validate_answer_matches_plan(task["plan_state"], text)
        record.update(body_plan_match=True, arrays=arrays, num_atoms=int(arrays["num_atoms"]))
        stage = "body_graph"
        graph, cif = api.graph_from_arrays(arrays, process_one)
        graph["sample_idx"] = task["sample_idx"]
        graph["r03_integrated_metadata"] = {"sample_idx": task["sample_idx"], "ordinal": task["ordinal"],
                                            "attempt_id": task["attempt_id"], "schedule_sha256": task["schedule_sha256"]}
        record.update(status="succeeded", parsed=True, attempt_status="complete", reason=None,
                      earliest_failure_stage=None, body_graph_complete=True, cif=cif)
        return record, graph
    except (ImportError, MemoryError):
        raise
    except Exception as exc:
        record.update(status="failed", parsed=False, attempt_status="body_failure",
                      reason=type(exc).__name__, message=str(exc), earliest_failure_stage=stage)
        return record, None


def summarize(records: Sequence[Mapping[str, Any]], *, elapsed: float) -> dict[str, Any]:
    requested = len(records)
    counts = {"requested_samples": requested,
              "decoded_samples": sum(bool(row.get("body_generation_complete")) for row in records),
              "parse_success": sum(bool(row.get("body_plan_match")) for row in records),
              "graph_success": sum(bool(row.get("body_graph_complete")) for row in records),
              "planner_failures": sum(not row["body_eligible"] for row in records),
              "construction_constraint_failures": sum(row.get("attempt_status") == "construction_constraint_failure" for row in records)}
    counts.update(schema=SCHEMA, time_sec=elapsed, denominator=requested,
                  pymatgen_success=counts["graph_success"], valid_array_count=counts["graph_success"],
                  parse_rate=counts["parse_success"] / max(1, requested),
                  graph_acceptance_rate=counts["graph_success"] / max(1, requested),
                  failures=dict(Counter(str(row.get("reason") or "unknown") for row in records if not row.get("body_graph_complete"))),
                  denominator_policy="all_input_requests_including_planner_failures",
                  temperature=0.7, cfg_scale=0.0, remasking="low_confidence", generation_policy="d2_safe_axis")
    return counts


def validate_b0_checkpoint(checkpoint: Path) -> dict[str, Any]:
    adapter = checkpoint / "adapter_model.safetensors"
    if adapter.stat().st_size != B0_BYTES:
        raise ValueError("original B0 adapter byte size changed")
    for name, expected in (("adapter_model.safetensors", B0_ADAPTER_SHA256),
                           ("tokenizer.json", B0_TOKENIZER_JSON_SHA256),
                           ("tokenizer_config.json", B0_TOKENIZER_CONFIG_SHA256)):
        require_file_hash(checkpoint / name, expected)
    config = json.loads((checkpoint / "adapter_config.json").read_text(encoding="utf-8"))
    saved = [str(name) for name in config.get("modules_to_save", [])]
    if (config.get("r") != 8 or config.get("lora_alpha") != 32
            or not any("wte" in name for name in saved) or not any("ff_out" in name for name in saved)):
        raise ValueError("B0 adapter configuration must restore its original trained embedding/head tables")
    return {"checkpoint": str(checkpoint.resolve()), "adapter_sha256": B0_ADAPTER_SHA256,
            "adapter_bytes": B0_BYTES, "tokenizer_json_sha256": B0_TOKENIZER_JSON_SHA256,
            "adapter_config_sha256": file_sha256(checkpoint / "adapter_config.json"),
            "trained_embeddings_and_head": "loaded_by_original_B0_PEFT_loader"}


def native_revision_slots(task: Mapping[str, Any], body_ids: Sequence[int], tokenizer: Any) -> list[int]:
    """Bind predicted species to the actual unchanged R03 element canvas."""
    source = task["source_row"]
    control = source.get("r03_control")
    if not isinstance(control, Mapping) or control.get("status") != "ok":
        raise ValueError("candidate repair requires a successful R03 controller program")
    if control.get("scope") != "post_construction_only" or control.get("sweeps") != 1:
        raise ValueError("controller attempted to change the approved revision scope")
    plan = task["plan_state"]
    order = source.get("species_program")
    elements = list(plan["elements"])
    if (not isinstance(order, list) or len(order) != len(set(order))
            or set(order) != set(elements) or len(order) != len(set(elements))):
        raise ValueError("controller program must permute the original Plan species exactly")
    expected = [str(symbol) for symbol, count in zip(elements, plan["counts"]) for _ in range(int(count))]
    if len(body_ids) != 7 + 4 * len(expected):
        raise ValueError("repair body cardinality differs from original Plan")
    vocab = tokenizer.get_vocab()
    if any(int(body_ids[7 + 4 * slot]) != int(vocab[f"<E_{symbol}>"]) for slot, symbol in enumerate(expected)):
        raise ValueError("repair body slots differ from original R03 element prefill")
    selected = list(reversed(order[:2]))
    slots = [expected.index(symbol) for symbol in selected]
    if control.get("revision_species") != selected:
        raise ValueError("exported revision species disagree with the predicted program")
    if control.get("slot_mapping") == "original_plan_element_order_counts" and control.get("revision_slots") != slots:
        raise ValueError("exported native revision slots disagree with unchanged R03 prefill")
    return slots


def _sample_supported_scalar(vector: Any, *, base_seed: int, stage: str, noise_api: Any) -> int:
    import torch

    minimum = torch.finfo(vector.dtype).min
    legal = torch.isfinite(vector) & (vector > minimum)
    if not bool(legal.any()):
        raise ValueError("cannot sample an empty supported scalar")
    uniform = noise_api.paired_uniform(
        int(base_seed), stage=stage, step=0, shape=vector.shape, device=vector.device, dtype=torch.float64,
    ).clamp(min=torch.finfo(torch.float64).tiny, max=1.0 - torch.finfo(torch.float64).eps)
    # Same categorical temperature as the offline conditional objective.
    scores = vector.double() / 0.7 - torch.log(-torch.log(uniform))
    scores = scores.masked_fill(~legal, -torch.inf)
    return int(scores.argmax())


def repair_batch(
    model: Any, tokenizer: Any, tasks: Sequence[Mapping[str, Any]], bodies: Sequence[Sequence[int]],
    *, constraints: Any, noise_api: Any, physics_api: Any = None,
) -> tuple[list[list[int]], list[dict[str, Any]]]:
    """One bounded, batched XYZ revision per selected species anchor.

    Context aliases may be canonicalized for the shared trained repair view.
    Only the active XYZ is copied back; every other original token is kept.
    A failed support check rolls back the entire active XYZ transaction.
    """
    import torch
    if physics_api is None:
        from crystal_dlm import r03_physics_transfer as physics_api
    if len(tasks) != len(bodies) or not tasks:
        raise ValueError("repair tasks/body rows must align and be nonempty")
    if len(tasks) > 8 or len({int(task["plan_state"]["N"]) for task in tasks}) != 1:
        raise ValueError("repair batches must retain homogeneous N and at most eight rows")
    current = [[int(value) for value in body] for body in bodies]
    programs: list[list[int]] = []
    reports = []
    for task, body in zip(tasks, current):
        report = {"status": "ok", "transactions": [], "sweeps": 1,
                  "support_protocol": dict(physics_api.REPAIR_SUPPORT_PROTOCOL),
                  "initial_schedule_modified": False,
                  "context_alias_policy": "100_to_0_for_model_view_only_restore_all_nonactive_output_tokens"}
        try:
            programs.append(native_revision_slots(task, body, tokenizer))
            report["geometry_before"] = physics_api.geometry_support_report(body, constraints=constraints)
        except (ValueError, TypeError, KeyError) as exc:
            programs.append([])
            report.update(status="controller_failure", reason=f"{type(exc).__name__}:{exc}")
        reports.append(report)
    device = next(model.parameters()).device
    model.eval()
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("repair inference model must be frozen")
    with torch.no_grad():
        for sweep_position in range(2):
            selected = [index for index, slots in enumerate(programs) if sweep_position < len(slots)]
            if not selected:
                continue
            previous = {index: current[index].copy() for index in selected}
            staged = {index: current[index].copy() for index in selected}
            transactions = {}
            pending = list(selected)
            for index in selected:
                slot = programs[index][sweep_position]
                canonical = physics_api.canonicalize_body_aliases(current[index], constraints=constraints)
                transactions[index] = {
                    "slot": slot, "visit_index": sweep_position, "positions": [8 + 4 * slot + axis for axis in range(3)],
                    "context_alias_positions": [p for p, (a, b) in enumerate(zip(current[index], canonical)) if a != b],
                    "original_xyz": current[index][8 + 4 * slot:11 + 4 * slot],
                    "scalar_steps": [], "committed": False, "rollback_reason": None,
                }
            for axis in range(3):
                if not pending:
                    break
                views = [physics_api.repair_scalar_example(
                    staged[index], programs[index][sweep_position], axis,
                    plan_state=tasks[index]["plan_state"], constraints=constraints,
                ) for index in pending]
                encoded = tokenizer([view["prompt"] for view in views], padding=True,
                                    add_special_tokens=False, return_tensors="pt")
                prompt_ids = encoded["input_ids"].to(device)
                attention = encoded["attention_mask"].to(device)
                suffix = torch.tensor([view["input_body"] for view in views], dtype=torch.long, device=device)
                inputs = torch.cat((prompt_ids, suffix), dim=1)
                attention = torch.cat((attention, torch.ones_like(suffix)), dim=1)
                if inputs.shape[1] > 382:
                    raise RuntimeError("repair input exceeds unchanged training context; no truncation allowed")
                raw = model(inputs, attention_mask=attention).logits
                next_pending = []
                for row_index, (index, view) in enumerate(zip(pending, views)):
                    vector, support = physics_api.supported_scalar_logits(
                        raw[row_index], view["input_body"], prompt_ids.shape[1],
                        view["num_atoms"], view["position"], constraints=constraints,
                    )
                    step = {"axis": "XYZ"[axis], "position": view["position"], **support}
                    if not support["available"]:
                        transactions[index]["rollback_reason"] = support["reason"]
                    else:
                        chosen = _sample_supported_scalar(
                            vector, base_seed=tasks[index]["body_noise_seed"],
                            stage=f"r03_repair_sweep1_slot_{programs[index][sweep_position]}_axis_{axis}",
                            noise_api=noise_api,
                        )
                        staged[index][view["position"]] = chosen
                        step["sampled_token_id"] = chosen
                        next_pending.append(index)
                    transactions[index]["scalar_steps"].append(step)
                pending = next_pending
                del raw
            completed = set(pending)
            for index in selected:
                transaction = transactions[index]
                if index in completed:
                    support = physics_api.geometry_support_report(staged[index], constraints=constraints)
                    transaction["geometry_after_staged_xyz"] = support
                    if support["supported"]:
                        allowed = set(transaction["positions"])
                        if any(old != new for position, (old, new) in enumerate(zip(previous[index], staged[index]))
                               if position not in allowed):
                            raise RuntimeError("repair transaction changed a non-active original token")
                        current[index] = staged[index]
                        transaction["committed"] = True
                    else:
                        transaction["rollback_reason"] = "final_geometry_outside_support:" + str(support["reason"])
                transaction["final_xyz"] = current[index][transaction["positions"][0]:transaction["positions"][-1] + 1]
                transaction["nonactive_original_tokens_unchanged"] = True
                reports[index]["transactions"].append(transaction)
    for index, report in enumerate(reports):
        if report["status"] == "ok":
            report["geometry_after"] = physics_api.geometry_support_report(current[index], constraints=constraints)
            report["attempted_anchors"] = len(report["transactions"])
            report["committed_anchors"] = sum(transaction["committed"] for transaction in report["transactions"])
            report["rolled_back_anchors"] = report["attempted_anchors"] - report["committed_anchors"]
    return current, reports


def validate_repair_checkpoint(checkpoint: Path) -> dict[str, Any]:
    from crystal_dlm.r03_physics_transfer import REPAIR_SUPPORT_PROTOCOL, REPAIR_VIEW_SCHEMA, TRANSFER_SCHEMA

    receipt = json.loads((checkpoint / "R03_REPAIR_TRANSFER.json").read_text(encoding="utf-8"))
    if (receipt.get("method") != TRANSFER_SCHEMA or receipt.get("eligible_policy") is not True
            or receipt.get("role") != "repair_only"
            or receipt.get("construction_checkpoint", {}).get("adapter_sha256") != B0_ADAPTER_SHA256
            or receipt.get("support_protocol") != REPAIR_SUPPORT_PROTOCOL
            or receipt.get("repair_view_schema") != REPAIR_VIEW_SCHEMA
            or receipt.get("final_tables_equal_to_B0") != {"embedding": True, "head": True}
            or receipt.get("frozen_parameter_version_changes") != []
            or receipt.get("new_sampling") is not False or receipt.get("new_physics_labels") is not False):
        raise ValueError("repair checkpoint lacks its accepted B0/physical-transfer contract")
    require_file_hash(checkpoint / "adapter_model.safetensors", receipt["adapter_sha256"])
    require_file_hash(checkpoint / "tokenizer.json", B0_TOKENIZER_JSON_SHA256)
    return receipt


def write_endpoint(
    output_dir: Path, tasks: Sequence[Mapping[str, Any]], records: Mapping[int, Mapping[str, Any]],
    graphs: Mapping[int, Any], *, runtime: Any, elapsed: float, prefix: str = "",
) -> dict[str, Any]:
    import torch

    ordered = [records[index] for index in range(len(tasks))]
    ordered_graphs = [graphs[index] for index in sorted(graphs)]
    arrays = [records[index]["arrays"] for index in sorted(graphs)]
    metrics = summarize(ordered, elapsed=elapsed)
    if ordered and ordered[0]['schema'] == EDITOR_PANEL_SCHEMA:
        metrics.update(schema=EDITOR_PANEL_SCHEMA, generation_policy='frozen_B0_D2_then_autonomous_editor',
                       editor_attempts=sum(row.get('editor_attempted') is True for row in ordered),
                       editor_changed=sum(bool(row.get('expert_trace',{}).get('changed_numeric_tokens')) for row in ordered))
    repairs = [record["repair_trace"] for record in ordered if "repair_trace" in record]
    if repairs:
        metrics["repair"] = {
            "scope": "post_construction_only", "controller_failures": sum(row["status"] != "ok" for row in repairs),
            "attempted_anchors": sum(row.get("attempted_anchors", 0) for row in repairs),
            "committed_anchors": sum(row.get("committed_anchors", 0) for row in repairs),
            "rolled_back_anchors": sum(row.get("rolled_back_anchors", 0) for row in repairs),
            "geometry_supported_after": sum(row.get("geometry_after", {}).get("supported") is True for row in repairs),
        }
    write_rows(output_dir / f"{prefix}raw_generations.jsonl", ordered)
    if not prefix:
        write_rows(output_dir / "body_attempts.jsonl", ordered)
        write_rows(output_dir / "failure_cases.jsonl", [row for row in ordered if not row["parsed"]])
    write_rows(output_dir / f"{prefix}valid_arrays.jsonl", arrays)
    torch.save(ordered_graphs, output_dir / f"{prefix}proposal_graphs.pt")
    if arrays:
        with frozen_imports(runtime):
            payload = runtime.modules["crystal_dlm.dynamic_crystal"].arrays_to_torch_payload(arrays)
        payload["sample_indices"] = [records[index]["sample_idx"] for index in sorted(graphs)]
        torch.save(payload, output_dir / f"{prefix}raw_dlm_samples.pt")
    write_json(output_dir / f"{prefix}sample_metrics.json", metrics)
    return metrics


def editor_panel_tasks(source_rows, cohort, ledger, *, seed):
    """Preserve every frozen request, including complete bodies with failed graphs."""
    tasks = []
    if len(source_rows) != len(cohort) or len(source_rows) != len(ledger):
        raise ValueError('editor source, Plan and seed ledgers differ in length')
    for index, (body, plan, noise) in enumerate(zip(source_rows, cohort, ledger)):
        if (body['sample_idx'] != index or body['ordinal'] != index
                or plan['cohort_ordinal'] != index or noise['sample_idx'] != index
                or body['attempt_id'] != plan['planner_attempt_id']
                or body['body_noise_seed'] != noise['body_noise_seed']):
            raise ValueError('editor source identity or original request order changed')
        complete = bool(plan.get('body_eligible') and body.get('body_generation_complete')
                        and body.get('body_plan_match'))
        original = body.get('raw_body_token_ids')
        if complete and (not isinstance(original,list) or len(original) != 7+4*int(plan['plan_state']['N'])):
            raise ValueError('completed frozen body lacks its exact original token sequence')
        editor_seed = int.from_bytes(hashlib.sha256(
            f'{seed}:{index}:{noise["body_noise_seed"]}'.encode()).digest()[:8], 'big')
        tasks.append({'ordinal': index, 'sample_idx': index, 'attempt_id': body['attempt_id'],
                      'body_noise_seed': body['body_noise_seed'], 'editor_seed': editor_seed,
                      'refiner_noise_seed': noise['refiner_noise_seed'], 'eligible': complete,
                      'reason': None if complete else body.get('reason','original_body_unavailable'),
                      'source_row': dict(plan, seed=plan.get('planner_sampling_seed')),
                      'plan_state': plan.get('plan_state'), 'body_prompt': plan.get('body_prompt'),
                      'body_prompt_sha256': plan.get('body_prompt_sha256'), 'original_body': original,
                      'schedule_sha256': None})
    return tasks


def skipped_editor_record(task, original):
    record = dict(original)
    record.update(schema=EDITOR_PANEL_SCHEMA,original_source_schema=original['schema'],
        purpose='evaluation',editor_attempted=False,editor_eligible=False,
        editor_skip_reason='original_body_incomplete_or_plan_mismatch',
        plan_state=task['plan_state'],body_prompt=task['body_prompt'],planner_record=task['source_row'],
        parsed=original.get('body_plan_match') is True,
        attempt_status='planner_failure' if not original.get('body_eligible') else 'body_failure',
        refiner_noise_seed=task['refiner_noise_seed'])
    return record


def registered_checkpoint_initializer(checkpoint, contract):
    if 'initialization' in contract:
        return contract['initialization'], None
    # The first deployed full-data stage predates the initializer receipt. Its
    # immutable completed stage still binds TRAIN_CONFIG and the actual argv.
    config_path = next((parent/'TRAIN_CONFIG.json' for parent in checkpoint.parents
                        if (parent/'TRAIN_CONFIG.json').is_file()), None)
    if config_path is None:
        raise ValueError('older editor checkpoint lacks its registered initializer provenance')
    config = json.loads(config_path.read_text())
    expected = file_sha256(config_path)
    matches = []
    for path in config_path.parent.parent.glob('*.stage.json'):
        stage = json.loads(path.read_text())
        if stage.get('returncode') == 0 and any(Path(pin['path']).resolve() == config_path.resolve()
                and pin.get('sha256') == expected for pin in stage.get('outputs',[])):
            matches.append((path,stage))
    if (len(matches) != 1 or config.get('train_files') != contract['train_files']
            or config.get('dev_files') != contract['dev_files']
            or config.get('b0_identity',{}).get('adapter_sha256') != B0_ADAPTER_SHA256):
        raise ValueError('older editor initializer is not bound to a completed training stage')
    path, stage = matches[0]
    parent = config['args'].get('checkpoint')
    if parent is None:
        if (config.get('initialization_kind') != 'original_B0'
                or '--checkpoint' in stage['command'] or '--resume-state' in stage['command']):
            raise ValueError('older training stage did not start from the original B0')
        initializer = {'kind':'original_B0','adapter_sha256':B0_ADAPTER_SHA256}
    else:
        if ('--checkpoint' not in stage['command']
                or stage['command'][stage['command'].index('--checkpoint')+1] != parent):
            raise ValueError('older warmstart checkpoint differs from the registered command')
        initializer = {'kind':'editor_checkpoint','path':parent,
                       'receipt_sha256':file_sha256(Path(parent)/'CHECKPOINT_FINAL.json')}
    return initializer, {'training_config_sha256':expected,'stage_receipt':str(path),'stage_sha256':file_sha256(path)}


def validate_editor_panel_holdout(checkpoint, cohort):
    from crystal_dlm.expert_edit_data import composition_key
    forbidden = {composition_key(row['plan_state']) for row in cohort if row.get('body_eligible')}
    examined, lineage, seen_checkpoints, seen_files = [], [], set(), set()
    while checkpoint is not None:
        checkpoint = checkpoint.resolve()
        if checkpoint in seen_checkpoints:
            raise ValueError('editor initializer provenance contains a cycle')
        seen_checkpoints.add(checkpoint)
        receipt = checkpoint/'CHECKPOINT_FINAL.json'
        if not (checkpoint/'_CHECKPOINT_SUCCESS').is_file() or not receipt.is_file():
            raise ValueError('formal editor requires complete ancestor training checkpoints')
        checkpoint_receipt = json.loads(receipt.read_text())
        model_files = checkpoint_receipt.get('model_files_sha256',{})
        required = {'adapter_model.safetensors','adapter_config.json','expert_edit_modules.pt',
                    'expert_edit_config.json','periodic_state.pt','periodic_state_config.json',
                    'EXPERT_EDITOR.json','roundtrip_probe.pt'}
        if not required.issubset(model_files):
            raise ValueError('editor receipt does not bind all actual model and probe files')
        for name,digest in model_files.items():
            if Path(name).name != name:
                raise ValueError('editor model file identity escaped its checkpoint')
            require_file_hash(checkpoint/name,digest)
        contract = checkpoint_receipt['contract']
        for split in ('train','dev'):
            for pin in contract[split+'_files']:
                key = (split,pin['path'],pin['sha256'])
                if key in seen_files:
                    continue
                seen_files.add(key)
                path = Path(pin['path'])
                require_file_hash(path,pin['sha256'])
                require_file_hash(path.parent/'DATA_FINAL.json',pin['report_sha256'])
                rows = read_rows(path) if path.stat().st_size else []
                if any(row['source_split'] != split or row['composition_key'] in forbidden for row in rows):
                    raise ValueError('formal cohort composition was used for editor training or model selection')
                examined.append({'split':split,'rows':len(rows),**pin})
        initializer, legacy_binding = registered_checkpoint_initializer(checkpoint,contract)
        lineage.append({'checkpoint':str(checkpoint),'receipt_sha256':file_sha256(receipt),
                        'initialization':initializer,'older_stage_binding':legacy_binding})
        if initializer.get('kind') == 'original_B0' and initializer.get('adapter_sha256') == B0_ADAPTER_SHA256:
            checkpoint = None
        elif initializer.get('kind') == 'editor_checkpoint':
            checkpoint = Path(initializer['path'])
            require_file_hash(checkpoint/'CHECKPOINT_FINAL.json',initializer['receipt_sha256'])
        else:
            raise ValueError('editor initializer provenance does not terminate at the original B0')
    return {'lineage':lineage,'files':examined,'evaluation_compositions_absent_from_train_and_dev':True,
            'scope':'all_editing_stages_and_their_development_sets; original_B0_pretraining_is_preexisting'}


def run_editor_panel(argv):
    """Apply the trained autonomous editor to an explicitly pinned legacy panel."""
    import os
    import torch
    import torch.distributed as dist
    from crystal_dlm.expert_edit import load_editor_model, edit_structures
    from scripts.export_r03_evaluation_inputs import export_legacy_panel
    parser = argparse.ArgumentParser(description=run_editor_panel.__doc__)
    parser.add_argument('--editor-source-manifest', type=Path, required=True)
    parser.add_argument('--editor-source-sha256', required=True)
    parser.add_argument('--editor-checkpoint', type=Path, required=True)
    parser.add_argument('--frozen-runtime-root', type=Path, required=True)
    parser.add_argument('--base-model', type=Path, required=True)
    parser.add_argument('--b0-checkpoint', type=Path, required=True)
    parser.add_argument('--crysllmgen-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--expected-requests', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--max-calls', type=int, default=160)
    parser.add_argument('--block-size', type=int, choices=(1,4,8), default=1)
    parser.add_argument('--accept-threshold', type=float, default=.5)
    args = parser.parse_args(argv)
    rank, world, local_rank = (int(os.environ.get(name, default)) for name, default in
                              (('RANK','0'),('WORLD_SIZE','1'),('LOCAL_RANK','0')))
    if not torch.cuda.is_available() or not 1 <= world <= 6 or not 1 <= args.batch_size <= 8:
        raise ValueError('editor panel requires 1..6 GPUs and batches in 1..8')
    require_file_hash(args.editor_source_manifest, args.editor_source_sha256)
    manifest = json.loads(args.editor_source_manifest.read_text())
    if manifest.get('arm') != 'candidate' or manifest.get('endpoint') != 'native':
        raise ValueError('editor input must be the frozen original B0 D2 native panel')
    # The explicit legacy adapter checks all original schemas, rich prompts,
    # complete input hashes and ordinals. Its geometry results never gate editing.
    _, source_validation = export_legacy_panel(manifest, endpoint='native',
        expected_requests=args.expected_requests, method_id=manifest['method_id'])
    source_rows = read_rows(Path(manifest['files']['body']['path']))
    cohort = read_rows(Path(manifest['files']['cohort']['path']))
    ledger = read_rows(Path(manifest['files']['seed_ledger']['path']))
    tasks = editor_panel_tasks(source_rows, cohort, ledger, seed=args.seed)
    torch.cuda.set_device(local_rank)
    if world > 1:
        dist.init_process_group('nccl')
    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        holdout = validate_editor_panel_holdout(args.editor_checkpoint, cohort)
        write_json(args.output_dir/'run_config.json', {'schema':EDITOR_PANEL_SCHEMA,
            'expected_requests':args.expected_requests, 'purpose':'evaluation',
            'source_manifest':str(args.editor_source_manifest), 'source_manifest_sha256':args.editor_source_sha256,
            'source_validation':source_validation, 'b0':validate_b0_checkpoint(args.b0_checkpoint),
            'editor_checkpoint':str(args.editor_checkpoint), 'holdout_validation':holdout,
            'seed':args.seed, 'max_calls':args.max_calls, 'block_size':args.block_size,
            'sampling_batch_size':args.batch_size, 'accept_threshold':args.accept_threshold,
            'tasks':['G','S'], 'scope_policy':'learned', 'accept_all':False,
            'online_physics_calls':0, 'external_geometry_gate':False, 'world_size':world})
        write_rows(args.output_dir/'attempt_ledger.jsonl', ledger)
    if world > 1:
        dist.barrier()
    device = torch.device('cuda',local_rank)
    model, tokenizer = load_editor_model(args.base_model,args.editor_checkpoint,device,trainable=False)
    model.eval()
    runtime = load_frozen_runtime(args.frozen_runtime_root)
    with frozen_imports(runtime):
        process_one = runtime.module.import_process_one(args.crysllmgen_dir)
        runtime.module.assert_body_tokenizer_identity(tokenizer,expected_vocab_sha256=B0_VOCAB_SHA256)
    local = [task for task in tasks if task['ordinal'] % world == rank]
    eligible = [task for task in local if task['eligible']]
    requests = [{'prompt':task['body_prompt'], 'body':task['original_body'],
                 'num_sites':task['plan_state']['N'], 'tasks':('G','S'), 'seed':task['editor_seed']}
                for task in eligible]
    def progress(completed,total,batches):
        if completed % 8 == 0 or completed == total:
            print(json.dumps({'rank':rank,'completed':completed,'requested':total,'forward_batches':batches}),flush=True)
    started = time.monotonic()
    sampled = edit_structures(model,tokenizer,requests,allowed_modes=model.training_modes,
        max_calls=args.max_calls,block_size=args.block_size,batch_size=args.batch_size,
        accept_threshold=args.accept_threshold,progress=progress)
    records, graphs = {}, {}
    for task in local:
        if not task['eligible']:
            record = skipped_editor_record(task,source_rows[task['ordinal']])
            records[task['ordinal']] = record
    for task, output in zip(eligible,sampled['results']):
        with frozen_imports(runtime):
            record, graph = materialize_record(task,output['body'],runtime=runtime,tokenizer=tokenizer,process_one=process_one)
        record.update(schema=EDITOR_PANEL_SCHEMA,purpose='evaluation',editor_attempted=True,
            generation_policy='frozen_B0_D2_then_autonomous_editor',editor_seed=task['editor_seed'],
            original_body_token_ids=task['original_body'],expert_trace=output,repair_used=True,
            refiner_noise_seed=task['refiner_noise_seed'])
        records[task['ordinal']] = record
        if graph is not None:
            graph['refiner_noise_seed'] = task['refiner_noise_seed']
            graphs[task['ordinal']] = graph
    write_rows(args.output_dir/f'editor_records.rank{rank}.jsonl',[records[i] for i in sorted(records)])
    torch.save(graphs,args.output_dir/f'editor_graphs.rank{rank}.pt')
    write_json(args.output_dir/f'editor_metrics.rank{rank}.json',{'forward_batches':sampled['forward_batches'],
               'forward_rows':sampled['forward_rows'],'seconds':time.monotonic()-started,'requests':len(local)})
    if world > 1:
        dist.barrier()
    if rank == 0:
        all_records, all_graphs = {}, {}
        for worker in range(world):
            for row in read_rows(args.output_dir/f'editor_records.rank{worker}.jsonl'):
                if row['ordinal'] in all_records:
                    raise ValueError('editor worker produced a duplicate global request')
                all_records[row['ordinal']] = row
            worker_graphs = torch.load(args.output_dir/f'editor_graphs.rank{worker}.pt',map_location='cpu',weights_only=False)
            if set(all_graphs) & set(worker_graphs):
                raise ValueError('editor worker duplicated a proposal graph')
            all_graphs.update(worker_graphs)
        if set(all_records) != set(range(args.expected_requests)):
            raise ValueError('editor failed to preserve the entire original request ledger')
        metrics = write_endpoint(args.output_dir,tasks,all_records,all_graphs,runtime=runtime,elapsed=time.monotonic()-started)
        write_json(args.output_dir/'EDITOR_FINAL.json',metrics)
        (args.output_dir/'_SUCCESS').touch()
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


def main() -> None:
    if '--editor-source-manifest' in sys.argv[1:]:
        return run_editor_panel(sys.argv[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-runtime-root", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--b0-checkpoint", type=Path, required=True)
    parser.add_argument("--plans-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--crysllmgen-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--expected-requests", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--repair", action="store_true", help="Execute the exported Llama anchor program after R03 construction")
    parser.add_argument("--geometry-support", action="store_true", help="Use the shared fixed support in post-construction XYZ repairs only")
    parser.add_argument("--construction-geometry", action="store_true", help="Apply periodic alias aggregation and 0.5 A PBC support during the frozen constructor")
    parser.add_argument("--repair-checkpoint", type=Path, help="Accepted repair-only P adapter; omission uses unadapted original B0 for G")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if not 1 <= args.batch_size <= 8 or args.expected_requests < 1:
        raise ValueError("batch size must be in 1..8 and request denominator must be positive")
    if bool(args.repair) != bool(args.geometry_support) or (args.repair_checkpoint and not args.repair):
        raise ValueError("candidate requires both --repair and --geometry-support; a repair checkpoint cannot alter construction")
    if args.construction_geometry and args.batch_size != 1:
        raise ValueError("--construction-geometry requires --batch-size 1; no-support attempts are retained individually")
    rows = read_rows(args.plans_jsonl)
    if len(rows) != args.expected_requests:
        raise ValueError("input request ledger does not match the frozen denominator")
    runtime = load_frozen_runtime(args.frozen_runtime_root)
    with frozen_imports(runtime):
        tasks = prepare_tasks(rows, runtime, seed=args.seed)
    batches = make_batches(tasks, batch_size=args.batch_size)
    checkpoint_identity = validate_b0_checkpoint(args.b0_checkpoint)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.output_dir / "run_config.json", {
        "schema": SCHEMA, "frozen_runtime": runtime.provenance, "b0": checkpoint_identity,
        "base_model": str(args.base_model.resolve()), "plans_jsonl": str(args.plans_jsonl.resolve()),
        "plans_sha256": file_sha256(args.plans_jsonl), "expected_requests": args.expected_requests,
        "seed": args.seed, "max_batch_size": args.batch_size,
        "temperature": 0.7, "cfg_scale": 0.0, "remasking": "low_confidence",
        "geometry_support_scope": ("construction_and_repair" if args.repair else "construction") if args.construction_geometry
                                  else ("post_construction_repair_only" if args.repair else "original_constructor_masks_only"),
        "construction_geometry_enabled": bool(args.construction_geometry),
        "post_construction_repair": bool(args.repair),
        "repair_checkpoint": str(args.repair_checkpoint.resolve()) if args.repair_checkpoint else None,
    })
    write_rows(args.output_dir / "attempt_ledger.jsonl", [{key: value for key, value in task.items() if key != "source_row"} for task in tasks])
    write_json(args.output_dir / "batch_partition.json", [[task["sample_idx"] for task in batch] for batch in batches])

    import torch
    geometry_api = None
    if args.construction_geometry:
        # Bind current support helpers before historical package names are installed.
        from crystal_dlm import r03_geometry_bridge as geometry_api
    no_support_error = geometry_api.GeometryNoLegalSupport if geometry_api is not None else ()

    with frozen_imports(runtime):
        api = runtime.module
        model, tokenizer = api.load_model_and_tokenizer(str(args.base_model), str(args.b0_checkpoint), torch.device(args.device))
        if tokenizer.pad_token_id == api.MASK_TOKEN_ID:
            raise RuntimeError("B0 padding token collides with mask token")
        identity = api.assert_body_tokenizer_identity(tokenizer, expected_vocab_sha256=B0_VOCAB_SHA256)
        if identity["vocab_size"] != 128830:
            raise ValueError("B0 vocabulary size changed")
        write_json(args.output_dir / "body_tokenizer_identity.json", identity)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        process_one = api.import_process_one(args.crysllmgen_dir)
        constraints = api.build_dynamic_lightweight_constraints(
            tokenizer, duplicate_coordinate_mask=True, lattice_volume_mask=True, min_lattice_rad=1e-4,
        )
    started = time.monotonic()
    records = {task["ordinal"]: base_record(task) for task in tasks if not task["eligible"]}
    graphs = {}
    progress_path = args.output_dir / "body_progress.jsonl"
    for batch_index, batch in enumerate(batches):
        try:
            with frozen_imports(runtime):
                suffix, batch_meta = construct_batch(model, tokenizer, batch, runtime,
                                                     constraints=constraints, geometry_api=geometry_api)
                for row_index, (task, body_ids) in enumerate(zip(batch, suffix.tolist())):
                    record, graph = materialize_record(task, body_ids, runtime=runtime, tokenizer=tokenizer, process_one=process_one)
                    record["prefill_token_ids"] = {position: values[row_index] for position, values in batch_meta["prefill"].items()}
                    record["body_prompt_token_count"] = batch_meta["prompt_token_lengths"][row_index]
                    if "construction_geometry" in batch_meta:
                        record["construction_geometry"] = batch_meta["construction_geometry"]
                    records[task["ordinal"]] = record
                    if graph is not None:
                        graphs[task["ordinal"]] = graph
                    with progress_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        except no_support_error as error:
            if len(batch) != 1:
                raise RuntimeError("construction no-support failure is not a singleton") from error
            task = batch[0]
            record = construction_failure_record(task, error.to_dict())
            records[task["ordinal"]] = record
            with progress_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        progress = {"event": "r03_body_progress", "batches_completed": batch_index + 1,
                    "total_batches": len(batches), "requests_completed": len(records),
                    "expected_requests": len(tasks), "elapsed_seconds": round(time.monotonic() - started, 2)}
        write_json(args.output_dir / "progress.json", progress)
        print(json.dumps(progress), flush=True)
    if args.repair:
        write_endpoint(args.output_dir, tasks, records, graphs, runtime=runtime,
                       elapsed=time.monotonic() - started, prefix="construction_")
        from crystal_dlm import r03_physics_transfer as physics_api
        repair_constraints = physics_api.build_repair_constraints(tokenizer)
        if args.repair_checkpoint:
            receipt = validate_repair_checkpoint(args.repair_checkpoint)
            with frozen_imports(runtime):
                repair_model, repair_tokenizer = api.load_model_and_tokenizer(
                    str(args.base_model), str(args.repair_checkpoint), torch.device(args.device),
                )
                api.assert_body_tokenizer_identity(repair_tokenizer, expected_vocab_sha256=B0_VOCAB_SHA256)
            if (tokenizer.get_vocab() != repair_tokenizer.get_vocab()
                    or not torch.equal(model.get_input_embeddings().weight, repair_model.get_input_embeddings().weight)
                    or not torch.equal(model.get_output_embeddings().weight, repair_model.get_output_embeddings().weight)):
                raise RuntimeError("loaded repair checkpoint changed the original trained B0 vocabulary/embedding/head")
            del model
            model = repair_model
            del repair_model
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            model.eval()
            if str(args.device).startswith("cuda"):
                torch.cuda.empty_cache()
            write_json(args.output_dir / "repair_checkpoint_identity.json", receipt)
        for batch_index, batch in enumerate(batches):
            batch = [task for task in batch if records[task["ordinal"]].get("body_generation_complete")]
            if not batch:
                continue
            bodies = [records[task["ordinal"]]["raw_body_token_ids"] for task in batch]
            revised, traces = repair_batch(
                model, tokenizer, batch, bodies, constraints=repair_constraints,
                noise_api=runtime.modules["paired_noise"], physics_api=physics_api,
            )
            for task, native_ids, final_ids, trace in zip(batch, bodies, revised, traces):
                ordinal = task["ordinal"]
                original = records[ordinal]
                if trace["status"] != "ok":
                    record = {**original, "parsed": False, "status": "failed", "attempt_status": "controller_failure",
                              "body_graph_complete": False, "reason": "controller_failure", "message": trace["reason"]}
                    graphs.pop(ordinal, None)
                elif final_ids == native_ids:
                    record = dict(original)
                else:
                    with frozen_imports(runtime):
                        record, graph = materialize_record(task, final_ids, runtime=runtime, tokenizer=tokenizer, process_one=process_one)
                    if graph is None:
                        graphs.pop(ordinal, None)
                    else:
                        graphs[ordinal] = graph
                record.update(repair_used=trace["status"] == "ok", repair_trace=trace,
                              construction_raw_body_token_ids=native_ids,
                              construction_status=original["status"],
                              prefill_token_ids=original["prefill_token_ids"],
                              body_prompt_token_count=original["body_prompt_token_count"],
                              repair_checkpoint=str(args.repair_checkpoint or args.b0_checkpoint))
                if "construction_geometry" in original:
                    record["construction_geometry"] = original["construction_geometry"]
                records[ordinal] = record
                with (args.output_dir / "repair_progress.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            progress = {"event": "r03_repair_progress", "batches_completed": batch_index + 1,
                        "total_batches": len(batches), "elapsed_seconds": round(time.monotonic() - started, 2)}
            write_json(args.output_dir / "progress.json", progress)
            print(json.dumps(progress), flush=True)
    metrics = write_endpoint(args.output_dir, tasks, records, graphs, runtime=runtime, elapsed=time.monotonic() - started)
    print(json.dumps(metrics, sort_keys=True), flush=True)
    if metrics.get("repair", {}).get("controller_failures"):
        (args.output_dir / "_FAILED").touch()
        raise SystemExit(2)
    (args.output_dir / "_SUCCESS").touch()


if __name__ == "__main__":
    main()
