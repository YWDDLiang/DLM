#!/usr/bin/env python3
"""Independent native paths and fresh-process replay through the same runtime."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crystal_dlm.programmed_path_data import (
    compile_condition, load_path_model, path_seed, read_jsonl,
    trace_summary, trace_terminal_body, validate_completed_body,
)
from crystal_dlm.sampling_layout import sampling_batches


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--checkpoint-path", required=True)
    p.add_argument("--conditions", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--candidates", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--condition-start", type=int, default=0)
    p.add_argument("--condition-stop", type=int, default=1024)
    p.add_argument("--collection-round", type=int, default=0)
    p.add_argument("--seed", type=int, default=20260905)
    p.add_argument("--purpose", choices=("train", "evaluation"), default="train")
    p.add_argument("--reference-closure", action="store_true")
    p.add_argument("--temperature", type=float, default=.7)
    p.add_argument("--world-size", type=int, default=int(os.environ.get("WORLD_SIZE", 1)))
    p.add_argument("--sampling-layout-world-size", type=int,
                   help="Freeze reference batch membership before redistributing whole batches to workers")
    p.add_argument("--merge-only", action="store_true")
    p.add_argument("--replay-jsonl", type=Path)
    p.add_argument("--replay-tolerance", type=float, default=1e-6)
    p.add_argument("--repair-roots-jsonl", type=Path, nargs="+")
    p.add_argument("--full-cell-repair", action="store_true")
    p.add_argument("--short-contact-spec", type=Path)
    p.add_argument("--legacy-layout-world-size", type=int)
    p.add_argument("--mixture-peer-checkpoint", type=Path)
    p.add_argument("--construction-only", action="store_true")
    return p.parse_args()


def conditions_for_run(args):
    rows = read_jsonl(args.conditions)
    if args.purpose == "evaluation":
        if any(row.get("source_split") == "train" for row in rows):
            raise ValueError("train path conditions cannot be renamed as evaluation")
        rows = [dict(row, group_id=row.get("group_id", f"eval:{row['sample_idx']}"),
                     source_split="evaluation", source_row_idx=int(row["sample_idx"])) for row in rows]
    if not 0 <= args.condition_start < args.condition_stop <= len(rows):
        raise ValueError("condition range is outside the frozen source pool")
    selected = list(enumerate(rows))[args.condition_start:args.condition_stop]
    if len({str(row["group_id"]) for _, row in selected}) != len(selected):
        raise ValueError("duplicate condition identity")
    if args.purpose == "train" and any(row.get("source_split") != "train" for _, row in selected):
        raise ValueError("training collection cannot use heldout conditions")
    if getattr(args, "repair_roots_jsonl", None):
        from crystal_dlm.self_repair_data import attach_fixed_repair_roots
        parents = []
        for path in args.repair_roots_jsonl:
            if not (path.parent / "_SUCCESS").is_file():
                raise ValueError("repair root collection is incomplete")
            parents.extend(read_jsonl(path))
        selected = attach_fixed_repair_roots(selected, parents)
    return selected


def merge(args):
    expected = {(str(row["group_id"]), j) for _, row in conditions_for_run(args) for j in range(args.candidates)}
    records = [row for rank in range(args.world_size)
               for row in read_jsonl(args.output_dir / f"records.rank{rank}.jsonl")]
    keys = [(str(row["group_id"]), row["candidate_index"]) for row in records]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("missing or duplicate requested path occurrences")
    if len({row["trajectory_id"] for row in records}) != len(records):
        raise ValueError("duplicate trajectory identity")
    records.sort(key=lambda row: (row["condition_ordinal"], row["candidate_index"]))
    with (args.output_dir / "paths.jsonl").open("x", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {"requested": len(records), "successful": sum(row["success"] for row in records),
              "conditions": len(expected) // args.candidates, "candidates": args.candidates,
              "condition_start": args.condition_start, "condition_stop": args.condition_stop,
              "collection_round": args.collection_round, "checkpoint": args.checkpoint_path,
              "seed": args.seed, "purpose": args.purpose, "inference_mlip": False,
              "execution_world_size": args.world_size,
              "sampling_layout_world_size": args.sampling_layout_world_size,
              "reference_closure": args.reference_closure,
              "full_cell_repair": args.full_cell_repair,
              "outcome_selection": False, "failures_retained": True}
    if getattr(args, "short_contact_spec", None):
        from crystal_dlm.short_contact_sampling import ContactSpec
        report.update(short_contact_policy=ContactSpec.load(args.short_contact_spec).metadata(),
                      legacy_layout_world_size=args.legacy_layout_world_size,
                      effective_sampling_layout_world_size=args.sampling_layout_world_size or args.legacy_layout_world_size)
    if getattr(args,"mixture_peer_checkpoint",None):
        report.update(probability_mixture={"schema":"k4k8_equal_same_state_probability_mixture_v1",
                      "checkpoints":[str(args.checkpoint_path),str(args.mixture_peer_checkpoint)],"weights":[.5,.5],
                      "temperature":.7,"same_state":True,"contact":False},
                      effective_sampling_layout_world_size=args.sampling_layout_world_size or args.legacy_layout_world_size,
                      component_model_forward_calls=[sum(row.get("mixture_counted_forward_calls",[0,0])[i] for row in records) for i in range(2)])
    if getattr(args,"construction_only",False):
        report.update(method=args._construction_method,path_mode="construction_only",construction_only=True)
    if getattr(args, "repair_roots_jsonl", None):
        report.update(path_mode="self_repair_support_check", diagnostic_only=True,
                      repair_roots_jsonl=[str(p) for p in args.repair_roots_jsonl],
                      root_selection="fixed_candidate_index_zero_no_outcomes",
                      seed_namespace="self-repair-v1:parent_trajectory_id")
    (args.output_dir / "SAMPLE_FINAL.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "_SUCCESS").touch()
    print(json.dumps(report), flush=True)


def make_sampler(model, tokenizer, constraints, compiled, seeds, temperature, contact_spec=None,
                 *, probe_only=False, collect_vectors=False, mixture_peer_model=None):
    from crystal_dlm.fixed_slot import MASK_TOKEN_ID
    from crystal_dlm.programmed_path_runtime import ProgrammedPathSampler
    from crystal_dlm.r5_dynamic_length import exact_dynamic_schema_constraints
    first = compiled[0]
    sampler_type, extra = ProgrammedPathSampler, {}
    if contact_spec is not None:
        from crystal_dlm.short_contact_sampling import ShortContactProgrammedPathSampler
        sampler_type = ShortContactProgrammedPathSampler
        extra = dict(contact_spec=contact_spec, probe_only=probe_only, collect_vectors=collect_vectors)
    if mixture_peer_model is not None:
        if contact_spec is not None:
            raise ValueError("probability mixture cannot be combined with contact tilt")
        from crystal_dlm.probability_mixture_sampling import ProbabilityMixtureProgrammedPathSampler
        sampler_type=ProbabilityMixtureProgrammedPathSampler
        extra=dict(peer_model=mixture_peer_model)
    return sampler_type(
        model, prompt_length=len(first["prompt_token_ids"]), gen_length=len(first["initial_body"]),
        mask_id=MASK_TOKEN_ID, programs=[c["program"] for c in compiled],
        allowed_token_ids=exact_dynamic_schema_constraints(tokenizer, first["program"].num_atoms),
        atom_count_grammar=None, constraints=constraints, temperature=temperature, sampling_seeds=seeds, **extra,
    )


def replay(args, model, tokenizer, constraints, device):
    import torch
    from crystal_dlm.fixed_slot import MASK_TOKEN_ID
    from crystal_dlm.programmed_path_runtime import replay_scalar_states
    maximum, total, state_effect = 0., 0, 0.
    mixture_calls=[0,0]
    rows = read_jsonl(args.replay_jsonl)
    for record in rows:
        if getattr(args,"construction_only",False):
            if record.get("method")!=args._construction_method or any(e.get("phase")!="construct" for e in record["trace"]["events"]):
                raise ValueError("construction-only replay received another method or phase")
        compiled = compile_condition(record, tokenizer, mask_id=MASK_TOKEN_ID, purpose=args.purpose)
        prefix = compiled["prompt_token_ids"]
        if prefix != record["prompt_token_ids"]:
            raise ValueError("fresh tokenizer changed the native prompt")
        contact_spec = getattr(args, "_contact_settings", None)
        if contact_spec is not None and record["trace"].get("short_contact_policy", {}).get("spec_sha256") != contact_spec.spec_sha256:
            raise ValueError("replay record differs from the frozen short-contact specification")
        if getattr(args,"mixture_peer_checkpoint",None):
            recorded=record.get("probability_mixture",{})
            if (recorded.get("schema")!="k4k8_equal_same_state_probability_mixture_v1" or recorded.get("weights")!=[.5,.5]
                    or recorded.get("temperature")!=.7 or recorded.get("contact") is not False
                    or [str(Path(p).resolve()) for p in recorded.get("checkpoints",[])] !=
                       [str(Path(args.checkpoint_path).resolve()),str(Path(args.mixture_peer_checkpoint).resolve())]):
                raise ValueError("replay record differs from the fixed K4/K8 mixture identity")
        sampler = make_sampler(model, tokenizer, constraints, [compiled], [record["sampling_seed"]],
                               record["trace"]["temperature"], contact_spec,
                               mixture_peer_model=getattr(args,"_mixture_model",None))
        for state in replay_scalar_states(record["trace"]):
            x = torch.tensor([prefix + state["input_body"]], device=device)
            old = torch.tensor([prefix + state["old_body"]], device=device)
            sampler._prepare(x)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                logits, bad = sampler.processed_logits(x, old, {0: state["position"]},
                                                      {0: state["transaction_positions"]}, torch.ones_like(x),
                                                      phase=state["phase"])
                actual = float(torch.log_softmax(logits[0, len(prefix) + state["position"]].double() / state["temperature"], -1)[state["target_token"]])
            difference = abs(actual - state["recorded_log_probability"])
            if bad or not torch.isfinite(torch.tensor(actual)) or difference > args.replay_tolerance:
                raise RuntimeError(f"fresh-process replay differs at {record['trajectory_id']}:{state['decision_index']}: {difference}, bad={bad}")
            if state["phase"] == "cooperative" and state_effect == 0:
                from crystal_dlm.state_conditioned_model import context_from_programs
                context = context_from_programs(old, prompt_length=len(prefix), num_sites=compiled["program"].num_atoms,
                    programs=[compiled["program"]], active_positions={0: state["transaction_positions"]})
                with torch.no_grad():
                    encoded = model.state_conditioner(**model.geometry_inputs(context))
                    state_effect = float(encoded["cell_embedding"].abs().sum() + encoded["site_embeddings"].abs().sum())
            maximum, total = max(maximum, difference), total + 1
        if trace_terminal_body(record["trace"]) != record["final_body_token_ids"]:
            raise RuntimeError("trace replay does not reconstruct the deployed endpoint")
        if getattr(args,"mixture_peer_checkpoint",None):
            mixture_calls=[a+b for a,b in zip(mixture_calls,sampler.component_forward_calls)]
    if total == 0 or (state_effect == 0 and not getattr(args,"construction_only",False)):
        raise RuntimeError("probe did not exercise the trained periodic state input")
    report = {"kind": "fresh_process_base_lora_conditioner_replay", "paths": len(rows),
              "all_recorded_decisions_checked": total, "maximum_logp_error": maximum,
              "trained_state_residual_l1": state_effect, "checkpoint": args.checkpoint_path,
              "scope": "all decisions of the recorded engineering probe paths"}
    if getattr(args,"construction_only",False):
        report.update(method=args._construction_method,path_mode="construction_only",
                      state_residual_check="cooperative residual is not required for construction-only replay")
    if getattr(args,"mixture_peer_checkpoint",None):
        report.update(component_model_forward_calls=mixture_calls,
                      probability_mixture={"schema":"k4k8_equal_same_state_probability_mixture_v1","weights":[.5,.5],
                        "checkpoints":[str(args.checkpoint_path),str(args.mixture_peer_checkpoint)],"temperature":.7,"contact":False})
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "REPLAY.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "_SUCCESS").touch()
    print(json.dumps(report), flush=True)


def main():
    args = parse_args()
    if args.construction_only:
        if args.purpose!="evaluation" or args.reference_closure or args.repair_roots_jsonl or args.full_cell_repair or args.short_contact_spec or args.mixture_peer_checkpoint or args.temperature!=.7:
            raise ValueError("construction-only stage ablation uses an original legacy K4/K8 evaluation policy")
        from crystal_dlm.short_contact_sampling import validate_legacy_contact_policy
        source=validate_legacy_contact_policy(args.checkpoint_path)
        if source.get("collection_round") not in (0,1):raise ValueError("unknown original legacy policy round")
        args._construction_method={0:"k4_construction_only_v1",1:"k8_construction_only_v1"}[source["collection_round"]]
    if args.legacy_layout_world_size is not None and (args.legacy_layout_world_size < 1 or args.sampling_layout_world_size is not None):
        raise ValueError("choose one positive explicit or legacy logical layout")
    if args.mixture_peer_checkpoint:
        if args.short_contact_spec or args.purpose!="evaluation" or args.reference_closure or args.repair_roots_jsonl or args.full_cell_repair or args.temperature!=.7:
            raise ValueError("equal K4/K8 mixture retains original evaluation phases and has no contact tilt")
        from crystal_dlm.short_contact_sampling import validate_legacy_contact_policy
        left=validate_legacy_contact_policy(args.checkpoint_path)
        right=validate_legacy_contact_policy(args.mixture_peer_checkpoint)
        if left.get("collection_round")!=0 or right.get("collection_round")!=1:
            raise ValueError("mixture components must be ordered completed K4 then K8")
    if args.short_contact_spec:
        if args.purpose != "evaluation" or args.reference_closure or args.repair_roots_jsonl:
            raise ValueError("short-contact deployment is an evaluation-only legacy K4/K8 candidate")
        from crystal_dlm.short_contact_sampling import ContactSpec, validate_legacy_contact_policy
        args._contact_settings = ContactSpec.load(args.short_contact_spec)
        validate_legacy_contact_policy(args.checkpoint_path)
    if args.full_cell_repair and args.reference_closure:
        raise ValueError("full-cell repair is not part of the historical reference policy")
    if (Path(args.checkpoint_path) / "periodic_repair_config.json").is_file() and not args.full_cell_repair:
        raise ValueError("repair checkpoints require their registered full-cell deployment")
    if args.candidates < 1 or args.batch_size < 1 or args.temperature <= 0:
        raise ValueError("positive occurrence count, batch size and likelihood temperature required")
    if args.reference_closure and (args.purpose != "evaluation" or args.candidates != 1 or args.replay_jsonl):
        raise ValueError("the old closure is an evaluation-only reference, not a path teacher")
    repair_only = bool(args.repair_roots_jsonl)
    if repair_only and (args.purpose != "train" or args.reference_closure or args.collection_round != 2
                        or args.candidates != 4 or args.batch_size != 4):
        raise ValueError("self-repair support check requires train-only round2 K4/batch4, without legacy closure")
    if args.merge_only:
        merge(args)
        return
    if "SLURM_JOB_ID" not in os.environ or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("model sampling/replay requires its declared GPU allocation")
    import torch
    from crystal_dlm.fixed_slot import MASK_TOKEN_ID
    from scripts.sample_llada_dynamic_crystals import build_dynamic_lightweight_constraints
    rank, local = int(os.environ.get("RANK", 0)), int(os.environ.get("LOCAL_RANK", 0))
    torch.set_num_threads(2)
    torch.cuda.set_device(local)
    torch.manual_seed(args.seed)
    device = torch.device("cuda", local)
    if args.reference_closure:
        from scripts.sample_llada_dynamic_crystals import load_model_and_tokenizer
        from crystal_dlm.programmed_path_reference import UnconditionedReferenceModel
        base, tokenizer = load_model_and_tokenizer(args.model_path, args.checkpoint_path, device)
        model = UnconditionedReferenceModel(base).requires_grad_(False).eval()
    else:
        model, tokenizer = load_path_model(args.model_path, args.checkpoint_path, device)
    if args.mixture_peer_checkpoint:
        args._mixture_model, peer_tokenizer=load_path_model(args.model_path,args.mixture_peer_checkpoint,device)
        if tokenizer.get_vocab()!=peer_tokenizer.get_vocab():
            raise ValueError("mixture tokenizer vocabularies differ")
    constraints = build_dynamic_lightweight_constraints(
        tokenizer, duplicate_coordinate_mask=True, lattice_volume_mask=True, min_lattice_rad=1e-4,
        canonicalize_periodic_alias=True, pbc_min_distance_mask=True, pbc_min_distance_A=.5, pbc_image_radius=2,
    )
    if args.replay_jsonl:
        replay(args, model, tokenizer, constraints, device)
        return
    scheduled_items = []
    effective_layout = args.sampling_layout_world_size or args.legacy_layout_world_size
    for ordinal, row in conditions_for_run(args):
        if effective_layout is None and ordinal % args.world_size != rank:
            continue
        c = compile_condition(row, tokenizer, mask_id=MASK_TOKEN_ID, purpose=args.purpose)
        if args.mixture_peer_checkpoint and peer_tokenizer(c["prompt"],add_special_tokens=False)["input_ids"]!=c["prompt_token_ids"]:
            raise ValueError("mixture tokenizers encode a condition differently")
        if repair_only:
            from crystal_dlm.self_repair_data import install_repair_initial_body
            c = install_repair_initial_body(c, tokenizer)
        for j in range(args.candidates):
            scheduled_items.append((ordinal, j, c))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started, completed, successful = time.monotonic(), 0, 0
    with (args.output_dir / f"records.rank{rank}.jsonl").open("x", encoding="utf-8") as handle:
        for batch in sampling_batches(scheduled_items, batch_size=args.batch_size, rank=rank,
                                      world_size=args.world_size,
                                      layout_world_size=effective_layout):
            compiled = [c for _, _, c in batch]
            seeds = [path_seed(args.seed, c["record"]["group_id"], args.collection_round, j) for _, j, c in batch]
            x = torch.tensor([c["prompt_token_ids"] + c["initial_body"] for c in compiled], device=device)
            sampler = make_sampler(model, tokenizer, constraints, compiled, seeds, args.temperature,
                                   getattr(args, "_contact_settings", None),mixture_peer_model=getattr(args,"_mixture_model",None))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if repair_only and not compiled[0]["record"]["repair_parent_success"]:
                    if any(c["record"]["repair_parent_success"] for c in compiled):
                        raise ValueError("failed roots must not be mixed with live repair roots")
                    result = x
                    traces = [{"schema": "programmed_crystal_attempt_path_v1",
                               "initial_body": c["initial_body"], "mask_id": MASK_TOKEN_ID,
                               "temperature": args.temperature, "element_order": list(c["program"].element_order),
                               "events": [], "success": False, "failure": "self_repair_parent_unsuccessful"}
                              for c in compiled]
                else:
                    result, traces = sampler.run(x, torch.ones_like(x), construct=not repair_only,
                                                 cooperative=not args.reference_closure and not hasattr(model, "raw_initialization") and not args.construction_only,
                                                 closure=not args.reference_closure and not hasattr(model, "raw_initialization") and not args.construction_only,
                                                 full_cell_repair=args.full_cell_repair)
                reference_logs = [None] * len(batch)
                if args.reference_closure:
                    from crystal_dlm.programmed_path_reference import close_reference
                    result, reference_logs = close_reference(model, result, traces,
                        programs=[c["program"] for c in compiled], seeds=seeds, prompt_length=sampler.prompt_length,
                        attention_mask=torch.ones_like(x), allowed=sampler.allowed_ids, constraints=constraints,
                        temperature=args.temperature)
            for row_index, (ordinal, candidate, c) in enumerate(batch):
                body_ids = result[row_index, len(c["prompt_token_ids"]):].tolist()
                trace = traces[row_index]
                if not args.reference_closure and trace_terminal_body(trace) != body_ids:
                    raise RuntimeError("recorded attempted trace does not reconstruct endpoint")
                body = tokenizer.decode(body_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                if trace["success"]:
                    validate_completed_body(body, c)
                record = {**c["record"], "prompt": c["prompt"], "prompt_token_ids": c["prompt_token_ids"],
                          "condition_ordinal": ordinal, "candidate_index": candidate,
                          "trajectory_id": f"{c['record']['group_id']}:{args.collection_round}:{candidate}",
                          "collection_round": args.collection_round, "sampling_seed": seeds[row_index],
                          "sampling_batch_size": len(batch),
                          "sampling_layout_world_size": args.sampling_layout_world_size,
                          "num_atoms": c["program"].num_atoms, "checkpoint": args.checkpoint_path,
                          "success": trace["success"], "body": body, "final_body_token_ids": body_ids,
                          "trace": trace, "trace_summary": trace_summary(trace),
                          "trace_scope": "predictor_only_legacy_revision_ledger" if args.reference_closure else "full_attempted_path",
                          "legacy_reference_closure": reference_logs[row_index]}
                if repair_only:
                    from crystal_dlm.self_repair_data import repair_net_change
                    record.update(path_mode="self_repair_support_check", diagnostic_only=True,
                                  trace_scope="full_attempted_self_repair",
                                  **repair_net_change(trace["initial_body"], body_ids))
                if args.short_contact_spec:
                    record.update(short_contact_policy=args._contact_settings.metadata(),
                                  legacy_layout_world_size=args.legacy_layout_world_size,
                                  effective_sampling_layout_world_size=effective_layout)
                if args.mixture_peer_checkpoint:
                    record.update(probability_mixture={"schema":"k4k8_equal_same_state_probability_mixture_v1",
                        "checkpoints":[str(args.checkpoint_path),str(args.mixture_peer_checkpoint)],"weights":[.5,.5],
                        "temperature":.7,"same_state":True,"contact":False},
                        effective_sampling_layout_world_size=effective_layout,
                        mixture_counted_forward_calls=list(sampler.component_forward_calls) if row_index==0 else [0,0])
                if args.construction_only:
                    record.update(method=args._construction_method,path_mode="construction_only",construction_only=True,
                                  trace_scope="construction_only_attempt",parseable=bool(trace["success"]),
                                  native_execution_success=bool(trace["success"]))
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                completed += 1
                successful += int(trace["success"])
            handle.flush()
            print(json.dumps({"rank": rank, "completed": completed, "successful": successful,
                              "elapsed_seconds": time.monotonic() - started}), flush=True)
    (args.output_dir / f"_SUCCESS.rank{rank}").touch()


if __name__ == "__main__":
    main()
