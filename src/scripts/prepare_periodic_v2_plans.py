#!/usr/bin/env python3
"""Prepare every original MP20 source with frozen, sampled Planner conditions.

Run under the separately declared preprocessing allocation. This performs no
training, chemical resampling, composition deduplication, relaxation or scoring.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import sys
import time

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch
import torch.distributed as dist

from crystal_dlm.c3fd_llama_fused_plan import STABILITY_CONDITIONS
from crystal_dlm.c3fd_llama_typed_planner import SOFT_FIELDS
from crystal_dlm.periodic_v2_plan_data import (
    SCHEMA, SAMPLING, FALLBACK, TYPED_INPUT_KEYS, answer_sha256,
    build_source_requests, collate_prediction_requests, finish_source,
    sample_soft_fields, summarize_outputs, validate_clean_source,
)
from crystal_dlm.semantic_composition_head import SemanticHeadFlags
from crystal_dlm.species_program_pointer import PlanConditionedSpeciesPointer, SpeciesPointerConfig
from scripts.train_c3fd_llama_typed_planner import load_frozen_c3fd, sha256_file
from scripts.train_spad_species_pointer import load_frozen_planner


EXPECTED_ROWS = {"train": 27136, "val": 9047}


def read_rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"non-object source record: {path}")
    return rows


def write_rows(path, rows):
    with Path(path).open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def input_files(args):
    files = {
        "c3fd_checkpoint": args.c3fd_checkpoint,
        "c3fd_vocabulary": args.vocabulary,
        "typed_config": args.planner_final / "typed_residual_config.json",
        "typed_state": args.planner_final / "typed_residual_state.pt",
        "llama_adapter_config": args.planner_final / "llama_adapter" / "adapter_config.json",
        "llama_adapter_weights": args.planner_final / "llama_adapter" / "adapter_model.safetensors",
        "species_pointer_state": args.pointer_state,
        "dlm_vocab_tokens": args.sft_data_dir / "vocab_tokens.txt",
        "sft_manifest": args.sft_data_dir / "manifest.json",
        "pointer_manifest": args.pointer_data_dir / "manifest.json",
        "preprocessing_script": Path(__file__).resolve(),
        "preprocessing_helper": SRC / "crystal_dlm" / "periodic_v2_plan_data.py",
        "frozen_planner_loader": SRC / "scripts" / "train_spad_species_pointer.py",
        "frozen_c3fd_loader": SRC / "scripts" / "train_c3fd_llama_typed_planner.py",
        "typed_planner_code": SRC / "crystal_dlm" / "c3fd_llama_typed_planner.py",
        "pointer_code": SRC / "crystal_dlm" / "species_program_pointer.py",
    }
    for split in EXPECTED_ROWS:
        files[f"sft_{split}"] = args.sft_data_dir / f"{split}.jsonl"
        files[f"pointer_{split}"] = args.pointer_data_dir / f"{split}.jsonl"
    model_weights = sorted(set(args.llama_model.glob("*.safetensors"))
                           | set(args.llama_model.glob("pytorch_model*.bin")))
    if not model_weights:
        raise ValueError("the original local Planner Llama weights are unavailable")
    for path in [args.llama_model / "config.json", *model_weights]:
        files["llama_base/" + path.name] = path
    for path in args.llama_model.glob("*.index.json"):
        files["llama_base/" + path.name] = path
    for name, path in files.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing immutable input {name}: {path}")
    return files


def assert_frozen_modules(*modules):
    for module in modules:
        if module.training or any(parameter.requires_grad or parameter.grad is not None
                                  for parameter in module.parameters()):
            raise RuntimeError("all preprocessing models must remain frozen, in eval mode, without gradients")


def load_modules(args, device):
    bundle = load_frozen_c3fd(
        checkpoint_path=args.c3fd_checkpoint, vocabulary_path=args.vocabulary,
        checkpoint_sha256=sha256_file(args.c3fd_checkpoint),
        vocabulary_sha256=sha256_file(args.vocabulary),
    )
    bundle.model.to(device)
    bundle = replace(bundle, context=bundle.context.to(device))
    config_payload = json.loads((args.planner_final / "typed_residual_config.json").read_text(encoding="utf-8"))
    if config_payload.get("schema") != "c3fd_llama_typed_planner_final_config_v1":
        raise ValueError("frozen Planner config schema differs")
    if config_payload.get("proposal_state_encoding") != "zero_query_then_frozen_stratum_index_plus_one":
        raise ValueError("frozen Planner proposal-state encoding differs")
    goal_to_id = config_payload.get("stability_goal_to_id")
    if goal_to_id != {value: index for index, value in enumerate(STABILITY_CONDITIONS)}:
        raise ValueError("frozen Planner stability-goal encoding differs")
    llama, typed, typed_config = load_frozen_planner(
        llama_model=args.llama_model, planner_final=args.planner_final, device=device,
    )
    state = torch.load(args.pointer_state, map_location="cpu", weights_only=False)
    if state.get("schema") != "spad_species_pointer_state_v1":
        raise ValueError("frozen species pointer state schema differs")
    pointer_config = SpeciesPointerConfig(**state["config"])
    if (pointer_config.llama_hidden_size != typed_config.llama_hidden_size
            or int(llama.config.hidden_size) != typed_config.llama_hidden_size
            or typed_config.num_proposal_states != len(bundle.stratum_to_index) + 1
            or typed_config.num_proposal_strata != len(bundle.stratum_to_index)
            or typed_config.num_species != int(bundle.model.config.num_species)
            or typed_config.max_count != int(bundle.model.config.max_count)
            or typed_config.ledger_feature_size != 6):
        raise ValueError("frozen C3FD, Llama, typed residual and pointer contracts differ")
    pointer_dims = (pointer_config.num_lattice_systems, pointer_config.num_spacegroup_buckets,
                    pointer_config.num_volume_per_atom_bins)
    for field, pointer_dim in zip(SOFT_FIELDS, pointer_dims):
        if pointer_dim != len(bundle.vocabulary["soft_vocabulary"][field]) or pointer_dim != typed_config.soft_head_dims[field]:
            raise ValueError("frozen soft vocabularies differ")
    pointer = PlanConditionedSpeciesPointer(pointer_config).to(device).eval()
    pointer.load_state_dict(state["state_dict"], strict=True)
    pointer.requires_grad_(False)
    assert_frozen_modules(bundle.model, llama, typed, pointer)
    return bundle, llama, typed, pointer, goal_to_id


@torch.inference_mode()
def predict_batch(requests, *, bundle, llama, typed, pointer, goal_to_id, seed, device):
    """Run the original chemical history; pointer inputs are freshly sampled."""
    assert_frozen_modules(bundle.model, llama, typed, pointer)
    batch = collate_prediction_requests(requests, bundle, goal_to_id=goal_to_id, device=device)
    embeddings = typed.typed_inputs_embeds(
        stability_goal_ids=batch["stability_goal_ids"],
        proposal_state_ids=batch["proposal_state_ids"],
        previous_species_indices=batch["previous_species_indices"],
        previous_count_values=batch["previous_count_values"],
        ledger_features=batch["ledger_features"],
    ).to(dtype=next(llama.parameters()).dtype)
    hidden = llama(inputs_embeds=embeddings, attention_mask=batch["attention_mask"],
                   output_hidden_states=True, use_cache=False, return_dict=True).hidden_states[-1]
    residual = typed(hidden.to(dtype=next(typed.parameters()).dtype),
                     soft_position_indices=batch["soft_position_indices"])
    base_output = bundle.model(
        bundle.context.expand(len(requests), -1),
        previous_species_indices=batch["previous_species_indices"],
        previous_count_values=batch["previous_count_values"],
        previous_n_values=batch["previous_n_values"],
        ledger_features=batch["ledger_features"], flags=SemanticHeadFlags(use_physics=True),
    )
    row_ids = torch.arange(len(requests), device=device)
    positions = batch["soft_position_indices"]
    sampled, audits = [], []
    for row, request in enumerate(requests):
        chosen, audit = sample_soft_fields(
            {field: base_output.rich_logits[field][row, positions[row]] for field in SOFT_FIELDS},
            {field: residual.soft_fields[field][row] for field in SOFT_FIELDS},
            bundle.vocabulary["soft_vocabulary"], seed=seed,
            split=request.source_split, source_row_idx=request.source_row_idx,
        )
        sampled.append(chosen)
        audits.append(audit)
    actual_soft_ids = torch.tensor([[row[field] for field in SOFT_FIELDS] for row in sampled],
                                  dtype=torch.long, device=device)
    orders = pointer.decode(
        hidden[row_ids, positions].float(), batch["pointer_atomic_numbers"],
        batch["pointer_counts"], batch["pointer_valid_mask"], actual_soft_ids,
    ).cpu()
    output = []
    for row, request in enumerate(requests):
        count = len(request.source["plan_state"]["elements"])
        indices = [int(value) for value in orders[row, :count].tolist()]
        if sorted(indices) != list(range(count)):
            raise RuntimeError("frozen pointer returned an invalid element permutation")
        output.append({
            "source_row_idx": request.source_row_idx, "source_split": request.source_split,
            "soft_ids": sampled[row],
            "soft_values": {field: str(bundle.vocabulary["soft_vocabulary"][field][sampled[row][field]])
                            for field in SOFT_FIELDS},
            "species_program_indices": indices, "soft_sampling_audit": audits[row],
        })
    assert_frozen_modules(bundle.model, llama, typed, pointer)
    return output


def merge_split(shards, source_rows, *, split):
    sources = {row["source_row_idx"]: row for row in source_rows}
    by_source = {}
    for shard in shards:
        for row in shard:
            index = row["source_row_idx"]
            if index not in sources or index in by_source:
                raise ValueError("merged preprocessing lost source uniqueness or changed its denominator")
            original = sources[index]
            validate_clean_source(row, split=split)
            evidence = row["condition_prediction"]
            if (row["answer"] != original["answer"]
                    or evidence["clean_answer_sha256_before"] != answer_sha256(original["answer"])
                    or evidence["clean_answer_sha256_after"] != answer_sha256(original["answer"])):
                raise ValueError("a preprocessing shard changed the exact source clean target")
            original_hard = {key: value for key, value in original["plan_state"].items() if key not in SOFT_FIELDS}
            new_hard = {key: value for key, value in row["plan_state"].items() if key not in SOFT_FIELDS}
            if original_hard != new_hard:
                raise ValueError("preprocessing changed non-soft original Plan fields")
            by_source[index] = row
    if set(by_source) != set(sources):
        raise ValueError("preprocessing did not retain all original source rows")
    return [by_source[row["source_row_idx"]] for row in source_rows]


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pointer-data-dir", "sft-data-dir", "c3fd-checkpoint", "vocabulary",
                 "llama-model", "planner-final", "pointer-state", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260906)
    return parser


def main():
    args = build_parser().parse_args()
    if args.batch_size < 1 or args.seed < 0:
        raise ValueError("batch size must be positive and seed nonnegative")
    if "SLURM_JOB_ID" not in os.environ or not os.environ.get("CUDA_VISIBLE_DEVICES") or not torch.cuda.is_available():
        raise RuntimeError("frozen Planner preparation requires its declared GPU allocation")
    world, local = int(os.environ.get("WORLD_SIZE", 1)), int(os.environ.get("LOCAL_RANK", 0))
    if not 1 <= world <= 6:
        raise ValueError("preprocessing world size must fit the declared six-card budget")
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    torch.set_num_threads(2)
    if world > 1:
        dist.init_process_group("nccl")
    rank = dist.get_rank() if world > 1 else 0
    for directory in (args.pointer_data_dir, args.sft_data_dir, args.planner_final):
        if not (directory / "_SUCCESS").is_file():
            raise ValueError(f"required original input is incomplete: {directory}")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    files = input_files(args)
    input_hashes = {name: sha256_file(path) for name, path in files.items()} if rank == 0 else None
    bundle, llama, typed, pointer, goal_to_id = load_modules(args, device)
    if world > 1:
        dist.barrier()  # All ranks must pass the non-existing-output check before creation.
    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    if world > 1:
        dist.barrier()
    started = time.monotonic()
    sources_by_split, request_reports = {}, {}
    for split, expected in EXPECTED_ROWS.items():
        source_rows = read_rows(args.sft_data_dir / f"{split}.jsonl")
        pointer_rows = read_rows(args.pointer_data_dir / f"{split}.jsonl")
        requests = build_source_requests(
            source_rows, pointer_rows, bundle, split=split, goal_to_id=goal_to_id,
            expected_rows=expected, max_elements=pointer.config.max_elements,
            max_count=pointer.config.max_count,
        )
        sources_by_split[split] = source_rows
        request_reports[split] = {"original_pointer_metadata_rows": len(pointer_rows)}
        local_requests = [request for request in requests if request.ordinal % world == rank]
        shard_path = args.output_dir / f"{split}.rank{rank}.jsonl"
        with shard_path.open("x", encoding="utf-8") as stream:
            for offset in range(0, len(local_requests), args.batch_size):
                chunk = local_requests[offset:offset + args.batch_size]
                predictive = [request for request in chunk if request.status != FALLBACK]
                predictions = (predict_batch(predictive, bundle=bundle, llama=llama, typed=typed,
                                             pointer=pointer, goal_to_id=goal_to_id,
                                             seed=args.seed, device=device) if predictive else [])
                by_source = {row["source_row_idx"]: row for row in predictions}
                for request in chunk:
                    row = finish_source(request, by_source.get(request.source_row_idx))
                    stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
                stream.flush()
                if offset == 0 or (offset // args.batch_size + 1) % 10 == 0:
                    print(json.dumps({"rank": rank, "split": split,
                                      "completed": min(offset + args.batch_size, len(local_requests)),
                                      "rank_requests": len(local_requests),
                                      "elapsed_seconds": time.monotonic() - started}), flush=True)
    assert_frozen_modules(bundle.model, llama, typed, pointer)
    if world > 1:
        dist.barrier()
    if rank == 0:
        reports = {}
        for split in EXPECTED_ROWS:
            combined = merge_split([read_rows(args.output_dir / f"{split}.rank{r}.jsonl") for r in range(world)],
                                   sources_by_split[split], split=split)
            write_rows(args.output_dir / f"{split}.jsonl", combined)
            write_rows(args.output_dir / f"coverage.{split}.jsonl", [
                {"source_row_idx": row["source_row_idx"], "source_split": row["source_split"],
                 "species_program_source": row["species_program_source"],
                 "soft_plan_source": row["soft_plan_source"], **row["condition_prediction"]}
                for row in combined
            ])
            reports[split] = {**request_reports[split], **summarize_outputs(combined)}
        shutil.copyfile(args.sft_data_dir / "vocab_tokens.txt", args.output_dir / "vocab_tokens.txt")
        after_hashes = {name: sha256_file(path) for name, path in files.items()}
        if after_hashes != input_hashes:
            raise RuntimeError("an original input or frozen model file changed during preprocessing")
        output_hashes = {path.name: sha256_file(path) for path in args.output_dir.iterdir() if path.is_file()}
        manifest = {
            "schema": SCHEMA, "splits": reports,
            "seed": args.seed, "sampling": SAMPLING,
            "sampling_transform_order": "legal_unit_weight_PoE_then_temperature_then_top_p_then_multinomial",
            "legacy_224_difference": "legacy soft MAP replaced by independent per-source sampled soft IDs",
            "pointer_uses_actual_sampled_soft_ids": True,
            "typed_input_whitelist": list(TYPED_INPUT_KEYS),
            "model_input_excludes": ["contact_tree_order", "teacher_soft_targets", "clean_crystal_geometry", "energy_labels"],
            "fallback_policy": "same_split_composition_family_explicit_goal_original_metadata_reuse_else_canonical_teacher_soft",
            "metadata_reuse_draw_seed": "recipient_split_source_row_idx_field_seed",
            "goal_defaulting": False, "source_composition_deduplication": False,
            "all_source_clean_answers_preserved": True,
            "all_sources_predicted": all(report["all_sources_predicted"] for report in reports.values()),
            "planner_training": False, "composition_resampled": False, "outcomes_read": False,
            "generated_path_files_read": False,
            "input_files": {name: str(path) for name, path in files.items()},
            "input_sha256_before": input_hashes, "input_sha256_after": after_hashes,
            "output_sha256": output_hashes, "world_size": world, "batch_size": args.batch_size,
            "floating_point_note": "sampling RNG is independent of batch/world order; kernel roundoff is not an exact-bitwise guarantee",
            "elapsed_seconds": time.monotonic() - started, "slurm_job_id": os.environ["SLURM_JOB_ID"],
        }
        (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (args.output_dir / "_SUCCESS").touch()
        print(json.dumps(manifest, sort_keys=True), flush=True)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
