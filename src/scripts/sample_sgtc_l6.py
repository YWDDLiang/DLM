#!/usr/bin/env python3
"""Sample one matched SGTC L6/L7 cell from minimal certified compositions."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from crystal_dlm.ctv_rollout import collect_ctv_branch_states  # noqa: E402
from crystal_dlm.fixed_slot import MASK_TOKEN_ID  # noqa: E402
from crystal_dlm.r5_dynamic_length import (  # noqa: E402
    count_prefill_for_batch,
    exact_body_token_count,
    exact_dynamic_generation_schedule,
    exact_dynamic_schema_constraints,
    validate_answer_matches_plan,
)
from crystal_dlm.sgtc_sampling import (  # noqa: E402
    matched_base_noise_group,
    validate_sgtc_attempts,
    validate_sgtc_denominator,
    validate_sgtc_plan_rows,
)
from scripts.sample_llada_dynamic_crystals import (  # noqa: E402
    build_dynamic_lightweight_constraints,
    graph_from_arrays,
    import_process_one,
    load_model_and_tokenizer,
)
from scripts.sample_llada_r5_exact_length import element_prefill_for_batch  # noqa: E402


def write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--prompt-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--crysllmgen-dir", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--reference-checkpoint-path")
    parser.add_argument("--late-guidance-scale", type=float, default=0.0)
    parser.add_argument(
        "--late-guidance-remaining-mask-threshold", type=float, default=0.0
    )
    args = parser.parse_args()

    denominator = validate_sgtc_denominator(args.num_samples)
    if not torch.cuda.is_available():
        raise RuntimeError("SGTC L6 sampling requires one CUDA device")
    device = torch.device("cuda", 0)
    model, tokenizer = load_model_and_tokenizer(
        args.model_path, args.checkpoint_path, device
    )
    reference_model = None
    if args.reference_checkpoint_path:
        if (
            float(args.late_guidance_scale) != 0.5
            or float(args.late_guidance_remaining_mask_threshold) != 0.25
        ):
            raise RuntimeError("SGTC fallback freezes late guidance at scale=0.5, threshold=0.25")
        reference_model, reference_tokenizer = load_model_and_tokenizer(
            args.model_path, args.reference_checkpoint_path, device
        )
        if reference_tokenizer.get_vocab() != tokenizer.get_vocab():
            raise RuntimeError("policy/reference tokenizer vocabularies differ")
    elif (
        float(args.late_guidance_scale) != 0.0
        or float(args.late_guidance_remaining_mask_threshold) != 0.0
    ):
        raise RuntimeError("late guidance parameters require --reference-checkpoint-path")
    process_one = import_process_one(args.crysllmgen_dir)
    plans = [
        json.loads(line)
        for line in args.prompt_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    plan_accounting = validate_sgtc_plan_rows(plans, expected=denominator)
    lightweight = build_dynamic_lightweight_constraints(
        tokenizer,
        duplicate_coordinate_mask=True,
        lattice_volume_mask=True,
        min_lattice_rad=1e-4,
    )
    attempts: list[dict[str, Any]] = []
    graphs: list[dict[str, Any]] = []
    failures: Counter[str] = Counter()
    guided_steps_total = 0
    denoise_steps_total = 0
    guided_attempts = 0
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    for ordinal, row in enumerate(plans):
        plan = dict(row["plan_state"])
        num_atoms = int(plan["N"])
        composition_id = str(row["reduced_composition_identity"])
        source_sample_idx = int(row.get("sample_idx", ordinal))
        prompt_text = str(row["prompt"]).rstrip() + "\n"
        encoded = tokenizer(
            [prompt_text], add_special_tokens=False, padding=True, return_tensors="pt"
        )
        prompt = encoded["input_ids"].to(device)
        attention = encoded["attention_mask"].to(device)
        generation_length = exact_body_token_count(num_atoms)
        allowed = exact_dynamic_schema_constraints(tokenizer, num_atoms)
        prefill = count_prefill_for_batch(tokenizer, num_atoms, 1)
        prefill.update(element_prefill_for_batch(tokenizer, [plan]))
        schedule = exact_dynamic_generation_schedule(num_atoms)
        base_noise_group = matched_base_noise_group(
            seed=int(args.seed),
            composition_id=composition_id,
            sample_idx=ordinal,
        )
        rollout_diagnostics: dict[str, Any] = {}
        final_tokens, snapshots = collect_ctv_branch_states(
            model,
            prompt,
            attention_mask=attention,
            num_atoms=num_atoms,
            gen_length=generation_length,
            temperature=float(args.temperature),
            mask_id=MASK_TOKEN_ID,
            allowed_token_ids_by_generation_pos=allowed,
            prefill_token_ids_by_generation_pos=prefill,
            generation_position_groups=schedule,
            lightweight_decoding_constraints=lightweight,
            base_noise_group=base_noise_group,
            milestones=(),
            reference_model=reference_model,
            late_guidance_scale=float(args.late_guidance_scale),
            late_guidance_remaining_mask_threshold=float(
                args.late_guidance_remaining_mask_threshold
            ),
            rollout_diagnostics=rollout_diagnostics,
        )
        guided_steps_total += int(rollout_diagnostics["guided_denoise_steps"])
        denoise_steps_total += int(rollout_diagnostics["total_denoise_steps"])
        guided_attempts += int(rollout_diagnostics["guided_denoise_steps"] > 0)
        if snapshots:
            raise RuntimeError("SGTC base sampler unexpectedly captured CTV states")
        text = tokenizer.batch_decode(
            final_tokens[:, prompt.shape[1] :],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )[0]
        record: dict[str, Any] = {
            "schema": "h1a2_sgtc_body_attempt_v1",
            "ordinal": ordinal,
            "sample_idx": ordinal,
            "source_sample_idx": source_sample_idx,
            "composition_id": composition_id,
            "plan_state": plan,
            "text": text,
            "parsed": False,
            "body_noise_seed": int(args.seed),
            "retry_or_replacement_used": False,
            "late_guidance": dict(rollout_diagnostics),
        }
        try:
            arrays = validate_answer_matches_plan(plan, text)
            graph, cif = graph_from_arrays(arrays, process_one)
            graph["sample_idx"] = ordinal
            graph["source_sample_idx"] = source_sample_idx
            graphs.append(graph)
            record.update(
                {
                    "parsed": True,
                    "num_atoms": int(arrays["num_atoms"]),
                    "cif": cif,
                }
            )
        except Exception as exc:  # noqa: BLE001 - failures stay in denominator.
            reason = f"{type(exc).__name__}:{str(exc)}"
            failures[reason] += 1
            record["reason"] = reason
        attempts.append(record)

    accounting = validate_sgtc_attempts(attempts, expected=denominator)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_jsonl(output / "raw_generations.jsonl", attempts)
    torch.save(graphs, output / "proposal_graphs.pt")
    manifest = {
        "schema": "h1a2_sgtc_body_manifest_v1",
        **accounting,
        **plan_accounting,
        "denominator": denominator,
        "graphs": len(graphs),
        "failures": dict(failures.most_common()),
        "seed": int(args.seed),
        "temperature": float(args.temperature),
        "exact_composition": True,
        "generation_schedule": "exact_axis",
        "guided_attempts": int(guided_attempts),
        "guided_denoise_steps": int(guided_steps_total),
        "late_guidance_remaining_mask_threshold": float(
            args.late_guidance_remaining_mask_threshold
        ),
        "late_guidance_scale": float(args.late_guidance_scale),
        "reference_checkpoint_path": args.reference_checkpoint_path,
        "total_denoise_steps": int(denoise_steps_total),
        "elapsed_seconds": time.time() - started,
        "max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "max_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "max_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "attempts_sha256": hashlib.sha256(
            (output / "raw_generations.jsonl").read_bytes()
        ).hexdigest(),
    }
    (output / "SGTC_BODY_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "_SUCCESS").touch()
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
