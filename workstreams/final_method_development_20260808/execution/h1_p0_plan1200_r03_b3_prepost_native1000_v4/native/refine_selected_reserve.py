#!/usr/bin/env python3
"""Refine only selected reserve candidates not already refined by V3."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from native_protocol import (
    NATIVE_DENOMINATOR,
    PREFIX_COUNT,
    candidate_seed,
    read_json,
    read_jsonl,
    sha256_file,
    validate_arm,
    validate_repeat,
    write_json_exclusive,
    write_jsonl_exclusive,
)
from protocol import ordered_rows, validate_config
from refine1000 import _write_payload
from scripts.refine_dlm_with_crysllmgen import (
    ProposalDataset,
    lattices_to_params_shape,
    setup_crysllmgen_imports,
)


def _load_selected_reserve_graphs(
    path: Path,
    *,
    arm: str,
    repeat: int,
    reserve_ordinals: set[int],
) -> dict[int, dict[str, Any]]:
    loaded = torch.load(path.resolve(), map_location="cpu")
    if not isinstance(loaded, list) or len(loaded) != NATIVE_DENOMINATOR:
        raise ValueError("selected proposal graph payload changed")
    graphs: dict[int, dict[str, Any]] = {}
    all_ordinals: set[int] = set()
    for wrapper in loaded:
        if not isinstance(wrapper, Mapping) or not isinstance(wrapper.get("graph"), Mapping):
            raise ValueError("selected proposal graph record is malformed")
        ordinal = int(wrapper.get("ordinal", -1))
        all_ordinals.add(ordinal)
        if ordinal not in reserve_ordinals:
            continue
        graph = dict(wrapper["graph"])
        metadata = graph.get("h1_plan1200_prepost_metadata")
        if (
            ordinal in graphs
            or not isinstance(metadata, Mapping)
            or int(metadata.get("ordinal", -1)) != ordinal
            or int(metadata.get("candidate_rank", -1)) < PREFIX_COUNT
            or str(metadata.get("arm")) != arm
            or int(metadata.get("repeat", -1)) != repeat
            or str(metadata.get("schedule_arm")) != "D2_SAFE_AXIS"
        ):
            raise ValueError(f"reserve graph identity changed at native ordinal {ordinal}")
        graphs[ordinal] = graph
    if all_ordinals != set(range(NATIVE_DENOMINATOR)) or set(graphs) != reserve_ordinals:
        raise ValueError("selected reserve graphs do not match selected body rows")
    return graphs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True)
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument("--body-config", type=Path, required=True)
    parser.add_argument("--selected-body-attempts", type=Path, required=True)
    parser.add_argument("--selected-proposal-graphs", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--crysllmgen-dir", type=Path, required=True)
    args = parser.parse_args()

    arm = validate_arm(args.arm)
    repeat = validate_repeat(args.repeat)
    config = read_json(args.body_config.resolve())
    validate_config(config)
    refiner = config["refiner"]
    if (
        args.checkpoint.resolve() != Path(refiner["checkpoint"]).resolve()
        or args.checkpoint_sha256 != refiner["checkpoint_sha256"]
        or int(refiner["timesteps"]) != 1000
        or int(refiner["diffusion_steps"]) != 800
        or int(refiner["num_evals"]) != 1
        or int(refiner["effective_batch_size"]) != 1
    ):
        raise ValueError("frozen refiner identity or configuration changed")

    attempts = ordered_rows(
        read_jsonl(args.selected_body_attempts.resolve()), ordinal_field="ordinal"
    )
    reserve_rows = [
        row for row in attempts if int(row.get("source_candidate_rank", -1)) >= PREFIX_COUNT
    ]
    reserve_ordinals = {int(row["ordinal"]) for row in reserve_rows}
    if (
        any(row.get("status") != "succeeded" for row in attempts)
        or any(
            int(row.get("source_refiner_noise_seed", -1))
            != candidate_seed(repeat, int(row["source_candidate_rank"]), "refiner")
            for row in attempts
        )
    ):
        raise ValueError("selected native body ledger changed")
    graph_map = _load_selected_reserve_graphs(
        args.selected_proposal_graphs,
        arm=arm,
        repeat=repeat,
        reserve_ordinals=reserve_ordinals,
    )

    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    write_json_exclusive(
        output / "run_config.json",
        {
            "schema": "h1_plan1200_native_reserve_refiner_run_v1",
            "arm": arm,
            "repeat": repeat,
            "selected_native_denominator": NATIVE_DENOMINATOR,
            "selected_reserve_count": len(reserve_rows),
            "selected_reserve_native_ordinals": sorted(reserve_ordinals),
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256_recorded": args.checkpoint_sha256,
            "timesteps": 1000,
            "diff_steps": 800,
            "num_evals": 1,
            "effective_batch_size": 1,
            "prefix_refinement_reused": True,
            "same_candidate_retry": False,
            "replacement": False,
            "repair": False,
            "filter": False,
            "rerank": False,
        },
    )

    output_attempts: list[dict[str, Any]] = []
    refined_samples: list[dict[str, Any]] = []
    started = time.monotonic()
    if reserve_rows:
        _, CSPDiffusion, Data, DataLoader = setup_crysllmgen_imports(
            args.crysllmgen_dir.resolve()
        )
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("native reserve refiner expects exactly one CUDA device")
        if "A800" not in torch.cuda.get_device_name(0):
            raise RuntimeError("native reserve refiner expects an A800")
        device = torch.device("cuda:0")
        model = CSPDiffusion(1000, "train").to(device)
        model.device = device
        checkpoint = torch.load(args.checkpoint.resolve(), map_location=device)
        model.load_state_dict(
            checkpoint["model"] if "model" in checkpoint else checkpoint
        )
        model.eval()
        with torch.no_grad():
            for row in tqdm(reserve_rows, desc=f"{arm} repeat{repeat} native reserve refine800"):
                ordinal = int(row["ordinal"])
                rank = int(row["source_candidate_rank"])
                seed = candidate_seed(repeat, rank, "refiner")
                record = {
                    "schema": "h1_plan1200_native_reserve_refinement_attempt_v1",
                    "arm": arm,
                    "repeat": repeat,
                    "ordinal": ordinal,
                    "native_ordinal": ordinal,
                    "candidate_rank": rank,
                    "refiner_sampling_seed": seed,
                    "attempt_status": "failed",
                    "refiner_status": "failed",
                    "refiner_complete": False,
                    "retry_used": False,
                    "replacement_used": False,
                }
                try:
                    random.seed(seed)
                    np.random.seed(seed % (2**32))
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
                    dataset = ProposalDataset([graph_map[ordinal]], Data)
                    loader = DataLoader(dataset, batch_size=1, shuffle=False)
                    batch = next(iter(loader)).to(device)
                    sampled, _ = model.sample(batch, diff_steps=800)
                    lattices = sampled["lattices"].detach().cpu().unsqueeze(0)
                    lengths, angles = lattices_to_params_shape(lattices)
                    refined_samples.append(
                        {
                            "sample_idx": ordinal,
                            "candidate_rank": rank,
                            "frac_coords": sampled["frac_coords"].detach().cpu().unsqueeze(0),
                            "num_atoms": sampled["num_atoms"].detach().cpu().unsqueeze(0),
                            "atom_types": sampled["atom_types"].detach().cpu().unsqueeze(0),
                            "lengths": lengths,
                            "angles": angles,
                        }
                    )
                    record.update(
                        {
                            "attempt_status": "complete",
                            "refiner_status": "complete",
                            "refiner_complete": True,
                            "failure_reason": None,
                            "failure_message": None,
                        }
                    )
                except Exception as exc:  # noqa: BLE001
                    record.update(
                        {
                            "failure_reason": type(exc).__name__,
                            "failure_message": str(exc),
                        }
                    )
                output_attempts.append(record)

    elapsed = time.monotonic() - started
    write_jsonl_exclusive(output / "refinement_attempts.jsonl", output_attempts)
    payload = _write_payload(output, refined_samples, elapsed=elapsed)
    complete = sum(row.get("refiner_complete") is True for row in output_attempts)
    report = {
        "schema": "h1_plan1200_native_reserve_refinement_report_v1",
        "status": "complete" if complete == len(reserve_rows) else "failed_closed",
        "arm": arm,
        "repeat": repeat,
        "native_denominator": NATIVE_DENOMINATOR,
        "prefix_refinements_reused": NATIVE_DENOMINATOR - len(reserve_rows),
        "selected_reserve_count": len(reserve_rows),
        "reserve_refiner_complete": complete,
        "all_selected_candidates_refined_after_merge": complete == len(reserve_rows),
        "time_sec": elapsed,
        "diff_steps": 800,
        "output_file": str(payload),
        "output_sha256": sha256_file(payload),
        "same_candidate_retry": False,
        "replacement": False,
        "repair_filter_rerank": False,
    }
    write_json_exclusive(output / "refinement_metrics.json", report)
    print(json.dumps(report, sort_keys=True))
    if complete != len(reserve_rows):
        raise RuntimeError("not every selected reserve candidate completed refinement")


if __name__ == "__main__":
    main()
