#!/usr/bin/env python3
"""Refine one four-cell body ledger with model_494 and frozen ordinal seeds."""

from __future__ import annotations

import argparse
import json
import random
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from protocol import (
    DENOMINATOR,
    ordered_rows,
    read_json,
    read_jsonl,
    validate_cell,
    validate_config,
    write_json_exclusive,
    write_jsonl_exclusive,
)
from scripts.refine_dlm_with_crysllmgen import (
    ProposalDataset,
    lattices_to_params_shape,
    setup_crysllmgen_imports,
)


def _load_graphs(
    path: Path, *, cell: str, success_ordinals: set[int]
) -> dict[int, dict[str, Any]]:
    loaded = torch.load(path, map_location="cpu")
    if not isinstance(loaded, list):
        raise TypeError("proposal graph payload is not a list")
    graphs: dict[int, dict[str, Any]] = {}
    for record in loaded:
        if not isinstance(record, dict) or not isinstance(record.get("graph"), dict):
            raise TypeError("proposal graph record is malformed")
        ordinal = int(record.get("ordinal", -1))
        graph = dict(record["graph"])
        metadata = graph.get("h1_ef_fourcell_metadata")
        if (
            ordinal in graphs
            or not isinstance(metadata, Mapping)
            or int(metadata.get("ordinal", -1)) != ordinal
            or str(metadata.get("cell")) != cell
            or str(metadata.get("schedule_arm")) != "D2_SAFE_AXIS"
        ):
            raise ValueError("proposal graph identity changed")
        graphs[ordinal] = graph
    if set(graphs) != success_ordinals:
        raise ValueError("proposal graphs do not exactly match body successes")
    return graphs


def _write_payload(
    output_dir: Path, samples: list[dict[str, Any]], *, elapsed: float
) -> Path:
    ordered = sorted(samples, key=lambda row: int(row["sample_idx"]))
    if ordered:
        payload = {
            "frac_coords": torch.cat([row["frac_coords"] for row in ordered], dim=1),
            "num_atoms": torch.cat([row["num_atoms"] for row in ordered], dim=1),
            "atom_types": torch.cat([row["atom_types"] for row in ordered], dim=1),
            "lengths": torch.cat([row["lengths"] for row in ordered], dim=1),
            "angles": torch.cat([row["angles"] for row in ordered], dim=1),
            "sample_idx": torch.tensor(
                [int(row["sample_idx"]) for row in ordered], dtype=torch.long
            ),
            "time": float(elapsed),
        }
    else:
        payload = {
            "frac_coords": torch.empty((1, 0, 3), dtype=torch.float32),
            "num_atoms": torch.empty((1, 0), dtype=torch.long),
            "atom_types": torch.empty((1, 0), dtype=torch.long),
            "lengths": torch.empty((1, 0, 3), dtype=torch.float32),
            "angles": torch.empty((1, 0, 3), dtype=torch.float32),
            "sample_idx": torch.empty((0,), dtype=torch.long),
            "time": float(elapsed),
        }
    output = output_dir / f"dlm_refined_mp_{len(ordered)}.pt"
    torch.save(payload, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--body-attempts", type=Path, required=True)
    parser.add_argument("--proposal-graphs", type=Path, required=True)
    parser.add_argument("--attempt-ledger", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--crysllmgen-dir", type=Path, required=True)
    args = parser.parse_args()

    cell = validate_cell(args.cell)
    config = read_json(args.config.resolve())
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
        read_jsonl(args.body_attempts.resolve()), ordinal_field="ordinal"
    )
    ledger = ordered_rows(
        read_jsonl(args.attempt_ledger.resolve()), ordinal_field="sample_idx"
    )
    success_ordinals = {
        ordinal
        for ordinal, row in enumerate(attempts)
        if row.get("status") == "succeeded"
    }
    seed_field = config["attempt_ledger"]["refiner_seed_field"]
    body_seed_field = config["attempt_ledger"]["body_seed_field"]
    if any(
        str(body_row.get("cell")) != cell
        or int(seed_row.get("ordinal", -1)) != ordinal
        or not isinstance(seed_row.get(seed_field), int)
        or (
            body_row.get("body_eligible") is True
            and int(body_row.get("body_noise_seed", -1))
            != int(seed_row.get(body_seed_field, -2))
        )
        for ordinal, (body_row, seed_row) in enumerate(
            zip(attempts, ledger, strict=True)
        )
    ):
        raise ValueError("body attempts and common seed ledger are not paired")
    graph_map = _load_graphs(
        args.proposal_graphs.resolve(),
        cell=cell,
        success_ordinals=success_ordinals,
    )

    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    write_json_exclusive(
        output / "run_config.json",
        {
            "schema": "h1_ef_fourcell_refiner_run_v1",
            "cell": cell,
            "body_attempts": str(args.body_attempts.resolve()),
            "proposal_graphs": str(args.proposal_graphs.resolve()),
            "attempt_ledger": str(args.attempt_ledger.resolve()),
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256_recorded": args.checkpoint_sha256,
            "checkpoint_rehashed": False,
            "num_samples": DENOMINATOR,
            "timesteps": 1000,
            "diff_steps": 800,
            "num_evals": 1,
            "effective_batch_size": 1,
            "seed_mode": refiner["seed_mode"],
            "retry": False,
            "replacement": False,
            "repair": False,
            "filter": False,
            "rerank": False,
        },
    )

    _, CSPDiffusion, Data, DataLoader = setup_crysllmgen_imports(
        args.crysllmgen_dir.resolve()
    )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("four-cell refiner expects exactly one CUDA device")
    if "A800" not in torch.cuda.get_device_name(0):
        raise RuntimeError("four-cell refiner expects an A800")
    device = torch.device("cuda:0")
    model = CSPDiffusion(1000, "train").to(device)
    model.device = device
    checkpoint = torch.load(args.checkpoint.resolve(), map_location=device)
    model.load_state_dict(
        checkpoint["model"] if "model" in checkpoint else checkpoint
    )
    model.eval()

    output_attempts: list[dict[str, Any]] = []
    refined_samples: list[dict[str, Any]] = []
    started = time.monotonic()
    with torch.no_grad():
        for ordinal in tqdm(range(DENOMINATOR), desc=f"{cell} refine800"):
            body = dict(attempts[ordinal])
            seed_record = ledger[ordinal]
            seed = int(seed_record[seed_field])
            record = {
                **body,
                "schema": "h1_ef_fourcell_refinement_attempt_v1",
                "cell": cell,
                "sample_idx": ordinal,
                "ordinal": ordinal,
                "refiner_sampling_seed": seed,
                "refiner_seed_source": "frozen_h1_attempt_ledger",
            }
            if body.get("status") != "succeeded":
                record.update(
                    {
                        "attempt_status": "failed",
                        "refiner_status": "not_started",
                        "refiner_complete": False,
                        "failure_reason": str(
                            body.get("reason") or "upstream_failed"
                        ),
                        "failure_message": str(body.get("reason") or ""),
                    }
                )
                output_attempts.append(record)
                continue

            record.update(
                {
                    "attempt_status": "failed",
                    "earliest_failure_stage": "refiner",
                    "refiner_status": "failed",
                    "refiner_complete": False,
                }
            )
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
                        "cell": cell,
                        "frac_coords": sampled["frac_coords"]
                        .detach()
                        .cpu()
                        .unsqueeze(0),
                        "num_atoms": sampled["num_atoms"]
                        .detach()
                        .cpu()
                        .unsqueeze(0),
                        "atom_types": sampled["atom_types"]
                        .detach()
                        .cpu()
                        .unsqueeze(0),
                        "lengths": lengths,
                        "angles": angles,
                    }
                )
                record.update(
                    {
                        "attempt_status": "complete",
                        "earliest_failure_stage": None,
                        "failure_reason": None,
                        "failure_message": None,
                        "refiner_status": "complete",
                        "refiner_complete": True,
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
    failures: dict[str, int] = {}
    for record in output_attempts:
        if record.get("attempt_status") == "complete":
            continue
        stage = str(record.get("earliest_failure_stage") or "upstream")
        reason = str(record.get("failure_reason") or "unknown").split(":", 1)[0]
        key = f"{stage}:{reason}"
        failures[key] = failures.get(key, 0) + 1
    report = {
        "schema": "h1_ef_fourcell_refiner_metrics_v1",
        "status": "complete",
        "cell": cell,
        "all_attempt_denominator": DENOMINATOR,
        "body_complete": len(success_ordinals),
        "refiner_complete": len(refined_samples),
        "failures": failures,
        "time_sec": elapsed,
        "diff_steps": 800,
        "num_evals": 1,
        "effective_batch_size": 1,
        "output_file": str(payload),
        "retry": False,
        "replacement": False,
        "repair": False,
        "filter": False,
        "rerank": False,
    }
    write_json_exclusive(output / "refinement_metrics.json", report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
