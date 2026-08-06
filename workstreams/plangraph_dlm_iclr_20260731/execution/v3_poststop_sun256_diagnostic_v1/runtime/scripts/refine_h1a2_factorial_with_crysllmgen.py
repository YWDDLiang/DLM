#!/usr/bin/env python3
"""Refine one H1-A2 factorial arm with ordinal-stable CrysLLMGen noise."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm

from crystal_dlm.fixed_slot import write_json  # noqa: E402
from crystal_dlm.h1a2_factorial_contract import FACTORIAL_ARMS  # noqa: E402
from crystal_dlm.h1a2_factorial_runtime import (  # noqa: E402
    ordered_single_arm_attempts,
    read_jsonl_objects,
)
from scripts.refine_dlm_with_crysllmgen import (  # noqa: E402
    ProposalDataset,
    init_distributed,
    lattices_to_params_shape,
    setup_crysllmgen_imports,
)


FROZEN_DIFF_STEPS = 800
FROZEN_TIMESTEPS = 1000
FROZEN_NUM_EVALS = 1


def _rank_path(
    output_dir: Path,
    filename: str,
    rank: int,
    distributed: bool,
) -> Path:
    if not distributed:
        return output_dir / filename
    path = Path(filename)
    return output_dir / f"{path.stem}.rank{rank}{path.suffix}"


def _write_jsonl(path: Path, records: list[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _graph_identity(graph: Mapping[str, Any]) -> tuple[int, str]:
    metadata = graph.get("h1a2_metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("proposal graph lacks H1-A2 metadata")
    return int(metadata["sample_idx"]), str(metadata["factorial_arm"])


def _load_body_graphs(
    path: Path,
    *,
    body_attempts: list[Mapping[str, Any]],
    factorial_arm: str,
) -> dict[int, dict[str, Any]]:
    loaded = torch.load(path, map_location="cpu")
    if not isinstance(loaded, list):
        raise TypeError("proposal graph payload is not a list")
    graphs: dict[int, dict[str, Any]] = {}
    for graph in loaded:
        if not isinstance(graph, dict):
            raise TypeError("proposal graph row is not a dictionary")
        sample_idx, arm = _graph_identity(graph)
        if arm != factorial_arm:
            raise ValueError("proposal graph factorial arm mismatch")
        if sample_idx in graphs:
            raise ValueError(f"duplicate proposal graph ordinal {sample_idx}")
        graphs[sample_idx] = graph
    expected = {
        int(record["sample_idx"])
        for record in body_attempts
        if record.get("body_status") == "complete"
    }
    if set(graphs) != expected:
        raise ValueError("proposal graphs do not exactly match body-success ordinals")
    return graphs


def _refiner_metrics(
    attempts: list[Mapping[str, Any]],
    *,
    elapsed: float,
    rank: int,
    world_size: int,
    distributed: bool,
) -> dict[str, Any]:
    body_complete = sum(
        record.get("body_status") == "complete" for record in attempts
    )
    refiner_complete = sum(
        record.get("refiner_status") == "complete" for record in attempts
    )
    failures: dict[str, int] = {}
    for record in attempts:
        if record.get("attempt_status") == "failed":
            stage = str(record.get("earliest_failure_stage") or "unknown")
            reason = str(record.get("failure_reason") or "unknown")
            key = f"{stage}:{reason}"
            failures[key] = failures.get(key, 0) + 1
    denominator = len(attempts)
    return {
        "schema": "h1a2_factorial_refiner_metrics_v1",
        "all_attempt_denominator": denominator,
        "body_complete": body_complete,
        "refiner_complete": refiner_complete,
        "body_completion_rate_all_attempt": body_complete / max(1, denominator),
        "refiner_completion_rate_all_attempt": refiner_complete
        / max(1, denominator),
        "failures": failures,
        "time_sec": float(elapsed),
        "rank": int(rank),
        "world_size": int(world_size),
        "distributed": bool(distributed),
        "diff_steps": FROZEN_DIFF_STEPS,
        "num_evals": FROZEN_NUM_EVALS,
        "effective_batch_size": 1,
        "retry": False,
        "replacement": False,
        "repair": False,
        "filter": False,
        "rerank": False,
    }


def _write_final_refined_payload(
    output_dir: Path,
    samples: list[dict[str, Any]],
    *,
    elapsed: float,
) -> Path:
    ordered = sorted(samples, key=lambda row: int(row["sample_idx"]))
    if ordered:
        payload = {
            "frac_coords": torch.cat(
                [row["frac_coords"] for row in ordered],
                dim=1,
            ),
            "num_atoms": torch.cat(
                [row["num_atoms"] for row in ordered],
                dim=1,
            ),
            "atom_types": torch.cat(
                [row["atom_types"] for row in ordered],
                dim=1,
            ),
            "lengths": torch.cat(
                [row["lengths"] for row in ordered],
                dim=1,
            ),
            "angles": torch.cat(
                [row["angles"] for row in ordered],
                dim=1,
            ),
            "sample_idx": torch.tensor(
                [int(row["sample_idx"]) for row in ordered],
                dtype=torch.long,
            ),
            "time": float(elapsed),
        }
    else:
        payload = {
            "frac_coords": torch.empty(
                (FROZEN_NUM_EVALS, 0, 3),
                dtype=torch.float32,
            ),
            "num_atoms": torch.empty(
                (FROZEN_NUM_EVALS, 0),
                dtype=torch.long,
            ),
            "atom_types": torch.empty(
                (FROZEN_NUM_EVALS, 0),
                dtype=torch.long,
            ),
            "lengths": torch.empty(
                (FROZEN_NUM_EVALS, 0, 3),
                dtype=torch.float32,
            ),
            "angles": torch.empty(
                (FROZEN_NUM_EVALS, 0, 3),
                dtype=torch.float32,
            ),
            "sample_idx": torch.empty((0,), dtype=torch.long),
            "time": float(elapsed),
        }
    output_file = output_dir / f"dlm_refined_mp_{len(ordered)}.pt"
    torch.save(payload, output_file)
    return output_file


def _merge_refiner_outputs(
    output_dir: Path,
    *,
    world_size: int,
    expected_count: int,
    factorial_arm: str,
    elapsed: float,
) -> None:
    attempt_rows: list[Mapping[str, Any]] = []
    samples: list[dict[str, Any]] = []
    for rank in range(int(world_size)):
        attempt_path = _rank_path(
            output_dir,
            "refinement_attempts.jsonl",
            rank,
            True,
        )
        sample_path = _rank_path(
            output_dir,
            "refined_samples.pt",
            rank,
            True,
        )
        if not attempt_path.exists():
            raise FileNotFoundError(f"missing refinement rank output: {attempt_path}")
        attempt_rows.extend(read_jsonl_objects(attempt_path))
        if sample_path.exists():
            loaded = torch.load(sample_path, map_location="cpu")
            if not isinstance(loaded, list):
                raise TypeError(f"{sample_path} is not a refined-sample list")
            samples.extend(loaded)

    attempts = ordered_single_arm_attempts(
        attempt_rows,
        expected_count=int(expected_count),
        expected_factorial_arm=factorial_arm,
    )
    successful = {
        int(record["sample_idx"])
        for record in attempts
        if record.get("refiner_status") == "complete"
    }
    sample_map: dict[int, dict[str, Any]] = {}
    for sample in samples:
        idx = int(sample["sample_idx"])
        if sample.get("factorial_arm") != factorial_arm:
            raise ValueError("refined sample arm mismatch")
        if idx in sample_map:
            raise ValueError(f"duplicate refined ordinal {idx}")
        sample_map[idx] = sample
    if set(sample_map) != successful:
        raise ValueError("refined samples do not match refiner-success attempts")

    ordered_samples = [sample_map[idx] for idx in sorted(successful)]
    _write_jsonl(output_dir / "refinement_attempts.jsonl", list(attempts))
    output_file = _write_final_refined_payload(
        output_dir,
        ordered_samples,
        elapsed=elapsed,
    )
    metrics = _refiner_metrics(
        list(attempts),
        elapsed=elapsed,
        rank=0,
        world_size=int(world_size),
        distributed=True,
    )
    metrics["output_file"] = str(output_file)
    write_json(str(output_dir / "refinement_metrics.json"), metrics)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--body-attempts", type=Path, required=True)
    parser.add_argument("--proposal-graphs", type=Path, required=True)
    parser.add_argument("--factorial-arm", choices=FACTORIAL_ARMS, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--crysllmgen-dir",
        type=Path,
        default=PROJECT_ROOT / "reference/crysllmgen",
    )
    parser.add_argument("--num-samples", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if int(args.num_samples) <= 0:
        raise ValueError("--num-samples must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dist_info = init_distributed()
    rank = int(dist_info["rank"])
    world_size = int(dist_info["world_size"])
    distributed = bool(dist_info["distributed"])
    is_main = bool(dist_info["is_main"])

    body_attempts = ordered_single_arm_attempts(
        read_jsonl_objects(args.body_attempts),
        expected_count=int(args.num_samples),
        expected_factorial_arm=args.factorial_arm,
    )
    graph_map = _load_body_graphs(
        args.proposal_graphs,
        body_attempts=list(body_attempts),
        factorial_arm=args.factorial_arm,
    )
    if is_main:
        write_json(
            str(args.output_dir / "run_config.json"),
            {
                "schema": "h1a2_factorial_refiner_run_v1",
                "body_attempts": str(args.body_attempts),
                "proposal_graphs": str(args.proposal_graphs),
                "factorial_arm": args.factorial_arm,
                "checkpoint": str(args.checkpoint),
                "checkpoint_sha256": args.checkpoint_sha256,
                "num_samples": int(args.num_samples),
                "timesteps": FROZEN_TIMESTEPS,
                "diff_steps": FROZEN_DIFF_STEPS,
                "num_evals": FROZEN_NUM_EVALS,
                "effective_batch_size": 1,
                "seed_mode": "stateless_ordinal_v1",
                "distributed": distributed,
                "world_size": world_size,
                "retry": False,
                "replacement": False,
                "repair": False,
                "filter": False,
                "rerank": False,
            },
        )

    config, CSPDiffusion, Data, DataLoader = setup_crysllmgen_imports(
        args.crysllmgen_dir
    )
    device = dist_info["device"]
    model = CSPDiffusion(FROZEN_TIMESTEPS, "train").to(device)
    model.device = device
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(
        checkpoint["model"] if "model" in checkpoint else checkpoint
    )
    model.eval()

    assigned = [
        sample_idx
        for sample_idx in range(int(args.num_samples))
        if sample_idx % world_size == rank
    ]
    attempts: list[dict[str, Any]] = []
    refined_samples: list[dict[str, Any]] = []
    start = time.time()
    progress = tqdm(
        assigned,
        desc=f"H1-A2 {args.factorial_arm} refiner rank{rank}",
        disable=distributed and not is_main,
    )
    with torch.no_grad():
        for sample_idx in progress:
            body_attempt = dict(body_attempts[sample_idx])
            if body_attempt.get("body_status") != "complete":
                body_attempt["refiner_status"] = "not_started"
                body_attempt["refiner_complete"] = False
                attempts.append(body_attempt)
                continue

            refiner_seed = int(body_attempt["refiner_sampling_seed"])
            body_attempt.update(
                {
                    "attempt_status": "failed",
                    "earliest_failure_stage": "refiner",
                    "refiner_status": "failed",
                    "refiner_complete": False,
                }
            )
            try:
                random.seed(refiner_seed)
                np.random.seed(refiner_seed % (2**32))
                torch.manual_seed(refiner_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(refiner_seed)

                dataset = ProposalDataset([graph_map[sample_idx]], Data)
                loader = DataLoader(dataset, batch_size=1, shuffle=False)
                batch = next(iter(loader)).to(device)
                outputs, _ = model.sample(batch, diff_steps=FROZEN_DIFF_STEPS)
                lattices = outputs["lattices"].detach().cpu().unsqueeze(0)
                lengths, angles = lattices_to_params_shape(lattices)
                refined_samples.append(
                    {
                        "sample_idx": int(sample_idx),
                        "evaluation_order": int(body_attempt["evaluation_order"]),
                        "factorial_arm": args.factorial_arm,
                        "frac_coords": outputs["frac_coords"]
                        .detach()
                        .cpu()
                        .unsqueeze(0),
                        "num_atoms": outputs["num_atoms"]
                        .detach()
                        .cpu()
                        .unsqueeze(0),
                        "atom_types": outputs["atom_types"]
                        .detach()
                        .cpu()
                        .unsqueeze(0),
                        "lengths": lengths,
                        "angles": angles,
                    }
                )
                body_attempt.update(
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
                body_attempt.update(
                    {
                        "attempt_status": "failed",
                        "earliest_failure_stage": "refiner",
                        "failure_reason": type(exc).__name__,
                        "failure_message": str(exc),
                        "refiner_status": "failed",
                        "refiner_complete": False,
                    }
                )
            attempts.append(body_attempt)

    elapsed = time.time() - start
    attempt_path = _rank_path(
        args.output_dir,
        "refinement_attempts.jsonl",
        rank,
        distributed,
    )
    sample_path = _rank_path(
        args.output_dir,
        "refined_samples.pt",
        rank,
        distributed,
    )
    _write_jsonl(attempt_path, attempts)
    torch.save(refined_samples, sample_path)
    write_json(
        str(
            _rank_path(
                args.output_dir,
                "refinement_metrics.json",
                rank,
                distributed,
            )
        ),
        _refiner_metrics(
            attempts,
            elapsed=elapsed,
            rank=rank,
            world_size=world_size,
            distributed=distributed,
        ),
    )

    if distributed:
        dist.barrier()
        if is_main:
            _merge_refiner_outputs(
                args.output_dir,
                world_size=world_size,
                expected_count=int(args.num_samples),
                factorial_arm=args.factorial_arm,
                elapsed=elapsed,
            )
        dist.barrier()
        dist.destroy_process_group()
    else:
        ordered = ordered_single_arm_attempts(
            attempts,
            expected_count=int(args.num_samples),
            expected_factorial_arm=args.factorial_arm,
        )
        _write_jsonl(args.output_dir / "refinement_attempts.jsonl", list(ordered))
        output_file = _write_final_refined_payload(
            args.output_dir,
            refined_samples,
            elapsed=elapsed,
        )
        metrics = _refiner_metrics(
            list(ordered),
            elapsed=elapsed,
            rank=0,
            world_size=1,
            distributed=False,
        )
        metrics["output_file"] = str(output_file)
        write_json(str(args.output_dir / "refinement_metrics.json"), metrics)


if __name__ == "__main__":
    main()
