#!/usr/bin/env python3
"""Project one frozen R03 body realization into four refiner processes.

Only process-label metadata changes.  Body text, proposal tensors, ordinals,
and the frozen body/refiner seed vectors remain identical across processes.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from protocol import (
    DENOMINATOR,
    REPEATS,
    canonical_sha256,
    ordered_rows,
    paired_seed,
    read_jsonl,
    sha256_file,
    write_json_exclusive,
    write_jsonl_exclusive,
)


def _scientific_digest(value: Any, digest: hashlib._Hash) -> None:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(b"tensor\0")
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(array.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(array.tobytes(order="C"))
        return
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest.update(b"ndarray\0")
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(array.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(array.tobytes(order="C"))
        return
    if isinstance(value, Mapping):
        digest.update(b"mapping\0")
        for key in sorted(value, key=lambda item: str(item)):
            if str(key) == "repeat":
                continue
            _scientific_digest(str(key), digest)
            _scientific_digest(value[key], digest)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        digest.update(b"sequence\0")
        for item in value:
            _scientific_digest(item, digest)
        return
    digest.update(type(value).__name__.encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(value, ensure_ascii=True, sort_keys=True).encode("utf-8"))
    digest.update(b"\0")


def scientific_sha256(value: Any) -> str:
    digest = hashlib.sha256()
    _scientific_digest(value, digest)
    return digest.hexdigest()


def _write_torch_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--body-attempts", type=Path, required=True)
    parser.add_argument("--proposal-graphs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    attempts = ordered_rows(
        read_jsonl(args.body_attempts.resolve()), ordinal_field="ordinal"
    )
    graphs = torch.load(args.proposal_graphs.resolve(), map_location="cpu")
    if not isinstance(graphs, list):
        raise TypeError("proposal graph payload is not a list")
    success_ordinals = {
        ordinal for ordinal, row in enumerate(attempts) if row.get("status") == "succeeded"
    }
    graph_ordinals: set[int] = set()
    for record in graphs:
        if not isinstance(record, dict) or not isinstance(record.get("graph"), dict):
            raise TypeError("proposal graph record is malformed")
        ordinal = int(record.get("ordinal", -1))
        metadata = record["graph"].get("h1_plan1200_prepost_metadata")
        if (
            ordinal in graph_ordinals
            or not isinstance(metadata, Mapping)
            or int(record.get("repeat", -1)) != 0
            or int(metadata.get("repeat", -1)) != 0
            or str(record.get("arm")) != "R03"
            or str(metadata.get("arm")) != "R03"
            or int(metadata.get("ordinal", -1)) != ordinal
            or str(metadata.get("schedule_arm")) != "D2_SAFE_AXIS"
        ):
            raise ValueError("base proposal graph identity changed")
        graph_ordinals.add(ordinal)
    if graph_ordinals != success_ordinals:
        raise ValueError("proposal graphs do not exactly match body successes")
    if any(
        str(row.get("arm")) != "R03"
        or int(row.get("repeat", -1)) != 0
        or int(row.get("body_noise_seed", -1)) != paired_seed(0, ordinal, "body")
        or int(row.get("refiner_noise_seed", -1))
        != paired_seed(0, ordinal, "refiner")
        for ordinal, row in enumerate(attempts)
    ):
        raise ValueError("base body/seed identity changed")

    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    process_reports: list[dict[str, Any]] = []
    scientific_attempt_hashes: list[str] = []
    scientific_graph_hashes: list[str] = []
    for repeat in REPEATS:
        process_dir = output / f"repeat_{repeat}"
        process_dir.mkdir()
        process_attempts: list[dict[str, Any]] = []
        for ordinal, source_row in enumerate(attempts):
            row = dict(source_row)
            row["repeat"] = repeat
            if (
                int(row["body_noise_seed"]) != paired_seed(repeat, ordinal, "body")
                or int(row["refiner_noise_seed"])
                != paired_seed(repeat, ordinal, "refiner")
            ):
                raise ValueError("projected seed vector changed")
            process_attempts.append(row)
        process_graphs = copy.deepcopy(graphs)
        for record in process_graphs:
            record["repeat"] = repeat
            record["graph"]["h1_plan1200_prepost_metadata"]["repeat"] = repeat

        attempts_path = process_dir / "body_attempts.jsonl"
        graphs_path = process_dir / "proposal_graphs.pt"
        write_jsonl_exclusive(attempts_path, process_attempts)
        _write_torch_exclusive(graphs_path, process_graphs)
        attempt_science = canonical_sha256(
            [{key: value for key, value in row.items() if key != "repeat"} for row in process_attempts]
        )
        graph_science = scientific_sha256(process_graphs)
        scientific_attempt_hashes.append(attempt_science)
        scientific_graph_hashes.append(graph_science)
        process_reports.append(
            {
                "repeat": repeat,
                "attempts": DENOMINATOR,
                "body_successes": len(success_ordinals),
                "body_attempts": str(attempts_path),
                "body_attempts_sha256": sha256_file(attempts_path),
                "body_attempts_scientific_sha256": attempt_science,
                "proposal_graphs": str(graphs_path),
                "proposal_graphs_sha256": sha256_file(graphs_path),
                "proposal_graphs_scientific_sha256": graph_science,
                "body_seed_vector_sha256": canonical_sha256(
                    [int(row["body_noise_seed"]) for row in process_attempts]
                ),
                "refiner_seed_vector_sha256": canonical_sha256(
                    [int(row["refiner_noise_seed"]) for row in process_attempts]
                ),
            }
        )
    if len(set(scientific_attempt_hashes)) != 1 or len(set(scientific_graph_hashes)) != 1:
        raise ValueError("projected process inputs are not scientifically identical")
    if len({row["body_seed_vector_sha256"] for row in process_reports}) != 1:
        raise ValueError("body seed vectors differ across processes")
    if len({row["refiner_seed_vector_sha256"] for row in process_reports}) != 1:
        raise ValueError("refiner seed vectors differ across processes")

    report = {
        "schema": "h1a2_historical_topology_process_projection_v1",
        "status": "complete",
        "attempts": DENOMINATOR,
        "body_process_realizations": 1,
        "refiner_process_realizations": 4,
        "identical_body_and_proposal_graphs": True,
        "identical_body_seed_vector": True,
        "identical_refiner_seed_vector": True,
        "base_body_attempts_sha256": sha256_file(args.body_attempts.resolve()),
        "base_proposal_graphs_sha256": sha256_file(args.proposal_graphs.resolve()),
        "body_attempts_scientific_sha256": scientific_attempt_hashes[0],
        "proposal_graphs_scientific_sha256": scientific_graph_hashes[0],
        "processes": process_reports,
        "retry_replacement_repair_filter_rerank": False,
    }
    report["report_payload_sha256"] = canonical_sha256(report)
    write_json_exclusive(output / "projection_report.json", report)
    (output / "projection_SUCCESS").touch(exist_ok=False)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
