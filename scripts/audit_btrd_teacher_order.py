#!/usr/bin/env python3
"""Audit BTRD proposal/refiner identity and ordered species preservation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def refined_sequences(payload: dict) -> dict[int, list[int]]:
    indices = torch.as_tensor(payload["sample_indices"]).reshape(-1).tolist()
    counts = torch.as_tensor(payload["num_atoms"])[0].reshape(-1).tolist()
    species = torch.as_tensor(payload["atom_types"])[0].reshape(-1).tolist()
    if len(indices) != len(counts) or len(indices) != len(set(int(x) for x in indices)):
        raise ValueError("refined sample index/count accounting is not one-to-one")
    result: dict[int, list[int]] = {}
    cursor = 0
    for raw_index, raw_count in zip(indices, counts, strict=True):
        index = int(raw_index)
        count = int(raw_count)
        if count <= 0 or cursor + count > len(species):
            raise ValueError("refined atom-axis accounting is invalid")
        result[index] = [int(value) for value in species[cursor : cursor + count]]
        cursor += count
    if cursor != len(species):
        raise ValueError("refined atom-axis has unconsumed entries")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposal-graphs", type=Path, required=True)
    parser.add_argument("--refined-pt", type=Path, required=True)
    parser.add_argument("--accounting-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-requested", type=int, default=6144)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)

    proposals_path = args.proposal_graphs.resolve()
    refined_path = args.refined_pt.resolve()
    accounting_path = args.accounting_jsonl.resolve()
    proposals = torch.load(proposals_path, map_location="cpu")
    accounting = read_jsonl(accounting_path)
    if len(accounting) != args.expected_requested:
        raise ValueError("BTRD requested accounting changed")
    indices = [int(row["btrd_index"]) for row in accounting]
    if indices != list(range(args.expected_requested)):
        raise ValueError("BTRD requested indices are not closed and ordered")

    proposal_species: dict[int, list[int]] = {}
    for graph in proposals:
        index = int(graph["btrd_index"])
        if index in proposal_species or not 0 <= index < args.expected_requested:
            raise ValueError("proposal index is duplicate or out of range")
        sequence = [int(value) for value in torch.as_tensor(graph["a_type"]).reshape(-1)]
        expected_count = int(torch.as_tensor(graph["n_atom"]).reshape(-1)[0])
        if expected_count != len(sequence):
            raise ValueError("proposal N/species accounting changed")
        proposal_species[index] = sequence

    parsed_indices = {
        int(row["btrd_index"]) for row in accounting if row.get("parsed") is True
    }
    if parsed_indices != set(proposal_species):
        raise ValueError("parsed accounting does not match proposal graph indices")

    refined = refined_sequences(torch.load(refined_path, map_location="cpu"))
    if not set(refined).issubset(proposal_species):
        raise ValueError("refiner emitted an unknown proposal index")
    mismatches = [
        index
        for index, sequence in refined.items()
        if sequence != proposal_species[index]
    ]
    if mismatches:
        raise ValueError(f"refiner changed ordered species at indices {mismatches[:20]}")

    missing_refiner = sorted(set(proposal_species) - set(refined))
    body_failures = sorted(set(range(args.expected_requested)) - set(proposal_species))
    report = {
        "schema": "btrd_teacher_order_audit_v1",
        "status": "pass",
        "requested": args.expected_requested,
        "parsed_proposals": len(proposal_species),
        "refined_targets": len(refined),
        "body_failures": len(body_failures),
        "refiner_missing": len(missing_refiner),
        "ordered_species_mismatches": 0,
        "proposal_indices_unique_in_range": True,
        "refined_indices_unique_in_range": True,
        "inputs": {
            "proposal_graphs_sha256": sha256_file(proposals_path),
            "refined_sha256": sha256_file(refined_path),
            "accounting_sha256": sha256_file(accounting_path),
        },
        "fallback_policy": (
            "body/refiner missing rows may use registered MP20 anchors; "
            "identity corruption fails closed"
        ),
    }
    output.mkdir(parents=True)
    (output / "BTRD_TEACHER_ORDER_AUDIT.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "_SUCCESS").touch()
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
