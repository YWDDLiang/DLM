#!/usr/bin/env python3
"""Freeze the first parseable SPAD bodies before any outcome evaluation.

This is the denominator constructor for the preregistered paper-scale endpoint.
It scans a fixed Planner/DLM stream in source order, keeps the first requested
number of independently parseable CIFs, and reindexes only those rows and their
proposal graphs.  No energy, hull, novelty, or S.U.N. value is accepted as an
input or consulted by the selection rule.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _indexed(
    rows: Sequence[Mapping[str, Any]], *, name: str
) -> dict[int, dict[str, Any]]:
    indexed: dict[int, dict[str, Any]] = {}
    for row in rows:
        idx = int(row["sample_idx"])
        if idx in indexed:
            raise ValueError(f"duplicate {name} sample_idx={idx}")
        indexed[idx] = dict(row)
    return indexed


def select_first_parseable(
    body_rows: Sequence[Mapping[str, Any]],
    planner_ledger: Sequence[Mapping[str, Any]],
    proposal_graphs: Sequence[Mapping[str, Any]],
    *,
    target: int,
    parse_cif: Callable[[str], None],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Return selected bodies/graphs, accounting rows, and a frozen report."""

    if target <= 0:
        raise ValueError("target must be positive")
    ledger = _indexed(planner_ledger, name="Planner ledger")
    if set(ledger) != set(range(len(planner_ledger))):
        raise ValueError("Planner ledger must cover contiguous source ordinals")
    bodies = _indexed(body_rows, name="body")
    if not set(bodies) <= set(ledger):
        raise ValueError("body rows contain out-of-ledger sample indices")

    parsed_body_indices = [
        int(row["sample_idx"]) for row in body_rows if row.get("parsed") is True
    ]
    if len(parsed_body_indices) != len(proposal_graphs):
        raise ValueError("parsed body/proposal graph count mismatch")
    graph_by_source: dict[int, dict[str, Any]] = {}
    for source_idx, graph in zip(parsed_body_indices, proposal_graphs, strict=True):
        if source_idx in graph_by_source:
            raise ValueError(f"duplicate proposal graph source index={source_idx}")
        graph_by_source[source_idx] = dict(graph)

    selected_rows: list[dict[str, Any]] = []
    selected_graphs: list[dict[str, Any]] = []
    accounting: list[dict[str, Any]] = []
    cutoff: int | None = None

    for source_idx in range(len(planner_ledger)):
        planner = ledger[source_idx]
        body = bodies.get(source_idx)
        record: dict[str, Any] = {
            "source_attempt_idx": source_idx,
            "planner_valid": planner.get("planner_valid") is True,
            "selected_index": None,
            "selection_basis": "parser_only",
        }
        if cutoff is not None:
            record["status"] = "not_examined_after_target"
            accounting.append(record)
            continue

        if planner.get("planner_valid") is not True:
            if body is not None:
                raise ValueError(
                    f"invalid Planner row unexpectedly has body sample_idx={source_idx}"
                )
            record["status"] = "planner_invalid"
            accounting.append(record)
            continue
        if body is None:
            raise ValueError(f"valid Planner row misses body sample_idx={source_idx}")
        if body.get("parsed") is not True or not body.get("cif"):
            record["status"] = "body_parser_failure"
            accounting.append(record)
            continue
        graph = graph_by_source.get(source_idx)
        if graph is None:
            raise ValueError(f"parsed body misses proposal graph sample_idx={source_idx}")
        try:
            parse_cif(str(body["cif"]))
        except Exception as exc:  # parser failure is the only permitted exclusion
            record["status"] = "independent_cif_parser_failure"
            record["parser_error_type"] = type(exc).__name__
            accounting.append(record)
            continue

        accepted_idx = len(selected_rows)
        selected = deepcopy(body)
        selected["source_attempt_idx"] = source_idx
        selected["accepted_index"] = accepted_idx
        selected["sample_idx"] = accepted_idx
        selected["paper_scale_parser_only_selection"] = True
        selected_graph = deepcopy(graph)
        selected_graph["source_attempt_idx"] = source_idx
        selected_graph["accepted_index"] = accepted_idx
        selected_graph["sample_idx"] = accepted_idx
        selected_rows.append(selected)
        selected_graphs.append(selected_graph)
        record["status"] = "selected_parseable_cif"
        record["selected_index"] = accepted_idx
        accounting.append(record)
        if len(selected_rows) == target:
            cutoff = source_idx

    if len(selected_rows) != target or cutoff is None:
        raise ValueError(
            f"fixed stream contains only {len(selected_rows)} parseable CIFs; target={target}"
        )
    examined = cutoff + 1
    before_cutoff = accounting[:examined]
    report = {
        "schema": "spad_first_parseable_body_v1",
        "source_requests_available": len(planner_ledger),
        "source_requests_examined": examined,
        "unexamined_after_target": len(planner_ledger) - examined,
        "target_parseable_cifs": target,
        "selected_parseable_cifs": len(selected_rows),
        "discarded_before_target": examined - target,
        "planner_invalid_before_target": sum(
            row["status"] == "planner_invalid" for row in before_cutoff
        ),
        "body_parser_failures_before_target": sum(
            row["status"] == "body_parser_failure" for row in before_cutoff
        ),
        "independent_parser_failures_before_target": sum(
            row["status"] == "independent_cif_parser_failure"
            for row in before_cutoff
        ),
        "selection_rule": "first_target_independently_parseable_cifs_in_fixed_source_order",
        "selection_basis": "CIF_parser_only_before_any_outcome_evaluation",
        "energy_hull_novelty_sun_read": False,
        "outcome_based_replacement_or_reranking": False,
        "conditional_valid_denominator": target,
    }
    return selected_rows, selected_graphs, accounting, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--body-dir", type=Path, required=True)
    parser.add_argument("--planner-ledger", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target", type=int, default=1000)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    import torch  # noqa: PLC0415
    from pymatgen.core import Structure  # noqa: PLC0415

    graphs = torch.load(args.body_dir / "proposal_graphs.pt", map_location="cpu")

    def parse_cif(value: str) -> None:
        Structure.from_str(value, fmt="cif")

    rows, selected_graphs, accounting, report = select_first_parseable(
        read_jsonl(args.body_dir / "raw_generations.jsonl"),
        read_jsonl(args.planner_ledger),
        graphs,
        target=int(args.target),
        parse_cif=parse_cif,
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "raw_generations.jsonl").open(
        "x", encoding="utf-8", newline="\n"
    ) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    with (args.output_dir / "all_requested_accounting.jsonl").open(
        "x", encoding="utf-8", newline="\n"
    ) as handle:
        for row in accounting:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    torch.save(selected_graphs, args.output_dir / "proposal_graphs.pt")
    (args.output_dir / "manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "_SUCCESS").touch()
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
