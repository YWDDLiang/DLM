#!/usr/bin/env python3
"""Label SPAD-E legal backfill actions with terminal model494/CHGNet energy."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from pymatgen.core import Lattice, Structure

from crystal_dlm.dynamic_crystal import arrays_to_structure, parse_dynamic_answer
from crystal_dlm.spad_program import coordinate_positions


SCHEMA = "spad_energy_group_v1"


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError(path)
                yield value


def prediction(model: Any, structures: list[Structure]) -> list[dict[str, Any] | None]:
    output: list[dict[str, Any] | None] = []
    for start in range(0, len(structures), 16):
        chunk = structures[start : start + 16]
        try:
            values = model.predict_structure(chunk, task="efsm", batch_size=16)
            if isinstance(values, dict):
                values = [values]
            output.extend(values)
        except Exception:
            for structure in chunk:
                try:
                    output.append(model.predict_structure(structure, task="efsm"))
                except Exception:
                    output.append(None)
    if len(output) != len(structures):
        raise RuntimeError("CHGNet prediction count changed")
    return output


def refined_structures(metrics_path: Path) -> dict[int, Structure]:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    payload = torch.load(Path(metrics["output_file"]), map_location="cpu")
    sample_indices = torch.as_tensor(payload["sample_indices"]).reshape(-1).tolist()
    num_atoms = torch.as_tensor(payload["num_atoms"])
    atom_types = torch.as_tensor(payload["atom_types"])
    frac_coords = torch.as_tensor(payload["frac_coords"])
    lengths = torch.as_tensor(payload["lengths"])
    angles = torch.as_tensor(payload["angles"])
    if num_atoms.ndim != 2 or int(num_atoms.shape[0]) != 1:
        raise ValueError("SPAD-E expects one model494 evaluation")
    if len(sample_indices) != int(num_atoms.shape[1]):
        raise ValueError("refined sample accounting changed")
    output: dict[int, Structure] = {}
    offset = 0
    for column, sample_idx in enumerate(sample_indices):
        count = int(num_atoms[0, column].item())
        atoms = atom_types[0, offset : offset + count].tolist()
        coords = frac_coords[0, offset : offset + count].tolist()
        lens = lengths[0, column].tolist()
        angs = angles[0, column].tolist()
        offset += count
        try:
            lattice = Lattice.from_parameters(
                *[float(value) for value in lens],
                *[float(value) for value in angs],
            )
            structure = Structure(
                lattice,
                [int(value) for value in atoms],
                coords,
                coords_are_cartesian=False,
                to_unit_cell=True,
            )
            if len(structure) != count or not math.isfinite(float(structure.volume)):
                raise ValueError("invalid model494 endpoint")
            output[int(sample_idx)] = structure
        except Exception:
            continue
    if offset != int(atom_types.shape[1]):
        raise ValueError("refined tensor has trailing atoms")
    return output


def finite_prediction(value: dict[str, Any] | None) -> tuple[float | None, np.ndarray | None]:
    if value is None:
        return None, None
    energy = float(np.asarray(value["e"]).reshape(()))
    force = np.asarray(value["f"], dtype=float)
    if not math.isfinite(energy) or force.ndim != 2 or force.shape[1] != 3:
        return None, None
    if not np.isfinite(force).all():
        return None, None
    return energy, force


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-actions", type=Path, required=True)
    parser.add_argument("--refinement-metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    actions = list(iter_jsonl(args.candidate_actions))
    if len(actions) != 8192 or {int(row["sample_idx"]) for row in actions} != set(
        range(8192)
    ):
        raise ValueError("SPAD-E action denominator changed")
    endpoints = refined_structures(args.refinement_metrics)

    valid_rows = [row for row in actions if row.get("valid_action") is True]
    raw_structures = [
        arrays_to_structure(parse_dynamic_answer(str(row["answer"]), strict=True))
        for row in valid_rows
    ]
    endpoint_rows = [row for row in valid_rows if int(row["sample_idx"]) in endpoints]
    endpoint_values = [endpoints[int(row["sample_idx"])] for row in endpoint_rows]

    from chgnet.model.model import CHGNet

    model = CHGNet.load(use_device=args.device, check_cuda_mem=True, verbose=False)
    raw_predictions = prediction(model, raw_structures)
    endpoint_predictions = prediction(model, endpoint_values)
    raw_by_idx = {
        int(row["sample_idx"]): finite_prediction(value)
        for row, value in zip(valid_rows, raw_predictions, strict=True)
    }
    endpoint_by_idx = {
        int(row["sample_idx"]): finite_prediction(value)[0]
        for row, value in zip(endpoint_rows, endpoint_predictions, strict=True)
    }

    groups: list[dict[str, Any]] = []
    labelled_actions: list[dict[str, Any]] = []
    for group_idx in range(2048):
        candidates = [actions[group_idx * 4 + index] for index in range(4)]
        if [int(row["candidate_idx"]) for row in candidates] != list(range(4)):
            raise ValueError("candidate order changed")
        source_answer = candidates[0].get("source_answer")
        slot = candidates[0].get("backfill_slot")
        active = None if slot is None else coordinate_positions(int(slot))
        source_tokens = (
            None
            if not isinstance(source_answer, str)
            else parse_dynamic_answer(source_answer, strict=True)["tokens"]
        )
        group_candidates = []
        for row in candidates:
            flat_idx = int(row["sample_idx"])
            energy_raw, force = raw_by_idx.get(flat_idx, (None, None))
            terminal_energy = endpoint_by_idx.get(flat_idx)
            selected_force = None
            action_triplet = None
            differing: list[int] = []
            if row.get("valid_action") is True:
                tokens = parse_dynamic_answer(str(row["answer"]), strict=True)["tokens"]
                if source_tokens is None or active is None:
                    raise RuntimeError("valid action lacks common-state metadata")
                differing = [
                    index
                    for index, (left, right) in enumerate(
                        zip(source_tokens, tokens, strict=True)
                    )
                    if left != right
                ]
                if not set(differing) <= set(active):
                    raise RuntimeError("action changed tokens outside backfill XYZ")
                action_triplet = [str(tokens[position]) for position in active]
                if force is not None and int(slot) < int(force.shape[0]):
                    selected_force = [float(value) for value in force[int(slot)]]
            labelled = {
                **row,
                "raw_chgnet_energy_per_atom": energy_raw,
                "selected_site_raw_force_eV_per_A": selected_force,
                "model494_endpoint_known": flat_idx in endpoints,
                "terminal_chgnet_energy_per_atom": terminal_energy,
                "energy_known": terminal_energy is not None,
                "action_triplet_tokens": action_triplet,
                "active_positions": None if active is None else list(active),
                "differing_positions": differing,
            }
            labelled_actions.append(labelled)
            group_candidates.append(
                {
                    "candidate_idx": int(row["candidate_idx"]),
                    "sample_idx": flat_idx,
                    "mandatory_noop": bool(row["mandatory_noop"]),
                    "valid_action": bool(row["valid_action"]),
                    "answer": row.get("answer"),
                    "action_triplet_tokens": action_triplet,
                    "differing_positions": differing,
                    "terminal_energy_per_atom": terminal_energy,
                    "energy_known": terminal_energy is not None,
                    "raw_energy_per_atom": energy_raw,
                    "selected_site_raw_force_eV_per_A": selected_force,
                }
            )
        legal_known = sum(
            item["valid_action"] and item["energy_known"] for item in group_candidates
        )
        trainable = (
            group_candidates[0]["valid_action"]
            and group_candidates[0]["energy_known"]
            and legal_known >= 2
            and active is not None
        )
        groups.append(
            {
                "schema": SCHEMA,
                "group_idx": group_idx,
                "prompt": candidates[0].get("prompt"),
                "plan_state": candidates[0].get("plan_state"),
                "species_program": candidates[0].get("species_program"),
                "source_answer": source_answer,
                "backfill_slot": slot,
                "active_positions": None if active is None else list(active),
                "K": 4,
                "mandatory_noop": True,
                "validity_before_energy": True,
                "trainable": bool(trainable),
                "legal_known_actions": int(legal_known),
                "candidates": group_candidates,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name, rows in (("labelled_actions.jsonl", labelled_actions), ("groups.jsonl", groups)):
        with (args.output_dir / name).open("x", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
    report = {
        "schema": SCHEMA,
        "groups": 2048,
        "candidates": 8192,
        "valid_actions": sum(row.get("valid_action") is True for row in actions),
        "model494_endpoints": len(endpoints),
        "terminal_energy_known": len(endpoint_by_idx),
        "raw_energy_known": len(raw_by_idx),
        "trainable_groups": sum(row["trainable"] is True for row in groups),
        "untrainable_groups": sum(row["trainable"] is not True for row in groups),
        "legal_known_histogram": dict(
            Counter(str(row["legal_known_actions"]) for row in groups)
        ),
        "invalid_actions_zero_energy_support": True,
        "unknown_energy_preserved": True,
        "force_role": "diagnostic_only",
        "official_or_test_outcomes_read": False,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "_SUCCESS").touch()
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
