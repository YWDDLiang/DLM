#!/usr/bin/env python3
"""Build BTRD SFT rows from tau200 model494 outputs and MP20 anchors."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

import torch

from crystal_dlm.dynamic_crystal import Z_TO_SYMBOL, arrays_to_dynamic_answer


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _first_eval(value: torch.Tensor, *, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value).detach().cpu()
    if tensor.ndim == 0 or int(tensor.shape[0]) != 1:
        raise ValueError(f"{name} must have one frozen refiner evaluation")
    return tensor[0]


def refined_geometry_by_index(payload):
    indices = torch.as_tensor(payload["sample_indices"]).detach().cpu().reshape(-1)
    num_atoms = _first_eval(payload["num_atoms"], name="num_atoms").reshape(-1)
    lengths = _first_eval(payload["lengths"], name="lengths")
    angles = _first_eval(payload["angles"], name="angles")
    atom_types = _first_eval(payload["atom_types"], name="atom_types").reshape(-1)
    frac = _first_eval(payload["frac_coords"], name="frac_coords").reshape(-1, 3)
    count = len(indices)
    if not (len(num_atoms) == len(lengths) == len(angles) == count):
        raise ValueError("refiner graph-axis accounting changed")
    if int(num_atoms.sum()) != len(atom_types) or len(atom_types) != len(frac):
        raise ValueError("refiner atom-axis accounting changed")
    result = {}
    cursor = 0
    for position, raw_index in enumerate(indices.tolist()):
        index = int(raw_index)
        if index in result:
            raise ValueError("duplicate BTRD refined sample index")
        atoms = int(num_atoms[position])
        species = [Z_TO_SYMBOL[int(value)] for value in atom_types[cursor : cursor + atoms]]
        coordinates = frac[cursor : cursor + atoms].tolist()
        cursor += atoms
        result[index] = {
            "lengths": lengths[position].tolist(),
            "angles": angles[position].tolist(),
            "species": species,
            "frac_coords": coordinates,
        }
    return result


def expected_counts(plan_state):
    return Counter(
        {
            str(element): int(count)
            for element, count in zip(
                plan_state["elements"], plan_state["counts"], strict=True
            )
        }
    )


def build_rows(selected, refined):
    output = []
    audit = Counter()
    for source in selected:
        row = dict(source)
        requested_mode = str(row["btrd_target_mode"])
        index = int(row["btrd_index"])
        effective = "mp20_anchor"
        reason = "registered_anchor"
        if requested_mode == "model494_tau200":
            geometry = refined.get(index)
            if geometry is None:
                reason = "tau200_missing_fallback_anchor"
            elif Counter(geometry["species"]) != expected_counts(row["plan_state"]):
                reason = "tau200_atom_multiset_mismatch_fallback_anchor"
            else:
                answer, diagnostics = arrays_to_dynamic_answer(
                    geometry["lengths"],
                    geometry["angles"],
                    geometry["species"],
                    geometry["frac_coords"],
                )
                row["answer"] = answer
                row["answer_sha256"] = sha256_text(answer)
                row["btrd_encode_diagnostics"] = diagnostics.to_dict()
                effective = "model494_tau200"
                reason = "tau200_teacher"
        row["btrd_requested_target_mode"] = requested_mode
        row["btrd_effective_target_mode"] = effective
        row["btrd_target_reason"] = reason
        row["btrd_teacher_steps"] = 200 if effective == "model494_tau200" else 0
        row["btrd_energy_label_used"] = False
        row["sample_weight"] = 1.0
        audit[reason] += 1
        output.append(row)
    return output, dict(sorted(audit.items()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset-jsonl", type=Path, required=True)
    parser.add_argument("--refined-pt", type=Path, required=True)
    parser.add_argument("--source-data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    subset = args.subset_jsonl.resolve()
    refined_path = args.refined_pt.resolve()
    source_data = args.source_data_dir.resolve()
    selected = read_jsonl(subset)
    if len(selected) != 8192:
        raise ValueError("BTRD selected denominator changed")
    refined = refined_geometry_by_index(torch.load(refined_path, map_location="cpu"))
    rows, audit = build_rows(selected, refined)
    output.mkdir(parents=True)
    train_path = output / "train.jsonl"
    train_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    for source in source_data.iterdir():
        if not source.is_file() or source.name in {
            "train.jsonl",
            "manifest.json",
            "SHA256SUMS",
            "_SUCCESS",
        }:
            continue
        shutil.copy2(source, output / source.name)
    if not (output / "val.jsonl").is_file() or not (output / "vocab_tokens.txt").is_file():
        raise FileNotFoundError("source DLM validation/vocabulary assets are incomplete")
    manifest = {
        "schema": "btrd_sft_data_v1",
        "status": "complete",
        "train_rows": len(rows),
        "validation_rows": len(read_jsonl(output / "val.jsonl")),
        "requested_tau200_rows": sum(
            row["btrd_requested_target_mode"] == "model494_tau200" for row in rows
        ),
        "effective_tau200_rows": sum(
            row["btrd_effective_target_mode"] == "model494_tau200" for row in rows
        ),
        "anchor_or_fallback_rows": sum(
            row["btrd_effective_target_mode"] == "mp20_anchor" for row in rows
        ),
        "audit": audit,
        "energy_labels_used": False,
        "row_selection_by_teacher_outcome": False,
        "subset_sha256": sha256_file(subset),
        "refined_sha256": sha256_file(refined_path),
        "train_sha256": sha256_file(train_path),
        "validation_sha256": sha256_file(output / "val.jsonl"),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    files = sorted(path for path in output.iterdir() if path.is_file())
    (output / "SHA256SUMS").write_text(
        "".join(f"{sha256_file(path)}  {path.name}\n" for path in files),
        encoding="utf-8",
    )
    (output / "_SUCCESS").touch()
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
