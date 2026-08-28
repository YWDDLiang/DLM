#!/usr/bin/env python3
"""Assemble all Direct-valid CHGNet-labelled same-Plan structures for critic audit."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def ordinal_from_attempt(row: dict[str, Any]) -> int:
    if row.get("ordinal") is not None:
        return int(row["ordinal"])
    return int(str(row["attempt_id"]).rsplit("-", 1)[-1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-cohort", type=Path, required=True)
    parser.add_argument("--body-run", type=Path, required=True)
    parser.add_argument("--eval-run", type=Path, required=True)
    parser.add_argument("--extra-body-run", type=Path, required=True)
    parser.add_argument("--extra-eval-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    import torch
    from ase.io import write as ase_write
    from pymatgen.core import Lattice, Structure
    from pymatgen.io.ase import AseAtomsAdaptor

    cohort_rows = read_jsonl(args.plan_cohort)
    cohort = {int(row["sample_idx"]): row for row in cohort_rows}
    if len(cohort) != 256 or set(cohort) != set(range(256)):
        raise ValueError("critic cohort must preserve sample_idx 0..255")

    atoms_rows: list[Any] = []
    metadata: list[dict[str, Any]] = []
    stream_reports: list[dict[str, int]] = []
    plan_counts: Counter[int] = Counter()
    split_counts: Counter[str] = Counter()
    for stream in range(8):
        body_root = args.body_run if stream < 4 else args.extra_body_run
        eval_root = args.eval_run if stream < 4 else args.extra_eval_run
        body = {
            int(row["sample_idx"]): row
            for row in read_jsonl(body_root / f"stream{stream}/body/raw_generations.jsonl")
        }
        full_root = eval_root / f"stream{stream}/evaluation/full_reconstructed"
        labels = {int(row["ordinal"]): row for row in read_jsonl(full_root / "attempt_labels_preofficial.jsonl")}
        direct = {
            ordinal_from_attempt(row): row
            for row in read_jsonl(eval_root / f"stream{stream}/evaluation/direct/attempt_metrics.jsonl")
        }
        input_manifest = json.loads((full_root / "input_manifest.json").read_text(encoding="utf-8"))
        attempt_manifest = {
            int(row["generation_ordinal"]): row for row in input_manifest["attempt_records"]
        }
        payload = torch.load(full_root / "all_attempts.pt", map_location="cpu")
        arrays = {key: value[0] for key, value in payload.items()}
        num_atoms = [int(value) for value in arrays["num_atoms"].tolist()]
        offsets: list[int] = []
        cursor = 0
        for count in num_atoms:
            offsets.append(cursor)
            cursor += int(count)
        if cursor != int(arrays["frac_coords"].shape[0]) or len(num_atoms) != 256:
            raise ValueError(f"stream{stream} tensor ledger changed")
        if set(body) != set(range(256)) or set(labels) != set(range(256)) or set(direct) != set(range(256)):
            raise ValueError(f"stream{stream} ordinal coverage changed")

        eligible = 0
        for sample_idx in range(256):
            source = body[sample_idx]
            label = labels[sample_idx]
            metric = direct[sample_idx]
            manifest_row = attempt_manifest[sample_idx]
            plan_row = cohort[sample_idx]
            if canonical_sha256(source.get("plan_state")) != str(plan_row["source_plan_state_sha256"]):
                raise ValueError(f"stream{stream} Plan identity changed at {sample_idx}")
            energy = label.get("chgnet_energy_per_atom")
            if not (
                source.get("parsed") is True
                and label.get("reconstructed") is True
                and label.get("chgnet_relaxation_known") is True
                and energy is not None
                and metric.get("valid") is True
                and manifest_row.get("reconstructed_index") is not None
            ):
                continue
            ordinal = int(manifest_row["pt_record_ordinal"])
            count = num_atoms[ordinal]
            if count <= 0:
                raise ValueError("eligible structure has nonpositive atom count")
            start = offsets[ordinal]
            stop = start + count
            lattice = Lattice.from_parameters(
                *[float(value) for value in arrays["lengths"][ordinal].tolist()],
                *[float(value) for value in arrays["angles"][ordinal].tolist()],
            )
            structure = Structure(
                lattice,
                [int(value) for value in arrays["atom_types"][start:stop].tolist()],
                arrays["frac_coords"][start:stop].tolist(),
                coords_are_cartesian=False,
                to_unit_cell=True,
            )
            atoms = AseAtomsAdaptor.get_atoms(structure)
            record_index = len(metadata)
            atoms.info.update(
                {
                    "record_index": record_index,
                    "stream": stream,
                    "sample_idx": sample_idx,
                    "pair_split": str(plan_row["pair_split"]),
                }
            )
            atoms_rows.append(atoms)
            plan = plan_row["plan_state"]
            metadata.append(
                {
                    "schema": "h1a2_noisy_critic_feasibility_row_v1",
                    "record_index": record_index,
                    "stream": stream,
                    "sample_idx": sample_idx,
                    "pair_split": str(plan_row["pair_split"]),
                    "source_plan_state_sha256": plan_row["source_plan_state_sha256"],
                    "formula": str(plan.get("formula")),
                    "family": str(plan.get("anion_framework") or "other"),
                    "arity": len(plan.get("elements") or ()),
                    "N": int(plan.get("N") or 0),
                    "charge_bucket": str(plan.get("charge_bucket") or "unknown"),
                    "chgnet_relaxed_energy_per_atom": float(energy),
                    "structure_sha256": str(manifest_row["structure_sha256"]),
                    "body_text_sha256": hashlib.sha256(str(source.get("text") or "").encode("utf-8")).hexdigest(),
                }
            )
            eligible += 1
            plan_counts[sample_idx] += 1
            split_counts[str(plan_row["pair_split"])] += 1
        stream_reports.append({"stream": stream, "eligible": eligible})

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    extxyz = output / "critic_feasibility.extxyz"
    ase_write(extxyz, atoms_rows, format="extxyz")
    with (output / "metadata.jsonl").open("x", encoding="utf-8") as handle:
        for row in metadata:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {
        "schema": "h1a2_noisy_critic_feasibility_manifest_v1",
        "plan_cohort": str(args.plan_cohort.resolve()),
        "body_runs": [str(args.body_run.resolve()), str(args.extra_body_run.resolve())],
        "eval_runs": [str(args.eval_run.resolve()), str(args.extra_eval_run.resolve())],
        "streams": 8,
        "eligible_structures": len(metadata),
        "eligible_by_split": dict(sorted(split_counts.items())),
        "plans_with_at_least_two": sum(value >= 2 for value in plan_counts.values()),
        "plans_with_at_least_three": sum(value >= 3 for value in plan_counts.values()),
        "stream_reports": stream_reports,
        "structure_stage": "model494_tau800_output_before common CHGNet relaxation",
        "label": "compatible CHGNet energy after common CHGNet relaxation",
        "formula_disjoint_from_l6_l7": True,
        "unknown_policy": "missing; never high energy",
        "novelty_used": False,
        "extxyz_sha256": hashlib.sha256(extxyz.read_bytes()).hexdigest(),
        "metadata_sha256": hashlib.sha256((output / "metadata.jsonl").read_bytes()).hexdigest(),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    stem = "NOISY_CRITIC_FEASIBILITY_DATA_MANIFEST"
    (output / f"{stem}.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output / f"{stem}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("stream", "eligible"))
        writer.writeheader()
        writer.writerows(stream_reports)
    lines = [
        "# Noisy-state critic feasibility data",
        "",
        f"- Eligible structures: `{len(metadata)}`",
        f"- Train/validation: `{dict(sorted(split_counts.items()))}`",
        f"- Plans with >=2 / >=3 eligible structures: `{sum(value >= 2 for value in plan_counts.values())}/{sum(value >= 3 for value in plan_counts.values())}`",
        f"- Stream counts: `{stream_reports}`",
        "- Structure stage: `model494 tau800 output before common CHGNet relaxation`",
        "- Label: `compatible CHGNet energy after common CHGNet relaxation`",
        "",
        "Unknown labels are missing, novelty is unused, and the formula-group split was frozen before generation.",
    ]
    (output / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output / "_SUCCESS").touch()
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
