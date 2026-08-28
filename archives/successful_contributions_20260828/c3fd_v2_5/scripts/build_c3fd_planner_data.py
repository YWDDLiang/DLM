#!/usr/bin/env python3
"""Build typed C³FD-v2 Planner supervision without formula BPE tokens."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crystal_dlm.ccfd_v2 import compile_plan_actions, replay_actions  # noqa: E402
from crystal_dlm.composition_pair_prior import (  # noqa: E402
    CompositionPairPrior,
    ValenceNode,
)
from crystal_dlm.species_physics import (  # noqa: E402
    feature_names,
    species_physics_matrix,
)


SOFT_FIELDS = (
    "anion_framework",
    "charge_bucket",
    "lattice_system",
    "spacegroup_bucket",
    "volume_per_atom_bin",
)
UNKNOWN_SOFT = "<UNKNOWN>"


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError(f"non-object row in {path}")
                yield value


def plan_from_row(row: Mapping[str, Any]) -> Mapping[str, Any] | None:
    value = row.get("plan_state") or row.get("r5_plan_state")
    return value if isinstance(value, Mapping) else None


def compile_row(row: Mapping[str, Any], row_index: int) -> dict[str, Any]:
    plan = plan_from_row(row)
    proposal_n = None
    if plan is not None and plan.get("N") not in (None, ""):
        try:
            proposal_n = int(plan["N"])
        except Exception:  # noqa: BLE001 - invalid proposal is audited downstream.
            proposal_n = None
    output: dict[str, Any] = {
        "source_row_idx": row.get("source_row_idx", row.get("row_idx", row_index)),
        "sample_weight": float(row.get("sample_weight", 1.0) or 1.0),
        "plan_state": None if plan is None else dict(plan),
        "composition_supervision": False,
        "certificate_class": "missing_plan_state",
        "compile_error": None,
        "N": proposal_n,
        "nodes": [],
        "counts": [],
        "soft_values": {
            field: UNKNOWN_SOFT if plan is None else str(plan.get(field) or UNKNOWN_SOFT)
            for field in SOFT_FIELDS
        },
    }
    if plan is None:
        return output
    try:
        actions, metadata = compile_plan_actions(plan)
        state = replay_actions(actions)
        certificate = state.certificate()
    except Exception as exc:  # noqa: BLE001 - failures remain in the denominator.
        output["certificate_class"] = "uncompiled"
        output["compile_error"] = f"{type(exc).__name__}:{str(exc)}"
        return output
    nodes = [ValenceNode.from_token(token) for token in state.tokens]
    output.update(
        {
            "composition_supervision": certificate.benchmark_compatible,
            "certificate_class": certificate.certificate_class,
            "assignment_source": metadata.get("assignment_source"),
            "N": int(state.target_atoms or 0),
            "nodes": [
                {
                    "atomic_number": int(node.atomic_number),
                    "oxidation_state": int(node.oxidation_state),
                }
                for node in nodes
            ],
            "counts": [int(token.count) for token in state.tokens],
        }
    )
    return output


def node_from_record(value: Mapping[str, Any]) -> ValenceNode:
    return ValenceNode(int(value["atomic_number"]), int(value["oxidation_state"]))


def ledger_steps(
    nodes: Sequence[ValenceNode],
    counts: Sequence[int],
    *,
    target_n: int,
    target_arity: int,
) -> list[dict[str, Any]]:
    if len(nodes) != len(counts) or len(nodes) != int(target_arity):
        raise ValueError("benchmark semantic sequence must match exact arity")
    remaining_atoms = int(target_n)
    net_charge = 0
    remaining_species = int(target_arity)
    branch = "unset"
    steps = [
        {
            "remaining_atoms": remaining_atoms,
            "net_charge": net_charge,
            "remaining_species": remaining_species,
            "branch": branch,
        },
        {
            "remaining_atoms": remaining_atoms,
            "net_charge": net_charge,
            "remaining_species": remaining_species,
            "branch": branch,
        },
    ]
    for node, raw_count in zip(nodes, counts):
        count = int(raw_count)
        token_branch = "alloy" if int(node.oxidation_state) == 0 else "ionic"
        if branch == "unset":
            branch = token_branch
        elif branch != token_branch:
            raise ValueError("semantic ledger mixed alloy and ionic branches")
        remaining_atoms -= count
        net_charge += int(node.oxidation_state) * count
        remaining_species -= 1
        if remaining_atoms < 0 or remaining_species < 0:
            raise ValueError("semantic ledger underflow")
        steps.append(
            {
                "remaining_atoms": remaining_atoms,
                "net_charge": net_charge,
                "remaining_species": remaining_species,
                "branch": branch,
            }
        )
    if remaining_atoms != 0 or net_charge != 0 or remaining_species != 0:
        raise ValueError("semantic ledger did not terminate exactly")
    return steps


def build_vocabulary(train_rows: list[dict[str, Any]]) -> dict[str, Any]:
    supervised = [row for row in train_rows if row["composition_supervision"]]
    nodes = tuple(
        sorted(
            {
                node_from_record(value)
                for row in supervised
                for value in row["nodes"]
            }
        )
    )
    node_to_id = {node: index for index, node in enumerate(nodes)}
    pair_prior = CompositionPairPrior.fit(
        [tuple(node_from_record(value) for value in row["nodes"]) for row in supervised]
    )
    soft_vocabulary = {}
    for field in SOFT_FIELDS:
        values = sorted({str(row["soft_values"][field]) for row in train_rows})
        if UNKNOWN_SOFT not in values:
            values.append(UNKNOWN_SOFT)
        soft_vocabulary[field] = values
    return {
        "schema": "h1a2_c3fd_semantic_vocabulary_v1",
        "formula_bpe": False,
        "species": [
            {
                "id": int(node_to_id[node]),
                "atomic_number": int(node.atomic_number),
                "oxidation_state": int(node.oxidation_state),
                "label": node.label,
            }
            for node in nodes
        ],
        "species_eos_id": len(nodes),
        "N_values": list(range(1, 21)),
        "count_values": list(range(1, 21)),
        "physics": {
            "schema": "aufbau_ionic_features_v1",
            "feature_names": list(feature_names()),
            "matrix": [list(row) for row in species_physics_matrix(nodes)],
        },
        "pair_prior": pair_prior.to_dict(),
        "soft_vocabulary": soft_vocabulary,
    }


def encode_rows(
    rows: list[dict[str, Any]], vocabulary: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    node_to_id = {
        (int(row["atomic_number"]), int(row["oxidation_state"])): int(row["id"])
        for row in vocabulary["species"]
    }
    soft_to_id = {
        field: {str(value): index for index, value in enumerate(values)}
        for field, values in vocabulary["soft_vocabulary"].items()
    }
    encoded: list[dict[str, Any]] = []
    supervised = in_vocab = 0
    oov_nodes: Counter[str] = Counter()
    soft_oov: Counter[str] = Counter()
    certificate_classes: Counter[str] = Counter()
    for row in rows:
        certificate_classes[str(row["certificate_class"])] += 1
        species_ids: list[int] = []
        row_in_vocab = True
        for value in row["nodes"]:
            key = (int(value["atomic_number"]), int(value["oxidation_state"]))
            if key not in node_to_id:
                oov_nodes[f"{key[0]}|{key[1]}"] += 1
                row_in_vocab = False
                continue
            species_ids.append(int(node_to_id[key]))
        composition_supervision = bool(row["composition_supervision"] and row_in_vocab)
        supervised += int(row["composition_supervision"])
        in_vocab += int(composition_supervision)
        soft_labels = {}
        for field in SOFT_FIELDS:
            value = str(row["soft_values"][field])
            mapping = soft_to_id[field]
            if value not in mapping:
                soft_oov[f"{field}:{value}"] += 1
                value = UNKNOWN_SOFT
            soft_labels[field] = int(mapping[value])
        plan = row.get("plan_state") or {}
        target_n = None if row.get("N") is None else int(row["N"])
        target_arity = len(plan.get("elements") or ()) or len(row.get("nodes") or ())
        family_target = int(soft_labels["anion_framework"])
        proposal_supervision = bool(
            target_n is not None
            and 1 <= target_n <= 20
            and 1 <= int(target_arity) <= 7
        )
        semantic_nodes = [node_from_record(value) for value in row["nodes"]]
        semantic_ledger: list[dict[str, Any]] = []
        if composition_supervision:
            semantic_ledger = ledger_steps(
                semantic_nodes,
                row["counts"],
                target_n=int(target_n),
                target_arity=int(target_arity),
            )
        encoded.append(
            {
                "schema": "h1a2_c3fd_semantic_row_v1",
                "source_row_idx": row["source_row_idx"],
                "sample_weight": row["sample_weight"],
                "plan_state": row["plan_state"],
                "certificate_class": row["certificate_class"],
                "composition_supervision": composition_supervision,
                "proposal_supervision": proposal_supervision,
                "compile_error": row["compile_error"],
                "N_target": None if row["N"] is None else int(row["N"]),
                "proposal_targets": {
                    "family": family_target,
                    "N": target_n,
                    "arity": int(target_arity),
                },
                "species_labels": species_ids,
                "count_targets": [int(value) for value in row["counts"]],
                "ledger_steps": semantic_ledger,
                "soft_labels": soft_labels,
            }
        )
    return encoded, {
        "rows": len(rows),
        "benchmark_supervised": supervised,
        "benchmark_supervised_in_vocab": in_vocab,
        "benchmark_in_vocab_rate": 0.0 if supervised == 0 else in_vocab / supervised,
        "oov_nodes": dict(oov_nodes.most_common()),
        "soft_oov": dict(soft_oov.most_common()),
        "certificate_classes": dict(sorted(certificate_classes.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    split_rows: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "val", "test"):
        path = args.input_dir / f"{split}.jsonl"
        if not path.is_file():
            continue
        split_rows[split] = [
            compile_row(row, index) for index, row in enumerate(iter_jsonl(path))
        ]
    if "train" not in split_rows or "val" not in split_rows:
        raise ValueError("C3FD data requires train and validation splits")
    vocabulary = build_vocabulary(split_rows["train"])
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifests = {}
    for split, rows in split_rows.items():
        encoded, manifest = encode_rows(rows, vocabulary)
        manifests[split] = manifest
        with (args.output_dir / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for row in encoded:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    gate = {
        "formula_bpe_disabled": vocabulary["formula_bpe"] is False,
        "train_benchmark_rows_in_vocab_100pct": manifests["train"]["benchmark_in_vocab_rate"] == 1.0,
        "val_benchmark_rows_in_vocab_at_least_99_5pct": manifests["val"]["benchmark_in_vocab_rate"] >= 0.995,
        "physics_features_finite": all(
            all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in row)
            for row in vocabulary["physics"]["matrix"]
        ),
        "pair_prior_train_only": True,
    }
    gate["planner_training_data_authorized"] = all(gate.values())
    payload = {
        "schema": "h1a2_c3fd_planner_data_manifest_v1",
        "input_dir": str(args.input_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "splits": manifests,
        "vocabulary": vocabulary,
        "gate": gate,
    }
    (args.output_dir / "vocabulary.json").write_text(
        json.dumps(vocabulary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not gate["planner_training_data_authorized"]:
        raise RuntimeError("C3FD Planner data gate failed")
    (args.output_dir / "_SUCCESS").touch()
    print(json.dumps({"splits": manifests, "gate": gate}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
