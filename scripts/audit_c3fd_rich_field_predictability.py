#!/usr/bin/env python3
"""Audit C3FD rich-head predictability and deployed-field redundancy on CPU."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


LATTICE_TO_SPACEGROUP = {
    "triclinic": "sg_001_002",
    "monoclinic": "sg_003_015",
    "orthorhombic": "sg_016_074",
    "tetragonal": "sg_075_142",
    "trigonal": "sg_143_167",
    "hexagonal": "sg_168_194",
    "cubic": "sg_195_230",
}
ACTIVE_SAMPLED_FIELDS = ("lattice_system", "volume_per_atom_bin")
HARD_DERIVED_FIELDS = ("anion_framework", "charge_bucket")
COMPILER_DERIVED_FIELDS = ("spacegroup_bucket",)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def class_entropy(labels: Sequence[int]) -> float:
    counts = Counter(int(label) for label in labels)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum(
        (count / total) * math.log(count / total)
        for count in counts.values()
        if count
    )


def conditional_entropy(left: Sequence[int], right: Sequence[int]) -> float:
    """Return H(right | left) in nats for paired discrete observations."""
    if len(left) != len(right):
        raise ValueError("conditional entropy arrays differ in length")
    if not left:
        return 0.0
    total = len(left)
    grouped: dict[int, Counter[int]] = {}
    for left_value, right_value in zip(left, right):
        grouped.setdefault(int(left_value), Counter())[int(right_value)] += 1
    value = 0.0
    for counts in grouped.values():
        group_total = sum(counts.values())
        group_entropy = -sum(
            (count / group_total) * math.log(count / group_total)
            for count in counts.values()
            if count
        )
        value += group_total / total * group_entropy
    return value


def joint_tvd(
    left_a: Sequence[int],
    left_b: Sequence[int],
    right_a: Sequence[int],
    right_b: Sequence[int],
) -> float:
    if not (len(left_a) == len(left_b) == len(right_a) == len(right_b)):
        raise ValueError("joint TVD arrays differ in length")
    if not left_a:
        return 0.0
    left = Counter(zip((int(value) for value in left_a), (int(value) for value in left_b)))
    right = Counter(zip((int(value) for value in right_a), (int(value) for value in right_b)))
    total = len(left_a)
    return 0.5 * sum(
        abs(left.get(key, 0) / total - right.get(key, 0) / total)
        for key in set(left) | set(right)
    )


def expected_calibration_error(
    confidences: Sequence[float],
    correct: Sequence[bool],
    *,
    bins: int = 10,
) -> float:
    if len(confidences) != len(correct):
        raise ValueError("confidence/correct lengths differ")
    if bins <= 0:
        raise ValueError("bins must be positive")
    if not confidences:
        return 0.0
    total = len(confidences)
    value = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        selected = [
            row
            for row, confidence in enumerate(confidences)
            if confidence >= lower
            and (confidence < upper or (index == bins - 1 and confidence <= upper))
        ]
        if not selected:
            continue
        accuracy = sum(bool(correct[row]) for row in selected) / len(selected)
        confidence = sum(float(confidences[row]) for row in selected) / len(selected)
        value += len(selected) / total * abs(accuracy - confidence)
    return value


def summarize_predictions(
    targets: Sequence[int],
    predictions: Sequence[int],
    confidences: Sequence[float],
    *,
    nll_sum: float,
    labels: Sequence[str],
    ordered: bool = False,
) -> dict[str, Any]:
    if not (len(targets) == len(predictions) == len(confidences)):
        raise ValueError("prediction arrays differ in length")
    if not targets:
        raise ValueError("no targets")
    count = len(targets)
    correct = [int(left) == int(right) for left, right in zip(targets, predictions)]
    target_counts = Counter(int(value) for value in targets)
    prediction_counts = Counter(int(value) for value in predictions)
    majority = max(target_counts.values()) / count
    accuracy = sum(correct) / count
    confusion = [[0 for _ in labels] for _ in labels]
    for target, prediction in zip(targets, predictions):
        confusion[int(target)][int(prediction)] += 1
    payload: dict[str, Any] = {
        "n": count,
        "nll": float(nll_sum) / count,
        "accuracy": accuracy,
        "majority_accuracy": majority,
        "accuracy_minus_majority_pp": 100.0 * (accuracy - majority),
        "ece_10": expected_calibration_error(confidences, correct, bins=10),
        "target_entropy_nats": class_entropy(targets),
        "prediction_entropy_nats": class_entropy(predictions),
        "labels": list(labels),
        "target_counts": {
            str(labels[index]): int(target_counts.get(index, 0))
            for index in range(len(labels))
        },
        "prediction_counts": {
            str(labels[index]): int(prediction_counts.get(index, 0))
            for index in range(len(labels))
        },
        "confusion": confusion,
    }
    if ordered:
        errors = [abs(int(left) - int(right)) for left, right in zip(targets, predictions)]
        payload["ordinal_mae_bins"] = sum(errors) / count
        payload["within_one_bin"] = sum(error <= 1 for error in errors) / count
    return payload


def summarize_spacegroup_relation(
    *,
    lattice_targets: Sequence[int],
    lattice_predictions: Sequence[int],
    sg_targets: Sequence[int],
    sg_predictions: Sequence[int],
    lattice_labels: Sequence[str],
    sg_labels: Sequence[str],
) -> dict[str, Any]:
    lengths = {
        len(lattice_targets),
        len(lattice_predictions),
        len(sg_targets),
        len(sg_predictions),
    }
    if len(lengths) != 1 or not lattice_targets:
        raise ValueError("spacegroup relation arrays differ or are empty")
    sg_to_id = {str(label): index for index, label in enumerate(sg_labels)}

    def derived(lattice_id: int) -> int | None:
        lattice = str(lattice_labels[int(lattice_id)])
        spacegroup = LATTICE_TO_SPACEGROUP.get(lattice)
        if spacegroup is None or spacegroup not in sg_to_id:
            return None
        return int(sg_to_id[spacegroup])

    target_derived = [derived(value) for value in lattice_targets]
    predicted_derived = [derived(value) for value in lattice_predictions]
    count = len(lattice_targets)
    return {
        "n": count,
        "independent_lattice_and_sg_joint_accuracy": sum(
            lattice_prediction == lattice_target and sg_prediction == sg_target
            for lattice_target, lattice_prediction, sg_target, sg_prediction in zip(
                lattice_targets,
                lattice_predictions,
                sg_targets,
                sg_predictions,
            )
        )
        / count,
        "independent_predicted_joint_tvd_from_target": joint_tvd(
            lattice_targets,
            sg_targets,
            lattice_predictions,
            sg_predictions,
        ),
        "target_sg_conditional_entropy_given_metric_lattice_nats": conditional_entropy(
            lattice_targets,
            sg_targets,
        ),
        "independent_predicted_sg_conditional_entropy_given_metric_lattice_nats": conditional_entropy(
            lattice_predictions,
            sg_predictions,
        ),
        "lattice_derived_sg_coverage": sum(value is not None for value in predicted_derived)
        / count,
        "target_one_to_one_lattice_sg_map_agreement": sum(
            left is not None and left == right
            for left, right in zip(target_derived, sg_targets)
        )
        / count,
        "independent_sg_head_accuracy": sum(
            left == right for left, right in zip(sg_predictions, sg_targets)
        )
        / count,
        "current_compiler_sg_accuracy_against_target": sum(
            left is not None and left == right
            for left, right in zip(predicted_derived, sg_targets)
        )
        / count,
        "independent_vs_lattice_derived_agreement": sum(
            right is not None and left == right
            for left, right in zip(sg_predictions, predicted_derived)
        )
        / count,
        "current_compiler_sg_incremental_entropy_given_metric_lattice_nats": 0.0,
        "current_compiler_sg_is_deterministic_output": True,
        "semantic_note": (
            "lattice_system is a metric-cell class from lengths/angles; "
            "spacegroup_bucket is a symmetry label from metadata. "
            "One-to-one map disagreement is not automatically invalid."
        ),
    }


def load_training_module() -> Any:
    path = PROJECT_ROOT / "scripts" / "train_c3fd_planner.py"
    spec = importlib.util.spec_from_file_location("train_c3fd_planner_for_audit", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_checkpoint_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("checkpoint must be NAME=PATH")
    name, raw_path = value.split("=", 1)
    if not name or not raw_path:
        raise argparse.ArgumentTypeError("checkpoint must be NAME=PATH")
    return name, Path(raw_path)


def run_checkpoint(
    *,
    name: str,
    checkpoint_path: Path,
    data_dir: Path,
    split: str,
    batch_size: int,
) -> dict[str, Any]:
    # Torch remains lazy so pure metric unit tests do not require the GPU runtime.
    import torch
    import torch.nn.functional as functional
    from torch.utils.data import DataLoader

    from crystal_dlm.c3fd_planner_model import C3FDPlannerConfig, C3FDPlannerModel
    from crystal_dlm.semantic_composition_head import SemanticHeadFlags

    training = load_training_module()
    checkpoint_path = checkpoint_path.resolve()
    data_dir = data_dir.resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("schema") != "h1a2_c3fd_planner_checkpoint_v1":
        raise ValueError(f"unexpected checkpoint schema at {checkpoint_path}")
    vocabulary_path = data_dir / "vocabulary.json"
    vocabulary = json.loads(vocabulary_path.read_text(encoding="utf-8"))
    vocabulary_sha = sha256_file(vocabulary_path)
    if str(checkpoint.get("vocabulary_sha256")) != vocabulary_sha:
        raise ValueError("checkpoint/vocabulary SHA mismatch")
    config = C3FDPlannerConfig(**checkpoint["config"])
    physics = torch.tensor(vocabulary["physics"]["matrix"], dtype=torch.float32)
    model = C3FDPlannerModel(config, physics_features=physics)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    context = torch.as_tensor(checkpoint["context"], dtype=torch.float32)
    if tuple(context.shape[:1]) != (1,):
        raise ValueError("checkpoint context must contain exactly one row")
    soft_values: Mapping[str, Sequence[str]] = vocabulary["soft_vocabulary"]
    soft_fields = tuple(sorted(soft_values))
    dataset_path = data_dir / f"{split}.jsonl"
    dataset = training.C3FDDataset(dataset_path)
    eos_id = int(vocabulary["species_eos_id"])
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=lambda rows: training.collate(
            rows,
            eos_species_id=eos_id,
            soft_fields=soft_fields,
        ),
    )
    collected = {
        field: {"target": [], "prediction": [], "confidence": [], "nll_sum": 0.0}
        for field in soft_fields
    }
    with torch.inference_mode():
        for batch in loader:
            output = model(
                context.expand(batch["n_targets"].shape[0], -1),
                previous_species_indices=batch["previous_species_indices"],
                previous_count_values=batch["previous_count_values"],
                previous_n_values=batch["previous_n_values"],
                ledger_features=batch["ledger_features"],
                flags=SemanticHeadFlags(use_physics=True),
            )
            for field in soft_fields:
                target = batch[f"rich:{field}"]
                valid = target != -100
                logits = output.rich_logits[field][valid].float()
                target_values = target[valid].long()
                if not bool(valid.any().item()):
                    continue
                probabilities = torch.softmax(logits, dim=-1)
                confidence, prediction = probabilities.max(dim=-1)
                values = collected[field]
                values["target"].extend(int(item) for item in target_values.tolist())
                values["prediction"].extend(int(item) for item in prediction.tolist())
                values["confidence"].extend(float(item) for item in confidence.tolist())
                values["nll_sum"] += float(
                    functional.cross_entropy(logits, target_values, reduction="sum").item()
                )
    fields = {
        field: summarize_predictions(
            values["target"],
            values["prediction"],
            values["confidence"],
            nll_sum=float(values["nll_sum"]),
            labels=[str(item) for item in soft_values[field]],
            ordered=field == "volume_per_atom_bin",
        )
        for field, values in collected.items()
    }
    relation = summarize_spacegroup_relation(
        lattice_targets=collected["lattice_system"]["target"],
        lattice_predictions=collected["lattice_system"]["prediction"],
        sg_targets=collected["spacegroup_bucket"]["target"],
        sg_predictions=collected["spacegroup_bucket"]["prediction"],
        lattice_labels=[str(item) for item in soft_values["lattice_system"]],
        sg_labels=[str(item) for item in soft_values["spacegroup_bucket"]],
    )
    checkpoint_calibration = sorted((checkpoint.get("calibration") or {}).keys())
    return {
        "name": name,
        "seed": int(checkpoint.get("seed", -1)),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "data_dir": str(data_dir),
        "dataset": str(dataset_path),
        "dataset_sha256": sha256_file(dataset_path),
        "vocabulary_sha256": vocabulary_sha,
        "split": split,
        "rows": len(dataset),
        "device": "cpu",
        "fields": fields,
        "spacegroup_relation": relation,
        "checkpoint_calibration_heads": checkpoint_calibration,
        "active_rich_heads_calibrated_in_checkpoint": sorted(
            set(checkpoint_calibration) & set(ACTIVE_SAMPLED_FIELDS)
        ),
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# C3FD rich-field predictability audit",
        "",
        "This is a CPU-only validation audit. It reads no generated-structure or",
        "stability outcomes and has no downstream pass threshold.",
        "",
        "## Deployed semantics",
        "",
        "- sampled rich logits: `lattice_system`, `volume_per_atom_bin`;",
        "- hard-derived lines: `anion_framework`, `charge_bucket`;",
        "- currently compiler-derived line: `spacegroup_bucket` from metric lattice system;",
        "- target `lattice_system` is derived from cell metric, while target SG comes from symmetry metadata;",
        "- therefore one-to-one metric/SG disagreement is not automatically a physical inconsistency.",
        "",
        "## Validation metrics",
        "",
        "| checkpoint | field | NLL | accuracy | majority | gain (pp) | ECE | ordinal MAE |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for checkpoint in report["checkpoints"]:
        for field, metrics in checkpoint["fields"].items():
            ordinal = metrics.get("ordinal_mae_bins")
            lines.append(
                "| {name} | {field} | {nll:.4f} | {accuracy:.4f} | {majority:.4f} | "
                "{gain:+.2f} | {ece:.4f} | {ordinal} |".format(
                    name=checkpoint["name"],
                    field=field,
                    nll=metrics["nll"],
                    accuracy=metrics["accuracy"],
                    majority=metrics["majority_accuracy"],
                    gain=metrics["accuracy_minus_majority_pp"],
                    ece=metrics["ece_10"],
                    ordinal="" if ordinal is None else f"{ordinal:.4f}",
                )
            )
    lines.extend(["", "## Space-group redundancy", ""])
    for checkpoint in report["checkpoints"]:
        relation = checkpoint["spacegroup_relation"]
        lines.extend(
            [
                f"### {checkpoint['name']}",
                "",
                f"- target one-to-one metric/SG map agreement: {relation['target_one_to_one_lattice_sg_map_agreement']:.4f}",
                f"- unused independent SG-head accuracy: {relation['independent_sg_head_accuracy']:.4f}",
                f"- current compiler-derived SG accuracy: {relation['current_compiler_sg_accuracy_against_target']:.4f}",
                f"- independent-head vs derived agreement: {relation['independent_vs_lattice_derived_agreement']:.4f}",
                f"- target H(SG | metric lattice): {relation['target_sg_conditional_entropy_given_metric_lattice_nats']:.4f} nats",
                f"- independent predicted joint TVD: {relation['independent_predicted_joint_tvd_from_target']:.4f}",
                f"- active rich heads with checkpoint temperature calibration: {checkpoint['active_rich_heads_calibrated_in_checkpoint']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation boundary",
            "",
            "Teacher-forced predictability is necessary for a useful predicted field, but",
            "it does not establish causal stability value. That requires the frozen matched",
            "development canary. No field subset is selected from test outcomes.",
            "The independent SG head and the compiler-derived SG are distinct deployed choices;",
            "their validation accuracy must not be conflated with a hard lattice/SG consistency rule.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        action="append",
        type=parse_checkpoint_arg,
        required=True,
        help="repeat NAME=PATH for each frozen checkpoint",
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    checkpoints = [
        run_checkpoint(
            name=name,
            checkpoint_path=path,
            data_dir=args.data_dir,
            split=args.split,
            batch_size=int(args.batch_size),
        )
        for name, path in args.checkpoint
    ]
    report = {
        "schema": "h1a2_c3fd_rich_field_predictability_audit_v2",
        "checkpoints": checkpoints,
        "deployed_semantics": {
            "active_sampled_fields": list(ACTIVE_SAMPLED_FIELDS),
            "hard_derived_fields": list(HARD_DERIVED_FIELDS),
            "compiler_derived_fields": list(COMPILER_DERIVED_FIELDS),
        },
        "outcomes_read": False,
        "gpu_jobs_used": 0,
        "device": "cpu",
        "has_downstream_pass_threshold": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    json_path = args.output_dir / "C3FD_RICH_FIELD_PREDICTABILITY_AUDIT.json"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path = args.output_dir / "C3FD_RICH_FIELD_PREDICTABILITY_AUDIT.md"
    md_path.write_text(render_markdown(report), encoding="utf-8", newline="\n")
    csv_path = args.output_dir / "C3FD_RICH_FIELD_PREDICTABILITY_AUDIT.csv"
    with csv_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "checkpoint",
                "seed",
                "field",
                "n",
                "nll",
                "accuracy",
                "majority_accuracy",
                "accuracy_minus_majority_pp",
                "ece_10",
                "ordinal_mae_bins",
            ),
        )
        writer.writeheader()
        for checkpoint in checkpoints:
            for field, metrics in checkpoint["fields"].items():
                writer.writerow(
                    {
                        "checkpoint": checkpoint["name"],
                        "seed": checkpoint["seed"],
                        "field": field,
                        "n": metrics["n"],
                        "nll": metrics["nll"],
                        "accuracy": metrics["accuracy"],
                        "majority_accuracy": metrics["majority_accuracy"],
                        "accuracy_minus_majority_pp": metrics["accuracy_minus_majority_pp"],
                        "ece_10": metrics["ece_10"],
                        "ordinal_mae_bins": metrics.get("ordinal_mae_bins", ""),
                    }
                )
    outputs = {
        path.name: sha256_file(path)
        for path in (json_path, md_path, csv_path)
    }
    success_path = args.output_dir / "_SUCCESS"
    success_path.write_text(canonical_json(outputs) + "\n", encoding="utf-8")
    print(json.dumps({"outputs": outputs, "checkpoints": len(checkpoints)}, indent=2))


if __name__ == "__main__":
    main()
