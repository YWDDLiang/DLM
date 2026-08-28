#!/usr/bin/env python3
"""Evaluate the frozen two-head CTV value gate on disjoint validation Plans."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from crystal_dlm.ctv_q_head import (  # noqa: E402
    advantage_is_supported,
    build_q_head,
    pairwise_order_accuracy,
    plan_bootstrap_spearman,
)
from crystal_dlm.ctv_value_data import spearman  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def mean_targets(label_rows: list[dict[str, Any]]) -> dict[tuple[str, int], float]:
    values: dict[tuple[str, int], list[float]] = {}
    for row in label_rows:
        if row.get("chgnet_relaxation_known") is not True:
            continue
        value = row.get("chgnet_energy_per_atom")
        if value is None or not math.isfinite(float(value)):
            continue
        values.setdefault((str(row["state_id"]), int(row["action_token"])), []).append(
            float(value)
        )
    return {key: sum(group) / len(group) for key, group in values.items()}


def predict_state(
    *,
    model: Any,
    summary: dict[str, Any],
    state_feature: torch.Tensor,
    action_features: torch.Tensor,
    probabilities: torch.Tensor,
    milestone: float,
    families: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    count = int(action_features.shape[0])
    q_value, advantage = model(
        state_feature.reshape(1, -1),
        action_features.reshape(1, count, -1),
        probabilities.clamp_min(math.exp(-30.0)).log().reshape(1, count, 1),
        torch.full(
            (1, count, 1),
            float(milestone),
            device=action_features.device,
            dtype=torch.float32,
        ),
        torch.nn.functional.one_hot(families.reshape(1, count), num_classes=9).to(
            dtype=torch.float32
        ),
    )
    scale = float(summary["energy_scale"])
    center = float(summary["energy_center"])
    return q_value[0] * scale + center, advantage[0] * scale


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--q-heads", type=Path, required=True)
    parser.add_argument("--validation-labels", type=Path, required=True)
    parser.add_argument("--dataset-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    artifact = torch.load(args.features, map_location="cpu", weights_only=False)
    checkpoint = torch.load(args.q_heads, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != "h1a2_ctv_q_heads_v1":
        raise ValueError("CTV Q-head checkpoint schema changed")
    models = []
    for state_dict in checkpoint["head_state_dicts"]:
        model = build_q_head(256).to(device)
        model.load_state_dict(state_dict)
        models.append(model.eval())
    summaries = checkpoint["head_summaries"]
    basic_support = set(int(value) for value in checkpoint["shared_supported_token_ids"])
    targets = mean_targets(read_jsonl(args.validation_labels))
    audit = json.loads(args.dataset_audit.read_text(encoding="utf-8"))

    state_features = artifact["state_features"].to(dtype=torch.float32)
    token_ids = [int(value) for value in artifact["action_token_ids"].tolist()]
    action_feature_by_token = {
        token: feature.to(dtype=torch.float32)
        for token, feature in zip(token_ids, artifact["action_features"])
    }
    family_by_token = {
        token: int(family)
        for token, family in zip(
            token_ids, artifact["action_token_families"].tolist()
        )
    }
    prediction_rows = []
    state_rows = []
    plan_pairs: dict[int, list[tuple[float, float]]] = {}
    total_supported_mass = 0.0
    guided_states = 0
    pair_correct = 0.0
    pair_comparisons = 0
    state_spearman = []
    stratum_pairs: dict[str, list[tuple[float, int, float]]] = {
        "oxide": [],
        "sulfide": [],
        "n13_20": [],
    }

    with torch.no_grad():
        for feature_index, record in enumerate(artifact["state_records"]):
            if record["split"] != "validation":
                continue
            legal = [int(value) for value in record["legal_token_ids"]]
            probabilities = torch.tensor(
                record["legal_probabilities"], device=device, dtype=torch.float32
            )
            action_features = torch.stack(
                [action_feature_by_token[token] for token in legal]
            ).to(device)
            families = torch.tensor(
                [family_by_token[token] for token in legal],
                device=device,
                dtype=torch.long,
            )
            state_feature = state_features[feature_index].to(device)
            predictions = [
                predict_state(
                    model=model,
                    summary=summary,
                    state_feature=state_feature,
                    action_features=action_features,
                    probabilities=probabilities,
                    milestone=float(record["milestone"]),
                    families=families,
                )
                for model, summary in zip(models, summaries)
            ]
            supported = []
            for token_index, token in enumerate(legal):
                supported.append(
                    token in basic_support
                    and advantage_is_supported(
                        float(predictions[0][1][token_index].cpu().item()),
                        float(predictions[1][1][token_index].cpu().item()),
                    )
                )
            supported_mass = sum(
                probability
                for probability, is_supported in zip(
                    record["legal_probabilities"], supported
                )
                if is_supported
            )
            total_supported_mass += supported_mass
            guided = supported_mass >= 0.70
            guided_states += int(guided)

            legal_index = {token: index for index, token in enumerate(legal)}
            observed_tokens = [int(value) for value in record["action_token_ids"]]
            observed_truth = []
            observed_prediction = []
            for token in observed_tokens:
                truth = targets.get((str(record["state_id"]), token))
                if truth is None:
                    raise ValueError("CTV validation action target is missing")
                index = legal_index[token]
                estimate = 0.5 * (
                    float(predictions[0][0][index].cpu().item())
                    + float(predictions[1][0][index].cpu().item())
                )
                observed_truth.append(float(truth))
                observed_prediction.append(estimate)
                prediction_rows.append(
                    {
                        "state_id": str(record["state_id"]),
                        "plan_ordinal": int(record["plan_ordinal"]),
                        "composition_id": str(record["composition_id"]),
                        "milestone": float(record["milestone"]),
                        "num_atoms": int(record["num_atoms"]),
                        "action_token": token,
                        "base_probability": float(record["legal_probabilities"][index]),
                        "observed_energy": float(truth),
                        "head0_q": float(predictions[0][0][index].cpu().item()),
                        "head1_q": float(predictions[1][0][index].cpu().item()),
                        "head0_advantage": float(
                            predictions[0][1][index].cpu().item()
                        ),
                        "head1_advantage": float(
                            predictions[1][1][index].cpu().item()
                        ),
                        "supported": bool(supported[index]),
                        "state_supported_mass": float(supported_mass),
                    }
                )
                plan_pairs.setdefault(int(record["plan_ordinal"]), []).append(
                    (estimate, float(truth))
                )
            accuracy, comparisons, correct = pairwise_order_accuracy(
                observed_prediction, observed_truth
            )
            if accuracy is not None:
                pair_correct += correct
                pair_comparisons += comparisons
            rho = spearman(observed_prediction, observed_truth)
            if rho is not None:
                state_spearman.append(float(rho))
            composition_numbers = {
                int(part.split(":", 1)[0])
                for part in str(record["composition_id"]).split("|")
                if ":" in part
            }
            state_rows.append(
                {
                    "state_id": str(record["state_id"]),
                    "plan_ordinal": int(record["plan_ordinal"]),
                    "milestone": float(record["milestone"]),
                    "supported_mass": float(supported_mass),
                    "guided": bool(guided),
                    "pairwise_accuracy": accuracy,
                    "pairwise_comparisons": comparisons,
                    "spearman": rho,
                }
            )
            for name, present in (
                ("oxide", 8 in composition_numbers),
                ("sulfide", 16 in composition_numbers),
                ("n13_20", 13 <= int(record["num_atoms"]) <= 20),
            ):
                if present and accuracy is not None:
                    stratum_pairs[name].append((correct, comparisons, accuracy))

    if len(state_rows) != 64 or len(prediction_rows) != 512:
        raise ValueError("CTV Q validation denominator changed")
    bootstrap = plan_bootstrap_spearman(plan_pairs, draws=2000, seed=76017)
    validation_audit = audit["validation"]
    raw_cross_agreement = validation_audit.get("cross_pair_agreement")
    feasibility_has_both_classes = int(validation_audit["direct_valid"]) not in (
        0,
        int(validation_audit["branches"]),
    )
    feasibility_auroc = None
    symmetry_rank_agreement = None
    pairwise_auc = pair_correct / pair_comparisons if pair_comparisons else None
    coverage = guided_states / len(state_rows)
    fallback = 1.0 - coverage
    strata = {}
    for name, values in stratum_pairs.items():
        comparisons = sum(value[1] for value in values)
        correct = sum(value[0] for value in values)
        strata[name] = {
            "states": len(values),
            "comparisons": comparisons,
            "pairwise_auc": correct / comparisons if comparisons else None,
            "gate_applicable": len(values) >= 5 and comparisons >= 20,
            "direction_positive": (correct / comparisons) > 0.5
            if comparisons
            else None,
        }

    gates = {
        "spearman_lcb_positive": bootstrap["lcb_95"] is not None
        and float(bootstrap["lcb_95"]) > 0.0,
        "pairwise_auc_gt_0_60": pairwise_auc is not None and pairwise_auc > 0.60,
        "cross_continuation_gt_0_60": raw_cross_agreement is not None
        and float(raw_cross_agreement) > 0.60,
        "feasibility_auroc_gt_0_70": feasibility_auroc is not None
        and feasibility_auroc > 0.70,
        "symmetry_rank_gt_0_90": symmetry_rank_agreement is not None
        and symmetry_rank_agreement > 0.90,
        "guided_state_coverage_ge_0_60": coverage >= 0.60,
        "projected_fallback_le_0_40": fallback <= 0.40,
        "hard_strata_direction": all(
            (not value["gate_applicable"]) or value["direction_positive"] is True
            for value in strata.values()
        ),
    }
    all_pre_symmetry = all(
        value for key, value in gates.items() if key != "symmetry_rank_gt_0_90"
    )
    symmetry_status = (
        "required_next" if all_pre_symmetry else "not_run_due_to_prior_hard_gate"
    )
    q_gate_pass = all(gates.values())
    gamma = None

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    prediction_path = output / "CTV_Q_VALIDATION_PREDICTIONS.csv"
    with prediction_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0]))
        writer.writeheader()
        writer.writerows(prediction_rows)
    state_path = output / "CTV_Q_VALIDATION_STATES.csv"
    with state_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(state_rows[0]))
        writer.writeheader()
        writer.writerows(state_rows)

    report = {
        "schema": "h1a2_ctv_q_gate_v1",
        "validation_plans": 32,
        "validation_states": 64,
        "validation_observed_actions": 512,
        "spearman": bootstrap,
        "mean_state_spearman": sum(state_spearman) / len(state_spearman),
        "pairwise_auc": pairwise_auc,
        "pairwise_comparisons": pair_comparisons,
        "raw_cross_continuation_agreement": raw_cross_agreement,
        "raw_cross_continuation_comparisons": validation_audit.get(
            "cross_pair_comparisons"
        ),
        "feasibility_has_both_classes": feasibility_has_both_classes,
        "feasibility_auroc": feasibility_auroc,
        "symmetry_rank_agreement": symmetry_rank_agreement,
        "symmetry_status": symmetry_status,
        "mean_supported_mass": total_supported_mass / len(state_rows),
        "guided_states": guided_states,
        "guided_state_coverage": coverage,
        "projected_fallback": fallback,
        "strata": strata,
        "gates": gates,
        "q_gate_pass": q_gate_pass,
        "selected_gamma": gamma,
        "l6_authorized": q_gate_pass and gamma is not None,
    }
    (output / "CTV_Q_GATE_FINAL.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# CTV Q-head validation gate",
        "",
        f"- Plan-bootstrap Spearman: `{bootstrap['point']}` "
        f"(95% LCB `{bootstrap['lcb_95']}`)",
        f"- Pairwise AUC: `{pairwise_auc}` over `{pair_comparisons}` comparisons",
        f"- Raw cross-continuation agreement: `{raw_cross_agreement}`",
        f"- Guided-state coverage / fallback: `{coverage}` / `{fallback}`",
        f"- Feasibility AUROC: `{feasibility_auroc}` "
        f"(both classes: `{feasibility_has_both_classes}`)",
        f"- Symmetry: `{symmetry_status}`",
        f"- Q gate pass: `{q_gate_pass}`",
        f"- Selected gamma: `{gamma}`",
        f"- L6 authorized: `{report['l6_authorized']}`",
    ]
    (output / "CTV_Q_GATE_FINAL.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    (output / "_SUCCESS").touch()
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
