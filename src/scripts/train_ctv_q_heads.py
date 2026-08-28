#!/usr/bin/env python3
"""Fit the two frozen-architecture CTV Q heads on disjoint Plan groups."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F

from crystal_dlm.ctv_q_head import (  # noqa: E402
    build_q_head,
    disjoint_plan_group,
    robust_scale,
    supported_token_ids,
    token_support_counts,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean_targets(label_rows: list[dict[str, Any]]) -> dict[tuple[str, int], float]:
    values: dict[tuple[str, int], list[float]] = {}
    for row in label_rows:
        if row.get("chgnet_relaxation_known") is not True:
            continue
        energy = row.get("chgnet_energy_per_atom")
        if energy is None or not math.isfinite(float(energy)):
            continue
        key = (str(row["state_id"]), int(row["action_token"]))
        values.setdefault(key, []).append(float(energy))
    return {key: sum(group) / len(group) for key, group in values.items()}


def state_training_rows(
    artifact: dict[str, Any], label_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    targets = mean_targets(label_rows)
    rows = []
    for feature_index, record in enumerate(artifact["state_records"]):
        if record["split"] != "train":
            continue
        energies = []
        for token in record["action_token_ids"]:
            value = targets.get((str(record["state_id"]), int(token)))
            if value is None:
                raise ValueError("CTV train state has a missing action target")
            energies.append(float(value))
        rows.append(
            {
                "feature_index": feature_index,
                "state_id": str(record["state_id"]),
                "composition_id": str(record["composition_id"]),
                "plan_ordinal": int(record["plan_ordinal"]),
                "milestone": float(record["milestone"]),
                "action_token_ids": [int(value) for value in record["action_token_ids"]],
                "action_probabilities": [
                    float(value) for value in record["action_probabilities"]
                ],
                "energies": energies,
            }
        )
    if len(rows) != 256:
        raise ValueError("CTV Q training requires exactly 256 train states")
    return rows


def tensorize_group(
    *,
    rows: list[dict[str, Any]],
    state_features: torch.Tensor,
    action_feature_by_token: dict[int, torch.Tensor],
    family_by_token: dict[int, int],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    state_tensor = torch.stack(
        [state_features[int(row["feature_index"])] for row in rows]
    ).to(device=device, dtype=torch.float32)
    action_tensor = torch.stack(
        [
            torch.stack(
                [action_feature_by_token[int(token)] for token in row["action_token_ids"]]
            )
            for row in rows
        ]
    ).to(device=device, dtype=torch.float32)
    probabilities = torch.tensor(
        [row["action_probabilities"] for row in rows],
        device=device,
        dtype=torch.float32,
    ).clamp_min_(math.exp(-30.0))
    milestones = torch.tensor(
        [[row["milestone"]] * 8 for row in rows],
        device=device,
        dtype=torch.float32,
    ).unsqueeze(-1)
    family = torch.zeros(len(rows), 8, 9, device=device, dtype=torch.float32)
    for state_index, row in enumerate(rows):
        for action_index, token in enumerate(row["action_token_ids"]):
            family[state_index, action_index, family_by_token[int(token)]] = 1.0
    targets = torch.tensor(
        [row["energies"] for row in rows], device=device, dtype=torch.float32
    )
    return {
        "state": state_tensor,
        "action": action_tensor,
        "log_probability": probabilities.log().unsqueeze(-1),
        "milestone": milestones,
        "family": family,
        "target": targets,
    }


def fit_head(
    *,
    group: int,
    seed: int,
    rows: list[dict[str, Any]],
    tensors: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[Any, dict[str, Any]]:
    if len(rows) != 128 or len({row["plan_ordinal"] for row in rows}) != 64:
        raise ValueError(f"CTV Q head {group} does not have 64 Plans/128 states")
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    model = build_q_head(256).to(device)
    target_values = tensors["target"].detach().cpu().reshape(-1).tolist()
    center, scale = robust_scale(target_values)
    target = (tensors["target"] - center) / scale
    centered_target = target - target.mean(dim=1, keepdim=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, weight_decay=1e-3
    )
    losses = []
    model.train()
    for update in range(512):
        q_value, advantage = model(
            tensors["state"],
            tensors["action"],
            tensors["log_probability"],
            tensors["milestone"],
            tensors["family"],
        )
        absolute_loss = F.smooth_l1_loss(q_value, target)
        advantage_loss = F.smooth_l1_loss(advantage, centered_target)
        loss = absolute_loss + advantage_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if update in (0, 127, 255, 383, 511):
            losses.append(
                {
                    "update": update + 1,
                    "loss": float(loss.detach().cpu().item()),
                    "absolute_loss": float(absolute_loss.detach().cpu().item()),
                    "advantage_loss": float(advantage_loss.detach().cpu().item()),
                }
            )
    return model.eval(), {
        "group": int(group),
        "seed": int(seed),
        "plans": 64,
        "states": 128,
        "energy_center": center,
        "energy_scale": scale,
        "loss_trace": losses,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--train-labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CTV Q training requires one CUDA device")
    device = torch.device("cuda", 0)
    artifact = torch.load(args.features, map_location="cpu", weights_only=False)
    if artifact.get("schema") != "h1a2_ctv_state_action_features_v1":
        raise ValueError("CTV feature artifact schema changed")
    if int(artifact.get("projection_dim", -1)) != 256:
        raise ValueError("CTV feature projection changed")
    label_rows = read_jsonl(args.train_labels)
    rows = state_training_rows(artifact, label_rows)
    action_feature_by_token = {
        int(token): feature
        for token, feature in zip(
            artifact["action_token_ids"].tolist(), artifact["action_features"]
        )
    }
    family_by_token = {
        int(token): int(family)
        for token, family in zip(
            artifact["action_token_ids"].tolist(),
            artifact["action_token_families"].tolist(),
        )
    }
    state_features = artifact["state_features"]

    models = []
    summaries = []
    support = []
    for group, seed in ((0, 74017), (1, 75017)):
        group_rows = [
            row for row in rows if disjoint_plan_group(row["plan_ordinal"]) == group
        ]
        flat_rows = [
            {
                "action_token": token,
                "plan_ordinal": row["plan_ordinal"],
                "energy": energy,
            }
            for row in group_rows
            for token, energy in zip(row["action_token_ids"], row["energies"])
        ]
        counts = token_support_counts(flat_rows)
        tensors = tensorize_group(
            rows=group_rows,
            state_features=state_features,
            action_feature_by_token=action_feature_by_token,
            family_by_token=family_by_token,
            device=device,
        )
        model, summary = fit_head(
            group=group,
            seed=seed,
            rows=group_rows,
            tensors=tensors,
            device=device,
        )
        models.append(model)
        summaries.append(summary)
        support.append(counts)
    shared_support = supported_token_ids(support[0], support[1])

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = output / "CTV_Q_HEADS.pt"
    torch.save(
        {
            "schema": "h1a2_ctv_q_heads_v1",
            "architecture": "layernorm_baseline64_advantage128_v1",
            "projection_dim": 256,
            "head_state_dicts": [model.state_dict() for model in models],
            "head_summaries": summaries,
            "support_counts": support,
            "shared_supported_token_ids": sorted(shared_support),
            "support_contract": {
                "minimum_known_returns_per_head": 8,
                "minimum_unique_plans_per_head": 4,
            },
        },
        checkpoint,
    )
    manifest = {
        "schema": "h1a2_ctv_q_heads_manifest_v1",
        "heads": 2,
        "groups_disjoint": True,
        "group_rule": "frozen_plan_ordinal_mod_2",
        "updates_per_head": 512,
        "validation_used_for_training": False,
        "shared_supported_tokens": len(shared_support),
        "head_summaries": summaries,
        "features_sha256": sha256(args.features),
        "train_labels_sha256": sha256(args.train_labels),
        "checkpoint_sha256": sha256(checkpoint),
    }
    (output / "CTV_Q_HEADS_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "_SUCCESS").touch()
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
