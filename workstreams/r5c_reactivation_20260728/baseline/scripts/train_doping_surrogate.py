#!/usr/bin/env python3
"""Train a small PyTorch ensemble surrogate for directed doping ranking."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from crystal_dlm.doping import (
    all_candidate_combos,
    combo_from_name,
    combo_name,
    feature_vector,
    ranking_tuple,
    read_jsonl,
    status_priority,
    write_json,
)

PROPERTY_KEYS = ("band_gap", "formation_energy", "defect_formation_energy")


def read_cal_records(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8", errors="ignore") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            record["line_no"] = line_no
            try:
                combo = combo_from_name(record.get("name", ""))
            except Exception:
                continue
            record["combo"] = combo
            record["combo_name"] = combo_name(combo)
            for key in PROPERTY_KEYS:
                if key in record:
                    record[key] = float(record[key])
            rows.append(record)
    return rows


def raw_cal_record_count(path: Path) -> int:
    count = 0
    with path.open(encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def stratified_success_split(success_rows: List[Dict[str, Any]], seed: int) -> Dict[str, List[Dict[str, Any]]]:
    rng = random.Random(seed)
    by_label = {True: [], False: []}
    for row in success_rows:
        by_label[1.7 <= float(row["band_gap"]) <= 2.7].append(row)
    splits = {"train": [], "val": [], "test": []}
    for rows in by_label.values():
        rng.shuffle(rows)
        n_val = round(len(rows) * 0.15)
        n_test = round(len(rows) * 0.15)
        splits["val"].extend(rows[:n_val])
        splits["test"].extend(rows[n_val : n_val + n_test])
        splits["train"].extend(rows[n_val + n_test :])
    for rows in splits.values():
        rng.shuffle(rows)
    return splits


def summarize_records(
    cal_records: List[Dict[str, Any]],
    compact_records: List[Dict[str, Any]],
    data_root: Path,
) -> Dict[str, Dict[str, Any]]:
    by_combo: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in cal_records:
        by_combo[record["combo_name"]].append(record)
    success_by_combo = {combo_name(row["dopants"]): row for row in compact_records}

    candidates: Dict[str, Dict[str, Any]] = {}
    for combo in all_candidate_combos():
        name = combo_name(combo)
        records = by_combo.get(name, [])
        status = status_priority(record.get("state") for record in records)
        success_record = success_by_combo.get(name)
        source_cif = None
        true_properties = None
        if success_record:
            status = "SUCCESS"
            source_cif = success_record.get("source_cif")
            true_properties = {
                key: float(success_record["properties"][key])
                for key in PROPERTY_KEYS
            }
        elif records:
            source_cif = records[0].get("source_cif")
        candidates[name] = {
            "name": name,
            "dopants": list(combo),
            "status": status,
            "cal_record_count": len(records),
            "source_cif": source_cif,
            "matched_dft_cif_path": (
                str(data_root / "cif_data" / source_cif)
                if source_cif and (data_root / "cif_data" / source_cif).exists()
                else None
            ),
            "true_properties": true_properties,
            "target_band_gap": (
                None if true_properties is None else 1.7 <= true_properties["band_gap"] <= 2.7
            ),
        }
    return candidates


def make_tensor(rows: Sequence[Mapping[str, Any]], torch):
    return torch.tensor([feature_vector(row["dopants"]) for row in rows], dtype=torch.float32)


def train_one_model(
    seed: int,
    status_rows: List[Dict[str, Any]],
    success_train_rows: List[Dict[str, Any]],
    property_mean: Dict[str, float],
    property_std: Dict[str, float],
    epochs: int,
    lr: float,
    hidden_dim: int,
):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(seed)
    random.seed(seed)

    class Surrogate(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(91, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
            )
            self.success = nn.Linear(hidden_dim, 1)
            self.target = nn.Linear(hidden_dim, 1)
            self.props = nn.Linear(hidden_dim, 3)

        def forward(self, x):
            h = self.net(x)
            return self.success(h).squeeze(-1), self.target(h).squeeze(-1), self.props(h)

    model = Surrogate()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    x_status = make_tensor(status_rows, torch)
    y_success = torch.tensor([1.0 if row["status"] == "SUCCESS" else 0.0 for row in status_rows], dtype=torch.float32)
    x_success = make_tensor(success_train_rows, torch)
    y_target = torch.tensor(
        [1.0 if 1.7 <= float(row["band_gap"]) <= 2.7 else 0.0 for row in success_train_rows],
        dtype=torch.float32,
    )
    y_props = torch.tensor(
        [
            [
                (float(row[key]) - property_mean[key]) / property_std[key]
                for key in PROPERTY_KEYS
            ]
            for row in success_train_rows
        ],
        dtype=torch.float32,
    )

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        success_logits, _, _ = model(x_status)
        _, target_logits, prop_pred = model(x_success)
        loss = (
            F.binary_cross_entropy_with_logits(success_logits, y_success)
            + F.binary_cross_entropy_with_logits(target_logits, y_target)
            + F.mse_loss(prop_pred, y_props)
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    model.eval()
    return model


def predict_ensemble(
    models: Sequence[Any],
    rows: List[Dict[str, Any]],
    property_mean: Dict[str, float],
    property_std: Dict[str, float],
) -> List[Dict[str, Any]]:
    import torch

    x = make_tensor(rows, torch)
    success_probs = []
    target_probs = []
    prop_preds = []
    with torch.no_grad():
        for model in models:
            success_logits, target_logits, props = model(x)
            success_probs.append(torch.sigmoid(success_logits))
            target_probs.append(torch.sigmoid(target_logits))
            prop_preds.append(props)
    success_stack = torch.stack(success_probs)
    target_stack = torch.stack(target_probs)
    prop_stack = torch.stack(prop_preds)
    prop_mean_norm = prop_stack.mean(dim=0)
    prop_std_norm = prop_stack.std(dim=0)
    predictions = []
    for idx, row in enumerate(rows):
        props = {}
        prop_unc = 0.0
        for key_idx, key in enumerate(PROPERTY_KEYS):
            props[key] = float(prop_mean_norm[idx, key_idx]) * property_std[key] + property_mean[key]
            prop_unc += float(prop_std_norm[idx, key_idx]) * property_std[key]
        predictions.append(
            {
                **row,
                "p_success": float(success_stack[:, idx].mean()),
                "p_band_gap_target": float(target_stack[:, idx].mean()),
                "pred_band_gap": props["band_gap"],
                "pred_formation_energy": props["formation_energy"],
                "pred_defect_formation_energy": props["defect_formation_energy"],
                "uncertainty": prop_unc / len(PROPERTY_KEYS),
            }
        )
    return predictions


def mae(rows: Sequence[Mapping[str, Any]], key: str, pred_key: str) -> float:
    values = [
        abs(float(row["true_properties"][key]) - float(row[pred_key]))
        for row in rows
        if row.get("true_properties") is not None
    ]
    return sum(values) / max(1, len(values))


def evaluate_predictions(
    predicted: List[Dict[str, Any]],
    success_splits: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    test_names = {row["combo_name"] for row in success_splits["test"]}
    test_rows = [row for row in predicted if row["name"] in test_names and row["true_properties"] is not None]
    for row in test_rows:
        row["actual_target_band_gap"] = bool(row["target_band_gap"])
        row["pred_target_band_gap"] = bool(row["p_band_gap_target"] >= 0.5)
    tp = sum(1 for row in test_rows if row["actual_target_band_gap"] and row["pred_target_band_gap"])
    fp = sum(1 for row in test_rows if (not row["actual_target_band_gap"]) and row["pred_target_band_gap"])
    tn = sum(1 for row in test_rows if (not row["actual_target_band_gap"]) and (not row["pred_target_band_gap"]))
    fn = sum(1 for row in test_rows if row["actual_target_band_gap"] and (not row["pred_target_band_gap"]))
    ranked_test = sorted(test_rows, key=ranking_tuple, reverse=True)
    top20 = ranked_test[: min(20, len(ranked_test))]
    base_target_rate = sum(row["actual_target_band_gap"] for row in test_rows) / max(1, len(test_rows))
    top20_target_rate = sum(row["actual_target_band_gap"] for row in top20) / max(1, len(top20))
    return {
        "heldout_count": len(test_rows),
        "base_target_rate": base_target_rate,
        "top20_target_rate": top20_target_rate,
        "target_precision": tp / max(1, tp + fp),
        "target_recall": tp / max(1, tp + fn),
        "target_accuracy": (tp + tn) / max(1, len(test_rows)),
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "band_gap_mae": mae(test_rows, "band_gap", "pred_band_gap"),
        "formation_energy_mae": mae(test_rows, "formation_energy", "pred_formation_energy"),
        "defect_formation_energy_mae": mae(test_rows, "defect_formation_energy", "pred_defect_formation_energy"),
        "gate_a_pass": top20_target_rate >= max(0.50, base_target_rate),
        "heldout_top20": [
            {
                "name": row["name"],
                "dopants": row["dopants"],
                "actual_target_band_gap": row["actual_target_band_gap"],
                "true_properties": row["true_properties"],
                "p_band_gap_target": row["p_band_gap_target"],
                "pred_band_gap": row["pred_band_gap"],
                "pred_formation_energy": row["pred_formation_energy"],
                "pred_defect_formation_energy": row["pred_defect_formation_energy"],
            }
            for row in top20
        ],
    }


def write_review(path: Path, metrics: Mapping[str, Any]) -> None:
    status = "CONTINUE" if metrics["gate_a_pass"] else "WATCH"
    text = f"""# AI Review: compact doping surrogate

状态：`{status}`

本轮训练了 16-member PyTorch ensemble surrogate/ranker，用于把 compact DLM 生成的掺杂组合按定向性质排序。排序目标固定为：先提高 `p_success` 与 `p_band_gap_target`，再降低 `formation_energy`，最后提高 `defect_formation_energy`，并用 ensemble uncertainty 做轻微惩罚。

## 关键指标

- heldout_count：{metrics['heldout_count']}
- base_target_rate：{metrics['base_target_rate']:.4f}
- heldout top20 target rate：{metrics['top20_target_rate']:.4f}
- band_gap MAE：{metrics['band_gap_mae']:.4f}
- formation_energy MAE：{metrics['formation_energy_mae']:.4f}
- defect_formation_energy MAE：{metrics['defect_formation_energy_mae']:.4f}
- Gate A：{'通过' if metrics['gate_a_pass'] else '未通过'}

## 诊断

如果 Gate A 未通过，系统仍然可以进入 `known-label mode`，也就是只基于已有 DFT SUCCESS 标签做推荐；但 `surrogate-exploration mode` 的 unknown/failed 候选必须标记为 `needs_dft=true`，不能声称性质已被确认。
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("/zhdd/home/hengzhang/data/crystal"))
    parser.add_argument("--compact-jsonl", type=Path, default=PROJECT_ROOT / "data/doping_crystal/compact_records.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--notes-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=20260519)
    parser.add_argument("--ensemble-size", type=int, default=16)
    parser.add_argument("--seed-start", type=int, default=2026051900)
    parser.add_argument("--epochs", type=int, default=700)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--hidden-dim", type=int, default=96)
    args = parser.parse_args()

    import torch

    args.output_dir.mkdir(parents=True, exist_ok=True)
    notes_dir = args.notes_dir or args.output_dir.parent.parent / "notes"
    compact_records = read_jsonl(args.compact_jsonl)
    cal_path = args.data_root / "cal_data (1).txt"
    raw_cal_count = raw_cal_record_count(cal_path)
    cal_records = read_cal_records(cal_path)
    candidates_by_name = summarize_records(cal_records, compact_records, args.data_root)
    success_rows = [
        {
            **candidate,
            "combo_name": name,
            **candidate["true_properties"],
        }
        for name, candidate in candidates_by_name.items()
        if candidate["status"] == "SUCCESS" and candidate["true_properties"] is not None
    ]
    status_rows = [
        candidate
        for candidate in candidates_by_name.values()
        if candidate["status"] in {"SUCCESS", "FAIL", "NOT CAL"}
    ]
    success_splits = stratified_success_split(success_rows, args.seed)
    train_props = success_splits["train"]
    property_mean = {key: sum(float(row[key]) for row in train_props) / len(train_props) for key in PROPERTY_KEYS}
    property_std = {}
    for key in PROPERTY_KEYS:
        variance = sum((float(row[key]) - property_mean[key]) ** 2 for row in train_props) / max(1, len(train_props) - 1)
        property_std[key] = max(1e-6, math.sqrt(variance))

    models = []
    for offset in range(args.ensemble_size):
        models.append(
            train_one_model(
                seed=args.seed_start + offset,
                status_rows=status_rows,
                success_train_rows=success_splits["train"],
                property_mean=property_mean,
                property_std=property_std,
                epochs=args.epochs,
                lr=args.lr,
                hidden_dim=args.hidden_dim,
            )
        )

    candidate_rows = list(candidates_by_name.values())
    predicted = predict_ensemble(models, candidate_rows, property_mean, property_std)
    predicted_sorted = sorted(predicted, key=ranking_tuple, reverse=True)
    for rank, row in enumerate(predicted_sorted, start=1):
        row["rank"] = rank
        row["ranking_tuple"] = list(ranking_tuple(row))
        row["needs_dft"] = row["status"] != "SUCCESS"

    metrics = evaluate_predictions(predicted, success_splits)
    metrics.update(
        {
            "cal_record_count": raw_cal_count,
            "parsed_cal_record_count": len(cal_records),
            "compact_success_count": len(compact_records),
            "candidate_count": len(candidate_rows),
            "status_counts": {
                status: sum(1 for row in candidate_rows if row["status"] == status)
                for status in ("SUCCESS", "FAIL", "NOT CAL", "UNKNOWN")
            },
            "ensemble_size": args.ensemble_size,
            "seed_start": args.seed_start,
            "property_mean": property_mean,
            "property_std": property_std,
        }
    )

    write_json(args.output_dir / "eval_metrics.json", metrics)
    write_json(args.output_dir / "top_candidates.json", {"candidates": predicted_sorted})
    write_review(notes_dir / "ai_review.md", metrics)
    print(json.dumps({"metrics": metrics, "top5": predicted_sorted[:5]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
