#!/usr/bin/env python3
"""Finalize and hash the terminal two-seed D3PO training run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


SEEDS = (81017, 81018)
UPDATES = 348


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finalize(run: Path) -> dict[str, Any]:
    root = run.resolve()
    if not (root / "_SUCCESS").is_file():
        raise RuntimeError("two-seed run _SUCCESS is missing")
    if (root / "D3PO_TRAIN_FINAL.json").exists():
        raise FileExistsError(root / "D3PO_TRAIN_FINAL.json")
    two_seed = read_json(root / "D3PO_TWO_SEED_MANIFEST.json")
    if two_seed.get("status") != "success" or two_seed.get("sequential") is not True:
        raise RuntimeError("two-seed manifest is not a successful serial run")

    rows: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    for seed in SEEDS:
        seed_root = root / f"seed{seed}"
        manifest_path = seed_root / "D3PO_TRAIN_MANIFEST.json"
        manifest = read_json(manifest_path)
        if (
            manifest.get("status") != "success"
            or int(manifest.get("seed", -1)) != seed
            or int(manifest.get("optimizer_updates", -1)) != UPDATES
            or manifest.get("search_or_selection") is not False
            or manifest.get("step0_canary", {}).get("passed") is not True
        ):
            raise RuntimeError(f"seed{seed} is not a frozen successful terminal")
        checkpoint = manifest.get("checkpoint") or {}
        adapter = Path(str(checkpoint.get("policy_adapter_path") or ""))
        model_path = adapter / "adapter_model.safetensors"
        config_path = adapter / "adapter_config.json"
        for path in (manifest_path, seed_root / "training_log.jsonl", model_path, config_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        observed_model_hash = sha256_file(model_path)
        observed_config_hash = sha256_file(config_path)
        if observed_model_hash != checkpoint.get("adapter_model_sha256"):
            raise RuntimeError(f"seed{seed} adapter model hash mismatch")
        if observed_config_hash != checkpoint.get("adapter_config_sha256"):
            raise RuntimeError(f"seed{seed} adapter config hash mismatch")
        validation = manifest.get("validation") or {}
        rows.append(
            {
                "seed": seed,
                "updates": UPDATES,
                "elapsed_seconds": float(manifest["elapsed_seconds"]),
                "preference_accuracy": float(
                    validation["preference_accuracy_with_half_credit_for_ties"]
                ),
                "mean_margin": float(validation["mean_reference_corrected_margin"]),
                "preference_loss": float(validation["preference_loss"]),
                "winner_anchor_loss": float(validation["winner_anchor_loss"]),
                "policy_adapter_path": str(adapter),
                "adapter_model_sha256": observed_model_hash,
                "adapter_config_sha256": observed_config_hash,
                "selected": False,
            }
        )
        for label, path in (
            (f"seed{seed}.manifest", manifest_path),
            (f"seed{seed}.training_log", seed_root / "training_log.jsonl"),
            (f"seed{seed}.adapter_model", model_path),
            (f"seed{seed}.adapter_config", config_path),
        ):
            hashes[label] = sha256_file(path)

    for label, path in (
        ("two_seed_manifest", root / "D3PO_TWO_SEED_MANIFEST.json"),
        ("scientific_contract", root / "scientific_contract.tsv"),
        ("inputs", root / "inputs.sha256"),
    ):
        hashes[label] = sha256_file(path)
    gpu_hours = sum(row["elapsed_seconds"] for row in rows) / 3600.0
    report = {
        "schema": "h1a2_d3po_two_seed_train_final_v1",
        "status": "success",
        "scientific_result_available": False,
        "training_seeds": list(SEEDS),
        "sequential": True,
        "checkpoint_or_seed_selection": False,
        "rows": rows,
        "resources": {
            "gpus": 1,
            "cpus": 8,
            "expected_gpu_hours": 10.0,
            "scheduler_kill_ceiling_gpu_hours": 10.0,
            "observed_training_gpu_hours": gpu_hours,
        },
        "hashes": dict(sorted(hashes.items())),
    }
    json_path = root / "D3PO_TRAIN_FINAL.json"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (root / "D3PO_TRAIN_FINAL.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# D3PO two-seed training terminal",
        "",
        "Status: **engineering success; no scientific generation result yet**.",
        "",
        "| Seed | Updates | Validation pref. acc. | Mean margin | Pref. loss | Anchor loss | Selected |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['seed']} | {row['updates']} | {row['preference_accuracy']:.6f} | "
            f"{row['mean_margin']:.6f} | {row['preference_loss']:.6f} | "
            f"{row['winner_anchor_loss']:.6f} | no |"
        )
    lines.extend(
        [
            "",
            f"Observed training GPU-hours: `{gpu_hours:.4f}` on one A800.",
            "Validation is disclosed only; neither seed nor checkpoint is selected.",
        ]
    )
    (root / "D3PO_TRAIN_FINAL.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    output_hashes = {
        name: sha256_file(root / name)
        for name in ("D3PO_TRAIN_FINAL.json", "D3PO_TRAIN_FINAL.csv", "D3PO_TRAIN_FINAL.md")
    }
    (root / "D3PO_TRAIN_FINAL_OUTPUTS.sha256").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(output_hashes.items())),
        encoding="utf-8",
    )
    (root / "_TRAIN_FINAL_SUCCESS").touch(exist_ok=False)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    result = finalize(args.run)
    print(json.dumps({"status": result["status"], "run": str(args.run.resolve())}))


if __name__ == "__main__":
    main()
