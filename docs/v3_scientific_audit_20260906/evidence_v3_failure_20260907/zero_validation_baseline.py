"""Reconstruct the exact saved validation zero-v/u baseline on CPU only.

Accept train_40064 or train_40064/train.  Uses its archived source, monitored
keys, original continuous identities, weights, time centres and source RNGs.
No tokenizer/model/checkpoint weights are instantiated and no GPU is queried.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path

from diagnose_geometry_field import zero_baseline


def file_hash(path):
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path)
    args = parser.parse_args()
    train = args.train_dir.resolve()
    if not (train / "training_config.json").is_file():
        train = train / "train"
    config_path = train / "training_config.json"
    log_path = train / "validation_log.jsonl"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validation = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    source_root = args.source_root or (train.parent / "code")
    if not (source_root / "src/crystal_dlm/mixed_geometry_diffusion.py").is_file():
        raise ValueError("the archived numerical source is required; set --source-root explicitly")
    identity = Path(config["val_identity"])
    prepared = Path(config["val_data"])
    receipts = {}
    for name, path in (("val_identity", identity), ("val_data", prepared)):
        actual = file_hash(path)
        expected = config["input_sha256"][name]
        if actual != expected:
            raise ValueError(f"{name} bytes differ from the training manifest")
        receipts[name] = {"path": str(path), "sha256": actual, "matches_training_manifest": True}
    final_path = train / "TRAIN_FINAL.json"
    final = json.loads(final_path.read_text(encoding="utf-8"))
    state_path = Path(final["policy_path"]) / "periodic_state_config.json"
    state_config = json.loads(state_path.read_text(encoding="utf-8"))
    max_sites = int(state_config["max_sites"])
    reference = zero_baseline(config, identity, prepared, max_sites, source_root)
    keys = [tuple(key) for key in config["monitor_source_keys"]]
    if len(keys) != len(set(keys)) or reference["sources"] != config["validation_sources"]:
        raise ValueError("monitor source identity/coverage mismatch")
    comparisons = []
    for event in validation:
        bands = []
        for baseline in reference["bands"]:
            index = baseline["band"]
            prefix = f"t_band_{index}_geometry_"
            weight = float(event[prefix + "weight_sum"])
            if weight != baseline["weight"] or event[prefix+"real_states"] != len(keys):
                raise ValueError("saved and reconstructed validation populations differ")
            result = {**baseline, "model_v_MSE": event[prefix+"lattice_sum"] / weight,
                      "model_u_MSE": event[prefix+"coordinates_sum"] / weight}
            for field in ("v", "u"):
                zero, mse = result[f"zero_{field}_MSE"], result[f"model_{field}_MSE"]
                result[f"{field}_absolute_improvement_over_zero"] = zero - mse
                result[f"{field}_relative_improvement_over_zero"] = (1-mse/zero) if zero > 1e-12 else None
                result[f"{field}_prediction_RMS_lower_bound"] = abs(math.sqrt(mse)-math.sqrt(zero))
                result[f"{field}_prediction_RMS_upper_bound"] = math.sqrt(mse)+math.sqrt(zero)
            result["relative_u_skill_suppressed_for_near_zero_target"] = baseline["zero_u_MSE"] <= 1e-12
            bands.append(result)
        comparisons.append({"epoch": event["epoch"], "step": event["step"], "bands": bands})
    result = {"schema": "mixed_geometry_zero_validation_v1", "model_forwards": 0, "GPU_calls": 0,
              "training_code_commit": config["code_commit"], "train_dir": str(train),
              "training_config_sha256": file_hash(config_path), "validation_log_sha256": file_hash(log_path),
              "state_config_sha256": file_hash(state_path), "inputs": receipts,
              "normalizer_sha256_canonical_json": sha256(json.dumps(config["normalizer"], sort_keys=True,
                                                                     separators=(",", ":")).encode()).hexdigest(),
              "source_hashes": {relative: file_hash(source_root/relative) for relative in
                                ("src/crystal_dlm/mixed_geometry_diffusion.py",
                                 "src/crystal_dlm/mixed_geometry_training_data.py",
                                 "src/scripts/train_mixed_geometry_dlm.py")},
              "monitor_source_keys": [list(key) for key in keys], "monitor_source_count": len(keys),
              "exact_band_weight_coverage": True, "baseline": reference, "comparisons": comparisons,
              "limits": ["Zero-field comparison measures predictive skill on this saved validation population.",
                         "It does not determine Bayes irreducible variance or final SUN.",
                         "Prediction norm bounds use the L2 triangle inequality; they are not measured field norms."]}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/"ZERO_VALIDATION_BASELINE.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    lines = ["# Saved validation versus zero geometry fields", "", "CPU only; no model forwards or GPU calls.", "",
             "| Epoch | t | Zero v MSE | Model v MSE | v relative gain | Zero u MSE | Model u MSE | u relative gain |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for comparison in comparisons:
        for band in comparison["bands"]:
            gains = ["near-zero target" if band[f"{field}_relative_improvement_over_zero"] is None
                     else f"{100*band[f'{field}_relative_improvement_over_zero']:.5f}%" for field in ("v", "u")]
            lines.append(f"| {comparison['epoch']} | {band['time']:.6g} | {band['zero_v_MSE']:.9g} | "
                         f"{band['model_v_MSE']:.9g} | {gains[0]} | {band['zero_u_MSE']:.9g} | "
                         f"{band['model_u_MSE']:.9g} | {gains[1]} |")
    (args.output_dir/"ZERO_VALIDATION_BASELINE.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps({"model_forwards": 0, "GPU_calls": 0, "sources": len(keys),
                      "max_sites": max_sites, "torch": reference["torch_version"], "comparisons": comparisons},
                     ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
