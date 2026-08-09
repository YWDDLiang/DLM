#!/usr/bin/env python3
"""Score a later DLM checkpoint on the byte-frozen B0 state panels."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import torch  # noqa: E402

from evaluate_state_panels import (  # noqa: E402
    aggregate_scores,
    assert_body_tokenizer_identity,
    load_model_and_tokenizer,
    read_json,
    read_jsonl,
    require_sha,
    score_states,
    sha256_file,
    validate_runtime,
    write_json_exclusive,
    write_jsonl_exclusive,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-adapter-sha256", required=True)
    parser.add_argument("--frozen-panel-root", type=Path, required=True)
    parser.add_argument("--frozen-panel-manifest-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    started = time.monotonic()
    device = validate_runtime()
    config = read_json(args.config.resolve())
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    panel_root = args.frozen_panel_root.resolve()
    manifest_path = require_sha(
        panel_root / "state_panel_manifest.json",
        args.frozen_panel_manifest_sha256,
        "frozen B0 state-panel manifest",
    )
    manifest = read_json(manifest_path)
    if (
        manifest.get("status") != "complete"
        or manifest.get("checkpoint_arm") != "B0"
        or manifest.get("automatic_b3_submission") is not False
    ):
        raise ValueError("frozen B0 state-panel manifest changed")
    files = manifest["files"]
    synthetic_path = require_sha(
        panel_root / "synthetic_states.jsonl",
        files["synthetic_states.jsonl"],
        "frozen synthetic states",
    )
    actual_path = require_sha(
        panel_root / "actual_rollout_states.jsonl",
        files["actual_rollout_states.jsonl"],
        "frozen actual rollout states",
    )
    b0_terminal_path = panel_root / "terminal_report.json"
    if not b0_terminal_path.is_file():
        raise FileNotFoundError("frozen B0 state-panel terminal is missing")
    b0_terminal = read_json(b0_terminal_path)
    if b0_terminal.get("status") != "complete":
        raise ValueError("frozen B0 state-panel terminal is incomplete")

    checkpoint = args.checkpoint.resolve()
    adapter = checkpoint / "adapter_model.safetensors"
    if not adapter.is_file() or adapter.stat().st_size <= 0:
        raise ValueError("candidate adapter is missing or empty")
    model, tokenizer = load_model_and_tokenizer(
        str(Path(config["base_model"]).resolve()), str(checkpoint), device
    )
    tokenizer_identity = assert_body_tokenizer_identity(
        tokenizer,
        expected_vocab_sha256=config["checkpoint"]["tokenizer_vocab_sha256"],
    )
    if int(tokenizer_identity["vocab_size"]) != int(
        config["checkpoint"]["tokenizer_size"]
    ):
        raise ValueError("candidate tokenizer identity changed")
    write_json_exclusive(output / "tokenizer_identity.json", tokenizer_identity)

    states = [*read_jsonl(synthetic_path), *read_jsonl(actual_path)]
    if len({str(row["state_id"]) for row in states}) != len(states):
        raise ValueError("frozen state IDs are not unique")
    scored = score_states(
        states,
        model=model,
        batch_size=int(config["decoder"]["score_batch_size"]),
    )
    score_path = output / f"{args.arm}_state_scores.jsonl"
    write_jsonl_exclusive(score_path, scored)
    states_by_id = {str(row["state_id"]): row for row in states}
    summary = aggregate_scores(scored, states_by_id=states_by_id)
    if set(summary) != set(config["panels"]["panel_types"]):
        raise ValueError("candidate scoring lost a panel")
    if any(
        not math.isfinite(panel["token_weighted_mean_nll"])
        for panel in summary.values()
    ):
        raise ValueError("candidate state-panel NLL is non-finite")

    b0_summary = b0_terminal["summary"]
    safe_delta = (
        summary["safe_axis_synthetic"]["token_weighted_mean_nll"]
        - b0_summary["safe_axis_synthetic"]["token_weighted_mean_nll"]
    )
    actual_delta = (
        summary["safe_axis_actual_b0"]["token_weighted_mean_nll"]
        - b0_summary["safe_axis_actual_b0"]["token_weighted_mean_nll"]
    )
    report = {
        "schema": "evidence_first_dlm_frozen_state_panel_candidate_terminal_v1",
        "status": "complete",
        "arm": str(args.arm),
        "checkpoint_path": str(checkpoint),
        "checkpoint_adapter_sha256_recorded": str(
            args.checkpoint_adapter_sha256
        ),
        "frozen_panel_manifest_sha256": sha256_file(manifest_path),
        "state_scores_sha256": sha256_file(score_path),
        "summary": summary,
        "versus_b0": {
            "safe_axis_synthetic_token_weighted_nll_delta": safe_delta,
            "safe_axis_actual_b0_token_weighted_nll_delta": actual_delta,
            "safe_axis_synthetic_point_improved": safe_delta < 0.0,
            "safe_axis_actual_point_improved": actual_delta < 0.0,
            "both_required_point_estimates_improved": (
                safe_delta < 0.0 and actual_delta < 0.0
            ),
        },
        "cuda_peak_memory_bytes": int(torch.cuda.max_memory_reserved(0)),
        "walltime_s": time.monotonic() - started,
        "training": False,
        "sun": False,
        "automatic_ratio_or_downstream_submission": False,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    write_json_exclusive(output / "terminal_report.json", report)
    (output / "_SUCCESS").touch(exist_ok=False)
    print(json.dumps(report, indent=2, sort_keys=True))
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
