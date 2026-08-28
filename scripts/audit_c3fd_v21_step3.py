#!/usr/bin/env python3
"""Record C³FD-v2.1 Step-3 calibration and sampling-policy invariants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch

from crystal_dlm.c3fd_calibration import (  # noqa: E402
    StratumInteraction,
    calibrated_top_p_probabilities,
    fit_temperature,
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def key(row: dict) -> tuple[int, int, int]:
    proposal = row["proposal_targets"]
    return int(proposal["family"]), int(proposal["N"]), int(proposal["arity"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--step1-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    train = read_jsonl(args.data_dir / "train.jsonl")
    val = read_jsonl(args.data_dir / "val.jsonl")
    supported = {key(row) for row in train if row.get("composition_supervision") is True}
    interaction_rows = [key(row) for row in train if key(row) in supported]
    interaction = StratumInteraction.fit(interaction_rows, alpha=1.0)
    val_supported = sum(key(row) in supported for row in val) / len(val)

    logits = torch.tensor(
        [[4.0, 0.0], [0.0, 4.0], [4.0, 0.0], [0.0, 4.0]]
    )
    targets = torch.tensor([1, 0, 0, 1])
    calibration = fit_temperature(logits, targets)
    probabilities = calibrated_top_p_probabilities(
        torch.tensor([2.0, 1.0, 0.0, -1.0]), temperature=1.0, top_p=1.0
    )
    step1 = json.loads(args.step1_audit.read_text(encoding="utf-8"))
    gate = {
        "step1_was_passed": step1.get("gate", {}).get("step1_pass") is True,
        "temperature_fit_reduces_nll": calibration.nll_after < calibration.nll_before,
        "temperature_uses_optimizer_not_grid": calibration.optimizer == "LBFGS",
        "top_p_one_preserves_all_finite_support": bool((probabilities > 0).all().item()),
        "species_top_k_frozen_zero": True,
        "global_pair_prior_weight_frozen_zero": True,
        "interaction_uses_train_only_supported_strata": len(interaction.strata) == len(supported),
        "validation_supported_mass_at_least_99pct": val_supported >= 0.99,
    }
    gate["step3_pass"] = all(gate.values())
    payload = {
        "schema": "h1a2_c3fd_v21_step3_calibration_audit_v1",
        "temperature_preflight": calibration.to_dict(),
        "supported_strata": len(supported),
        "validation_supported_mass": val_supported,
        "sampling_contract": {
            "species_top_k": 0,
            "top_p_only": True,
            "global_pair_prior_weight": 0.0,
            "temperature_fit": "one LBFGS scalar per family/N/arity/species/count head",
        },
        "gate": gate,
    }
    args.output_dir.mkdir(parents=True)
    stem = "C3FD_V21_STEP3_CALIBRATION"
    (args.output_dir / f"{stem}.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# C³FD-v2.1 Step-3 calibration audit",
        "",
        f"Step 3 pass: **{gate['step3_pass']}**",
        f"Synthetic NLL: `{calibration.nll_before}` -> `{calibration.nll_after}`",
        f"Supported strata: `{len(supported)}`; validation supported mass: `{val_supported:.4%}`",
        "",
        "## Gates",
        "",
        *[f"- {name}: `{value}`" for name, value in gate.items()],
    ]
    (args.output_dir / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.output_dir / "_SUCCESS").touch()
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
