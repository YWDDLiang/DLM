#!/usr/bin/env python3
"""Record the C³FD-v2.1 Step-2 model invariant audit."""

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

from crystal_dlm.ccfd import FormulaToken  # noqa: E402
from crystal_dlm.ccfd_v2 import (  # noqa: E402
    BenchmarkReachability,
    CCFDv2State,
    SetAtomCount,
)
from crystal_dlm.semantic_composition_head import SemanticCompositionHead  # noqa: E402


def benchmark_true(_elements, _counts):
    return {"valid": True, "reason": "synthetic_invariant"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    torch.manual_seed(20260828)
    head = SemanticCompositionHead(
        hidden_size=8,
        num_species=3,
        num_families=7,
        max_arity=7,
        ledger_feature_size=6,
        decoder_layers=1,
        decoder_heads=2,
        decoder_dropout=0.0,
    )
    head.eval()
    hidden = torch.zeros(2, 3, 8)
    previous_species = torch.tensor([[-1, -1, 0], [-1, -1, 0]])
    previous_counts = torch.tensor([[0, 0, 1], [0, 0, 1]])
    previous_n = torch.tensor([[0, 2, 0], [0, 2, 0]])
    ledger = torch.zeros(2, 3, 6)
    ledger[1, 1:, 0] = 1.0
    with torch.inference_mode():
        output = head(
            hidden,
            previous_species_indices=previous_species,
            previous_count_values=previous_counts,
            previous_n_values=previous_n,
            ledger_features=ledger,
        )
    ledger_delta = float(
        (output.species_logits[0] - output.species_logits[1]).abs().max().item()
    )

    oxygen = FormulaToken.from_symbol("O", -2, 2)
    iron = FormulaToken.from_symbol("Fe", 2, 2)
    oracle = BenchmarkReachability(
        ((oxygen.atomic_number, oxygen.oxidation_state), (iron.atomic_number, iron.oxidation_state))
    )
    partial = CCFDv2State.start().apply(SetAtomCount(4)).apply(oxygen)
    unconstrained = oracle.legal_species_counts(
        partial, benchmark_validator=benchmark_true, max_species=3
    )
    exact_three = oracle.legal_species_counts(
        partial,
        benchmark_validator=benchmark_true,
        max_species=3,
        target_arity=3,
    )
    terminal = partial.apply(iron).end()
    certificate = terminal.certificate(benchmark_validator=benchmark_true)
    gate = {
        "family_head_shape_7": tuple(output.family_logits.shape) == (2, 7),
        "arity_head_shape_7": tuple(output.arity_logits.shape) == (2, 7),
        "ledger_changes_species_logits": ledger_delta > 0.0,
        "shorter_complete_path_legal_without_exact_arity": iron in unconstrained,
        "shorter_complete_path_masked_for_target_arity3": iron not in exact_three,
        "N_charge_certificate_unchanged": bool(
            certificate.benchmark_compatible
            and terminal.target_atoms == 4
            and terminal.emitted_atoms == 4
            and terminal.net_charge == 0
        ),
    }
    gate["step2_pass"] = all(gate.values())
    payload = {
        "schema": "h1a2_c3fd_v21_step2_model_audit_v1",
        "ledger_feature_order": [
            "remaining_atoms_over_20",
            "net_charge_over_160",
            "remaining_species_over_7",
            "branch_unset",
            "branch_ionic",
            "branch_alloy",
        ],
        "ledger_max_logit_delta": ledger_delta,
        "gate": gate,
    }
    args.output_dir.mkdir(parents=True)
    stem = "C3FD_V21_STEP2_MODEL_AUDIT"
    (args.output_dir / f"{stem}.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# C³FD-v2.1 Step-2 model audit",
        "",
        f"Step 2 pass: **{gate['step2_pass']}**",
        f"Ledger intervention max species-logit delta: `{ledger_delta}`",
        "",
        "## Gates",
        "",
        *[f"- {key}: `{value}`" for key, value in gate.items()],
    ]
    (args.output_dir / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.output_dir / "_SUCCESS").touch()
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
