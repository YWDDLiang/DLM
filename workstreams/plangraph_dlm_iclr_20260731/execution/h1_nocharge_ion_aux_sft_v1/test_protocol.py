from __future__ import annotations

import json
from pathlib import Path
import unittest

from crystal_dlm.h1_llm_planner import (
    H1_PLANNER_PROMPT_STYLE_RICH_NOCHARGE,
    H1_PLANNER_PROMPT_STYLE_RICH_PLAN,
)
from crystal_dlm.ordinal_rng import derive_ordinal_seed


ROOT = Path(__file__).resolve().parent


class NochargeIonAuxExecutionProtocolTest(unittest.TestCase):
    def test_config_keeps_dlm_and_automatic_rl_frozen(self) -> None:
        config = json.loads((ROOT / "CONFIG.json").read_text(encoding="utf-8"))
        authorization = json.loads((ROOT / "AUTHORIZATION.json").read_text(encoding="utf-8"))
        self.assertFalse(config["downstream"]["body_dlm_changed"])
        self.assertFalse(config["automatic_rl"])
        self.assertFalse(config["automatic_downstream"])
        self.assertFalse(authorization["explicit_constraints"]["generated_charge_field"])
        self.assertEqual(config["training"]["fixed_endpoint"], "checkpoint-0400")
        self.assertEqual(
            config["training"]["batch_size"]
            * config["training"]["gradient_accumulation"]
            * config["training"]["updates"],
            config["data"]["train_records_per_arm"],
        )

    def test_ledgers_are_exact_independent_and_stateless(self) -> None:
        config = json.loads((ROOT / "CONFIG.json").read_text(encoding="utf-8"))
        observed_base_seeds: set[int] = set()
        for denominator, filename, seed_key in (
            (64, "LEDGER64.json", "stage64_base_seed"),
            (256, "LEDGER256.json", "stage256_base_seed"),
        ):
            ledger = json.loads((ROOT / filename).read_text(encoding="utf-8"))
            base_seed = int(config["planner"][seed_key])
            observed_base_seeds.add(base_seed)
            self.assertEqual(ledger["denominator"], denominator)
            self.assertTrue(ledger["independent_from_training_and_other_stages"])
            self.assertEqual([row["ordinal"] for row in ledger["rows"]], list(range(denominator)))
            self.assertEqual(
                [row["planner_sampling_seed"] for row in ledger["rows"]],
                [
                    derive_ordinal_seed(
                        base_seed,
                        sample_idx=ordinal,
                        stage="planner_sampling",
                        role="shared",
                    )
                    for ordinal in range(denominator)
                ],
            )
        self.assertEqual(len(observed_base_seeds), 2)
        self.assertNotIn(config["training"]["seed"], observed_base_seeds)

    def test_initial_submission_stops_at_raw64(self) -> None:
        initial = (ROOT / "submit_once.sh").read_text(encoding="utf-8")
        followup = (ROOT / "submit_256_once.sh").read_text(encoding="utf-8")
        self.assertIn("planner64.sbatch", initial)
        self.assertNotIn("planner256.sbatch", initial)
        self.assertIn("planner256.sbatch", followup)
        self.assertIn("planner64/_SUCCESS", followup)
        self.assertIn("planner_gate_pass", followup)
        self.assertNotIn("downstream.sbatch", initial + followup)
        self.assertNotIn("rl.sbatch", initial + followup)

    def test_dual_evaluator_firewall_is_explicit(self) -> None:
        data = (ROOT / "data.sbatch").read_text(encoding="utf-8")
        self.assertIn("${LEGACY_PYTHON}\" scripts/export_h1_nocharge_mp20_legacy_snapshot.py", data)
        self.assertIn("${SMACT4_PYTHON}\" scripts/build_h1_nocharge_ion_aux_sft_data.py", data)
        self.assertIn("--legacy-snapshot-dir", data)
        self.assertIn("${LEGACY_PYTHON}\" scripts/audit_h1_nocharge_sft_tokenizer.py", data)

    def test_prompt_and_sampling_contracts(self) -> None:
        self.assertEqual(H1_PLANNER_PROMPT_STYLE_RICH_PLAN, "h1_rich_plan_v1")
        self.assertEqual(
            H1_PLANNER_PROMPT_STYLE_RICH_NOCHARGE,
            "h1_rich_nocharge_plan_v1",
        )
        resolver = (ROOT / "resolve_fixed_adapter.py").read_text(encoding="utf-8")
        self.assertIn("checkpoint-0400", resolver)
        for stage in (64, 256):
            script = (ROOT / f"planner{stage}.sbatch").read_text(encoding="utf-8")
            self.assertIn("--no-include-sample-id", script)
            self.assertIn("--seed-mode stateless_ordinal_v1", script)
            self.assertIn("--formula-constraint-mode off", script)
            self.assertIn("--temperature 0.9 --top-p 0.95 --top-k 50", script)
            self.assertIn("resolve_fixed_adapter.py", script)

    def test_source_freeze_excludes_data_models_and_runs(self) -> None:
        freeze = (ROOT / "freeze_source.py").read_text(encoding="utf-8")
        self.assertNotIn('root / "reference"', freeze)
        self.assertNotIn('root / "runs"', freeze)
        self.assertIn('root / "crystal_dlm"', freeze)
        self.assertIn('root / "scripts"', freeze)
        self.assertIn('root / "tests"', freeze)


if __name__ == "__main__":
    unittest.main()
