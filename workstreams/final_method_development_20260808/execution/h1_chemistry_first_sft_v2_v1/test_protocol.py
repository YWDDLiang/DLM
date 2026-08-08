from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from crystal_dlm.h1_chemistry_first_sft import (
    H1_CHEMISTRY_FIRST_INFERENCE_MESSAGES_SHA256,
    H1_CHEMISTRY_FIRST_SFT_SEED,
)
from crystal_dlm.ordinal_rng import derive_ordinal_seed


ROOT = Path(__file__).resolve().parent


class ChemistryFirstExecutionProtocolTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = json.loads((ROOT / "CONFIG.json").read_text(encoding="utf-8"))
        self.authorization = json.loads(
            (ROOT / "AUTHORIZATION.json").read_text(encoding="utf-8")
        )

    def test_identity_and_protected_roots(self) -> None:
        self.assertEqual(self.config["identity"], "h1_chemistry_first_sft_v2_v1")
        self.assertEqual(
            self.config["p0_adapter_weight_sha256"],
            "65766c7485bd5ad8e180f3f5d99b83bef0488c251acd9278cb8bc2ad2518aa3a",
        )
        self.assertEqual(
            self.config["protected_body_b0_sha256"],
            "5c39976b6ab237cbab32cbfeb1c23a557571e1c7d2b60c1e60cbb450166ae76d",
        )
        self.assertEqual(
            self.config["protected_refiner_model494_sha256"],
            "573e9b10af64b266b7c6cde4d0f8bdd8a7388fa98d36e2e82db341af3e511e7e",
        )

    def test_both_candidates_always_execute_and_no_rl(self) -> None:
        self.assertEqual(
            self.authorization["always_execute_candidates"], ["sft_v2", "sft_v2_c"]
        )
        self.assertTrue(self.config["training"]["always_execute"])
        self.assertFalse(self.config["automatic_rl"])
        self.assertFalse(self.config["automatic_downstream"])
        self.assertFalse(self.config["body_dlm_changed"])

    def test_prompt_is_exact_six_line_nocharge(self) -> None:
        self.assertFalse(self.config["prompt"]["generated_charge_field"])
        self.assertEqual(
            self.config["prompt"]["generated_lines"],
            ["formula", "anion", "lattice", "spacegroup", "volume", "end"],
        )
        self.assertEqual(
            self.config["prompt"]["inference_messages_sha256"],
            H1_CHEMISTRY_FIRST_INFERENCE_MESSAGES_SHA256,
        )

    def test_one_epoch_optimizer_contract(self) -> None:
        training = self.config["training"]
        self.assertEqual(training["seed"], H1_CHEMISTRY_FIRST_SFT_SEED)
        self.assertEqual(training["batch_size"], 1)
        self.assertEqual(training["gradient_accumulation"], 8)
        self.assertEqual(training["learning_rate"], 2e-6)
        self.assertEqual(training["epochs"], 1)
        self.assertFalse(training["drop_last"])
        self.assertFalse(training["repeat_records"])
        self.assertFalse(training["intermediate_checkpoints"])
        self.assertFalse(training["generated_metric_checkpoint_selection"])

    def test_ledgers_are_exact_stateless_and_independent(self) -> None:
        seeds = set()
        for denominator, filename, key in (
            (64, "LEDGER64.json", "stage64_base_seed"),
            (256, "LEDGER256.json", "stage256_base_seed"),
        ):
            ledger = json.loads((ROOT / filename).read_text(encoding="utf-8"))
            base_seed = int(self.config["planner"][key])
            seeds.add(base_seed)
            self.assertEqual(ledger["denominator"], denominator)
            self.assertEqual(
                [row["ordinal"] for row in ledger["rows"]], list(range(denominator))
            )
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
        self.assertEqual(len(seeds), 2)
        self.assertNotIn(self.config["training"]["seed"], seeds)

    def test_resources_match_registered_limits(self) -> None:
        train = (ROOT / "train.sbatch").read_text(encoding="utf-8")
        smoke = (ROOT / "smoke.sbatch").read_text(encoding="utf-8")
        self.assertIn("#SBATCH --partition=gpu_long", train)
        self.assertIn("#SBATCH --cpus-per-task=4", train)
        self.assertIn("#SBATCH --cpus-per-task=4", smoke)
        for stage in (64, 256):
            planner = (ROOT / f"planner{stage}.sbatch").read_text(encoding="utf-8")
            self.assertIn("#SBATCH --cpus-per-task=8", planner)
            self.assertIn("#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1", planner)
        initial = (ROOT / "submit_once.sh").read_text(encoding="utf-8")
        self.assertIn("--array=0-1%2", initial)
        self.assertIn("--array=0-2%2", initial)

    def test_dual_runtime_data_firewall(self) -> None:
        data = (ROOT / "data.sbatch").read_text(encoding="utf-8")
        self.assertIn(
            '"${LEGACY_PYTHON}" scripts/export_h1_nocharge_mp20_legacy_snapshot.py',
            data,
        )
        self.assertIn(
            '"${SMACT4_PYTHON}" scripts/build_h1_chemistry_first_sft_data.py',
            data,
        )
        self.assertIn("--tokenizer-path", data)

    def test_exact_smact4_wheel_is_frozen_and_offline(self) -> None:
        runtime = self.config["shared_smact4_runtime"]
        wheel = ROOT / "vendor" / runtime["wheel"]
        bundle_freeze = json.loads(
            (ROOT / "SMACT4_RUNTIME_BUNDLE_FREEZE_RECORD.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            hashlib.sha256(wheel.read_bytes()).hexdigest(),
            runtime["wheel_sha256"],
        )
        self.assertFalse(runtime["network_install"])
        self.assertFalse(runtime["global_environment_mutation"])
        self.assertEqual(runtime["base_python_requirement"], ">=3.11,<3.14")
        self.assertEqual(runtime["portable_python_version"], "3.12.13")
        self.assertEqual(runtime["wheel_count"], 54)
        self.assertEqual(
            runtime["bundle_archive_sha256"], bundle_freeze["archive_sha256"]
        )
        self.assertEqual(
            runtime["bundle_manifest_sha256"],
            bundle_freeze["bundle_manifest_sha256"],
        )
        prepare = (ROOT / "prepare_shared_smact4_runtime.py").read_text(
            encoding="utf-8"
        )
        preflight = (ROOT / "preflight.py").read_text(encoding="utf-8")
        submissions = "".join(
            (ROOT / name).read_text(encoding="utf-8")
            for name in ("submit_once.sh", "submit_256_once.sh")
        )
        self.assertIn("safe_regular_tar_members", prepare)
        self.assertIn("safe_python_tar_members", prepare)
        self.assertIn('"--no-index"', prepare)
        self.assertIn('"--no-deps"', prepare)
        self.assertIn('"PYTHONNOUSERSITE": "1"', prepare)
        self.assertIn("build_root.rename(output_root)", prepare)
        self.assertNotIn("--base-python", prepare)
        self.assertIn("validate_shared_smact4_runtime", preflight)
        self.assertIn("shared_smact4_runtime_terminal", preflight)
        self.assertEqual(submissions.count('test -f "${SMACT4_RUNTIME_ROOT}/_SUCCESS"'), 2)

    def test_initial_submission_stops_at_raw64(self) -> None:
        initial = (ROOT / "submit_once.sh").read_text(encoding="utf-8")
        followup = (ROOT / "submit_256_once.sh").read_text(encoding="utf-8")
        self.assertIn("planner64.sbatch", initial)
        self.assertNotIn("planner256.sbatch", initial)
        self.assertIn("planner256.sbatch", followup)
        self.assertIn("passing_candidates", followup)
        self.assertNotIn("downstream.sbatch", initial + followup)
        self.assertNotIn("rl.sbatch", initial + followup)

    def test_fixed_adapter_resolver_supports_named_peft_subdirectory(self) -> None:
        from workstreams.final_method_development_20260808.execution.h1_chemistry_first_sft_v2_v1.resolve_fixed_adapter import (
            resolve_fixed_adapter,
        )

        with tempfile.TemporaryDirectory() as directory:
            training = Path(directory)
            endpoint = training / "checkpoint-000123"
            adapter = endpoint / "candidate"
            adapter.mkdir(parents=True)
            weight = adapter / "adapter_model.safetensors"
            weight.write_bytes(b"frozen-fixture-weight")
            digest = hashlib.sha256(weight.read_bytes()).hexdigest()
            (endpoint / "adapter_location.json").write_text(
                json.dumps(
                    {
                        "adapter_directory_relative": "candidate",
                        "weight_file": weight.name,
                    }
                ),
                encoding="utf-8",
            )
            (training / "terminal_report.json").write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "candidate": "sft_v2",
                        "generated_metric_checkpoint_selection": False,
                        "intermediate_checkpoint_count": 0,
                        "conditional_structural_anchor_nll_gate": {"passed": True},
                        "fixed_endpoint": endpoint.name,
                        "fixed_adapter_relative": "checkpoint-000123/candidate",
                        "fixed_adapter_weight_file": weight.name,
                        "fixed_adapter_weight_sha256": digest,
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(resolve_fixed_adapter(training, "sft_v2"), adapter)

    def test_anchor_scientific_stop_does_not_block_other_candidate(self) -> None:
        train = (ROOT / "train.sbatch").read_text(encoding="utf-8")
        planner = (ROOT / "planner64.sbatch").read_text(encoding="utf-8")
        assembly = (ROOT / "assemble_stage.sbatch").read_text(encoding="utf-8")
        self.assertIn("train_${CANDIDATE}_SCIENTIFIC_STOP", train)
        self.assertIn("SKIPPED_TRAINING_SCIENTIFIC_STOP", planner)
        self.assertIn("write_stage_training_stop.py", assembly)

    def test_sampling_contract_is_raw_one_sample(self) -> None:
        for stage in (64, 256):
            script = (ROOT / f"planner{stage}.sbatch").read_text(encoding="utf-8")
            self.assertIn("--batch-size 1", script)
            self.assertIn("--no-include-sample-id", script)
            self.assertIn("--seed-mode stateless_ordinal_v1", script)
            self.assertIn("--formula-constraint-mode off", script)
            self.assertIn("--temperature 0.9 --top-p 0.95 --top-k 50", script)

    def test_source_freeze_excludes_assets_and_runs(self) -> None:
        freeze = (ROOT / "freeze_source.py").read_text(encoding="utf-8")
        self.assertNotIn('root / "reference"', freeze)
        self.assertNotIn('root / "runs"', freeze)
        self.assertIn('root / "crystal_dlm"', freeze)
        self.assertIn('root / "scripts"', freeze)
        self.assertIn("WORKSTREAM_REL", freeze)


if __name__ == "__main__":
    unittest.main()
