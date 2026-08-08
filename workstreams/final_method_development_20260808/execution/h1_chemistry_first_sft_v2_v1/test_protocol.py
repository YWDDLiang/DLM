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
        self.assertEqual(
            self.config["identity"],
            "h1_chemistry_first_sft_v2_smact_split_v2",
        )
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
        engineering = (ROOT / "submit_once.sh").read_text(encoding="utf-8")
        science64 = (ROOT / "submit_training64_once.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("--array=0-1%2", engineering)
        self.assertIn("--array=0-1%2", science64)
        self.assertIn("--array=0-2%2", science64)

    def test_reference_adapter_uses_same_fp32_load_path_then_freezes(self) -> None:
        trainer = (
            ROOT.parents[3] / "scripts" / "llama_h1_chemistry_first_sft.py"
        ).read_text(encoding="utf-8")
        reference_load = trainer.split('adapter_name="reference",', 1)[1].split(
            ")", 1
        )[0]
        self.assertIn("is_trainable=True", reference_load)
        self.assertIn("autocast_adapter_dtype=True", reference_load)
        self.assertIn(
            'if ".reference." in name:\n            parameter.requires_grad_(False)',
            trainer,
        )

    def test_cross_machine_evaluator_firewall(self) -> None:
        snapshot = (ROOT / "snapshot.sbatch").read_text(encoding="utf-8")
        data = (ROOT / "data.sbatch").read_text(encoding="utf-8")
        self.assertIn(
            '"${LEGACY_PYTHON}" scripts/export_h1_nocharge_mp20_legacy_snapshot.py',
            snapshot,
        )
        self.assertIn(
            '"${LEGACY_PYTHON}" scripts/build_h1_chemistry_first_sft_data.py',
            data,
        )
        self.assertIn("--smact4-witness-bundle-dir", data)
        self.assertIn("--expected-source-inventory-sha256", data)
        self.assertIn("--tokenizer-path", data)
        assembly = (ROOT / "assemble_stage.sbatch").read_text(encoding="utf-8")
        self.assertIn("verify_h1_local_smact4_stage_audit_bundle.py", assembly)
        a800_scripts = "".join(
            path.read_text(encoding="utf-8")
            for path in ROOT.iterdir()
            if path.suffix in {".sh", ".sbatch"}
        )
        self.assertNotIn("SMACT4_PYTHON", a800_scripts)
        self.assertNotIn("audit_h1_chemistry_first_planner_smact4.py", assembly)
        for obsolete_runtime_entry in (
            "prepare_runtime_on_a800.sh",
            "prepare_shared_smact4_runtime.py",
            "freeze_smact4_runtime_bundle.py",
            "shared_smact4_python.sh",
            "smact4_runtime_requirements.in",
        ):
            self.assertFalse((ROOT / obsolete_runtime_entry).exists())

    def test_exact_smact4_is_local_only_and_sha_bound(self) -> None:
        runtime = self.config["local_smact4_ledger"]
        wheel = ROOT / "vendor" / runtime["wheel"]
        self.assertEqual(
            hashlib.sha256(wheel.read_bytes()).hexdigest(),
            runtime["wheel_sha256"],
        )
        self.assertEqual(runtime["execution_location"], "local_windows_only")
        self.assertFalse(runtime["a800_execution"])
        self.assertFalse(self.config["a800_smact4_execution"])
        self.assertEqual(runtime["contract_sha256"], self.config["evaluators"]["secondary"]["contract_sha256"])
        preflight = (ROOT / "preflight.py").read_text(encoding="utf-8")
        submissions = "".join(
            (ROOT / name).read_text(encoding="utf-8")
            for name in (
                "submit_snapshot_once.sh",
                "submit_once.sh",
                "submit_training64_once.sh",
                "submit_assemble64_once.sh",
                "submit_256_once.sh",
                "submit_assemble256_once.sh",
            )
        )
        self.assertNotIn("validate_shared_smact4_runtime", preflight)
        self.assertNotIn("SMACT4_PYTHON", submissions)
        self.assertNotIn("smact4-python", submissions)
        self.assertIn("LOCAL_SMACT4_WITNESS_ROOT", submissions)
        self.assertIn("LOCAL_SMACT4_AUDIT_ROOT", submissions)

    def test_submissions_stop_at_each_local_smact4_boundary(self) -> None:
        snapshot = (ROOT / "submit_snapshot_once.sh").read_text(encoding="utf-8")
        engineering = (ROOT / "submit_once.sh").read_text(encoding="utf-8")
        initial = (ROOT / "submit_training64_once.sh").read_text(encoding="utf-8")
        assembly64 = (ROOT / "submit_assemble64_once.sh").read_text(encoding="utf-8")
        followup = (ROOT / "submit_256_once.sh").read_text(encoding="utf-8")
        assembly256 = (ROOT / "submit_assemble256_once.sh").read_text(encoding="utf-8")
        self.assertIn("snapshot.sbatch", snapshot)
        self.assertIn("smoke.sbatch", engineering)
        self.assertNotIn("train.sbatch", engineering)
        self.assertNotIn("planner64.sbatch", engineering)
        self.assertIn("sacct_smoke_before_science64.txt", initial)
        self.assertIn('test "${matches}" = "COMPLETED|0:0"', initial)
        self.assertIn("train.sbatch", initial)
        self.assertIn("planner64.sbatch", initial)
        self.assertNotIn("assemble64.sbatch", initial)
        self.assertNotIn("planner256.sbatch", initial)
        self.assertIn("assemble64.sbatch", assembly64)
        self.assertIn("sacct_planner64_before_assembly.txt", assembly64)
        self.assertIn("planner256.sbatch", followup)
        self.assertNotIn("assemble256.sbatch", followup)
        self.assertIn("assemble256.sbatch", assembly256)
        self.assertIn("sacct_planner256_before_assembly.txt", assembly256)
        self.assertIn("passing_candidates", followup)
        all_submissions = snapshot + engineering + initial + assembly64 + followup + assembly256
        self.assertNotIn("downstream.sbatch", all_submissions)
        self.assertNotIn("rl.sbatch", all_submissions)

    def test_v4_smoke_repair_reuses_data_but_resubmits_gpu_smoke(self) -> None:
        repair = json.loads(
            (ROOT / "SMOKE_IDENTITY_REPAIR_V4.json").read_text(encoding="utf-8")
        )
        self.assertFalse(repair["repair"]["models_changed"])
        self.assertFalse(repair["repair"]["data_changed"])
        self.assertFalse(repair["repair"]["optimizer_changed"])
        self.assertFalse(repair["repair"]["smact4_execution_on_a800"])
        reuse = (ROOT / "reuse_parent_data_on_a800.sh").read_text(encoding="utf-8")
        smoke = (ROOT / "submit_identity_repair_smoke_once.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("byte_identical_reuse_only", reuse)
        self.assertIn("EXPECTED_PARENT_DATA_AUDIT_SHA256", reuse)
        self.assertNotIn("data.sbatch", smoke)
        self.assertIn("smoke.sbatch", smoke)
        self.assertNotIn("train.sbatch", smoke)
        self.assertNotIn("planner64.sbatch", smoke)
        self.assertNotIn("SMACT4_PYTHON", reuse + smoke)

    def test_v5_source_gate_path_repair_is_closed(self) -> None:
        repair = json.loads(
            (ROOT / "SOURCE_GATE_PATH_REPAIR_V5.json").read_text(encoding="utf-8")
        )
        self.assertFalse(repair["repair"]["models_changed"])
        self.assertFalse(repair["repair"]["training_code_changed"])
        self.assertFalse(repair["repair"]["data_changed"])
        self.assertFalse(repair["repair"]["smact4_execution_on_a800"])
        old_run = "20260808_h1_chemistry_first_sft_v2_smact_split_v2_identity_repair_v4"
        new_run = "20260808_h1_chemistry_first_sft_v2_smact_split_v2_source_gate_path_repair_v5"
        active_paths = [ROOT / "CONFIG.json"]
        active_paths.extend(ROOT.glob("*.sh"))
        active_paths.extend(ROOT.glob("*.sbatch"))
        for path in active_paths:
            text = path.read_text(encoding="utf-8")
            self.assertNotIn(old_run, text, path.name)
        self.assertEqual(self.config["run_root"].rsplit("/", 1)[-1], new_run)
        audit = (ROOT / "audit_source_on_a800.sh").read_text(encoding="utf-8")
        self.assertIn('ISOLATED_ROOT="${RUN_ROOT}/isolated_archive_test"', audit)
        bootstrap = (ROOT / "bootstrap_source_on_a800.sh").read_text(encoding="utf-8")
        self.assertIn("SOURCE_GATE_PATH_REPAIR_V5.json", bootstrap)
        self.assertNotIn("smact4_400_runtime", audit + bootstrap)

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
