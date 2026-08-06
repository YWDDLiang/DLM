from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from crystal_dlm.wqcodiff.crysllmgen.gate import (
    GateALock,
    PATCH_ALLOWED_AUTHORIZATIONS,
    audit_authorized_patch_record,
    audit_source_sync_record,
    build_gate_a_lock,
)
from scripts.a800.install_authorized_patch import ALLOWED_AUTHORIZATIONS


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/experiments/wyckoff_codiffusion/protocol_v4.yaml"
REGISTRY = ROOT / "configs/experiments/wyckoff_codiffusion/experiment_registry_v2.yaml"


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class CrysLLMGenGateALockTests(unittest.TestCase):
    def test_thread_environment_supersession_authorization_is_registered(self) -> None:
        self.assertIn(
            "user_thread_env_superseding_diagnostics_2026-07-22",
            PATCH_ALLOWED_AUTHORIZATIONS,
        )

    def test_parent_csp_sun256_authorization_is_registered(self) -> None:
        self.assertIn(
            "user_wq_parent_csp_sun256_v1_2026-07-24",
            PATCH_ALLOWED_AUTHORIZATIONS,
        )

    def test_iclr_mlip_free_mechanism_authorization_is_registered(self) -> None:
        self.assertIn(
            "user_iclr_mlip_free_wq_mechanism64_2026-07-25",
            PATCH_ALLOWED_AUTHORIZATIONS,
        )

    def test_existing22_chgnet_sun_authorization_is_registered(self) -> None:
        self.assertIn(
            "user_wq_existing22_chgnet_sun_v1_2026-07-25",
            PATCH_ALLOWED_AUTHORIZATIONS,
        )
        self.assertIn(
            "user_wq_existing22_mp_completion_v1_2026-07-25",
            PATCH_ALLOWED_AUTHORIZATIONS,
        )

    def test_bridge_parity_sup28054_authorization_is_registered(self) -> None:
        self.assertIn(
            "user_wq_schedule_correct_bridge_parity_sup28054_v1_2026-07-26",
            PATCH_ALLOWED_AUTHORIZATIONS,
        )

    def test_wyckoff_tangent_remote_install_authorization_is_registered(
        self,
    ) -> None:
        self.assertIn(
            "user_wq_wyckoff_tangent_bridge_preflight_v1_remote_install_2026-07-26",
            PATCH_ALLOWED_AUTHORIZATIONS,
        )

    def test_wyckoff_tangent_audit_amendment_and_submit_is_registered(
        self,
    ) -> None:
        self.assertIn(
            "user_wq_wyckoff_tangent_bridge_preflight_v1_audit_amendment_and_submit_2026-07-26",
            PATCH_ALLOWED_AUTHORIZATIONS,
        )

    def test_wyckoff_chart_retraction_v2_authorization_is_registered(
        self,
    ) -> None:
        self.assertIn(
            "user_wq_wyckoff_chart_retraction_preflight_sup28185_v2_2026-07-26",
            PATCH_ALLOWED_AUTHORIZATIONS,
        )

    def test_wtb256_and_sup28194_authorizations_are_registered(self) -> None:
        expected = {
            (
                "user_wq_wyckoff_chart_retraction_confirmatory256_v1_"
                "local_preparation_2026-07-26"
            ),
            (
                "user_wq_wyckoff_chart_retraction_confirmatory256_"
                "sup28194_v1_2026-07-27"
            ),
        }
        self.assertTrue(expected.issubset(PATCH_ALLOWED_AUTHORIZATIONS))
        self.assertTrue(expected.issubset(ALLOWED_AUTHORIZATIONS))

    def test_wtb_identity_mechanics_sup28195_authorization_is_registered(
        self,
    ) -> None:
        authorization = (
            "user_wq_wyckoff_identity_mechanics_sup28195_v1_2026-07-27"
        )
        self.assertIn(authorization, PATCH_ALLOWED_AUTHORIZATIONS)
        self.assertIn(authorization, ALLOWED_AUTHORIZATIONS)

    def test_wq_charge_stop_paired64_authorization_is_registered(self) -> None:
        authorization = "user_wq_llm_charge_stop_paired64_v1_2026-07-27"
        self.assertIn(authorization, PATCH_ALLOWED_AUTHORIZATIONS)
        self.assertIn(authorization, ALLOWED_AUTHORIZATIONS)

    def test_wq_formula_plan_sft_pilot_authorization_is_registered(self) -> None:
        authorization = "user_wq_formula_plan_sft_pilot_v1_2026-07-27"
        self.assertIn(authorization, PATCH_ALLOWED_AUTHORIZATIONS)
        self.assertIn(authorization, ALLOWED_AUTHORIZATIONS)

    def test_wq_formula_plan_one_epoch_authorization_is_registered(self) -> None:
        authorization = "user_wq_formula_plan_sft_one_epoch_v2_2026-07-27"
        self.assertIn(authorization, PATCH_ALLOWED_AUTHORIZATIONS)
        self.assertIn(authorization, ALLOWED_AUTHORIZATIONS)

    def test_installer_and_runtime_gate_authorization_registries_match(
        self,
    ) -> None:
        self.assertSetEqual(
            set(ALLOWED_AUTHORIZATIONS),
            set(PATCH_ALLOWED_AUTHORIZATIONS),
        )

    def test_authorized_patch_preserves_base_identity_and_locks_overlay(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_file = root / "scripts" / "fixture.py"
            source_file.parent.mkdir(parents=True)
            source_file.write_text("base\n", encoding="utf-8")
            base_sha = hashlib.sha256(source_file.read_bytes()).hexdigest()
            source_record = _write(
                root / "source.json",
                {
                    "schema": "wqcodiff_source_manifest_v1",
                    "bundle_sha256": "a" * 64,
                    "files": [
                        {
                            "path": "scripts/fixture.py",
                            "bytes": source_file.stat().st_size,
                            "sha256": base_sha,
                        }
                    ],
                },
            )
            source_file.write_text("authorized patch\n", encoding="utf-8")
            patched_sha = hashlib.sha256(source_file.read_bytes()).hexdigest()
            manifest_sha = "b" * 64
            record = (
                root
                / ".artifacts"
                / "source_sync"
                / f"authorized_patch_{manifest_sha}.json"
            )
            record.parent.mkdir(parents=True)
            _write(
                record,
                {
                    "schema": "wqcodiff_authorized_patch_v1",
                    "authorization": "user_r5c_a100_sun_and_three_epoch_checkpoint_selection_2026-07-21",
                    "base_source_bundle_sha256": "a" * 64,
                    "manifest_sha256": manifest_sha,
                    "files": [
                        {
                            "path": "scripts/fixture.py",
                            "bytes": source_file.stat().st_size,
                            "sha256": patched_sha,
                        }
                    ],
                },
            )
            patch = audit_authorized_patch_record(
                project_root=root,
                manifest_sha256=manifest_sha,
                base_source_bundle_sha256="a" * 64,
            )
            self.assertTrue(patch["ok"])
            source = audit_source_sync_record(
                source_record,
                project_root=root,
                authorized_overrides=patch["overrides"],
            )
            self.assertTrue(source["ok"])
            source_file.write_text("tampered\n", encoding="utf-8")
            self.assertFalse(
                audit_authorized_patch_record(
                    project_root=root,
                    manifest_sha256=manifest_sha,
                    base_source_bundle_sha256="a" * 64,
                )["ok"]
            )

    def test_complete_gate_lock_binds_source_and_wq_smoke_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            readme = ROOT / "README.md"
            source = _write(
                work / "source.json",
                {
                    "schema": "wqcodiff_source_manifest_v1",
                    "bundle_sha256": "a" * 64,
                    "files": [
                        {
                            "path": "README.md",
                            "bytes": readme.stat().st_size,
                            "sha256": hashlib.sha256(readme.read_bytes()).hexdigest(),
                        }
                    ],
                },
            )
            parity = _write(
                work / "parity.json",
                {
                    "schema": "crysllmgen_disabled_extension_parity_audit_v1",
                    "ok": True,
                    "errors": [],
                },
            )
            llama = _write(
                work / "llama.json",
                {
                    "schema": "crysllmgen_llama_gate_a_report_v1",
                    "ok": True,
                    "offline": True,
                    "blas_threads": 1,
                    "model": {"adapter_changes_logits": True},
                },
            )
            grammar = _write(
                work / "grammar.json",
                {
                    "schema": "crysllmgen_wq_grammar_gate_report_v1",
                    "ok": True,
                    "transitions": {"transitions": 1_000_000, "illegal_generated": 0},
                    "catalog": {"roundtrip_passed": 230},
                },
            )
            atom = _write(
                work / "atom.json",
                {
                    "schema": "crysllmgen_lora_training_report_v1",
                    "run_role": "smoke",
                    "representation": "atom",
                    "optimizer": {"completed_global_step": 100},
                    "runtime": {"threads": 1, "offline": True},
                    "model": {"adapter_sha256": "b" * 64},
                },
            )
            wq = _write(
                work / "wq.json",
                {
                    "schema": "crysllmgen_lora_training_report_v1",
                    "run_role": "smoke",
                    "representation": "wyckoff",
                    "optimizer": {"completed_global_step": 100},
                    "runtime": {"threads": 1, "offline": True},
                    "model": {"adapter_sha256": "c" * 64},
                },
            )
            constrained = _write(
                work / "constrained.json",
                {
                    "schema": "crysllmgen_constrained_gate_report_v1",
                    "ok": True,
                    "submitted_attempts": 256,
                    "terminal_attempts": 256,
                    "parsed_attempts": 256,
                    "topology_legal_attempts": 256,
                    "retry_or_replacement_used": False,
                    "model": {"adapter_model_sha256": "c" * 64},
                },
            )
            payload = build_gate_a_lock(
                project_root=ROOT,
                source_sync_record=source,
                protocol_path=PROTOCOL,
                registry_path=REGISTRY,
                parity_audit_path=parity,
                llama_report_path=llama,
                grammar_report_path=grammar,
                atom_smoke_report_path=atom,
                wq_smoke_report_path=wq,
                constrained_report_path=constrained,
            )
            self.assertTrue(payload["ok"])
            lock = _write(work / "gate_a.lock.json", payload)
            loaded = GateALock.load(
                lock,
                project_root=ROOT,
                protocol_path=PROTOCOL,
            )
            self.assertEqual(loaded.source_bundle_sha256, "a" * 64)
            constrained.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                GateALock.load(lock, project_root=ROOT, protocol_path=PROTOCOL)


if __name__ == "__main__":
    unittest.main()
