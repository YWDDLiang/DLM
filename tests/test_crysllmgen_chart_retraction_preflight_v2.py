from __future__ import annotations

import ast
import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = (
    ROOT
    / "scripts"
    / "a800"
    / "run_wq_wyckoff_chart_retraction_preflight_sup28185_v2.py"
)
CONTRACT = (
    ROOT
    / "configs"
    / "experiments"
    / "wyckoff_codiffusion"
    / "wq_wyckoff_chart_retraction_preflight_sup28185_v2.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ChartRetractionPreflightV2StaticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = RUNNER.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)
        cls.contract = json.loads(CONTRACT.read_text(encoding="utf-8"))

    def test_new_identity_is_local_only_and_not_confirmatory(self) -> None:
        self.assertEqual(
            self.contract["identity"],
            "wq_wyckoff_chart_retraction_preflight_sup28185_v2",
        )
        self.assertEqual(self.contract["supersedes_job_id"], 28185)
        self.assertEqual(
            self.contract["status"],
            "local_built_remote_execution_not_authorized",
        )
        self.assertFalse(
            self.contract["authorization"]["slurm_submission_authorized"]
        )
        evidence = self.contract["evidence_classification"]
        self.assertTrue(evidence["development_panel_reused"])
        self.assertFalse(evidence["confirmatory_evidence"])
        self.assertEqual(
            evidence["reuse_purpose"],
            "mechanics_regression_only",
        )

    def test_contract_hash_binds_every_execution_source(self) -> None:
        implementation = self.contract["implementation"]
        for field in (
            "v2_runner_source",
            "legacy_execution_engine_source",
            "tangent_bridge_source",
            "runtime_source",
        ):
            self.assertEqual(
                _sha256(ROOT / implementation[field]),
                implementation[f"{field}_sha256"],
            )
        self.assertEqual(
            implementation["method"],
            "global_chart_retraction_v1",
        )
        self.assertFalse(
            implementation["first_order_lattice_chart_update_used"]
        )
        self.assertFalse(
            implementation["lattice_tikhonov_regularization_used"]
        )

    def test_runner_has_no_submission_training_generation_or_network_path(
        self,
    ) -> None:
        lowered = self.source.lower()
        for forbidden in (
            "sbatch",
            "srun",
            "subprocess",
            "requests.",
            "mprester",
            ".backward(",
            "optimizer",
            ".generate(",
            ".propose(",
        ):
            self.assertNotIn(forbidden, lowered)
        self.assertFalse(
            any(isinstance(node, ast.While) for node in ast.walk(self.tree))
        )

    def test_new_audits_and_fail_closed_gates_are_present(self) -> None:
        gates = self.contract["mechanics_gates"]
        self.assertEqual(
            gates["required_lattice_projection_method"],
            "global_chart_retraction_v1",
        )
        self.assertLessEqual(
            gates["primitive_transform_consistency_max_abs_error"],
            1.0e-12,
        )
        self.assertLessEqual(
            gates["primitive_lattice_consistency_relative_error"],
            1.0e-12,
        )
        self.assertEqual(gates["primitive_lattice_max_abs_entry"], 100.0)
        for token in (
            "maximum_lattice_chart_update_norm",
            "maximum_lattice_retracted_update_norm",
            "maximum_primitive_transform_consistency_max_abs_error",
            "maximum_primitive_lattice_consistency_relative_error",
            "maximum_primitive_lattice_scale",
            "global_chart_retraction",
            "lattice_scale_safety",
        ):
            self.assertIn(token, self.source)

    def test_old_failure_and_no_retry_boundary_are_exact(self) -> None:
        failure = self.contract["immutable_failure_reference"]
        self.assertEqual(failure["job_id"], 28185)
        self.assertEqual(failure["F_succeeded_cells"], 32)
        self.assertEqual(failure["T_succeeded_cells"], 31)
        self.assertFalse(failure["retry_or_replacement_used"])
        self.assertEqual(
            failure["unique_failed_cell"]["cell_id"],
            "b-e2a6902801f056e71703433a",
        )
        self.assertTrue(all(self.contract["forbidden_actions"].values()))
        self.assertEqual(self.contract["matrix"]["retry"], 0)
        self.assertEqual(self.contract["matrix"]["replacement"], 0)

    def test_future_resource_envelope_respects_permanent_cpu_gate(self) -> None:
        resources = self.contract[
            "future_resource_envelope_not_authorized"
        ]
        self.assertEqual(resources["a800"], 1)
        self.assertLessEqual(resources["cpus"], 8 * resources["a800"])


if __name__ == "__main__":
    unittest.main()
