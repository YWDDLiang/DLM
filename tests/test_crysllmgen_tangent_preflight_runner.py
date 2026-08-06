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
    / "run_wq_wyckoff_tangent_bridge_preflight_v1.py"
)
CONTRACT = (
    ROOT
    / "configs"
    / "experiments"
    / "wyckoff_codiffusion"
    / "wq_wyckoff_tangent_bridge_preflight_v1.json"
)
FROZEN_V1_TERMINAL_AUDIT = (
    ROOT
    / "runs"
    / "remote_audit"
    / "20260726_wq_wyckoff_tangent_bridge_preflight_v1"
    / "terminal_audit_job28185.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class WyckoffTangentPreflightRunnerStaticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = RUNNER.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)
        cls.contract = json.loads(CONTRACT.read_text(encoding="utf-8"))

    def test_contract_is_local_only_and_hash_binds_frozen_v1_artifact(
        self,
    ) -> None:
        self.assertEqual(
            self.contract["status"],
            "local_built_remote_execution_not_authorized",
        )
        self.assertFalse(
            self.contract["authorization"]["remote_transfer_authorized"]
        )
        self.assertFalse(
            self.contract["authorization"]["slurm_submission_authorized"]
        )
        implementation = self.contract["implementation"]
        self.assertEqual(
            _sha256(ROOT / implementation["runner_source"]),
            implementation["runner_source_sha256"],
        )
        # The working-tree source now carries the independently identified v2
        # correction. Historical v1 remains bound to its immutable terminal
        # audit rather than being retroactively rewritten.
        self.assertNotEqual(
            _sha256(ROOT / implementation["tangent_bridge_source"]),
            implementation["tangent_bridge_source_sha256"],
        )
        terminal = json.loads(
            FROZEN_V1_TERMINAL_AUDIT.read_text(encoding="utf-8")
        )
        self.assertEqual(
            terminal["frozen_scientific_identity"][
                "tangent_bridge_source_sha256"
            ],
            implementation["tangent_bridge_source_sha256"],
        )

    def test_runner_has_no_submission_network_or_optimization_path(self) -> None:
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

    def test_u_is_exact_hash_bound_and_never_rerun(self) -> None:
        reference = self.contract["immutable_u_reference"]
        self.assertEqual(reference["job_id"], 28081)
        self.assertFalse(reference["rerun"])
        for field in (
            "terminal_audit_sha256",
            "selection_manifest_sha256",
            "attempts_jsonl_sha256",
            "terminal_report_sha256",
        ):
            self.assertEqual(len(reference[field]), 64)
        self.assertIn("_validate_u_reference", self.source)
        self.assertIn('"u_rerun": False', self.source)
        self.assertNotIn("run_parent_reverse_from_noisy_state", self.source)

    def test_new_matrix_is_exact_f_and_t_with_frozen_call_budgets(self) -> None:
        matrix = self.contract["matrix"]
        gates = self.contract["mechanics_gates"]
        self.assertEqual(matrix["new_arms"], ["F", "T"])
        self.assertEqual(matrix["new_cells_per_arm"], 32)
        self.assertEqual(matrix["new_total_cells"], 64)
        self.assertEqual(gates["F_parent_decoder_calls_per_cell"], 0)
        self.assertEqual(gates["F_projection_calls_per_cell"], 1)
        self.assertEqual(gates["T_parent_decoder_calls_per_cell"], 64)
        self.assertEqual(gates["T_projection_calls_per_cell"], 64)
        self.assertIn("_run_final_projection", self.source)
        self.assertIn("_run_tangent_trajectory", self.source)
        self.assertIn("run_parent_reverse_on_wyckoff_manifold", self.source)

    def test_failure_evidence_is_append_only_and_never_retried(self) -> None:
        self.assertIn('attempts_path.open("x"', self.source)
        self.assertIn("os.fsync", self.source)
        self.assertIn('"status": "failed"', self.source)
        self.assertIn('"retry_or_replacement_used": False', self.source)
        self.assertTrue(all(self.contract["forbidden_actions"].values()))

    def test_resource_and_scientific_mechanics_gates_are_fail_closed(self) -> None:
        resources = self.contract[
            "future_resource_envelope_not_authorized"
        ]
        self.assertEqual(resources["a800"], 1)
        self.assertLessEqual(resources["cpus"], 8 * resources["a800"])
        gates = self.contract["mechanics_gates"]
        self.assertEqual(gates["terminal_cells_per_arm"], 32)
        self.assertEqual(gates["successful_cells_per_arm"], 32)
        self.assertLessEqual(
            gates["fixed_site_drift_max_abs"],
            1.0e-8,
        )
        self.assertIn("all(bool(value) for value in gate_checks.values())", self.source)
        self.assertIn("raise SystemExit(2)", self.source)


if __name__ == "__main__":
    unittest.main()
