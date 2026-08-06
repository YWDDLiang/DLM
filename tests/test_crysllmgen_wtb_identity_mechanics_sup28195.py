from __future__ import annotations

import hashlib
import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT
    / "configs"
    / "experiments"
    / "wyckoff_codiffusion"
    / "wq_wyckoff_identity_mechanics_sup28195_v1.json"
)
AUTHORIZATION = (
    ROOT
    / "diagnostics"
    / "authorization_records"
    / "wq_wyckoff_identity_mechanics_sup28195_v1.json"
)
RUNNER = (
    ROOT / "scripts" / "a800" / "run_wq_wyckoff_identity_mechanics_sup28195_v1.py"
)
SBATCH = (
    ROOT
    / "scripts"
    / "a800"
    / "wq_wyckoff_identity_mechanics_sup28195_v1"
    / "mechanics.sbatch"
)
SUBMIT = SBATCH.with_name("submit_once.sh")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class WTBIdentityMechanicsSup28195Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
        cls.authorization = json.loads(AUTHORIZATION.read_text(encoding="utf-8"))
        cls.runner = RUNNER.read_text(encoding="utf-8")
        cls.sbatch = SBATCH.read_text(encoding="utf-8")
        cls.submit = SUBMIT.read_text(encoding="utf-8")

    def test_contract_makes_order_diagnostic_but_real_identity_fail_closed(
        self,
    ) -> None:
        identity = self.contract["identity_contract"]
        self.assertEqual(identity["composition_identity"], "element_count_multiset")
        self.assertEqual(
            identity["topology_identity"],
            "exact_species_wyckoff_topology_hash",
        )
        self.assertEqual(
            identity["legacy_ordered_atom_identity_role"],
            "diagnostic_only",
        )
        self.assertIs(identity["legacy_ordered_atom_mismatch_blocking"], False)
        self.assertIs(
            identity["true_composition_topology_or_atom_count_change_blocking"],
            True,
        )
        self.assertEqual(
            identity["parent_input_order"],
            "canonical_orbit_expansion",
        )

    def test_failed_job_is_immutable_and_reused_only_for_development(self) -> None:
        lineage = self.contract["lineage"]
        self.assertEqual(lineage["supersedes_failed_job_id"], 28195)
        self.assertEqual(
            lineage["job28195_decision"],
            "invalid_integrity_stop_no_retry",
        )
        self.assertIs(lineage["job28195_reinterpreted"], False)
        self.assertIs(lineage["development_panel_reused"], True)
        self.assertIs(lineage["confirmatory_evidence"], False)
        self.assertTrue(lineage["future_confirmation_requires_nonoverlapping_panel"])
        self.assertEqual(
            lineage["terminal_failure_audit_sha256"],
            "124bb6e02d612687cd25a21b57b57e64773eff5836c788b8f5998754f1da76c9",
        )

    def test_frozen_source_and_matrix_are_exact(self) -> None:
        frozen = self.contract["frozen_job28195_sources"]
        self.assertEqual(frozen["source_rows"], 256)
        self.assertEqual(
            frozen["source_attempts_sha256"],
            "3246a24d2595ae760e15f402222d6730a2a0fdbc404636254a7fa995559d56f2",
        )
        self.assertEqual(
            frozen["source_report_sha256"],
            "9c6d5a9f2570f73b87ffa4c2bac898499588ea803cdf5803103408ce22323be9",
        )
        matrix = self.contract["matrix"]
        self.assertEqual(matrix["source_identity_audit_attempts"], 256)
        self.assertEqual(matrix["mechanics_attempts_per_arm"], 32)
        self.assertEqual(
            (matrix["start_ordinal"], matrix["end_ordinal_inclusive"]),
            (512, 543),
        )
        self.assertEqual(set(matrix["arms"]), {"R", "U", "T"})
        self.assertEqual(matrix["new_proposals"], 0)
        self.assertEqual(matrix["retry"], 0)
        self.assertEqual(matrix["replacement"], 0)

    def test_contract_pins_every_implementation_byte(self) -> None:
        for entry in self.contract["implementation"].values():
            path = ROOT / entry["path"]
            self.assertTrue(path.is_file(), path)
            self.assertEqual(_sha(path), entry["sha256"], path)
        self.assertEqual(
            self.contract["authorization"]["record_sha256"],
            _sha(AUTHORIZATION),
        )

    def test_runner_normalizes_before_parent_input_and_never_promotes(self) -> None:
        required = (
            "composition_multiset_signature",
            "canonical_storage=True",
            "canonical_orbit_expansion",
            "legacy_order_mismatch_is_blocking",
            "automatic_confirmatory_authorized",
            "job28195_reinterpreted",
        )
        for value in required:
            self.assertIn(value, self.runner)
        self.assertIn(
            "legacy.composition_signature = composition_multiset_signature",
            self.runner,
        )
        self.assertNotIn("run_crysllmgen_metrics", self.runner)
        self.assertNotIn("run_crysllmgen_a100_sun", self.runner)

    def test_sbatch_resource_and_stage_boundary(self) -> None:
        self.assertIn("#SBATCH --partition=gpu", self.sbatch)
        self.assertIn("#SBATCH --cpus-per-task=8", self.sbatch)
        self.assertIn("#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1", self.sbatch)
        self.assertIn("#SBATCH --mem=64G", self.sbatch)
        self.assertIn("#SBATCH --time=01:00:00", self.sbatch)
        self.assertNotIn("#SBATCH --array", self.sbatch)
        self.assertEqual(
            self.sbatch.count(
                "python \"$RUNNER\" \\\n"
            ),
            1,
        )
        for forbidden in (
            "run_crysllmgen_metrics.py",
            "run_crysllmgen_a100_sun.py",
            "scancel",
            "mp-api",
        ):
            self.assertNotIn(forbidden, self.sbatch)

    def test_submit_is_exclusive_and_calls_sbatch_once(self) -> None:
        claim_write = self.submit.index('with Path(sys.argv[1]).open("x"')
        sbatch_command = self.submit.index('command="sbatch --parsable')
        self.assertLess(claim_write, sbatch_command)
        executable_sbatch = re.findall(r"^\s*sbatch --parsable \\\\?$", self.submit, re.M)
        self.assertEqual(len(executable_sbatch), 1)
        self.assertIn("same WTB identity-v2 job already exists", self.submit)
        self.assertIn("queue_policy", self.submit)
        self.assertNotIn("scancel", self.submit)
        self.assertNotIn("submit_once.sh", self.submit)

    def test_authorization_is_one_development_job_not_new_science(self) -> None:
        self.assertEqual(
            self.authorization["execution_identity"],
            "wq_wyckoff_identity_mechanics_sup28195_v1",
        )
        boundary = self.authorization["scientific_boundary"]
        self.assertIs(boundary["job28195_reinterpreted"], False)
        self.assertIs(boundary["confirmatory_evidence"], False)
        self.assertEqual(boundary["new_proposals"], 0)
        self.assertIs(boundary["training"], False)
        self.assertIs(boundary["automatic_confirmatory_submission"], False)
        self.assertIn(
            "automatic submission of a new 256-attempt confirmatory evaluation",
            self.authorization["not_authorized"],
        )


if __name__ == "__main__":
    unittest.main()
