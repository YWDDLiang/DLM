from __future__ import annotations

import hashlib
import json
import re
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT
    / "configs"
    / "experiments"
    / "wyckoff_codiffusion"
    / "wq_wyckoff_chart_retraction_preflight_sup28185_v2.json"
)
AUTHORIZATION = (
    ROOT
    / "diagnostics"
    / "authorization_records"
    / "wq_wyckoff_chart_retraction_preflight_sup28185_v2.json"
)
JOB_DIR = (
    ROOT
    / "scripts"
    / "a800"
    / "wq_wyckoff_chart_retraction_preflight_sup28185_v2"
)
SBATCH = JOB_DIR / "preflight.sbatch"
SUBMIT = JOB_DIR / "submit_once.sh"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ChartRetractionSubmissionV2StaticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
        cls.authorization = json.loads(
            AUTHORIZATION.read_text(encoding="utf-8")
        )
        cls.sbatch = SBATCH.read_text(encoding="utf-8")
        cls.submit = SUBMIT.read_text(encoding="utf-8")

    def test_shells_parse_and_remain_bash42_compatible(self) -> None:
        for path in (SBATCH, SUBMIT):
            result = subprocess.run(
                ["bash", "-n", str(path)],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
        combined = self.sbatch + self.submit
        for forbidden in (
            "declare -A",
            "mapfile",
            "readarray",
            "coproc",
            "wait -n",
            "${!prefix@}",
        ):
            self.assertNotIn(forbidden, combined)

    def test_authorization_is_separate_from_local_only_contract(self) -> None:
        self.assertFalse(
            self.contract["authorization"]["remote_transfer_authorized"]
        )
        self.assertFalse(
            self.contract["authorization"]["slurm_submission_authorized"]
        )
        self.assertEqual(self.authorization["user_quote"], "继续")
        job = self.authorization["authorized_job"]
        self.assertEqual(
            job["identity"],
            "wq_wyckoff_chart_retraction_preflight_sup28185_v2",
        )
        self.assertEqual(job["evidence_class"], "development_mechanics_regression_only")
        self.assertFalse(job["retry_or_replacement_allowed"])

    def test_resource_gate_is_exact_and_precedes_claim(self) -> None:
        self.assertIn("#SBATCH --partition=gpu", self.sbatch)
        self.assertIn(
            "#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1",
            self.sbatch,
        )
        self.assertIn("#SBATCH --cpus-per-task=8", self.sbatch)
        self.assertIn("#SBATCH --mem=64G", self.sbatch)
        self.assertIn("#SBATCH --time=01:00:00", self.sbatch)
        self.assertNotIn("#SBATCH --array", self.sbatch)
        resource_gate = self.submit.index("cpus > 8 * gpus")
        claim = self.submit.index('path.open("x"')
        actual_sbatch = self.submit.index('job_id="$(')
        self.assertLess(resource_gate, claim)
        self.assertLess(claim, actual_sbatch)

    def test_every_execution_identity_is_hash_bound(self) -> None:
        for expected in (
            _sha256(CONTRACT),
            _sha256(AUTHORIZATION),
            self.contract["implementation"]["v2_runner_source_sha256"],
            self.contract["implementation"][
                "legacy_execution_engine_source_sha256"
            ],
            self.contract["implementation"]["tangent_bridge_source_sha256"],
            self.contract["implementation"]["runtime_source_sha256"],
            self.contract["source_panel"]["generation_jsonl_sha256"],
            self.contract["parent"]["checkpoint_sha256"],
            self.contract["immutable_u_reference"]["terminal_audit_sha256"],
            self.contract["immutable_u_reference"][
                "selection_manifest_sha256"
            ],
            self.contract["immutable_u_reference"]["attempts_jsonl_sha256"],
            self.contract["immutable_u_reference"]["terminal_report_sha256"],
            self.contract["immutable_failure_reference"][
                "terminal_audit_sha256"
            ],
        ):
            self.assertIn(expected, self.sbatch)
            self.assertIn(expected, self.submit)

    def test_unique_identity_ignores_unrelated_queue(self) -> None:
        self.assertIn('test ! -e "$RECORD"', self.submit)
        self.assertIn('test ! -e "$CLAIM"', self.submit)
        self.assertIn('test ! -e "$OUTPUT"', self.submit)
        self.assertIn(
            "same WTB-32 v2 job identity already exists",
            self.submit,
        )
        self.assertIn('-v name="$JOB_NAME"', self.submit)
        self.assertNotRegex(
            self.submit,
            re.compile(r"(pending|running).*(<=|-[lg]e)", re.IGNORECASE),
        )

    def test_exactly_one_development_job_and_no_retry_path(self) -> None:
        self.assertEqual(self.submit.count("sbatch --parsable"), 2)
        combined = (self.sbatch + self.submit).lower()
        for forbidden in (
            "scancel",
            "scontrol update",
            "torchrun",
            "mprester",
            "optimizer",
        ):
            self.assertNotIn(forbidden, combined)
        self.assertIn(
            '"evidence_class": "development_mechanics_regression_only"',
            self.submit,
        )
        self.assertIn('"confirmatory_evidence": False', self.submit)
        self.assertIn('"retry_or_replacement_allowed": False', self.submit)
        self.assertIn('"training_submitted": False', self.submit)

    def test_terminal_gate_requires_v2_method_and_complete_arms(self) -> None:
        self.assertIn(
            '"wq_wyckoff_chart_retraction_preflight_terminal_v2"',
            self.sbatch,
        )
        self.assertIn(
            '"global_chart_retraction_v1"',
            self.sbatch,
        )
        self.assertIn('set(arms) != {"F", "T"}', self.sbatch)
        self.assertIn(
            'arms[arm].get("succeeded_cells") != 32',
            self.sbatch,
        )
        self.assertIn(
            "not all(value is True for value in checks.values())",
            self.sbatch,
        )


if __name__ == "__main__":
    unittest.main()
