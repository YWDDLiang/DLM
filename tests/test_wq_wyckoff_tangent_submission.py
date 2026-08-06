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
    / "wq_wyckoff_tangent_bridge_preflight_v1.json"
)
JOB_DIR = (
    ROOT
    / "scripts"
    / "a800"
    / "wq_wyckoff_tangent_bridge_preflight_v1"
)
SBATCH = JOB_DIR / "preflight.sbatch"
SUBMIT = JOB_DIR / "submit_once.sh"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class WyckoffTangentSubmissionStaticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
        cls.sbatch = SBATCH.read_text(encoding="utf-8")
        cls.submit = SUBMIT.read_text(encoding="utf-8")

    def test_shells_parse_and_avoid_post_bash42_features(self) -> None:
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

    def test_contract_and_all_immutable_inputs_are_hash_bound(self) -> None:
        self.assertIn(_sha256(CONTRACT), self.sbatch)
        self.assertIn(_sha256(CONTRACT), self.submit)
        for expected in (
            self.contract["implementation"]["runner_source_sha256"],
            self.contract["implementation"]["tangent_bridge_source_sha256"],
            self.contract["source_panel"]["generation_jsonl_sha256"],
            self.contract["parent"]["checkpoint_sha256"],
            self.contract["immutable_u_reference"]["terminal_audit_sha256"],
            self.contract["immutable_u_reference"][
                "selection_manifest_sha256"
            ],
            self.contract["immutable_u_reference"]["attempts_jsonl_sha256"],
            self.contract["immutable_u_reference"]["terminal_report_sha256"],
        ):
            self.assertIn(expected, self.sbatch)
            self.assertIn(expected, self.submit)

    def test_authorization_and_atomic_install_are_mandatory(self) -> None:
        for source in (self.sbatch, self.submit):
            self.assertIn("WTB32_PATCH_SHA256", source)
            self.assertIn("WTB32_AUTHORIZATION_SHA256", source)
            self.assertIn("authorized_patch_", source)
            self.assertIn("submission_authorization_record.json", source)
        self.assertFalse(
            self.contract["authorization"]["remote_transfer_authorized"]
        )
        self.assertFalse(
            self.contract["authorization"]["slurm_submission_authorized"]
        )

    def test_unique_identity_does_not_gate_on_unrelated_queue(self) -> None:
        self.assertIn('test ! -e "$RECORD"', self.submit)
        self.assertIn('test ! -e "$CLAIM"', self.submit)
        self.assertIn('test ! -e "$OUTPUT"', self.submit)
        self.assertIn(
            "same WTB-32 job identity already exists",
            self.submit,
        )
        self.assertIn('-v name="$JOB_NAME"', self.submit)
        self.assertNotRegex(
            self.submit,
            re.compile(r"(pending|running).*(<=|-[lg]e)", re.IGNORECASE),
        )
        self.assertNotIn("QOSMaxSubmitJobPerUserLimit", self.submit)

    def test_exactly_one_job_and_no_retry_or_mutation_path(self) -> None:
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
        self.assertIn('"retry_or_replacement_allowed": False', self.submit)
        self.assertIn('"u_rerun": False', self.submit)
        self.assertIn('"training_submitted": False', self.submit)

    def test_terminal_gate_requires_both_complete_32_cell_arms(self) -> None:
        self.assertIn('set(arms) != {"F", "T"}', self.sbatch)
        self.assertIn(
            'arms[arm].get("terminal_cells") != 32',
            self.sbatch,
        )
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
