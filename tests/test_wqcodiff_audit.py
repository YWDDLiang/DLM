from __future__ import annotations

import unittest

from crystal_dlm.wqcodiff.audit import _required_claim_evidence


class WorkflowClaimEvidenceTests(unittest.TestCase):
    def _complete(self) -> dict[str, bool]:
        return _required_claim_evidence(
            ledger_results=[{"ok": True}],
            artifact_result={"ok": True, "records": 1},
            formal_reports=[
                {"schema": "wqcodiff_formal_audit_v1", "passed": True},
                {"schema": "wqcodiff_pyxtal_chart_audit_v1", "passed": True},
            ],
            dataset_reports=[
                {"schema": "wqcodiff_p1_dataset_audit_v1", "passed": True}
            ],
            source_result={"ok": True},
            asset_result={"ok": True},
            revision_result={"ok": True},
            aggregate_result={"oral_eligible": True},
        )

    def test_complete_evidence_bundle_passes_every_requirement(self) -> None:
        self.assertTrue(all(self._complete().values()))

    def test_empty_or_wrong_schema_reports_cannot_enable_claim(self) -> None:
        evidence = _required_claim_evidence(
            ledger_results=[],
            artifact_result={"ok": True, "records": 0},
            formal_reports=[{"schema": "unrelated", "passed": True}],
            dataset_reports=[],
            source_result=None,
            asset_result=None,
            revision_result=None,
            aggregate_result={"oral_eligible": True},
        )
        self.assertFalse(evidence["attempt_ledgers"])
        self.assertFalse(evidence["attempt_artifacts"])
        self.assertFalse(evidence["formal_and_chart_gates"])
        self.assertFalse(evidence["p1_dataset_gate"])
        self.assertFalse(evidence["source_manifest"])
        self.assertFalse(evidence["mlip_asset_lock"])
        self.assertFalse(evidence["revision_threshold_lock"])

    def test_each_required_formal_schema_is_indispensable(self) -> None:
        evidence = _required_claim_evidence(
            ledger_results=[{"ok": True}],
            artifact_result={"ok": True, "records": 1},
            formal_reports=[
                {"schema": "wqcodiff_formal_audit_v1", "passed": True}
            ],
            dataset_reports=[
                {"schema": "wqcodiff_p1_dataset_audit_v1", "passed": True}
            ],
            source_result={"ok": True},
            asset_result={"ok": True},
            revision_result={"ok": True},
            aggregate_result={"oral_eligible": True},
        )
        self.assertFalse(evidence["formal_and_chart_gates"])


if __name__ == "__main__":
    unittest.main()
