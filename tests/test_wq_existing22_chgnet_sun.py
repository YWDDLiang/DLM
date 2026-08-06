from __future__ import annotations

import copy
import json
import re
import tempfile
import unittest
from pathlib import Path

from diagnostics.accept_wq_existing22_chgnet_sun import scientific_decision
from diagnostics.prepare_wq_existing22_chgnet_sun import (
    Existing22SunInputError,
    build_attempt_rows,
    write_json_exclusive,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = (
    ROOT
    / "configs"
    / "experiments"
    / "wyckoff_codiffusion"
    / "wq_existing22_chgnet_sun_v1.json"
)
SBATCH = (
    ROOT
    / "scripts"
    / "a800"
    / "wq_existing22_chgnet_sun_v1"
    / "evaluate.sbatch"
)
SUBMIT = SBATCH.with_name("submit_once.sh")


def frozen_contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def source_rows(contract: dict) -> tuple[list[dict], list[dict]]:
    failed = {
        int(value)
        for value in contract["denominator"][
            "frozen_structural_failure_ordinals"
        ]
    }
    structures: list[dict] = []
    metrics: list[dict] = []
    for index, ordinal in enumerate(
        contract["denominator"]["expected_ordinals"]
    ):
        attempt_id = f"attempt-{index:02d}"
        structures.append(
            {
                "attempt_id": attempt_id,
                "ordinal": ordinal,
                "status": "succeeded",
                "structure": {
                    "@module": "pymatgen.core.structure",
                    "@class": "Structure",
                    "fixture_ordinal": ordinal,
                },
                "retry_or_replacement_used": False,
            }
        )
        valid = ordinal not in failed
        metrics.append(
            {
                "attempt_id": attempt_id,
                "ordinal": ordinal,
                "projected_formula": f"X{index + 1}Y",
                "comp_valid": True,
                "struct_valid": valid,
                "valid": valid,
                "fingerprint_valid": True,
                "reason": "" if valid else "structure_invalid",
                "retry_or_replacement_used": False,
            }
        )
    return structures, metrics


class ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = frozen_contract()
        cls.sbatch = SBATCH.read_text(encoding="utf-8")
        cls.submit = SUBMIT.read_text(encoding="utf-8")

    def test_preserves_formal_fail_as_separate_user_continuation(self) -> None:
        self.assertEqual(
            self.contract["frozen_history"]["formal_survival_result"],
            "FAIL",
        )
        self.assertFalse(
            self.contract["authorization"]["formal_survival_gate_rewritten"]
        )
        self.assertTrue(
            self.contract["authorization"][
                "user_accepted_exploratory_continuation"
            ]
        )
        observed = self.contract["frozen_history"]["observed_survival"]
        self.assertEqual(observed["structural_valid_count"], 17)
        self.assertEqual(observed["joint_valid_count"], 17)

    def test_freezes_all22_and_pre_chgnet_failure_placeholders(self) -> None:
        denominator = self.contract["denominator"]
        self.assertEqual(denominator["attempts"], 22)
        self.assertEqual(denominator["reconstructed_structures_exact"], 17)
        self.assertEqual(
            denominator["frozen_structural_failure_ordinals"],
            [262, 295, 323, 328, 445],
        )
        self.assertIn(
            "cannot be relaxed", denominator["frozen_failure_handling"]
        )

    def test_directional_thresholds_use_exact_historical_reference(self) -> None:
        reference = self.contract["historical_directional_reference"]
        self.assertEqual(reference["strict"]["count"], 90)
        self.assertEqual(reference["strict"]["attempts"], 1000)
        self.assertEqual(reference["meta_like"]["count"], 461)
        self.assertEqual(reference["meta_like"]["attempts"], 1000)
        self.assertEqual(
            self.contract["decision_rule"]["minimum_strict_full_sun_count"],
            2,
        )
        self.assertEqual(
            self.contract["decision_rule"]["minimum_meta_full_sun_count"],
            11,
        )

    def test_resource_and_environment_are_fail_closed(self) -> None:
        resources = self.contract["resources"]
        self.assertEqual(resources["a800"], 1)
        self.assertEqual(resources["cpus"], 8)
        self.assertLessEqual(resources["cpus"], 8 * resources["a800"])
        self.assertIn("#SBATCH --cpus-per-task=8", self.sbatch)
        self.assertIn("#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1", self.sbatch)
        self.assertIn("conda activate diff_meets_diff", self.sbatch)
        self.assertNotIn("conda activate crysllm", self.sbatch)
        self.assertIn("unset MP_API_KEY", self.sbatch)
        self.assertIn("unset PMG_MAPI_KEY", self.sbatch)
        self.assertIn("job_cpus > 8 * job_gpus", self.submit)
        self.assertNotIn("QOSMaxSubmitJobPerUserLimit", self.submit)

    def test_training_marker_preflight_allows_only_evaluator_assets(
        self,
    ) -> None:
        pattern = re.compile(
            r"(?<![A-Za-z])train(?:ing)?(?![A-Za-z])"
            r"|fine[-_ ]?tune|optimizer|backward",
            re.IGNORECASE,
        )
        markers = []
        for line in self.sbatch.splitlines():
            if not pattern.search(line):
                continue
            normalized = line.strip()
            if normalized.endswith("\\"):
                normalized = normalized[:-1].rstrip()
            markers.append(normalized)
        self.assertEqual(
            markers,
            [
                "--train-csv reference/crysllmgen/data/mp_20/train.csv",
                (
                    "--training-index-cache "
                    "reference/crysllmgen/data/mp_20/.cache/"
                    "train.csv.3a814f7b7bf29b1a.training_index.pkl"
                ),
            ],
        )
        self.assertNotIn(
            "grep -Eiq 'train|fine[-_ ]?tune|optimizer|backward'",
            self.submit,
        )
        self.assertIn("if markers != expected:", self.submit)


class AdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.contract = frozen_contract()
        self.structures, self.metrics = source_rows(self.contract)

    def test_maps_seventeen_successes_and_five_fixed_failures(self) -> None:
        generation, manifest = build_attempt_rows(
            self.contract, self.structures, self.metrics
        )
        self.assertEqual(len(generation), 22)
        succeeded = [row for row in generation if row["status"] == "succeeded"]
        failed = [row for row in generation if row["status"] == "failed"]
        self.assertEqual(len(succeeded), 17)
        self.assertEqual(len(failed), 5)
        self.assertTrue(all("structure" in row for row in succeeded))
        self.assertTrue(all("structure" not in row for row in failed))
        self.assertTrue(
            all(
                row["reason"] == "frozen_pre_chgnet_structure_invalid"
                for row in failed
            )
        )
        self.assertEqual(manifest["reconstructed_structures"], 17)
        self.assertEqual(manifest["failed_placeholders"], 5)
        self.assertFalse(manifest["geometry_repair_or_rescue"])
        self.assertFalse(manifest["retry_or_replacement_used"])

    def test_rejects_attempt_order_change(self) -> None:
        structures = copy.deepcopy(self.structures)
        structures[0], structures[1] = structures[1], structures[0]
        with self.assertRaisesRegex(
            Existing22SunInputError, "identity/order"
        ):
            build_attempt_rows(self.contract, structures, self.metrics)

    def test_rejects_rescuing_known_structural_failure(self) -> None:
        metrics = copy.deepcopy(self.metrics)
        metrics[0]["struct_valid"] = True
        metrics[0]["valid"] = True
        metrics[0]["reason"] = ""
        with self.assertRaisesRegex(
            Existing22SunInputError, "structural-failure identity"
        ):
            build_attempt_rows(self.contract, self.structures, metrics)

    def test_exclusive_output_guard(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "sealed.json"
            write_json_exclusive(target, {"ok": True})
            with self.assertRaises(FileExistsError):
                write_json_exclusive(target, {"ok": False})


class DecisionTests(unittest.TestCase):
    def test_pass_requires_both_lower_bounds(self) -> None:
        self.assertEqual(
            scientific_decision(
                strict_count=2,
                meta_count=11,
                unknown_count=0,
                minimum_strict=2,
                minimum_meta=11,
            ),
            "PASS",
        )

    def test_fail_when_even_optimistic_unknowns_cannot_reach_target(self) -> None:
        self.assertEqual(
            scientific_decision(
                strict_count=0,
                meta_count=7,
                unknown_count=1,
                minimum_strict=2,
                minimum_meta=11,
            ),
            "FAIL",
        )

    def test_inconclusive_when_unknowns_can_change_decision(self) -> None:
        self.assertEqual(
            scientific_decision(
                strict_count=1,
                meta_count=9,
                unknown_count=2,
                minimum_strict=2,
                minimum_meta=11,
            ),
            "INCONCLUSIVE_MP_COVERAGE",
        )


if __name__ == "__main__":
    unittest.main()
