from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from crystal_dlm.wqcodiff.guidance_contract import (
    GuidanceContractError,
    load_and_validate_guidance_contract,
    validate_guidance_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs"
    / "experiments"
    / "wyckoff_codiffusion"
    / "mattersim_guidance_chgnet_eval_v1.json"
)


class MatterSimGuidanceContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = json.loads(CONFIG.read_text(encoding="utf-8"))

    def mutate(self) -> dict[str, object]:
        return copy.deepcopy(self.payload)

    def test_frozen_contract_passes(self) -> None:
        result = load_and_validate_guidance_contract(CONFIG)
        self.assertTrue(result["ok"])
        self.assertEqual(result["guide"], "mattersim")
        self.assertEqual(result["headline_evaluator"], "chgnet")
        self.assertEqual(result["max_cpus_per_a800"], 8)
        self.assertTrue(result["mlip_free_training"])
        self.assertFalse(result["long_training_authorized"])
        self.assertEqual(len(result["file_sha256"]), 64)

    def test_rejects_role_swap(self) -> None:
        payload = self.mutate()
        payload["roles"]["guidance"]["model"] = "chgnet"
        with self.assertRaisesRegex(GuidanceContractError, "must be MatterSim"):
            validate_guidance_contract(payload)

    def test_rejects_chgnet_in_guidance_tuning(self) -> None:
        payload = self.mutate()
        payload["guidance_contract"]["development_selection_sources"].append(
            "chgnet_energy"
        )
        with self.assertRaisesRegex(GuidanceContractError, "cannot be used to tune"):
            validate_guidance_contract(payload)

    def test_rejects_mattersim_as_headline_evaluator(self) -> None:
        payload = self.mutate()
        payload["evaluation_firewall"]["mattersim_used_as_headline_evaluator"] = True
        with self.assertRaisesRegex(GuidanceContractError, "must be false"):
            validate_guidance_contract(payload)

    def test_rejects_more_than_eight_cpus_per_a800(self) -> None:
        payload = self.mutate()
        payload["resource_contract"]["job_profiles"][0]["cpus"] = 9
        with self.assertRaisesRegex(GuidanceContractError, "exceeds 8 CPUs"):
            validate_guidance_contract(payload)

    def test_rejects_cross_composition_energy_ranking(self) -> None:
        payload = self.mutate()
        payload["guidance_contract"]["cross_composition_energy_ranking"] = True
        with self.assertRaisesRegex(GuidanceContractError, "cross-composition"):
            validate_guidance_contract(payload)

    def test_rejects_retry_or_unguided_fallback(self) -> None:
        payload = self.mutate()
        payload["guidance_contract"]["retry_enabled"] = True
        with self.assertRaisesRegex(GuidanceContractError, "retry_enabled"):
            validate_guidance_contract(payload)

        payload = self.mutate()
        payload["guidance_contract"]["failure_policy"] = "fallback_to_unguided"
        with self.assertRaisesRegex(GuidanceContractError, "cannot hide"):
            validate_guidance_contract(payload)

    def test_rejects_unaudited_lattice_guidance(self) -> None:
        payload = self.mutate()
        payload["guidance_contract"]["lattice_guidance_enabled"] = True
        with self.assertRaisesRegex(GuidanceContractError, "lattice guidance"):
            validate_guidance_contract(payload)

    def test_rejects_long_training_authorization(self) -> None:
        payload = self.mutate()
        payload["new_diffusion_training_policy"][
            "long_training_authorized_by_this_file"
        ] = True
        with self.assertRaisesRegex(GuidanceContractError, "cannot authorize"):
            validate_guidance_contract(payload)

    def test_rejects_mattersim_training_or_distillation(self) -> None:
        payload = self.mutate()
        payload["new_diffusion_training_policy"][
            "mattersim_allowed_in_training_or_distillation"
        ] = True
        with self.assertRaisesRegex(GuidanceContractError, "cannot enter training"):
            validate_guidance_contract(payload)

        payload = self.mutate()
        payload["new_diffusion_training_policy"][
            "allowed_training_signal_sources"
        ].append("MatterSim_force_distillation")
        with self.assertRaisesRegex(GuidanceContractError, "MLIP-derived"):
            validate_guidance_contract(payload)


if __name__ == "__main__":
    unittest.main()
