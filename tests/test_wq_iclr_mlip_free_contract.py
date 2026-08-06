from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from crystal_dlm.wqcodiff.mlip_free_contract import (
    MLIPFreeContractError,
    load_and_validate_mlip_free_contract,
    validate_mlip_free_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs"
    / "experiments"
    / "wyckoff_codiffusion"
    / "wq_iclr_mlip_free_v1.json"
)


class MLIPFreeExperimentContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = json.loads(CONFIG.read_text(encoding="utf-8"))

    def mutate(self) -> dict[str, object]:
        return copy.deepcopy(self.payload)

    def test_frozen_contract_passes(self) -> None:
        result = load_and_validate_mlip_free_contract(CONFIG)
        self.assertTrue(result["training_mlip_free"])
        self.assertTrue(result["sampling_mlip_free"])
        self.assertEqual(result["heldout_evaluator"], "chgnet")
        self.assertEqual(result["minimum_no_neutral_recovered"], 24)
        self.assertFalse(result["remote_submission_authorized"])
        self.assertFalse(result["long_training_authorized"])

    def test_rejects_mlip_training_signal(self) -> None:
        payload = self.mutate()
        payload["training_contract"]["allowed_signal_sources"].append(
            "MatterSim_force_target"
        )
        with self.assertRaisesRegex(MLIPFreeContractError, "forbidden training"):
            validate_mlip_free_contract(payload)

    def test_rejects_energy_guidance_or_reranking(self) -> None:
        for key in ("energy_guidance_enabled", "reranking_enabled"):
            payload = self.mutate()
            payload["sampling_contract"][key] = True
            with self.assertRaisesRegex(MLIPFreeContractError, key):
                validate_mlip_free_contract(payload)

    def test_rejects_retry_and_best_of(self) -> None:
        for key in ("retry_enabled", "replacement_enabled", "best_of_n_enabled"):
            payload = self.mutate()
            payload["sampling_contract"][key] = True
            with self.assertRaisesRegex(MLIPFreeContractError, key):
                validate_mlip_free_contract(payload)

    def test_projector_only_handles_charge_neutrality_failures(self) -> None:
        payload = self.mutate()
        payload["composition_projection"][
            "applicable_source_reason"
        ] = "pauling_fail_or_ratio_rejected"
        with self.assertRaisesRegex(MLIPFreeContractError, "only act"):
            validate_mlip_free_contract(payload)

    def test_rejects_incomplete_topology_preservation(self) -> None:
        payload = self.mutate()
        payload["composition_projection"]["preserve"].remove("free_coordinates")
        with self.assertRaisesRegex(MLIPFreeContractError, "preservation"):
            validate_mlip_free_contract(payload)

    def test_rejects_changed_objective_order(self) -> None:
        payload = self.mutate()
        payload["composition_projection"]["objective_order"][0:2] = [
            "affected_primitive_atom_count",
            "changed_orbit_count",
        ]
        with self.assertRaisesRegex(MLIPFreeContractError, "objective order"):
            validate_mlip_free_contract(payload)

    def test_rejects_chgnet_training_or_checkpoint_selection(self) -> None:
        payload = self.mutate()
        payload["evaluation_contract"]["forbidden_uses"].remove("checkpoint_selection")
        with self.assertRaisesRegex(MLIPFreeContractError, "firewall"):
            validate_mlip_free_contract(payload)

    def test_rejects_resource_limit_change(self) -> None:
        payload = self.mutate()
        payload["resource_contract"]["max_cpus_per_a800"] = 16
        with self.assertRaisesRegex(MLIPFreeContractError, "8 per A800"):
            validate_mlip_free_contract(payload)

    def test_rejects_remote_or_long_training_authorization(self) -> None:
        for key in ("remote_submission", "long_training"):
            payload = self.mutate()
            payload["authorization"][key] = True
            with self.assertRaisesRegex(MLIPFreeContractError, "not authorized"):
                validate_mlip_free_contract(payload)


if __name__ == "__main__":
    unittest.main()
