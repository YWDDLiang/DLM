from __future__ import annotations

import copy
import hashlib
import unittest
from pathlib import Path

from crystal_dlm.wqcodiff.protocol import load_protocol, validate_protocol


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/experiments/wyckoff_codiffusion/protocol_v3.yaml"


class ProtocolTests(unittest.TestCase):
    def test_active_protocol_is_registered_v3(self) -> None:
        protocol = load_protocol(PROTOCOL)
        self.assertEqual(protocol.schema_version, 3)
        self.assertEqual(len(protocol.sha256), 64)

    def test_space_group_rollback_cannot_be_silently_enabled(self) -> None:
        protocol = load_protocol(PROTOCOL)
        changed = copy.deepcopy(protocol.data)
        changed["wyckoff_representation"]["committed_space_group_change"] = True
        with self.assertRaisesRegex(ValueError, "SG rollback"):
            validate_protocol(changed)

    def test_gpu_hour_ceiling_cannot_be_raised(self) -> None:
        protocol = load_protocol(PROTOCOL)
        changed = copy.deepcopy(protocol.data)
        changed["compute_funnel"]["usable_a800_gpu_hours_hard_ceiling"] = 2200
        with self.assertRaisesRegex(ValueError, "GPU-hour"):
            validate_protocol(changed)

    def test_numeric_thread_lock_cannot_be_raised(self) -> None:
        protocol = load_protocol(PROTOCOL)
        changed = copy.deepcopy(protocol.data)
        changed["execution"]["threads"]["OPENBLAS_NUM_THREADS"] = 64
        with self.assertRaisesRegex(ValueError, "thread lock"):
            validate_protocol(changed)

    def test_day7_method_registry_rejects_duplicate_cells(self) -> None:
        protocol = load_protocol(PROTOCOL)
        changed = copy.deepcopy(protocol.data)
        changed["day7_falsification"]["methods"].append("B-WQ-D3PM")
        with self.assertRaisesRegex(ValueError, "contains duplicates"):
            validate_protocol(changed)

    def test_wrapped_score_cannot_silently_revert_to_euclidean_target(self) -> None:
        protocol = load_protocol(PROTOCOL)
        changed = copy.deepcopy(protocol.data)
        changed["models"]["continuous"]["wrapped_score_target"] = "unwrapped_gaussian"
        with self.assertRaisesRegex(ValueError, "wrapped-Gaussian"):
            validate_protocol(changed)

    def test_wrapped_score_loss_cannot_drop_sigma_squared_weighting(self) -> None:
        protocol = load_protocol(PROTOCOL)
        changed = copy.deepcopy(protocol.data)
        changed["models"]["continuous"]["wrapped_score_loss_weight"] = "uniform"
        with self.assertRaisesRegex(ValueError, "loss weighting"):
            validate_protocol(changed)

    def test_periodic_coordinate_likelihood_cannot_revert_to_euclidean(self) -> None:
        protocol = load_protocol(PROTOCOL)
        changed = copy.deepcopy(protocol.data)
        changed["models"]["continuous"][
            "periodic_coordinate_likelihood"
        ] = "ordinary_gaussian"
        with self.assertRaisesRegex(ValueError, "periodic-coordinate likelihood"):
            validate_protocol(changed)

    def test_periodic_coordinate_scale_floor_is_frozen(self) -> None:
        protocol = load_protocol(PROTOCOL)
        changed = copy.deepcopy(protocol.data)
        changed["models"]["continuous"]["periodic_coordinate_scale_min"] = 1.0e-6
        with self.assertRaisesRegex(ValueError, "scale bounds"):
            validate_protocol(changed)

    def test_mattersim_cannot_be_moved_back_into_the_mace_process(self) -> None:
        protocol = load_protocol(PROTOCOL)
        changed = copy.deepcopy(protocol.data)
        changed["mlip"]["runtime_isolation"]["one_evaluator_per_process"] = False
        with self.assertRaisesRegex(ValueError, "process-isolation"):
            validate_protocol(changed)

    def test_failed_wheelhouse_lock_cannot_be_promoted_to_active(self) -> None:
        protocol = load_protocol(PROTOCOL)
        changed = copy.deepcopy(protocol.data)
        core = changed["mlip"]["runtime_isolation"]["core_environment"]
        core["wheelhouse_lock"] = "wheelhouse_lock.json"
        with self.assertRaisesRegex(ValueError, "core evaluator runtime"):
            validate_protocol(changed)

    def test_mattersim_compatibility_pins_cannot_float(self) -> None:
        protocol = load_protocol(PROTOCOL)
        changed = copy.deepcopy(protocol.data)
        runtime = changed["mlip"]["runtime_isolation"]["mattersim_environment"]
        runtime["compatibility_pins"]["ase"]["version"] = "3.28.0"
        with self.assertRaisesRegex(ValueError, "MatterSim inference-runtime"):
            validate_protocol(changed)

    def test_r5c_sun_executor_is_hash_locked(self) -> None:
        protocol = load_protocol(PROTOCOL)
        contract = protocol.data["mlip"]["sun_executor"]
        script = ROOT / contract["script"]
        self.assertEqual(
            hashlib.sha256(script.read_bytes()).hexdigest(),
            contract["script_sha256"],
        )

        changed = copy.deepcopy(protocol.data)
        changed["mlip"]["sun_executor"]["frozen_arguments"][
            "max_natoms_per_batch"
        ] = 1024
        with self.assertRaisesRegex(ValueError, "R5-C S.U.N. executor"):
            validate_protocol(changed)


if __name__ == "__main__":
    unittest.main()
