from __future__ import annotations

import copy
import unittest
from pathlib import Path

from crystal_dlm.wqcodiff.crysllmgen.protocol import (
    load_protocol_v4,
    validate_protocol_v4,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/experiments/wyckoff_codiffusion/protocol_v4.yaml"


class CrysLLMGenProtocolV4Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = load_protocol_v4(PROTOCOL)

    def test_registered_protocol_loads_with_bound_artifacts(self) -> None:
        self.assertEqual(self.protocol.name, "crysllmgen_wyckoff_georev_v4")
        self.assertEqual(len(self.protocol.sha256), 64)

    def test_initial_mask_cannot_return(self) -> None:
        changed = copy.deepcopy(self.protocol.data)
        changed["scope"]["initial_global_mask"] = "enabled"
        with self.assertRaisesRegex(ValueError, "initial MASK"):
            validate_protocol_v4(changed)

    def test_retry_and_replacement_remain_forbidden(self) -> None:
        changed = copy.deepcopy(self.protocol.data)
        changed["attempt_contract"]["replacement_sampling"] = "allowed"
        with self.assertRaisesRegex(ValueError, "replacement"):
            validate_protocol_v4(changed)

    def test_rank16_seven_projection_match_cannot_be_weakened(self) -> None:
        changed = copy.deepcopy(self.protocol.data)
        changed["llama_training"]["lora"]["rank"] = 8
        with self.assertRaisesRegex(ValueError, "LoRA capacity"):
            validate_protocol_v4(changed)
        changed = copy.deepcopy(self.protocol.data)
        changed["llama_training"]["lora"]["target_modules"] = ["q_proj", "v_proj"]
        with self.assertRaisesRegex(ValueError, "target modules"):
            validate_protocol_v4(changed)

    def test_r5c_sun_and_thread_contracts_are_locked(self) -> None:
        changed = copy.deepcopy(self.protocol.data)
        changed["evaluation"]["sun_executor"]["script_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "R5-C"):
            validate_protocol_v4(changed)
        changed = copy.deepcopy(self.protocol.data)
        changed["execution"]["threads"]["OPENBLAS_NUM_THREADS"] = 2
        with self.assertRaisesRegex(ValueError, "thread"):
            validate_protocol_v4(changed)

    def test_parent_scheduler_and_reverse_horizon_are_distinct(self) -> None:
        csp = self.protocol.data["assets"]["cspdiffusion"]
        self.assertEqual(csp["scheduler_timesteps"], 1000)
        self.assertEqual(csp["official_reverse_start_timestep"], 800)
        self.assertEqual(csp["upstream_run_type"], "train")
        changed = copy.deepcopy(self.protocol.data)
        changed["assets"]["cspdiffusion"]["scheduler_timesteps"] = 800
        with self.assertRaisesRegex(ValueError, "scheduler"):
            validate_protocol_v4(changed)
        changed = copy.deepcopy(self.protocol.data)
        changed["assets"]["cspdiffusion"]["upstream_run_type"] = "sample"
        with self.assertRaisesRegex(ValueError, "run type"):
            validate_protocol_v4(changed)


if __name__ == "__main__":
    unittest.main()
