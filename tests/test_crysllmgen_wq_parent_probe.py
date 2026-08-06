import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from crystal_dlm.wqcodiff.crysllmgen.atom_sampling import (
    expanded_state_to_parent_batch,
)
from crystal_dlm.wqcodiff.crysllmgen.wq_parent_probe import (
    WQParentCSPProbeConfig,
    _derived_subseed,
    probe,
)
from crystal_dlm.wqcodiff.runtime import ExpandedState


class WQParentCSPProbeTests(unittest.TestCase):
    def test_expanded_state_conversion_preserves_primitive_geometry(self) -> None:
        expanded = ExpandedState(
            conventional_lattice=np.diag([4.0, 5.0, 6.0]),
            primitive_lattice=np.diag([4.0, 5.0, 6.0]),
            fractional_coordinates=np.asarray(
                [[0.0, 0.0, 0.0], [0.25, 0.5, 0.75]],
                dtype=np.float64,
            ),
            atomic_numbers=np.asarray([6, 8], dtype=np.int32),
            atom_to_orbit=np.asarray([0, 0], dtype=np.int64),
            orbit_jacobians=(np.zeros((2, 3, 1), dtype=np.float64),),
            redetected_space_group=1,
        )
        batch = expanded_state_to_parent_batch(expanded, torch.device("cpu"))
        self.assertEqual(batch.num_graphs, 1)
        self.assertEqual(batch.num_nodes, 2)
        self.assertEqual(batch.atom_types.tolist(), [6, 8])
        np.testing.assert_allclose(batch.lengths.numpy(), [[4.0, 5.0, 6.0]])
        np.testing.assert_allclose(batch.angles.numpy(), [[90.0, 90.0, 90.0]])
        np.testing.assert_allclose(
            batch.frac_coords.numpy(),
            expanded.fractional_coordinates,
        )

    def test_probe_uses_same_proposal_subseed_as_wq_sampler(self) -> None:
        seed = 123456789
        raw = f"{seed}:proposal:-1".encode("utf-8")
        import hashlib

        expected = (
            int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")
            & ((1 << 63) - 1)
        )
        self.assertEqual(_derived_subseed(seed, "proposal", -1), expected)

    def test_config_rejects_non_hash_identity(self) -> None:
        with self.assertRaisesRegex(ValueError, "lowercase SHA256"):
            WQParentCSPProbeConfig(
                protocol_path="protocol.yaml",
                gate_a_lock="gate.json",
                csp_checkpoint="model.pt",
                llama_root="llama",
                llama_adapter="adapter",
                output_jsonl="output.jsonl",
                attempt_ledger="ledger.jsonl",
                report_path="report.json",
                experiment_id="probe",
                pairing_id="paired",
                training_seed=11,
                sampling_seed=101,
                attempts=4,
                adapter_training_execution_patch_sha256="x",
                diagnostic_execution_patch_sha256="0" * 64,
            )

    def test_probe_forwards_diagnostic_patch_identity_to_gate(self) -> None:
        protocol = (
            Path(__file__).resolve().parents[1]
            / "configs/experiments/wyckoff_codiffusion/protocol_v4.yaml"
        )
        config = WQParentCSPProbeConfig(
            protocol_path=str(protocol),
            gate_a_lock="gate.json",
            csp_checkpoint="model.pt",
            llama_root="llama",
            llama_adapter="adapter",
            output_jsonl="output.jsonl",
            attempt_ledger="ledger.jsonl",
            report_path="report.json",
            experiment_id="probe-sup27407",
            pairing_id="paired",
            training_seed=11,
            sampling_seed=101,
            attempts=4,
            adapter_training_execution_patch_sha256="1" * 64,
            diagnostic_execution_patch_sha256="2" * 64,
        )
        with (
            mock.patch(
                "crystal_dlm.wqcodiff.crysllmgen.wq_parent_probe.load_protocol_v4"
            ),
            mock.patch(
                "crystal_dlm.wqcodiff.crysllmgen.wq_parent_probe.GateALock.load"
            ) as gate_load,
            mock.patch(
                "crystal_dlm.wqcodiff.crysllmgen.wq_parent_probe.torch.cuda.is_available",
                return_value=False,
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "requires CUDA"):
                probe(config)
        gate_load.assert_called_once_with(
            config.gate_a_lock,
            project_root=protocol.resolve().parents[3],
            protocol_path=config.protocol_path,
            execution_patch_manifest_sha256=(
                config.diagnostic_execution_patch_sha256
            ),
        )


if __name__ == "__main__":
    unittest.main()
