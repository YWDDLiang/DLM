from __future__ import annotations

import unittest

import torch
from torch import nn

from crystal_dlm.wqcodiff.training import ExponentialMovingAverage


class ExponentialMovingAverageTests(unittest.TestCase):
    def test_checkpoint_state_is_relocated_to_requested_model_device(self) -> None:
        source_model = nn.Linear(3, 2)
        source = ExponentialMovingAverage(source_model, 0.999)
        state = source.state_dict()

        target_model = nn.Linear(3, 2)
        target = ExponentialMovingAverage(target_model, 0.999)
        target.load_state_dict(state, device=next(target_model.parameters()).device)

        expected_device = next(target_model.parameters()).device
        self.assertTrue(target.shadow)
        self.assertTrue(
            all(value.device == expected_device for value in target.shadow.values())
        )
        # The regression occurred on the first in-place update after loading.
        target.update(target_model)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA regression runs in Slurm")
    def test_cpu_checkpoint_state_can_update_cuda_model(self) -> None:
        cpu_model = nn.Linear(3, 2)
        state = ExponentialMovingAverage(cpu_model, 0.999).state_dict()

        cuda_model = nn.Linear(3, 2).cuda()
        target = ExponentialMovingAverage(cuda_model, 0.999)
        target.load_state_dict(state, device=next(cuda_model.parameters()).device)
        target.update(cuda_model)
        self.assertTrue(all(value.is_cuda for value in target.shadow.values()))


if __name__ == "__main__":
    unittest.main()
