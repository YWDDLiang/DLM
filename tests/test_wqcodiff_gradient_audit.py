from __future__ import annotations

import math
import unittest

import torch
from torch import nn

from scripts.a800.audit_component_gradients import (
    _parameter_group_norm,
    summarize_records,
)


class ComponentGradientAuditTests(unittest.TestCase):
    def test_parameter_group_norm_uses_only_selected_gradients(self) -> None:
        model = nn.Sequential(nn.Linear(2, 2, bias=False), nn.Linear(2, 1, bias=False))
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        value = _parameter_group_norm(
            tuple(model.named_parameters()),
            lambda name: name.startswith("0."),
        )
        self.assertAlmostEqual(value, math.sqrt(4.0), places=6)

    def test_summary_keeps_term_specific_distributions(self) -> None:
        records = [
            {
                "term": "bridge",
                "loss": float(index),
                "global_grad_norm": float(index + 1),
                "shared_backbone_grad_norm": float(index + 2),
                "task_specific_grad_norm": float(index + 3),
            }
            for index in range(1, 5)
        ]
        summary = summarize_records(records)
        self.assertEqual(summary["bridge"]["loss"]["max"], 4.0)
        self.assertEqual(summary["bridge"]["global_grad_norm"]["min"], 2.0)


if __name__ == "__main__":
    unittest.main()
