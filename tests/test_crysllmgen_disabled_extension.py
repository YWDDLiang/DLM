from __future__ import annotations

import unittest

from crystal_dlm.wqcodiff.crysllmgen.disabled_extension import (
    DisabledExtensionConfig,
    DisabledExtensionRefiner,
)


class _Model:
    def __init__(self) -> None:
        self.decoder_calls = 0
        self.sample_calls = 0

    def decoder(self, *values):
        self.decoder_calls += 1
        return values

    def sample(self, batch, *, step_lr, diff_steps):
        self.sample_calls += 1
        return {"batch": batch, "step_lr": step_lr, "diff_steps": diff_steps}


class DisabledExtensionTests(unittest.TestCase):
    def test_config_rejects_any_enabled_extension(self) -> None:
        for name in (
            "wyckoff_wrapper_enabled",
            "topology_feedback_enabled",
            "attempt_replacement_enabled",
        ):
            with self.subTest(name=name), self.assertRaises(ValueError):
                DisabledExtensionConfig(**{name: True})

    def test_refiner_is_a_transparent_single_call_adapter(self) -> None:
        model = _Model()
        wrapper = DisabledExtensionRefiner(model)
        values = (1, 2, 3, 4, 5, 6)
        self.assertEqual(wrapper.decoder_step(*values), values)
        self.assertEqual(model.decoder_calls, 1)
        self.assertEqual(
            wrapper.sample("batch", step_lr=1.0e-5, diff_steps=4),
            {"batch": "batch", "step_lr": 1.0e-5, "diff_steps": 4},
        )
        self.assertEqual(model.sample_calls, 1)


if __name__ == "__main__":
    unittest.main()
