from __future__ import annotations

import copy
import unittest

import torch
from transformers import get_cosine_schedule_with_warmup

from crystal_dlm.h1_chemistry_first_sft import record_order_sha256
from crystal_dlm.peft_adapter_identity import adapter_value_sha256
from scripts.llama_h1_chemistry_first_sft import audited_warmup_optimizer_step


BASE_LR = 2e-6
WARMUP_STEPS = 135
TOTAL_UPDATES = 4505


class TinyAdapterPair(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = torch.nn.ModuleDict(
            {
                "candidate": torch.nn.Linear(2, 2, bias=False),
                "reference": torch.nn.Linear(2, 2, bias=False),
            }
        )
        with torch.no_grad():
            self.block.reference.weight.copy_(self.block.candidate.weight)
        self.block.reference.weight.requires_grad_(False)
        self.active_adapter = "candidate"

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.block.candidate(value)


def make_runtime() -> tuple[TinyAdapterPair, list[torch.nn.Parameter], torch.optim.Optimizer, object]:
    torch.manual_seed(26080817)
    model = TinyAdapterPair()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=BASE_LR, weight_decay=0.0)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=TOTAL_UPDATES,
    )
    return model, trainable, optimizer, scheduler


def loss_for(model: TinyAdapterPair) -> torch.Tensor:
    value = torch.tensor([[1.0, -2.0], [0.5, 3.0]])
    target = torch.tensor([[0.25, -0.75], [1.25, 0.5]])
    return torch.square(model(value) - target).mean()


def optimizer_state_signature(model: TinyAdapterPair, optimizer) -> dict[str, tuple]:
    result = {}
    for name, parameter in model.named_parameters():
        if ".candidate." not in name:
            continue
        state = optimizer.state.get(parameter) or {}
        result[name] = (
            float(state["step"].item()) if state else None,
            state["exp_avg"].detach().clone() if state else None,
            state["exp_avg_sq"].detach().clone() if state else None,
        )
    return result


class ChemistryFirstOptimizerAuditTest(unittest.TestCase):
    def _audit_step(self, runtime, step: int):
        model, trainable, optimizer, scheduler = runtime
        loss_for(model).backward()
        ids = [f"record-{index:02d}" for index in range(step * 8)]
        initial_reference = getattr(self, "initial_reference", None)
        initial_candidate = getattr(self, "initial_candidate", None)
        if initial_reference is None:
            initial_reference = adapter_value_sha256(model, "reference")
            initial_candidate = adapter_value_sha256(model, "candidate")
            self.initial_reference = initial_reference
            self.initial_candidate = initial_candidate
        return audited_warmup_optimizer_step(
            model,
            trainable,
            optimizer,
            scheduler,
            step_number=step,
            base_lr=BASE_LR,
            warmup_steps=WARMUP_STEPS,
            reference_sha_initial=initial_reference,
            candidate_sha_initial=initial_candidate,
            consumed_record_ids=ids,
            expected_record_order_sha256=record_order_sha256(
                [{"record_id": value} for value in ids]
            ),
        )

    def setUp(self) -> None:
        self.initial_reference = None
        self.initial_candidate = None

    def test_scheduler_initializes_at_registered_zero_lr(self) -> None:
        _, _, optimizer, scheduler = make_runtime()
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.0)
        self.assertEqual(scheduler.last_epoch, 0)

    def test_zero_lr_step_advances_moments_without_changing_adapter(self) -> None:
        runtime = make_runtime()
        report = self._audit_step(runtime, 1)
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["lr_used"], 0.0)
        self.assertAlmostEqual(report["lr_next"], BASE_LR / WARMUP_STEPS)
        self.assertFalse(report["candidate_changed"])
        self.assertEqual(report["candidate_sha_after"], report["candidate_sha_initial"])
        self.assertEqual(report["reference_sha_after"], report["reference_sha_initial"])
        self.assertTrue(report["optimizer_state_after"]["passed"])
        self.assertGreater(
            report["optimizer_state_after"]["nonzero_moment_tensor_count"], 0
        )

    def test_second_step_is_first_positive_parameter_update(self) -> None:
        runtime = make_runtime()
        first = self._audit_step(runtime, 1)
        second = self._audit_step(runtime, 2)
        self.assertTrue(first["passed"], first)
        self.assertTrue(second["passed"], second)
        self.assertAlmostEqual(second["lr_used"], BASE_LR / WARMUP_STEPS)
        self.assertAlmostEqual(second["lr_next"], 2 * BASE_LR / WARMUP_STEPS)
        self.assertTrue(second["candidate_changed"])
        self.assertNotEqual(second["candidate_sha_after"], second["candidate_sha_initial"])
        self.assertEqual(second["reference_sha_after"], second["reference_sha_initial"])
        self.assertEqual(second["optimizer_state_after"]["step_values_min"], 2.0)

    def test_audit_instrumentation_preserves_two_step_trajectory(self) -> None:
        audited = make_runtime()
        plain_model = copy.deepcopy(audited[0])
        plain_trainable = [
            parameter for parameter in plain_model.parameters() if parameter.requires_grad
        ]
        plain_optimizer = torch.optim.AdamW(
            plain_trainable, lr=BASE_LR, weight_decay=0.0
        )
        plain_scheduler = get_cosine_schedule_with_warmup(
            plain_optimizer,
            num_warmup_steps=WARMUP_STEPS,
            num_training_steps=TOTAL_UPDATES,
        )
        for step in (1, 2):
            report = self._audit_step(audited, step)
            self.assertTrue(report["passed"], report)
            loss_for(plain_model).backward()
            torch.nn.utils.clip_grad_norm_(plain_trainable, 1.0)
            plain_optimizer.step()
            plain_scheduler.step()
            plain_optimizer.zero_grad(set_to_none=True)
        self.assertEqual(
            adapter_value_sha256(audited[0], "candidate"),
            adapter_value_sha256(plain_model, "candidate"),
        )
        self.assertEqual(audited[3].last_epoch, plain_scheduler.last_epoch)
        self.assertEqual(audited[2].param_groups[0]["lr"], plain_optimizer.param_groups[0]["lr"])
        left = optimizer_state_signature(audited[0], audited[2])
        right = optimizer_state_signature(plain_model, plain_optimizer)
        self.assertEqual(set(left), set(right))
        for name in left:
            self.assertEqual(left[name][0], right[name][0])
            self.assertTrue(torch.equal(left[name][1], right[name][1]))
            self.assertTrue(torch.equal(left[name][2], right[name][2]))

    def test_all_none_gradient_fails_before_optimizer_step(self) -> None:
        runtime = make_runtime()
        model = runtime[0]
        ids = [f"record-{index:02d}" for index in range(8)]
        report = audited_warmup_optimizer_step(
            *runtime,
            step_number=1,
            base_lr=BASE_LR,
            warmup_steps=WARMUP_STEPS,
            reference_sha_initial=adapter_value_sha256(model, "reference"),
            candidate_sha_initial=adapter_value_sha256(model, "candidate"),
            consumed_record_ids=ids,
            expected_record_order_sha256=record_order_sha256(
                [{"record_id": value} for value in ids]
            ),
        )
        self.assertFalse(report["passed"])
        self.assertFalse(report["optimizer_step_performed"])
        self.assertIn("candidate_gradient_health", report["failures"])

    def test_all_zero_gradient_fails_before_optimizer_step(self) -> None:
        runtime = make_runtime()
        model = runtime[0]
        (model.block.candidate.weight.sum() * 0.0).backward()
        ids = [f"record-{index:02d}" for index in range(8)]
        report = audited_warmup_optimizer_step(
            *runtime,
            step_number=1,
            base_lr=BASE_LR,
            warmup_steps=WARMUP_STEPS,
            reference_sha_initial=adapter_value_sha256(model, "reference"),
            candidate_sha_initial=adapter_value_sha256(model, "candidate"),
            consumed_record_ids=ids,
            expected_record_order_sha256=record_order_sha256(
                [{"record_id": value} for value in ids]
            ),
        )
        self.assertFalse(report["passed"])
        self.assertFalse(report["optimizer_step_performed"])
        self.assertIn("zero_or_nonfinite_gradient_norm", report["failures"])

    def test_reference_gradient_and_scheduler_reordering_fail_closed(self) -> None:
        runtime = make_runtime()
        model = runtime[0]
        loss_for(model).backward()
        model.block.reference.weight.grad = torch.ones_like(model.block.reference.weight)
        runtime[3].step()
        ids = [f"record-{index:02d}" for index in range(8)]
        report = audited_warmup_optimizer_step(
            *runtime,
            step_number=1,
            base_lr=BASE_LR,
            warmup_steps=WARMUP_STEPS,
            reference_sha_initial=adapter_value_sha256(model, "reference"),
            candidate_sha_initial=adapter_value_sha256(model, "candidate"),
            consumed_record_ids=ids,
            expected_record_order_sha256=record_order_sha256(
                [{"record_id": value} for value in ids]
            ),
        )
        self.assertFalse(report["passed"])
        self.assertFalse(report["optimizer_step_performed"])
        self.assertIn("candidate_gradient_health", report["failures"])
        self.assertIn("lr_used", report["failures"])
        self.assertIn("scheduler_epoch_before", report["failures"])

    def test_optimizer_noop_at_positive_step_is_detected(self) -> None:
        runtime = make_runtime()
        first = self._audit_step(runtime, 1)
        self.assertTrue(first["passed"], first)
        loss_for(runtime[0]).backward()
        runtime[2].step = lambda closure=None: None
        ids = [f"record-{index:02d}" for index in range(16)]
        second = audited_warmup_optimizer_step(
            *runtime,
            step_number=2,
            base_lr=BASE_LR,
            warmup_steps=WARMUP_STEPS,
            reference_sha_initial=self.initial_reference,
            candidate_sha_initial=self.initial_candidate,
            consumed_record_ids=ids,
            expected_record_order_sha256=record_order_sha256(
                [{"record_id": value} for value in ids]
            ),
        )
        self.assertFalse(second["passed"])
        self.assertIn("optimizer_state_after", second["failures"])
        self.assertIn("candidate_change_semantics", second["failures"])


if __name__ == "__main__":
    unittest.main()
