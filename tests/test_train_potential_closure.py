from collections import Counter
import importlib.util
import math
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "src" / "scripts" / "train_potential_closure.py"
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
TRANSFORMERS_AVAILABLE = importlib.util.find_spec("transformers") is not None
MODULE = None
if TORCH_AVAILABLE and TRANSFORMERS_AVAILABLE:
    SPEC = importlib.util.spec_from_file_location("train_potential_closure", SCRIPT)
    MODULE = importlib.util.module_from_spec(SPEC)
    assert SPEC and SPEC.loader
    sys.modules[SPEC.name] = MODULE
    SPEC.loader.exec_module(MODULE)


@unittest.skipUnless(
    TORCH_AVAILABLE and TRANSFORMERS_AVAILABLE,
    "torch/transformers are unavailable",
)
class PotentialClosureTrainerTest(unittest.TestCase):
    def strata(self):
        result = {}
        cursor = 0
        for stratum in MODULE.EXPECTED_STRATA:
            result[stratum] = list(
                range(cursor, cursor + MODULE.EXPECTED_GROUPS_PER_STRATUM)
            )
            cursor += MODULE.EXPECTED_GROUPS_PER_STRATUM
        return result

    def test_frozen_four_step_schedule(self):
        self.assertEqual(
            [MODULE.optimizer_objective(step) for step in range(1, 9)],
            [
                "clean_ce",
                "cell",
                "clean_ce",
                "site",
                "clean_ce",
                "cell",
                "clean_ce",
                "site",
            ],
        )
        counts = Counter(
            MODULE.optimizer_objective(step)
            for step in range(1, MODULE.TOTAL_UPDATES + 1)
        )
        self.assertEqual(
            counts,
            Counter({"clean_ce": 1024, "cell": 512, "site": 512}),
        )

    def test_potential_stream_is_strict_three_clean_three_on_policy(self):
        streams = MODULE.TransactionBatchStreams(
            self.strata(), mode="potential_closed", seed=7
        )
        exposure = Counter()
        seen = {
            stratum: Counter() for stratum in MODULE.EXPECTED_STRATA
        }
        for kind in MODULE.KINDS:
            for _ in range(MODULE.CYCLES):
                batch = streams.take(kind)
                self.assertEqual(
                    Counter(batch.domains),
                    Counter({"mp20_clean": 3, "on_policy": 3}),
                )
                for index, domain in zip(
                    batch.group_indices, batch.domains, strict=True
                ):
                    stratum = f"{domain}_{kind}"
                    exposure[stratum] += 1
                    seen[stratum][index] += 1
        streams.assert_exhausted()
        for stratum in MODULE.EXPECTED_STRATA:
            self.assertEqual(exposure[stratum], 1536)
            self.assertEqual(set(seen[stratum].values()), {3})

    def test_control_stream_reads_only_clean_and_matches_six_microbatches(self):
        streams = MODULE.TransactionBatchStreams(
            self.strata(), mode="closure_control", seed=7
        )
        exposure = Counter()
        for kind in MODULE.KINDS:
            for _ in range(MODULE.CYCLES):
                batch = streams.take(kind)
                self.assertEqual(len(batch.group_indices), 6)
                self.assertEqual(set(batch.domains), {"mp20_clean"})
                exposure[kind] += len(batch.group_indices)
        streams.assert_exhausted()
        self.assertEqual(exposure, Counter({"cell": 3072, "site": 3072}))

    def test_lr_warmup_reaches_five_e_minus_six_at_update_100(self):
        self.assertAlmostEqual(MODULE.learning_rate_for_update(1), 5e-8)
        self.assertAlmostEqual(MODULE.learning_rate_for_update(50), 2.5e-6)
        self.assertAlmostEqual(MODULE.learning_rate_for_update(100), 5e-6)
        self.assertAlmostEqual(MODULE.learning_rate_for_update(2048), 5e-6)

    def test_transaction_teacher_ce_is_length_normalized(self):
        scores = MODULE.torch.tensor([-6.0, -3.0], requires_grad=True)
        lattice = MODULE.transaction_clean_ce(
            scores, target_index=0, transaction_length=6
        )
        site = MODULE.transaction_clean_ce(
            scores, target_index=1, transaction_length=3
        )
        self.assertAlmostEqual(float(lattice.detach()), 1.0)
        self.assertAlmostEqual(float(site.detach()), 1.0)
        (lattice + site).backward()
        self.assertTrue(MODULE.torch.isfinite(scores.grad).all())

    def test_probe_decision_uses_median_ratios_cosines_and_kl(self):
        passing = [
            {
                "clean_grad_norm": 2.0,
                "cell_grad_norm": 1.0,
                "site_grad_norm": 4.0,
                "cell_to_clean_ratio": 0.5,
                "site_to_clean_ratio": 2.0,
                "clean_cell_cosine": 0.1,
                "clean_site_cosine": 0.2,
                "max_teacher_kl_nats": 0.05,
            }
            for _ in range(5)
        ]
        self.assertTrue(MODULE.probe_statistic_decision(passing)["passed"])
        failing = [dict(row) for row in passing]
        for row in failing:
            row["clean_site_cosine"] = -0.6
        self.assertFalse(MODULE.probe_statistic_decision(failing)["passed"])

    def test_sequential_joint_scorer_supports_three_and_six_tokens(self):
        torch = MODULE.torch

        class Output:
            def __init__(self, logits):
                self.logits = logits

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(1.0))

            def forward(self, input_ids, attention_mask):
                del attention_mask
                vocab = 16
                values = torch.arange(vocab, dtype=torch.float32)
                logits = self.scale * values.reshape(1, 1, -1)
                return Output(logits.expand(input_ids.shape[0], input_ids.shape[1], -1))

        class Runtime:
            def __init__(self):
                self.model = TinyModel()

            def activate_policy(self, trainable):
                self.model.scale.requires_grad_(trainable)

            def activate_reference(self):
                self.model.scale.requires_grad_(False)

        for width in (3, 6):
            runtime = Runtime()
            batch = {
                "input_ids": torch.zeros((width + 2,), dtype=torch.long),
                "attention_mask": torch.ones((width + 2,), dtype=torch.long),
                "active_absolute": torch.arange(1, width + 1),
                "action_tokens": torch.stack(
                    (
                        torch.arange(1, width + 1),
                        torch.arange(2, width + 2),
                    )
                ),
            }
            scores = MODULE.sequential_action_scores(
                runtime, batch, reference=False
            )
            self.assertEqual(tuple(scores.shape), (2,))
            self.assertTrue(torch.isfinite(scores).all())
            scores.sum().backward()
            self.assertIsNotNone(runtime.model.scale.grad)

    def test_slurm_contracts_pin_resources_probe_and_modes(self):
        probe = (ROOT / "slurm" / "185_probe_potential_closure_gradients.sbatch").read_text()
        train = (ROOT / "slurm" / "186_train_potential_closure.sbatch").read_text()
        self.assertIn("--gres=gpu:NVIDIAA800-SXM4-80GB:1", probe)
        self.assertIn("--cpus-per-task=8", probe)
        self.assertIn("--probe-only", probe)
        self.assertIn("--gres=gpu:NVIDIAA800-SXM4-80GB:2", train)
        self.assertIn("--cpus-per-task=8", train)
        self.assertIn("closure_control", train)
        self.assertIn("potential_closed", train)
        self.assertIn("formal_action_pool_gate", train)
        self.assertIn("PROBE_FINAL.json", train)


if __name__ == "__main__":
    unittest.main()
