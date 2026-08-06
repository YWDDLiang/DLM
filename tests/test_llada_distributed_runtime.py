import unittest
from unittest import mock


try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - local mirror may omit Torch.
    torch = None


if torch is not None:
    from scripts.llada_sft import DistributedNoPaddingSampler, evaluate
else:
    DistributedNoPaddingSampler = None
    evaluate = None


@unittest.skipIf(torch is None, "torch is not installed in this environment")
class LLaDADistributedRuntimeTests(unittest.TestCase):
    def test_no_padding_sampler_keeps_odd_denominator_unique(self):
        dataset = list(range(5))
        rank0 = DistributedNoPaddingSampler(
            dataset,
            num_replicas=2,
            rank=0,
        )
        rank1 = DistributedNoPaddingSampler(
            dataset,
            num_replicas=2,
            rank=1,
        )
        self.assertEqual(list(rank0), [0, 2, 4])
        self.assertEqual(list(rank1), [1, 3])
        self.assertEqual(sorted([*rank0, *rank1]), list(range(5)))

    def test_evaluate_uses_sample_weight_mass_not_batch_means(self):
        class ToyModel:
            def eval(self):
                return self

            def train(self):
                return self

        loader = [
            {
                "input_ids": torch.zeros((2, 1), dtype=torch.long),
                "sample_weights": torch.ones((2,), dtype=torch.float32),
            },
            {
                "input_ids": torch.zeros((1, 1), dtype=torch.long),
                "sample_weights": torch.ones((1,), dtype=torch.float32),
            },
        ]
        with mock.patch(
            "scripts.llada_sft.compute_loss",
            side_effect=[torch.tensor(1.0), torch.tensor(3.0)],
        ):
            observed = evaluate(
                ToyModel(),
                loader,
                torch.device("cpu"),
                max_batches=2,
                loss_config={},
                distributed=False,
            )
        self.assertAlmostEqual(observed, 5.0 / 3.0, places=7)


if __name__ == "__main__":
    unittest.main()
