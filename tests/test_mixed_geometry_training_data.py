"""Independent accounting and continuous-source checks for the mixed trainer."""
from copy import deepcopy
import hashlib
from types import SimpleNamespace
import unittest

import torch

from crystal_dlm.dynamic_crystal import arrays_to_dynamic_tokens
from crystal_dlm.manifold_corruption import lattice_matrix_from_parameters
from crystal_dlm.mixed_geometry_diffusion import LatticeNormalizer
from crystal_dlm.mixed_geometry_training_data import MixedBatchSchedule, MixedGeometrySources, materialize_mixed_batch
from scripts.train_mixed_geometry_dlm import mixed_batch_loss
from test_state_programmed_runtime import TinyTokenizer, constraints


class Tokenizer(TinyTokenizer):
    def __init__(self):
        super().__init__()
        self.pad_token_id = self.vocab["<PAD>"]
        self.mask_token_id = self.mask_id

    def __call__(self, text, **kwargs):
        return {"input_ids": [self.vocab["<E_H>"]]}


def source_fixture(index=0):
    fractional = [[.003137, .202314, .398721], [.507913, .604822, .796132]]
    lengths, angles = [4.0234, 5.0432, 6.0321], [85.213, 95.327, 105.413]
    tokens, _ = arrays_to_dynamic_tokens(lengths, angles, ["H", "H"], fractional)
    answer = " ".join(tokens)
    row = {"source_row_idx": index, "source_split": "train", "prompt": "original Plan",
           "answer": answer, "source_answer": answer,
           "plan_state": {"N": 2, "elements": ["H"], "counts": [2]},
           "species_program": ["H"], "species_program_source": "frozen_planner",
           "sample_weight": 1., "energy_above_hull": -999., "forced_mask_positions": [1]}
    identity = {"source_split": "train", "source_row_idx": index, "identity_verified": True,
                "target_answer_sha256_bytes": hashlib.sha256(answer.encode()).hexdigest(),
                "continuous_aligned": {"lattice_matrix_A": lattice_matrix_from_parameters(lengths, angles).tolist(),
                                       "species": ["H", "H"], "fractional": fractional}}
    return row, identity


def sources(rows=None, identities=None):
    row, identity = source_fixture()
    tokenizer = Tokenizer()
    support = constraints(tokenizer)
    support.update(canonicalize_periodic_alias=True)
    normalizer = LatticeNormalizer((0.,)*6, (1.,)*6, (1.,)*6, 1e-6, 1., 1)
    data = MixedGeometrySources(rows or [row], identities or [identity], tokenizer, support, normalizer,
                                split="train", seed=1917)
    return data, tokenizer


class ScheduleTests(unittest.TestCase):
    def collect(self, world, count=27):
        schedule = MixedBatchSchedule(count, world)
        result = {}
        real = {"token": [], "geometry": []}
        padding = {"token": 0, "geometry": 0}
        for rank in range(world):
            for window, micro, mode, slots in schedule.rank_batches(rank=rank, seed=7, epoch=1):
                result.setdefault((window, mode), []).extend((micro // 2 * world + rank, item) for item in slots)
                for index, padded in slots:
                    if padded:
                        padding[mode] += 1
                    else:
                        real[mode].append(index)
        normalized = {key: [item for _, item in sorted(value)] for key, value in result.items()}
        return schedule, normalized, real, padding

    def test_four_and_six_cards_have_identical_source_windows(self):
        four, a, real, padding = self.collect(4)
        six, b, _, _ = self.collect(6)
        self.assertEqual(a, b)
        self.assertEqual((four.accumulation, six.accumulation), (6, 4))
        for mode in ("token", "geometry"):
            self.assertEqual(sorted(real[mode]), list(range(27)))
            self.assertEqual(padding[mode], 9)
        for window in range(four.updates_per_epoch):
            self.assertEqual(a[(window, "token")], a[(window, "geometry")])

    def test_distributed_accumulation_and_padding_equal_population_gradient(self):
        count = 27
        expected = sum((index + 1) * (mode + 1) for index in range(count) for mode in range(2)) / (2 * count)
        for world in (4, 6):
            schedule = MixedBatchSchedule(count, world)
            parameter = torch.tensor(1., dtype=torch.float64, requires_grad=True)
            for rank in range(world):
                for _, _, mode, slots in schedule.rank_batches(rank=rank, seed=7, epoch=1):
                    coefficients = [(index + 1) * (1 if mode == "token" else 2) * (not padded) for index, padded in slots]
                    loss = parameter * sum(coefficients) / len(slots)
                    (loss * schedule.epoch_normalization / schedule.accumulation / world
                     / schedule.updates_per_epoch).backward()
            self.assertAlmostEqual(float(parameter.grad), expected, places=11)

    def test_official_source_budget_keeps_all_sources(self):
        for world in (4, 6):
            manifest = MixedBatchSchedule(27136, world).manifest(2)
            self.assertEqual(manifest["updates"], 4524)
            self.assertEqual(manifest["effective_states"], 108544)
            self.assertEqual(manifest["padding_per_epoch"], 16)


class SourceTests(unittest.TestCase):
    def test_continuous_values_survive_and_never_enter_token_scaffold(self):
        data, tokenizer = sources()
        self.assertEqual(float(data.fractional[0, 0, 0]), .003137)
        example = data.example(0, mode="geometry", epoch=0)
        self.assertNotIn("energy_above_hull", example)
        self.assertNotIn("forced_mask_positions", example)
        for position in data.sources[0].transaction_positions:
            self.assertEqual(example["input_body"][position], tokenizer.mask_id)
        batch = materialize_mixed_batch([example], tokenizer, device="cpu")
        self.assertEqual(batch["geometry_state"].z.dtype, torch.float32)
        self.assertEqual(batch["geometry_state"].atom_mask.sum().item(), 2)
        self.assertEqual(batch["species"][0, :2].tolist(), [1, 1])
        self.assertEqual(int(batch["geometry_context"].active_token_mask.sum()), 12)

    def test_noise_depends_on_source_and_epoch_not_order_or_padding(self):
        row0, identity0 = source_fixture(0)
        row1, identity1 = source_fixture(1)
        a, _ = sources([row0, row1], [identity0, identity1])
        b, _ = sources([row1, row0], [identity0, identity1])
        x = a.example(0, mode="geometry", epoch=3)["noisy_geometry"]
        y = b.example(1, mode="geometry", epoch=3, is_padding=True)["noisy_geometry"]
        for name in ("z", "fractional", "t"):
            self.assertTrue(torch.equal(getattr(x.state, name), getattr(y.state, name)))
        self.assertTrue(torch.equal(x.v_target, y.v_target))
        self.assertFalse(torch.equal(x.state.z, a.example(0, mode="geometry", epoch=4)["noisy_geometry"].state.z))

    def test_identity_missing_duplicate_or_target_mutation_is_rejected(self):
        row, identity = source_fixture()
        for candidate in ({**identity, "identity_verified": False},
                          {**identity, "target_answer_sha256_bytes": "changed"},
                          {**identity, "source_row_idx": 9}):
            with self.assertRaises(ValueError):
                sources([row], [candidate])
        with self.assertRaises(ValueError):
            sources([row], [identity, deepcopy(identity)])

    def test_geometry_padding_is_a_fixed_batch_mean_and_has_zero_gradient(self):
        data, tokenizer = sources()
        examples = [data.example(0, mode="geometry", epoch=0, is_padding=pad) for pad in (False, True)]
        batch = materialize_mixed_batch(examples, tokenizer, device="cpu")
        v = torch.ones(2, 6, requires_grad=True)
        u = torch.ones(2, 20, 3, requires_grad=True)
        batch["v_target"].zero_()
        batch["u_target"].zero_()
        loss, _, _ = mixed_batch_loss(SimpleNamespace(v_prediction=v, u_prediction=u), batch, None)
        self.assertAlmostEqual(float(loss.detach()), .5)
        loss.backward()
        self.assertEqual(float(v.grad[1].abs().sum()), 0.)
        self.assertEqual(float(u.grad[1].abs().sum()), 0.)
        self.assertGreater(float(v.grad[0].abs().sum()), 0.)
        self.assertEqual(float(u.grad[0, 2:].abs().sum()), 0.)


if __name__ == "__main__":
    unittest.main()
