from __future__ import annotations

from types import SimpleNamespace
import unittest


try:
    import torch

    from crystal_dlm.deployed_transaction_scoring import (
        score_deployed_transaction_actions,
    )
except ModuleNotFoundError:
    torch = None
    score_deployed_transaction_actions = None


_TorchModuleBase = torch.nn.Module if torch is not None else object


class FakeTokenizer:
    """Small atomic crystal vocabulary; no external tokenizer is loaded."""

    def __init__(self) -> None:
        self.vocab = {
            "<N_001>": 1,
            "<N_002>": 2,
            "<LA_000>": 3,
            "<LA_010>": 4,
            "<LA_040>": 5,
            "<LB_000>": 6,
            "<LB_010>": 7,
            "<LB_040>": 8,
            "<LC_000>": 9,
            "<LC_010>": 10,
            "<LC_040>": 11,
            "<AA_090>": 12,
            "<AB_090>": 13,
            "<AG_090>": 14,
            "<AG_001>": 15,
            "<E_A>": 20,
            "<E_B>": 21,
            "<X_000>": 30,
            "<X_040>": 31,
            "<X_050>": 32,
            "<X_100>": 33,
            "<X_010>": 34,
            "<Y_000>": 40,
            "<Y_040>": 41,
            "<Y_050>": 42,
            "<Y_100>": 43,
            "<Y_010>": 44,
            "<Z_000>": 50,
            "<Z_040>": 51,
            "<Z_050>": 52,
            "<Z_100>": 53,
            "<Z_010>": 54,
            "<OUTSIDE_SCHEMA>": 60,
            "<PROMPT>": 70,
        }

    def get_vocab(self):
        return dict(self.vocab)


@unittest.skipIf(torch is None, "torch unavailable")
class DeployedTransactionScoringTest(unittest.TestCase):
    VOCAB_SIZE = 128
    MASK_ID = 127

    class StaticModel(_TorchModuleBase):
        def __init__(self, base_logits=None, *, prefix_aware=False):
            super().__init__()
            self.output = torch.nn.Embedding(
                DeployedTransactionScoringTest.VOCAB_SIZE, 1
            )
            self.scale = torch.nn.Parameter(torch.tensor(1.0))
            if base_logits is None:
                base_logits = torch.zeros(
                    DeployedTransactionScoringTest.VOCAB_SIZE,
                    dtype=torch.float32,
                )
            self.register_buffer("base_logits", base_logits.float())
            self.prefix_aware = bool(prefix_aware)
            self.inputs = []
            self.attention_masks = []

        def get_output_embeddings(self):
            return self.output

        def forward(self, input_ids, attention_mask=None):
            self.inputs.append(input_ids.detach().clone())
            if attention_mask is not None:
                self.attention_masks.append(attention_mask.detach().clone())
            batch, length = input_ids.shape
            logits = (
                self.scale
                * self.base_logits.to(input_ids.device).reshape(1, 1, -1)
            ).expand(batch, length, -1).clone()
            if self.prefix_aware:
                x_position = 1 + 8
                y_position = 1 + 9
                x_is_50 = (input_ids[:, x_position] == 32).to(logits.dtype)
                logits[:, y_position, 40] += (1.0 - x_is_50) * 2.0
                logits[:, y_position, 42] += x_is_50 * 2.0
            return SimpleNamespace(logits=logits)

    def setUp(self):
        self.tokenizer = FakeTokenizer()
        self.vocab = self.tokenizer.get_vocab()

    def schema(self, num_atoms):
        v = self.vocab
        allowed = [
            [v[f"<N_{num_atoms:03d}>"]],
            [v["<LA_000>"], v["<LA_010>"], v["<LA_040>"]],
            [v["<LB_000>"], v["<LB_010>"], v["<LB_040>"]],
            [v["<LC_000>"], v["<LC_010>"], v["<LC_040>"]],
            [v["<AA_090>"]],
            [v["<AB_090>"]],
            [v["<AG_001>"], v["<AG_090>"]],
        ]
        for _ in range(num_atoms):
            allowed.extend(
                [
                    [v["<E_A>"], v["<E_B>"]],
                    [v["<X_000>"], v["<X_010>"], v["<X_040>"], v["<X_050>"], v["<X_100>"]],
                    [v["<Y_000>"], v["<Y_010>"], v["<Y_040>"], v["<Y_050>"], v["<Y_100>"]],
                    [v["<Z_000>"], v["<Z_010>"], v["<Z_040>"], v["<Z_050>"], v["<Z_100>"]],
                ]
            )
        return allowed

    def constraints(self):
        v = self.vocab
        coord_tokens = {
            "X": {
                v["<X_000>"]: 0,
                v["<X_010>"]: 10,
                v["<X_040>"]: 40,
                v["<X_050>"]: 50,
                v["<X_100>"]: 100,
            },
            "Y": {
                v["<Y_000>"]: 0,
                v["<Y_010>"]: 10,
                v["<Y_040>"]: 40,
                v["<Y_050>"]: 50,
                v["<Y_100>"]: 100,
            },
            "Z": {
                v["<Z_000>"]: 0,
                v["<Z_010>"]: 10,
                v["<Z_040>"]: 40,
                v["<Z_050>"]: 50,
                v["<Z_100>"]: 100,
            },
        }
        return {
            "representation": "dynamic_v1",
            "duplicate_coordinate_mask": True,
            "lattice_volume_mask": True,
            "canonicalize_periodic_alias": True,
            "pbc_min_distance_mask": True,
            "pbc_min_distance_A": 0.5,
            "pbc_image_radius": 2,
            "min_lattice_rad": 1.0e-4,
            "max_atoms": 2,
            "body_offset": 0,
            "coord_period": 100,
            "count_token_to_n": {v["<N_001>"]: 1, v["<N_002>"]: 2},
            "coord_token_to_bin": coord_tokens,
            "coord_bin_to_token_id": {
                axis: {value: token for token, value in mapping.items()}
                for axis, mapping in coord_tokens.items()
            },
            "coordinate_alias_token_ids": {
                "X": (v["<X_000>"], v["<X_100>"]),
                "Y": (v["<Y_000>"], v["<Y_100>"]),
                "Z": (v["<Z_000>"], v["<Z_100>"]),
            },
            "length_token_to_bin": {
                "LA": {v["<LA_000>"]: 0, v["<LA_010>"]: 10, v["<LA_040>"]: 40},
                "LB": {v["<LB_000>"]: 0, v["<LB_010>"]: 10, v["<LB_040>"]: 40},
                "LC": {v["<LC_000>"]: 0, v["<LC_010>"]: 10, v["<LC_040>"]: 40},
            },
            "length_step": 0.1,
            "angle_token_to_bin": {
                "AA": {v["<AA_090>"]: 90},
                "AB": {v["<AB_090>"]: 90},
                "AG": {v["<AG_001>"]: 1, v["<AG_090>"]: 90},
            },
            "gamma_bin_to_token_id": {1: v["<AG_001>"], 90: v["<AG_090>"]},
            "z_bin_to_token_id": {
                value: token for token, value in coord_tokens["Z"].items()
            },
            "zero_length_token_ids_by_position": {
                1: v["<LA_000>"],
                2: v["<LB_000>"],
                3: v["<LC_000>"],
            },
        }

    def complete(self, num_atoms=1):
        v = self.vocab
        suffix = [
            v[f"<N_{num_atoms:03d}>"],
            v["<LA_040>"],
            v["<LB_040>"],
            v["<LC_040>"],
            v["<AA_090>"],
            v["<AB_090>"],
            v["<AG_090>"],
            v["<E_A>"],
            v["<X_000>"],
            v["<Y_000>"],
            v["<Z_000>"],
        ]
        if num_atoms == 2:
            suffix.extend(
                [
                    v["<E_B>"],
                    v["<X_050>"],
                    v["<Y_050>"],
                    v["<Z_050>"],
                ]
            )
        return torch.tensor([v["<PROMPT>"]] + suffix, dtype=torch.long)

    def score(self, model, *, complete, positions, actions, allowed=None):
        num_atoms = (len(complete) - 1 - 7) // 4
        return score_deployed_transaction_actions(
            model,
            complete,
            prompt_length=1,
            gen_length=7 + 4 * num_atoms,
            generation_positions=positions,
            action_token_ids=actions,
            attention_mask=torch.ones((1,), dtype=torch.long),
            mask_id=self.MASK_ID,
            allowed_token_ids_by_generation_pos=(
                allowed if allowed is not None else self.schema(num_atoms)
            ),
            atom_count_grammar=None,
            lightweight_decoding_constraints=self.constraints(),
        )

    def test_temperature_and_schema_mask_define_normalizer(self):
        v = self.vocab
        base = torch.zeros(self.VOCAB_SIZE)
        base[v["<X_050>"]] = 0.7
        base[v["<OUTSIDE_SCHEMA>"]] = 100.0
        model = self.StaticModel(base)
        allowed = self.schema(1)
        allowed[8] = [v["<X_000>"], v["<X_050>"]]
        allowed[9] = [v["<Y_050>"]]
        allowed[10] = [v["<Z_050>"]]
        result = self.score(
            model,
            complete=self.complete(1),
            positions=(8, 9, 10),
            actions=[[v["<X_050>"], v["<Y_050>"], v["<Z_050>"]]],
            allowed=allowed,
        )
        expected = torch.log_softmax(torch.tensor([0.0, 0.7]) / 0.7, dim=0)[1]
        untempered = torch.log_softmax(torch.tensor([0.0, 0.7]), dim=0)[1]
        self.assertAlmostEqual(result.action_logprobs[0].item(), expected.item(), places=6)
        self.assertNotAlmostEqual(result.action_logprobs[0].item(), untempered.item(), places=3)
        self.assertTrue(result.action_audits[0].valid)
        (-result.candidate_log_mass).backward()
        self.assertIsNotNone(model.scale.grad)
        self.assertTrue(torch.isfinite(model.scale.grad))
        self.assertNotEqual(float(model.scale.grad), 0.0)

    def test_later_component_probability_depends_on_committed_prefix(self):
        v = self.vocab
        model = self.StaticModel(prefix_aware=True)
        allowed = self.schema(1)
        allowed[8] = [v["<X_000>"], v["<X_050>"]]
        allowed[9] = [v["<Y_000>"], v["<Y_050>"]]
        allowed[10] = [v["<Z_000>"]]
        result = self.score(
            model,
            complete=self.complete(1),
            positions=(8, 9, 10),
            actions=[
                [v["<X_000>"], v["<Y_000>"], v["<Z_000>"]],
                [v["<X_050>"], v["<Y_000>"], v["<Z_000>"]],
            ],
            allowed=allowed,
        )
        left_y = result.action_audits[0].component_logprobs[1]
        right_y = result.action_audits[1].component_logprobs[1]
        self.assertIsNotNone(left_y)
        self.assertIsNotNone(right_y)
        self.assertGreater(left_y, right_y)
        self.assertTrue(bool((model.inputs[0][:, 1 + 8 : 1 + 11] == self.MASK_ID).all()))
        self.assertEqual(int(model.inputs[1][0, 1 + 8]), v["<X_000>"])
        self.assertEqual(int(model.inputs[1][1, 1 + 8]), v["<X_050>"])
        self.assertTrue(bool((model.inputs[1][:, 1 + 9 : 1 + 11] == self.MASK_ID).all()))
        self.assertTrue(all(bool(mask.all()) for mask in model.attention_masks))

    def test_schema_masked_target_is_explicit_invalid_not_nan(self):
        v = self.vocab
        result = self.score(
            self.StaticModel(),
            complete=self.complete(1),
            positions=(8, 9, 10),
            actions=[[v["<OUTSIDE_SCHEMA>"], v["<Y_000>"], v["<Z_000>"]]],
        )
        self.assertTrue(torch.isneginf(result.action_logprobs[0]))
        self.assertTrue(torch.isneginf(result.candidate_log_mass))
        self.assertFalse(bool(torch.isnan(result.action_logprobs).any()))
        self.assertEqual(result.action_audits[0].invalid_step, 0)
        self.assertEqual(result.action_audits[0].invalid_reason, "schema_masked_target")

    def test_pbc_collision_path_is_negative_infinity(self):
        v = self.vocab
        result = self.score(
            self.StaticModel(),
            complete=self.complete(2),
            positions=(12, 13, 14),
            actions=[
                [v["<X_050>"], v["<Y_050>"], v["<Z_050>"]],
                [v["<X_000>"], v["<Y_000>"], v["<Z_010>"]],
            ],
        )
        self.assertTrue(result.action_audits[0].valid)
        self.assertFalse(result.action_audits[1].valid)
        self.assertEqual(result.action_audits[1].invalid_step, 2)
        self.assertEqual(
            result.action_audits[1].invalid_reason,
            "dynamic_geometry_masked_target",
        )
        self.assertTrue(torch.isneginf(result.action_logprobs[1]))
        self.assertFalse(bool(torch.isnan(result.action_logprobs).any()))

    def test_zero_lattice_length_is_rejected_by_dynamic_lattice_mask(self):
        v = self.vocab
        result = self.score(
            self.StaticModel(),
            complete=self.complete(1),
            positions=(1, 2, 3, 4, 5, 6),
            actions=[
                [
                    v["<LA_000>"],
                    v["<LB_040>"],
                    v["<LC_040>"],
                    v["<AA_090>"],
                    v["<AB_090>"],
                    v["<AG_090>"],
                ]
            ],
        )
        self.assertFalse(result.action_audits[0].valid)
        self.assertEqual(result.action_audits[0].invalid_step, 0)
        self.assertEqual(
            result.action_audits[0].invalid_reason,
            "dynamic_geometry_masked_target",
        )
        self.assertTrue(torch.isneginf(result.action_logprobs[0]))

    def test_cell_final_geometry_fallback_marks_path_invalid(self):
        v = self.vocab
        complete = self.complete(2).clone()
        complete[1 + 12] = v["<X_040>"]
        complete[1 + 13] = v["<Y_000>"]
        complete[1 + 14] = v["<Z_000>"]
        result = self.score(
            self.StaticModel(),
            complete=complete,
            positions=(1, 2, 3, 4, 5, 6),
            actions=[
                [
                    v["<LA_010>"],
                    v["<LB_010>"],
                    v["<LC_010>"],
                    v["<AA_090>"],
                    v["<AB_090>"],
                    v["<AG_090>"],
                ]
            ],
        )
        self.assertFalse(result.action_audits[0].valid)
        self.assertEqual(
            result.action_audits[0].invalid_reason,
            "cell_geometry_unsupported",
        )
        self.assertTrue(torch.isneginf(result.action_logprobs[0]))

    def test_xyz_and_cell_complete_transaction_widths_are_supported(self):
        v = self.vocab
        model = self.StaticModel()
        xyz = self.score(
            model,
            complete=self.complete(1),
            positions=(8, 9, 10),
            actions=[[v["<X_050>"], v["<Y_050>"], v["<Z_050>"]]],
        )
        cell = self.score(
            model,
            complete=self.complete(1),
            positions=(1, 2, 3, 4, 5, 6),
            actions=[
                [
                    v["<LA_040>"],
                    v["<LB_040>"],
                    v["<LC_040>"],
                    v["<AA_090>"],
                    v["<AB_090>"],
                    v["<AG_090>"],
                ]
            ],
        )
        self.assertEqual(xyz.transaction_kind, "xyz")
        self.assertEqual(cell.transaction_kind, "cell")
        self.assertEqual(len(xyz.action_audits[0].component_logprobs), 3)
        self.assertEqual(len(cell.action_audits[0].component_logprobs), 6)
        self.assertTrue(torch.isfinite(xyz.action_logprobs).all())
        self.assertTrue(torch.isfinite(cell.action_logprobs).all())

    def test_candidate_logsumexp_remains_finite_when_probabilities_underflow(self):
        v = self.vocab
        base = torch.zeros(self.VOCAB_SIZE)
        for token in (
            v["<X_000>"],
            v["<X_050>"],
            v["<Y_000>"],
            v["<Y_050>"],
            v["<Z_000>"],
            v["<Z_050>"],
            v["<X_100>"],
            v["<Y_100>"],
            v["<Z_100>"],
        ):
            base[token] = -1000.0
        result = self.score(
            self.StaticModel(base),
            complete=self.complete(1),
            positions=(8, 9, 10),
            actions=[
                [v["<X_000>"], v["<Y_000>"], v["<Z_000>"]],
                [v["<X_050>"], v["<Y_050>"], v["<Z_050>"]],
            ],
        )
        self.assertTrue(torch.isfinite(result.action_logprobs).all())
        self.assertTrue(torch.isfinite(result.candidate_log_mass))
        self.assertEqual(float(torch.exp(result.action_logprobs).sum()), 0.0)
        self.assertTrue(
            torch.allclose(
                result.candidate_log_mass,
                torch.logsumexp(result.action_logprobs, dim=0),
            )
        )

    def test_duplicate_paths_are_counted_once_in_candidate_mass(self):
        v = self.vocab
        action = [v["<X_050>"], v["<Y_050>"], v["<Z_050>"]]
        result = self.score(
            self.StaticModel(),
            complete=self.complete(1),
            positions=(8, 9, 10),
            actions=[action, action],
        )
        self.assertEqual(result.action_audits[1].duplicate_of, 0)
        self.assertEqual(result.unique_valid_action_count, 1)
        self.assertTrue(
            torch.allclose(result.candidate_log_mass, result.action_logprobs[0])
        )


if __name__ == "__main__":
    unittest.main()
